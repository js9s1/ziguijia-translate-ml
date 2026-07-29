import logging
import multiprocessing
import os
import shutil
import signal
import sqlite3
import sys as _sys
import threading
import time
import uuid
from collections.abc import Callable
from queue import Queue

import psutil

# ── Use "spawn" to avoid inheriting CUDA/ROCm state via fork ──
# On Linux, multiprocessing defaults to "fork", which copies the parent's
# entire virtual address space into every child.  When the parent has
# PyTorch + ROCm loaded (11+ GiB virtual memory), each child inherits
# that reservation.  With daemon=False children, orphan accumulation
# after GPU hangs quickly leads to multiple 72 GiB processes.
#
# "spawn" creates a fresh Python interpreter with no inherited state,
# so children start with a clean ~500 MiB footprint.
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # already set in a parent context

from db_schema import ConnectionManager, init_jobs_schema
from middleware import _DEFAULT_PARAMS
from singleton import singleton
from valkey_util import publish_job_status

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "jobs.db")

logger = logging.getLogger(__name__)

from job_checkpoint import (
    _purge_step_artifacts,
    clear_checkpoint_for_file,
    clear_edited_srt_files,
    get_checkpoint,
    get_checkpoint_edited,
    get_edited_srt_files,
    invalidate_checkpoints_after,
    set_checkpoint,
    set_checkpoint_edited,
    set_edited_srt_file,
)
from job_orphan import (
    cleanup_orphan_processes,
    kill_process_group,
    kill_processes_by_output_dir,
)
from job_types import _JOB_TYPE_LABELS, _SKIP_QUEUE_INIT, JobStatus, _get_job_type_label, _get_run_func
from job_worker import (
    _WORKER_HEARTBEAT_STALE,
    _now_str,
    _run_job,
    _safe_close_proc,
    _safe_close_psutil_procs,
    _try_mark_failed,
)


def _job_process_wrapper(job_data: dict, run_func_name: str):
    """Entry point for a child process executing a single job.

    Creates its own process group so that all subprocesses spawned by the
    job handler can be killed as a group on cancellation.

    Exits with 0 on success, non-zero on failure.

    With ``spawn`` start method the child is a fresh Python interpreter
    — no forked CUDA state, no inherited DB connection.  We set
    ``_SKIP_QUEUE_INIT`` so that when the handler later calls
    ``get_job_queue()`` the singleton skips the harmful startup
    bookkeeping (orphan cleanup, status resets).
    """
    global _SKIP_QUEUE_INIT
    _SKIP_QUEUE_INIT = True

    os.setpgid(os.getpid(), os.getpid())

    run_func = _get_run_func(run_func_name)
    if run_func is None:
        _sys.exit(1)
    run_func(job_data)



@singleton
class JobQueue:
    def __init__(self):
        self._conn = ConnectionManager(DB_FILE)
        self._queue = Queue()
        self._worker_thread: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._cancel_lock = threading.Lock()
        self._current_access_code: str | None = None
        self._running = False
        self._heartbeat_ts = 0.0
        self._graceful_shutdown = False
        self._shutdown_timeout = 60  # seconds to wait for current job
        self._shutdown_done = threading.Event()  # set when worker finishes
        self._last_gpu_reset_ts = 0.0  # cooldown for _reset_gpu_state
        self._init_db()
        self._load_pending_jobs()
        if not _SKIP_QUEUE_INIT:
            self._ensure_worker()

    def _get_conn(self) -> sqlite3.Connection:
        return self._conn.get()

    def _load_pending_jobs(self):
        if _SKIP_QUEUE_INIT:
            return

        conn = self._get_conn()

        # Kill orphan subprocesses left behind by the dead server instance
        # BEFORE resetting statuses so the orphans are still findable via
        # their PROCESSING output_dir.
        self._cleanup_orphan_processes()

        # Also kill any children that were in PROCESSING state — the parent
        # died and the non-daemon child might still be running.  Scan their
        # output directories and terminate any processes referencing them.
        processing_rows = conn.execute(
            "SELECT output_dir FROM jobs WHERE status = ? AND output_dir IS NOT NULL", (JobStatus.PROCESSING.value,)
        ).fetchall()
        for row in processing_rows:
            self._kill_processes_by_output_dir(row["output_dir"])

        # Now find any jobs that were still PROCESSING when the server died —
        # reset them to PENDING so they can be retried.
        now = _now_str()
        conn.execute(
            "UPDATE jobs SET status = ?, status_changed_at = ? WHERE status = ?",
            (JobStatus.PENDING.value, now, JobStatus.PROCESSING.value),
        )
        conn.commit()

        rows = conn.execute("SELECT access_code FROM jobs WHERE status = ?", (JobStatus.PENDING.value,)).fetchall()
        for row in rows:
            self._queue.put(row[0])
        logger.info(f"Loaded {len(rows)} pending jobs from database")

    def _cleanup_orphan_processes(self):
        return cleanup_orphan_processes(self)

    def _init_db(self):
        conn = self._get_conn()
        init_jobs_schema(conn)

    def _generate_access_code(self) -> str:
        return str(uuid.uuid4())[:8].upper()

    def _find_failed_ocr_job(self, video_number: str, user_id: int) -> tuple[str, str] | None:
        """Return (access_code, output_dir) of a failed ning OCR job for the same video+user, if any."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT access_code, output_dir FROM jobs WHERE video_number = ? AND user_id = ? AND run_func_name = ? AND status = ? ORDER BY created_at DESC LIMIT 1",
            (video_number, user_id, "_run_video_ning_ocr_job", JobStatus.FAILED.value),
        ).fetchone()
        if row:
            return row[0], row[1]
        return None

    def _find_failed_ning_job(self, video_number: str, user_id: int) -> tuple[str, str] | None:
        """Return (access_code, output_dir) of a failed ning SRT-translate job for the same video+user, if any."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT access_code, output_dir FROM jobs WHERE video_number = ? AND user_id = ? AND run_func_name = ? AND status = ? ORDER BY created_at DESC LIMIT 1",
            (video_number, user_id, "_run_video_job", JobStatus.FAILED.value),
        ).fetchone()
        if row:
            return row[0], row[1]
        return None

    def add_job(self, job_data: dict, run_func: Callable[[dict], None], user_id: int = None) -> str:
        conn = self._get_conn()
        access_code = job_data.get("access_code") or self._generate_access_code()

        run_func_name = run_func.__name__

        # Preserve existing checkpoint on resubmit — OR REPLACE would otherwise
        # wipe it since the INSERT column list omits the checkpoint column.
        existing_checkpoint = conn.execute(
            "SELECT checkpoint FROM jobs WHERE access_code = ?", (access_code,)
        ).fetchone()
        prev_ckpt = existing_checkpoint[0] if existing_checkpoint else ""

        now = _now_str()
        conn.execute(
            """
            INSERT OR REPLACE INTO jobs (access_code, srt_path, output_dir, temperature, status, error, run_func_name, video_number, video_file, user_id, text, blur, target_language, cfg_weight, exaggeration, start_trim, end_trim, cached_path, filename, checkpoint, created_at, status_changed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                access_code,
                job_data.get("srt_path"),
                job_data.get("output_dir"),
                job_data.get("temperature"),
                JobStatus.PENDING.value,
                None,
                run_func_name,
                job_data.get("video_number"),
                job_data.get("video_file"),
                user_id,
                job_data.get("text"),
                job_data.get("blur", "yes"),
                job_data.get("target_language", "en"),
                job_data.get("cfg_weight", _DEFAULT_PARAMS["cfg_weight"]),
                job_data.get("exaggeration", _DEFAULT_PARAMS["exaggeration"]),
                job_data.get("start_trim"),
                job_data.get("end_trim"),
                job_data.get("cached_path"),
                job_data.get("filename"),
                prev_ckpt,
                now,
                now,
            ),
        )
        conn.commit()

        self._queue.put(access_code)
        # Ensure the worker thread is alive (it may have died silently)
        self._ensure_worker()

        return access_code

    def _kill_processes_by_output_dir(self, output_dir: str, sig: int = signal.SIGTERM):
        return kill_processes_by_output_dir(output_dir, sig)

    def _kill_process_group(self, proc, output_dir: str | None = None):
        return kill_process_group(self, proc, output_dir)

    def _close_conn(self):
        """Close the current thread-local DB connection, if open.

        Used after ``os.fork()`` so the child process doesn't share
        the parent's connection — each process manages its own.
        """
        self._conn.close()

    def _is_worker_healthy(self) -> bool:
        """Return True if the worker thread is alive and has checked in recently."""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            return False
        elapsed = time.monotonic() - self._heartbeat_ts
        if elapsed > _WORKER_HEARTBEAT_STALE:
            logger.warning("Worker thread heartbeat stale (%.1fs since last pulse)", elapsed)
            return False
        return True

    def _ensure_worker(self):
        if self._worker_thread is not None and not self._is_worker_healthy():
            logger.warning("Worker thread dead or stale — restarting")
            self._running = False
            old_thread = self._worker_thread
            self._worker_thread = None
            # Let the old thread exit cleanly (daemon thread won't block
            # process shutdown, but joining prevents reference leaks)
            try:
                old_thread.join(timeout=5)
            except RuntimeError:
                pass  # thread not started or already joined
        if not self._running:
            self._running = True
            self._heartbeat_ts = time.monotonic()
            self._worker_thread = threading.Thread(
                target=self._process_queue,
                daemon=True,
                name="jobqueue-worker",
            )
            self._worker_thread.start()

    def _try_mark_failed(self, access_code, error_msg):
        return _try_mark_failed(self, access_code, error_msg)

    def _process_queue(self):
        import queue as std_queue

        _last_orphan_check = time.monotonic()
        _ORPHAN_CHECK_INTERVAL = 120  # every 2 minutes
        while self._running:
            self._heartbeat_ts = time.monotonic()

            # Periodic orphan reaper — cleans up subprocesses that
            # outlived their job (GPU hang, kill -9 on parent, etc.)
            if time.monotonic() - _last_orphan_check > _ORPHAN_CHECK_INTERVAL:
                _last_orphan_check = time.monotonic()
                self._cleanup_orphan_processes()

            if self._graceful_shutdown:
                # Don't dequeue new jobs — let the current one finish, then exit
                with self._cancel_lock:
                    if self._current_access_code is not None:
                        time.sleep(1)
                        continue
                break
            try:
                access_code = self._queue.get(timeout=1)
            except std_queue.Empty:
                continue  # timeout, just re-check self._running
            with self._cancel_lock:
                self._current_access_code = access_code
                self._cancel_event.clear()
            try:
                self._process_job(access_code)
            except BaseException as e:
                logger.exception(f"_process_job {access_code} raised {type(e).__name__}: {e}")
                self._try_mark_failed(access_code, str(e)[:500])
            finally:
                with self._cancel_lock:
                    self._current_access_code = None
        # Drain the queue on exit so any remaining items don't block
        # the queue's internal Condition and leak a thread reference.
        while True:
            try:
                self._queue.get_nowait()
            except std_queue.Empty:
                break
        self._shutdown_done.set()

    def _process_job(self, access_code: str):
        return _run_job(self, access_code, _job_process_wrapper)

    def set_checkpoint(self, access_code: str, checkpoint: str):
        return set_checkpoint(self, access_code, checkpoint)

    def clear_checkpoint_for_file(self, access_code: str, file_path: str):
        return clear_checkpoint_for_file(self, access_code, file_path)

    def invalidate_checkpoints_after(self, access_code: str, step: str):
        return invalidate_checkpoints_after(self, access_code, step)

    def _purge_step_artifacts(self, output_dir: str, steps: set[str]):
        return _purge_step_artifacts(self, output_dir, steps)

    def set_checkpoint_edited(self, access_code: str, edited: bool = True):
        return set_checkpoint_edited(self, access_code, edited)

    def get_checkpoint_edited(self, access_code: str) -> bool:
        return get_checkpoint_edited(self, access_code)

    def set_edited_srt_file(self, access_code: str, filename: str):
        return set_edited_srt_file(self, access_code, filename)

    def clear_edited_srt_files(self, access_code: str):
        return clear_edited_srt_files(self, access_code)

    def get_edited_srt_files(self, access_code: str) -> list[str]:
        return get_edited_srt_files(self, access_code)

    def get_checkpoint(self, access_code: str) -> str:
        return get_checkpoint(self, access_code)

    def update_job_progress(self, access_code: str, progress: str):
        conn = self._get_conn()
        conn.execute("UPDATE jobs SET progress = ? WHERE access_code = ?", (progress, access_code))
        conn.commit()
        publish_job_status(access_code, "progress", progress=progress)

    def update_target_language(self, access_code: str, lang: str):
        """Update the target_language field for a job (e.g. after language detection)."""
        conn = self._get_conn()
        conn.execute("UPDATE jobs SET target_language = ? WHERE access_code = ?", (lang, access_code))
        conn.commit()

    def get_status(self, access_code: str) -> dict | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT access_code, status, error, output_dir, progress, target_language, created_at, temperature, cfg_weight, exaggeration, checkpoint, checkpoint_edited, edited_srt_files FROM jobs WHERE access_code = ?",
            (access_code,),
        ).fetchone()

        if not row:
            return None

        # Count how many pending jobs are ahead of this one in the queue
        queue_position = None
        if row["status"] == JobStatus.PENDING.value and row["created_at"]:
            queue_position = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = ? AND created_at < ?",
                (JobStatus.PENDING.value, row["created_at"]),
            ).fetchone()[0]

        return {
            "access_code": row["access_code"],
            "status": row["status"],
            "error": row["error"],
            "output_dir": row["output_dir"],
            "progress": row["progress"],
            "target_language": row["target_language"],
            "queue_position": queue_position,
            "temperature": row["temperature"],
            "cfg_weight": row["cfg_weight"],
            "exaggeration": row["exaggeration"],
            "checkpoint": row["checkpoint"] or "",
            "checkpoint_edited": bool(row["checkpoint_edited"]),
            "edited_srt_files": row["edited_srt_files"].split(",") if row["edited_srt_files"] else [],
        }

    def get_user_jobs(self, user_id: int) -> list:
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT access_code, run_func_name, status, error, output_dir, created_at, status_changed_at
            FROM jobs WHERE user_id = ? AND status != ?
            ORDER BY COALESCE(status_changed_at, created_at) DESC
        """,
            (user_id, JobStatus.DELETED.value),
        ).fetchall()
        type_map = _JOB_TYPE_LABELS
        return [
            {
                "access_code": r[0],
                "type": type_map.get(r[1], r[1] or "未知"),
                "status": r[2],
                "error": r[3],
                "output_dir": r[4],
                "created_at": r[5],
                "status_changed_at": r[6],
            }
            for r in rows
        ]

    def cancel_job(self, access_code: str) -> dict:
        conn = self._get_conn()
        row = conn.execute("SELECT status, output_dir FROM jobs WHERE access_code = ?", (access_code,)).fetchone()

        if not row:
            return {"success": False, "error": "Job not found"}

        status, output_dir = row

        if status not in (JobStatus.PENDING.value, JobStatus.PROCESSING.value):
            return {"success": False, "error": f"Job is already {status}, cannot cancel"}

        if output_dir:
            # Kill any process whose cmdline references this output dir,
            # including subprocesses spawned by the job handler.
            killed_pids = set()
            killed_procs = []
            for proc in psutil.process_iter(["pid", "cmdline"]):
                try:
                    if output_dir in " ".join(proc.info["cmdline"] or []):
                        # Kill the entire process group so children don't orphan
                        try:
                            pgid = os.getpgid(proc.info["pid"])
                            os.killpg(pgid, signal.SIGTERM)
                        except (ProcessLookupError, PermissionError, AttributeError):
                            proc.send_signal(signal.SIGTERM)
                        killed_pids.add(proc.info["pid"])
                        killed_procs.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
            if killed_pids:
                _, alive = psutil.wait_procs(
                    [p for p in psutil.process_iter(["pid"]) if p.info["pid"] in killed_pids],
                    timeout=3,
                )
                for p in alive:
                    try:
                        pgid = os.getpgid(p.pid)
                        os.killpg(pgid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError, OSError):
                        try:
                            p.kill()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                _safe_close_psutil_procs(killed_procs)

        # Signal the worker thread to skip completion logic
        with self._cancel_lock:
            if access_code == self._current_access_code:
                self._cancel_event.set()

        now = _now_str()
        conn.execute(
            "UPDATE jobs SET status = ?, error = ?, cancelled_at = ?, status_changed_at = ? WHERE access_code = ? AND (status = ? OR status = ?)",
            (
                JobStatus.CANCELLED.value,
                "Cancelled by user",
                now,
                now,
                access_code,
                JobStatus.PENDING.value,
                JobStatus.PROCESSING.value,
            ),
        )
        conn.commit()
        publish_job_status(access_code, JobStatus.CANCELLED.value)

        return {"success": True, "message": "Job cancelled"}

    def resubmit_job(self, access_code: str) -> dict:
        conn = self._get_conn()
        row = conn.execute("SELECT status FROM jobs WHERE access_code = ?", (access_code,)).fetchone()

        if not row:
            return {"success": False, "error": "Job not found"}

        if row[0] == JobStatus.DELETED.value:
            return {"success": False, "error": "Job has been deleted"}

        # Allow resubmit for failed, cancelled, or completed jobs (with checkpoint_edited flag)
        if row[0] == JobStatus.COMPLETED.value:
            # Must have checkpoint_edited flag set
            ckpt_row = conn.execute(
                "SELECT checkpoint_edited FROM jobs WHERE access_code = ?", (access_code,)
            ).fetchone()
            if not ckpt_row or not ckpt_row[0]:
                return {
                    "success": False,
                    "error": "Job is completed, only failed, cancelled or checkpoint-edited jobs can be resubmitted",
                }
        elif row[0] not in (JobStatus.FAILED.value, JobStatus.CANCELLED.value):
            return {
                "success": False,
                "error": f"Job is {row[0]}, only failed, cancelled or checkpoint-edited jobs can be resubmitted",
            }

        now = _now_str()
        conn.execute(
            "UPDATE jobs SET status = ?, error = NULL, status_changed_at = ? WHERE access_code = ?",
            (JobStatus.PENDING.value, now, access_code),
        )
        conn.commit()

        self._queue.put(access_code)
        # Ensure the worker thread is alive (it may have died silently)
        self._ensure_worker()

        return {"success": True, "message": "Job resubmitted"}

    def delete_job(self, access_code: str) -> dict:
        conn = self._get_conn()
        row = conn.execute("SELECT status FROM jobs WHERE access_code = ?", (access_code,)).fetchone()

        if not row:
            return {"success": False, "error": "Job not found"}

        # Mark as deleted without removing files or DB row
        now = _now_str()
        conn.execute(
            "UPDATE jobs SET status = ?, error = ?, deleted_at = ?, status_changed_at = ? WHERE access_code = ?",
            (JobStatus.DELETED.value, "Deleted by user", now, now, access_code),
        )
        conn.commit()

        return {"success": True, "message": "Job hidden"}

    def shutdown(self, timeout: int | None = None):
        """Initiate graceful shutdown.

        Signals the worker thread to stop dequeuing new jobs and waits
        for the currently-running job to finish (up to *timeout* seconds).
        If the job doesn't finish in time, it is terminated and marked
        PENDING so it auto-resumes on next startup.
        """
        if timeout is not None:
            self._shutdown_timeout = timeout
        self._graceful_shutdown = True
        self._running = False
        logger.info(
            "Graceful shutdown initiated — waiting up to %ds for current job",
            self._shutdown_timeout,
        )
        # Wait for the worker thread to finish (it will exit once the
        # current job completes or times out).
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=self._shutdown_timeout + 5)
        if self._shutdown_done.is_set():
            logger.info("Graceful shutdown complete")
        else:
            logger.warning("Graceful shutdown timed out — worker may still be running")

    def stop(self):
        self._running = False

    def clear_job_queue(self, dry_run: bool = False) -> dict:
        """Remove all deleted jobs from database and their output directories.

        Args:
            dry_run: If True, report what would be removed without doing anything.

        Returns a summary of what was cleaned up.
        """
        conn = self._get_conn()

        # Find all deleted jobs with their output directories
        rows = conn.execute(
            "SELECT access_code, output_dir FROM jobs WHERE status = ?", (JobStatus.DELETED.value,)
        ).fetchall()

        if not rows:
            return {"success": True, "message": "No deleted jobs found", "jobs_removed": 0, "dirs_removed": 0}

        if dry_run:
            dirs_found = sum(1 for r in rows if r["output_dir"] and os.path.exists(r["output_dir"]))
            return {
                "success": True,
                "message": f"Would remove {len(rows)} jobs and {dirs_found} directories",
                "jobs_removed": len(rows),
                "dirs_removed": dirs_found,
            }

        jobs_removed = 0
        dirs_removed = 0
        errors = []

        # Remove output directories
        for row in rows:
            access_code, output_dir = row["access_code"], row["output_dir"]

            if output_dir and os.path.exists(output_dir):
                try:
                    shutil.rmtree(output_dir)
                    dirs_removed += 1
                    logger.info(f"Removed output directory for deleted job {access_code}: {output_dir}")
                except Exception as e:
                    error_msg = f"Failed to remove directory {output_dir} for job {access_code}: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)

        # Remove job entries from database
        try:
            cursor = conn.execute("DELETE FROM jobs WHERE status = ?", (JobStatus.DELETED.value,))
            jobs_removed = cursor.rowcount
            conn.commit()
            logger.info(f"Removed {jobs_removed} deleted job entries from database")
        except Exception as e:
            error_msg = f"Failed to remove job entries from database: {e}"
            errors.append(error_msg)
            logger.error(error_msg)

        result = {
            "success": len(errors) == 0,
            "jobs_removed": jobs_removed,
            "dirs_removed": dirs_removed,
        }

        if errors:
            result["errors"] = errors
            result["message"] = f"Completed with {len(errors)} errors"
        else:
            result["message"] = f"Successfully removed {jobs_removed} jobs and {dirs_removed} directories"

        return result


def get_job_queue() -> JobQueue:
    return JobQueue()
