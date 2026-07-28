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
from enum import Enum
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

from config import CHECKPOINT_ORDER, FILENAME_TO_CHECKPOINT_STEP
from db_schema import JOB_COLUMNS, ConnectionManager, init_jobs_schema
from middleware import _DEFAULT_PARAMS
from valkey_util import publish_job_status
from singleton import singleton

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "jobs.db")

_SKIP_QUEUE_INIT = False
"""When True, ``_load_pending_jobs`` is a no-op (spawn child context).

With ``fork`` the child inherits the parent's already-initialised
JobQueue singleton so ``__init__`` never runs.  With ``spawn`` every
module import creates a fresh instance, and we must **not** let the
child tamper with job statuses or kill orphans that belong to the
parent process.
"""

logger = logging.getLogger(__name__)


def _extract_failure_reason(output_dir: str | None) -> str | None:
    """Read ``job.log`` from *output_dir* and extract a human-readable failure reason.

    Returns a short error summary string, or None if no log is available.
    """
    if not output_dir:
        return None
    log_path = os.path.join(output_dir, "job.log")
    if not os.path.isfile(log_path):
        return None
    try:
        with open(log_path, errors="replace") as f:
            # Read the last ~8 KiB — enough for a traceback + surrounding context
            f.seek(0, os.SEEK_END)
            size = f.tell()
            start = max(0, size - 8192)
            f.seek(start)
            tail = f.read()
    except OSError:
        return None

    lines = tail.splitlines()
    # Walk backwards looking for error markers
    captured: list[str] = []
    in_traceback = False
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            if in_traceback:
                continue
            break  # blank line outside traceback → end of relevant section

        is_tb = stripped.startswith("Traceback ") or stripped.startswith("  File ") or (
            in_traceback and not stripped.startswith("---")
        )
        is_error = any(
            marker in stripped
            for marker in (
                "Error:",
                "error:",
                "RuntimeError",
                "CalledProcessError",
                "Conversion failed!",
                "No filtered frames",
                "Invalid argument",
                "cannot",
                "failed",
                "FAILED",
                "missing:",
            )
        )
        if is_tb or is_error:
            in_traceback = True
            captured.append(stripped)
            continue
        if in_traceback:
            # We hit non-error/non-tb content — stop collecting
            break

    if captured:
        # Reverse back to chronological order, take last few lines
        captured.reverse()
        # Keep the most relevant ~5 lines
        keep = captured[-5:]
        return "; ".join(keep)[:500]
    return None


def _reset_gpu_state():
    """Reset the ROCm/HIP GPU driver state between jobs.

    Long-running jobs (especially video processing) can leave the AMD
    Renoir iGPU (gfx90c) ROCm driver in an unstable state, causing
    subsequent jobs to crash with SIGABRT during heavier GPU workloads
    even though a lightweight health probe passes.

    Uses the shared ``gpu_probe.run_gpu_probe`` — a heavier workload
    (256–1024 tensors) than the TTS pre-flight check.
    """
    from gpu_probe import run_gpu_probe

    r = run_gpu_probe([256, 512, 1024], reset_peak_memory=True)
    if r is None:
        return
    if "OK" in r.stdout:
        logger.info("GPU state reset between jobs — ok")
    else:
        logger.warning(
            "GPU state reset between jobs — probe failed: stderr=%s stdout=%s",
            r.stderr.strip()[:200], r.stdout.strip()[:200],
        )


def _reset_gpu_state_cooldown(jq) -> bool:
    """Conditionally reset GPU state with a 60 s cooldown.
    Returns True if a reset was attempted.
    """
    now = time.monotonic()
    if now - jq._last_gpu_reset_ts < 60:
        return False
    jq._last_gpu_reset_ts = now
    _reset_gpu_state()
    return True

def _now_str() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')

# Worker thread heartbeat: if no pulse within this many seconds, consider it dead.
_WORKER_HEARTBEAT_STALE = 30


def _safe_close_proc(proc: multiprocessing.Process | None):
    """Close a multiprocessing.Process to release OS resources.

    Must be called after ``proc.join()`` regardless of whether the process
    exited normally or was killed.  On POSIX, failing to close a non-daemon
    Process leaks a pipe fd and keeps the process in the process table as
    a zombie until the parent exits.
    """
    if proc is None:
        return
    try:
        proc.close()
    except Exception:
        pass


def _safe_close_psutil_procs(procs):
    """Close psutil.Process objects to release their cached /proc handles."""
    for p in procs:
        try:
            if isinstance(p, psutil.Process):
                p.cmdline.cache_clear() if hasattr(p.cmdline, 'cache_clear') else None
        except Exception:
            pass


# ── Handler registry ────────────────────────────────────────
# Each entry is ``(module_name, attr_name)`` so the import happens
# lazily and only the *needed* handler module is loaded.  This keeps
# the spawn child from importing PyTorch (audio_job.py) for every
# job type and blowing up virtual memory for no reason.
_HANDLER_MODULES: dict[str, tuple[str, str]] = {
    "_run_gen_audio":                            ("audio_job",     "_run_gen_audio"),
    "_run_audio_segmentation_job":               ("audio_job",     "_run_audio_segmentation_job"),
    "_run_tts_job":                              ("tts_job",       "_run_tts_job"),
    "_run_video_job":                            ("video_ning_job","_run_video_job"),
    "_run_video_ning_ocr_job":                   ("video_ning_job","_run_video_ning_ocr_job"),
    "_run_video_ning_ocr_translate_only_job":    ("video_ning_job","_run_video_ning_ocr_translate_only_job"),
    "_run_video_custom_job":                     ("video_custom_job","_run_video_custom_job"),
    "_run_video_auto_job":                       ("video_custom_job","_run_video_auto_job"),
    "_run_video_ocr_job":                        ("video_custom_job","_run_video_ocr_job"),
    "_run_ocr_only_job":                         ("video_ocr_job", "_run_ocr_only_job"),
}

_JOB_TYPE_LABELS: dict[str, str] = {
    "_run_gen_audio": "音频生成",
    "_run_video_job": "宁视频翻译",
    "_run_video_custom_job": "自定义视频",
    "_run_tts_job": "语音合成",
    "_run_video_auto_job": "自动翻译视频",
    "_run_audio_segmentation_job": "音频分段合成",
    "_run_video_ocr_job": "OCR翻译视频",
    "_run_video_ning_ocr_job": "宁视频OCR翻译",
    "_run_video_ning_ocr_translate_only_job": "宁视频OCR仅翻译",
    "_run_ocr_only_job": "视频OCR提取字幕",
}


class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DELETED = "deleted"


def _get_run_func(name: str) -> Callable | None:
    """Lazily import and return a single job handler function.

    Uses ``importlib`` so only the module containing *name* is loaded.
    This avoids pulling ``torch`` / ``ChatterboxMultilingualTTS`` into
    every spawn child for job types that don't need them.
    """
    spec = _HANDLER_MODULES.get(name)
    if spec is None:
        return None
    module_name, attr_name = spec
    import importlib
    mod = importlib.import_module(module_name)
    return getattr(mod, attr_name)


def _get_job_type_label(run_func_name: str) -> str:
    return _JOB_TYPE_LABELS.get(run_func_name, run_func_name or "未知")


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
            "SELECT output_dir FROM jobs WHERE status = ? AND output_dir IS NOT NULL",
            (JobStatus.PROCESSING.value,)
        ).fetchall()
        for row in processing_rows:
            self._kill_processes_by_output_dir(row["output_dir"])

        # Now find any jobs that were still PROCESSING when the server died —
        # reset them to PENDING so they can be retried.
        now = _now_str()
        conn.execute(
            "UPDATE jobs SET status = ?, status_changed_at = ? WHERE status = ?",
            (JobStatus.PENDING.value, now, JobStatus.PROCESSING.value)
        )
        conn.commit()

        rows = conn.execute(
            "SELECT access_code FROM jobs WHERE status = ?",
            (JobStatus.PENDING.value,)
        ).fetchall()
        for row in rows:
            self._queue.put(row[0])
        logger.info(f"Loaded {len(rows)} pending jobs from database")

    def _cleanup_orphan_processes(self):
        """Kill subprocesses orphaned by a server crash / restart.

        Iterates all running processes and terminates any whose command-line
        references the output directory of a job that is no longer in a
        running/processing state.  This prevents resource leaks where
        ``gen_audio.py``, ``gen_video.py``, ffmpeg, etc. keep consuming
        CPU / GPU / memory after their parent died or the job was marked
        failed before the subprocess finished.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT DISTINCT output_dir FROM jobs WHERE status NOT IN (?, ?) AND output_dir IS NOT NULL",
            (JobStatus.PROCESSING.value, JobStatus.PENDING.value)
        ).fetchall()
        dirs = {r["output_dir"] for r in rows}
        if not dirs:
            return

        # Also match common subpaths that subprocesses write to
        extra_paths = set()
        for d in dirs:
            extra_paths.add(os.path.join(d, "audio_tracks"))
            extra_paths.add(os.path.join(d, "frames"))
        dirs |= extra_paths

        killed_pids = []
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmdline = " ".join(proc.info["cmdline"] or [])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            for od in dirs:
                if od in cmdline:
                    try:
                        proc.terminate()
                        killed_pids.append(proc)
                        logger.info(
                            "Orphan %d (%s) terminated (matches %s)",
                            proc.info["pid"],
                            os.path.basename(cmdline.split()[0] if cmdline else "?"),
                            od,
                        )
                    except psutil.NoSuchProcess:
                        pass
                    except Exception:
                        try:
                            proc.kill()
                            killed_pids.append(proc)
                        except Exception:
                            pass
                    break

        if killed_pids:
            # Give survivors 3 s, then SIGKILL the rest
            _, alive = psutil.wait_procs(
                killed_pids,
                timeout=3,
                callback=lambda p: logger.info("Orphan %d gracefully exited", p.pid),
            )
            for p in alive:
                try:
                    p.kill()
                    logger.warning("Orphan %d force-killed", p.pid)
                except Exception:
                    pass
            _safe_close_psutil_procs(killed_pids)
            logger.info(
                "Cleaned up %d orphan subprocess(es) from previous server instance", len(killed_pids)
            )

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
            (video_number, user_id, "_run_video_ning_ocr_job", JobStatus.FAILED.value)
        ).fetchone()
        if row:
            return row[0], row[1]
        return None

    def _find_failed_ning_job(self, video_number: str, user_id: int) -> tuple[str, str] | None:
        """Return (access_code, output_dir) of a failed ning SRT-translate job for the same video+user, if any."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT access_code, output_dir FROM jobs WHERE video_number = ? AND user_id = ? AND run_func_name = ? AND status = ? ORDER BY created_at DESC LIMIT 1",
            (video_number, user_id, "_run_video_job", JobStatus.FAILED.value)
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
        conn.execute("""
            INSERT OR REPLACE INTO jobs (access_code, srt_path, output_dir, temperature, status, error, run_func_name, video_number, video_file, user_id, text, blur, target_language, cfg_weight, exaggeration, start_trim, end_trim, cached_path, filename, checkpoint, created_at, status_changed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
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
        ))
        conn.commit()

        self._queue.put(access_code)
        # Ensure the worker thread is alive (it may have died silently)
        self._ensure_worker()

        return access_code

    @staticmethod
    def _kill_processes_by_output_dir(output_dir: str, sig: int = signal.SIGTERM):
        """Kill all running processes whose cmdline contains *output_dir*.

        Uses ``psutil`` to iterate running processes — this works even when
        the original parent process (``multiprocessing.Process``) has already
        exited but its children/descendants (shell scripts, ffmpeg, etc.) are
        still alive as orphans.
        """
        if not output_dir:
            return
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmdline = " ".join(proc.info["cmdline"] or [])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            if output_dir in cmdline:
                try:
                    os.kill(proc.info["pid"], sig)
                except (ProcessLookupError, PermissionError):
                    pass

    def _kill_process_group(self, proc: multiprocessing.Process, output_dir: str | None = None):
        """Kill *proc* and its entire process group, with a psutil-based
        fallback when the child process PID is no longer valid.

        After ``os.fork()`` the child calls ``os.setpgid(os.getpid(), os.getpid())``
        so that grandchildren (shell scripts, gen_audio.py, ffmpeg, etc.) share
        the same PGID.  When the child exits, ``os.getpgid(proc.pid)`` raises
        ``ProcessLookupError`` even though the PGID is still active.  In that
        case we fall back to scanning ``/proc`` via ``psutil``.
        """
        import signal as _sig

        # Try process-group kill first
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, _sig.SIGTERM)
            pgid_killed = True
        except (ProcessLookupError, PermissionError, AttributeError):
            pgid_killed = False

        # Fall back to psutil-based kill when PGID lookup failed.
        # Use self._kill_processes_by_output_dir (not JobQueue._kill…)
        # because the @singleton decorator replaces the class with a
        # function, so JobQueue._kill_processes_by_output_dir would be
        # an AttributeError on a function object.
        if not pgid_killed and output_dir:
            self._kill_processes_by_output_dir(output_dir, _sig.SIGTERM)

        proc.join(timeout=3)
        if proc.is_alive():
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, _sig.SIGKILL)
                pgid_killed = True
            except (ProcessLookupError, PermissionError, AttributeError):
                pgid_killed = False
            if not pgid_killed and output_dir:
                self._kill_processes_by_output_dir(output_dir, _sig.SIGKILL)
            proc.join(timeout=2)

        # Release OS resources (pipe fd, process-table entry).
        # Must be called after every proc.join() regardless of outcome.
        _safe_close_proc(proc)

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
            logger.warning(
                "Worker thread heartbeat stale (%.1fs since last pulse)", elapsed
            )
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
                target=self._process_queue, daemon=True,
                name="jobqueue-worker",
            )
            self._worker_thread.start()

    def _try_mark_failed(self, access_code, error_msg):
        """Best-effort mark job as FAILED, reconnecting if needed.

        Called from error-recovery paths where the worker's DB connection
        may be in a broken state (e.g. after a GPU hang destabilises the
        ROCm driver).  Failures here are logged but never re-raised, so a
        stuck ``"processing"`` row is the worst-case outcome and an operator
        can reset it manually.
        """
        now = _now_str()
        for attempt in range(2):
            try:
                conn = self._get_conn()
                conn.execute(
                    "UPDATE jobs SET status = ?, error = ?, failed_at = ?, status_changed_at = ? WHERE access_code = ?",
                    (JobStatus.FAILED.value, error_msg, now, now, access_code)
                )
                conn.commit()
                publish_job_status(access_code, JobStatus.FAILED.value, error=error_msg)
                return
            except Exception:
                if attempt == 0:
                    self._conn.close()
                else:
                    logger.critical(
                        "Cannot update job %s to FAILED — may stay stuck at processing",
                        access_code, exc_info=True,
                    )

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
            except Exception:
                continue
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
        conn = self._get_conn()

        row = conn.execute(
            f"SELECT {', '.join(JOB_COLUMNS)} FROM jobs WHERE access_code = ?", (access_code,)
        ).fetchone()

        if not row:
            logger.warning(f"Job {access_code} not found in database")
            return

        job = dict(zip(JOB_COLUMNS, row))

        run_func_name = job.get("run_func_name")
        try:
            run_func = _get_run_func(run_func_name) if run_func_name else None
        except Exception as e:
            logger.error(f"Job {access_code} failed to load handler '{run_func_name}': {e}")
            now = _now_str()
            conn.execute(
                "UPDATE jobs SET status = ?, error = ?, status_changed_at = ? WHERE access_code = ?",
                (JobStatus.FAILED.value, f"Handler import failed: {e}", now, access_code)
            )
            conn.commit()
            return

        if not run_func:
            logger.error(f"Job {access_code} has no run_func")
            now = _now_str()
            conn.execute(
                "UPDATE jobs SET status = ?, error = ?, status_changed_at = ? WHERE access_code = ?",
                (JobStatus.FAILED.value, "Job handler not found", now, access_code)
            )
            conn.commit()
            return

        now = _now_str()
        cursor = conn.execute(
            "UPDATE jobs SET status = ?, status_changed_at = ? WHERE access_code = ? AND status = ?",
            (JobStatus.PROCESSING.value, now, access_code, JobStatus.PENDING.value)
        )
        conn.commit()

        if cursor.rowcount == 0:
            logger.info(f"Job {access_code} already claimed by another worker, skipping")
            return

        publish_job_status(access_code, JobStatus.PROCESSING.value)

        job.pop("run_func_name", None)
        job.pop("created_at", None)

        # Run the job in a child process so we can detect cancellation
        # and abort without blocking the queue worker.
        # Non-daemon so the child survives parent process exit (e.g. gunicorn
        # worker recycle on SIGHUP).  The child creates its own process group
        # in _job_process_wrapper so it is immune to parent signal propagation.
        # Cancellation still works via _kill_process_group (os.killpg).
        proc = multiprocessing.Process(
            target=_job_process_wrapper,
            args=(job, run_func_name),
            daemon=False,
        )
        proc.start()

        cancelled = False
        shutdown_deadline = None
        try:
            while proc.is_alive():
                proc.join(timeout=5)
                self._heartbeat_ts = time.monotonic()

                # Check for graceful shutdown — set a deadline for the child
                if self._graceful_shutdown and shutdown_deadline is None:
                    shutdown_deadline = time.monotonic() + self._shutdown_timeout
                    logger.info(
                        "Job %s: graceful shutdown in progress — child has %ds to finish",
                        access_code, self._shutdown_timeout,
                    )

                if shutdown_deadline is not None and time.monotonic() >= shutdown_deadline:
                    logger.warning(
                        "Job %s: shutdown deadline exceeded — terminating child",
                        access_code,
                    )
                    cancelled = True

                if self._cancel_event.is_set():
                    cancelled = True
                else:
                    # Poll the DB — the job may have been cancelled externally
                    # (e.g. via TUI or another server instance).
                    try:
                        row = conn.execute(
                            "SELECT status FROM jobs WHERE access_code = ?",
                            (access_code,)
                        ).fetchone()
                        if row is None or row["status"] != JobStatus.PROCESSING.value:
                            cancelled = True
                    except Exception:
                        pass

                if cancelled:
                    # Kill the entire process group so subprocesses
                    # (ffmpeg, rapid_videocr, whisper-cli, etc.) don't linger.
                    output_dir = job.get("output_dir")
                    self._kill_process_group(proc, output_dir)
                    break

            if cancelled:
                if shutdown_deadline is not None:
                    # Graceful shutdown timeout — mark as PENDING so it resumes on restart
                    now = _now_str()
                    conn.execute(
                        "UPDATE jobs SET status = ?, error = NULL, status_changed_at = ? WHERE access_code = ?",
                        (JobStatus.PENDING.value, now, access_code)
                    )
                    conn.commit()
                    # Re-queue so it's picked up on next startup
                    self._queue.put(access_code)
                    publish_job_status(access_code, JobStatus.PENDING.value)
                    logger.info(f"Job {access_code} interrupted by shutdown — marked PENDING for resume")
                else:
                    # User cancellation
                    now = _now_str()
                    conn.execute(
                        "UPDATE jobs SET status = ?, error = ?, cancelled_at = ?, status_changed_at = ? WHERE access_code = ?",
                        (JobStatus.CANCELLED.value, "Cancelled by user", now, now, access_code)
                    )
                    conn.commit()
                    publish_job_status(access_code, JobStatus.CANCELLED.value)
                    logger.info(f"Job {access_code} was cancelled")
            else:
                # Save exitcode immediately — multiprocessing.Process auto-closes
                # after child exit, making later .exitcode / .is_alive() calls fail.
                exitcode = proc.exitcode
                if exitcode == 0:
                    now = _now_str()
                    conn.execute(
                        "UPDATE jobs SET status = ?, status_changed_at = ?, completed_at = ? WHERE access_code = ?",
                        (JobStatus.COMPLETED.value, now, now, access_code)
                    )
                    conn.commit()
                    publish_job_status(access_code, JobStatus.COMPLETED.value)
                    logger.info(f"Job {access_code} completed successfully")
                else:
                    # Kill the process group — grandchild subprocesses may linger
                    output_dir = job.get("output_dir")
                    self._kill_process_group(proc, output_dir)

                    sig = f" (exit {exitcode})" if exitcode is not None else ""
                    reason = _extract_failure_reason(output_dir)
                    if reason:
                        error_msg = f"{reason}{sig}"
                    else:
                        error_msg = f"Job process failed{sig}"
                    self._try_mark_failed(access_code, error_msg)
                    logger.warning(f"Job {access_code} failed with exit code {exitcode}")
        except Exception as e:
            # Ensure the child process is cleaned up on unexpected errors
            if proc.is_alive():
                output_dir = job.get("output_dir")
                self._kill_process_group(proc, output_dir)
            self._try_mark_failed(access_code, str(e)[:500])
            logger.error(f"Job {access_code} handler error: {e}")
        finally:
            # Release OS resources — without this the child process remains
            # as a zombie on POSIX, leaking a pipe fd and a process-table entry.
            _safe_close_proc(proc)

        # ── GPU state reset between jobs ─────────────────────────
        # After any job finishes (success, failure, or cancellation),
        # reset the ROCm/HIP GPU driver state so the next job doesn't
        # inherit a degraded GPU from a long-running predecessor.
        # Cooldown prevents repeatedly probing a hung GPU.
        _reset_gpu_state_cooldown(self)

        conn.commit()

    def set_checkpoint(self, access_code: str, checkpoint: str):
        """Record that a job has completed up to a certain step."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE jobs SET checkpoint = ? WHERE access_code = ?",
            (checkpoint, access_code)
        )
        conn.commit()

    def clear_checkpoint_for_file(self, access_code: str, file_path: str):
        """Remove checkpoint steps whose output file was deleted.

        When a user deletes a file from the result page, the corresponding
        checkpoint step is cleared so the step will re-run on resubmit.
        """
        ckpt = self.get_checkpoint(access_code)
        if not ckpt:
            return
        parts = [s for s in ckpt.split(",") if s]
        if not parts:
            return

        basename = os.path.basename(file_path)
        steps_to_clear = set()

        # Exact-match lookups shared with chatterbox_server.py
        step = FILENAME_TO_CHECKPOINT_STEP.get(basename)
        if step:
            steps_to_clear.add(step)

        # Pattern-based lookups for job-specific file types
        if basename == "output_modified.mp4":
            steps_to_clear.add("video")
        elif "_decompressed.mov" in basename:
            steps_to_clear.add("decompress")
        elif basename.endswith("_trimmed.mp4"):
            steps_to_clear.add("trim")
        elif basename.endswith(".mp4") and basename not in ("output_modified.mp4",):
            steps_to_clear.add("download")
        elif "audio" in file_path.replace("\\", "/").split("/"):
            steps_to_clear.add("audio")

        if not steps_to_clear:
            return

        new_parts = [p for p in parts if p not in steps_to_clear]
        if new_parts != parts:
            self.set_checkpoint(access_code, ",".join(new_parts))

    def invalidate_checkpoints_after(self, access_code: str, step: str):
        """Remove all checkpoint steps *after* *step*, keeping *step* intact.

        Also deletes the output artifacts of those steps so the job
        can cleanly re-generate them.

        The checkpoint order depends on the job type.  Built-in ordering:

            download < decompress < trim < extract_audio < whisper < ocr < translate < audio < video

        If *step* is not found, nothing changes.  This is called when a user
        edits a file belonging to *step* and saves it — everything after that
        step must be re-run, but the edited step itself is preserved since
        the new content is already saved.
        """
        ORDER = CHECKPOINT_ORDER
        ckpt = self.get_checkpoint(access_code)
        if not ckpt:
            logger.info("invalidate_checkpoints_after(%s, %s): no checkpoint", access_code, step)
        parts = [s for s in (ckpt or "").split(",") if s]

        try:
            idx = ORDER.index(step)
        except ValueError:
            new_parts = [p for p in parts if p != step]
        else:
            new_parts = [p for p in parts if p not in ORDER[idx+1:]]

        removed_from_ckpt = set(parts) - set(new_parts)
        # Steps after the edited step whose artifacts must be purged
        steps_to_regen = ORDER[idx+1:] if step in ORDER else []

        logger.info("invalidate_checkpoints_after(%s, %s): parts=%s, new=%s, removed_from_ckpt=%s, steps_to_regen=%s",
                     access_code, step, parts, new_parts, removed_from_ckpt, steps_to_regen)

        if new_parts != parts:
            self.set_checkpoint(access_code, ",".join(new_parts))

        # Purge artifacts for steps that will re-run
        if steps_to_regen:
            conn = self._get_conn()
            row = conn.execute("SELECT output_dir FROM jobs WHERE access_code = ?", (access_code,)).fetchone()
            output_dir = row[0] if row else None
            if output_dir and os.path.isdir(output_dir):
                self._purge_step_artifacts(output_dir, set(steps_to_regen))

    _STEP_ARTIFACTS = {
        "translate":  ["translated.srt"],
        "audio":      [
            "audio/output.wav",
            "audio/output_adjusted.srt",
            "audio/output-final-modified.srt",
            "audio/changed_segments.json",
            "audio/job.log",
            "audio_tracks/output.wav",
            "audio_tracks/output_adjusted.srt",
            "audio_tracks/output-final-modified.srt",
            "audio_tracks/changed_segments.json",
            "audio_tracks/job.log",
        ],
        "video":      ["output_modified.mp4", "output_final.mp4"],
    }

    def _purge_step_artifacts(self, output_dir: str, steps: set[str]):
        """Delete output files produced by the given checkpoint steps.

        For the audio step, only the final output files are removed;
        the ``tmp/`` subdirectory (holding per-segment cached wavs and
        meta JSONs) is preserved so unchanged segments can skip re-generation.
        """
        import shutil
        for step in steps:
            for rel in self._STEP_ARTIFACTS.get(step, []):
                path = os.path.join(output_dir, rel)
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                        logger.info("Purged directory: %s", path)
                    elif os.path.isfile(path):
                        os.remove(path)
                        logger.info("Purged file: %s", path)
                except Exception as e:
                    logger.warning("Failed to purge %s: %s", path, e)

    def set_checkpoint_edited(self, access_code: str, edited: bool = True):
        """Mark that the checkpoint has been edited (user edited an SRT)."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE jobs SET checkpoint_edited = ? WHERE access_code = ?",
            (1 if edited else 0, access_code)
        )
        conn.commit()

    def get_checkpoint_edited(self, access_code: str) -> bool:
        """Return True if the user has edited a checkpoint-level file."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT checkpoint_edited FROM jobs WHERE access_code = ?", (access_code,)
        ).fetchone()
        return bool(row and row[0])

    def set_edited_srt_file(self, access_code: str, filename: str):
        """Record that a specific SRT file has been edited by the user."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT edited_srt_files FROM jobs WHERE access_code = ?", (access_code,)
        ).fetchone()
        existing = row[0] if row and row[0] else ""
        files = set(f for f in existing.split(",") if f)
        files.add(filename)
        new_val = ",".join(sorted(files))
        conn.execute(
            "UPDATE jobs SET edited_srt_files = ? WHERE access_code = ?",
            (new_val, access_code)
        )
        conn.commit()

    def clear_edited_srt_files(self, access_code: str):
        """Clear all recorded edited SRT files (called on resubmit)."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE jobs SET edited_srt_files = '' WHERE access_code = ?",
            (access_code,)
        )
        conn.commit()

    def get_edited_srt_files(self, access_code: str) -> list[str]:
        """Return the list of edited SRT filenames for a job."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT edited_srt_files FROM jobs WHERE access_code = ?", (access_code,)
        ).fetchone()
        if row and row[0]:
            return [f for f in row[0].split(",") if f]
        return []

    def get_checkpoint(self, access_code: str) -> str:
        """Return the highest completed checkpoint step for a job."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT checkpoint FROM jobs WHERE access_code = ?", (access_code,)
        ).fetchone()
        return row[0] if row and row[0] else ""

    def update_job_progress(self, access_code: str, progress: str):
        conn = self._get_conn()
        conn.execute(
            "UPDATE jobs SET progress = ? WHERE access_code = ?",
            (progress, access_code)
        )
        conn.commit()
        publish_job_status(access_code, "progress", progress=progress)

    def update_target_language(self, access_code: str, lang: str):
        """Update the target_language field for a job (e.g. after language detection)."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE jobs SET target_language = ? WHERE access_code = ?",
            (lang, access_code)
        )
        conn.commit()

    def get_status(self, access_code: str) -> dict | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT access_code, status, error, output_dir, progress, target_language, created_at, temperature, cfg_weight, exaggeration, checkpoint, checkpoint_edited, edited_srt_files FROM jobs WHERE access_code = ?",
            (access_code,)
        ).fetchone()

        if not row:
            return None

        # Count how many pending jobs are ahead of this one in the queue
        queue_position = None
        if row["status"] == JobStatus.PENDING.value and row["created_at"]:
            queue_position = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = ? AND created_at < ?",
                (JobStatus.PENDING.value, row["created_at"])
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
        rows = conn.execute("""
            SELECT access_code, run_func_name, status, error, output_dir, created_at, status_changed_at
            FROM jobs WHERE user_id = ? AND status != ?
            ORDER BY COALESCE(status_changed_at, created_at) DESC
        """, (user_id, JobStatus.DELETED.value)).fetchall()
        type_map = _JOB_TYPE_LABELS
        return [{
            "access_code": r[0],
            "type": type_map.get(r[1], r[1] or "未知"),
            "status": r[2],
            "error": r[3],
            "output_dir": r[4],
            "created_at": r[5],
            "status_changed_at": r[6],
        } for r in rows]

    def cancel_job(self, access_code: str) -> dict:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT status, output_dir FROM jobs WHERE access_code = ?", (access_code,)
        ).fetchone()

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
            for proc in psutil.process_iter(['pid', 'cmdline']):
                try:
                    if output_dir in ' '.join(proc.info['cmdline'] or []):
                        # Kill the entire process group so children don't orphan
                        try:
                            pgid = os.getpgid(proc.info['pid'])
                            os.killpg(pgid, signal.SIGTERM)
                        except (ProcessLookupError, PermissionError, AttributeError):
                            proc.send_signal(signal.SIGTERM)
                        killed_pids.add(proc.info['pid'])
                        killed_procs.append(proc)
                except Exception:
                    pass
            if killed_pids:
                _, alive = psutil.wait_procs(
                    [p for p in psutil.process_iter(['pid']) if p.info['pid'] in killed_pids],
                    timeout=3,
                )
                for p in alive:
                    try:
                        pgid = os.getpgid(p.pid)
                        os.killpg(pgid, signal.SIGKILL)
                    except Exception:
                        try:
                            p.kill()
                        except Exception:
                            pass
                _safe_close_psutil_procs(killed_procs)

        # Signal the worker thread to skip completion logic
        with self._cancel_lock:
            if access_code == self._current_access_code:
                self._cancel_event.set()

        now = _now_str()
        conn.execute(
            "UPDATE jobs SET status = ?, error = ?, cancelled_at = ?, status_changed_at = ? WHERE access_code = ? AND (status = ? OR status = ?)",
            (JobStatus.CANCELLED.value, "Cancelled by user", now, now, access_code, JobStatus.PENDING.value, JobStatus.PROCESSING.value)
        )
        conn.commit()
        publish_job_status(access_code, JobStatus.CANCELLED.value)

        return {"success": True, "message": "Job cancelled"}

    def resubmit_job(self, access_code: str) -> dict:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT status FROM jobs WHERE access_code = ?", (access_code,)
        ).fetchone()

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
                return {"success": False, "error": "Job is completed, only failed, cancelled or checkpoint-edited jobs can be resubmitted"}
        elif row[0] not in (JobStatus.FAILED.value, JobStatus.CANCELLED.value):
            return {"success": False, "error": f"Job is {row[0]}, only failed, cancelled or checkpoint-edited jobs can be resubmitted"}

        now = _now_str()
        conn.execute(
            "UPDATE jobs SET status = ?, error = NULL, status_changed_at = ? WHERE access_code = ?",
            (JobStatus.PENDING.value, now, access_code)
        )
        conn.commit()

        self._queue.put(access_code)
        # Ensure the worker thread is alive (it may have died silently)
        self._ensure_worker()

        return {"success": True, "message": "Job resubmitted"}

    def delete_job(self, access_code: str) -> dict:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT status FROM jobs WHERE access_code = ?", (access_code,)
        ).fetchone()

        if not row:
            return {"success": False, "error": "Job not found"}

        # Mark as deleted without removing files or DB row
        now = _now_str()
        conn.execute(
            "UPDATE jobs SET status = ?, error = ?, deleted_at = ?, status_changed_at = ? WHERE access_code = ?",
            (JobStatus.DELETED.value, "Deleted by user", now, now, access_code)
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
            "SELECT access_code, output_dir FROM jobs WHERE status = ?",
            (JobStatus.DELETED.value,)
        ).fetchall()

        if not rows:
            return {"success": True, "message": "No deleted jobs found", "jobs_removed": 0, "dirs_removed": 0}

        if dry_run:
            dirs_found = sum(1 for r in rows if r["output_dir"] and os.path.exists(r["output_dir"]))
            return {"success": True, "message": f"Would remove {len(rows)} jobs and {dirs_found} directories",
                    "jobs_removed": len(rows), "dirs_removed": dirs_found}

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
            cursor = conn.execute(
                "DELETE FROM jobs WHERE status = ?",
                (JobStatus.DELETED.value,)
            )
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
