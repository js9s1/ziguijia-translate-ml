import logging
import multiprocessing
import os
import shutil
import signal
import sqlite3
import threading
import time
import uuid
from enum import Enum
from queue import Queue
from typing import Callable, Optional

import psutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "jobs.db")

logger = logging.getLogger(__name__)

_now_str = lambda: time.strftime('%Y-%m-%d %H:%M:%S')


_JOB_HANDLERS: dict[str, Callable] = {}


class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DELETED = "deleted"


def _register_handlers():
    """Lazily populate the job handler registry on first import."""
    from audio_job import _run_gen_audio, _run_audio_segmentation_job
    from tts_job import _run_tts_job
    from video_ning_job import _run_video_job, _run_video_ning_ocr_job
    from video_custom_job import _run_video_custom_job, _run_video_auto_job, _run_video_ocr_job
    from video_ocr_job import _run_ocr_only_job
    _JOB_HANDLERS.update({
        "_run_gen_audio": _run_gen_audio,
        "_run_video_job": _run_video_job,
        "_run_video_custom_job": _run_video_custom_job,
        "_run_tts_job": _run_tts_job,
        "_run_video_auto_job": _run_video_auto_job,
        "_run_audio_segmentation_job": _run_audio_segmentation_job,
        "_run_video_ocr_job": _run_video_ocr_job,
        "_run_video_ning_ocr_job": _run_video_ning_ocr_job,
        "_run_ocr_only_job": _run_ocr_only_job,
    })


_JOB_TYPE_LABELS: dict[str, str] = {
    "_run_gen_audio": "音频生成",
    "_run_video_job": "宁视频翻译",
    "_run_video_custom_job": "自定义视频",
    "_run_tts_job": "语音合成",
    "_run_video_auto_job": "自动翻译视频",
    "_run_audio_segmentation_job": "音频分段合成",
    "_run_video_ocr_job": "OCR翻译视频",
    "_run_video_ning_ocr_job": "宁视频OCR翻译",
    "_run_ocr_only_job": "视频OCR提取字幕",
}


def _get_run_func(name: str) -> Optional[Callable]:
    if not _JOB_HANDLERS:
        _register_handlers()
    return _JOB_HANDLERS.get(name)


def _get_job_type_label(run_func_name: str) -> str:
    return _JOB_TYPE_LABELS.get(run_func_name, run_func_name or "未知")


def _job_process_wrapper(job_data: dict, run_func_name: str):
    """Entry point for a child process executing a single job.

    Cleans up state inherited from the parent via fork (DB connection),
    then runs the job handler. Exits with 0 on success, non-zero on failure.

    Creates its own process group so that all subprocesses spawned by the
    job handler can be killed as a group on cancellation.
    """
    import os
    import sys

    # Create a new process group so the parent can kill the whole tree.
    os.setpgid(os.getpid(), os.getpid())

    # Close any DB connection inherited from the parent process.
    # The child opens its own connection when it needs one.
    jq = get_job_queue()
    jq._close_conn()

    run_func = _get_run_func(run_func_name)
    if run_func is None:
        sys.exit(1)
    run_func(job_data)


class JobQueue:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._local = threading.local()
        self._queue = Queue()
        self._worker_thread = None
        self._cancel_event = threading.Event()
        self._current_access_code: str | None = None
        self._running = False
        self._init_db()
        self._load_pending_jobs()
        self._ensure_worker()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(DB_FILE)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _load_pending_jobs(self):
        conn = self._get_conn()

        # Find any jobs that were still PROCESSING when the server died —
        # reset them to PENDING so they can be retried.
        now = _now_str()
        conn.execute(
            "UPDATE jobs SET status = ?, status_changed_at = ? WHERE status = ?",
            (JobStatus.PENDING.value, now, JobStatus.PROCESSING.value)
        )
        conn.commit()

        # Kill orphan subprocesses left behind by the dead server instance.
        self._cleanup_orphan_processes()

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
        references the output directory of a now-pending job.  This prevents
        resource leaks where ``gen_audio.py``, ``gen_video.py``, ffmpeg, etc.
        keep consuming CPU / GPU / memory after their parent died.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT DISTINCT output_dir FROM jobs WHERE status = ? AND output_dir IS NOT NULL",
            (JobStatus.PENDING.value,)
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
            logger.info(
                "Cleaned up %d orphan subprocess(es) from previous server instance", len(killed_pids)
            )

    def _init_db(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                access_code TEXT PRIMARY KEY,
                srt_path TEXT,
                output_dir TEXT,
                temperature REAL,
                status TEXT,
                error TEXT,
                run_func_name TEXT,
                video_number TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        for col in ("video_number", "video_file", "user_id", "text"):
            try:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} TEXT")
            except Exception:
                pass
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN progress TEXT")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN blur TEXT DEFAULT 'yes'")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN target_language TEXT DEFAULT 'en'")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN cfg_weight REAL DEFAULT 0.5")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN exaggeration REAL DEFAULT 0.5")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN completed_at TIMESTAMP")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN failed_at TIMESTAMP")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN cancelled_at TIMESTAMP")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN status_changed_at TIMESTAMP")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN deleted_at TIMESTAMP")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN checkpoint TEXT DEFAULT ''")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN checkpoint_edited INTEGER DEFAULT 0")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN edited_srt_files TEXT DEFAULT ''")
        except Exception:
            pass
        conn.commit()

    def _generate_access_code(self) -> str:
        return str(uuid.uuid4())[:8].upper()

    def _find_failed_ocr_job(self, video_number: str, user_id: int) -> Optional[tuple[str, str]]:
        """Return (access_code, output_dir) of a failed ning OCR job for the same video+user, if any."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT access_code, output_dir FROM jobs WHERE video_number = ? AND user_id = ? AND run_func_name = ? AND status = ? ORDER BY created_at DESC LIMIT 1",
            (video_number, user_id, "_run_video_ning_ocr_job", JobStatus.FAILED.value)
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

        conn.execute("""
            INSERT OR REPLACE INTO jobs (access_code, srt_path, output_dir, temperature, status, error, run_func_name, video_number, video_file, user_id, text, blur, target_language, cfg_weight, exaggeration, checkpoint, status_changed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            job_data.get("cfg_weight", 0.5),
            job_data.get("exaggeration", 0.5),
            prev_ckpt,
            _now_str(),
        ))
        conn.commit()

        self._queue.put(access_code)

        return access_code

    def _close_conn(self):
        """Close the current thread-local DB connection, if open.

        Used after ``os.fork()`` so the child process doesn't share
        the parent's connection — each process manages its own.
        """
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None

    def _ensure_worker(self):
        if not self._running:
            self._running = True
            self._worker_thread = threading.Thread(
                target=self._process_queue, daemon=True
            )
            self._worker_thread.start()

    def _process_queue(self):
        import queue as std_queue
        while self._running:
            try:
                access_code = self._queue.get(timeout=1)
            except std_queue.Empty:
                continue  # timeout, just re-check self._running
            except Exception:
                continue
            self._current_access_code = access_code
            self._cancel_event.clear()
            try:
                self._process_job(access_code)
            finally:
                self._current_access_code = None

    def _process_job(self, access_code: str):
        conn = self._get_conn()

        columns = ["access_code", "srt_path", "output_dir", "temperature", "status", "error", "run_func_name", "video_number", "created_at", "video_file", "user_id", "text", "blur", "target_language", "cfg_weight", "exaggeration"]
        row = conn.execute(
            f"SELECT {', '.join(columns)} FROM jobs WHERE access_code = ?", (access_code,)
        ).fetchone()

        if not row:
            logger.warning(f"Job {access_code} not found in database")
            return

        job = dict(zip(columns, row))

        run_func_name = job.get("run_func_name")
        run_func = _get_run_func(run_func_name) if run_func_name else None

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

        job.pop("run_func_name", None)
        job.pop("created_at", None)

        # Run the job in a child process so we can detect cancellation
        # and abort without blocking the queue worker.
        proc = multiprocessing.Process(
            target=_job_process_wrapper,
            args=(job, run_func_name),
            daemon=True,
        )
        proc.start()

        cancelled = False
        try:
            while proc.is_alive():
                proc.join(timeout=5)
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
                    import signal
                    try:
                        pgid = os.getpgid(proc.pid)
                        os.killpg(pgid, signal.SIGTERM)
                    except (ProcessLookupError, PermissionError, AttributeError):
                        pass
                    proc.join(timeout=3)
                    if proc.is_alive():
                        try:
                            pgid = os.getpgid(proc.pid)
                            os.killpg(pgid, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError, AttributeError):
                            pass
                        proc.join(timeout=2)
                    break

            if cancelled:
                now = _now_str()
                conn.execute(
                    "UPDATE jobs SET status = ?, error = ?, cancelled_at = ?, status_changed_at = ? WHERE access_code = ?",
                    (JobStatus.CANCELLED.value, "Cancelled by user", now, now, access_code)
                )
                logger.info(f"Job {access_code} was cancelled")
            elif proc.exitcode == 0:
                now = _now_str()
                conn.execute(
                    "UPDATE jobs SET status = ?, status_changed_at = ?, completed_at = ? WHERE access_code = ?",
                    (JobStatus.COMPLETED.value, now, now, access_code)
                )
                logger.info(f"Job {access_code} completed successfully")
            else:
                now = _now_str()
                sig = f" (exit {proc.exitcode})" if proc.exitcode is not None else ""
                conn.execute(
                    "UPDATE jobs SET status = ?, error = ?, failed_at = ?, status_changed_at = ? WHERE access_code = ?",
                    (JobStatus.FAILED.value, f"Job process failed{sig}", now, now, access_code)
                )
                logger.warning(f"Job {access_code} failed with exit code {proc.exitcode}")
        except Exception as e:
            # Ensure the child process is cleaned up on unexpected errors
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=3)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=2)
            now = _now_str()
            conn.execute(
                "UPDATE jobs SET status = ?, error = ?, failed_at = ?, status_changed_at = ? WHERE access_code = ?",
                (JobStatus.FAILED.value, str(e)[:500], now, now, access_code)
            )
            logger.error(f"Job {access_code} handler error: {e}")

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

        # Map deleted filenames back to checkpoint steps for ning OCR jobs
        if basename == "ocr_screen.srt":
            steps_to_clear.add("ocr")
        elif basename == "translated.srt":
            steps_to_clear.add("translate")
        elif basename == "output_modified.mp4":
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
        """Remove checkpoint *step* and all subsequent steps after a user edit.

        The checkpoint order depends on the job type.  Built-in ordering:

            download < decompress < trim < extract_audio < whisper < ocr < translate < audio < video

        If *step* is not found, nothing changes.  This is called when a user
        edits a file belonging to *step* and saves it — everything after that
        step must be re-run.
        """
        ORDER = ["download", "decompress", "trim", "extract_audio", "whisper", "ocr", "translate", "audio", "video"]
        ckpt = self.get_checkpoint(access_code)
        if not ckpt:
            return
        parts = [s for s in ckpt.split(",") if s]
        if not parts:
            return

        # If step isn't in our built-in order, fall back to removing step only
        try:
            idx = ORDER.index(step)
        except ValueError:
            # Not a standard step — remove just this one
            new_parts = [p for p in parts if p != step]
        else:
            new_parts = [p for p in parts if p not in ORDER[idx:]]

        if new_parts != parts:
            self.set_checkpoint(access_code, ",".join(new_parts))

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

    def update_target_language(self, access_code: str, lang: str):
        """Update the target_language field for a job (e.g. after language detection)."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE jobs SET target_language = ? WHERE access_code = ?",
            (lang, access_code)
        )
        conn.commit()

    def get_status(self, access_code: str) -> Optional[dict]:
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

        if status not in ("pending", "processing"):
            return {"success": False, "error": f"Job is already {status}, cannot cancel"}

        if output_dir:
            # Kill any process whose cmdline references this output dir,
            # including subprocesses spawned by the job handler.
            killed_pids = []
            for proc in psutil.process_iter(['pid', 'cmdline']):
                try:
                    if output_dir in ' '.join(proc.info['cmdline'] or []):
                        # Kill the entire process group so children don't orphan
                        try:
                            pgid = os.getpgid(proc.info['pid'])
                            os.killpg(pgid, signal.SIGTERM)
                        except (ProcessLookupError, PermissionError, AttributeError):
                            proc.send_signal(signal.SIGTERM)
                        killed_pids.append(proc.info['pid'])
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

        # Signal the worker thread to skip completion logic
        if access_code == self._current_access_code:
            self._cancel_event.set()

        now = _now_str()
        conn.execute(
            "UPDATE jobs SET status = ?, error = ?, cancelled_at = ?, status_changed_at = ? WHERE access_code = ? AND (status = ? OR status = ?)",
            (JobStatus.CANCELLED.value, "Cancelled by user", now, now, access_code, JobStatus.PENDING.value, JobStatus.PROCESSING.value)
        )
        conn.commit()

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

        # Allow resubmit for failed jobs, or completed jobs that have been
        # checkpoint-edited (user edited an SRT after completion).
        if row[0] == JobStatus.COMPLETED.value:
            # Must have checkpoint_edited flag set
            ckpt_row = conn.execute(
                "SELECT checkpoint_edited FROM jobs WHERE access_code = ?", (access_code,)
            ).fetchone()
            if not ckpt_row or not ckpt_row[0]:
                return {"success": False, "error": f"Job is completed, only failed or checkpoint-edited jobs can be resubmitted"}
        elif row[0] != JobStatus.FAILED.value:
            return {"success": False, "error": f"Job is {row[0]}, only failed or checkpoint-edited jobs can be resubmitted"}

        now = _now_str()
        conn.execute(
            "UPDATE jobs SET status = ?, error = NULL, status_changed_at = ? WHERE access_code = ?",
            (JobStatus.PENDING.value, now, access_code)
        )
        conn.commit()

        self._queue.put(access_code)

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
            "UPDATE jobs SET status = ?, error = ?, status_changed_at = ? WHERE access_code = ?",
            (JobStatus.DELETED.value, "Deleted by user", now, access_code)
        )
        conn.commit()

        return {"success": True, "message": "Job hidden"}

    def stop(self):
        self._running = False


def get_job_queue() -> JobQueue:
    return JobQueue()
