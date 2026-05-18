import logging
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


class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"


def _get_run_func(name: str) -> Optional[Callable]:
    from srt_action import _run_gen_audio, _run_video_job, _run_video_custom_job, _run_tts_job, _run_video_auto_job, _run_audio_segmentation_job
    if name == "_run_gen_audio":
        return _run_gen_audio
    if name == "_run_video_job":
        return _run_video_job
    if name == "_run_video_custom_job":
        return _run_video_custom_job
    if name == "_run_tts_job":
        return _run_tts_job
    if name == "_run_video_auto_job":
        return _run_video_auto_job
    if name == "_run_audio_segmentation_job":
        return _run_audio_segmentation_job
    return None


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
        conn.execute(
            "UPDATE jobs SET status = ? WHERE status = ?",
            (JobStatus.PENDING.value, JobStatus.PROCESSING.value)
        )
        conn.commit()

        rows = conn.execute(
            "SELECT access_code FROM jobs WHERE status = ?",
            (JobStatus.PENDING.value,)
        ).fetchall()
        for row in rows:
            self._queue.put(row[0])
        logger.info(f"Loaded {len(rows)} pending jobs from database")

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
            conn.execute("ALTER TABLE jobs ADD COLUMN deleted_at TIMESTAMP")
        except Exception:
            pass
        conn.commit()

    def _generate_access_code(self) -> str:
        return str(uuid.uuid4())[:8].upper()

    def add_job(self, job_data: dict, run_func: Callable[[dict], None], user_id: int = None) -> str:
        conn = self._get_conn()
        access_code = self._generate_access_code()

        run_func_name = None
        if run_func.__name__ == "_run_gen_audio":
            run_func_name = "_run_gen_audio"
        elif run_func.__name__ == "_run_video_job":
            run_func_name = "_run_video_job"
        elif run_func.__name__ == "_run_video_custom_job":
            run_func_name = "_run_video_custom_job"
        elif run_func.__name__ == "_run_tts_job":
            run_func_name = "_run_tts_job"
        elif run_func.__name__ == "_run_video_auto_job":
            run_func_name = "_run_video_auto_job"
        elif run_func.__name__ == "_run_audio_segmentation_job":
            run_func_name = "_run_audio_segmentation_job"

        conn.execute("""
            INSERT INTO jobs (access_code, srt_path, output_dir, temperature, status, error, run_func_name, video_number, video_file, user_id, text, blur, target_language, cfg_weight, exaggeration)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        ))
        conn.commit()

        self._queue.put(access_code)

        return access_code

    def _ensure_worker(self):
        if not self._running:
            self._running = True
            self._worker_thread = threading.Thread(
                target=self._process_queue, daemon=True
            )
            self._worker_thread.start()

    def _process_queue(self):
        while self._running:
            try:
                access_code = self._queue.get(timeout=1)
                self._current_access_code = access_code
                self._cancel_event.clear()
                self._process_job(access_code)
                self._current_access_code = None
            except Exception:
                continue

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
            conn.execute(
                "UPDATE jobs SET status = ?, error = ? WHERE access_code = ?",
                (JobStatus.FAILED.value, "Job handler not found", access_code)
            )
            conn.commit()
            return

        cursor = conn.execute(
            "UPDATE jobs SET status = ? WHERE access_code = ? AND status = ?",
            (JobStatus.PROCESSING.value, access_code, JobStatus.PENDING.value)
        )
        conn.commit()

        if cursor.rowcount == 0:
            logger.info(f"Job {access_code} already claimed by another worker, skipping")
            return

        try:
            job.pop("run_func_name", None)
            job.pop("created_at", None)
            run_func(job)
            # After run_func returns, check if we were cancelled mid-execution
            if self._cancel_event.is_set():
                logger.info(f"Job {access_code} was cancelled, marking as failed")
                now = time.strftime('%Y-%m-%d %H:%M:%S')
                conn.execute(
                    "UPDATE jobs SET status = ?, error = ? WHERE access_code = ?",
                    (JobStatus.FAILED.value, "Cancelled by user", access_code)
                )
                conn.execute(
                    "UPDATE jobs SET cancelled_at = ? WHERE access_code = ?",
                    (now, access_code)
                )
            else:
                conn.execute(
                    "UPDATE jobs SET status = ? WHERE access_code = ?",
                    (JobStatus.COMPLETED.value, access_code)
                )
                now = time.strftime('%Y-%m-%d %H:%M:%S')
                conn.execute(
                    "UPDATE jobs SET completed_at = ? WHERE access_code = ?",
                    (now, access_code)
                )
                logger.info(f"Job {access_code} completed successfully")
        except Exception as e:
            now = time.strftime('%Y-%m-%d %H:%M:%S')
            conn.execute(
                "UPDATE jobs SET status = ?, error = ?, failed_at = ? WHERE access_code = ?",
                (JobStatus.FAILED.value, str(e)[:500], now, access_code)
            )
            logger.error(f"Job {access_code} failed: {e}")

        conn.commit()

    def update_job_progress(self, access_code: str, progress: str):
        conn = self._get_conn()
        conn.execute(
            "UPDATE jobs SET progress = ? WHERE access_code = ?",
            (progress, access_code)
        )
        conn.commit()

    def get_status(self, access_code: str) -> Optional[dict]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT access_code, status, error, output_dir, progress, target_language, created_at, temperature, cfg_weight, exaggeration FROM jobs WHERE access_code = ?",
            (access_code,)
        ).fetchone()

        if not row:
            return None

        # Count how many pending jobs are ahead of this one in the queue
        queue_position = None
        created_at = row[6] if len(row) > 6 else None
        if row[1] == JobStatus.PENDING.value and created_at:
            queue_position = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = ? AND created_at < ?",
                (JobStatus.PENDING.value, created_at)
            ).fetchone()[0]

        return {
            "access_code": row[0],
            "status": row[1],
            "error": row[2],
            "output_dir": row[3],
            "progress": row[4],
            "target_language": row[5] if len(row) > 5 else None,
            "queue_position": queue_position,
            "temperature": row[7] if len(row) > 7 else None,
            "cfg_weight": row[8] if len(row) > 8 else None,
            "exaggeration": row[9] if len(row) > 9 else None,
        }

    def get_user_jobs(self, user_id: int) -> list:
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT access_code, run_func_name, status, error, output_dir, created_at
            FROM jobs WHERE user_id = ? AND status != ?
            ORDER BY created_at DESC
        """, (user_id, JobStatus.DELETED.value)).fetchall()
        type_map = {
            "_run_gen_audio": "音频生成",
            "_run_video_job": "宁视频翻译",
            "_run_video_custom_job": "自定义视频",
            "_run_tts_job": "语音合成",
            "_run_video_auto_job": "自动翻译视频",
            "_run_audio_segmentation_job": "音频分段合成",
        }
        return [{
            "access_code": r[0],
            "type": type_map.get(r[1], r[1] or "未知"),
            "status": r[2],
            "error": r[3],
            "output_dir": r[4],
            "created_at": r[5],
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
            for proc in psutil.process_iter(['pid', 'cmdline']):
                try:
                    if output_dir in ' '.join(proc.info['cmdline'] or []):
                        proc.send_signal(signal.SIGTERM)
                except Exception:
                    pass

        # Signal the worker thread to skip completion logic
        if access_code == self._current_access_code:
            self._cancel_event.set()

        now = time.strftime('%Y-%m-%d %H:%M:%S')
        conn.execute(
            "UPDATE jobs SET status = ?, error = ?, cancelled_at = ? WHERE access_code = ? AND (status = ? OR status = ?)",
            (JobStatus.FAILED.value, "Cancelled by user", now, access_code, JobStatus.PENDING.value, JobStatus.PROCESSING.value)
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

        if row[0] != JobStatus.FAILED.value:
            return {"success": False, "error": f"Job is {row[0]}, only failed jobs can be resubmitted"}

        conn.execute(
            "UPDATE jobs SET status = ?, error = NULL WHERE access_code = ?",
            (JobStatus.PENDING.value, access_code)
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
        conn.execute(
            "UPDATE jobs SET status = ?, error = ? WHERE access_code = ?",
            (JobStatus.DELETED.value, "Deleted by user", access_code)
        )
        conn.commit()

        return {"success": True, "message": "Job hidden"}

    def stop(self):
        self._running = False


def get_job_queue() -> JobQueue:
    return JobQueue()
