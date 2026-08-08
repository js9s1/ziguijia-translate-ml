"""Worker process management — subprocess launch, heartbeat monitoring, GPU reset.

Extracted from ``jobqueue.py`` to reduce that module's size (~1324 → ~1000 lines).
Functions that were previously methods on ``JobQueue`` take the instance as their
first parameter (``jq``) and access its attributes directly.
"""

import logging
import multiprocessing  # used by _safe_close_proc type annotation
import os
import sqlite3
import time

import psutil
from db_schema import JOB_COLUMNS
from job_types import JobStatus, _get_run_func
from valkey_util import publish_job_status

logger = logging.getLogger(__name__)

_WORKER_HEARTBEAT_STALE = 30
"""Seconds after which a worker thread is considered dead."""


# ── Helpers ──────────────────────────────────────────────────


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _safe_close_proc(proc: multiprocessing.Process | None):
    """Release OS resources held by a ``multiprocessing.Process``.

    ``multiprocessing.Process`` owns a pipe fd and a PID table entry.
    Without explicit ``close()`` the child process will remain as
    a zombie until the parent exits.
    """
    if proc is None:
        return
    try:
        proc.close()
    except (OSError, ValueError):
        pass


def _safe_close_psutil_procs(procs):
    """Close psutil.Process objects to release their cached /proc handles."""
    for p in procs:
        try:
            if isinstance(p, psutil.Process):
                p.cmdline.cache_clear() if hasattr(p.cmdline, "cache_clear") else None
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def check_job_valid(jq, access_code: str):
    """Raise RuntimeError if *access_code* is no longer PENDING or PROCESSING.

    Call this from within long-running job handlers (e.g. CPU-fallback
    TTS generation) so that a job marked FAILED or CANCELLED in the DB
    (by a watchdog restart duplicate, or by a user cancel) is aborted
    instead of continuing to burn CPU indefinitely.
    """
    conn = jq._get_conn()
    row = conn.execute("SELECT status FROM jobs WHERE access_code = ?", (access_code,)).fetchone()
    if not row:
        raise RuntimeError(f"Job {access_code} no longer exists in database")
    if row[0] not in (JobStatus.PENDING.value, JobStatus.PROCESSING.value):
        raise RuntimeError(f"Job {access_code} is {row[0]}, aborting")


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

        is_tb = (
            stripped.startswith("Traceback ")
            or stripped.startswith("  File ")
            or (in_traceback and not stripped.startswith("---"))
        )
        if is_tb:
            in_traceback = True
            continue

        in_traceback = False
        if (
            "Error" in stripped
            or "error" in stripped
            or "failed" in stripped.lower()
            or "killed" in stripped.lower()
            or "SIGTERM" in stripped
            or "SIGKILL" in stripped
            or "SIGABRT" in stripped
        ):
            captured.append(stripped)
            if len(captured) >= 3:
                break

    if captured:
        captured.reverse()
        return "; ".join(captured)
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
            r.stderr.strip()[:200],
            r.stdout.strip()[:200],
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


# ── Job lifecycle functions (take JobQueue instance) ────────


def _try_mark_failed(jq, access_code, error_msg):
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
            conn = jq._get_conn()
            conn.execute(
                "UPDATE jobs SET status = ?, error = ?, failed_at = ?, status_changed_at = ? WHERE access_code = ?",
                (JobStatus.FAILED.value, error_msg, now, now, access_code),
            )
            conn.commit()
            publish_job_status(access_code, JobStatus.FAILED.value, error=error_msg)
            return
        except sqlite3.Error:
            if attempt == 0:
                jq._conn.close()
            else:
                logger.critical(
                    "Cannot update job %s to FAILED — may stay stuck at processing",
                    access_code,
                    exc_info=True,
                )


def _run_job(jq, access_code: str):
    """Execute a single job directly in the worker thread.

    Runs the job handler in-process — no multiprocessing child.  This
    avoids the unreliable HIP context re-initialisation that causes
    SIGSEGV on AMD Renoir iGPUs (gfx90c via ROCm) when a fresh spawn
    child loads the Chatterbox TTS model to GPU.

    If a genuine crash takes down this process, gunicorn restarts the
    worker, and checkpointed jobs resume on next startup.
    """
    conn = jq._get_conn()

    row = conn.execute(f"SELECT {', '.join(JOB_COLUMNS)} FROM jobs WHERE access_code = ?", (access_code,)).fetchone()

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
            (JobStatus.FAILED.value, f"Handler import failed: {e}", now, access_code),
        )
        conn.commit()
        return

    if not run_func:
        logger.error(f"Job {access_code} has no run_func")
        now = _now_str()
        conn.execute(
            "UPDATE jobs SET status = ?, error = ?, status_changed_at = ? WHERE access_code = ?",
            (JobStatus.FAILED.value, "Job handler not found", now, access_code),
        )
        conn.commit()
        return

    now = _now_str()
    cursor = conn.execute(
        "UPDATE jobs SET status = ?, status_changed_at = ? WHERE access_code = ? AND status = ?",
        (JobStatus.PROCESSING.value, now, access_code, JobStatus.PENDING.value),
    )
    conn.commit()

    if cursor.rowcount == 0:
        logger.info(f"Job {access_code} already claimed by another worker, skipping")
        return

    publish_job_status(access_code, JobStatus.PROCESSING.value)

    job.pop("run_func_name", None)
    job.pop("created_at", None)

    # Execute the job handler directly in this thread.  No more
    # multiprocessing child — the spawn boundary causes unreliable
    # HIP context re-initialisation on AMD Renoir iGPUs (gfx90c via
    # ROCm), leading to silent SIGSEGV during GPU model loads.  Running
    # in-process keeps the existing (working) HIP context.
    #
    # If this process crashes (e.g. genuine GPU bug), gunicorn
    # restarts the worker, and checkpointed jobs resume at the failed
    # step on the next startup without redoing completed work.
    success = False
    try:
        run_func(job)
        success = True
    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Job {access_code} handler '{run_func_name}' raised {type(e).__name__}: {e}")
        raise

    if success:
        now = _now_str()
        conn.execute(
            "UPDATE jobs SET status = ?, status_changed_at = ?, completed_at = ? WHERE access_code = ?",
            (JobStatus.COMPLETED.value, now, now, access_code),
        )
        conn.commit()
        publish_job_status(access_code, JobStatus.COMPLETED.value)
        logger.info(f"Job {access_code} completed successfully")

    # ── GPU state reset between jobs ─────────────────────────
    _reset_gpu_state_cooldown(jq)

    conn.commit()
