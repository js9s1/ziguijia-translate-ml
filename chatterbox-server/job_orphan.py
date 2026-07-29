"""Orphan process cleanup — killing subprocesses left behind by dead/crashed jobs.

Extracted from ``jobqueue.py``.  All functions that need the ``JobQueue``
instance take it as their first parameter (``jq``).
"""

import logging
import os
import signal

import psutil
from job_types import JobStatus
from job_worker import _safe_close_proc, _safe_close_psutil_procs

logger = logging.getLogger(__name__)


def kill_processes_by_output_dir(output_dir: str, sig: int = signal.SIGTERM):
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


def kill_process_group(jq, proc, output_dir: str | None = None):
    """Kill *proc* and its entire process group, with a psutil-based
    fallback when the child process PID is no longer valid."""
    import signal as _sig

    # Try process-group kill first
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, _sig.SIGTERM)
        pgid_killed = True
    except (ProcessLookupError, PermissionError, AttributeError):
        pgid_killed = False

    # Fall back to psutil-based kill when PGID lookup failed.
    if not pgid_killed and output_dir:
        kill_processes_by_output_dir(output_dir, _sig.SIGTERM)

    proc.join(timeout=3)
    if proc.is_alive():
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, _sig.SIGKILL)
            pgid_killed = True
        except (ProcessLookupError, PermissionError, AttributeError):
            pgid_killed = False
        if not pgid_killed and output_dir:
            kill_processes_by_output_dir(output_dir, _sig.SIGKILL)
        proc.join(timeout=2)

    # Release OS resources (pipe fd, process-table entry).
    # Must be called after every proc.join() regardless of outcome.
    _safe_close_proc(proc)


def cleanup_orphan_processes(jq):
    """Kill subprocesses orphaned by a server crash / restart.

    Iterates all running processes and terminates any whose command-line
    references the output directory of a job that is no longer in a
    running/processing state.  This prevents resource leaks where
    ``gen_audio.py``, ``gen_video.py``, ffmpeg, etc. keep consuming
    CPU / GPU / memory after their parent died or the job was marked
    failed before the subprocess finished.
    """
    conn = jq._get_conn()
    rows = conn.execute(
        "SELECT DISTINCT output_dir FROM jobs WHERE status NOT IN (?, ?) AND output_dir IS NOT NULL",
        (JobStatus.PROCESSING.value, JobStatus.PENDING.value),
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
                except (psutil.AccessDenied, psutil.ZombieProcess):
                    try:
                        proc.kill()
                        killed_pids.append(proc)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
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
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        _safe_close_psutil_procs(killed_pids)
        logger.info("Cleaned up %d orphan subprocess(es) from previous server instance", len(killed_pids))
