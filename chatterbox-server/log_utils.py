"""Shared job logging — writes to {output_dir}/job.log, stderr, and progress store."""

import logging
import os

logger = logging.getLogger(__name__)


def job_log(access_code: str, output_dir: str, msg: str):
    """Append msg to job.log, log to stderr, and update job progress."""
    log_path = os.path.join(output_dir, "job.log")
    with open(log_path, "a") as f:
        f.write(msg + "\n")
        f.flush()
    logger.info(f"[Job {access_code}] {msg}")
    from jobqueue import get_job_queue  # defer to avoid circular import
    get_job_queue().update_job_progress(access_code, msg)


def job_log_lines(access_code: str, output_dir: str, lines: list[str]):
    """Write multiple lines to job.log in a single file open/close cycle.
    Logs each line to stderr individually, but issues a single aggregate
    progress update instead of one per line."""
    if not lines:
        return
    log_path = os.path.join(output_dir, "job.log")
    with open(log_path, "a") as f:
        for line in lines:
            f.write(line + "\n")
    for line in lines:
        logger.info(f"[Job {access_code}] {line}")
    from jobqueue import get_job_queue  # defer to avoid circular import
    get_job_queue().update_job_progress(access_code, f"Captured {len(lines)} output lines")
