"""Shared job logging — writes to {output_dir}/job.log, stderr, and progress store."""

import logging
import os
from contextlib import contextmanager

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


@contextmanager
def redirect_logging_to_file(log_path: str):
    """Temporarily add a file handler to the root logger so that all
    ``logging.*`` output (model loading, transformers messages, etc.)
    is written to *log_path* in addition to the existing handlers.

    This catches what ``redirect_stdout`` / ``redirect_stderr`` miss —
    anything emitted via ``logging.info/warning/error`` by libraries
    such as ``transformers``, ``torch``, or ``hy_mt``.

    Usage::

        with redirect_logging_to_file("/path/to/job.log"):
            run_model_translation(...)

    """
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    fh.setFormatter(logging.Formatter(fmt))
    logging.getLogger().addHandler(fh)
    try:
        yield
    finally:
        fh.close()
        logging.getLogger().removeHandler(fh)
