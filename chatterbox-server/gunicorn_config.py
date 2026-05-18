import os
import logging

logger = logging.getLogger(__name__)


def post_fork(server, worker):
    """Called after a worker is forked. Reset the JobQueue singleton so each
    worker gets its own worker thread and DB connection."""
    from jobqueue import JobQueue
    # Reset the singleton so the forked worker re-initializes its own state
    JobQueue._instance = None
    JobQueue._lock = __import__("threading").Lock()
    logger.info(f"Worker {worker.pid} forked — JobQueue reset for this process")
