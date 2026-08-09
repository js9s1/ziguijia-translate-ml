import json
import logging

import flask.globals as flask_globals


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None)
        if request_id:
            log_entry["request_id"] = request_id
        user_id = getattr(record, "user_id", None)
        if user_id:
            log_entry["user_id"] = user_id
        for key in ("endpoint", "method", "status", "duration_ms", "ip", "error"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["traceback"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


class RequestIDFilter(logging.Filter):
    """Inject ``request_id`` (and ``user_id``) into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            g = flask_globals.g
            record.request_id = g.request_id
            record.user_id = getattr(g, "user_id", None)
        except RuntimeError:
            record.request_id = "-"
            record.user_id = None
        return True


def post_fork(server, worker):
    """Called after a worker is forked. Reset the JobQueue singleton so each
    worker gets its own worker thread and DB connection. Also reset the
    NingAudio singleton and Indonesian model handle so no stale GPU
    references leak from the parent process. Finally install
    structured JSON logging on gunicorn's existing handler."""
    from jobqueue import JobQueue

    JobQueue.clear()
    # Eagerly initialise the JobQueue so the worker thread starts processing
    # pending jobs immediately, without waiting for a web request.
    from jobqueue import get_job_queue

    get_job_queue()

    # Reset TTS model singletons — gen_audio now runs as a subprocess
    # on its own Python 3.13, so the worker never holds GPU handles.
    try:
        from gpu_manage import _clear_all_models

        _clear_all_models()
    except ImportError:
        pass  # OK — torch/gpu_manage not needed in server worker

    fmt = JSONFormatter()
    req_filter = RequestIDFilter()

    # Find gunicorn's error handler to reuse its file descriptor
    gunicorn_error_logger = logging.getLogger("gunicorn.error")
    gunicorn_handler = gunicorn_error_logger.handlers[0] if gunicorn_error_logger.handlers else None

    if gunicorn_handler:
        # Install our formatter/filter on gunicorn's handler
        gunicorn_handler.setFormatter(fmt)
        gunicorn_handler.addFilter(req_filter)

        # Add the same handler to application loggers so they write to the same file
        for name in (
            "chatterbox_server",
            "auth",
            "jobqueue",
            "valkey_util",
            "audio_job",
            "tts_job",
            "video_ning_job",
            "video_custom_job",
            "video_ocr_job",
            "video_util",
            "pipeline",
            "db_schema",
            "log_utils",
        ):
            app_logger = logging.getLogger(name)
            app_logger.handlers.clear()
            app_logger.addHandler(gunicorn_handler)
            app_logger.propagate = False
            app_logger.setLevel(logging.INFO)

    server.log.info("Worker %d forked — JobQueue reset, structured logging installed", worker.pid)


def worker_exit(server, worker):
    """Called when a worker is shutting down.

    Triggers graceful shutdown of the job queue — waits for the current
    job to finish (up to shutdown_timeout seconds) before the worker exits.
    If the job doesn't finish, it's marked PENDING for automatic resume.
    """
    from jobqueue import get_job_queue

    try:
        jq = get_job_queue()
        jq.shutdown()
    except Exception as e:
        server.log.warning("Error during graceful shutdown: %s", e)
