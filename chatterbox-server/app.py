"""Application factory for the chatterbox Flask server.

Usage:
    from app import create_app
    app = create_app()
"""

import logging
import os
import time
import uuid

from flask import Flask, g, jsonify, request, session
from redis_util import is_available as redis_available


def _get_secret_key() -> str:
    """Get the Flask secret key — persist to file so sessions survive restarts."""
    env_key = os.environ.get("FLASK_SECRET_KEY")
    if env_key:
        return env_key
    base_dir = os.path.dirname(os.path.abspath(__file__))
    key_file = os.path.join(os.path.dirname(base_dir), ".secret_key")
    legacy = os.path.join(base_dir, ".secret_key")
    if os.path.exists(legacy) and not os.path.exists(key_file):
        os.rename(legacy, key_file)
    if os.path.exists(key_file):
        with open(key_file) as f:
            return f.read().strip()
    new_key = os.urandom(24).hex()
    with open(key_file, "w") as f:
        f.write(new_key)
    os.chmod(key_file, 0o600)
    return new_key


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__, static_folder=None)
    app.secret_key = _get_secret_key()
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("FLASK_ENV") == "production",
        PERMANENT_SESSION_LIFETIME=86400 * 7,
    )

    if redis_available():
        from cachelib.file import FileSystemCache

        from flask_session import Session
        app.config["SESSION_TYPE"] = "cachelib"
        app.config["SESSION_CACHELIB"] = FileSystemCache(
            cache_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "flask_session"),
            threshold=500,
        )
        Session(app)

    # ── Request lifecycle logging ──────────────────────────
    logger = logging.getLogger("chatterbox_server")

    @app.before_request
    def _assign_request_id():
        g.request_id = uuid.uuid4().hex[:12]
        g.request_start = time.monotonic()
        g.user_id = session.get("user_id")
        logger.info(
            "→ %s %s",
            request.method,
            request.path,
            extra={"method": request.method, "endpoint": request.path, "ip": request.remote_addr},
        )

    @app.after_request
    def _log_response(response):
        duration_ms = round((time.monotonic() - getattr(g, "request_start", 0)) * 1000)
        if response.status_code >= 500:
            response.headers["X-Request-Id"] = getattr(g, "request_id", "-")
            return response
        logger.info(
            "← %s %s %d (%dms)",
            request.method,
            request.path,
            response.status_code,
            duration_ms,
            extra={
                "method": request.method,
                "endpoint": request.path,
                "status": response.status_code,
                "duration_ms": duration_ms,
                "ip": request.remote_addr,
            },
        )
        response.headers["X-Request-Id"] = getattr(g, "request_id", "-")
        return response

    @app.errorhandler(500)
    def _log_internal_error(e):
        logger.error(
            "Unhandled 500 on %s %s",
            request.method,
            request.path,
            exc_info=True,
            extra={
                "method": request.method,
                "endpoint": request.path,
                "status": 500,
                "ip": request.remote_addr,
                "error": str(e),
            },
        )
        return jsonify({"error": "Internal server error"}), 500

    # ── Register route blueprints ──────────────────────────
    from routes import register_all
    register_all(app)

    return app
