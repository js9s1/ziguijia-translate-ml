"""Application factory for the chatterbox Flask server.

Usage:
    from app import create_app
    app = create_app()
"""

import json
import logging
import os
import time
import uuid

from flask import Flask, g, jsonify, redirect, request, session
from valkey_util import is_available as valkey_available


def _is_ajax():
    return (
        request.headers.get("X-Requested-With") == "XMLHttpRequest"
        or request.headers.get("Accept") == "application/json"
    )


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
    fd = os.open(key_file, os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(new_key)
    except Exception:
        os.close(fd)
        os.unlink(key_file)
        raise
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

    if valkey_available():
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

    @app.before_request
    def _handle_options():
        if request.method == "OPTIONS":
            resp = app.make_default_options_response()
            resp.headers["Access-Control-Allow-Origin"] = "*"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            return resp

    @app.after_request
    def _log_response(response):
        duration_ms = round((time.monotonic() - getattr(g, "request_start", 0)) * 1000)

        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"

        if response.status_code >= 500:
            response.headers["X-Request-Id"] = getattr(g, "request_id", "-")
            return response

        is_json = response.content_type and "application/json" in response.content_type
        if is_json and not _is_ajax() and request.method == "POST":
            try:
                body = json.loads(response.get_data(as_text=True))
                code = body.get("access_code")
                if code:
                    return redirect("/result?code=" + code)
                if response.status_code == 401:
                    return redirect("/auth/login?next=" + request.path)
            except Exception:
                pass

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
