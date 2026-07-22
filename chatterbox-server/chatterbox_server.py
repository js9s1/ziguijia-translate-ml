"""Chatterbox server — ML video translation + TTS web application.

Module-level ``app`` is exported for gunicorn compatibility
(``gunicorn chatterbox_server:app``).

The application is built by ``app.create_app()``, which wires together
Flask configuration, session management, logging middleware, and
all route blueprints.
"""

from app import create_app

# Backward-compatible build_all_static_srt import for gunicorn_config.
from oldrun import build_all_static_srt as _build_all_static_srt  # noqa: F401

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5600, debug=True)
