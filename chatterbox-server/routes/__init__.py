"""Route blueprints — registered by ``app.create_app()``."""

from .api import api_bp
from .auth import auth_bp
from .files import files_bp
from .jobs import jobs_bp
from .srt import srt_bp
from .tts import tts_bp
from .video import video_bp


def register_all(app):
    """Register all route blueprints on the Flask app."""
    app.register_blueprint(auth_bp)
    app.register_blueprint(tts_bp)
    app.register_blueprint(srt_bp)
    app.register_blueprint(video_bp)
    app.register_blueprint(files_bp)
    app.register_blueprint(jobs_bp)
    app.register_blueprint(api_bp)  # must be last (catch-all static route)
