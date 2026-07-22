"""Shared test fixtures — Flask test client, in-memory DB, auth helpers."""

import os
import sys
import tempfile
import shutil
import sqlite3

import pytest

# ── Prevent config.py from loading real .env and setting GPU env vars ──
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "9.0.0")
os.environ.setdefault("HSA_XNACK", "0")
os.environ.setdefault("ROCBLAS_USE_HIPBLASLT", "0")
os.environ["FLASK_SECRET_KEY"] = "test-secret-key-42"
os.environ["VALKEY_PASSWORD"] = ""

# Ensure chatterbox-server is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "chatterbox-server"))


# ── Prevent JobQueue singleton from starting worker thread ──
import jobqueue as _jq  # noqa: E402
_jq._SKIP_QUEUE_INIT = True


@pytest.fixture(scope="session")
def temp_db_dir():
    """Create a temporary directory for test databases."""
    tmpdir = tempfile.mkdtemp(prefix="chatterbox_test_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(autouse=True)
def _patch_db_paths(temp_db_dir, monkeypatch):
    """Redirect DB paths to temp directory for all tests.

    This must run BEFORE the app is created because JobQueue and
    UserManager are singletons that open their DB files in __init__.
    """
    # Patch the paths in every module that references DB_FILE
    monkeypatch.setattr("jobqueue.DB_FILE",
                        os.path.join(temp_db_dir, "jobs.db"))
    monkeypatch.setattr("auth.DB_FILE",
                        os.path.join(temp_db_dir, "users.db"))

    # Also patch the config to use temp dirs for audio/video
    monkeypatch.setattr("config.AUDIO_TRACKS_DIR",
                        os.path.join(temp_db_dir, "audio_tracks"))
    monkeypatch.setattr("config.VIDEO_DIR",
                        os.path.join(temp_db_dir, "video"))
    monkeypatch.setattr("config.ASSETS_DIR",
                        os.path.join(temp_db_dir, "assets"))
    monkeypatch.setattr("config.SMTP_HOST", "")
    monkeypatch.setattr("config.SMTP_PORT", 587)

    # Clear singletons before each test module so DBs are fresh
    from jobqueue import JobQueue
    from auth import UserManager
    from audio_utils import NingAudio
    JobQueue.clear()
    UserManager.clear()
    NingAudio.clear()

    yield

    JobQueue.clear()
    UserManager.clear()
    NingAudio.clear()


@pytest.fixture
def app():
    """Create a Flask app configured for testing."""
    from app import create_app
    app = create_app()
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test-secret"
    yield app
    # Stop the job queue worker if it started
    try:
        from jobqueue import get_job_queue
        get_job_queue().stop()
    except Exception:
        pass


@pytest.fixture
def client(app):
    """Flask test client."""
    return app.test_client()


@pytest.fixture
def isolated_db(temp_db_dir):
    """Create a fresh SQLite database for unit tests.

    Each test gets its own file so table names don't clash.
    """
    import uuid
    db_path = os.path.join(temp_db_dir, f"unittest_{uuid.uuid4().hex[:8]}.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture
def auth_client(client):
    """Test client with a logged-in user session, CSRF token, and rate-limit bypass.

    Creates a user, verifies them, then pushes the session.
    Rate limiting is disabled by setting a huge in-memory limit so tests
    sharing the same IP don't get blocked.
    """
    # Bypass rate limit for test
    from middleware import _ip_limiter, _email_limiter, RATE_LIMIT_MAX
    _ip_limiter.limit = 10000
    _email_limiter.limit = 10000

    user_data = {"email": "test@example.com", "password": "test123456"}
    client.post("/auth/register", json=user_data)

    from auth import get_user_manager
    um = get_user_manager()
    conn = um._get_conn()
    conn.execute(
        "UPDATE users SET verified = 1, verification_code = NULL WHERE email = ?",
        (user_data["email"],),
    )
    conn.commit()

    with client.session_transaction() as sess:
        sess["user_id"] = 1
        sess["user_email"] = user_data["email"]
        sess["_csrf_token"] = "test-csrf-token"

    return client, user_data


@pytest.fixture
def csrf_headers():
    """Headers that include a valid CSRF token for the test session."""
    return {"X-CSRF-Token": "test-csrf-token", "Content-Type": "application/json"}
