from flask import Flask, request, send_file, jsonify, send_from_directory, session, abort, Response, g
from functools import wraps
import json
import logging
import os
import secrets
import subprocess
import sys
import time
import uuid
from collections import defaultdict

import importlib
from jobqueue import get_job_queue
from auth import get_user_manager
from config import AUDIO_TRACKS_DIR, VIDEO_DIR, FILENAME_TO_CHECKPOINT_STEP
from redis_util import (
    InMemoryRateLimiter, is_available as redis_available,
    cache_get, cache_set, get_redis, get_session_redis,
)

# ── Deferred imports ──────────────────────────────────────
# All processing modules are imported lazily to avoid loading
# heavy dependencies (PyTorch, etc.) until a request actually
# needs them.  Import at the top of a handler with:
#     process_audio_file = _lazy("audio_job", "process_audio_file")

_MODULES: dict[str, object] = {}


def _lazy(module_name: str, attr: str):
    """Deferred import: load the module on first access and return the attribute."""
    import_key = f"{module_name}.{attr}"
    if import_key not in _MODULES:
        mod = importlib.import_module(module_name)
        _MODULES[import_key] = getattr(mod, attr)
    return _MODULES[import_key]

# ── Structured logging ─────────────────────────────────────
# Handler/filter setup and JSONFormatter live in gunicorn_config.py post_fork.

logger = logging.getLogger("chatterbox_server")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Rate limiter ──────────────────────────────────────────
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 10     # max requests per window per IP
EMAIL_RATE_LIMIT_WINDOW = 3600  # 1 hour
EMAIL_RATE_LIMIT_MAX = 3        # max 3 reset requests per email per hour

# In-memory fallback when Redis is unavailable
_ip_limiter = InMemoryRateLimiter(RATE_LIMIT_MAX, RATE_LIMIT_WINDOW)
_email_limiter = InMemoryRateLimiter(EMAIL_RATE_LIMIT_MAX, EMAIL_RATE_LIMIT_WINDOW)


def rate_limit(f):
    """Rate limiter: uses Redis when available, in-memory fallback otherwise."""
    @wraps(f)
    def decorated(*args, **kwargs):
        ip = request.remote_addr or "unknown"
        key = f"rl:ip:{ip}"

        # Try Redis first
        r = get_redis()
        if r is not None:
            try:
                pipe = r.pipeline()
                pipe.incr(key)
                pipe.expire(key, RATE_LIMIT_WINDOW)
                count, _ = pipe.execute()
                if count > RATE_LIMIT_MAX:
                    logger.warning(f"Rate limit exceeded for IP {ip}")
                    return jsonify({"error": "请求过于频繁，请稍后再试"}), 429
                return f(*args, **kwargs)
            except redis.RedisError:
                pass  # fall through to in-memory

        # In-memory fallback (Redis unavailable or errored)
        if not _ip_limiter.check(ip):
            logger.warning(f"Rate limit exceeded for IP {ip}")
            return jsonify({"error": "请求过于频繁，请稍后再试"}), 429
        return f(*args, **kwargs)
    return decorated


def email_rate_limit(email: str) -> tuple[bool, str]:
    """Check per-email rate limit for password reset. Returns (allowed, error_message).

    Tries Redis first, falls back to in-memory when Redis is down.
    """
    key = f"rl:email:{email}"

    # Try Redis first
    r = get_redis()
    if r is not None:
        try:
            pipe = r.pipeline()
            pipe.incr(key)
            pipe.expire(key, EMAIL_RATE_LIMIT_WINDOW)
            count, _ = pipe.execute()
            if count > EMAIL_RATE_LIMIT_MAX:
                logger.warning(f"Email rate limit exceeded for {email}")
                return False, "该邮箱的密码重置请求过于频繁，请稍后再试"
            return True, ""
        except redis.RedisError:
            pass  # fall through to in-memory

    # In-memory fallback
    if not _email_limiter.check(email):
        logger.warning(f"Email rate limit exceeded for {email}")
        return False, "该邮箱的密码重置请求过于频繁，请稍后再试"

    return True, ""


def _get_secret_key() -> str:
    """Get the Flask secret key — persist to file so sessions survive restarts.

    The key file is stored one directory above BASE_DIR (the project root)
    so it is outside the web-served directory tree and the catch-all static
    route can never serve it.
    """
    env_key = os.environ.get("FLASK_SECRET_KEY")
    if env_key:
        return env_key
    key_file = os.path.join(os.path.dirname(BASE_DIR), ".secret_key")
    # Migrate from legacy location inside BASE_DIR if present
    legacy = os.path.join(BASE_DIR, ".secret_key")
    if os.path.exists(legacy) and not os.path.exists(key_file):
        os.rename(legacy, key_file)
    if os.path.exists(key_file):
        with open(key_file, "r") as f:
            return f.read().strip()
    new_key = os.urandom(24).hex()
    with open(key_file, "w") as f:
        f.write(new_key)
    os.chmod(key_file, 0o600)
    return new_key


app = Flask(__name__, static_folder=None)
app.secret_key = _get_secret_key()
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("FLASK_ENV") == "production",
    PERMANENT_SESSION_LIFETIME=86400 * 7,
)

# Use Redis for server-side sessions when available; otherwise Flask's
# default signed-cookie session (no extra config needed).
if redis_available():
    from flask_session import Session
    app.config["SESSION_TYPE"] = "redis"
    app.config["SESSION_REDIS"] = get_session_redis()
    app.config["SESSION_KEY_PREFIX"] = "session:"
    Session(app)


# ── Request lifecycle logging ──────────────────────────────

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
    # Skip logging for 500s — the errorhandler already logged with full traceback
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


HTML_DIR = os.path.join(BASE_DIR, "html")


# ── CSRF protection ───────────────────────────────────────

def _generate_csrf_token() -> str:
    """Return a random CSRF token, creating one if the session has none."""
    if "_csrf_token" not in session:
        session["_csrf_token"] = secrets.token_hex(32)
    return session["_csrf_token"]


def csrf_required(f):
    """Decorator: require a valid CSRF token on state-changing POST requests.

    The token must be supplied via the ``X-CSRF-Token`` header.
    Auth-related endpoints (/auth/*) are exempt because they establish
    the session in the first place.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("X-CSRF-Token", "")
        expected = session.get("_csrf_token", "")
        if not expected or not token or not secrets.compare_digest(token, expected):
            logger.warning(f"CSRF token mismatch from {request.remote_addr}")
            return jsonify({"error": "CSRF token missing or invalid"}), 403
        return f(*args, **kwargs)
    return decorated


# ── Shared parameter parsing ──────────────────────────

_DEFAULT_PARAMS = {
    "temperature": 0.6,
    "target_language": "en",
    "cfg_weight": 0.5,
    "exaggeration": 0.5,
}

MAX_TEXT_LENGTH = 500


def _parse_float(source: dict, key: str, default: float) -> float:
    """Parse a float parameter from *source*, raising 400 on invalid input."""
    raw = source.get(key, default)
    try:
        return float(raw)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid {key}: {raw!r}")


def _parse_job_params(source: dict) -> dict:
    """Parse common job parameters from request.form or JSON body."""
    try:
        return {
            "temperature": _parse_float(source, "temperature", _DEFAULT_PARAMS["temperature"]),
            "target_language": source.get("target_language", _DEFAULT_PARAMS["target_language"]),
            "cfg_weight": _parse_float(source, "cfg_weight", _DEFAULT_PARAMS["cfg_weight"]),
            "exaggeration": _parse_float(source, "exaggeration", _DEFAULT_PARAMS["exaggeration"]),
        }
    except ValueError as e:
        abort(400, description=str(e))


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "请先登录"}), 401
        return f(*args, **kwargs)
    return decorated


@app.route("/", methods=["GET"])
def index():
    """Serve the landing page"""
    return send_from_directory(HTML_DIR, "index.html")


@app.route("/index.html", methods=["GET"])
def index_html():
    """Serve the landing page at /index.html"""
    return send_from_directory(HTML_DIR, "index.html")


@app.route("/tts", methods=["GET"])
def tts_page():
    """Serve the TTS form page"""
    return send_from_directory(HTML_DIR, "ningSound.html")


@app.route("/audio/process", methods=["POST"])
@login_required
def audio_process():
    try:
        if "file" in request.files:
            file = request.files["file"]
            content = file.read().decode("utf-8")
            params = _parse_job_params(request.form)
            process_audio_file = _lazy("audio_job", "process_audio_file")
            result = process_audio_file(content, file.filename, params["temperature"], session["user_id"],
                                        target_language=params["target_language"], cfg_weight=params["cfg_weight"], exaggeration=params["exaggeration"])
            return jsonify(result)
        else:
            data = request.get_json()
            text = data.get("text", "")
            if not text:
                return jsonify({"error": "Missing text"}), 400
            if len(text) > MAX_TEXT_LENGTH:
                return jsonify({"error": f"文字长度超过限制（最多{MAX_TEXT_LENGTH}字符）"}), 400
            process_tts = _lazy("tts_job", "process_tts")
            params = _parse_job_params(data)
            result = process_tts(text, data.get("filename", "output.wav"), session["user_id"],
                                 temperature=params["temperature"],
                                 target_language=params["target_language"],
                                 cfg_weight=params["cfg_weight"],
                                 exaggeration=params["exaggeration"])
            return jsonify(result)
    except Exception as e:
        logger.error(f"Audio process error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/tts/process", methods=["POST"])
@login_required
def tts_process():
    try:
        data = request.get_json()
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "Missing text"}), 400
        if len(text) > MAX_TEXT_LENGTH:
            return jsonify({"error": f"文字长度超过限制（最多{MAX_TEXT_LENGTH}字符）"}), 400
        process_tts = _lazy("tts_job", "process_tts")
        params = _parse_job_params(data)
        result = process_tts(text, data.get("filename", "output.wav"), session["user_id"],
                             temperature=params["temperature"],
                             target_language=params["target_language"],
                             cfg_weight=params["cfg_weight"],
                             exaggeration=params["exaggeration"])
        return jsonify(result)
    except Exception as e:
        logger.error(f"TTS process error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/tts/status/<access_code>", methods=["GET"])
def tts_status(access_code):
    status = get_job_queue().get_status(access_code)
    if not status:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(status)


@app.route("/tts/status-stream/<access_code>", methods=["GET"])
def tts_status_stream(access_code):
    """SSE endpoint: pushes job status updates until the job finishes."""
    def generate():
        r = get_redis()
        if r is None:
            # No Redis — fall back to single-poll
            status = get_job_queue().get_status(access_code)
            yield f"data: {json.dumps(status or {'error': 'Job not found'})}\n\n"
            return

        pubsub = r.pubsub()
        pubsub.subscribe(f"job:{access_code}")
        terminal = {"completed", "failed", "cancelled", "deleted"}

        # Send current status immediately
        status = get_job_queue().get_status(access_code)
        if status:
            yield f"data: {json.dumps(status)}\n\n"
            if status.get("status") in terminal:
                pubsub.unsubscribe()
                return

        # Listen for updates
        for message in pubsub.listen():
            if message["type"] != "message":
                continue
            try:
                data = json.loads(message["data"])
                yield f"data: {json.dumps(data)}\n\n"
                if data.get("status") in terminal:
                    break
            except (json.JSONDecodeError, TypeError):
                pass
        pubsub.unsubscribe()

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint (GPU status cached for 60s in Redis)."""
    import importlib
    cached = cache_get("health:gpu")
    if cached:
        cuda_ok = cached == "1"
    else:
        cuda_ok = importlib.import_module("torch").cuda.is_available()
        cache_set("health:gpu", "1" if cuda_ok else "0", ttl=60)
    return jsonify({
        "status": "healthy", "message": "Server is running",
        "cuda": cuda_ok,
        "hsa_override": os.environ.get("HSA_OVERRIDE_GFX_VERSION", "not set"),
    })


@app.route("/result", methods=["GET"])
def result_page():
    """Serve the result page"""
    return send_from_directory(HTML_DIR, "result.html")


# ── File path security ────────────────────────────────

ALLOWED_FILE_DIRS = [
    os.path.realpath(BASE_DIR),
    os.path.realpath(AUDIO_TRACKS_DIR) if os.path.exists(AUDIO_TRACKS_DIR) else None,
    os.path.realpath(VIDEO_DIR) if VIDEO_DIR and os.path.exists(VIDEO_DIR) else None,
]
ALLOWED_FILE_DIRS = [d for d in ALLOWED_FILE_DIRS if d]


def _safe_file_path(requested_path: str) -> str | None:
    """Resolve a file path and verify it falls within allowed directories."""
    resolved = os.path.realpath(requested_path)
    for allowed in ALLOWED_FILE_DIRS:
        if resolved.startswith(allowed + "/") or resolved == allowed:
            if os.path.isfile(resolved):
                return resolved
    logger.warning(f"Blocked path traversal attempt: {requested_path}")
    return None


def _get_video_metadata(path: str) -> dict:
    """Get duration (seconds) and resolution (WxH) of a video file via ffprobe.

    Returns dict with ``duration`` (float) and ``resolution`` (str, e.g. "1280x720").
    On any error returns empty dict so the caller still works.
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", path],
            capture_output=True, text=True, timeout=30,
        )
        info = json.loads(result.stdout)
        duration = float(info.get("format", {}).get("duration", 0))
        resolution = ""
        for s in info.get("streams", []):
            if s.get("codec_type") == "video":
                w = s.get("width", 0)
                h = s.get("height", 0)
                if w and h:
                    resolution = f"{w}x{h}"
                break
        meta = {}
        if duration:
            mins = int(duration // 60)
            secs = int(duration % 60)
            meta["duration"] = round(duration, 1)
            meta["duration_str"] = f"{mins}分{secs}秒"
        if resolution:
            meta["resolution"] = resolution
        return meta
    except Exception:
        return {}


@app.route("/files/list", methods=["GET"])
@login_required
def files_list():
    """List files in an allowed directory (non-recursive, non-tmp)"""
    try:
        dir_path = request.args.get("dir")
        if not dir_path:
            return jsonify({"error": "No directory specified"}), 400

        resolved = os.path.realpath(dir_path)
        allowed = False
        for d in ALLOWED_FILE_DIRS:
            if resolved.startswith(d + "/") or resolved == d:
                allowed = True
                break
        if not allowed:
            return jsonify({"error": "Directory not allowed"}), 403

        if not os.path.exists(resolved) or not os.path.isdir(resolved):
            return jsonify({"error": "Directory not found"}), 404

        files = []
        for name in os.listdir(resolved):
            if name.startswith('.'):
                continue
            file_path = os.path.join(resolved, name)
            if not os.path.isfile(file_path):
                continue
            files.append({
                "name": name,
                "size": os.path.getsize(file_path)
            })

        return jsonify({"files": files})
    except Exception as e:
        logger.error(f"Error listing files: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/files/read", methods=["GET"])
@login_required
def files_read():
    """Read file contents (only within allowed directories)"""
    try:
        file_path = request.args.get("path")
        if not file_path:
            return jsonify({"error": "No file specified"}), 400

        safe_path = _safe_file_path(file_path)
        if not safe_path:
            return jsonify({"error": "File not allowed or not found"}), 404

        with open(safe_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content, 200, {'Content-Type': 'text/plain; charset=utf-8'}
    except Exception as e:
        logger.error(f"Error reading file: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/files/download", methods=["GET"])
@login_required
def files_download():
    """Download a file (only within allowed directories)"""
    try:
        file_path = request.args.get("path")
        if not file_path:
            return jsonify({"error": "No file specified"}), 400

        safe_path = _safe_file_path(file_path)
        if not safe_path:
            return jsonify({"error": "File not allowed or not found"}), 404

        return send_file(safe_path, as_attachment=True)
    except Exception as e:
        logger.error(f"Error downloading file: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/files/delete", methods=["POST"])
@login_required
@csrf_required
def files_delete():
    """Delete a file (only within allowed directories)."""
    try:
        data = request.get_json()
        file_path = data.get("path")
        if not file_path:
            return jsonify({"success": False, "error": "No file specified"}), 400

        safe_path = _safe_file_path(file_path)
        if not safe_path:
            return jsonify({"success": False, "error": "File not allowed or not found"}), 404

        os.remove(safe_path)

        # If the job uses checkpoints, clear the relevant checkpoint step
        access_code = data.get("access_code")
        if access_code:
            get_job_queue().clear_checkpoint_for_file(access_code, file_path)

        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error deleting file: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# ── SRT save/edit endpoint ────────────────────────────────

# Mapping defined in config.py (shared with jobqueue.py).
# Invalidate downstream checkpoints when an SRT is edited.


@app.route("/files/save-srt", methods=["POST"])
@login_required
@csrf_required
def files_save_srt():
    """Save edited SRT content and invalidate downstream checkpoints.

    Expected JSON: { "path": "<full-path>", "content": "<new-srt-text>", "access_code": "<code>" }

    The step mapped from the filename determines which checkpoints are
    invalidated:
      - ocr_screen.srt   → invalidate from "ocr" onward → re-run translate+audio+video
      - whisper.srt       → invalidate from "whisper" onward → re-run translate+audio+video
      - translated.srt    → invalidate from "translate" onward → re-run audio+video
      - output_adjusted.srt → invalidate from "audio" onward → re-run audio+video

    After saving, the job is flagged as checkpoint_edited so the UI can show
    a "resubmit" button instead of "edit".
    """
    try:
        data = request.get_json()
        file_path = data.get("path")
        content = data.get("content")
        access_code = data.get("access_code")

        if not file_path or content is None:
            return jsonify({"success": False, "error": "Missing path or content"}), 400

        safe_path = _safe_file_path(file_path)
        if not safe_path:
            return jsonify({"success": False, "error": "File not allowed or not found"}), 404

        # Validate that the file has an .srt extension
        if not safe_path.lower().endswith(".srt"):
            return jsonify({"success": False, "error": "Only .srt files can be saved via this endpoint"}), 400

        # Write the new content
        with open(safe_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # Invalidate downstream checkpoints
        jq = get_job_queue()
        basename = os.path.basename(safe_path)
        step = FILENAME_TO_CHECKPOINT_STEP.get(basename)
        if step:
            jq.invalidate_checkpoints_after(access_code, step)

        # Mark as edited so UI switches from "edit" → "resubmit" for this specific file
        if access_code:
            jq.set_checkpoint_edited(access_code, True)
            jq.set_edited_srt_file(access_code, basename)

        return jsonify({"success": True, "message": "File saved, downstream steps marked for re-run"})
    except Exception as e:
        logger.error(f"Error saving SRT: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/files/srt-resubmit/<access_code>", methods=["POST"])
@login_required
@csrf_required
def files_srt_resubmit(access_code):
    """Resubmit a job after editing a checkpoint-level SRT file.

    The job status is set back to pending so the worker picks it up.
    Because downstream checkpoints were already invalidated by the save,
    the job will resume from the right step.
    """
    try:
        jq = get_job_queue()
        result = jq.resubmit_job(access_code)
        if result["success"]:
            # Clear the edited flag and per-file edit tracking
            jq.set_checkpoint_edited(access_code, False)
            jq.clear_edited_srt_files(access_code)
            return jsonify(result)
        return jsonify(result), 400
    except Exception as e:
        logger.error(f"Error resubmitting SRT-edited job {access_code}: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════
# SRT page and processing
# ═══════════════════════════════════════════

@app.route("/srt", methods=["GET"])
def srt_page():
    """Serve the SRT page"""
    return send_from_directory(HTML_DIR, "srt.html")


@app.route("/srt/process", methods=["POST"])
@login_required
def srt_process():
    try:
        if "srt_file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        srt_file = request.files["srt_file"]
        params = _parse_job_params(request.form)

        process_srt_file = _lazy("audio_job", "process_srt_file")
        result = process_srt_file(srt_file, params["temperature"], session["user_id"],
                                  target_language=params["target_language"], cfg_weight=params["cfg_weight"], exaggeration=params["exaggeration"])
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing SRT: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/srt/status/<access_code>", methods=["GET"])
def srt_status(access_code):
    status = get_job_queue().get_status(access_code)
    if not status:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(status)


@app.route("/srt/resubmit/<access_code>", methods=["POST"])
def srt_resubmit(access_code):
    try:
        result = get_job_queue().resubmit_job(access_code)
        if result["success"]:
            return jsonify(result)
        return jsonify(result), 400
    except Exception as e:
        logger.error(f"Error resubmitting job {access_code}: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════
# Video pages and processing
# ═══════════════════════════════════════════

@app.route("/video/ning", methods=["GET"])
def video_ning_page():
    return send_from_directory(HTML_DIR, "ningVideo.html")


@app.route("/video/ning/process", methods=["POST"])
@login_required
def video_ning_process():
    try:
        if "srt_file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        number = request.form.get("number")
        if not number:
            return jsonify({"error": "No video number provided"}), 400

        srt_file = request.files["srt_file"]
        params = _parse_job_params(request.form)

        process_video_ning = _lazy("video_ning_job", "process_video_ning")
        blur = request.form.get("blur", "yes")
        result = process_video_ning(number, srt_file, params["temperature"], session["user_id"], blur,
                                    target_language=params["target_language"], cfg_weight=params["cfg_weight"], exaggeration=params["exaggeration"])
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/video/ning/ocr-process", methods=["POST"])
@login_required
def video_ning_ocr_process():
    try:
        number = request.form.get("number")
        if not number:
            return jsonify({"error": "No video number provided"}), 400

        params = _parse_job_params(request.form)

        process_video_ning_ocr = _lazy("video_ning_job", "process_video_ning_ocr")
        find_cached = _lazy("video_ning_job", "_find_cached_video")
        blur = request.form.get("blur", "yes")
        start_trim = request.form.get("start_trim", "12.25")
        end_trim = request.form.get("end_trim", "40.0")

        # Check if user already decided to use a cached file
        cached_path = request.form.get("cached_path", "")
        if cached_path:
            result = process_video_ning_ocr(number, params["temperature"], session["user_id"], blur,
                                            target_language=params["target_language"], cfg_weight=params["cfg_weight"], exaggeration=params["exaggeration"],
                                            start_trim=float(start_trim), end_trim=float(end_trim),
                                            cached_path=cached_path)
            return jsonify(result)

        # Check if user explicitly said to ignore cache
        no_cache_check = request.form.get("no_cache_check", "false") == "true"
        if no_cache_check:
            result = process_video_ning_ocr(number, params["temperature"], session["user_id"], blur,
                                            target_language=params["target_language"], cfg_weight=params["cfg_weight"], exaggeration=params["exaggeration"],
                                            start_trim=float(start_trim), end_trim=float(end_trim))
            return jsonify(result)

        # First-time submission: check for cached video files
        cached = find_cached(number)
        if cached:
            meta = _get_video_metadata(cached)
            meta_parts = []
            if meta.get("duration_str"):
                meta_parts.append(f"时长 {meta['duration_str']}")
            if meta.get("resolution"):
                meta_parts.append(f"分辨率 {meta['resolution']}")
            meta_str = "，".join(meta_parts) if meta_parts else ""
            basename = os.path.basename(cached)
            msg = f"已找到缓存的视频文件 ({basename})"
            if meta_str:
                msg += f"\n{meta_str}"
            msg += "\n是否使用该已下载的版本？"
            return jsonify({
                "cached_found": True,
                "paths": [cached],
                "number": number,
                "message": msg,
                "metadata": meta,
            })

        # No cached file found — proceed normally
        result = process_video_ning_ocr(number, params["temperature"], session["user_id"], blur,
                                        target_language=params["target_language"], cfg_weight=params["cfg_weight"], exaggeration=params["exaggeration"],
                                        start_trim=float(start_trim), end_trim=float(end_trim))
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error OCR processing ning video: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/video/custom", methods=["GET"])
def video_custom_page():
    return send_from_directory(HTML_DIR, "userVideo.html")


@app.route("/video/custom/process", methods=["POST"])
@login_required
def video_custom_process():
    try:
        if "video_file" not in request.files:
            return jsonify({"error": "No video file uploaded"}), 400
        if "srt_file" not in request.files:
            return jsonify({"error": "No SRT file uploaded"}), 400

        video_file = request.files["video_file"]
        srt_file = request.files["srt_file"]
        params = _parse_job_params(request.form)

        process_video_custom = _lazy("video_custom_job", "process_video_custom")
        result = process_video_custom(video_file, srt_file, params["temperature"], session["user_id"],
                                      target_language=params["target_language"], cfg_weight=params["cfg_weight"], exaggeration=params["exaggeration"])
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing custom video: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/video/custom/auto-process", methods=["POST"])
@login_required
def video_custom_auto_process():
    try:
        if "video_file" not in request.files:
            return jsonify({"error": "No video file uploaded"}), 400

        video_file = request.files["video_file"]
        params = _parse_job_params(request.form)

        process_video_auto = _lazy("video_custom_job", "process_video_auto")
        result = process_video_auto(video_file, params["temperature"], session["user_id"],
                                    target_language=params["target_language"], cfg_weight=params["cfg_weight"], exaggeration=params["exaggeration"])
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error auto processing video: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════
# OCR-only: extract subtitles from video
# ═══════════════════════════════════════════

@app.route("/video/ocr", methods=["GET"])
def video_ocr_page():
    """Serve the OCR-only page (extract subtitles from video)."""
    return send_from_directory(HTML_DIR, "videoOCR.html")


@app.route("/video/ocr/process", methods=["POST"])
@login_required
def video_ocr_process():
    try:
        if "video_file" not in request.files:
            return jsonify({"error": "No video file uploaded"}), 400

        video_file = request.files["video_file"]
        process_ocr_only = _lazy("video_ocr_job", "process_ocr_only")
        result = process_ocr_only(video_file, session["user_id"])
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing OCR-only job: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/video/custom/ocr-process", methods=["POST"])
@login_required
def video_custom_ocr_process():
    try:
        if "video_file" not in request.files:
            return jsonify({"error": "No video file uploaded"}), 400

        video_file = request.files["video_file"]
        params = _parse_job_params(request.form)

        process_video_ocr = _lazy("video_custom_job", "process_video_ocr")
        result = process_video_ocr(video_file, params["temperature"], session["user_id"],
                                   target_language=params["target_language"], cfg_weight=params["cfg_weight"], exaggeration=params["exaggeration"])
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error OCR processing video: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════
# Auth routes
# ═══════════════════════════════════════════

@app.route("/auth/register", methods=["GET", "POST"])
@rate_limit
def auth_register():
    if request.method == "GET":
        return send_from_directory(HTML_DIR, "register.html")
    data = request.get_json()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    if not email or not password:
        return jsonify({"success": False, "error": "请填写邮箱和密码"})
    result = get_user_manager().register(email, password)
    return jsonify(result)


@app.route("/auth/verify", methods=["POST"])
@rate_limit
def auth_verify():
    data = request.get_json()
    email = data.get("email", "").strip().lower()
    code = data.get("code", "").strip()
    result = get_user_manager().verify(email, code)
    return jsonify(result)


@app.route("/auth/login", methods=["GET", "POST"])
@rate_limit
def auth_login():
    if request.method == "GET":
        return send_from_directory(HTML_DIR, "login.html")
    data = request.get_json()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    result = get_user_manager().login(email, password)
    if result["success"]:
        session["user_id"] = result["user"]["id"]
        session["user_email"] = result["user"]["email"]
    return jsonify(result)


@app.route("/auth/logout", methods=["POST"])
def auth_logout():
    session.clear()
    # Regenerate session ID to prevent session fixation
    session.regenerate() if hasattr(session, 'regenerate') else None
    return jsonify({"success": True})


@app.route("/auth/change-password", methods=["GET", "POST"])
@login_required
def auth_change_password():
    if request.method == "GET":
        return send_from_directory(HTML_DIR, "change-password.html")
    data = request.get_json()
    old_password = data.get("old_password", "")
    new_password = data.get("new_password", "")
    email = session.get("user_email", "")
    result = get_user_manager().change_password(email, old_password, new_password)
    return jsonify(result)


@app.route("/auth/resend", methods=["POST"])
@rate_limit
def auth_resend():
    data = request.get_json()
    email = data.get("email", "").strip().lower()
    result = get_user_manager().resend_code(email)
    return jsonify(result)


@app.route("/auth/me", methods=["GET"])
def auth_me():
    if "user_id" not in session:
        return jsonify({"authenticated": False})
    return jsonify({
        "authenticated": True,
        "user": {"id": session["user_id"], "email": session["user_email"]},
    })


@app.route("/auth/csrf-token", methods=["GET"])
def auth_csrf_token():
    """Return a CSRF token for the current session."""
    return jsonify({"csrf_token": _generate_csrf_token()})


@app.route("/auth/reset-password", methods=["GET", "POST"])
@rate_limit
def auth_reset_password():
    """GET → serve the reset-password request page (enter email)
    POST → send reset code to email"""
    if request.method == "GET":
        return send_from_directory(HTML_DIR, "reset-password.html")
    data = request.get_json()
    email = data.get("email", "").strip().lower()
    if not email:
        return jsonify({"success": False, "error": "请输入邮箱"})
    
    allowed, error_msg = email_rate_limit(email)
    if not allowed:
        return jsonify({"success": False, "error": error_msg})
    
    result = get_user_manager().request_reset(email)
    return jsonify(result)


@app.route("/auth/reset-password/confirm", methods=["POST"])
@rate_limit
def auth_reset_password_confirm():
    """Verify reset code and set new password."""
    data = request.get_json()
    email = data.get("email", "").strip().lower()
    code = data.get("code", "").strip()
    new_password = data.get("new_password", "")
    if not email or not code or not new_password:
        return jsonify({"success": False, "error": "请填写所有字段"})
    result = get_user_manager().reset_password(email, code, new_password)
    return jsonify(result)


# ═══════════════════════════════════════════
# My Jobs & Job management
# ═══════════════════════════════════════════

@app.route("/my-jobs", methods=["GET"])
def my_jobs_page():
    return send_from_directory(HTML_DIR, "my-jobs.html")


@app.route("/api/my-jobs", methods=["GET"])
@login_required
def api_my_jobs():
    user_id = session["user_id"]
    jobs = get_job_queue().get_user_jobs(user_id)
    return jsonify({"jobs": jobs})


@app.route("/api/jobs/<access_code>/cancel", methods=["POST"])
@login_required
@csrf_required
def api_job_cancel(access_code):
    result = get_job_queue().cancel_job(access_code)
    return jsonify(result)


@app.route("/api/jobs/<access_code>/delete", methods=["POST"])
@login_required
@csrf_required
def api_job_delete(access_code):
    result = get_job_queue().delete_job(access_code)
    return jsonify(result)


# ═══════════════════════════════════════════
# Static files (catch-all — must be last)
# ═══════════════════════════════════════════

@app.route("/<path:filename>", methods=["GET"])
def serve_static(filename):
    """Serve static files (css, js, etc.) from the html directory with no-cache."""
    resp = send_from_directory(HTML_DIR, filename)
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5600, debug=True)
