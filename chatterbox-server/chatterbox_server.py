from flask import Flask, request, send_file, jsonify, send_from_directory, session, abort
from functools import wraps
import logging
import os
import sys
import time
from collections import defaultdict

from audio_job import process_srt_file
from jobqueue import get_job_queue
from auth import get_user_manager
from config import AUDIO_TRACKS_DIR, VIDEO_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Rate limiter ──────────────────────────────────────────
_rate_limit_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 10     # max requests per window per IP


def rate_limit(f):
    """Simple in-memory rate limiter: max RATE_LIMIT_MAX requests per RATE_LIMIT_WINDOW per IP."""
    @wraps(f)
    def decorated(*args, **kwargs):
        ip = request.remote_addr or "unknown"
        now = time.time()
        window_start = now - RATE_LIMIT_WINDOW
        timestamps = _rate_limit_store[ip]
        # Prune old entries
        _rate_limit_store[ip] = [t for t in timestamps if t > window_start]
        if len(_rate_limit_store[ip]) >= RATE_LIMIT_MAX:
            logger.warning(f"Rate limit exceeded for IP {ip}")
            return jsonify({"error": "请求过于频繁，请稍后再试"}), 429
        _rate_limit_store[ip].append(now)
        return f(*args, **kwargs)
    return decorated


def _get_secret_key() -> str:
    """Get the Flask secret key — persist to file so sessions survive restarts."""
    env_key = os.environ.get("FLASK_SECRET_KEY")
    if env_key:
        return env_key
    key_file = os.path.join(BASE_DIR, ".secret_key")
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
    PERMANENT_SESSION_LIFETIME=86400 * 7,
)


HTML_DIR = os.path.join(BASE_DIR, "html")


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
            temperature = float(request.form.get("temperature", 0.8))
            target_language = request.form.get("target_language", "en")
            cfg_weight = float(request.form.get("cfg_weight", 0.5))
            exaggeration = float(request.form.get("exaggeration", 0.5))
            from audio_job import process_audio_file
            result = process_audio_file(content, file.filename, temperature, session["user_id"],
                                        target_language=target_language, cfg_weight=cfg_weight, exaggeration=exaggeration)
            return jsonify(result)
        else:
            data = request.get_json()
            text = data.get("text", "")
            if not text:
                return jsonify({"error": "Missing text"}), 400
            from tts_job import process_tts
            result = process_tts(text, data.get("filename", "output.wav"), session["user_id"],
                                 temperature=float(data.get("temperature", 0.8)),
                                 target_language=data.get("target_language", "en"),
                                 cfg_weight=float(data.get("cfg_weight", 0.5)),
                                 exaggeration=float(data.get("exaggeration", 0.5)))
            return jsonify(result)
    except Exception as e:
        logger.error(f"Audio process error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/tts/process", methods=["POST"])
@login_required
def tts_process():
    try:
        data = request.get_json()
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "Missing text"}), 400
        from tts_job import process_tts
        result = process_tts(text, data.get("filename", "output.wav"), session["user_id"],
                             temperature=float(data.get("temperature", 0.8)),
                             target_language=data.get("target_language", "en"),
                             cfg_weight=float(data.get("cfg_weight", 0.5)),
                             exaggeration=float(data.get("exaggeration", 0.5)))
        return jsonify(result)
    except Exception as e:
        logger.error(f"TTS process error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/tts/status/<access_code>", methods=["GET"])
def tts_status(access_code):
    status = get_job_queue().get_status(access_code)
    if not status:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(status)


@app.route("/tts/stream/<access_code>", methods=["GET"])
def tts_stream(access_code):
    import glob
    status = get_job_queue().get_status(access_code)
    if not status:
        return jsonify({"error": "Job not found"}), 404
    output_dir = status.get("output_dir")
    if not output_dir:
        return jsonify({"error": "No output directory"}), 404
    wav_files = glob.glob(os.path.join(output_dir, "*.wav"))
    if not wav_files:
        return jsonify({"error": "No audio file found"}), 404
    return send_file(wav_files[0], mimetype="audio/wav")


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Server is running"})


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
        logger.error(f"Error listing files: {e}")
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
        logger.error(f"Error reading file: {e}")
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
        logger.error(f"Error downloading file: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/files/delete", methods=["POST"])
@login_required
def files_delete():
    """Delete a file (only within allowed directories)"""
    try:
        data = request.get_json()
        file_path = data.get("path")
        if not file_path:
            return jsonify({"success": False, "error": "No file specified"}), 400

        safe_path = _safe_file_path(file_path)
        if not safe_path:
            return jsonify({"success": False, "error": "File not allowed or not found"}), 404

        os.remove(safe_path)
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


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
        temperature = float(request.form.get("temperature", 0.6))
        target_language = request.form.get("target_language", "en")
        cfg_weight = float(request.form.get("cfg_weight", 0.5))
        exaggeration = float(request.form.get("exaggeration", 0.5))

        result = process_srt_file(srt_file, temperature, session["user_id"],
                                  target_language=target_language, cfg_weight=cfg_weight, exaggeration=exaggeration)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing SRT: {str(e)}")
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
        logger.error(f"Error resubmitting job {access_code}: {str(e)}")
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
        temperature = float(request.form.get("temperature", 0.6))
        target_language = request.form.get("target_language", "en")
        cfg_weight = float(request.form.get("cfg_weight", 0.5))
        exaggeration = float(request.form.get("exaggeration", 0.5))

        from video_ning_job import process_video_ning
        blur = request.form.get("blur", "yes")
        result = process_video_ning(number, srt_file, temperature, session["user_id"], blur,
                                    target_language=target_language, cfg_weight=cfg_weight, exaggeration=exaggeration)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/video/ning/ocr-process", methods=["POST"])
@login_required
def video_ning_ocr_process():
    try:
        number = request.form.get("number")
        if not number:
            return jsonify({"error": "No video number provided"}), 400

        temperature = float(request.form.get("temperature", 0.6))
        target_language = request.form.get("target_language", "en")
        cfg_weight = float(request.form.get("cfg_weight", 0.5))
        exaggeration = float(request.form.get("exaggeration", 0.5))

        from video_ning_job import process_video_ning_ocr
        blur = request.form.get("blur", "yes")
        crop = request.form.get("crop", "")
        result = process_video_ning_ocr(number, temperature, session["user_id"], blur,
                                        target_language=target_language, cfg_weight=cfg_weight,
                                        exaggeration=exaggeration, crop=crop)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error OCR processing ning video: {str(e)}")
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
        temperature = float(request.form.get("temperature", 0.6))
        target_language = request.form.get("target_language", "en")
        cfg_weight = float(request.form.get("cfg_weight", 0.5))
        exaggeration = float(request.form.get("exaggeration", 0.5))

        from video_custom_job import process_video_custom
        result = process_video_custom(video_file, srt_file, temperature, session["user_id"],
                                      target_language=target_language, cfg_weight=cfg_weight, exaggeration=exaggeration)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing custom video: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/video/custom/auto-process", methods=["POST"])
@login_required
def video_custom_auto_process():
    try:
        if "video_file" not in request.files:
            return jsonify({"error": "No video file uploaded"}), 400

        video_file = request.files["video_file"]
        temperature = float(request.form.get("temperature", 0.6))
        target_language = request.form.get("target_language", "en")
        cfg_weight = float(request.form.get("cfg_weight", 0.5))
        exaggeration = float(request.form.get("exaggeration", 0.5))

        from video_custom_job import process_video_auto
        result = process_video_auto(video_file, temperature, session["user_id"],
                                    target_language=target_language, cfg_weight=cfg_weight, exaggeration=exaggeration)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error auto processing video: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/video/custom/ocr-process", methods=["POST"])
@login_required
def video_custom_ocr_process():
    try:
        if "video_file" not in request.files:
            return jsonify({"error": "No video file uploaded"}), 400

        video_file = request.files["video_file"]
        temperature = float(request.form.get("temperature", 0.6))
        target_language = request.form.get("target_language", "en")
        cfg_weight = float(request.form.get("cfg_weight", 0.5))
        exaggeration = float(request.form.get("exaggeration", 0.5))

        from video_custom_job import process_video_ocr
        result = process_video_ocr(video_file, temperature, session["user_id"],
                                   target_language=target_language, cfg_weight=cfg_weight, exaggeration=exaggeration)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error OCR processing video: {str(e)}")
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
    return jsonify({"success": True})


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
def api_job_cancel(access_code):
    result = get_job_queue().cancel_job(access_code)
    return jsonify(result)


@app.route("/api/jobs/<access_code>/delete", methods=["POST"])
@login_required
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
