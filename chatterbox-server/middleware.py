"""Shared middleware — rate limiting, CSRF, login, validation, file security.

Extracted from chatterbox_server.py so routes can live in their own modules.
"""

import logging
import os
import re
import secrets
import subprocess
import tempfile
from functools import wraps

import valkey
from config import AUDIO_TRACKS_DIR, VIDEO_DIR, is_screen_recording_filename, validate_upload_filename
from flask import abort, jsonify, request, session
from valkey_util import (
    InMemoryRateLimiter,
    check_rate_limit,
    get_valkey,
)

logger = logging.getLogger("chatterbox_server")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Rate limiter ──────────────────────────────────────────

RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX = 10
EMAIL_RATE_LIMIT_WINDOW = 3600
EMAIL_RATE_LIMIT_MAX = 3

_ip_limiter = InMemoryRateLimiter(RATE_LIMIT_MAX, RATE_LIMIT_WINDOW)
_email_limiter = InMemoryRateLimiter(EMAIL_RATE_LIMIT_MAX, EMAIL_RATE_LIMIT_WINDOW)


def _check_ip_rate_limit(ip_key: str) -> bool:
    r = get_valkey()
    if r is not None:
        try:
            return check_rate_limit(ip_key, RATE_LIMIT_MAX, RATE_LIMIT_WINDOW)
        except valkey.ValkeyError:
            pass
    return _ip_limiter.check(ip_key)


def rate_limit(f):
    """Rate limiter: uses Redis when available, in-memory fallback otherwise."""

    @wraps(f)
    def decorated(*args, **kwargs):
        ip = request.remote_addr or "unknown"
        key = f"rl:ip:{ip}"
        if not _check_ip_rate_limit(key):
            logger.warning(f"Rate limit exceeded for IP {ip}")
            return jsonify({"error": "请求过于频繁，请稍后再试"}), 429
        return f(*args, **kwargs)

    return decorated


def email_rate_limit(email: str) -> tuple[bool, str]:
    """Check per-email rate limit for password reset. Returns (allowed, error_message)."""
    key = f"rl:email:{email}"
    r = get_valkey()
    allowed = True
    if r is not None:
        try:
            allowed = check_rate_limit(key, EMAIL_RATE_LIMIT_MAX, EMAIL_RATE_LIMIT_WINDOW)
        except valkey.ValkeyError:
            pass
    if not allowed:
        return False, "该邮箱的密码重置请求过于频繁，请稍后再试"
    if not _email_limiter.check(key):
        return False, "该邮箱的密码重置请求过于频繁，请稍后再试"
    return True, ""


# ── CSRF protection ───────────────────────────────────────


def generate_csrf_token() -> str:
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


# ── Login required ────────────────────────────────────────


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "请先登录"}), 401
        return f(*args, **kwargs)

    return decorated


# ── Shared parameter parsing ──────────────────────────────

_DEFAULT_PARAMS = {
    "temperature": 0.6,
    "target_language": "en",
    "cfg_weight": 0.25,
    "exaggeration": 0.3,
}


def parse_float_param(source: dict, key: str, default: float) -> float:
    """Parse a float parameter from *source*, raising 400 on invalid input."""
    raw = source.get(key, default)
    try:
        return float(raw)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid {key}: {raw!r}") from None


def parse_job_params(source: dict) -> dict:
    """Parse common job parameters from request.form or JSON body."""
    try:
        return {
            "temperature": parse_float_param(source, "temperature", _DEFAULT_PARAMS["temperature"]),
            "target_language": source.get("target_language", _DEFAULT_PARAMS["target_language"]),
            "cfg_weight": parse_float_param(source, "cfg_weight", _DEFAULT_PARAMS["cfg_weight"]),
            "exaggeration": parse_float_param(source, "exaggeration", _DEFAULT_PARAMS["exaggeration"]),
        }
    except ValueError as e:
        abort(400, description=str(e))


def get_audio_params(job_data: dict) -> dict:
    """Extract audio parameters from a job_data dict using shared defaults."""
    return {
        "temperature": job_data.get("temperature", _DEFAULT_PARAMS["temperature"]),
        "target_language": job_data.get("target_language", _DEFAULT_PARAMS["target_language"]),
        "cfg_weight": job_data.get("cfg_weight", _DEFAULT_PARAMS["cfg_weight"]),
        "exaggeration": job_data.get("exaggeration", _DEFAULT_PARAMS["exaggeration"]),
    }


# ── File path security ────────────────────────────────────

ALLOWED_FILE_DIRS = [
    os.path.realpath(BASE_DIR),
]
for _extra in (AUDIO_TRACKS_DIR, VIDEO_DIR):
    if _extra and os.path.exists(_extra):
        ALLOWED_FILE_DIRS.append(os.path.realpath(_extra))

_batch_dir = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "batch")
if os.path.exists(_batch_dir):
    ALLOWED_FILE_DIRS.append(os.path.realpath(_batch_dir))


def safe_file_path(requested_path: str) -> str | None:
    """Resolve a file path and verify it falls within allowed directories."""
    resolved = os.path.realpath(requested_path)
    for allowed in ALLOWED_FILE_DIRS:
        if resolved.startswith(allowed + "/") or resolved == allowed:
            if os.path.isfile(resolved):
                return resolved
    logger.warning(f"Blocked path traversal attempt: {requested_path}")
    return None


def get_video_metadata(path: str) -> dict:
    """Get duration (seconds) and resolution (WxH) of a video file via ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        info = __import__("json").loads(result.stdout)
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
    except (subprocess.TimeoutExpired, ValueError, OSError):
        return {}


# ── SRT validation ────────────────────────────────────────

_SRT_TIMING_RE = re.compile(r"\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}")


def validate_file_upload(uploaded_file, label: str = "file"):
    """Validate an uploaded file in one place.

    Performs, in order:
      1. filename validation (empty, reserved ``output*`` prefix, path traversal)
      2. non-empty content check
      3. kind-specific checks (video codec, SRT timing/language/format)

    Raises ValueError with a user-facing message on any failure.
    """
    filename = uploaded_file.filename or ""
    validate_upload_filename(filename)
    if os.path.basename(filename) != filename:
        raise ValueError(f"Uploaded {label} filename contains path components: '{filename}'")

    content = uploaded_file.read()
    uploaded_file.seek(0)
    if not content:
        raise ValueError(f"Uploaded {label} file is empty (0 bytes). Please check the file and try again.")

    if "video" in label.lower():
        if is_screen_recording_filename(filename):
            raise ValueError(
                f"检测到录屏文件 '{filename}'。请勿上传录屏视频：录屏视频处理会消耗大量资源，"
                f"请上传原始视频文件。"
            )
        _validate_video_codec(content)

    if "srt" in label.lower():
        text = content.decode("utf-8", errors="replace")
        validate_srt_content(text, label)


def validate_srt_content(text: str, label: str = "SRT"):
    """Validate SRT text: timing lines, single language, and full parse for
    formatting errors (code fences, malformed indexes, bad timestamps)."""
    text = text.lstrip("\ufeff")
    if not _SRT_TIMING_RE.search(text):
        raise ValueError(
            f"Uploaded {label} file does not appear to be a valid SRT file. "
            f"Expected timing lines like '00:00:01,000 --> 00:00:03,000'."
        )
    _validate_srt_language(text)

    import srt

    try:
        list(srt.parse(text))
    except Exception as e:
        raise ValueError(f"Uploaded {label} file has formatting errors: {e}") from e


def _validate_video_codec(content: bytes):
    import json

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        try:
            result = None
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-select_streams", "v:0", tmp.name],
                capture_output=True,
                text=True,
                timeout=30,
            )
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
    if result is None:
        raise ValueError("Cannot probe video codec — ffprobe timed out or failed.")
    if result.returncode != 0:
        raise ValueError("Cannot probe video codec — file may be corrupt.")
    info = json.loads(result.stdout)
    streams = info.get("streams", [])
    if not streams:
        raise ValueError("No video stream found in uploaded file.")


def _validate_srt_language(text: str):
    from language_utils import UNICODE_SCRIPTS

    lines = text.splitlines()
    script_counts: dict[str, int] = {}
    total_chars = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.isdigit():
            continue
        if "-->" in stripped:
            continue
        for name, pattern in UNICODE_SCRIPTS.items():
            matches = len(pattern.findall(stripped))
            if matches:
                script_counts[name] = script_counts.get(name, 0) + matches
                total_chars += matches

    # Only reject if there's substantial mixing (>15% minority script).
    # Chinese OCR SRTs commonly contain occasional Latin letters
    # (URLs, abbreviations, brand names) which should not block saving.
    if len(script_counts) > 1:
        total = sum(script_counts.values())
        for name, count in script_counts.items():
            if total > 0 and count / total < 0.15:
                continue  # minority script, ignore
            # All remaining scripts must be compatible
            if name in ("CJK", "Latin"):
                continue  # CJK + occasional Latin is expected in OCR
            # Any other script found above threshold — that's genuinely mixed
            raise ValueError(
                f"SRT file contains mixed languages (found {name} + "
                + ", ".join(script_counts.keys())
                + "). Please upload an SRT file in a single language."
            )
        # Only Latin + CJK found — that's acceptable
