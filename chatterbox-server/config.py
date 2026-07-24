"""Centralized configuration for paths and environment variables."""

import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

# ── ROCm environment for AMD Renoir APU (gfx90c) ────────────
# Must be set before any PyTorch/ROCm import.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "9.0.0")
os.environ.setdefault("HSA_XNACK", "0")
os.environ.setdefault("ROCBLAS_USE_HIPBLASLT", "0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # /home/js9s/子归家/code_ml

# ── Directories ──────────────────────────────────────────────
AUDIO_TRACKS_DIR = os.environ.get(
    "AUDIO_TRACKS_DIR",
    os.path.abspath(os.path.join(PROJECT_ROOT, "..", "audio_tracks")),  # /home/js9s/子归家/audio_tracks
)
VIDEO_DIR = os.environ.get(
    "VIDEO_DIR",
    os.path.abspath(os.path.join(PROJECT_ROOT, "..", "video")),  # /home/js9s/子归家/video
)
ASSETS_DIR = os.environ.get(
    "ASSETS_DIR",
    os.path.abspath(os.path.join(PROJECT_ROOT, "..", "assets")),  # /home/js9s/子归家/assets
)

# ── File paths ───────────────────────────────────────────────
AUDIO_PROMPT_PATH = os.environ.get(
    "AUDIO_PROMPT_PATH",
    os.path.join(ASSETS_DIR, "std_ning.wav"),
)
GEN_AUDIO_SCRIPT = os.environ.get(
    "GEN_AUDIO_SCRIPT",
    os.path.join(PROJECT_ROOT, "gen_audio.py"),
)
GEN_VIDEO_SCRIPT = os.environ.get(
    "GEN_VIDEO_SCRIPT",
    os.path.join(PROJECT_ROOT, "gen_video.py"),
)
GEN_VIDEO_ORIG_SCRIPT = os.environ.get(
    "GEN_VIDEO_ORIG_SCRIPT",
    os.path.join(PROJECT_ROOT, "gen_video_orig.sh"),
)

# ── External tools ───────────────────────────────────────────
PYTHON_BIN = os.environ.get(
    "PYTHON_BIN",
    "__PYTHON_BIN__",
)
WHISPER_MODEL = os.environ.get(
    "WHISPER_MODEL",
    "__WHISPER_MODEL__",
)
WHISPER_OV_DEVICE = os.environ.get("WHISPER_OV_DEVICE", "CPU")
HY_MT_DIR = os.environ.get(
    "HY_MT_DIR",
    "__HY_MT_DIR__",
)
HY_MT_BACKEND = os.environ.get("HY_MT_BACKEND", "openvino")  # "openvino" | "pytorch"
RAPID_VIDEOCR_PIPELINE_SCRIPT = os.environ.get(
    "RAPID_VIDEOCR_PIPELINE_SCRIPT",
    os.path.join(PROJECT_ROOT, "rapid_videocr_pipeline.sh"),
)
RAPID_VIDEOCR_BIN = os.environ.get(
    "RAPID_VIDEOCR_BIN",
    "/home/js9s/.pyenv/versions/3.11.14/bin/rapid_videocr",
)

# ── Default ports ──────────────────────────────────────────
PORT = 18789

# ── Language code → full name mapping for translation ─────
LANG_MAP = {
    "ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "fi": "Finnish", "fr": "French",
    "he": "Hebrew", "hi": "Hindi", "it": "Italian", "ja": "Japanese",
    "ko": "Korean", "ms": "Malay", "nl": "Dutch", "no": "Norwegian",
    "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "sv": "Swedish",
    "sw": "Swahili", "tr": "Turkish", "zh": "Chinese",
    "vi": "Vietnamese", "th": "Thai", "id": "Indonesian",
}

# ── SMTP / Email ─────────────────────────────────────────────
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
SMTP_FROM = os.environ.get("SMTP_FROM", SMTP_USER)

# ── Valkey (Redis-compatible) ──────────────────────────────────
# Prefer VALKEY_URL env var (full connection URL).
# If not set, construct from host/port/db/password components —
# this avoids embedding the password in a Python string that could
# be logged or leaked.
VALKEY_URL = os.environ.get("VALKEY_URL", "")
VALKEY_HOST = os.environ.get("VALKEY_HOST", "localhost")
VALKEY_PORT = int(os.environ.get("VALKEY_PORT", "6379"))
VALKEY_DB = int(os.environ.get("VALKEY_DB", "0"))
VALKEY_PASSWORD = os.environ.get("VALKEY_PASSWORD", "")

# ── SRT filename → checkpoint step mapping ─────────────────
# Shared between jobqueue.py (clear_checkpoint_for_file) and
# chatterbox_server.py (SRT save/edit endpoint).
FILENAME_TO_CHECKPOINT_STEP = {
    "ocr_screen.srt": "ocr",
    "translated.srt": "translate",
    "whisper.srt": "whisper",
    "output_adjusted.srt": "audio",
}

# ── Checkpoint step ordering ─────────────────────────────────
# Canonical order of checkpoint steps. Used by jobqueue.py for
# invalidation and by jobs_tui.py for purge logic. The single
# definition here keeps both in sync.
CHECKPOINT_ORDER = [
    "download", "decompress", "trim", "extract_audio",
    "whisper", "ocr", "translate", "audio", "video",
]

MARKER_INTRO = "杨宁随缘开示"
MARKER_OUTRO = "子归家全体编制人员"
