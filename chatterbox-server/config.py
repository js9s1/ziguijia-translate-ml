"""Centralized configuration for paths and environment variables."""

import os
import re

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

# ── ROCm environment for AMD Strix Halo APU (gfx1151) ────────
# ROCm 7.2+ has native support for gfx1151 — no GFX override needed.
# Must be set before any PyTorch/ROCm import.
if "LD_LIBRARY_PATH" in os.environ:
    os.environ["LD_LIBRARY_PATH"] = "/opt/rocm/rocm/lib" + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ["LD_LIBRARY_PATH"] else "")
else:
    os.environ["LD_LIBRARY_PATH"] = "/opt/rocm/rocm/lib"

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
# GEN_AUDIO_PYTHON: Python 3.11 for TTS subprocess (pyenv-managed)
GEN_AUDIO_PYTHON = os.environ.get(
    "GEN_AUDIO_PYTHON",
    os.path.expanduser("~/.pyenv/versions/3.11.14/bin/python3.11"),
)
# TRANSLATE_PYTHON: Python 3.11 for HY-MT translation subprocess (ROCm GPU)
TRANSLATE_PYTHON = os.environ.get(
    "TRANSLATE_PYTHON",
    os.path.expanduser("~/.pyenv/versions/3.11.14/bin/python3.11"),
)
# PYTHON_BIN: system Python for everything else (gen_video, download, etc.)
PYTHON_BIN = os.environ.get(
    "PYTHON_BIN",
    "/usr/bin/python3",
)
WHISPER_MODEL = os.environ.get(
    "WHISPER_MODEL",
    os.path.expanduser("~/.local/share/whisper-models/ggml-medium.bin"),
)
WHISPER_OV_DEVICE = os.environ.get("WHISPER_OV_DEVICE", "CPU")
HY_MT_DIR = os.environ.get(
    "HY_MT_DIR",
    "/home/ziguijia/src/HY-MT",
)
HY_MT_BACKEND = os.environ.get("HY_MT_BACKEND", "pytorch")  # "openvino" | "pytorch"
RAPID_VIDEOCR_PIPELINE_SCRIPT = os.environ.get(
    "RAPID_VIDEOCR_PIPELINE_SCRIPT",
    os.path.join(PROJECT_ROOT, "rapid_videocr_pipeline.sh"),
)
RAPID_VIDEOCR_BIN = os.environ.get(
    "RAPID_VIDEOCR_BIN",
    os.path.expanduser("~/.local/bin/rapid_videocr"),
)

# ── Default ports ──────────────────────────────────────────
PORT = 18789

from language_utils import LANG_MAP

# Backward-compatible re-export (canonical definition lives in language_utils.py)

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
    "download",
    "decompress",
    "trim",
    "extract_audio",
    "whisper",
    "ocr",
    "translate",
    "audio",
    "video",
]

MARKER_INTRO = "杨宁随缘开示"
MARKER_OUTRO = "子归家全体编制人员"


def validate_upload_filename(filename: str) -> None:
    """Raise ValueError if the filename is disallowed.

    - filenames starting with ``output`` (case‑insensitive) are reserved
      for pipeline‑generated files and must not be uploaded by users.
    """
    if not filename:
        raise ValueError("文件名为空")
    basename = os.path.basename(filename)
    if basename.lower().startswith("output"):
        raise ValueError(f"文件名不能以 'output' 开头，'{basename}' 是系统保留前缀")


_SCREEN_RECORD_RE = re.compile(
    r"(screen[\s\-_]*record|screencast|screenrec|screen[\s\-_]*(capture|shot)|RPReplay|录屏|屏幕录制)",
    re.IGNORECASE,
)


def is_screen_recording_filename(filename: str) -> bool:
    """Return True if the filename looks like a screen recording
    (iOS 'Screen Recording ...', Android 'Screencast ...', 录屏/屏幕录制, etc.)."""
    return bool(_SCREEN_RECORD_RE.search(os.path.basename(filename)))
