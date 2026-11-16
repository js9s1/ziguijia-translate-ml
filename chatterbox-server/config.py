"""Centralized configuration for paths and environment variables."""

import os
import re

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

# ── ROCm environment for AMD Strix Halo APU (gfx1151) ────────
# ROCm 7.2+ has native support for gfx1151 — no GFX override needed.
# Must be set before any PyTorch/ROCm import.  Shared config lives in
# rocm.env (repo root), applied by rocm_env.setup().
import sys as _sys

_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rocm_env import setup as _rocm_setup  # noqa: E402

_rocm_setup()

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
# Chatterbox multilingual T3 checkpoint: "v3" (current) or "v2" (legacy).
# V3 requires chatterbox-tts from upstream git; PyPI 0.1.7 has no V3 support.
CHATTERBOX_T3_MODEL = os.environ.get("CHATTERBOX_T3_MODEL", "v3")

# ── VoxCPM2 (Cantonese / yue only) ───────────────────────────
# The 2B model is a separate engine used only when target_language == "yue".
# It is shared across daemon worker threads (one model instance, serialized
# generation) because two 2B instances would exhaust this host's RAM.
VOXCPM_MODEL = os.environ.get("VOXCPM_MODEL", "openbmb/VoxCPM2")
VOXCPM_CFG_VALUE = float(os.environ.get("VOXCPM_CFG_VALUE", "2.0"))
VOXCPM_INFERENCE_TIMESTEPS = int(os.environ.get("VOXCPM_INFERENCE_TIMESTEPS", "10"))
# -1 lets VoxCPM sample a fresh seed each request; 42 is deterministic.
VOXCPM_SEED = int(os.environ.get("VOXCPM_SEED", "42"))
VOXCPM_OPTIMIZE = os.environ.get("VOXCPM_OPTIMIZE", "0").lower() in ("1", "true", "yes")
VOXCPM_RETRY_BADCASE = os.environ.get("VOXCPM_RETRY_BADCASE", "0").lower() in ("1", "true", "yes")
GEN_AUDIO_SCRIPT = os.environ.get(
    "GEN_AUDIO_SCRIPT",
    os.path.join(PROJECT_ROOT, "gen_audio", "gen_audio.py"),
)
GEN_AUDIO_DAEMON_SCRIPT = os.environ.get(
    "GEN_AUDIO_DAEMON_SCRIPT",
    os.path.join(PROJECT_ROOT, "gen_audio", "gen_audio_daemon.py"),
)


def _gen_audio_daemon_runtime_dir():
    """Per-user tmpfs dir (0700) for the TTS daemon socket + pid."""
    base = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    if not base or not os.path.isdir(base):
        base = "/tmp"
    return os.path.join(base, "gen_audio_daemon")


GEN_AUDIO_DAEMON_SOCK = os.environ.get(
    "GEN_AUDIO_DAEMON_SOCK",
    os.path.join(_gen_audio_daemon_runtime_dir(), "daemon.sock"),
)
GEN_AUDIO_DAEMON_PID = os.environ.get(
    "GEN_AUDIO_DAEMON_PID",
    os.path.join(_gen_audio_daemon_runtime_dir(), "daemon.pid"),
)


def _translate_daemon_runtime_dir():
    """Per-user tmpfs dir (0700) for the translate daemon socket + pid."""
    base = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    if not base or not os.path.isdir(base):
        base = "/tmp"
    return os.path.join(base, "translate_daemon")


TRANSLATE_DAEMON_SCRIPT = os.environ.get(
    "TRANSLATE_DAEMON_SCRIPT",
    os.path.join(PROJECT_ROOT, "translate", "translate_daemon.py"),
)
TRANSLATE_DAEMON_SOCK = os.environ.get(
    "TRANSLATE_DAEMON_SOCK",
    os.path.join(_translate_daemon_runtime_dir(), "daemon.sock"),
)
TRANSLATE_DAEMON_PID = os.environ.get(
    "TRANSLATE_DAEMON_PID",
    os.path.join(_translate_daemon_runtime_dir(), "daemon.pid"),
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
# Primary ASR binary: whisper.cpp built with the Vulkan backend.  The pipeline
# forces fp32 (GGML_VK_DISABLE_F16=1) and enables flash-attention; if it fails
# it falls back to WHISPER_CPU_BIN with WHISPER_CPU_THREADS threads.
WHISPER_BIN = os.environ.get(
    "WHISPER_BIN",
    os.path.expanduser("~/.local/opt/whisper.cpp-vulkan/whisper-cli"),
)
# Directory holding libwhisper.so/libggml*.so for a relocatable build.  Only
# added to LD_LIBRARY_PATH when libwhisper.so.1 is actually present there.
WHISPER_LIB_DIR = os.environ.get(
    "WHISPER_LIB_DIR",
    os.path.dirname(os.path.realpath(WHISPER_BIN)),
)
WHISPER_CPU_BIN = os.environ.get("WHISPER_CPU_BIN", "/usr/bin/whisper-cli")
WHISPER_CPU_THREADS = int(os.environ.get("WHISPER_CPU_THREADS") or "16")
WHISPER_OV_DEVICE = os.environ.get("WHISPER_OV_DEVICE", "CPU")
# GPU gate for the primary Vulkan backend: skip Vulkan and use the CPU
# fallback while the iGPU is hot or a GPU daemon has active work.  Set an
# active threshold to 0 to disable that check.  Translate is gated because
# HY-MT falls back to GPU when the NPU fails; set it to 0 if your translate
# daemon stays on the NPU.
WHISPER_GPU_MAX_TEMP = float(
    os.environ.get("WHISPER_GPU_MAX_TEMP") or os.environ.get("GEN_AUDIO_IDLE_TEMPERATURE") or "76"
)
WHISPER_GATE_TTS_ACTIVE = int(os.environ.get("WHISPER_GATE_TTS_ACTIVE") or "1")
WHISPER_GATE_OCR_ACTIVE = int(os.environ.get("WHISPER_GATE_OCR_ACTIVE") or "1")
WHISPER_GATE_TRANSLATE_ACTIVE = int(os.environ.get("WHISPER_GATE_TRANSLATE_ACTIVE") or "1")
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
RAPID_VIDEOCR_DAEMON_CLIENT = os.environ.get(
    "RAPID_VIDEOCR_DAEMON_CLIENT",
    os.path.join(PROJECT_ROOT, "daemon_ocr_client.py"),
)


def _ocr_daemon_runtime_dir():
    """Per-user tmpfs dir for the shared OCR daemon socket."""
    base = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    if not base or not os.path.isdir(base):
        base = "/tmp"
    return os.path.join(base, "rapid_videocr_daemon")


RAPID_VIDEOCR_DAEMON_SOCK = os.environ.get(
    "RAPID_VIDEOCR_DAEMON_SOCK",
    os.path.join(_ocr_daemon_runtime_dir(), "daemon.sock"),
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

# ── gen_audio step timeouts ────────────────────────────────
# run_gen_audio_step (pipeline.py) guards the TTS subprocess with two
# timeouts: a stall limit (no new log output while the process is
# alive → hung) and a total cap scaled by SRT segment count.
GEN_AUDIO_STALL_TIMEOUT = float(os.environ.get("GEN_AUDIO_STALL_TIMEOUT", "900"))
GEN_AUDIO_SEGMENT_BUDGET = float(os.environ.get("GEN_AUDIO_SEGMENT_BUDGET", "180"))
GEN_AUDIO_MIN_TOTAL_TIMEOUT = float(os.environ.get("GEN_AUDIO_MIN_TOTAL_TIMEOUT", "7200"))

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
