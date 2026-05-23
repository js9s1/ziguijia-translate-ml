"""Centralized configuration for paths and environment variables."""

import os

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
HY_MT_DIR = os.environ.get(
    "HY_MT_DIR",
    "__HY_MT_DIR__",
)
RAPID_VIDEOCR_PIPELINE_SCRIPT = os.environ.get(
    "RAPID_VIDEOCR_PIPELINE_SCRIPT",
    os.path.join(PROJECT_ROOT, "rapid_videocr_pipeline.sh"),
)
RAPID_VIDEOCR_BIN = os.environ.get(
    "RAPID_VIDEOCR_BIN",
    "/home/js9s/.local/bin/rapid_videocr",
)

# ── Default ports ──────────────────────────────────────────
PORT = 18789
