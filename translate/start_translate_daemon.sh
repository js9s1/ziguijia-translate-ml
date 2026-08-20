#!/bin/bash
# Start the warm translate daemon (translate_daemon.py) on Python 3.11 + ROCm env.
#
# Usage:
#   ./start_translate_daemon.sh                 # foreground, MAX_JOBS=2
#   TRANSLATE_MAX_JOBS=2 ./start_translate_daemon.sh
#   nohup ./start_translate_daemon.sh >> ~/logs/translate_daemon.log 2>&1 &
#
# The daemon keeps the HY-MT model resident on GPU and translates SRT files
# over a Unix socket ($XDG_RUNTIME_DIR/translate_daemon/).  translate_srt.py
# jobs auto-detect the daemon and start it on demand when it is not running
# (it also exits once the GPU is hot while idle — TRANSLATE_IDLE_TEMPERATURE).
set -e

DAEMON_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DAEMON_DIR/../rocm_env.sh"
export TRANSLATE_MAX_JOBS="${TRANSLATE_MAX_JOBS:-2}"
# Slot 0 runs on the NPU (npu-engine.service), remaining slots on the GPU.
export TRANSLATE_NPU_SLOT="${TRANSLATE_NPU_SLOT:-1}"

TRANSLATE_PYTHON="${TRANSLATE_PYTHON:-$HOME/.pyenv/versions/3.11.14/bin/python3.11}"

exec "$TRANSLATE_PYTHON" -u "$DAEMON_DIR/translate_daemon.py" "$@"
