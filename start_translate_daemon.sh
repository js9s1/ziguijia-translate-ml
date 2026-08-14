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
# (it also exits after TRANSLATE_IDLE_TIMEOUT seconds idle).
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/rocm_env.sh"
export TRANSLATE_MAX_JOBS="${TRANSLATE_MAX_JOBS:-2}"

TRANSLATE_PYTHON="${TRANSLATE_PYTHON:-$HOME/.pyenv/versions/3.11.14/bin/python3.11}"

exec "$TRANSLATE_PYTHON" -u "$SCRIPT_DIR/translate_daemon.py" "$@"
