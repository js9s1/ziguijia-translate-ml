#!/bin/bash
# Start the warm TTS daemon (gen_audio_daemon.py) on Python 3.11 + ROCm env.
#
# Usage:
#   ./start_gen_audio_daemon.sh                 # foreground, MAX_JOBS=2
#   GEN_AUDIO_MAX_JOBS=2 ./start_gen_audio_daemon.sh
#   nohup ./start_gen_audio_daemon.sh >> ~/logs/gen_audio_daemon.log 2>&1 &
#
# The daemon keeps the Chatterbox TTS model(s) resident on GPU and serves
# text-to-wav requests over a Unix socket ($XDG_RUNTIME_DIR/gen_audio_daemon/).
# gen_audio.py jobs auto-detect the daemon and start it on demand when it is
# not running (it also exits after GEN_AUDIO_IDLE_TIMEOUT seconds idle).
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../rocm_env.sh"
export GEN_AUDIO_MAX_JOBS="${GEN_AUDIO_MAX_JOBS:-2}"

GEN_AUDIO_PYTHON="${GEN_AUDIO_PYTHON:-$HOME/.pyenv/versions/3.11.14/bin/python3.11}"

exec "$GEN_AUDIO_PYTHON" -u "$SCRIPT_DIR/gen_audio_daemon.py" "$@"
