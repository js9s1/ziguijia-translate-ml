#!/bin/bash

echo "Stopping servers..."

pkill -f "gunicorn.*chatterbox_server:app" 2>/dev/null

sleep 1

if pgrep -f "gunicorn.*chatterbox" > /dev/null 2>&1; then
    echo "Warning: gunicorn still running, force killing..."
    pkill -9 -f "gunicorn" 2>/dev/null
fi

# Kill orphaned gen_audio subprocesses (spawned by gunicorn worker,
# survive because they're not in the same process group)
if pgrep -f "gen_audio.py" > /dev/null 2>&1; then
    echo "Killing orphaned gen_audio processes..."
    pkill -f "gen_audio.py" 2>/dev/null
fi

# Stop Valkey if it was started by start_server.sh (only if no other clients)
# Load password from .env
if [ -f "${HOME}/子归家/code_ml/.env" ]; then
    set -a; source "${HOME}/子归家/code_ml/.env"; set +a
fi
VALKEYPW="${VALKEY_PASSWORD:-}"

if valkey-cli -a "$VALKEYPW" --no-auth-warning ping > /dev/null 2>&1; then
    echo "Stopping Valkey..."
    valkey-cli -a "$VALKEYPW" --no-auth-warning shutdown nosave 2>/dev/null
fi

echo "Servers stopped"
ps aux | grep -E "gunicorn|valkey" | grep -v grep || echo "All stopped"