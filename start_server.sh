#!/bin/bash

cd ${HOME}/子归家/code_ml/chatterbox-server

export PYTHONPATH="${HOME}/子归家/code_ml/chatterbox-server:$PYTHONPATH"

export FLASK_SECRET_KEY="chatterbox-fixed-secret-key-2024"
export HF_TOKEN="$(cat ${HOME}/src/chatterbox/hf_t)"

# PyTorch memory management — enable expandable segments to reduce fragmentation
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ROCm: override gfx version for AMD Renoir iGPU (gfx90c → gfx900)
source "${HOME}/子归家/code_ml/rocm_env.sh"

#if ! docker ps --format '{{.Names}}' | grep -q qdrant; then
#    echo "Starting qdrant..."
#    docker rm qdrant 2>/dev/null
#    docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_data:/qdrant qdrant/qdrant
#    sleep 3
#fi
#
#echo "Checking if indexing is needed..."
#COLLECTION_STATUS=$(curl -s "http://localhost:6333/collections/video_transcriptions")
#POINTS_COUNT=$(echo "$COLLECTION_STATUS" | grep -o '"points_count":[0-9]*' | grep -o '[0-9]*')
#if [ -n "$POINTS_COUNT" ] && [ "$POINTS_COUNT" -gt 0 ]; then
#    echo "Collection exists with $POINTS_COUNT points, skipping indexing."
#else
#    echo "Collection not found or empty. Skipping indexing."
#fi
#
#if ! curl -s "http://localhost:18789/health" > /dev/null 2>&1; then
#    echo "Starting NingInferenceServer..."
#    nohup ${HOME}/.pyenv/versions/3.11.14/bin/python3.11 ${HOME}/子归家/code/chatterbox-server/ning_inference_server.py \
#        --port 18789 \
#        --audio_prompt ${HOME}/子归家/assets/std_ning.wav \
#        --device cpu \
#        > ${HOME}/logs/chatterbox-server/ning_server.log 2>&1 &
#    sleep 5
#else
#    echo "NingInferenceServer already running"
#fi

# Kill any existing gunicorn on our port to avoid bind conflicts
PID_FILE=/tmp/gunicorn.pid
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Stopping old gunicorn (PID $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null
        sleep 1
    fi
fi
# Also kill any stray gunicorn bound to our port
for OLD_PID in $(lsof -ti :5600 2>/dev/null || true); do
    echo "Killing process $OLD_PID on port 5600..."
    kill "$OLD_PID" 2>/dev/null
done
sleep 1

# Start Valkey if not already running
# Load password from .env
if [ -f "${HOME}/子归家/code_ml/.env" ]; then
    set -a; source "${HOME}/子归家/code_ml/.env"; set +a
fi
VALKEYPW="${VALKEY_PASSWORD:-}"

if ! valkey-cli -a "$VALKEYPW" --no-auth-warning ping > /dev/null 2>&1; then
    echo "Starting Valkey..."
    if [ -n "$VALKEYPW" ]; then
        valkey-server --daemonize yes --loglevel warning --requirepass "$VALKEYPW"
    else
        valkey-server --daemonize yes --loglevel warning
    fi
    # Clean up stale dump files that can crash valkey on startup
    rm -f dump.rdb /home/js9s/子归家/code_ml/dump.rdb
    # Wait for valkey to be ready (may take a few seconds)
    for i in 1 2 3 4 5; do
        sleep 1
        if valkey-cli -a "$VALKEYPW" --no-auth-warning ping > /dev/null 2>&1; then
            echo "Valkey started"
            break
        fi
    done
    if ! valkey-cli -a "$VALKEYPW" --no-auth-warning ping > /dev/null 2>&1; then
        echo "Warning: Valkey failed to start — server will use in-memory fallbacks"
    fi
else
    echo "Valkey already running"
fi

echo "Starting chatterbox server..."

${HOME}/.local/bin/gunicorn \
    -c ${HOME}/子归家/code_ml/chatterbox-server/gunicorn_config.py \
    --access-logfile ${HOME}/logs/chatterbox-server/access.log \
    --error-logfile ${HOME}/logs/chatterbox-server/error.log \
    --log-level info \
    -w 1 -k gthread --threads 8 --timeout 300 -b 0.0.0.0:5600 'chatterbox_server:app' --daemon --pid "$PID_FILE"

sleep 2

if pgrep -f "gunicorn.*chatterbox_server" > /dev/null; then
    echo "Server started on port 5600"
    echo "Logs:"
    echo "  - ${HOME}/logs/chatterbox-server/ning_server.log"
    echo "  - ${HOME}/logs/chatterbox-server/access.log"
    echo "  - ${HOME}/logs/chatterbox-server/error.log"
else
    echo "Failed to start server. Check logs."
fi
