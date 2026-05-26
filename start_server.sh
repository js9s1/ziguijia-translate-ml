#!/bin/bash

cd ${HOME}/子归家/code_ml/chatterbox-server

export NING_SERVER_URL="http://127.0.0.1:18789"
export PYTHONPATH="${HOME}/子归家/code_ml/chatterbox-server:$PYTHONPATH"

# SMTP config for email verification
export SMTP_HOST="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USER="your-email@example.com"
export SMTP_PASS='your-app-password'
export SMTP_FROM="your-email@example.com"
export FLASK_SECRET_KEY="chatterbox-fixed-secret-key-2024"
export HF_TOKEN="$(cat ${HOME}/src/chatterbox/hf_t)"

# ── RapidVideOCR ────────────────────────────────────────────
export RAPID_VIDEOCR_BIN="${HOME}/.local/bin/rapid_videocr"

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

echo "Starting chatterbox server..."

${HOME}/.pyenv/versions/3.11.14/bin/gunicorn \
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
