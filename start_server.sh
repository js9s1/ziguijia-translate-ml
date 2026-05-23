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

# ROCm 7.x doesn't ship gfx900 kernels needed for this iGPU (gfx90c).
# Force CPU fallback — `_choose_device` in audio_utils handles this.
export HIP_VISIBLE_DEVICES=""

# PyTorch memory management — enable expandable segments to reduce fragmentation
export PYTORCH_ALLOC_CONF=expandable_segments:True

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

echo "Starting chatterbox server..."

${HOME}/.pyenv/versions/3.11.14/bin/gunicorn \
    --access-logfile ${HOME}/logs/chatterbox-server/access.log \
    --error-logfile ${HOME}/logs/chatterbox-server/error.log \
    --log-level info \
    -w 1 -k gthread --threads 8 --timeout 300 -b 0.0.0.0:5600 'chatterbox_server:app' --daemon --pid /tmp/gunicorn.pid

sleep 2

if pgrep -f gunicorn > /dev/null; then
    echo "Server started on port 5600"
    echo "Logs:"
    echo "  - ${HOME}/logs/chatterbox-server/ning_server.log"
    echo "  - ${HOME}/logs/chatterbox-server/access.log"
    echo "  - ${HOME}/logs/chatterbox-server/error.log"
else
    echo "Failed to start server. Check logs."
fi
