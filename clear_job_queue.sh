#!/bin/bash

echo "Clearing job queue..."

pkill -f gen_video_orig.sh 2>/dev/null
pkill -f gen_audio.py 2>/dev/null

cd ${HOME}/子归家/code_ml/chatterbox-server

echo "Removing jobs database..."
rm -f jobs.db

echo "Job queue cleared."
echo ""
echo "Log files:"
echo "  - /tmp/gunicorn.log"
echo "  - ${HOME}/子归家/code_ml/chatterbox-server/access.log"
echo "  - ${HOME}/子归家/code_ml/chatterbox-server/error.log"
echo ""
echo "Note: Restart gunicorn to reset the job queue worker."