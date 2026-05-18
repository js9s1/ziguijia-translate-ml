#!/bin/bash

echo "Stopping servers..."

pkill -f "gunicorn.*chatterbox_server:app" 2>/dev/null

sleep 1

if pgrep -f "gunicorn.*chatterbox" > /dev/null 2>&1; then
    echo "Warning: gunicorn still running, force killing..."
    pkill -9 -f "gunicorn" 2>/dev/null
fi

echo "Servers stopped"
ps aux | grep -E "gunicorn" | grep -v grep || echo "All stopped"