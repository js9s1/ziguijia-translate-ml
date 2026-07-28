#!/usr/bin/env python3
"""
Background job to rebuild static SRT list pages for oldrun processing.

Run every 6 hours via fcron.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "chatterbox-server"))

from oldrun import build_all_static_srt

if __name__ == "__main__":
    build_all_static_srt()
