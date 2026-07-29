#!/usr/bin/env python3
"""
Background job to rebuild static SRT list pages for oldrun processing.

Run every 6 hours via fcron.
"""

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "chatterbox-server"))

from oldrun import build_all_static_srt

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("build_srt_pages started")
    build_all_static_srt()
    logger.info("build_srt_pages finished")
