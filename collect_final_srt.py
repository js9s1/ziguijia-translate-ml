#!/usr/bin/env python3
"""Collect the final modified SRT from every completed job to a top-level
output directory so users can browse/download them easily.

The "final modified SRT" is ``output_adjusted.srt`` — the timing-adjusted
SRT produced by the audio pipeline (or copied back to the output dir by
``run_video_ckpt``).  Each collected file is named::

    {access_code}_output-final-modified.srt

Run from the project root (``code_ml/``) or from anywhere; paths are
resolved relative to the project root.

Usage::

    python collect_final_srt.py          # copy all final SRTs
    python collect_final_srt.py --dry    # preview only
"""

import argparse
import os
import shutil
import sqlite3
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_ROOT, "chatterbox-server", "jobs.db")
OUTPUT_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "output_srt")

# The name used for the final modified SRT in the top-level output dir
FINAL_FILENAME = "output-final-modified.srt"

# The source SRT file inside each job's output directory
SOURCE_SRT = "output_adjusted.srt"


def collect(dry_run: bool = False) -> int:
    """Copy all final SRTs to *OUTPUT_DIR*.  Returns count of files copied."""
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT access_code, output_dir, video_number, video_file, run_func_name "
        "FROM jobs WHERE status = 'completed' AND output_dir IS NOT NULL "
        "ORDER BY created_at DESC"
    ).fetchall()
    conn.close()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    copied = 0
    skipped_reason: dict[str, int] = {}

    for row in rows:
        ac = row["access_code"]
        output_dir = row["output_dir"]
        video_number = row["video_number"] or ""
        video_file = row["video_file"] or ""

        src_path = os.path.join(output_dir, SOURCE_SRT)
        if not os.path.isfile(src_path):
            skipped_reason["no_output_adjusted_srt"] = \
                skipped_reason.get("no_output_adjusted_srt", 0) + 1
            continue

        # Build a descriptive filename
        label_parts = []
        if video_number:
            label_parts.append(video_number)
        label_parts.append(ac)
        label = "_".join(label_parts)

        dst_name = f"{label}_{FINAL_FILENAME}"
        dst_path = os.path.join(OUTPUT_DIR, dst_name)

        if dry_run:
            print(f"  [dry-run] would copy: {src_path}")
            print(f"             → {dst_path}")
            copied += 1
            continue

        try:
            shutil.copy2(src_path, dst_path)
            print(f"  ✓ {ac}: {src_path}")
            print(f"      → {dst_path}")
            copied += 1
        except OSError as e:
            print(f"  ✗ {ac}: {e}", file=sys.stderr)
            skipped_reason["copy_error"] = \
                skipped_reason.get("copy_error", 0) + 1

    # Summary
    total = len(rows)
    print()
    print(f"── Summary ──────────────────────────────────")
    print(f"  Total completed jobs with output_dir: {total}")
    print(f"  Copied: {copied}")
    for reason, count in sorted(skipped_reason.items()):
        print(f"  Skipped ({reason}): {count}")

    if copied:
        print(f"  Output directory: {OUTPUT_DIR}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Collect final modified SRTs to top-level output directory."
    )
    parser.add_argument(
        "--dry", "-n", action="store_true",
        help="Preview what would be copied without copying."
    )
    args = parser.parse_args()

    if args.dry:
        print(f"DRY RUN — no files will be copied\n")
    else:
        print(f"Collecting final SRTs → {OUTPUT_DIR}\n")

    return collect(dry_run=args.dry)


if __name__ == "__main__":
    sys.exit(main())
