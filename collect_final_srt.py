#!/usr/bin/env python3
"""For every completed job that has ``output_adjusted.srt`` (the final
SRT from the audio pipeline), copy it as ``output-final-modified.srt``
at the job's own top-level output directory so users can find and
download it easily.

Usage::

    python collect_final_srt.py          # perform the copies
    python collect_final_srt.py --dry    # preview only
"""

import argparse
import os
import shutil
import sqlite3
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_ROOT, "chatterbox-server", "jobs.db")

SOURCE_NAME = "output_adjusted.srt"
DEST_NAME = "output-final-modified.srt"


def _find_srt(output_dir: str) -> str | None:
    """Find ``output_adjusted.srt`` anywhere inside *output_dir*.

    Preference order:
      1. Directly at the top level  (e.g. ``{output_dir}/output_adjusted.srt``)
      2. Inside ``audio_tracks/``    (e.g. ``{output_dir}/audio_tracks/output_adjusted.srt``)
      3. Inside ``audio/``           (e.g. ``{output_dir}/audio/output_adjusted.srt``)
    """
    for sub in ("", "audio_tracks", "audio"):
        candidate = os.path.join(output_dir, sub, SOURCE_NAME)
        if os.path.isfile(candidate):
            return candidate
    return None


def collect(dry_run: bool = False) -> int:
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT access_code, output_dir FROM jobs "
        "WHERE status = 'completed' AND output_dir IS NOT NULL "
        "ORDER BY created_at DESC"
    ).fetchall()
    conn.close()

    copied = 0
    skipped_reason: dict[str, int] = {}

    for row in rows:
        ac = row["access_code"]
        output_dir = row["output_dir"]

        src_path = _find_srt(output_dir)
        if src_path is None:
            skipped_reason["no_output_adjusted_srt"] = skipped_reason.get("no_output_adjusted_srt", 0) + 1
            continue

        dst_path = os.path.join(output_dir, DEST_NAME)
        if dry_run:
            print(f"  [dry-run] {ac}: {src_path}")
            print(f"             → {dst_path}")
            copied += 1
            continue

        try:
            shutil.copy2(src_path, dst_path)
            print(f"  ✓ {ac}: {dst_path}")
            copied += 1
        except OSError as e:
            print(f"  ✗ {ac}: {e}", file=sys.stderr)
            skipped_reason["copy_error"] = skipped_reason.get("copy_error", 0) + 1

    total = len(rows)
    print()
    print("── Summary ──────────────────────────────────")
    print(f"  Completed jobs with output_dir: {total}")
    print(f"  output-final-modified.srt created: {copied}")
    for reason, count in sorted(skipped_reason.items()):
        print(f"  Skipped ({reason}): {count}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Create output-final-modified.srt in each completed job's directory.")
    parser.add_argument("--dry", "-n", action="store_true", help="Preview without copying.")
    args = parser.parse_args()
    if args.dry:
        print("DRY RUN\n")
    return collect(dry_run=args.dry)


if __name__ == "__main__":
    sys.exit(main())
