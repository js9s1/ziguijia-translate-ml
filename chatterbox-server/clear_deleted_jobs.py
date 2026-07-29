#!/usr/bin/env python3
"""
Administrative script to clear deleted jobs from the queue.

Delegates to ``JobQueue.clear_job_queue()`` so the logic lives in one place.

Usage:
    python clear_deleted_jobs.py [--dry-run]

Options:
    --dry-run    Show what would be removed without actually removing anything
"""

import argparse
import sys

from jobqueue import get_job_queue


def main():
    parser = argparse.ArgumentParser(description="Clear deleted jobs from the job queue")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be removed without actually removing anything"
    )
    args = parser.parse_args()

    jq = get_job_queue()
    result = jq.clear_job_queue(dry_run=args.dry_run)

    if args.dry_run:
        print("DRY RUN MODE - No changes will be made")
        print("=" * 50)

    if result["success"]:
        print("✓ " + result["message"])
        print(f"  Jobs removed: {result['jobs_removed']}")
        print(f"  Directories removed: {result['dirs_removed']}")
    else:
        print("✗ " + result.get("message", "Failed"))
        if "error" in result:
            print(f"  Error: {result['error']}")
        if "errors" in result:
            for error in result["errors"]:
                print(f"  Error: {error}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
