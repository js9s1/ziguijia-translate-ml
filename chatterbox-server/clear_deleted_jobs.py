#!/usr/bin/env python3
"""
Administrative script to clear deleted jobs from the queue.

This script removes all jobs marked as "deleted" from the database
and removes their associated output directories from the filesystem.

Usage:
    python clear_deleted_jobs.py [--dry-run]
    
Options:
    --dry-run    Show what would be removed without actually removing anything
"""

import argparse
import os
import shutil
import sqlite3
import sys
from pathlib import Path

# Database and status constants
HERE = Path(__file__).resolve().parent
DB_FILE = HERE / "jobs.db"
DELETED_STATUS = "deleted"


def clear_deleted_jobs(dry_run=False):
    """Clear all deleted jobs and their output directories."""
    
    if not DB_FILE.exists():
        return {"success": False, "error": f"Database file not found: {DB_FILE}"}
    
    # Connect to database
    try:
        conn = sqlite3.connect(str(DB_FILE))
        conn.row_factory = sqlite3.Row
    except Exception as e:
        return {"success": False, "error": f"Failed to connect to database: {e}"}
    
    # Find all deleted jobs with their output directories
    try:
        rows = conn.execute(
            "SELECT access_code, output_dir FROM jobs WHERE status = ?",
            (DELETED_STATUS,)
        ).fetchall()
    except Exception as e:
        conn.close()
        return {"success": False, "error": f"Failed to query database: {e}"}
    
    if not rows:
        conn.close()
        return {"success": True, "message": "No deleted jobs found", "jobs_removed": 0, "dirs_removed": 0}
    
    jobs_removed = 0
    dirs_removed = 0
    errors = []
    
    if dry_run:
        print(f"Found {len(rows)} deleted jobs:")
        dirs_to_remove = 0
        
        for row in rows:
            access_code, output_dir = row["access_code"], row["output_dir"]
            print(f"  Job: {access_code}")
            
            if output_dir:
                if os.path.exists(output_dir):
                    print(f"    Would remove directory: {output_dir}")
                    dirs_to_remove += 1
                else:
                    print(f"    Directory not found: {output_dir}")
            else:
                print("    No output directory")
        
        conn.close()
        return {
            "success": True,
            "message": f"Would remove {len(rows)} jobs and {dirs_to_remove} directories",
            "jobs_removed": len(rows),
            "dirs_removed": dirs_to_remove
        }
    
    # Remove output directories
    for row in rows:
        access_code, output_dir = row["access_code"], row["output_dir"]
        
        if output_dir and os.path.exists(output_dir):
            try:
                shutil.rmtree(output_dir)
                dirs_removed += 1
                print(f"Removed output directory for deleted job {access_code}: {output_dir}")
            except Exception as e:
                error_msg = f"Failed to remove directory {output_dir} for job {access_code}: {e}"
                errors.append(error_msg)
                print(f"Error: {error_msg}")
    
    # Remove job entries from database
    try:
        cursor = conn.execute(
            "DELETE FROM jobs WHERE status = ?",
            (DELETED_STATUS,)
        )
        jobs_removed = cursor.rowcount
        conn.commit()
        print(f"Removed {jobs_removed} deleted job entries from database")
    except Exception as e:
        error_msg = f"Failed to remove job entries from database: {e}"
        errors.append(error_msg)
        print(f"Error: {error_msg}")
    finally:
        conn.close()
    
    result = {
        "success": len(errors) == 0,
        "jobs_removed": jobs_removed,
        "dirs_removed": dirs_removed,
    }
    
    if errors:
        result["errors"] = errors
        result["message"] = f"Completed with {len(errors)} errors"
    else:
        result["message"] = f"Successfully removed {jobs_removed} jobs and {dirs_removed} directories"
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Clear deleted jobs from the job queue")
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show what would be removed without actually removing anything"
    )
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN MODE - No changes will be made")
        print("=" * 50)
    else:
        print("Clearing deleted jobs...")
    
    result = clear_deleted_jobs(dry_run=args.dry_run)
    
    if result["success"]:
        print("✓ " + result["message"])
        if not args.dry_run:
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