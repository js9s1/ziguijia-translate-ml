#!/bin/bash

# cleanup_old_deleted_jobs.sh
# Remove deleted job outputs and database entries older than a specified time period.
# Default: jobs deleted more than 7 days ago.
#
# Usage:
#   ./cleanup_old_deleted_jobs.sh [--days N] [--dry-run] [--verbose] [--force]
#
# Options:
#   --days N      Number of days to keep deleted jobs (default: 7)
#   --dry-run     Show what would be removed without actually removing
#   --verbose     Print detailed output
#   --force       Remove ALL deleted jobs regardless of age
#   --help        Show this help message
#
# Examples:
#   # Remove deleted jobs older than 7 days (default)
#   ./cleanup_old_deleted_jobs.sh
#
#   # Remove deleted jobs older than 30 days
#   ./cleanup_old_deleted_jobs.sh --days 30
#
#   # Preview what would be removed
#   ./cleanup_old_deleted_jobs.sh --dry-run --verbose
#
#   # Force clean ALL deleted jobs (use with caution!)
#   ./cleanup_old_deleted_jobs.sh --force
#
#   # Daily cleanup via cron (run at 2 AM daily):
#   0 2 * * * /path/to/chatterbox-server/cleanup_old_deleted_jobs.sh --days 7 >> /var/log/job_cleanup.log 2>&1

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DB_FILE="${SCRIPT_DIR}/chatterbox-server/jobs.db"

# ─── Default values ───────────────────────────────────────────────
DAYS_KEEP=7
DRY_RUN=false
VERBOSE=false
FORCE_CLEAN=false

# ─── Color codes for output ───────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ─── Helper functions ─────────────────────────────────────────────
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_verbose() {
    if [[ "$VERBOSE" == true ]]; then
        echo -e "[DEBUG] $1"
    fi
}

show_help() {
    head -20 "$0" | grep "^#" | sed 's/^# \?//'
    exit 0
}

# ─── Parse arguments ──────────────────────────────────────────────
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --days)
                DAYS_KEEP="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --force)
                FORCE_CLEAN=true
                shift
                ;;
            --help)
                show_help
                ;;
            *)
                log_error "Unknown argument: $1"
                show_help
                ;;
        esac
    done

    # Validate DAYS_KEEP is a positive integer (unless --force is used)
    if [[ "$FORCE_CLEAN" == false ]]; then
        if ! [[ "$DAYS_KEEP" =~ ^[0-9]+$ ]] || [[ "$DAYS_KEEP" -lt 1 ]]; then
            log_error "Invalid days value: $DAYS_KEEP (must be positive integer)"
            exit 1
        fi
    fi
}

# ─── Check dependencies ──────────────────────────────────────────
check_dependencies() {
    if ! command -v sqlite3 &> /dev/null; then
        log_error "sqlite3 command not found. Please install sqlite3."
        exit 1
    fi

    if [[ ! -f "$DB_FILE" ]]; then
        log_error "Database file not found: $DB_FILE"
        exit 1
    fi
}

# ─── Calculate cutoff timestamp ──────────────────────────────────
get_cutoff_date() {
    local days_ago
    # Calculate date N days ago in ISO format
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux: GNU date
        days_ago=$(date -d "${DAYS_KEEP} days ago" +"%Y-%m-%d %H:%M:%S")
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS: BSD date
        days_ago=$(date -v-"${DAYS_KEEP}"d +"%Y-%m-%d %H:%M:%S")
    else
        # Fallback: use python for portability
        days_ago=$(python3 -c "from datetime import datetime, timedelta; print((datetime.now() - timedelta(days=$DAYS_KEEP)).strftime('%Y-%m-%d %H:%M:%S'))")
    fi
    echo "$days_ago"
}

# ─── Find old deleted jobs ───────────────────────────────────────
find_old_deleted_jobs() {
    local cutoff_date="$1"

    # If force clean, return all deleted jobs
    if [[ "$FORCE_CLEAN" == true ]]; then
        sqlite3 -separator '|' "$DB_FILE" <<EOF
SELECT access_code, output_dir, 
       COALESCE(deleted_at, created_at, 'unknown') as effective_date
FROM jobs
WHERE status = 'deleted'
ORDER BY effective_date ASC;
EOF
        return
    fi

    # Find jobs where:
    # 1. deleted_at exists and is older than cutoff, OR
    # 2. deleted_at is NULL but created_at is older than cutoff (legacy jobs), OR
    # 3. Both timestamps are NULL (very old legacy jobs, always clean up)
    sqlite3 -separator '|' "$DB_FILE" <<EOF
SELECT access_code, output_dir, 
       COALESCE(deleted_at, created_at, 'unknown') as effective_date
FROM jobs
WHERE status = 'deleted'
  AND (
    (deleted_at IS NOT NULL AND deleted_at < '${cutoff_date}')
    OR 
    (deleted_at IS NULL AND created_at IS NOT NULL AND created_at < '${cutoff_date}')
    OR
    (deleted_at IS NULL AND created_at IS NULL)
  )
ORDER BY effective_date ASC;
EOF
}

# ─── Count total deleted jobs ────────────────────────────────────
count_all_deleted_jobs() {
    sqlite3 "$DB_FILE" "SELECT COUNT(*) FROM jobs WHERE status = 'deleted';"
}

# ─── Remove output directory ─────────────────────────────────────
remove_output_dir() {
    local dir="$1"
    local job_id="$2"

    if [[ -z "$dir" ]] || [[ "$dir" == "NULL" ]]; then
        log_verbose "No output directory for job $job_id"
        return 0
    fi

    if [[ ! -d "$dir" ]]; then
        log_verbose "Directory already missing: $dir"
        return 0
    fi

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would remove directory: $dir"
        return 0
    fi

    if rm -rf "$dir" 2>/dev/null; then
        log_verbose "Removed directory: $dir"
        return 0
    else
        log_warn "Failed to remove directory: $dir"
        return 1
    fi
}

# ─── Delete job from database ────────────────────────────────────
delete_job_from_db() {
    local job_id="$1"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would delete job from database: $job_id"
        return 0
    fi

    sqlite3 "$DB_FILE" "DELETE FROM jobs WHERE access_code = '${job_id}';"
    return 0
}

# ─── Main cleanup function ───────────────────────────────────────
main_cleanup() {
    local cutoff_date
    cutoff_date=$(get_cutoff_date)

    local total_deleted
    total_deleted=$(count_all_deleted_jobs)
    log_info "Total deleted jobs in database: $total_deleted"
    log_info "Cutoff date (jobs deleted before): $cutoff_date"
    log_info "Keeping deleted jobs for: $DAYS_KEEP days"
    echo ""

    if [[ "$total_deleted" -eq 0 ]]; then
        log_info "No deleted jobs found in database."
        return 0
    fi

    # Get jobs to process
    local jobs_data
    jobs_data=$(find_old_deleted_jobs "$cutoff_date")

    if [[ -z "$jobs_data" ]]; then
        log_info "No deleted jobs older than $DAYS_KEEP days found."
        return 0
    fi

    # Count jobs to process
    local job_count
    job_count=$(echo "$jobs_data" | wc -l)
    log_info "Found $job_count deleted job(s) older than $DAYS_KEEP days"
    echo ""

    if [[ "$DRY_RUN" == true ]]; then
        log_warn "DRY RUN MODE - No changes will be made"
        echo "=========================================="
    fi

    local removed_count=0
    local error_count=0

    # Process each job
    while IFS='|' read -r access_code output_dir deleted_at; do
        # Skip empty lines
        [[ -z "$access_code" ]] && continue

        log_verbose "Processing job: $access_code (deleted: $deleted_at)"

        # Remove output directory
        if ! remove_output_dir "$output_dir" "$access_code"; then
            ((error_count++))
            continue
        fi

        # Delete from database
        if ! delete_job_from_db "$access_code"; then
            ((error_count++))
            continue
        fi

        ((++removed_count))
        log_info "Processed job: $access_code"
    done <<< "$jobs_data"

    echo ""
    echo "=========================================="

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would remove: $removed_count job(s)"
        [[ "$error_count" -gt 0 ]] && log_warn "[DRY RUN] Errors encountered: $error_count"
    else
        log_info "Cleanup complete!"
        log_info "Jobs removed: $removed_count"
        [[ "$error_count" -gt 0 ]] && log_warn "Errors encountered: $error_count"
    fi

    return "$error_count"
}

# ─── Entry point ─────────────────────────────────────────────────
main() {
    parse_args "$@"
    check_dependencies
    main_cleanup
}

main "$@"
