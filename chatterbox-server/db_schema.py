"""Database schema creation and migration for jobs.db and users.db.

All CREATE TABLE and ALTER TABLE statements live here so that schema
changes are tracked in one place and errors are handled with precision
rather than swallowed wholesale.

Also provides ``ConnectionManager`` — a thin wrapper that owns a
thread-local SQLite connection with standard pragmas (WAL, busy_timeout).
Both ``JobQueue`` and ``UserManager`` delegate to it instead of
duplicating the same boilerplate.
"""

import sqlite3
import logging
import threading

logger = logging.getLogger(__name__)


# ── Connection management ────────────────────────────────────────

class ConnectionManager:
    """Thread-local SQLite connection with WAL mode and busy timeout.

    One instance per database file.  Safe to share across threads —
    each thread gets its own ``sqlite3.Connection``.
    """

    def __init__(self, db_file: str):
        self._db_file = db_file
        self._local = threading.local()

    def get(self) -> sqlite3.Connection:
        """Return the current thread's connection, creating it if needed."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_file)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def close(self):
        """Close the current thread's connection, if open.

        Call this after ``os.fork()`` so the child doesn't share the
        parent's connection.
        """
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None


# ── Schema migration helpers ─────────────────────────────────────


def _is_duplicate_column_error(exc: sqlite3.OperationalError) -> bool:
    """Return True if *exc* indicates the column already exists."""
    msg = str(exc).lower()
    return "duplicate column" in msg or "already exists" in msg


def add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, col_type: str):
    """Add *column* to *table* if it doesn't already exist.

    Raises ``sqlite3.OperationalError`` for anything other than a
    duplicate-column condition (e.g. syntax errors, missing table).
    """
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
    except sqlite3.OperationalError as e:
        if _is_duplicate_column_error(e):
            logger.debug("Column %s.%s already exists, skipping", table, column)
        else:
            raise


# ── jobs table (used by jobqueue.JobQueue) ───────────────────────

def _create_jobs_table(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            access_code TEXT PRIMARY KEY,
            srt_path TEXT,
            output_dir TEXT,
            temperature REAL,
            status TEXT,
            error TEXT,
            run_func_name TEXT,
            video_number TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


def _migrate_jobs_table(conn: sqlite3.Connection):
    text_cols = ("video_number", "video_file", "user_id", "text")
    for col in text_cols:
        add_column_if_missing(conn, "jobs", col, "TEXT")

    add_column_if_missing(conn, "jobs", "progress", "TEXT")
    add_column_if_missing(conn, "jobs", "blur", "TEXT DEFAULT 'yes'")
    add_column_if_missing(conn, "jobs", "target_language", "TEXT DEFAULT 'en'")
    add_column_if_missing(conn, "jobs", "cfg_weight", "REAL DEFAULT 0.5")
    add_column_if_missing(conn, "jobs", "exaggeration", "REAL DEFAULT 0.5")
    add_column_if_missing(conn, "jobs", "completed_at", "TIMESTAMP")
    add_column_if_missing(conn, "jobs", "failed_at", "TIMESTAMP")
    add_column_if_missing(conn, "jobs", "cancelled_at", "TIMESTAMP")
    add_column_if_missing(conn, "jobs", "status_changed_at", "TIMESTAMP")
    add_column_if_missing(conn, "jobs", "deleted_at", "TIMESTAMP")
    add_column_if_missing(conn, "jobs", "checkpoint", "TEXT DEFAULT ''")
    add_column_if_missing(conn, "jobs", "checkpoint_edited", "INTEGER DEFAULT 0")
    add_column_if_missing(conn, "jobs", "edited_srt_files", "TEXT DEFAULT ''")
    add_column_if_missing(conn, "jobs", "start_trim", "REAL DEFAULT NULL")
    add_column_if_missing(conn, "jobs", "end_trim", "REAL DEFAULT NULL")
    add_column_if_missing(conn, "jobs", "cached_path", "TEXT")
    add_column_if_missing(conn, "jobs", "filename", "TEXT")

    # Fix legacy UTC created_at vs local status_changed_at mismatch.
    # created_at was set by CURRENT_TIMESTAMP (UTC) while status_changed_at
    # used local time.  Align created_at to status_changed_at when available.
    conn.execute("""
        UPDATE jobs
        SET created_at = status_changed_at
        WHERE status_changed_at IS NOT NULL
          AND created_at IS NOT NULL
          AND created_at != status_changed_at
    """)


# ── users table (used by auth.UserManager) ──────────────────────

def _create_users_table(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            verified INTEGER DEFAULT 0,
            verification_code TEXT,
            reset_code TEXT,
            reset_code_expires REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


def _migrate_users_table(conn: sqlite3.Connection):
    add_column_if_missing(conn, "users", "reset_code", "TEXT")
    add_column_if_missing(conn, "users", "reset_code_expires", "REAL")


# ── public API ───────────────────────────────────────────────────

# Canonical column list for the jobs table. Shared between schema
# migration and query construction so there is a single source of truth.
JOB_COLUMNS = [
    "access_code", "srt_path", "output_dir", "temperature", "status",
    "error", "run_func_name", "video_number", "created_at", "video_file",
    "user_id", "text", "blur", "target_language", "cfg_weight",
    "exaggeration", "start_trim", "end_trim", "cached_path", "filename",
]


def init_jobs_schema(conn: sqlite3.Connection):
    """Create and migrate the ``jobs`` table."""
    _create_jobs_table(conn)
    _migrate_jobs_table(conn)
    conn.commit()


def init_users_schema(conn: sqlite3.Connection):
    """Create and migrate the ``users`` table."""
    _create_users_table(conn)
    _migrate_users_table(conn)
    conn.commit()
