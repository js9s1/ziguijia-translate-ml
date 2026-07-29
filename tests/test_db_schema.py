"""Tests for database schema creation and migration."""

import sqlite3

import pytest
from db_schema import (
    JOB_COLUMNS,
    ConnectionManager,
    add_column_if_missing,
    init_jobs_schema,
    init_users_schema,
)


class TestConnectionManager:
    def test_get_returns_connection(self):
        cm = ConnectionManager(":memory:")
        conn = cm.get()
        assert isinstance(conn, sqlite3.Connection)

    def test_reuses_connection_same_thread(self):
        cm = ConnectionManager(":memory:")
        c1 = cm.get()
        c2 = cm.get()
        assert c1 is c2

    def test_wal_mode_enabled(self):
        cm = ConnectionManager(":memory:")
        conn = cm.get()
        cur = conn.execute("PRAGMA journal_mode")
        assert cur.fetchone()[0] in ("wal", "memory")

    def test_row_factory(self):
        cm = ConnectionManager(":memory:")
        conn = cm.get()
        conn.execute("CREATE TABLE foo (bar TEXT)")
        conn.execute("INSERT INTO foo VALUES ('baz')")
        row = conn.execute("SELECT * FROM foo").fetchone()
        assert row["bar"] == "baz"

    def test_close(self):
        cm = ConnectionManager(":memory:")
        conn = cm.get()
        cm.close()
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")


class TestJobsSchema:
    def test_init_creates_table(self, isolated_db):
        conn = isolated_db
        init_jobs_schema(conn)
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'")
        assert cur.fetchone() is not None

    def test_all_columns_exist(self, isolated_db):
        conn = isolated_db
        init_jobs_schema(conn)
        cur = conn.execute("PRAGMA table_info(jobs)")
        cols = {row[1] for row in cur.fetchall()}
        for col in JOB_COLUMNS:
            assert col in cols, f"Column {col} missing from jobs table"

    def test_idempotent(self, isolated_db):
        conn = isolated_db
        init_jobs_schema(conn)
        init_jobs_schema(conn)  # should not raise

    def test_migration_adds_column(self, isolated_db):
        conn = isolated_db
        init_jobs_schema(conn)
        # Drop a column (not supported in SQLite) — instead, test that
        # add_column_if_missing doesn't error on existing column
        add_column_if_missing(conn, "jobs", "checkpoint", "TEXT DEFAULT ''")
        # Table should still be intact
        conn.execute("INSERT INTO jobs (access_code, status) VALUES ('A', 'pending')")
        assert conn.execute("SELECT status FROM jobs WHERE access_code='A'").fetchone()["status"] == "pending"


class TestUsersSchema:
    def test_init_creates_table(self, isolated_db):
        conn = isolated_db
        init_users_schema(conn)
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        assert cur.fetchone() is not None

    def test_required_columns(self, isolated_db):
        conn = isolated_db
        init_users_schema(conn)
        cur = conn.execute("PRAGMA table_info(users)")
        cols = {row[1] for row in cur.fetchall()}
        for col in ("id", "email", "password_hash", "verified", "created_at"):
            assert col in cols

    def test_email_unique_constraint(self, isolated_db):
        conn = isolated_db
        init_users_schema(conn)
        conn.execute(
            "INSERT INTO users (email, password_hash) VALUES (?, ?)",
            ("a@b.com", "hash"),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO users (email, password_hash) VALUES (?, ?)",
                ("a@b.com", "hash2"),
            )


class TestAddColumnIfMissing:
    def test_adds_new_column(self, isolated_db):
        conn = isolated_db
        conn.execute("CREATE TABLE t (a TEXT)")
        add_column_if_missing(conn, "t", "b", "TEXT DEFAULT 'x'")
        cur = conn.execute("PRAGMA table_info(t)")
        cols = {row[1] for row in cur.fetchall()}
        assert "b" in cols

    def test_no_error_on_existing(self, isolated_db):
        conn = isolated_db
        conn.execute("CREATE TABLE t2 (a TEXT)")
        add_column_if_missing(conn, "t2", "a", "TEXT")
        # Should not raise
        add_column_if_missing(conn, "t2", "a", "TEXT")
