"""Tests for jobs_tui.py — load_jobs, State, render_summary, show_user_jobs_tui."""

import os
import sys
import sqlite3
import tempfile
import importlib

import pytest

JOBS_SCHEMA = """
CREATE TABLE jobs (
    access_code TEXT PRIMARY KEY,
    srt_path TEXT,
    output_dir TEXT,
    temperature REAL,
    status TEXT,
    error TEXT,
    run_func_name TEXT,
    video_number TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    video_file TEXT,
    user_id INTEGER,
    text TEXT,
    progress TEXT,
    blur TEXT DEFAULT 'yes',
    target_language TEXT DEFAULT 'en',
    cfg_weight REAL DEFAULT 0.5,
    exaggeration REAL DEFAULT 0.5,
    completed_at TIMESTAMP,
    failed_at TIMESTAMP,
    cancelled_at TIMESTAMP,
    deleted_at TIMESTAMP,
    checkpoint TEXT DEFAULT '',
    checkpoint_edited INTEGER DEFAULT 0,
    edited_srt_files TEXT DEFAULT '',
    status_changed_at TIMESTAMP,
    start_trim REAL DEFAULT 12.25,
    end_trim REAL DEFAULT 40.0,
    cached_path TEXT,
    filename TEXT
)
"""

USERS_SCHEMA = """
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    verified INTEGER DEFAULT 0,
    verification_code TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reset_code TEXT,
    reset_code_expires TEXT
)
"""


@pytest.fixture(scope="session")
def _tui_env():
    """Bootstrap temp databases so jobs_tui can be imported.

    Because DB paths are resolved at module level, this fixture MUST run
    before the first import of jobs_tui.
    """
    tmp = tempfile.mkdtemp(prefix="tuitest_")
    jobs_db = os.path.join(tmp, "jobs.db")
    users_db = os.path.join(tmp, "users.db")

    # Create schema
    conn = sqlite3.connect(jobs_db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(JOBS_SCHEMA)
    conn.close()

    conn = sqlite3.connect(users_db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(USERS_SCHEMA)
    conn.close()

    old_jobs = os.environ.get("JOBS_DB")
    old_users = os.environ.get("USERS_DB")
    os.environ["JOBS_DB"] = jobs_db
    os.environ["USERS_DB"] = users_db

    # Force reimport in case another test already cached it
    for mod in list(sys.modules):
        if mod in ("jobs_tui", "user_tui") or mod.startswith("jobs_tui."):
            sys.modules.pop(mod, None)

    yield jobs_db, users_db

    if old_jobs:
        os.environ["JOBS_DB"] = old_jobs
    else:
        os.environ.pop("JOBS_DB", None)
    if old_users:
        os.environ["USERS_DB"] = old_users
    else:
        os.environ.pop("USERS_DB", None)

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def tuimod(_tui_env):
    """Import jobs_tui after environment is set up."""
    sys.path.insert(0, os.getcwd())
    import jobs_tui
    return jobs_tui


@pytest.fixture
def seed_db(_tui_env):
    """Seed the temp database with test data."""
    jobs_db, users_db = _tui_env

    conn = sqlite3.connect(users_db)
    conn.execute("DELETE FROM users")
    conn.execute("INSERT INTO users (id, email, password_hash, verified) VALUES (1, 'alice@test.com', 'x', 1)")
    conn.execute("INSERT INTO users (id, email, password_hash, verified) VALUES (2, 'bob@test.com', 'x', 1)")
    conn.execute("INSERT INTO users (id, email, password_hash, verified) VALUES (3, 'carol@test.com', 'x', 0)")
    conn.commit()
    conn.close()

    conn = sqlite3.connect(jobs_db)
    conn.execute("DELETE FROM jobs")
    # User 1: 3 jobs
    conn.execute(
        "INSERT INTO jobs (access_code, run_func_name, status, user_id, created_at, output_dir) "
        "VALUES ('AAA00001', '_run_gen_audio', 'completed', 1, '2024-01-01 10:00:00', '/tmp/a1')"
    )
    conn.execute(
        "INSERT INTO jobs (access_code, run_func_name, status, user_id, created_at, output_dir) "
        "VALUES ('AAA00002', '_run_video_job', 'processing', 1, '2024-01-02 10:00:00', '/tmp/a2')"
    )
    conn.execute(
        "INSERT INTO jobs (access_code, run_func_name, status, user_id, created_at, output_dir) "
        "VALUES ('AAA00003', '_run_tts_job', 'failed', 1, '2024-01-03 10:00:00', '/tmp/a3')"
    )
    # User 2: 2 jobs
    conn.execute(
        "INSERT INTO jobs (access_code, run_func_name, status, user_id, created_at, output_dir) "
        "VALUES ('BBB00001', '_run_gen_audio', 'pending', 2, '2024-01-04 10:00:00', '/tmp/b1')"
    )
    conn.execute(
        "INSERT INTO jobs (access_code, run_func_name, status, user_id, created_at, output_dir) "
        "VALUES ('BBB00002', '_run_gen_audio', 'completed', 2, '2024-01-05 10:00:00', '/tmp/b2')"
    )
    # User 3: 1 job
    conn.execute(
        "INSERT INTO jobs (access_code, run_func_name, status, user_id, created_at, output_dir) "
        "VALUES ('CCC00001', '_run_video_job', 'cancelled', 3, '2024-01-06 10:00:00', '/tmp/c1')"
    )
    conn.commit()
    conn.close()


# ── load_jobs tests ────────────────────────────────────────

class TestLoadJobsBackwardCompat:
    def test_load_all(self, tuimod, seed_db):
        jobs, total = tuimod.load_jobs()
        assert total == 6
        assert len(jobs) == 6

    def test_load_with_limit(self, tuimod, seed_db):
        jobs, total = tuimod.load_jobs(limit=3)
        assert total == 6
        assert len(jobs) == 3

    def test_load_all_returns_job_objects(self, tuimod, seed_db):
        jobs, _ = tuimod.load_jobs()
        assert len(jobs) > 0
        j = jobs[0]
        assert hasattr(j, "access_code")
        assert hasattr(j, "status")
        assert hasattr(j, "user_id")
        # username should be populated from users table
        assert isinstance(j.username, str)


class TestLoadJobsUserFilter:
    def test_filter_user_1(self, tuimod, seed_db):
        jobs, total = tuimod.load_jobs(user_id=1)
        assert total == 3
        assert len(jobs) == 3
        assert all(j.user_id == 1 for j in jobs)

    def test_filter_user_2(self, tuimod, seed_db):
        jobs, total = tuimod.load_jobs(user_id=2)
        assert total == 2
        assert len(jobs) == 2
        assert all(j.user_id == 2 for j in jobs)

    def test_filter_user_3(self, tuimod, seed_db):
        jobs, total = tuimod.load_jobs(user_id=3)
        assert total == 1
        assert len(jobs) == 1
        assert jobs[0].access_code == "CCC00001"

    def test_filter_nonexistent_user(self, tuimod, seed_db):
        jobs, total = tuimod.load_jobs(user_id=999)
        assert total == 0
        assert jobs == []


class TestLoadJobsCombinedFilter:
    def test_search_within_user(self, tuimod, seed_db):
        jobs, total = tuimod.load_jobs(search="AAA", user_id=1)
        assert total == 3
        # All of user 1's jobs start with AAA
        assert all(j.access_code.startswith("AAA") for j in jobs)

    def test_search_cross_user(self, tuimod, seed_db):
        """Search should be scoped to user_id when both are given."""
        jobs, total = tuimod.load_jobs(search="BBB", user_id=1)
        assert total == 0  # User 1 has no BBB codes

    def test_search_returns_user_jobs(self, tuimod, seed_db):
        jobs, total = tuimod.load_jobs(search="CCC", user_id=3)
        assert total == 1
        assert jobs[0].access_code == "CCC00001"
        assert jobs[0].user_id == 3

    def test_search_ignores_user_filter_when_user_none(self, tuimod, seed_db):
        jobs, total = tuimod.load_jobs(search="BBB", user_id=None)
        assert total == 2  # Both of user 2's jobs start with BBB


class TestLoadJobsOrdering:
    def test_processing_first(self, tuimod, seed_db):
        jobs, _ = tuimod.load_jobs()
        # processing jobs should come first
        first_statuses = [j.status for j in jobs[:2]]
        assert "processing" in first_statuses
        assert jobs[0].status == "processing"


# ── State tests ─────────────────────────────────────────────

class TestStateBackwardCompat:
    def test_no_user_id_no_label(self, tuimod, seed_db):
        jobs, total = tuimod.load_jobs()
        s = tuimod.State(jobs, total)
        assert s.user_id is None
        assert s.display_label == ""

    def test_reload_without_user_id(self, tuimod, seed_db):
        jobs, total = tuimod.load_jobs()
        s = tuimod.State(jobs, total)
        s.reload(reset_page=False)
        assert s.user_id is None
        assert len(s.all) == 6


class TestStateUserFilter:
    def test_state_with_user_id(self, tuimod, seed_db):
        jobs, total = tuimod.load_jobs(user_id=1)
        s = tuimod.State(jobs, total, user_id=1, display_label="用户: alice")
        assert s.user_id == 1
        assert s.display_label == "用户: alice"
        assert len(s.all) == 3

    def test_reload_preserves_user_filter(self, tuimod, seed_db):
        jobs, total = tuimod.load_jobs(user_id=2)
        s = tuimod.State(jobs, total, user_id=2)
        s.reload(reset_page=False)
        assert s.user_id == 2
        assert len(s.all) == 2
        assert all(j.user_id == 2 for j in s.all)

    def test_reload_reset_page(self, tuimod, seed_db):
        jobs, total = tuimod.load_jobs()
        s = tuimod.State(jobs, total)
        s.page = 1  # navigate away
        s.reload(reset_page=True)
        assert s.page == 0

    def test_reload_no_reset_page_within_bounds(self, tuimod, seed_db):
        jobs, total = tuimod.load_jobs()
        s = tuimod.State(jobs, total)
        s.reload(reset_page=False)
        assert s.page == 0  # stays on valid page

    def test_reload_clamp_page_when_out_of_bounds(self, tuimod, seed_db):
        jobs, total = tuimod.load_jobs()
        s = tuimod.State(jobs, total)
        s.page = 999
        s.reload(reset_page=False)
        assert s.page == s.pages - 1

    def test_display_label_persists_after_reload(self, tuimod, seed_db):
        jobs, total = tuimod.load_jobs(user_id=1)
        s = tuimod.State(jobs, total, user_id=1, display_label="用户: alice")
        s.reload()
        assert s.display_label == "用户: alice"


class TestStateProperties:
    def test_visible_reverses(self, tuimod, seed_db):
        jobs, total = tuimod.load_jobs()
        s = tuimod.State(jobs, total)
        # visible is reversed — last element of page slice is first
        assert len(s.visible) <= tuimod.PAGE_SIZE
        first_slice = s.all[s.start:s.end]
        if first_slice:
            assert s.visible[-1] == first_slice[0]

    def test_selected_job(self, tuimod, seed_db):
        jobs, total = tuimod.load_jobs()
        s = tuimod.State(jobs, total)
        s.cursor = 0
        assert s.selected_job is not None

    def test_clamp_cursor_empty(self, tuimod, seed_db):
        s = tuimod.State([], 0)
        s.cursor = 999
        s.clamp_cursor()
        assert s.cursor == 0

    def test_clamp_cursor_at_bottom(self, tuimod, seed_db):
        jobs, total = tuimod.load_jobs()
        s = tuimod.State(jobs, total)
        s.clamp_cursor(at_bottom=True)
        assert s.cursor == len(s.visible) - 1


# ── render_summary tests ────────────────────────────────────

class TestRenderSummary:
    def test_no_label_no_search(self, tuimod, seed_db):
        jobs, _ = tuimod.load_jobs()
        panel = tuimod.render_summary(jobs)
        assert panel is not None

    def test_with_label(self, tuimod, seed_db):
        jobs, _ = tuimod.load_jobs(user_id=1)
        panel = tuimod.render_summary(jobs, label="用户: alice")
        assert panel is not None

    def test_with_search(self, tuimod, seed_db):
        jobs, _ = tuimod.load_jobs()
        panel = tuimod.render_summary(jobs, search_query="AAA")
        assert panel is not None

    def test_with_both_label_and_search(self, tuimod, seed_db):
        jobs, _ = tuimod.load_jobs()
        panel = tuimod.render_summary(jobs, label="用户: alice", search_query="AAA")
        assert panel is not None

    def test_stats_accurate(self, tuimod, seed_db):
        jobs, _ = tuimod.load_jobs()
        stats: dict[str, int] = {}
        for j in jobs:
            stats[j.status] = stats.get(j.status, 0) + 1
        assert stats.get("pending", 0) == 1
        assert stats.get("processing", 0) == 1
        assert stats.get("completed", 0) == 2
        assert stats.get("failed", 0) == 1
        assert stats.get("cancelled", 0) == 1


# ── show_user_jobs_tui tests ─────────────────────────────────

class TestShowUserJobsTui:
    def test_function_is_callable(self, tuimod):
        assert callable(tuimod.show_user_jobs_tui)

    def test_calls_interactive_not_explode(self, tuimod, seed_db):
        """show_user_jobs_tui is a thin wrapper — just verify it imports cleanly."""
        import inspect
        source = inspect.getsource(tuimod.show_user_jobs_tui)
        assert "interactive(console" in source
        assert "user_id=" in source
        assert "display_label=" in source

    def test_no_load_user_jobs_reference(self, tuimod):
        """Verify we no longer depend on the removed load_user_jobs import."""
        import inspect
        source = inspect.getsource(tuimod.show_user_jobs_tui)
        assert "load_user_jobs" not in source


# ── interactive() entry point tests ──────────────────────────

class TestInteractiveEntry:
    def test_accepts_user_id_optional(self, tuimod):
        """interactive() should accept user_id=None by default."""
        import inspect
        sig = inspect.signature(tuimod.interactive)
        params = list(sig.parameters.keys())
        assert "user_id" in params
        assert "display_label" in params

    def test_accepts_only_console(self, tuimod):
        """interactive(console) should still work (backward compat)."""
        import inspect
        sig = inspect.signature(tuimod.interactive)
        # No required params besides console
        required = [
            p for p, v in sig.parameters.items()
            if v.default is inspect.Parameter.empty
        ]
        assert required == ["console"]


# ── Edge case tests ─────────────────────────────────────────

class TestEdgeCases:
    def test_state_no_jobs(self, tuimod):
        s = tuimod.State([], 0)
        assert s.total == 0
        assert s.pages == 1
        assert s.visible == []
        assert s.selected_job is None

    def test_load_jobs_no_results(self, tuimod, seed_db):
        jobs, total = tuimod.load_jobs(search="ZZZZZZZZ")
        assert total == 0
        assert jobs == []

    def test_user_with_no_jobs(self, tuimod, seed_db):
        jobs, total = tuimod.load_jobs(user_id=99)
        assert total == 0
        assert jobs == []
        # State should handle empty gracefully
        s = tuimod.State(jobs, total, user_id=99)
        assert s.pages == 1
        assert s.user_id == 99
