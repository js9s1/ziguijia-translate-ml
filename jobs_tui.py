#!/usr/bin/env python3
"""
Job queue TUI — cursor-based, paginated, keyboard-driven.

Usage:
    python jobs_tui.py [-h]
    python jobs_tui.py --list
    python jobs_tui.py --status <code>
    python jobs_tui.py --log <code>
    python jobs_tui.py --cancel <code>
    python jobs_tui.py --delete <code>
    python jobs_tui.py --watch <code>
"""

from __future__ import annotations

import argparse
import os
import select
import shutil
import signal
import sqlite3
import subprocess
import sys
import termios
import time
import tty
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import psutil
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ── Paths ─────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent
DB = Path(os.environ.get("JOBS_DB", str(HERE / "chatterbox-server" / "jobs.db")))
USERS_DB = Path(os.environ.get("USERS_DB", str(HERE / "chatterbox-server" / "users.db")))
MAX_LOAD_JOBS = 500
PAGE_SIZE = 10
AUTO_REFRESH_SEC = 3

# Import canonical checkpoint order from config to keep in sync
import sys as _sys
_sys.path.insert(0, str(HERE / "chatterbox-server"))
from config import CHECKPOINT_ORDER as _CKPT_ORDER
VALID_CHECKPOINT_STEPS = set(_CKPT_ORDER)
_CHECKPOINT_ORDER = list(_CKPT_ORDER)  # ordered list for "which steps are after X"

# ── Data types ────────────────────────────────────────────

_TYPE_MAP = {
    "_run_gen_audio": "音频生成",
    "_run_video_job": "宁视频翻译",
    "_run_video_custom_job": "自定义视频",
    "_run_tts_job": "语音合成",
    "_run_video_auto_job": "自动翻译视频",
    "_run_audio_file_job": "音频文件合成",
    "_run_audio_segmentation_job": "音频文件合成",
    "_run_video_ocr_job": "OCR翻译视频",
    "_run_video_ning_ocr_job": "宁视频OCR翻译",
    "_run_video_ning_ocr_translate_only_job": "宁视频OCR仅翻译",
    "_run_ocr_only_job": "视频OCR提取字幕",
}

_NING_VIDEO_TYPES = {
    "_run_video_job",
    "_run_video_ning_ocr_job",
    "_run_video_ning_ocr_translate_only_job",
}
_STATUS_STYLE_MAP = {
    "pending": "yellow",
    "processing": "cyan bold",
    "completed": "green",
    "failed": "red bold",
    "cancelled": "red bold",
    "deleted": "white dim",
}
_STS_LABEL_MAP = {
    "pending": "等待中",
    "processing": "处理中",
    "completed": "已完成",
    "failed": "已失败",
    "cancelled": "已取消",
    "deleted": "已删除",
}


@dataclass(kw_only=True)
class Job:
    access_code: str
    created_at: str | None
    error: str | None
    output_dir: str | None
    run_func_name: str | None
    srt_path: str | None
    status: str
    user_id: int | None
    status_changed_at: str | None = None
    target_language: str | None = None
    username: str = ""
    temperature: float | None = None
    cfg_weight: float | None = None
    exaggeration: float | None = None
    start_trim: float | None = None
    end_trim: float | None = None
    text: str | None = None

    @property
    def display_type(self) -> str:
        return _TYPE_MAP.get(self.run_func_name or "", self.run_func_name or "未知")

    @property
    def status_style(self) -> str:
        return _STATUS_STYLE_MAP.get(self.status, "white")

    @property
    def status_label(self) -> str:
        return _STS_LABEL_MAP.get(self.status, self.status)


# ── DB helpers ────────────────────────────────────────────

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


def count_jobs() -> int:
    conn = get_conn()
    row = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()
    conn.close()
    return row[0]


def _batch_load_usernames(user_ids: list[int | None]) -> dict[int, str]:
    """Load usernames for all given user_ids in one query."""
    ids = sorted({uid for uid in user_ids if uid is not None})
    if not ids:
        return {}
    try:
        conn = sqlite3.connect(str(USERS_DB))
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT id, email FROM users WHERE id IN ({placeholders})", ids
        ).fetchall()
        conn.close()
        return {str(row[0]): (row[1].split("@")[0][:8] if row[1] else "") for row in rows}
    except Exception:
        return {}


def load_jobs(limit: int = MAX_LOAD_JOBS, offset: int = 0, search: str = "") -> tuple[list[Job], int]:
    """Return (jobs_page, total_count). If search is given, filter by access_code LIKE."""
    conn = get_conn()
    order_clause = """
        ORDER BY
            CASE WHEN status = 'processing' THEN 0 ELSE 1 END,
            created_at DESC
    """
    if search:
        pattern = f"%{search}%"
        total = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE access_code LIKE ?", (pattern,)
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT access_code, run_func_name, status, error, output_dir, srt_path, created_at, user_id, status_changed_at, "
            "temperature, cfg_weight, exaggeration, start_trim, end_trim, text, target_language "
            "FROM jobs WHERE access_code LIKE ? " + order_clause + " LIMIT ? OFFSET ?",
            (pattern, limit, offset),
        ).fetchall()
    else:
        total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        rows = conn.execute(
            "SELECT access_code, run_func_name, status, error, output_dir, srt_path, created_at, user_id, status_changed_at, "
            "temperature, cfg_weight, exaggeration, start_trim, end_trim, text, target_language "
            "FROM jobs " + order_clause + " LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    conn.close()
    jobs = [Job(**dict(r)) for r in rows]
    # Batch-load usernames
    username_map = _batch_load_usernames([j.user_id for j in jobs])
    for j in jobs:
        j.username = username_map.get(j.user_id, "")
    return jobs, total


def get_job(code: str) -> Job | None:
    conn = get_conn()
    r = conn.execute(
        "SELECT access_code, run_func_name, status, error, output_dir, srt_path, created_at, user_id, status_changed_at, "
        "temperature, cfg_weight, exaggeration, start_trim, end_trim, text, target_language "
        "FROM jobs WHERE access_code = ?",
        (code.upper(),),
    ).fetchone()
    conn.close()
    if r is None:
        return None
    job = Job(**dict(r))
    username_map = _batch_load_usernames([job.user_id])
    job.username = username_map.get(job.user_id, "")
    return job


def cancel_job(code: str) -> str:
    code = code.upper()
    conn = get_conn()
    job = conn.execute(
        "SELECT status, output_dir FROM jobs WHERE access_code = ?", (code,)
    ).fetchone()
    if not job:
        conn.close()
        return f"任务 {code} 不存在"
    status, output_dir = job["status"], job["output_dir"]
    if status not in ("pending", "processing"):
        conn.close()
        return f"任务 {code} 已经是 {status}，无法取消"
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    if output_dir and status == "processing":
        try:
            for proc in psutil.process_iter(["pid", "cmdline"]):
                try:
                    if output_dir in " ".join(proc.info["cmdline"] or []):
                        proc.send_signal(signal.SIGTERM)
                except Exception:
                    pass
        except Exception:
            pass
    conn.execute(
        "UPDATE jobs SET status = ?, error = ?, cancelled_at = ?, status_changed_at = ? WHERE access_code = ?",
        ("cancelled", "用户取消", now, now, code),
    )
    conn.commit()
    conn.close()
    return f"✅ 任务 {code} 已取消"


def _invalidated_steps(keep_steps: list[str]) -> list[str]:
    """Return steps not in keep_steps, in checkpoint order."""
    kept = set(keep_steps)
    return [s for s in _CHECKPOINT_ORDER if s not in kept]


def _purge_step_output(output_dir: str, video_number: str, step: str):
    """Delete files/dirs associated with a single checkpoint step.

    For the "audio" step, only final output files inside ``audio/`` or
    ``audio_tracks/`` are removed.  The ``tmp/`` subdirectory (per-segment
    cached wavs and meta JSONs) is preserved so unchanged segments skip
    re-generation on resubmit.
    """
    import shutil
    paths = []
    if step == "download":
        paths.append(os.path.join(output_dir, f"{video_number}.mp4"))
    elif step == "trim":
        paths.append(os.path.join(output_dir, f"{video_number}_trimmed.mp4"))
    elif step == "ocr":
        paths.append(os.path.join(output_dir, "ocr_screen.srt"))
        paths.append(os.path.join(output_dir, "frames"))
    elif step == "translate":
        paths.append(os.path.join(output_dir, "translated.srt"))
    elif step == "audio":
        # Remove only final output files — preserve tmp/ cache
        _AUDIO_OUTPUT_FILES = [
            "output.wav",
            "output_adjusted.srt",
            "output-final-modified.srt",
            "changed_segments.json",
            "job.log",
        ]
        for subdir in ("audio", "audio_tracks"):
            ad = os.path.join(output_dir, subdir)
            if os.path.isdir(ad):
                for fn in _AUDIO_OUTPUT_FILES:
                    fp = os.path.join(ad, fn)
                    if os.path.isfile(fp):
                        paths.append(fp)
    elif step == "video":
        paths.append(os.path.join(output_dir, "output_modified.mp4"))
        paths.append(os.path.join(output_dir, "output_final.mp4"))
    for p in paths:
        if not p:
            continue
        if os.path.isfile(p):
            os.remove(p)
        elif os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)


def purge_invalidated_output(output_dir: str, keep_steps: list[str]):
    """Delete output files for checkpoint steps not in keep_steps."""
    if not output_dir or not os.path.isdir(output_dir):
        return
    # Extract video_number from output_dir name (pattern: {number}-{access_code})
    base = os.path.basename(output_dir)
    video_number = base.split("-")[0] if "-" in base else ""
    for step in _invalidated_steps(keep_steps):
        _purge_step_output(output_dir, video_number, step)


def get_checkpoint(code: str) -> str:
    """Return the checkpoint string for a job, filtered to only valid current steps."""
    conn = get_conn()
    row = conn.execute(
        "SELECT checkpoint FROM jobs WHERE access_code = ?", (code.upper(),)
    ).fetchone()
    conn.close()
    if not row or not row[0]:
        return ""
    steps = [s.strip() for s in row[0].split(",") if s.strip() in VALID_CHECKPOINT_STEPS]
    return ",".join(steps)


def resubmit_job(code: str, keep_steps: list[str] | None = None) -> str:
    """Resubmit a failed/completed job.

    If *keep_steps* is given, the checkpoint column is overwritten with only
    those steps (all others are invalidated).  Pass an empty list to clear
    all checkpoints (restart from scratch).
    """
    code = code.upper()
    conn = get_conn()
    job = conn.execute(
        "SELECT status, checkpoint_edited, output_dir FROM jobs WHERE access_code = ?", (code,)
    ).fetchone()
    if not job:
        conn.close()
        return f"任务 {code} 不存在"

    status = job["status"]
    checkpoint_edited = job["checkpoint_edited"]

    if status == "deleted":
        conn.close()
        return f"任务 {code} 已被删除，无法重新提交"

    if status not in ("failed", "completed", "cancelled"):
        conn.close()
        return f"任务 {code} 状态为 {status}，只能重新提交失败、已完成或已取消的任务"

    output_dir = job["output_dir"]

    # Purge output files for invalidated steps before updating checkpoint DB
    if keep_steps is not None:
        purge_invalidated_output(output_dir, keep_steps)

    # Update checkpoint and mark as edited, let server update status and queue
    if keep_steps is not None:
        new_checkpoint = ",".join(s for s in keep_steps if s in VALID_CHECKPOINT_STEPS)
        conn.execute(
            "UPDATE jobs SET checkpoint = ?, checkpoint_edited = 1 WHERE access_code = ?",
            (new_checkpoint, code),
        )
    else:
        # Mark as edited so server allows resubmit
        conn.execute(
            "UPDATE jobs SET checkpoint_edited = 1 WHERE access_code = ?",
            (code,),
        )
    conn.commit()
    conn.close()

    # Notify server via HTTP to update status and queue the job
    try:
        import urllib.request
        import json
        req = urllib.request.Request(
            f"http://localhost:5600/srt/resubmit/{code}",
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            if resp.status == 200 and data.get("success"):
                return f"✅ 任务 {code} 已重新提交"
            else:
                error_msg = data.get("error", "未知错误")
                return f"⚠️ 检查点已更新，但服务器拒绝: {error_msg}"
    except urllib.error.HTTPError as e:
        error_data = e.read().decode('utf-8') if e.fp else ""
        return f"⚠️ 检查点已更新，但服务器返回错误 ({e.code}): {error_data}"
    except Exception as e:
        # Server might not be running or connection failed
        return f"⚠️ 检查点已更新，但无法通知服务器: {e}"


def delete_job(code: str) -> str:
    code = code.upper()
    conn = get_conn()
    job = conn.execute(
        "SELECT status FROM jobs WHERE access_code = ?", (code,)
    ).fetchone()
    if not job:
        conn.close()
        return f"任务 {code} 不存在"
    # Soft delete: mark as deleted, consistent with the web server
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    conn.execute(
        "UPDATE jobs SET status = ?, error = ?, deleted_at = ?, status_changed_at = ? WHERE access_code = ?",
        ("deleted", "用户删除", now, now, code),
    )
    conn.commit()
    conn.close()
    return f"✅ 任务 {code} 已删除（软删除）"


LOG_CANDIDATES = ["job.log", "process.log", "output.log"]


def list_output_files(output_dir: str) -> list[dict]:
    """List files in the output directory with name, size, path. Returns empty list if dir missing."""
    if not output_dir or not os.path.isdir(output_dir):
        return []
    result = []
    try:
        for entry in sorted(os.scandir(output_dir), key=lambda e: (not e.is_dir(), e.name)):
            if entry.name.startswith("."):
                continue
            kind = "📁" if entry.is_dir() else "📄"
            size = ""
            if entry.is_file():
                sz = entry.stat().st_size
                if sz < 1024:
                    size = f"{sz}B"
                elif sz < 1024 * 1024:
                    size = f"{sz / 1024:.1f}KB"
                else:
                    size = f"{sz / 1024 / 1024:.1f}MB"
            result.append({"name": entry.name, "path": entry.path, "size": size, "kind": kind, "is_dir": entry.is_dir()})
    except PermissionError:
        pass
    return result


def find_log_file(output_dir: str) -> str | None:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    for c in LOG_CANDIDATES:
        p = os.path.join(output_dir, c)
        if os.path.isfile(p):
            return p
    audio_dir = os.path.join(output_dir, "audio_tracks")
    if os.path.isdir(audio_dir):
        for c in LOG_CANDIDATES:
            p = os.path.join(audio_dir, c)
            if os.path.isfile(p):
                return p
    return None


def read_log(output_dir: str, max_lines: int = 500) -> str:
    path = find_log_file(output_dir)
    if path is None:
        return "(无日志文件)"
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
        return f"... (截断，仅显示最近 {max_lines} 行)\n" + "".join(lines)
    return "".join(lines)


# ── Pagination & page state ──────────────────────────────


class State:
    def __init__(self, jobs: list[Job], total: int):
        self.all = jobs  # DB returns newest first; reverse within each page
        self.total = total
        self.pages = max(1, (self.total + PAGE_SIZE - 1) // PAGE_SIZE)
        self.page = 0
        self.cursor = 0  # row index within current page (0=top/oldest)
        self.menu_open = False
        self.menu_cursor = 0
        self.focus: str = "table"  # "table" | "nav"
        self.nav_cursor = 0
        self.auto_refresh = False
        self._message: str | None = None
        self._message_time: float = 0.0
        self.search_query: str = ""
        # Orphan tracking
        self._orphans: list[dict] = []
        self._orphans_last_scan: float = 0.0

    def reload(self, reset_page=True):
        self.all, self.total = load_jobs(search=self.search_query)
        self.pages = max(1, (self.total + PAGE_SIZE - 1) // PAGE_SIZE)
        if reset_page:
            self.page = 0
        elif self.page >= self.pages:
            self.page = self.pages - 1
        self.clamp_cursor()

    @property
    def start(self) -> int:
        return self.page * PAGE_SIZE

    @property
    def end(self) -> int:
        return min(self.start + PAGE_SIZE, self.total)

    @property
    def visible(self) -> list[Job]:
        """Jobs on this page, newest at bottom (#1 at top, #10 or fewer at bottom)."""
        return list(reversed(self.all[self.start : self.end]))

    @property
    def selected_job(self) -> Job | None:
        idx = self.start + (len(self.visible) - 1 - self.cursor)
        if 0 <= idx < self.total:
            return self.all[idx]
        return None

    def clamp_cursor(self, at_bottom=False):
        count = len(self.visible)
        if count == 0:
            self.cursor = 0
        elif at_bottom or self.cursor >= count:
            self.cursor = count - 1

    def message(self, text: str):
        self._message = text
        self._message_time = time.monotonic()

    def clear_expired_message(self):
        if self._message and (time.monotonic() - self._message_time) > 5:
            self._message = None

    # ── Menu popup cursor ──────────────────────────────

    def menu_options(self) -> list[tuple[str, str]]:
        """Return list of (action_id, label) for the popup menu."""
        opts: list[tuple[str, str]] = [("detail", r"\[D]详情"), ("open_dir", r"\[o]打开目录")]
        if self.selected_job and self.selected_job.status in ("pending", "processing"):
            opts.append(("cancel", r"\[k]取消"))
        if self.selected_job and self.selected_job.status in ("failed", "completed", "cancelled"):
            opts.append(("resubmit", r"\[s]重新提交"))
        if self.selected_job and self.selected_job.user_id is not None:
            opts.append(("user_jobs", r"\[u]该用户任务"))
        opts.append(("delete", r"\[r]删除"))
        opts.append(("close", r"\[c]关闭"))
        return opts

    def clamp_menu_cursor(self):
        max_idx = len(self.menu_options()) - 1
        if self.menu_cursor < 0:
            self.menu_cursor = 0
        if self.menu_cursor > max_idx:
            self.menu_cursor = max_idx

    # ── Nav bar cursor ─────────────────────────────────

    NAV_ITEMS: list[tuple[str, str, str]] = [
        ("search", "/", "搜索"),
        ("refresh", "N", "刷新"),
        ("quit", "Q", "退出"),
    ]

    def nav_options(self) -> list[tuple[str, str]]:
        return [(aid, label) for aid, key, label in self.NAV_ITEMS]

    def clamp_nav_cursor(self):
        max_idx = len(self.NAV_ITEMS) - 1
        if self.nav_cursor < 0:
            self.nav_cursor = 0
        if self.nav_cursor > max_idx:
            self.nav_cursor = max_idx

    def execute_nav_action(self, action: str) -> Optional[str]:
        """Execute a nav action. Returns an optional message string."""
        if action == "search":
            return None  # handled by caller
        elif action == "refresh":
            self.search_query = ""
            self.reload()
            return "✅ 已刷新"
        elif action == "quit":
            return None


# ── Terminal raw mode ─────────────────────────────────────


@contextmanager
def raw_mode():
    """Set terminal to raw input mode without breaking output processing.

    Unlike tty.setraw(), this preserves OPOST/ONLCR so Rich's Live rendering
    (which relies on \n → \r\n translation) is not corrupted.
    """
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    new = termios.tcgetattr(fd)
    # Input: disable break, CR→NL, parity stripping, stripping, XON/XOFF
    new[tty.IFLAG] &= ~(termios.BRKINT | termios.ICRNL | termios.INPCK |
                         termios.ISTRIP | termios.IXON)
    # Output: PRESERVE OPOST (don't touch OFLAG) — keeps \n → \r\n
    # Control: 8-bit chars
    new[tty.CFLAG] &= ~(termios.CSIZE | termios.PARENB)
    new[tty.CFLAG] |= termios.CS8
    # Local: disable echo, canonical mode, extended input, signals
    new[tty.LFLAG] &= ~(termios.ECHO | termios.ICANON | termios.IEXTEN |
                        termios.ISIG)
    # Read: return immediately with at least 1 byte
    new[tty.CC][termios.VMIN] = 1
    new[tty.CC][termios.VTIME] = 0
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, new)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def read_key(timeout: float | None = None) -> str:
    """Read a single keypress. Returns semantic names for special keys.

    If *timeout* is not None, returns an empty string when no key is
    pressed within that many seconds.
    """
    fd = sys.stdin.fileno()

    if timeout is not None:
        r, _, _ = select.select([fd], [], [], timeout)
        if not r:
            return ""

    ch = os.read(fd, 1)
    if not ch:
        return ""
    ch = ch.decode("utf-8", errors="replace")
    if ch != "\x1b":
        return ch

    # Escape sequence — collect remaining bytes with a short deadline
    seq = b""
    deadline = time.monotonic() + 0.3
    while time.monotonic() < deadline:
        r, _, _ = select.select([fd], [], [], 0.03)
        if r:
            seq += os.read(fd, 1)
        elif seq:
            break
    s = seq.decode("utf-8", errors="replace")

    # CSI sequences
    if s == "[A":
        return "UP"
    if s == "[B":
        return "DOWN"
    if s == "[C":
        return "RIGHT"
    if s == "[D":
        return "LEFT"
    if s in ("[5~", "[5;5~"):
        return "PGUP"
    if s in ("[6~", "[6;5~"):
        return "PGDN"
    if s in ("[H", "[7~"):
        return "HOME"
    if s in ("[F", "[8~"):
        return "END"
    if s in ("[2~", "[2;5~"):
        return "INS"
    if s == "[3~":
        return "DEL"
    if s == "OA":
        return "UP"
    if s == "OB":
        return "DOWN"
    if s == "OC":
        return "RIGHT"
    if s == "OD":
        return "LEFT"
    if s == "OH":
        return "HOME"
    if s == "OF":
        return "END"
    if not s:
        return "ESC"
    return "\x1b" + s


# ── Orphan detection ──────────────────────────────────────

_ORPHAN_CACHE: tuple[float, list[dict]] = (0.0, [])  # (cached_at, orphans)
_ORPHAN_CACHE_TTL = 5  # seconds


def detect_orphans(force: bool = False) -> list[dict]:
    """Find running processes whose cmdline references a job output_dir,
    where the job is no longer in a running/pending state.

    Returns list of dicts: {access_code, output_dir, pid, cmd, cpu_percent, memory_rss, runtime}.
    Cached for _ORPHAN_CACHE_TTL seconds to avoid excessive psutil iteration.
    """
    global _ORPHAN_CACHE
    now = time.monotonic()
    if not force and (now - _ORPHAN_CACHE[0]) < _ORPHAN_CACHE_TTL:
        return _ORPHAN_CACHE[1]

    # Load all non-PENDING, non-PROCESSING jobs with an output_dir
    conn = get_conn()
    rows = conn.execute(
        "SELECT access_code, output_dir FROM jobs WHERE status NOT IN ('pending', 'processing') AND output_dir IS NOT NULL"
    ).fetchall()
    conn.close()

    dir_to_code = {r["output_dir"]: r["access_code"] for r in rows if r["output_dir"]}

    orphans = []
    try:
        for proc in psutil.process_iter(["pid", "cmdline", "cpu_percent", "memory_info", "create_time"]):
            try:
                cmdline = " ".join(proc.info["cmdline"] or [])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            if not cmdline:
                continue
            # Skip terminal/UI processes that happen to have the output_dir
            # in their cwd — they aren't runaway job subprocesses.
            skip_procs = {"ghostty", "wezterm-gui", "alacritty", "kitty", "konsole",
                          "gnome-terminal", "xfce4-terminal", "tmux", "screen"}
            cmd_base = os.path.basename(cmdline.split()[0]) if cmdline.split() else ""
            if cmd_base in skip_procs:
                continue
            for output_dir, access_code in dir_to_code.items():
                if output_dir in cmdline:
                    try:
                        runtime = time.time() - proc.info["create_time"]
                    except Exception:
                        runtime = 0
                    rss = proc.info["memory_info"].rss if proc.info["memory_info"] else 0
                    cpu = proc.info["cpu_percent"] or 0
                    short_cmd = os.path.basename(cmdline.split()[0]) if cmdline.split() else "?"
                    orphans.append({
                        "access_code": access_code,
                        "output_dir": output_dir,
                        "pid": proc.info["pid"],
                        "cmd": short_cmd,
                        "cpu_percent": cpu,
                        "memory_rss_mb": rss / (1024 * 1024),
                        "runtime_sec": runtime,
                    })
                    break  # one match per proc is enough
    except Exception:
        pass

    _ORPHAN_CACHE = (now, orphans)
    return orphans


def render_orphan_warning(orphans: list[dict]) -> Panel | None:
    """Render a warning panel if orphan processes are found."""
    if not orphans:
        return None
    lines = []
    total_cpu = 0
    total_mem = 0
    for o in orphans:
        rt = _format_duration(o["runtime_sec"]) if o["runtime_sec"] else "?"
        lines.append(
            f"  PID {o['pid']}  [bold red]{o['cmd']}[/]  "
            f"[yellow]{o['cpu_percent']:.0f}% CPU[/]  "
            f"[magenta]{o['memory_rss_mb']:.0f} MB[/]  "
            f"[dim]运行 {rt}[/]  "
            f"[cyan]{o['access_code']}[/]"
        )
        total_cpu += o["cpu_percent"]
        total_mem += o["memory_rss_mb"]
    header = (
        f"[bold red]⚠ 发现 {len(orphans)} 个孤儿进程[/] "
        f"([yellow]{total_cpu:.0f}% CPU[/], [magenta]{total_mem:.0f} MB[/]) "
        f"[dim]N=刷新  K=杀掉[/]"
    )
    return Panel(
        Align.left("\n".join([header] + lines)),
        border_style="red",
        padding=(0, 1),
    )


def render_summary(jobs: list[Job], search_query: str = "") -> Panel:
    total = len(jobs)
    pending = sum(1 for j in jobs if j.status == "pending")
    processing = sum(1 for j in jobs if j.status == "processing")
    completed = sum(1 for j in jobs if j.status == "completed")
    failed = sum(1 for j in jobs if j.status == "failed")
    parts = [
        f"总计 [bold]{total}[/]",
        f"[yellow]等待 {pending}[/]",
        f"[cyan]处理中 {processing}[/]",
        f"[green]完成 {completed}[/]",
        f"[red]失败 {failed}[/]",
    ]
    if search_query:
        parts.insert(0, f"[cyan]🔍 {search_query}[/]")
    return Panel(Align.center("  |  ".join(parts)), border_style="dim", padding=(0, 0))


def render_table(s: State) -> Table:
    t = Table(box=box.SIMPLE, header_style="bold", show_edge=False, padding=(0, 1))
    t.add_column("", width=2)
    t.add_column("#", style="dim", width=4)
    t.add_column("类型", width=14)
    t.add_column("用户", width=8)
    t.add_column("访问码", width=9)
    t.add_column("状态", width=10, no_wrap=True)
    t.add_column("错误", width=28, overflow="fold")
    t.add_column("时间", width=16, no_wrap=True)

    for i, j in enumerate(s.visible):
        is_cursor = i == s.cursor
        idx = s.start + (len(s.visible) - 1 - i) + 1
        row_style = "reverse" if is_cursor else ""
        marker = "▸" if is_cursor else " "

        t.add_row(
            Text(marker, style="cyan bold" if is_cursor else ""),
            str(idx),
            j.display_type,
            j.username,
            Text(j.access_code, style="bold"),
            Text(j.status_label, style=j.status_style),
            (j.error or "")[:60],
            (j.status_changed_at or j.created_at or "")[:16],
            style=row_style,
        )
    return t


def render_footer(s: State) -> Panel:
    page_info = f"第 [bold]{s.page + 1}[/]/{s.pages} 页  ({s.total} 任务)"

    # Render nav items with cursor
    nav_parts = []
    for i, (aid, key, label) in enumerate(s.NAV_ITEMS):
        text = f" {key}={label} "
        if s.focus == "nav" and i == s.nav_cursor:
            text = f"[reverse]{text}[/]"
        else:
            text = f"[dim]{text}[/]"
        nav_parts.append(text)

    help_text = (
        "[bold]Tab[/] 切换焦点  "
        + "  ".join(nav_parts)
    )
    return Panel(f"  {page_info}  |  {help_text}", border_style="dim", padding=(0, 0))


def render_all(s: State) -> list:
    items = [render_summary(s.all, s.search_query), "", render_table(s), "", render_footer(s)]
    # Orphan warning, if any
    now = time.monotonic()
    if s._orphans_last_scan and (now - s._orphans_last_scan) > _ORPHAN_CACHE_TTL:
        s._orphans = detect_orphans()
        s._orphans_last_scan = now
    orphan_panel = render_orphan_warning(s._orphans)
    if orphan_panel:
        items.insert(1, orphan_panel)
        items.insert(2, "")
    if s._message:
        items.append(
            Panel(
                Align.center(Text(s._message, style="bold")),
                border_style="green" if s._message.startswith("✅") else "red",
                padding=(0, 0),
            )
        )
    return items


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}秒"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}分{seconds % 60:.0f}秒"
    elif seconds < 86400:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        return f"{h:.0f}小时{m:.0f}分"
    else:
        d = seconds // 86400
        h = (seconds % 86400) // 3600
        return f"{d:.0f}天{h:.0f}小时"


def _elapsed_since(dt_str: str | None) -> str | None:
    """Return human-readable elapsed time from *dt_str* to now, or None."""
    if not dt_str:
        return None
    try:
        dt = time.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
        epoch = time.mktime(dt)
        elapsed = time.time() - epoch
        if elapsed < 0:
            return None
        return _format_duration(elapsed)
    except (ValueError, OSError):
        return None


def render_detail(job: Job, mode: str) -> Panel:
    if mode == "detail":
        lines = [
            f"访问码: {job.access_code}",
            f"类型: {job.display_type}",
            f"状态: {job.status_label}",
            f"创建时间: {job.created_at or 'N/A'}",
            f"状态变更: {job.status_changed_at or 'N/A'}",
            f"输出目录: {job.output_dir or 'N/A'}",
            f"SRT路径: {job.srt_path or 'N/A'}",
            f"用户: {job.username or 'N/A'}",
        ]
        # Show elapsed time for active statuses
        if job.status in ("processing", "pending"):
            elapsed = _elapsed_since(job.status_changed_at)
            if elapsed:
                label = "处理中" if job.status == "processing" else "等待中"
                # Insert right after the status line (index 2)
                lines.insert(3, f"已{label}: [bold]{elapsed}[/]")
        # Show optional numeric parameters
        param_lines = []
        if job.temperature is not None:
            param_lines.append(f"温度: {job.temperature}")
        if job.cfg_weight is not None:
            param_lines.append(f"CFG权重: {job.cfg_weight}")
        if job.exaggeration is not None:
            param_lines.append(f"夸张度: {job.exaggeration}")
        if job.start_trim is not None and job.run_func_name in _NING_VIDEO_TYPES:
            param_lines.append(f"开始裁剪: {job.start_trim}s")
        if job.end_trim is not None and job.run_func_name in _NING_VIDEO_TYPES:
            param_lines.append(f"结束裁剪: {job.end_trim}s")
        if param_lines:
            lines.append("")
            lines.extend(param_lines)
        # ── Show input text for TTS jobs ────────────────────────
        if job.display_type == "语音合成" and job.text:
            lines.append("")
            # Truncate long text for display
            display_text = job.text[:1000]
            if len(job.text) > 1000:
                display_text += "..."
            lines.append(f"输入文本:\n[italic]{display_text}[/]")
        if job.error:
            lines.append(f"\n错误信息:\n{job.error}")
        return Panel(
            "\n".join(lines),
            title=f"详情 — {job.access_code}",
            border_style="cyan",
        )
    elif mode == "log":
        log_text = read_log(job.output_dir) if job.output_dir else "(无日志文件)"
        return Panel(
            log_text[:5000],
            title=f"日志 — {job.access_code}",
            border_style="yellow",
        )
    return Panel("")


def load_user_jobs(user_id: int) -> list[Job]:
    """Load all jobs for a given user_id."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT access_code, run_func_name, status, error, output_dir, srt_path, created_at, user_id, status_changed_at, "
        "text, target_language "
        "FROM jobs WHERE user_id = ?"
        " ORDER BY"
        "   CASE WHEN status = 'processing' THEN 0 ELSE 1 END,"
        "   COALESCE(status_changed_at, created_at) DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    jobs = [Job(**dict(r)) for r in rows]
    username_map = _batch_load_usernames([j.user_id for j in jobs])
    for j in jobs:
        j.username = username_map.get(j.user_id, "")
    return jobs


def show_user_jobs_tui(current_job: Job, state: State, console: Console):
    """Show a paginated job list filtered to the current job's user."""
    jobs = load_user_jobs(current_job.user_id)
    if not jobs:
        console.clear()
        console.print(f"[yellow]用户 {current_job.username} 没有其他任务[/]")
        console.print("\n[dim]按任意键返回...[/dim]")
        os.read(sys.stdin.fileno(), 1)
        return

    total = len(jobs)
    pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = 0
    cursor = 0

    while True:
        start = page * PAGE_SIZE
        end = min(start + PAGE_SIZE, total)
        page_jobs = jobs[start:end]

        # Build a temporary State-like view for rendering
        class TempState:
            def __init__(self):
                self.all = jobs
                self.total = total
                self.pages = pages
                self.page = page
                self.cursor = cursor
                self.start = start
                self.end = end
                self.visible = list(reversed(page_jobs))
                self.search_query = ""
                self._message = None
                self._message_time = 0.0

        s = TempState()

        console.clear()
        console.print(render_summary(jobs, f"用户: {current_job.username}"))
        console.print()
        console.print(render_table(s))
        console.print()
        console.print(Align.center("[dim]↑↓ 移动  ←→翻页  Q=返回[/dim]"))

        with raw_mode():
            key = read_key()

        if key in ("q", "Q", "ESC", "\x1b"):
            return

        elif key == "UP":
            if cursor > 0:
                cursor -= 1
            elif page > 0:
                page -= 1
                cursor = PAGE_SIZE - 1

        elif key == "DOWN":
            max_cursor = len(s.visible) - 1
            if cursor < max_cursor:
                cursor += 1
            elif page < pages - 1:
                page += 1
                cursor = 0

        elif key == "LEFT":
            if page > 0:
                page -= 1
                cursor = 0

        elif key == "RIGHT":
            if page + 1 < pages:
                page += 1
                cursor = 0


# ── Output browser ──────────────────────────────────────

AUDIO_VIDEO_EXTS = {".mp4", ".wav", ".mp3", ".m4a", ".avi", ".mkv", ".mov"}


def run_output_browser(job: Job, console: Console):
    """Interactive file browser for the job output directory. Recursive for subdirs.
    Files are paginated with 20 per page."""
    PAGE = 20
    stack = [job.output_dir]
    cursor = 0
    page = 0

    while True:
        current_dir = stack[-1]
        files = list_output_files(current_dir)
        if not files:
            console.clear()
            console.print(
                Panel("(空目录)", title=f"输出 — {os.path.basename(current_dir)}", border_style="blue")
            )
            console.print("\n[dim]按任意键返回...[/dim]")
            os.read(sys.stdin.fileno(), 1)
            if len(stack) > 1:
                stack.pop()
                cursor = 0
                page = 0
                continue
            else:
                return

        pages = max(1, (len(files) + PAGE - 1) // PAGE)
        if page >= pages:
            page = pages - 1

        # Clamp cursor
        if cursor >= len(files):
            cursor = len(files) - 1
        if cursor < 0:
            cursor = 0

        # Paginate
        start = page * PAGE
        end = min(start + PAGE, len(files))
        page_files = files[start:end]
        rel_cursor = cursor - start

        console.clear()
        # Render file listing
        lines = []
        for i, f in enumerate(page_files):
            name = f["name"]
            size = f["size"]
            kind = f["kind"]
            sized = f"  [dim]{size}[/]" if size else ""
            entry_line = f"  {kind} {name}{sized}"
            if i == rel_cursor:
                lines.append(f"[reverse] {kind} {name}{sized} [/]")
            else:
                lines.append(entry_line)

        title_parts = []
        for d in stack[1:]:
            rel = os.path.relpath(d, job.output_dir)
            if rel and rel != ".":
                title_parts.append(rel)
        title = f"📂 {job.access_code}" + ("/" + "/".join(title_parts) if title_parts else "")
        if pages > 1:
            title += f"  (第{page+1}/{pages}页)"
        console.print(Panel("\n".join(lines), title=title, border_style="blue"))
        console.print(Align.center("[dim]↑↓ 选择  ←→翻页  Enter=打开/播放  D=删除  ESC/Bksp=返回  Q=退出[/dim]"))

        key = read_key()

        if key in ("q", "Q"):
            return

        elif key == "UP":
            cursor -= 1
        elif key == "DOWN":
            cursor += 1

        elif key == "LEFT" or key == "\x7f" or key == "\b":
            # Back / backspace at page 0 = go up a dir
            if page > 0:
                page -= 1
                cursor = page * PAGE + rel_cursor
                if cursor < 0:
                    cursor = 0
            elif len(stack) > 1:
                stack.pop()
                cursor = 0
                page = 0
            continue

        elif key == "RIGHT":
            if page + 1 < pages:
                page += 1
                cursor = page * PAGE + rel_cursor
            continue

        elif key in ("\r", "\n"):
            entry = files[cursor]
            fpath = entry["path"]
            if entry["is_dir"]:
                stack.append(fpath)
                cursor = 0
                page = 0
                continue

            # File — check extension
            ext = os.path.splitext(fpath)[1].lower()
            if ext in AUDIO_VIDEO_EXTS:
                # Play with mpv — background process, detach
                console.clear()
                console.print(Panel(
                    Align.center(f"▶ 播放: [bold]{entry['name']}[/]"),
                    border_style="green",
                ))
                console.print("\n[dim]按 Enter 启动 mpv...[/dim]")
                os.read(sys.stdin.fileno(), 1)
                import subprocess
                subprocess.Popen(
                    ["mpv", "--keep-open=yes", fpath],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                )
            else:
                # Text files — open with less in ghostty
                console.clear()
                console.print(Panel(
                    Align.center(f"📝 打开: [bold]{entry['name']}[/]"),
                    border_style="yellow",
                ))
                console.print("\n[dim]按 Enter 启动 ghostty...[/dim]")
                os.read(sys.stdin.fileno(), 1)
                import subprocess
                subprocess.Popen(
                    ["ghostty", "-e", "sh", "-c", f"cat '{fpath}' | sed 's/\\r/\\n/g' | less"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                )

        elif key in ("d", "D"):
            entry = files[cursor]
            fpath = entry["path"]
            name = entry["name"]
            # Confirmation
            console.clear()
            console.print(Panel(
                Align.center(f"[bold red]确定要删除 {name} 吗？[/]\n\n[reverse] Y/Enter=确认  N=取消 [/]"),
                border_style="red",
                padding=(1, 2),
                width=50,
            ))
            console.print(Align.center("[dim]Y/Enter=确认  N=取消[/dim]"))
            while True:
                ch = os.read(sys.stdin.fileno(), 1)
                if not ch:
                    continue
                ch = ch.decode("utf-8", errors="replace").lower()
                if ch in ("y", "\r", "\n"):
                    try:
                        os.remove(fpath)
                    except Exception as e:
                        console.clear()
                        console.print(Panel(
                            Align.center(f"[red]删除失败: {e}[/]"),
                            border_style="red",
                        ))
                        console.print("\n[dim]按任意键返回...[/dim]")
                        os.read(sys.stdin.fileno(), 1)
                    break
                elif ch in ("n", "\x1b", "q"):
                    break
            continue

        elif key == "ESC":
            if len(stack) > 1:
                stack.pop()
                cursor = 0
            else:
                return

        # Clamp cursor after movement
        if cursor < 0:
            cursor = 0
        if cursor >= len(files):
            cursor = len(files) - 1


def make_menu_popup(s: State) -> Panel:
    """Narrow popup with cursor-based action options."""
    job = s.selected_job
    if not job:
        return Panel("")
    opts = s.menu_options()
    rendered = []
    for i, (aid, label) in enumerate(opts):
        text = f" {label} "
        if i == s.menu_cursor:
            rendered.append(f"[reverse]{text}[/]")
        else:
            rendered.append(text)
    return Panel(
        "  ".join(rendered),
        title=f"⚡ {job.access_code}",
        border_style="yellow",
        padding=(0, 1),
        width=70,
    )


def render_menu_overlay(s: State) -> Group:
    """Full layout when the menu popup is open."""
    job = s.selected_job
    items = render_all(s)
    if job:
        items.append("")
        items.append(Align.center(make_menu_popup(s)))
        items.append(Align.center("[dim]← → 选择  Enter=确认  D/l/o/k/s/r/c=直接操作  Q=关闭[/dim]"))
    return Group(*items)


# ── Menu interaction ─────────────────────────────────────


def run_menu_screen(s: State, console: Console):
    """Popup overlay mode — ↑↓ navigates jobs, ←→ navigates popup options."""
    job = s.selected_job
    if not job:
        s.menu_open = False
        return

    s.menu_cursor = 0
    s.clamp_menu_cursor()

    while True:
        console.clear()
        console.print(render_summary(s.all, s.search_query))
        console.print()
        console.print(render_table(s))
        console.print()
        console.print(Align.center(make_menu_popup(s)))
        console.print(Align.center("[dim]↑↓ 选择任务  ← → 选择操作  Enter=确认  D/l/o/k/s/r/c=直接操作[/dim]"))

        key = read_key()

        if key in ("q", "Q", "ESC", "\x1b"):
            s.menu_open = False
            return

        # ── Direct action keystrokes ──────────────────────
        if key in ("d", "D"):
            action = "detail"
        elif key in ("o", "O"):
            action = "open_dir"
        elif key in ("k", "K"):
            action = "cancel"
        elif key in ("s",):
            action = "resubmit"
        elif key in ("u", "U"):
            action = "user_jobs"
        elif key in ("r",):
            action = "delete"
        elif key in ("c", "C"):
            action = "close"
        else:
            action = None

        if action:
            if action == "close":
                s.menu_open = False
                return

            elif action == "detail":
                console.clear()
                console.print(render_detail(job, "detail"))
                console.print("\n[dim]按任意键返回...[/dim]")
                os.read(sys.stdin.fileno(), 1)

            elif action == "open_dir":
                if job.output_dir:
                    subprocess.Popen(["ghostty", "--working-directory=" + job.output_dir,
                                     "-e", "bash", "-c", "echo Open: " + job.access_code + "; exec $SHELL"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            elif action == "output":
                run_output_browser(job, console)

            elif action == "user_jobs":
                show_user_jobs_tui(job, s, console)
                continue

            elif action == "cancel":
                cancel_job(job.access_code)
                s._message = f"✅ 任务 {job.access_code} 已取消"
                s.menu_open = False
                s.reload(reset_page=False)
                return

            elif action == "resubmit":
                checkpoint_str = get_checkpoint(job.access_code)
                steps = [st.strip() for st in checkpoint_str.split(",") if st.strip()] if checkpoint_str else []

                if steps:
                    while True:
                        console.clear()
                        console.print(render_summary(s.all, s.search_query))
                        console.print()
                        console.print(render_table(s))
                        console.print()
                        lines = [f"[bold yellow]重新提交任务 {job.access_code}[/]", ""]
                        lines.append("[dim]选择从哪个步骤重新开始（该步骤及之后将被清除）：[/]")
                        lines.append("")
                        for i, step in enumerate(steps):
                            lines.append(f"  {i+1}. {step}")
                        lines.append(f"  0. [从头开始]")
                        lines.append("")
                        lines.append("[dim]输入数字  Enter=确认  ESC=取消[/dim]")
                        console.print(Panel("\n".join(lines), border_style="yellow", padding=(1, 2), width=60))
                        
                        # Read a single digit
                        ch = os.read(sys.stdin.fileno(), 1)
                        if not ch:
                            continue
                        c = ch.decode("utf-8", errors="replace")
                        if c == "\x1b":
                            break
                        elif c.isdigit():
                            choice = int(c)
                            if choice == 0:
                                keep_steps = []
                            elif 1 <= choice <= len(steps):
                                keep_steps = steps[:choice-1]
                            else:
                                continue
                            try:
                                msg = resubmit_job(job.access_code, keep_steps=keep_steps)
                                s.message(msg)
                                s.menu_open = False
                                s.reload(reset_page=False)
                                return
                            except Exception as e:
                                s.message(f"❌ 重新提交失败: {e}")
                                s.menu_open = False
                                s.reload(reset_page=False)
                                return
                else:
                    console.clear()
                    console.print(render_summary(s.all, s.search_query))
                    console.print()
                    console.print(render_table(s))
                    console.print()
                    confirm_panel = Panel(
                        Align.center(
                            f"[bold yellow]确定要重新提交任务 {job.access_code} 吗？[/]\n\n"
                            f"[dim]无检查点，将从头开始。[/]\n\n"
                            f"[reverse] Y/Enter=确认  N=取消 [/]"
                        ),
                        border_style="yellow", padding=(1, 2), width=60,
                    )
                    console.print(Align.center(confirm_panel))
                    while True:
                        ch = os.read(sys.stdin.fileno(), 1)
                        if not ch:
                            continue
                        ch = ch.decode("utf-8", errors="replace").lower()
                        if ch in ("y", "\r", "\n"):
                            msg = resubmit_job(job.access_code)
                            s._message = msg
                            s.menu_open = False
                            s.reload(reset_page=False)
                            return
                        elif ch in ("n", "\x1b", "q"):
                            break
                continue

            elif action == "delete":
                console.clear()
                console.print(render_summary(s.all, s.search_query))
                console.print()
                console.print(render_table(s))
                console.print()
                confirm_panel = Panel(
                    Align.center(f"[bold yellow]确定要删除任务 {job.access_code} 吗？[/]\n\n[reverse] Y/Enter=确认  N=取消 [/]"),
                    border_style="red",
                    padding=(1, 2),
                    width=50,
                )
                console.print(Align.center(confirm_panel))
                while True:
                    ch = os.read(sys.stdin.fileno(), 1)
                    if not ch:
                        continue
                    ch = ch.decode("utf-8", errors="replace").lower()
                    if ch in ("y", "\r", "\n"):
                        delete_job(job.access_code)
                        s._message = f"✅ 任务 {job.access_code} 已删除"
                        s.menu_open = False
                        s.reload(reset_page=False)
                        return
                    elif ch in ("n", "\x1b", "q"):
                        break
                continue
            continue

        # ── Job list navigation ──────────────────────────
        if key == "UP":
            if s.cursor > 0:
                s.cursor -= 1
            elif s.page < s.pages - 1:
                s.page += 1
                s.cursor = len(s.visible) - 1
            s.menu_cursor = 0
            s.clamp_menu_cursor()

        elif key == "DOWN":
            max_cursor = len(s.visible) - 1
            if s.cursor < max_cursor:
                s.cursor += 1
            elif s.page > 0:
                s.page -= 1
                s.cursor = 0
            s.menu_cursor = 0
            s.clamp_menu_cursor()

        # ── Popup navigation ─────────────────────────────
        elif key in ("LEFT",):
            s.menu_cursor -= 1
            s.clamp_menu_cursor()

        elif key in ("RIGHT",):
            s.menu_cursor += 1
            s.clamp_menu_cursor()

        # ── Confirm action ───────────────────────────────
        elif key in ("\r", "\n"):
            job = s.selected_job
            if not job:
                continue
            opts = s.menu_options()
            if s.menu_cursor >= len(opts):
                continue
            action = opts[s.menu_cursor][0]

            if action == "close":
                s.menu_open = False
                return

            elif action == "detail":
                console.clear()
                console.print(render_detail(job, "detail"))
                console.print("\n[dim]按任意键返回...[/dim]")
                os.read(sys.stdin.fileno(), 1)

            elif action == "open_dir":
                if job.output_dir:
                    subprocess.Popen(["ghostty", "--working-directory=" + job.output_dir,
                                     "-e", "bash", "-c", "echo Open: " + job.access_code + "; exec $SHELL"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            elif action == "output":
                run_output_browser(job, console)

            elif action == "user_jobs":
                show_user_jobs_tui(job, s, console)
                continue

            elif action == "cancel":
                cancel_job(job.access_code)
                s._message = f"✅ 任务 {job.access_code} 已取消"
                s.menu_open = False
                s.reload(reset_page=False)
                return

            elif action == "resubmit":
                checkpoint_str = get_checkpoint(job.access_code)
                steps = [st.strip() for st in checkpoint_str.split(",") if st.strip()] if checkpoint_str else []

                if steps:
                    while True:
                        console.clear()
                        console.print(render_summary(s.all, s.search_query))
                        console.print()
                        console.print(render_table(s))
                        console.print()
                        lines = [f"[bold yellow]重新提交任务 {job.access_code}[/]", ""]
                        lines.append("[dim]选择从哪个步骤重新开始（该步骤及之后将被清除）：[/]")
                        lines.append("")
                        for i, step in enumerate(steps):
                            lines.append(f"  {i+1}. {step}")
                        lines.append(f"  0. [从头开始]")
                        lines.append("")
                        lines.append("[dim]输入数字  Enter=确认  ESC=取消[/dim]")
                        console.print(Panel("\n".join(lines), border_style="yellow", padding=(1, 2), width=60))
                        
                        # Read a single digit
                        ch = os.read(sys.stdin.fileno(), 1)
                        if not ch:
                            continue
                        c = ch.decode("utf-8", errors="replace")
                        if c == "\x1b":
                            break
                        elif c.isdigit():
                            choice = int(c)
                            if choice == 0:
                                keep_steps = []
                            elif 1 <= choice <= len(steps):
                                keep_steps = steps[:choice-1]
                            else:
                                continue
                            try:
                                msg = resubmit_job(job.access_code, keep_steps=keep_steps)
                                s.message(msg)
                                s.menu_open = False
                                s.reload(reset_page=False)
                                return
                            except Exception as e:
                                s.message(f"❌ 重新提交失败: {e}")
                                s.menu_open = False
                                s.reload(reset_page=False)
                                return
                else:
                    console.clear()
                    console.print(render_summary(s.all, s.search_query))
                    console.print()
                    console.print(render_table(s))
                    console.print()
                    confirm_panel = Panel(
                        Align.center(
                            f"[bold yellow]确定要重新提交任务 {job.access_code} 吗？[/]\n\n"
                            f"[dim]无检查点，将从头开始。[/]\n\n"
                            f"[reverse] Y/Enter=确认  N=取消 [/]"
                        ),
                        border_style="yellow", padding=(1, 2), width=60,
                    )
                    console.print(Align.center(confirm_panel))
                    while True:
                        ch = os.read(sys.stdin.fileno(), 1)
                        if not ch:
                            continue
                        ch = ch.decode("utf-8", errors="replace").lower()
                        if ch in ("y", "\r", "\n"):
                            msg = resubmit_job(job.access_code)
                            s._message = msg
                            s.menu_open = False
                            s.reload(reset_page=False)
                            return
                        elif ch in ("n", "\x1b", "q"):
                            break
                continue

            elif action == "delete":
                # ── Confirmation dialog ──
                console.clear()
                console.print(render_summary(s.all, s.search_query))
                console.print()
                console.print(render_table(s))
                console.print()
                confirm_panel = Panel(
                    Align.center(f"[bold yellow]确定要删除任务 {job.access_code} 吗？[/]\n\n[reverse] Y/Enter=确认  N=取消 [/]"),
                    border_style="red",
                    padding=(1, 2),
                    width=50,
                )
                console.print(Align.center(confirm_panel))
                while True:
                    ch = os.read(sys.stdin.fileno(), 1)
                    if not ch:
                        continue
                    ch = ch.decode("utf-8", errors="replace").lower()
                    if ch in ("y", "\r", "\n"):
                        delete_job(job.access_code)
                        s._message = f"✅ 任务 {job.access_code} 已删除"
                        s.menu_open = False
                        s.reload(reset_page=False)
                        return
                    elif ch in ("n", "\x1b", "q"):
                        break
                continue


# ── Search mode ────────────────────────────────────────


def run_search(s: State, console: Console):
    """Interactive search input — each keystroke filters the job list in real-time."""
    query = ""
    while True:
        console.clear()
        console.print(render_summary(s.all, s.search_query))
        console.print()
        console.print(render_table(s))
        console.print()
        search_prompt = f"🔍 搜索访问码: [reverse] {query}▌ " if query else "🔍 搜索访问码: [dim]输入访问码...[/]"
        console.print(Align.center(Panel(search_prompt, border_style="cyan", padding=(0, 1), width=50)))
        console.print(Align.center("[dim]输入搜索  ESC=清除  Enter=关闭[/dim]"))

        ch = os.read(sys.stdin.fileno(), 1)
        if not ch:
            continue
        ch = ch.decode("utf-8", errors="replace")

        if ch == "\x1b":
            # Could be ESC or escape sequence — check for more bytes
            import select
            r, _, _ = select.select([sys.stdin.fileno()], [], [], 0.1)
            if not r:
                # Pure ESC — clear search, keep list filtered
                s.search_query = ""
                s.reload()
                return
            else:
                # Escape sequence, consume and ignore
                seq = os.read(sys.stdin.fileno(), 1)
                continue

        if ch in ("\r", "\n"):
            return  # Exit search, keep filter

        if ch == "\x7f" or ch == "\b":
            # Backspace
            query = query[:-1]
            s.search_query = query
            s.reload()
            continue

        # Printable character
        if ch.isprintable():
            query += ch.upper()
            s.search_query = query
            s.reload()


# ── Main interactive loop ─────────────────────────────────


def interactive(console: Console):
    raw_jobs, total = load_jobs()
    if not raw_jobs:
        console.print("[yellow]数据库中没有任务[/]")
        return

    s = State(raw_jobs, total)
    s.clamp_cursor(at_bottom=True)
    s._orphans = detect_orphans(force=True)
    s._orphans_last_scan = time.monotonic()
    console.clear()
    last_auto_refresh = time.monotonic()

    with raw_mode():
        while True:
            s.clear_expired_message()

            # Auto-refresh check
            if s.auto_refresh:
                now = time.monotonic()
                if now - last_auto_refresh >= AUTO_REFRESH_SEC:
                    s.reload(reset_page=False)
                    last_auto_refresh = now
                    console.clear()
                    console.print(Group(*render_all(s)))

            if s.menu_open:
                run_menu_screen(s, console)
                console.clear()
                continue

            console.clear()
            console.print(Group(*render_all(s)))

            # Determine read timeout for auto-refresh responsiveness
            timeout = None
            if s.auto_refresh:
                elapsed = time.monotonic() - last_auto_refresh
                timeout = max(0.05, AUTO_REFRESH_SEC - elapsed)

            key = read_key(timeout=timeout)

            if key == "":
                continue

            if key in ("q", "Q") and s.focus != "nav":
                break

            # ── Tab: switch focus ────────────────────────
            if key == "\t":
                s.focus = "nav" if s.focus == "table" else "table"
                s.nav_cursor = 0
                s.clamp_nav_cursor()
                continue

            # ── Focus: Navigation bar ────────────────────
            if s.focus == "nav":
                if key in ("LEFT", "UP"):
                    s.nav_cursor -= 1
                    s.clamp_nav_cursor()
                elif key in ("RIGHT", "DOWN"):
                    s.nav_cursor += 1
                    s.clamp_nav_cursor()
                elif key in ("\r", "\n", " "):
                    items = s.NAV_ITEMS
                    if 0 <= s.nav_cursor < len(items):
                        action = items[s.nav_cursor][0]
                        if action == "search":
                            run_search(s, console)
                            console.clear()
                            continue
                        msg = s.execute_nav_action(action)
                        if msg is None:
                            break  # quit
                        elif msg:
                            s.message(msg)
                elif key == "\x1b":  # ESC -> focus back to table
                    s.focus = "table"
                continue  # redraw

            # ── Focus: Table ─────────────────────────────
            if key == "UP":
                if s.cursor > 0:
                    s.cursor -= 1
                elif s.page < s.pages - 1:
                    s.page += 1
                    s.cursor = len(s.visible) - 1

            elif key == "DOWN":
                max_cursor = len(s.visible) - 1
                if s.cursor < max_cursor:
                    s.cursor += 1
                elif s.page > 0:
                    s.page -= 1
                    s.cursor = 0

            elif key in ("\r", "\n", "RIGHT"):
                if s.selected_job:
                    s.menu_open = True

            elif key == "/":
                run_search(s, console)
                console.clear()
                continue

            elif key in ("n", "N"):
                s.search_query = ""
                s._orphans = detect_orphans(force=True)
                s._orphans_last_scan = time.monotonic()
                s.reload()
                s.message("✅ 已刷新")

            elif key in ("k", "K"):
                orphans = detect_orphans(force=True)
                if not orphans:
                    s.message("没有孤儿进程需要清理")
                else:
                    killed = 0
                    for o in orphans:
                        try:
                            os.kill(o["pid"], signal.SIGTERM)
                            killed += 1
                        except ProcessLookupError:
                            killed += 1  # already dead
                        except Exception as e:
                            s.message(f"❌ 无法杀掉 PID {o['pid']}: {e}")
                    if killed:
                        # Give them a moment, then SIGKILL survivors
                        time.sleep(1)
                        for o in detect_orphans(force=True):
                            try:
                                os.kill(o["pid"], signal.SIGKILL)
                                killed += 1
                            except Exception:
                                pass
                    s._orphans = detect_orphans(force=True)
                    s._orphans_last_scan = time.monotonic()
                    remaining = len(s._orphans)
                    if remaining:
                        s.message(f"⚠ 杀掉了 {killed - remaining} 个孤儿进程，{remaining} 个幸存")
                    else:
                        s.message(f"✅ 已杀掉 {killed} 个孤儿进程")

            elif key in ("r", "R"):
                s.auto_refresh = not s.auto_refresh
                last_auto_refresh = time.monotonic()
                label = "开启" if s.auto_refresh else "关闭"
                s.message(f"🔁 自动刷新已{label}")

            elif key == "ESC":
                if s.menu_open:
                    s.menu_open = False


# ── One-shot CLI modes ────────────────────────────────────


def list_mode(console: Console):
    jobs, total = load_jobs()
    if not jobs:
        console.print("[yellow]数据库中没有任务[/]")
        return
    s = State(jobs, total)
    console.print(render_summary(jobs, ""))
    console.print()
    console.print(render_table(s))


def status_mode(code: str, console: Console):
    job = get_job(code.upper())
    if not job:
        console.print(f"[red]任务 {code.upper()} 不存在[/]")
        return
    console.print(render_detail(job, "detail"))


def log_mode(code: str, console: Console):
    job = get_job(code.upper())
    if not job:
        console.print(f"[red]任务 {code.upper()} 不存在[/]")
        return
    console.print(render_detail(job, "log"))


def watch_mode(code: str, interval: int, console: Console):
    code = code.upper()
    try:
        while True:
            job = get_job(code)
            if not job:
                console.print(f"[red]任务 {code} 不存在[/]")
                break
            console.clear()
            total = count_jobs()
            console.print(
                Panel(
                    Align.center(f"总计 [bold]{total}[/]"),
                    border_style="dim",
                    padding=(0, 0),
                )
            )
            console.print()
            console.print(render_detail(job, "detail"))
            if job.status in ("completed", "failed"):
                status_text = (
                    "[bold green]✓ 任务已完成[/]"
                    if job.status == "completed"
                    else "[bold red]✗ 任务已失败[/]"
                )
                console.print(f"\n{status_text}")
                break
            time.sleep(max(1, interval))
    except KeyboardInterrupt:
        pass


# ── Entry point ───────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="📋 任务队列 TUI")
    parser.add_argument("--list", "-l", action="store_true", help="列出所有任务")
    parser.add_argument("--status", "-s", type=str, help="查看任务状态")
    parser.add_argument("--log", type=str, help="查看任务日志")
    parser.add_argument("--cancel", "-k", type=str, help="取消任务")
    parser.add_argument("--delete", "-d", type=str, help="删除任务")
    parser.add_argument("--purge", action="store_true", help="清除所有已删除的任务")
    parser.add_argument("--dry-run", action="store_true", help="配合 --purge 使用，预览会清除的内容而不实际删除")
    parser.add_argument("--watch", "-w", type=str, help="持续监控任务状态")
    parser.add_argument("--interval", "-i", type=int, default=5, help="监控间隔秒数")
    args = parser.parse_args()

    console = Console()

    if args.list:
        list_mode(console)
    elif args.status:
        status_mode(args.status.upper(), console)
    elif args.log:
        log_mode(args.log.upper(), console)
    elif args.cancel:
        print(cancel_job(args.cancel.upper()))
    elif args.delete:
        print(delete_job(args.delete.upper()))
    elif args.purge:
        try:
            # Import and delegate to the unified JobQueue.clear_job_queue()
            sys.path.insert(0, str(HERE / "chatterbox-server"))
            from jobqueue import get_job_queue
            jq = get_job_queue()
            result = jq.clear_job_queue(dry_run=args.dry_run)
            if result["success"]:
                print("✓ " + result["message"])
            else:
                print("✗ " + result.get("message", "Failed"))
                if "error" in result:
                    print(f"  Error: {result['error']}")
                if "errors" in result:
                    for e in result["errors"]:
                        print(f"  Error: {e}")
        except Exception as e:
            print(f"✗ 清除失败: {e}")
    elif args.watch:
        watch_mode(args.watch.upper(), args.interval, console)
    else:
        interactive(console)


if __name__ == "__main__":
    main()
