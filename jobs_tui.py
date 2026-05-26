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

# ── Data types ────────────────────────────────────────────

_TYPE_MAP = {
    "_run_gen_audio": "音频生成",
    "_run_video_job": "宁视频翻译",
    "_run_video_custom_job": "自定义视频",
    "_run_tts_job": "语音合成",
    "_run_video_auto_job": "自动翻译视频",
    "_run_audio_file_job": "音频文件合成",
    "_run_audio_segmentation_job": "音频文件合成",
}
_STATUS_STYLE_MAP = {
    "pending": "yellow",
    "processing": "cyan bold",
    "completed": "green",
    "failed": "red bold",
    "deleted": "white dim",
}
_STATUS_LABEL_MAP = {
    "pending": "等待中",
    "processing": "处理中",
    "completed": "已完成",
    "failed": "已失败",
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

    @property
    def display_type(self) -> str:
        return _TYPE_MAP.get(self.run_func_name or "", self.run_func_name or "未知")

    @property
    def status_style(self) -> str:
        return _STATUS_STYLE_MAP.get(self.status, "white")

    @property
    def status_label(self) -> str:
        return _STATUS_LABEL_MAP.get(self.status, self.status)


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
            COALESCE(status_changed_at, created_at) DESC
    """
    if search:
        pattern = f"%{search}%"
        total = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE access_code LIKE ?", (pattern,)
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT access_code, run_func_name, status, error, output_dir, srt_path, created_at, user_id, status_changed_at "
            "FROM jobs WHERE access_code LIKE ? " + order_clause + " LIMIT ? OFFSET ?",
            (pattern, limit, offset),
        ).fetchall()
    else:
        total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        rows = conn.execute(
            "SELECT access_code, run_func_name, status, error, output_dir, srt_path, created_at, user_id, status_changed_at "
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
        "SELECT access_code, run_func_name, status, error, output_dir, srt_path, created_at, user_id, status_changed_at "
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
        "UPDATE jobs SET status = ?, error = ?, status_changed_at = ? WHERE access_code = ?",
        ("failed", "用户取消", time.strftime('%Y-%m-%d %H:%M:%S'), code),
    )
    conn.commit()
    conn.close()
    return f"✅ 任务 {code} 已取消"


def delete_job(code: str) -> str:
    code = code.upper()
    conn = get_conn()
    job = conn.execute(
        "SELECT output_dir FROM jobs WHERE access_code = ?", (code,)
    ).fetchone()
    if not job:
        conn.close()
        return f"任务 {code} 不存在"
    output_dir = job["output_dir"]
    conn.execute("DELETE FROM jobs WHERE access_code = ?", (code,))
    conn.commit()
    conn.close()
    if output_dir and os.path.isdir(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
    return f"✅ 任务 {code} 已删除"


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

    def reload(self):
        self.all, self.total = load_jobs(search=self.search_query)
        self.pages = max(1, (self.total + PAGE_SIZE - 1) // PAGE_SIZE)
        self.clamp_cursor()
        self.page = 0

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


# ── Render functions ──────────────────────────────────────


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
    if s._message:
        items.append(
            Panel(
                Align.center(Text(s._message, style="bold")),
                border_style="green" if s._message.startswith("✅") else "red",
                padding=(0, 0),
            )
        )
    return items


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
        "SELECT access_code, run_func_name, status, error, output_dir, srt_path, created_at, user_id, status_changed_at "
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
        items.append(Align.center("[dim]← → 选择  Enter=确认  D/l/o/k/r/c=直接操作  Q=关闭[/dim]"))
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
        console.print(Align.center("[dim]↑↓ 选择任务  ← → 选择操作  Enter=确认  D/l/o/k/r/c=直接操作[/dim]"))

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
        elif key in ("u", "U"):
            action = "user_jobs"
        elif key in ("r", "R"):
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
                s.reload()
                return

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
                        s.reload()
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
                s.reload()
                return

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
                        s.reload()
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
    console.clear()
    last_auto_refresh = time.monotonic()

    with raw_mode():
        while True:
            s.clear_expired_message()

            # Auto-refresh check
            if s.auto_refresh:
                now = time.monotonic()
                if now - last_auto_refresh >= AUTO_REFRESH_SEC:
                    s.reload()
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
                s.reload()
                s.message("✅ 已刷新")

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
    elif args.watch:
        watch_mode(args.watch.upper(), args.interval, console)
    else:
        interactive(console)


if __name__ == "__main__":
    main()
