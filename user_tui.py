#!/usr/bin/env python3
"""
User list TUI — cursor-based, paginated, keyboard-driven.

Usage:
    python user_tui.py
"""

from __future__ import annotations

import os
import select
import sqlite3
import sys
import termios
import tty
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ── Paths ─────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent
USERS_DB = Path(os.environ.get("USERS_DB", str(HERE / "chatterbox-server" / "users.db")))
JOBS_DB = Path(os.environ.get("JOBS_DB", str(HERE / "chatterbox-server" / "jobs.db")))
PAGE_SIZE = 20


# ── Data types ────────────────────────────────────────────

@dataclass
class User:
    id: int
    email: str
    verified: int
    created_at: str | None


# ── DB helpers ────────────────────────────────────────────

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(USERS_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


def get_jobs_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(JOBS_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


def load_users(limit: int = 500, offset: int = 0) -> tuple[list[User], int]:
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    rows = conn.execute(
        "SELECT id, email, verified, created_at FROM users ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    conn.close()
    users = [User(**dict(r)) for r in rows]
    return users, total


_TYPE_MAP = {
    "_run_gen_audio": "音频生成",
    "_run_video_job": "宁视频翻译",
    "_run_video_custom_job": "自定义视频",
    "_run_tts_job": "语音合成",
    "_run_video_auto_job": "自动翻译视频",
    "_run_audio_segmentation_job": "音频文件合成",
}
_STATUS_STYLE = {
    "pending": "yellow",
    "processing": "cyan bold",
    "completed": "green",
    "failed": "red bold",
    "deleted": "white dim",
}
_STATUS_LABEL = {
    "pending": "等待中",
    "processing": "处理中",
    "completed": "已完成",
    "failed": "已失败",
    "deleted": "已删除",
}


def load_user_jobs(user_id: int) -> list[dict]:
    conn = get_jobs_conn()
    rows = conn.execute(
        "SELECT access_code, run_func_name, status, error, created_at "
        "FROM jobs WHERE user_id = ? AND status != 'deleted' ORDER BY created_at DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    jobs = []
    for r in rows:
        d = dict(r)
        d["type"] = _TYPE_MAP.get(d.get("run_func_name"), d.get("run_func_name") or "未知")
        jobs.append(d)
    return jobs


# ── Rendering ─────────────────────────────────────────────

def render_table(users: list[User], cursor: int, page: int, pages: int, total: int) -> Panel:
    t = Table(box=box.SIMPLE, header_style="bold", show_edge=False, padding=(0, 1))
    t.add_column("", width=2)
    t.add_column("#", style="dim", width=4)
    t.add_column("ID", width=4)
    t.add_column("邮箱", width=32, overflow="fold")
    t.add_column("已验证", width=8)
    t.add_column("注册时间", width=20, no_wrap=True)

    for i, u in enumerate(users):
        is_cursor = i == cursor
        row_style = "reverse" if is_cursor else ""
        marker = "▸" if is_cursor else " "
        idx = page * PAGE_SIZE + i + 1
        verified = "✓" if u.verified else "✗"
        t.add_row(
            Text(marker, style="cyan bold" if is_cursor else ""),
            str(idx),
            str(u.id),
            u.email,
            Text(verified, style="green" if u.verified else "red"),
            u.created_at or "",
            style=row_style,
        )

    summary = f"总计 {total}  |  已验证 {sum(1 for u in users if u.verified)}/{total}"
    header = Panel(summary, padding=(0, 1))
    return Panel(t, title=f"用户列表 — 第 {page+1}/{pages} 页", border_style="cyan")


# ── Key input ─────────────────────────────────────────────

@contextmanager
def raw_mode():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


_KEY_MAP: dict[str, str] = {
    "\x1b[A": "UP",
    "\x1b[B": "DOWN",
    "\x1b[C": "RIGHT",
    "\x1b[D": "LEFT",
    "\x1b": "ESC",
    "q": "q",
    "Q": "Q",
    "n": "n",
    "N": "N",
    "\r": "ENTER",
    "\n": "ENTER",
}


def read_key() -> str:
    """Read a single keypress from stdin (must be in raw mode)."""
    seq = os.read(sys.stdin.fileno(), 1)
    if seq == b"\x1b":
        rest = b""
        while select.select([sys.stdin], [], [], 0)[0]:
            rest += os.read(sys.stdin.fileno(), 1)
        if rest:
            return _KEY_MAP.get("\x1b" + rest.decode(), "ESC")
        return "ESC"
    ch = seq.decode("utf-8", errors="replace")
    return _KEY_MAP.get(ch, ch)


# ── Job listing screen ───────────────────────────────────

def show_user_jobs(user: User, console: Console):
    """Show all jobs belonging to a user. Press any key to return."""
    jobs = load_user_jobs(user.id)
    if not jobs:
        console.clear()
        console.print(f"[yellow]用户 {user.email} 没有任何任务[/]")
        console.print("\n[dim]按任意键返回...[/dim]")
        os.read(sys.stdin.fileno(), 1)
        return

    console.clear()
    title = f"📋 {user.email} 的任务 ({len(jobs)})"

    t = Table(box=box.SIMPLE, header_style="bold", show_edge=False, padding=(0, 1))
    t.add_column("#", style="dim", width=4)
    t.add_column("访问码", width=9)
    t.add_column("类型", width=14)
    t.add_column("状态", width=10, no_wrap=True)
    t.add_column("错误", width=28, overflow="fold")
    t.add_column("时间", width=16, no_wrap=True)

    for i, j in enumerate(jobs):
        status_style = _STATUS_STYLE.get(j["status"], "white")
        status_label = _STATUS_LABEL.get(j["status"], j["status"])
        t.add_row(
            str(i + 1),
            Text(j.get("access_code", ""), style="bold"),
            j.get("type", ""),
            Text(status_label, style=status_style),
            (j.get("error") or "")[:60],
            j.get("created_at") or "",
        )

    console.print(Panel(t, title=title, border_style="cyan"))
    console.print("\n[dim]按任意键返回用户列表...[/dim]")
    os.read(sys.stdin.fileno(), 1)


# ── Main ──────────────────────────────────────────────────

def main():
    console = Console()
    users, total = load_users()
    if not users:
        console.print("[yellow]数据库中没有用户[/]")
        return

    pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = 0
    cursor = 0

    while True:
        start = page * PAGE_SIZE
        end = min(start + PAGE_SIZE, total)
        page_users = users[start:end]

        console.clear()
        console.print(render_table(page_users, cursor, page, pages, total))
        console.print()
        console.print(Align.center("[dim]↑↓ 上下移动  N=刷新  Q=退出[/dim]"))

        with raw_mode():
            key = read_key()

        if key in ("q", "Q"):
            break

        elif key == "UP":
            if cursor > 0:
                cursor -= 1
            elif page > 0:
                page -= 1
                cursor = PAGE_SIZE - 1

        elif key == "DOWN":
            max_cursor = len(page_users) - 1
            if cursor < max_cursor:
                cursor += 1
            elif page < pages - 1:
                page += 1
                cursor = 0

        elif key in ("ENTER",):
            u = page_users[cursor]
            show_user_jobs(u, console)
            # Reload user list after returning
            users, total = load_users()
            pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)

        elif key in ("n", "N"):
            users, total = load_users()
            pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)


if __name__ == "__main__":
    main()
