#!/usr/bin/env python3
"""Client for the shared warm ROCm OCR daemon (lives in
code_ml/rapid_videocr_daemon, serves both batch and code_ml).
Pure stdlib — runs under any python3.

Commands:
  daemon_ocr_client.py ping
      Attach/start check. Exit 0 if the daemon answers a ping.
  daemon_ocr_client.py shutdown
      Graceful shutdown request (finishes in-flight jobs first).
  daemon_ocr_client.py ocr IMG_DIR SAVE_DIR FILE_NAME
      Run one OCR job, waiting with backoff while the daemon is busy.
      Exit 0 on success (writes SAVE_DIR/FILE_NAME.srt), non-zero otherwise.

When an OCR request arrives and no daemon is running, this client launches
the daemon itself (one-shot subprocess, logs to ~/logs/rapid_videocr_daemon.log)
and waits for it to become ready.

Runtime socket/pid live in $XDG_RUNTIME_DIR/rapid_videocr_daemon/ (tmpfs).
"""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

DAEMON_DIR = Path(__file__).resolve().parent / "rapid_videocr_daemon"


def _runtime_dir():
    base = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    if not base or not Path(base).is_dir():
        base = "/tmp"
    return Path(base) / "rapid_videocr_daemon"


DAEMON_SOCK = Path(
    os.environ.get("RAPID_VIDEOCR_DAEMON_SOCK", _runtime_dir() / "daemon.sock")
).resolve()
DAEMON_SCRIPT = Path(
    os.environ.get("RAPID_VIDEOCR_DAEMON_SCRIPT", DAEMON_DIR / "rapid_videocr_daemon.py")
).resolve()
DAEMON_PYTHON = os.environ.get(
    "RAPID_VIDEOCR_DAEMON_PYTHON", DAEMON_DIR / ".venv-rocm/bin/python"
)
DAEMON_LOG = Path(
    os.environ.get(
        "RAPID_VIDEOCR_DAEMON_LOG",
        os.path.expanduser("~/logs/rapid_videocr_daemon.log"),
    )
).resolve()

DAEMON_PING_TIMEOUT = float(os.environ.get("DAEMON_PING_TIMEOUT", "10"))
DAEMON_START_WAIT = float(os.environ.get("DAEMON_START_WAIT", "120"))
DAEMON_JOB_TIMEOUT = float(os.environ.get("DAEMON_JOB_TIMEOUT", str(4 * 3600)))
DAEMON_MAX_JOBS = os.environ.get("DAEMON_MAX_JOBS", "2")


def _daemon_request(payload, timeout=30):
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(str(DAEMON_SOCK))
        s.sendall((json.dumps(payload) + "\n").encode())
        data = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
        s.close()
        if not data:
            return None
        return json.loads(data.split(b"\n", 1)[0].decode())
    except Exception:
        try:
            s.close()
        except Exception:
            pass
        return None


def _daemon_alive():
    resp = _daemon_request({"cmd": "ping"}, timeout=DAEMON_PING_TIMEOUT)
    return resp is not None and resp.get("ok") is True


def _ensure_daemon():
    """Attach to a running daemon or launch one. Returns True when ready."""
    if _daemon_alive():
        return True

    DAEMON_LOG.parent.mkdir(parents=True, exist_ok=True)
    logf = open(DAEMON_LOG, "a")
    env = dict(os.environ)
    env["RAPID_VIDEOCR_DAEMON_SOCK"] = str(DAEMON_SOCK)
    env["DAEMON_MAX_JOBS"] = DAEMON_MAX_JOBS
    try:
        subprocess.Popen(
            [DAEMON_PYTHON, str(DAEMON_SCRIPT)],
            stdout=logf,
            stderr=logf,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            env=env,
        )
    except Exception:
        return False

    deadline = time.time() + DAEMON_START_WAIT
    while time.time() < deadline:
        if _daemon_alive():
            return True
        time.sleep(1)
    return False


def _shutdown_daemon():
    if not _daemon_alive():
        print("daemon not running")
        return 0
    _daemon_request({"cmd": "shutdown"}, timeout=DAEMON_PING_TIMEOUT)
    deadline = time.time() + DAEMON_PING_TIMEOUT
    while time.time() < deadline and DAEMON_SOCK.exists():
        time.sleep(0.5)
    print("shutdown requested")
    return 0


def _run_ocr(img_dir, save_dir, file_name):
    if not _ensure_daemon():
        print("daemon failed to start", file=sys.stderr)
        return 1

    deadline = time.time() + DAEMON_JOB_TIMEOUT
    attempts = 0
    while True:
        resp = _daemon_request(
            {
                "cmd": "ocr",
                "img_dir": str(img_dir),
                "save_dir": str(save_dir),
                "file_name": str(file_name),
                "out_format": "srt",
            },
            timeout=DAEMON_JOB_TIMEOUT,
        )
        if resp is None:
            return 1
        if resp.get("code") == "busy":
            attempts += 1
            retry_after = float(resp.get("retry_after", 5))
            if time.time() + retry_after >= deadline:
                print(
                    f"daemon busy ({attempts} attempts) - timed out waiting",
                    file=sys.stderr,
                )
                return 1
            time.sleep(retry_after)
            continue
        if not resp.get("ok"):
            print(resp.get("error", "ocr failed"), file=sys.stderr)
            return 1
        return 0


def main(argv):
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2

    cmd = argv[1]
    if cmd == "ping":
        return 0 if _daemon_alive() else 1
    if cmd == "shutdown":
        return _shutdown_daemon()
    if cmd == "ocr" and len(argv) == 5:
        return _run_ocr(argv[2], argv[3], argv[4])
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))