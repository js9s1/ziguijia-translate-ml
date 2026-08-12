#!/home/ziguijia/子归家/code_ml/rapid_videocr_daemon/.venv-rocm/bin/python
"""
Warm ROCm OCR daemon, shared by the batch pipeline and code_ml.

Loads rapid_videocr + the rollback torch/ROCm models ONCE per worker and
keeps them resident, serving OCR jobs over a Unix socket. Up to
DAEMON_MAX_JOBS (default 2) jobs run concurrently, one per worker thread
(each worker owns its own RapidVideOCR instance — required because the
extractor keeps per-call mutable state).

Daemon does NOT do anything while idle — it blocks on accept(), so CPU
usage is ~0% when no job is running.

Runtime files: socket + pid live in $XDG_RUNTIME_DIR/rapid_videocr_daemon/
(tmpfs, 0700, auto-cleaned on reboot); the log is ~/logs/rapid_videocr_daemon.log.

Protocol (newline-delimited JSON per request, connection-per-request):
  request  {"cmd": "ping"}
  response {"ok": true, "engine": "torch/rocm", "device": "..."}

  request  {"cmd": "ocr", "img_dir": "...", "save_dir": "...",
            "file_name": "result_chunk0", "out_format": "srt"}
  response {"ok": true, "srt": "/path/to/result_chunk0.srt"}
    or     {"ok": false, "error": "..."}
    or     {"ok": false, "code": "busy", "retry_after": 5}
           (all DAEMON_MAX_JOBS workers busy — client should retry later;
            the request was NOT queued or run)

  request  {"cmd": "shutdown"}
  response {"ok": true}

Oversized / malformed requests are answered with an error and dropped.
SIGTERM or SIGINT stops the daemon cleanly after in-flight jobs finish.
"""
import json
import os
import select
import signal
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if sys.stdout is not None:
    sys.stdout.reconfigure(line_buffering=True)
if sys.stderr is not None:
    sys.stderr.reconfigure(line_buffering=True)

os.environ.setdefault(
    "LD_LIBRARY_PATH",
    f"/opt/rocm/rocm/lib{os.environ.get('LD_LIBRARY_PATH', '')}",
)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rapid_videocr.export import OutputFormat  # noqa: E402
from rapid_videocr.main import RapidVideOCR, RapidVideOCRInput  # noqa: E402

from rapid_videocr_rocm import build_ocr_params  # noqa: E402

def _runtime_dir():
    """Per-user tmpfs dir (0700) for the socket + pid. Auto-cleaned on boot."""
    base = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    if not base or not Path(base).is_dir():
        base = "/tmp"
    return Path(base) / "rapid_videocr_daemon"


DAEMON_SOCK = Path(
    os.environ.get("RAPID_VIDEOCR_DAEMON_SOCK", _runtime_dir() / "daemon.sock")
).resolve()
DAEMON_PID = Path(
    os.environ.get("RAPID_VIDEOCR_DAEMON_PID", _runtime_dir() / "daemon.pid")
).resolve()

MAX_REQ_BYTES = 1 << 20  # 1 MB cap per request
MAX_JOBS = max(1, int(os.environ.get("DAEMON_MAX_JOBS", "2")))

_QUIT = False

_WORKER_TL = threading.local()  # one RapidVideOCR instance per worker thread
OCR_POOL = None  # ThreadPoolExecutor, max_workers == MAX_JOBS
JOB_SLOTS = None  # BoundedSemaphore(MAX_JOBS) — admits one job per worker


def _init_worker():
    ocr_params = build_ocr_params()
    _WORKER_TL.extractor = RapidVideOCR(
        RapidVideOCRInput(
            is_batch_rec=False,
            batch_size=10,
            out_format=OutputFormat.SRT.value,
            ocr_params=ocr_params,
        )
    )


def _worker_extractor():
    return _WORKER_TL.extractor


def _set_quit(_signum, _frame):
    global _QUIT
    _QUIT = True


def _pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError, PermissionError):
        return False


def _read_request(conn):
    buf = b""
    conn.settimeout(60)
    while True:
        try:
            chunk = conn.recv(65536)
        except socket.timeout:
            return None
        except OSError:
            return None
        if not chunk:
            break
        buf += chunk
        if b"\n" in buf or len(buf) > MAX_REQ_BYTES:
            break
    if not buf:
        return None
    try:
        return json.loads(buf.split(b"\n", 1)[0].decode())
    except (ValueError, UnicodeDecodeError):
        return {"cmd": "__bad__"}


def _send_response(conn, resp):
    try:
        conn.sendall((json.dumps(resp) + "\n").encode())
    except OSError:
        pass
    finally:
        try:
            conn.close()
        except OSError:
            pass


def _submit_job(conn, req):
    """Run one OCR job on a worker. The worker replies and frees the slot."""

    def run_job():
        try:
            img_dir = Path(req["img_dir"]).resolve()
            save_dir = Path(req["save_dir"]).resolve()
            save_name = str(req.get("file_name", "result"))
            out_format = str(req.get("out_format", OutputFormat.SRT.value))
            t0 = time.time()
            _worker_extractor()(img_dir, save_dir, save_name)
            srt_path = save_dir / f"{save_name}.srt"
            print(
                f"[daemon] OCR done {save_name} ({time.time() - t0:.1f}s)"
            )
            _send_response(
                conn,
                {
                    "ok": True,
                    "srt": str(srt_path) if srt_path.exists() else None,
                    "format": out_format,
                },
            )
            return
        except Exception as e:  # noqa: BLE001 - report any failure to the client
            _send_response(conn, {"ok": False, "error": f"{type(e).__name__}: {e}"})
            return
        finally:
            JOB_SLOTS.release()

    try:
        OCR_POOL.submit(run_job)
    except RuntimeError:  # executor shut down mid-request
        JOB_SLOTS.release()
        _send_response(conn, {"ok": False, "error": "daemon shutting down"})


def _handle(conn, device_name):
    req = _read_request(conn)
    if req is None:
        return
    cmd = req.get("cmd")

    if cmd == "ping":
        _send_response(
            conn, {"ok": True, "engine": "torch/rocm", "device": device_name}
        )
        return

    if cmd == "__bad__":
        _send_response(conn, {"ok": False, "error": "malformed request"})
        return

    if cmd == "shutdown":
        _send_response(conn, {"ok": True})
        global _QUIT
        _QUIT = True
        return

    if cmd == "ocr":
        if not JOB_SLOTS.acquire(blocking=False):
            print("[daemon] busy — all workers occupied, rejecting request")
            _send_response(
                conn,
                {
                    "ok": False,
                    "code": "busy",
                    "error": "daemon busy - retry later",
                    "retry_after": 5,
                },
            )
            return
        _submit_job(conn, req)
        return

    _send_response(conn, {"ok": False, "error": f"unknown cmd: {cmd}"})


def main():
    global _QUIT, OCR_POOL, JOB_SLOTS

    if DAEMON_SOCK.exists():
        try:
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            probe.settimeout(0.5)
            probe.connect(str(DAEMON_SOCK))
            probe.sendall(b'{"cmd": "ping"}\n')
            probe.close()
            print(f"[daemon] already running ({DAEMON_SOCK}), exiting")
            return 0
        except OSError:
            pass
        finally:
            try:
                probe.close()
            except OSError:
                pass
        DAEMON_SOCK.unlink()

    import torch

    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0) or "cuda:0"

    JOB_SLOTS = threading.BoundedSemaphore(MAX_JOBS)
    OCR_POOL = ThreadPoolExecutor(
        max_workers=MAX_JOBS, initializer=_init_worker, thread_name_prefix="ocr"
    )
    conn_pool = ThreadPoolExecutor(
        max_workers=min(64, MAX_JOBS * 8), thread_name_prefix="conn"
    )

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    DAEMON_SOCK.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    server.bind(str(DAEMON_SOCK))
    server.listen(4)
    server.settimeout(0.5)

    try:
        DAEMON_PID.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        DAEMON_PID.write_text(str(os.getpid()))
    except OSError as e:
        print(f"[daemon] cannot write {DAEMON_PID}: {e}")

    signal.signal(signal.SIGTERM, _set_quit)
    signal.signal(signal.SIGINT, _set_quit)

    print(
        f"[daemon] ROCm OCR daemon ready on {DAEMON_SOCK} "
        f"(device: {device_name}, max concurrent jobs: {MAX_JOBS})"
    )

    while not _QUIT:
        try:
            conn, _ = server.accept()
        except socket.timeout:
            continue
        except OSError:
            if _QUIT:
                break
            continue
        try:
            conn_pool.submit(_handle, conn, device_name)
        except RuntimeError:
            try:
                conn.close()
            except OSError:
                pass

    conn_pool.shutdown(wait=True)
    OCR_POOL.shutdown(wait=True)
    server.close()
    try:
        DAEMON_SOCK.unlink()
    except OSError:
        pass
    print("[daemon] stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())