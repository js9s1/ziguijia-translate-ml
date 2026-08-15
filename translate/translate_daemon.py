#!/usr/bin/env python3
"""Warm translate daemon for SRT translation jobs (HY-MT).

Keeps the HY-MT 1.8B model resident on GPU and translates whole SRT files
over a Unix socket, so per-job translate_srt.py subprocesses no longer pay
model load cost.  Up to TRANSLATE_MAX_JOBS (default 2) files translate
concurrently, one per worker thread; each thread owns its own model
instance (HY-MT keeps module-global state, so instances cannot be shared).

A global thermal gate pauses all workers while the GPU is above
TRANSLATE_TEMP_LIMIT and resumes them once it cools to
TRANSLATE_COOLDOWN_TARGET.

While idle (no jobs running), the daemon releases the GPU once the card
reaches TRANSLATE_IDLE_TEMPERATURE (default 76°C): an idle resident model
still keeps the box warm (ROCm's HSA exception-monitor thread busy-polls
once any GPU work has happened).  This only applies after a model has
been loaded — a fresh daemon holds no GPU state.  A prewarm grace
(TRANSLATE_IDLE_GRACE_SECS, default 600 s) starts when the model is
prewarmed (ensure_model) and shields the release so an enqueued job has
time to start; above TRANSLATE_IDLE_CRITICAL_TEMP (default 98°C) the
grace is ignored and the GPU is released immediately.  Clients restart
the daemon on demand (translate_srt.py auto mode does this).

Runtime files: socket + pid live in $XDG_RUNTIME_DIR/translate_daemon/
(tmpfs, 0700, auto-cleaned on reboot).

Protocol (newline-delimited JSON per request, connection-per-request):
  request  {"cmd": "ping"}
  response {"ok": true, "engine": "hy-mt", "device": "..."}

  request  {"cmd": "ensure_model"}
  response {"ok": true, "device": "cuda"}
    or     {"ok": false, "code": "busy", "retry_after": 5}
           (all worker slots busy — model will load on first translate)

  request  {"cmd": "translate", "input_path": "...", "output_path": "...",
            "language": "English", "intro": null, "outro": null}
  response {"ok": true, "output_path": "...", "segments": 42}
    or     {"ok": false, "error": "..."}
    or     {"ok": false, "code": "busy", "retry_after": 5}
           (all worker slots busy — client should retry later)

  request  {"cmd": "shutdown"}
  response {"ok": true}

SIGTERM or SIGINT stops the daemon cleanly after in-flight jobs finish.
"""
import contextlib
import gc
import json
import os
import signal
import socket
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if sys.stdout is not None:
    sys.stdout.reconfigure(line_buffering=True)
if sys.stderr is not None:
    sys.stderr.reconfigure(line_buffering=True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "chatterbox-server"))
sys.path.insert(0, os.path.expanduser("~/src/HY-MT"))

from rocm_env import setup as _rocm_setup  # noqa: E402

_rocm_setup()  # before transformers/torch load ROCm libs

from config import TRANSLATE_DAEMON_PID, TRANSLATE_DAEMON_SOCK  # noqa: E402
from gpu_thermal import ThermalGate  # noqa: E402

# Import at module level (main thread) — translate_srt registers signal
# handlers and imports hy_mt/transformers at import time.
from translate_srt import translate_srt_file  # noqa: E402

DAEMON_SOCK = Path(TRANSLATE_DAEMON_SOCK).resolve()
DAEMON_PID = Path(TRANSLATE_DAEMON_PID).resolve()

MAX_REQ_BYTES = 1 << 20  # 1 MB cap per request
MAX_JOBS = max(1, int(os.environ.get("TRANSLATE_MAX_JOBS", "2")))

# Thermal gate + idle release (global — all workers pause together).
# Shared logic lives in gpu_thermal.ThermalGate; see module docstring.
GATE = ThermalGate(
    temp_limit=float(os.environ.get("TRANSLATE_TEMP_LIMIT", "90")),
    cooldown_target=float(os.environ.get("TRANSLATE_COOLDOWN_TARGET", "70")),
    poll_secs=float(os.environ.get("TRANSLATE_POLL_SECS", "10")),
    idle_temperature=float(os.environ.get("TRANSLATE_IDLE_TEMPERATURE", "76")),
    idle_critical_temp=float(os.environ.get("TRANSLATE_IDLE_CRITICAL_TEMP", "98")),
    idle_grace_secs=float(os.environ.get("TRANSLATE_IDLE_GRACE_SECS", "600")),
)
_THERMAL_BLOCKED = GATE.blocked  # workers wait on this while the gate is closed

_QUIT = False
_ACTIVE = 0
_ACTIVE_LOCK = threading.Lock()
_MODEL_LOADED = threading.Event()  # set once any worker has loaded the model

_WORKER_TL = threading.local()  # one HY-MT model per worker thread
TTS_POOL = None  # ThreadPoolExecutor, max_workers == MAX_JOBS
JOB_SLOTS = None  # BoundedSemaphore(MAX_JOBS) — admits one job per worker


# ── Per-thread model management ─────────────────────────────


def _init_worker():
    _WORKER_TL.model = None
    _WORKER_TL.tokenizer = None


def _thread_model():
    """Load (or keep) the HY-MT model on this worker thread."""
    if getattr(_WORKER_TL, "model", None) is not None:
        return _WORKER_TL.model, _WORKER_TL.tokenizer

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import hy_mt

    warnings.filterwarnings("ignore")
    model_path = hy_mt.MODEL_PATH
    _WORKER_TL.tokenizer = AutoTokenizer.from_pretrained(model_path)

    device = "cpu"
    if torch.cuda.is_available():
        try:
            t = torch.zeros(1, device="cuda")
            del t
            device = "cuda"
        except Exception:
            pass

    print(f"[daemon] loading HY-MT model to {device}...")
    if device == "cuda":
        try:
            try:
                import accelerate  # noqa: F401

                _WORKER_TL.model = AutoModelForCausalLM.from_pretrained(
                    model_path, device_map="cuda", torch_dtype=torch.float16
                )
            except ImportError:
                _WORKER_TL.model = AutoModelForCausalLM.from_pretrained(
                    model_path, torch_dtype=torch.float16
                ).to("cuda")
        except Exception as exc:
            print(f"[daemon] CUDA load failed ({exc}) — falling back to CPU")
            _WORKER_TL.model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        _WORKER_TL.model = AutoModelForCausalLM.from_pretrained(model_path)
    _MODEL_LOADED.set()
    print(f"[daemon] HY-MT model ready ({device})")
    return _WORKER_TL.model, _WORKER_TL.tokenizer


class _ThreadBackend:
    """Per-thread translation backend with the same retry logic as the
    in-process client (translate_srt._translate_with_retry)."""

    def translate(self, text, target_language):
        from translate_srt import _translate_with_retry

        model, tokenizer = _thread_model()
        return _translate_with_retry(model, tokenizer, text, target_language)


# ── Thermal gate + idle watchdog ────────────────────────────


def _monitor():
    global _QUIT
    while not _QUIT:
        if GATE.poll(lambda: _ACTIVE, _MODEL_LOADED.is_set):
            _QUIT = True
            return


# ── Socket protocol ─────────────────────────────────────────


def _read_request(conn):
    buf = b""
    conn.settimeout(60)
    while True:
        try:
            chunk = conn.recv(65536)
        except socket.timeout:
            print("[daemon] request read timed out (60s) — closing connection")
            try:
                conn.close()
            except OSError:
                pass
            return None
        except OSError:
            return None
        if not chunk:
            break
        buf += chunk
        if b"\n" in buf or len(buf) > MAX_REQ_BYTES:
            break
    if not buf:
        try:
            conn.close()
        except OSError:
            pass
        return None
    try:
        req = json.loads(buf.split(b"\n", 1)[0].decode())
        print(f"[daemon] request: {req.get('cmd')}")
        return req
    except (ValueError, UnicodeDecodeError):
        try:
            conn.close()
        except OSError:
            pass
        return {"cmd": "__bad__"}


def _send_response(conn, resp):
    try:
        conn.sendall((json.dumps(resp) + "\n").encode())
        print(f"[daemon] response sent: {str(resp)[:120]}")
    except OSError as e:
        print(f"[daemon] response send failed: {e}")
    finally:
        try:
            conn.close()
        except OSError:
            pass


def _submit_ensure_model(conn, req):
    """Load the HY-MT model on a worker thread (slot-protected)."""

    def run_job():
        try:
            model, _ = _thread_model()
            GATE.arm_grace()
            device = "cpu"
            with contextlib.suppress(StopIteration, AttributeError):
                device = str(next(model.parameters()).device)
            _send_response(conn, {"ok": True, "device": device})
        except Exception as e:  # noqa: BLE001
            _send_response(conn, {"ok": False, "error": f"{type(e).__name__}: {e}"})
        finally:
            JOB_SLOTS.release()

    try:
        TTS_POOL.submit(run_job)
        print("[daemon] ensure_model: run_job submitted to worker pool")
    except Exception as e:  # executor shut down mid-request
        print(f"[daemon] ensure_model: submit failed ({type(e).__name__}: {e})")
        with contextlib.suppress(ValueError):
            JOB_SLOTS.release()
        _send_response(conn, {"ok": False, "error": "daemon shutting down"})


def _submit_translate(conn, req):
    def run_job():
        global _ACTIVE
        try:
            with _ACTIVE_LOCK:
                _ACTIVE += 1
            input_path = Path(str(req["input_path"])).resolve()
            output_path = Path(str(req["output_path"])).resolve()
            lang = str(req.get("language", "English"))
            intro = req.get("intro") or None
            outro = req.get("outro") or None

            while _THERMAL_BLOCKED.is_set():
                if _QUIT:
                    raise RuntimeError("daemon shutting down")
                time.sleep(5)

            t0 = time.time()
            translate_srt_file(
                str(input_path),
                str(output_path),
                lang,
                intro_marker=intro,
                outro_marker=outro,
                backend=_ThreadBackend(),
            )
            print(f"[daemon] translate done ({lang}, {time.time() - t0:.1f}s)")
            _send_response(conn, {"ok": True, "output_path": str(output_path)})
        except Exception as e:  # noqa: BLE001 - report any failure to the client
            _send_response(conn, {"ok": False, "error": f"{type(e).__name__}: {e}"})
        finally:
            with _ACTIVE_LOCK:
                _ACTIVE -= 1
            JOB_SLOTS.release()

    try:
        TTS_POOL.submit(run_job)
        print("[daemon] translate: run_job submitted to worker pool")
    except Exception as e:  # executor shut down mid-request
        print(f"[daemon] translate: submit failed ({type(e).__name__}: {e})")
        try:
            JOB_SLOTS.release()
        except ValueError:
            pass
        _send_response(conn, {"ok": False, "error": "daemon shutting down"})


def _handle(conn, device_name):
    req = _read_request(conn)
    if req is None:
        return
    cmd = req.get("cmd")

    if cmd == "ping":
        _send_response(
            conn,
            {
                "ok": True,
                "engine": "hy-mt",
                "device": device_name,
                "max_jobs": MAX_JOBS,
                "active": _ACTIVE,
            },
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

    if cmd == "ensure_model":
        if not JOB_SLOTS.acquire(blocking=False):
            print("[daemon] ensure_model: busy — all worker slots occupied")
            _send_response(
                conn, {"ok": False, "code": "busy", "error": "daemon busy - retry later", "retry_after": 5}
            )
            return
        _submit_ensure_model(conn, req)
        return

    if cmd == "translate":
        if not JOB_SLOTS.acquire(blocking=False):
            print("[daemon] translate: busy — all worker slots occupied")
            _send_response(
                conn, {"ok": False, "code": "busy", "error": "daemon busy - retry later", "retry_after": 5}
            )
            return
        _submit_translate(conn, req)
        return

    _send_response(conn, {"ok": False, "error": f"unknown cmd: {cmd}"})


def _set_quit(_signum, _frame):
    global _QUIT
    _QUIT = True


def main():
    global _QUIT, TTS_POOL, JOB_SLOTS

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
    TTS_POOL = ThreadPoolExecutor(
        max_workers=MAX_JOBS, initializer=_init_worker, thread_name_prefix="translate"
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

    threading.Thread(target=_monitor, daemon=True, name="monitor").start()

    print(
        f"[daemon] translate daemon ready on {DAEMON_SOCK} "
        f"(device: {device_name}, max concurrent jobs: {MAX_JOBS}, "
        f"temp limit: {GATE.temp_limit:.0f}°C, idle temp: {GATE.idle_temperature:.0f}°C, "
        f"idle grace: {GATE.idle_grace_secs:.0f}s, critical: {GATE.idle_critical_temp:.0f}°C)"
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
    TTS_POOL.shutdown(wait=True)
    server.close()
    try:
        DAEMON_SOCK.unlink()
    except OSError:
        pass
    try:
        DAEMON_PID.unlink()
    except OSError:
        pass
    print("[daemon] stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
