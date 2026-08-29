#!/usr/bin/env python3
"""Warm translate daemon for SRT translation jobs (HY-MT).

Translates whole SRT files over a Unix socket so per-job translate_srt.py
subprocesses never pay model load cost.  One NPU worker slot (HY-MT on the
AMD NPU via npu-engine.service) plus TRANSLATE_MAX_JOBS - 1 GPU worker
slots (HY-MT on the iGPU via ROCm).  Jobs are dispatched NPU-first: a
single job always runs on the NPU; the GPU slot(s) are used only when the
NPU is already busy.  The NPU slot falls back to the GPU if the engine
cannot be reached.  Each worker owns its own model/engine handle (HY-MT
keeps module-global state, so instances cannot be shared).

A global thermal gate pauses all workers while the GPU is above
TRANSLATE_TEMP_LIMIT and resumes them once it cools to
TRANSLATE_COOLDOWN_TARGET.

While idle (no jobs running), the daemon shuts down once the card
reaches TRANSLATE_IDLE_TEMPERATURE (default 76°C): an idle resident model
still keeps the box warm (ROCm's HSA exception-monitor thread busy-polls
once any GPU work has happened) and exiting the process is the only way
to stop it.  This only applies after a GPU model has been loaded — a fresh
daemon holds no GPU state.  A prewarm grace (TRANSLATE_IDLE_GRACE_SECS,
default 600 s) starts when the model is prewarmed (ensure_model) and
shields the shutdown so an enqueued job has time to start; above
TRANSLATE_IDLE_CRITICAL_TEMP (default 98°C) the grace is ignored and
the daemon shuts down immediately.  Independently of temperature, an
idle timeout (TRANSLATE_IDLE_TIMEOUT_SECS, default 300 s) starts when a
translate job finishes — if the daemon is still idle when it expires it
exits anyway, so a cool-but-unused daemon does not hold the GPU forever
(set to 0 to disable).  Clients restart the daemon on demand
(translate_srt.py auto mode does this).

A GPU hang watchdog (shared HangGuard in gpu_thermal.py — the same guard
the gen_audio daemon uses) covers a daemon that never responds at all:
when a GPU translate/ensure_model job runs past
TRANSLATE_HANG_TIMEOUT_SECS (default 6 h) it is stuck (e.g. hung GPU
kernel) and the daemon shuts itself down — SIGTERM first, then SIGKILL
after TRANSLATE_HANG_GRACE_SECS (default 90 s) if the stuck worker
thread prevents a clean interpreter exit.  Clients restart the daemon on
demand, so the next attempt gets a fresh process.  Only GPU jobs trigger
this: the NPU slot talks to the separate npu-engine.service daemon
(which keeps its own timeouts) and is left untouched.

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
from gpu_thermal import HangGuard, ThermalGate, log_system_temp  # noqa: E402

# Import at module level (main thread) — translate_srt registers signal
# handlers and imports hy_mt/transformers at import time.
from translate_srt import translate_srt_file  # noqa: E402

DAEMON_SOCK = Path(TRANSLATE_DAEMON_SOCK).resolve()
DAEMON_PID = Path(TRANSLATE_DAEMON_PID).resolve()

MAX_REQ_BYTES = 1 << 20  # 1 MB cap per request
MAX_JOBS = max(1, int(os.environ.get("TRANSLATE_MAX_JOBS", "2")))

# Hang watchdog (shared HangGuard, see gpu_thermal): a GPU
# translate/ensure_model job running longer than this is treated as stuck
# (e.g. hung GPU kernel) and the daemon kills itself — SIGTERM, then
# SIGKILL after the grace period if the stuck worker thread still prevents
# a clean interpreter exit.  GPU only — NPU jobs go through the separate
# npu-engine.service and are never armed here.
HANG = HangGuard(
    hang_timeout_secs=float(os.environ.get("TRANSLATE_HANG_TIMEOUT_SECS", str(6 * 3600))),
    grace_secs=float(os.environ.get("TRANSLATE_HANG_GRACE_SECS", "90")),
    check_secs=float(os.environ.get("TRANSLATE_HANG_CHECK_SECS", "10")),
)

# Worker pools: one NPU worker (HY-MT on the NPU engine) plus
# TRANSLATE_MAX_JOBS - 1 GPU workers.  Jobs are dispatched NPU-first
# (single jobs always go to the NPU; the GPU is only used when the NPU
# is busy).  The NPU worker falls back to its own GPU model if the
# engine cannot be reached.
NPU_SLOT = os.environ.get("TRANSLATE_NPU_SLOT", "1") == "1"
N_GPU_WORKERS = MAX_JOBS - 1 if NPU_SLOT else MAX_JOBS

# Thermal gate + idle release (global — all workers pause together).
# Shared logic lives in gpu_thermal.ThermalGate; see module docstring.
GATE = ThermalGate(
    temp_limit=float(os.environ.get("TRANSLATE_TEMP_LIMIT", "90")),
    cooldown_target=float(os.environ.get("TRANSLATE_COOLDOWN_TARGET", "70")),
    poll_secs=float(os.environ.get("TRANSLATE_POLL_SECS", "10")),
    idle_temperature=float(os.environ.get("TRANSLATE_IDLE_TEMPERATURE", "76")),
    idle_critical_temp=float(os.environ.get("TRANSLATE_IDLE_CRITICAL_TEMP", "98")),
    idle_grace_secs=float(os.environ.get("TRANSLATE_IDLE_GRACE_SECS", "600")),
    idle_timeout_secs=float(os.environ.get("TRANSLATE_IDLE_TIMEOUT_SECS", "300")),
)
_THERMAL_BLOCKED = GATE.blocked  # workers wait on this while the gate is closed

_QUIT = False
_ACTIVE = 0
_ACTIVE_LOCK = threading.Lock()
_MODEL_LOADED = threading.Event()  # set once any worker has loaded the model


def _request_quit():
    global _QUIT
    _QUIT = True

_WORKER_TL = threading.local()  # one HY-MT model per worker thread
NPU_POOL = None  # ThreadPoolExecutor(1) — NPU worker (when NPU_SLOT)
GPU_POOL = None  # ThreadPoolExecutor(N_GPU_WORKERS) — GPU workers
NPU_SLOT_SEM = None  # BoundedSemaphore(1) — NPU worker busy flag
GPU_SLOT_SEM = None  # BoundedSemaphore(N_GPU_WORKERS) — GPU workers busy flag


# ── Per-thread model management ─────────────────────────────


def _init_worker(npu=False):
    _WORKER_TL.model = None
    _WORKER_TL.tokenizer = None
    _WORKER_TL.npu = npu
    _WORKER_TL.npu_client = None
    _WORKER_TL.npu_down = False


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
    in-process client (translate_srt._translate_with_retry).  NPU workers
    route to the NPU engine (falling back to the in-process GPU model);
    GPU workers use the in-process GPU model."""

    def __init__(self, npu=False):
        self._npu = npu

    def translate(self, text, target_language):
        from translate_srt import _translate_with_retry

        if self._npu and not getattr(_WORKER_TL, "npu_down", True):
            if _WORKER_TL.npu_client is None:
                from npu_client import NpuEngineClient

                _WORKER_TL.npu_client = NpuEngineClient()
            if _WORKER_TL.npu_client.ensure_engine():
                return _WORKER_TL.npu_client.translate(text, target_language)
            _WORKER_TL.npu_down = True
        model, tokenizer = _thread_model()
        return _translate_with_retry(model, tokenizer, text, target_language)


# ── Thermal gate + idle watchdog ────────────────────────────


def _monitor():
    global _QUIT
    while not _QUIT:
        if GATE.poll(lambda: _ACTIVE, _MODEL_LOADED.is_set):
            _request_quit()
            return


def _hang_watchdog():
    """GPU hang self-kill (shared HangGuard, see gpu_thermal — the same
    guard the gen_audio daemon uses).

    A GPU worker stuck in a ROCm call never returns; when such a job
    exceeds the hang timeout the guard requests a clean exit (SIGTERM)
    and force-kills after the grace period if the stuck worker thread
    prevents interpreter shutdown.  Only GPU jobs are armed, so NPU jobs
    never trigger this.  Clients restart the daemon on demand."""
    HANG.watch(request_quit=_request_quit, is_quit=lambda: _QUIT)


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


def _release_slot(npu):
    sem = NPU_SLOT_SEM if npu else GPU_SLOT_SEM
    if sem is not None:
        sem.release()


def _run_ensure_model(conn, req, npu):
    """Load the model on a worker thread (slot-protected)."""
    global _ACTIVE
    with _ACTIVE_LOCK:
        _ACTIVE += 1
        if not npu:
            HANG.arm()
    try:
        if npu:
            from npu_client import NpuEngineClient, NpuEngineUnavailable

            if _WORKER_TL.npu_client is None:
                _WORKER_TL.npu_client = NpuEngineClient()
            try:
                resp = _WORKER_TL.npu_client.ensure_model()
                GATE.arm_grace()
                _send_response(conn, {"ok": True, "device": resp.get("device", "npu")})
                return
            except NpuEngineUnavailable as e:
                print(f"[daemon] NPU ensure_model failed ({e}) — using GPU")
                _WORKER_TL.npu_down = True
        model, _ = _thread_model()
        GATE.arm_grace()
        device = "cpu"
        with contextlib.suppress(StopIteration, AttributeError):
            device = str(next(model.parameters()).device)
        _send_response(conn, {"ok": True, "device": device})
    except Exception as e:  # noqa: BLE001
        _send_response(conn, {"ok": False, "error": f"{type(e).__name__}: {e}"})
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE -= 1
            if not npu:
                HANG.disarm()
        _release_slot(npu)


def _run_translate(conn, req, npu):
    global _ACTIVE
    try:
        with _ACTIVE_LOCK:
            _ACTIVE += 1
            if not npu:
                HANG.arm()
        GATE.disarm_grace()  # prewarm is consumed — idle-hot shutdown may fire once idle again
        GATE.disarm_idle_timeout()  # a job is here — idle countdown re-arms when it finishes
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
            backend=_ThreadBackend(npu=npu),
        )
        print(f"[daemon] translate done ({lang}, {'npu' if npu else 'gpu'}, {time.time() - t0:.1f}s)")
        _send_response(conn, {"ok": True, "output_path": str(output_path)})
    except Exception as e:  # noqa: BLE001 - report any failure to the client
        _send_response(conn, {"ok": False, "error": f"{type(e).__name__}: {e}"})
    finally:
        GATE.arm_idle_timeout()  # idle again — exit after the idle countdown even while cool
        with _ACTIVE_LOCK:
            _ACTIVE -= 1
            if not npu:
                HANG.disarm()
        _release_slot(npu)


def _log_future_error(fut):
    try:
        fut.result()
    except Exception as e:  # noqa: BLE001 - job functions handle their own errors
        print(f"[daemon] worker job crashed: {type(e).__name__}: {e}")


def _dispatch(conn, req, cmd):
    """NPU-first job routing: NPU slot if free, else a GPU slot, else busy.
    ensure_model never falls through to the GPU worker while the NPU slot
    exists — prewarming the GPU model for no reason just triggers the HSA
    busy-poll.  It returns busy instead; the client can retry later."""
    npu = None
    if NPU_SLOT_SEM is not None and NPU_SLOT_SEM.acquire(blocking=False):
        npu = True
    elif (
        (cmd != "ensure_model" or NPU_SLOT_SEM is None)
        and GPU_SLOT_SEM is not None
        and GPU_SLOT_SEM.acquire(blocking=False)
    ):
        npu = False
    else:
        print(f"[daemon] {cmd}: busy — all worker slots occupied")
        _send_response(
            conn, {"ok": False, "code": "busy", "error": "daemon busy - retry later", "retry_after": 5}
        )
        return

    pool = NPU_POOL if npu else GPU_POOL
    fn = _run_translate if cmd == "translate" else _run_ensure_model
    try:
        fut = pool.submit(fn, conn, req, npu)
        fut.add_done_callback(_log_future_error)
        print(f"[daemon] {cmd}: run_job submitted to {'npu' if npu else 'gpu'} worker")
    except Exception as e:  # executor shut down mid-request
        print(f"[daemon] {cmd}: submit failed ({type(e).__name__}: {e})")
        _release_slot(npu)
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
        _dispatch(conn, req, "ensure_model")
        return

    if cmd == "translate":
        _dispatch(conn, req, "translate")
        return

    _send_response(conn, {"ok": False, "error": f"unknown cmd: {cmd}"})


def _set_quit(_signum, _frame):
    _request_quit()


def main():
    global _QUIT, NPU_POOL, GPU_POOL, NPU_SLOT_SEM, GPU_SLOT_SEM

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
    if NPU_SLOT:
        device_name = f"npu + {device_name} (npu-first)"

    if NPU_SLOT:
        NPU_SLOT_SEM = threading.BoundedSemaphore(1)
        NPU_POOL = ThreadPoolExecutor(
            max_workers=1,
            initializer=_init_worker,
            initargs=(True,),
            thread_name_prefix="translate-npu",
        )
    if N_GPU_WORKERS > 0:
        GPU_SLOT_SEM = threading.BoundedSemaphore(N_GPU_WORKERS)
        GPU_POOL = ThreadPoolExecutor(
            max_workers=N_GPU_WORKERS,
            initializer=_init_worker,
            initargs=(False,),
            thread_name_prefix="translate-gpu",
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
    threading.Thread(target=_hang_watchdog, daemon=True, name="hang-watchdog").start()

    print(
        f"[daemon] translate daemon ready on {DAEMON_SOCK} "
        f"(device: {device_name}, max concurrent jobs: {MAX_JOBS}, "
        f"npu slot: {'on' if NPU_SLOT else 'off'}, gpu workers: {N_GPU_WORKERS}, "
        f"temp limit: {GATE.temp_limit:.0f}°C, idle temp: {GATE.idle_temperature:.0f}°C, "
        f"idle grace: {GATE.idle_grace_secs:.0f}s, critical: {GATE.idle_critical_temp:.0f}°C, "
        f"idle timeout: {GATE.idle_timeout_secs:.0f}s, "
        f"gpu hang kill: {HANG.hang_timeout_secs / 3600:.1f}h + {HANG.grace_secs:.0f}s grace)"
    )
    log_system_temp("startup")

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
    if NPU_POOL is not None:
        NPU_POOL.shutdown(wait=True)
    if GPU_POOL is not None:
        GPU_POOL.shutdown(wait=True)
    server.close()
    try:
        DAEMON_SOCK.unlink()
    except OSError:
        pass
    try:
        DAEMON_PID.unlink()
    except OSError:
        pass
    log_system_temp("shutdown")
    print("[daemon] stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
