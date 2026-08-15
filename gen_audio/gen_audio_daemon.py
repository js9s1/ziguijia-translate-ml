#!/usr/bin/env python3
"""Warm TTS daemon for gen_audio jobs.

Keeps the Chatterbox TTS model(s) resident on GPU and serves text-to-wav
requests over a Unix socket, so per-job gen_audio.py subprocesses no longer
pay model load cost.  Up to GEN_AUDIO_MAX_JOBS (default 2) requests run
concurrently, one per worker thread; each thread owns its own model
instance (Chatterbox keeps mutable per-call state, so instances cannot be
shared).  A thread lazily loads the model for the requested language and
swaps when the next request needs the other language.

A global thermal gate pauses all workers while the GPU is above
GEN_AUDIO_TEMP_LIMIT and resumes them once it cools to
GEN_AUDIO_COOLDOWN_TARGET.

The daemon exits after GEN_AUDIO_IDLE_TIMEOUT seconds without jobs
(default 300; 0 disables).  This avoids burning a core while idle:
ROCm's HSA exception-monitor thread busy-polls once any GPU work has
happened, which keeps the box warm.  Clients restart the daemon on
demand (gen_audio.py auto mode does this).

Runtime files: socket + pid live in $XDG_RUNTIME_DIR/gen_audio_daemon/
(tmpfs, 0700, auto-cleaned on reboot).

Protocol (newline-delimited JSON per request, connection-per-request):
  request  {"cmd": "ping"}
  response {"ok": true, "engine": "chatterbox", "device": "..."}

  request  {"cmd": "ensure_model", "language": "en"}
  response {"ok": true, "sr": 24000, "device": "cuda"}

  request  {"cmd": "tts", "text": "...", "language": "en",
            "prompt_file": "/path/or/null", "temperature": 0.6,
            "cfg_weight": 0.5, "exaggeration": 0.5,
            "output_path": "/path/segment.wav"}
  response {"ok": true, "duration": 1.23, "sr": 24000}
    or     {"ok": false, "error": "..."}
    or     {"ok": false, "code": "busy", "retry_after": 5}
           (all worker slots busy — client should retry later;
            the request was NOT queued or run)

  request  {"cmd": "shutdown"}
  response {"ok": true}

SIGTERM or SIGINT stops the daemon cleanly after in-flight jobs finish.
"""
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
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "chatterbox-server"))

from config import AUDIO_PROMPT_PATH, GEN_AUDIO_DAEMON_PID, GEN_AUDIO_DAEMON_SOCK  # noqa: E402
from gpu_thermal import get_gpu_temp  # noqa: E402

DAEMON_SOCK = Path(GEN_AUDIO_DAEMON_SOCK).resolve()
DAEMON_PID = Path(GEN_AUDIO_DAEMON_PID).resolve()

MAX_REQ_BYTES = 1 << 20  # 1 MB cap per request
MAX_JOBS = max(1, int(os.environ.get("GEN_AUDIO_MAX_JOBS", "2")))

# Idle shutdown: the ROCm HSA exception-monitor thread busy-polls (~1 core,
# heat) once any GPU work has happened, so the daemon exits after this many
# seconds without jobs.  0 = never exit.  The client restarts it on demand.
IDLE_TIMEOUT = float(os.environ.get("GEN_AUDIO_IDLE_TIMEOUT", "300"))

# Thermal gate (global — all workers pause together)
GPU_TEMP_LIMIT = float(os.environ.get("GEN_AUDIO_TEMP_LIMIT", "90"))  # °C
GPU_COOLDOWN_TARGET = float(os.environ.get("GEN_AUDIO_COOLDOWN_TARGET", "70"))  # °C
GPU_POLL_SECS = float(os.environ.get("GEN_AUDIO_POLL_SECS", "10"))

_QUIT = False
_THERMAL_BLOCKED = threading.Event()
_ACTIVE = 0
_ACTIVE_LOCK = threading.Lock()
_LAST_FUTURES = []  # debugging: recently submitted tts futures
_LAST_ACTIVE_TS = time.monotonic()  # last job completion — for idle shutdown

_WORKER_TL = threading.local()  # one TTS model per worker thread
TTS_POOL = None  # ThreadPoolExecutor, max_workers == MAX_JOBS
JOB_SLOTS = None  # BoundedSemaphore(MAX_JOBS) — admits one job per worker

# ── Generation / load coordination ─────────────────────────
# Loading a second model instance while another thread is running GPU
# inference has produced permanently NaN ("non-finite audio") instances
# on ROCm (job 06D55E25: one worker always failed).  Generations may run
# concurrently with each other, but a model load waits for all in-flight
# generations to finish and blocks new ones from starting.
_GEN_COND = threading.Condition()
_GENS_ACTIVE = 0
_LOAD_ACTIVE = False
_LOAD_MUTEX = threading.Lock()  # serialize loads against each other


class _GenerationGuard:
    """Context manager marking a generation as in-flight (see above)."""

    def __enter__(self):
        global _GENS_ACTIVE
        with _GEN_COND:
            while _LOAD_ACTIVE:
                _GEN_COND.wait()
            _GENS_ACTIVE += 1

    def __exit__(self, *_exc):
        global _GENS_ACTIVE
        with _GEN_COND:
            _GENS_ACTIVE -= 1
            _GEN_COND.notify_all()


# ── Per-thread model management ─────────────────────────────


def _init_worker():
    _WORKER_TL.model = None
    _WORKER_TL.kind = None


def _get_device(m) -> str:
    try:
        return str(m.device)
    except AttributeError:
        pass
    try:
        return str(next(m.parameters()).device)
    except (StopIteration, AttributeError):
        pass
    for sub in ("t3", "s3gen", "model"):
        try:
            sm = getattr(m, sub)
            return str(next(sm.parameters()).device)
        except (StopIteration, AttributeError):
            pass
    return "cpu"


def _choose_device() -> str:
    import torch

    if not torch.cuda.is_available():
        print("[daemon] CUDA not available — using CPU")
        return "cpu"
    try:
        from gpu_manage import _GPU_MIN_FREE_MEM_GiB

        free, _ = torch.cuda.mem_get_info()
        free_gb = free / (1024**3)
        if free_gb < _GPU_MIN_FREE_MEM_GiB:
            print(
                f"[daemon] GPU free memory {free_gb:.1f} GiB — too low "
                f"(need {_GPU_MIN_FREE_MEM_GiB:.1f} GiB), using CPU"
            )
            return "cpu"
    except ImportError:
        pass
    return "cuda"


def _load_multilingual():
    import torch

    warnings.filterwarnings("ignore")
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    device = _choose_device()
    print(f"[daemon] loading multilingual TTS model to {device}...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    model.prepare_conditionals(AUDIO_PROMPT_PATH)
    print(f"[daemon] multilingual TTS model ready ({device})")
    return model


def _load_indonesian():
    import torch

    warnings.filterwarnings("ignore")
    from chatterbox.tts import ChatterboxTTS

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    device = _choose_device()
    print(f"[daemon] loading Indonesian TTS model to {device}...")
    model = ChatterboxTTS.from_pretrained(device=device)
    checkpoint_path = hf_hub_download(
        repo_id="grandhigh/Chatterbox-TTS-Indonesian",
        filename="t3_cfg.safetensors",
    )
    t3_state = load_file(checkpoint_path, device="cpu")
    model.t3.load_state_dict(t3_state)
    torch.cuda.empty_cache()
    print(f"[daemon] Indonesian TTS model ready ({_get_device(model)})")
    return model


def _ensure_thread_model(lang: str, force: bool = False):
    """Load (or keep) the model for *lang* on this worker thread.

    Loads are coordinated against in-flight generations: ``_LOAD_ACTIVE``
    blocks new generations, the loader waits for active ones to finish,
    and ``_LOAD_MUTEX`` serializes loads against each other.
    """
    global _LOAD_ACTIVE
    kind = getattr(_WORKER_TL, "kind", None)
    model = getattr(_WORKER_TL, "model", None)
    if not force and kind == lang and model is not None:
        return model

    with _GEN_COND:
        _LOAD_ACTIVE = True
        _GEN_COND.notify_all()
    try:
        while True:
            with _GEN_COND:
                if _GENS_ACTIVE == 0:
                    break
                _GEN_COND.wait()

        with _LOAD_MUTEX:
            import torch

            if model is not None:
                del model
                _WORKER_TL.model = None
                _WORKER_TL.kind = None
                gc.collect()
                torch.cuda.empty_cache()

            _WORKER_TL.model = _load_indonesian() if lang == "id" else _load_multilingual()
            _WORKER_TL.kind = lang
            return _WORKER_TL.model
    finally:
        with _GEN_COND:
            _LOAD_ACTIVE = False
            _GEN_COND.notify_all()


# ── Generation helpers ──────────────────────────────────────


def _generate_indonesian(model, text, prompt_file, temperature):
    kwargs = {}
    audio_prompt = prompt_file if prompt_file else AUDIO_PROMPT_PATH
    if audio_prompt and os.path.exists(audio_prompt):
        kwargs["audio_prompt_path"] = audio_prompt
    try:
        with _GenerationGuard():
            return model.generate(text, temperature=temperature, **kwargs)
    except TypeError:
        with _GenerationGuard():
            return model.generate(text, **kwargs)


def _generate_with_retry(model, text, target_language, temperature, cfg_weight, exaggeration, attempts=3):
    for attempt in range(attempts):
        try:
            with _GenerationGuard():
                return model.generate(
                    text,
                    language_id=target_language,
                    temperature=temperature + attempt * 0.02,
                    cfg_weight=cfg_weight,
                    exaggeration=exaggeration,
                )
        except Exception as e:
            if "not finite" not in str(e):
                raise
            print(
                f"[daemon] non-finite audio (attempt {attempt + 1}/{attempts}) — "
                f"reloading this worker's model and retrying"
            )
            model = _ensure_thread_model(target_language, force=True)
    raise RuntimeError(f"model produced non-finite audio {attempts} times")


# ── Thermal gate ────────────────────────────────────────────


def _thermal_monitor():
    global _QUIT, _LAST_ACTIVE_TS
    while not _QUIT:
        temp = get_gpu_temp()
        if temp is not None:
            if temp >= GPU_TEMP_LIMIT and not _THERMAL_BLOCKED.is_set():
                print(f"[daemon] GPU temp {temp:.0f}°C ≥ limit {GPU_TEMP_LIMIT:.0f}°C — pausing all workers")
                _THERMAL_BLOCKED.set()
            elif temp <= GPU_COOLDOWN_TARGET and _THERMAL_BLOCKED.is_set():
                print(f"[daemon] GPU cooled to {temp:.0f}°C — resuming workers")
                _THERMAL_BLOCKED.clear()
        if IDLE_TIMEOUT > 0:
            idle = time.monotonic() - _LAST_ACTIVE_TS
            if idle >= IDLE_TIMEOUT and _ACTIVE == 0:
                print(f"[daemon] idle for {idle:.0f}s ≥ {IDLE_TIMEOUT:.0f}s — shutting down")
                _QUIT = True
                return
        time.sleep(GPU_POLL_SECS)


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
    """Load the requested model on a worker thread (slot-protected)."""

    def run_job():
        global _LAST_ACTIVE_TS
        try:
            lang = str(req.get("language", "en"))
            model = _ensure_thread_model(lang)
            _send_response(conn, {"ok": True, "sr": int(model.sr), "device": _get_device(model)})
        except Exception as e:  # noqa: BLE001
            _send_response(conn, {"ok": False, "error": f"{type(e).__name__}: {e}"})
        finally:
            _LAST_ACTIVE_TS = time.monotonic()
            JOB_SLOTS.release()

    try:
        TTS_POOL.submit(run_job)
    except RuntimeError:  # executor shut down mid-request
        JOB_SLOTS.release()
        _send_response(conn, {"ok": False, "error": "daemon shutting down"})


def _submit_tts(conn, req):
    def run_job():
        global _ACTIVE, _LAST_ACTIVE_TS
        try:
            with _ACTIVE_LOCK:
                _ACTIVE += 1
            print(f"[daemon] tts: run_job started (thread={threading.get_ident()})")
            text = str(req.get("text", ""))
            lang = str(req.get("language", "en"))
            prompt_file = req.get("prompt_file") or None
            temperature = float(req.get("temperature", 0.6))
            cfg_weight = float(req.get("cfg_weight", 0.5))
            exaggeration = float(req.get("exaggeration", 0.5))
            output_path = Path(str(req["output_path"])).resolve()

            # Thermal gate — pause while the GPU is over temp (monitor
            # thread sets/clears _THERMAL_BLOCKED; workers wait while set).
            while _THERMAL_BLOCKED.is_set():
                if _QUIT:
                    raise RuntimeError("daemon shutting down")
                time.sleep(5)

            model = _ensure_thread_model(lang)
            t0 = time.time()
            if lang == "id":
                wav = _generate_indonesian(model, text, prompt_file, temperature)
            else:
                if prompt_file:
                    with _GenerationGuard():
                        model.prepare_conditionals(prompt_file)
                wav = _generate_with_retry(model, text, lang, temperature, cfg_weight, exaggeration)

            sr = int(model.sr)
            import soundfile as sf

            sf.write(str(output_path), wav.squeeze(0).cpu().numpy(), sr)
            duration = float(wav.shape[1] / sr)
            print(f"[daemon] tts done ({lang}, {time.time() - t0:.1f}s)")
            _send_response(conn, {"ok": True, "duration": duration, "sr": sr})
        except Exception as e:  # noqa: BLE001 - report any failure to the client
            _send_response(conn, {"ok": False, "error": f"{type(e).__name__}: {e}"})
        finally:
            with _ACTIVE_LOCK:
                _ACTIVE -= 1
            _LAST_ACTIVE_TS = time.monotonic()
            JOB_SLOTS.release()

    try:
        fut = TTS_POOL.submit(run_job)
        _LAST_FUTURES.append(fut)
        del _LAST_FUTURES[:-5]
        print(f"[daemon] tts: run_job submitted to TTS pool (future={id(fut)})")
    except Exception as e:  # executor shut down mid-request
        print(f"[daemon] tts: submit failed ({type(e).__name__}: {e})")
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
                "engine": "chatterbox",
                "device": device_name,
                "max_jobs": MAX_JOBS,
                "active": _ACTIVE,
            },
        )
        return

    if cmd == "debug":
        import concurrent.futures as _cf

        qsize = TTS_POOL._work_queue.qsize() if TTS_POOL else -1
        info = {
            "ok": True,
            "pool_id": id(TTS_POOL) if TTS_POOL else None,
            "work_queue_id": id(TTS_POOL._work_queue) if TTS_POOL else None,
            "work_queue_qsize": qsize,
            "max_workers": TTS_POOL._max_workers if TTS_POOL else None,
            "threads": [t.ident for t in (TTS_POOL._threads or []) if TTS_POOL] ,
            "pool_shutdown": TTS_POOL._shutdown if TTS_POOL else None,
        }
        if TTS_POOL:
            done = {"ran": False}

            def _probe():
                done["ran"] = True

            try:
                fut = TTS_POOL.submit(_probe)
                fut.result(timeout=5)
                info["probe_task"] = "ran"
            except Exception as e:
                info["probe_task"] = f"FAILED: {type(e).__name__}: {e}"
        finfo = []
        for f in _LAST_FUTURES[-5:]:
            try:
                exc = f.exception(timeout=0)
            except Exception as e:
                exc = f"{type(e).__name__}: {e}"
            finfo.append({"done": f.done(), "cancelled": f.cancelled(), "exception": str(exc)[:200]})
        info["last_futures"] = finfo
        _send_response(conn, info)
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
            _send_response(
                conn, {"ok": False, "code": "busy", "error": "daemon busy - retry later", "retry_after": 5}
            )
            return
        _submit_ensure_model(conn, req)
        return

    if cmd == "tts":
        if not JOB_SLOTS.acquire(blocking=False):
            print("[daemon] tts: busy — all worker slots occupied")
            _send_response(
                conn, {"ok": False, "code": "busy", "error": "daemon busy - retry later", "retry_after": 5}
            )
            return
        print("[daemon] tts: slot acquired, submitting to TTS pool")
        _submit_tts(conn, req)
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
        max_workers=MAX_JOBS, initializer=_init_worker, thread_name_prefix="tts"
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

    threading.Thread(target=_thermal_monitor, daemon=True, name="thermal").start()

    print(
        f"[daemon] TTS daemon ready on {DAEMON_SOCK} "
        f"(device: {device_name}, max concurrent jobs: {MAX_JOBS}, "
        f"temp limit: {GPU_TEMP_LIMIT:.0f}°C, idle timeout: {IDLE_TIMEOUT:.0f}s)"
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
