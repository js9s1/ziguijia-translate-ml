#!/usr/bin/env python3
"""
Batch pipeline: read numbers from a file, download each video from ziguijia.com,
run rapid_videocr to generate SRT, translate to English, and record successes/failures.

Requires Python 3.11+ with transformers / optimum-intel (HY-MT) installed.
Usage:
    ./batch_pipeline.py numbers.txt          # fresh run
    ./batch_pipeline.py numbers.txt --force  # re-process even if already done
    ./batch_pipeline.py --translate-run run_20260703_115744  # translate existing SRTs
"""

import sys
import os
import re
import json
import socket
import subprocess
import threading
import time
import shutil
from pathlib import Path

import signal

if sys.stdout is not None:
    sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Paths — set in main() under run_{timestamp}/, adjust in your environment
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RUN_DIR = None
VIDEO_DIR = None
SRT_DIR = None
FRAMES_DIR = None
RESULTS_FILE = None
SUCCESS_FILE = None
FAILED_FILE = None
SKIPPED_FILE = None
RUN_CACHE_FILE = PROJECT_ROOT / "run.cache"

# Official translations from an older run — numbers whose videos already
# have an official English SRT here are skipped entirely.
OFFICIAL_EN_DIR = Path(
    os.environ.get(
        "OFFICIAL_EN_DIR",
        str(PROJECT_ROOT / "oldrun" / "official_trans" / "en"),
    )
)


# Binary for OCR — ROCm-backed copy of rapid_videocr (code_ml/rapid_videocr_daemon)
RAPID_VIDEOCR_BIN = os.environ.get(
    "RAPID_VIDEOCR_BIN",
    "/home/ziguijia/子归家/code_ml/rapid_videocr_daemon/rapid_videocr_rocm.py",
)
OCR_ENGINE = os.environ.get("OCR_ENGINE", "torch")  # ROCm backend

# Warm ROCm OCR daemon (home: /home/ziguijia/子归家/code_ml/rapid_videocr_daemon):
# keeps torch/ROCm models loaded between OCR jobs.
# OCR_MODE=subprocess disables the daemon and runs one subprocess per chunk.
OCR_MODE = os.environ.get("OCR_MODE", "daemon").strip().lower()
DAEMON_DIR = Path("/home/ziguijia/子归家/code_ml/rapid_videocr_daemon")
DAEMON_PYTHON = os.environ.get(
    "DAEMON_PYTHON",
    str(DAEMON_DIR / ".venv-rocm/bin/python"),
)
DAEMON_SCRIPT = DAEMON_DIR / "rapid_videocr_daemon.py"
_RUNTIME_DIR = Path(
    os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
)
DAEMON_SOCK = Path(
    os.environ.get(
        "RAPID_VIDEOCR_DAEMON_SOCK", _RUNTIME_DIR / "rapid_videocr_daemon" / "daemon.sock"
    )
)
DAEMON_PID_FILE = Path(
    os.environ.get(
        "RAPID_VIDEOCR_DAEMON_PID", _RUNTIME_DIR / "rapid_videocr_daemon" / "daemon.pid"
    )
)
DAEMON_LOG = Path(os.path.expanduser("~/logs/rapid_videocr_daemon.log"))
DAEMON_PING_TIMEOUT = float(os.environ.get("DAEMON_PING_TIMEOUT", "10"))
DAEMON_START_WAIT = float(os.environ.get("DAEMON_START_WAIT", "120"))
DAEMON_JOB_TIMEOUT = float(os.environ.get("DAEMON_JOB_TIMEOUT", str(4 * 3600)))
PYTHON311 = os.environ.get("PYTHON311", "/usr/bin/env python3")

# Warm translate daemon (home: /home/ziguijia/子归家/code_ml) — keeps the
# HY-MT model resident on GPU and translates SRT files over a Unix socket.
# The daemon exits when idle above TRANSLATE_IDLE_TEMPERATURE (ROCm's HSA
# busy-poll keeps the box warm while the process lives, so between-job
# cooling requires the exit).  The batch sets the temperature to 75°C —
# the same cooldown target it waits for between jobs — and relaunches the
# daemon on demand.  To hide the restart + model-load cost, the batch
# prewarms the daemon while the last PREWARM_CHUNKS_AHEAD OCR chunks run
# (see run_ocr); the daemon's prewarm grace shields it from the idle-hot
# shutdown during that window.  The daemon stays warm after the batch
# ends — the idle-hot watchdog alone releases the GPU when it actually
# gets hot (use --stop-daemon to force it down).
TRANSLATE_DAEMON_DIR = Path("/home/ziguijia/子归家/code_ml")
TRANSLATE_DAEMON_PYTHON = os.environ.get(
    "TRANSLATE_DAEMON_PYTHON",
    str(Path("~/.pyenv/versions/3.11.14/bin/python3.11").expanduser()),
)
TRANSLATE_DAEMON_SCRIPT = Path(
    os.environ.get(
        "TRANSLATE_DAEMON_SCRIPT",
        str(TRANSLATE_DAEMON_DIR / "translate" / "translate_daemon.py"),
    )
)
TRANSLATE_DAEMON_SOCK = Path(
    os.environ.get(
        "TRANSLATE_DAEMON_SOCK", _RUNTIME_DIR / "translate_daemon" / "daemon.sock"
    )
)
TRANSLATE_DAEMON_PID = Path(
    os.environ.get(
        "TRANSLATE_DAEMON_PID", _RUNTIME_DIR / "translate_daemon" / "daemon.pid"
    )
)
TRANSLATE_DAEMON_LOG = Path(os.path.expanduser("~/logs/translate_daemon.log"))
TRANSLATE_DAEMON_START_WAIT = float(os.environ.get("TRANSLATE_DAEMON_START_WAIT", "300"))
# Max time spent waiting for a free worker slot (queued on 'busy').
# The socket read for an accepted job is unbounded (see run_daemon_translate).
TRANSLATE_DAEMON_JOB_TIMEOUT = float(os.environ.get("TRANSLATE_DAEMON_JOB_TIMEOUT", "21600"))

# Fallback: per-file translate_srt.py subprocess (--mode auto also uses the daemon)
TRANSLATE_PYTHON = os.environ.get(
    "TRANSLATE_PYTHON",
    str(Path("~/.pyenv/versions/3.11.14/bin/python3.11").expanduser()),
)
TRANSLATE_SCRIPT = os.environ.get(
    "TRANSLATE_SCRIPT",
    os.path.join(os.path.dirname(__file__), "..", "code_ml", "translate", "translate_srt.py"),
)

# OCR settings
OCR_FPS = 3
OCR_CROP_PCT = 25           # keep lower 25% of frame
OCR_CHUNK_SIZE = 800        # frames per rapid_videocr chunk
PREWARM_CHUNKS_AHEAD = 1    # prewarm the translate daemon while the last N OCR chunks run

# Translation settings
TARGET_LANG_NAME = os.environ.get("TARGET_LANG_NAME", "English")
SRT_EN_SUFFIX = ".en.srt"
SRT_BI_SUFFIX = ".zh+en.srt"
SKIP_TRANSLATE = False
MARK_START = "杨宁随缘开示"
MARK_START_EN = "Yang Ning"
MARK_END = "子归家全体编制人员"
MARK_END_EN = "All staff members who have returned home"

# Thermal protection
THERMAL_LIMIT = float(os.environ.get("THERMAL_LIMIT", "91"))  # °C — pause if exceeded
THERMAL_POLL_SECS = float(os.environ.get("THERMAL_POLL_SECS", "10"))  # seconds between temp checks
COOLDOWN_TARGET = float(os.environ.get("COOLDOWN_TARGET", "75"))  # °C — resume when below this
_stop_requested = False          # set by signal handler
_paused_for_thermal = False      # set when waiting for cooldown

# ===================================================================
# Download — delegate to pre-process/download_orig.py
# ===================================================================

DOWNLOAD_SCRIPT = PROJECT_ROOT.parent / "pre-process" / "download_orig.py"


def normalize_video_number(number):
    """Normalize a video number: digits are zero-padded to 5, letter prefixes uppercased."""
    number = number.strip()
    if re.fullmatch(r"\d+", number):
        return number.zfill(5)
    match = re.fullmatch(r"([A-Za-z])(\d+)", number)
    if match:
        return match.group(1).upper() + match.group(2)
    return None


def download_video(number, output_dir, force=False):
    """
    Download video for *number* from ziguijia.com into *output_dir*.
    Returns (ok: bool, video_path: Path | None)
    """
    number = normalize_video_number(number)
    if not number:
        print(f"  Invalid video number")
        return False, None

    out_path = output_dir / f"{number}.mp4"
    if out_path.exists() and not force:
        print(f"  Video already exists: {out_path}")
        if _is_html_file(out_path):
            print(f"  Existing file is HTML — re-downloading")
        else:
            return True, out_path

    if not DOWNLOAD_SCRIPT.exists():
        print(f"  Download script not found: {DOWNLOAD_SCRIPT}")
        return False, None

    cmd = [
        sys.executable, str(DOWNLOAD_SCRIPT),
        number, str(output_dir),
    ]
    print(f"  Downloading via {DOWNLOAD_SCRIPT.name}…")
    try:
        result = subprocess.run(cmd, timeout=660)
    except Exception as e:
        print(f"  Download failed: {e}")
        if out_path.exists():
            out_path.unlink()
        return False, None

    if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        print(f"  Download failed (exit {result.returncode})")
        if out_path.exists():
            out_path.unlink()
        return False, None

    if _is_html_file(out_path):
        print("  ERROR: Downloaded file is HTML, not video")
        out_path.unlink()
        return False, None

    return True, out_path


def _is_html_file(path):
    try:
        with open(path, "rb") as f:
            head = f.read(512).lstrip()
        return head.startswith(b"<!DOCTYPE") or head.startswith(b"<html")
    except Exception:
        return False


# ===================================================================
# Warm ROCm OCR daemon client
# ===================================================================

def _daemon_request(payload, timeout=30, sock_path=None, quiet=False):
    """Send one JSON request to a daemon socket. Returns response dict or None."""
    sock_path = sock_path or DAEMON_SOCK
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(str(sock_path))
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
    except Exception as e:
        if not quiet:
            print(f"    daemon request failed: {e}")
        try:
            s.close()
        except Exception:
            pass
        return None


def _daemon_alive():
    resp = _daemon_request({"cmd": "ping"}, timeout=DAEMON_PING_TIMEOUT)
    return resp is not None and resp.get("ok") is True


def _daemon_pid_running():
    try:
        pid = int(DAEMON_PID_FILE.read_text().strip())
    except Exception:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop_ocr_daemon():
    """Stop the warm daemon (if running). Cleans up socket/pid residue."""
    print(f"  Stopping ROCm OCR daemon ({DAEMON_SOCK.name})…")
    _daemon_request({"cmd": "shutdown"}, timeout=DAEMON_PING_TIMEOUT)
    for p in (DAEMON_SOCK, DAEMON_PID_FILE):
        try:
            p.unlink()
        except OSError:
            pass
    print("  Daemon stopped")


def start_ocr_daemon():
    """Attach to a running daemon or launch one. Returns True when ready."""
    if DAEMON_SOCK.exists() and _daemon_alive():
        return True
    if _daemon_pid_running():
        print("  Daemon process alive — waiting for socket to appear…")
        deadline = time.time() + DAEMON_START_WAIT
        while time.time() < deadline:
            if _daemon_alive():
                return True
            time.sleep(1)
        print("  Daemon socket never appeared — removing stale socket")
        try:
            DAEMON_SOCK.unlink()
        except OSError:
            pass
    else:
        for p in (DAEMON_SOCK, DAEMON_PID_FILE):
            try:
                p.unlink()
            except OSError:
                pass

    DAEMON_LOG.parent.mkdir(parents=True, exist_ok=True)
    logf = open(DAEMON_LOG, "a")
    print(f"  Launching ROCm OCR daemon ({DAEMON_PYTHON} → {DAEMON_LOG} )…")
    try:
        subprocess.Popen(
            [DAEMON_PYTHON, str(DAEMON_SCRIPT)],
            stdout=logf,
            stderr=logf,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"  Failed to launch daemon: {e}")
        return False

    deadline = time.time() + DAEMON_START_WAIT
    while time.time() < deadline:
        if _daemon_alive():
            print(f"  Daemon ready ({DAEMON_SOCK.name})")
            return True
        time.sleep(1)
    print("  Daemon did not become ready")
    return False


def run_daemon_ocr(chunk_dir, prefix):
    """Run one OCR chunk through the warm daemon. Retries while the daemon
    is busy (all workers occupied). Returns (ok: bool, error: str | None)."""
    deadline = time.time() + DAEMON_JOB_TIMEOUT
    attempts = 0
    while True:
        resp = _daemon_request(
            {
                "cmd": "ocr",
                "img_dir": str(chunk_dir),
                "save_dir": str(chunk_dir),
                "file_name": prefix,
                "out_format": "srt",
            },
            timeout=DAEMON_JOB_TIMEOUT,
        )
        if resp is None:
            return False, "daemon unreachable"
        if resp.get("code") == "busy":
            attempts += 1
            retry_after = float(resp.get("retry_after", 5))
            if time.time() + retry_after >= deadline:
                return False, (
                    f"daemon busy ({attempts} attempts) - "
                    "timed out waiting for a worker slot"
                )
            print(
                f"    daemon busy ({attempts}), retrying in {retry_after:.0f}s…"
            )
            time.sleep(retry_after)
            continue
        if not resp.get("ok"):
            return False, str(resp.get("error", "ocr failed"))
        return True, None


# ===================================================================
# OCR pipeline (adapted from code_ml/rapid_videocr_pipeline.sh)
# ===================================================================

def run_ocr(video_path, output_srt, keep_frames=False):
    """
    Extract frames, run rapid_videocr, merge duplicates → SRT file.
    Returns (ok: bool, message: str)
    """
    output_srt = Path(output_srt)
    frames_dir = FRAMES_DIR / video_path.stem
    video_path = Path(video_path).resolve()

    print(f"  OCR engine: {OCR_ENGINE} (rapid_videocr ROCm copy)")

    # ---- 1. Extract frames ----
    print(f"  [1/4] Extracting frames (fps={OCR_FPS}, lower {OCR_CROP_PCT}%)…")
    crop_filter = (
        f"crop=iw:ih*{OCR_CROP_PCT}/100:0:ih*(100-{OCR_CROP_PCT})/100"
    )
    extract_filter = f"fps={OCR_FPS},{crop_filter}"

    frames_dir.mkdir(parents=True, exist_ok=True)
    # Clean old frames
    for old in frames_dir.glob("*.png"):
        old.unlink()

    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", extract_filter,
        "-c:v", "png",
        str(frames_dir / "frame_%05d.png"),
    ]
    try:
        subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if e.stderr else ""
        print(f"  ffmpeg failed: {stderr[-300:]}")
        return False, "ffmpeg extract failed"

    # ---- 2. Rename frames to VSF naming convention ----
    print(f"  [2/4] Renaming frames to VSF convention…")
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        return False, "no frames extracted"

    ms_per_frame = 1000.0 / OCR_FPS
    for i, p in enumerate(frame_files):
        start_ms = int(round(i * ms_per_frame))
        end_ms = int(round((i + 1) * ms_per_frame))
        h, rem = divmod(start_ms, 3600000)
        m, rem = divmod(rem, 60000)
        s, mmm = divmod(rem, 1000)
        h2, rem2 = divmod(end_ms, 3600000)
        m2, rem2 = divmod(rem2, 60000)
        s2, mmm2 = divmod(rem2, 1000)
        dst = frames_dir / (
            f"{h}_{m:02d}_{s:02d}_{mmm:03d}__"
            f"{h2}_{m2:02d}_{s2:02d}_{mmm2:03d}_"
            f"0000000000000000000000000.png"
        )
        p.rename(dst)

    # ---- 3. Run rapid_videocr in chunks ----
    daemon_ok = False
    if OCR_MODE == "daemon" and start_ocr_daemon():
        daemon_ok = True
    if daemon_ok:
        print(f"  [3/4] Running rapid_videocr (warm ROCm daemon)…")
    else:
        print(f"  [3/4] Running rapid_videocr (per-chunk subprocess)…")
    all_frames = sorted(frames_dir.glob("*.png"))
    total_frames = len(all_frames)
    chunk_count = (total_frames + OCR_CHUNK_SIZE - 1) // OCR_CHUNK_SIZE
    print(f"    {total_frames} frames → {chunk_count} chunk(s) of {OCR_CHUNK_SIZE}")

    temp_dir = frames_dir / "_ocr_temp"
    temp_dir.mkdir(exist_ok=True)
    chunk_srts = []
    prewarm_started = False

    for chunk_idx in range(chunk_count):
        start = chunk_idx * OCR_CHUNK_SIZE
        end = min(start + OCR_CHUNK_SIZE, total_frames)

        chunk_dir = temp_dir / f"chunk_{chunk_idx}"
        chunk_dir.mkdir(exist_ok=True)

        for f in all_frames[start:end]:
            f.rename(chunk_dir / f.name)

        # Prewarm the translate daemon while the last OCR chunks run, so
        # the translate step that follows starts with a warm model
        # instead of paying the daemon restart + model load.
        if (
            not SKIP_TRANSLATE
            and not prewarm_started
            and chunk_count - chunk_idx <= PREWARM_CHUNKS_AHEAD
        ):
            prewarm_started = True
            print("    Prewarming translate daemon (background)…")
            threading.Thread(
                target=prewarm_translate_daemon,
                daemon=True,
                name="translate-prewarm",
            ).start()

        prefix = f"result_chunk{chunk_idx}"
        print(f"    Chunk {chunk_idx + 1}/{chunk_count}: frames {start + 1}–{end}")
        if daemon_ok:
            ok, err = run_daemon_ocr(chunk_dir, prefix)
            if not ok:
                print(f"    daemon OCR failed ({err})")
        else:
            try:
                subprocess.run(
                    [
                        RAPID_VIDEOCR_BIN, "-i", str(chunk_dir),
                        "-s", str(chunk_dir), "-f", prefix, "-o", "srt",
                    ],
                    capture_output=True,
                    timeout=1800,
                )
            except Exception as e:
                print(f"    rapid_videocr failed: {e}")

        # Find the output SRT
        possible_srts = list(chunk_dir.glob(f"{prefix}.srt")) + list(
            (chunk_dir / "outputs").glob(f"{prefix}.srt")
        )
        for srt in possible_srts:
            chunk_srts.append(srt)
            break

    if not chunk_srts:
        return False, "no SRT produced"

    # Concatenate chunks
    raw_srt = temp_dir / "result.srt"
    with open(raw_srt, "w") as out:
        for srt in chunk_srts:
            out.write(srt.read_text())
            out.write("\n")

    print(f"    Raw SRT: {raw_srt} ({len(chunk_srts)} chunk(s))")

    # ---- 4. Merge adjacent duplicate segments ----
    print(f"  [4/4] Merging adjacent duplicates…")
    merged = _merge_srt(raw_srt, output_srt)
    print(f"    Before: {_count_srt_entries(raw_srt)}  After: {_count_srt_entries(output_srt)}")

    # ---- Cleanup ----
    if not keep_frames:
        shutil.rmtree(frames_dir)
    else:
        # Restore frames
        for chunk_dir in temp_dir.glob("chunk_*"):
            for f in chunk_dir.glob("*.png"):
                f.rename(frames_dir / f.name)

    return True, str(output_srt)


def _parse_srt_blocks(text):
    """Parse SRT text into list of (time_line, text)."""
    entries = []
    for block in text.strip().split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue
        time_line = lines[1]
        text = "\n".join(lines[2:]).strip()
        entries.append((time_line, text))
    return entries


def _clean_text(t):
    import re as _re
    t = _re.sub(r"^1[\n\s]*", "", t)
    t = _re.sub(r"^[\d\W]+", "", t)
    return t.strip()


def _is_similar(a, b, threshold=0.75):
    from difflib import SequenceMatcher
    if not a or not b:
        return False
    if a == b:
        return True

    def norm(s):
        return re.sub(r"[^\u4e00-\u9fff\uff00-\uffef\u3000-\u303f]", "", s)

    na, nb = norm(a), norm(b)
    if not na or not nb:
        return False
    return SequenceMatcher(None, na, nb).ratio() >= threshold


def _count_srt_entries(srt_path):
    try:
        text = Path(srt_path).read_text()
        return len(_parse_srt_blocks(text))
    except Exception:
        return 0


def _merge_srt(raw_srt_path, output_srt_path):
    text = Path(raw_srt_path).read_text()
    # Normalize timestamps: rapid_videocr may use '.' instead of ',' for ms
    text = _normalize_srt_timestamps(text)
    entries = _parse_srt_blocks(text)
    merged = []

    for time_line, text in entries:
        cleaned = _clean_text(text)
        if not cleaned:
            continue
        if merged and _is_similar(cleaned, merged[-1][1]):
            end_time = time_line.split(" --> ")[1]
            prev_start = merged[-1][0].split(" --> ")[0]
            best = cleaned if len(cleaned) > len(merged[-1][1]) else merged[-1][1]
            merged[-1] = (f"{prev_start} --> {end_time}", best)
        else:
            merged.append((time_line, cleaned))

    out_lines = []
    for i, (time_line, text) in enumerate(merged, 1):
        out_lines.append(str(i))
        out_lines.append(time_line)
        out_lines.append(text)
        out_lines.append("")

    Path(output_srt_path).write_text("\n".join(out_lines))
    return merged


# ===================================================================
# Thermal monitoring & graceful shutdown
# ===================================================================

def _get_cpu_temp():
    """Read CPU temperature via 'sensors -j' or /sys/class/thermal. Returns float °C or None."""
    try:
        result = subprocess.run(
            ["sensors", "-j"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            # Priority: prefer k10temp / acpitz / coretemp chip names
            cpu_temps = []   # from known CPU sensors
            other_temps = [] # from other sensors with temp readings
            for chip_name, chip in data.items():
                if not isinstance(chip, dict):
                    continue
                is_cpu = any(t in chip_name.lower() for t in ("k10temp", "acpitz", "coretemp"))
                for key, val in chip.items():
                    if not isinstance(val, dict):
                        continue
                    for subkey, subval in val.items():
                        if not isinstance(subval, (int, float)):
                            continue
                        # Filter absurd values (sensor max/limit values like 65261)
                        if subval <= 0 or subval > 200:
                            continue
                        subkey_l = subkey.lower()
                        if any(t in subkey_l for t in ("tctl", "tdie", "temp", "package", "core")):
                            (cpu_temps if is_cpu else other_temps).append(float(subval))
            # Prefer CPU sensor readings, fall back to any plausible temp
            candidates = cpu_temps or other_temps
            if candidates:
                return max(candidates)
    except Exception:
        pass

    try:
        for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
            try:
                ztype = (zone / "type").read_text().strip()
                if ztype in ("acpitz", "x86_pkg_temp", "cpu-thermal"):
                    raw = int((zone / "temp").read_text().strip())
                    return raw / 1000.0
            except Exception:
                continue
    except Exception:
        pass

    return None


def _thermal_check():
    """Check temperature and pause if too high. Returns True if ok, False if stop requested."""
    global _paused_for_thermal
    temp = _get_cpu_temp()
    if temp is None:
        return True
    if temp >= THERMAL_LIMIT:
        print(f"\n  ⚠ CPU temp {temp:.0f}°C ≥ limit {THERMAL_LIMIT:.0f}°C — pausing for cooldown…")
        _paused_for_thermal = True
        while temp is not None and temp > COOLDOWN_TARGET:
            if _stop_requested:
                return False
            time.sleep(THERMAL_POLL_SECS)
            temp = _get_cpu_temp()
            if temp is not None:
                print(f"\r  Cooling… {temp:.0f}°C (target ≤{COOLDOWN_TARGET:.0f}°C)", end="", flush=True)
        if temp is not None:
            print(f"\n  ✓ Cooled to {temp:.0f}°C — resuming")
        _paused_for_thermal = False
    return True


def _signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM — set stop flag for graceful shutdown."""
    global _stop_requested
    sig_name = signal.Signals(signum).name
    if _stop_requested:
        print(f"\n  Second {sig_name} received — forcing exit.", file=sys.stderr)
        os._exit(1)
    _stop_requested = True
    print(f"\n\n  ⏸ {sig_name} received — will stop gracefully after current task. Press again to force-quit.",
          file=sys.stderr)


# ===================================================================
# Translation (via warm translate daemon, fallback: subprocess)
# ===================================================================

def _translate_daemon_alive(quiet=False):
    resp = _daemon_request(
        {"cmd": "ping"},
        timeout=DAEMON_PING_TIMEOUT,
        sock_path=TRANSLATE_DAEMON_SOCK,
        quiet=quiet,
    )
    return resp is not None and resp.get("ok") is True


def _translate_daemon_pid_running():
    try:
        pid = int(TRANSLATE_DAEMON_PID.read_text().strip())
    except Exception:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop_translate_daemon():
    """Stop the warm translate daemon (if running). Cleans up socket/pid residue."""
    print(f"  Stopping translate daemon ({TRANSLATE_DAEMON_SOCK.name})…")
    _daemon_request(
        {"cmd": "shutdown"},
        timeout=DAEMON_PING_TIMEOUT,
        sock_path=TRANSLATE_DAEMON_SOCK,
    )
    for p in (TRANSLATE_DAEMON_SOCK, TRANSLATE_DAEMON_PID):
        try:
            p.unlink()
        except OSError:
            pass
    print("  Translate daemon stopped")


_TRANSLATE_DAEMON_PROC = None  # last Popen handle — poll() reaps it so it can't zombie


def ensure_translate_daemon():
    """Attach to a running translate daemon or launch one. Returns True when ready."""
    global _TRANSLATE_DAEMON_PROC
    if _TRANSLATE_DAEMON_PROC is not None:
        _TRANSLATE_DAEMON_PROC.poll()  # reap a previously exited daemon
    if TRANSLATE_DAEMON_SOCK.exists() and _translate_daemon_alive():
        return True
    if _translate_daemon_pid_running():
        print("  Translate daemon alive — waiting for socket to appear…")
        deadline = time.time() + TRANSLATE_DAEMON_START_WAIT
        while time.time() < deadline:
            if _translate_daemon_alive(quiet=True):
                return True
            time.sleep(1)
        print("  Translate daemon socket never appeared — removing stale socket")
        try:
            TRANSLATE_DAEMON_SOCK.unlink()
        except OSError:
            pass
    else:
        for p in (TRANSLATE_DAEMON_SOCK, TRANSLATE_DAEMON_PID):
            try:
                p.unlink()
            except OSError:
                pass

    TRANSLATE_DAEMON_LOG.parent.mkdir(parents=True, exist_ok=True)
    logf = open(TRANSLATE_DAEMON_LOG, "a")
    env = dict(os.environ)
    env["TRANSLATE_IDLE_TEMPERATURE"] = "75"  # exit while cooling between jobs; relaunched on demand
    print(
        f"  Launching translate daemon ({TRANSLATE_DAEMON_PYTHON} → {TRANSLATE_DAEMON_LOG})…"
    )
    try:
        _TRANSLATE_DAEMON_PROC = subprocess.Popen(
            [TRANSLATE_DAEMON_PYTHON, "-u", str(TRANSLATE_DAEMON_SCRIPT)],
            stdout=logf,
            stderr=logf,
            env=env,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"  Failed to launch translate daemon: {e}")
        return False

    deadline = time.time() + TRANSLATE_DAEMON_START_WAIT
    while time.time() < deadline:
        if _translate_daemon_alive(quiet=True):
            print(f"  Translate daemon ready ({TRANSLATE_DAEMON_SOCK.name})")
            return True
        time.sleep(1)
    print("  Translate daemon did not become ready")
    return False


def prewarm_translate_daemon():
    """Start the translate daemon and load the HY-MT model in the
    background (see run_ocr).  The daemon's prewarm grace shields it
    from the idle-hot release while OCR finishes; failures are harmless
    — translate_srt() falls back to launching the daemon itself."""
    if ensure_translate_daemon():
        resp = _daemon_request(
            {"cmd": "ensure_model"},
            timeout=1800,
            sock_path=TRANSLATE_DAEMON_SOCK,
        )
        if resp and resp.get("ok"):
            print(f"  Translate daemon prewarmed ({resp.get('device')})")
        else:
            print(f"  Translate daemon prewarm not ready ({resp or 'no response'})")


def run_daemon_translate(srt_path, en_path, lang, intro=None, outro=None):
    """Translate one SRT through the warm translate daemon. Retries while the
    daemon is busy (all worker slots occupied). Returns (ok, error | None)."""
    deadline = time.time() + TRANSLATE_DAEMON_JOB_TIMEOUT
    attempts = 0
    while True:
        resp = _daemon_request(
            {
                "cmd": "translate",
                "input_path": str(srt_path),
                "output_path": str(en_path),
                "language": lang,
                "intro": intro,
                "outro": outro,
            },
            # While the job runs no client timeout may apply — a translation
            # can legitimately take hours.  The daemon replies 'busy' promptly
            # when queued; once a slot is free the socket stays open until the
            # job completes (a dead daemon breaks the socket, handled below).
            timeout=None,
            sock_path=TRANSLATE_DAEMON_SOCK,
        )
        if resp is None:
            return False, "translate daemon unreachable"
        if resp.get("code") == "busy":
            attempts += 1
            retry_after = float(resp.get("retry_after", 5))
            if time.time() + retry_after >= deadline:
                return False, (
                    f"translate daemon busy ({attempts} attempts) - "
                    "timed out waiting for a worker slot"
                )
            print(
                f"    translate daemon busy ({attempts}), retrying in {retry_after:.0f}s…"
            )
            time.sleep(retry_after)
            continue
        if not resp.get("ok"):
            return False, str(resp.get("error", "translate failed"))
        return True, None


def translate_srt(srt_path, en_path):
    """Translate one SRT via the warm daemon, falling back to a
    translate_srt.py subprocess. Returns (ok, error | None)."""
    if ensure_translate_daemon():
        ok, err = run_daemon_translate(
            srt_path, en_path, TARGET_LANG_NAME, MARK_START, MARK_END
        )
        if ok:
            return True, None
        print(f"  Translate daemon failed ({err}) — falling back to translate_srt.py subprocess")
    else:
        print(f"  Translate daemon unavailable — using translate_srt.py subprocess")
    try:
        subprocess.run(
            [
                TRANSLATE_PYTHON, TRANSLATE_SCRIPT,
                str(srt_path), str(en_path),
                "-l", TARGET_LANG_NAME,
                "--intro", MARK_START,
                "--outro", MARK_END,
            ],
            check=True,
            # No cap while the job is translating (same rationale as above).
            timeout=None,
        )
        return True, None
    except Exception as e:
        return False, str(e)


def _normalize_srt_timestamps(text):
    text = re.sub(r"(\d{1,2}:\d{1,2}:\d{1,2})[.:](\d{3})", r"\1,\2", text)
    text = re.sub(
        r"(\d{1,2}:\d{1,2}:\d{1,2},\d{3})\s*->\s*(\d{1,2}:\d{1,2}:\d{1,2},\d{3})",
        r"\1 --> \2", text,
    )
    return text


def read_srt_text(path):
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8-sig", "utf-16-le", "utf-16-be", "gbk", "gb2312", "gb18030", "utf-8"):
        try:
            text = raw.decode(enc)
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            return _normalize_srt_timestamps(text)
        except (UnicodeDecodeError, LookupError):
            continue
    text = raw.decode("utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _normalize_srt_timestamps(text)

def merge_bilingual_srt(zh_srt_path, en_srt_path, bilingual_srt_path):
    """Combine zh + en SRTs using en.srt as the driver.
    Empty en segments stay empty; non-empty en segments get matched with
    the corresponding zh segment by block index (not by position)."""
    zh_content = read_srt_text(zh_srt_path)
    en_content = read_srt_text(en_srt_path)

    def parse_srt(text):
        blocks = re.split(r"\n\n", text.strip())
        result = []
        for b in blocks:
            lines = b.split("\n")
            while lines and lines[0].strip() == "":
                lines.pop(0)
            while lines and lines[-1].strip() == "":
                lines.pop()
            if len(lines) >= 2:
                idx = lines[0]
                time_range = lines[1]
                text = "\n".join(lines[2:]) if len(lines) >= 3 else ""
                result.append((idx, time_range, text))
        return result

    zh_blocks = parse_srt(zh_content)
    en_blocks = parse_srt(en_content)

    zh_lookup = {idx: text for idx, _, text in zh_blocks}

    combined = []
    for idx, time_range, en_text in en_blocks:
        if en_text.strip():
            zh_text = zh_lookup.get(idx, "")
            combined.append(f"{idx}\n{time_range}\n{zh_text}\n{en_text}")
        else:
            combined.append(f"{idx}\n{time_range}")

    with open(bilingual_srt_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(combined) + "\n")


# ===================================================================
# Batch processing
# ===================================================================

def load_numbers(file_path):
    """Read numbers from file (one per line, skip blank/comment lines)."""
    numbers = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Extract the first number (optionally letter-prefixed) from the line
            m = re.match(r"([A-Za-z]?\d+)", line)
            if m:
                number = normalize_video_number(m.group(1))
                if number:
                    numbers.append(number)
    return numbers


def load_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {"successful": [], "failed": {}, "timestamp": ""}


def save_results(results):
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(SUCCESS_FILE, "w") as f:
        for entry in results.get("successful", []):
            f.write(f"{entry['number']}\n")
    with open(FAILED_FILE, "w") as f:
        for num, reason in results.get("failed", {}).items():
            f.write(f"{num}: {reason}\n")


def load_run_cache():
    if RUN_CACHE_FILE.exists():
        with open(RUN_CACHE_FILE) as f:
            return json.load(f)
    return {"run_dir": "", "items": {}}


def save_run_cache(cache):
    with open(RUN_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


ALL_STAGES = ("download", "ocr", "translate")


def _item_done(item):
    if not isinstance(item, dict):
        return False
    return all(item.get(stage) == "ok" for stage in ALL_STAGES)


_official_en_names = None


def _official_en_names_cached():
    """List of file names in OFFICIAL_EN_DIR, cached for the run."""
    global _official_en_names
    if _official_en_names is None:
        if OFFICIAL_EN_DIR.is_dir():
            _official_en_names = sorted(p.name for p in OFFICIAL_EN_DIR.iterdir())
        else:
            print(f"  Note: official translation dir not found: {OFFICIAL_EN_DIR}")
            _official_en_names = []
    return _official_en_names


def find_official_en(number):
    """Return the official English SRT file name containing '№{number}', or None."""
    marker = f"\u2116{number}"
    for name in _official_en_names_cached():
        if marker in name:
            return name
    return None


def process_number(number, force=False, status=None, cache=None):
    """Download + OCR for a single number. Returns (ok, detail, status).
    *status* is a dict like {"download": "ok", "ocr": "", "translate": ""} from run.cache."""
    if status is None:
        status = {}
    print(f"\n{'=' * 60}")
    print(f"Processing: {number}")
    print(f"{'=' * 60}")
    t0 = time.time()

    def save():
        if cache is not None:
            cache["items"][number] = status
            save_run_cache(cache)

    def stage(name):
        return "" if force else status.get(name, "")

    # ---- Stop / Thermal check ----
    if _stop_requested:
        return False, "stopped by user", status
    if not _thermal_check():
        return False, "stopped (thermal)", status

    # ---- Download ----
    video_path = None
    if stage("download") == "ok":
        video_path = VIDEO_DIR / f"{number}.mp4"
        if video_path.exists() and not _is_html_file(video_path):
            print(f"  Download: already done (cached)")
        else:
            status["download"] = ""
            save()
            video_path = None

    if video_path is None:
        ok, video_path = download_video(number, VIDEO_DIR, force=force)
        if not ok:
            elapsed = time.time() - t0
            print(f"  → FAILED after {elapsed:.0f}s (download)")
            return ok, "download failed", status
        status["download"] = "ok"
        save()

    # ---- Stop / Thermal re-check after download ----
    if _stop_requested:
        return False, "stopped by user", status

    # ---- OCR ----
    srt_path = SRT_DIR / f"{number}.srt"
    if stage("ocr") == "ok" and srt_path.exists():
        print(f"  OCR: already done (cached)")
    elif srt_path.exists() and not force:
        print(f"  SRT already exists: {srt_path}")
    else:
        ok, msg = run_ocr(video_path, srt_path)
        if not ok:
            elapsed = time.time() - t0
            print(f"  → FAILED after {elapsed:.0f}s ({msg})")
            return False, msg, status
    status["ocr"] = "ok"
    save()

    # ---- Translate ----
    if SKIP_TRANSLATE:
        print(f"  Skipping translation (--skip-translate)")
    else:
        en_srt_path = SRT_DIR / f"{number}{SRT_EN_SUFFIX}"
        bi_srt_path = SRT_DIR / f"{number}{SRT_BI_SUFFIX}"

        if stage("translate") == "ok" and en_srt_path.exists():
            print(f"  Translate: already done (cached)")
        elif not force and en_srt_path.exists():
            print(f"  EN SRT already exists: {en_srt_path}")
            status["translate"] = "ok"
            save()
        else:
            print(f"  Translating to {TARGET_LANG_NAME}…")
            ok, err = translate_srt(srt_path, en_srt_path)
            if not ok:
                print(f"  Translation failed: {err}")
                elapsed = time.time() - t0
                print(f"  → PARTIAL ({elapsed:.0f}s) — OCR ok, translation failed")
                return True, f"{srt_path} (translation failed: {err})", status
            status["translate"] = "ok"
            save()

        if bi_srt_path.exists() and not force:
            print(f"  Bilingual SRT already exists: {bi_srt_path}")
        else:
            print(f"  Merging bilingual SRT…")
            merge_bilingual_srt(str(srt_path), str(en_srt_path), str(bi_srt_path))
            print(f"  → {bi_srt_path}")

    elapsed = time.time() - t0
    print(f"  → SUCCESS ({elapsed:.0f}s)")
    return True, str(srt_path), status


def translate_existing_run(run_dir):
    """Translate all SRT files in an existing run directory."""
    run_path = Path(run_dir)
    srt_dir = run_path / "srts"
    if not srt_dir.is_dir():
        print(f"Error: srts/ not found in {run_path}")
        sys.exit(1)

    srt_files = sorted(
        p for p in srt_dir.glob("*.srt")
        if not p.name.endswith(SRT_EN_SUFFIX) and not p.name.endswith(SRT_BI_SUFFIX)
    )
    if not srt_files:
        print(f"No SRT files found in {srt_dir}")
        return

    print(f"Translate-only mode — target: {TARGET_LANG_NAME}")
    print(f"Run dir:  {run_path}")
    print(f"SRTs:     {len(srt_files)} files")
    print()

    ok_count = 0
    fail_count = 0
    t0 = time.time()

    for i, srt_path in enumerate(srt_files, 1):
        if _stop_requested:
            print(" — stopping (progress not lost, just rerun)")
            break

        en_path = srt_path.with_suffix(f"{SRT_EN_SUFFIX}")
        bi_path = srt_path.with_suffix(f"{SRT_BI_SUFFIX}")
        number = srt_path.stem
        print(f"[{i}/{len(srt_files)}] {number}", end="")

        need_en = not en_path.exists()
        need_bi = not bi_path.exists()

        if not need_en and not need_bi:
            print(f" — already done")
            ok_count += 1
            continue

        if need_en:
            print(f"\n  Translating…")
            ok, err = translate_srt(srt_path, en_path)
            if not ok:
                fail_count += 1
                print(f"  → FAILED: {err}")
                continue
            print(f"  → {en_path}")
        else:
            print(f" — en exists")

        if need_bi:
            print(f"  Merging bilingual…")
            merge_bilingual_srt(str(srt_path), str(en_path), str(bi_path))
            print(f"  → {bi_path}")
        else:
            print(f" — zh+en exists")

        ok_count += 1
        _thermal_check()

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done. {ok_count} ok, {fail_count} failed ({elapsed:.0f}s)")


def main():
    import argparse

    global TARGET_LANG_NAME, SKIP_TRANSLATE

    parser = argparse.ArgumentParser(
        description="Batch download + OCR + translate pipeline for ziguijia.com videos"
    )
    parser.add_argument(
        "numbers_file",
        nargs="?",
        default="numbers.txt",
        help="File containing numbers, one per line (default: numbers.txt)",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-process even if already done",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep extracted frames after OCR",
    )
    parser.add_argument(
        "--target-language",
        default=TARGET_LANG_NAME,
        help=f"Target language for translation (default: {TARGET_LANG_NAME})",
    )
    parser.add_argument(
        "--skip-translate",
        action="store_true",
        help="Skip translation step (OCR only)",
    )
    parser.add_argument(
        "--translate-run",
        metavar="RUN_DIR",
        default=None,
        help="Translate existing SRT files in a previous run directory (skip download + OCR)",
    )
    parser.add_argument(
        "--stop-daemon",
        action="store_true",
        help="Stop the warm ROCm OCR + translate daemons and exit",
    )
    args = parser.parse_args()
    TARGET_LANG_NAME = args.target_language
    SKIP_TRANSLATE = args.skip_translate

    if args.stop_daemon:
        stop_ocr_daemon()
        stop_translate_daemon()
        return

    print(f"Starting job at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if args.translate_run:
        translate_existing_run(args.translate_run)
        print(f"Job completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return

    numbers_path = Path(args.numbers_file)
    if not numbers_path.exists():
        print(f"Error: file not found: {numbers_path}")
        sys.exit(1)

    numbers = load_numbers(numbers_path)
    if not numbers:
        print("Error: no valid numbers found in file")
        sys.exit(1)

    seen = set()
    unique = [n for n in numbers if not (n in seen or seen.add(n))]
    if len(unique) < len(numbers):
        print(f"Note: {len(numbers) - len(unique)} duplicate(s) removed")
    numbers = unique

    # ---- Set up run directory from cache ----
    global RUN_DIR, VIDEO_DIR, SRT_DIR, FRAMES_DIR, RESULTS_FILE, SUCCESS_FILE, FAILED_FILE, SKIPPED_FILE
    cache = load_run_cache()
    run_ts = time.strftime("%Y%m%d_%H%M%S")

    if cache["run_dir"] and not args.force:
        old_dir = PROJECT_ROOT / cache["run_dir"]
        if not old_dir.exists():
            print(f"Cache is messed up, there supposed to be a run directory"
                  f" {old_dir} but it does not exist")
            sys.exit(1)
        all_done = all(_item_done(cache["items"].get(n, {})) for n in numbers)
        if all_done:
            print("All video numbers are processed, please run move_batch.sh")
            sys.exit(0)
        new_dir = PROJECT_ROOT / f"run_{run_ts}"
        old_dir.rename(new_dir)
        RUN_DIR = new_dir
        cache["run_dir"] = str(RUN_DIR.name)
        save_run_cache(cache)
        print(f"Resuming (renamed {old_dir.name} → {new_dir.name})")
    else:
        RUN_DIR = PROJECT_ROOT / f"run_{run_ts}"
        cache = {"run_dir": str(RUN_DIR.name), "items": {}}

    VIDEO_DIR = RUN_DIR / "videos"
    SRT_DIR = RUN_DIR / "srts"
    FRAMES_DIR = RUN_DIR / "frames"
    RESULTS_FILE = RUN_DIR / "results.json"
    SUCCESS_FILE = RUN_DIR / "success.txt"
    FAILED_FILE = RUN_DIR / "failed.txt"
    SKIPPED_FILE = RUN_DIR / "skipped.txt"

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    SRT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Run dir:    {RUN_DIR}")
    print(f"Numbers to process: {len(numbers)}")
    print(f"  Video dir: {VIDEO_DIR}")
    print(f"  SRT dir:   {SRT_DIR}")

    results = load_results()
    already_done = {e["number"] for e in results.get("successful", [])}

    total = len(numbers)
    ok_count = 0
    fail_count = 0

    overall_start = time.time()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    temp = _get_cpu_temp()
    if temp is not None:
        print(f"CPU temp:   {temp:.0f}°C (limit: {THERMAL_LIMIT:.0f}°C, cooldown target: {COOLDOWN_TARGET:.0f}°C)")
    print(f"Stop:       Press Ctrl+C once for graceful stop, twice to force-quit")
    print(f"Env:        THERMAL_LIMIT={THERMAL_LIMIT}  THERMAL_POLL_SECS={THERMAL_POLL_SECS}  COOLDOWN_TARGET={COOLDOWN_TARGET}")
    print()

    for i, number in enumerate(numbers, 1):
        item_status = cache["items"].get(number, {})
        if _item_done(item_status) and not args.force:
            print(f"\n[{i}/{total}] {number} — already done (cached), skipping")
            ok_count += 1
            continue

        official_en = find_official_en(number) if not args.force else None
        if official_en:
            print(f"\n[{i}/{total}] {number} — official English translation exists"
                  f" ({official_en}), skipping")
            cache["items"][number] = {stage: "ok" for stage in ALL_STAGES}
            save_run_cache(cache)
            with open(SKIPPED_FILE, "a") as f:
                f.write(f"{number}: {official_en}\n")
            ok_count += 1
            continue

        if number in already_done and not args.force:
            print(f"\n[{i}/{total}] {number} — already done, skipping")
            ok_count += 1
            continue

        print(f"\n[{i}/{total}]", end="")
        ok, detail, item_status = process_number(number, force=args.force, status=item_status, cache=cache)

        if ok:
            ok_count += 1
            results["successful"] = [
                e for e in results["successful"] if e["number"] != number
            ]
            results["successful"].append({"number": number, "srt": detail})
            results["failed"].pop(number, None)
        else:
            fail_count += 1
            results["failed"][number] = detail
            results["successful"] = [
                e for e in results["successful"] if e["number"] != number
            ]

        results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_results(results)

        if _stop_requested:
            print(f"\n  ⏸ Graceful stop after {i}/{total} — progress saved to {RESULTS_FILE}")
            break

    # ---- Summary ----
    total_elapsed = time.time() - overall_start
    print(f"\n{'=' * 60}")
    if _stop_requested:
        print(f"BATCH INTERRUPTED ({ok_count + fail_count}/{total} processed)")
    else:
        print(f"BATCH COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total:      {total}")
    print(f"Successful:  {ok_count}")
    print(f"Failed:      {fail_count}")
    print(f"Duration:    {total_elapsed:.0f}s")
    print(f"Results:    {RESULTS_FILE}")

    if results.get("failed"):
        print(f"\nFailed numbers:")
        for num, reason in results["failed"].items():
            print(f"  {num}: {reason}")

    print(f"Job completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
