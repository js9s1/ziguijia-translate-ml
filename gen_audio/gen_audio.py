import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rocm_env import setup as _rocm_setup

_rocm_setup()  # before any torch import

import soundfile as sf
import srt
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chatterbox-server"))

from config import ASSETS_DIR as CFG_ASSETS_DIR
from config import AUDIO_PROMPT_PATH as CFG_AUDIO_PROMPT_PATH
from config import GEN_AUDIO_DAEMON_SOCK
from gpu_thermal import get_gpu_temp

# Inline read_srt_text to avoid pulling in the full server import chain
# (video_util → jobqueue → middleware → flask). gen_audio is a standalone
# subprocess — it doesn't need the Flask server.
import re as _re

_TIMESTAMP_RE = _re.compile(r"(\d{2}:\d{2}:\d{2})[.,](\d{3})")


def _read_srt_text(path: str) -> str:
    """Read SRT with encoding fallback, normalize timestamps to comma-milliseconds."""
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8-sig", "utf-16-le", "utf-16-be", "gbk", "gb2312", "gb18030", "utf-8"):
        try:
            text = raw.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    else:
        text = raw.decode("utf-8", errors="replace")
    text = text.replace("\r\n", "\n")
    text = _TIMESTAMP_RE.sub(r"\1,\2", text)
    return text

CHANGED_THRESHOLD = 0.05

# Thermal protection — Strix Halo iGPU (gfx1151) can handle sustained load
# but still benefits from thermal monitoring on fanless/compact systems.
GPU_TEMP_LIMIT = float(os.environ.get("GPU_TEMP_LIMIT", "90"))  # °C — pause if exceeded
GPU_COOLDOWN_TARGET = float(os.environ.get("GPU_COOLDOWN_TARGET", "70"))  # °C — resume when below
GPU_POLL_SECS = float(os.environ.get("GPU_POLL_SECS", "10"))
_STOP_REQUESTED = False


def _get_gpu_temp():
    """Read AMD GPU (amdgpu) temperature in °C via hwmon. Returns float or None."""
    return get_gpu_temp()


def _signal_handler(signum, frame):
    global _STOP_REQUESTED
    sig_name = signal.Signals(signum).name
    print(f"\n\n  ⏸ {sig_name} received — will stop after current segment. Press again to force-quit.", file=sys.stderr)
    if _STOP_REQUESTED:
        print(f"\n  Second {sig_name} — forcing exit.", file=sys.stderr)
        os._exit(1)
    _STOP_REQUESTED = True


def _thermal_check():
    """Check GPU temp; pause and cool down if needed. Returns True to continue."""
    global _STOP_REQUESTED
    temp = _get_gpu_temp()
    if temp is None:
        return not _STOP_REQUESTED
    if temp >= GPU_TEMP_LIMIT:
        print(f"\n  ⚠ GPU temp {temp:.0f}°C ≥ limit {GPU_TEMP_LIMIT:.0f}°C — pausing for cooldown…")
        while temp is not None and temp > GPU_COOLDOWN_TARGET:
            if _STOP_REQUESTED:
                return False
            time.sleep(GPU_POLL_SECS)
            temp = _get_gpu_temp()
            if temp is not None:
                print(f"\r  Cooling… GPU {temp:.0f}°C (target ≤{GPU_COOLDOWN_TARGET:.0f}°C)", end="", flush=True)
        if temp is not None:
            print(f"\n  ✓ GPU cooled to {temp:.0f}°C — resuming")
    return not _STOP_REQUESTED


# ── Per-job cache (single JSON, in-memory lookups) ────────────

# One ``cache_meta.json`` per job (in ``tmp/``).  Maps segment index
# to {content, chunks, duration} — the content fingerprint determines
# whether a ``combined_segment_*.wav`` needs regeneration.


def _cache_meta_path(output_dir):
    return os.path.join(output_dir, "tmp", "cache_meta.json")


def _combined_seg_path(output_dir, seg_idx):
    return os.path.join(output_dir, "tmp", f"combined_segment_{seg_idx}.wav")


def _load_cache(output_dir):
    """Return the in-memory cache dict, or an empty one."""
    path = _cache_meta_path(output_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, KeyError):
        return {}


def _save_cache(output_dir, cache):
    """Write the cache dict to disk."""
    path = _cache_meta_path(output_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _check_cache(cache, seg_idx, clean_content, output_dir):
    """In-memory cache lookup.  Returns (is_hit, wav_duration).

    A hit requires the WAV file on disk *and* the content to match.

    If a WAV exists but no cache entry (e.g. crash recovery), the WAV is
    reused and a cache entry is created on the fly.

    Silent WAVs are never reused — the TTS model sometimes "succeeds"
    while producing near-silence, and reusing that would leave audible
    holes in the final audio on every resubmit.
    """
    wav_path = _combined_seg_path(output_dir, seg_idx)
    if not os.path.exists(wav_path):
        return False, 0.0
    seg_cache = cache.get(str(seg_idx))
    if seg_cache:
        # Fallback silence (TTS failed) is never reused — retry generation.
        if seg_cache.get("fallback"):
            return False, 0.0
        if seg_cache.get("content") != clean_content:
            return False, 0.0
        if _cached_wav_is_silent(wav_path):
            print(f"  ↪ cache: seg {seg_idx} wav is silent — regenerating")
            return False, 0.0
        return True, seg_cache.get("duration", 0.0)
    # WAV exists but no cache entry — recover orphaned WAV from crash
    try:
        info = sf.info(wav_path)
    except OSError:
        return False, 0.0
    if _cached_wav_is_silent(wav_path):
        return False, 0.0
    _set_cache(cache, seg_idx, clean_content, info.duration)
    _save_cache(output_dir, cache)
    return True, info.duration


def _set_cache(cache, seg_idx, clean_content, wav_duration, fallback=False):
    cache[str(seg_idx)] = {
        "content": clean_content,
        "duration": wav_duration,
        "fallback": fallback,
    }


def _migrate_cache(cache, new_subs, output_dir):
    """Remap cache entries when SRT segment indices shift.

    For each segment in *new_subs*, match by content against the old
    *cache*.  If found at a different index, rename the corresponding
    WAV file and move the cache entry.

    Old entries with no match in the new SRT are discarded.
    New segments with no match are left empty for regeneration.

    Uses a FIFO list per content value to handle duplicate text across
    multiple segments without one overwriting another.
    """
    if not cache:
        return

    # Build content → list of (old_idx, duration) from old cache
    # (list preserves order for duplicate content)
    old_map: dict[str, list[tuple[int, float]]] = {}
    for old_idx_str, entry in list(cache.items()):
        old_idx = int(old_idx_str)
        content = entry.get("content", "")
        if content:
            old_map.setdefault(content, []).append((old_idx, entry.get("duration", 0.0)))

    # Match each new segment against old cache
    for i, sub in enumerate(new_subs):
        _, clean_content = extract_speaker(sub.content)
        if not clean_content.strip():
            continue

        entries = old_map.get(clean_content)
        if entries:
            old_idx, dur = entries.pop(0)  # FIFO match
            if not entries:
                del old_map[clean_content]
            old_entry = cache.get(str(old_idx), {})
            if old_idx != i:
                old_wav = _combined_seg_path(output_dir, old_idx)
                new_wav = _combined_seg_path(output_dir, i)
                if os.path.exists(old_wav) and not os.path.exists(new_wav):
                    os.rename(old_wav, new_wav)
                    print(f"  ↪ cache: moved seg {old_idx} → seg {i} (same content)")
                elif os.path.exists(new_wav):
                    os.remove(old_wav)
            cache[str(i)] = {
                "content": clean_content,
                "duration": dur,
                "fallback": old_entry.get("fallback", False),
            }
        else:
            # No match — delete stale WAV so _check_cache won't orphan-recover it
            cache.pop(str(i), None)
            stale_wav = _combined_seg_path(output_dir, i)
            if os.path.exists(stale_wav):
                os.remove(stale_wav)

    # Remove stale entries (remaining entries in old_map had no match)
    for entries in old_map.values():
        for old_idx, _ in entries:
            old_key = str(old_idx)
            if old_key in cache:
                stale_wav = _combined_seg_path(output_dir, old_idx)
                if os.path.exists(stale_wav):
                    os.remove(stale_wav)
                del cache[old_key]
                print(f"  ↪ cache: removed stale seg {old_idx}")

    _save_cache(output_dir, cache)


# ── Processing ───────────────────────────────────────────────


def extract_speaker(content):
    m = re.match(r"^\s*(\w+)\s*:\s*(.*)", content, re.DOTALL)
    if m:
        return m.group(1).lower(), m.group(2)
    return None, content


def get_speaker_prompt(speaker, default_prompt=None, assets_dir=CFG_ASSETS_DIR):
    """Look up {speaker}_voice.wav in assets_dir. Returns path if found, else default_prompt."""
    if speaker:
        prompt_path = os.path.join(assets_dir, f"{speaker}_voice.wav")
        if os.path.exists(prompt_path):
            print(f"  Using speaker prompt: {prompt_path}")
            return prompt_path
    return default_prompt


def split_text(text, max_len=120):
    if len(text) <= max_len:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break

        chunk = text[:max_len]
        last_boundary = -1
        for sep in (". ", "! ", "? ", ".\n", "!\n", "?\n", ".\r\n", "!\r\n", "?\r\n"):
            pos = chunk.rfind(sep)
            if pos > last_boundary:
                last_boundary = pos + 1

        if last_boundary > 0:
            chunks.append(text[:last_boundary])
            text = text[last_boundary:].lstrip()
        else:
            last_space = chunk.rfind(" ")
            if last_space > 0:
                chunks.append(text[:last_space])
                text = text[last_space:].lstrip()
            else:
                chunks.append(chunk)
                text = text[max_len:].lstrip()

    return chunks


def load_subs(srt_path):
    """Parse SRT file preserving empty-content entries (unlike srt.parse which drops them).

    Also auto-repairs malformed blocks where a segment absorbed the next one
    due to a missing blank-line separator.
    """
    content = _read_srt_text(srt_path)
    # Split on blank lines to get raw entry blocks
    blocks = re.split(r"\n\n+", content.strip())
    subs = []
    for block in blocks:
        lines = block.strip().split("\n", 2)
        if len(lines) < 2:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue
        ts_match = re.match(r"(\d{1,2}:\d{1,2}:\d{1,2},\d{3})\s*-->\s*(\d{1,2}:\d{1,2}:\d{1,2},\d{3})", lines[1])
        if not ts_match:
            continue
        start = srt.srt_timestamp_to_timedelta(ts_match.group(1))
        end = srt.srt_timestamp_to_timedelta(ts_match.group(2))
        sub_content = lines[2] if len(lines) > 2 else ""
        subs.append(srt.Subtitle(index=idx, start=start, end=end, content=sub_content))

    # Auto-repair swallowed segments
    _swallowed_pat = re.compile(
        r"\n(\d+)\n(\d{1,2}:\d{1,2}:\d{1,2},\d{3}\s*-->\s*\d{1,2}:\d{1,2}:\d{1,2},\d{3})\n(.+)",
        re.DOTALL,
    )
    repaired_subs = []
    for sub in subs:
        m = _swallowed_pat.search(sub.content)
        if m:
            next_idx = int(m.group(1))
            ts_str = m.group(2)
            remaining = m.group(3).rstrip()
            ts_m = re.match(
                r"(\d{1,2}:\d{1,2}:\d{1,2},\d{3})\s*-->\s*(\d{1,2}:\d{1,2}:\d{1,2},\d{3})",
                ts_str,
            )
            if ts_m:
                split_pos = sub.content.index("\n" + m.group(1) + "\n")
                current_text = sub.content[:split_pos].strip()
                repaired_subs.append(srt.Subtitle(
                    index=sub.index, start=sub.start, end=sub.end, content=current_text,
                ))
                repaired_subs.append(srt.Subtitle(
                    index=next_idx,
                    start=srt.srt_timestamp_to_timedelta(ts_m.group(1)),
                    end=srt.srt_timestamp_to_timedelta(ts_m.group(2)),
                    content=remaining,
                ))
                print(f"  ⚠ SRT repair: split swallowed segment {sub.index} → "
                      f"[{sub.index}] + recovered [{next_idx}]")
                continue
        repaired_subs.append(sub)

    return repaired_subs


def generate_silence(duration_sec, sample_rate):
    num_frames = round(duration_sec * sample_rate)
    return torch.zeros(1, num_frames)


def _is_silent_audio(wav_data, threshold_db: float = -45.0, min_voiced_frac: float = 0.05):
    """Return True if *wav_data* is essentially silent.

    A clip counts as voiced only if at least *min_voiced_frac* of its
    samples exceed *threshold_db*.  The fraction test is robust against a
    single loud click inside an otherwise silent clip (unlike RMS), which
    is how the TTS model occasionally "succeeds" while producing silence
    (regression: job 1E606E46 segments 29/37).
    """
    if wav_data.numel() == 0:
        return True
    voiced = (wav_data.abs() > 10 ** (threshold_db / 20)).float().mean().item()
    return voiced < min_voiced_frac


def _cached_wav_is_silent(wav_path: str) -> bool:
    """Return True if the cached WAV on disk is essentially silent."""
    try:
        data, _ = sf.read(wav_path, dtype="float32")
    except OSError:
        return False
    return _is_silent_audio(torch.from_numpy(data))


_SILENCE_MARKER_RE = re.compile(r"<(\d+(?:\.\d+)?)>\s*")


def _generate_chunk_with_markers(
    backend,
    text,
    wav_path,
    temperature,
    prompt_file,
    target_language,
    cfg_weight,
    exaggeration,
    sample_rate,
):
    """Generate one chunk, honouring inline ``<seconds>`` silence markers.

    Text like ``"你好<1.5>世界"`` generates "你好", then 1.5 s of silence,
    then "世界".  Returns (wav_tensor[1, n], duration_s) — the combined
    waveform including inserted silences.
    """
    parts = _SILENCE_MARKER_RE.split(text)
    if len(parts) == 1:
        return backend.generate(
            text,
            wav_path,
            temperature,
            prompt_file=prompt_file,
            target_language=target_language,
            cfg_weight=cfg_weight,
            exaggeration=exaggeration,
        )

    segs = []
    total = 0.0
    first = parts[0].strip()
    if first:
        wav, dur = backend.generate(
            first,
            wav_path,
            temperature,
            prompt_file=prompt_file,
            target_language=target_language,
            cfg_weight=cfg_weight,
            exaggeration=exaggeration,
        )
        segs.append(wav)
        total += dur

    i = 1
    while i < len(parts) - 1:
        silence_sec = float(parts[i])
        seg_text = parts[i + 1].strip()
        if seg_text:
            part_path = f"{os.path.splitext(wav_path)[0]}_p{i}.wav"
            wav, dur = backend.generate(
                seg_text,
                part_path,
                temperature,
                prompt_file=prompt_file,
                target_language=target_language,
                cfg_weight=cfg_weight,
                exaggeration=exaggeration,
            )
            segs.append(wav)
            total += dur
        if silence_sec > 0:
            segs.append(generate_silence(silence_sec, sample_rate))
            total += silence_sec
        i += 2

    if not segs:
        return generate_silence(0.0, sample_rate), 0.0
    wav = torch.cat(segs, dim=1) if len(segs) > 1 else segs[0]
    return wav, total


def combine_audio_segments(segments_info, total_duration, sample_rate):
    max_end = 0
    for seg in segments_info:
        end_sample = round((seg["new_start"] + seg["wav_duration"]) * sample_rate)
        if end_sample > max_end:
            max_end = end_sample

    needed_duration = max_end / sample_rate
    actual_duration = max(total_duration, needed_duration)

    if actual_duration <= 0:
        actual_duration = 1

    combined = generate_silence(actual_duration, sample_rate)
    for seg in segments_info:
        wav_data_np, _ = sf.read(seg["wav_path"], dtype="float32")
        wav_data = torch.from_numpy(wav_data_np).unsqueeze(0)
        start_sample = round(seg["new_start"] * sample_rate)
        end_sample = start_sample + wav_data.shape[1]
        if end_sample > combined.shape[1]:
            new_combined = torch.zeros(1, end_sample)
            new_combined[:, : combined.shape[1]] = combined
            combined = new_combined
        combined[:, start_sample:end_sample] = wav_data
    return combined


def save_audio(output_path, wav_tensor, sample_rate):
    sf.write(output_path, wav_tensor.squeeze(0).cpu().numpy(), sample_rate)


# ── TTS backends ────────────────────────────────────────────
# Backend objects expose:
#   sample_rate -> int (direct mode; property)
#   ensure_model(lang) -> sample_rate (daemon mode)
#   generate(text, wav_path, temperature, prompt_file, target_language,
#            cfg_weight, exaggeration) -> (wav_tensor[1, n], duration_s)


class _DaemonUnavailable(RuntimeError):
    """Raised when the TTS daemon cannot be reached — abort the job so it
    can be retried later (cached segments keep completed work)."""


class _DaemonTTSClient:
    def __init__(self, sock_path: str, auto_start: bool = False):
        self._sock = sock_path
        self._auto_start = auto_start

    def ping(self, timeout: float = 3.0) -> dict | None:
        """Return the ping response, or None if the daemon is unreachable."""
        try:
            return self._request({"cmd": "ping"}, timeout=timeout)
        except _DaemonUnavailable:
            return None

    def ensure_daemon(self, start_wait: float = 120.0) -> bool:
        """Attach to a running daemon or launch one. Returns True when ready."""
        if self.ping() is not None:
            return True

        from config import GEN_AUDIO_DAEMON_SCRIPT, GEN_AUDIO_PYTHON

        log_path = os.path.join(
            os.path.expanduser("~"), "logs", "gen_audio_daemon.log"
        )
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as logf:
            try:
                subprocess.Popen(
                    [GEN_AUDIO_PYTHON, "-u", GEN_AUDIO_DAEMON_SCRIPT],
                    stdout=logf,
                    stderr=logf,
                    start_new_session=True,
                    stdin=subprocess.DEVNULL,
                )
            except Exception as e:
                print(f"  ⚠ failed to start gen_audio daemon: {e}")
                return False

        deadline = time.time() + start_wait
        while time.time() < deadline:
            if self.ping() is not None:
                return True
            time.sleep(1)
        return False

    def _request(self, payload: dict, timeout: float = 600.0) -> dict:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.settimeout(timeout)
            s.connect(self._sock)
            s.sendall((json.dumps(payload) + "\n").encode())
            buf = b""
            while True:
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
                if b"\n" in buf:
                    break
            if not buf:
                raise _DaemonUnavailable("daemon closed connection without response")
            return json.loads(buf.split(b"\n", 1)[0].decode())
        except (FileNotFoundError, ConnectionRefusedError, socket.timeout, OSError) as e:
            raise _DaemonUnavailable(f"daemon unreachable: {e}") from None
        finally:
            try:
                s.close()
            except OSError:
                pass

    def _request_retry(self, payload: dict, timeout: float = 600.0) -> dict:
        while True:
            resp = self._request(payload, timeout)
            if resp.get("code") == "busy":
                if _STOP_REQUESTED:
                    raise SystemExit(1)
                retry_after = float(resp.get("retry_after", 5))
                print(f"  ⏳ daemon busy — retrying in {retry_after:.0f}s")
                time.sleep(retry_after)
                continue
            return resp

    def ensure_model(self, lang: str) -> int:
        resp = self._request_retry({"cmd": "ensure_model", "language": lang}, timeout=1800.0)
        if not resp.get("ok"):
            raise _DaemonUnavailable(f"daemon ensure_model failed: {resp.get('error')}")
        print(f"  ✓ daemon model ready ({resp.get('device')}, sr={resp.get('sr')})")
        return int(resp["sr"])

    def generate(
        self,
        text: str,
        wav_path: str,
        temperature: float,
        prompt_file: str | None,
        target_language: str,
        cfg_weight: float,
        exaggeration: float,
    ):
        resp = self._request_retry(
            {
                "cmd": "tts",
                "text": text,
                "language": target_language,
                "prompt_file": prompt_file,
                "temperature": temperature,
                "cfg_weight": cfg_weight,
                "exaggeration": exaggeration,
                "output_path": wav_path,
            },
            timeout=1800.0,
        )
        if not resp.get("ok"):
            raise RuntimeError(f"daemon tts failed: {resp.get('error')}")
        wav_np, _ = sf.read(wav_path, dtype="float32")
        wav = torch.from_numpy(wav_np).unsqueeze(0)
        return wav, float(resp["duration"])


def process_with_direct(
    srt_path,
    audio_prompt,
    temperature,
    output_dir,
    assets_dir=CFG_ASSETS_DIR,
    target_language="en",
    cfg_weight=0.5,
    exaggeration=0.5,
    backend=None,
):
    """Process an SRT into audio via the TTS backend (gen_audio daemon client)."""
    subs = load_subs(srt_path)
    if not subs:
        print("No subtitles found in SRT file")
        return

    # Ensure tmp subdirectory exists (main() does this, but direct callers may not)
    os.makedirs(os.path.join(output_dir, "tmp"), exist_ok=True)

    # ── Load cache and check which segments need generation ───
    # Defer GPU / model init until we know there is uncached work.
    cache = _load_cache(output_dir)
    _migrate_cache(cache, subs, output_dir)

    uncached_count = 0
    for i, sub in enumerate(subs):
        _, clean_content = extract_speaker(sub.content)
        if not clean_content.strip():
            continue
        cached, _ = _check_cache(cache, i, clean_content, output_dir)
        if not cached:
            uncached_count += 1

    if uncached_count == 0:
        # ── Every segment is cached — skip GPU entirely ────
        cached_wavs = [
            _combined_seg_path(output_dir, i)
            for i in range(len(subs))
            if os.path.exists(_combined_seg_path(output_dir, i))
        ]
        if not cached_wavs:
            print("No cached audio found")
            return
        info = sf.info(cached_wavs[0])
        sample_rate = info.samplerate

        adjusted_subs = []
        segments_info = []
        changed_segments = []
        accumulated_offset = 0.0

        for i, sub in enumerate(subs):
            _, clean_content = extract_speaker(sub.content)
            orig_start = sub.start.total_seconds()
            orig_duration = (sub.end - sub.start).total_seconds()
            seg_wav_path = _combined_seg_path(output_dir, i)

            if not clean_content.strip():
                total_wav_duration = orig_duration
            else:
                _, total_wav_duration = _check_cache(cache, i, clean_content, output_dir)

            new_start = orig_start + accumulated_offset

            if abs(total_wav_duration - orig_duration) > CHANGED_THRESHOLD:
                changed_segments.append(
                    {
                        "segment": i,
                        "time": orig_start,
                        "orig_duration": orig_duration,
                        "new_duration": total_wav_duration,
                        "diff": total_wav_duration - orig_duration,
                    }
                )

            adjusted_subs.append(
                srt.Subtitle(
                    index=i,
                    start=timedelta(seconds=new_start),
                    end=timedelta(seconds=new_start + total_wav_duration),
                    content=sub.content,
                )
            )

            segments_info.append(
                {
                    "wav_path": seg_wav_path,
                    "wav_duration": total_wav_duration,
                    "new_start": new_start,
                }
            )

            if total_wav_duration > orig_duration:
                accumulated_offset += total_wav_duration - orig_duration

        original_total = (subs[-1].end - subs[0].start).total_seconds()
        total_duration = original_total + accumulated_offset
        combined_tensor = combine_audio_segments(segments_info, total_duration, sample_rate)
        _save_cache(output_dir, cache)
        return combined_tensor, adjusted_subs, changed_segments, sample_rate, total_duration

    # ── Some segments need generation — init TTS backend ─────
    sample_rate = backend.ensure_model(target_language)

    adjusted_subs = []
    segments_info = []
    changed_segments = []
    accumulated_offset = 0.0

    seg_counter = 0
    for i, sub in enumerate(subs):
        speaker, clean_content = extract_speaker(sub.content)
        display = f"[{speaker}] " if speaker else ""
        chunks = split_text(clean_content, 120)
        print(f"Processing segment {i}: {display}{clean_content[:50]}... ({len(chunks)} chunk(s))")

        orig_start = sub.start.total_seconds()
        orig_duration = (sub.end - sub.start).total_seconds()

        new_start = orig_start + accumulated_offset

        seg_wav_path = _combined_seg_path(output_dir, i)

        if not clean_content.strip():
            # Empty segment — generate silence and preserve timing
            silence_wav = generate_silence(orig_duration, sample_rate)
            save_audio(seg_wav_path, silence_wav, sample_rate)
            # Remove stale cache entry for empty segments
            cache.pop(str(i), None)
            adjusted_subs.append(
                srt.Subtitle(
                    index=i,
                    start=timedelta(seconds=new_start),
                    end=timedelta(seconds=new_start + orig_duration),
                    content=sub.content,
                )
            )
            segments_info.append(
                {
                    "wav_path": seg_wav_path,
                    "wav_duration": orig_duration,
                    "new_start": new_start,
                }
            )
            continue

        # Non-verbal / sound-effect text → silence.  Only text with NO
        # letters at all (pure symbols/punctuation, e.g. "♪♪", "……")
        # is treated as non-verbal.  Any letter-based text — however
        # short ("So,", "Right?", "分かりますか。") — is real speech and
        # must be generated (regressions: jobs 1E606E46, 9BAB19F2).
        _alpha_count = sum(1 for c in clean_content if c.isalpha())
        if _alpha_count == 0:
            print(f"  ↪ segment {i}: non-verbal ('{clean_content.strip()}') → silence ({orig_duration:.1f}s)")
            silence_wav = generate_silence(orig_duration, sample_rate)
            save_audio(seg_wav_path, silence_wav, sample_rate)
            _set_cache(cache, i, clean_content, orig_duration)
            _save_cache(output_dir, cache)
            adjusted_subs.append(
                srt.Subtitle(
                    index=i,
                    start=timedelta(seconds=new_start),
                    end=timedelta(seconds=new_start + orig_duration),
                    content=sub.content,
                )
            )
            segments_info.append(
                {
                    "wav_path": seg_wav_path,
                    "wav_duration": orig_duration,
                    "new_start": new_start,
                }
            )
            continue

        # ── Check per-segment cache ──
        cached, cached_duration = _check_cache(cache, i, clean_content, output_dir)

        if cached:
            print(f"  ↪ segment {i} cached (duration: {cached_duration:.2f}s), skipping generation")
            total_wav_duration = cached_duration
            # Advance seg_counter past chunk slots this segment would use
            seg_counter += len(chunks)
        else:
            # ── GPU thermal check before heavy work ──
            if not _thermal_check():
                print("  → Stopped by signal")
                raise SystemExit(1)

            prompt = get_speaker_prompt(speaker, audio_prompt, assets_dir)

            chunk_wavs = []
            total_wav_duration = 0.0
            seg_failed = False
            for ci, chunk in enumerate(chunks):
                wav_path = os.path.join(output_dir, "tmp", f"segment_{seg_counter}.wav")
                has_speech = any(p.strip() for p in _SILENCE_MARKER_RE.split(chunk))
                wav_data = None
                wav_duration = 0.0
                recoveries = 0
                for attempt in range(3):
                    try:
                        wav_data, wav_duration = _generate_chunk_with_markers(
                            backend,
                            chunk,
                            wav_path,
                            temperature + attempt * 0.02,
                            prompt_file=prompt,
                            target_language=target_language,
                            cfg_weight=cfg_weight,
                            exaggeration=exaggeration,
                            sample_rate=sample_rate,
                        )
                    except _DaemonUnavailable as e:
                        # The daemon process died mid-request (e.g. a ROCm
                        # GPU fault on an out-of-range flow token, as in
                        # job 9BAB19F2).  Restart it and retry the chunk
                        # instead of aborting the whole job — segments
                        # completed so far stay cached for a resubmit.
                        if recoveries >= 2:
                            raise
                        recoveries += 1
                        print(
                            f"  ⚠ TTS daemon died ({e}) — restarting it and "
                            f"retrying (recovery {recoveries}/2)"
                        )
                        if _STOP_REQUESTED:
                            raise SystemExit(1)
                        try:
                            if not backend.ensure_daemon():
                                raise _DaemonUnavailable("failed to restart TTS daemon")
                            backend.ensure_model(target_language)
                        except _DaemonUnavailable as e2:
                            print(f"  ⚠ daemon recovery failed: {e2}")
                            raise
                        continue
                    except (RuntimeError, SystemExit) as e:
                        # Stuck-loop or other generation error — log and use silence.
                        # Don't fail the whole job for one bad segment.
                        print(f"  ⚠ segment {i} TTS failed: {e}")
                        wav_data = None
                        break
                    if not has_speech or not _is_silent_audio(wav_data):
                        break
                    print(
                        f"  ⚠ segment {i} chunk produced silent audio "
                        f"(attempt {attempt + 1}/3) — retrying"
                    )
                    wav_data = None
                if wav_data is None:
                    # All attempts failed or produced silence — use silence
                    # for this segment rather than failing the whole job.
                    print(f"     Using silence for this segment ({orig_duration:.1f}s)")
                    chunk_wavs = [generate_silence(orig_duration, sample_rate)]
                    total_wav_duration = orig_duration
                    seg_counter += len(chunks)
                    seg_failed = True
                    break  # exit chunk loop, use silence combined_wav below
                if wav_data.dim() == 1:
                    wav_data = wav_data.unsqueeze(0)
                chunk_wavs.append(wav_data)
                total_wav_duration += wav_duration
                seg_counter += 1

            combined_wav = torch.cat(chunk_wavs, dim=1) if len(chunk_wavs) > 1 else chunk_wavs[0]

            # Pad with silence if generated audio is shorter than original duration
            if total_wav_duration < orig_duration:
                silence_duration = orig_duration - total_wav_duration
                silence = generate_silence(silence_duration, sample_rate)
                combined_wav = torch.cat([combined_wav, silence], dim=1)
                total_wav_duration = orig_duration

            save_audio(seg_wav_path, combined_wav, sample_rate)

            # Update in-memory cache and persist immediately.
            # Fallback silence is recorded but flagged so the next run
            # retries TTS instead of reusing the silent WAV.
            _set_cache(cache, i, clean_content, total_wav_duration, fallback=seg_failed)
            if seg_failed:
                print("     (fallback silence not reused — TTS will be retried on the next run)")
            _save_cache(output_dir, cache)

        new_start = orig_start + accumulated_offset
        duration_diff = total_wav_duration - orig_duration

        if abs(duration_diff) > CHANGED_THRESHOLD:
            changed_segments.append(
                {
                    "segment": i,
                    "time": orig_start,
                    "orig_duration": orig_duration,
                    "new_duration": total_wav_duration,
                    "diff": duration_diff,
                }
            )

        adjusted_subs.append(
            srt.Subtitle(
                index=i,
                start=timedelta(seconds=new_start),
                end=timedelta(seconds=new_start + total_wav_duration),
                content=sub.content,
            )
        )

        segments_info.append(
            {
                "wav_path": seg_wav_path,
                "wav_duration": total_wav_duration,
                "new_start": new_start,
            }
        )

        if total_wav_duration > orig_duration:
            accumulated_offset += total_wav_duration - orig_duration

    original_total = (subs[-1].end - subs[0].start).total_seconds()
    total_duration = original_total + accumulated_offset

    if original_total > 0 and total_duration / original_total > 3.0:
        print(
            f"ERROR: Generated audio duration ({total_duration:.2f}s) is more than 3x the original SRT duration ({original_total:.2f}s). TTS model likely failed."
        )
        raise RuntimeError(
            f"TTS duration inflation: {total_duration:.2f}s vs original {original_total:.2f}s (ratio: {total_duration / original_total:.1f}x)"
        )

    combined_tensor = combine_audio_segments(segments_info, total_duration, sample_rate)

    # Persist cache to disk so the next run can benefit from it
    _save_cache(output_dir, cache)

    return combined_tensor, adjusted_subs, changed_segments, sample_rate, total_duration


def process_text(
    text,
    audio_prompt,
    temperature,
    output_dir,
    assets_dir=CFG_ASSETS_DIR,
    target_language="en",
    cfg_weight=0.5,
    exaggeration=0.5,
    backend=None,
):
    """Generate audio from plain text — no SRT input, no SRT output.

    Returns (wav_tensor[1, n], sample_rate, total_duration), or None on
    total failure of every chunk.  Inline ``<seconds>`` silence markers
    are honoured the same way as in SRT mode.
    """
    sample_rate = backend.ensure_model(target_language)

    os.makedirs(os.path.join(output_dir, "tmp"), exist_ok=True)
    chunks = split_text(text, 120)
    prompt = get_speaker_prompt(None, audio_prompt, assets_dir)
    print(f"Text mode: {len(chunks)} chunk(s)")

    chunk_wavs = []
    total_duration = 0.0
    for ci, chunk in enumerate(chunks):
        if not _thermal_check():
            print("  → Stopped by signal")
            raise SystemExit(1)

        wav_path = os.path.join(output_dir, "tmp", f"text_chunk_{ci}.wav")
        has_speech = any(p.strip() for p in _SILENCE_MARKER_RE.split(chunk))
        wav_data = None
        wav_duration = 0.0
        for attempt in range(3):
            try:
                wav_data, wav_duration = _generate_chunk_with_markers(
                    backend,
                    chunk,
                    wav_path,
                    temperature + attempt * 0.02,
                    prompt_file=prompt,
                    target_language=target_language,
                    cfg_weight=cfg_weight,
                    exaggeration=exaggeration,
                    sample_rate=sample_rate,
                )
            except _DaemonUnavailable:
                raise
            except (RuntimeError, SystemExit) as e:
                print(f"  ⚠ chunk {ci} TTS failed: {e}")
                wav_data = None
                break
            if not has_speech or not _is_silent_audio(wav_data):
                break
            print(
                f"  ⚠ chunk {ci} produced silent audio "
                f"(attempt {attempt + 1}/3) — retrying"
            )
            wav_data = None

        if wav_data is None:
            # All attempts failed — skip this chunk instead of failing the job.
            print(f"     Skipping chunk {ci} (TTS failed)")
            continue
        if wav_data.dim() == 1:
            wav_data = wav_data.unsqueeze(0)
        chunk_wavs.append(wav_data)
        total_duration += wav_duration

    if not chunk_wavs:
        print("ERROR: no audio generated for text")
        return None

    wav = torch.cat(chunk_wavs, dim=1) if len(chunk_wavs) > 1 else chunk_wavs[0]
    return wav, sample_rate, total_duration


def main():
    parser = argparse.ArgumentParser(description="Generate audio from SRT segments and output adjusted SRT")
    parser.add_argument(
        "srt",
        nargs="?",
        help="Input SRT file (omit when using --text)",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Plain text to synthesize directly to WAV (no SRT input/output)",
    )
    parser.add_argument(
        "--audio_prompt",
        default=CFG_AUDIO_PROMPT_PATH,
        help="Reference audio file for voice cloning (default/no-speaker)",
    )
    parser.add_argument(
        "--assets_dir",
        default=CFG_ASSETS_DIR,
        help="Directory containing {speaker}_voice.wav files for speaker-specific prompts",
    )
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for audio generation")
    parser.add_argument(
        "--target_language", type=str, default="en", help="Target language code (e.g. en, zh, ja, etc.)"
    )
    parser.add_argument("--cfg_weight", type=float, default=0.5, help="CFG weight for generation")
    parser.add_argument("--exaggeration", type=float, default=0.5, help="Exaggeration level for voice characteristics")
    parser.add_argument("--output_dir", default="./output", help="Output directory")
    parser.add_argument("--output_srt", default="output_adjusted.srt")
    parser.add_argument("--output_wav", default="output.wav")
    parser.add_argument("--changed_json", default="changed_segments.json")
    parser.add_argument(
        "--mode",
        choices=["auto", "daemon"],
        default="auto",
        help="TTS backend: auto (start daemon if needed), daemon (require running daemon)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tmp_dir = os.path.join(args.output_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Register signal handlers
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Show GPU thermal status
    gpu_temp = _get_gpu_temp()
    if gpu_temp is not None:
        print(
            f"GPU temp:   {gpu_temp:.0f}°C (limit: {GPU_TEMP_LIMIT:.0f}°C, cooldown target: {GPU_COOLDOWN_TARGET:.0f}°C)"
        )
    print("Stop:       Ctrl+C once for graceful stop per segment, twice to force-quit")
    print(
        f"Env:        GPU_TEMP_LIMIT={GPU_TEMP_LIMIT}  GPU_COOLDOWN_TARGET={GPU_COOLDOWN_TARGET}  GPU_POLL_SECS={GPU_POLL_SECS}"
    )
    print()

    client = _DaemonTTSClient(GEN_AUDIO_DAEMON_SOCK, auto_start=(args.mode == "auto"))
    if args.mode == "auto" and client.ping() is None:
        print(f"gen_audio daemon not running — starting it ({GEN_AUDIO_DAEMON_SOCK})")
        client.ensure_daemon()
    resp = client.ping()
    if resp is not None and resp.get("ok"):
        print(
            f"Using gen_audio daemon ({GEN_AUDIO_DAEMON_SOCK}, "
            f"engine={resp.get('engine')}, device={resp.get('device')}, max_jobs={resp.get('max_jobs')})"
        )
        backend = client
    else:
        print(f"ERROR: gen_audio daemon not reachable at {GEN_AUDIO_DAEMON_SOCK}", file=sys.stderr)
        return 1

    if args.text is not None:
        result = process_text(
            args.text,
            args.audio_prompt,
            args.temperature,
            args.output_dir,
            args.assets_dir,
            target_language=args.target_language,
            cfg_weight=args.cfg_weight,
            exaggeration=args.exaggeration,
            backend=backend,
        )
        if result is None:
            return 1
        wav_tensor, sample_rate, total_duration = result
        save_audio(os.path.join(args.output_dir, args.output_wav), wav_tensor, sample_rate)
        print(f"Combined WAV: {args.output_wav} (duration: {total_duration:.2f}s)")
        return 0

    if not args.srt:
        print("ERROR: either an SRT file or --text must be provided", file=sys.stderr)
        return 1

    result = process_with_direct(
        args.srt,
        args.audio_prompt,
        args.temperature,
        args.output_dir,
        args.assets_dir,
        target_language=args.target_language,
        cfg_weight=args.cfg_weight,
        exaggeration=args.exaggeration,
        backend=backend,
    )

    if result is None:
        return

    combined_tensor, adjusted_subs, changed_segments, sample_rate, total_duration = result

    save_audio(os.path.join(args.output_dir, args.output_wav), combined_tensor, sample_rate)

    with open(os.path.join(args.output_dir, args.output_srt), "w", encoding="utf-8") as f:
        # Write SRT manually to preserve empty-content entries
        # (srt.compose silently drops them)
        for i, sub in enumerate(adjusted_subs):
            f.write(f"{i + 1}\n")
            f.write(f"{srt.timedelta_to_srt_timestamp(sub.start)} --> {srt.timedelta_to_srt_timestamp(sub.end)}\n")
            f.write(sub.content + "\n\n")

    with open(os.path.join(args.output_dir, args.changed_json), "w") as f:
        json.dump(changed_segments, f)

    print(f"Generated {len(adjusted_subs)} audio segments in {args.output_dir}")
    print(f"Adjusted SRT: {args.output_srt}")
    print(f"Combined WAV: {args.output_wav} (duration: {total_duration:.2f}s)")
    print(f"Changed segments: {[s['segment'] for s in changed_segments]}")
    for s in changed_segments:
        print(
            f"  Segment {s['segment']} at {s['time']:.2f}s: {s['orig_duration']:.2f}s -> {s['new_duration']:.2f}s (diff: {s['diff']:+.2f}s)"
        )


if __name__ == "__main__":
    main()
