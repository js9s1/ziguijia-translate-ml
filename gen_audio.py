import argparse
import json
import os
import re
import signal
import sys
import time
from datetime import timedelta
from pathlib import Path

import srt
import torch
import soundfile as sf

sys.path.append(os.path.join(os.path.dirname(__file__), "chatterbox-server"))

from config import ASSETS_DIR as CFG_ASSETS_DIR, AUDIO_PROMPT_PATH as CFG_AUDIO_PROMPT_PATH
from video_util import read_srt_text

CHANGED_THRESHOLD = 0.05

# Thermal protection — Renoir iGPU (gfx90c via ROCm) is prone to GPU hangs
# under sustained load.  Pause between segments when the GPU gets too hot.
GPU_TEMP_LIMIT = float(os.environ.get("GPU_TEMP_LIMIT", "80"))   # °C — pause if exceeded
GPU_COOLDOWN_TARGET = float(os.environ.get("GPU_COOLDOWN_TARGET", "60"))  # °C — resume when below
GPU_POLL_SECS = float(os.environ.get("GPU_POLL_SECS", "10"))
_STOP_REQUESTED = False


def _get_gpu_temp():
    """Read AMD GPU (amdgpu) temperature in °C via hwmon. Returns float or None."""
    try:
        for card in Path("/sys/class/drm").glob("card*"):
            hwmons = list(card.glob("device/hwmon/hwmon*"))
            for hw in hwmons:
                try:
                    name = (hw / "name").read_text().strip()
                    if name == "amdgpu":
                        raw = int((hw / "temp1_input").read_text().strip())
                        return raw / 1000.0
                except Exception:
                    continue
    except Exception:
        pass
    return None


def _signal_handler(signum, frame):
    global _STOP_REQUESTED
    sig_name = signal.Signals(signum).name
    print(f"\n\n  ⏸ {sig_name} received — will stop after current segment. Press again to force-quit.",
          file=sys.stderr)
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
                print(f"\r  Cooling… GPU {temp:.0f}°C (target ≤{GPU_COOLDOWN_TARGET:.0f}°C)",
                      end="", flush=True)
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
        with open(path, "r", encoding="utf-8") as f:
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
    """
    wav_path = _combined_seg_path(output_dir, seg_idx)
    if not os.path.exists(wav_path):
        return False, 0.0
    seg_cache = cache.get(str(seg_idx))
    if seg_cache:
        if seg_cache.get("content") == clean_content:
            return True, seg_cache.get("duration", 0.0)
        return False, 0.0
    # WAV exists but no cache entry — recover orphaned WAV from crash
    try:
        info = sf.info(wav_path)
        _set_cache(cache, seg_idx, clean_content, info.duration)
        _save_cache(output_dir, cache)
        return True, info.duration
    except Exception:
        return False, 0.0


def _set_cache(cache, seg_idx, clean_content, wav_duration):
    cache[str(seg_idx)] = {
        "content": clean_content,
        "duration": wav_duration,
    }


def _migrate_cache(cache, new_subs, output_dir):
    """Remap cache entries when SRT segment indices shift.

    For each segment in *new_subs*, match by content against the old
    *cache*.  If found at a different index, rename the corresponding
    WAV file and move the cache entry.

    Old entries with no match in the new SRT are discarded.
    New segments with no match are left empty for regeneration.
    """
    if not cache:
        return

    # Build content → (old_idx, duration) from old cache
    old_map = {}
    for old_idx_str, entry in list(cache.items()):
        old_idx = int(old_idx_str)
        content = entry.get("content", "")
        if content:
            old_map[content] = (old_idx, entry.get("duration", 0.0))

    # Match each new segment against old cache
    for i, sub in enumerate(new_subs):
        _, clean_content = extract_speaker(sub.content)
        if not clean_content.strip():
            continue

        if clean_content in old_map:
            old_idx, dur = old_map.pop(clean_content)
            if old_idx != i:
                old_wav = _combined_seg_path(output_dir, old_idx)
                new_wav = _combined_seg_path(output_dir, i)
                if os.path.exists(old_wav) and not os.path.exists(new_wav):
                    os.rename(old_wav, new_wav)
                    print(f"  ↪ cache: moved seg {old_idx} → seg {i} (same content)")
                elif os.path.exists(new_wav):
                    os.remove(old_wav)
            cache[str(i)] = {"content": clean_content, "duration": dur}
        else:
            # No match — clear so it gets regenerated
            cache.pop(str(i), None)

    # Remove stale entries
    for (old_idx, _) in old_map.values():
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
        for sep in ('. ', '! ', '? ', '.\n', '!\n', '?\n', '.\r\n', '!\r\n', '?\r\n'):
            pos = chunk.rfind(sep)
            if pos > last_boundary:
                last_boundary = pos + 1

        if last_boundary > 0:
            chunks.append(text[:last_boundary])
            text = text[last_boundary:].lstrip()
        else:
            last_space = chunk.rfind(' ')
            if last_space > 0:
                chunks.append(text[:last_space])
                text = text[last_space:].lstrip()
            else:
                chunks.append(chunk)
                text = text[max_len:].lstrip()

    return chunks


def load_subs(srt_path):
    """Parse SRT file preserving empty-content entries (unlike srt.parse which drops them)."""
    content = read_srt_text(srt_path)
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
    return subs


def generate_silence(duration_sec, sample_rate):
    num_frames = round(duration_sec * sample_rate)
    return torch.zeros(1, num_frames)


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
            new_combined[:, :combined.shape[1]] = combined
            combined = new_combined
        combined[:, start_sample:end_sample] = wav_data
    return combined


def save_audio(output_path, wav_tensor, sample_rate):
    sf.write(output_path, wav_tensor.squeeze(0).cpu().numpy(), sample_rate)


def process_with_direct(srt_path, audio_prompt, temperature, output_dir, assets_dir=CFG_ASSETS_DIR, target_language="en", cfg_weight=0.5, exaggeration=0.5):
    subs = load_subs(srt_path)
    if not subs:
        print("No subtitles found in SRT file")
        return

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
        cached_wavs = [_combined_seg_path(output_dir, i) for i in range(len(subs))
                       if os.path.exists(_combined_seg_path(output_dir, i))]
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
                changed_segments.append({
                    "segment": i,
                    "time": orig_start,
                    "orig_duration": orig_duration,
                    "new_duration": total_wav_duration,
                    "diff": total_wav_duration - orig_duration,
                })

            adjusted_subs.append(
                srt.Subtitle(index=i, start=timedelta(seconds=new_start),
                             end=timedelta(seconds=new_start + total_wav_duration),
                             content=sub.content)
            )

            segments_info.append({
                "wav_path": seg_wav_path,
                "wav_duration": total_wav_duration,
                "new_start": new_start,
            })

            if total_wav_duration > orig_duration:
                accumulated_offset += total_wav_duration - orig_duration

        original_total = (subs[-1].end - subs[0].start).total_seconds()
        total_duration = original_total + accumulated_offset
        combined_tensor = combine_audio_segments(segments_info, total_duration, sample_rate)
        _save_cache(output_dir, cache)
        return combined_tensor, adjusted_subs, changed_segments, sample_rate, total_duration

    # ── Some segments need generation — init GPU model ──────
    from audio_utils import NingAudio
    import gpu_manage as _gm

    audio = NingAudio(audio_prompt=audio_prompt)
    audio._ensure_model(target_language)
    if target_language == "id":
        sample_rate = _gm._indonesian_model.sr
    else:
        sample_rate = audio.sample_rate

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
                srt.Subtitle(index=i, start=timedelta(seconds=new_start),
                             end=timedelta(seconds=new_start + orig_duration),
                             content=sub.content)
            )
            segments_info.append({
                "wav_path": seg_wav_path,
                "wav_duration": orig_duration,
                "new_start": new_start,
            })
            continue

        # ── Check per-segment cache ──
        cached, cached_duration = _check_cache(
            cache, i, clean_content, output_dir)

        if cached:
            print(f"  ↪ segment {i} cached (duration: {cached_duration:.2f}s), skipping generation")
            total_wav_duration = cached_duration
            # Advance seg_counter past chunk slots this segment would use
            seg_counter += len(chunks)
        else:
            # ── GPU thermal check before heavy work ──
            if not _thermal_check():
                print(f"  → Stopped by signal")
                raise SystemExit(1)

            prompt = get_speaker_prompt(speaker, audio_prompt, assets_dir)

            chunk_wavs = []
            total_wav_duration = 0.0
            for chunk in chunks:
                wav_path = os.path.join(output_dir, "tmp", f"segment_{seg_counter}.wav")
                wav_data, wav_duration = audio.generate_audio(
                    chunk, wav_path, sample_rate, temperature, prompt_file=prompt,
                    target_language=target_language, cfg_weight=cfg_weight, exaggeration=exaggeration
                )
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

            # Update in-memory cache and persist immediately
            _set_cache(cache, i, clean_content, total_wav_duration)
            _save_cache(output_dir, cache)

        new_start = orig_start + accumulated_offset
        duration_diff = total_wav_duration - orig_duration

        if abs(duration_diff) > CHANGED_THRESHOLD:
            changed_segments.append({
                "segment": i,
                "time": orig_start,
                "orig_duration": orig_duration,
                "new_duration": total_wav_duration,
                "diff": duration_diff
            })

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
        print(f"ERROR: Generated audio duration ({total_duration:.2f}s) is more than 3x the original SRT duration ({original_total:.2f}s). TTS model likely failed.")
        raise RuntimeError(f"TTS duration inflation: {total_duration:.2f}s vs original {original_total:.2f}s (ratio: {total_duration/original_total:.1f}x)")

    combined_tensor = combine_audio_segments(
        segments_info, total_duration, sample_rate
    )

    # Persist cache to disk so the next run can benefit from it
    _save_cache(output_dir, cache)

    return combined_tensor, adjusted_subs, changed_segments, sample_rate, total_duration


def main():
    parser = argparse.ArgumentParser(
        description="Generate audio from SRT segments and output adjusted SRT"
    )
    parser.add_argument("srt", help="Input SRT file")
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
    parser.add_argument("--temperature", type=float, default=0.8,
        help="Temperature for audio generation")
    parser.add_argument("--target_language", type=str, default="en",
        help="Target language code (e.g. en, zh, ja, etc.)")
    parser.add_argument("--cfg_weight", type=float, default=0.5,
        help="CFG weight for generation")
    parser.add_argument("--exaggeration", type=float, default=0.5,
        help="Exaggeration level for voice characteristics")
    parser.add_argument("--output_dir", default="./output", help="Output directory")
    parser.add_argument("--output_srt", default="output_adjusted.srt")
    parser.add_argument("--output_wav", default="output.wav")
    parser.add_argument("--changed_json", default="changed_segments.json")
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
        print(f"GPU temp:   {gpu_temp:.0f}°C (limit: {GPU_TEMP_LIMIT:.0f}°C, cooldown target: {GPU_COOLDOWN_TARGET:.0f}°C)")
    print(f"Stop:       Ctrl+C once for graceful stop per segment, twice to force-quit")
    print(f"Env:        GPU_TEMP_LIMIT={GPU_TEMP_LIMIT}  GPU_COOLDOWN_TARGET={GPU_COOLDOWN_TARGET}  GPU_POLL_SECS={GPU_POLL_SECS}")
    print()

    print("Using direct NingAudio (in-process)")
    result = process_with_direct(args.srt, args.audio_prompt, args.temperature, args.output_dir, args.assets_dir,
                                 target_language=args.target_language, cfg_weight=args.cfg_weight, exaggeration=args.exaggeration)

    if result is None:
        return

    combined_tensor, adjusted_subs, changed_segments, sample_rate, total_duration = result

    save_audio(os.path.join(args.output_dir, args.output_wav), combined_tensor, sample_rate)

    with open(os.path.join(args.output_dir, args.output_srt), "w", encoding="utf-8") as f:
        # Write SRT manually to preserve empty-content entries
        # (srt.compose silently drops them)
        for i, sub in enumerate(adjusted_subs):
            f.write(f"{i+1}\n")
            f.write(f"{srt.timedelta_to_srt_timestamp(sub.start)} --> {srt.timedelta_to_srt_timestamp(sub.end)}\n")
            f.write(sub.content + "\n\n")

    with open(os.path.join(args.output_dir, args.changed_json), "w") as f:
        json.dump(changed_segments, f)

    print(f"Generated {len(adjusted_subs)} audio segments in {args.output_dir}")
    print(f"Adjusted SRT: {args.output_srt}")
    print(f"Combined WAV: {args.output_wav} (duration: {total_duration:.2f}s)")
    print(f"Changed segments: {[s['segment'] for s in changed_segments]}")
    for s in changed_segments:
        print(f"  Segment {s['segment']} at {s['time']:.2f}s: {s['orig_duration']:.2f}s -> {s['new_duration']:.2f}s (diff: {s['diff']:+.2f}s)")


if __name__ == "__main__":
    main()