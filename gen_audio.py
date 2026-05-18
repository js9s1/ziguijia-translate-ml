import argparse
import io
import json
import os
import re
import sys
import time
from datetime import timedelta

import srt
import torch
import torchaudio as ta

sys.path.append(os.path.join(os.path.dirname(__file__), "chatterbox-server"))

from config import ASSETS_DIR as CFG_ASSETS_DIR, AUDIO_PROMPT_PATH as CFG_AUDIO_PROMPT_PATH

CHANGED_THRESHOLD = 0.05

ASSETS_DIR = CFG_ASSETS_DIR


def extract_speaker(content):
    m = re.match(r"^\s*(\w+)\s*:\s*(.*)", content, re.DOTALL)
    if m:
        return m.group(1).lower(), m.group(2)
    return None, content


def get_speaker_prompt(speaker, default_prompt=None, assets_dir=ASSETS_DIR):
    """Look up {speaker}_voice.wav in assets_dir. Returns path if found, else default_prompt."""
    if speaker:
        prompt_path = os.path.join(assets_dir, f"{speaker}_voice.wav")
        if os.path.exists(prompt_path):
            print(f"  Using speaker prompt: {prompt_path}")
            return prompt_path
    return default_prompt


def split_text(text, max_len=500):
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
    with open(srt_path, "r", encoding="utf-8") as f:
        return list(srt.parse(f.read()))


def generate_silence(duration_sec, sample_rate):
    num_frames = int(duration_sec * sample_rate)
    return torch.zeros(1, num_frames)


def combine_audio_segments(segments_info, total_duration, sample_rate):
    max_end = 0
    for seg in segments_info:
        end_sample = int((seg["new_start"] + seg["wav_duration"]) * sample_rate)
        if end_sample > max_end:
            max_end = end_sample

    needed_duration = max_end / sample_rate
    actual_duration = max(total_duration, needed_duration)

    if actual_duration <= 0:
        actual_duration = 1

    combined = generate_silence(actual_duration, sample_rate)
    for seg in segments_info:
        wav_data = seg["wav_data"]
        if wav_data.dim() == 1:
            wav_data = wav_data.unsqueeze(0)
        start_sample = int(seg["new_start"] * sample_rate)
        end_sample = start_sample + wav_data.shape[1]
        if end_sample > combined.shape[1]:
            new_combined = torch.zeros(1, end_sample)
            new_combined[:, :combined.shape[1]] = combined
            combined = new_combined
        combined[:, start_sample:end_sample] = wav_data
    return combined


def save_audio(output_path, wav_tensor, sample_rate):
    ta.save(output_path, wav_tensor, sample_rate)


def process_with_direct(srt_path, audio_prompt, temperature, output_dir, assets_dir=ASSETS_DIR, target_language="en", cfg_weight=0.5, exaggeration=0.5):
    from audio_utils import NingAudio

    audio = NingAudio(audio_prompt=audio_prompt)
    audio.setup()
    sample_rate = audio.sample_rate

    subs = load_subs(srt_path)
    if not subs:
        print("No subtitles found in SRT file")
        return

    adjusted_subs = []
    segments_info = []
    changed_segments = []
    accumulated_offset = 0.0

    seg_counter = 0
    for i, sub in enumerate(subs):
        speaker, clean_content = extract_speaker(sub.content)
        display = f"[{speaker}] " if speaker else ""
        chunks = split_text(clean_content, 220)
        print(f"Processing segment {i}: {display}{clean_content[:50]}... ({len(chunks)} chunk(s))")

        orig_start = sub.start.total_seconds()
        orig_duration = (sub.end - sub.start).total_seconds()

        prompt = get_speaker_prompt(speaker, audio_prompt, assets_dir)

        chunk_wavs = []
        total_wav_duration = 0.0
        for j, chunk in enumerate(chunks):
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
                "wav_data": combined_wav,
                "wav_duration": total_wav_duration,
                "new_start": new_start,
            }
        )

        accumulated_offset += total_wav_duration - orig_duration

    original_total = (subs[-1].end - subs[0].start).total_seconds()
    total_duration = original_total + accumulated_offset

    if original_total > 0 and total_duration / original_total > 3.0:
        print(f"ERROR: Generated audio duration ({total_duration:.2f}s) is more than 3x the original SRT duration ({original_total:.2f}s). TTS model likely failed.")
        raise RuntimeError(f"TTS duration inflation: {total_duration:.2f}s vs original {original_total:.2f}s (ratio: {total_duration/original_total:.1f}x)")

    combined_tensor = combine_audio_segments(
        segments_info, total_duration, sample_rate
    )

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
        default=ASSETS_DIR,
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

    print("Using direct NingAudio (in-process)")
    result = process_with_direct(args.srt, args.audio_prompt, args.temperature, args.output_dir, args.assets_dir,
                                 target_language=args.target_language, cfg_weight=args.cfg_weight, exaggeration=args.exaggeration)

    if result is None:
        return

    combined_tensor, adjusted_subs, changed_segments, sample_rate, total_duration = result

    save_audio(os.path.join(args.output_dir, args.output_wav), combined_tensor, sample_rate)

    with open(os.path.join(args.output_dir, args.output_srt), "w", encoding="utf-8") as f:
        f.write(srt.compose(adjusted_subs))

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