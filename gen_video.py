#!/usr/bin/env python3
"""
Modify video track to match adjusted audio timing.

Reads the original SRT and adjusted SRT (from gen_audio.py) along with
changed_segments.json, then stretches the video track segments where
audio duration differs from original.

Usage:
    python gen_video.py <video_file> <original_srt> <adjusted_srt> <changed_json> [--output OUTPUT]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import shutil
from datetime import timedelta

import srt


def parse_srt_time(time_str):
    """Parse SRT time format (HH:MM:SS,mmm) to timedelta."""
    parts = time_str.replace(',', '.').split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)


def format_srt_time(td):
    """Format timedelta to SRT time format (HH:MM:SS,mmm)."""
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def get_video_info(video_file):
    """Get video duration using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", video_file,
        ],
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    duration = float(data["format"]["duration"])
    width = height = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            width = stream.get("width")
            height = stream.get("height")
            break
    return {"duration": duration, "width": width, "height": height}


def calculate_segments(original_subs, adjusted_subs, changed_indices):
    """
    Calculate which video segments need stretching and by how much.

    Returns list of dicts with:
        - index: segment index
        - orig_start, orig_end: original timing
        - adj_start, adj_end: adjusted timing (from gen_audio output)
        - stretch_factor: how much to stretch video
        - is_changed: whether this segment is in changed_indices
    """
    segments = []

    for i in range(len(original_subs)):
        orig = original_subs[i]
        adj = adjusted_subs[i]

        orig_duration = (orig.end - orig.start).total_seconds()
        adj_duration = (adj.end - adj.start).total_seconds()

        # Calculate stretch factor (adj_duration / orig_duration)
        if orig_duration > 0:
            stretch_factor = adj_duration / orig_duration
        else:
            stretch_factor = 1.0

        segments.append({
            "index": i,
            "orig_start": orig.start.total_seconds(),
            "orig_end": orig.end.total_seconds(),
            "orig_duration": orig_duration,
            "adj_start": adj.start.total_seconds(),
            "adj_end": adj.end.total_seconds(),
            "adj_duration": adj_duration,
            "stretch_factor": stretch_factor,
            "is_changed": i in changed_indices,
        })

    return segments


def process_video(video_file, segments, output_file):
    """Process video by cutting, stretching, and concatenating segments."""
    temp_dir = tempfile.mkdtemp(prefix="gen_video_")

    try:
        video_info = get_video_info(video_file)
        video_duration = video_info["duration"]

        segment_files = []
        segment_data = []

        # Build segment list: leading gap, subtitle segments, gaps, trailing
        cumulative_time = 0.0

        # Leading segment (0 to first subtitle start)
        if segments:
            first_sub_start = segments[0]["orig_start"]
            if first_sub_start > 0:
                segment_data.append({
                    "start": 0,
                    "end": first_sub_start,
                    "stretch": 1.0
                })
                cumulative_time += first_sub_start

        # Subtitle segments with gaps
        for i, seg in enumerate(segments):
            start = seg["orig_start"]
            end = seg["orig_end"]
            stretch = seg["stretch_factor"]
            is_changed = seg["is_changed"]

            # Gap before this subtitle
            if i > 0:
                prev_end = segments[i-1]["orig_end"]
                if start > prev_end:
                    segment_data.append({
                        "start": prev_end,
                        "end": start,
                        "stretch": 1.0
                    })
                    cumulative_time += (start - prev_end)

            # This subtitle segment
            seg_duration = end - start
            actual_stretch = stretch if is_changed else 1.0
            segment_data.append({
                "start": start,
                "end": end,
                "stretch": actual_stretch
            })
            cumulative_time += seg_duration * actual_stretch

        # Trailing segment
        if segments:
            last_sub_end = segments[-1]["orig_end"]
            if last_sub_end < video_duration:
                segment_data.append({
                    "start": last_sub_end,
                    "end": video_duration,
                    "stretch": 1.0
                })

        # Process each segment - apply stretch filter and reset timestamps
        for i, seg in enumerate(segment_data):
            output_seg = os.path.join(temp_dir, f"seg_{i:04d}.mp4")

            if seg["stretch"] != 1.0:
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(seg["start"]),
                    "-to", str(seg["end"]),
                    "-i", video_file,
                    "-vf", f"setpts={seg['stretch']}*PTS,setpts=PTS-STARTPTS",
                    "-c:v", "libx265", "-crf", "23", "-preset", "fast",
                    "-an",
                    "-r", "24",
                    output_seg
                ]
            else:
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(seg["start"]),
                    "-to", str(seg["end"]),
                    "-i", video_file,
                    "-vf", "setpts=PTS-STARTPTS",
                    "-c:v", "libx265", "-crf", "23", "-preset", "fast",
                    "-an",
                    "-r", "24",
                    output_seg
                ]

            subprocess.run(cmd, check=True, capture_output=True)
            segment_files.append(output_seg)

        # Concatenate with concat filter
        if len(segment_files) == 1:
            shutil.copy(segment_files[0], output_file)
        else:
            filter_str = "".join([f"[{i}:v]" for i in range(len(segment_files))]) + \
                        f"concat=n={len(segment_files)}:v=1:a=0[outv]"

            cmd = ["ffmpeg", "-y"]
            for f in segment_files:
                cmd.extend(["-i", f])
            cmd.extend([
                "-filter_complex", filter_str,
                "-map", "[outv]",
                "-c:v", "libx265", "-crf", "23", "-preset", "fast",
                "-r", "24",
                output_file
            ])

            subprocess.run(cmd, check=True, capture_output=True)

        print(f"Modified video saved to: {output_file}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Modify video track to match adjusted audio timing"
    )
    parser.add_argument(
        "video_file",
        help="Input video file (decompressed mp4)",
    )
    parser.add_argument(
        "original_srt",
        help="Original SRT file (input to gen_audio)",
    )
    parser.add_argument(
        "adjusted_srt",
        help="Adjusted SRT file (output from gen_audio)",
    )
    parser.add_argument(
        "changed_json",
        help="JSON file listing changed segment indices",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output video file (default: input_video_modified.mp4)",
    )
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.video_file):
        print(f"Error: Video file not found: {args.video_file}")
        sys.exit(1)

    if not os.path.exists(args.original_srt):
        print(f"Error: Original SRT file not found: {args.original_srt}")
        sys.exit(1)

    if not os.path.exists(args.adjusted_srt):
        print(f"Error: Adjusted SRT file not found: {args.adjusted_srt}")
        sys.exit(1)

    if not os.path.exists(args.changed_json):
        print(f"Error: Changed segments JSON not found: {args.changed_json}")
        sys.exit(1)

    output_file = args.output or args.video_file.replace(".mp4", "_modified.mp4")

    # Load SRT files
    with open(args.original_srt, "r", encoding="utf-8") as f:
        original_subs = list(srt.parse(f.read()))

    with open(args.adjusted_srt, "r", encoding="utf-8") as f:
        adjusted_subs = list(srt.parse(f.read()))

    with open(args.changed_json, "r") as f:
        changed_data = json.load(f)
        if isinstance(changed_data, list) and len(changed_data) > 0:
            if isinstance(changed_data[0], dict):
                changed_indices = set(item["segment"] for item in changed_data)
            else:
                changed_indices = set(changed_data)
        else:
            changed_indices = set()

    # Get video info
    video_info = get_video_info(args.video_file)
    print(f"Video duration: {video_info['duration']:.2f}s")
    print(f"Total segments: {len(original_subs)}")
    print(f"Changed segments: {len(changed_indices)}")

    # Calculate segment stretch factors
    segments = calculate_segments(original_subs, adjusted_subs, changed_indices)

    # Show changed segments with their stretch factors
    print("\nChanged segments:")
    for seg in segments:
        if seg["is_changed"]:
            print(f"  Segment {seg['index']}: "
                  f"orig_dur={seg['orig_duration']:.3f}s, "
                  f"adj_dur={seg['adj_duration']:.3f}s, "
                  f"stretch={seg['stretch_factor']:.3f}x")

    # Process video
    process_video(args.video_file, segments, output_file)

    # Final step: add audio and SRT subtitles to the modified video
    audio_wav = os.path.join(os.path.dirname(args.adjusted_srt), "output.wav")
    final_output = output_file.replace("_modified.mp4", "_final.mp4")
    temp_output = output_file.replace("_modified.mp4", "_temp.mp4")

    subprocess.run([
        "ffmpeg", "-y",
        "-i", output_file,
        "-i", audio_wav,
        "-c:v", "libx265", "-crf", "23", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v", "-map", "1:a",
        temp_output
    ], check=True)

    subprocess.run([
        "ffmpeg", "-y",
        "-i", temp_output,
        "-i", args.adjusted_srt,
        "-c", "copy",
        "-c:s", "mov_text",
        "-metadata:s:s:0", "language=eng",
        final_output
    ], check=True)

    os.remove(temp_output)

    print(f"Final video with audio and subtitles saved to: {final_output}")


if __name__ == "__main__":
    main()