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

import srt

sys.path.append(os.path.join(os.path.dirname(__file__), "chatterbox-server"))
from video_util import read_srt_text

VAAPI_DEVICE = "/dev/dri/renderD128"


def get_video_info(video_file):
    """Get video duration using ffprobe."""
    # Reject empty files early with a clear error
    file_size = os.path.getsize(video_file)
    if file_size == 0:
        raise ValueError(f"Video file is empty (0 bytes): {video_file}")
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", video_file,
        ],
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    if "format" not in data:
        raise ValueError(f"Cannot read video file (ffprobe returned no format info): {video_file}")
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

        # Calculate stretch factor (adj_duration / orig_duration).
        # Never squeeze — only stretch to accommodate longer audio.
        if orig_duration > 0:
            stretch_factor = max(1.0, adj_duration / orig_duration)
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
    """Process video by cutting, stretching, and concatenating segments.

    Each subtitle segment's video source is the original timing (*orig_start* … *orig_end*),
    stretched by *adj_duration / orig_duration* so the video length matches the adjusted audio.
    Gaps between subtitles use the adjusted timing so that segment *i* is placed at
    :math:`adj\\_start[i]` in the output — this eliminates systematic drift over hundreds of
    segments without needing a post-hoc correction pass.
    """
    temp_dir = tempfile.mkdtemp(prefix="gen_video_")

    try:
        video_info = get_video_info(video_file)
        video_duration = video_info["duration"]

        segment_data = []

        # ── Build segment list ──────────────────────────────
        # Track the output position explicitly so that gap stretching
        # absorbs any drift introduced by the max(1.0, …) clamp on
        # subtitle segments.
        video_cumul = 0.0

        # ── Filter segments past video end ──────────────────
        # SRT timestamps may extend beyond the actual video duration
        # (e.g. user uploads a 4:30 SRT for a 2:46 video).
        # Clamp / discard segments accordingly instead of crashing ffmpeg.
        valid_segments = []
        for seg in segments:
            if seg["orig_start"] >= video_duration:
                break  # past end — stop here
            if seg["orig_end"] > video_duration:
                # Clamp the segment to video end
                excess = seg["orig_end"] - video_duration
                seg = dict(seg)
                seg["orig_end"] = video_duration
                seg["orig_duration"] = seg["orig_end"] - seg["orig_start"]
                seg["adj_end"] = max(seg["adj_start"],
                    seg["adj_end"] - excess * seg.get("stretch_factor", 1.0))
                seg["adj_duration"] = seg["adj_end"] - seg["adj_start"]
            valid_segments.append(seg)

        # ── Leading gap ────────────────────────────────────
        # Leading gap: from video start to the first subtitle's *adjusted* start.
        if valid_segments:
            first_adj_start = valid_segments[0]["adj_start"]
            first_orig_start = valid_segments[0]["orig_start"]
            if first_adj_start > 0 and first_orig_start > 0:
                stretch = first_adj_start / first_orig_start
                segment_data.append({
                    "start": 0,
                    "end": min(first_orig_start, video_duration),
                    "stretch": stretch,
                    "is_subtitle": False
                })
                video_cumul += first_orig_start * stretch

        # ── Subtitle segments with inter-segment gaps ──────
        for i, seg in enumerate(valid_segments):
            orig_start = seg["orig_start"]
            orig_end   = seg["orig_end"]
            adj_start  = seg["adj_start"]
            adj_end    = seg["adj_end"]
            orig_dur   = seg["orig_duration"]
            adj_dur    = seg["adj_duration"]

            # Gap before this subtitle.
            if i > 0:
                prev_orig_end  = valid_segments[i-1]["orig_end"]
                orig_gap = orig_start - prev_orig_end
                needed_gap = adj_start - video_cumul

                if orig_gap > 0.001:
                    gap_start = prev_orig_end
                    gap_end   = min(orig_start, video_duration)
                elif needed_gap > 0.001:
                    gap_start = max(0, prev_orig_end - 0.04)
                    gap_end   = min(prev_orig_end, video_duration)
                    orig_gap  = gap_end - gap_start
                else:
                    orig_gap = 0.0  # skip this gap

                if orig_gap > 0:
                    stretch = needed_gap / orig_gap
                    stretch = max(0.0, min(100.0, stretch))
                    segment_data.append({
                        "start": gap_start,
                        "end": gap_end,
                        "stretch": stretch,
                        "is_subtitle": False
                    })
                    video_cumul += orig_gap * stretch

            # Subtitle segment.
            stretch = adj_dur / orig_dur if orig_dur > 0 else 1.0
            stretch = max(1.0, min(10.0, stretch))
            segment_data.append({
                "start": orig_start,
                "end": min(orig_end, video_duration),
                "stretch": stretch,
                "is_subtitle": True
            })
            video_cumul += orig_dur * stretch

        # ── Trailing gap ───────────────────────────────────
        if valid_segments:
            last_orig_end = valid_segments[-1]["orig_end"]
            if last_orig_end < video_duration:
                segment_data.append({
                    "start": last_orig_end,
                    "end": video_duration,
                    "stretch": 1.0,
                    "is_subtitle": False
                })

        # ── Validate total matches audio ────────────────────
        total_duration = sum(
            (s["end"] - s["start"]) * s["stretch"] for s in segment_data
        )
        if valid_segments:
            expected_end = valid_segments[-1]["adj_end"]
            drift = total_duration - expected_end
            if abs(drift) > 1.0:
                print(f"Warning: predicted video length {total_duration:.2f}s "
                      f"differs from adjusted audio end {expected_end:.2f}s "
                      f"by {drift:+.2f}s")
            if abs(drift) <= 1.0 and abs(drift) > 0.01:
                print(f"Timing check: video={total_duration:.2f}s  "
                      f"audio-end={expected_end:.2f}s  drift={drift:+.3f}s")

        # ── Build filter-graph batches ──────────────────────
        # Instead of one ffmpeg process per segment (1215× VAAPI init),
        # we pack many segments into a single filter_complex.  Each batch
        # feeds the source video once per segment with -ss/-to for
        # keyframe-seeking, then trim+setpts → concat → hevc_vaapi encode.
        # Result: ~29 ffmpeg processes instead of ~1215, with one VAAPI
        # init per batch instead of one per segment.
        MAX_BATCH_SEGMENTS = 42  # -ss/-to pairs + filter_complex ≈ 180 args, well under OS limits

        batch_files = []
        for batch_idx in range(0, len(segment_data), MAX_BATCH_SEGMENTS):
            batch = segment_data[batch_idx:batch_idx + MAX_BATCH_SEGMENTS]
            batch_out = os.path.join(temp_dir, f"batch_{batch_idx:04d}.mp4")

            # Build the command
            cmd = ["ffmpeg", "-y", "-hwaccel_output_format", "nv12"]
            for seg in batch:
                cmd.extend(["-ss", str(seg["start"]), "-to", str(seg["end"]),
                            "-i", video_file])
            cmd.append("-vaapi_device")
            cmd.append(VAAPI_DEVICE)

            # Build filter_complex: setpts per input → concat → hwupload
            # -hwaccel_output_format nv12 ensures the decoder outputs software
            # nv12 frames that setpts can consume directly.
            filter_parts = []
            for i, seg in enumerate(batch):
                stretch = seg["stretch"]
                if stretch != 1.0:
                    filter_parts.append(
                        f"[{i}:v]setpts={stretch}*PTS,setpts=PTS-STARTPTS[v{i}]")
                else:
                    filter_parts.append(
                        f"[{i}:v]setpts=PTS-STARTPTS[v{i}]")
            n = len(batch)
            concat_labels = "".join(f"[v{i}]" for i in range(n))
            filter_parts.append(
                f"{concat_labels}concat=n={n}:v=1:a=0[outv]")
            filter_parts.append("[outv]format=nv12,hwupload[hw]")

            cmd.extend(["-filter_complex", ";".join(filter_parts)])
            cmd.extend(["-map", "[hw]", "-c:v", "hevc_vaapi", "-qp", "23",
                        "-r", "24", batch_out])

            print(f"  Batch {batch_idx:04d}: {n} segments → single ffmpeg")
            result = subprocess.run(cmd, check=False, capture_output=True, text=True,
                                     timeout=3600)
            if result.returncode != 0:
                print(f"FFmpeg batch error: {result.stderr[:50000]}",
                      file=sys.stderr)
                result.check_returncode()

            # Validate
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_streams", "-select_streams", "v", batch_out],
                capture_output=True, text=True,
            )
            probe_data = json.loads(probe.stdout) if probe.stdout.strip() else {"streams": []}
            if not probe_data.get("streams"):
                print(f"Warning: batch {batch_idx:04d} has no video stream, skipping")
                continue
            batch_files.append(batch_out)

        # ── Hierarchical concat of batch outputs ─────────────
        if len(batch_files) == 1:
            shutil.copy(batch_files[0], output_file)
        else:
            # Same MAX size, but now there are ~29 batches instead of 1215
            batch_list = batch_files
            level = 0
            while len(batch_list) > 1:
                level += 1
                next_batch = []
                for bi in range(0, len(batch_list), MAX_BATCH_SEGMENTS):
                    group = batch_list[bi:bi + MAX_BATCH_SEGMENTS]
                    if len(group) == 1:
                        next_batch.append(group[0])
                        continue
                    out_name = os.path.join(temp_dir, f"L{level}_batch{bi:04d}.mp4")
                    n = len(group)
                    concat_inputs = "".join(f"[{i}:v]" for i in range(n))
                    if n > 1:
                        filter_str = f"{concat_inputs}concat=n={n}:v=1:a=0[outv];[outv]format=nv12,hwupload[hw]"
                    else:
                        filter_str = f"[0:v]format=nv12,hwupload[hw]"
                    cmd = ["ffmpeg", "-y",
                           "-hwaccel_output_format", "nv12",
                           "-vaapi_device", VAAPI_DEVICE]
                    for sf in group:
                        cmd.extend(["-i", sf])
                    cmd.extend([
                        "-filter_complex", filter_str,
                        "-map", "[hw]",
                        "-c:v", "hevc_vaapi", "-qp", "23",
                        "-r", "24",
                        out_name,
                    ])
                    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
                    if result.returncode != 0:
                        print(f"FFmpeg error (L{level} batch {bi}): {result.stderr[:50000]}", file=sys.stderr)
                        result.check_returncode()
                    next_batch.append(out_name)
                batch_list = next_batch

            shutil.copy(batch_list[0], output_file)

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
    parser.add_argument(
        "--audio-wav",
        default=None,
        help="Audio WAV file (default: <adjusted_srt_dir>/output.wav)",
    )
    parser.add_argument(
        "--blur",
        action="store_true",
        default=False,
        help="Apply delogo filter to blur Chinese text before final mux",
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

    # Guard: input and output must not point to the same file — ffmpeg -y
    # opens the output for writing first, which truncates the input.
    if os.path.abspath(args.video_file) == os.path.abspath(output_file):
        _fallback = args.video_file + ".source"
        print(f"WARNING: input path == output path, renamed source to {_fallback}")
        os.rename(args.video_file, _fallback)
        # Re-point args.video_file to the renamed source
        args.video_file = _fallback

    # Validate inputs
    if not os.path.exists(args.video_file):
        print(f"Error: Video file not found: {args.video_file}")
        sys.exit(1)

    # Load SRT files
    original_subs = list(srt.parse(read_srt_text(args.original_srt)))

    adjusted_subs = list(srt.parse(read_srt_text(args.adjusted_srt)))

    if len(original_subs) != len(adjusted_subs):
        print(f"Error: Segment count mismatch: {len(original_subs)} original vs {len(adjusted_subs)} adjusted")
        print("The original and adjusted SRT files must have the same number of segments.")
        sys.exit(1)

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
    audio_wav = args.audio_wav or os.path.join(os.path.dirname(args.adjusted_srt), "output.wav")
    final_output = output_file.replace("_modified.mp4", "_final.mp4")
    temp_output = output_file.replace("_modified.mp4", "_temp.mp4")

    # Get exact audio duration so we can trim the video precisely.
    # This is a safety net: the per-segment adjusted timing should already
    # produce a video that matches the audio length, but ffmpeg frame
    # rounding in the encoder can add up to ~1 frame per batch.
    try:
        audio_info = get_video_info(audio_wav)
        audio_duration = audio_info["duration"]
    except Exception:
        audio_duration = None

    try:
        if args.blur:
            print("Applying delogo filter to blur Chinese text...")
            # The delogo filter is software-only and doesn't play well
            # with ffmpeg's auto-inserted VAAPI scaler when hwupload
            # is in the graph.  Run the blur pass entirely on CPU
            # then encode with software x265 — simpler and reliable.
            blur_cmd = [
                "ffmpeg", "-y",
                "-i", output_file,
                "-i", audio_wav,
                "-vf", "hwdownload,format=yuv420p,delogo=x=100:y=600:w=1060:h=80:show=0",
                "-c:v", "libx265", "-crf", "23", "-preset", "fast",
                "-c:a", "aac", "-b:a", "192k",
            ]
            if audio_duration:
                blur_cmd.extend(["-t", str(audio_duration), temp_output])
            else:
                blur_cmd.extend(["-shortest", temp_output])
            subprocess.run(blur_cmd, check=True)
        else:
            mux_cmd = [
                "ffmpeg", "-y",
                "-i", output_file,
                "-i", audio_wav,
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-map", "0:v", "-map", "1:a",
                "-shortest",
                temp_output,
            ]
            subprocess.run(mux_cmd, check=True)

        subprocess.run([
            "ffmpeg", "-y",
            "-i", temp_output,
            "-i", args.adjusted_srt,
            "-c", "copy",
            "-c:s", "mov_text",
            "-metadata:s:s:0", "language=eng",
            final_output
        ], check=True)
    finally:
        if os.path.exists(temp_output):
            os.remove(temp_output)

    print(f"Final video with audio and subtitles saved to: {final_output}")


if __name__ == "__main__":
    main()
