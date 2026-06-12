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
        # Every entry maps a source time-range to the output with a stretch factor.
        # We use *adjusted* start/end for placement so that cumulated time stays
        # synchronised with the audio WAV.

        # Leading gap: from video start to the first subtitle's *adjusted* start.
        if segments:
            first_adj_start = segments[0]["adj_start"]
            first_orig_start = segments[0]["orig_start"]
            if first_adj_start > 0 and first_orig_start > 0:
                stretch = first_adj_start / first_orig_start
                segment_data.append({
                    "start": 0,
                    "end": first_orig_start,
                    "stretch": stretch,
                    "is_subtitle": False
                })

        # Subtitle segments with inter-segment gaps
        for i, seg in enumerate(segments):
            orig_start = seg["orig_start"]
            orig_end   = seg["orig_end"]
            adj_start  = seg["adj_start"]
            adj_end    = seg["adj_end"]
            orig_dur   = seg["orig_duration"]
            adj_dur    = seg["adj_duration"]

            # Gap before this subtitle (adjusted timing)
            if i > 0:
                prev_orig_end  = segments[i-1]["orig_end"]
                prev_adj_end   = segments[i-1]["adj_end"]
                orig_gap = orig_start - prev_orig_end
                adj_gap  = adj_start - prev_adj_end

                # Determine source time-range for the gap.
                # Normally the original video already has a gap between
                # subtitles.  When the original segments are back-to-back
                # but the adjusted audio inserts a pause, we steal the
                # last frame of the previous segment as a freeze-frame
                # and stretch it to fill the pause.
                if orig_gap > 0.001:
                    gap_start = prev_orig_end
                    gap_end   = orig_start
                elif adj_gap > 0.001:
                    # No original gap — use the tail 1 frame of the prev segment.
                    gap_start = max(0, prev_orig_end - 0.04)
                    gap_end   = prev_orig_end
                    orig_gap  = gap_end - gap_start
                else:
                    orig_gap = 0.0  # skip this gap

                if orig_gap > 0:
                    stretch = adj_gap / orig_gap
                    stretch = max(0.0, min(100.0, stretch))
                    segment_data.append({
                        "start": gap_start,
                        "end": gap_end,
                        "stretch": stretch,
                        "is_subtitle": False
                    })

            # Subtitle segment: stretch = adj_dur / orig_dur.
            # We allow stretch < 1.0 (slight speed-up) so the video stays
            # locked to the adjusted audio timing.  Extreme values are
            # clamped to avoid obvious artifacts.
            stretch = adj_dur / orig_dur if orig_dur > 0 else 1.0
            stretch = max(0.1, min(10.0, stretch))
            segment_data.append({
                "start": orig_start,
                "end": orig_end,
                "stretch": stretch,
                "is_subtitle": True
            })

        # Trailing gap: from last subtitle's adjusted end to end of video.
        if segments:
            last_orig_end = segments[-1]["orig_end"]
            last_adj_end  = segments[-1]["adj_end"]
            if last_orig_end < video_duration:
                # Keep trailing gap at 1× unless we have adjusted timing past
                # the last subtitle (rare — usually the trailing gap is credits / silence).
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
        if segments:
            expected_end = segments[-1]["adj_end"]
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
            blur_cmd = [
                "ffmpeg", "-y",
                "-i", output_file,
                "-i", audio_wav,
                "-vf", "format=nv12,hwupload,delogo=x=100:y=600:w=1060:h=80:show=0",
                "-vaapi_device", VAAPI_DEVICE,
                "-c:v", "hevc_vaapi", "-qp", "23",
                "-c:a", "aac", "-b:a", "192k",
                "-map", "0:v", "-map", "1:a",
            ]
            if audio_duration:
                blur_cmd.extend(["-t", str(audio_duration)])
            else:
                blur_cmd.append("-shortest")
            subprocess.run(blur_cmd, check=True)
        else:
            mux_cmd = [
                "ffmpeg", "-y",
                "-i", output_file,
                "-i", audio_wav,
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-map", "0:v", "-map", "1:a",
            ]
            if audio_duration:
                mux_cmd.extend(["-t", str(audio_duration)])
            else:
                mux_cmd.append("-shortest")
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
