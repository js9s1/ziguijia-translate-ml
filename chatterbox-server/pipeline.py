"""Shared subprocess pipeline steps for audio/video job handlers.

Checkpoint-aware wrappers
-------------------------
``run_audio_ckpt``, ``run_video_ckpt``, ``run_download_ckpt``, ``run_ocr_ckpt``,
``run_translate_ckpt``, ``run_extract_audio_ckpt``, and ``run_whisper_ckpt``
combine subprocess invocation with ``CheckpointHelper`` check-skip-mark cycles
so every job handler delegates to these instead of repeating boilerplate.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time

from config import (
    AUDIO_PROMPT_PATH,
    GEN_AUDIO_MIN_TOTAL_TIMEOUT,
    GEN_AUDIO_PYTHON,
    GEN_AUDIO_SEGMENT_BUDGET,
    GEN_AUDIO_STALL_TIMEOUT,
    GEN_VIDEO_SCRIPT,
    LANG_MAP,
    PROJECT_ROOT,
    PYTHON_BIN,
    RAPID_VIDEOCR_BIN,
    RAPID_VIDEOCR_PIPELINE_SCRIPT,
    TRANSLATE_PYTHON,
    WHISPER_MODEL,
    WHISPER_OV_DEVICE,
)
from jobqueue import get_job_queue
from log_utils import job_log, job_log_lines
from segment_utils import build_segment_defs

# ── Low-level subprocess wrappers ──────────────────────────


def run_gen_audio_step(
    srt_path: str,
    output_dir: str,
    temperature: float,
    access_code: str,
    audio_prompt: str = AUDIO_PROMPT_PATH,
    target_language: str = "en",
    cfg_weight: float = 0.5,
    exaggeration: float = 0.5,
    output_srt: str = "output_adjusted.srt",
    output_wav: str = "output.wav",
    changed_json: str = "changed_segments.json",
    stall_timeout: float = GEN_AUDIO_STALL_TIMEOUT,
    per_segment_budget: float = GEN_AUDIO_SEGMENT_BUDGET,
    min_total_timeout: float = GEN_AUDIO_MIN_TOTAL_TIMEOUT,
) -> dict[str, str]:
    """Generate audio from SRT (subprocess — dedicated GPU context).

    Runs gen_audio.py as a subprocess on its own Python interpreter so
    TTS keeps its GPU context separate from the server worker.

    A fixed wall-clock timeout does not fit a process whose runtime
    scales with job size (a 46-segment SRT takes ~7 min; a 1215-segment
    one takes hours).  Instead:

    - *stall_timeout*: gen_audio.py prints per-segment progress lines
      continuously (unbuffered).  If the log file stops growing for this
      long while the process is still alive, it is hung → terminate.
    - Total safety cap: ``max(min_total_timeout, per_segment_budget ×
      number of SRT segments)`` — only reached by a job that keeps
      producing output but never finishes.

    On timeout the subprocess gets SIGTERM first (gen_audio.py stops
    after the current segment and persists its per-segment WAV cache,
    so a resubmit resumes cheaply), then SIGKILL after a grace period.
    """
    os.makedirs(output_dir, exist_ok=True)

    srt_path_out = os.path.join(output_dir, output_srt)
    wav_path_out = os.path.join(output_dir, output_wav)
    changed_json_out = os.path.join(output_dir, changed_json)

    log_path = os.path.join(output_dir, "job.log")
    job_log(access_code, output_dir, "--- gen_audio ---")

    gen_audio_script = os.path.join(PROJECT_ROOT, "gen_audio", "gen_audio.py")
    assets_dir = os.path.join(PROJECT_ROOT, "..", "assets")

    cmd = [
        GEN_AUDIO_PYTHON,
        "-u",
        gen_audio_script,
        srt_path,
        "--audio_prompt", audio_prompt,
        "--temperature", str(temperature),
        "--output_dir", output_dir,
        "--assets_dir", assets_dir,
        "--target_language", target_language,
        "--cfg_weight", str(cfg_weight),
        "--exaggeration", str(exaggeration),
        "--output_srt", output_srt,
        "--output_wav", output_wav,
        "--changed_json", changed_json,
    ]

    # ── Size-based total timeout, scaled by job size ──────────
    n_segments = 0
    try:
        import srt

        from video_util import read_srt_text

        n_segments = len(list(srt.parse(read_srt_text(srt_path))))
    except Exception:
        n_segments = 0
    total_timeout = max(min_total_timeout, per_segment_budget * n_segments)

    def _stop(process: subprocess.Popen, reason: str) -> RuntimeError:
        process.terminate()
        try:
            process.wait(timeout=90)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        return RuntimeError(reason)

    with open(log_path, "a") as proc_log:
        proc_log.write(f"+ {' '.join(cmd)}\n")
        proc_log.flush()
        proc_pos = proc_log.tell()

        process = subprocess.Popen(
            cmd,
            stdout=proc_log,
            stderr=subprocess.STDOUT,
            text=True,
        )

        start = time.monotonic()
        last_size = proc_pos
        last_growth = start
        while True:
            try:
                process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - start
                cur_size = os.path.getsize(log_path)
                if cur_size > last_size:
                    last_size = cur_size
                    last_growth = time.monotonic()
                stalled_for = time.monotonic() - last_growth
                if stalled_for > stall_timeout:
                    raise _stop(
                        process,
                        f"gen_audio stalled: no output for {stalled_for / 60:.1f}min "
                        f"(elapsed {elapsed / 60:.1f}min)",
                    ) from None
                if elapsed > total_timeout:
                    raise _stop(
                        process,
                        f"gen_audio timed out after {elapsed / 60:.1f}min "
                        f"({n_segments} segments)",
                    ) from None
                get_job_queue().update_job_progress(
                    access_code, f"正在生成音频... ({int(elapsed) // 60}分{int(elapsed) % 60}秒)"
                )

    # Read back only the output written by this subprocess invocation
    with open(log_path) as f:
        f.seek(proc_pos)
        sub_out = f.read()

    output_lines = sub_out.strip().splitlines()
    if output_lines:
        job_log_lines(access_code, output_dir, output_lines)

    if process.returncode != 0:
        raise RuntimeError(sub_out[:500] if sub_out else f"gen_audio exited with code {process.returncode}")

    job_log(access_code, output_dir, "gen_audio subprocess completed")

    return {
        "output_srt_path": srt_path_out,
        "output_wav_path": wav_path_out,
        "changed_json_path": changed_json_out,
    }


def run_gen_video_step(
    video_file: str,
    srt_path: str,
    adjusted_srt: str,
    changed_json: str,
    output_path: str,
    access_code: str,
    blur: bool = False,
    timeout: float = 43200,  # 12 h safety limit
):
    """Run the gen_video.py subprocess to produce the final video.

    Args:
        blur: When True, passes ``--blur`` to gen_video.py so that
              speaker background is blurred behind subtitles.
    """
    # Write both stdout and stderr directly to job.log to avoid pipe
    # buffer deadlock and unnecessary in-memory buffering.
    output_dir = os.path.dirname(output_path)
    log_path = os.path.join(output_dir, "job.log")
    job_log(access_code, output_dir, f"--- gen_video {os.path.basename(output_path)} ---")

    with open(log_path, "a") as proc_log:
        proc_pos = proc_log.tell()

        cmd = [
            PYTHON_BIN,
            GEN_VIDEO_SCRIPT,
            video_file,
            srt_path,
            adjusted_srt,
            changed_json,
            "--output",
            output_path,
        ]
        if blur:
            cmd.append("--blur")

        process = subprocess.Popen(
            cmd,
            stdout=proc_log,
            stderr=proc_log,
            text=True,
        )

        # Poll subprocess with periodic progress updates
        start = time.monotonic()
        while True:
            try:
                process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                elapsed = int(time.monotonic() - start)
                if elapsed > timeout:
                    process.kill()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise RuntimeError(f"gen_video timed out after {elapsed // 3600}h") from None
                get_job_queue().update_job_progress(access_code, f"正在合成视频... ({elapsed // 60}分{elapsed % 60}秒)")

    # Read back only the output written by this subprocess invocation
    with open(log_path) as f:
        f.seek(proc_pos)
        sub_out = f.read()

    output_lines = sub_out.strip().splitlines()
    if output_lines:
        job_log_lines(access_code, output_dir, output_lines)

    if process.returncode != 0:
        raise RuntimeError(sub_out[:500] if sub_out else "gen_video failed")

    if not os.path.exists(output_path):
        raise RuntimeError(f"gen_video completed but output file missing: {output_path}")


def validate_files(expected_files: list[str], label: str = ""):
    """Check that all expected files exist. Raises RuntimeError if any are missing."""
    missing = [f for f in expected_files if not os.path.exists(f)]
    if missing:
        raise RuntimeError(f"{label} output files missing: {', '.join(missing)}")


# ── Checkpoint-aware wrappers ──────────────────────────────


def _audio_fallback_paths(audio_dir: str) -> dict[str, str]:
    """Return the paths dict used when the audio checkpoint is already done."""
    return {
        "output_srt_path": os.path.join(audio_dir, "output_adjusted.srt"),
        "changed_json_path": os.path.join(audio_dir, "changed_segments.json"),
    }


def run_audio_ckpt(
    translated_srt: str,
    output_dir: str,
    temperature: float,
    access_code: str,
    target_language: str = "en",
    cfg_weight: float = 0.5,
    exaggeration: float = 0.5,
    ckpt=None,
    audio_subdir: str = "audio_tracks",
) -> dict[str, str]:
    """Run the audio generation step if the checkpoint is not yet done.

    When *ckpt* is provided and the ``"audio"`` step is already marked,
    returns the expected output paths without re-running.

    Returns the same dict as ``run_gen_audio_step``: ``output_srt_path``,
    ``output_wav_path``, ``changed_json_path``.
    """
    audio_dir = os.path.join(output_dir, audio_subdir)
    if ckpt and ckpt.done("audio"):
        job_log(access_code, output_dir, "  ↪ audio already done, skipping")
        return _audio_fallback_paths(audio_dir)

    job_log(access_code, output_dir, "Generating audio from translated SRT...")
    audio_out = run_gen_audio_step(
        translated_srt,
        audio_dir,
        temperature,
        access_code,
        target_language=target_language,
        cfg_weight=cfg_weight,
        exaggeration=exaggeration,
    )
    if ckpt:
        ckpt.mark("audio")
    return audio_out


def run_video_ckpt(
    video_file: str,
    translated_srt: str,
    audio_out: dict[str, str],
    output_dir: str,
    access_code: str,
    ckpt=None,
    output_filename: str = "output_modified.mp4",
    blur: bool = False,
):
    """Run the video generation step if the checkpoint is not yet done.

    When *ckpt* is provided and the ``"video"`` step is already marked,
    this is a no-op (the caller still has the existing output files).

    After running (or skipping), the adjusted SRT is copied to *output_dir*
    as a convenience so callers don't need to repeat ``shutil.copy2``.
    """
    video_output = os.path.join(output_dir, output_filename)
    if ckpt and ckpt.done("video"):
        job_log(access_code, output_dir, "  ↪ video already done, skipping")
    else:
        job_log(access_code, output_dir, "Processing video...")
        run_gen_video_step(
            video_file,
            translated_srt,
            audio_out["output_srt_path"],
            audio_out["changed_json_path"],
            video_output,
            access_code,
            blur=blur,
        )
        if ckpt:
            ckpt.mark("video")

    # Copy the adjusted SRT to the output directory so it's visible
    # in the file browser without navigating into audio_tracks/.
    shutil.copy2(audio_out["output_srt_path"], output_dir)


# ── Download step ───────────────────────────────────────


def run_download_ckpt(
    video_number: str,
    output_dir: str,
    access_code: str,
    ckpt,
    proc_log,
    job_data: dict,
) -> str:
    """Download original video (or use cached copy). Returns the video path.

    Checkpoint-aware: skips if ``"download"`` is already marked.
    """
    video_path = os.path.join(output_dir, f"{video_number}.mp4")
    if ckpt and ckpt.done("download"):
        job_log(access_code, output_dir, "  ↪ download already done, skipping")
        return video_path

    cached_path = job_data.get("cached_path")
    if cached_path and os.path.isfile(cached_path):
        job_log(access_code, output_dir, f"Using cached video: {cached_path}")
        shutil.copy2(cached_path, video_path)
    else:
        job_log(access_code, output_dir, "Downloading original video...")
        download_script = os.path.join(PROJECT_ROOT, "..", "pre-process", "download_orig.py")
        codec = job_data.get("codec", "mp4")
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            result = subprocess.run(
                [PYTHON_BIN, download_script, video_number, output_dir, "--codec", codec],
                stdout=proc_log,
                stderr=proc_log,
                timeout=3600,
            )
            if result.returncode == 0 and os.path.exists(video_path):
                break
            if attempt < max_attempts:
                job_log(access_code, output_dir, f"Download failed, retrying ({attempt}/{max_attempts})...")
                time.sleep(20)

    if not os.path.exists(video_path):
        raise RuntimeError(f"Downloaded video not found: {video_path}")
    if ckpt:
        ckpt.mark("download")
    return video_path


# ── OCR step ─────────────────────────────────────────────


def run_ocr_ckpt(
    video_path: str,
    output_dir: str,
    access_code: str,
    ckpt,
    proc_log,
    ocr_srt_name: str = "ocr_screen.srt",
) -> str:
    """Run RapidVideOCR pipeline. Returns path to the generated SRT.

    Checkpoint-aware: skips if ``"ocr"`` is already marked.
    """
    ocr_srt = os.path.join(output_dir, ocr_srt_name)
    if ckpt and ckpt.done("ocr"):
        job_log(access_code, output_dir, "  ↪ OCR already done, skipping")
        return ocr_srt

    job_log(access_code, output_dir, "Running RapidVideOCR pipeline...")
    frames_dir = os.path.join(output_dir, "frames")
    subprocess.run(
        ["/usr/bin/bash", RAPID_VIDEOCR_PIPELINE_SCRIPT, "-i", video_path, "-o", ocr_srt, "-d", frames_dir],
        stdout=proc_log,
        stderr=proc_log,
        timeout=14400,
        env={**os.environ, "RAPID_VIDEOCR_BIN": RAPID_VIDEOCR_BIN},
        check=True,
    )
    if not os.path.exists(ocr_srt):
        raise RuntimeError("RapidVideOCR pipeline failed to generate SRT")
    if ckpt:
        ckpt.mark("ocr")
    return ocr_srt


# ── Translate step ───────────────────────────────────────


def run_translate_ckpt(
    input_srt: str,
    output_dir: str,
    access_code: str,
    ckpt,
    proc_log,
    log_file: str,
    target_language: str,
    translated_name: str = "translated.srt",
    intro_marker: str | None = None,
    outro_marker: str | None = None,
) -> str:
    """Translate an SRT file via HY-MT (subprocess on ROCm Python 3.11).

    Checkpoint-aware: skips if ``"translate"`` is already marked.
    """
    translated_srt = os.path.join(output_dir, translated_name)
    if ckpt and ckpt.done("translate"):
        job_log(access_code, output_dir, "  ↪ translation already done, skipping")
        return translated_srt

    job_log(
        access_code,
        output_dir,
        "Translating subtitles (GPU)..." + (" (intro/outro markers)" if intro_marker or outro_marker else ""),
    )
    target_language_name = LANG_MAP.get(target_language, target_language)

    translate_script = os.path.join(PROJECT_ROOT, "translate", "translate_srt.py")
    cmd = [
        TRANSLATE_PYTHON,
        "-u",
        translate_script,
        input_srt,
        translated_srt,
        "-l", target_language_name,
    ]
    if intro_marker:
        cmd.extend(["--intro", intro_marker])
    if outro_marker:
        cmd.extend(["--outro", outro_marker])

    with open(log_file, "a") as log_fh:
        log_fh.write(f"+ {' '.join(cmd)}\n")
        log_fh.flush()

    with open(log_file, "a") as log_fh:
        result = subprocess.run(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if result.returncode != 0:
        raise RuntimeError(f"translate_srt exited with code {result.returncode}")

    job_log(access_code, output_dir, "  ✓ translation complete")
    if ckpt:
        ckpt.mark("translate")
    return translated_srt


# ── Extract-audio step ───────────────────────────────────


def run_extract_audio_ckpt(
    video_file: str,
    output_dir: str,
    access_code: str,
    ckpt,
    proc_log,
    audio_name: str = "audio.wav",
) -> str:
    """Extract audio from a video file via ffmpeg. Returns path to the wav.

    Checkpoint-aware: skips if ``"extract_audio"`` is already marked.
    """
    audio_path = os.path.join(output_dir, audio_name)
    if ckpt and ckpt.done("extract_audio"):
        job_log(access_code, output_dir, "  ↪ extract_audio already done, skipping")
        return audio_path

    job_log(access_code, output_dir, "Extracting audio from video...")
    result = subprocess.run(
        ["ffmpeg", "-i", video_file, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path, "-y"],
        stdout=proc_log,
        stderr=subprocess.PIPE,
        text=True,
        timeout=3600,
    )
    if result.returncode != 0:
        if result.stderr:
            proc_log.write(result.stderr)
        result.check_returncode()
    if ckpt:
        ckpt.mark("extract_audio")
    return audio_path


# ── Whisper step ─────────────────────────────────────────


def run_whisper_ckpt(
    audio_path: str,
    output_dir: str,
    access_code: str,
    ckpt,
    proc_log,
    whisper_srt_name: str = "whisper.srt",
) -> str:
    """Run whisper-cli speech recognition. Returns path to the generated SRT.

    Checkpoint-aware: skips if ``"whisper"`` is already marked.
    """
    whisper_srt = os.path.join(output_dir, whisper_srt_name)
    if ckpt and ckpt.done("whisper"):
        job_log(access_code, output_dir, "  ↪ whisper already done, skipping")
        return whisper_srt

    job_log(access_code, output_dir, f"Running whisper speech recognition (OpenVINO device={WHISPER_OV_DEVICE})...")
    subprocess.run(
        [
            "whisper-cli",
            "-m",
            WHISPER_MODEL,
            "-f",
            audio_path,
            "-osrt",
            "-of",
            whisper_srt.replace(".srt", ""),
            "-l",
            "zh",
            "-oved",
            WHISPER_OV_DEVICE,
        ],
        stdout=proc_log,
        stderr=proc_log,
        timeout=7200,
        check=True,
    )
    if not os.path.exists(whisper_srt):
        raise RuntimeError("Whisper failed to generate SRT")
    if ckpt:
        ckpt.mark("whisper")
    return whisper_srt


# ── adjust_original_audio helper ─────────────────────────


def _adjust_original_audio_nonfatal(video_path, translated_srt, audio_out, output_dir, access_code):
    """Call adjust_original_audio, logging a warning on failure (non-fatal)."""
    try:
        adjust_original_audio(
            video_path, translated_srt, audio_out["output_srt_path"], output_dir, access_code=access_code
        )
    except subprocess.SubprocessError:
        job_log(access_code, output_dir, "Warning: zh audio adjustment failed (non-fatal)")


# ── Original zh audio adjustment ────────────────────────────


def _get_video_duration(video_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def _build_atempo_filter(stretch: float) -> str:
    """Build an ffmpeg atempo filter chain for the given stretch factor.

    atempo supports 0.5–2.0 per instance.  For factors outside this range
    we chain multiple instances:

    - If *stretch* > 2.0: chain ``atempo=2.0`` filters then the remainder.
    - If *stretch* < 0.5: chain ``atempo=0.5`` filters then the remainder.
    """
    if stretch <= 0:
        return "atempo=0.5"
    if 0.5 <= stretch <= 2.0:
        return f"atempo={stretch:.6f}"
    parts = []
    remaining = stretch
    if stretch > 2.0:
        while remaining > 2.0:
            parts.append("atempo=2.0")
            remaining /= 2.0
    else:  # stretch < 0.5
        while remaining < 0.5:
            parts.append("atempo=0.5")
            remaining /= 0.5
    parts.append(f"atempo={remaining:.6f}")
    return ",".join(parts)


def adjust_original_audio(
    video_path: str,
    original_srt_path: str,
    adjusted_srt_path: str,
    output_dir: str,
    access_code: str = "",
    audio_offset: float = 0.0,
):
    """Extract original zh audio and stretch it to match the adjusted SRT timing.

    Mirrors the same segment-stretch logic that ``gen_video.py`` applies to
    the video track (leading gap, subtitle segments, inter-subtitle gaps,
    trailing gap), but operates on the audio stream using the *atempo* filter
    instead of *setpts*.

    Parameters
    ----------
    video_path:
        Source video file containing the original zh audio track.
    original_srt_path:
        Original SRT (input to gen_audio / the translated SRT).
    adjusted_srt_path:
        Adjusted SRT (output from gen_audio).
    output_dir:
        Directory where ``orig_zh_adjusted.wav`` will be written.
    access_code:
        Job access code for logging (optional).
    audio_offset:
        Seconds to skip from the start of *video_path* before extracting
        audio (used when the video file has extra leading content that was
        trimmed away before OCR runs, e.g. the ning-video intro trim).
    """
    import srt
    from video_util import read_srt_text

    job_log(access_code, output_dir, "Adjusting original zh audio...")

    # ── Parse SRTs ───────────────────────────────────────────
    orig_subs = list(srt.parse(read_srt_text(original_srt_path)))
    adj_subs = list(srt.parse(read_srt_text(adjusted_srt_path)))

    if len(orig_subs) != len(adj_subs):
        job_log(
            access_code,
            output_dir,
            f"Segment count mismatch ({len(orig_subs)} vs {len(adj_subs)}), skipping zh audio adjustment",
        )
        return

    if not orig_subs:
        return

    # ── Extract audio from video ─────────────────────────────
    orig_audio_path = os.path.join(output_dir, "orig_zh_full.wav")
    extract_cmd = ["ffmpeg", "-y"]
    if audio_offset > 0:
        extract_cmd.extend(["-ss", str(audio_offset)])
    extract_cmd.extend(
        [
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            orig_audio_path,
        ]
    )
    subprocess.run(extract_cmd, check=True, capture_output=True)

    if not os.path.exists(orig_audio_path):
        job_log(access_code, output_dir, "Audio extraction produced no file, skipping")
        return

    # ── Build segment list (mirrors gen_video.py process_video) ──
    video_duration = _get_video_duration(video_path) - audio_offset
    seg_defs = build_segment_defs(
        orig_starts=[s.start.total_seconds() for s in orig_subs],
        orig_ends=[s.end.total_seconds() for s in orig_subs],
        adj_starts=[s.start.total_seconds() for s in adj_subs],
        adj_ends=[s.end.total_seconds() for s in adj_subs],
        video_duration=video_duration,
        offset=audio_offset,
    )

    # ── Process audio segments ───────────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix="adj_zh_audio_")
    try:
        seg_files = []
        for idx, seg in enumerate(seg_defs):
            if seg["end"] - seg["start"] <= 0:
                continue
            seg_path = os.path.join(tmp_dir, f"seg_{idx:04d}.wav")
            # atempo is INVERSE of setpts: atempo=2.0 → faster/shorter,
            # setpts=2.0 → slower/longer.  We want audio to stretch the same
            # way as video, so use 1/stretch for the atempo factor.
            atempo = _build_atempo_filter(1.0 / seg["stretch"]) if seg["stretch"] > 0 else _build_atempo_filter(100.0)
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(seg["start"]),
                    "-to",
                    str(seg["end"]),
                    "-i",
                    orig_audio_path,
                    "-af",
                    atempo,
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    seg_path,
                ],
                check=True,
                capture_output=True,
            )
            seg_files.append(seg_path)

        if not seg_files:
            job_log(access_code, output_dir, "No audio segments produced, skipping")
            return

        # ── Concatenate ───────────────────────────────────
        concat_list = os.path.join(tmp_dir, "concat.txt")
        with open(concat_list, "w") as f:
            for sf in seg_files:
                f.write(f"file '{sf}'\n")

        out_path = os.path.join(output_dir, "orig_zh_adjusted.wav")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_list,
                "-c",
                "copy",
                out_path,
            ],
            check=True,
            capture_output=True,
        )

        # ── Correct accumulated atempo rounding error ─────────
        # Per-segment atempo (especially on short 1-3 s segments)
        # drifts a few percent, and over ~70 segments the drift
        # compounds to a 1-2 s mismatch.  Apply a one-shot atempo
        # on the concatenated track so its duration matches the
        # designed timeline (sum of stretched segment durations).
        expected_total = sum((s["end"] - s["start"]) * s["stretch"] for s in seg_defs)
        actual_total = _get_video_duration(out_path)
        if expected_total > 0 and abs(actual_total - expected_total) > 0.05:
            factor = actual_total / expected_total
            job_log(
                access_code,
                output_dir,
                f"  Correcting zh audio duration: {actual_total:.2f}s → {expected_total:.2f}s (atempo={factor:.4f})",
            )
            fix_path = os.path.join(tmp_dir, "orig_zh_fixed.wav")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    out_path,
                    "-af",
                    _build_atempo_filter(factor),
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    fix_path,
                ],
                check=True,
                capture_output=True,
            )
            shutil.move(fix_path, out_path)

        job_log(access_code, output_dir, f"  ✓ orig_zh_adjusted.wav saved ({len(seg_files)} segments)")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Shared OCR pipeline helpers ─────────────────────────


def run_ocr_translate_step(
    video_path: str,
    output_dir: str,
    access_code: str,
    ckpt,
    proc_log,
    log_file: str,
    target_language: str,
    intro_marker: str = "",
    outro_marker: str = "",
) -> str:
    """Run OCR → translate on a video, return path to translated SRT."""
    return run_translate_ckpt(
        run_ocr_ckpt(video_path, output_dir, access_code, ckpt, proc_log),
        output_dir,
        access_code,
        ckpt,
        proc_log,
        log_file,
        target_language,
        intro_marker=intro_marker,
        outro_marker=outro_marker,
    )


def run_ocr_only_step(
    video_path: str,
    output_dir: str,
    access_code: str,
    ckpt,
    proc_log,
) -> str:
    """Run OCR only — no translation, no audio, no video."""
    return run_ocr_ckpt(video_path, output_dir, access_code, ckpt, proc_log)


def run_ocr_full_pipeline(
    video_path: str,
    output_dir: str,
    access_code: str,
    ap: dict,
    ckpt,
    proc_log,
    log_file: str,
    intro_marker: str = "",
    outro_marker: str = "",
    audio_subdir: str = "audio",
    blur: bool = False,
    validate_label: str = "",
) -> None:
    """Run OCR → translate → audio → video pipeline."""
    translated_srt = run_ocr_translate_step(
        video_path, output_dir, access_code, ckpt, proc_log, log_file,
        ap["target_language"], intro_marker=intro_marker, outro_marker=outro_marker,
    )
    audio_out = run_audio_ckpt(
        translated_srt,
        output_dir,
        ap["temperature"],
        access_code,
        target_language=ap["target_language"],
        cfg_weight=ap["cfg_weight"],
        exaggeration=ap["exaggeration"],
        ckpt=ckpt,
        audio_subdir=audio_subdir,
    )
    run_video_ckpt(
        video_path,
        translated_srt,
        audio_out,
        output_dir,
        access_code,
        ckpt=ckpt,
        output_filename="output_modified.mp4",
        blur=blur,
    )
    _adjust_original_audio_nonfatal(video_path, translated_srt, audio_out, output_dir, access_code)

    if validate_label:
        validate_files(
            [
                audio_out["output_srt_path"],
                audio_out["output_wav_path"],
                os.path.join(output_dir, "output_modified.mp4"),
            ],
            label=validate_label,
        )

    job_log(access_code, output_dir, "Done!")
