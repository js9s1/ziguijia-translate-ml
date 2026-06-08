"""Shared subprocess pipeline steps for audio/video job handlers.

Checkpoint-aware wrappers
-------------------------
``run_audio_ckpt`` and ``run_video_ckpt`` combine the pipeline subprocess
invocation with a ``CheckpointHelper`` check-skip-mark cycle and fallback
path construction.  Every checkpoint-using job handler (``_run_video_auto_job``,
``_run_video_ocr_job``, ``_run_video_ning_ocr_job``) delegates to these
instead of repeating the same boilerplate.
"""

import os
import shutil
import subprocess
import time

from log_utils import job_log, job_log_lines
from config import (
    PYTHON_BIN,
    GEN_AUDIO_SCRIPT,
    GEN_VIDEO_SCRIPT,
    AUDIO_PROMPT_PATH,
)


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
    timeout: int = 3600,
) -> dict[str, str]:
    """Run the gen_audio.py subprocess and return paths to generated files.
    Progress is updated every 30 seconds while waiting.

    Returns:
        dict with keys: output_srt_path, output_wav_path, changed_json_path

    Raises:
        RuntimeError if the subprocess fails or output files are missing.
    """
    os.makedirs(output_dir, exist_ok=True)

    srt_path_out = os.path.join(output_dir, output_srt)
    wav_path_out = os.path.join(output_dir, output_wav)
    changed_json_out = os.path.join(output_dir, changed_json)

    # Write both stdout and stderr directly to job.log to avoid pipe
    # buffer deadlock and unnecessary in-memory buffering.
    log_path = os.path.join(output_dir, "job.log")
    job_log(access_code, output_dir, f"--- gen_audio ---")

    with open(log_path, "a") as proc_log:
        proc_pos = proc_log.tell()

        process = subprocess.Popen(
            [
                PYTHON_BIN,
                GEN_AUDIO_SCRIPT,
                srt_path,
                "--audio_prompt", audio_prompt,
                "--output_dir", output_dir,
                "--output_srt", output_srt,
                "--output_wav", output_wav,
                "--changed_json", changed_json,
                "--temperature", str(temperature),
                "--target_language", target_language,
                "--cfg_weight", str(cfg_weight),
                "--exaggeration", str(exaggeration),
            ],
            stdout=proc_log,
            stderr=proc_log,
            text=True,
            env={
                **os.environ,
                "HSA_OVERRIDE_GFX_VERSION": "9.0.0",
                "HSA_XNACK": "0",
                "ROCBLAS_USE_HIPBLASLT": "0",
            },
        )

        # Poll subprocess with periodic progress updates
        from jobqueue import get_job_queue
        start = time.monotonic()
        while True:
            try:
                process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                elapsed = int(time.monotonic() - start)
                get_job_queue().update_job_progress(
                    access_code,
                    f"正在合成音频... ({elapsed // 60}分{elapsed % 60}秒)"
                )

    # Read back only the output written by this subprocess invocation
    with open(log_path, "r") as f:
        f.seek(proc_pos)
        sub_out = f.read()

    output_lines = sub_out.strip().splitlines()
    if output_lines:
        job_log_lines(access_code, output_dir, output_lines)

    if process.returncode != 0:
        raise RuntimeError(sub_out[:500] if sub_out else "gen_audio failed")

    expected = [srt_path_out, wav_path_out]
    missing = [f for f in expected if not os.path.exists(f)]
    if missing:
        raise RuntimeError(f"gen_audio completed but output files missing: {', '.join(missing)}")

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
    timeout: int = 7200,
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
            "--output", output_path,
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
        from jobqueue import get_job_queue
        start = time.monotonic()
        while True:
            try:
                process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                elapsed = int(time.monotonic() - start)
                get_job_queue().update_job_progress(
                    access_code,
                    f"正在合成视频... ({elapsed // 60}分{elapsed % 60}秒)"
                )

    # Read back only the output written by this subprocess invocation
    with open(log_path, "r") as f:
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
    timeout: int = 3600,
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
        translated_srt, audio_dir, temperature, access_code,
        target_language=target_language,
        cfg_weight=cfg_weight,
        exaggeration=exaggeration,
        timeout=timeout,
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
            video_file, translated_srt,
            audio_out["output_srt_path"], audio_out["changed_json_path"],
            video_output, access_code,
            blur=blur,
        )
        if ckpt:
            ckpt.mark("video")

    # Copy the adjusted SRT to the output directory so it's visible
    # in the file browser without navigating into audio_tracks/.
    shutil.copy2(audio_out["output_srt_path"], output_dir)
