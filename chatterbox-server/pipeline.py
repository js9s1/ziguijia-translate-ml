"""Shared subprocess pipeline steps for audio/video job handlers."""

import logging
import os
import subprocess
import time
from typing import Optional

from config import (
    PYTHON_BIN,
    GEN_AUDIO_SCRIPT,
    GEN_VIDEO_SCRIPT,
    AUDIO_PROMPT_PATH,
)

logger = logging.getLogger(__name__)


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

    # Redirect stderr to job.log to prevent pipe buffer deadlock
    log_path = os.path.join(output_dir, "job.log")
    job_log = open(log_path, "a")
    job_log.write(f"\n--- gen_audio {access_code} ---\n")
    job_log.flush()

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
        stdout=subprocess.PIPE,
        stderr=job_log,
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
                f"正在合成音频... ({elapsed // 60}分{elapsed % 60}秒)"
            )

    stdout, _ = process.communicate()
    job_log.close()

    for line in stdout.strip().splitlines():
        logger.info(f"[Job {access_code}] {line}")

    if process.returncode != 0:
        stderr_text = open(log_path).read()
        for line in stderr_text.strip().splitlines():
            logger.error(f"[Job {access_code}] {line}")
        raise RuntimeError(stderr_text[:500] if stderr_text else "gen_audio failed")

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
    timeout: int = 7200,
):
    """Run the gen_video.py subprocess to produce the final video."""
    # Redirect stderr to job.log to prevent pipe buffer deadlock
    log_path = os.path.join(os.path.dirname(output_path), "job.log")
    job_log = open(log_path, "a")
    job_log.write(f"\n--- gen_video {os.path.basename(output_path)} ---\n")
    job_log.flush()
    process = subprocess.Popen(
        [
            PYTHON_BIN,
            GEN_VIDEO_SCRIPT,
            video_file,
            srt_path,
            adjusted_srt,
            changed_json,
            "--output", output_path,
        ],
        stdout=subprocess.PIPE,
        stderr=job_log,
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

    stdout, _ = process.communicate()
    job_log.close()

    for line in stdout.strip().splitlines():
        logger.info(f"[Job {access_code}] {line}")

    if process.returncode != 0:
        stderr_text = open(log_path).read()
        for line in stderr_text.strip().splitlines():
            logger.error(f"[Job {access_code}] {line}")
        raise RuntimeError(stderr_text[:500] if stderr_text else "gen_video failed")

    if not os.path.exists(output_path):
        raise RuntimeError(f"gen_video completed but output file missing: {output_path}")


def validate_files(expected_files: list[str], label: str = ""):
    """Check that all expected files exist. Raises RuntimeError if any are missing."""
    missing = [f for f in expected_files if not os.path.exists(f)]
    if missing:
        raise RuntimeError(f"{label} output files missing: {', '.join(missing)}")
