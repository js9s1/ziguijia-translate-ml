"""Ning-video job — video synthesis from a preset video number + SRT."""

import json
import os
import re
import shutil
import subprocess
import time
import uuid

from jobqueue import get_job_queue, JobStatus
from log_utils import job_log, job_log_lines
from config import VIDEO_DIR, GEN_VIDEO_ORIG_SCRIPT, RAPID_VIDEOCR_PIPELINE_SCRIPT, PROJECT_ROOT, RAPID_VIDEOCR_BIN, PYTHON_BIN, LANG_MAP
from pipeline import run_audio_ckpt, run_video_ckpt, validate_files, adjust_original_audio
from video_util import CheckpointHelper, translate_srt_file, open_proc_log


# Markers used to auto-detect intro/outro boundaries in OCR SRT content
_MARKER_INTRO = "杨宁随缘开示"
_MARKER_OUTRO = "子归家全体编制人员"


def _find_cached_video(number: str) -> str | None:
    """Search VIDEO_DIR for an existing {number}.mp4 in any subdirectory.

    Returns the full path to the first cached copy found, or None if none exist.
    """
    if not os.path.isdir(VIDEO_DIR):
        return None
    target = f"{number}.mp4"
    for entry in os.listdir(VIDEO_DIR):
        subdir = os.path.join(VIDEO_DIR, entry)
        if os.path.isdir(subdir):
            candidate = os.path.join(subdir, target)
            if os.path.isfile(candidate):
                return candidate
    return None


def _download_video(ckpt, job_data: dict, video_number: str,
                    access_code: str, output_dir: str, proc_log) -> str:
    """Download original video (or use cached copy). Returns path to the video."""
    if ckpt.done("download"):
        job_log(access_code, output_dir, "  ↪ download already done, skipping")
        return os.path.join(output_dir, f"{video_number}.mp4")

    cached_path = job_data.get("cached_path")
    if cached_path and os.path.isfile(cached_path):
        job_log(access_code, output_dir, f"Using cached video: {cached_path}")
        shutil.copy2(cached_path, os.path.join(output_dir, f"{video_number}.mp4"))
    else:
        job_log(access_code, output_dir, "Downloading original video...")
        download_script = os.path.join(PROJECT_ROOT, "..", "pre-process", "download_orig.py")
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            result = subprocess.run(
                [PYTHON_BIN, download_script, video_number, output_dir],
                stdout=proc_log, stderr=proc_log, timeout=3600,
            )
            video_path = os.path.join(output_dir, f"{video_number}.mp4")
            if result.returncode == 0 and os.path.exists(video_path):
                break
            if attempt < max_attempts:
                job_log(access_code, output_dir,
                        f"Download failed, retrying ({attempt}/{max_attempts})...")
                time.sleep(20)

    video_path = os.path.join(output_dir, f"{video_number}.mp4")
    if not os.path.exists(video_path):
        raise RuntimeError(f"Downloaded video not found: {video_path}")
    ckpt.mark("download")
    return video_path


def _run_video_job(job_data: dict):
    video_number = job_data["video_number"]
    access_code = job_data["access_code"]
    srt_path = job_data["srt_path"]
    output_dir = job_data["output_dir"]
    temperature = job_data.get("temperature", 0.8)
    target_language = job_data.get("target_language", "en")
    cfg_weight = job_data.get("cfg_weight", 0.5)
    exaggeration = job_data.get("exaggeration", 0.5)
    blur = job_data.get("blur", "yes")

    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "job.log")

    with open_proc_log(log_file) as (proc_log, _):
        valid_steps = ["download", "audio", "video"]
        ckpt = CheckpointHelper(access_code, output_dir, valid_steps)

        # Step 1: Download (or use cached) original video
        video_path = _download_video(ckpt, job_data, video_number, access_code, output_dir, proc_log)

        # Step 2: Generate audio from the user's SRT
        audio_out = run_audio_ckpt(
            srt_path, output_dir, temperature, access_code,
            target_language=target_language,
            cfg_weight=cfg_weight, exaggeration=exaggeration,
            ckpt=ckpt,
            audio_subdir="audio",
        )

        # Step 3: Process video with stretched segments
        run_video_ckpt(video_path, srt_path, audio_out, output_dir,
                       access_code, ckpt=ckpt,
                       output_filename="output_modified.mp4",
                       blur=(blur == "yes"))

        # Step 4: Adjust original zh audio (non-fatal)
        try:
            adjust_original_audio(video_path, srt_path,
                                  audio_out["output_srt_path"], output_dir,
                                  access_code=access_code)
        except Exception:
            job_log(access_code, output_dir, "Warning: zh audio adjustment failed (non-fatal)")

    audio_dir = os.path.join(output_dir, "audio")
    validate_files([
        os.path.join(audio_dir, "output_adjusted.srt"),
        os.path.join(audio_dir, "output.wav"),
        os.path.join(output_dir, "output_modified.mp4"),
    ], label="宁视频翻译")

    # Copy adjusted SRT to top-level so the user can access it
    shutil.copy2(os.path.join(audio_dir, "output_adjusted.srt"), output_dir)
    job_log(access_code, output_dir, "Done!")


def process_video_ning(number: str, srt_file, temperature: float, user_id: int = None, blur: str = "yes", target_language: str = "en", cfg_weight: float = 0.5, exaggeration: float = 0.5, cached_path: str | None = None) -> dict:
    # Reuse an existing failed job for the same video+user so checkpoints carry over
    jq = get_job_queue()
    existing = jq._find_failed_ning_job(number, user_id)
    if existing:
        access_code, output_dir = existing
        job_log_lines(access_code, output_dir, [f"--- resubmit (temperature={temperature}, lang={target_language}) ---"])
    else:
        access_code = str(uuid.uuid4())[:8].upper()
        output_dir = os.path.join(VIDEO_DIR, f"{number}-{access_code}")
        os.makedirs(output_dir, exist_ok=True)

    srt_path = os.path.join(output_dir, srt_file.filename)
    srt_file.save(srt_path)

    job_data = {
        "video_number": number,
        "srt_path": srt_path,
        "output_dir": output_dir,
        "access_code": access_code,
        "temperature": temperature,
        "blur": blur,
        "target_language": target_language,
        "cfg_weight": cfg_weight,
        "exaggeration": exaggeration,
    }

    if cached_path:
        job_data["cached_path"] = cached_path

    job_access_code = jq.add_job(job_data, _run_video_job, user_id)
    return {"access_code": job_access_code, "message": "Job queued successfully"}


def _run_video_ning_ocr_job(job_data: dict):
    video_number = job_data["video_number"]
    access_code = job_data["access_code"]
    output_dir = job_data["output_dir"]
    temperature = job_data.get("temperature", 0.8)
    target_language = job_data.get("target_language", "en")
    cfg_weight = job_data.get("cfg_weight", 0.5)
    exaggeration = job_data.get("exaggeration", 0.5)
    blur = job_data.get("blur", "yes")

    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "job.log")

    with open_proc_log(log_file) as (proc_log, _):
        valid_steps = ["download", "ocr", "translate", "audio", "video"]
        ckpt = CheckpointHelper(access_code, output_dir, valid_steps)

        # Step 1: Download (or use cached) original video
        video_path = _download_video(ckpt, job_data, video_number, access_code, output_dir, proc_log)

        # Step 2: Run rapid_videocr_pipeline.sh on full video to generate OCR SRT
        ocr_srt = os.path.join(output_dir, "ocr_screen.srt")
        if not ckpt.done("ocr"):
            job_log(access_code, output_dir, "Running RapidVideOCR pipeline...")
            frames_dir = os.path.join(output_dir, "frames")
            subprocess.run(
                ["/usr/bin/bash", RAPID_VIDEOCR_PIPELINE_SCRIPT, "-i", video_path,
                 "-o", ocr_srt, "-d", frames_dir],
                stdout=proc_log, stderr=proc_log, timeout=14400,
                env={**os.environ, "RAPID_VIDEOCR_BIN": RAPID_VIDEOCR_BIN},
            )
            if not os.path.exists(ocr_srt):
                raise RuntimeError("RapidVideOCR pipeline failed to generate SRT")
            ckpt.mark("ocr")
        else:
            job_log(access_code, output_dir, "  ↪ OCR already done, skipping")

        # Step 3: Translate OCR SRT via HY-MT (header/trailer auto-trimmed)
        translated_srt = os.path.join(output_dir, "translated.srt")
        if not ckpt.done("translate"):
            job_log(access_code, output_dir, "Translating OCR subtitles (intro/outro auto-detected)...")
            target_language_name = LANG_MAP.get(target_language, target_language)
            translate_srt_file(ocr_srt, translated_srt, access_code, output_dir,
                               target_language_name, proc_log, log_file,
                               intro_marker=_MARKER_INTRO, outro_marker=_MARKER_OUTRO)
            ckpt.mark("translate")
        else:
            job_log(access_code, output_dir, "  ↪ translation already done, skipping")

        # Step 4: Generate audio from translated SRT
        audio_out = run_audio_ckpt(
            translated_srt, output_dir, temperature, access_code,
            target_language=target_language,
            cfg_weight=cfg_weight, exaggeration=exaggeration,
            ckpt=ckpt,
            audio_subdir="audio",
        )

        # Step 5: Process full video with stretched segments
        run_video_ckpt(video_path, translated_srt, audio_out, output_dir,
                       access_code, ckpt=ckpt, output_filename="output_modified.mp4",
                       blur=(blur == "yes"))

        # Adjust original zh audio to match video stretch
        try:
            adjust_original_audio(video_path, translated_srt,
                                  audio_out["output_srt_path"], output_dir,
                                  access_code=access_code)
        except Exception:
            job_log(access_code, output_dir, "Warning: zh audio adjustment failed (non-fatal)")

    validate_files([
        audio_out["output_srt_path"],
        audio_out["output_wav_path"],
        os.path.join(output_dir, "output_modified.mp4"),
    ], label="宁视频OCR翻译")
    job_log(access_code, output_dir, "Done!")


def process_video_ning_ocr(number: str, temperature: float, user_id: int = None, blur: str = "yes", target_language: str = "en", cfg_weight: float = 0.5, exaggeration: float = 0.5, cached_path: str | None = None) -> dict:
    # Reuse an existing failed job for the same video+user so checkpoints carry over
    jq = get_job_queue()
    existing = jq._find_failed_ocr_job(number, user_id)
    if existing:
        access_code, output_dir = existing
        # Invalidate checkpoints from "download" onward so job re-runs
        # (old trim step is gone; download checkpoint stays for cached video reuse)
        jq.invalidate_checkpoints_after(access_code, "download")
        job_log_lines(access_code, output_dir, [f"--- resubmit (temperature={temperature}, lang={target_language}) ---"])
    else:
        access_code = str(uuid.uuid4())[:8].upper()
        output_dir = os.path.join(VIDEO_DIR, f"{number}-{access_code}")
        os.makedirs(output_dir, exist_ok=True)

    job_data = {
        "video_number": number,
        "output_dir": output_dir,
        "access_code": access_code,
        "temperature": temperature,
        "blur": blur,
        "target_language": target_language,
        "cfg_weight": cfg_weight,
        "exaggeration": exaggeration,
    }

    if cached_path:
        job_data["cached_path"] = cached_path

    job_access_code = jq.add_job(job_data, _run_video_ning_ocr_job, user_id)
    return {"access_code": job_access_code, "message": "OCR translation job queued"}


# ── OCR-only (download, OCR, translate, then stop) ──────────

def _run_video_ning_ocr_translate_only_job(job_data: dict):
    """Download, run OCR on full video, translate with marker trimming — stop after translate."""
    video_number = job_data["video_number"]
    access_code = job_data["access_code"]
    output_dir = job_data["output_dir"]
    temperature = job_data.get("temperature", 0.8)
    target_language = job_data.get("target_language", "en")
    cfg_weight = job_data.get("cfg_weight", 0.5)
    exaggeration = job_data.get("exaggeration", 0.5)

    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "job.log")

    with open_proc_log(log_file) as (proc_log, _):
        valid_steps = ["download", "ocr", "translate"]
        ckpt = CheckpointHelper(access_code, output_dir, valid_steps)

        # Step 1: Download
        video_path = _download_video(ckpt, job_data, video_number, access_code, output_dir, proc_log)

        # Step 2: OCR on full video
        ocr_srt = os.path.join(output_dir, "ocr_screen.srt")
        if not ckpt.done("ocr"):
            job_log(access_code, output_dir, "Running RapidVideOCR pipeline...")
            frames_dir = os.path.join(output_dir, "frames")
            subprocess.run(
                ["/usr/bin/bash", RAPID_VIDEOCR_PIPELINE_SCRIPT, "-i", video_path,
                 "-o", ocr_srt, "-d", frames_dir],
                stdout=proc_log, stderr=proc_log, timeout=14400,
                env={**os.environ, "RAPID_VIDEOCR_BIN": RAPID_VIDEOCR_BIN},
            )
            if not os.path.exists(ocr_srt):
                raise RuntimeError("RapidVideOCR pipeline failed to generate SRT")
            ckpt.mark("ocr")
        else:
            job_log(access_code, output_dir, "  ↪ OCR already done, skipping")

        # Step 3: Translate with intro/outro marker detection
        translated_srt = os.path.join(output_dir, "translated.srt")
        if not ckpt.done("translate"):
            job_log(access_code, output_dir, "Translating OCR subtitles (intro/outro auto-detected)...")
            target_language_name = LANG_MAP.get(target_language, target_language)
            translate_srt_file(ocr_srt, translated_srt, access_code, output_dir,
                               target_language_name, proc_log, log_file,
                               intro_marker=_MARKER_INTRO, outro_marker=_MARKER_OUTRO)
            ckpt.mark("translate")
        else:
            job_log(access_code, output_dir, "  ↪ translation already done, skipping")

        # Copy translated SRT to top-level for easy access
        if os.path.exists(translated_srt):
            shutil.copy2(translated_srt, os.path.join(output_dir, "output_adjusted.srt"))

    job_log(access_code, output_dir, "Done! OCR → translation complete (audio/video skipped)")


def process_video_ning_ocr_translate_only(number: str, temperature: float, user_id: int = None, blur: str = "yes", target_language: str = "en", cfg_weight: float = 0.5, exaggeration: float = 0.5, cached_path: str | None = None) -> dict:
    access_code = str(uuid.uuid4())[:8].upper()
    output_dir = os.path.join(VIDEO_DIR, f"{number}-{access_code}")
    os.makedirs(output_dir, exist_ok=True)

    job_data = {
        "video_number": number,
        "output_dir": output_dir,
        "access_code": access_code,
        "temperature": temperature,
        "target_language": target_language,
        "cfg_weight": cfg_weight,
        "exaggeration": exaggeration,
    }

    if cached_path:
        job_data["cached_path"] = cached_path

    job_access_code = get_job_queue().add_job(job_data, _run_video_ning_ocr_translate_only_job, user_id)
    return {"access_code": job_access_code, "message": "OCR translate-only job queued"}
