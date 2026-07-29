"""Ning-video job — video synthesis from a preset video number + SRT."""

import os
import shutil
import uuid

from config import MARKER_INTRO, MARKER_OUTRO, VIDEO_DIR
from jobqueue import get_job_queue
from log_utils import job_log, job_log_lines
from middleware import get_audio_params
from pipeline import (
    _adjust_original_audio_nonfatal,
    run_audio_ckpt,
    run_download_ckpt,
    run_ocr_ckpt,
    run_translate_ckpt,
    run_video_ckpt,
    validate_files,
)
from video_util import CheckpointHelper, open_proc_log


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


def _run_video_job(job_data: dict):
    video_number = job_data["video_number"]
    access_code = job_data["access_code"]
    srt_path = job_data["srt_path"]
    output_dir = job_data["output_dir"]
    ap = get_audio_params(job_data)
    blur = job_data.get("blur", "yes")

    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "job.log")

    with open_proc_log(log_file) as (proc_log, _):
        valid_steps = ["download", "audio", "video"]
        ckpt = CheckpointHelper(access_code, output_dir, valid_steps)

        video_path = run_download_ckpt(video_number, output_dir, access_code, ckpt, proc_log, job_data)
        audio_out = run_audio_ckpt(
            srt_path,
            output_dir,
            ap["temperature"],
            access_code,
            target_language=ap["target_language"],
            cfg_weight=ap["cfg_weight"],
            exaggeration=ap["exaggeration"],
            ckpt=ckpt,
            audio_subdir="audio",
        )
        run_video_ckpt(
            video_path,
            srt_path,
            audio_out,
            output_dir,
            access_code,
            ckpt=ckpt,
            output_filename="output_modified.mp4",
            blur=(blur == "yes"),
        )
        _adjust_original_audio_nonfatal(video_path, srt_path, audio_out, output_dir, access_code)

    validate_files(
        [
            os.path.join(output_dir, "audio", "output_adjusted.srt"),
            os.path.join(output_dir, "audio", "output.wav"),
            os.path.join(output_dir, "output_modified.mp4"),
        ],
        label="宁视频翻译",
    )
    shutil.copy2(os.path.join(output_dir, "audio", "output_adjusted.srt"), output_dir)
    job_log(access_code, output_dir, "Done!")


def process_video_ning(
    number: str,
    srt_file,
    temperature: float,
    user_id: int = None,
    blur: str = "yes",
    target_language: str = "en",
    cfg_weight: float = 0.5,
    exaggeration: float = 0.5,
    cached_path: str | None = None,
) -> dict:
    jq = get_job_queue()
    existing = jq._find_failed_ning_job(number, user_id)
    if existing:
        access_code, output_dir = existing
        job_log_lines(
            access_code, output_dir, [f"--- resubmit (temperature={temperature}, lang={target_language}) ---"]
        )
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
    ap = get_audio_params(job_data)
    blur = job_data.get("blur", "yes")

    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "job.log")

    with open_proc_log(log_file) as (proc_log, _):
        valid_steps = ["download", "ocr", "translate", "audio", "video"]
        ckpt = CheckpointHelper(access_code, output_dir, valid_steps)

        video_path = run_download_ckpt(video_number, output_dir, access_code, ckpt, proc_log, job_data)
        translated_srt = run_translate_ckpt(
            run_ocr_ckpt(video_path, output_dir, access_code, ckpt, proc_log),
            output_dir,
            access_code,
            ckpt,
            proc_log,
            log_file,
            ap["target_language"],
            intro_marker=MARKER_INTRO,
            outro_marker=MARKER_OUTRO,
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
            audio_subdir="audio",
        )
        run_video_ckpt(
            video_path,
            translated_srt,
            audio_out,
            output_dir,
            access_code,
            ckpt=ckpt,
            output_filename="output_modified.mp4",
            blur=(blur == "yes"),
        )
        _adjust_original_audio_nonfatal(video_path, translated_srt, audio_out, output_dir, access_code)

    validate_files(
        [
            audio_out["output_srt_path"],
            audio_out["output_wav_path"],
            os.path.join(output_dir, "output_modified.mp4"),
        ],
        label="宁视频OCR翻译",
    )
    job_log(access_code, output_dir, "Done!")


def process_video_ning_ocr(
    number: str,
    temperature: float,
    user_id: int = None,
    blur: str = "yes",
    target_language: str = "en",
    cfg_weight: float = 0.5,
    exaggeration: float = 0.5,
    cached_path: str | None = None,
) -> dict:
    jq = get_job_queue()
    existing = jq._find_failed_ocr_job(number, user_id)
    if existing:
        access_code, output_dir = existing
        jq.invalidate_checkpoints_after(access_code, "download")
        job_log_lines(
            access_code, output_dir, [f"--- resubmit (temperature={temperature}, lang={target_language}) ---"]
        )
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


def _run_video_ning_ocr_translate_only_job(job_data: dict):
    """Download, run OCR on full video, translate with marker trimming — stop after translate."""
    video_number = job_data["video_number"]
    access_code = job_data["access_code"]
    output_dir = job_data["output_dir"]
    target_language = job_data.get("target_language", "en")

    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "job.log")

    with open_proc_log(log_file) as (proc_log, _):
        valid_steps = ["download", "ocr", "translate"]
        ckpt = CheckpointHelper(access_code, output_dir, valid_steps)

        video_path = run_download_ckpt(video_number, output_dir, access_code, ckpt, proc_log, job_data)
        translated_srt = run_translate_ckpt(
            run_ocr_ckpt(video_path, output_dir, access_code, ckpt, proc_log),
            output_dir,
            access_code,
            ckpt,
            proc_log,
            log_file,
            target_language,
            intro_marker=MARKER_INTRO,
            outro_marker=MARKER_OUTRO,
        )
        if os.path.exists(translated_srt):
            shutil.copy2(translated_srt, os.path.join(output_dir, "output_adjusted.srt"))

    job_log(access_code, output_dir, "Done! OCR → translation complete (audio/video skipped)")


def process_video_ning_ocr_translate_only(
    number: str,
    temperature: float,
    user_id: int = None,
    blur: str = "yes",
    target_language: str = "en",
    cfg_weight: float = 0.5,
    exaggeration: float = 0.5,
    cached_path: str | None = None,
) -> dict:
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
