"""Custom/auto video jobs — user video + SRT, or auto-extract+translate+generate."""

import os
import uuid

from config import (
    MARKER_INTRO,
    MARKER_OUTRO,
    VIDEO_DIR,
)
from jobqueue import get_job_queue
from log_utils import job_log
from pipeline import (
    _adjust_original_audio_nonfatal,
    run_audio_ckpt,
    run_extract_audio_ckpt,
    run_gen_audio_step,
    run_ocr_ckpt,
    run_translate_ckpt,
    run_video_ckpt,
    run_whisper_ckpt,
)
from video_util import CheckpointHelper, open_proc_log


def _run_video_custom_job(job_data: dict):
    video_file = job_data["video_file"]
    access_code = job_data["access_code"]
    srt_path = job_data["srt_path"]
    output_dir = job_data["output_dir"]

    os.makedirs(output_dir, exist_ok=True)
    gen_audio_dir = os.path.join(output_dir, "audio_tracks")
    job_log(access_code, output_dir, "Step 1: Generating audio from SRT")
    audio_out = run_gen_audio_step(srt_path, gen_audio_dir, job_data.get("temperature", 0.8), access_code,
                                   target_language=job_data.get("target_language", "en"),
                                   cfg_weight=job_data.get("cfg_weight", 0.5),
                                   exaggeration=job_data.get("exaggeration", 0.5))

    job_log(access_code, output_dir, "Step 2: Processing video")
    run_video_ckpt(video_file, srt_path, audio_out, output_dir, access_code,
                   output_filename="output_modified.mp4")
    _adjust_original_audio_nonfatal(video_file, srt_path, audio_out, output_dir, access_code)
    job_log(access_code, output_dir, "Done!")


def process_video_custom(video_file, srt_file, temperature: float, user_id: int = None, target_language: str = "en", cfg_weight: float = 0.5, exaggeration: float = 0.5) -> dict:
    access_code = str(uuid.uuid4())[:8].upper()
    output_dir = os.path.join(VIDEO_DIR, access_code)
    os.makedirs(output_dir, exist_ok=True)

    video_path = os.path.join(output_dir, video_file.filename)
    video_file.save(video_path)

    srt_path = os.path.join(output_dir, srt_file.filename)
    srt_file.save(srt_path)

    job_data = {
        "video_file": video_path,
        "srt_path": srt_path,
        "output_dir": output_dir,
        "access_code": access_code,
        "temperature": temperature,
        "target_language": target_language,
        "cfg_weight": cfg_weight,
        "exaggeration": exaggeration,
    }
    job_access_code = get_job_queue().add_job(job_data, _run_video_custom_job, user_id)
    return {"access_code": job_access_code, "message": "Job queued successfully"}


def _run_video_auto_job(job_data: dict):
    video_file = job_data["video_file"]
    access_code = job_data["access_code"]
    output_dir = job_data["output_dir"]
    temperature = job_data.get("temperature", 0.8)
    target_language = job_data.get("target_language", "en")
    cfg_weight = job_data.get("cfg_weight", 0.5)
    exaggeration = job_data.get("exaggeration", 0.5)

    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "job.log")

    with open_proc_log(log_file) as (proc_log, _):
        ckpt = CheckpointHelper(access_code, output_dir)

        audio_path = run_extract_audio_ckpt(video_file, output_dir, access_code, ckpt, proc_log)
        translated_srt = run_translate_ckpt(
            run_whisper_ckpt(audio_path, output_dir, access_code, ckpt, proc_log),
            output_dir, access_code, ckpt, proc_log, log_file, target_language,
            intro_marker=MARKER_INTRO, outro_marker=MARKER_OUTRO,
        )
        audio_out = run_audio_ckpt(
            translated_srt, output_dir, temperature, access_code,
            target_language=target_language, cfg_weight=cfg_weight,
            exaggeration=exaggeration, ckpt=ckpt, audio_subdir="audio_tracks",
        )
        run_video_ckpt(video_file, translated_srt, audio_out, output_dir, access_code,
                       ckpt=ckpt, output_filename="output_modified.mp4")
        _adjust_original_audio_nonfatal(video_file, translated_srt, audio_out, output_dir, access_code)

    job_log(access_code, output_dir, "Done!")


def process_video_auto(video_file, temperature: float, user_id: int = None, target_language: str = "en", cfg_weight: float = 0.5, exaggeration: float = 0.5) -> dict:
    access_code = str(uuid.uuid4())[:8].upper()
    output_dir = os.path.join(VIDEO_DIR, access_code)
    os.makedirs(output_dir, exist_ok=True)

    video_path = os.path.join(output_dir, video_file.filename)
    video_file.save(video_path)

    job_data = {
        "access_code": access_code,
        "video_file": video_path,
        "output_dir": output_dir,
        "temperature": temperature,
        "user_id": user_id,
        "target_language": target_language,
        "cfg_weight": cfg_weight,
        "exaggeration": exaggeration,
    }
    job_access_code = get_job_queue().add_job(job_data, _run_video_auto_job, user_id)
    return {"access_code": job_access_code, "message": "Auto translation job queued"}


def _run_video_ocr_job(job_data: dict):
    video_file = job_data["video_file"]
    access_code = job_data["access_code"]
    output_dir = job_data["output_dir"]
    temperature = job_data.get("temperature", 0.8)
    target_language = job_data.get("target_language", "en")
    cfg_weight = job_data.get("cfg_weight", 0.5)
    exaggeration = job_data.get("exaggeration", 0.5)

    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "job.log")

    with open_proc_log(log_file) as (proc_log, _):
        valid_steps = ["ocr", "translate", "audio", "video"]
        ckpt = CheckpointHelper(access_code, output_dir, valid_steps)

        translated_srt = run_translate_ckpt(
            run_ocr_ckpt(video_file, output_dir, access_code, ckpt, proc_log),
            output_dir, access_code, ckpt, proc_log, log_file, target_language,
            intro_marker=MARKER_INTRO, outro_marker=MARKER_OUTRO,
        )
        audio_out = run_audio_ckpt(
            translated_srt, output_dir, temperature, access_code,
            target_language=target_language, cfg_weight=cfg_weight,
            exaggeration=exaggeration, ckpt=ckpt, audio_subdir="audio_tracks",
        )
        run_video_ckpt(video_file, translated_srt, audio_out, output_dir, access_code,
                       ckpt=ckpt, output_filename="output_modified.mp4")
        _adjust_original_audio_nonfatal(video_file, translated_srt, audio_out, output_dir, access_code)

    job_log(access_code, output_dir, "Done!")


def process_video_ocr(video_file, temperature: float, user_id: int = None, target_language: str = "en", cfg_weight: float = 0.5, exaggeration: float = 0.5) -> dict:
    access_code = str(uuid.uuid4())[:8].upper()
    output_dir = os.path.join(VIDEO_DIR, access_code)
    os.makedirs(output_dir, exist_ok=True)

    video_path = os.path.join(output_dir, video_file.filename)
    video_file.save(video_path)

    job_data = {
        "access_code": access_code,
        "video_file": video_path,
        "output_dir": output_dir,
        "temperature": temperature,
        "user_id": user_id,
        "target_language": target_language,
        "cfg_weight": cfg_weight,
        "exaggeration": exaggeration,
    }
    job_access_code = get_job_queue().add_job(job_data, _run_video_ocr_job, user_id)
    return {"access_code": job_access_code, "message": "OCR translation job queued"}
