"""Custom/auto video jobs — user video + SRT, or auto-extract+translate+generate."""

import os
import subprocess
import uuid

from jobqueue import get_job_queue
from log_utils import job_log
from config import VIDEO_DIR, WHISPER_MODEL, WHISPER_OV_DEVICE, RAPID_VIDEOCR_PIPELINE_SCRIPT, LANG_MAP
from pipeline import run_audio_ckpt, run_video_ckpt, run_gen_audio_step
from video_util import CheckpointHelper, translate_srt_file, open_proc_log


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
    job_log(access_code, output_dir, "Done!")


def process_video_custom(video_file, srt_file, temperature: float, user_id: int = None, target_language: str = "en", cfg_weight: float = 0.5, exaggeration: float = 0.5) -> dict:
    access_code = str(uuid.uuid4())[:8].upper()
    output_dir = os.path.join(VIDEO_DIR, access_code)
    os.makedirs(output_dir, exist_ok=True)

    # Save under original filename (unique per access_code output_dir)
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

    with open_proc_log(log_file) as (proc_log, log_file):
        ckpt = CheckpointHelper(access_code, output_dir)

        # Step 1: Extract audio from video
        audio_path = os.path.join(output_dir, "audio.wav")
        if not ckpt.done("extract_audio"):
            job_log(access_code, output_dir, "Extracting audio from video...")
            subprocess.run(
                ["ffmpeg", "-i", video_file, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path, "-y"],
                stdout=proc_log, stderr=proc_log, timeout=3600,
            )
            ckpt.mark("extract_audio")
        else:
            job_log(access_code, output_dir, "  ↪ extract_audio already done, skipping")

        # Step 2: Run whisper speech recognition
        whisper_srt = os.path.join(output_dir, "whisper.srt")
        if not ckpt.done("whisper"):
            job_log(access_code, output_dir, f"Running whisper speech recognition (OpenVINO device={WHISPER_OV_DEVICE})...")
            subprocess.run(
                ["whisper-cli", "-m", WHISPER_MODEL, "-f", audio_path, "-osrt", "-of", whisper_srt.replace(".srt", ""), "-l", "zh",
                 "-oved", WHISPER_OV_DEVICE],
                stdout=proc_log, stderr=proc_log, timeout=7200,
            )
            if not os.path.exists(whisper_srt):
                raise RuntimeError("Whisper failed to generate SRT")
            ckpt.mark("whisper")
        else:
            job_log(access_code, output_dir, "  ↪ whisper already done, skipping")

        # Step 3: Translate subtitles
        translated_srt = os.path.join(output_dir, "translated.srt")
        if not ckpt.done("translate"):
            job_log(access_code, output_dir, "Translating subtitles...")
            target_language_name = LANG_MAP.get(target_language, target_language)
            translate_srt_file(whisper_srt, translated_srt, access_code, output_dir,
                               target_language_name, proc_log, log_file)
            ckpt.mark("translate")
        else:
            job_log(access_code, output_dir, "  ↪ translation already done, skipping")

        # Step 4: Generate audio from translated SRT
        audio_out = run_audio_ckpt(
            translated_srt, output_dir, temperature, access_code,
            target_language=target_language,
            cfg_weight=cfg_weight, exaggeration=exaggeration,
            ckpt=ckpt,
            audio_subdir="audio_tracks",
        )

        # Step 5: Process video
        run_video_ckpt(video_file, translated_srt, audio_out, output_dir, access_code,
                       ckpt=ckpt, output_filename="output_modified.mp4")

    job_log(access_code, output_dir, "Done!")


def process_video_auto(video_file, temperature: float, user_id: int = None, target_language: str = "en", cfg_weight: float = 0.5, exaggeration: float = 0.5) -> dict:
    access_code = str(uuid.uuid4())[:8].upper()
    output_dir = os.path.join(VIDEO_DIR, access_code)
    os.makedirs(output_dir, exist_ok=True)

    # Save under original filename (unique per access_code output_dir)
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

    with open_proc_log(log_file) as (proc_log, log_file):
        valid_steps = ["ocr", "translate", "audio", "video"]
        ckpt = CheckpointHelper(access_code, output_dir, valid_steps)

        # Step 1: Run rapid_videocr_pipeline.sh to generate OCR SRT
        ocr_srt = os.path.join(output_dir, "ocr_screen.srt")
        if not ckpt.done("ocr"):
            job_log(access_code, output_dir, "Running RapidVideOCR pipeline...")
            frames_dir = os.path.join(output_dir, "frames")
            subprocess.run(
                ["/usr/bin/bash", RAPID_VIDEOCR_PIPELINE_SCRIPT, "-i", video_file,
                 "-o", ocr_srt, "-d", frames_dir],
                stdout=proc_log, stderr=proc_log, timeout=14400,
            )
            if not os.path.exists(ocr_srt):
                raise RuntimeError("RapidVideOCR pipeline failed to generate SRT")
            ckpt.mark("ocr")
        else:
            job_log(access_code, output_dir, "  ↪ OCR already done, skipping")

        # Step 2: Translate OCR SRT via HY-MT
        translated_srt = os.path.join(output_dir, "translated.srt")
        if not ckpt.done("translate"):
            job_log(access_code, output_dir, "Translating OCR subtitles...")
            target_language_name = LANG_MAP.get(target_language, target_language)
            translate_srt_file(ocr_srt, translated_srt, access_code, output_dir,
                               target_language_name, proc_log, log_file)
            ckpt.mark("translate")
        else:
            job_log(access_code, output_dir, "  ↪ translation already done, skipping")

        # Step 3: Generate audio from translated SRT
        audio_out = run_audio_ckpt(
            translated_srt, output_dir, temperature, access_code,
            target_language=target_language,
            cfg_weight=cfg_weight, exaggeration=exaggeration,
            ckpt=ckpt,
            audio_subdir="audio_tracks",
        )

        # Step 4: Process video
        run_video_ckpt(video_file, translated_srt, audio_out, output_dir, access_code,
                       ckpt=ckpt, output_filename="output_modified.mp4")

    job_log(access_code, output_dir, "Done!")


def process_video_ocr(video_file, temperature: float, user_id: int = None, target_language: str = "en", cfg_weight: float = 0.5, exaggeration: float = 0.5) -> dict:
    access_code = str(uuid.uuid4())[:8].upper()
    output_dir = os.path.join(VIDEO_DIR, access_code)
    os.makedirs(output_dir, exist_ok=True)

    # Save under original filename (unique per access_code output_dir)
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
