"""Ning-video job — video synthesis from a preset video number + SRT."""

import json
import os
import shutil
import subprocess
import time
import uuid

from jobqueue import get_job_queue, JobStatus
from log_utils import job_log, job_log_lines
from config import VIDEO_DIR, GEN_VIDEO_ORIG_SCRIPT, RAPID_VIDEOCR_PIPELINE_SCRIPT, PROJECT_ROOT, RAPID_VIDEOCR_BIN, PYTHON_BIN, LANG_MAP
from pipeline import validate_files
from video_util import CheckpointHelper, translate_srt_file, open_proc_log


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
    blur = job_data.get("blur", "yes")
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "job.log")

    with open_proc_log(log_file) as (proc_log, _):
        process = subprocess.Popen(
            ["/usr/bin/bash", GEN_VIDEO_ORIG_SCRIPT, video_number, srt_path, output_dir,
             str(job_data.get("temperature", 0.8)), blur,
             str(job_data.get("target_language", "en")),
             str(job_data.get("cfg_weight", 0.5)),
             str(job_data.get("exaggeration", 0.5))],
            stdout=proc_log, stderr=proc_log,
        )
        start = time.monotonic()
        while True:
            try:
                process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                elapsed = int(time.monotonic() - start)
                get_job_queue().update_job_progress(access_code, f"正在执行视频处理脚本... ({elapsed // 60}分{elapsed % 60}秒)")

    audio_dir = os.path.join(output_dir, "audio")
    validate_files([
        os.path.join(audio_dir, "output_adjusted.srt"),
        os.path.join(audio_dir, "output.wav"),
        os.path.join(output_dir, "output_final.mp4"),
    ], label="宁视频翻译")


def process_video_ning(number: str, srt_file, temperature: float, user_id: int = None, blur: str = "yes", target_language: str = "en", cfg_weight: float = 0.5, exaggeration: float = 0.5) -> dict:
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

    job_access_code = get_job_queue().add_job(job_data, _run_video_job, user_id)

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
        valid_steps = ["download", "trim", "ocr", "translate", "audio", "video"]
        ckpt = CheckpointHelper(access_code, output_dir, valid_steps)

        # Step 1: Download (or use cached) original video
        if not ckpt.done("download"):
            cached_path = job_data.get("cached_path")
            if cached_path and os.path.isfile(cached_path):
                job_log(access_code, output_dir, f"Using cached video: {cached_path}")
                shutil.copy2(cached_path, os.path.join(output_dir, f"{video_number}.mp4"))
            else:
                job_log(access_code, output_dir, "Downloading original video...")
                download_script = os.path.join(PROJECT_ROOT, "..", "pre-process", "download_orig.py")
                subprocess.run(
                    [PYTHON_BIN, download_script, video_number, output_dir],
                    stdout=proc_log, stderr=proc_log, timeout=3600,
                )
            video_path = os.path.join(output_dir, f"{video_number}.mp4")
            if not os.path.exists(video_path):
                raise RuntimeError(f"Downloaded video not found: {video_path}")
            ckpt.mark("download")
        else:
            job_log(access_code, output_dir, "  ↪ download already done, skipping")
            video_path = os.path.join(output_dir, f"{video_number}.mp4")

        # Step 2: Trim video
        trimmed_path = os.path.join(output_dir, f"{video_number}_trimmed.mp4")
        if not ckpt.done("trim"):
            start_trim = float(job_data.get("start_trim", 12.25))
            end_trim = float(job_data.get("end_trim", 40.0))
            job_log(access_code, output_dir, f"Trimming video (remove first {start_trim}s, last {end_trim}s)...")
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
                capture_output=True, text=True, timeout=60,
            )
            info = json.loads(result.stdout)
            orig_duration = float(info["format"]["duration"])
            if orig_duration > start_trim + end_trim:
                end_time = orig_duration - end_trim
                subprocess.run(
                    ["ffmpeg", "-y",
                     "-ss", str(start_trim), "-to", str(end_time),
                     "-i", video_path,
                     "-c:v", "libx265", "-crf", "23", "-preset", "fast",
                     "-an",
                     "-r", "24",
                     trimmed_path],
                    stdout=proc_log, stderr=proc_log, timeout=3600,
                )
                if os.path.exists(trimmed_path):
                    job_log(access_code, output_dir, f"Trimmed: {start_trim}s from start, {end_trim}s from end (was {orig_duration:.1f}s)")
                else:
                    job_log(access_code, output_dir, "Trim failed, using original video as-is")
                    trimmed_path = video_path
            else:
                job_log(access_code, output_dir, f"Video too short ({orig_duration:.1f}s), skipping trim")
                trimmed_path = video_path
            ckpt.mark("trim")
        else:
            job_log(access_code, output_dir, "  ↪ trim already done, skipping")
            if not os.path.exists(trimmed_path):
                trimmed_path = video_path

        # Step 3: Run rapid_videocr_pipeline.sh to generate OCR SRT
        ocr_srt = os.path.join(output_dir, "ocr_screen.srt")
        if not ckpt.done("ocr"):
            job_log(access_code, output_dir, "Running RapidVideOCR pipeline...")
            frames_dir = os.path.join(output_dir, "frames")
            subprocess.run(
                ["/usr/bin/bash", RAPID_VIDEOCR_PIPELINE_SCRIPT, "-i", trimmed_path,
                 "-o", ocr_srt, "-d", frames_dir],
                stdout=proc_log, stderr=proc_log, timeout=14400,
                env={**os.environ, "RAPID_VIDEOCR_BIN": RAPID_VIDEOCR_BIN},
            )
            if not os.path.exists(ocr_srt):
                raise RuntimeError("RapidVideOCR pipeline failed to generate SRT")
            ckpt.mark("ocr")
        else:
            job_log(access_code, output_dir, "  ↪ OCR already done, skipping")

        # Step 4: Translate OCR SRT via HY-MT
        translated_srt = os.path.join(output_dir, "translated.srt")
        if not ckpt.done("translate"):
            job_log(access_code, output_dir, "Translating OCR subtitles...")
            target_language_name = LANG_MAP.get(target_language, target_language)
            translate_srt_file(ocr_srt, translated_srt, access_code, output_dir,
                               target_language_name, proc_log, log_file)
            ckpt.mark("translate")
        else:
            job_log(access_code, output_dir, "  ↪ translation already done, skipping")

        # Step 5: Generate audio from translated SRT
        audio_dir = os.path.join(output_dir, "audio")
        adjusted_srt = os.path.join(audio_dir, "output_adjusted.srt")
        if not ckpt.done("audio"):
            job_log(access_code, output_dir, "Generating audio from translated SRT...")
            gen_audio_script = os.path.join(PROJECT_ROOT, "gen_audio.py")
            audio_prompt = os.path.join(PROJECT_ROOT, "..", "assets", "std_ning.wav")
            subprocess.run(
                [PYTHON_BIN, gen_audio_script, translated_srt,
                 "--audio_prompt", audio_prompt,
                 "--output_dir", audio_dir,
                 "--output_srt", "output_adjusted.srt",
                 "--output_wav", "output.wav",
                 "--changed_json", "changed_segments.json",
                 "--temperature", str(temperature),
                 "--target_language", target_language,
                 "--cfg_weight", str(cfg_weight),
                 "--exaggeration", str(exaggeration)],
                stdout=proc_log, stderr=proc_log, timeout=7200,
            )
            ckpt.mark("audio")
        else:
            job_log(access_code, output_dir, "  ↪ audio already done, skipping")

        # Step 6: Process video with stretched segments
        output_modified = os.path.join(output_dir, "output_modified.mp4")
        if not ckpt.done("video"):
            job_log(access_code, output_dir, "Processing video...")
            changed_json = os.path.join(audio_dir, "changed_segments.json")
            gen_video_script = os.path.join(PROJECT_ROOT, "gen_video.py")
            cmd = [PYTHON_BIN, gen_video_script, trimmed_path, translated_srt, adjusted_srt, changed_json,
                   "--output", output_modified]
            if blur == "yes":
                cmd.append("--blur")
            subprocess.run(
                cmd,
                stdout=proc_log, stderr=proc_log, timeout=7200,
            )
            ckpt.mark("video")
        else:
            job_log(access_code, output_dir, "  ↪ video already done, skipping")

    validate_files([
        adjusted_srt,
        os.path.join(audio_dir, "output.wav"),
        output_modified,
    ], label="宁视频OCR翻译")
    job_log(access_code, output_dir, "Done!")


def process_video_ning_ocr(number: str, temperature: float, user_id: int = None, blur: str = "yes", target_language: str = "en", cfg_weight: float = 0.5, exaggeration: float = 0.5, start_trim: float = 12.25, end_trim: float = 40.0, cached_path: str | None = None) -> dict:
    # Reuse an existing failed job for the same video+user so checkpoints carry over
    jq = get_job_queue()
    existing = jq._find_failed_ocr_job(number, user_id)
    if existing:
        access_code, output_dir = existing
        # Invalidate checkpoints from "trim" onward so the job re-runs
        # trimming with the (potentially new) start_trim/end_trim values.
        jq.invalidate_checkpoints_after(access_code, "download")
        job_log_lines(access_code, output_dir, [f"--- resubmit (temperature={temperature}, lang={target_language}, start_trim={start_trim}, end_trim={end_trim}) ---"])
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
        "start_trim": start_trim,
        "end_trim": end_trim,
    }

    if cached_path:
        job_data["cached_path"] = cached_path

    job_access_code = jq.add_job(job_data, _run_video_ning_ocr_job, user_id)
    return {"access_code": job_access_code, "message": "OCR translation job queued"}
