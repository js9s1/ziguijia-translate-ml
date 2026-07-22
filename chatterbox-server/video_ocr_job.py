"""OCR-only job — extracts subtitles from video using rapid_videocr_pipeline.sh"""

import os
import subprocess
import uuid

from config import RAPID_VIDEOCR_PIPELINE_SCRIPT, VIDEO_DIR
from jobqueue import get_job_queue
from log_utils import job_log
from video_util import open_proc_log, read_srt_text


def _detect_srt_language(srt_path: str) -> str:
    """Read the SRT and detect the dominant language via Unicode ranges.

    Returns a language code (zh, ja, ko, ar, ru, el, he, hi, th, en) or 'en' as fallback.
    """
    from language_utils import detect_dominant_script, is_srt_index_line, is_srt_timing_line

    try:
        content = read_srt_text(srt_path)
    except Exception:
        return "en"

    # Remove SRT timing lines and indices — keep only text lines
    text_lines = []
    for line in content.splitlines():
        line = line.strip()
        if not line or is_srt_index_line(line) or is_srt_timing_line(line):
            continue
        text_lines.append(line)
    text = "\n".join(text_lines)
    if not text.strip():
        return "en"

    return detect_dominant_script(text)


def _run_ocr_only_job(job_data: dict):
    """Run the rapid_videocr_pipeline.sh on the video and produce an SRT."""
    video_file = job_data["video_file"]
    access_code = job_data["access_code"]
    output_dir = job_data["output_dir"]

    # Derive output SRT name from the video filename
    video_basename = os.path.basename(video_file)
    base = os.path.splitext(video_basename)[0]
    output_srt_name = job_data.get("output_srt_name", base + "_ocr.srt")

    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "job.log")

    with open_proc_log(log_file) as (proc_log, _):
        job_log(access_code, output_dir, "Running RapidVideOCR pipeline to extract subtitles...")

        ocr_srt = os.path.join(output_dir, output_srt_name)
        frames_dir = os.path.join(output_dir, "frames")

        result = subprocess.run(
            ["/usr/bin/bash", RAPID_VIDEOCR_PIPELINE_SCRIPT, "-i", video_file,
             "-o", ocr_srt, "-d", frames_dir],
            stdout=proc_log, stderr=proc_log, timeout=14400,
        )

    if result.returncode != 0:
        raise RuntimeError(f"RapidVideOCR pipeline failed (exit {result.returncode})")

    if not os.path.exists(ocr_srt):
        raise RuntimeError("RapidVideOCR pipeline completed but SRT not found")

    job_log(access_code, output_dir, f"Done! SRT saved to: {ocr_srt}")

    # Detect the dominant language in the OCR'd SRT and store it
    detected_lang = _detect_srt_language(ocr_srt)
    job_log(access_code, output_dir, f"Detected language: {detected_lang}")
    jq = get_job_queue()
    jq.update_target_language(access_code, detected_lang)


def process_ocr_only(video_file, user_id: int = None) -> dict:
    """Accept an uploaded video file, queue an OCR-only job, return the access code."""
    access_code = str(uuid.uuid4())[:8].upper()
    output_dir = os.path.join(VIDEO_DIR, access_code + "_ocr")
    os.makedirs(output_dir, exist_ok=True)

    # Keep the original filename (replace extension with .mp4 if needed for safety)
    orig_name = video_file.filename or "video.mp4"
    video_ext = os.path.splitext(orig_name)[1] or ".mp4"
    safe_name = orig_name  # keep original name
    video_path = os.path.join(output_dir, safe_name)
    video_file.save(video_path)

    # Output SRT name: same base name, _ocr.srt suffix
    base = os.path.splitext(orig_name)[0]
    output_srt_name = base + "_ocr.srt"

    job_data = {
        "access_code": access_code,
        "video_file": video_path,
        "output_dir": output_dir,
        "output_srt_name": output_srt_name,
        "user_id": user_id,
    }

    job_access_code = get_job_queue().add_job(job_data, _run_ocr_only_job, user_id)
    return {"access_code": job_access_code, "message": "OCR job queued"}
