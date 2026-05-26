"""Ning-video job — video synthesis from a preset video number + SRT."""

import json
import os
import re
import subprocess
import sys
import time
from contextlib import redirect_stdout, redirect_stderr

from jobqueue import get_job_queue, JobStatus
from log_utils import job_log, redirect_logging_to_file
from config import VIDEO_DIR, GEN_VIDEO_ORIG_SCRIPT, RAPID_VIDEOCR_PIPELINE_SCRIPT, HY_MT_DIR, PROJECT_ROOT, RAPID_VIDEOCR_BIN
from pipeline import validate_files

_LANG_MAP = {
    "ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "fi": "Finnish", "fr": "French",
    "he": "Hebrew", "hi": "Hindi", "it": "Italian", "ja": "Japanese",
    "ko": "Korean", "ms": "Malay", "nl": "Dutch", "no": "Norwegian",
    "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "sv": "Swedish",
    "sw": "Swahili", "tr": "Turkish", "zh": "Chinese",
    "vi": "Vietnamese", "th": "Thai",
}


def _looks_untranslated(text: str, source_has_cjk: bool = True) -> bool:
    if not source_has_cjk:
        return False
    cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return cjk_count >= 3


def _run_video_job(job_data: dict):
    video_number = job_data["video_number"]
    access_code = job_data["access_code"]
    srt_path = job_data["srt_path"]
    output_dir = job_data["output_dir"]
    blur = job_data.get("blur", "yes")
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "job.log")
    proc_log = open(log_file, "a")

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
    proc_log.close()

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
    proc_log = open(log_file, "a")

    # ── Checkpoint helpers (persisted in jobs.db) ──────────────
    def _done(name: str) -> bool:
        ckpt = get_job_queue().get_checkpoint(access_code)
        steps = ["download", "decompress", "trim", "ocr", "translate", "audio", "video"]
        parts = ckpt.split(",") if ckpt else []
        return name in steps and name in parts

    def _mark(name: str):
        ckpt = get_job_queue().get_checkpoint(access_code)
        parts = ([s for s in ckpt.split(",") if s] if ckpt else []) + [name]
        get_job_queue().set_checkpoint(access_code, ",".join(parts))
        job_log(access_code, output_dir, f"  ✓ checkpoint {name}")

    # Step 1: Download the original video
    if not _done("download"):
        job_log(access_code, output_dir, "Downloading original video...")
        download_script = os.path.join(PROJECT_ROOT, "..", "pre-process", "download_orig.py")
        subprocess.run(
            ["/usr/bin/python3", download_script, video_number, output_dir],
            stdout=proc_log, stderr=proc_log, timeout=3600,
        )
        video_path = os.path.join(output_dir, f"{video_number}.mp4")
        if not os.path.exists(video_path):
            raise RuntimeError(f"Downloaded video not found: {video_path}")
        _mark("download")
    else:
        job_log(access_code, output_dir, "  ↪ download already done, skipping")

    # Step 2: Decompress video
    if not _done("decompress"):
        job_log(access_code, output_dir, "Decompressing video...")
        dcbl_script = os.path.join(PROJECT_ROOT, "..", "pre-process", "dcbl.sh")
        decompressed_path = os.path.join(output_dir, f"{video_number}_decompressed.mov")
        subprocess.run(
            ["/usr/bin/bash", dcbl_script, video_path, decompressed_path, blur],
            stdout=proc_log, stderr=proc_log, timeout=7200,
        )
        _mark("decompress")
    else:
        decompressed_path = os.path.join(output_dir, f"{video_number}_decompressed.mov")
        job_log(access_code, output_dir, "  ↪ decompress already done, skipping")

    # Step 2.5: Trim decompressed video — remove first 12.25s and last 45s
    if not _done("trim"):
        job_log(access_code, output_dir, "Trimming decompressed video (remove first 12.25s, last 45s)...")
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", decompressed_path],
            capture_output=True, text=True, timeout=60,
        )
        info = json.loads(result.stdout)
        orig_duration = float(info["format"]["duration"])
        start_trim = 12.25
        end_trim = 45.0
        trimmed_path = os.path.join(output_dir, f"{video_number}_trimmed.mp4")
        if orig_duration > start_trim + end_trim:
            end_time = orig_duration - end_trim
            subprocess.run(
                ["ffmpeg", "-y",
                 "-ss", str(start_trim), "-to", str(end_time),
                 "-i", decompressed_path,
                 "-c:v", "libx265", "-crf", "23", "-preset", "fast",
                 "-an",
                 "-r", "24",
                 trimmed_path],
                stdout=proc_log, stderr=proc_log, timeout=3600,
            )
            if os.path.exists(trimmed_path):
                decompressed_path = trimmed_path
                job_log(access_code, output_dir, f"Trimmed: {start_trim}s from start, {end_trim}s from end (was {orig_duration:.1f}s)")
            else:
                job_log(access_code, output_dir, "Trim failed, using decompressed video as-is")
        else:
            job_log(access_code, output_dir, f"Video too short ({orig_duration:.1f}s), skipping trim")
        _mark("trim")
    else:
        trimmed_path = os.path.join(output_dir, f"{video_number}_trimmed.mp4")
        if os.path.exists(trimmed_path):
            decompressed_path = trimmed_path
        job_log(access_code, output_dir, "  ↪ trim already done, skipping")

    # Step 3: Run rapid_videocr_pipeline.sh to generate OCR SRT
    ocr_srt = os.path.join(output_dir, "ocr_screen.srt")
    if not _done("ocr"):
        job_log(access_code, output_dir, "Running RapidVideOCR pipeline...")
        frames_dir = os.path.join(output_dir, "frames")
        subprocess.run(
            ["/usr/bin/bash", RAPID_VIDEOCR_PIPELINE_SCRIPT, "-i", decompressed_path,
             "-o", ocr_srt, "-d", frames_dir],
            stdout=proc_log, stderr=proc_log, timeout=14400,
            env={**os.environ, "RAPID_VIDEOCR_BIN": RAPID_VIDEOCR_BIN},
        )
        if not os.path.exists(ocr_srt):
            raise RuntimeError("RapidVideOCR pipeline failed to generate SRT")
        _mark("ocr")
    else:
        job_log(access_code, output_dir, "  ↪ OCR already done, skipping")

    # Step 4: Translate OCR SRT via HY-MT
    translated_srt = os.path.join(output_dir, "translated.srt")
    if not _done("translate"):
        job_log(access_code, output_dir, "Translating OCR subtitles...")
        sys.path.insert(0, HY_MT_DIR)
        import importlib
        hy_mt = importlib.import_module("hy_mt")
        target_language_name = _LANG_MAP.get(target_language, target_language)

        def _translate_segment(text: str) -> str:
            for attempt in range(3):
                if attempt == 0:
                    result = hy_mt.translate_zh(text, target_language_name)
                elif attempt == 1:
                    result = hy_mt.translate(text, target_language_name)
                else:
                    model, tokenizer = hy_mt._get_model()
                    messages = [
                        {"role": "user",
                         "content": f"Translate the following Chinese sentence into {target_language_name}. Output ONLY the {target_language_name} translation, nothing else:\n\n{text}"},
                    ]
                    tokenized_chat = tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=False, return_tensors="pt"
                    )
                    outputs = model.generate(tokenized_chat.to(model.device), **hy_mt.GENERATION_KWARGS)
                    result = tokenizer.decode(outputs[0][len(tokenized_chat[0]):], skip_special_tokens=True)
                if not _looks_untranslated(result):
                    return result
            return result

        with redirect_stdout(proc_log), redirect_stderr(proc_log), redirect_logging_to_file(log_file):
            with open(ocr_srt, "r", encoding="utf-8") as f:
                content = f.read()
            blocks = re.split(r"\n\n", content.strip())
            translated_blocks = []
            for block in blocks:
                lines = block.split("\n")
                if len(lines) >= 3:
                    idx = lines[0]
                    time_range = lines[1]
                    text = "\n".join(lines[2:])
                    translated = _translate_segment(text)
                    translated_blocks.append(f"{idx}\n{time_range}\n{translated}")
            with open(translated_srt, "w", encoding="utf-8") as f:
                f.write("\n\n".join(translated_blocks) + "\n")
            hy_mt.unload_model()
        _mark("translate")
    else:
        job_log(access_code, output_dir, "  ↪ translation already done, skipping")

    # Step 5: Generate audio from translated SRT
    audio_dir = os.path.join(output_dir, "audio")
    adjusted_srt = os.path.join(audio_dir, "output_adjusted.srt")
    if not _done("audio"):
        job_log(access_code, output_dir, "Generating audio from translated SRT...")
        gen_audio_script = os.path.join(PROJECT_ROOT, "gen_audio.py")
        audio_prompt = os.path.join(PROJECT_ROOT, "..", "assets", "std_ning.wav")
        subprocess.run(
            ["/home/js9s/.pyenv/versions/3.11.14/bin/python3.11", gen_audio_script, translated_srt,
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
        _mark("audio")
    else:
        job_log(access_code, output_dir, "  ↪ audio already done, skipping")

    # Step 6: Process video with stretched segments
    output_modified = os.path.join(output_dir, "output_modified.mp4")
    if not _done("video"):
        job_log(access_code, output_dir, "Processing video...")
        changed_json = os.path.join(audio_dir, "changed_segments.json")
        gen_video_script = os.path.join(PROJECT_ROOT, "gen_video.py")
        subprocess.run(
            ["/usr/bin/python3", gen_video_script, decompressed_path, translated_srt, adjusted_srt, changed_json,
             "--output", output_modified],
            stdout=proc_log, stderr=proc_log, timeout=7200,
        )
        _mark("video")
    else:
        job_log(access_code, output_dir, "  ↪ video already done, skipping")

    proc_log.close()

    validate_files([
        adjusted_srt,
        os.path.join(audio_dir, "output.wav"),
        output_modified,
    ], label="宁视频OCR翻译")
    job_log(access_code, output_dir, "Done!")


def process_video_ning_ocr(number: str, temperature: float, user_id: int = None, blur: str = "yes", target_language: str = "en", cfg_weight: float = 0.5, exaggeration: float = 0.5) -> dict:
    # Reuse an existing failed job for the same video+user so checkpoints carry over
    jq = get_job_queue()
    existing = jq._find_failed_ocr_job(number, user_id)
    if existing:
        access_code, output_dir = existing
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

    job_access_code = jq.add_job(job_data, _run_video_ning_ocr_job, user_id)
    return {"access_code": job_access_code, "message": "OCR translation job queued"}
