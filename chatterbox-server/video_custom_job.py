"""Custom/auto video jobs — user video + SRT, or auto-extract+translate+generate."""

import os
import re
import subprocess
import sys
import uuid
from contextlib import redirect_stdout, redirect_stderr

from jobqueue import get_job_queue
from log_utils import job_log, redirect_logging_to_file
from config import VIDEO_DIR, WHISPER_MODEL, HY_MT_DIR, RAPID_VIDEOCR_PIPELINE_SCRIPT
from pipeline import run_gen_audio_step, run_gen_video_step

# ── Language code → full name mapping for translation prompt ──
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
    """Heuristic: if source was CJK and output still has CJK, model likely refused to translate."""
    if not source_has_cjk:
        return False
    cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return cjk_count >= 3


def _run_video_custom_job(job_data: dict):
    video_file = job_data["video_file"]
    access_code = job_data["access_code"]
    srt_path = job_data["srt_path"]
    output_dir = job_data["output_dir"]

    os.makedirs(output_dir, exist_ok=True)
    job_log(access_code, output_dir, "Step 1: Generating audio from SRT")
    gen_audio_dir = os.path.join(output_dir, "audio_tracks")
    audio_out = run_gen_audio_step(srt_path, gen_audio_dir, job_data.get("temperature", 0.8), access_code,
                                   target_language=job_data.get("target_language", "en"),
                                   cfg_weight=job_data.get("cfg_weight", 0.5),
                                   exaggeration=job_data.get("exaggeration", 0.5))

    job_log(access_code, output_dir, "Step 2: Processing video")
    video_output = os.path.join(output_dir, "output_modified.mp4")
    run_gen_video_step(video_file, srt_path, audio_out["output_srt_path"], audio_out["changed_json_path"], video_output, access_code)
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
    proc_log = open(log_file, "a")

    # ── Checkpoint helpers (persisted in jobs.db) ──────────────
    def _done(name: str) -> bool:
        ckpt = get_job_queue().get_checkpoint(access_code)
        parts = ckpt.split(",") if ckpt else []
        return name in parts

    def _mark(name: str):
        ckpt = get_job_queue().get_checkpoint(access_code)
        parts = ([s for s in ckpt.split(",") if s] if ckpt else []) + [name]
        get_job_queue().set_checkpoint(access_code, ",".join(parts))
        job_log(access_code, output_dir, f"  ✓ checkpoint {name}")

    # Step 1: Extract audio from video
    audio_path = os.path.join(output_dir, "audio.wav")
    if not _done("extract_audio"):
        job_log(access_code, output_dir, "Extracting audio from video...")
        subprocess.run(
            ["ffmpeg", "-i", video_file, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path, "-y"],
            stdout=proc_log, stderr=proc_log, timeout=3600,
        )
        _mark("extract_audio")
    else:
        job_log(access_code, output_dir, "  ↪ extract_audio already done, skipping")

    # Step 2: Run whisper speech recognition
    whisper_srt = os.path.join(output_dir, "whisper.srt")
    if not _done("whisper"):
        job_log(access_code, output_dir, "Running whisper speech recognition...")
        subprocess.run(
            ["whisper-cli", "-m", WHISPER_MODEL, "-f", audio_path, "-osrt", "-of", whisper_srt.replace(".srt", ""), "-l", "zh"],
            stdout=proc_log, stderr=proc_log, timeout=7200,
        )
        if not os.path.exists(whisper_srt):
            raise RuntimeError("Whisper failed to generate SRT")
        _mark("whisper")
    else:
        job_log(access_code, output_dir, "  ↪ whisper already done, skipping")

    # Step 3: Translate subtitles
    translated_srt = os.path.join(output_dir, "translated.srt")
    if not _done("translate"):
        job_log(access_code, output_dir, "Translating subtitles...")
        sys.path.insert(0, HY_MT_DIR)
        import importlib
        hy_mt = importlib.import_module("hy_mt")
        target_language_name = _LANG_MAP.get(target_language, target_language)

        def _translate_segment(text: str) -> str:
            """Translate a segment with retry until output actually changes language."""
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
            with open(whisper_srt, "r", encoding="utf-8") as f:
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

    # Step 4: Generate audio from translated SRT
    gen_audio_dir = os.path.join(output_dir, "audio_tracks")
    if not _done("audio"):
        job_log(access_code, output_dir, "Step 1: Generating audio from translated SRT")
        audio_out = run_gen_audio_step(translated_srt, gen_audio_dir, temperature, access_code,
                                       target_language=target_language,
                                       cfg_weight=cfg_weight,
                                       exaggeration=exaggeration)
        _mark("audio")
    else:
        audio_out = {
            "output_srt_path": os.path.join(gen_audio_dir, "output_adjusted.srt"),
            "changed_json_path": os.path.join(gen_audio_dir, "changed_segments.json"),
        }
        job_log(access_code, output_dir, "  ↪ audio already done, skipping")

    # Step 5: Process video
    if not _done("video"):
        job_log(access_code, output_dir, "Step 2: Processing video")
        video_output = os.path.join(output_dir, "output_modified.mp4")
        run_gen_video_step(video_file, translated_srt, audio_out["output_srt_path"], audio_out["changed_json_path"], video_output, access_code)
        _mark("video")
    else:
        job_log(access_code, output_dir, "  ↪ video already done, skipping")

    proc_log.close()
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
    proc_log = open(log_file, "a")

    # ── Checkpoint helpers (persisted in jobs.db) ──────────────
    def _done(name: str) -> bool:
        ckpt = get_job_queue().get_checkpoint(access_code)
        steps = ["ocr", "translate", "audio", "video"]
        parts = ckpt.split(",") if ckpt else []
        return name in steps and name in parts

    def _mark(name: str):
        ckpt = get_job_queue().get_checkpoint(access_code)
        parts = ([s for s in ckpt.split(",") if s] if ckpt else []) + [name]
        get_job_queue().set_checkpoint(access_code, ",".join(parts))
        job_log(access_code, output_dir, f"  ✓ checkpoint {name}")

    # Step 1: Run rapid_videocr_pipeline.sh to generate OCR SRT
    ocr_srt = os.path.join(output_dir, "ocr_screen.srt")
    if not _done("ocr"):
        job_log(access_code, output_dir, "Running RapidVideOCR pipeline...")
        frames_dir = os.path.join(output_dir, "frames")
        subprocess.run(
            ["/usr/bin/bash", RAPID_VIDEOCR_PIPELINE_SCRIPT, "-i", video_file,
             "-o", ocr_srt, "-d", frames_dir],
            stdout=proc_log, stderr=proc_log, timeout=14400,
        )
        if not os.path.exists(ocr_srt):
            raise RuntimeError("RapidVideOCR pipeline failed to generate SRT")
        _mark("ocr")
    else:
        job_log(access_code, output_dir, "  ↪ OCR already done, skipping")

    # Step 2: Translate OCR SRT via HY-MT
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

    # Step 3: Generate audio from translated SRT
    if not _done("audio"):
        job_log(access_code, output_dir, "Step 3: Generating audio from translated SRT")
        gen_audio_dir = os.path.join(output_dir, "audio_tracks")
        audio_out = run_gen_audio_step(translated_srt, gen_audio_dir, temperature, access_code,
                                       target_language=target_language,
                                       cfg_weight=cfg_weight,
                                       exaggeration=exaggeration)
        _mark("audio")
    else:
        gen_audio_dir = os.path.join(output_dir, "audio_tracks")
        audio_out = {
            "output_srt_path": os.path.join(gen_audio_dir, "output_adjusted.srt"),
            "changed_json_path": os.path.join(gen_audio_dir, "changed_segments.json"),
        }
        job_log(access_code, output_dir, "  ↪ audio already done, skipping")

    # Step 4: Process video
    if not _done("video"):
        job_log(access_code, output_dir, "Step 4: Processing video")
        video_output = os.path.join(output_dir, "output_modified.mp4")
        run_gen_video_step(video_file, translated_srt, audio_out["output_srt_path"], audio_out["changed_json_path"], video_output, access_code)
        _mark("video")
    else:
        job_log(access_code, output_dir, "  ↪ video already done, skipping")

    proc_log.close()
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
