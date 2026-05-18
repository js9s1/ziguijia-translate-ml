import logging
import os
import subprocess
import sys
import uuid

from jobqueue import get_job_queue
from config import (
    AUDIO_TRACKS_DIR,
    VIDEO_DIR,
    GEN_VIDEO_ORIG_SCRIPT,
    WHISPER_MODEL,
    HY_MT_DIR,
)
from pipeline import run_gen_audio_step, run_gen_video_step, validate_files

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

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


def _run_video_job(job_data: dict):
    video_number = job_data["video_number"]
    access_code = job_data["access_code"]
    srt_path = job_data["srt_path"]
    output_dir = job_data["output_dir"]
    blur = job_data.get("blur", "yes")
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "job.log")
    job_log = open(log_file, "a")

    import time
    process = subprocess.Popen(
        ["/usr/bin/bash", GEN_VIDEO_ORIG_SCRIPT, video_number, srt_path, output_dir,
         str(job_data.get("temperature", 0.8)), blur,
         str(job_data.get("target_language", "en")),
         str(job_data.get("cfg_weight", 0.5)),
         str(job_data.get("exaggeration", 0.5))],
        stdout=job_log, stderr=job_log,
    )
    start = time.monotonic()
    while True:
        try:
            process.wait(timeout=30)
            break
        except subprocess.TimeoutExpired:
            elapsed = int(time.monotonic() - start)
            get_job_queue().update_job_progress(access_code, f"正在执行视频处理脚本... ({elapsed // 60}分{elapsed % 60}秒)")
    job_log.close()

    audio_dir = os.path.join(output_dir, "audio")
    validate_files([
        os.path.join(audio_dir, "output_adjusted.srt"),
        os.path.join(audio_dir, "output.wav"),
        os.path.join(output_dir, "output_final.mp4"),
    ], label="宁视频翻译")


def _run_gen_audio(job_data: dict):
    run_gen_audio_step(
        srt_path=job_data["srt_path"],
        output_dir=job_data["output_dir"],
        temperature=job_data.get("temperature", 0.8),
        access_code=job_data["access_code"],
        target_language=job_data.get("target_language", "en"),
        cfg_weight=job_data.get("cfg_weight", 0.5),
        exaggeration=job_data.get("exaggeration", 0.5),
    )


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


def _run_video_custom_job(job_data: dict):
    video_file = job_data["video_file"]
    access_code = job_data["access_code"]
    srt_path = job_data["srt_path"]
    output_dir = job_data["output_dir"]
    log_file = os.path.join(output_dir, "job.log")

    def log(msg):
        with open(log_file, "a") as f:
            f.write(msg + "\n")
            f.flush()
        logger.info(f"[Job {access_code}] {msg}")
        get_job_queue().update_job_progress(access_code, msg)

    os.makedirs(output_dir, exist_ok=True)
    log("Step 1: Generating audio from SRT")
    gen_audio_dir = os.path.join(output_dir, "audio_tracks")
    audio_out = run_gen_audio_step(srt_path, gen_audio_dir, job_data.get("temperature", 0.8), access_code,
                                   target_language=job_data.get("target_language", "en"),
                                   cfg_weight=job_data.get("cfg_weight", 0.5),
                                   exaggeration=job_data.get("exaggeration", 0.5))

    log("Step 2: Processing video")
    video_output = os.path.join(output_dir, "output_modified.mp4")
    run_gen_video_step(video_file, srt_path, audio_out["output_srt_path"], audio_out["changed_json_path"], video_output, access_code)
    log("Done!")


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


def _run_tts_job(job_data: dict):
    from audio_utils import NingAudio
    from contextlib import redirect_stdout

    text = job_data["text"]
    output_dir = job_data["output_dir"]
    filename = job_data.get("filename", "output.wav")
    temperature = job_data.get("temperature", 0.8)
    target_language = job_data.get("target_language", "en")
    cfg_weight = job_data.get("cfg_weight", 0.5)
    exaggeration = job_data.get("exaggeration", 0.5)
    access_code = job_data["access_code"]
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "job.log")
    with open(log_file, "w") as lf:
        lf.write(f"[Job {access_code}] Starting TTS...\n")
        with redirect_stdout(lf):
            wav_data = NingAudio().text_to_wave(text, temperature=temperature, target_language=target_language, cfg_weight=cfg_weight, exaggeration=exaggeration)

    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
        f.write(wav_data.read())
    logger.info(f"TTS job {access_code} wrote {output_path}")


def process_tts(text: str, filename: str, user_id: int = None, temperature: float = 0.8, target_language: str = "en", cfg_weight: float = 0.5, exaggeration: float = 0.5) -> dict:
    access_code = str(uuid.uuid4())[:8].upper()
    output_dir = os.path.join(AUDIO_TRACKS_DIR, access_code)
    os.makedirs(output_dir, exist_ok=True)

    job_data = {
        "access_code": access_code,
        "text": text,
        "output_dir": output_dir,
        "filename": filename,
        "temperature": temperature,
        "target_language": target_language,
        "cfg_weight": cfg_weight,
        "exaggeration": exaggeration,
    }

    job_access_code = get_job_queue().add_job(job_data, _run_tts_job, user_id)
    return {"access_code": job_access_code, "message": "Job queued successfully"}


def _run_video_auto_job(job_data: dict):
    import re
    video_file = job_data["video_file"]
    access_code = job_data["access_code"]
    output_dir = job_data["output_dir"]
    temperature = job_data.get("temperature", 0.8)
    target_language = job_data.get("target_language", "en")
    cfg_weight = job_data.get("cfg_weight", 0.5)
    exaggeration = job_data.get("exaggeration", 0.5)
    log_file = os.path.join(output_dir, "job.log")

    def log(msg):
        with open(log_file, "a") as f:
            f.write(msg + "\n")
            f.flush()
        logger.info(f"[Job {access_code}] {msg}")
        get_job_queue().update_job_progress(access_code, msg)

    os.makedirs(output_dir, exist_ok=True)

    import time
    log_file = os.path.join(output_dir, "job.log")
    job_log = open(log_file, "a")

    def log(msg):
        print(msg, file=job_log, flush=True)
        logger.info(f"[Job {access_code}] {msg}")
        get_job_queue().update_job_progress(access_code, msg)

    log("Extracting audio from video...")
    audio_path = os.path.join(output_dir, "audio.wav")
    subprocess.run(
        ["ffmpeg", "-i", video_file, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path, "-y"],
        stdout=job_log, stderr=job_log, timeout=3600,
    )

    log("Running whisper speech recognition...")
    whisper_srt = os.path.join(output_dir, "whisper.srt")
    subprocess.run(
        ["whisper-cli", "-m", WHISPER_MODEL, "-f", audio_path, "-osrt", "-of", whisper_srt.replace(".srt", ""), "-l", "zh"],
        stdout=job_log, stderr=job_log, timeout=7200,
    )
    if not os.path.exists(whisper_srt):
        raise RuntimeError("Whisper failed to generate SRT")

    log("Translating subtitles...")
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
    translated_srt = os.path.join(output_dir, "translated.srt")
    with open(translated_srt, "w", encoding="utf-8") as f:
        f.write("\n\n".join(translated_blocks) + "\n")
    hy_mt.unload_model()

    log("Step 1: Generating audio from translated SRT")
    gen_audio_dir = os.path.join(output_dir, "audio_tracks")
    audio_out = run_gen_audio_step(translated_srt, gen_audio_dir, temperature, access_code,
                                   target_language=target_language,
                                   cfg_weight=cfg_weight,
                                   exaggeration=exaggeration)

    log("Step 2: Processing video")
    video_output = os.path.join(output_dir, "output_modified.mp4")
    run_gen_video_step(video_file, translated_srt, audio_out["output_srt_path"], audio_out["changed_json_path"], video_output, access_code)
    log("Done!")


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


def _split_text(text, max_len=500):
    if len(text) <= max_len:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break

        chunk = text[:max_len]
        last_boundary = -1
        for sep in ('. ', '! ', '? ', '.\n', '!\n', '?\n', '.\r\n', '!\r\n', '?\r\n'):
            pos = chunk.rfind(sep)
            if pos > last_boundary:
                last_boundary = pos + 1

        if last_boundary > 0:
            chunks.append(text[:last_boundary])
            text = text[last_boundary:].lstrip()
        else:
            last_space = chunk.rfind(' ')
            if last_space > 0:
                chunks.append(text[:last_space])
                text = text[last_space:].lstrip()
            else:
                chunks.append(chunk)
                text = text[max_len:].lstrip()

    return chunks


def _run_audio_segmentation_job(job_data: dict):
    import re
    import torch
    import torchaudio as ta
    from audio_utils import NingAudio

    content = job_data["content"]
    output_dir = job_data["output_dir"]
    filename = job_data.get("filename", "output.wav")
    temperature = job_data.get("temperature", 0.8)
    target_language = job_data.get("target_language", "en")
    cfg_weight = job_data.get("cfg_weight", 0.5)
    exaggeration = job_data.get("exaggeration", 0.5)
    os.makedirs(output_dir, exist_ok=True)

    ning = NingAudio()
    ning.setup()
    sample_rate = ning.sample_rate

    pattern = r'<(\d+(?:\.\d+)?)>\s*'
    parts = re.split(pattern, content)

    segments = []
    first_text = parts[0].strip() if parts and parts[0].strip() else ""
    if first_text:
        segments.append((0, first_text))

    i = 1
    while i < len(parts) - 1:
        silence_sec = float(parts[i])
        text = parts[i + 1].strip()
        if text:
            segments.append((silence_sec, text))
        i += 2

    if not segments:
        raise ValueError("No text content found in file")

    all_audio_parts = []
    for silence_sec, text in segments:
        chunks = _split_text(text, 500)
        for chunk in chunks:
            wav_bytes = ning.text_to_wave(chunk, temperature=temperature,
                                          target_language=target_language,
                                          cfg_weight=cfg_weight,
                                          exaggeration=exaggeration)
            wav, sr = ta.load(wav_bytes)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            all_audio_parts.append(wav)
        if silence_sec > 0:
            silence = ning.generate_silence(silence_sec, sample_rate)
            all_audio_parts.append(silence)

    combined = torch.cat(all_audio_parts, dim=1)

    output_path = os.path.join(output_dir, filename)
    ta.save(output_path, combined, sample_rate)
    logger.info(f"Audio segmentation job {job_data['access_code']} wrote {output_path}")


def process_audio_file(content: str, original_filename: str, temperature: float, user_id: int = None, target_language: str = "en", cfg_weight: float = 0.5, exaggeration: float = 0.5) -> dict:
    access_code = str(uuid.uuid4())[:8].upper()
    output_dir = os.path.join(AUDIO_TRACKS_DIR, access_code)
    os.makedirs(output_dir, exist_ok=True)

    job_data = {
        "access_code": access_code,
        "content": content,
        "output_dir": output_dir,
        "filename": "output.wav",
        "temperature": temperature,
        "target_language": target_language,
        "cfg_weight": cfg_weight,
        "exaggeration": exaggeration,
    }

    job_access_code = get_job_queue().add_job(job_data, _run_audio_segmentation_job, user_id)
    return {"access_code": job_access_code, "message": "Job queued successfully"}


def process_srt_file(srt_file, temperature: float, user_id: int = None, target_language: str = "en", cfg_weight: float = 0.5, exaggeration: float = 0.5) -> dict:
    access_code = str(uuid.uuid4())[:8].upper()
    output_dir = os.path.join(AUDIO_TRACKS_DIR, access_code)
    os.makedirs(output_dir, exist_ok=True)

    srt_path = os.path.join(output_dir, srt_file.filename)
    srt_file.save(srt_path)

    job_data = {
        "access_code": access_code,
        "srt_path": srt_path,
        "output_dir": output_dir,
        "temperature": temperature,
        "target_language": target_language,
        "cfg_weight": cfg_weight,
        "exaggeration": exaggeration,
    }

    job_access_code = get_job_queue().add_job(job_data, _run_gen_audio, user_id)

    return {"access_code": job_access_code, "message": "Job queued successfully"}