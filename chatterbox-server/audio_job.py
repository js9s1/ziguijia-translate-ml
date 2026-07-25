"""Audio-from-SRT and audio-segmentation jobs."""

import os
import re
import uuid

import torch
import torchaudio as ta
from audio_utils import NingAudio
from config import AUDIO_TRACKS_DIR
from jobqueue import get_job_queue
from log_utils import job_log
from pipeline import run_gen_audio_step


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
    content = job_data["content"]
    output_dir = job_data["output_dir"]
    filename = job_data.get("filename", "output.wav")
    temperature = job_data.get("temperature", 0.8)
    target_language = job_data.get("target_language", "en")
    cfg_weight = job_data.get("cfg_weight", 0.5)
    exaggeration = job_data.get("exaggeration", 0.5)
    os.makedirs(output_dir, exist_ok=True)

    ning = NingAudio()
    ning._ensure_model(target_language)
    if target_language == "id":
        import gpu_manage as _gm
        sample_rate = _gm._indonesian_model.sr
    else:
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
    job_log(job_data['access_code'], output_dir, f"Wrote {output_path}")


def process_audio_file(content: str, original_filename: str, temperature: float, user_id: int = None, target_language: str = "en", cfg_weight: float = 0.5, exaggeration: float = 0.5) -> dict:
    access_code = str(uuid.uuid4())[:8].upper()
    output_dir = os.path.join(AUDIO_TRACKS_DIR, access_code)
    os.makedirs(output_dir, exist_ok=True)

    # Use user-supplied filename, falling back to a unique name derived from the
    # original filename so concurrent submissions don't overwrite each other.
    if original_filename:
        base, ext = os.path.splitext(original_filename)
        if not ext:
            ext = ".wav"
        filename = f"{base}{ext}"
    else:
        filename = f"output_{access_code}.wav"

    job_data = {
        "access_code": access_code,
        "content": content,
        "output_dir": output_dir,
        "filename": filename,
        "temperature": temperature,
        "target_language": target_language,
        "cfg_weight": cfg_weight,
        "exaggeration": exaggeration,
    }

    job_access_code = get_job_queue().add_job(job_data, _run_audio_segmentation_job, user_id)
    return {"access_code": job_access_code, "message": "Job queued successfully"}
