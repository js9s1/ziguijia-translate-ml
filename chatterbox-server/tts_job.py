"""TTS job — text-to-wave, queued and run in background."""

import os
import uuid

from jobqueue import get_job_queue
from log_utils import job_log
from config import AUDIO_TRACKS_DIR


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

    job_log(access_code, output_dir, "Starting TTS...")
    log_file = os.path.join(output_dir, "job.log")
    with open(log_file, "a") as lf:
        with redirect_stdout(lf):
            wav_data = NingAudio().text_to_wave(text, temperature=temperature, target_language=target_language, cfg_weight=cfg_weight, exaggeration=exaggeration)

    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
        f.write(wav_data.read())
    job_log(access_code, output_dir, f"Wrote {output_path}")


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
