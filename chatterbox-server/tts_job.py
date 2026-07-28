"""TTS job — text-to-wave, queued and run in background."""

import os
import uuid

from config import AUDIO_TRACKS_DIR
from jobqueue import get_job_queue
from log_utils import job_log
from middleware import get_audio_params


def _run_tts_job(job_data: dict):
    from contextlib import redirect_stdout

    from audio_utils import NingAudio

    text = job_data["text"]
    output_dir = job_data["output_dir"]
    filename = job_data.get("filename", "output.wav")
    ap = get_audio_params(job_data)
    access_code = job_data["access_code"]
    os.makedirs(output_dir, exist_ok=True)

    job_log(access_code, output_dir, "Starting TTS...")
    log_file = os.path.join(output_dir, "job.log")
    with open(log_file, "a") as lf:
        with redirect_stdout(lf):
            wav_data = NingAudio().text_to_wave_with_silence(text, temperature=ap["temperature"], target_language=ap["target_language"], cfg_weight=ap["cfg_weight"], exaggeration=ap["exaggeration"])

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
