"""TTS job — text-to-wave, runs gen_audio.py subprocess on Python 3.11."""

import os
import subprocess
import uuid

from config import AUDIO_TRACKS_DIR, AUDIO_PROMPT_PATH, GEN_AUDIO_PYTHON, PROJECT_ROOT
from jobqueue import get_job_queue
from log_utils import job_log
from middleware import get_audio_params


def _run_tts_job(job_data: dict):
    text = job_data["text"]
    output_dir = job_data["output_dir"]
    filename = job_data.get("filename", "output.wav")
    ap = get_audio_params(job_data)
    access_code = job_data["access_code"]
    os.makedirs(output_dir, exist_ok=True)

    srt_path = os.path.join(output_dir, "input.srt")
    with open(srt_path, "w") as f:
        f.write(f"1\n00:00:00,000 --> 00:01:00,000\n{text}\n\n")

    gen_audio_script = os.path.join(PROJECT_ROOT, "gen_audio", "gen_audio.py")
    assets_dir = os.path.join(PROJECT_ROOT, "..", "assets")

    job_log(access_code, output_dir, "--- gen_audio (TTS) ---")
    cmd = [
        GEN_AUDIO_PYTHON, "-u", gen_audio_script, srt_path,
        "--audio_prompt", AUDIO_PROMPT_PATH,
        "--temperature", str(ap["temperature"]),
        "--output_dir", output_dir,
        "--assets_dir", assets_dir,
        "--target_language", ap["target_language"],
        "--cfg_weight", str(ap["cfg_weight"]),
        "--exaggeration", str(ap["exaggeration"]),
        "--output_wav", filename,
        "--output_srt", "output_adjusted.srt",
        "--changed_json", "changed_segments.json",
    ]

    log_path = os.path.join(output_dir, "job.log")
    with open(log_path, "a") as proc_log:
        proc_log.write(f"+ {' '.join(cmd)}\n")
        proc_log.flush()

    result = subprocess.run(cmd, stdout=open(log_path, "a"), stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"gen_audio exited with code {result.returncode}")

    job_log(access_code, output_dir, f"gen_audio subprocess completed")


def process_tts(
    text: str,
    filename: str,
    user_id: int = None,
    temperature: float = 0.8,
    target_language: str = "en",
    cfg_weight: float = 0.5,
    exaggeration: float = 0.5,
) -> dict:
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
