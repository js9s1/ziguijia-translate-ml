"""Audio-from-SRT and audio-segmentation jobs."""

import os
import re
import uuid

from config import AUDIO_TRACKS_DIR
from jobqueue import get_job_queue
from log_utils import job_log
from middleware import get_audio_params
from pipeline import run_gen_audio_step


def _run_gen_audio(job_data: dict):
    ap = get_audio_params(job_data)
    run_gen_audio_step(
        srt_path=job_data["srt_path"],
        output_dir=job_data["output_dir"],
        temperature=ap["temperature"],
        access_code=job_data["access_code"],
        target_language=ap["target_language"],
        cfg_weight=ap["cfg_weight"],
        exaggeration=ap["exaggeration"],
    )


def process_srt_file(
    srt_file,
    temperature: float,
    user_id: int = None,
    target_language: str = "en",
    cfg_weight: float = 0.5,
    exaggeration: float = 0.5,
) -> dict:
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
        for sep in (". ", "! ", "? ", ".\n", "!\n", "?\n", ".\r\n", "!\r\n", "?\r\n"):
            pos = chunk.rfind(sep)
            if pos > last_boundary:
                last_boundary = pos + 1

        if last_boundary > 0:
            chunks.append(text[:last_boundary])
            text = text[last_boundary:].lstrip()
        else:
            last_space = chunk.rfind(" ")
            if last_space > 0:
                chunks.append(text[:last_space])
                text = text[last_space:].lstrip()
            else:
                chunks.append(chunk)
                text = text[max_len:].lstrip()

    return chunks


def _run_audio_segmentation_job(job_data: dict):
    content = job_data["text"]
    output_dir = job_data["output_dir"]
    filename = job_data.get("filename", "output.wav")
    ap = get_audio_params(job_data)
    os.makedirs(output_dir, exist_ok=True)

    access_code = job_data["access_code"]
    job_log(access_code, output_dir, f"Starting audio segmentation: {len(content)} chars, language={ap['target_language']}")

    import subprocess as _sp

    from config import AUDIO_PROMPT_PATH, GEN_AUDIO_PYTHON, PROJECT_ROOT

    text = re.sub(r"<\d+(?:\.\d+)?>\s*", " ", content).strip()
    srt_path = os.path.join(output_dir, "input.srt")
    with open(srt_path, "w") as f:
        f.write(f"1\n00:00:00,000 --> 00:01:00,000\n{text}\n\n")

    gen_audio_script = os.path.join(PROJECT_ROOT, "gen_audio.py")
    assets_dir = os.path.join(PROJECT_ROOT, "..", "assets")

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

    result = _sp.run(cmd, stdout=open(log_path, "a"), stderr=_sp.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"gen_audio exited with code {result.returncode}")

    job_log(access_code, output_dir, "gen_audio subprocess completed")


def process_audio_file(
    content: str,
    original_filename: str,
    temperature: float,
    user_id: int = None,
    target_language: str = "en",
    cfg_weight: float = 0.5,
    exaggeration: float = 0.5,
) -> dict:
    access_code = str(uuid.uuid4())[:8].upper()
    output_dir = os.path.join(AUDIO_TRACKS_DIR, access_code)
    os.makedirs(output_dir, exist_ok=True)

    # Use user-supplied filename, falling back to a unique name derived from the
    # original filename so concurrent submissions don't overwrite each other.
    if original_filename:
        base, _ext = os.path.splitext(original_filename)
        filename = f"{base}.wav"
    else:
        filename = f"output_{access_code}.wav"

    job_data = {
        "access_code": access_code,
        "text": content,
        "output_dir": output_dir,
        "filename": filename,
        "temperature": temperature,
        "target_language": target_language,
        "cfg_weight": cfg_weight,
        "exaggeration": exaggeration,
    }

    job_access_code = get_job_queue().add_job(job_data, _run_audio_segmentation_job, user_id)
    return {"access_code": job_access_code, "message": "Job queued successfully"}
