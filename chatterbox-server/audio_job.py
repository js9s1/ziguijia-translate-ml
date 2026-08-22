"""Audio-from-SRT and audio-segmentation jobs."""

import os
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
    import time

    from config import (
        AUDIO_PROMPT_PATH,
        GEN_AUDIO_MIN_TOTAL_TIMEOUT,
        GEN_AUDIO_PYTHON,
        GEN_AUDIO_SEGMENT_BUDGET,
        GEN_AUDIO_STALL_TIMEOUT,
        PROJECT_ROOT,
    )

    gen_audio_script = os.path.join(PROJECT_ROOT, "gen_audio", "gen_audio.py")
    assets_dir = os.path.join(PROJECT_ROOT, "..", "assets")

    cmd = [
        GEN_AUDIO_PYTHON, "-u", gen_audio_script,
        "--text", content,
        "--audio_prompt", AUDIO_PROMPT_PATH,
        "--temperature", str(ap["temperature"]),
        "--output_dir", output_dir,
        "--assets_dir", assets_dir,
        "--target_language", ap["target_language"],
        "--cfg_weight", str(ap["cfg_weight"]),
        "--exaggeration", str(ap["exaggeration"]),
        "--output_wav", filename,
    ]

    # Size-based total timeout, scaled to the number of 120-char chunks
    # gen_audio.py will split the text into (mirrors run_gen_audio_step).
    n_chunks = max(1, -(-len(content) // 120))
    total_timeout = max(GEN_AUDIO_MIN_TOTAL_TIMEOUT, GEN_AUDIO_SEGMENT_BUDGET * n_chunks)

    def _stop(process: _sp.Popen, reason: str) -> RuntimeError:
        process.terminate()
        try:
            process.wait(timeout=90)
        except _sp.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=10)
            except _sp.TimeoutExpired:
                pass
        return RuntimeError(reason)

    log_path = os.path.join(output_dir, "job.log")
    with open(log_path, "a") as proc_log:
        proc_log.write(f"+ {' '.join(cmd)}\n")
        proc_log.flush()
        proc_pos = proc_log.tell()

        process = _sp.Popen(cmd, stdout=proc_log, stderr=_sp.STDOUT, text=True)

        start = time.monotonic()
        last_size = proc_pos
        last_growth = start
        while True:
            try:
                process.wait(timeout=30)
                break
            except _sp.TimeoutExpired:
                elapsed = time.monotonic() - start
                cur_size = os.path.getsize(log_path)
                if cur_size > last_size:
                    last_size = cur_size
                    last_growth = time.monotonic()
                stalled_for = time.monotonic() - last_growth
                if stalled_for > GEN_AUDIO_STALL_TIMEOUT:
                    raise _stop(
                        process,
                        f"gen_audio stalled: no output for {stalled_for / 60:.1f}min "
                        f"(elapsed {elapsed / 60:.1f}min)",
                    ) from None
                if elapsed > total_timeout:
                    raise _stop(
                        process,
                        f"gen_audio timed out after {elapsed / 60:.1f}min "
                        f"({n_chunks} chunks)",
                    ) from None
                get_job_queue().update_job_progress(
                    access_code, f"正在生成音频... ({int(elapsed) // 60}分{int(elapsed) % 60}秒)"
                )

    if process.returncode != 0:
        raise RuntimeError(f"gen_audio exited with code {process.returncode}")

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
