"""Background model pre-warming for the warm daemons, keyed by job type.

When a job is queued, only the daemon(s) that job will actually use are
pre-warmed:

- TTS (gen_audio) daemon      → audio jobs (gen_audio / tts / segmentation)
                                 and video pipelines with an audio step.
- Translate (HY-MT) daemon    → jobs with a translate step (OCR/whisper
                                 pipelines and translate-only jobs, unless
                                 ocr_only).

Jobs that need neither (OCR-only, SRT-only) pre-warm nothing.

For video pipelines the audio step runs long after enqueue (download →
OCR/whisper → translate), so the enqueue prewarm's grace would expire
before audio starts.  Those handlers therefore call prewarm_tts_async()
when the step *before* audio begins (the translate step, or download for
SRT-driven audio jobs) so the TTS model load overlaps it.

Everything here is best-effort and runs in a detached daemon thread:
failures never affect the job queue — the per-job subprocesses
(gen_audio.py, translate_srt.py) already fall back to starting the
daemon / loading the model themselves.
"""

import contextlib
import json
import logging
import os
import socket
import subprocess
import threading
import time

logger = logging.getLogger(__name__)

_PREWARM_LOCK = threading.Lock()
"""Serialize prewarm attempts across threads and gunicorn workers, so
multiple queued jobs don't race to spawn the same daemon."""


class _DaemonUnreachable(Exception):
    pass


def _request(sock_path: str, payload: dict, timeout: float = 60.0) -> dict:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.settimeout(timeout)
        s.connect(sock_path)
        s.sendall((json.dumps(payload) + "\n").encode())
        buf = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
            if b"\n" in buf:
                break
        if not buf:
            raise _DaemonUnreachable("daemon closed connection without response")
        return json.loads(buf.split(b"\n", 1)[0].decode())
    except (TimeoutError, FileNotFoundError, ConnectionRefusedError, OSError) as e:
        raise _DaemonUnreachable(f"daemon unreachable: {e}") from None
    finally:
        with contextlib.suppress(OSError):
            s.close()


def _ping(sock_path: str) -> bool:
    try:
        return bool(_request(sock_path, {"cmd": "ping"}, timeout=3.0).get("ok"))
    except _DaemonUnreachable:
        return False


def _start_daemon(sock_path: str, python: str, script: str, log_name: str) -> bool:
    """Launch a warm daemon subprocess and wait until it answers pings."""
    log_path = os.path.join(os.path.expanduser("~"), "logs", log_name)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    try:
        with open(log_path, "a") as logf:
            proc = subprocess.Popen(
                [python, "-u", script],
                stdout=logf,
                stderr=logf,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
            )
            threading.Thread(
                target=proc.wait, daemon=True, name=f"daemon-reap-{log_name}"
            ).start()
    except Exception:
        logger.exception("prewarm: failed to start %s", log_name)
        return False
    deadline = time.time() + 120
    while time.time() < deadline:
        if _ping(sock_path):
            return True
        time.sleep(1)
    logger.warning("prewarm: %s did not become ready within 120s", log_name)
    return False


def _ensure_daemon(sock_path: str, python: str, script: str, log_name: str) -> bool:
    if _ping(sock_path):
        return True
    return _start_daemon(sock_path, python, script, log_name)


def prewarm_tts(language: str):
    """Start the gen_audio daemon if needed and load the TTS model."""
    from config import (
        GEN_AUDIO_DAEMON_SCRIPT,
        GEN_AUDIO_DAEMON_SOCK,
        GEN_AUDIO_PYTHON,
    )

    if not _ensure_daemon(
        GEN_AUDIO_DAEMON_SOCK, GEN_AUDIO_PYTHON, GEN_AUDIO_DAEMON_SCRIPT, "gen_audio_daemon.log"
    ):
        return
    lang = str(language or "en")
    try:
        resp = _request(
            GEN_AUDIO_DAEMON_SOCK,
            {"cmd": "ensure_model", "language": lang},
            timeout=1800.0,
        )
        if resp.get("ok"):
            logger.info("prewarm: TTS model ready (%s)", resp.get("device"))
        else:
            logger.warning("prewarm: TTS ensure_model failed: %s", resp.get("error"))
    except _DaemonUnreachable as e:
        logger.warning("prewarm: TTS ensure_model unreachable: %s", e)


def prewarm_tts_async(language: str):
    """Detached, serialized TTS prewarm — call when the step *before* audio
    starts (e.g. the translate step, or download for SRT-driven audio jobs)
    so the model load overlaps it instead of blocking the audio step.

    Best-effort: failures never affect the job — gen_audio.py clients
    start the daemon on demand themselves.
    """

    def _worker():
        with _PREWARM_LOCK:
            try:
                prewarm_tts(language)
            except Exception:
                logger.exception("prewarm tts (async) failed")

    threading.Thread(target=_worker, daemon=True, name="daemon-prewarm-tts").start()


def prewarm_translate():
    """Start the translate daemon if needed and load the HY-MT model."""
    from config import (
        TRANSLATE_DAEMON_SCRIPT,
        TRANSLATE_DAEMON_SOCK,
        TRANSLATE_PYTHON,
    )

    if not _ensure_daemon(
        TRANSLATE_DAEMON_SOCK, TRANSLATE_PYTHON, TRANSLATE_DAEMON_SCRIPT, "translate_daemon.log"
    ):
        return
    try:
        resp = _request(TRANSLATE_DAEMON_SOCK, {"cmd": "ensure_model"}, timeout=1800.0)
        if resp.get("ok"):
            logger.info("prewarm: HY-MT model ready (%s)", resp.get("device"))
        else:
            logger.warning("prewarm: translate ensure_model failed: %s", resp.get("error"))
    except _DaemonUnreachable as e:
        logger.warning("prewarm: translate ensure_model unreachable: %s", e)


# Job types that use the TTS (gen_audio) daemon.
_TTS_JOB_TYPES = {
    "_run_gen_audio",
    "_run_tts_job",
    "_run_audio_segmentation_job",
    "_run_video_job",
    "_run_video_custom_job",
    "_run_video_auto_job",
    "_run_video_ocr_job",
    "_run_video_ning_ocr_job",
    "_run_video_ning_auto_job",
}

# Job types that use the translate (HY-MT) daemon.  Translate-only jobs
# only translate when ocr_only != "yes" (filtered in _run_prewarm).
_TRANSLATE_JOB_TYPES = {
    "_run_video_ocr_translate_only_job",
    "_run_video_ning_ocr_translate_only_job",
    "_run_video_ocr_job",
    "_run_video_ning_ocr_job",
    "_run_video_ning_auto_job",
    "_run_video_auto_job",
}


def prewarm_choices(run_func_name: str, job_data: dict) -> tuple[bool, bool]:
    """Return (needs_tts, needs_translate) for a queued job.

    Pure mapping, no I/O — unit-testable.
    """
    ocr_only = job_data.get("ocr_only") == "yes"
    needs_tts = run_func_name in _TTS_JOB_TYPES
    needs_translate = run_func_name in _TRANSLATE_JOB_TYPES and not ocr_only
    return needs_tts, needs_translate


def _run_prewarm(run_func_name: str, job_data: dict):
    language = job_data.get("target_language") or "en"
    needs_tts, needs_translate = prewarm_choices(run_func_name, job_data)
    if needs_tts:
        prewarm_tts(language)
    if needs_translate:
        prewarm_translate()


def prewarm_for_job(run_func_name: str, job_data: dict):
    """Non-blocking pre-warm of the daemon(s) a queued job will use."""
    if run_func_name not in _TTS_JOB_TYPES and run_func_name not in _TRANSLATE_JOB_TYPES:
        return

    def _worker():
        with _PREWARM_LOCK:
            try:
                _run_prewarm(run_func_name, job_data)
            except Exception:
                logger.exception("prewarm for %s failed", run_func_name)

    threading.Thread(target=_worker, daemon=True, name="daemon-prewarm").start()
