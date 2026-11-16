"""Preflight gate for the primary Vulkan Whisper backend.

Vulkan Whisper shares the iGPU with the TTS, OCR, and translation daemons.
Running it concurrently makes those jobs slower and pushes the APU toward
thermal throttling, so ``run_whisper_ckpt`` checks this gate first and uses
the CPU fallback while the GPU is hot or another GPU daemon is active.

``gpu_busy_percent`` is deliberately not used: on this APU it reads well
above zero even when the box is idle, so explicit daemon activity counts
(TTS/OCR/translate ping ``active``) are more reliable.
"""

import json
import os
import socket

from config import (
    GEN_AUDIO_DAEMON_SOCK,
    RAPID_VIDEOCR_DAEMON_SOCK,
    TRANSLATE_DAEMON_SOCK,
    WHISPER_GATE_OCR_ACTIVE,
    WHISPER_GATE_TRANSLATE_ACTIVE,
    WHISPER_GATE_TTS_ACTIVE,
    WHISPER_GPU_MAX_TEMP,
)

from gpu_thermal import get_gpu_temp

_PING_TIMEOUT = 0.5


def _daemon_status(sock_path: str) -> dict | None:
    """Return a daemon's ping response, or None when it is not running."""
    if not sock_path or not os.path.exists(sock_path):
        return None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(_PING_TIMEOUT)
            sock.connect(sock_path)
            sock.sendall(b'{"cmd": "ping"}\n')
            buf = b""
            while b"\n" not in buf:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
        if not buf:
            return None
        return json.loads(buf.split(b"\n", 1)[0].decode())
    except (OSError, ValueError):
        return None


def _active_count(status: dict | None) -> int:
    if not status or not status.get("ok"):
        return 0
    try:
        return int(status.get("active", 0))
    except (TypeError, ValueError):
        return 0


def gpu_block_reason() -> str | None:
    """Return why Vulkan Whisper should be skipped, or None when it is safe."""
    temp = get_gpu_temp()
    if temp is not None and temp >= WHISPER_GPU_MAX_TEMP:
        return f"GPU {temp:.0f}°C >= {WHISPER_GPU_MAX_TEMP:.0f}°C"

    checks = (
        ("TTS", GEN_AUDIO_DAEMON_SOCK, WHISPER_GATE_TTS_ACTIVE),
        ("OCR", RAPID_VIDEOCR_DAEMON_SOCK, WHISPER_GATE_OCR_ACTIVE),
        ("translate", TRANSLATE_DAEMON_SOCK, WHISPER_GATE_TRANSLATE_ACTIVE),
    )
    for label, sock_path, limit in checks:
        if limit <= 0:
            continue
        active = _active_count(_daemon_status(sock_path))
        if active >= limit:
            return f"{label} active ({active})"
    return None
