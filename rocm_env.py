"""Shared ROCm environment setup for all Python GPU entrypoints.

Single source of truth is ``rocm.env`` next to this file (KEY=VALUE,
sourced by shell scripts and parsed here).  Call :func:`setup` BEFORE
importing torch so the ROCm libraries are findable:

    from rocm_env import setup
    setup()
    import torch  # noqa: E402
"""

import os
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parent / "rocm.env"


def load() -> dict[str, str]:
    """Parse rocm.env into a dict (stdlib only — must run before torch)."""
    env: dict[str, str] = {}
    for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def setup() -> None:
    """Apply the canonical ROCm env. Idempotent; call before importing torch."""
    for key, value in load().items():
        os.environ.setdefault(key, value)

    rocm_lib = os.environ.get("ROCM_LIB", "")
    if not rocm_lib:
        return
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if cur.split(":", 1)[0] == rocm_lib:
        return  # already applied (e.g. by rocm_env.sh)
    os.environ["LD_LIBRARY_PATH"] = rocm_lib + (":" + cur if cur else "")
