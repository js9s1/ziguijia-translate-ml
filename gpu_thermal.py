"""Shared GPU thermal reader — used by gen_audio.py (client) and gen_audio_daemon.py."""

from pathlib import Path


def get_gpu_temp() -> float | None:
    """Read AMD GPU (amdgpu) temperature in °C via hwmon. Returns float or None."""
    try:
        for card in Path("/sys/class/drm").glob("card*"):
            hwmons = list(card.glob("device/hwmon/hwmon*"))
            for hw in hwmons:
                try:
                    name = (hw / "name").read_text().strip()
                    if name == "amdgpu":
                        raw = int((hw / "temp1_input").read_text().strip())
                        return raw / 1000.0
                except (OSError, ValueError):
                    continue
    except (OSError, FileNotFoundError):
        pass
    return None
