"""Shared GPU thermal logic — used by gen_audio.py (client) and the warm daemons.

``get_gpu_temp`` reads the GPU temperature; ``ThermalGate`` implements the
thermal gate + idle-temperature shutdown shared by translate_daemon.py and
gen_audio_daemon.py.
"""

import threading
import time
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


def get_cpu_temp() -> float | None:
    """Read CPU/package temperature in °C via ACPI thermal zones. Returns float or None."""
    try:
        for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
            try:
                ztype = (zone / "type").read_text().strip()
                if ztype in ("acpitz", "x86_pkg_temp", "cpu-thermal"):
                    return int((zone / "temp").read_text().strip()) / 1000.0
            except (OSError, ValueError):
                continue
    except (OSError, FileNotFoundError):
        pass
    return None


class ThermalGate:
    """Shared thermal gate + idle shutdown for the warm daemons.

    The daemon's monitor thread calls ``poll()`` in a loop; each call
    sleeps ``poll_secs`` and reads the GPU temperature:

    * at or above ``temp_limit`` the ``blocked`` event is set (workers
      pause); at or below ``cooldown_target`` it is cleared (resume).
    * returns True when the daemon should shut down: no active jobs
      (``active() == 0``), a model loaded (``model_loaded()``), and the
      GPU or CPU at or above ``idle_temperature`` for ``idle_hot_polls``
      consecutive polls — or at or above ``idle_critical_temp``, which
      bypasses the prewarm grace armed with ``arm_grace()``.  Exiting
      the process frees the GPU and stops ROCm's HSA exception-monitor
      busy-poll; clients restart the daemon on demand.

    The CPU sensor is included because an idle resident model keeps the
    box warm mainly through ROCm's HSA busy-poll (CPU load): on APUs the
    GPU edge sensor can read well below ``idle_temperature`` while the
    CPU package is still hot, so the shutdown would never fire.
    """

    def __init__(
        self,
        *,
        temp_limit: float,
        cooldown_target: float,
        poll_secs: float,
        idle_temperature: float,
        idle_critical_temp: float,
        idle_grace_secs: float,
        idle_hot_polls: int = 2,
    ):
        self.temp_limit = temp_limit
        self.cooldown_target = cooldown_target
        self.poll_secs = poll_secs
        self.idle_temperature = idle_temperature
        self.idle_critical_temp = idle_critical_temp
        self.idle_grace_secs = idle_grace_secs
        self.idle_hot_polls = idle_hot_polls
        self.blocked = threading.Event()
        self._prewarm_until = 0.0  # monotonic deadline — idle release suppressed until then
        self._hot_polls = 0

    def arm_grace(self) -> None:
        """Start the prewarm grace — call after a successful ensure_model."""
        self._prewarm_until = time.monotonic() + self.idle_grace_secs

    def disarm_grace(self) -> None:
        """End the prewarm grace — call once a job starts using the model,
        so the idle-hot shutdown can fire right after the job finishes."""
        self._prewarm_until = 0.0

    def poll(self, active, model_loaded) -> bool:
        """One monitoring cycle.  Returns True when the daemon should shut down."""
        temp = get_gpu_temp()
        cpu_temp = get_cpu_temp()
        hot = temp
        if cpu_temp is not None:
            hot = cpu_temp if hot is None else max(hot, cpu_temp)
        if hot is not None:
            if temp is not None:
                if temp >= self.temp_limit and not self.blocked.is_set():
                    print(f"[daemon] GPU temp {temp:.0f}°C ≥ limit {self.temp_limit:.0f}°C — pausing all workers")
                    self.blocked.set()
                elif temp <= self.cooldown_target and self.blocked.is_set():
                    print(f"[daemon] GPU cooled to {temp:.0f}°C — resuming workers")
                    self.blocked.clear()
            if active() == 0 and model_loaded():
                if hot >= self.idle_critical_temp or (
                    hot >= self.idle_temperature and time.monotonic() >= self._prewarm_until
                ):
                    self._hot_polls += 1
                    if self._hot_polls >= self.idle_hot_polls:
                        parts = []
                        if temp is not None:
                            parts.append(f"GPU {temp:.0f}°C")
                        if cpu_temp is not None:
                            parts.append(f"CPU {cpu_temp:.0f}°C")
                        print(
                            f"[daemon] idle and {'/'.join(parts) or '?'} ≥ {self.idle_temperature:.0f}°C "
                            f"— shutting down"
                        )
                        return True
                else:
                    self._hot_polls = 0
        time.sleep(self.poll_secs)
        return False
