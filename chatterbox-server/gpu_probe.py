"""Shared GPU health probe — used by both jobqueue (post-job reset) and audio_utils (pre-TTS check).

Both modules previously contained near-identical inline Python scripts executed
via ``subprocess.run([sys.executable, "-c", ...])``.  This module provides a
single ``run_gpu_probe`` function parameterised by workload intensity so each
caller picks the right variant without duplicating the subprocess scaffolding.
"""

import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)

_ROCm_ENV = {
    "PYTHONUNBUFFERED": "1",
    "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES", "0"),
}


def _build_probe_script(sizes: list[int], include_softmax: bool, reset_peak_memory: bool) -> str:
    """Build the inline Python script for GPU probing."""
    softmax_line = "        c = torch.softmax(c, dim=-1)    # reduction path\n" if include_softmax else ""
    reset_line = (
        (
            "    # Attempt HIP device reset (ROCm) — helps clear driver state\n"
            "    try:\n"
            "        torch.cuda.reset_peak_memory_stats()\n"
            "    except Exception:\n"
            "        pass\n"
        )
        if reset_peak_memory
        else ""
    )

    sizes_repr = repr(sizes)
    return (
        "import torch\n"
        "ok = True\n"
        "try:\n"
        f"    for size in {sizes_repr}:\n"
        "        a = torch.randn(size, size, device='cuda')\n"
        "        b = torch.randn(size, size, device='cuda')\n"
        "        c = a @ b                     # GEMM — catches HIPBLAS errors\n"
        "        c = torch.nn.functional.relu(c)  # activation path\n"
        + softmax_line
        + "        torch.cuda.synchronize()\n"
        "    torch.cuda.empty_cache()\n"
        "    torch.cuda.synchronize()\n" + reset_line + "except Exception as e:\n"
        "    print(f'GPU probe failed: {e}', flush=True)\n"
        "    ok = False\n"
        "print('OK' if ok else 'FAIL', flush=True)\n"
    )


def run_gpu_probe(
    sizes: list[int],
    *,
    include_softmax: bool = False,
    reset_peak_memory: bool = False,
    timeout: int = 30,
) -> subprocess.CompletedProcess | None:
    """Run a GPU health probe in a subprocess and return the result.

    Parameters
    ----------
    sizes:
        Tensor sizes to iterate through (e.g. ``[128, 256, 512, 768]``).
        Larger sizes stress the GPU more heavily.
    include_softmax:
        If True, add a softmax operation after each matmul+relu cycle.
    reset_peak_memory:
        If True, call ``torch.cuda.reset_peak_memory_stats()`` after the
        workload (helps clear ROCm driver state).
    timeout:
        Maximum seconds to wait for the subprocess.

    Returns
    -------
    ``subprocess.CompletedProcess`` on success, ``None`` on timeout or error.
    """
    probe_code = _build_probe_script(sizes, include_softmax, reset_peak_memory)
    try:
        return subprocess.run(
            [sys.executable, "-c", probe_code],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, **_ROCm_ENV},
        )
    except subprocess.TimeoutExpired:
        logger.warning("GPU probe timed out after %ds (GPU may be hung)", timeout)
        return None
    except FileNotFoundError:
        logger.debug("GPU probe skipped — python interpreter not found")
        return None
    except Exception as e:
        logger.warning("GPU probe error: %s", e)
        return None
