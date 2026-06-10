import io
import os
import sys
import threading
from typing import Optional

import srt
import torch
import torchaudio as ta

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from config import AUDIO_PROMPT_PATH
from singleton import singleton
from video_util import read_srt_text


def _check_gpu_healthy() -> bool:
    """Probe GPU health with a progressively heavier workload (with timeout).

    A single tiny 128×128 matmul can still pass on a degraded ROCm/HIP
    driver state after a long video job.  This test ramps up through
    multiple tensor sizes and runs a small inference pipeline:
    matmul → activation → softmax, repeated across sizes, with explicit
    ``torch.cuda.synchronize()`` after each step.

    Returns True only if all sizes pass, indicating the GPU is genuinely
    healthy enough for TTS inference.
    """
    import subprocess
    import sys
    probe_code = (
        "import os\n"
        "os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '9.0.0')\n"
        "os.environ.setdefault('HSA_XNACK', '0')\n"
        "os.environ.setdefault('ROCBLAS_USE_HIPBLASLT', '0')\n"
        "import torch\n"
        "ok = True\n"
        "try:\n"
        "    # Progressive workload — small → large, catching driver decay\n"
        "    for size in [128, 256, 512, 768]:\n"
        "        a = torch.randn(size, size, device='cuda')\n"
        "        b = torch.randn(size, size, device='cuda')\n"
        "        c = a @ b                     # GEMM — catches HIPBLAS errors\n"
        "        c = torch.nn.functional.relu(c)  # activation path\n"
        "        c = torch.softmax(c, dim=-1)    # reduction path\n"
        "        torch.cuda.synchronize()\n"
        "    torch.cuda.empty_cache()\n"
        "    print('OK', flush=True)\n"
        "except Exception as e:\n"
        "    print(f'FAIL: {e}', flush=True)"
    )
    try:
        r = subprocess.run(
            [sys.executable, "-c", probe_code],
            capture_output=True, text=True, timeout=30,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                "HSA_OVERRIDE_GFX_VERSION": "9.0.0",
                "HSA_XNACK": "0",
                "ROCBLAS_USE_HIPBLASLT": "0",  # gfx90c doesn't support hipBLASLt
                "PYTHONUNBUFFERED": "1",
            },
        )
        ok = "OK" in r.stdout
        if not ok:
            print(f"GPU health check failed — stderr: {r.stderr.strip()}")
        return ok
    except subprocess.TimeoutExpired:
        print("GPU health check timed out — GPU appears hung")
        return False
    except Exception as e:
        print(f"GPU health check error: {e}")
        return False


def _choose_device(preferred: str = "cuda") -> str:
    """Pick device: GPU if healthy with enough free memory, otherwise CPU.
    Performs a subprocess probe to catch GPU Hang conditions."""
    if preferred == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        print("CUDA not available — using CPU")
        return "cpu"
    # AMD APU iGPUs (Renoir/gfx90c, VanGogh/Dali) — historically unstable
    # with TTS on ROCm 7.x. With PyTorch ROCm 6.2 + HSA_XNACK=0 override,
    # GPU inference works correctly. This check is kept as a safety net
    # but no longer blocks by default.
    # See rocm_env.sh for required environment variables.
    if not _check_gpu_healthy():
        print("GPU unhealthy (hang detected) — falling back to CPU")
        return "cpu"
    try:
        free, total = torch.cuda.mem_get_info()
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        # Model needs roughly 6 GiB; leave some margin
        if free_gb < 6.6:
            print(f"GPU free memory {free_gb:.1f} GiB / {total_gb:.1f} GiB — too low, using CPU")
            return "cpu"
        return "cuda"
    except Exception:
        return "cpu"


# HF_TOKEN must be set via environment variable for HuggingFace authentication
# Example: export HF_TOKEN="your_token_here"


# Shared model state — lives at module level so it persists across
# singleton instances and works with the @singleton decorator.
_model = None
_default_audio_prompt_path = AUDIO_PROMPT_PATH


@singleton
class NingAudio:
    def __init__(self, audio_prompt: Optional[str] = None):
        global _default_audio_prompt_path
        self.audio_prompt_path = audio_prompt if audio_prompt else _default_audio_prompt_path
        self.model = None
        self.sample_rate = None

    def get_model(self, device: str = "cuda") -> ChatterboxMultilingualTTS:
        global _model
        if _model is not None:
            # Already loaded; if on CPU but GPU has freed up, reload to GPU
            if self.model and self.model.device == "cpu" and _choose_device("cuda") == "cuda":
                print("GPU memory available, reloading model to GPU")
                _model = None
                torch.cuda.empty_cache()
        if _model is None:
            import warnings
            warnings.filterwarnings("ignore")
            # Use _choose_device for all cases — it checks GPU health, model
            # compatibility (known-problem iGPUs), and free memory.
            actual_device = _choose_device(device)
            try:
                _model = ChatterboxMultilingualTTS.from_pretrained(device=actual_device)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if actual_device == "cuda":
                    print(f"CUDA OOM ({e}), falling back to CPU")
                    torch.cuda.empty_cache()
                    _model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
                else:
                    raise
            _model.prepare_conditionals(self.audio_prompt_path)
        self.model = _model
        self.sample_rate = self.model.sr
        return self.model

    def setup(self, device: str = "cuda"):
        self.get_model(device)

    def wav_to_bytes(self, wav: torch.Tensor, sample_rate: int) -> io.BytesIO:
        buffer = io.BytesIO()
        ta.save(buffer, wav, sample_rate, format='wav')
        buffer.seek(0)
        return buffer

    def text_to_wave(
        self,
        text: str,
        prompt_file: Optional[str] = None,
        temperature: float = 0.6,
        target_language: str = "en",
        cfg_weight: float = 0.5,
        exaggeration: float = 0.5,
    ) -> io.BytesIO:
        self.setup()
        if prompt_file:
            self.model.prepare_conditionals(prompt_file)
        wav = self.model.generate(
                text,
                language_id=target_language,
                temperature=temperature,
                cfg_weight=cfg_weight,
                exaggeration=exaggeration,
            )
        return self.wav_to_bytes(wav, self.model.sr)

    def generate_silence(self, duration_sec, sample_rate):
        num_frames = int(duration_sec * sample_rate)
        return torch.zeros(1, num_frames)

    def load_subs(self, srt_path):
        return list(srt.parse(read_srt_text(srt_path)))

    def generate_audio(self, text, output_path, sample_rate, temperature=None, prompt_file=None, target_language="en", cfg_weight=0.5, exaggeration=0.5):
        if temperature is None:
            temperature = self.model.temperature
        if prompt_file:
            self.model.prepare_conditionals(prompt_file)
        wav = self.model.generate(text, language_id=target_language, temperature=temperature, cfg_weight=cfg_weight, exaggeration=exaggeration)
        # Ensure wav is 2D tensor [1, samples]
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        ta.save(output_path, wav, sample_rate)
        wav_duration = wav.shape[1] / sample_rate
        return wav, wav_duration

    def combine_audio_segments(self, segments_info, total_duration, sample_rate):
        # Calculate actual needed duration from segments
        max_end = 0
        for seg in segments_info:
            end_sample = int((seg["new_start"] + seg["wav_duration"]) * sample_rate)
            if end_sample > max_end:
                max_end = end_sample

        # Use max of original duration and actual needed
        needed_duration = max_end / sample_rate
        actual_duration = max(total_duration, needed_duration)

        if actual_duration <= 0:
            actual_duration = 1

        combined = self.generate_silence(actual_duration, sample_rate)
        for seg in segments_info:
            wav_data = seg["wav_data"]
            if wav_data.dim() == 1:
                wav_data = wav_data.unsqueeze(0)
            start_sample = int(seg["new_start"] * sample_rate)
            end_sample = start_sample + wav_data.shape[1]
            if end_sample > combined.shape[1]:
                new_combined = torch.zeros(1, end_sample)
                new_combined[:, :combined.shape[1]] = combined
                combined = new_combined
            combined[:, start_sample:end_sample] = wav_data
        return combined

    def save_audio(self, output_path, wav_tensor, sample_rate):
        ta.save(output_path, wav_tensor, sample_rate)
