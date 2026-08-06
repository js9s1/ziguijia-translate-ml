"""GPU model management — one model at a time, lazy swap on language change.

All TTS model globals, loading, and GPU acquisition live here so the
single-worker job queue naturally serialises GPU access.  *NingAudio*
imports from this module; job functions call :func:`_release_gpu` after
each job to free VRAM for subprocesses (now unused — gen_audio runs
in-process, but kept for manual intervention).
"""

import gc
import os

import torch
from config import AUDIO_PROMPT_PATH

# ── Model globals (exactly one may be non-None) ──────────────

_model = None  # ChatterboxMultilingualTTS
_indonesian_model = None  # ChatterboxTTS (fine-tuned)


# ── GPU acquisition / release ───────────────────────────────


def _acquire_gpu_for(lang: str) -> None:
    """Ensure the model for *lang* is the ONLY loaded model, on GPU."""
    global _model, _indonesian_model

    if lang == "id":
        if _model is not None:
            del _model
            _model = None
            gc.collect()
            torch.cuda.empty_cache()
    else:
        if _indonesian_model is not None:
            del _indonesian_model
            _indonesian_model = None
            gc.collect()
            torch.cuda.empty_cache()

    if lang == "id":
        if _indonesian_model is None:
            _load_indonesian_model()
    else:
        if _model is None:
            _load_multilingual_model()


def _release_gpu() -> None:
    """Unload whichever model is on GPU so other processes can use it."""
    global _model, _indonesian_model
    if _model is not None:
        print("Releasing multilingual model from GPU...")
        del _model
    if _indonesian_model is not None:
        print("Releasing Indonesian model from GPU...")
        del _indonesian_model
    _model = None
    _indonesian_model = None
    torch.cuda.empty_cache()


def _get_device(m) -> str:
    """Get device string for a model. Tries .device, parameter devices,
    and submodule attributes (t3, s3gen, model)."""
    try:
        return str(m.device)
    except AttributeError:
        pass
    try:
        return str(next(m.parameters()).device)
    except (StopIteration, AttributeError):
        pass
    for sub in ("t3", "s3gen", "model"):
        try:
            sm = getattr(m, sub)
            return str(next(sm.parameters()).device)
        except (StopIteration, AttributeError):
            pass
    return "cpu"


# ── Model loaders ───────────────────────────────────────────


def _load_multilingual_model():
    """Load ChatterboxMultilingualTTS, choosing device via health/memory check.

    Calls ``_choose_device`` to decide whether GPU or CPU is safe.  This
    prevents native ROCm crashes (SIGSEGV) on unstable iGPUs like Renoir
    gfx90c by falling back to CPU when the GPU is degraded or low on memory.
    """
    global _model
    import warnings

    warnings.filterwarnings("ignore")
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    device = _choose_device("cuda")
    print(f"Loading multilingual TTS model to {device}...")
    _model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    _model.prepare_conditionals(AUDIO_PROMPT_PATH)
    print(f"Multilingual TTS model ready ({device}).")


def _load_indonesian_model():
    """Load Indonesian fine-tuned ChatterboxTTS via device health/memory check."""
    global _indonesian_model
    from chatterbox.tts import ChatterboxTTS
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    device = _choose_device("cuda")
    print(f"Loading Indonesian TTS model to {device}...")
    _indonesian_model = ChatterboxTTS.from_pretrained(device=device)

    checkpoint_path = hf_hub_download(
        repo_id="grandhigh/Chatterbox-TTS-Indonesian",
        filename="t3_cfg.safetensors",
    )
    t3_state = load_file(checkpoint_path, device="cpu")
    _indonesian_model.t3.load_state_dict(t3_state)
    torch.cuda.empty_cache()
    dev = _get_device(_indonesian_model)
    print(f"Indonesian TTS model ready ({dev}).")


# ── Indonesian generation helper ────────────────────────────


def _generate_indonesian(text: str, prompt_file: str | None = None, temperature: float = 0.6) -> torch.Tensor:
    """Generate Indonesian speech via the fine-tuned ChatterboxTTS model.

    Falls back to *AUDIO_PROMPT_PATH* (the same default voice as the
    multilingual model) when no explicit *prompt_file* is given.
    """
    model = _indonesian_model
    kwargs = {}
    audio_prompt = prompt_file if prompt_file else AUDIO_PROMPT_PATH
    if audio_prompt and os.path.exists(audio_prompt):
        kwargs["audio_prompt_path"] = audio_prompt
    try:
        wav = model.generate(text, temperature=temperature, **kwargs)
    except TypeError:
        wav = model.generate(text, **kwargs)
    return wav


def _switch_to_cpu() -> None:
    """Switch the multilingual model from GPU to CPU.

    No-op if model is already on CPU or not loaded.  Call this before a
    segment that needs CPU, then call :func:`_reload_gpu` to switch back.
    """
    global _model
    if _model is None:
        _load_multilingual_model()  # load straight to CPU — _choose_device will fall back
        return
    if _get_device(_model) != "cuda":
        return  # already on CPU
    print("Switching multilingual TTS model from GPU to CPU...")
    import warnings

    del _model
    _model = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    warnings.filterwarnings("ignore")
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    _model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
    _model.prepare_conditionals(AUDIO_PROMPT_PATH)
    print("Multilingual TTS model now on CPU.")


def _reload_gpu() -> bool:
    """Switch the multilingual model from CPU back to GPU.

    Returns True on success, False if GPU is not available / unhealthy.
    Call after a temporary CPU fallback for a single segment.
    """
    global _model
    import warnings

    if not torch.cuda.is_available():
        print("CUDA not available — cannot reload on GPU")
        return False
    device_choice = _choose_device("cuda")
    if device_choice != "cuda":
        print(f"GPU not healthy — keeping model on CPU (device check returned {device_choice})")
        return False

    # Unload current (CPU) model
    if _model is not None:
        del _model
        _model = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    warnings.filterwarnings("ignore")
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    try:
        print("Reloading multilingual TTS model on GPU...")
        _model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
        _model.prepare_conditionals(AUDIO_PROMPT_PATH)
        print("Multilingual TTS model reloaded on GPU.")
        return True
    except (torch.cuda.OutOfMemoryError, RuntimeError, torch.cuda.CudaError) as e:
        print(f"GPU reload failed ({e}), keeping CPU model")
        # Reload on CPU instead
        _model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
        _model.prepare_conditionals(AUDIO_PROMPT_PATH)
        return False


# ── Fork safety ─────────────────────────────────────────────


def _clear_indonesian_model():
    """Reset Indonesian model handle (used by gunicorn post_fork)."""
    global _indonesian_model
    _indonesian_model = None


def _clear_all_models():
    """Reset both model handles (used by gunicorn post_fork)."""
    global _model, _indonesian_model
    _model = None
    _indonesian_model = None


# ── Health / device probes ──────────────────────────────────


def _check_gpu_healthy() -> bool:
    """Probe GPU health with a progressively heavier workload (with timeout).

    Returns True only if all sizes pass, indicating the GPU is genuinely
    healthy enough for TTS inference.
    """
    from gpu_probe import run_gpu_probe

    r = run_gpu_probe([128, 256, 512, 768], include_softmax=True)
    if r is None:
        return False
    ok = "OK" in r.stdout
    if not ok:
        print(f"GPU health check failed — stderr: {r.stderr.strip()}")
    return ok


def _choose_device(preferred: str = "cuda") -> str:
    """Pick device: GPU if healthy with enough free memory, otherwise CPU."""
    if preferred == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        print("CUDA not available — using CPU")
        return "cpu"
    if not _check_gpu_healthy():
        print("GPU unhealthy (hang detected) — falling back to CPU")
        return "cpu"
    try:
        free, _ = torch.cuda.mem_get_info()
        free_gb = free / (1024**3)
        if free_gb < 6.6:
            print(f"GPU free memory {free_gb:.1f} GiB — too low, using CPU")
            return "cpu"
        return "cuda"
    except (torch.cuda.CudaError, RuntimeError, AttributeError):
        return "cpu"
