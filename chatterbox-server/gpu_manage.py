"""GPU model management — one model at a time, lazy swap on language change.

All TTS model globals, loading, and GPU acquisition live here.  This
module only runs inside the gen_audio.py subprocess — the server
worker never imports it.
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
    """
    global _model
    import warnings

    warnings.filterwarnings("ignore")
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    except ModuleNotFoundError:
        raise RuntimeError(
            "chatterbox-tts is not installed for this Python interpreter. "
            "In-process TTS requires chatterbox-tts. Use gen_audio.py subprocess instead."
        ) from None

    device = _choose_device("cuda")
    print(f"Loading multilingual TTS model to {device}...")
    _model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    _model.prepare_conditionals(AUDIO_PROMPT_PATH)
    print(f"Multilingual TTS model ready ({device}, S3Gen on cpu).")


def _load_indonesian_model():
    """Load Indonesian fine-tuned ChatterboxTTS via device health/memory check."""
    global _indonesian_model
    try:
        from chatterbox.tts import ChatterboxTTS
    except ModuleNotFoundError:
        raise RuntimeError("chatterbox-tts is not installed for this Python interpreter.") from None
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


# ── Device selection ────────────────────────────────────────

_GPU_MIN_FREE_MEM_GiB = float(os.environ.get("GPU_MIN_FREE_MEM_GiB", "1.0"))


def _choose_device(preferred: str = "cuda") -> str:
    """Pick device: GPU if enough free memory, otherwise CPU."""
    if preferred == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        print("CUDA not available — using CPU")
        return "cpu"
    try:
        free, _ = torch.cuda.mem_get_info()
        free_gb = free / (1024**3)
        if free_gb < _GPU_MIN_FREE_MEM_GiB:
            print(f"GPU free memory {free_gb:.1f} GiB — too low (need {_GPU_MIN_FREE_MEM_GiB:.1f} GiB), using CPU")
            return "cpu"
        return "cuda"
    except (torch.cuda.CudaError, RuntimeError, AttributeError):
        return "cpu"
