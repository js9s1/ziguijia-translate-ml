"""Audio parameter schemas for the job/UI layer.

The web UI renders engine-specific audio controls from these schemas instead
of hard-coding one flat set of sliders for every pipeline.  The engine is
resolved server-side from target_language; the browser only uses the returned
schema to decide what to render.

Keep this module dependency-light: it is imported by middleware and the API
routes, not by the GPU job subprocesses.
"""

from __future__ import annotations

import json
import math
from typing import Any

AUDIO_PIPELINES = {"tts", "srt", "video_ning", "video_custom"}

_ENGINE_LABELS = {
    "chatterbox": "Chatterbox",
    "chatterbox_id": "Chatterbox (Indonesian)",
    "voxcpm": "VoxCPM2",
}

_CHATTERBOX_FIELDS = [
    {
        "key": "temperature",
        "label": "温度",
        "type": "float",
        "min": 0.2,
        "max": 1.2,
        "step": 0.01,
        "default": 0.6,
        "help": "越高输出越随机，越低越稳定",
    },
    {
        "key": "cfg_weight",
        "label": "CFG 权重",
        "type": "float",
        "min": 0.0,
        "max": 1.5,
        "step": 0.01,
        "default": 0.25,
        "help": "控制生成音频与提示的契合度",
    },
    {
        "key": "exaggeration",
        "label": "口音夸张度",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "default": 0.3,
        "help": "口音/语气的夸张程度",
    },
]

_CHATTERBOX_ID_FIELDS = [
    {
        "key": "temperature",
        "label": "温度",
        "type": "float",
        "min": 0.2,
        "max": 1.2,
        "step": 0.01,
        "default": 0.6,
        "help": "印尼语微调模型使用温度控制输出随机性",
    },
]

_SCHEMAS = {
    "chatterbox": _CHATTERBOX_FIELDS,
    "chatterbox_id": _CHATTERBOX_ID_FIELDS,
}

_CANTONESE_CODES = {"yue", "zh-hk", "zh-yue", "cantonese"}


def resolve_audio_engine(target_language: str | None) -> str:
    """Map a target language to the audio engine that will handle it."""
    code = str(target_language or "").strip().lower()
    if code in _CANTONESE_CODES:
        return "voxcpm"
    if code == "id":
        return "chatterbox_id"
    return "chatterbox"


def _voxcpm_fields() -> list[dict[str, Any]]:
    # Import lazily so this module stays importable without the ROCm config
    # module in lightweight tooling/tests.
    try:
        from config import (
            VOXCPM_CFG_VALUE,
            VOXCPM_HIFI_CLONE,
            VOXCPM_INFERENCE_TIMESTEPS,
            VOXCPM_RETRY_BADCASE,
            VOXCPM_SEED,
        )
    except Exception:
        VOXCPM_CFG_VALUE = 2.0
        VOXCPM_HIFI_CLONE = True
        VOXCPM_INFERENCE_TIMESTEPS = 10
        VOXCPM_RETRY_BADCASE = True
        VOXCPM_SEED = 42

    return [
        {
            "key": "cfg_value",
            "label": "CFG / 引导强度",
            "type": "float",
            "min": 1.0,
            "max": 3.0,
            "step": 0.05,
            "default": float(VOXCPM_CFG_VALUE),
            "help": "VoxCPM2 CFG guidance；2.0 音色最稳，>2.5 易重复",
        },
        {
            "key": "inference_timesteps",
            "label": "推理步数",
            "type": "int",
            "min": 5,
            "max": 30,
            "step": 1,
            "default": int(VOXCPM_INFERENCE_TIMESTEPS),
            "help": "10 是质量/速度拐点；16+ 只提升细节",
        },
        {
            "key": "seed",
            "label": "随机种子",
            "type": "int",
            "min": -1,
            "max": 2147483647,
            "step": 1,
            "default": int(VOXCPM_SEED),
            "help": "-1 表示每次随机",
        },
        {
            "key": "retry_badcase",
            "label": "Badcase 重试",
            "type": "bool",
            "default": bool(VOXCPM_RETRY_BADCASE),
            "help": "音频明显过长时换种子重试（最多 3 次）",
        },
        {
            "key": "hifi_clone",
            "label": "Hi-fi 克隆",
            "type": "bool",
            "default": bool(VOXCPM_HIFI_CLONE),
            "help": "用参考片段+文稿复刻语气；关闭则仅克隆音色",
        },
    ]


def _fields_for(engine: str) -> list[dict[str, Any]]:
    if engine == "voxcpm":
        return _voxcpm_fields()
    return _SCHEMAS.get(engine, _CHATTERBOX_FIELDS)


def get_audio_param_schema(pipeline: str, target_language: str | None) -> dict[str, Any]:
    """Return the browser-facing schema for an audio pipeline/language."""
    pipeline = str(pipeline or "tts").strip().lower()
    if pipeline not in AUDIO_PIPELINES:
        raise ValueError(f"Unknown audio pipeline: {pipeline}")
    engine = resolve_audio_engine(target_language)
    return {
        "pipeline": pipeline,
        "language": str(target_language or "en"),
        "engine": engine,
        "engine_label": _ENGINE_LABELS.get(engine, engine),
        "fields": _fields_for(engine),
    }


def _clamp(value: float, low: float | None, high: float | None) -> float:
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


def _coerce(field: dict[str, Any], raw: Any) -> Any:
    kind = field.get("type")
    if kind == "bool":
        if isinstance(raw, str):
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        return bool(raw)
    value = int(float(raw)) if kind == "int" else float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{field['key']} must be finite")
    value = _clamp(value, field.get("min"), field.get("max"))
    return int(value) if kind == "int" else value


def _decode_raw(raw: Any) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, str):
        if raw.strip().lower() == "null":
            return {}
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("audio_params must be valid JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("audio_params must be a JSON object")
    return raw


def normalize_audio_params(
    raw: Any,
    target_language: str | None,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate/clamp a submitted params object and stamp its engine.

    *fallback* lets legacy flat form fields supply values for engines whose
    keys overlap those fields (Chatterbox temperature/cfg/exaggeration).
    The engine is always derived from the language; a client-supplied engine
    is ignored.
    """
    engine = resolve_audio_engine(target_language)
    submitted = _decode_raw(raw)
    fallback = fallback or {}
    out: dict[str, Any] = {"engine": engine}
    for field in _fields_for(engine):
        key = field["key"]
        value = submitted.get(key, fallback.get(key, field.get("default")))
        if value is None or value == "":
            value = field.get("default")
        out[key] = _coerce(field, value)
    return out


def normalize_job_params(
    raw: Any,
    target_language: str | None,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical nested job envelope.

    Shape::

        {
          "target_language": "yue",
          "audio": {...},
          "video": {"mode": ..., "blur": ..., "ocr_only": ...},
          "output": {"filename": ...}
        }

    ``raw`` may be a ``job_params``/``params`` JSON string, or a dict.  Old
    flat job rows are normalized through *fallback*.
    """
    fallback = fallback or {}
    effective_language = target_language or fallback.get("target_language") or "en"
    submitted = _decode_raw(raw)

    # Compatibility with a bare audio-params object (Phase 1 clients).
    if "audio" not in submitted and any(
        key in submitted
        for key in ("engine", "cfg_value", "inference_timesteps", "temperature", "cfg_weight", "exaggeration")
    ):
        audio_raw: Any = submitted
    else:
        audio_raw = submitted.get("audio")
    if audio_raw is None:
        audio_raw = fallback.get("audio_params")

    video_submitted = submitted.get("video")
    video_submitted = video_submitted if isinstance(video_submitted, dict) else {}
    video = {}
    for key in ("mode", "blur", "ocr_mode", "ocr_only"):
        value = video_submitted.get(key, fallback.get(key))
        if value is not None:
            video[key] = value

    output_submitted = submitted.get("output")
    output_submitted = output_submitted if isinstance(output_submitted, dict) else {}
    filename = output_submitted.get("filename", fallback.get("filename"))
    output = {"filename": filename} if filename is not None else {}

    out: dict[str, Any] = {
        "target_language": str(effective_language),
        "audio": normalize_audio_params(audio_raw, effective_language, fallback=fallback),
    }
    if video:
        out["video"] = video
    if output:
        out["output"] = output
    return out


def load_job_params(raw: Any) -> dict[str, Any] | None:
    """Parse a stored job envelope without filling defaults."""
    try:
        decoded = _decode_raw(raw)
    except ValueError:
        return None
    return decoded or None


def get_job_params(job_data: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical nested params for a runtime job dict."""
    cached = job_data.get("params")
    if isinstance(cached, dict) and "audio" in cached:
        return cached
    return normalize_job_params(
        job_data.get("job_params") or job_data.get("params"),
        job_data.get("target_language"),
        fallback=job_data,
    )


def get_video_params(job_data: dict[str, Any]) -> dict[str, Any]:
    """Return video options, preferring the nested envelope over flat columns."""
    params = get_job_params(job_data)
    video = params.get("video", {})
    out = {}
    for key in ("mode", "blur", "ocr_mode", "ocr_only"):
        value = video.get(key, job_data.get(key))
        if value is not None:
            out[key] = value
    return out


def get_output_params(job_data: dict[str, Any]) -> dict[str, Any]:
    """Return output options, preferring the nested envelope over flat columns."""
    params = get_job_params(job_data)
    output = params.get("output", {})
    filename = output.get("filename", job_data.get("filename"))
    return {"filename": filename} if filename is not None else {}
