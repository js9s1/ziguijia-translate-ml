"""Job type definitions shared between jobqueue and job_worker.

Extracted to avoid circular imports — both modules need ``JobStatus``,
the handler registry, and ``_get_run_func``.
"""

from collections.abc import Callable
from enum import Enum

_SKIP_QUEUE_INIT = False
"""When True, ``_load_pending_jobs`` is a no-op (spawn child context)."""


class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DELETED = "deleted"


# ── Handler registry ────────────────────────────────────────
# Each entry is ``(module_name, attr_name)`` so the import happens
# lazily and only the *needed* handler module is loaded.  This keeps
# the spawn child from importing PyTorch (audio_job.py) for every
# job type and blowing up virtual memory for no reason.
_HANDLER_MODULES: dict[str, tuple[str, str]] = {
    "_run_gen_audio": ("audio_job", "_run_gen_audio"),
    "_run_audio_segmentation_job": ("audio_job", "_run_audio_segmentation_job"),
    "_run_tts_job": ("tts_job", "_run_tts_job"),
    "_run_video_job": ("video_ning_job", "_run_video_job"),
    "_run_video_ning_ocr_job": ("video_ning_job", "_run_video_ning_ocr_job"),
    "_run_video_ning_ocr_translate_only_job": ("video_ning_job", "_run_video_ning_ocr_translate_only_job"),
    "_run_video_custom_job": ("video_custom_job", "_run_video_custom_job"),
    "_run_video_auto_job": ("video_custom_job", "_run_video_auto_job"),
    "_run_video_ocr_job": ("video_custom_job", "_run_video_ocr_job"),
    "_run_ocr_only_job": ("video_ocr_job", "_run_ocr_only_job"),
}

_JOB_TYPE_LABELS: dict[str, str] = {
    "_run_gen_audio": "音频生成",
    "_run_video_job": "宁视频翻译",
    "_run_video_custom_job": "自定义视频",
    "_run_tts_job": "语音合成",
    "_run_video_auto_job": "自动翻译视频",
    "_run_audio_segmentation_job": "音频分段合成",
    "_run_video_ocr_job": "OCR翻译视频",
    "_run_video_ning_ocr_job": "宁视频OCR翻译",
    "_run_video_ning_ocr_translate_only_job": "宁视频OCR仅翻译",
    "_run_ocr_only_job": "视频OCR提取字幕",
}


def _get_run_func(name: str) -> Callable | None:
    """Lazily import and return a single job handler function.

    Uses ``importlib`` so only the module containing *name* is loaded.
    This avoids pulling ``torch`` / ``ChatterboxMultilingualTTS`` into
    every spawn child for job types that don't need them.
    """
    spec = _HANDLER_MODULES.get(name)
    if spec is None:
        return None
    module_name, attr_name = spec
    import importlib

    mod = importlib.import_module(module_name)
    return getattr(mod, attr_name)


def _get_job_type_label(run_func_name: str) -> str:
    return _JOB_TYPE_LABELS.get(run_func_name, run_func_name or "未知")
