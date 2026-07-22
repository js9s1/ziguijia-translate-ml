"""Deferred imports for ML-heavy modules.

All processing modules are imported lazily to avoid loading
heavy dependencies (PyTorch, etc.) until a request actually needs them.

Usage:
    from lazy_imports import _lazy
    process_audio_file = _lazy("audio_job", "process_audio_file")
"""

import importlib

_MODULES: dict[str, object] = {}


def _lazy(module_name: str, attr: str):
    """Deferred import: load the module on first access and return the attribute."""
    import_key = f"{module_name}.{attr}"
    if import_key not in _MODULES:
        mod = importlib.import_module(module_name)
        _MODULES[import_key] = getattr(mod, attr)
    return _MODULES[import_key]
