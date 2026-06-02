"""Shared utilities for video job handlers."""

import os
import sys
import re
from contextlib import contextmanager

from jobqueue import get_job_queue
from log_utils import job_log
from config import HY_MT_DIR, LANG_MAP


@contextmanager
def open_proc_log(log_path: str):
    """Open *log_path* for appending and yield ``(file_handle, log_path)``.

    The handle is guaranteed to close on exit, even if the body raises.
    Use with ``subprocess.Popen(..., stdout=handle, stderr=handle)``.
    """
    fh = open(log_path, "a")
    try:
        yield fh, log_path
    finally:
        fh.close()


def looks_untranslated(text: str, source_has_cjk: bool = True) -> bool:
    """Heuristic: if source was CJK and output still has CJK, model likely refused to translate."""
    if not source_has_cjk:
        return False
    cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return cjk_count >= 3


def _get_hy_mt():
    """Import and return the hy_mt module (adds HY_MT_DIR to sys.path once)."""
    if HY_MT_DIR not in sys.path:
        sys.path.insert(0, HY_MT_DIR)
    import importlib
    return importlib.import_module("hy_mt")


def translate_segment(text: str, target_language: str, source_has_cjk: bool = True) -> str:
    """Translate a segment with up to 3 fallback strategies until output changes language.

    Uses hy_mt module loaded from HY_MT_DIR.
    """
    hy_mt = _get_hy_mt()
    result = text
    for attempt in range(3):
        if attempt == 0:
            result = hy_mt.translate_zh(text, target_language)
        elif attempt == 1:
            result = hy_mt.translate(text, target_language)
        else:
            model, tokenizer = hy_mt._get_model()
            messages = [
                {"role": "user",
                 "content": f"Translate the following Chinese sentence into {target_language}. Output ONLY the {target_language} translation, nothing else:\n\n{text}"},
            ]
            tokenized_chat = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False, return_tensors="pt"
            )
            outputs = model.generate(tokenized_chat.to(model.device), **hy_mt.GENERATION_KWARGS)
            result = tokenizer.decode(outputs[0][len(tokenized_chat[0]):], skip_special_tokens=True)
        if not looks_untranslated(result, source_has_cjk):
            return result
    return result


def translate_srt_file(srt_path: str, output_path: str, access_code: str, output_dir: str,
                       target_language_name: str, proc_log, log_file):
    """Translate all subtitle blocks in an SRT file and write the result.

    Args:
        srt_path: Path to input SRT file.
        output_path: Path to write translated SRT.
        access_code: Job access code for logging.
        output_dir: Job output directory for logging.
        target_language_name: Full language name (e.g. "English").
        proc_log: Open file handle for subprocess logging.
        log_file: Path to the log file for redirect_logging_to_file.
    """
    from contextlib import redirect_stdout, redirect_stderr
    from log_utils import redirect_logging_to_file

    hy_mt = _get_hy_mt()

    with redirect_stdout(proc_log), redirect_stderr(proc_log), redirect_logging_to_file(log_file):
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read()
        blocks = re.split(r"\n\n", content.strip())
        n_total = len([b for b in blocks if len(b.split("\n")) >= 3])
        translated_blocks = []
        count = 0
        for block in blocks:
            lines = block.split("\n")
            if len(lines) >= 3:
                idx = lines[0]
                time_range = lines[1]
                text = "\n".join(lines[2:])
                translated = translate_segment(text, target_language_name)
                translated_blocks.append(f"{idx}\n{time_range}\n{translated}")
                count += 1
                if count % 10 == 0 or count == n_total:
                    job_log(access_code, output_dir, f"  翻译进度: {count}/{n_total}")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(translated_blocks) + "\n")
        hy_mt.unload_model()


class CheckpointHelper:
    """Manages checkpoint steps for a job, persisted in jobs.db."""

    def __init__(self, access_code: str, output_dir: str, valid_steps: list[str] | None = None):
        self.access_code = access_code
        self.output_dir = output_dir
        self.valid_steps = valid_steps

    def done(self, name: str) -> bool:
        """Check if a checkpoint step has been completed."""
        ckpt = get_job_queue().get_checkpoint(self.access_code)
        parts = ckpt.split(",") if ckpt else []
        if self.valid_steps is not None:
            return name in self.valid_steps and name in parts
        return name in parts

    def mark(self, name: str):
        """Mark a checkpoint step as completed."""
        ckpt = get_job_queue().get_checkpoint(self.access_code)
        parts = ([s for s in ckpt.split(",") if s] if ckpt else []) + [name]
        get_job_queue().set_checkpoint(self.access_code, ",".join(parts))
        job_log(self.access_code, self.output_dir, f"  ✓ checkpoint {name}")
