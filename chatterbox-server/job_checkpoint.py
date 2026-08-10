"""Checkpoint CRUD — recording and invalidating pipeline step progress.

Extracted from ``jobqueue.py``.  All functions take the ``JobQueue`` instance
as their first parameter (``jq``) and access DB / attributes through it.
"""

import logging
import os

from config import CHECKPOINT_ORDER, FILENAME_TO_CHECKPOINT_STEP

logger = logging.getLogger(__name__)

_STEP_ARTIFACTS: dict[str, list[str]] = {
    "translate": ["translated.srt"],
    "audio": [
        "audio/output.wav",
        "audio/output_adjusted.srt",
        "audio/output-final-modified.srt",
        "audio/changed_segments.json",
        "audio_tracks/output.wav",
        "audio_tracks/output_adjusted.srt",
        "audio_tracks/output-final-modified.srt",
        "audio_tracks/changed_segments.json",
    ],
    "video": ["output_modified.mp4", "output_final.mp4"],
}


def set_checkpoint(jq, access_code: str, checkpoint: str):
    """Record that a job has completed up to a certain step."""
    conn = jq._get_conn()
    conn.execute("UPDATE jobs SET checkpoint = ? WHERE access_code = ?", (checkpoint, access_code))
    conn.commit()


def get_checkpoint(jq, access_code: str) -> str:
    """Return the highest completed checkpoint step for a job."""
    conn = jq._get_conn()
    row = conn.execute("SELECT checkpoint FROM jobs WHERE access_code = ?", (access_code,)).fetchone()
    return row[0] if row and row[0] else ""


def clear_checkpoint_for_file(jq, access_code: str, file_path: str):
    """Remove checkpoint steps whose output file was deleted.

    When a user deletes a file from the result page, the corresponding
    checkpoint step is cleared so the step will re-run on resubmit.
    """
    ckpt = get_checkpoint(jq, access_code)
    if not ckpt:
        return
    parts = [s for s in ckpt.split(",") if s]
    if not parts:
        return

    basename = os.path.basename(file_path)
    steps_to_clear: set[str] = set()

    # Exact-match lookups
    step = FILENAME_TO_CHECKPOINT_STEP.get(basename)
    if step:
        steps_to_clear.add(step)

    # Pattern-based lookups for job-specific file types
    if basename == "output_modified.mp4":
        steps_to_clear.add("video")
    elif "_decompressed.mov" in basename:
        steps_to_clear.add("decompress")
    elif basename.endswith("_trimmed.mp4"):
        steps_to_clear.add("trim")
    elif basename.endswith(".mp4") and basename not in ("output_modified.mp4",):
        steps_to_clear.add("download")
    elif "audio" in file_path.replace("\\", "/").split("/"):
        steps_to_clear.add("audio")

    if not steps_to_clear:
        return

    new_parts = [p for p in parts if p not in steps_to_clear]
    if new_parts != parts:
        set_checkpoint(jq, access_code, ",".join(new_parts))


def invalidate_checkpoints_after(jq, access_code: str, step: str):
    """Remove all checkpoint steps *after* *step*, keeping *step* intact.

    Also deletes the output artifacts of those steps so the job
    can cleanly re-generate them.
    """
    ORDER = CHECKPOINT_ORDER
    ckpt = get_checkpoint(jq, access_code)
    if not ckpt:
        logger.info("invalidate_checkpoints_after(%s, %s): no checkpoint", access_code, step)
    parts = [s for s in (ckpt or "").split(",") if s]

    try:
        idx = ORDER.index(step)
    except ValueError:
        new_parts = [p for p in parts if p != step]
    else:
        new_parts = [p for p in parts if p not in ORDER[idx + 1 :]]

    removed_from_ckpt = set(parts) - set(new_parts)
    steps_to_regen = ORDER[idx + 1 :] if step in ORDER else []

    logger.info(
        "invalidate_checkpoints_after(%s, %s): parts=%s, new=%s, removed_from_ckpt=%s, steps_to_regen=%s",
        access_code,
        step,
        parts,
        new_parts,
        removed_from_ckpt,
        steps_to_regen,
    )

    if new_parts != parts:
        set_checkpoint(jq, access_code, ",".join(new_parts))

    # Purge artifacts for steps that will re-run
    if steps_to_regen:
        conn = jq._get_conn()
        row = conn.execute("SELECT output_dir FROM jobs WHERE access_code = ?", (access_code,)).fetchone()
        output_dir = row[0] if row else None
        if output_dir and os.path.isdir(output_dir):
            _purge_step_artifacts(jq, output_dir, set(steps_to_regen))


def _purge_step_artifacts(jq, output_dir: str, steps: set[str]):
    """Delete output files produced by the given checkpoint steps.

    For the audio step, only the final output files are removed;
    the ``tmp/`` subdirectory (holding per-segment cached wavs and
    meta JSONs) is preserved so unchanged segments can skip re-generation.
    """
    import shutil

    for step in steps:
        for rel in _STEP_ARTIFACTS.get(step, []):
            path = os.path.join(output_dir, rel)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    logger.info("Purged directory: %s", path)
                elif os.path.isfile(path):
                    os.remove(path)
                    logger.info("Purged file: %s", path)
            except Exception as e:
                logger.warning("Failed to purge %s: %s", path, e)


def set_checkpoint_edited(jq, access_code: str, edited: bool = True):
    """Mark that the checkpoint has been edited (user edited an SRT)."""
    conn = jq._get_conn()
    conn.execute("UPDATE jobs SET checkpoint_edited = ? WHERE access_code = ?", (1 if edited else 0, access_code))
    conn.commit()


def get_checkpoint_edited(jq, access_code: str) -> bool:
    """Return True if the user has edited a checkpoint-level file."""
    conn = jq._get_conn()
    row = conn.execute("SELECT checkpoint_edited FROM jobs WHERE access_code = ?", (access_code,)).fetchone()
    return bool(row and row[0])


def set_edited_srt_file(jq, access_code: str, filename: str):
    """Record that a specific SRT file has been edited by the user."""
    conn = jq._get_conn()
    row = conn.execute("SELECT edited_srt_files FROM jobs WHERE access_code = ?", (access_code,)).fetchone()
    existing = row[0] if row and row[0] else ""
    files = set(f for f in existing.split(",") if f)
    files.add(filename)
    new_val = ",".join(sorted(files))
    conn.execute("UPDATE jobs SET edited_srt_files = ? WHERE access_code = ?", (new_val, access_code))
    conn.commit()


def clear_edited_srt_files(jq, access_code: str):
    """Clear all recorded edited SRT files (called on resubmit)."""
    conn = jq._get_conn()
    conn.execute("UPDATE jobs SET edited_srt_files = '' WHERE access_code = ?", (access_code,))
    conn.commit()


def get_edited_srt_files(jq, access_code: str) -> list[str]:
    """Return the list of edited SRT filenames for a job."""
    conn = jq._get_conn()
    row = conn.execute("SELECT edited_srt_files FROM jobs WHERE access_code = ?", (access_code,)).fetchone()
    if row and row[0]:
        return [f for f in row[0].split(",") if f]
    return []
