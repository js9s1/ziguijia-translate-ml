#!/usr/bin/env python3
"""Backfill segment_*_meta.json for all video jobs missing them.

Reads ``combined_segment_*.wav`` files, gets duration via ``ffprobe``,
extracts text/chunks/speaker from ``output_adjusted.srt``, pulls job
params from ``jobs.db``, and writes ``segment_*_meta.json``.

No heavy torch/tts imports — safe to run on a server.
"""

import argparse
import glob
import json
import os
import re
import sqlite3
import subprocess
import sys

# ── Helpers ──────────────────────────────────────────────────────


def _get_wav_duration(wav_path: str) -> float:
    """Return duration in seconds, or -1 on failure."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-select_streams", "a:0", wav_path],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if r.returncode != 0:
            return -1
        data = json.loads(r.stdout)
        return float(data["format"]["duration"])
    except Exception:
        return -1


def _extract_speaker(text: str):
    """Return (speaker, clean_text) or (None, text)."""
    m = re.match(r"^\s*(\w+)\s*:\s*(.*)", text, re.DOTALL)
    if m:
        return m.group(1).lower(), m.group(2)
    return None, text


def _split_text(text: str, max_len: int = 120):
    """Split text into chunks suitable for TTS (mirrors gen_audio.split_text)."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        chunk = text[:max_len]
        last_boundary = -1
        for sep in (". ", "! ", "? ", ".\n", "!\n", "?\n"):
            pos = chunk.rfind(sep)
            if pos > last_boundary:
                last_boundary = pos + 1
        if last_boundary > 0:
            chunks.append(text[:last_boundary])
            text = text[last_boundary:].lstrip()
        else:
            last_space = chunk.rfind(" ")
            if last_space > 0:
                chunks.append(text[:last_space])
                text = text[last_space:].lstrip()
            else:
                chunks.append(chunk)
                text = text[max_len:].lstrip()
    return chunks


# ── Main ─────────────────────────────────────────────────────────


def backfill_one(output_dir: str, db_path: str, dry_run: bool = False):
    """Backfill meta JSONs for a single job output directory."""
    audio_dir = os.path.join(output_dir, "audio")
    tmp_dir = os.path.join(audio_dir, "tmp")
    srt_path = os.path.join(audio_dir, "output_adjusted.srt")

    if not os.path.isdir(tmp_dir):
        return {"status": "skip", "reason": "no tmp dir"}

    wavs = sorted(glob.glob(os.path.join(tmp_dir, "combined_segment_*.wav")))
    if not wavs:
        return {"status": "skip", "reason": "no wav files"}

    metas_missing = 0
    metas_exist = 0
    for w in wavs:
        idx = int(re.search(r"combined_segment_(\d+)\.wav", w).group(1))
        meta = os.path.join(tmp_dir, f"segment_{idx}_meta.json")
        if os.path.exists(meta):
            metas_exist += 1
        else:
            metas_missing += 1

    if metas_missing == 0:
        return {"status": "ok", "reason": f"all {metas_exist} metas present"}

    # Load SRT if available
    import srt as srt_mod

    srt_map = {}
    if os.path.exists(srt_path):
        with open(srt_path, encoding="utf-8") as f:
            subs = list(srt_mod.parse(f.read()))
        for i, sub in enumerate(subs):
            srt_map[i] = sub.content.strip()

    # Load job params from DB
    access_code = os.path.basename(output_dir).rsplit("-", 1)[-1]
    params = {}
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT temperature, target_language, cfg_weight, exaggeration FROM jobs WHERE access_code = ?",
            (access_code,),
        ).fetchone()
        if row:
            params = dict(row)
        conn.close()
    except Exception:
        pass  # params stay at defaults

    temperature = params.get("temperature", 0.8)
    target_language = params.get("target_language", "en")
    cfg_weight = params.get("cfg_weight", 0.5)
    exaggeration = params.get("exaggeration", 0.5)

    # Build the single cache_meta.json
    created = 0
    errors = 0
    cache = {}
    for w in wavs:
        idx = int(re.search(r"combined_segment_(\d+)\.wav", w).group(1))

        dur = _get_wav_duration(w)
        if dur < 0:
            errors += 1
            print(f"  [ERROR] ffprobe failed for seg {idx}", file=sys.stderr)
            continue

        content = srt_map.get(idx, "")
        clean_content = _extract_speaker(content)[1]
        chunks = _split_text(clean_content)

        cache[str(idx)] = {
            "content": clean_content,
            "chunks": chunks,
            "duration": dur,
        }
        created += 1

    cache_path = os.path.join(tmp_dir, "cache_meta.json")
    old_format_exists = any(os.path.exists(os.path.join(tmp_dir, f"segment_{i}_meta.json")) for i in range(10000))

    if not dry_run:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    # Clean up old per-segment meta JSONs — they're now redundant
    for fname in os.listdir(tmp_dir):
        if fname.startswith("segment_") and fname.endswith("_meta.json"):
            if not dry_run:
                os.remove(os.path.join(tmp_dir, fname))
            created += 1  # count cleanup

    return {
        "status": "done" if created > 0 else "skip",
        "created": created,
        "errors": errors,
        "pre_existing": metas_exist,
    }


def main():
    parser = argparse.ArgumentParser(description="Backfill segment_*_meta.json files")
    parser.add_argument("--dry-run", action="store_true", help="Count only, don't write")
    parser.add_argument("--job", help="Specific access_code or output_dir")
    parser.add_argument("--all", action="store_true", help="Backfill all video jobs")
    args = parser.parse_args()

    BASE_VIDEO = "/home/js9s/子归家/video"
    DB = "/home/js9s/子归家/code_ml/chatterbox-server/jobs.db"

    if args.job:
        # Single job
        if os.path.isabs(args.job):
            out_dir = args.job
        else:
            # Find by access_code
            conn = sqlite3.connect(DB)
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT output_dir FROM jobs WHERE access_code = ?", (args.job,)).fetchone()
            conn.close()
            if not row:
                print(f"Job {args.job} not found in DB", file=sys.stderr)
                sys.exit(1)
            out_dir = row["output_dir"]

        result = backfill_one(out_dir, DB, dry_run=args.dry_run)
        print(f"{out_dir}: {result}")
        return

    if not args.all:
        parser.print_help()
        return

    # Find all video job directories with combined_segment_*.wav
    total_created = 0
    total_errors = 0
    job_dirs = sorted(glob.glob(os.path.join(BASE_VIDEO, "*/")))
    for jd in job_dirs:
        jd = jd.rstrip("/")
        access_code = os.path.basename(jd).rsplit("-", 1)[-1]
        result = backfill_one(jd, DB, dry_run=args.dry_run)
        if result["status"] == "done":
            total_created += result.get("created", 0)
            total_errors += result.get("errors", 0)
            print(
                f"{os.path.basename(jd)}: +{result['created']} meta JSONs "
                f"({result.get('pre_existing', 0)} already exist)"
                f"{' (' + str(result['errors']) + ' errors)' if result.get('errors') else ''}"
            )

    print()
    print(f"Total created: {total_created}")
    print(f"Total errors:  {total_errors}")


if __name__ == "__main__":
    main()
