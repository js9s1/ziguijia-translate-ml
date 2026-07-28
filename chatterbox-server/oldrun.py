"""Oldrun SRT list — scanning, indexing, and static HTML generation.

Extracted from chatterbox_server.py.
"""

import gzip
import json
import logging
import os
import pickle as _pickle

import jinja2

logger = logging.getLogger(__name__)

BASE_DIR_ = os.path.dirname(os.path.abspath(__file__))
HTML_DIR = os.path.join(BASE_DIR_, "html")
OLDRUN_SRT_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR_)), "batch", "oldrun")
OLDRUN_SRT_TIMESTAMP = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR_)), "batch", "list_updated.timestamp")

# Jinja2 template for the SRT list page, loaded from file.
_SRT_LIST_TEMPLATE_PATH = os.path.join(BASE_DIR_, "templates", "srt_list.html")
with open(_SRT_LIST_TEMPLATE_PATH, encoding="utf-8") as _f:
    _SRT_LIST_TEMPLATE = jinja2.Template(_f.read())


def _load_index() -> dict | None:
    """Load the incremental index from the timestamp file (pickle format)."""
    try:
        with open(OLDRUN_SRT_TIMESTAMP, "rb") as f:
            header = f.read(1)
            if header == b"\x80":
                f.seek(0)
                data = _pickle.load(f)
            elif header == b"{":
                return None
            else:
                return None
        if isinstance(data, dict) and "scanned_dirs" in data:
            return data
    except (OSError, _pickle.UnpicklingError, EOFError):
        pass
    return None


def _save_index(index: dict):
    """Atomically write the index into the timestamp file using pickle."""
    tmp = OLDRUN_SRT_TIMESTAMP + ".tmp"
    with open(tmp, "wb") as f:
        _pickle.dump(index, f, protocol=_pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, OLDRUN_SRT_TIMESTAMP)


def _scan_dir(dirpath: str) -> tuple[list[dict], list[dict], list[dict]]:
    """Walk a single directory tree and return (zh_entries, en_entries, zh_en_entries)."""
    zh, en, zh_en = [], [], []
    for root, _, filenames in os.walk(dirpath):
        for name in filenames:
            if not name.lower().endswith(".srt"):
                continue
            entry = {"name": name, "path": os.path.join(root, name)}
            low = name.lower()
            if low.endswith(".zh+en.srt"):
                zh_en.append(entry)
            elif low.endswith(".en.srt"):
                en.append(entry)
            else:
                zh.append(entry)
    return zh, en, zh_en


def _collect_incremental(force: bool = False) -> tuple[list[dict], list[dict], list[dict], bool]:
    """Return (zh, en, zh_en, changed) — walks new dirs or full rescan if forced."""
    index = _load_index()

    if not os.path.isdir(OLDRUN_SRT_DIR):
        return [], [], [], False
    current_dirs = sorted({
        os.path.join(OLDRUN_SRT_DIR, d)
        for d in os.listdir(OLDRUN_SRT_DIR)
        if os.path.isdir(os.path.join(OLDRUN_SRT_DIR, d))
    })

    if force or index is None:
        zh, en, zh_en = [], [], []
        for d in current_dirs:
            z, e, ze = _scan_dir(d)
            zh.extend(z)
            en.extend(e)
            zh_en.extend(ze)
        zh.sort(key=lambda x: x["name"])
        en.sort(key=lambda x: x["name"])
        zh_en.sort(key=lambda x: x["name"])
        _save_index({"zh": zh, "en": en, "zh_en": zh_en, "scanned_dirs": current_dirs})
        return zh, en, zh_en, True

    scanned = set(index["scanned_dirs"])
    new_dirs = [d for d in current_dirs if d not in scanned]

    if not new_dirs:
        return index["zh"], index["en"], index.get("zh_en", []), False

    zh = list(index["zh"])
    en = list(index["en"])
    zh_en = list(index.get("zh_en", []))
    for d in new_dirs:
        z, e, ze = _scan_dir(d)
        zh.extend(z)
        en.extend(e)
        zh_en.extend(ze)

    zh.sort(key=lambda x: x["name"])
    en.sort(key=lambda x: x["name"])
    zh_en.sort(key=lambda x: x["name"])

    _save_index({"zh": zh, "en": en, "zh_en": zh_en, "scanned_dirs": current_dirs})
    return zh, en, zh_en, True


def _write_static_html(lang: str, files: list[dict]):
    """Write a static HTML file to HTML_DIR with embedded data and search."""
    flag = "\U0001F1E8\U0001F1F3" if lang == "zh" else ("\U0001F1EC\U0001F1E7" if lang == "en" else "\U0001F1E8\U0001F1F3\U0001F1EC\U0001F1E7")
    title = f"字幕列表 - {flag} {lang}"

    LANGUAGES = [
        ("zh", "\U0001F1E8\U0001F1F3", "zh"),
        ("en", "\U0001F1EC\U0001F1E7", "en"),
        ("zh+en", "\U0001F1E8\U0001F1F3\U0001F1EC\U0001F1E7", "zh+en"),
    ]
    lang_options = [
        {"url": f"/srt-{l}.html", "flag": f, "label": lb, "active": l == lang}
        for l, f, lb in LANGUAGES
    ]

    html = _SRT_LIST_TEMPLATE.render(
        title=title,
        flag=flag,
        lang=lang,
        lang_options=lang_options,
    )

    filepath = os.path.join(HTML_DIR, f"srt-{lang}.html")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Wrote static srt list: %s (%d files)", filepath, len(files))

    json_bytes = json.dumps(files, ensure_ascii=False).encode("utf-8")
    gz_path = os.path.join(HTML_DIR, f"srt-{lang}.json.gz")
    with gzip.open(gz_path, "wb", compresslevel=9) as f:
        f.write(json_bytes)
    gz_size = os.path.getsize(gz_path)
    logger.info("Wrote compressed srt data: %s (%d bytes, %.0f%%)", gz_path, gz_size, 100 * gz_size / len(json_bytes))


def build_all_static_srt():
    """Rebuild static SRT list pages — gated on list_updated.timestamp.

    Called at startup and periodically (every 6 hours) via gunicorn_config.
    """
    try:
        ts_mtime = os.path.getmtime(OLDRUN_SRT_TIMESTAMP)
    except OSError:
        ts_mtime = None

    force_full = False
    if ts_mtime is not None:
        for lang in ("zh", "en", "zh+en"):
            html_path = os.path.join(HTML_DIR, f"srt-{lang}.html")
            if not os.path.isfile(html_path):
                force_full = True
                break
            if os.path.getmtime(html_path) < ts_mtime:
                force_full = True
                break
        else:
            return

    zh, en, zh_en, changed = _collect_incremental(force=force_full)
    if not changed:
        for lang in ("zh", "en", "zh+en"):
            if not os.path.isfile(os.path.join(HTML_DIR, f"srt-{lang}.html")):
                changed = True
                break
    if not changed:
        logger.info("mtime gate triggered rebuild, but no new dirs — forcing HTML rewrite anyway")

    _write_static_html("zh", zh)
    _write_static_html("en", en)
    _write_static_html("zh+en", zh_en)
