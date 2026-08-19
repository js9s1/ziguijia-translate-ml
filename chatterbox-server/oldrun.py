"""Oldrun SRT list — scanning, indexing, and static HTML generation.

Extracted from chatterbox_server.py.
"""

import gzip
import json
import logging
import os
import pickle as _pickle
import re

import jinja2

logger = logging.getLogger(__name__)

BASE_DIR_ = os.path.dirname(os.path.abspath(__file__))
HTML_DIR = os.path.join(BASE_DIR_, "html")
OLDRUN_SRT_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR_)), "batch", "oldrun")
OLDRUN_SRT_TIMESTAMP = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR_)), "batch", "list_updated.timestamp")
OFFICIAL_TRANS_DIR = os.path.join(OLDRUN_SRT_DIR, "official_trans")

ZH_FLAG = "\U0001f1e8\U0001f1f3"
EN_FLAG = "\U0001f1ec\U0001f1e7"

# Language codes whose flag is not the regional indicators of the code itself.
LANG_FLAG_OVERRIDES = {
    "en": EN_FLAG,  # English -> GB
    "ja": "\U0001f1ef\U0001f1f5",  # Japanese -> JP
    "ko": "\U0001f1f0\U0001f1f7",  # Korean -> KR
}

INDEX_MARKER_START = "<!-- gt-srt-links:start -->"
INDEX_MARKER_END = "<!-- gt-srt-links:end -->"
INDEX_ANCHOR = '<a href="/srt-zh+en.html" class="nav-dropdown-item">🇨🇳+🇬🇧 zh+en</a>'

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
    current_dirs = sorted(
        {
            os.path.join(OLDRUN_SRT_DIR, d)
            for d in os.listdir(OLDRUN_SRT_DIR)
            if os.path.isdir(os.path.join(OLDRUN_SRT_DIR, d)) and d != "official_trans"
        }
    )

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


def _cc_flag(cc: str) -> str:
    """Flag emoji for a language code (de -> 🇩🇪, ja -> 🇯🇵)."""
    cc = cc.lower()
    if cc in LANG_FLAG_OVERRIDES:
        return LANG_FLAG_OVERRIDES[cc]
    out = []
    for ch in cc:
        if "a" <= ch <= "z":
            out.append(chr(ord(ch) - ord("a") + 0x1F1E6))
        else:
            out.append(ch)
    return "".join(out)


def _flag_for_lang(lang: str) -> str:
    """Flag emoji(s) for a lang label (zh, en, zh+cc)."""
    if lang == "zh":
        return ZH_FLAG
    if lang == "en":
        return EN_FLAG
    if lang == "zh+en":
        return ZH_FLAG + EN_FLAG
    if lang.startswith("zh+"):
        return ZH_FLAG + _cc_flag(lang[3:])
    return lang


def _collect_official_trans() -> dict[str, list[dict]]:
    """Scan batch/oldrun/official_trans/{cc}/ and return {cc: [{name, path}]}.

    The official_trans directory holds officially translated (bilingual)
    subtitles copied over from the ground-truth pipeline — 'en' belongs on the
    zh+en page, other country codes get their own zh+{cc} page.
    """
    out: dict[str, list[dict]] = {}
    if not os.path.isdir(OFFICIAL_TRANS_DIR):
        return out
    for cc in sorted(os.listdir(OFFICIAL_TRANS_DIR)):
        cc_dir = os.path.join(OFFICIAL_TRANS_DIR, cc)
        if not os.path.isdir(cc_dir):
            continue
        files = []
        for dirpath, _, names in os.walk(cc_dir):
            for name in names:
                if name.lower().endswith(".srt"):
                    files.append({"name": name, "path": os.path.join(dirpath, name)})
        files.sort(key=lambda x: x["name"])
        out[cc] = files
    return out


def _merge_unique(base: list[dict], extra: list[dict]) -> list[dict]:
    """Append extra entries to base, deduped by path, sorted by name."""
    seen = {f["path"] for f in base}
    merged = list(base)
    for f in extra:
        if f["path"] not in seen:
            merged.append(f)
    merged.sort(key=lambda x: x["name"])
    return merged


def _update_index_dropdown(ccs: list[str]):
    """Refresh the zh+{cc} items (with flags) in the index.html subtitle dropdown."""
    index_path = os.path.join(HTML_DIR, "index.html")
    if not os.path.isfile(index_path):
        return

    with open(index_path, encoding="utf-8") as f:
        html = f.read()

    items = [
        f'                    <a href="/srt-zh+{cc}.html" class="nav-dropdown-item">{ZH_FLAG}+{_cc_flag(cc)} zh+{cc}</a>'
        for cc in ccs
    ]
    block = INDEX_MARKER_START + "\n" + "\n".join(items) + "\n" + INDEX_MARKER_END
    pattern = re.compile(re.escape(INDEX_MARKER_START) + r".*?" + re.escape(INDEX_MARKER_END), re.DOTALL)
    if pattern.search(html):
        html = pattern.sub(lambda _m: block, html)
    else:
        if INDEX_ANCHOR not in html:
            return
        html = html.replace(INDEX_ANCHOR, INDEX_ANCHOR + "\n" + block)

    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Updated index.html dropdown: zh+%s", ", ".join(ccs) or "()")


def _write_static_html(lang: str, files: list[dict]):
    """Write a static HTML file to HTML_DIR with embedded data and search."""
    flag = _flag_for_lang(lang)
    title = f"字幕列表 - {flag} {lang}"

    langs = [
        ("zh", ZH_FLAG, "zh"),
        ("en", EN_FLAG, "en"),
        ("zh+en", ZH_FLAG + EN_FLAG, "zh+en"),
    ]
    for cc in _collect_official_trans():
        if cc == "en":
            continue
        langs.append((f"zh+{cc}", ZH_FLAG + _cc_flag(cc), f"zh+{cc}"))
    lang_options = [
        {"url": f"/srt-{code}.html", "flag": lflag, "label": label, "active": code == lang}
        for code, lflag, label in langs
    ]

    html = _SRT_LIST_TEMPLATE.render(
        title=title,
        flag=flag,
        lang=lang,
        lang_options=lang_options,
        list_url=f"/srt-{lang}.html",
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
    Covers everything under batch/oldrun, including official_trans: 'en' is
    merged into the zh+en page, other country codes get their own zh+{cc} page.
    """
    try:
        ts_mtime = os.path.getmtime(OLDRUN_SRT_TIMESTAMP)
    except OSError:
        ts_mtime = None

    gt = _collect_official_trans()
    page_langs = ["zh", "en", "zh+en"] + [f"zh+{cc}" for cc in gt if cc != "en"]

    force_full = False
    if ts_mtime is not None:
        for lang in page_langs:
            html_path = os.path.join(HTML_DIR, f"srt-{lang}.html")
            if not os.path.isfile(html_path):
                force_full = True
                break
            if os.path.getmtime(html_path) < ts_mtime:
                force_full = True
                break
        else:
            logger.info("no rebuild needed — HTML files are up to date")
            return

    zh, en, zh_en, changed = _collect_incremental(force=force_full)
    if not changed:
        for lang in page_langs:
            if not os.path.isfile(os.path.join(HTML_DIR, f"srt-{lang}.html")):
                changed = True
                break
    if not changed:
        logger.info("mtime gate triggered rebuild, but no new dirs — forcing HTML rewrite anyway")

    zh_en = _merge_unique(zh_en, gt.get("en", []))

    _write_static_html("zh", zh)
    _write_static_html("en", en)
    _write_static_html("zh+en", zh_en)
    for cc, files in sorted(gt.items()):
        if cc == "en":
            continue
        _write_static_html(f"zh+{cc}", files)

    _update_index_dropdown(sorted(cc for cc in gt if cc != "en"))
