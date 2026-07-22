"""Shared language/script detection utilities.

Unicode script ranges used by both ``chatterbox_server._validate_srt_language``
and ``video_ocr_job._detect_srt_language``.
"""

import re
import os

# Language code → full name mapping (single source of truth for the backend).
# Frontend fetches this via /api/languages.
LANG_NAME_MAP: dict[str, str] = {
    "ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "fi": "Finnish", "fr": "French",
    "he": "Hebrew", "hi": "Hindi", "it": "Italian", "ja": "Japanese",
    "ko": "Korean", "ms": "Malay", "nl": "Dutch", "no": "Norwegian",
    "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "sv": "Swedish",
    "sw": "Swahili", "tr": "Turkish", "zh": "Chinese",
    "vi": "Vietnamese", "th": "Thai",
}

# Unicode script ranges for language detection/validation.
# Keys are script names; values are compiled regexes that match a single
# character in the range.
UNICODE_SCRIPTS: dict[str, re.Pattern] = {
    "CJK":        re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]'),
    "Latin":      re.compile(r'[a-zA-Z]'),
    "Cyrillic":   re.compile(r'[\u0400-\u04ff]'),
    "Arabic":     re.compile(r'[\u0600-\u06ff]'),
    "Devanagari": re.compile(r'[\u0900-\u097f]'),
    "Thai":       re.compile(r'[\u0e00-\u0e7f]'),
    "Greek":      re.compile(r'[\u0370-\u03ff]'),
    "Hebrew":     re.compile(r'[\u0590-\u05ff]'),
}

# Code-point ranges for more detailed detection (used by video_ocr_job).
# Each entry maps a two-letter language code to (start, end) codepoint ranges.
CODE_POINT_RANGES: list[tuple[str, int, int]] = [
    ("zh", 0x4E00, 0x9FFF),
    ("zh", 0x3400, 0x4DBF),
    ("ja", 0x3040, 0x309F),  # Hiragana
    ("ja", 0x30A0, 0x30FF),  # Katakana
    ("ko", 0xAC00, 0xD7AF),  # Hangul
    ("ar", 0x0600, 0x06FF),  # Arabic
    ("ru", 0x0400, 0x04FF),  # Cyrillic
    ("el", 0x0370, 0x03FF),  # Greek
    ("he", 0x0590, 0x05FF),  # Hebrew
    ("hi", 0x0900, 0x097F),  # Devanagari
    ("th", 0x0E00, 0x0E7F),  # Thai
]


def detect_dominant_script(text: str) -> str:
    """Detect the dominant language via Unicode code-point ranges.

    Returns a two-letter language code, or ``"en"`` as the fallback.
    """
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for lang, lo, hi in CODE_POINT_RANGES:
            if lo <= cp <= hi:
                counts[lang] = counts.get(lang, 0) + 1
                break

    if not counts:
        return "en"

    best = max(counts, key=counts.get)

    # Heuristic: ja+zh together → prefer ja (Hiragana/Katakana is distinctive)
    if best == "ja" and counts.get("zh", 0) > counts["ja"] * 3:
        return "zh"

    return best


def detect_mixed_scripts(text: str) -> set[str]:
    """Return set of script names found in *text* (used for validation).

    Returns an empty set if no recognisable scripts are found.
    """
    found: set[str] = set()
    for name, pattern in UNICODE_SCRIPTS.items():
        if pattern.search(text):
            found.add(name)
    return found


def is_srt_timing_line(line: str) -> bool:
    """Return True if *line* looks like an SRT timing line (contains ``-->``)."""
    return "-->" in line


def is_srt_index_line(line: str) -> bool:
    """Return True if *line* looks like an SRT subtitle index (digits only)."""
    stripped = line.strip()
    return bool(stripped) and stripped.isdigit()
