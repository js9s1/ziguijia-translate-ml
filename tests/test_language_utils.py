"""Tests for language/script detection utilities."""

import pytest

from language_utils import (
    detect_dominant_script,
    detect_mixed_scripts,
    is_srt_timing_line,
    is_srt_index_line,
    CODE_POINT_RANGES,
    UNICODE_SCRIPTS,
)


class TestDetectDominantScript:
    def test_chinese(self):
        assert detect_dominant_script("你好世界") == "zh"

    def test_arabic(self):
        assert detect_dominant_script("مرحبا") == "ar"

    def test_cyrillic(self):
        assert detect_dominant_script("привет") == "ru"

    def test_thai(self):
        assert detect_dominant_script("สวัสดี") == "th"

    def test_japanese_hiragana(self):
        assert detect_dominant_script("こんにちは") == "ja"

    def test_japanese_katakana(self):
        assert detect_dominant_script("コンニチハ") == "ja"

    def test_japanese_mixed_mostly_cjk(self):
        # When Chinese chars dominate Hiragana/Katakana, should return zh
        text = "中国日本韩国" * 10 + "こん"  # mostly CJK
        assert detect_dominant_script(text) == "zh"

    def test_english_fallback(self):
        assert detect_dominant_script("hello world") == "en"

    def test_empty_string(self):
        assert detect_dominant_script("") == "en"


class TestDetectMixedScripts:
    def test_single_script(self):
        assert detect_mixed_scripts("hello") == {"Latin"}

    def test_mixed_cjk_latin(self):
        result = detect_mixed_scripts("hello 你好")
        assert "Latin" in result
        assert "CJK" in result

    def test_empty(self):
        assert detect_mixed_scripts("") == set()

    def test_numbers_only(self):
        assert detect_mixed_scripts("12345") == set()


class TestSRTLineDetection:
    def test_timing_line(self):
        assert is_srt_timing_line("00:00:01,000 --> 00:00:03,000")

    def test_not_timing_line(self):
        assert not is_srt_timing_line("hello world")

    def test_index_digits(self):
        assert is_srt_index_line("1")
        assert is_srt_index_line("42")

    def test_not_index(self):
        assert not is_srt_index_line("abc")
        assert not is_srt_index_line("")


class TestCodePointRanges:
    def test_all_ranges_have_valid_codepoints(self):
        for lang, lo, hi in CODE_POINT_RANGES:
            assert 0 < lo <= hi <= 0x10FFFF

    def test_all_unicode_scripts_compiled(self):
        for name, pattern in UNICODE_SCRIPTS.items():
            assert name in ("CJK", "Latin", "Cyrillic", "Arabic", "Devanagari",
                            "Thai", "Greek", "Hebrew")
