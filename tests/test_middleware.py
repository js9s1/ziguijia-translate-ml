"""Tests for middleware: rate limiting, CSRF, login_required, file validation."""

import os
import shutil
import subprocess

import pytest
from middleware import (
    _SRT_TIMING_RE,
    DURATION_MISMATCH_MESSAGE,
    parse_float_param,
    parse_job_params,
    safe_file_path,
    validate_srt_content,
    validate_srt_target_language,
    validate_text_matches_target_language,
    validate_video_srt_duration,
)


class TestSafeFilePath:
    def test_allowed_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        # tmp_path won't be in ALLOWED_FILE_DIRS, so we add it
        from middleware import ALLOWED_FILE_DIRS as _dirs

        original = list(_dirs)
        _dirs.append(os.path.realpath(str(tmp_path)))
        try:
            result = safe_file_path(str(f))
            assert result is not None
        finally:
            _dirs[:] = original

    def test_blocked_path(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        result = safe_file_path(str(f))
        assert result is None

    def test_non_existent_file(self):
        result = safe_file_path("/nonexistent/file_12345.txt")
        assert result is None


class TestParseFloatParam:
    def test_valid_float(self):
        assert parse_float_param({"x": "1.5"}, "x", 0.0) == 1.5

    def test_default(self):
        assert parse_float_param({}, "missing", 0.8) == 0.8

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_float_param({"x": "abc"}, "x", 0.0)


class TestParseJobParams:
    def test_all_params(self):
        source = {"temperature": "0.7", "target_language": "zh", "cfg_weight": "0.3"}
        result = parse_job_params(source)
        assert result["temperature"] == 0.7
        assert result["target_language"] == "zh"
        assert result["cfg_weight"] == 0.3

    def test_defaults(self):
        result = parse_job_params({})
        assert result["temperature"] == 0.6
        assert result["target_language"] == "en"
        assert result["cfg_weight"] == 0.25
        assert result["exaggeration"] == 0.3


class TestSRTTimingRegex:
    def test_valid_timing(self):
        assert _SRT_TIMING_RE.search("00:00:01,000 --> 00:00:03,000")

    def test_dot_milliseconds(self):
        assert _SRT_TIMING_RE.search("00:00:01.000 --> 00:00:03.000")

    def test_single_digit_hours(self):
        # The regex requires 2-digit hours; single-digit hours are not a match
        assert not _SRT_TIMING_RE.search("0:00:01,000 --> 0:00:03,000")

    def test_invalid(self):
        assert not _SRT_TIMING_RE.search("hello world")


ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg/ffprobe not available"
)


def _make_mp4(path, seconds):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={seconds}:size=160x120:rate=10",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        capture_output=True,
        check=True,
    )


def _write_srt(path, end_seconds):
    from datetime import timedelta

    end = timedelta(seconds=end_seconds)
    hours, rem = divmod(end.seconds, 3600)
    minutes, secs = divmod(rem, 60)
    stamp = f"{hours:02d}:{minutes:02d}:{secs:02d},{int(end.microseconds / 1000):03d}"
    path.write_text(f"1\n00:00:00,000 --> {stamp}\nhello\n", encoding="utf-8")


class TestValidateSrtLanguage:
    def _srt(self, lines):
        return "1\n00:00:01,000 --> 00:00:03,000\n" + lines + "\n"

    def _multi_srt(self, cues):
        parts = []
        for i, (content, start) in enumerate(cues, 1):
            parts.append(f"{i}\n{start} --> 00:00:05,000\n{content}\n")
        return "\n".join(parts)

    def test_single_language_passes(self):
        validate_srt_content(self._srt("这是一段中文字幕"), "SRT")
        validate_srt_content(self._srt("Dies ist ein deutscher Untertitel"), "SRT")

    def test_occasional_latin_in_cjk_passes(self):
        text = self._multi_srt(
            [
                ("释迦牟尼证到最高境界就是众生本来都是佛", "00:00:01,000"),
                ("这世界真的是没有任何一个人知道这一点", "00:00:03,000"),
                ("你一直以为你是个众生", "00:00:05,000"),
                ("一直认为你流浪在其他道", "00:00:07,000"),
                ("比如 www.a.com OK 就是这样", "00:00:09,000"),
                ("所以未来的日子你不断地了解佛是什么", "00:00:11,000"),
            ]
        )
        validate_srt_content(text, "SRT")

    def test_bilingual_cjk_latin_rejected(self):
        with pytest.raises(ValueError, match="多种语言"):
            validate_srt_content(self._srt("你就是如来\nDu bist der Tathagata"), "SRT")

    def test_bilingual_other_scripts_rejected(self):
        with pytest.raises(ValueError, match="多种语言"):
            validate_srt_content(self._srt("你好 мир"), "SRT")


class TestValidateSrtTargetLanguage:
    def _srt(self, lines):
        return "1\n00:00:01,000 --> 00:00:03,000\n" + lines + "\n"

    def test_latin_srt_with_ja_target_rejected(self):
        with pytest.raises(ValueError, match="make them consistent"):
            validate_srt_target_language(
                self._srt("Tu ne peux pas trouver la vacuité."), "ja"
            )

    def test_latin_srt_with_latin_target_passes(self):
        validate_srt_target_language(
            self._srt("Tu ne peux pas trouver la vacuité."), "fr"
        )

    def test_japanese_srt_with_ja_target_passes(self):
        validate_srt_target_language(self._srt("今日はいい天気ですね。"), "ja")

    def test_chinese_srt_with_ja_target_rejected(self):
        with pytest.raises(ValueError, match="make them consistent"):
            validate_srt_target_language(self._srt("这是一段中文字幕。"), "ja")

    def test_empty_srt_skips(self):
        validate_srt_target_language("", "ja")


class TestValidateTextTargetLanguage:
    def test_chinese_text_with_en_target_rejected(self):
        with pytest.raises(ValueError, match="make them consistent"):
            validate_text_matches_target_language("这是一段中文字幕", "en")

    def test_french_text_with_ja_target_rejected(self):
        with pytest.raises(ValueError, match="make them consistent"):
            validate_text_matches_target_language("Bonjour tout le monde", "ja")

    def test_matching_text_passes(self):
        validate_text_matches_target_language("这是一段中文字幕", "zh")
        validate_text_matches_target_language("Bonjour tout le monde", "fr")

    def test_no_letters_skips(self):
        validate_text_matches_target_language("123 !!! 456", "zh")

    def test_unknown_target_skips(self):
        validate_text_matches_target_language("任意文本", "xx")


class TestValidateVideoSrtDuration:
    @ffmpeg
    def test_matching_durations_pass(self, tmp_path):
        _make_mp4(tmp_path / "v.mp4", 10)
        _write_srt(tmp_path / "v.srt", 10)
        validate_video_srt_duration(str(tmp_path / "v.mp4"), str(tmp_path / "v.srt"))

    @ffmpeg
    def test_within_tolerance_pass(self, tmp_path):
        _make_mp4(tmp_path / "v.mp4", 10)
        _write_srt(tmp_path / "v.srt", 10.4)
        validate_video_srt_duration(str(tmp_path / "v.mp4"), str(tmp_path / "v.srt"))

    @ffmpeg
    def test_beyond_tolerance_raises(self, tmp_path):
        _make_mp4(tmp_path / "v.mp4", 10)
        _write_srt(tmp_path / "v.srt", 11)
        with pytest.raises(ValueError, match=DURATION_MISMATCH_MESSAGE):
            validate_video_srt_duration(str(tmp_path / "v.mp4"), str(tmp_path / "v.srt"))

    @ffmpeg
    def test_srt_shorter_than_video_raises(self, tmp_path):
        _make_mp4(tmp_path / "v.mp4", 11)
        _write_srt(tmp_path / "v.srt", 10)
        with pytest.raises(ValueError, match=DURATION_MISMATCH_MESSAGE):
            validate_video_srt_duration(str(tmp_path / "v.mp4"), str(tmp_path / "v.srt"))

    def test_boundary_exactly_five_percent_passes(self, tmp_path, monkeypatch):
        monkeypatch.setattr("middleware._probe_video_duration", lambda p: 100.0)
        monkeypatch.setattr("middleware._srt_duration_seconds", lambda p: 95.0)
        validate_video_srt_duration("v", "s")

    def test_just_beyond_boundary_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("middleware._probe_video_duration", lambda p: 100.0)
        monkeypatch.setattr("middleware._srt_duration_seconds", lambda p: 94.9)
        with pytest.raises(ValueError, match=DURATION_MISMATCH_MESSAGE):
            validate_video_srt_duration("v", "s")

    def test_unprobeable_video_passes(self, tmp_path):
        _write_srt(tmp_path / "v.srt", 10)
        validate_video_srt_duration(str(tmp_path / "nonexistent.mp4"), str(tmp_path / "v.srt"))

    def test_unparseable_srt_passes(self, tmp_path):
        (tmp_path / "v.srt").write_text("not an srt", encoding="utf-8")
        validate_video_srt_duration(str(tmp_path / "nonexistent.mp4"), str(tmp_path / "v.srt"))
