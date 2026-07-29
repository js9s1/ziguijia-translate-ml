"""Tests for video_util: SRT reading, timestamp normalization, translation heuristics.

These tests target pure functions only and do NOT require the Flask app,
GPU, PyTorch, or database infrastructure.
"""

import os
import sys
from unittest import mock

# ── Module-level mocks so we can import video_util without pulling in
#     the full chatterbox-server dependency tree ──
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "chatterbox-server"))

_mock_valkey = mock.MagicMock()

_mock_redis = mock.MagicMock()
_mock_redis.publish_job_status = mock.MagicMock()

sys.modules["valkey"] = _mock_valkey
sys.modules["valkey_util"] = _mock_redis
sys.modules["psutil"] = mock.MagicMock()
sys.modules["log_utils"] = mock.MagicMock()
sys.modules["config"] = mock.MagicMock()

from video_util import (  # noqa: E402
    _normalize_srt_timestamps,
    looks_untranslated,
    open_proc_log,
    read_srt_text,
)


class TestNormalizeSRTTimestamps:
    def test_dot_milliseconds_to_comma(self):
        result = _normalize_srt_timestamps("00:00:01.000 --> 00:00:03.500")
        assert result == "00:00:01,000 --> 00:00:03,500"

    def test_colon_milliseconds_to_comma(self):
        result = _normalize_srt_timestamps("00:00:01:000 --> 00:00:03:500")
        assert result == "00:00:01,000 --> 00:00:03,500"

    def test_single_dash_arrow(self):
        result = _normalize_srt_timestamps("00:00:01,000 -> 00:00:03,500")
        assert result == "00:00:01,000 --> 00:00:03,500"

    def test_combined_fixes(self):
        result = _normalize_srt_timestamps("00:00:01.000 -> 00:00:03.500")
        assert result == "00:00:01,000 --> 00:00:03,500"

    def test_no_change_when_already_valid(self):
        text = "00:00:01,000 --> 00:00:03,500"
        assert _normalize_srt_timestamps(text) == text

    def test_single_digit_fields(self):
        result = _normalize_srt_timestamps("0:00:01,000 --> 0:00:03,000")
        assert "0:00:01,000" in result

    def test_full_srt_block_unaffected(self):
        block = "1\n00:00:00,000 --> 00:00:02,000\nHello world\n"
        assert _normalize_srt_timestamps(block) == block

    def test_ffmpeg_style_in_full_block(self):
        block = "1\n00:00:00.000 --> 00:00:02.000\nHello world\n"
        result = _normalize_srt_timestamps(block)
        assert "00:00:00,000 --> 00:00:02,000" in result


class TestReadSRTText:
    def _write_srt(self, dir_path, name, content, encoding="utf-8"):
        path = os.path.join(dir_path, name)
        with open(path, "w", encoding=encoding) as f:
            f.write(content)
        return path

    def test_basic_utf8(self, tmp_path):
        srt_content = "1\n00:00:01,000 --> 00:00:03,000\nTest\n"
        path = self._write_srt(str(tmp_path), "test.srt", srt_content)
        result = read_srt_text(path)
        assert "Test" in result

    def test_utf8_with_bom(self, tmp_path):
        srt_content = "1\n00:00:01,000 --> 00:00:03,000\nTest\n"
        path = os.path.join(str(tmp_path), "bom.srt")
        with open(path, "wb") as f:
            f.write(b"\xef\xbb\xbf" + srt_content.encode("utf-8"))
        result = read_srt_text(path)
        assert "Test" in result

    def test_cr_only_line_endings_normalized(self, tmp_path):
        # Mac-style line endings (\r only, no \n)
        srt_content = "1\r00:00:01,000 --> 00:00:03,000\rTest\r"
        path = os.path.join(str(tmp_path), "cr.srt")
        with open(path, "wb") as f:
            f.write(srt_content.encode("utf-8"))
        result = read_srt_text(path)
        assert "\r" not in result
        assert "\n" in result

    def test_line_endings_normalized(self, tmp_path):
        srt_content = "1\r\n00:00:01,000 --> 00:00:03,000\r\nTest\r\n"
        path = os.path.join(str(tmp_path), "crlf.srt")
        with open(path, "wb") as f:
            f.write(srt_content.encode("utf-8"))
        result = read_srt_text(path)
        assert "\r\n" not in result
        assert "\n" in result

    def test_normalizes_timestamps(self, tmp_path):
        srt_content = "1\n00:00:01.000 --> 00:00:03.000\nTest\n\n"
        path = self._write_srt(str(tmp_path), "dots.srt", srt_content)
        result = read_srt_text(path)
        assert "00:00:01,000" in result


class TestLooksUntranslated:
    def test_cjk_present(self):
        assert looks_untranslated("你好世界测试字符", source_has_cjk=True) is True

    def test_cjk_absent(self):
        assert looks_untranslated("Hello world text", source_has_cjk=True) is False

    def test_few_cjk_characters(self):
        assert looks_untranslated("你好", source_has_cjk=True) is False

    def test_source_not_cjk(self):
        assert looks_untranslated("你好世界测试", source_has_cjk=False) is False

    def test_mixed_cjk_and_latin(self):
        text = "Hello 你好世界测试字符"
        assert looks_untranslated(text, source_has_cjk=True) is True


class TestOpenProcLog:
    def test_opens_file_for_appending(self, tmp_path):
        log_path = os.path.join(str(tmp_path), "test.log")
        with open(log_path, "w") as f:
            f.write("existing\n")
        with open_proc_log(log_path) as (fh, path):
            fh.write("new line\n")
        with open(log_path) as f:
            content = f.read()
        assert "existing\n" in content
        assert "new line\n" in content

    def test_closes_file_after_use(self, tmp_path):
        log_path = os.path.join(str(tmp_path), "close.log")
        with open_proc_log(log_path) as (fh, path):
            fh.write("data\n")
        assert fh.closed is True

    def test_closes_on_exception(self, tmp_path):
        log_path = os.path.join(str(tmp_path), "exc.log")
        try:
            with open_proc_log(log_path) as (fh, path):
                fh.write("data\n")
                raise RuntimeError("test")
        except RuntimeError:
            pass
        assert fh.closed is True
