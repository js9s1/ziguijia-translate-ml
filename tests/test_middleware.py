"""Tests for middleware: rate limiting, CSRF, login_required, file validation."""

import os
import io
import tempfile

import pytest

from middleware import (
    safe_file_path,
    parse_float_param,
    parse_job_params,
    validate_file_upload,
    _SRT_TIMING_RE,
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
