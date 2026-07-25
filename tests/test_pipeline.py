"""Tests for pipeline: atempo filter builder, file validation.

These tests target pure functions only and do NOT require the Flask app,
GPU, PyTorch, subprocess execution, or database infrastructure.
"""

import os
import sys
import tempfile
from unittest import mock

import pytest

# ── Module-level mocks so we can import pipeline without pulling in
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

from pipeline import (  # noqa: E402
    _build_atempo_filter,
    validate_files,
)


class TestBuildAtempoFilter:
    def test_normal_range(self):
        result = _build_atempo_filter(1.5)
        assert "atempo=1.500000" in result

    def test_lower_bound(self):
        result = _build_atempo_filter(0.5)
        assert "atempo=0.500000" in result

    def test_upper_bound(self):
        result = _build_atempo_filter(2.0)
        assert "atempo=2.000000" in result

    def test_above_2_chains_multiple(self):
        result = _build_atempo_filter(3.0)
        assert "atempo=2.0" in result
        assert result.count(",") >= 1

    def test_above_4_chains_three(self):
        result = _build_atempo_filter(5.0)
        assert result.count(",") >= 2

    def test_below_point_five_chains(self):
        result = _build_atempo_filter(0.25)
        assert result.count(",") >= 1

    def test_exactly_one(self):
        result = _build_atempo_filter(1.0)
        assert "atempo=1.000000" in result

    def test_very_high_value(self):
        result = _build_atempo_filter(10.0)
        assert result.count(",") >= 2

    def test_very_low_value(self):
        result = _build_atempo_filter(0.1)
        assert result.count(",") >= 2

    def test_all_parts_use_6_decimal_places(self):
        result = _build_atempo_filter(1.5)
        parts = result.split(",")
        for part in parts:
            assert "=" in part
            value = part.split("=")[1]
            assert "." in value
            assert len(value.split(".")[1]) == 6


class TestValidateFiles:
    def test_all_files_exist(self):
        with tempfile.NamedTemporaryFile(delete=False) as f1, \
             tempfile.NamedTemporaryFile(delete=False) as f2:
            f1.write(b"a")
            f2.write(b"b")
        try:
            validate_files([f1.name, f2.name])
        finally:
            os.unlink(f1.name)
            os.unlink(f2.name)

    def test_missing_file_raises(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"a")
        try:
            with pytest.raises(RuntimeError, match="missing"):
                validate_files([f.name, "/nonexistent/path_xyz.123"])
        finally:
            os.unlink(f.name)

    def test_empty_list_passes(self):
        validate_files([])

    def test_label_included_in_error(self):
        with pytest.raises(RuntimeError, match="FooBar"):
            validate_files(["/nonexistent/path_123.xyz"], label="FooBar")
