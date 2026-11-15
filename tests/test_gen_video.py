"""Tests for gen_video delogo rectangle scaling."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from gen_video import (  # noqa: E402
    _build_concat_cmd,
    _build_segment_batch_cmd,
    _delogo_filter,
    _segment_filter,
)


def _parse_filter(filter_str: str) -> dict[str, int]:
    assert filter_str.startswith("format=yuv420p,delogo=")
    rect = {}
    for part in filter_str.split("delogo=", 1)[1].split(":show=", 1)[0].split(":"):
        key, value = part.split("=")
        rect[key] = int(value)
    return rect


def test_720p_matches_reference_rectangle():
    assert _parse_filter(_delogo_filter(1280, 720)) == {
        "x": 100,
        "y": 600,
        "w": 1060,
        "h": 80,
    }


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (320, 240),
        (640, 360),
        (854, 480),
        (1280, 720),
        (1920, 1080),
        (3840, 2160),
        (853, 481),  # odd dimensions
    ],
)
def test_rectangle_stays_inside_frame(width, height):
    rect = _parse_filter(_delogo_filter(width, height))
    assert rect["x"] >= 0
    assert rect["y"] >= 0
    assert rect["w"] >= 2
    assert rect["h"] >= 2
    assert rect["x"] + rect["w"] <= width
    assert rect["y"] + rect["h"] <= height
    # Still covers the same proportional band as the 720p reference.
    assert rect["y"] / height == pytest.approx(600 / 720, abs=0.01)
    assert (rect["y"] + rect["h"]) / height == pytest.approx(680 / 720, abs=0.01)


def test_854x480_rectangle():
    assert _parse_filter(_delogo_filter(854, 480)) == {
        "x": 66,
        "y": 400,
        "w": 706,
        "h": 52,
    }


@pytest.mark.parametrize(("width", "height"), [(None, 480), (854, None), (0, 0)])
def test_missing_dimensions_raise(width, height):
    with pytest.raises(RuntimeError, match="dimensions"):
        _delogo_filter(width, height)


_BATCH = [
    {"start": 0.0, "end": 2.0, "stretch": 1.0},
    {"start": 2.0, "end": 5.0, "stretch": 1.5},
]


def test_segment_filter_vaapi_and_cpu_labels():
    hw_filter, hw_label = _segment_filter(_BATCH, use_vaapi=True)
    assert "hwupload" in hw_filter
    assert hw_label == "[hw]"

    cpu_filter, cpu_label = _segment_filter(_BATCH, use_vaapi=False)
    assert "format=yuv420p" in cpu_filter
    assert "hwupload" not in cpu_filter
    assert cpu_label == "[out]"


def test_segment_batch_cmd_encoders():
    hw_cmd = _build_segment_batch_cmd("/tmp/in.mp4", _BATCH, "/tmp/out.mp4", use_vaapi=True)
    assert "h264_vaapi" in hw_cmd
    assert "-vaapi_device" in hw_cmd
    assert "libx264" not in hw_cmd

    cpu_cmd = _build_segment_batch_cmd("/tmp/in.mp4", _BATCH, "/tmp/out.mp4", use_vaapi=False)
    assert "libx264" in cpu_cmd
    assert "h264_vaapi" not in cpu_cmd
    assert "-vaapi_device" not in cpu_cmd


def test_concat_cmd_encoders():
    hw_cmd = _build_concat_cmd(["/tmp/a.mp4", "/tmp/b.mp4"], "/tmp/out.mp4", use_vaapi=True)
    assert any("concat=n=2" in part for part in hw_cmd)
    assert "h264_vaapi" in hw_cmd

    cpu_cmd = _build_concat_cmd(["/tmp/a.mp4", "/tmp/b.mp4"], "/tmp/out.mp4", use_vaapi=False)
    assert any("concat=n=2" in part for part in cpu_cmd)
    assert "libx264" in cpu_cmd
    assert "hwupload" not in " ".join(cpu_cmd)
