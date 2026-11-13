"""Tests for gen_video delogo rectangle scaling."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from gen_video import _delogo_filter  # noqa: E402


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
