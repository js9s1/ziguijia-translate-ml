"""Tests for pipeline: atempo filter builder, file validation.

These tests target pure functions only and do NOT require the Flask app,
GPU, PyTorch, subprocess execution, or database infrastructure.
"""

import os
import subprocess
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

import pipeline  # noqa: E402
from pipeline import (  # noqa: E402
    _build_atempo_filter,
    normalize_video_number,
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


class TestNormalizeVideoNumber:
    def test_numeric_unchanged(self):
        assert normalize_video_number("10001") == "10001"

    def test_lowercase_prefix_uppercased(self):
        assert normalize_video_number("f1002") == "F1002"

    def test_uppercase_prefix_unchanged(self):
        assert normalize_video_number("A0001") == "A0001"

    def test_whitespace_stripped(self):
        assert normalize_video_number(" g0001 ") == "G0001"


class TestValidateFiles:
    def test_all_files_exist(self):
        with tempfile.NamedTemporaryFile(delete=False) as f1, tempfile.NamedTemporaryFile(delete=False) as f2:
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


def _write_srt(path):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("1\n00:00:00,000 --> 00:00:01,000\n测试\n")


class TestRunWhisperCkpt:
    def test_vulkan_primary_success(self, tmp_path, monkeypatch):
        out = tmp_path / "out"
        out.mkdir()
        srt = out / "whisper.srt"

        def fake_run(cmd, **kwargs):
            _write_srt(srt)
            return mock.Mock(returncode=0)

        run = mock.Mock(side_effect=fake_run)
        monkeypatch.setattr(pipeline.subprocess, "run", run)
        monkeypatch.setattr(pipeline, "WHISPER_BIN", "/opt/vk/whisper-cli")
        monkeypatch.setattr(pipeline, "WHISPER_LIB_DIR", "")
        monkeypatch.setattr(pipeline, "gpu_block_reason", lambda: None)
        ckpt = mock.Mock()
        ckpt.done.return_value = False

        result = pipeline.run_whisper_ckpt("audio.wav", str(out), "CODE", ckpt, mock.Mock())

        assert result == str(srt)
        assert run.call_count == 1
        cmd = run.call_args.args[0]
        assert cmd[0] == "/opt/vk/whisper-cli"
        assert "-fa" in cmd
        assert run.call_args.kwargs["env"]["GGML_VK_DISABLE_F16"] == "1"
        ckpt.mark.assert_called_once_with("whisper")

    def test_falls_back_to_cpu_and_clears_partial_srt(self, tmp_path, monkeypatch):
        out = tmp_path / "out"
        out.mkdir()
        srt = out / "whisper.srt"
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if len(calls) == 1:
                srt.write_text("partial", encoding="utf-8")
                raise subprocess.CalledProcessError(1, cmd)
            assert not srt.exists(), "partial Vulkan SRT should be removed before fallback"
            _write_srt(srt)
            return mock.Mock(returncode=0)

        run = mock.Mock(side_effect=fake_run)
        monkeypatch.setattr(pipeline.subprocess, "run", run)
        monkeypatch.setattr(pipeline, "WHISPER_BIN", "/opt/vk/whisper-cli")
        monkeypatch.setattr(pipeline, "WHISPER_LIB_DIR", "")
        monkeypatch.setattr(pipeline, "WHISPER_CPU_BIN", "/usr/bin/whisper-cli")
        monkeypatch.setattr(pipeline, "WHISPER_CPU_THREADS", 16)
        monkeypatch.setattr(pipeline, "WHISPER_OV_DEVICE", "CPU")
        monkeypatch.setattr(pipeline, "gpu_block_reason", lambda: None)
        ckpt = mock.Mock()
        ckpt.done.return_value = False

        result = pipeline.run_whisper_ckpt("audio.wav", str(out), "CODE", ckpt, mock.Mock())

        assert result == str(srt)
        assert len(calls) == 2
        cpu_cmd = calls[1]
        assert cpu_cmd[0] == "/usr/bin/whisper-cli"
        assert cpu_cmd[cpu_cmd.index("-oved") + 1] == "CPU"
        assert cpu_cmd[cpu_cmd.index("-t") + 1] == "16"
        ckpt.mark.assert_called_once_with("whisper")

    def test_empty_vulkan_output_falls_back_to_cpu(self, tmp_path, monkeypatch):
        out = tmp_path / "out"
        out.mkdir()
        srt = out / "whisper.srt"
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if len(calls) == 1:
                srt.write_text("", encoding="utf-8")
            else:
                _write_srt(srt)
            return mock.Mock(returncode=0)

        monkeypatch.setattr(pipeline.subprocess, "run", mock.Mock(side_effect=fake_run))
        monkeypatch.setattr(pipeline, "WHISPER_BIN", "/opt/vk/whisper-cli")
        monkeypatch.setattr(pipeline, "WHISPER_LIB_DIR", "")
        monkeypatch.setattr(pipeline, "gpu_block_reason", lambda: None)
        ckpt = mock.Mock()
        ckpt.done.return_value = False

        pipeline.run_whisper_ckpt("audio.wav", str(out), "CODE", ckpt, mock.Mock())

        assert len(calls) == 2
        ckpt.mark.assert_called_once_with("whisper")

    def test_gpu_gate_skips_vulkan_and_uses_cpu(self, tmp_path, monkeypatch):
        out = tmp_path / "out"
        out.mkdir()
        srt = out / "whisper.srt"
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            _write_srt(srt)
            return mock.Mock(returncode=0)

        monkeypatch.setattr(pipeline.subprocess, "run", mock.Mock(side_effect=fake_run))
        monkeypatch.setattr(pipeline, "WHISPER_CPU_BIN", "/usr/bin/whisper-cli")
        monkeypatch.setattr(pipeline, "WHISPER_CPU_THREADS", 16)
        monkeypatch.setattr(pipeline, "WHISPER_OV_DEVICE", "CPU")
        monkeypatch.setattr(pipeline, "gpu_block_reason", lambda: "TTS active (1)")
        ckpt = mock.Mock()
        ckpt.done.return_value = False

        result = pipeline.run_whisper_ckpt("audio.wav", str(out), "CODE", ckpt, mock.Mock())

        assert result == str(srt)
        assert len(calls) == 1
        assert calls[0][0] == "/usr/bin/whisper-cli"
        assert "-fa" not in calls[0]
        assert calls[0][calls[0].index("-t") + 1] == "16"
        ckpt.mark.assert_called_once_with("whisper")

    def test_checkpoint_skip_does_not_invoke_backend(self, tmp_path, monkeypatch):
        out = tmp_path / "out"
        out.mkdir()
        run = mock.Mock()
        monkeypatch.setattr(pipeline.subprocess, "run", run)
        ckpt = mock.Mock()
        ckpt.done.return_value = True

        result = pipeline.run_whisper_ckpt("audio.wav", str(out), "CODE", ckpt, mock.Mock())

        assert result == str(out / "whisper.srt")
        run.assert_not_called()
        ckpt.mark.assert_not_called()


class TestRunTranslateCkpt:
    def test_retries_in_process_on_failure(self, tmp_path, monkeypatch):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return mock.Mock(returncode=1 if len(calls) == 1 else 0)

        monkeypatch.setattr(pipeline.subprocess, "run", mock.Mock(side_effect=fake_run))
        ckpt = mock.Mock()
        ckpt.done.return_value = False

        result = pipeline.run_translate_ckpt(
            "in.srt",
            str(tmp_path),
            "CODE",
            ckpt,
            mock.Mock(),
            str(tmp_path / "job.log"),
            "en",
        )

        assert len(calls) == 2
        assert "--mode" not in calls[0]
        assert calls[1][-2:] == ["--mode", "direct"]
        ckpt.mark.assert_called_once_with("translate")
        assert result == str(tmp_path / "translated.srt")

    def test_success_does_not_retry(self, tmp_path, monkeypatch):
        run = mock.Mock(return_value=mock.Mock(returncode=0))
        monkeypatch.setattr(pipeline.subprocess, "run", run)
        ckpt = mock.Mock()
        ckpt.done.return_value = False

        pipeline.run_translate_ckpt(
            "in.srt",
            str(tmp_path),
            "CODE",
            ckpt,
            mock.Mock(),
            str(tmp_path / "job.log"),
            "en",
        )

        assert run.call_count == 1
