"""Tests for the Vulkan Whisper preflight gate.

These are pure logic tests: GPU temperature and daemon pings are mocked.
"""

import os
import socket
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "chatterbox-server"))

import whisper_gate  # noqa: E402


def _set_limits(monkeypatch, *, temp=76.0, tts=1, ocr=1, translate=1):
    monkeypatch.setattr(whisper_gate, "WHISPER_GPU_MAX_TEMP", temp)
    monkeypatch.setattr(whisper_gate, "WHISPER_GATE_TTS_ACTIVE", tts)
    monkeypatch.setattr(whisper_gate, "WHISPER_GATE_OCR_ACTIVE", ocr)
    monkeypatch.setattr(whisper_gate, "WHISPER_GATE_TRANSLATE_ACTIVE", translate)


def _status_for(active_sock, active=1):
    def fake_status(path):
        if path == active_sock:
            return {"ok": True, "active": active}
        return None

    return fake_status


class TestDaemonStatus:
    def test_reads_ping_response(self, tmp_path):
        sock_path = str(tmp_path / "daemon.sock")
        ready = threading.Event()

        def serve():
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(sock_path)
            server.listen(1)
            ready.set()
            conn, _ = server.accept()
            with conn:
                conn.recv(4096)
                conn.sendall(b'{"ok": true, "active": 2}\n')
            server.close()

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        assert ready.wait(2)

        try:
            assert whisper_gate._daemon_status(sock_path) == {"ok": True, "active": 2}
        finally:
            thread.join(timeout=2)

    def test_missing_socket_returns_none(self, tmp_path):
        assert whisper_gate._daemon_status(str(tmp_path / "missing.sock")) is None


class TestGpuBlockReason:
    def test_hot_gpu_blocks(self, monkeypatch):
        _set_limits(monkeypatch)
        monkeypatch.setattr(whisper_gate, "get_gpu_temp", lambda: 80.0)
        monkeypatch.setattr(whisper_gate, "_daemon_status", lambda path: None)

        reason = whisper_gate.gpu_block_reason()

        assert reason is not None
        assert "GPU 80" in reason

    def test_cool_gpu_with_idle_daemons_allows(self, monkeypatch):
        _set_limits(monkeypatch)
        monkeypatch.setattr(whisper_gate, "get_gpu_temp", lambda: 60.0)
        monkeypatch.setattr(whisper_gate, "_daemon_status", lambda path: None)

        assert whisper_gate.gpu_block_reason() is None

    def test_unknown_temperature_does_not_block(self, monkeypatch):
        _set_limits(monkeypatch)
        monkeypatch.setattr(whisper_gate, "get_gpu_temp", lambda: None)
        monkeypatch.setattr(whisper_gate, "_daemon_status", lambda path: None)

        assert whisper_gate.gpu_block_reason() is None

    def test_tts_active_blocks(self, monkeypatch):
        _set_limits(monkeypatch)
        monkeypatch.setattr(whisper_gate, "get_gpu_temp", lambda: 60.0)
        monkeypatch.setattr(
            whisper_gate,
            "_daemon_status",
            _status_for(whisper_gate.GEN_AUDIO_DAEMON_SOCK),
        )

        assert whisper_gate.gpu_block_reason() == "TTS active (1)"

    def test_ocr_active_blocks(self, monkeypatch):
        _set_limits(monkeypatch)
        monkeypatch.setattr(whisper_gate, "get_gpu_temp", lambda: 60.0)
        monkeypatch.setattr(
            whisper_gate,
            "_daemon_status",
            _status_for(whisper_gate.RAPID_VIDEOCR_DAEMON_SOCK),
        )

        assert whisper_gate.gpu_block_reason() == "OCR active (1)"

    def test_translate_active_blocks(self, monkeypatch):
        _set_limits(monkeypatch)
        monkeypatch.setattr(whisper_gate, "get_gpu_temp", lambda: 60.0)
        monkeypatch.setattr(
            whisper_gate,
            "_daemon_status",
            _status_for(whisper_gate.TRANSLATE_DAEMON_SOCK),
        )

        assert whisper_gate.gpu_block_reason() == "translate active (1)"

    def test_active_below_threshold_allows(self, monkeypatch):
        _set_limits(monkeypatch, tts=2)
        monkeypatch.setattr(whisper_gate, "get_gpu_temp", lambda: 60.0)
        monkeypatch.setattr(
            whisper_gate,
            "_daemon_status",
            _status_for(whisper_gate.GEN_AUDIO_DAEMON_SOCK, active=1),
        )

        assert whisper_gate.gpu_block_reason() is None

    def test_zero_threshold_disables_check(self, monkeypatch):
        _set_limits(monkeypatch, tts=0)
        monkeypatch.setattr(whisper_gate, "get_gpu_temp", lambda: 60.0)
        monkeypatch.setattr(
            whisper_gate,
            "_daemon_status",
            _status_for(whisper_gate.GEN_AUDIO_DAEMON_SOCK, active=5),
        )

        assert whisper_gate.gpu_block_reason() is None

    def test_missing_active_field_treated_as_idle(self, monkeypatch):
        _set_limits(monkeypatch)
        monkeypatch.setattr(whisper_gate, "get_gpu_temp", lambda: 60.0)
        monkeypatch.setattr(
            whisper_gate,
            "_daemon_status",
            lambda path: {"ok": True},
        )

        assert whisper_gate.gpu_block_reason() is None
