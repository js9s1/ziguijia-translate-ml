"""Tests for Pydantic request validation schemas."""

import pytest
from pydantic import ValidationError

from schemas import (
    RegisterRequest, LoginRequest, VerifyRequest,
    ChangePasswordRequest, ResetPasswordRequest,
    ResetPasswordConfirmRequest, TTSRequest,
    FileDeleteRequest, SRTSaveRequest, JobParams, MAX_TEXT_LENGTH,
)


class TestRegisterRequest:
    def test_valid(self):
        r = RegisterRequest(email="a@b.com", password="secret123")
        assert r.email == "a@b.com"
        assert r.password == "secret123"

    def test_empty_email(self):
        with pytest.raises(ValidationError):
            RegisterRequest(email="", password="x")

    def test_empty_password(self):
        with pytest.raises(ValidationError):
            RegisterRequest(email="a@b.com", password="")


class TestTTSRequest:
    def test_valid(self):
        r = TTSRequest(text="hello", temperature=0.6, target_language="en")
        assert r.text == "hello"
        assert r.temperature == 0.6

    def test_text_too_long(self):
        with pytest.raises(ValidationError):
            TTSRequest(text="x" * (MAX_TEXT_LENGTH + 1))

    def test_temperature_bounds(self):
        with pytest.raises(ValidationError):
            TTSRequest(text="hi", temperature=3.0)

    def test_defaults(self):
        r = TTSRequest(text="hi")
        assert r.temperature == 0.6
        assert r.target_language == "en"
        assert r.cfg_weight == 0.5


class TestFileDeleteRequest:
    def test_valid(self):
        r = FileDeleteRequest(path="/tmp/foo.txt")
        assert r.path == "/tmp/foo.txt"
        assert r.access_code is None

    def test_requires_path(self):
        with pytest.raises(ValidationError):
            FileDeleteRequest(path="")


class TestSRTSaveRequest:
    def test_valid(self):
        r = SRTSaveRequest(path="/tmp/file.srt", content="[SRT]", access_code="ABC")
        assert r.path == "/tmp/file.srt"

    def test_empty_content(self):
        with pytest.raises(ValidationError):
            SRTSaveRequest(path="/tmp/x.srt", content="", access_code="A")


class TestJobParams:
    def test_valid(self):
        r = JobParams(temperature=0.8, target_language="zh")
        assert r.temperature == 0.8
        assert r.target_language == "zh"

    def test_defaults(self):
        r = JobParams()
        assert r.temperature == 0.6
        assert r.target_language == "en"
