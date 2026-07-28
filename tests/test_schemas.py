"""Tests for Pydantic request validation schemas."""

import pytest
from pydantic import ValidationError

from schemas import (
    RegisterRequest, LoginRequest, VerifyRequest,
    ChangePasswordRequest, ResetPasswordRequest,
    ResetPasswordConfirmRequest,
    FileDeleteRequest, SRTSaveRequest,
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
