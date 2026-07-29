"""Pydantic request validation schemas for API endpoints."""

from pydantic import BaseModel, Field

# ── Auth schemas ────────────────────────────────────────────


class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class VerifyRequest(BaseModel):
    email: str = Field(..., min_length=1)
    code: str = Field(..., min_length=1)


class ChangePasswordRequest(BaseModel):
    old_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=6)


class ResetPasswordRequest(BaseModel):
    email: str = Field(..., min_length=1)


class ResetPasswordConfirmRequest(BaseModel):
    email: str = Field(..., min_length=1)
    code: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=6)


class ResendCodeRequest(BaseModel):
    email: str = Field(..., min_length=1)


# ── File schemas ────────────────────────────────────────────


class FileDeleteRequest(BaseModel):
    path: str = Field(..., min_length=1)
    access_code: str | None = None


class SRTSaveRequest(BaseModel):
    path: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    access_code: str = Field(..., min_length=1)


class SRTOldrunDownloadRequest(BaseModel):
    files: list[dict] = Field(..., min_length=1)
