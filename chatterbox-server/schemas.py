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


# ── Job schemas ─────────────────────────────────────────────

MAX_TEXT_LENGTH = 500


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    filename: str = "output.wav"
    temperature: float = Field(0.6, ge=0.1, le=2.0)
    target_language: str = Field("en", min_length=2, max_length=5)
    cfg_weight: float = Field(0.5, ge=0.0, le=2.0)
    exaggeration: float = Field(0.5, ge=0.0, le=2.0)


class AudioTextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    filename: str = "output.wav"
    temperature: float = Field(0.6, ge=0.1, le=2.0)
    target_language: str = Field("en", min_length=2, max_length=5)
    cfg_weight: float = Field(0.5, ge=0.0, le=2.0)
    exaggeration: float = Field(0.5, ge=0.0, le=2.0)


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


# ── Job management schemas ──────────────────────────────────

class JobParams(BaseModel):
    temperature: float = Field(0.6, ge=0.1, le=2.0)
    target_language: str = Field("en", min_length=2, max_length=5)
    cfg_weight: float = Field(0.5, ge=0.0, le=2.0)
    exaggeration: float = Field(0.5, ge=0.0, le=2.0)
