"""Tests for auth routes: registration, login, verification, password reset."""

import pytest


@pytest.fixture(autouse=True)
def _bypass_rate_limit():
    """Disable rate limiting for auth tests — many tests share the same IP."""
    from middleware import _email_limiter, _ip_limiter

    old_ip = _ip_limiter.limit
    old_email = _email_limiter.limit
    _ip_limiter.limit = 10000
    _email_limiter.limit = 10000
    yield
    _ip_limiter.limit = old_ip
    _email_limiter.limit = old_email


class TestAuthMe:
    def test_unauthenticated(self, client):
        resp = client.get("/auth/me")
        data = resp.get_json()
        assert data["authenticated"] is False

    def test_authenticated(self, auth_client):
        client, _ = auth_client
        resp = client.get("/auth/me")
        data = resp.get_json()
        assert data["authenticated"] is True
        assert data["user"]["email"] == "test@example.com"


class TestCSRFToken:
    def test_returns_token(self, client):
        resp = client.get("/auth/csrf-token")
        data = resp.get_json()
        assert "csrf_token" in data
        assert len(data["csrf_token"]) > 0


class TestRegistration:
    def test_register_success(self, client):
        resp = client.post(
            "/auth/register",
            json={
                "email": "newuser@example.com",
                "password": "password123",
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True

        # Verify user exists in DB
        from auth import get_user_manager

        user = get_user_manager().get_user_by_email("newuser@example.com")
        assert user is not None
        assert user["email"] == "newuser@example.com"

    def test_register_missing_fields(self, client):
        resp = client.post("/auth/register", json={"email": "x@y.com"})
        data = resp.get_json()
        assert data["success"] is False

    def test_register_duplicate(self, client):
        client.post(
            "/auth/register",
            json={
                "email": "dup@example.com",
                "password": "pass123456",
            },
        )
        resp = client.post(
            "/auth/register",
            json={
                "email": "dup@example.com",
                "password": "another123",
            },
        )
        data = resp.get_json()
        assert data["success"] is False


class TestVerification:
    def test_verify_invalid_email(self, client):
        resp = client.post(
            "/auth/verify",
            json={
                "email": "no@user.com",
                "code": "123456",
            },
        )
        data = resp.get_json()
        assert data["success"] is False

    def test_verify_wrong_code(self, client):
        client.post(
            "/auth/register",
            json={
                "email": "v@test.com",
                "password": "password123",
            },
        )
        resp = client.post(
            "/auth/verify",
            json={
                "email": "v@test.com",
                "code": "000000",
            },
        )
        data = resp.get_json()
        assert data["success"] is False


class TestLogin:
    def test_login_unverified(self, client):
        client.post(
            "/auth/register",
            json={
                "email": "unverified@test.com",
                "password": "pass123456",
            },
        )
        resp = client.post(
            "/auth/login",
            json={
                "email": "unverified@test.com",
                "password": "pass123456",
            },
        )
        data = resp.get_json()
        assert data["success"] is False
        assert "need_verify" in data

    def test_login_wrong_password(self, client):
        client.post(
            "/auth/register",
            json={
                "email": "wrong@test.com",
                "password": "correct123",
            },
        )
        resp = client.post(
            "/auth/login",
            json={
                "email": "wrong@test.com",
                "password": "wrong456",
            },
        )
        data = resp.get_json()
        assert data["success"] is False

    def test_login_nonexistent_user(self, client):
        resp = client.post(
            "/auth/login",
            json={
                "email": "ghost@test.com",
                "password": "whatever",
            },
        )
        data = resp.get_json()
        assert data["success"] is False

    def test_login_success(self, client):
        from auth import get_user_manager

        email = "ok2@test.com"
        client.post(
            "/auth/register",
            json={
                "email": email,
                "password": "pass123456",
            },
        )
        um = get_user_manager()
        conn = um._get_conn()
        conn.execute(
            "UPDATE users SET verified = 1, verification_code = NULL WHERE email = ?",
            (email,),
        )
        conn.commit()

        resp = client.post(
            "/auth/login",
            json={
                "email": email,
                "password": "pass123456",
            },
        )
        data = resp.get_json()
        assert data["success"] is True
        assert data["user"]["email"] == email


class TestLogout:
    def test_logout(self, auth_client):
        client, _ = auth_client
        resp = client.post("/auth/logout")
        assert resp.status_code == 200
        # After logout, should be unauthenticated
        resp = client.get("/auth/me")
        assert resp.get_json()["authenticated"] is False


class TestChangePassword:
    def test_unauthenticated(self, client):
        resp = client.post(
            "/auth/change-password",
            json={
                "old_password": "old",
                "new_password": "new123456",
            },
        )
        assert resp.status_code == 401

    def test_wrong_old_password(self, auth_client):
        client, _ = auth_client
        resp = client.post(
            "/auth/change-password",
            json={
                "old_password": "wrong",
                "new_password": "new123456",
            },
        )
        data = resp.get_json()
        assert data["success"] is False

    def test_success(self, auth_client):
        client, user = auth_client
        resp = client.post(
            "/auth/change-password",
            json={
                "old_password": user["password"],
                "new_password": "brandnew789",
            },
        )
        data = resp.get_json()
        assert data["success"] is True


class TestResetPasswordFlow:
    def test_request_reset(self, client):
        client.post(
            "/auth/register",
            json={
                "email": "reset@test.com",
                "password": "pass123456",
            },
        )
        resp = client.post("/auth/reset-password", json={"email": "reset@test.com"})
        data = resp.get_json()
        assert data["success"] is True

    def test_confirm_nonexistent(self, client):
        resp = client.post(
            "/auth/reset-password/confirm",
            json={
                "email": "nope@test.com",
                "code": "123456",
                "new_password": "new123456",
            },
        )
        data = resp.get_json()
        assert data["success"] is False


class TestRouteMethods:
    """Test that GET pages return HTML and POST returns JSON."""

    def test_register_page(self, client):
        resp = client.get("/auth/register")
        assert resp.status_code == 200
        assert b"<html" in resp.data.lower() or b"<!doctype" in resp.data.lower()

    def test_login_page(self, client):
        resp = client.get("/auth/login")
        assert resp.status_code == 200
