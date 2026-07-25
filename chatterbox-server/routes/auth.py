"""Auth routes — registration, login, password reset, CSRF token."""

import logging

from flask import Blueprint, jsonify, request, send_from_directory, session
from middleware import email_rate_limit, login_required, rate_limit
from pydantic import ValidationError
from schemas import (
    ChangePasswordRequest,
    LoginRequest,
    RegisterRequest,
    ResendCodeRequest,
    ResetPasswordConfirmRequest,
    ResetPasswordRequest,
    VerifyRequest,
)

auth_bp = Blueprint("auth", __name__)
logger = logging.getLogger(__name__)

HTML_DIR = __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "html")


def _get_user_manager():
    from auth import get_user_manager
    return get_user_manager()


@auth_bp.route("/auth/register", methods=["GET", "POST"])
@rate_limit
def auth_register():
    if request.method == "GET":
        return send_from_directory(HTML_DIR, "register.html")
    try:
        req = RegisterRequest.model_validate(request.get_json(silent=True) or {})
    except ValidationError as e:
        return jsonify({"success": False, "error": f"请填写邮箱和密码 ({e.errors()[0]['msg']})"})
    email = req.email.strip().lower()
    password = req.password
    if not email or not password:
        return jsonify({"success": False, "error": "请填写邮箱和密码"})
    result = _get_user_manager().register(email, password)
    return jsonify(result)


@auth_bp.route("/auth/verify", methods=["POST"])
@rate_limit
def auth_verify():
    data = request.get_json(silent=True) or {}
    email = (data.get("email", "") or "").strip().lower()
    code = (data.get("code", "") or "").strip()
    result = _get_user_manager().verify(email, code)
    return jsonify(result)


@auth_bp.route("/auth/login", methods=["GET", "POST"])
@rate_limit
def auth_login():
    if request.method == "GET":
        return send_from_directory(HTML_DIR, "login.html")
    try:
        req = LoginRequest.model_validate(request.get_json(silent=True) or {})
    except ValidationError as e:
        return jsonify({"success": False, "error": f"请填写邮箱和密码 ({e.errors()[0]['msg']})"})
    email = req.email.strip().lower()
    password = req.password
    result = _get_user_manager().login(email, password)
    if result["success"]:
        session["user_id"] = result["user"]["id"]
        session["user_email"] = result["user"]["email"]
    return jsonify(result)


@auth_bp.route("/auth/logout", methods=["POST"])
def auth_logout():
    session.clear()
    if hasattr(session, "regenerate"):
        session.regenerate()
    return jsonify({"success": True})


@auth_bp.route("/auth/change-password", methods=["GET", "POST"])
@login_required
def auth_change_password():
    if request.method == "GET":
        return send_from_directory(HTML_DIR, "change-password.html")
    try:
        req = ChangePasswordRequest.model_validate(request.get_json(silent=True) or {})
    except ValidationError as e:
        return jsonify({"success": False, "error": f"请填写密码 ({e.errors()[0]['msg']})"})
    email = session.get("user_email", "")
    result = _get_user_manager().change_password(email, req.old_password, req.new_password)
    return jsonify(result)


@auth_bp.route("/auth/resend", methods=["POST"])
@rate_limit
def auth_resend():
    data = request.get_json(silent=True) or {}
    email = (data.get("email", "") or "").strip().lower()
    result = _get_user_manager().resend_code(email)
    return jsonify(result)


@auth_bp.route("/auth/me", methods=["GET"])
def auth_me():
    if "user_id" not in session:
        return jsonify({"authenticated": False})
    return jsonify({
        "authenticated": True,
        "user": {"id": session["user_id"], "email": session["user_email"]},
    })


@auth_bp.route("/auth/csrf-token", methods=["GET"])
def auth_csrf_token():
    """Return a CSRF token for the current session."""
    from middleware import generate_csrf_token
    return jsonify({"csrf_token": generate_csrf_token()})


@auth_bp.route("/auth/reset-password", methods=["GET", "POST"])
@rate_limit
def auth_reset_password():
    if request.method == "GET":
        return send_from_directory(HTML_DIR, "reset-password.html")
    try:
        req = ResetPasswordRequest.model_validate(request.get_json(silent=True) or {})
    except ValidationError as e:
        return jsonify({"success": False, "error": f"请输入邮箱 ({e.errors()[0]['msg']})"})
    email = req.email.strip().lower()
    allowed, error_msg = email_rate_limit(email)
    if not allowed:
        return jsonify({"success": False, "error": error_msg})

    result = _get_user_manager().request_reset(email)
    return jsonify(result)


@auth_bp.route("/auth/reset-password/confirm", methods=["POST"])
@rate_limit
def auth_reset_password_confirm():
    try:
        req = ResetPasswordConfirmRequest.model_validate(request.get_json(silent=True) or {})
    except ValidationError as e:
        return jsonify({"success": False, "error": f"请填写所有字段 ({e.errors()[0]['msg']})"})
    email = req.email.strip().lower()
    code = req.code.strip()
    new_password = req.new_password
    result = _get_user_manager().reset_password(email, code, new_password)
    return jsonify(result)
