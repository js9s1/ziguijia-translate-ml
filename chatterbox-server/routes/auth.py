"""Auth routes — registration, login, password reset, CSRF token."""

import logging

from flask import Blueprint, jsonify, request, send_from_directory, session
from middleware import email_rate_limit, login_required, rate_limit

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
    data = request.get_json()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    if not email or not password:
        return jsonify({"success": False, "error": "请填写邮箱和密码"})
    result = _get_user_manager().register(email, password)
    return jsonify(result)


@auth_bp.route("/auth/verify", methods=["POST"])
@rate_limit
def auth_verify():
    data = request.get_json()
    email = data.get("email", "").strip().lower()
    code = data.get("code", "").strip()
    result = _get_user_manager().verify(email, code)
    return jsonify(result)


@auth_bp.route("/auth/login", methods=["GET", "POST"])
@rate_limit
def auth_login():
    if request.method == "GET":
        return send_from_directory(HTML_DIR, "login.html")
    data = request.get_json()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
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
    data = request.get_json()
    old_password = data.get("old_password", "")
    new_password = data.get("new_password", "")
    email = session.get("user_email", "")
    result = _get_user_manager().change_password(email, old_password, new_password)
    return jsonify(result)


@auth_bp.route("/auth/resend", methods=["POST"])
@rate_limit
def auth_resend():
    data = request.get_json()
    email = data.get("email", "").strip().lower()
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
    data = request.get_json()
    email = data.get("email", "").strip().lower()
    if not email:
        return jsonify({"success": False, "error": "请输入邮箱"})

    allowed, error_msg = email_rate_limit(email)
    if not allowed:
        return jsonify({"success": False, "error": error_msg})

    result = _get_user_manager().request_reset(email)
    return jsonify(result)


@auth_bp.route("/auth/reset-password/confirm", methods=["POST"])
@rate_limit
def auth_reset_password_confirm():
    data = request.get_json()
    email = data.get("email", "").strip().lower()
    code = data.get("code", "").strip()
    new_password = data.get("new_password", "")
    if not email or not code or not new_password:
        return jsonify({"success": False, "error": "请填写所有字段"})
    result = _get_user_manager().reset_password(email, code, new_password)
    return jsonify(result)
