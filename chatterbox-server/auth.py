import logging
import os
import secrets
import smtplib
import string
import time
from email.mime.text import MIMEText

from config import SMTP_FROM, SMTP_HOST, SMTP_PASS, SMTP_PORT, SMTP_USER
from db_schema import ConnectionManager, init_users_schema
from singleton import singleton
from werkzeug.security import check_password_hash, generate_password_hash

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "users.db")

# Pre-computed dummy hash for timing-attack resistance on unknown-user lookups.
# Generated once so the format is always valid and modern.
_DUMMY_HASH = generate_password_hash("dummy-timing-guard")


@singleton
class UserManager:
    def __init__(self):
        self._conn = ConnectionManager(DB_FILE)
        self._init_db()

    def _get_conn(self):
        return self._conn.get()

    def _init_db(self):
        conn = self._get_conn()
        init_users_schema(conn)

    def register(self, email: str, password: str) -> dict:
        conn = self._get_conn()
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            return {"success": False, "error": "该邮箱已被注册"}

        password_hash = generate_password_hash(password)
        code = "".join(secrets.choice(string.digits) for _ in range(6))
        conn.execute(
            "INSERT INTO users (email, password_hash, verification_code) VALUES (?, ?, ?)",
            (email, password_hash, code),
        )
        conn.commit()

        sent = self._send_verification_email(email, code)
        result = {"success": True, "message": "注册成功，请查收验证邮件"}
        if not sent:
            result["message"] = "注册成功（邮件发送失败，请联系管理员）"
        return result

    def verify(self, email: str, code: str) -> dict:
        conn = self._get_conn()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if not user:
            return {"success": False, "error": "用户不存在"}
        if user["verified"]:
            return {"success": True, "message": "邮箱已验证"}
        if user["verification_code"] != code:
            return {"success": False, "error": "验证码错误"}
        conn.execute(
            "UPDATE users SET verified = 1, verification_code = NULL WHERE email = ?",
            (email,),
        )
        conn.commit()
        return {"success": True, "message": "邮箱验证成功"}

    def login(self, email: str, password: str) -> dict:
        conn = self._get_conn()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if not user:
            # Hash against a dummy to equalize timing with valid users
            check_password_hash(_DUMMY_HASH, password)
            return {"success": False, "error": "邮箱或密码错误"}
        if not check_password_hash(user["password_hash"], password):
            return {"success": False, "error": "邮箱或密码错误"}
        if not user["verified"]:
            result = {"success": False, "error": "邮箱未验证", "need_verify": True, "email": user["email"]}
            return result
        return {
            "success": True,
            "user": {"id": user["id"], "email": user["email"]},
        }

    def get_user_by_email(self, email: str):
        conn = self._get_conn()
        return conn.execute("SELECT id, email, verified FROM users WHERE email = ?", (email,)).fetchone()

    def change_password(self, email: str, old_password: str, new_password: str) -> dict:
        conn = self._get_conn()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if not user:
            return {"success": False, "error": "用户不存在"}
        if not check_password_hash(user["password_hash"], old_password):
            return {"success": False, "error": "原密码错误"}
        if len(new_password) < 6:
            return {"success": False, "error": "新密码至少6个字符"}
        new_hash = generate_password_hash(new_password)
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE email = ?",
            (new_hash, email),
        )
        conn.commit()
        logger.info(f"Password changed for {email}")
        return {"success": True, "message": "密码修改成功"}

    def resend_code(self, email: str) -> dict:
        conn = self._get_conn()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if not user:
            return {"success": False, "error": "用户不存在"}
        if user["verified"]:
            return {"success": True, "message": "邮箱已验证"}
        code = "".join(secrets.choice(string.digits) for _ in range(6))
        conn.execute(
            "UPDATE users SET verification_code = ? WHERE email = ?",
            (code, email),
        )
        conn.commit()
        sent = self._send_verification_email(email, code)
        result = {"success": True, "message": "验证码已重新发送"}
        if not sent:
            result["message"] = "邮件发送失败，请联系管理员"
        return result

    def request_reset(self, email: str) -> dict:
        """Generate a reset code and send it via email."""
        conn = self._get_conn()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if not user:
            # Don't reveal whether the email exists
            return {"success": True, "message": "如果该邮箱已注册，重置密码的邮件已发送"}
        code = "".join(secrets.choice(string.digits) for _ in range(6))
        expires = time.time() + 1800  # 30 minutes
        conn.execute(
            "UPDATE users SET reset_code = ?, reset_code_expires = ? WHERE email = ?",
            (code, expires, email),
        )
        conn.commit()
        sent = self._send_reset_email(email, code)
        if not sent:
            logger.warning(f"Failed to send reset email to {email}")
        return {"success": True, "message": "重置密码的邮件已发送，请查收"}

    def reset_password(self, email: str, code: str, new_password: str) -> dict:
        """Verify reset code and update password.

        Returns the same generic error for all failure modes to prevent
        email enumeration and timing oracle attacks.
        """
        conn = self._get_conn()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if not user:
            return {"success": False, "error": "链接无效或已过期"}
        if not user["reset_code"] or not user["reset_code_expires"]:
            return {"success": False, "error": "链接无效或已过期"}
        if time.time() > float(user["reset_code_expires"]):
            return {"success": False, "error": "链接无效或已过期"}
        if user["reset_code"] != code:
            return {"success": False, "error": "链接无效或已过期"}
        if len(new_password) < 6:
            return {"success": False, "error": "链接无效或已过期"}
        new_hash = generate_password_hash(new_password)
        conn.execute(
            "UPDATE users SET password_hash = ?, reset_code = NULL, reset_code_expires = NULL WHERE email = ?",
            (new_hash, email),
        )
        conn.commit()
        logger.info(f"Password reset for {email}")
        return {"success": True, "message": "密码重置成功"}

    def _send_email(self, email: str, subject: str, body: str) -> bool:
        """Send an email via SMTP. Returns True on success, False otherwise."""
        if not (SMTP_HOST and SMTP_USER):
            logger.warning(
                f"SMTP not configured (SMTP_HOST={SMTP_HOST!r}, SMTP_USER={SMTP_USER!r}), cannot send email to {email}"
            )
            return False
        try:
            msg = MIMEText(body, "plain", "utf-8")
            msg["Subject"] = subject
            msg["From"] = SMTP_FROM
            msg["To"] = email
            if SMTP_PORT == 465:
                with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
                    server.login(SMTP_USER, SMTP_PASS)
                    server.send_message(msg)
            else:
                with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                    server.starttls()
                    server.login(SMTP_USER, SMTP_PASS)
                    server.send_message(msg)
            return True
        except Exception as e:
            logger.warning(f"Failed to send email via SMTP: {e}")
            return False

    def _send_reset_email(self, email: str, code: str) -> bool:
        subject = "重置密码 - 宁师的视频翻译工具"
        body = f"""您好，

您请求了重置密码。

您的验证码是：{code}

验证码有效期为30分钟。如果这不是您本人的操作，请忽略此邮件。

宁师的视频翻译工具
"""
        sent = self._send_email(email, subject, body)
        if sent:
            logger.info(f"Reset email sent to {email}")
        else:
            logger.warning(f"Failed to send reset email to {email}")
        return sent

    def _send_verification_email(self, email: str, code: str) -> bool:
        subject = "验证您的邮箱 - 宁师的视频翻译工具"
        body = f"""您好，

感谢您注册宁师的视频翻译工具。

您的验证码是：{code}

请在验证页面输入此验证码完成邮箱验证。

如果这不是您本人的操作，请忽略此邮件。

宁师的视频翻译工具
"""
        sent = self._send_email(email, subject, body)
        if sent:
            logger.info(f"Verification email sent to {email}")
        else:
            logger.warning(f"Failed to send verification email to {email}")
        return sent


def get_user_manager() -> UserManager:
    return UserManager()
