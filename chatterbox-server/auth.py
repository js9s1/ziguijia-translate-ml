import logging
import os
import random
import smtplib
import sqlite3
import string
import threading
from email.mime.text import MIMEText

from werkzeug.security import check_password_hash, generate_password_hash

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "users.db")

SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
SMTP_FROM = os.environ.get("SMTP_FROM", SMTP_USER)


class UserManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._local = threading.local()
        self._init_db()

    def _get_conn(self):
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(DB_FILE)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                verified INTEGER DEFAULT 0,
                verification_code TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

    def register(self, email: str, password: str) -> dict:
        conn = self._get_conn()
        existing = conn.execute(
            "SELECT id FROM users WHERE email = ?", (email,)
        ).fetchone()
        if existing:
            return {"success": False, "error": "该邮箱已被注册"}

        password_hash = generate_password_hash(password)
        code = "".join(random.choices(string.digits, k=6))
        conn.execute(
            "INSERT INTO users (email, password_hash, verification_code) VALUES (?, ?, ?)",
            (email, password_hash, code),
        )
        conn.commit()

        sent = self._send_verification_email(email, code)
        result = {"success": True, "message": "注册成功，请查收验证邮件"}
        if not sent:
            result["message"] = "注册成功（邮件发送失败）"
            result["code"] = code
        return result

    def verify(self, email: str, code: str) -> dict:
        conn = self._get_conn()
        user = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()
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
        user = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()
        if not user:
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
        return conn.execute(
            "SELECT id, email, verified FROM users WHERE email = ?", (email,)
        ).fetchone()

    def resend_code(self, email: str) -> dict:
        conn = self._get_conn()
        user = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()
        if not user:
            return {"success": False, "error": "用户不存在"}
        if user["verified"]:
            return {"success": True, "message": "邮箱已验证"}
        code = "".join(random.choices(string.digits, k=6))
        conn.execute(
            "UPDATE users SET verification_code = ? WHERE email = ?",
            (code, email),
        )
        conn.commit()
        sent = self._send_verification_email(email, code)
        result = {"success": True, "message": "验证码已重新发送"}
        if not sent:
            result["message"] = "邮件发送失败"
            result["code"] = code
        return result

    def _send_verification_email(self, email: str, code: str) -> bool:
        subject = "验证您的邮箱 - 宁师的视频翻译工具"
        body = f"""您好，

感谢您注册宁师的视频翻译工具。

您的验证码是：{code}

请在验证页面输入此验证码完成邮箱验证。

如果这不是您本人的操作，请忽略此邮件。

宁师的视频翻译工具
"""
        if SMTP_HOST and SMTP_USER:
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
                logger.info(f"Verification email sent to {email}")
                return True
            except Exception as e:
                logger.warning(f"Failed to send email via SMTP: {e}")
        logger.info(f"Verification code for {email}: {code}")
        return False


def get_user_manager() -> UserManager:
    return UserManager()
