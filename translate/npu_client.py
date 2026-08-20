"""Client for the NPU translation engine (npu_engine.py).

Used by the translate daemon's NPU worker slot.  The engine runs as the
systemd system service npu-engine.service (started on demand via sudo —
NOPASSWD sudoers entry — because the engine needs LimitMEMLOCK=infinity).

Fallbacks to a local subprocess only if the engine socket exists but the
service isn't running and cannot be started; the caller decides whether
to fall back to the GPU worker instead.
"""
import json
import os
import socket
import subprocess
import time

DEFAULT_SOCK = os.path.join(
    os.environ.get("XDG_RUNTIME_DIR", os.path.expanduser("~/.run")),
    "npu_translate_engine.sock",
)


class NpuEngineUnavailable(RuntimeError):
    pass


class NpuEngineClient:
    def __init__(self, sock_path: str | None = None, auto_start: bool = True):
        self._sock = sock_path or os.environ.get("TRANSLATE_NPU_SOCK", DEFAULT_SOCK)
        self._auto_start = auto_start
        self._warned = False

    def _request(self, payload: dict, timeout: float = 600.0) -> dict:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.settimeout(timeout)
            s.connect(self._sock)
            s.sendall((json.dumps(payload) + "\n").encode())
            buf = b""
            while True:
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
                if b"\n" in buf:
                    break
            if not buf:
                raise NpuEngineUnavailable("engine closed connection without response")
            return json.loads(buf.split(b"\n", 1)[0].decode())
        except (FileNotFoundError, ConnectionRefusedError, socket.timeout, OSError) as e:
            raise NpuEngineUnavailable(f"engine unreachable: {e}") from None
        finally:
            try:
                s.close()
            except OSError:
                pass

    def ping(self, timeout: float = 3.0) -> dict | None:
        try:
            return self._request({"cmd": "ping"}, timeout=timeout)
        except NpuEngineUnavailable:
            return None

    def start_engine(self, wait: float = 600.0) -> bool:
        """Start npu-engine.service via sudo (NOPASSWD) and wait until it pings."""
        try:
            r = subprocess.run(
                ["sudo", "-n", "systemctl", "start", "npu-engine.service"],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode != 0:
                raise NpuEngineUnavailable(
                    f"sudo systemctl start failed: {r.stderr.strip()[:200]}"
                )
        except (subprocess.SubprocessError, OSError) as e:
            raise NpuEngineUnavailable(f"cannot start engine service: {e}") from None

        deadline = time.time() + wait
        while time.time() < deadline:
            if self.ping() is not None:
                return True
            time.sleep(2)
        raise NpuEngineUnavailable("engine service started but not answering ping")

    def ensure_engine(self, wait: float = 600.0) -> bool:
        if self.ping() is not None:
            return True
        if not self._auto_start:
            return False
        try:
            return self.start_engine(wait)
        except NpuEngineUnavailable as e:
            if not self._warned:
                print(f"[daemon] NPU engine unavailable ({e}) — falling back to GPU")
                self._warned = True
            return False

    def ensure_model(self, timeout: float = 3600.0) -> dict:
        if not self.ensure_engine():
            raise NpuEngineUnavailable("engine not reachable")
        return self._request({"cmd": "ensure_model"}, timeout=timeout)

    def translate(self, text: str, language: str, timeout: float = 3600.0) -> str:
        resp = self._request(
            {"cmd": "translate", "text": text, "language": language},
            timeout=timeout,
        )
        if not resp.get("ok"):
            raise NpuEngineUnavailable(f"engine translate failed: {resp.get('error')}")
        return str(resp.get("text", ""))
