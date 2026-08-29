#!/usr/bin/env python3
"""
Standalone SRT translation script for use as a ROCm GPU subprocess.

Translation runs through the warm translate daemon when available
(--mode auto starts it on demand); falls back to in-process HY-MT.

Usage:
    ./translate_srt.py input.srt output.srt -l English
    ./translate_srt.py input.srt output.srt -l English --intro "杨宁随缘开示" --outro "子归家全体编制人员"
"""

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.expanduser("~/src/HY-MT"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chatterbox-server"))

from rocm_env import setup as _rocm_setup  # noqa: E402

_rocm_setup()  # before hy_mt/torch load ROCm libs

import hy_mt  # noqa: E402

_STOP_REQUESTED = False


def _signal_handler(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


try:
    # Only valid in the main thread — imported as a module (e.g. by the
    # translate daemon worker) this raises, which is fine.
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
except ValueError:
    pass


def read_srt_text(path):
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8-sig", "utf-16-le", "utf-16-be", "gbk", "gb2312", "gb18030", "utf-8"):
        try:
            text = raw.decode(enc)
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            return _normalize_srt_timestamps(text)
        except (UnicodeDecodeError, LookupError):
            continue
    text = raw.decode("utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _normalize_srt_timestamps(text)


def _normalize_srt_timestamps(text):
    text = re.sub(r"(\d{1,2}:\d{1,2}:\d{1,2})[.:](\d{3})", r"\1,\2", text)
    text = re.sub(
        r"(\d{1,2}:\d{1,2}:\d{1,2},\d{3})\s*->\s*(\d{1,2}:\d{1,2}:\d{1,2},\d{3})",
        r"\1 --> \2",
        text,
    )
    return text


def looks_untranslated(text, source_has_cjk=True):
    if not source_has_cjk:
        return False
    cjk_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return cjk_count >= 3


def _prompt_zh(model, tokenizer, text, target_language):
    messages = [
        {
            "role": "user",
            "content": f"将以下文本翻译为{target_language}，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}",
        },
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_tensors="pt"
    ).to(model.device)
    outputs = model.generate(**inputs, **hy_mt.GENERATION_KWARGS)
    return tokenizer.decode(outputs[0][len(inputs.input_ids[0]) :], skip_special_tokens=True)


def _prompt_en(model, tokenizer, text, target_language):
    messages = [
        {
            "role": "user",
            "content": f"Translate the following segment into {target_language}, without additional explanation.\n\n{text}",
        },
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_tensors="pt"
    ).to(model.device)
    outputs = model.generate(**inputs, **hy_mt.GENERATION_KWARGS)
    return tokenizer.decode(outputs[0][len(inputs.input_ids[0]) :], skip_special_tokens=True)


def _prompt_raw(model, tokenizer, text, target_language):
    messages = [
        {
            "role": "user",
            "content": (
                f"Translate the following Chinese sentence into {target_language}. "
                f"Output ONLY the {target_language} translation, nothing else:\n\n{text}"
            ),
        },
    ]
    tokenized_chat = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_tensors="pt"
    )
    outputs = model.generate(**tokenized_chat.to(model.device), **hy_mt.GENERATION_KWARGS)
    return tokenizer.decode(outputs[0][len(tokenized_chat[0]) :], skip_special_tokens=True)


def _translate_with_retry(model, tokenizer, text, target_language, source_has_cjk=True):
    """Three attempts with escalating prompts, mirroring the legacy path."""
    result = text
    for attempt in range(3):
        if attempt == 0:
            result = _prompt_zh(model, tokenizer, text, target_language)
        elif attempt == 1:
            result = _prompt_en(model, tokenizer, text, target_language)
        else:
            result = _prompt_raw(model, tokenizer, text, target_language)
        if not looks_untranslated(result, source_has_cjk):
            return result
    return result


def translate_segment(text, target_language, source_has_cjk=True):
    model, tokenizer = hy_mt._get_model()
    return _translate_with_retry(model, tokenizer, text, target_language, source_has_cjk)


# ── Backends ─────────────────────────────────────────────────
# Backend objects expose: translate(text, target_language) -> str


class _DaemonUnavailable(RuntimeError):
    """Raised when the translate daemon cannot be reached."""


class _DaemonTranslateClient:
    def __init__(self, sock_path: str, auto_start: bool = False):
        self._sock = sock_path
        self._auto_start = auto_start

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
                raise _DaemonUnavailable("daemon closed connection without response")
            return json.loads(buf.split(b"\n", 1)[0].decode())
        except (FileNotFoundError, ConnectionRefusedError, socket.timeout, OSError) as e:
            raise _DaemonUnavailable(f"daemon unreachable: {e}") from None
        finally:
            try:
                s.close()
            except OSError:
                pass

    def ping(self, timeout: float = 3.0) -> dict | None:
        try:
            return self._request({"cmd": "ping"}, timeout=timeout)
        except _DaemonUnavailable:
            return None

    def ensure_daemon(self, start_wait: float = 120.0) -> bool:
        """Attach to a running daemon or launch one. Returns True when ready."""
        if self.ping() is not None:
            return True

        from config import TRANSLATE_DAEMON_SCRIPT, TRANSLATE_PYTHON

        log_path = os.path.join(os.path.expanduser("~"), "logs", "translate_daemon.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as logf:
            try:
                subprocess.Popen(
                    [TRANSLATE_PYTHON, "-u", TRANSLATE_DAEMON_SCRIPT],
                    stdout=logf,
                    stderr=logf,
                    start_new_session=True,
                    stdin=subprocess.DEVNULL,
                )
            except Exception as e:
                print(f"  ⚠ failed to start translate daemon: {e}")
                return False

        deadline = time.time() + start_wait
        while time.time() < deadline:
            if self.ping() is not None:
                return True
            time.sleep(1)
        return False

    def translate_file(self, input_path, output_path, target_language, intro_marker=None, outro_marker=None):
        deadline = time.time() + 6 * 3600  # 6 h safety
        while True:
            if _STOP_REQUESTED:
                raise SystemExit(1)
            try:
                resp = self._request(
                    {
                        "cmd": "translate",
                        "input_path": input_path,
                        "output_path": output_path,
                        "language": target_language,
                        "intro": intro_marker,
                        "outro": outro_marker,
                    },
                    timeout=None,
                )
            except _DaemonUnavailable as e:
                if self._auto_start and self.ensure_daemon():
                    continue
                raise
            if resp.get("code") == "busy":
                retry_after = float(resp.get("retry_after", 5))
                print(f"  ⏳ translate daemon busy — retrying in {retry_after:.0f}s")
                time.sleep(retry_after)
                continue
            if not resp.get("ok"):
                raise _DaemonUnavailable(f"daemon translate failed: {resp.get('error')}")
            return resp


class _DirectBackend:
    def translate(self, text, target_language):
        return translate_segment(text, target_language)


def translate_srt_file(input_path, output_path, target_language, intro_marker=None, outro_marker=None, backend=None):
    if backend is None:
        backend = _DirectBackend()
    content = read_srt_text(input_path)
    raw_blocks = re.split(r"\n\n", content.strip())
    valid_blocks = []
    for b in raw_blocks:
        lines = b.split("\n")
        while lines and lines[0].strip() == "":
            lines.pop(0)
        while lines and lines[-1].strip() == "":
            lines.pop()
        if len(lines) >= 2:
            idx = lines[0]
            time_range = lines[1]
            text = "\n".join(lines[2:]) if len(lines) >= 3 else ""
            valid_blocks.append((idx, time_range, text))

    n_total = len(valid_blocks)
    start_idx = None
    end_idx = None
    for i, (_, _, text) in enumerate(valid_blocks):
        if start_idx is None and intro_marker and intro_marker in text:
            start_idx = i
        if outro_marker and outro_marker in text:
            end_idx = i

    translated = []
    for count, (idx, time_range, text) in enumerate(valid_blocks, 1):
        i = count - 1
        if _STOP_REQUESTED:
            raise SystemExit(1)
        if start_idx is not None and i <= start_idx:
            ttext = ""
        elif end_idx is not None and i >= end_idx:
            ttext = ""
        else:
            ttext = backend.translate(text, target_language)
        translated.append(f"{idx}\n{time_range}\n{ttext}".rstrip("\n"))
        if count % 10 == 0 or count == n_total:
            print(f"  Translate: {count}/{n_total}", flush=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(translated) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Translate SRT subtitles via HY-MT (ROCm GPU)")
    parser.add_argument("input_srt", help="Input SRT file path")
    parser.add_argument("output_srt", help="Output translated SRT file path")
    parser.add_argument("-l", "--language", default="English", help="Target language (default: English)")
    parser.add_argument("--intro", default=None, help="Intro marker text — blocks up to and including this are emptied")
    parser.add_argument("--outro", default=None, help="Outro marker text — blocks from this onward are emptied")
    parser.add_argument(
        "--mode",
        choices=["auto", "daemon", "direct"],
        default="auto",
        help="Translation backend: auto (daemon with direct fallback), daemon (require daemon), direct (in-process)",
    )
    args = parser.parse_args()

    print(f"Translating: {args.input_srt} → {args.output_srt}")
    print(f"Target language: {args.language}")

    if args.mode in ("auto", "daemon"):
        from config import TRANSLATE_DAEMON_SOCK

        client = _DaemonTranslateClient(TRANSLATE_DAEMON_SOCK, auto_start=(args.mode == "auto"))
        if args.mode == "auto" and client.ping() is None:
            print(f"translate daemon not running — starting it ({TRANSLATE_DAEMON_SOCK})")
            client.ensure_daemon()
        resp = client.ping()
        if resp is not None and resp.get("ok"):
            print(
                f"Using translate daemon ({TRANSLATE_DAEMON_SOCK}, "
                f"engine={resp.get('engine')}, device={resp.get('device')}, max_jobs={resp.get('max_jobs')})"
            )
            print(flush=True)
            client.translate_file(args.input_srt, args.output_srt, args.language, args.intro, args.outro)
            print("Translation complete.", flush=True)
            return
        if args.mode == "daemon":
            print(f"ERROR: translate daemon not reachable at {TRANSLATE_DAEMON_SOCK}", file=sys.stderr)
            return 1
        print("WARNING: translate daemon not reachable — falling back to direct in-process mode")

    print("Using direct HY-MT (in-process)")
    print(f"Device: {hy_mt._get_model()[0].device}")
    print(flush=True)

    translate_srt_file(args.input_srt, args.output_srt, args.language, args.intro, args.outro)

    hy_mt.unload_model()
    print("Translation complete.", flush=True)


if __name__ == "__main__":
    main()
