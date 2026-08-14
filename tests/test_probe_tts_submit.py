"""Tests for the live-server TTS submit probe.

Unit tests cover the cachelib session-file injection helper without needing a
server. The live HTTP probe only runs when ``TTS_PROBE_URL`` is set::

    TTS_PROBE_URL=http://localhost:5600 pytest tests/test_probe_tts_submit.py

The safe probe sends an invalid ``temperature`` so the request is rejected
during parameter parsing and no real job is queued. A 400 with
``{"error": "Invalid temperature: 'abc'"}`` proves the form fields (text
included) reached the route. Set ``TTS_PROBE_REAL=1`` to also queue a real
job, and ``TTS_PROBE_SESSION_DIR`` to point at the server's flask-session
cache directory (default: auto-detect).

The module can also be run directly as the original CLI probe::

    python3 tests/test_probe_tts_submit.py --real
"""

import argparse
import hashlib
import os
import pickle
import re
import struct
import subprocess
import sys
import uuid

import pytest

DEFAULT_URL = "http://localhost:5600"
DEFAULT_TEXT = "love to code"


def default_session_dir():
    candidates = [
        os.path.expanduser("~/子归家/code_ml/flask_session"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "flask_session"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0]


def write_session(session_dir, sid, user_id=1):
    key = "session:" + sid
    filename = hashlib.md5(key.encode("utf-8")).hexdigest()
    path = os.path.join(session_dir, filename)
    data = {
        "user_id": user_id,
        "user_email": "probe@local",
        "_csrf_token": "probe123",
        "_permanent": False,
    }
    with open(path, "wb") as f:
        f.write(struct.pack("I", 0) + pickle.dumps(data, protocol=5))
    return path


def post_job(url, sid, text, temperature, target_language="en", filename="probe.wav"):
    return subprocess.run(
        [
            "curl",
            "-s",
            "-i",
            "-X",
            "POST",
            url.rstrip("/") + "/tts/process",
            "-H",
            "Cookie: session=" + sid,
            "-F",
            "text=" + text,
            "-F",
            "temperature=" + temperature,
            "-F",
            "target_language=" + target_language,
            "-F",
            "cfg_weight=0.25",
            "-F",
            "exaggeration=0.5",
            "-F",
            "filename=" + filename,
            "-F",
            "csrf_token=probe123",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )


def response_body(stdout):
    """Extract the response body from curl ``-i`` output."""
    return re.split(r"\r?\n\r?\n", stdout, maxsplit=1)[-1].strip()


# ── Unit tests: session-file injection ─────────────────────


class TestWriteSession:
    def test_filename_is_md5_of_session_key(self, tmp_path):
        sid = "probe-abc123"
        path = write_session(str(tmp_path), sid)
        expected = hashlib.md5(("session:" + sid).encode("utf-8")).hexdigest()
        assert os.path.basename(path) == expected

    def test_content_matches_cachelib_format(self, tmp_path):
        path = write_session(str(tmp_path), "probe-abc123", user_id=7)
        with open(path, "rb") as f:
            assert struct.unpack("I", f.read(4))[0] == 0
            data = pickle.load(f)
        assert data["user_id"] == 7
        assert data["user_email"] == "probe@local"
        assert data["_csrf_token"] == "probe123"

    def test_session_readable_by_cachelib(self, tmp_path):
        cachelib = pytest.importorskip("cachelib")
        sid = "probe-cachelib"
        write_session(str(tmp_path), sid)
        cache = cachelib.FileSystemCache(cache_dir=str(tmp_path))
        data = cache.get("session:" + sid)
        assert data["user_id"] == 1


# ── Live-server integration tests (opt-in) ─────────────────


def _live_url():
    url = os.environ.get("TTS_PROBE_URL")
    if not url:
        pytest.skip("TTS_PROBE_URL not set; skipping live probe")
    return url


def _live_session_dir():
    session_dir = os.environ.get("TTS_PROBE_SESSION_DIR") or default_session_dir()
    if not os.path.isdir(session_dir):
        pytest.skip(f"session dir not found: {session_dir}")
    return session_dir


def _inject_and_post(url, session_dir, text, temperature):
    sid = "probe-" + uuid.uuid4().hex[:12]
    path = write_session(session_dir, sid)
    try:
        result = post_job(url, sid, text, temperature)
    finally:
        if os.path.exists(path):
            os.remove(path)
    assert result.returncode == 0, result.stderr.strip()
    return result.stdout


def test_live_safe_probe_rejects_invalid_temperature():
    url = _live_url()
    session_dir = _live_session_dir()
    stdout = _inject_and_post(url, session_dir, DEFAULT_TEXT, "abc")
    assert stdout.splitlines()[0].endswith("400")
    assert "Invalid temperature: 'abc'" in response_body(stdout)


@pytest.mark.skipif(not os.environ.get("TTS_PROBE_REAL"), reason="set TTS_PROBE_REAL=1 to queue a real job")
def test_live_real_job_is_queued():
    url = _live_url()
    session_dir = _live_session_dir()
    stdout = _inject_and_post(url, session_dir, DEFAULT_TEXT, "0.6")
    body = response_body(stdout)
    assert '"access_code"' in body


# ── CLI entry point (kept from the original probe script) ──


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="server base URL")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="text to submit")
    parser.add_argument("--temperature", default="abc", help="temperature value ('abc' keeps the probe safe)")
    parser.add_argument("--target-language", default="en")
    parser.add_argument("--filename", default="probe.wav")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--session-dir", default=None, help="flask-session cache dir (default: auto-detect)")
    parser.add_argument("--real", action="store_true", help="send a valid temperature so a real job is queued")
    parser.add_argument("--keep-session", action="store_true", help="do not delete the injected session file")
    args = parser.parse_args()

    if args.real:
        args.temperature = "0.6"

    session_dir = args.session_dir or default_session_dir()
    if not os.path.isdir(session_dir):
        print(f"error: session dir not found: {session_dir}", file=sys.stderr)
        return 1

    sid = "probe-" + uuid.uuid4().hex[:12]
    path = write_session(session_dir, sid, user_id=args.user_id)
    print(f"injected session {sid} -> {path}")

    result = post_job(args.url, sid, args.text, args.temperature, args.target_language, args.filename)

    if not args.keep_session:
        os.remove(path)
        print("removed injected session")

    if result.returncode != 0:
        print(f"curl failed: {result.stderr.strip()}", file=sys.stderr)
        return 1

    print("--- response ---")
    lines = result.stdout.splitlines()
    print(lines[0] if lines else "(empty)")
    for line in lines[1:]:
        if line.lower().startswith("location:"):
            print(line)
    body = response_body(result.stdout)
    if body:
        print(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
