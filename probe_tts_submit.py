"""Probe the live chatterbox server: inject a flask-session, submit a TTS job.

Writes a server-side session file directly into the app's cachelib
FileSystemCache directory (the same storage the running server reads), then
POSTs a multipart form identical to what the browser sends, and prints the
server's response.

Safe by default: it sends an invalid ``temperature`` so the request is
rejected during parameter parsing and no real job is queued. A response of
``{"error":"Invalid temperature: 'abc'"}`` proves the form fields (text
included) reached the route. Pass ``--real`` to actually queue a job.

Usage:
    python3 probe_tts_submit.py                 # safe probe
    python3 probe_tts_submit.py --text "hello"  # custom text
    python3 probe_tts_submit.py --real          # queue a real job
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

DEFAULT_URL = "http://localhost:5600"
DEFAULT_TEXT = "love to code"


def default_session_dir():
    candidates = [
        os.path.expanduser("~/子归家/code_ml/flask_session"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "flask_session"),
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
    body = re.split(r"\r?\n\r?\n", result.stdout, maxsplit=1)[-1].strip()
    if body:
        print(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
