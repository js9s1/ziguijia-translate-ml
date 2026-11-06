#!/usr/bin/env python3
"""
Downloads video from ziguijia.com.

Numeric numbers use https://www.ziguijia.com/chatroom/shoot/NUMBER.
Numbers starting with a letter use the section matching that prefix:
  A… → https://ziguijia.com/fahua/A…
  F… → https://ziguijia.com/jingang/F…
  G… → https://ziguijia.com/wuliangshou/G…
"""

import sys
import re
import os
import subprocess
import urllib3
import requests
from pathlib import Path

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

PREFIX_SECTIONS = {
    "A": "fahua",
    "F": "jingang",
    "G": "wuliangshou",
}


def page_url_for(number):
    """Return the ziguijia page URL for a video number, or None if invalid."""
    number = number.strip()

    if re.fullmatch(r"\d+", number):
        return f"https://www.ziguijia.com/chatroom/shoot/{number}"

    match = re.fullmatch(r"([A-Za-z])(\d+)", number)
    if match:
        prefix = match.group(1).upper()
        section = PREFIX_SECTIONS.get(prefix)
        if section:
            return f"https://ziguijia.com/{section}/{prefix}{match.group(2)}"

    return None


def _is_html_file(path):
    """Check whether a file is HTML (not a valid video)."""
    try:
        with open(path, "rb") as f:
            head = f.read(512).lstrip()
        return head.startswith(b"<!DOCTYPE") or head.startswith(b"<html") or head.startswith(b"<")
    except Exception:
        return False


def download_file(url, filename=None):
    if not filename:
        filename = Path(url).name or "video.mp4"

    print(f"Downloading: {url}")
    print(f"      to file: {filename}")

    try:
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-v", "error",
            "-user_agent", HEADERS["User-Agent"],
            "-headers", f"Referer: https://www.ziguijia.com/\r\nAccept: {HEADERS['Accept']}\r\nAccept-Language: {HEADERS['Accept-Language']}",
            "-i", url,
            "-c", "copy",
            filename,
        ]
        print(f"  Downloading via ffmpeg…")
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0 or not os.path.exists(filename) or os.path.getsize(filename) == 0:
            stderr_tail = result.stderr.strip().split("\n")[-3:] if result.stderr else []
            print(f"  ffmpeg failed: {stderr_tail}")
            return False

        print(f"Done → {filename}")

        if _is_html_file(filename):
            print("ERROR: Downloaded file is HTML, not a valid video!")
            print("  The download server returned a web page instead of video data.")
            print("  Showing first 2000 chars of the response:\n")
            with open(filename, "r", encoding="utf-8", errors="replace") as f:
                print(f.read(2000))
            return False

        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download video from ziguijia.com")
    parser.add_argument("number", help="Video number (digits, or letter prefix + digits, e.g. F1001)")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--codec", choices=["m4v", "mp4"], default="mp4",
                        help="Video codec: m4v=H.265 (unused), mp4=H.264 (default)")
    parser.add_argument("--resolution", type=int, default=720,
                        help="Video resolution (default: 720)")
    args = parser.parse_args()

    number = args.number.strip()
    page_url = page_url_for(number)
    if not page_url:
        print("Error: please provide a numeric video number or a supported letter prefix (A, F, G) followed by digits")
        sys.exit(1)

    number = number[0].upper() + number[1:] if number[0].isalpha() else number

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Fetching page → {page_url}")

    session = requests.Session()
    session.headers.update(HEADERS)

    try:
        r = session.get(page_url, timeout=20, verify=False)
        r.raise_for_status()
    except Exception as e:
        print(f"Cannot access page: {e}")
        sys.exit(1)

    html = r.text

    title_match = re.search(r"<h2>(.*?)</h2>", html)
    title = title_match.group(1) if title_match else f"video_{number}"

    print(f"Title: {title}")

    src_match = re.search(r'<source type="video/mp4" src="([^"]+)"', html)
    if not src_match:
        print("No video source found in page")
        sys.exit(1)

    video_url = src_match.group(1).replace("&amp;", "&")
    video_url = re.sub(r"height=\d+", f"height={args.resolution}", video_url)

    if "#" in video_url:
        video_url = video_url.split("#")[0]

    print(f"Video URL → {video_url}")

    out_name = os.path.join(output_dir, f"{number}.mp4")
    if not download_file(video_url, out_name):
        print("download failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
