#!/usr/bin/env python3
"""
Standalone SRT translation script for use as a ROCm GPU subprocess.

Usage:
    ./translate_srt.py input.srt output.srt -l English
    ./translate_srt.py input.srt output.srt -l English --intro "杨宁随缘开示" --outro "子归家全体编制人员"
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.expanduser("~/src/HY-MT"))

from rocm_env import setup as _rocm_setup  # noqa: E402

_rocm_setup()  # before hy_mt/torch load ROCm libs

import hy_mt  # noqa: E402


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


def translate_segment(text, target_language, source_has_cjk=True):
    result = text
    for attempt in range(3):
        if attempt == 0:
            result = hy_mt.translate_zh(text, target_language)
        elif attempt == 1:
            result = hy_mt.translate(text, target_language)
        else:
            model, tokenizer = hy_mt._get_model()
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
            outputs = model.generate(tokenized_chat.to(model.device), **hy_mt.GENERATION_KWARGS)
            result = tokenizer.decode(
                outputs[0][len(tokenized_chat[0]) :], skip_special_tokens=True
            )
        if not looks_untranslated(result, source_has_cjk):
            return result
    return result


def translate_srt_file(input_path, output_path, target_language, intro_marker=None, outro_marker=None):
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
        if start_idx is not None and i <= start_idx:
            ttext = ""
        elif end_idx is not None and i >= end_idx:
            ttext = ""
        else:
            ttext = translate_segment(text, target_language)
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
    args = parser.parse_args()

    print(f"Translating: {args.input_srt} → {args.output_srt}")
    print(f"Target language: {args.language}")
    print(f"Device: {hy_mt._get_model()[0].device}")
    print(flush=True)

    translate_srt_file(args.input_srt, args.output_srt, args.language, args.intro, args.outro)

    hy_mt.unload_model()
    print("Translation complete.", flush=True)


if __name__ == "__main__":
    main()
