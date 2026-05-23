#!/usr/bin/env bash
#
# rapid_videocr_pipeline.sh — Full OCR pipeline for a video file.
#
# Extracts frames from the video, renames them to VideoSubFinder naming
# convention, runs rapid_videocr, then merges adjacent duplicate segments.
#
# Usage:
#   ./rapid_videocr_pipeline.sh -i input.mp4 [options]
#
# Options:
#   -i, --input FILE     Input video file (required)
#   -d, --dir DIR        Temp directory for frames (default: ./frames)
#   -f, --fps N          Frames per second (default: 3)
#   -o, --output FILE    Output SRT file (default: ./output.srt)
#   -p, --prefix STR     Output filename prefix for intermediate SRT (default: result)
#   -s, --save-dir DIR   Save raw OCR output to DIR (default: temp dir, auto-cleaned)
#   --keep-frames        Don't delete the frames directory after processing
#   -h, --help           Show this help
#

set -euo pipefail

# ---- defaults ----
FPS=3
FRAMES_DIR="./frames"
OUTPUT_SRT="./output.srt"
PREFIX="result"
SAVE_DIR=""
KEEP_FRAMES=0
RAPID_VIDEOCR_BIN="${RAPID_VIDEOCR_BIN:-rapid_videocr}"

# ---- parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input)     INPUT_VIDEO="$2"; shift 2 ;;
        -d|--dir)       FRAMES_DIR="$2";  shift 2 ;;
        -f|--fps)       FPS="$2";         shift 2 ;;
        -o|--output)    OUTPUT_SRT="$2";  shift 2 ;;
        -p|--prefix)    PREFIX="$2";      shift 2 ;;
        -s|--save-dir)  SAVE_DIR="$2";    shift 2 ;;
        --keep-frames)  KEEP_FRAMES=1;    shift   ;;
        -h|--help)      head -20 "$0";    exit 0   ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "${INPUT_VIDEO:-}" ]]; then
    echo "Error: --input is required"
    exit 1
fi

if [[ ! -f "$INPUT_VIDEO" ]]; then
    echo "Error: input file not found: $INPUT_VIDEO"
    exit 1
fi

# ---- resolve absolute paths ----
INPUT_VIDEO="$(realpath "$INPUT_VIDEO")"
FRAMES_DIR="$(realpath -m "$FRAMES_DIR")"
OUTPUT_SRT="$(realpath -m "$OUTPUT_SRT")"

echo "=== RapidVideOCR Pipeline ==="
echo "Input:      $INPUT_VIDEO"
echo "Frames:     $FPS fps -> $FRAMES_DIR"
echo "Output:     $OUTPUT_SRT"
if [[ -n "$SAVE_DIR" ]]; then
    echo "Raw save:  $SAVE_DIR"
fi
echo ""

# ---- 1. Extract frames at given FPS ----
echo "[1/4] Extracting frames at ${FPS}fps..."
mkdir -p "$FRAMES_DIR"
rm -f "$FRAMES_DIR"/*.png
ffmpeg -y -i "$INPUT_VIDEO" -vf "fps=${FPS}" "$FRAMES_DIR/frame_%05d.png" 2>&1 | tail -1

# ---- 2. Rename frames to VSF naming convention ----
echo "[2/4] Renaming frames to VSF convention..."
FRAME_COUNT=$(python3 -c "
import os
from pathlib import Path

d = Path('${FRAMES_DIR}')
frames = sorted(d.glob('frame_*.png'))
n = len(frames)
ms_per_frame = 1000.0 / ${FPS}
zeros = '0000000000000000000000000'

for i, p in enumerate(frames):
    start_ms = round(i * ms_per_frame)
    end_ms   = round((i + 1) * ms_per_frame)

    h, rem   = divmod(start_ms, 3600000)
    m, rem   = divmod(rem, 60000)
    s, mmm   = divmod(rem, 1000)

    h2, rem2 = divmod(end_ms, 3600000)
    m2, rem2 = divmod(rem2, 60000)
    s2, mmm2 = divmod(rem2, 1000)

    dst = f'{h}_{m:02d}_{s:02d}_{mmm:03d}__{h2}_{m2:02d}_{s2:02d}_{mmm2:03d}_{zeros}.png'
    os.rename(str(p), str(d / dst))

print(n)
")
echo "Renamed ${FRAME_COUNT} frames"

# ---- 3. Run rapid_videocr in chunks ----
echo "[3/4] Running rapid_videocr in chunks of 800 frames..."
if [[ -n "$SAVE_DIR" ]]; then
    RAW_OUTDIR="$(realpath -m "$SAVE_DIR")"
    mkdir -p "$RAW_OUTDIR"
else
    RAW_OUTDIR="$(mktemp -d -t rvoc_XXXXXX)"
fi

# Gather sorted frame list
FRAME_FILES=()
while IFS= read -r -d '' f; do
    FRAME_FILES+=("$f")
done < <(find "$FRAMES_DIR" -maxdepth 1 -name '*.png' -print0 | sort -z)

TOTAL_FRAMES=${#FRAME_FILES[@]}
CHUNK_SIZE=800
CHUNK_COUNT=$(( (TOTAL_FRAMES + CHUNK_SIZE - 1) / CHUNK_SIZE ))
echo "  Splitting ${TOTAL_FRAMES} frames into ${CHUNK_COUNT} chunks of ${CHUNK_SIZE}..."

ALL_SRTS=()
for ((chunk=0; chunk<CHUNK_COUNT; chunk++)); do
    start=$((chunk * CHUNK_SIZE))
    end=$((start + CHUNK_SIZE))
    (( end > TOTAL_FRAMES )) && end=$TOTAL_FRAMES

    CHUNK_DIR="${RAW_OUTDIR}/chunk_${chunk}"
    mkdir -p "$CHUNK_DIR"

    # Move frames for this chunk
    for ((i=start; i<end; i++)); do
        mv "${FRAME_FILES[$i]}" "$CHUNK_DIR/"
    done

    CHUNK_PREFIX="${PREFIX}_chunk${chunk}"
    echo "  Chunk $((chunk+1))/${CHUNK_COUNT}: ${CHUNK_PREFIX} (frames $((start+1))-${end})"
    "$RAPID_VIDEOCR_BIN" -i "$CHUNK_DIR" -s "$CHUNK_DIR" -f "$CHUNK_PREFIX" -o srt || true

    CHUNK_SRT="${CHUNK_DIR}/${CHUNK_PREFIX}.srt"
    [[ ! -f "$CHUNK_SRT" ]] && CHUNK_SRT="${CHUNK_DIR}/outputs/${CHUNK_PREFIX}.srt"
    if [[ -f "$CHUNK_SRT" ]]; then
        ALL_SRTS+=("$CHUNK_SRT")
    else
        echo "  Warning: no SRT output for chunk ${chunk}"
    fi
done

if [[ ${#ALL_SRTS[@]} -eq 0 ]]; then
    echo "Error: no SRT produced from any chunk"
    exit 1
fi

# Concatenate all chunk SRTs into one
RAW_SRT="${RAW_OUTDIR}/${PREFIX}.srt"
> "$RAW_SRT"
for srt in "${ALL_SRTS[@]}"; do
    cat "$srt" >> "$RAW_SRT"
    echo "" >> "$RAW_SRT"
done

echo "Raw SRT: $RAW_SRT (${#ALL_SRTS[@]} chunks, ${TOTAL_FRAMES} frames)"

# ---- 4. Merge adjacent duplicate / near-duplicate segments ----
echo "[4/4] Merging adjacent duplicate segments..."
python3 -c "
import re
from difflib import SequenceMatcher

with open('${RAW_SRT}') as f:
    raw = f.read()

entries = []
for block in raw.strip().split('\n\n'):
    lines = block.strip().split('\n')
    if len(lines) < 2:
        continue
    time_line = lines[1]
    text = '\n'.join(lines[2:]).strip()
    entries.append((time_line, text))

def clean_text(t):
    t = re.sub(r'^1[\n\s]*', '', t)
    t = re.sub(r'^[\d\W]+', '', t)
    return t.strip()

def is_similar(a, b, threshold=0.75):
    \"\"\"Check if two texts are similar enough to merge.\"\"\"
    if not a or not b:
        return False
    if a == b:
        return True
    # Normalize for comparison: keep only Chinese chars and common punct
    def norm(s):
        return re.sub(r'[^\u4e00-\u9fff\uff00-\uffef\u3000-\u303f]', '', s)
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return False
    return SequenceMatcher(None, na, nb).ratio() >= threshold

merged = []
for time_line, text in entries:
    cleaned = clean_text(text)
    if not cleaned:
        continue
    if merged and is_similar(cleaned, merged[-1][1]):
        # Merge: extend end time, keep the longer/more complete text
        end_time = time_line.split(' --> ')[1]
        prev_start = merged[-1][0].split(' --> ')[0]
        best_text = cleaned if len(cleaned) > len(merged[-1][1]) else merged[-1][1]
        merged[-1] = (prev_start + ' --> ' + end_time, best_text)
    else:
        merged.append((time_line, cleaned))

out_lines = []
for i, (time_line, text) in enumerate(merged, 1):
    out_lines.append(str(i))
    out_lines.append(time_line)
    out_lines.append(text)
    out_lines.append('')

with open('${OUTPUT_SRT}', 'w') as f:
    f.write('\n'.join(out_lines))

print(f'Before: {len(entries)} entries')
print(f'After:  {len(merged)} entries')
"

echo ""
echo "Done! Merged SRT saved to: ${OUTPUT_SRT}"

# ---- Restore frames if --keep-frames was requested ----
if [[ "$KEEP_FRAMES" -eq 1 ]]; then
    for ((chunk=0; chunk<CHUNK_COUNT; chunk++)); do
        CHUNK_DIR="${RAW_OUTDIR}/chunk_${chunk}"
        if [[ -d "$CHUNK_DIR" ]]; then
            mv "$CHUNK_DIR"/*.png "$FRAMES_DIR/" 2>/dev/null || true
        fi
    done
fi

# ---- Cleanup ----
if [[ "$KEEP_FRAMES" -eq 0 ]]; then
    echo "Cleaning up frames directory..."
    rm -rf "$FRAMES_DIR"
fi
# Clean up temp raw output dir (skip if user provided --save-dir)
if [[ -z "$SAVE_DIR" ]]; then
    rm -rf "${RAW_OUTDIR:-}"
fi
