#!/bin/bash

if [ $# -lt 4 ]; then
    echo "Usage: $0 <number> <srt_file> <output_dir> <temperature> [blur_chinese] [target_language] [cfg_weight] [exaggeration]"
    echo "  number           - Video number or ID"
    echo "  srt_file         - Path to SRT file"
    echo "  output_dir       - Output directory path"
    echo "  temperature      - Temperature for audio generation (default: 0.6)"
    echo "  blur_chinese     - Blur Chinese subtitles (yes/no, default: yes)"
    echo "  target_language  - Target language code (default: en)"
    echo "  cfg_weight       - CFG weight (default: 0.5)"
    echo "  exaggeration     - Exaggeration level (default: 0.5)"
    exit 1
fi

NUMBER=$1
SRT_FILE="$2"
OUTPUT_DIR="$3"
TEMPERATURE=${4:-0.6}
BLUR_CHINESE=${5:-yes}
TARGET_LANGUAGE=${6:-en}
CFG_WEIGHT=${7:-0.5}
EXAGGERATION=${8:-0.5}
AUDIO_DIR="$OUTPUT_DIR/audio"
echo "Number: $NUMBER"
echo "SRT File: $SRT_FILE"
echo "Output Dir: $OUTPUT_DIR"

if [ ! -f "$SRT_FILE" ]; then
    echo "Error: SRT file not found: $SRT_FILE"
    exit 1
fi

BASE_DIR="${HOME}/子归家/code_ml"
SERVER_DIR="$BASE_DIR/chatterbox-server"
source "${BASE_DIR}/rocm_env.sh"
mkdir -p "$OUTPUT_DIR" "$AUDIO_DIR"
echo "Step 1: Generating audio from SRT..."
PYTHONUNBUFFERED=1 ${HOME}/.pyenv/versions/3.11.14/bin/python3.11 ${HOME}/子归家/code_ml/gen_audio.py "$SRT_FILE" --audio_prompt ${HOME}/子归家/assets/std_ning.wav --output_dir "$AUDIO_DIR" --output_srt output_adjusted.srt --output_wav output.wav --changed_json changed_segments.json --temperature "$TEMPERATURE" --target_language "$TARGET_LANGUAGE" --cfg_weight "$CFG_WEIGHT" --exaggeration "$EXAGGERATION"
if [ $? -ne 0 ]; then echo "Error: gen_audio.py failed"; exit 1; fi
echo "Step 2: Downloading video..."
/usr/bin/python3 ${HOME}/子归家/pre-process/download_orig.py "$NUMBER" "$OUTPUT_DIR" --codec "${CODEC:-mp4}"
if [ $? -ne 0 ]; then echo "Error: download_orig.py failed"; exit 1; fi
echo "Step 3: Processing video with stretched segments..."
VIDEO_FILE="$OUTPUT_DIR/${NUMBER}.mp4"
ADJUSTED_SRT="$AUDIO_DIR/output_adjusted.srt"
CHANGED_JSON="$AUDIO_DIR/changed_segments.json"
OUTPUT_MODIFIED="$OUTPUT_DIR/output_modified.mp4"
if [ "$BLUR_CHINESE" = "yes" ]; then
    /usr/bin/python3 ${HOME}/子归家/code_ml/gen_video.py "$VIDEO_FILE" "$SRT_FILE" "$ADJUSTED_SRT" "$CHANGED_JSON" --output "$OUTPUT_MODIFIED" --blur
else
    /usr/bin/python3 ${HOME}/子归家/code_ml/gen_video.py "$VIDEO_FILE" "$SRT_FILE" "$ADJUSTED_SRT" "$CHANGED_JSON" --output "$OUTPUT_MODIFIED"
fi
if [ $? -ne 0 ]; then echo "Error: gen_video.py failed"; exit 1; fi
echo "Step 4: Adjusting original zh audio..."
/usr/bin/python3 -c "
import sys
sys.path.insert(0, '${SERVER_DIR}')
sys.path.insert(0, '${BASE_DIR}')
from pipeline import adjust_original_audio
adjust_original_audio('${VIDEO_FILE}', '${SRT_FILE}', '${ADJUSTED_SRT}', '${OUTPUT_DIR}')
" || echo "Warning: zh audio adjustment failed (non-fatal)"
echo "Done! All files saved to: $OUTPUT_DIR"
