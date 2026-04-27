#!/bin/bash
# Generate TTS audio for a spoken-text file
# Usage: generate-audio.sh <input.txt> <output.mp3> [voice]
set -e

INPUT="$1"
OUTPUT="$2"
VOICE="${3:-will}"
DICT_DIR="$(dirname "$0")/.."
WORK_DIR=$(mktemp -d)

if [ -z "$INPUT" ] || [ -z "$OUTPUT" ]; then
    echo "Usage: $0 <input.txt> <output.mp3> [voice]"
    exit 1
fi

echo "=== SP Study Guide TTS Generator ==="
echo "Input:  $INPUT"
echo "Output: $OUTPUT"
echo "Voice:  $VOICE"

echo "-> Applying pronunciation dictionary..."
python3 "$DICT_DIR/scripts/apply-pronunciation.py" "$INPUT" "$DICT_DIR/pronunciation.txt" > "$WORK_DIR/pronounced.txt"

echo "-> Chunking text..."
python3 "$DICT_DIR/scripts/chunk-text.py" "$WORK_DIR/pronounced.txt" --max-chars 2000 --output-dir "$WORK_DIR" --prefix chunk

echo "-> Generating audio chunks..."
for chunk in "$WORK_DIR"/chunk_*.txt; do
    base=$(basename "$chunk" .txt)
    echo "  Processing $base..."
    kokoro-tts "$(cat "$chunk")" --voice "$VOICE" --format mp3 --output "$WORK_DIR/${base}.mp3"
done

echo "-> Concatenating..."
ls "$WORK_DIR"/chunk_*.mp3 | sort | sed "s|.*|file '&'|" > "$WORK_DIR/concat.txt"
ffmpeg -y -f concat -safe 0 -i "$WORK_DIR/concat.txt" -c:a libmp3lame -b:a 128k "$OUTPUT" 2>/dev/null

DURATION=$(ffprobe -v quiet -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUTPUT" 2>/dev/null)
SIZE=$(ls -lh "$OUTPUT" | awk '{print $5}')
MINS=$(echo "$DURATION / 60" | bc)
SECS=$(echo "$DURATION % 60 / 1" | bc)
echo "=== Done ==="
echo "File:     $OUTPUT"
echo "Duration: ${MINS}m ${SECS}s"
echo "Size:     $SIZE"

rm -rf "$WORK_DIR"
