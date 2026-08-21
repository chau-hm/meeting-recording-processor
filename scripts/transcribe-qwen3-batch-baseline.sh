#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <input-directory> [output-directory]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

INPUT_DIR="$1"
if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Error: input directory not found: $INPUT_DIR" >&2
  exit 1
fi
INPUT_DIR="$(cd "$INPUT_DIR" && pwd)"

OUTPUT_DIR="${2:-$ROOT_DIR/output/qwen3-baseline}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

MODEL="${ASR_MODEL:-Qwen/Qwen3-ASR-1.7B}"
LANGUAGE="${ASR_LANGUAGE:-Cantonese}"
CONTEXT_FILE="${ASR_CONTEXT_FILE:-$ROOT_DIR/profiles/generic.txt}"

export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HOME"

files=()
while IFS= read -r -d '' file; do
  files+=("$file")
done < <(
  find "$INPUT_DIR" -maxdepth 1 -type f \
    \( -iname '*.wav' -o -iname '*.m4a' -o -iname '*.mp3' -o -iname '*.flac' -o -iname '*.mp4' -o -iname '*.mov' \) \
    -print0 | sort -z
)

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No supported audio/video files found in: $INPUT_DIR" >&2
  exit 1
fi

args=(
  "${files[@]}"
  --model "$MODEL"
  --timestamps
  -f all
  -o "$OUTPUT_DIR"
  --verbose
)

if [[ -n "$LANGUAGE" && "$LANGUAGE" != "auto" ]]; then
  args+=(--language "$LANGUAGE")
fi

if [[ -f "$CONTEXT_FILE" ]]; then
  CONTEXT="$(tr '\n' ' ' < "$CONTEXT_FILE" | tr -s ' ' | sed 's/^ //; s/ $//')"
  if [[ -n "$CONTEXT" ]]; then
    args+=(--context "$CONTEXT")
  fi
fi

echo "Baseline: Qwen3 only; no fallback or per-file run manifest yet."
echo "Files:    ${#files[@]}"
echo "Model:    $MODEL"
echo "Language: ${LANGUAGE:-auto}"
echo "Output:   $OUTPUT_DIR"
echo

uv run mlx-qwen3-asr "${args[@]}"

echo
echo "Baseline batch transcription completed: $OUTPUT_DIR"
echo "Raw ASR output has not been manually verified."
echo "Workflow stopped at Step 1; no cleanup or meeting notes were generated."

