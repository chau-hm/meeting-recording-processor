#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <audio-or-video-file> [output-directory]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

INPUT="$1"
if [[ ! -f "$INPUT" ]]; then
  echo "Error: input file not found: $INPUT" >&2
  exit 1
fi
INPUT="$(cd "$(dirname "$INPUT")" && pwd)/$(basename "$INPUT")"

case "${INPUT##*.}" in
  wav|WAV|m4a|M4A|mp3|MP3|flac|FLAC|mp4|MP4|mov|MOV) ;;
  *)
    echo "Error: unsupported input extension: $INPUT" >&2
    exit 1
    ;;
esac

OUTPUT_DIR="${2:-$ROOT_DIR/output/qwen3-baseline}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

MODEL="${ASR_MODEL:-Qwen/Qwen3-ASR-1.7B}"
LANGUAGE="${ASR_LANGUAGE:-Cantonese}"
CONTEXT_FILE="${ASR_CONTEXT_FILE:-$ROOT_DIR/profiles/generic.txt}"

export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HOME"

args=(
  "$INPUT"
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

echo "Baseline: Qwen3 only; no fallback or run manifest yet."
echo "Input:    $INPUT"
echo "Model:    $MODEL"
echo "Language: ${LANGUAGE:-auto}"
echo "Output:   $OUTPUT_DIR"
echo "Cache:    $HF_HOME"
echo

uv run mlx-qwen3-asr "${args[@]}"

echo
echo "Baseline transcription completed: $OUTPUT_DIR"
echo "Raw ASR output has not been manually verified."
echo "Workflow stopped at Step 1; no cleanup or meeting notes were generated."

