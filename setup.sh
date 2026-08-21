#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Error: Meeting Recording Processor v1 requires macOS." >&2
  exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "Error: Meeting Recording Processor v1 requires Apple Silicon (arm64)." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  cat >&2 <<'MSG'
Error: uv is not installed.
Install it first, for example:
  brew install uv
MSG
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  cat >&2 <<'MSG'
Error: ffmpeg and ffprobe are required.
Install them first, for example:
  brew install ffmpeg
MSG
  exit 1
fi

export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.cache/uv}"
mkdir -p "$HF_HOME"

echo "Creating/updating the uv environment..."
uv sync

echo
echo "Checking installed Python packages..."
uv run python -c 'import huggingface_hub, mlx_audio, mlx_qwen3_asr, opencc; print("Python packages: OK")'

echo
echo "Setup complete."
echo "Model cache: $HF_HOME"
echo "Next: ./scripts/download-models.sh"
