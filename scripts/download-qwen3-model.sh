#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

args=(--asr qwen3)
if [[ -n "${ASR_MODEL:-}" ]]; then
  args+=(--qwen-model "$ASR_MODEL")
fi
exec uv run mrp download-model "${args[@]}" "$@"
