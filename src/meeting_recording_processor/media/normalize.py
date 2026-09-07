"""Extract the selected audio stream as canonical 16 kHz mono PCM WAV."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Callable

from ..errors import MediaError
from .probe import MediaMetadata

TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
TARGET_CODEC = "pcm_s16le"

Runner = Callable[..., subprocess.CompletedProcess[str]]


def normalize_audio(
    input_path: Path,
    metadata: MediaMetadata,
    destination: Path,
    *,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    runner: Runner = subprocess.run,
) -> Path:
    if target_sample_rate <= 0:
        raise MediaError(f"音訊 sample rate 無效：{target_sample_rate}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-map",
        f"0:{metadata.selected_audio_stream}",
        "-vn",
        "-ac",
        str(TARGET_CHANNELS),
        "-ar",
        str(target_sample_rate),
        "-c:a",
        TARGET_CODEC,
        "-y",
        str(destination),
    ]
    try:
        completed = runner(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise MediaError(f"無法執行 ffmpeg：{exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or "ffmpeg failed").strip()
        raise MediaError(f"音訊抽取／標準化失敗：{detail}")
    if not destination.is_file() or destination.stat().st_size == 0:
        raise MediaError("ffmpeg 冇產生有效 normalized WAV")
    return destination
