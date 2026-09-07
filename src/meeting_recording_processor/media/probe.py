"""Inspect media with ffprobe and select its first usable audio stream."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Callable

from ..errors import MediaError


@dataclass(frozen=True, slots=True)
class AudioStream:
    index: int
    codec: str
    channels: int | None
    sample_rate: int | None
    language: str | None = None
    is_default: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MediaMetadata:
    format_name: str
    duration_seconds: float | None
    audio_streams: tuple[AudioStream, ...]
    selected_audio_stream: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_name": self.format_name,
            "duration_seconds": self.duration_seconds,
            "audio_streams": [stream.to_dict() for stream in self.audio_streams],
            "selected_audio_stream": self.selected_audio_stream,
        }


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _optional_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    try:
        result = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return result if result is not None and math.isfinite(result) and result >= 0 else None


def probe_media(path: Path, *, runner: Runner = subprocess.run) -> MediaMetadata:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=format_name,duration:stream=index,codec_type,codec_name,channels,sample_rate:stream_tags=language:stream_disposition=default",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = runner(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise MediaError(f"無法執行 ffprobe：{exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or "ffprobe failed").strip()
        raise MediaError(f"無法讀取媒體資料：{detail}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise MediaError("ffprobe 回傳咗無效 JSON") from exc

    streams: list[AudioStream] = []
    for item in payload.get("streams", []):
        if item.get("codec_type") != "audio":
            continue
        tags = item.get("tags") or {}
        disposition = item.get("disposition") or {}
        stream_index = _optional_int(item.get("index"))
        if stream_index is None:
            continue
        streams.append(
            AudioStream(
                index=stream_index,
                codec=str(item.get("codec_name") or "unknown"),
                channels=_optional_int(item.get("channels")),
                sample_rate=_optional_int(item.get("sample_rate")),
                language=str(tags.get("language")) if tags.get("language") else None,
                is_default=bool(disposition.get("default", 0)),
            )
        )
    if not streams:
        raise MediaError("輸入檔案冇可用 audio stream")

    selected = next((stream for stream in streams if stream.is_default), streams[0])
    format_data = payload.get("format") or {}
    return MediaMetadata(
        format_name=str(format_data.get("format_name") or "unknown"),
        duration_seconds=_optional_float(format_data.get("duration")),
        audio_streams=tuple(streams),
        selected_audio_stream=selected.index,
    )


def ffprobe_version(*, runner: Runner = subprocess.run) -> str | None:
    try:
        completed = runner(
            ["ffprobe", "-version"], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    if completed.returncode != 0 or not completed.stdout:
        return None
    return completed.stdout.splitlines()[0].strip()
