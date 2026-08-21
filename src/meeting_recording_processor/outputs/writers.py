"""Export canonical transcript JSON to plain text and SubRip captions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import ExportConfig
from ..errors import OutputExistsError
from ..schema_io import atomic_write_text, load_package

SUPPORTED_FORMATS = frozenset({"txt", "srt"})


def format_srt_timestamp(seconds: float) -> str:
    total_milliseconds = max(0, round(float(seconds) * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def render_srt(segments: list[dict[str, Any]]) -> str:
    cues: list[str] = []
    for index, segment in enumerate(segments, start=1):
        start = format_srt_timestamp(float(segment["start"]))
        end = format_srt_timestamp(float(segment["end"]))
        text = str(segment["text"]).strip()
        cues.append(f"{index}\n{start} --> {end}\n{text}")
    return "\n\n".join(cues) + "\n"


def export_transcript(config: ExportConfig) -> tuple[Path, Path]:
    payload = load_package(config.transcript_path.resolve(), require_completed=True)
    transcript = payload["transcript"]
    output_dir = config.resolved_output_dir.resolve()
    txt_path = output_dir / f"{config.base_name}.txt"
    srt_path = output_dir / f"{config.base_name}.srt"
    if not config.overwrite:
        existing = [path for path in (txt_path, srt_path) if path.exists()]
        if existing:
            raise OutputExistsError(
                f"輸出已存在：{', '.join(str(path) for path in existing)}；如要取代請加 --overwrite"
            )
    text = str(transcript["text"]).rstrip() + "\n"
    srt = render_srt(transcript["segments"])
    atomic_write_text(txt_path, text, overwrite=config.overwrite)
    atomic_write_text(srt_path, srt, overwrite=config.overwrite)
    return txt_path, srt_path
