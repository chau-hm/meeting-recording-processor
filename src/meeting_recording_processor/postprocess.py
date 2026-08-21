"""Deterministic text normalization without translation or content rewriting."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable

from .errors import BackendError
from .schemas import BackendResult, TranscriptSegment


_SPACE_RE = re.compile(r"[ \t\f\v]+")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SENTENCE_ENDINGS = frozenset("。！？!?")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = _CONTROL_RE.sub("", text)
    lines = [_SPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _default_traditional_converter() -> Callable[[str], str]:
    try:
        from opencc import OpenCC
    except ImportError as exc:
        raise BackendError("缺少 OpenCC，無法保證繁體中文輸出；請重新執行 uv sync") from exc
    converter = OpenCC("s2hk")
    return converter.convert


def to_traditional(
    text: str, *, converter: Callable[[str], str] | None = None
) -> str:
    return (converter or _default_traditional_converter())(text)


def _visible_length(text: str) -> int:
    return sum(1 for character in text if not character.isspace())


def _split_long_unit(text: str, max_characters: int) -> list[str]:
    result: list[str] = []
    remaining = text.strip()
    while _visible_length(remaining) > max_characters:
        visible = 0
        cut = 0
        last_space = -1
        for index, character in enumerate(remaining):
            if character.isspace():
                last_space = index
            else:
                visible += 1
            if visible >= max_characters:
                cut = index + 1
                break
        if last_space > 0 and last_space >= cut // 2:
            cut = last_space
        result.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        result.append(remaining)
    return result


def split_text_for_cues(text: str, *, max_characters: int = 42) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    units: list[str] = []
    buffer: list[str] = []
    for character in text:
        buffer.append(character)
        if character in _SENTENCE_ENDINGS or character == "\n":
            unit = "".join(buffer).strip()
            if unit:
                units.extend(_split_long_unit(unit, max_characters))
            buffer = []
    tail = "".join(buffer).strip()
    if tail:
        units.extend(_split_long_unit(tail, max_characters))
    return units


def _estimate_segments(
    text: str,
    *,
    start: float,
    end: float,
    timing_source: str,
) -> list[TranscriptSegment]:
    cues = split_text_for_cues(text)
    if not cues:
        return []
    duration = max(0.001, end - start)
    weights = [max(1, _visible_length(cue)) for cue in cues]
    total_weight = sum(weights)
    current = start
    segments: list[TranscriptSegment] = []
    for index, (cue, weight) in enumerate(zip(cues, weights, strict=True)):
        segment_end = end if index == len(cues) - 1 else current + duration * weight / total_weight
        segments.append(
            TranscriptSegment(
                start=round(current, 3),
                end=round(max(segment_end, current + 0.001), 3),
                text=cue,
                timing_source=timing_source,
            )
        )
        current = segment_end
    return segments


def _group_model_segments(
    segments: Iterable[TranscriptSegment],
    *,
    max_duration: float = 7.0,
    max_characters: int = 42,
) -> list[TranscriptSegment]:
    result: list[TranscriptSegment] = []
    buffer: list[TranscriptSegment] = []

    def join_buffer() -> str:
        joined = ""
        for item in buffer:
            current = item.text
            if not joined or not current:
                joined += current
                continue
            if current[:1].isspace():
                joined += current
            elif (
                joined[-1:].isascii()
                and joined[-1:].isalnum()
                and current[:1].isascii()
                and current[:1].isalnum()
            ):
                joined += " " + current
            else:
                joined += current
        return joined

    def flush() -> None:
        if not buffer:
            return
        raw = join_buffer()
        text = clean_text(raw)
        if text:
            result.append(
                TranscriptSegment(
                    start=buffer[0].start,
                    end=buffer[-1].end,
                    text=text,
                    timing_source="model",
                )
            )
        buffer.clear()

    for segment in segments:
        if not segment.text.strip():
            continue
        buffer.append(segment)
        combined = join_buffer()
        duration = buffer[-1].end - buffer[0].start
        if (
            duration >= max_duration
            or _visible_length(combined) >= max_characters
            or combined.rstrip()[-1:] in _SENTENCE_ENDINGS
        ):
            flush()
    flush()
    return result


def postprocess_result(
    result: BackendResult,
    *,
    audio_duration: float,
    converter: Callable[[str], str] | None = None,
) -> tuple[str, tuple[TranscriptSegment, ...], tuple[str, ...]]:
    traditional_converter = converter or _default_traditional_converter()
    text = traditional_converter(clean_text(result.text))

    raw_segments = [
        TranscriptSegment(
            start=max(0.0, float(segment.start)),
            end=max(float(segment.start) + 0.001, float(segment.end)),
            text=traditional_converter(clean_text(segment.text)),
            timing_source=segment.timing_source,
        )
        for segment in result.segments
        if clean_text(segment.text)
    ]

    if raw_segments and all(segment.timing_source == "model" for segment in raw_segments):
        segments = _group_model_segments(raw_segments)
    elif raw_segments:
        segments = []
        for segment in raw_segments:
            segments.extend(
                _estimate_segments(
                    segment.text,
                    start=segment.start,
                    end=segment.end,
                    timing_source="estimated_from_chunk",
                )
            )
    else:
        segments = _estimate_segments(
            text,
            start=0.0,
            end=max(audio_duration, 0.001),
            timing_source="estimated_from_duration",
        )

    warnings = list(result.warnings)
    if any(segment.timing_source.startswith("estimated") for segment in segments):
        warnings.append("字幕時間由音訊分段估算，並非 word-level model timestamp")
    return text, tuple(segments), tuple(dict.fromkeys(warnings))
