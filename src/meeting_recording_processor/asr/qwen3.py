"""Qwen3-ASR adapter using the pinned native MLX runtime."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

from ..config import DEFAULT_QWEN_ALIGNER_MODEL, DEFAULT_QWEN_MODEL
from ..errors import BackendError
from ..progress import ProgressEvent, ProgressPhase
from ..schemas import (
    MIN_SEGMENT_DURATION,
    BackendResult,
    RawTranscriptSegment,
    TranscriptSegment,
)
from .base import ProgressCallback, isolate_progress_callback

BACKEND_NAME = "qwen3"
DEFAULT_MODEL_ID = DEFAULT_QWEN_MODEL
DEFAULT_ALIGNER_MODEL_ID = DEFAULT_QWEN_ALIGNER_MODEL
RUNTIME_PACKAGE = "mlx-qwen3-asr==0.3.5"


def _value(item: object, name: str, default: Any = None) -> Any:
    return item.get(name, default) if isinstance(item, dict) else getattr(item, name, default)


def _finite_number(value: object) -> float | None:
    if value is None or isinstance(value, (str, bytes, bool)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _diagnostic_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return f"<{type(value).__name__}>"
    return number if math.isfinite(number) else repr(value)


def _collect_raw_segments(
    items: object,
    *,
    timing_source: str,
) -> tuple[tuple[RawTranscriptSegment, ...], tuple[dict[str, Any], ...]]:
    if items is None:
        return (), ()
    if not isinstance(items, (list, tuple)):
        return (
            (),
            (
                {
                    "source": timing_source,
                    "index": None,
                    "reason": "timing records are not an array",
                    "value_type": type(items).__name__,
                },
            ),
        )

    raw_segments: list[RawTranscriptSegment] = []
    diagnostics: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        text_value = _value(item, "text")
        start_value = _value(item, "start")
        end_value = _value(item, "end")
        text = text_value if isinstance(text_value, str) else None
        start = _finite_number(start_value)
        end = _finite_number(end_value)
        reasons: list[str] = []
        if text is None or not text.strip():
            reasons.append("text is empty or not a string")
        if start is None:
            reasons.append("start is not a finite number")
        if end is None:
            reasons.append("end is not a finite number")
        if reasons:
            diagnostic: dict[str, Any] = {
                "source": timing_source,
                "index": index,
                "reason": "; ".join(reasons),
                "start": _diagnostic_value(start_value),
                "end": _diagnostic_value(end_value),
            }
            if text is not None:
                diagnostic["text"] = text
            diagnostics.append(diagnostic)
            continue
        raw_segments.append(
            RawTranscriptSegment(
                start=start,
                end=end,
                text=text,
                timing_source=timing_source,
            )
        )
    return tuple(raw_segments), tuple(diagnostics)


def _word_timing(
    items: object,
) -> tuple[
    tuple[RawTranscriptSegment, ...],
    tuple[TranscriptSegment, ...],
    int,
    tuple[dict[str, Any], ...],
    str | None,
]:
    raw_segments, diagnostics = _collect_raw_segments(items, timing_source="model")
    if items is None or (isinstance(items, (list, tuple)) and not items):
        return raw_segments, (), 0, diagnostics, None
    if diagnostics:
        return (
            raw_segments,
            (),
            0,
            diagnostics,
            f"{len(diagnostics)} malformed or non-finite timing record(s)",
        )

    canonical: list[TranscriptSegment] = []
    previous_start: float | None = None
    repair_count = 0
    for index, segment in enumerate(raw_segments):
        if segment.start < 0:
            return raw_segments, (), 0, diagnostics, f"segment {index} has a negative start"
        if segment.end < segment.start:
            return (
                raw_segments,
                (),
                0,
                diagnostics,
                f"segment {index} ends before it starts",
            )
        if previous_start is not None and segment.start < previous_start:
            return (
                raw_segments,
                (),
                0,
                diagnostics,
                f"segment {index} starts before the previous segment",
            )
        end = segment.end
        if end == segment.start:
            end = segment.start + MIN_SEGMENT_DURATION
            repair_count += 1
        canonical.append(
            TranscriptSegment(
                start=segment.start,
                end=end,
                text=segment.text,
                timing_source="model",
            )
        )
        previous_start = segment.start
    return tuple(raw_segments), tuple(canonical), repair_count, diagnostics, None


def _chunk_timing(
    items: object,
) -> tuple[
    tuple[RawTranscriptSegment, ...],
    tuple[TranscriptSegment, ...],
    tuple[dict[str, Any], ...],
    str | None,
]:
    raw_segments, diagnostics = _collect_raw_segments(items, timing_source="chunk")
    if items is None or (isinstance(items, (list, tuple)) and not items):
        return raw_segments, (), diagnostics, None
    if diagnostics:
        return (
            raw_segments,
            (),
            diagnostics,
            f"{len(diagnostics)} malformed or non-finite timing record(s)",
        )

    canonical: list[TranscriptSegment] = []
    previous_start: float | None = None
    for index, segment in enumerate(raw_segments):
        if segment.start < 0:
            return (
                raw_segments,
                (),
                diagnostics,
                f"chunk {index} has a negative start",
            )
        if segment.end <= segment.start:
            return (
                raw_segments,
                (),
                diagnostics,
                f"chunk {index} does not have positive duration",
            )
        if previous_start is not None and segment.start < previous_start:
            return (
                raw_segments,
                (),
                diagnostics,
                f"chunk {index} starts before the previous chunk",
            )
        canonical.append(
            TranscriptSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                timing_source="chunk",
            )
        )
        previous_start = segment.start
    return tuple(raw_segments), tuple(canonical), diagnostics, None


def _optional_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number >= 0 and math.isfinite(number) else None


def _progress_event(payload: object) -> ProgressEvent | None:
    if not isinstance(payload, dict):
        return None

    event_name = str(payload.get("event", ""))
    if event_name not in {
        "chunks_prepared",
        "chunk_started",
        "chunk_completed",
        "completed",
    }:
        return None

    processed = _optional_number(payload.get("processed_audio_sec"))
    total = _optional_number(payload.get("audio_duration_sec"))
    current = processed
    unit = "seconds"
    if event_name == "chunks_prepared" and current is None and total is not None:
        current = 0.0
    if current is None or total is None or total <= 0:
        current = _optional_number(payload.get("chunk_index"))
        total = _optional_number(payload.get("total_chunks"))
        unit = "chunks"

    chunk_index = payload.get("chunk_index")
    total_chunks = payload.get("total_chunks")
    if event_name == "chunks_prepared":
        message = "Transcribing..."
    elif chunk_index is not None and total_chunks is not None:
        message = f"Transcribing chunk {chunk_index}/{total_chunks}..."
    else:
        message = "Transcribing..."
    return ProgressEvent(
        phase=ProgressPhase.TRANSCRIBING,
        current=current,
        total=total,
        unit=unit,
        message=message,
        determinate=current is not None and total is not None and total > 0,
    )


def _safe_progress_callback(
    progress_callback: ProgressCallback | None,
) -> Callable[[dict[str, Any]], None] | None:
    isolated = isolate_progress_callback(progress_callback)
    if isolated is None:
        return None

    enabled = True

    def on_progress(payload: dict[str, Any]) -> None:
        nonlocal enabled
        if not enabled:
            return
        try:
            event = _progress_event(payload)
            if event is not None:
                isolated(event)
        except Exception:
            enabled = False

    return on_progress


class Qwen3Backend:
    name = BACKEND_NAME

    def __init__(
        self,
        *,
        model_id: str,
        model_path: Path,
        aligner_model_id: str = DEFAULT_ALIGNER_MODEL_ID,
        aligner_path: Path | None = None,
        aligner_snapshot: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.aligner_model_id = aligner_model_id
        self.aligner_path = aligner_path
        self.aligner_snapshot = aligner_snapshot
        self.verbose = verbose

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str,
        profile_text: str | None,
        progress_callback: ProgressCallback | None = None,
    ) -> BackendResult:
        aligner_metadata = {
            "aligner_model": self.aligner_model_id,
            "aligner_snapshot": self.aligner_snapshot,
        }
        if self.aligner_path is None:
            raise BackendError(
                "Qwen3 timestamp transcription requires a locally resolved forced "
                "aligner; run download-model --asr qwen3 first",
                metadata=aligner_metadata,
            )

        on_progress = _safe_progress_callback(progress_callback)
        try:
            from mlx_qwen3_asr import transcribe

            result = transcribe(
                str(audio_path),
                model=str(self.model_path),
                language=None if language.lower() == "auto" else language,
                context=profile_text or None,
                return_timestamps=True,
                return_chunks=True,
                forced_aligner=str(self.aligner_path),
                verbose=self.verbose,
                on_progress=on_progress,
            )
        except Exception as exc:
            raise BackendError(f"Qwen3-ASR 執行失敗：{exc}") from exc

        word_items = getattr(result, "segments", None)
        chunk_items = getattr(result, "chunks", None)
        (
            raw_word_segments,
            word_segments,
            zero_duration_repairs,
            word_diagnostics,
            word_rejection_reason,
        ) = _word_timing(word_items)
        (
            raw_chunk_segments,
            chunk_segments,
            chunk_diagnostics,
            chunk_rejection_reason,
        ) = _chunk_timing(chunk_items)

        if word_segments:
            segments = word_segments
            timestamp_source = "word_repaired" if zero_duration_repairs else "word"
        elif chunk_segments:
            segments = chunk_segments
            timestamp_source = "chunk"
        else:
            segments = ()
            timestamp_source = "none"

        raw_segments = (
            raw_word_segments
            if isinstance(word_items, (list, tuple)) and word_items
            else raw_chunk_segments
        )

        metadata = {
            **aligner_metadata,
            "finish_reason": getattr(result, "finish_reason", None),
            "truncated": bool(getattr(result, "truncated", False)),
            "timestamp_source": timestamp_source,
            "raw_word_segment_count": (
                len(word_items) if isinstance(word_items, (list, tuple)) else 0
            ),
            "usable_word_segment_count": len(word_segments),
            "zero_duration_repair_count": zero_duration_repairs,
            "word_timing_rejected": word_rejection_reason is not None,
            "chunk_timing_used": bool(chunk_segments),
            "raw_chunk_segment_count": (
                len(chunk_items) if isinstance(chunk_items, (list, tuple)) else 0
            ),
        }
        if word_rejection_reason is not None:
            metadata["word_timing_rejection_reason"] = word_rejection_reason
        if word_diagnostics:
            metadata["raw_timing_diagnostics"] = list(word_diagnostics)
        if chunk_diagnostics:
            metadata["raw_chunk_timing_diagnostics"] = list(chunk_diagnostics)
        warnings: list[str] = []
        if metadata["truncated"]:
            warnings.append("Qwen3-ASR 回報輸出可能因 token budget 截斷")
        if zero_duration_repairs:
            warnings.append(
                "Qwen3 model word timing 有 "
                f"{zero_duration_repairs} 個 zero-duration segment；canonical timing "
                f"已按 {MIN_SEGMENT_DURATION:.3f}s 最小時長修復"
            )
        if word_rejection_reason is not None:
            if chunk_segments:
                warnings.append(
                    "Qwen3 model word timing 已整體拒絕："
                    f"{word_rejection_reason}；改用 chunk timing"
                )
            else:
                warnings.append(
                    "Qwen3 model word timing 已整體拒絕："
                    f"{word_rejection_reason}；改由音訊總時長估算 timing"
                )
        if chunk_rejection_reason is not None:
            warnings.append(f"Qwen3 chunk timing 無法信任：{chunk_rejection_reason}")
        if not segments and word_items is not None and not word_rejection_reason:
            warnings.append("Qwen3 沒有可用 timing；改由音訊總時長估算 timing")
        return BackendResult(
            language=str(getattr(result, "language", None) or language),
            backend=self.name,
            model=self.model_id,
            text=str(getattr(result, "text", "") or ""),
            segments=segments,
            metadata=metadata,
            warnings=tuple(warnings),
            raw_segments=raw_segments,
        )
