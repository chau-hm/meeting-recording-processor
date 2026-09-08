"""Qwen3-ASR adapter using the pinned native MLX runtime."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

from ..config import DEFAULT_QWEN_ALIGNER_MODEL, DEFAULT_QWEN_MODEL
from ..errors import BackendError
from ..progress import ProgressEvent, ProgressPhase
from ..schemas import BackendResult, TranscriptSegment
from .base import ProgressCallback, isolate_progress_callback

BACKEND_NAME = "qwen3"
DEFAULT_MODEL_ID = DEFAULT_QWEN_MODEL
DEFAULT_ALIGNER_MODEL_ID = DEFAULT_QWEN_ALIGNER_MODEL
RUNTIME_PACKAGE = "mlx-qwen3-asr==0.3.5"


def _value(item: object, name: str, default: Any = None) -> Any:
    return item.get(name, default) if isinstance(item, dict) else getattr(item, name, default)


def _segments_from_result(items: object, *, timing_source: str) -> tuple[TranscriptSegment, ...]:
    if not isinstance(items, (list, tuple)):
        return ()
    segments: list[TranscriptSegment] = []
    for item in items:
        text = str(_value(item, "text", "") or "")
        start = _value(item, "start")
        end = _value(item, "end")
        try:
            if text.strip() and start is not None and end is not None:
                segments.append(
                    TranscriptSegment(
                        start=float(start),
                        end=float(end),
                        text=text,
                        timing_source=timing_source,
                    )
                )
        except (TypeError, ValueError):
            continue
    return tuple(segments)


def _optional_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
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

        segments = _segments_from_result(getattr(result, "segments", None), timing_source="model")
        if not segments:
            segments = _segments_from_result(getattr(result, "chunks", None), timing_source="chunk")

        metadata = {
            **aligner_metadata,
            "finish_reason": getattr(result, "finish_reason", None),
            "truncated": bool(getattr(result, "truncated", False)),
            "timestamp_source": "word" if segments and segments[0].timing_source == "model" else "chunk_or_none",
        }
        warnings: list[str] = []
        if metadata["truncated"]:
            warnings.append("Qwen3-ASR 回報輸出可能因 token budget 截斷")
        return BackendResult(
            language=str(getattr(result, "language", None) or language),
            backend=self.name,
            model=self.model_id,
            text=str(getattr(result, "text", "") or ""),
            segments=segments,
            metadata=metadata,
            warnings=tuple(warnings),
        )
