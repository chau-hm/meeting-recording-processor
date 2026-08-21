"""Qwen3-ASR adapter using the pinned native MLX runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..errors import BackendError
from ..schemas import BackendResult, TranscriptSegment

BACKEND_NAME = "qwen3"
DEFAULT_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
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


class Qwen3Backend:
    name = BACKEND_NAME

    def __init__(self, *, model_id: str, model_path: Path, verbose: bool = False) -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.verbose = verbose

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str,
        profile_text: str | None,
    ) -> BackendResult:
        try:
            from mlx_qwen3_asr import transcribe

            result = transcribe(
                str(audio_path),
                model=str(self.model_path),
                language=None if language.lower() == "auto" else language,
                context=profile_text or None,
                return_timestamps=True,
                return_chunks=True,
                verbose=self.verbose,
            )
        except Exception as exc:
            raise BackendError(f"Qwen3-ASR 執行失敗：{exc}") from exc

        segments = _segments_from_result(getattr(result, "segments", None), timing_source="model")
        if not segments:
            segments = _segments_from_result(getattr(result, "chunks", None), timing_source="chunk")

        metadata = {
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
