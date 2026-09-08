"""SenseVoiceSmall MLX fallback adapter with coarse chunk timestamps."""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
from typing import Any
import wave

from ..errors import BackendError, MediaError
from ..progress import ProgressEvent, ProgressPhase
from ..schemas import BackendResult, TranscriptSegment
from .base import ProgressCallback, isolate_progress_callback

BACKEND_NAME = "sensevoice"
DEFAULT_MODEL_ID = "mlx-community/SenseVoiceSmall"

_LANGUAGE_MAP = {
    "auto": "auto",
    "cantonese": "yue",
    "yue": "yue",
    "chinese": "zh",
    "mandarin": "zh",
    "english": "en",
    "japanese": "ja",
    "korean": "ko",
}


def _rich_metadata(result: object) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    items = getattr(result, "segments", None)
    if isinstance(items, (list, tuple)) and items and isinstance(items[0], dict):
        first = items[0]
        metadata.update(
            {
                key: first.get(key)
                for key in ("language", "emotion", "event")
                if first.get(key) is not None
            }
        )
    if metadata.get("language") is None:
        language = getattr(result, "language", None)
        if language is not None:
            metadata["language"] = language
    return metadata


def _write_chunk(
    source: wave.Wave_read,
    destination: Path,
    *,
    frame_count: int,
) -> int:
    raw = source.readframes(frame_count)
    if not raw:
        return 0
    with wave.open(str(destination), "wb") as target:
        target.setnchannels(source.getnchannels())
        target.setsampwidth(source.getsampwidth())
        target.setframerate(source.getframerate())
        target.writeframes(raw)
    return len(raw) // (source.getnchannels() * source.getsampwidth())


class SenseVoiceBackend:
    name = BACKEND_NAME

    def __init__(
        self,
        *,
        model_id: str,
        model_path: Path,
        verbose: bool = False,
        chunk_seconds: float = 30.0,
    ) -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.verbose = verbose
        if not math.isfinite(chunk_seconds) or chunk_seconds <= 0:
            raise ValueError("SenseVoice chunk_seconds must be a positive finite number")
        self.chunk_seconds = chunk_seconds

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str,
        profile_text: str | None,
        progress_callback: ProgressCallback | None = None,
    ) -> BackendResult:
        language_code = _LANGUAGE_MAP.get(language.strip().lower())
        if language_code is None:
            raise BackendError(f"SenseVoice 唔支援 language value：{language}")

        warnings: list[str] = []
        if profile_text:
            warnings.append("SenseVoice adapter 不支援 context hotwords；已保留設定但冇注入 model")

        try:
            from mlx_audio.stt import load

            model = load(str(self.model_path))
        except Exception as exc:
            raise BackendError(
                f"SenseVoice model 載入失敗：{exc}",
                metadata={
                    "model_path": str(self.model_path),
                    "language": language_code,
                },
                warnings=tuple(warnings),
            ) from exc

        report_progress = isolate_progress_callback(progress_callback)

        segments: list[TranscriptSegment] = []
        chunk_metadata: list[dict[str, Any]] = []
        texts: list[str] = []
        try:
            source = wave.open(str(audio_path), "rb")
        except (OSError, wave.Error) as exc:
            raise MediaError(f"SenseVoice 無法讀取 normalized WAV：{exc}") from exc

        with source, tempfile.TemporaryDirectory(
            prefix="sensevoice-chunks-", dir=str(audio_path.parent)
        ) as temporary:
            sample_rate = source.getframerate()
            if source.getnchannels() != 1 or source.getsampwidth() != 2:
                raise MediaError("SenseVoice input 必須係 16-bit mono WAV")
            if sample_rate <= 0:
                raise MediaError("SenseVoice input WAV sample rate 無效")
            frames_per_chunk = max(1, int(sample_rate * self.chunk_seconds))
            total_duration = source.getnframes() / sample_rate if sample_rate else None
            if report_progress is not None:
                report_progress(
                    ProgressEvent(
                        phase=ProgressPhase.TRANSCRIBING,
                        current=0.0,
                        total=total_duration,
                        unit="seconds",
                        message="Transcribing...",
                    )
                )
            offset_frames = 0
            chunk_index = 0
            previous_start: float | None = None
            while True:
                chunk_path = Path(temporary) / f"chunk-{chunk_index:05d}.wav"
                written_frames = _write_chunk(
                    source, chunk_path, frame_count=frames_per_chunk
                )
                if written_frames == 0:
                    break
                start = offset_frames / sample_rate
                end = (offset_frames + written_frames) / sample_rate
                try:
                    result = model.generate(
                        str(chunk_path),
                        language=language_code,
                        use_itn=False,
                        verbose=self.verbose,
                    )
                except Exception as exc:
                    failure_metadata = {
                        "failed_chunk_index": chunk_index,
                        "completed_chunk_count": len(chunk_metadata),
                        "processed_audio_seconds": offset_frames / sample_rate,
                        "total_audio_seconds": total_duration,
                        "completed_chunks": list(chunk_metadata),
                        "language": language_code,
                    }
                    raise BackendError(
                        f"SenseVoice 第 {chunk_index + 1} 段轉錄失敗：{exc}",
                        metadata=failure_metadata,
                        warnings=tuple(warnings),
                    ) from exc
                text = str(getattr(result, "text", "") or "")
                rich = _rich_metadata(result)
                chunk_metadata.append(
                    {
                        "index": chunk_index,
                        "start": start,
                        "end": end,
                        **rich,
                    }
                )
                if (
                    not math.isfinite(start)
                    or not math.isfinite(end)
                    or start < 0
                    or end <= start
                    or (previous_start is not None and start < previous_start)
                ):
                    raise BackendError(
                        "SenseVoice 產生咗無效 chunk timing",
                        metadata={
                            "failed_chunk_index": chunk_index,
                            "completed_chunk_count": len(chunk_metadata) - 1,
                            "processed_audio_seconds": offset_frames / sample_rate,
                            "total_audio_seconds": total_duration,
                            "completed_chunks": list(chunk_metadata[:-1]),
                            "language": language_code,
                        },
                        warnings=tuple(warnings),
                    )
                previous_start = start
                if text.strip():
                    texts.append(text)
                    segments.append(
                        TranscriptSegment(
                            start=start,
                            end=end,
                            text=text,
                            timing_source="chunk",
                        )
                    )
                offset_frames += written_frames
                chunk_index += 1
                if report_progress is not None:
                    report_progress(
                        ProgressEvent(
                            phase=ProgressPhase.TRANSCRIBING,
                            current=offset_frames / sample_rate,
                            total=total_duration,
                            unit="seconds",
                            message=f"Transcribing chunk {chunk_index}...",
                        )
                    )

        detected_languages = [
            item["language"] for item in chunk_metadata if item.get("language")
        ]
        detected_language = (
            max(set(detected_languages), key=detected_languages.count)
            if detected_languages
            else language_code
        )
        warnings.append("SenseVoice 冇 word-level timestamp；JSON 內保留 chunk timing provenance")
        return BackendResult(
            language=str(detected_language),
            backend=self.name,
            model=self.model_id,
            text="\n".join(texts),
            segments=tuple(segments),
            metadata={
                "chunks": chunk_metadata,
                "chunk_count": len(chunk_metadata),
                "total_audio_seconds": total_duration,
                "timestamp_source": "chunk",
            },
            warnings=tuple(warnings),
        )
