"""VibeVoice-ASR adapter using the native Transformers implementation."""

from __future__ import annotations

from contextlib import nullcontext
import importlib
import math
from pathlib import Path
from typing import Any
import wave

from ..errors import BackendError
from ..progress import ProgressEvent, ProgressPhase
from ..schemas import BackendResult, TranscriptSegment
from .base import ProgressCallback, isolate_progress_callback

BACKEND_NAME = "vibevoice"
DEFAULT_MODEL_ID = "microsoft/VibeVoice-ASR-HF"
RUNTIME_PACKAGE = "transformers>=5.3.0,<5.4.0"
TARGET_SAMPLE_RATE = 24_000
MAX_DURATION_SECONDS = 60 * 60

_SPEAKER_METADATA_WARNING = (
    "VibeVoice speaker attribution is preserved in attempt metadata; "
    "it is not promoted to the canonical transcript schema"
)
_LANGUAGE_METADATA_WARNING = (
    "VibeVoice is multilingual and does not use --language for conditioning; "
    "the requested value is retained as metadata only"
)


def _wav_duration(audio_path: Path) -> float | None:
    try:
        with wave.open(str(audio_path), "rb") as source:
            sample_rate = source.getframerate()
            frame_count = source.getnframes()
    except (OSError, wave.Error) as exc:
        raise BackendError(f"VibeVoice 無法讀取 normalized WAV：{exc}") from exc
    if sample_rate <= 0:
        return None
    duration = frame_count / sample_rate
    return duration if math.isfinite(duration) and duration >= 0 else None


def _finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first_text(value: object) -> str:
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        value = value[0]
    return value if isinstance(value, str) else str(value or "")


def _first_parsed(value: object) -> object:
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        if isinstance(value[0], dict):
            return value
        return value[0]
    return value


def _snapshot_from_path(path: Path) -> str | None:
    return path.name if path.parent.name == "snapshots" else None


def _parsed_segments(
    parsed: object,
) -> tuple[tuple[TranscriptSegment, ...], list[dict[str, object]]]:
    if not isinstance(parsed, list):
        return (), []

    segments: list[TranscriptSegment] = []
    structured: list[dict[str, object]] = []
    previous_start = -1.0
    for item in parsed:
        if not isinstance(item, dict):
            continue
        text = item.get("Content")
        start = _finite_number(item.get("Start"))
        end = _finite_number(item.get("End"))
        if (
            not isinstance(text, str)
            or not text.strip()
            or start is None
            or end is None
            or start < 0
            or end <= start
            or start < previous_start
        ):
            continue
        speaker = item.get("Speaker")
        if not isinstance(speaker, (str, int, float, bool)) and speaker is not None:
            speaker = str(speaker)
        segment = TranscriptSegment(
            start=start,
            end=end,
            text=text,
            timing_source="model",
        )
        segments.append(segment)
        structured.append(
            {
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": text,
            }
        )
        previous_start = start
    return tuple(segments), structured


def _mps_available(torch_module: Any) -> bool:
    try:
        return bool(torch_module.backends.mps.is_available())
    except (AttributeError, RuntimeError):
        return False


class VibeVoiceBackend:
    name = BACKEND_NAME

    def __init__(
        self,
        *,
        model_id: str,
        model_path: Path,
        verbose: bool = False,
    ) -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.verbose = verbose

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str,
        profile_text: str | None,
        progress_callback: ProgressCallback | None = None,
    ) -> BackendResult:
        duration = _wav_duration(audio_path)
        if duration is not None and duration > MAX_DURATION_SECONDS:
            raise BackendError(
                "VibeVoice 目前只支援單次最多 60 分鐘音訊；"
                f"normalized WAV 長度為 {duration / 60:.1f} 分鐘"
            )

        try:
            torch_module = importlib.import_module("torch")
        except ImportError as exc:
            raise BackendError("VibeVoice 需要 PyTorch；請重新執行 uv sync") from exc
        if not _mps_available(torch_module):
            raise BackendError("VibeVoice 需要可用嘅 Apple Silicon MPS backend")

        try:
            transformers = importlib.import_module("transformers")
            processor_class = getattr(transformers, "AutoProcessor")
            model_class = getattr(transformers, "VibeVoiceAsrForConditionalGeneration")
        except (ImportError, AttributeError) as exc:
            raise BackendError(
                "VibeVoice 需要 transformers>=5.3.0,<5.4.0 原生 ASR API"
            ) from exc

        try:
            processor = processor_class.from_pretrained(
                str(self.model_path),
                local_files_only=True,
            )
            model = model_class.from_pretrained(
                str(self.model_path),
                local_files_only=True,
            )
            model = model.to("mps")
            model.eval()
        except Exception as exc:
            raise BackendError(f"VibeVoice model 載入失敗：{exc}") from exc

        device = getattr(model, "device", "mps")
        if not str(device).startswith("mps"):
            raise BackendError(f"VibeVoice 未能使用 MPS device：{device}")
        dtype = getattr(model, "dtype", None)

        report_progress = isolate_progress_callback(progress_callback)
        if report_progress is not None:
            report_progress(
                ProgressEvent(
                    phase=ProgressPhase.TRANSCRIBING,
                    message="Transcribing...",
                    determinate=False,
                )
            )

        try:
            inputs = processor.apply_transcription_request(
                audio=str(audio_path),
                prompt=profile_text or None,
            )
            if dtype is None:
                inputs = inputs.to(device)
            else:
                inputs = inputs.to(device, dtype)
            input_ids = inputs["input_ids"]
            input_length = int(input_ids.shape[1])
            inference_mode = getattr(torch_module, "inference_mode", None)
            context = inference_mode() if callable(inference_mode) else nullcontext()
            with context:
                output_ids = model.generate(**inputs)
            generated_ids = output_ids[:, input_length:]
        except Exception as exc:
            raise BackendError(f"VibeVoice transcription 失敗：{exc}") from exc

        try:
            raw_output = _first_text(processor.decode(generated_ids))
        except Exception as exc:
            raise BackendError(f"VibeVoice raw output decode 失敗：{exc}") from exc

        warnings = [_SPEAKER_METADATA_WARNING, _LANGUAGE_METADATA_WARNING]
        try:
            parsed = _first_parsed(
                processor.decode(generated_ids, return_format="parsed")
            )
        except Exception as exc:
            parsed = None
            warnings.append(f"VibeVoice structured output parse 失敗：{exc}")

        segments, structured_segments = _parsed_segments(parsed)
        if not segments:
            warnings.append(
                "VibeVoice structured output 冇提供可用 model timestamps；"
                "post-processing 會按 project contract 處理"
            )

        try:
            text = _first_text(
                processor.decode(generated_ids, return_format="transcription_only")
            )
        except Exception as exc:
            raise BackendError(f"VibeVoice transcription-only decode 失敗：{exc}") from exc

        metadata = {
            "device": str(device),
            "dtype": str(dtype) if dtype is not None else None,
            "torch_version": getattr(torch_module, "__version__", None),
            "transformers_version": getattr(transformers, "__version__", None),
            "model_id": self.model_id,
            "model_snapshot": _snapshot_from_path(self.model_path),
            "context_provided": bool(profile_text),
            "language_mode": "metadata_only",
            "requested_language": language,
            "raw_output": raw_output,
            "structured_segments": structured_segments,
        }
        return BackendResult(
            language=language,
            backend=self.name,
            model=self.model_id,
            text=text,
            segments=segments,
            metadata=metadata,
            warnings=tuple(warnings),
        )
