"""Offline extract orchestration with objective Qwen3 → SenseVoice fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any, Callable

from .asr import Qwen3Backend, SenseVoiceBackend
from .config import AsrMode, ExtractConfig, SUPPORTED_EXTENSIONS
from .errors import ConfigurationError, OutputExistsError, TranscriptionFailed
from .manifest import new_run_id, sha256_file, tool_metadata, utc_now_iso
from .media.normalize import normalize_audio
from .media.probe import MediaMetadata, ffprobe_version, probe_media
from .media.signal import AudioSignalStats, analyze_wav_signal
from .models import ResolvedModel, resolve_cached_model
from .postprocess import postprocess_result
from .progress import ProgressPhase, ProgressReporter, format_duration
from .quality import inspect_text
from .runtime import require_apple_silicon, require_media_commands
from .schema_io import SCHEMA_VERSION, write_package
from .schemas import AttemptRecord, BackendResult


@dataclass(frozen=True, slots=True)
class ExtractResult:
    output_path: Path
    selected_backend: str | None
    fallback_used: bool
    warnings: tuple[str, ...]


BackendFactory = Callable[[str, str, Path, bool], Any]
ModelResolver = Callable[[str, Path], ResolvedModel]
ProgressFactory = Callable[[ExtractConfig], ProgressReporter]


def default_progress_factory(config: ExtractConfig) -> ProgressReporter:
    return ProgressReporter(mode=config.progress_mode)


def default_backend_factory(
    backend: str, model_id: str, model_path: Path, verbose: bool
):
    if backend == AsrMode.QWEN3.value:
        return Qwen3Backend(model_id=model_id, model_path=model_path, verbose=verbose)
    if backend == AsrMode.SENSEVOICE.value:
        return SenseVoiceBackend(model_id=model_id, model_path=model_path, verbose=verbose)
    raise ConfigurationError(f"未知 ASR backend：{backend}")


class Extractor:
    """Injectable orchestrator so tests never need MLX models or commands."""

    def __init__(
        self,
        *,
        platform_validator: Callable[[], None] = require_apple_silicon,
        media_commands_validator: Callable[[], None] = require_media_commands,
        probe: Callable[[Path], MediaMetadata] = probe_media,
        normalizer: Callable[[Path, MediaMetadata, Path], Path] = normalize_audio,
        signal_analyzer: Callable[[Path], AudioSignalStats] = analyze_wav_signal,
        model_resolver: ModelResolver = resolve_cached_model,
        backend_factory: BackendFactory = default_backend_factory,
        postprocessor: Callable[..., tuple[str, tuple, tuple]] = postprocess_result,
        file_hasher: Callable[[Path], str] = sha256_file,
        probe_version: Callable[[], str | None] = ffprobe_version,
        progress_factory: ProgressFactory = default_progress_factory,
    ) -> None:
        self.platform_validator = platform_validator
        self.media_commands_validator = media_commands_validator
        self.probe = probe
        self.normalizer = normalizer
        self.signal_analyzer = signal_analyzer
        self.model_resolver = model_resolver
        self.backend_factory = backend_factory
        self.postprocessor = postprocessor
        self.file_hasher = file_hasher
        self.probe_version = probe_version
        self.progress_factory = progress_factory

    def extract(self, config: ExtractConfig) -> ExtractResult:
        input_path = config.input_path.expanduser().resolve()
        output_path = config.output_path.expanduser().resolve()
        work_root = config.work_dir.expanduser().resolve()
        cache_dir = config.cache_dir.expanduser().resolve()
        context_file = config.context_file.expanduser().resolve() if config.context_file else None

        self._validate_request(input_path, output_path, context_file, config.overwrite)
        progress = self.progress_factory(config)
        work_path: Path | None = None
        try:
            progress.start(message=f"Preparing {input_path.name}...")
            self.platform_validator()
            self.media_commands_validator()

            progress.emit_phase(
                ProgressPhase.PREPARING,
                message="Inspecting media...",
            )
            media = self.probe(input_path)
            run_id = new_run_id()
            work_path = work_root / f"{input_path.stem}-{run_id}"
            work_path.mkdir(parents=True, exist_ok=False)
            normalized_path = work_path / "audio-16k-mono.wav"

            attempts: list[AttemptRecord] = []
            selected_result: BackendResult | None = None
            selected_attempt_id: str | None = None
            pipeline_error: str | None = None
            all_warnings: list[str] = []

            self.normalizer(input_path, media, normalized_path)
            signal = self.signal_analyzer(normalized_path)
            duration = media.duration_seconds
            if duration is None and signal.duration_seconds > 0:
                duration = signal.duration_seconds
            progress.emit_phase(
                ProgressPhase.PREPARING,
                current=0.0,
                total=duration,
                message=(
                    f"Audio ready; duration {format_duration(duration)}."
                    if duration is not None
                    else "Audio ready; duration unknown."
                ),
            )
            profile_text = self._read_context(context_file)
            sequence = self._backend_sequence(config.asr_mode)

            for index, backend_name in enumerate(sequence, start=1):
                model_id = self._model_id(config, backend_name)
                attempt_id = f"attempt-{index}-{backend_name}"
                started_at = utc_now_iso()
                started = perf_counter()
                resolved: ResolvedModel | None = None
                backend_result: BackendResult | None = None
                error: str | None = None
                try:
                    if index == 1:
                        progress.emit_phase(
                            ProgressPhase.LOADING_MODEL,
                            message=f"Loading {backend_name} model...",
                        )
                    resolved = self.model_resolver(model_id, cache_dir)
                    backend = self.backend_factory(
                        backend_name, model_id, resolved.path, config.verbose
                    )
                    backend_result = backend.transcribe(
                        normalized_path,
                        language=config.language,
                        profile_text=profile_text,
                        progress_callback=progress.emit if progress.enabled else None,
                    )
                    quality_report = inspect_text(
                        backend_result.text,
                        active_audio_seconds=signal.active_audio_seconds,
                    )
                    quality = quality_report.to_dict()
                    attempt_status = "hard_failure" if quality_report.hard_failure else "completed"
                except Exception as exc:
                    error = str(exc)
                    quality = inspect_text(
                        "", active_audio_seconds=signal.active_audio_seconds
                    ).to_dict()
                    quality["reasons"] = ["backend_error"]
                    quality["hard_failure"] = True
                    attempt_status = "error"

                runtime_seconds = perf_counter() - started
                completed_at = utc_now_iso()
                attempt = AttemptRecord(
                    attempt_id=attempt_id,
                    backend=backend_name,
                    model=model_id,
                    model_snapshot=resolved.snapshot if resolved else None,
                    status=attempt_status,
                    started_at=started_at,
                    completed_at=completed_at,
                    runtime_seconds=round(runtime_seconds, 3),
                    raw_text=backend_result.text if backend_result else "",
                    raw_segments=backend_result.segments if backend_result else (),
                    quality=quality,
                    metadata=backend_result.metadata if backend_result else {},
                    warnings=backend_result.warnings if backend_result else (),
                    error=error,
                )
                attempts.append(attempt)
                all_warnings.extend(attempt.warnings)

                if backend_result is not None and not quality["hard_failure"]:
                    selected_result = backend_result
                    selected_attempt_id = attempt_id
                    break
                if config.asr_mode is not AsrMode.AUTO:
                    break
                if index < len(sequence):
                    fallback_reason = (
                        "failed objective quality gate"
                        if backend_result is not None
                        else "failed"
                    )
                    progress.emit_phase(
                        ProgressPhase.FALLBACK,
                        message=(
                            f"{backend_name} {fallback_reason}; "
                            f"falling back to {sequence[index]} and loading model..."
                        ),
                    )

            transcript_payload: dict[str, Any] | None = None
            if selected_result is not None:
                try:
                    text, segments, post_warnings = self.postprocessor(
                        selected_result,
                        audio_duration=signal.duration_seconds,
                    )
                    if not text or not segments:
                        raise ConfigurationError("post-processing 冇產生有效 transcript／segments")
                    all_warnings.extend(post_warnings)
                    transcript_payload = {
                        "language": selected_result.language,
                        "text": text,
                        "segments": [segment.to_dict() for segment in segments],
                    }
                except Exception as exc:
                    pipeline_error = f"post-processing 失敗：{exc}"
                    selected_result = None
                    selected_attempt_id = None

            if selected_result is None and pipeline_error is None:
                pipeline_error = "all_asr_attempts_failed"

            status = "completed" if selected_result is not None else "failed"
            progress.emit_phase(
                ProgressPhase.WRITING_OUTPUT,
                message="Writing transcript files...",
            )
            package = self._build_package(
                status=status,
                run_id=run_id,
                input_path=input_path,
                output_path=output_path,
                media=media,
                signal=signal,
                config=config,
                context_file=context_file,
                attempts=attempts,
                selected_attempt_id=selected_attempt_id,
                transcript=transcript_payload,
                warnings=tuple(dict.fromkeys(all_warnings)),
                error=pipeline_error,
                kept_work_path=work_path if config.keep_work_files else None,
            )
            write_package(output_path, package, overwrite=config.overwrite)

            if status != "completed":
                failure_detail = pipeline_error or next(
                    (
                        attempt.error
                        for attempt in reversed(attempts)
                        if attempt.error
                    ),
                    "all ASR attempts failed",
                )
                progress.fail(failure_detail)
                raise TranscriptionFailed(
                    "所有可用 ASR attempt 都未能產生有效 transcript",
                    diagnostic_path=output_path,
                )
            progress.complete()
            return ExtractResult(
                output_path=output_path,
                selected_backend=selected_result.backend,
                fallback_used=(
                    config.asr_mode is AsrMode.AUTO
                    and selected_result.backend == AsrMode.SENSEVOICE.value
                ),
                warnings=tuple(dict.fromkeys(all_warnings)),
            )
        except KeyboardInterrupt:
            progress.fail("Interrupted")
            raise
        except Exception as exc:
            if not progress.finished:
                progress.fail(str(exc))
            raise
        finally:
            if not config.keep_work_files and work_path is not None and work_path.exists():
                shutil.rmtree(work_path)

    @staticmethod
    def _validate_request(
        input_path: Path,
        output_path: Path,
        context_file: Path | None,
        overwrite: bool,
    ) -> None:
        if not input_path.is_file():
            raise ConfigurationError(f"搵唔到 input file：{input_path}")
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ConfigurationError(f"不支援 {input_path.suffix}；支援格式：{supported}")
        if context_file is not None and not context_file.is_file():
            raise ConfigurationError(f"搵唔到 context file：{context_file}")
        if output_path.exists() and not overwrite:
            raise OutputExistsError(f"輸出已存在：{output_path}；如要取代請加 --overwrite")

    @staticmethod
    def _read_context(path: Path | None) -> str | None:
        if path is None:
            return None
        content = path.read_text(encoding="utf-8").strip()
        return content or None

    @staticmethod
    def _backend_sequence(mode: AsrMode) -> tuple[str, ...]:
        if mode is AsrMode.AUTO:
            return (AsrMode.QWEN3.value, AsrMode.SENSEVOICE.value)
        return (mode.value,)

    @staticmethod
    def _model_id(config: ExtractConfig, backend: str) -> str:
        if backend == AsrMode.QWEN3.value:
            return config.qwen_model
        if backend == AsrMode.SENSEVOICE.value:
            return config.sensevoice_model
        raise ConfigurationError(f"未知 backend：{backend}")

    def _build_package(
        self,
        *,
        status: str,
        run_id: str,
        input_path: Path,
        output_path: Path,
        media: MediaMetadata,
        signal: AudioSignalStats,
        config: ExtractConfig,
        context_file: Path | None,
        attempts: list[AttemptRecord],
        selected_attempt_id: str | None,
        transcript: dict[str, Any] | None,
        warnings: tuple[str, ...],
        error: str | None,
        kept_work_path: Path | None,
    ) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "source": {
                "path": str(input_path),
                "name": input_path.name,
                "size_bytes": input_path.stat().st_size,
                "sha256": self.file_hasher(input_path),
                "media": media.to_dict(),
                "signal": signal.to_dict(),
            },
            "request": {
                "asr": config.asr_mode.value,
                "language": config.language,
                "context_file": str(context_file) if context_file else None,
                "context_sha256": self.file_hasher(context_file) if context_file else None,
                "models": {
                    "qwen3": config.qwen_model,
                    "sensevoice": config.sensevoice_model,
                },
                "offline": True,
            },
            "attempts": [attempt.to_dict() for attempt in attempts],
            "selected_attempt_id": selected_attempt_id,
            "transcript": transcript,
            "processing": {
                "audio": "16 kHz mono PCM WAV",
                "steps": [
                    "Unicode NFC",
                    "whitespace/control-character cleanup",
                    "OpenCC s2hk Traditional Chinese conversion",
                    "subtitle cue grouping without translation or rewriting",
                ],
                "kept_work_path": str(kept_work_path) if kept_work_path else None,
            },
            "warnings": list(warnings),
            "error": error,
            "output": str(output_path),
            "tool": tool_metadata(ffprobe=self.probe_version()),
        }


def extract(config: ExtractConfig) -> ExtractResult:
    return Extractor().extract(config)
