"""Configuration types shared by the CLI and pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .errors import ConfigurationError


class AsrMode(StrEnum):
    QWEN3 = "qwen3"
    SENSEVOICE = "sensevoice"
    VIBEVOICE = "vibevoice"
    AUTO = "auto"


SUPPORTED_EXTENSIONS = frozenset({".wav", ".m4a", ".mp3", ".flac", ".mp4", ".mov"})
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_QWEN_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
DEFAULT_SENSEVOICE_MODEL = "mlx-community/SenseVoiceSmall"
DEFAULT_VIBEVOICE_MODEL = "microsoft/VibeVoice-ASR-HF"
DEFAULT_VIBEVOICE_ACOUSTIC_CHUNK_SIZE = 64_000
VIBEVOICE_ACOUSTIC_TOKENIZER_HOP_SIZE = 3_200


def validate_vibevoice_acoustic_chunk_size(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(
            "VibeVoice acoustic tokenizer chunk size must be an integer"
        )
    if value <= 0:
        raise ConfigurationError(
            "VibeVoice acoustic tokenizer chunk size must be greater than zero"
        )
    if value % VIBEVOICE_ACOUSTIC_TOKENIZER_HOP_SIZE:
        raise ConfigurationError(
            "VibeVoice acoustic tokenizer chunk size must be a multiple of "
            f"{VIBEVOICE_ACOUSTIC_TOKENIZER_HOP_SIZE} samples"
        )
    return value


def project_root(start: Path | None = None) -> Path:
    """Return the nearest repository root, falling back to the current directory."""

    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file() and "meeting-recording-processor" in pyproject.read_text(
            encoding="utf-8", errors="ignore"
        ):
            return candidate
    return current


@dataclass(frozen=True, slots=True)
class ExtractConfig:
    input_path: Path
    output_dir: Path
    work_dir: Path
    cache_dir: Path
    asr_mode: AsrMode = AsrMode.AUTO
    language: str = "Cantonese"
    context_file: Path | None = None
    qwen_model: str = DEFAULT_QWEN_MODEL
    qwen_aligner_model: str = DEFAULT_QWEN_ALIGNER_MODEL
    sensevoice_model: str = DEFAULT_SENSEVOICE_MODEL
    vibevoice_model: str = DEFAULT_VIBEVOICE_MODEL
    vibevoice_acoustic_chunk_size: int = DEFAULT_VIBEVOICE_ACOUSTIC_CHUNK_SIZE
    keep_work_files: bool = False
    overwrite: bool = False
    verbose: bool = False
    progress_mode: str | None = None

    def __post_init__(self) -> None:
        validate_vibevoice_acoustic_chunk_size(self.vibevoice_acoustic_chunk_size)

    @property
    def output_path(self) -> Path:
        return self.output_dir / f"{self.input_path.stem}.transcript.json"


@dataclass(frozen=True, slots=True)
class ExportConfig:
    transcript_path: Path
    output_dir: Path | None = None
    overwrite: bool = False

    @property
    def base_name(self) -> str:
        name = self.transcript_path.name
        suffix = ".transcript.json"
        return name[: -len(suffix)] if name.endswith(suffix) else self.transcript_path.stem

    @property
    def resolved_output_dir(self) -> Path:
        return self.output_dir or self.transcript_path.parent
