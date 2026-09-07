"""Configuration types shared by the CLI and pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class AsrMode(StrEnum):
    QWEN3 = "qwen3"
    SENSEVOICE = "sensevoice"
    AUTO = "auto"


SUPPORTED_EXTENSIONS = frozenset({".wav", ".m4a", ".mp3", ".flac", ".mp4", ".mov"})
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_SENSEVOICE_MODEL = "mlx-community/SenseVoiceSmall"


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
    sensevoice_model: str = DEFAULT_SENSEVOICE_MODEL
    keep_work_files: bool = False
    overwrite: bool = False
    verbose: bool = False
    progress_mode: str | None = None

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
