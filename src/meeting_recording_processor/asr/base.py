"""Backend-neutral ASR protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol

from ..progress import ProgressEvent
from ..schemas import BackendResult


ProgressCallback = Callable[[ProgressEvent], None]


class AsrBackend(Protocol):
    name: str
    model_id: str

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str,
        profile_text: str | None,
        progress_callback: ProgressCallback | None = None,
    ) -> BackendResult: ...
