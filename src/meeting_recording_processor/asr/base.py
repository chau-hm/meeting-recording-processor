"""Backend-neutral ASR protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..schemas import BackendResult


class AsrBackend(Protocol):
    name: str
    model_id: str

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str,
        profile_text: str | None,
    ) -> BackendResult: ...
