"""Backend-neutral ASR protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol

from ..progress import ProgressEvent
from ..schemas import BackendResult


ProgressCallback = Callable[[ProgressEvent], None]


def isolate_progress_callback(
    callback: ProgressCallback | None,
) -> ProgressCallback | None:
    """Keep non-critical callback failures outside the inference boundary."""

    if callback is None:
        return None

    enabled = True

    def report(event: ProgressEvent) -> None:
        nonlocal enabled
        if not enabled:
            return
        try:
            callback(event)
        except Exception:
            enabled = False

    return report


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
