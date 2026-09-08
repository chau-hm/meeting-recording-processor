"""Canonical, backend-neutral transcript types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


MIN_SEGMENT_DURATION = 0.001


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    timing_source: str = "model"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RawTranscriptSegment:
    """Backend timing preserved for diagnostics without canonical invariants."""

    start: float
    end: float
    text: str
    timing_source: str = "model"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BackendResult:
    language: str
    backend: str
    model: str
    text: str
    segments: tuple[TranscriptSegment, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    raw_segments: tuple[RawTranscriptSegment, ...] | None = None


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    attempt_id: str
    backend: str
    model: str
    status: str
    started_at: str
    completed_at: str
    runtime_seconds: float
    model_snapshot: str | None = None
    raw_text: str = ""
    raw_segments: tuple[RawTranscriptSegment, ...] = field(default_factory=tuple)
    quality: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["raw_segments"] = [segment.to_dict() for segment in self.raw_segments]
        return result
