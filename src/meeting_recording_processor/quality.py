"""Objective hard-failure checks; not a subjective transcript scorer."""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata
from typing import Any


@dataclass(frozen=True, slots=True)
class QualityReport:
    hard_failure: bool
    reasons: tuple[str, ...]
    non_whitespace_characters: int
    substantive_characters: int
    punctuation_or_symbol_ratio: float
    active_audio_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hard_failure": self.hard_failure,
            "reasons": list(self.reasons),
            "non_whitespace_characters": self.non_whitespace_characters,
            "substantive_characters": self.substantive_characters,
            "punctuation_or_symbol_ratio": self.punctuation_or_symbol_ratio,
            "active_audio_seconds": self.active_audio_seconds,
        }


def inspect_text(
    text: str,
    *,
    punctuation_threshold: float = 0.90,
    active_audio_seconds: float | None = None,
    active_audio_threshold: float = 10.0,
    minimum_substantive_characters: int = 3,
) -> QualityReport:
    """Detect objective failures without attempting subjective accuracy scoring."""

    characters = [character for character in text if not character.isspace()]
    total = len(characters)
    substantive = sum(
        1 for character in characters if unicodedata.category(character)[0] in {"L", "N"}
    )
    punctuation_or_symbols = sum(
        1 for character in characters if unicodedata.category(character)[0] in {"P", "S"}
    )
    ratio = punctuation_or_symbols / total if total else 0.0

    reasons: list[str] = []
    if total == 0:
        reasons.append("empty_output")
    if total > 0 and ratio > punctuation_threshold:
        reasons.append("punctuation_collapse")
    if (
        active_audio_seconds is not None
        and active_audio_seconds >= active_audio_threshold
        and substantive < minimum_substantive_characters
    ):
        reasons.append("near_zero_text_with_active_audio")

    return QualityReport(
        hard_failure=bool(reasons),
        reasons=tuple(reasons),
        non_whitespace_characters=total,
        substantive_characters=substantive,
        punctuation_or_symbol_ratio=ratio,
        active_audio_seconds=active_audio_seconds,
    )
