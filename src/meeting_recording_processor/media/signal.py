"""Lightweight objective signal statistics for hard-failure detection."""

from __future__ import annotations

from array import array
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import sys
from typing import Any
import wave

from ..errors import MediaError


@dataclass(frozen=True, slots=True)
class AudioSignalStats:
    duration_seconds: float
    rms: float
    peak: float
    active_audio_seconds: float
    active_ratio: float
    activity_threshold: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_wav_signal(
    path: Path,
    *,
    frame_seconds: float = 0.10,
    activity_threshold: float = 0.01,
) -> AudioSignalStats:
    try:
        stream = wave.open(str(path), "rb")
    except (OSError, wave.Error) as exc:
        raise MediaError(f"無法分析 normalized WAV：{exc}") from exc

    with stream:
        channels = stream.getnchannels()
        sample_width = stream.getsampwidth()
        sample_rate = stream.getframerate()
        frame_count = stream.getnframes()
        if channels != 1 or sample_width != 2 or sample_rate <= 0:
            raise MediaError("normalized WAV 必須係 16-bit mono PCM")

        samples_per_window = max(1, int(sample_rate * frame_seconds))
        total_samples = 0
        sum_squares = 0.0
        peak_sample = 0
        active_samples = 0

        while True:
            raw = stream.readframes(samples_per_window)
            if not raw:
                break
            samples = array("h")
            samples.frombytes(raw)
            if sys.byteorder != "little":
                samples.byteswap()
            if not samples:
                continue
            window_squares = sum(sample * sample for sample in samples)
            window_rms = math.sqrt(window_squares / len(samples)) / 32768.0
            if window_rms >= activity_threshold:
                active_samples += len(samples)
            total_samples += len(samples)
            sum_squares += window_squares
            peak_sample = max(peak_sample, max(abs(sample) for sample in samples))

    duration = frame_count / sample_rate
    rms = math.sqrt(sum_squares / total_samples) / 32768.0 if total_samples else 0.0
    active_seconds = active_samples / sample_rate
    return AudioSignalStats(
        duration_seconds=duration,
        rms=rms,
        peak=peak_sample / 32768.0,
        active_audio_seconds=active_seconds,
        active_ratio=(active_seconds / duration) if duration else 0.0,
        activity_threshold=activity_threshold,
    )
