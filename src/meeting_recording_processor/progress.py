"""Backend-neutral progress events and terminal renderers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
import math
import os
import sys
from threading import Event, RLock, Thread, current_thread
import time
from typing import Callable, TextIO

from .errors import ConfigurationError


class ProgressPhase(StrEnum):
    PREPARING = "preparing"
    LOADING_MODEL = "loading-model"
    TRANSCRIBING = "transcribing"
    WRITING_OUTPUT = "writing-output"
    COMPLETED = "completed"
    FAILED = "failed"


class ProgressMode(StrEnum):
    AUTO = "auto"
    ON = "on"
    OFF = "off"


Clock = Callable[[], float]

_PHASE_ORDER = {
    ProgressPhase.PREPARING.value: 0,
    ProgressPhase.LOADING_MODEL.value: 1,
    ProgressPhase.TRANSCRIBING.value: 2,
    ProgressPhase.WRITING_OUTPUT.value: 3,
    ProgressPhase.COMPLETED.value: 4,
    ProgressPhase.FAILED.value: 4,
}
_TERMINAL_PHASES = {
    ProgressPhase.COMPLETED.value,
    ProgressPhase.FAILED.value,
}


def resolve_progress_mode(value: str | ProgressMode | None = None) -> ProgressMode:
    """Resolve an explicit mode or ``ASR_PROGRESS`` without changing defaults."""

    raw = os.environ.get("ASR_PROGRESS", ProgressMode.AUTO.value) if value is None else value
    normalized = str(raw).strip().lower()
    if not normalized:
        normalized = ProgressMode.AUTO.value
    try:
        return ProgressMode(normalized)
    except ValueError as exc:
        allowed = ", ".join(mode.value for mode in ProgressMode)
        raise ConfigurationError(
            f"ASR_PROGRESS 必須係 {allowed} 其中一個；收到：{raw!r}"
        ) from exc


def format_duration(seconds: float | int | None) -> str:
    """Format seconds as ``MM:SS`` or ``H:MM:SS``.

    Unknown, negative, or non-finite values use the same placeholder as the
    upstream CLI rather than displaying a misleading duration.
    """

    if seconds is None:
        return "--:--"
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "--:--"
    if not math.isfinite(value) or value < 0:
        return "--:--"

    total = int(value)
    hours, remainder = divmod(total, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes:02d}:{remaining_seconds:02d}"


def progress_percentage(
    current: float | int | None,
    total: float | int | None,
) -> float | None:
    """Return a clamped percentage, or ``None`` when no reliable total exists."""

    if current is None or total is None:
        return None
    try:
        current_value = float(current)
        total_value = float(total)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(current_value) or not math.isfinite(total_value):
        return None
    if total_value <= 0:
        return None
    return min(max(current_value / total_value * 100.0, 0.0), 100.0)


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """A backend-independent snapshot of work completed so far."""

    phase: str
    current: float | None = None
    total: float | None = None
    unit: str = "seconds"
    elapsed: float = 0.0
    message: str = ""
    determinate: bool | None = None
    file_index: int | None = None
    file_total: int | None = None
    file_name: str | None = None

    def __post_init__(self) -> None:
        phase = self.phase.value if isinstance(self.phase, ProgressPhase) else str(self.phase)
        object.__setattr__(self, "phase", phase)
        if self.determinate is None:
            object.__setattr__(
                self,
                "determinate",
                progress_percentage(self.current, self.total) is not None,
            )
        elif self.determinate and progress_percentage(self.current, self.total) is None:
            object.__setattr__(self, "determinate", False)

    @property
    def percentage(self) -> float | None:
        if not self.determinate:
            return None
        return progress_percentage(self.current, self.total)


class ProgressRenderer:
    """Render events as a compact TTY line or plain log records."""

    def __init__(
        self,
        *,
        mode: str | ProgressMode | None = None,
        stream: TextIO | None = None,
        log_interval_seconds: float = 15.0,
        log_step_percentage: float = 10.0,
    ) -> None:
        self.mode = resolve_progress_mode(mode)
        self.stream = stream if stream is not None else sys.stderr
        self._disabled = False
        isatty = getattr(self.stream, "isatty", None)
        try:
            self.interactive = bool(isatty()) if callable(isatty) else False
        except Exception:
            self.interactive = False
            self._disabled = True
        self.log_interval_seconds = max(0.0, log_interval_seconds)
        self.log_step_percentage = max(0.0, log_step_percentage)
        self._last_phase: str | None = None
        self._last_percentage: float | None = None
        self._last_elapsed = 0.0
        self._last_message: str | None = None
        self._last_line_width = 0
        self._line_active = False
        self._closed = False

    @property
    def enabled(self) -> bool:
        return (
            self.mode is not ProgressMode.OFF
            and not self._disabled
            and not self._closed
        )

    def render(self, event: ProgressEvent) -> None:
        if not self.enabled or self._closed:
            return
        try:
            if not self._phase_is_allowed(event.phase):
                return
            self._render_now(event)
        except Exception:
            self.disable()

    def disable(self) -> None:
        """Permanently stop progress output after a renderer failure."""

        self._disabled = True
        self._line_active = False
        self._closed = True

    def _render_now(self, event: ProgressEvent) -> None:
        if not self.interactive and not self._should_log(event):
            return

        text = self._format_event(event)
        if self.interactive:
            self._write_tty(text, final=event.phase in {
                ProgressPhase.COMPLETED.value,
                ProgressPhase.FAILED.value,
            })
        else:
            self.stream.write(
                f"{self._file_prefix(event)}{self._plain_prefix(event)} "
                f"{self._format_plain_event(event)}\n"
            )
            self.stream.flush()
        self._last_phase = event.phase
        if event.percentage is not None:
            self._last_percentage = event.percentage
        elif event.phase != ProgressPhase.TRANSCRIBING.value:
            self._last_percentage = None
        self._last_elapsed = event.elapsed
        self._last_message = event.message

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self.interactive and self._line_active:
                self.stream.write("\n")
                self.stream.flush()
        except Exception:
            self.disable()
        finally:
            self._line_active = False
            self._closed = True

    def _should_log(self, event: ProgressEvent) -> bool:
        if self._last_phase != event.phase:
            return True
        if event.message != self._last_message:
            return True
        if event.phase in {ProgressPhase.COMPLETED.value, ProgressPhase.FAILED.value}:
            return True

        percentage = event.percentage
        if percentage is not None:
            if self._last_percentage is None:
                return True
            if percentage >= self._last_percentage + self.log_step_percentage:
                return True
            if percentage >= 100.0 and self._last_percentage < 100.0:
                return True
        return event.elapsed - self._last_elapsed >= self.log_interval_seconds

    def _phase_is_allowed(self, phase: str) -> bool:
        if self._last_phase is None:
            return True
        if self._last_phase in _TERMINAL_PHASES and phase != self._last_phase:
            return False
        previous_order = _PHASE_ORDER.get(self._last_phase)
        current_order = _PHASE_ORDER.get(phase)
        return (
            previous_order is None
            or current_order is None
            or current_order >= previous_order
        )

    def _write_tty(self, text: str, *, final: bool) -> None:
        padding = max(0, self._last_line_width - len(text))
        self.stream.write(f"\r{text}{' ' * padding}")
        if final:
            self.stream.write("\n")
            self._line_active = False
            self._last_line_width = 0
        else:
            self._line_active = True
            self._last_line_width = max(self._last_line_width, len(text))
        self.stream.flush()

    @staticmethod
    def _plain_prefix(event: ProgressEvent) -> str:
        phase_names = {
            ProgressPhase.TRANSCRIBING.value: "transcribe",
            ProgressPhase.LOADING_MODEL.value: "load-model",
            ProgressPhase.WRITING_OUTPUT.value: "write-output",
        }
        return f"[{phase_names.get(event.phase, event.phase)}]"

    @staticmethod
    def _file_prefix(event: ProgressEvent) -> str:
        if (
            event.file_index is None
            or event.file_total is None
            or event.file_name is None
        ):
            return ""
        return f"File {event.file_index} of {event.file_total}: {event.file_name} | "

    def _format_event(self, event: ProgressEvent) -> str:
        context = self._file_prefix(event)
        if event.phase == ProgressPhase.TRANSCRIBING.value:
            return context + self._format_transcribing(event)
        if event.phase == ProgressPhase.COMPLETED.value:
            return context + f"Completed in {format_duration(event.elapsed)}"
        if event.phase == ProgressPhase.FAILED.value:
            detail = f": {self._clean_message(event.message)}" if event.message else ""
            return context + f"Transcription failed after {format_duration(event.elapsed)}{detail}"

        message = self._clean_message(event.message) or event.phase.replace("-", " ").capitalize()
        return context + f"{message}  Elapsed: {format_duration(event.elapsed)}"

    def _format_plain_event(self, event: ProgressEvent) -> str:
        if event.phase == ProgressPhase.TRANSCRIBING.value:
            percentage = event.percentage
            if percentage is None:
                return (
                    "Transcribing... "
                    f"progress=indeterminate elapsed={format_duration(event.elapsed)}"
                )
            displayed_percentage = min(100, max(0, int(percentage + 0.5)))
            processed = (
                f" processed={format_duration(event.current)}/"
                f"{format_duration(event.total)}"
                if event.unit == "seconds"
                else ""
            )
            return (
                f"Transcribing progress={displayed_percentage}%{processed} "
                f"elapsed={format_duration(event.elapsed)}"
            )
        if event.phase == ProgressPhase.COMPLETED.value:
            return f"Completed elapsed={format_duration(event.elapsed)}"
        if event.phase == ProgressPhase.FAILED.value:
            detail = (
                f" message={self._clean_message(event.message)}"
                if event.message
                else ""
            )
            return f"Transcription failed elapsed={format_duration(event.elapsed)}{detail}"
        message = self._clean_message(event.message) or event.phase.replace("-", " ")
        return f"{message} elapsed={format_duration(event.elapsed)}"

    @staticmethod
    def _clean_message(message: str) -> str:
        return message.replace("\x1b", "").replace("\r", " ").replace("\n", " ")

    def _format_transcribing(self, event: ProgressEvent) -> str:
        percentage = event.percentage
        if percentage is None:
            return f"Transcribing... Elapsed: {format_duration(event.elapsed)}"

        displayed_percentage = min(100, max(0, int(percentage + 0.5)))
        bar_width = 20
        filled = int(round(displayed_percentage / 100 * bar_width))
        bar = "#" * filled + "-" * (bar_width - filled)
        progress_text = f"Transcribing [{bar}] {displayed_percentage}%"
        if event.unit == "seconds":
            progress_text += (
                f"  {format_duration(event.current)} / {format_duration(event.total)}"
            )
        elif event.current is not None and event.total is not None:
            progress_text += f"  {event.current:g} / {event.total:g} {event.unit}"
        return f"{progress_text}  Elapsed: {format_duration(event.elapsed)}"


class ProgressReporter:
    """Track elapsed time and add batch context before rendering events."""

    def __init__(
        self,
        *,
        mode: str | ProgressMode | None = None,
        stream: TextIO | None = None,
        clock: Clock = time.monotonic,
        file_index: int | None = None,
        file_total: int | None = None,
        file_name: str | None = None,
        heartbeat_seconds: float = 1.0,
    ) -> None:
        self.renderer = ProgressRenderer(mode=mode, stream=stream)
        self._clock = clock
        self._started_at: float | None = None
        self._finished = False
        self._lock = RLock()
        self._heartbeat_seconds = max(0.1, heartbeat_seconds)
        self._heartbeat_stop = Event()
        self._heartbeat_thread: Thread | None = None
        self._last_event: ProgressEvent | None = None
        self.file_index = file_index
        self.file_total = file_total
        self.file_name = file_name

    @property
    def enabled(self) -> bool:
        return self.renderer.enabled

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def current_phase(self) -> str | None:
        with self._lock:
            return self._last_event.phase if self._last_event is not None else None

    def start(self, message: str = "Preparing audio...") -> None:
        if self._finished:
            return
        if self._started_at is None:
            self._started_at = self._clock()
        self.emit_phase(ProgressPhase.PREPARING, message=message)
        if self.enabled:
            self._start_heartbeat()

    def emit_phase(
        self,
        phase: ProgressPhase | str,
        *,
        current: float | None = None,
        total: float | None = None,
        unit: str = "seconds",
        message: str = "",
        determinate: bool | None = None,
    ) -> None:
        self.emit(
            ProgressEvent(
                phase=phase,
                current=current,
                total=total,
                unit=unit,
                message=message,
                determinate=determinate,
            )
        )

    def emit(self, event: ProgressEvent) -> None:
        with self._lock:
            if self._finished:
                return
            if self._started_at is None:
                self._started_at = self._clock()
            enriched = replace(
                event,
                elapsed=max(0.0, self._clock() - self._started_at),
                file_index=event.file_index
                if event.file_index is not None
                else self.file_index,
                file_total=event.file_total
                if event.file_total is not None
                else self.file_total,
                file_name=event.file_name if event.file_name is not None else self.file_name,
            )
            if self._last_event is not None and not self._phase_is_allowed(
                self._last_event.phase, enriched.phase
            ):
                return
            self._last_event = enriched
            try:
                self.renderer.render(enriched)
            except Exception:
                self.renderer.disable()

    def complete(self, message: str = "") -> None:
        if self._finished:
            return
        self._stop_heartbeat()
        self.emit_phase(ProgressPhase.COMPLETED, message=message)
        self._close_renderer()
        self._finished = True

    def fail(self, message: str = "") -> None:
        if self._finished:
            return
        self._stop_heartbeat()
        self.emit_phase(ProgressPhase.FAILED, message=message)
        self._close_renderer()
        self._finished = True

    def close(self) -> None:
        if not self._finished:
            self._stop_heartbeat()
            self._close_renderer()
            self._finished = True

    def _close_renderer(self) -> None:
        try:
            self.renderer.close()
        except Exception:
            self.renderer.disable()

    def _start_heartbeat(self) -> None:
        if self._heartbeat_thread is not None:
            return
        self._heartbeat_thread = Thread(
            target=self._heartbeat_loop,
            name="asr-progress-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        thread = self._heartbeat_thread
        if thread is not None and thread is not current_thread():
            thread.join(timeout=self._heartbeat_seconds + 0.5)
        self._heartbeat_thread = None

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(self._heartbeat_seconds):
            if not self.renderer.enabled:
                return
            self._heartbeat_tick()

    def _heartbeat_tick(self) -> None:
        """Render the current event while holding the state lock."""

        with self._lock:
            if (
                self._finished
                or self._last_event is None
                or not self.renderer.enabled
            ):
                return
            started_at = self._started_at
            enriched = replace(
                self._last_event,
                elapsed=max(
                    0.0,
                    self._clock() - (started_at if started_at is not None else self._clock()),
                ),
            )
            self._last_event = enriched
            try:
                self.renderer.render(enriched)
            except Exception:
                self.renderer.disable()

    @staticmethod
    def _phase_is_allowed(previous: str, current: str) -> bool:
        if previous in _TERMINAL_PHASES and current != previous:
            return False
        previous_order = _PHASE_ORDER.get(previous)
        current_order = _PHASE_ORDER.get(current)
        return (
            previous_order is None
            or current_order is None
            or current_order >= previous_order
        )
