from io import StringIO
import unittest
from unittest.mock import patch

from meeting_recording_processor.asr.base import isolate_progress_callback
from meeting_recording_processor.asr.qwen3 import _progress_event
from meeting_recording_processor.progress import (
    ProgressEvent,
    ProgressMode,
    ProgressPhase,
    ProgressRenderer,
    ProgressReporter,
    format_duration,
    progress_percentage,
)


class TtyStringIO(StringIO):
    def isatty(self) -> bool:
        return True


class BrokenStream:
    def __init__(self) -> None:
        self.write_calls = 0

    def isatty(self) -> bool:
        return False

    def write(self, _value: str) -> int:
        self.write_calls += 1
        raise BrokenPipeError("closed stream")

    def flush(self) -> None:
        raise BrokenPipeError("closed stream")


class ProgressTests(unittest.TestCase):
    def test_format_duration(self) -> None:
        self.assertEqual(format_duration(47), "00:47")
        self.assertEqual(format_duration(751), "12:31")
        self.assertEqual(format_duration(3822), "1:03:42")
        self.assertEqual(format_duration(None), "--:--")

    def test_percentage_is_clamped(self) -> None:
        self.assertEqual(progress_percentage(-1, 10), 0.0)
        self.assertEqual(progress_percentage(15, 10), 100.0)
        self.assertIsNone(progress_percentage(1, None))
        self.assertIsNone(progress_percentage(1, 0))

    def test_unknown_total_is_indeterminate(self) -> None:
        event = ProgressEvent(
            phase=ProgressPhase.TRANSCRIBING,
            current=12,
            total=None,
            elapsed=5,
        )
        self.assertFalse(event.determinate)
        self.assertIsNone(event.percentage)

        output = StringIO()
        ProgressRenderer(mode=ProgressMode.ON, stream=output).render(event)
        self.assertIn("Transcribing...", output.getvalue())
        self.assertNotIn("%", output.getvalue())

    def test_tty_and_plain_rendering_use_different_line_styles(self) -> None:
        event = ProgressEvent(
            phase=ProgressPhase.TRANSCRIBING,
            current=5,
            total=10,
            elapsed=3,
        )
        tty_output = TtyStringIO()
        tty_renderer = ProgressRenderer(mode="on", stream=tty_output)
        tty_renderer.render(event)
        tty_renderer.render(
            ProgressEvent(phase=ProgressPhase.COMPLETED, elapsed=4)
        )
        self.assertIn("\r", tty_output.getvalue())

        plain_output = StringIO()
        plain_renderer = ProgressRenderer(mode="on", stream=plain_output)
        plain_renderer.render(event)
        plain_renderer.render(
            ProgressEvent(phase=ProgressPhase.COMPLETED, elapsed=4)
        )
        self.assertNotIn("\r", plain_output.getvalue())
        self.assertNotIn("\x1b", plain_output.getvalue())

    def test_disabled_mode_is_silent(self) -> None:
        output = StringIO()
        reporter = ProgressReporter(mode=ProgressMode.OFF, stream=output)
        reporter.start()
        reporter.complete()
        self.assertEqual(output.getvalue(), "")

    def test_renderer_disables_after_first_stream_failure(self) -> None:
        stream = BrokenStream()
        renderer = ProgressRenderer(mode="on", stream=stream)
        event = ProgressEvent(phase=ProgressPhase.PREPARING, message="Preparing")
        renderer.render(event)
        renderer.render(event)
        self.assertFalse(renderer.enabled)
        self.assertEqual(stream.write_calls, 1)

    def test_progress_callback_preserves_keyboard_interrupt(self) -> None:
        def interrupt(_event: ProgressEvent) -> None:
            raise KeyboardInterrupt

        callback = isolate_progress_callback(interrupt)
        assert callback is not None
        with self.assertRaises(KeyboardInterrupt):
            callback(ProgressEvent(phase=ProgressPhase.PREPARING))

    def test_failure_does_not_report_completion(self) -> None:
        output = StringIO()
        reporter = ProgressReporter(mode="on", stream=output)
        reporter.start()
        reporter.fail("backend error")
        rendered = output.getvalue()
        self.assertIn("Transcription failed", rendered)
        self.assertNotIn("Completed", rendered)

    def test_final_percentage_is_rendered_before_failure(self) -> None:
        output = StringIO()
        renderer = ProgressRenderer(mode="on", stream=output)
        renderer.render(
            ProgressEvent(
                phase=ProgressPhase.TRANSCRIBING,
                current=10,
                total=10,
            )
        )
        renderer.render(
            ProgressEvent(phase=ProgressPhase.FAILED, message="backend error")
        )
        rendered = output.getvalue()
        self.assertIn("progress=100%", rendered)
        self.assertIn("Transcription failed", rendered)
        self.assertNotIn("Completed", rendered)

    def test_success_phases_are_rendered_in_order(self) -> None:
        output = StringIO()
        renderer = ProgressRenderer(mode="on", stream=output)
        renderer.render(ProgressEvent(phase=ProgressPhase.PREPARING, message="Preparing"))
        renderer.render(
            ProgressEvent(phase=ProgressPhase.LOADING_MODEL, message="Loading")
        )
        renderer.render(
            ProgressEvent(
                phase=ProgressPhase.TRANSCRIBING,
                current=10,
                total=10,
            )
        )
        renderer.render(
            ProgressEvent(
                phase=ProgressPhase.WRITING_OUTPUT,
                message="Writing transcript files...",
            )
        )
        renderer.render(ProgressEvent(phase=ProgressPhase.COMPLETED))
        rendered = output.getvalue()
        markers = [
            "[preparing]",
            "[load-model]",
            "progress=100%",
            "[write-output]",
            "Completed elapsed=",
        ]
        positions = [rendered.index(marker) for marker in markers]
        self.assertEqual(positions, sorted(positions))

    def test_fallback_phase_resets_percentage_for_next_backend(self) -> None:
        output = StringIO()
        renderer = ProgressRenderer(mode="on", stream=output)
        renderer.render(
            ProgressEvent(
                phase=ProgressPhase.TRANSCRIBING,
                current=10,
                total=10,
                message="Qwen3 transcribing...",
            )
        )
        renderer.render(
            ProgressEvent(
                phase=ProgressPhase.FALLBACK,
                message="Fallback to SenseVoice; loading model...",
            )
        )
        renderer.render(
            ProgressEvent(
                phase=ProgressPhase.TRANSCRIBING,
                current=0,
                total=10,
                message="SenseVoice transcribing...",
            )
        )
        renderer.render(
            ProgressEvent(
                phase=ProgressPhase.TRANSCRIBING,
                current=1,
                total=10,
                message="SenseVoice transcribing...",
            )
        )

        lines = output.getvalue().splitlines()
        fallback_line = next(line for line in lines if line.startswith("[fallback]"))
        self.assertIn("progress=100%", output.getvalue())
        self.assertNotIn("%", fallback_line)
        self.assertIn("progress=0%", output.getvalue())
        self.assertIn("progress=10%", output.getvalue())

    def test_fallback_phase_allows_same_rank_transition_but_rejects_regression(self) -> None:
        output = StringIO()
        renderer = ProgressRenderer(mode="on", stream=output)
        renderer.render(
            ProgressEvent(phase=ProgressPhase.TRANSCRIBING, current=1, total=2)
        )
        renderer.render(
            ProgressEvent(
                phase=ProgressPhase.FALLBACK,
                message="Fallback to SenseVoice; loading model...",
            )
        )
        renderer.render(
            ProgressEvent(phase=ProgressPhase.TRANSCRIBING, current=0, total=2)
        )
        renderer.render(
            ProgressEvent(
                phase=ProgressPhase.WRITING_OUTPUT,
                message="Writing transcript files...",
            )
        )
        rendered_before_regression = output.getvalue()

        renderer.render(
            ProgressEvent(
                phase=ProgressPhase.TRANSCRIBING,
                current=2,
                total=2,
                message="stale transcription",
            )
        )

        self.assertEqual(output.getvalue(), rendered_before_regression)

    def test_batch_context_is_preserved_on_events(self) -> None:
        output = StringIO()
        reporter = ProgressReporter(
            mode="on",
            stream=output,
            file_index=3,
            file_total=8,
            file_name="meeting-03.m4a",
        )
        reporter.emit(
            ProgressEvent(
                phase=ProgressPhase.TRANSCRIBING,
                current=1,
                total=2,
            )
        )
        self.assertEqual(reporter.file_index, 3)
        self.assertIn("File 3 of 8: meeting-03.m4a", output.getvalue())
        self.assertIn("progress=50%", output.getvalue())

    def test_heartbeat_renders_current_phase_after_transition(self) -> None:
        reporter = ProgressReporter(mode="on", stream=StringIO())
        rendered_events: list[ProgressEvent] = []
        with patch.object(
            reporter.renderer,
            "render",
            side_effect=lambda event: rendered_events.append(event),
        ):
            reporter.emit(
                ProgressEvent(
                    phase=ProgressPhase.TRANSCRIBING,
                    current=1,
                    total=2,
                )
            )
            reporter.emit_phase(
                ProgressPhase.WRITING_OUTPUT,
                message="Writing transcript files...",
            )
            reporter._heartbeat_tick()

        self.assertEqual(
            [event.phase for event in rendered_events],
            [
                ProgressPhase.TRANSCRIBING.value,
                ProgressPhase.WRITING_OUTPUT.value,
                ProgressPhase.WRITING_OUTPUT.value,
            ],
        )

    def test_heartbeat_refreshes_fallback_then_current_backend(self) -> None:
        reporter = ProgressReporter(mode="on", stream=StringIO())
        rendered_events: list[ProgressEvent] = []
        with patch.object(
            reporter.renderer,
            "render",
            side_effect=lambda event: rendered_events.append(event),
        ):
            reporter.emit(
                ProgressEvent(
                    phase=ProgressPhase.TRANSCRIBING,
                    current=10,
                    total=10,
                    message="Qwen3 transcribing...",
                )
            )
            reporter.emit_phase(
                ProgressPhase.FALLBACK,
                message="Fallback to SenseVoice; loading model...",
            )
            reporter._heartbeat_tick()
            reporter.emit(
                ProgressEvent(
                    phase=ProgressPhase.TRANSCRIBING,
                    current=0,
                    total=10,
                    message="SenseVoice transcribing...",
                )
            )
            reporter._heartbeat_tick()

        self.assertEqual(
            [event.phase for event in rendered_events],
            [
                ProgressPhase.TRANSCRIBING.value,
                ProgressPhase.FALLBACK.value,
                ProgressPhase.FALLBACK.value,
                ProgressPhase.TRANSCRIBING.value,
                ProgressPhase.TRANSCRIBING.value,
            ],
        )
        self.assertEqual(
            rendered_events[2].message,
            "Fallback to SenseVoice; loading model...",
        )
        self.assertEqual(rendered_events[4].message, "SenseVoice transcribing...")

    def test_qwen_structured_progress_uses_processed_audio(self) -> None:
        event = _progress_event(
            {
                "event": "chunk_completed",
                "processed_audio_sec": 30,
                "audio_duration_sec": 60,
                "chunk_index": 1,
                "total_chunks": 2,
            }
        )
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.current, 30)
        self.assertEqual(event.total, 60)
        self.assertEqual(event.unit, "seconds")
        self.assertTrue(event.determinate)


if __name__ == "__main__":
    unittest.main()
