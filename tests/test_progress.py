from io import StringIO
import unittest

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

    def test_failure_does_not_report_completion(self) -> None:
        output = StringIO()
        reporter = ProgressReporter(mode="on", stream=output)
        reporter.start()
        reporter.fail("backend error")
        rendered = output.getvalue()
        self.assertIn("Transcription failed", rendered)
        self.assertNotIn("Completed in", rendered)

    def test_pending_final_percentage_is_discarded_on_failure(self) -> None:
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
        self.assertNotIn("progress=100%", rendered)
        self.assertIn("Transcription failed", rendered)

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
        self.assertIn("progress=50%", output.getvalue())

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
