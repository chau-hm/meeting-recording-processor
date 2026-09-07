from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from meeting_recording_processor.cli import _run_batch, build_parser


class CliTests(unittest.TestCase):
    def test_extract_defaults_to_auto(self) -> None:
        args = build_parser().parse_args(["extract", "meeting.m4a"])
        self.assertEqual(args.command, "extract")
        self.assertEqual(args.asr, "auto")
        self.assertEqual(args.language, "Cantonese")

    def test_export_parser(self) -> None:
        args = build_parser().parse_args(["export", "meeting.transcript.json"])
        self.assertEqual(args.command, "export")

    def test_batch_parser_preserves_transcription_options(self) -> None:
        args = build_parser().parse_args(
            ["batch", "recordings", "--asr", "qwen3", "--progress", "off"]
        )
        self.assertEqual(args.command, "batch")
        self.assertEqual(args.asr, "qwen3")
        self.assertEqual(args.progress_mode, "off")

    def test_batch_reports_file_index_and_total(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            input_dir = Path(temporary) / "recordings"
            output_dir = Path(temporary) / "output"
            input_dir.mkdir()
            (input_dir / "meeting-02.m4a").write_bytes(b"two")
            (input_dir / "meeting-01.m4a").write_bytes(b"one")
            args = build_parser().parse_args(
                ["batch", str(input_dir), "--output-dir", str(output_dir)]
            )
            output = StringIO()
            with (
                patch(
                    "meeting_recording_processor.cli.extract",
                    side_effect=lambda config: SimpleNamespace(
                        output_path=config.output_path
                    ),
                ) as mocked_extract,
                redirect_stderr(output),
                redirect_stdout(StringIO()),
            ):
                result = _run_batch(args)

            self.assertEqual(result, 0)
            self.assertEqual(mocked_extract.call_count, 2)
            self.assertIn("File 1 of 2: meeting-01.m4a", output.getvalue())
            self.assertIn("File 2 of 2: meeting-02.m4a", output.getvalue())


if __name__ == "__main__":
    unittest.main()
