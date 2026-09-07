from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from meeting_recording_processor.cli import _run_batch, _run_download, build_parser
from meeting_recording_processor.errors import ConfigurationError
from meeting_recording_processor.models import ResolvedModel


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

    def test_vibevoice_parser_and_model_override(self) -> None:
        args = build_parser().parse_args(
            [
                "extract",
                "meeting.m4a",
                "--asr",
                "vibevoice",
                "--vibevoice-model",
                "local/vibevoice",
            ]
        )
        self.assertEqual(args.asr, "vibevoice")
        self.assertEqual(args.vibevoice_model, "local/vibevoice")

    def test_download_parser_supports_vibevoice_and_all(self) -> None:
        vibevoice = build_parser().parse_args(
            ["download-model", "--asr", "vibevoice"]
        )
        all_models = build_parser().parse_args(["download-model", "--asr", "all"])
        self.assertEqual(vibevoice.asr, "vibevoice")
        self.assertEqual(all_models.asr, "all")

    def test_download_all_includes_vibevoice(self) -> None:
        args = build_parser().parse_args(["download-model", "--asr", "all"])
        downloaded: list[tuple[str, Path]] = []
        with (
            patch("meeting_recording_processor.cli.require_apple_silicon"),
            patch(
                "meeting_recording_processor.cli.download_model",
                side_effect=lambda model_id, cache_dir: (
                    downloaded.append((model_id, cache_dir))
                    or ResolvedModel(model_id, cache_dir, "snapshot")
                ),
            ),
            patch("meeting_recording_processor.cli.directory_size", return_value=0),
            patch("meeting_recording_processor.cli.human_size", return_value="0 B"),
        ):
            result = _run_download(args)

        self.assertEqual(result, 0)
        self.assertEqual(
            [model_id for model_id, _cache_dir in downloaded],
            [
                args.qwen_model,
                args.sensevoice_model,
                args.vibevoice_model,
            ],
        )

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

    def _assert_batch_collision(self, names: tuple[str, str], *, overwrite: bool) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            input_dir = Path(temporary) / "recordings"
            output_dir = Path(temporary) / "output"
            input_dir.mkdir()
            for name in names:
                (input_dir / name).write_bytes(b"fixture")
            command = ["batch", str(input_dir), "--output-dir", str(output_dir)]
            if overwrite:
                command.append("--overwrite")
            args = build_parser().parse_args(command)
            with patch("meeting_recording_processor.cli.extract") as mocked_extract:
                with self.assertRaises(ConfigurationError) as caught:
                    _run_batch(args)

            message = str(caught.exception)
            self.assertIn("batch output collision", message)
            for name in names:
                self.assertIn(name, message)
            self.assertIn(".transcript.json", message)
            self.assertEqual(mocked_extract.call_count, 0)

    def test_batch_rejects_same_stem_collision_before_extract(self) -> None:
        self._assert_batch_collision(("meeting.m4a", "meeting.mp3"), overwrite=False)

    def test_batch_rejects_same_stem_collision_even_with_overwrite(self) -> None:
        self._assert_batch_collision(("meeting.m4a", "meeting.mp3"), overwrite=True)

    def test_batch_rejects_case_equivalent_destination_collision(self) -> None:
        self._assert_batch_collision(("Meeting.m4a", "meeting.wav"), overwrite=False)


if __name__ == "__main__":
    unittest.main()
