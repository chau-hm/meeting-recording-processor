import json
from pathlib import Path
import tempfile
import unittest

from meeting_recording_processor.config import ExportConfig
from meeting_recording_processor.errors import OutputExistsError, SchemaError
from meeting_recording_processor.outputs.writers import export_transcript, format_srt_timestamp
from meeting_recording_processor.schema_io import load_package, write_package


def completed_package() -> dict:
    return {
        "schema_version": "1.0",
        "status": "completed",
        "run_id": "test-run",
        "attempts": [
            {
                "attempt_id": "attempt-1-qwen3",
                "backend": "qwen3",
                "model": "test/qwen3",
                "status": "completed",
                "raw_segments": [],
            }
        ],
        "selected_attempt_id": "attempt-1-qwen3",
        "transcript": {
            "language": "Cantonese",
            "text": "第一句。\nSecond line.",
            "segments": [
                {"start": 0.0, "end": 1.25, "text": "第一句。", "timing_source": "model"},
                {"start": 1.25, "end": 2.5, "text": "Second line.", "timing_source": "model"},
            ],
        },
    }


class SchemaAndExportTests(unittest.TestCase):
    def test_machine_readable_schema_is_valid_json(self) -> None:
        schema_path = Path(__file__).parents[1] / "schemas/transcript-v1.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(schema["properties"]["schema_version"]["const"], "1.0")

    def test_package_round_trip_preserves_unicode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "meeting.transcript.json"
            write_package(path, completed_package())
            loaded = load_package(path, require_completed=True)
            self.assertEqual(loaded["transcript"]["text"], "第一句。\nSecond line.")

    def test_failed_package_cannot_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "failed.transcript.json"
            payload = {
                "schema_version": "1.0",
                "status": "failed",
                "run_id": "failed-run",
                "attempts": [],
                "selected_attempt_id": None,
                "transcript": None,
            }
            write_package(path, payload)
            with self.assertRaises(SchemaError):
                export_transcript(ExportConfig(path))

    def test_export_writes_txt_and_srt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "meeting.transcript.json"
            write_package(path, completed_package())
            txt, srt = export_transcript(ExportConfig(path))
            self.assertEqual(txt.read_text(encoding="utf-8"), "第一句。\nSecond line.\n")
            rendered = srt.read_text(encoding="utf-8")
            self.assertIn("00:00:00,000 --> 00:00:01,250", rendered)
            self.assertIn("第一句。", rendered)
            self.assertIn("Second line.", rendered)
            with self.assertRaises(OutputExistsError):
                export_transcript(ExportConfig(path))

    def test_srt_timestamp_rounding(self) -> None:
        self.assertEqual(format_srt_timestamp(3661.2346), "01:01:01,235")


if __name__ == "__main__":
    unittest.main()
