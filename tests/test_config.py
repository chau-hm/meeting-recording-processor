from pathlib import Path
import unittest

from meeting_recording_processor.config import (
    DEFAULT_VIBEVOICE_MODEL,
    AsrMode,
    ExportConfig,
    ExtractConfig,
)


class ConfigTests(unittest.TestCase):
    def test_extract_output_name(self) -> None:
        config = ExtractConfig(
            input_path=Path("meeting.v2.mp4"),
            output_dir=Path("out"),
            work_dir=Path("work"),
            cache_dir=Path("cache"),
        )
        self.assertEqual(config.asr_mode, AsrMode.AUTO)
        self.assertEqual(config.vibevoice_model, DEFAULT_VIBEVOICE_MODEL)
        self.assertEqual(config.output_path, Path("out/meeting.v2.transcript.json"))

    def test_export_strips_transcript_suffix(self) -> None:
        config = ExportConfig(Path("out/meeting.transcript.json"))
        self.assertEqual(config.base_name, "meeting")
        self.assertEqual(config.resolved_output_dir, Path("out"))


if __name__ == "__main__":
    unittest.main()
