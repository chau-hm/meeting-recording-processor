from pathlib import Path
import unittest

from meeting_recording_processor.config import (
    DEFAULT_QWEN_ALIGNER_MODEL,
    DEFAULT_VIBEVOICE_ACOUSTIC_CHUNK_SIZE,
    DEFAULT_VIBEVOICE_MODEL,
    AsrMode,
    ExportConfig,
    ExtractConfig,
)
from meeting_recording_processor.errors import ConfigurationError


class ConfigTests(unittest.TestCase):
    def test_extract_output_name(self) -> None:
        config = ExtractConfig(
            input_path=Path("meeting.v2.mp4"),
            output_dir=Path("out"),
            work_dir=Path("work"),
            cache_dir=Path("cache"),
        )
        self.assertEqual(config.asr_mode, AsrMode.AUTO)
        self.assertEqual(config.qwen_aligner_model, DEFAULT_QWEN_ALIGNER_MODEL)
        self.assertEqual(config.vibevoice_model, DEFAULT_VIBEVOICE_MODEL)
        self.assertEqual(
            config.vibevoice_acoustic_chunk_size,
            DEFAULT_VIBEVOICE_ACOUSTIC_CHUNK_SIZE,
        )
        self.assertEqual(config.output_path, Path("out/meeting.v2.transcript.json"))

    def test_invalid_vibevoice_acoustic_chunk_size_is_rejected(self) -> None:
        for value in (0, -3200, 65000):
            with self.subTest(value=value):
                with self.assertRaises(ConfigurationError):
                    ExtractConfig(
                        input_path=Path("meeting.m4a"),
                        output_dir=Path("out"),
                        work_dir=Path("work"),
                        cache_dir=Path("cache"),
                        vibevoice_acoustic_chunk_size=value,
                    )

    def test_export_strips_transcript_suffix(self) -> None:
        config = ExportConfig(Path("out/meeting.transcript.json"))
        self.assertEqual(config.base_name, "meeting")
        self.assertEqual(config.resolved_output_dir, Path("out"))


if __name__ == "__main__":
    unittest.main()
