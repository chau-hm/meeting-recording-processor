from array import array
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import wave

from meeting_recording_processor.errors import MediaError
from meeting_recording_processor.media.normalize import normalize_audio
from meeting_recording_processor.media.probe import MediaMetadata, probe_media
from meeting_recording_processor.media.signal import analyze_wav_signal


class MediaTests(unittest.TestCase):
    def test_probe_selects_default_audio_stream(self) -> None:
        payload = {
            "format": {"format_name": "mov,mp4", "duration": "12.5"},
            "streams": [
                {"index": 0, "codec_type": "video", "codec_name": "h264"},
                {
                    "index": 1,
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "channels": 2,
                    "sample_rate": "48000",
                    "tags": {"language": "yue"},
                    "disposition": {"default": 1},
                },
            ],
        }

        def runner(*_args, **_kwargs):
            return subprocess.CompletedProcess([], 0, json.dumps(payload), "")

        metadata = probe_media(Path("meeting.mp4"), runner=runner)
        self.assertEqual(metadata.selected_audio_stream, 1)
        self.assertEqual(metadata.duration_seconds, 12.5)
        self.assertEqual(metadata.audio_streams[0].language, "yue")

    def test_probe_rejects_media_without_audio(self) -> None:
        payload = {"format": {}, "streams": [{"index": 0, "codec_type": "video"}]}

        def runner(*_args, **_kwargs):
            return subprocess.CompletedProcess([], 0, json.dumps(payload), "")

        with self.assertRaises(MediaError):
            probe_media(Path("silent.mp4"), runner=runner)

    def test_normalize_builds_explicit_stream_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "normalized.wav"
            metadata = MediaMetadata("mov", 1.0, (), 3)
            captured = []

            def runner(command, **_kwargs):
                captured.extend(command)
                destination.write_bytes(b"RIFF-placeholder")
                return subprocess.CompletedProcess(command, 0, "", "")

            normalize_audio(Path("meeting.mp4"), metadata, destination, runner=runner)
            self.assertIn("0:3", captured)
            self.assertIn("16000", captured)
            self.assertTrue(destination.exists())

    def test_signal_analysis_reports_active_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tone.wav"
            samples = array("h", [5000, -5000] * 8000)
            with wave.open(str(path), "wb") as target:
                target.setnchannels(1)
                target.setsampwidth(2)
                target.setframerate(16000)
                target.writeframes(samples.tobytes())
            stats = analyze_wav_signal(path)
            self.assertAlmostEqual(stats.duration_seconds, 1.0, places=2)
            self.assertGreater(stats.active_audio_seconds, 0.9)
            self.assertGreater(stats.rms, 0.1)


if __name__ == "__main__":
    unittest.main()
