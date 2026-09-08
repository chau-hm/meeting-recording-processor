from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import wave
import unittest
from unittest.mock import patch

from meeting_recording_processor.asr.sensevoice import SenseVoiceBackend
from meeting_recording_processor.errors import BackendError


def write_wav(path: Path, seconds: float, *, sample_rate: int = 16_000) -> None:
    frame_count = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"\0\0" * frame_count)


class FakeSenseVoiceModel:
    def __init__(self, results: list[SimpleNamespace | Exception]) -> None:
        self.results = results
        self.calls: list[dict[str, object]] = []

    def generate(self, path: str, **kwargs):
        with wave.open(path, "rb") as source:
            duration = source.getnframes() / source.getframerate()
        self.calls.append({"path": path, "duration": duration, **kwargs})
        result = self.results[len(self.calls) - 1]
        if isinstance(result, Exception):
            raise result
        return result


def result(
    text: str,
    *,
    language: str = "yue",
    emotion: str | None = "neutral",
    event: str | None = "Speech",
) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        language=language,
        segments=[
            {
                "text": text,
                "language": language,
                "emotion": emotion,
                "event": event,
            }
        ],
    )


class SenseVoiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.audio = self.root / "audio.wav"
        self.model_path = self.root / "sensevoice-snapshot"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_backend(
        self,
        model: FakeSenseVoiceModel,
        *,
        language: str = "Cantonese",
        profile_text: str | None = None,
        progress: list | None = None,
    ):
        loaded_paths: list[str] = []

        def load(path: str):
            loaded_paths.append(path)
            return model

        module = SimpleNamespace(load=load)
        with patch.dict(sys.modules, {"mlx_audio.stt": module}):
            backend_result = SenseVoiceBackend(
                model_id="test/sensevoice",
                model_path=self.model_path,
            ).transcribe(
                self.audio,
                language=language,
                profile_text=profile_text,
                progress_callback=progress.append if progress is not None else None,
            )
        return backend_result, loaded_paths

    def test_local_model_load_and_cantonese_language_mapping(self) -> None:
        write_wav(self.audio, 1.0)
        model = FakeSenseVoiceModel([result("廣東話")])

        backend_result, loaded_paths = self.run_backend(model)

        self.assertEqual(loaded_paths, [str(self.model_path)])
        self.assertEqual(model.calls[0]["language"], "yue")
        self.assertEqual(backend_result.language, "yue")
        self.assertEqual(backend_result.metadata["timestamp_source"], "chunk")

        model = FakeSenseVoiceModel([result("廣東話")])
        self.run_backend(model, language="yue")
        self.assertEqual(model.calls[0]["language"], "yue")

    def test_supported_language_values_and_unsupported_language(self) -> None:
        write_wav(self.audio, 0.1)
        for language, expected in (
            ("auto", "auto"),
            ("Chinese", "zh"),
            ("Mandarin", "zh"),
            ("English", "en"),
            ("Japanese", "ja"),
            ("Korean", "ko"),
        ):
            with self.subTest(language=language):
                model = FakeSenseVoiceModel([result("text", language=expected)])
                self.run_backend(model, language=language)
                self.assertEqual(model.calls[0]["language"], expected)

        model = FakeSenseVoiceModel([result("不應執行")])
        with self.assertRaises(BackendError) as caught:
            self.run_backend(model, language="French")
        self.assertIn("French", str(caught.exception))
        self.assertEqual(model.calls, [])

    def test_thirty_second_chunk_boundaries_text_order_metadata_and_progress(self) -> None:
        write_wav(self.audio, 65.0)
        model = FakeSenseVoiceModel(
            [
                result("第一段"),
                result(""),
                result("第三段", emotion=None, event=None),
            ]
        )
        progress: list = []

        backend_result, _ = self.run_backend(model, progress=progress)

        self.assertEqual([round(call["duration"], 3) for call in model.calls], [30.0, 30.0, 5.0])
        chunks = backend_result.metadata["chunks"]
        self.assertEqual(
            [(item["index"], item["start"], item["end"]) for item in chunks],
            [(0, 0.0, 30.0), (1, 30.0, 60.0), (2, 60.0, 65.0)],
        )
        self.assertEqual(backend_result.text, "第一段\n第三段")
        self.assertEqual(len(backend_result.segments), 2)
        self.assertTrue(
            all(segment.timing_source == "chunk" for segment in backend_result.segments)
        )
        self.assertEqual(chunks[0]["language"], "yue")
        self.assertEqual(chunks[0]["emotion"], "neutral")
        self.assertEqual(chunks[0]["event"], "Speech")
        self.assertNotIn("emotion", chunks[2])
        self.assertNotIn("event", chunks[2])
        processed = [event.current for event in progress]
        self.assertEqual(processed, [0.0, 30.0, 60.0, 65.0])
        self.assertEqual(processed, sorted(processed))

    def test_context_warning_is_explicit_and_not_injected(self) -> None:
        write_wav(self.audio, 1.0)
        model = FakeSenseVoiceModel([result("text")])

        backend_result, _ = self.run_backend(model, profile_text="technical terms")

        self.assertTrue(any("context hotwords" in warning for warning in backend_result.warnings))
        self.assertNotIn("context", model.calls[0])

    def test_late_chunk_failure_preserves_completed_work_diagnostics(self) -> None:
        write_wav(self.audio, 65.0)
        model = FakeSenseVoiceModel(
            [
                result("第一段"),
                RuntimeError("decoder crashed"),
                result("不應執行"),
            ]
        )

        with self.assertRaises(BackendError) as caught:
            self.run_backend(model)

        metadata = caught.exception.metadata
        self.assertEqual(metadata["failed_chunk_index"], 1)
        self.assertEqual(metadata["completed_chunk_count"], 1)
        self.assertEqual(metadata["processed_audio_seconds"], 30.0)
        self.assertEqual(metadata["total_audio_seconds"], 65.0)
        self.assertEqual(metadata["completed_chunks"][0]["end"], 30.0)
        self.assertNotIn("partial transcript", metadata)


if __name__ == "__main__":
    unittest.main()
