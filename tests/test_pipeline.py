from pathlib import Path
import tempfile
import unittest

from meeting_recording_processor.config import AsrMode, ExtractConfig
from meeting_recording_processor.errors import TranscriptionFailed
from meeting_recording_processor.media.probe import AudioStream, MediaMetadata
from meeting_recording_processor.media.signal import AudioSignalStats
from meeting_recording_processor.models import ResolvedModel
from meeting_recording_processor.pipeline import Extractor
from meeting_recording_processor.postprocess import postprocess_result
from meeting_recording_processor.schema_io import load_package
from meeting_recording_processor.schemas import BackendResult, TranscriptSegment


class FakeBackend:
    def __init__(self, result: BackendResult | Exception) -> None:
        self.result = result
        self.calls = 0

    def transcribe(self, *_args, **_kwargs) -> BackendResult:
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def result(backend: str, text: str) -> BackendResult:
    return BackendResult(
        language="Cantonese" if backend == "qwen3" else "yue",
        backend=backend,
        model=f"test/{backend}",
        text=text,
        segments=(TranscriptSegment(0.0, 5.0, text, "model" if backend == "qwen3" else "chunk"),),
    )


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input = self.root / "meeting.m4a"
        self.input.write_bytes(b"private fixture bytes")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def extractor(self, backends: dict[str, FakeBackend]) -> Extractor:
        metadata = MediaMetadata(
            "mov,mp4,m4a", 5.0, (AudioStream(0, "aac", 1, 16000, "yue", True),), 0
        )
        signal = AudioSignalStats(5.0, 0.2, 0.5, 5.0, 1.0, 0.01)

        def normalizer(_source, _metadata, destination):
            destination.write_bytes(b"normalized")
            return destination

        def resolver(model_id, _cache):
            return ResolvedModel(model_id, self.root / "model", "snapshot-test")

        def factory(name, _model_id, _model_path, _verbose):
            return backends[name]

        return Extractor(
            platform_validator=lambda: None,
            media_commands_validator=lambda: None,
            probe=lambda _path: metadata,
            normalizer=normalizer,
            signal_analyzer=lambda _path: signal,
            model_resolver=resolver,
            backend_factory=factory,
            postprocessor=lambda backend_result, audio_duration: postprocess_result(
                backend_result, audio_duration=audio_duration, converter=lambda value: value
            ),
            probe_version=lambda: "ffprobe test",
        )

    def config(self, mode: AsrMode = AsrMode.AUTO) -> ExtractConfig:
        return ExtractConfig(
            input_path=self.input,
            output_dir=self.root / "output",
            work_dir=self.root / "work",
            cache_dir=self.root / "cache",
            asr_mode=mode,
        )

    def test_auto_accepts_qwen_without_running_fallback(self) -> None:
        qwen = FakeBackend(result("qwen3", "我哋今日開始開會。"))
        sensevoice = FakeBackend(result("sensevoice", "不應執行。"))
        extracted = self.extractor({"qwen3": qwen, "sensevoice": sensevoice}).extract(
            self.config()
        )
        self.assertEqual(extracted.selected_backend, "qwen3")
        self.assertFalse(extracted.fallback_used)
        self.assertEqual(qwen.calls, 1)
        self.assertEqual(sensevoice.calls, 0)

    def test_auto_falls_back_on_punctuation_collapse(self) -> None:
        qwen = FakeBackend(result("qwen3", "!!!!!!!!!!"))
        sensevoice = FakeBackend(result("sensevoice", "我哋今日開始開會。"))
        extracted = self.extractor({"qwen3": qwen, "sensevoice": sensevoice}).extract(
            self.config()
        )
        self.assertTrue(extracted.fallback_used)
        package = load_package(extracted.output_path, require_completed=True)
        self.assertEqual(len(package["attempts"]), 2)
        self.assertEqual(package["attempts"][0]["status"], "hard_failure")
        self.assertEqual(package["selected_attempt_id"], "attempt-2-sensevoice")

    def test_subjectively_imperfect_but_substantive_text_does_not_fallback(self) -> None:
        qwen = FakeBackend(result("qwen3", "呢個 technical name 可能聽錯咗。"))
        sensevoice = FakeBackend(result("sensevoice", "fallback"))
        self.extractor({"qwen3": qwen, "sensevoice": sensevoice}).extract(self.config())
        self.assertEqual(sensevoice.calls, 0)

    def test_explicit_qwen_failure_does_not_fallback(self) -> None:
        qwen = FakeBackend(result("qwen3", "!!!!!!!!!!"))
        sensevoice = FakeBackend(result("sensevoice", "我哋今日開始開會。"))
        extractor = self.extractor({"qwen3": qwen, "sensevoice": sensevoice})
        with self.assertRaises(TranscriptionFailed) as caught:
            extractor.extract(self.config(AsrMode.QWEN3))
        self.assertEqual(sensevoice.calls, 0)
        package = load_package(caught.exception.diagnostic_path)
        self.assertEqual(package["status"], "failed")

    def test_backend_exception_is_preserved_before_fallback(self) -> None:
        qwen = FakeBackend(RuntimeError("model crash"))
        sensevoice = FakeBackend(result("sensevoice", "後備辨識成功。"))
        extracted = self.extractor({"qwen3": qwen, "sensevoice": sensevoice}).extract(
            self.config()
        )
        package = load_package(extracted.output_path, require_completed=True)
        self.assertEqual(package["attempts"][0]["status"], "error")
        self.assertIn("model crash", package["attempts"][0]["error"])


if __name__ == "__main__":
    unittest.main()
