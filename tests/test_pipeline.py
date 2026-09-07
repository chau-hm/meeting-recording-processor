from pathlib import Path
from io import StringIO
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from meeting_recording_processor.asr.qwen3 import Qwen3Backend
from meeting_recording_processor.config import AsrMode, ExtractConfig
from meeting_recording_processor.errors import TranscriptionFailed
from meeting_recording_processor.media.probe import AudioStream, MediaMetadata
from meeting_recording_processor.media.signal import AudioSignalStats
from meeting_recording_processor.models import ResolvedModel
from meeting_recording_processor.pipeline import Extractor
from meeting_recording_processor.postprocess import postprocess_result
from meeting_recording_processor.progress import ProgressEvent, ProgressPhase, ProgressReporter
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


class BrokenStream:
    def isatty(self) -> bool:
        return False

    def write(self, _value: str) -> int:
        raise BrokenPipeError("closed stderr")

    def flush(self) -> None:
        raise BrokenPipeError("closed stderr")


class ReportingBackend(FakeBackend):
    def __init__(
        self,
        result: BackendResult | Exception,
        events: tuple[ProgressEvent, ...],
    ) -> None:
        super().__init__(result)
        self.events = events

    def transcribe(self, *_args, **kwargs) -> BackendResult:
        callback = kwargs.get("progress_callback")
        if callback is not None:
            for event in self.events:
                callback(event)
        return super().transcribe(*_args, **kwargs)


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

    def extractor(
        self,
        backends: dict[str, FakeBackend],
        *,
        media_duration: float | None = 5.0,
        progress_factory=None,
    ) -> Extractor:
        metadata = MediaMetadata(
            "mov,mp4,m4a",
            media_duration,
            (AudioStream(0, "aac", 1, 16000, "yue", True),),
            0,
        )
        signal = AudioSignalStats(5.0, 0.2, 0.5, 5.0, 1.0, 0.01)

        def normalizer(_source, _metadata, destination):
            destination.write_bytes(b"normalized")
            return destination

        def resolver(model_id, _cache):
            return ResolvedModel(model_id, self.root / "model", "snapshot-test")

        def factory(name, _model_id, _model_path, _verbose):
            return backends[name]

        extractor = Extractor(
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
        if progress_factory is not None:
            extractor.progress_factory = progress_factory
        return extractor

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

    def test_unknown_media_duration_does_not_abort_asr(self) -> None:
        qwen = FakeBackend(result("qwen3", "我哋今日開始開會。"))
        extracted = self.extractor(
            {"qwen3": qwen, "sensevoice": FakeBackend(result("sensevoice", "fallback"))},
            media_duration=None,
        ).extract(self.config(AsrMode.QWEN3))
        self.assertEqual(extracted.selected_backend, "qwen3")

    def test_backend_failure_does_not_report_completion(self) -> None:
        qwen = FakeBackend(RuntimeError("model crash"))
        output = StringIO()
        reporter = ProgressReporter(mode="on", stream=output)
        extractor = self.extractor(
            {"qwen3": qwen},
            progress_factory=lambda _config: reporter,
        )
        with self.assertRaises(TranscriptionFailed):
            extractor.extract(self.config(AsrMode.QWEN3))
        self.assertIn("Transcription failed", output.getvalue())
        self.assertNotIn("Completed", output.getvalue())

    def test_broken_progress_stream_does_not_trigger_fallback(self) -> None:
        qwen = FakeBackend(result("qwen3", "我哋今日開始開會。"))
        sensevoice = FakeBackend(result("sensevoice", "不應執行。"))
        reporter = ProgressReporter(mode="on", stream=BrokenStream())
        extracted = self.extractor(
            {"qwen3": qwen, "sensevoice": sensevoice},
            progress_factory=lambda _config: reporter,
        ).extract(self.config())
        self.assertEqual(extracted.selected_backend, "qwen3")
        self.assertFalse(extracted.fallback_used)
        self.assertEqual(qwen.calls, 1)
        self.assertEqual(sensevoice.calls, 0)
        self.assertFalse(reporter.enabled)

    def test_fallback_does_not_regress_rendered_phases(self) -> None:
        qwen = ReportingBackend(
            result("qwen3", "!!!!!!!!!!"),
            (
                ProgressEvent(
                    phase=ProgressPhase.TRANSCRIBING,
                    current=5,
                    total=5,
                ),
            ),
        )
        sensevoice = FakeBackend(result("sensevoice", "後備辨識成功。"))
        output = StringIO()
        reporter = ProgressReporter(mode="on", stream=output)
        extracted = self.extractor(
            {"qwen3": qwen, "sensevoice": sensevoice},
            progress_factory=lambda _config: reporter,
        ).extract(self.config())
        self.assertEqual(extracted.selected_backend, "sensevoice")
        rendered = output.getvalue()
        markers = [
            "[transcribe]",
            "[write-output]",
            "Completed elapsed=",
        ]
        positions = [rendered.index(marker) for marker in markers]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn(
            "[load-model]",
            rendered[positions[0] :],
        )

    def test_qwen_progress_callback_failure_does_not_fail_inference(self) -> None:
        transcript = "我哋今日開始開會。"

        def fake_transcribe(_audio_path: str, **kwargs):
            kwargs["on_progress"](
                {
                    "event": "chunk_completed",
                    "processed_audio_sec": 1,
                    "audio_duration_sec": 2,
                }
            )
            return SimpleNamespace(
                language="Cantonese",
                text=transcript,
                segments=[
                    SimpleNamespace(start=0.0, end=1.0, text=transcript),
                ],
                chunks=None,
                finish_reason="stop",
                truncated=False,
            )

        def broken_callback(_event) -> None:
            raise BrokenPipeError("closed stderr")

        module = SimpleNamespace(transcribe=fake_transcribe)
        with patch.dict(sys.modules, {"mlx_qwen3_asr": module}):
            backend_result = Qwen3Backend(
                model_id="test/qwen3",
                model_path=self.root / "model",
            ).transcribe(
                self.input,
                language="Cantonese",
                profile_text=None,
                progress_callback=broken_callback,
            )

        self.assertEqual(backend_result.text, transcript)


if __name__ == "__main__":
    unittest.main()
