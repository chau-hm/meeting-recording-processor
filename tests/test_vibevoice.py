from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import wave

from meeting_recording_processor.asr.vibevoice import (
    MAX_DURATION_SECONDS,
    VibeVoiceBackend,
)
from meeting_recording_processor.config import (
    DEFAULT_VIBEVOICE_ACOUSTIC_CHUNK_SIZE,
)
from meeting_recording_processor.errors import BackendError, ConfigurationError
from meeting_recording_processor.progress import ProgressPhase


class FakeTensor:
    shape = (1, 4)

    def __getitem__(self, _key):
        return self


class FakeBatch(dict):
    def __init__(self) -> None:
        super().__init__(input_ids=FakeTensor())
        self.to_args = None

    def to(self, *args):
        self.to_args = args
        return self


class FakeMpsOutOfMemoryError(RuntimeError):
    pass


class FakeProcessor:
    from_pretrained_calls: list[tuple[str, dict]] = []
    request_calls: list[tuple[str, str | None]] = []
    raw_payload: object = [
        'assistant[{"Start":0.0,"End":5.2,"Speaker":0,"Content":"我哋今日開始開會。"}]'
    ]
    parsed_payload: object = [
        {
            "Start": 0.0,
            "End": 5.2,
            "Speaker": 0,
            "Content": "我哋今日開始開會。",
        },
        {
            "Start": 5.2,
            "End": 9.1,
            "Speaker": 1,
            "Content": "OK，下一個 item。",
        },
    ]
    transcription_payload: object = ["我哋今日開始開會。 OK，下一個 item。"]

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        cls.from_pretrained_calls.append((path, kwargs))
        return cls()

    def apply_transcription_request(self, *, audio: str, prompt: str | None):
        self.request_calls.append((audio, prompt))
        return FakeBatch()

    def decode(self, _generated_ids, return_format: str = "raw"):
        if return_format == "parsed":
            return [self.parsed_payload]
        if return_format == "transcription_only":
            return self.transcription_payload
        return self.raw_payload


class FakeModel:
    from_pretrained_calls: list[tuple[str, dict]] = []
    to_calls: list[str] = []
    generate_calls: list[dict] = []
    generation_error: Exception | None = None

    def __init__(self, dtype: object) -> None:
        self.device = "cpu"
        self.dtype = dtype

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        cls.from_pretrained_calls.append((path, kwargs))
        return cls(kwargs["dtype"])

    def to(self, device: str):
        self.to_calls.append(device)
        self.device = device
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        if self.generation_error is not None:
            raise self.generation_error
        return FakeTensor()


def fake_runtime(*, mps_available: bool = True) -> tuple[ModuleType, ModuleType]:
    torch_module = ModuleType("torch")
    torch_module.__version__ = "2.test"
    torch_module.float32 = "torch.float32"
    torch_module.OutOfMemoryError = FakeMpsOutOfMemoryError
    mps_module = SimpleNamespace(
        is_available=lambda: mps_available,
        current_allocated_memory=lambda: 123,
        driver_allocated_memory=lambda: 456,
    )
    torch_module.backends = SimpleNamespace(mps=mps_module)
    torch_module.mps = mps_module
    torch_module.inference_mode = nullcontext

    transformers_module = ModuleType("transformers")
    transformers_module.__version__ = "5.3.0"
    transformers_module.AutoProcessor = FakeProcessor
    transformers_module.VibeVoiceAsrForConditionalGeneration = FakeModel
    return torch_module, transformers_module


class VibeVoiceTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeProcessor.from_pretrained_calls.clear()
        FakeProcessor.request_calls.clear()
        FakeModel.from_pretrained_calls.clear()
        FakeModel.to_calls.clear()
        FakeModel.generate_calls.clear()
        FakeModel.generation_error = None
        FakeProcessor.raw_payload = [
            'assistant[{"Start":0.0,"End":5.2,"Speaker":0,"Content":"我哋今日開始開會。"}]'
        ]
        FakeProcessor.parsed_payload = [
            {
                "Start": 0.0,
                "End": 5.2,
                "Speaker": 0,
                "Content": "我哋今日開始開會。",
            },
            {
                "Start": 5.2,
                "End": 9.1,
                "Speaker": 1,
                "Content": "OK，下一個 item。",
            },
        ]
        FakeProcessor.transcription_payload = ["我哋今日開始開會。 OK，下一個 item。"]
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.audio = self.root / "audio-24k.wav"
        with wave.open(str(self.audio), "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(24000)
            target.writeframes(b"\0\0" * 24000)
        self.model_path = self.root / "hub" / "snapshots" / "snapshot-test"
        self.model_path.mkdir(parents=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_native_api_maps_text_segments_speakers_and_raw_output(self) -> None:
        torch_module, transformers_module = fake_runtime()
        progress_events = []
        with patch.dict(
            "sys.modules",
            {"torch": torch_module, "transformers": transformers_module},
        ):
            result = VibeVoiceBackend(
                model_id="microsoft/VibeVoice-ASR-HF",
                model_path=self.model_path,
            ).transcribe(
                self.audio,
                language="Cantonese",
                profile_text="VibeVoice technical vocabulary",
                progress_callback=progress_events.append,
            )

        self.assertEqual(result.text, "我哋今日開始開會。 OK，下一個 item。")
        self.assertEqual(len(result.segments), 2)
        self.assertTrue(all(segment.timing_source == "model" for segment in result.segments))
        self.assertEqual(result.metadata["structured_segments"][1]["speaker"], 1)
        self.assertIn("assistant", result.metadata["raw_output"])
        self.assertEqual(result.metadata["model_snapshot"], "snapshot-test")
        self.assertTrue(result.metadata["context_provided"])
        self.assertEqual(result.language, "und")
        self.assertEqual(result.metadata["language_mode"], "not_detected")
        self.assertEqual(result.metadata["requested_language"], "Cantonese")
        self.assertEqual(result.metadata["dtype"], "torch.float32")
        self.assertEqual(
            FakeModel.generate_calls[0]["acoustic_tokenizer_chunk_size"],
            DEFAULT_VIBEVOICE_ACOUSTIC_CHUNK_SIZE,
        )
        self.assertEqual(FakeModel.to_calls, ["mps"])
        self.assertEqual(
            FakeModel.from_pretrained_calls[0][1]["dtype"],
            torch_module.float32,
        )
        self.assertEqual(FakeProcessor.from_pretrained_calls[0][0], str(self.model_path))
        self.assertTrue(
            all(
                call[1]["local_files_only"]
                for call in FakeProcessor.from_pretrained_calls
            )
        )
        self.assertEqual(
            FakeProcessor.request_calls,
            [(str(self.audio), "VibeVoice technical vocabulary")],
        )
        self.assertEqual(progress_events[0].phase, ProgressPhase.TRANSCRIBING.value)
        self.assertFalse(progress_events[0].determinate)
        self.assertIsNone(progress_events[0].percentage)

    def test_configured_acoustic_chunk_size_reaches_generation(self) -> None:
        torch_module, transformers_module = fake_runtime()
        with patch.dict(
            "sys.modules",
            {"torch": torch_module, "transformers": transformers_module},
        ):
            VibeVoiceBackend(
                model_id="microsoft/VibeVoice-ASR-HF",
                model_path=self.model_path,
                acoustic_tokenizer_chunk_size=32000,
            ).transcribe(
                self.audio,
                language="Cantonese",
                profile_text=None,
            )

        self.assertEqual(
            FakeModel.generate_calls[0]["acoustic_tokenizer_chunk_size"],
            32000,
        )

    def test_invalid_acoustic_chunk_size_is_rejected_before_model_load(self) -> None:
        for value in (0, -3200, 65000):
            with self.subTest(value=value):
                with self.assertRaises(ConfigurationError):
                    VibeVoiceBackend(
                        model_id="microsoft/VibeVoice-ASR-HF",
                        model_path=self.model_path,
                        acoustic_tokenizer_chunk_size=value,
                    )
        self.assertEqual(FakeModel.from_pretrained_calls, [])

    def test_generation_failure_preserves_runtime_and_memory_metadata(self) -> None:
        FakeModel.generation_error = FakeMpsOutOfMemoryError("MPS out of memory")
        torch_module, transformers_module = fake_runtime()
        with patch.dict(
            "sys.modules",
            {"torch": torch_module, "transformers": transformers_module},
        ):
            with self.assertRaises(BackendError) as caught:
                VibeVoiceBackend(
                    model_id="microsoft/VibeVoice-ASR-HF",
                    model_path=self.model_path,
                    acoustic_tokenizer_chunk_size=32000,
                ).transcribe(
                    self.audio,
                    language="Cantonese",
                    profile_text=None,
                )

        metadata = caught.exception.metadata
        self.assertEqual(metadata["device"], "mps")
        self.assertEqual(metadata["dtype"], "torch.float32")
        self.assertEqual(metadata["torch_version"], "2.test")
        self.assertEqual(metadata["transformers_version"], "5.3.0")
        self.assertEqual(metadata["model_id"], "microsoft/VibeVoice-ASR-HF")
        self.assertEqual(metadata["model_snapshot"], "snapshot-test")
        self.assertEqual(metadata["audio_duration_seconds"], 1.0)
        self.assertFalse(metadata["context_provided"])
        self.assertEqual(metadata["acoustic_tokenizer_chunk_size"], 32000)
        self.assertEqual(metadata["failure_kind"], "mps_out_of_memory")
        self.assertEqual(metadata["mps_current_allocated_memory_bytes"], 123)
        self.assertEqual(metadata["mps_driver_allocated_memory_bytes"], 456)

    def test_malformed_record_rejects_all_model_timing_but_keeps_full_text(self) -> None:
        FakeProcessor.parsed_payload = [
            {
                "Start": 0.0,
                "End": 5.0,
                "Speaker": 0,
                "Content": "第一段。",
            },
            {
                "Start": 5.0,
                "End": "malformed",
                "Speaker": 1,
                "Content": "中間段落。",
            },
            {
                "Start": 10.0,
                "End": 15.0,
                "Speaker": 0,
                "Content": "第三段。",
            },
        ]
        FakeProcessor.transcription_payload = ["第一段。 中間段落。 第三段。"]
        torch_module, transformers_module = fake_runtime()
        with patch.dict(
            "sys.modules",
            {"torch": torch_module, "transformers": transformers_module},
        ):
            result = VibeVoiceBackend(
                model_id="microsoft/VibeVoice-ASR-HF",
                model_path=self.model_path,
            ).transcribe(
                self.audio,
                language="Cantonese",
                profile_text=None,
            )

        self.assertEqual(result.text, "第一段。 中間段落。 第三段。")
        self.assertEqual(result.segments, ())
        self.assertEqual(result.metadata["structured_segments"], [])
        self.assertFalse(result.metadata["structured_parse_valid"])
        self.assertTrue(any("model timestamps" in warning for warning in result.warnings))

    def test_parsed_raw_string_is_not_accepted_as_structured_output(self) -> None:
        raw_markup = 'assistant\n[{"Start":0.0,"End":5.0,"Speaker":0,"Content":"第一段。"}]'
        FakeProcessor.parsed_payload = raw_markup
        FakeProcessor.transcription_payload = ["第一段。"]
        torch_module, transformers_module = fake_runtime()
        with patch.dict(
            "sys.modules",
            {"torch": torch_module, "transformers": transformers_module},
        ):
            result = VibeVoiceBackend(
                model_id="microsoft/VibeVoice-ASR-HF",
                model_path=self.model_path,
            ).transcribe(
                self.audio,
                language="Cantonese",
                profile_text=None,
            )

        self.assertEqual(result.text, "第一段。")
        self.assertEqual(result.segments, ())
        self.assertFalse(result.metadata["structured_parse_valid"])
        self.assertEqual(result.metadata["structured_output"], [raw_markup])

    def test_transcription_only_raw_markup_is_rejected_and_preserved(self) -> None:
        raw_markup = 'assistant\n[{"Start":0.0,"End":5.0,"Speaker":0,"Content":"第一段。"}]'
        FakeProcessor.raw_payload = [raw_markup]
        FakeProcessor.parsed_payload = raw_markup
        FakeProcessor.transcription_payload = [raw_markup]
        torch_module, transformers_module = fake_runtime()
        with patch.dict(
            "sys.modules",
            {"torch": torch_module, "transformers": transformers_module},
        ):
            with self.assertRaisesRegex(BackendError, "raw model markup") as caught:
                VibeVoiceBackend(
                    model_id="microsoft/VibeVoice-ASR-HF",
                    model_path=self.model_path,
                ).transcribe(
                    self.audio,
                    language="Cantonese",
                    profile_text=None,
                )

        self.assertEqual(caught.exception.metadata["raw_output"], raw_markup)
        self.assertEqual(
            caught.exception.metadata["transcription_only_output"],
            [raw_markup],
        )

    def test_mps_is_required_and_no_cpu_fallback_is_used(self) -> None:
        torch_module, transformers_module = fake_runtime(mps_available=False)
        with patch.dict(
            "sys.modules",
            {"torch": torch_module, "transformers": transformers_module},
        ):
            with self.assertRaisesRegex(BackendError, "MPS"):
                VibeVoiceBackend(
                    model_id="microsoft/VibeVoice-ASR-HF",
                    model_path=self.model_path,
                ).transcribe(
                    self.audio,
                    language="Cantonese",
                    profile_text=None,
                )
        self.assertEqual(FakeModel.from_pretrained_calls, [])

    def test_duration_limit_fails_before_model_load(self) -> None:
        torch_module, transformers_module = fake_runtime()
        with (
            patch.dict(
                "sys.modules",
                {"torch": torch_module, "transformers": transformers_module},
            ),
            patch(
                "meeting_recording_processor.asr.vibevoice._wav_duration",
                return_value=MAX_DURATION_SECONDS + 1,
            ),
        ):
            with self.assertRaisesRegex(BackendError, "60 分鐘"):
                VibeVoiceBackend(
                    model_id="microsoft/VibeVoice-ASR-HF",
                    model_path=self.model_path,
                ).transcribe(
                    self.audio,
                    language="Cantonese",
                    profile_text=None,
                )
        self.assertEqual(FakeModel.from_pretrained_calls, [])


if __name__ == "__main__":
    unittest.main()
