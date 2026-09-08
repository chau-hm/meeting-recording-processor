import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from meeting_recording_processor.asr.qwen3 import Qwen3Backend
from meeting_recording_processor.postprocess import postprocess_result


class Qwen3TimingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.audio = self.root / "audio.wav"
        self.audio.write_bytes(b"fixture")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def transcribe(self, result: SimpleNamespace):
        module = SimpleNamespace(transcribe=lambda *_args, **_kwargs: result)
        with patch.dict(sys.modules, {"mlx_qwen3_asr": module}):
            return Qwen3Backend(
                model_id="test/qwen3",
                model_path=self.root / "model",
                aligner_path=self.root / "aligner",
                aligner_snapshot="aligner-snapshot",
            ).transcribe(
                self.audio,
                language="Cantonese",
                profile_text=None,
            )

    @staticmethod
    def result(*, segments, chunks=None) -> SimpleNamespace:
        return SimpleNamespace(
            language="Cantonese",
            text="我哋今日開始開會。",
            segments=segments,
            chunks=chunks,
            finish_reason="stop",
            truncated=False,
        )

    def test_zero_duration_word_is_repaired_only_in_canonical_timing(self) -> None:
        result = self.transcribe(
            self.result(
                segments=[
                    SimpleNamespace(start=0.0, end=0.5, text="我哋"),
                    SimpleNamespace(start=0.5, end=0.5, text="今日"),
                    SimpleNamespace(start=0.5, end=0.9, text="開始開會。"),
                ]
            )
        )

        self.assertEqual(result.raw_segments[1].start, 0.5)
        self.assertEqual(result.raw_segments[1].end, 0.5)
        self.assertEqual(result.segments[1].start, 0.5)
        self.assertEqual(result.segments[1].end, 0.501)
        self.assertEqual(result.metadata["zero_duration_repair_count"], 1)
        self.assertEqual(result.metadata["timestamp_source"], "word_repaired")
        self.assertTrue(any("zero-duration" in warning for warning in result.warnings))

    def test_serious_word_timing_is_rejected_as_a_set_and_chunk_timing_is_used(self) -> None:
        result = self.transcribe(
            self.result(
                segments=[
                    SimpleNamespace(start=0.0, end=0.5, text="第一"),
                    SimpleNamespace(start=0.8, end=1.0, text="第二"),
                    SimpleNamespace(start=0.6, end=0.9, text="第三"),
                ],
                chunks=[
                    SimpleNamespace(start=0.0, end=1.0, text="我哋今日開始開會。"),
                ],
            )
        )

        self.assertEqual(
            [(item.start, item.end) for item in result.raw_segments],
            [(0.0, 0.5), (0.8, 1.0), (0.6, 0.9)],
        )
        self.assertEqual(result.metadata["word_timing_rejected"], True)
        self.assertEqual(result.metadata["chunk_timing_used"], True)
        self.assertEqual(result.segments[0].timing_source, "chunk")
        self.assertTrue(any("整體拒絕" in warning for warning in result.warnings))

    def test_invalid_chunk_timing_leaves_text_for_duration_estimation(self) -> None:
        result = self.transcribe(
            self.result(
                segments=[
                    SimpleNamespace(start=0.0, end=0.5, text="第一"),
                    SimpleNamespace(start=0.8, end=1.0, text="第二"),
                    SimpleNamespace(start=0.6, end=0.9, text="第三"),
                ],
                chunks=[
                    SimpleNamespace(start=0.0, end=0.0, text="我哋今日開始開會。"),
                ],
            )
        )

        self.assertEqual(result.segments, ())
        self.assertEqual(result.metadata["timestamp_source"], "none")
        self.assertTrue(any("chunk timing" in warning for warning in result.warnings))
        _, segments, warnings = postprocess_result(
            result,
            audio_duration=5.0,
            converter=lambda value: value,
        )
        self.assertTrue(segments)
        self.assertTrue(
            all(item.timing_source == "estimated_from_duration" for item in segments)
        )
        self.assertTrue(warnings)

    def test_non_finite_and_malformed_timing_is_diagnostic_and_json_safe(self) -> None:
        result = self.transcribe(
            self.result(
                segments=[
                    SimpleNamespace(start=0.0, end=0.5, text="第一"),
                    SimpleNamespace(start=float("nan"), end=float("inf"), text="第二"),
                    SimpleNamespace(start=None, end=1.0, text="第三"),
                ]
            )
        )

        self.assertEqual(len(result.raw_segments), 1)
        self.assertEqual(result.segments, ())
        self.assertEqual(len(result.metadata["raw_timing_diagnostics"]), 2)
        json.dumps(result.metadata, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
