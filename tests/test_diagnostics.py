from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from meeting_recording_processor.diagnostics import doctor_report
from meeting_recording_processor.models import ResolvedModel


class DiagnosticsTests(unittest.TestCase):
    def test_doctor_reports_vibevoice_runtime_and_model_separately(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)

            def resolve(model_id, _cache):
                return ResolvedModel(model_id, cache_dir / "snapshot", "snapshot-test")

            with (
                patch(
                    "meeting_recording_processor.diagnostics.resolve_cached_model",
                    side_effect=resolve,
                ),
                patch(
                    "meeting_recording_processor.diagnostics._distribution_version",
                    side_effect=lambda name: f"{name}-version",
                ),
                patch(
                    "meeting_recording_processor.diagnostics.util.find_spec",
                    return_value=object(),
                ),
            ):
                report = doctor_report(cache_dir)

        self.assertTrue(report["packages"]["torch"]["available"])
        self.assertEqual(report["packages"]["transformers"]["version"], "transformers-version")
        self.assertTrue(report["models"]["vibevoice"]["available"])
        self.assertEqual(report["models"]["vibevoice"]["snapshot"], "snapshot-test")

    def test_doctor_distinguishes_missing_vibevoice_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)

            def resolve(model_id, _cache):
                if model_id == "microsoft/VibeVoice-ASR-HF":
                    raise RuntimeError("model missing")
                return ResolvedModel(model_id, cache_dir / "snapshot", "snapshot-test")

            with patch(
                "meeting_recording_processor.diagnostics.resolve_cached_model",
                side_effect=resolve,
            ):
                report = doctor_report(cache_dir)

        self.assertFalse(report["models"]["vibevoice"]["available"])
        self.assertEqual(report["models"]["vibevoice"]["detail"], "model missing")


if __name__ == "__main__":
    unittest.main()
