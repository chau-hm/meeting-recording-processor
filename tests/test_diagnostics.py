from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from meeting_recording_processor.diagnostics import doctor_report
from meeting_recording_processor.models import ResolvedModel


class DiagnosticsTests(unittest.TestCase):
    def _doctor_report(self, *, torch_module, torch_spec=True):
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)

            def resolve(model_id, _cache):
                return ResolvedModel(model_id, cache_dir / "snapshot", "snapshot-test")

            def find_spec(name):
                if name == "torch" and not torch_spec:
                    return None
                return object()

            with (
                patch(
                    "meeting_recording_processor.diagnostics.resolve_cached_model",
                    side_effect=resolve,
                ),
                patch("meeting_recording_processor.diagnostics.util.find_spec", side_effect=find_spec),
                patch("meeting_recording_processor.diagnostics.platform.system", return_value="Darwin"),
                patch("meeting_recording_processor.diagnostics.platform.machine", return_value="arm64"),
                patch("meeting_recording_processor.diagnostics.shutil.which", return_value="/usr/bin/tool"),
                patch("meeting_recording_processor.diagnostics.importlib.import_module", return_value=torch_module),
            ):
                return doctor_report(cache_dir)

    def test_doctor_reports_vibevoice_runtime_and_model_separately(self) -> None:
        torch_module = SimpleNamespace(
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: True),
            ),
        )
        report = self._doctor_report(torch_module=torch_module)

        self.assertTrue(report["packages"]["torch"]["available"])
        self.assertTrue(report["models"]["vibevoice"]["available"])
        self.assertEqual(report["models"]["vibevoice"]["snapshot"], "snapshot-test")
        self.assertTrue(report["runtime"]["vibevoice-mps"]["available"])
        self.assertTrue(report["healthy"])

    def test_doctor_marks_mps_unavailable_as_unhealthy(self) -> None:
        torch_module = SimpleNamespace(
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: False),
            ),
        )
        report = self._doctor_report(torch_module=torch_module)

        self.assertFalse(report["runtime"]["vibevoice-mps"]["available"])
        self.assertIn("returned false", report["runtime"]["vibevoice-mps"]["detail"])
        self.assertFalse(report["healthy"])

    def test_doctor_handles_missing_torch_without_importing_it(self) -> None:
        report = self._doctor_report(torch_module=object(), torch_spec=False)

        self.assertFalse(report["packages"]["torch"]["available"])
        self.assertFalse(report["runtime"]["vibevoice-mps"]["available"])
        self.assertIn("not installed", report["runtime"]["vibevoice-mps"]["detail"])
        self.assertFalse(report["healthy"])

    def test_doctor_handles_mps_runtime_exception(self) -> None:
        def unavailable():
            raise RuntimeError("MPS probe failed")

        torch_module = SimpleNamespace(
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=unavailable),
            ),
        )
        report = self._doctor_report(torch_module=torch_module)

        self.assertFalse(report["runtime"]["vibevoice-mps"]["available"])
        self.assertIn("MPS capability check failed", report["runtime"]["vibevoice-mps"]["detail"])

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
