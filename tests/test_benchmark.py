import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from meeting_recording_processor.benchmark import run_benchmark
from meeting_recording_processor.config import BenchmarkConfig, AsrMode
from meeting_recording_processor.errors import OutputExistsError, TranscriptionFailed
from meeting_recording_processor.media.probe import MediaMetadata
from meeting_recording_processor.models import (
    ModelAssetStatus,
    ModelSetStatus,
    build_model_inventory,
)
from meeting_recording_processor.schema_io import write_package


def completed_package(backend: str, text: str) -> dict:
    return {
        "schema_version": "1.0",
        "status": "completed",
        "run_id": f"run-{backend}",
        "attempts": [
            {
                "attempt_id": f"attempt-1-{backend}",
                "backend": backend,
                "model": f"test/{backend}",
                "status": "completed",
                "raw_segments": [],
            }
        ],
        "selected_attempt_id": f"attempt-1-{backend}",
        "transcript": {
            "language": "Cantonese",
            "text": text,
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": text,
                    "timing_source": "model",
                }
            ],
        },
    }


class BenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input = self.root / "meeting.m4a"
        self.input.write_bytes(b"fixture")
        self.inventory = build_model_inventory()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def config(self, *, include_experimental: bool = False, overwrite: bool = False):
        return BenchmarkConfig(
            input_path=self.input,
            output_dir=self.root / "output",
            work_dir=self.root / "work",
            cache_dir=self.root / "cache",
            include_experimental=include_experimental,
            overwrite=overwrite,
        )

    def statuses(self, *, missing: set[str] = set()) -> tuple[ModelSetStatus, ...]:
        return tuple(
            ModelSetStatus(
                model_set=model_set,
                assets=tuple(
                    ModelAssetStatus(asset, asset.model_id not in missing)
                    for asset in model_set.assets
                ),
            )
            for model_set in self.inventory
        )

    def execute(
        self,
        *,
        statuses: tuple[ModelSetStatus, ...] | None = None,
        extractor=None,
        include_experimental: bool = False,
        overwrite: bool = False,
        clock_values: tuple[float, ...] = (0.0, 10.0, 10.0, 15.0),
    ):
        calls = []
        if extractor is None:
            def extractor(config):
                calls.append(config)
                config.output_dir.mkdir(parents=True, exist_ok=True)
                write_package(
                    config.output_path,
                    completed_package(config.asr_mode.value, f"{config.asr_mode.value} text"),
                )
                return SimpleNamespace(output_path=config.output_path)

        clock_values_iter = iter(clock_values)
        result = run_benchmark(
            self.config(
                include_experimental=include_experimental,
                overwrite=overwrite,
            ),
            extractor=extractor,
            media_probe=lambda _path: MediaMetadata("m4a", 10.0, (), 0),
            model_status_inspector=lambda _cache, _inventory: (
                statuses if statuses is not None else self.statuses()
            ),
            runtime_checker=lambda _model_set: None,
            clock=lambda: next(clock_values_iter),
        )
        return result, calls

    def test_runs_production_backends_explicitly_and_writes_isolated_report(self) -> None:
        result, calls = self.execute()

        self.assertEqual(
            [config.asr_mode for config in calls],
            [AsrMode.QWEN3, AsrMode.SENSEVOICE],
        )
        self.assertEqual(
            [config.output_dir.name for config in calls],
            ["qwen3", "sensevoice"],
        )
        self.assertEqual(result.results[0]["status"], "passed")
        self.assertEqual(result.results[1]["status"], "passed")
        self.assertEqual(result.results[2]["status"], "skipped")
        self.assertEqual(result.results[0]["elapsed_seconds"], 10.0)
        self.assertEqual(result.results[0]["realtime_factor"], 1.0)
        self.assertEqual(result.results[0]["character_count"], len("qwen3 text"))
        self.assertEqual(
            result.results[0]["output"],
            "qwen3/meeting.transcript.json",
        )
        report = json.loads(result.report_path.read_text(encoding="utf-8"))
        self.assertEqual(
            [item["backend"] for item in report["results"]],
            ["qwen3", "sensevoice", "vibevoice"],
        )
        self.assertEqual(result.passed_count, 2)

    def test_missing_qwen_does_not_block_sensevoice(self) -> None:
        missing_qwen = {
            self.inventory[0].assets[0].model_id,
            self.inventory[0].assets[1].model_id,
        }
        result, calls = self.execute(statuses=self.statuses(missing=missing_qwen))

        self.assertEqual([config.asr_mode for config in calls], [AsrMode.SENSEVOICE])
        self.assertEqual(result.results[0]["status"], "skipped")
        self.assertIn("Qwen/Qwen3-ASR-1.7B", result.results[0]["reason"])
        self.assertEqual(result.results[1]["status"], "passed")

    def test_failed_backend_does_not_stop_later_backend(self) -> None:
        calls = []

        def extractor(config):
            calls.append(config.asr_mode)
            if config.asr_mode is AsrMode.QWEN3:
                raise TranscriptionFailed("qwen failed")
            config.output_dir.mkdir(parents=True, exist_ok=True)
            write_package(
                config.output_path,
                completed_package("sensevoice", "sensevoice text"),
            )
            return SimpleNamespace(output_path=config.output_path)

        result, _unused = self.execute(
            extractor=extractor,
            clock_values=(0.0, 2.0, 2.0, 5.0),
        )

        self.assertEqual(calls, [AsrMode.QWEN3, AsrMode.SENSEVOICE])
        self.assertEqual(result.results[0]["status"], "failed")
        self.assertEqual(result.results[1]["status"], "passed")
        self.assertEqual(result.passed_count, 1)

    def test_include_experimental_makes_cached_vibevoice_eligible(self) -> None:
        result, calls = self.execute(
            include_experimental=True,
            clock_values=(0, 1, 1, 2, 2, 3),
        )

        self.assertEqual(
            [config.asr_mode for config in calls],
            [AsrMode.QWEN3, AsrMode.SENSEVOICE, AsrMode.VIBEVOICE],
        )
        self.assertEqual(result.results[2]["status"], "passed")
        self.assertEqual(result.results[2]["output"], "vibevoice/meeting.transcript.json")

    def test_include_experimental_still_skips_missing_vibevoice(self) -> None:
        missing = {self.inventory[2].assets[0].model_id}
        result, calls = self.execute(
            statuses=self.statuses(missing=missing),
            include_experimental=True,
        )

        self.assertEqual(
            [config.asr_mode for config in calls],
            [AsrMode.QWEN3, AsrMode.SENSEVOICE],
        )
        self.assertEqual(result.results[2]["status"], "skipped")
        self.assertIn("microsoft/VibeVoice-ASR-HF", result.results[2]["reason"])

    def test_existing_output_requires_overwrite_before_inference(self) -> None:
        destination = (
            self.root
            / "output"
            / "benchmark"
            / "meeting"
            / "qwen3"
            / "meeting.transcript.json"
        )
        destination.parent.mkdir(parents=True)
        destination.write_text("existing", encoding="utf-8")
        calls = []

        def extractor(config):
            calls.append(config)
            raise AssertionError("inference should not start")

        with self.assertRaises(OutputExistsError):
            self.execute(extractor=extractor)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
