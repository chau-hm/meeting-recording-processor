from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from meeting_recording_processor.models import (
    ModelAssetStatus,
    ModelSetStatus,
    ResolvedModel,
    build_model_inventory,
    clear_model_set,
    inspect_model_sets,
)


class FakeStrategy:
    def __init__(self, expected_freed_size: int) -> None:
        self.expected_freed_size = expected_freed_size
        self.execute_calls = 0

    def execute(self) -> None:
        self.execute_calls += 1


class CombinedStrategy(FakeStrategy):
    def __init__(self, strategies: list[FakeStrategy]) -> None:
        super().__init__(sum(strategy.expected_freed_size for strategy in strategies))
        self.strategies = strategies

    def execute(self) -> None:
        super().execute()
        for strategy in self.strategies:
            strategy.execute()


@dataclass(frozen=True)
class FakeRevision:
    commit_hash: str


class FakeRepository:
    def __init__(self, repo_id: str, revisions: tuple[str, ...], strategy: FakeStrategy) -> None:
        self.repo_id = repo_id
        self.repo_type = "model"
        self.revisions = frozenset(FakeRevision(revision) for revision in revisions)
        self.size_on_disk = 100
        self.strategy = strategy
        self.delete_calls: list[tuple[str, ...]] = []

    def delete_revisions(self, *revisions: str) -> FakeStrategy:
        self.delete_calls.append(revisions)
        return self.strategy


class FakeCacheInfo:
    def __init__(
        self,
        repositories: set[FakeRepository],
        incomplete_files: tuple[object, ...] = (),
    ) -> None:
        self.repos = repositories
        self.incomplete_files = incomplete_files

    def delete_revisions(self, *revisions: str) -> FakeStrategy:
        selected: list[FakeStrategy] = []
        revision_set = set(revisions)
        for repository in self.repos:
            matching = tuple(
                revision.commit_hash
                for revision in repository.revisions
                if revision.commit_hash in revision_set
            )
            if matching:
                selected.append(repository.delete_revisions(*sorted(matching)))

        return CombinedStrategy(selected)


def _statuses(inventory, *, missing: set[str] = set()) -> tuple[ModelSetStatus, ...]:
    return tuple(
        ModelSetStatus(
            model_set=model_set,
            assets=tuple(
                ModelAssetStatus(
                    asset=asset,
                    cached=asset.model_id not in missing,
                    complete=asset.model_id not in missing,
                )
                for asset in model_set.assets
            ),
        )
        for model_set in inventory
    )


class ModelLifecycleTests(unittest.TestCase):
    def test_inventory_contains_complete_qwen_set_and_applies_overrides(self) -> None:
        inventory = build_model_inventory(
            qwen_model="local/qwen",
            qwen_aligner_model="local/aligner",
            sensevoice_model="local/sensevoice",
            vibevoice_model="local/vibevoice",
        )

        self.assertEqual([model_set.backend for model_set in inventory], [
            "qwen3",
            "sensevoice",
            "vibevoice",
        ])
        self.assertEqual(inventory[0].model_ids, ("local/qwen", "local/aligner"))
        self.assertEqual(inventory[1].model_ids, ("local/sensevoice",))
        self.assertEqual(inventory[2].model_ids, ("local/vibevoice",))
        self.assertTrue(inventory[2].experimental)

    def test_incomplete_qwen_set_is_not_installed(self) -> None:
        inventory = build_model_inventory()
        qwen, sensevoice, _vibevoice = inventory
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            (cache_dir / "hub").mkdir()
            asr_repository = FakeRepository(
                qwen.assets[0].model_id,
                ("asr-revision",),
                FakeStrategy(10),
            )
            with patch(
                "huggingface_hub.scan_cache_dir",
                return_value=SimpleNamespace(repos={asr_repository}),
            ), patch(
                "meeting_recording_processor.models.resolve_cached_model",
                return_value=ResolvedModel(
                    qwen.assets[0].model_id,
                    cache_dir / "hub" / "snapshot",
                    "asr-revision",
                ),
            ):
                statuses = inspect_model_sets(cache_dir, (qwen, sensevoice))

        self.assertFalse(statuses[0].installed)
        self.assertTrue(statuses[0].assets[0].installed)
        self.assertFalse(statuses[0].assets[1].installed)
        self.assertFalse(statuses[1].installed)

    def test_model_completeness_resolution_is_local_only(self) -> None:
        sensevoice = build_model_inventory()[1]
        repository = FakeRepository(
            sensevoice.assets[0].model_id,
            ("sensevoice-revision",),
            FakeStrategy(10),
        )

        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            (cache_dir / "hub").mkdir()
            with (
                patch(
                    "huggingface_hub.scan_cache_dir",
                    return_value=SimpleNamespace(repos={repository}),
                ),
                patch(
                    "huggingface_hub.snapshot_download",
                    return_value=str(cache_dir / "hub" / "snapshot"),
                ) as snapshot_download,
                patch(
                    "meeting_recording_processor.models.download_model",
                    side_effect=AssertionError("model inspection must not download"),
                ),
            ):
                status = inspect_model_sets(cache_dir, (sensevoice,))[0]

        self.assertTrue(status.installed)
        snapshot_download.assert_called_once_with(
            repo_id=sensevoice.assets[0].model_id,
            cache_dir=str((cache_dir / "hub").resolve()),
            local_files_only=True,
        )

    def test_clear_model_deletes_only_selected_repositories_and_all_revisions(self) -> None:
        inventory = build_model_inventory()
        qwen, sensevoice, _vibevoice = inventory
        qwen_strategy = FakeStrategy(123)
        sensevoice_strategy = FakeStrategy(456)
        unrelated_strategy = FakeStrategy(789)
        qwen_repository = FakeRepository(
            qwen.assets[0].model_id,
            ("qwen-b", "qwen-a"),
            qwen_strategy,
        )
        aligner_repository = FakeRepository(
            qwen.assets[1].model_id,
            ("aligner-a",),
            FakeStrategy(234),
        )
        sensevoice_repository = FakeRepository(
            sensevoice.assets[0].model_id,
            ("sense-a",),
            sensevoice_strategy,
        )
        unrelated_repository = FakeRepository(
            "unrelated/model",
            ("unrelated-a",),
            unrelated_strategy,
        )

        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            (cache_dir / "hub").mkdir()
            with patch(
                "huggingface_hub.scan_cache_dir",
                return_value=FakeCacheInfo(
                    {
                        qwen_repository,
                        aligner_repository,
                        sensevoice_repository,
                        unrelated_repository,
                    }
                ),
            ), patch(
                "meeting_recording_processor.models.resolve_cached_model",
                side_effect=lambda model_id, _cache_dir: ResolvedModel(
                    model_id,
                    cache_dir / "hub" / "snapshot",
                    "snapshot",
                ),
            ):
                result = clear_model_set(qwen, cache_dir)

        self.assertEqual(qwen_repository.delete_calls, [("qwen-a", "qwen-b")])
        self.assertEqual(aligner_repository.delete_calls, [("aligner-a",)])
        self.assertEqual(sensevoice_repository.delete_calls, [])
        self.assertEqual(unrelated_repository.delete_calls, [])
        self.assertEqual(qwen_strategy.execute_calls, 1)
        self.assertEqual(aligner_repository.strategy.execute_calls, 1)
        self.assertEqual(result.expected_freed_bytes, 357)

    def test_clear_model_dry_run_does_not_execute_deletion_and_reports_absence(self) -> None:
        inventory = build_model_inventory()
        qwen = inventory[0]
        strategy = FakeStrategy(512)
        asr_repository = FakeRepository(qwen.assets[0].model_id, ("asr-a",), strategy)

        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            (cache_dir / "hub").mkdir()
            unrelated = cache_dir / "unrelated.txt"
            unrelated.write_bytes(b"keep")
            with patch(
                "huggingface_hub.scan_cache_dir",
                return_value=FakeCacheInfo({asr_repository}),
            ), patch(
                "meeting_recording_processor.models.resolve_cached_model",
                return_value=ResolvedModel(
                    qwen.assets[0].model_id,
                    cache_dir / "hub" / "snapshot",
                    "asr-a",
                ),
            ):
                result = clear_model_set(qwen, cache_dir, dry_run=True)

            self.assertTrue(unrelated.exists())

        self.assertTrue(result.assets[0].installed)
        self.assertFalse(result.assets[1].installed)
        self.assertEqual(result.expected_freed_bytes, 512)
        self.assertEqual(strategy.execute_calls, 0)
        self.assertTrue(result.dry_run)

    def test_clear_model_removes_selected_incomplete_files_only(self) -> None:
        vibevoice = build_model_inventory()[2]

        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            hub_dir = cache_dir / "hub"
            selected = (
                hub_dir
                / "models--microsoft--VibeVoice-ASR-HF"
                / "blobs"
                / "selected.incomplete"
            )
            unrelated = (
                hub_dir
                / "models--unrelated--model"
                / "blobs"
                / "unrelated.incomplete"
            )
            selected.parent.mkdir(parents=True)
            unrelated.parent.mkdir(parents=True)
            selected.write_bytes(b"selected")
            unrelated.write_bytes(b"unrelated")
            cache_info = SimpleNamespace(
                repos=(),
                incomplete_files=(
                    SimpleNamespace(file_path=selected, size_on_disk=selected.stat().st_size),
                    SimpleNamespace(file_path=unrelated, size_on_disk=unrelated.stat().st_size),
                ),
            )
            with patch(
                "huggingface_hub.scan_cache_dir",
                return_value=cache_info,
            ):
                result = clear_model_set(vibevoice, cache_dir)

            self.assertFalse(selected.exists())
            self.assertTrue(unrelated.exists())

        self.assertFalse(result.assets[0].cached)
        self.assertFalse(result.assets[0].installed)
        self.assertEqual(result.assets[0].incomplete_file_count, 1)
        self.assertEqual(result.expected_freed_bytes, len(b"selected"))

    def test_clear_model_dry_run_includes_selected_incomplete_size(self) -> None:
        sensevoice = build_model_inventory()[1]

        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            selected = (
                cache_dir
                / "hub"
                / "models--mlx-community--SenseVoiceSmall"
                / "blobs"
                / "selected.incomplete"
            )
            selected.parent.mkdir(parents=True)
            selected.write_bytes(b"partial model")
            with patch(
                "huggingface_hub.scan_cache_dir",
                return_value=SimpleNamespace(
                    repos=(),
                    incomplete_files=(
                        SimpleNamespace(
                            file_path=selected,
                            size_on_disk=selected.stat().st_size,
                        ),
                    ),
                ),
            ):
                result = clear_model_set(sensevoice, cache_dir, dry_run=True)

            self.assertTrue(selected.exists())

        self.assertEqual(result.expected_freed_bytes, len(b"partial model"))
        self.assertEqual(result.assets[0].incomplete_file_count, 1)
        self.assertEqual(
            result.assets[0].incomplete_size_bytes,
            len(b"partial model"),
        )
        self.assertEqual(result.freed_bytes, 0)

    def test_clear_model_qwen_mixed_complete_and_incomplete_assets_isolated(self) -> None:
        qwen = build_model_inventory()[0]
        aligner_strategy = FakeStrategy(23)
        aligner_repository = FakeRepository(
            qwen.assets[1].model_id,
            ("aligner-a",),
            aligner_strategy,
        )
        asr_repository = FakeRepository(
            qwen.assets[0].model_id,
            ("asr-a",),
            FakeStrategy(31),
        )

        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            selected = (
                cache_dir
                / "hub"
                / "models--Qwen--Qwen3-ASR-1.7B"
                / "blobs"
                / "selected.incomplete"
            )
            unrelated = (
                cache_dir
                / "hub"
                / "models--other--model"
                / "blobs"
                / "unrelated.incomplete"
            )
            selected.parent.mkdir(parents=True)
            unrelated.parent.mkdir(parents=True)
            selected.write_bytes(b"qwen partial")
            unrelated.write_bytes(b"keep")
            with (
                patch(
                    "huggingface_hub.scan_cache_dir",
                    return_value=FakeCacheInfo(
                        {asr_repository, aligner_repository},
                        incomplete_files=(
                            SimpleNamespace(
                                file_path=selected,
                                size_on_disk=selected.stat().st_size,
                            ),
                            SimpleNamespace(
                                file_path=unrelated,
                                size_on_disk=unrelated.stat().st_size,
                            ),
                        ),
                    ),
                ),
                patch(
                    "meeting_recording_processor.models.resolve_cached_model",
                    return_value=ResolvedModel(
                        qwen.assets[0].model_id,
                        cache_dir / "hub" / "snapshot",
                        "snapshot",
                    ),
                ),
            ):
                result = clear_model_set(qwen, cache_dir)

            self.assertFalse(selected.exists())
            self.assertTrue(unrelated.exists())

        self.assertEqual(
            result.expected_freed_bytes,
            31 + 23 + len(b"qwen partial"),
        )
        self.assertEqual(aligner_repository.delete_calls, [("aligner-a",)])
        self.assertEqual(aligner_strategy.execute_calls, 1)

    def test_clear_model_partial_qwen_set_removes_aligner_and_asr_incomplete_file(self) -> None:
        qwen = build_model_inventory()[0]
        aligner_strategy = FakeStrategy(17)
        aligner_repository = FakeRepository(
            qwen.assets[1].model_id,
            ("aligner-a",),
            aligner_strategy,
        )

        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            selected = (
                cache_dir
                / "hub"
                / "models--Qwen--Qwen3-ASR-1.7B"
                / "blobs"
                / "asr.incomplete"
            )
            selected.parent.mkdir(parents=True)
            selected.write_bytes(b"asr partial")
            with (
                patch(
                    "huggingface_hub.scan_cache_dir",
                    return_value=FakeCacheInfo(
                        {aligner_repository},
                        incomplete_files=(
                            SimpleNamespace(
                                file_path=selected,
                                size_on_disk=selected.stat().st_size,
                            ),
                        ),
                    ),
                ),
                patch(
                    "meeting_recording_processor.models.resolve_cached_model",
                    return_value=ResolvedModel(
                        qwen.assets[1].model_id,
                        cache_dir / "hub" / "snapshot",
                        "snapshot",
                    ),
                ),
            ):
                result = clear_model_set(qwen, cache_dir)

            self.assertFalse(selected.exists())

        self.assertFalse(result.assets[0].cached)
        self.assertEqual(result.assets[0].incomplete_file_count, 1)
        self.assertTrue(result.assets[1].cached)
        self.assertEqual(aligner_strategy.execute_calls, 1)
        self.assertEqual(
            result.expected_freed_bytes,
            17 + len(b"asr partial"),
        )


if __name__ == "__main__":
    unittest.main()
