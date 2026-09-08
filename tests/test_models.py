from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from meeting_recording_processor.models import (
    ModelAssetStatus,
    ModelSetStatus,
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
    def __init__(self, repositories: set[FakeRepository]) -> None:
        self.repos = repositories

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
                ModelAssetStatus(asset, asset.model_id not in missing)
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
            ):
                statuses = inspect_model_sets(cache_dir, (qwen, sensevoice))

        self.assertFalse(statuses[0].installed)
        self.assertTrue(statuses[0].assets[0].installed)
        self.assertFalse(statuses[0].assets[1].installed)
        self.assertFalse(statuses[1].installed)

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
            ):
                result = clear_model_set(qwen, cache_dir, dry_run=True)

            self.assertTrue(unrelated.exists())

        self.assertTrue(result.assets[0].installed)
        self.assertFalse(result.assets[1].installed)
        self.assertEqual(result.expected_freed_bytes, 512)
        self.assertEqual(strategy.execute_calls, 0)
        self.assertTrue(result.dry_run)


if __name__ == "__main__":
    unittest.main()
