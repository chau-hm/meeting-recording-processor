"""ASR model inventory, cache resolution, and lifecycle operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_QWEN_ALIGNER_MODEL,
    DEFAULT_QWEN_MODEL,
    DEFAULT_SENSEVOICE_MODEL,
    DEFAULT_VIBEVOICE_MODEL,
)
from .errors import ModelUnavailableError
from .runtime import configure_huggingface_cache


@dataclass(frozen=True, slots=True)
class ModelAsset:
    """One Hugging Face repository required by an ASR model set."""

    name: str
    model_id: str


@dataclass(frozen=True, slots=True)
class AsrModelSet:
    """The complete local assets and runtime hints for one backend."""

    backend: str
    assets: tuple[ModelAsset, ...]
    experimental: bool = False
    runtime_modules: tuple[str, ...] = ()
    requires_mps: bool = False

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(asset.model_id for asset in self.assets)


@dataclass(frozen=True, slots=True)
class ModelAssetStatus:
    asset: ModelAsset
    installed: bool
    size_bytes: int = 0
    revision_count: int = 0


@dataclass(frozen=True, slots=True)
class ModelSetStatus:
    model_set: AsrModelSet
    assets: tuple[ModelAssetStatus, ...]

    @property
    def installed(self) -> bool:
        return all(asset.installed for asset in self.assets)

    @property
    def missing_assets(self) -> tuple[ModelAsset, ...]:
        return tuple(asset.asset for asset in self.assets if not asset.installed)


@dataclass(frozen=True, slots=True)
class ClearModelResult:
    model_set: AsrModelSet
    assets: tuple[ModelAssetStatus, ...]
    expected_freed_bytes: int
    freed_bytes: int
    before_cache_size: int
    after_cache_size: int
    dry_run: bool


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    model_id: str
    path: Path
    snapshot: str | None


def _snapshot_name(path: Path) -> str | None:
    if path.parent.name == "snapshots":
        return path.name
    return None


def build_model_inventory(
    *,
    qwen_model: str = DEFAULT_QWEN_MODEL,
    qwen_aligner_model: str = DEFAULT_QWEN_ALIGNER_MODEL,
    sensevoice_model: str = DEFAULT_SENSEVOICE_MODEL,
    vibevoice_model: str = DEFAULT_VIBEVOICE_MODEL,
) -> tuple[AsrModelSet, ...]:
    """Build the canonical inventory while applying all CLI model overrides."""

    return (
        AsrModelSet(
            backend="qwen3",
            assets=(
                ModelAsset("qwen3", qwen_model),
                ModelAsset("qwen3-aligner", qwen_aligner_model),
            ),
            runtime_modules=("mlx_qwen3_asr",),
        ),
        AsrModelSet(
            backend="sensevoice",
            assets=(ModelAsset("sensevoice", sensevoice_model),),
            runtime_modules=("mlx_audio",),
        ),
        AsrModelSet(
            backend="vibevoice",
            assets=(ModelAsset("vibevoice", vibevoice_model),),
            experimental=True,
            runtime_modules=("torch", "transformers"),
            requires_mps=True,
        ),
    )


def select_model_sets(
    inventory: tuple[AsrModelSet, ...],
    selection: str,
) -> tuple[AsrModelSet, ...]:
    if selection == "all":
        return inventory
    selected = tuple(model_set for model_set in inventory if model_set.backend == selection)
    if not selected:
        raise ValueError(f"Unknown ASR model set: {selection}")
    return selected


def _scan_cache_info(cache_dir: Path) -> Any | None:
    hub_dir = cache_dir.expanduser().resolve() / "hub"
    if not hub_dir.exists():
        return None
    if not hub_dir.is_dir():
        raise ModelUnavailableError(f"Hugging Face cache path is not a directory: {hub_dir}")

    from huggingface_hub import scan_cache_dir

    return scan_cache_dir(hub_dir)


def _model_repositories(cache_info: Any | None) -> dict[str, Any]:
    if cache_info is None:
        return {}
    return {
        repository.repo_id: repository
        for repository in cache_info.repos
        if getattr(repository, "repo_type", None) == "model"
    }


def _revision_hashes(repository: Any) -> tuple[str, ...]:
    return tuple(
        sorted(
            revision.commit_hash
            for revision in getattr(repository, "revisions", ())
            if isinstance(getattr(revision, "commit_hash", None), str)
            and revision.commit_hash
        )
    )


def inspect_model_sets(
    cache_dir: Path,
    model_sets: tuple[AsrModelSet, ...],
) -> tuple[ModelSetStatus, ...]:
    """Inspect exact cached repositories without resolving or downloading models."""

    cache_info = _scan_cache_info(cache_dir)
    repositories = _model_repositories(cache_info)
    statuses: list[ModelSetStatus] = []
    for model_set in model_sets:
        assets: list[ModelAssetStatus] = []
        for asset in model_set.assets:
            repository = repositories.get(asset.model_id)
            revisions = _revision_hashes(repository) if repository is not None else ()
            size_bytes = int(getattr(repository, "size_on_disk", 0) or 0) if revisions else 0
            assets.append(
                ModelAssetStatus(
                    asset=asset,
                    installed=bool(revisions),
                    size_bytes=max(size_bytes, 0),
                    revision_count=len(revisions),
                )
            )
        statuses.append(ModelSetStatus(model_set=model_set, assets=tuple(assets)))
    return tuple(statuses)


def inspect_model_set(cache_dir: Path, model_set: AsrModelSet) -> ModelSetStatus:
    return inspect_model_sets(cache_dir, (model_set,))[0]


def resolve_cached_model(model_id: str, cache_dir: Path) -> ResolvedModel:
    configure_huggingface_cache(cache_dir, offline=True)
    try:
        from huggingface_hub import snapshot_download

        resolved = Path(
            snapshot_download(
                repo_id=model_id,
                cache_dir=str((cache_dir / "hub").resolve()),
                local_files_only=True,
            )
        ).resolve()
    except Exception as exc:
        raise ModelUnavailableError(
            f"本機 cache 未有完整 model：{model_id}；請先執行 download-model"
        ) from exc
    return ResolvedModel(model_id=model_id, path=resolved, snapshot=_snapshot_name(resolved))


def download_model(model_id: str, cache_dir: Path) -> ResolvedModel:
    configure_huggingface_cache(cache_dir, offline=False)
    from huggingface_hub import snapshot_download

    resolved = Path(
        snapshot_download(repo_id=model_id, cache_dir=str((cache_dir / "hub").resolve()))
    ).resolve()
    return ResolvedModel(model_id=model_id, path=resolved, snapshot=_snapshot_name(resolved))


def clear_model_set(
    model_set: AsrModelSet,
    cache_dir: Path,
    *,
    dry_run: bool = False,
) -> ClearModelResult:
    """Remove all cached revisions for exactly the repositories in ``model_set``."""

    cache_info = _scan_cache_info(cache_dir)
    repositories = _model_repositories(cache_info)
    assets: list[ModelAssetStatus] = []
    planned_repositories: set[str] = set()
    revisions_to_delete: list[str] = []

    for asset in model_set.assets:
        repository = repositories.get(asset.model_id)
        revisions = _revision_hashes(repository) if repository is not None else ()
        size_bytes = int(getattr(repository, "size_on_disk", 0) or 0) if revisions else 0
        assets.append(
            ModelAssetStatus(
                asset=asset,
                installed=bool(revisions),
                size_bytes=max(size_bytes, 0),
                revision_count=len(revisions),
            )
        )
        if not revisions or asset.model_id in planned_repositories:
            continue
        revisions_to_delete.extend(revisions)
        planned_repositories.add(asset.model_id)

    strategies: list[Any] = []
    expected_freed_bytes = 0
    if cache_info is not None and revisions_to_delete:
        strategy = cache_info.delete_revisions(*revisions_to_delete)
        strategies.append(strategy)
        expected_freed_bytes = max(int(strategy.expected_freed_size), 0)

    before_cache_size = directory_size(cache_dir)
    if not dry_run:
        for strategy in strategies:
            strategy.execute()
    after_cache_size = directory_size(cache_dir)
    freed_bytes = (
        0
        if dry_run
        else max(before_cache_size - after_cache_size, 0)
    )
    return ClearModelResult(
        model_set=model_set,
        assets=tuple(assets),
        expected_freed_bytes=expected_freed_bytes,
        freed_bytes=freed_bytes,
        before_cache_size=before_cache_size,
        after_cache_size=after_cache_size,
        dry_run=dry_run,
    )


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{size} B"
