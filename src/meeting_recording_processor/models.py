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
    cached: bool
    complete: bool | None = None
    size_bytes: int = 0
    revision_count: int = 0
    incomplete_size_bytes: int = 0
    incomplete_file_count: int = 0

    @property
    def installed(self) -> bool:
        """Whether the asset resolves completely from the local cache."""

        return self.cached if self.complete is None else self.complete


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


@dataclass(frozen=True, slots=True)
class _IncompleteCacheFile:
    path: Path
    size_bytes: int


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


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _selected_repository_paths(
    cache_dir: Path,
    cache_info: Any | None,
    model_ids: tuple[str, ...],
) -> dict[str, Path]:
    hub_dir = (cache_dir.expanduser().resolve() / "hub").resolve()
    repositories = _model_repositories(cache_info)
    paths: dict[str, Path] = {}

    from huggingface_hub.file_download import repo_folder_name

    for model_id in model_ids:
        repository = repositories.get(model_id)
        if repository is None:
            repository_path = hub_dir / repo_folder_name(
                repo_id=model_id,
                repo_type="model",
            )
        else:
            repository_path_value = getattr(repository, "repo_path", None)
            repository_path = (
                Path(repository_path_value)
                if repository_path_value is not None
                else hub_dir
                / repo_folder_name(repo_id=model_id, repo_type="model")
            )
        try:
            resolved_repository_path = repository_path.resolve(strict=False)
        except (OSError, RuntimeError):
            continue
        if _path_is_within(resolved_repository_path, hub_dir):
            paths[model_id] = resolved_repository_path
    return paths


def _incomplete_files_by_model(
    cache_dir: Path,
    cache_info: Any | None,
    model_ids: tuple[str, ...],
) -> dict[str, tuple[_IncompleteCacheFile, ...]]:
    if cache_info is None:
        return {}

    hub_dir = (cache_dir.expanduser().resolve() / "hub").resolve()
    repository_paths = _selected_repository_paths(cache_dir, cache_info, model_ids)
    incomplete_by_model: dict[str, list[_IncompleteCacheFile]] = {
        model_id: [] for model_id in model_ids
    }
    seen_paths: set[Path] = set()

    for item in getattr(cache_info, "incomplete_files", ()):
        raw_path = Path(getattr(item, "file_path", ""))
        if not raw_path.is_absolute() or raw_path.is_symlink():
            continue
        try:
            resolved_path = raw_path.resolve(strict=False)
        except (OSError, RuntimeError):
            continue
        if (
            not resolved_path.name.endswith(".incomplete")
            or not resolved_path.is_file()
            or not _path_is_within(resolved_path, hub_dir)
        ):
            continue

        for model_id, repository_path in repository_paths.items():
            blobs_path = repository_path / "blobs"
            if resolved_path.parent != blobs_path or resolved_path in seen_paths:
                continue
            size_bytes = max(int(getattr(item, "size_on_disk", 0) or 0), 0)
            incomplete_by_model[model_id].append(
                _IncompleteCacheFile(path=raw_path, size_bytes=size_bytes)
            )
            seen_paths.add(resolved_path)
            break

    return {
        model_id: tuple(sorted(files, key=lambda item: str(item.path)))
        for model_id, files in incomplete_by_model.items()
        if files
    }


def _asset_status(
    asset: ModelAsset,
    repository: Any | None,
    incomplete_files: tuple[_IncompleteCacheFile, ...],
    cache_dir: Path,
) -> ModelAssetStatus:
    revisions = _revision_hashes(repository) if repository is not None else ()
    cached = bool(revisions)
    complete = False
    if cached:
        try:
            resolve_cached_model(asset.model_id, cache_dir)
        except ModelUnavailableError:
            pass
        else:
            complete = True
    size_bytes = int(getattr(repository, "size_on_disk", 0) or 0) if cached else 0
    return ModelAssetStatus(
        asset=asset,
        cached=cached,
        complete=complete,
        size_bytes=max(size_bytes, 0),
        revision_count=len(revisions),
        incomplete_size_bytes=sum(item.size_bytes for item in incomplete_files),
        incomplete_file_count=len(incomplete_files),
    )


def inspect_model_sets(
    cache_dir: Path,
    model_sets: tuple[AsrModelSet, ...],
) -> tuple[ModelSetStatus, ...]:
    """Inspect cache presence and complete local model resolution without downloading."""

    cache_info = _scan_cache_info(cache_dir)
    repositories = _model_repositories(cache_info)
    model_ids = tuple(
        asset.model_id for model_set in model_sets for asset in model_set.assets
    )
    incomplete_by_model = _incomplete_files_by_model(cache_dir, cache_info, model_ids)
    statuses: list[ModelSetStatus] = []
    for model_set in model_sets:
        assets: list[ModelAssetStatus] = []
        for asset in model_set.assets:
            repository = repositories.get(asset.model_id)
            assets.append(
                _asset_status(
                    asset,
                    repository,
                    incomplete_by_model.get(asset.model_id, ()),
                    cache_dir,
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
    """Remove cached revisions and incomplete downloads for exactly ``model_set``."""

    cache_info = _scan_cache_info(cache_dir)
    repositories = _model_repositories(cache_info)
    incomplete_by_model = _incomplete_files_by_model(
        cache_dir,
        cache_info,
        tuple(asset.model_id for asset in model_set.assets),
    )
    assets: list[ModelAssetStatus] = []
    planned_repositories: set[str] = set()
    revisions_to_delete: list[str] = []
    incomplete_files: list[_IncompleteCacheFile] = []

    for asset in model_set.assets:
        repository = repositories.get(asset.model_id)
        revisions = _revision_hashes(repository) if repository is not None else ()
        selected_incomplete_files = incomplete_by_model.get(asset.model_id, ())
        assets.append(
            _asset_status(
                asset,
                repository,
                selected_incomplete_files,
                cache_dir,
            )
        )
        if revisions and asset.model_id not in planned_repositories:
            revisions_to_delete.extend(revisions)
            planned_repositories.add(asset.model_id)
        incomplete_files.extend(selected_incomplete_files)

    strategies: list[Any] = []
    expected_freed_bytes = 0
    if cache_info is not None and revisions_to_delete:
        strategy = cache_info.delete_revisions(*revisions_to_delete)
        strategies.append(strategy)
        expected_freed_bytes = max(int(strategy.expected_freed_size), 0)
    unique_incomplete_files = {
        item.path: item for item in incomplete_files
    }
    expected_freed_bytes += sum(
        item.size_bytes for item in unique_incomplete_files.values()
    )

    before_cache_size = directory_size(cache_dir)
    if not dry_run:
        for item in unique_incomplete_files.values():
            try:
                item.path.unlink(missing_ok=True)
            except OSError as exc:
                raise ModelUnavailableError(
                    f"無法移除 incomplete model cache file：{item.path}"
                ) from exc
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
