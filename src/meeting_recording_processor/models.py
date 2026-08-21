"""Explicit model download and offline cache resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .errors import ModelUnavailableError
from .runtime import configure_huggingface_cache


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    model_id: str
    path: Path
    snapshot: str | None


def _snapshot_name(path: Path) -> str | None:
    if path.parent.name == "snapshots":
        return path.name
    return None


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
