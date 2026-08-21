"""Provenance helpers embedded in each canonical transcript JSON."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from importlib import metadata
import platform
from pathlib import Path
import sys
from uuid import uuid4

from . import __version__


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid4().hex[:8]}"


def package_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def tool_metadata(*, ffprobe: str | None) -> dict[str, object]:
    return {
        "name": "meeting-recording-processor",
        "version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "mlx-qwen3-asr": package_version("mlx-qwen3-asr"),
            "mlx-audio": package_version("mlx-audio"),
            "opencc-python-reimplemented": package_version("opencc-python-reimplemented"),
            "huggingface-hub": package_version("huggingface-hub"),
        },
        "ffprobe": ffprobe,
    }
