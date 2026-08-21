"""Platform, environment and command-runtime helpers."""

from __future__ import annotations

import os
from pathlib import Path
import platform
import shutil
import sys

from .errors import PlatformError


def require_apple_silicon() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise PlatformError("轉錄只支援 Apple Silicon macOS（arm64）")
    if sys.version_info < (3, 11):
        raise PlatformError("需要 Python 3.11 或以上版本")


def require_media_commands() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise PlatformError(f"缺少必要指令：{', '.join(missing)}；請先安裝 ffmpeg")


def configure_huggingface_cache(cache_dir: Path, *, offline: bool) -> None:
    cache_dir = cache_dir.resolve()
    hub_dir = cache_dir / "hub"
    hub_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(hub_dir)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ.pop("HF_DATASETS_OFFLINE", None)


def command_available(name: str) -> bool:
    return shutil.which(name) is not None
