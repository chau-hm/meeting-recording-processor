"""Non-mutating runtime diagnostics for setup verification."""

from __future__ import annotations

from importlib import metadata, util
from pathlib import Path
import platform
import shutil
import sys
from typing import Any

from .config import (
    DEFAULT_QWEN_MODEL,
    DEFAULT_SENSEVOICE_MODEL,
    DEFAULT_VIBEVOICE_MODEL,
)
from .models import resolve_cached_model


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def doctor_report(cache_dir: Path) -> dict[str, Any]:
    models: dict[str, dict[str, Any]] = {}
    for backend, model_id in (
        ("qwen3", DEFAULT_QWEN_MODEL),
        ("sensevoice", DEFAULT_SENSEVOICE_MODEL),
        ("vibevoice", DEFAULT_VIBEVOICE_MODEL),
    ):
        try:
            resolved = resolve_cached_model(model_id, cache_dir)
        except Exception as exc:
            models[backend] = {"model": model_id, "available": False, "detail": str(exc)}
        else:
            models[backend] = {
                "model": model_id,
                "available": True,
                "snapshot": resolved.snapshot,
                "path": str(resolved.path),
            }

    report: dict[str, Any] = {
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "supported": platform.system() == "Darwin" and platform.machine() == "arm64",
        },
        "python": {
            "version": sys.version.split()[0],
            "supported": (3, 11) <= sys.version_info[:2] < (3, 14),
        },
        "commands": {
            name: {"available": shutil.which(name) is not None, "path": shutil.which(name)}
            for name in ("uv", "ffmpeg", "ffprobe")
        },
        "packages": {
            "mlx-qwen3-asr": {
                "available": util.find_spec("mlx_qwen3_asr") is not None,
                "version": _distribution_version("mlx-qwen3-asr"),
            },
            "mlx-audio": {
                "available": util.find_spec("mlx_audio") is not None,
                "version": _distribution_version("mlx-audio"),
            },
            "opencc-python-reimplemented": {
                "available": util.find_spec("opencc") is not None,
                "version": _distribution_version("opencc-python-reimplemented"),
            },
            "torch": {
                "available": util.find_spec("torch") is not None,
                "version": _distribution_version("torch"),
            },
            "transformers": {
                "available": util.find_spec("transformers") is not None,
                "version": _distribution_version("transformers"),
            },
        },
        "models": models,
        "cache_dir": str(cache_dir.resolve()),
    }
    report["healthy"] = bool(
        report["platform"]["supported"]
        and report["python"]["supported"]
        and all(item["available"] for item in report["commands"].values())
        and all(item["available"] for item in report["packages"].values())
        and all(item["available"] for item in report["models"].values())
    )
    return report
