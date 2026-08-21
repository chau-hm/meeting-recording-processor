"""Read, validate and atomically write canonical transcript JSON."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .errors import OutputExistsError, SchemaError


SCHEMA_VERSION = "1.0"
ATTEMPT_STATUSES = frozenset({"completed", "hard_failure", "error"})
TIMING_SOURCES = frozenset(
    {"model", "chunk", "estimated_from_chunk", "estimated_from_duration"}
)


def _require_nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{label} 必須係非空白 string")
    return value


def _validate_segments(segments: object, *, label: str) -> None:
    if not isinstance(segments, list):
        raise SchemaError(f"{label} 必須係 array")
    previous_start = -1.0
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise SchemaError(f"{label}[{index}] 必須係 object")
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SchemaError(f"{label}[{index}] 時間格式無效") from exc
        if start < 0 or end <= start or start < previous_start:
            raise SchemaError(f"{label}[{index}] 時間範圍無效或次序錯誤")
        _require_nonempty_string(segment.get("text"), f"{label}[{index}].text")
        timing_source = segment.get("timing_source")
        if timing_source not in TIMING_SOURCES:
            raise SchemaError(f"{label}[{index}].timing_source 無效")
        previous_start = start


def validate_package(payload: object, *, require_completed: bool = False) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SchemaError("transcript JSON 頂層必須係 object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError(f"不支援 schema_version：{payload.get('schema_version')!r}")
    status = payload.get("status")
    if status not in {"completed", "failed"}:
        raise SchemaError("status 必須係 completed 或 failed")
    _require_nonempty_string(payload.get("run_id"), "run_id")
    if require_completed and status != "completed":
        raise SchemaError("呢份 transcript JSON 未成功完成，唔可以 export")
    attempts = payload.get("attempts")
    if not isinstance(attempts, list):
        raise SchemaError("attempts 必須係 array")
    attempt_statuses: dict[str, str] = {}
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, dict):
            raise SchemaError(f"attempts[{index}] 必須係 object")
        attempt_id = _require_nonempty_string(
            attempt.get("attempt_id"), f"attempts[{index}].attempt_id"
        )
        if attempt_id in attempt_statuses:
            raise SchemaError(f"attempt_id 重複：{attempt_id}")
        _require_nonempty_string(attempt.get("backend"), f"attempts[{index}].backend")
        _require_nonempty_string(attempt.get("model"), f"attempts[{index}].model")
        attempt_status = attempt.get("status")
        if attempt_status not in ATTEMPT_STATUSES:
            raise SchemaError(f"attempts[{index}].status 無效")
        raw_segments = attempt.get("raw_segments")
        if raw_segments is not None:
            _validate_segments(raw_segments, label=f"attempts[{index}].raw_segments")
        attempt_statuses[attempt_id] = str(attempt_status)

    transcript = payload.get("transcript")
    if status == "completed":
        if not attempts:
            raise SchemaError("completed package 必須至少包含一個 attempt")
        selected_attempt_id = _require_nonempty_string(
            payload.get("selected_attempt_id"), "selected_attempt_id"
        )
        if attempt_statuses.get(selected_attempt_id) != "completed":
            raise SchemaError("selected_attempt_id 必須指向 completed attempt")
        if not isinstance(transcript, dict):
            raise SchemaError("completed transcript 必須包含 transcript object")
        _require_nonempty_string(transcript.get("language"), "transcript.language")
        _require_nonempty_string(transcript.get("text"), "transcript.text")
        segments = transcript.get("segments")
        if not isinstance(segments, list) or not segments:
            raise SchemaError("transcript.segments 必須係非空白 array")
        _validate_segments(segments, label="transcript.segments")
    else:
        if payload.get("selected_attempt_id") is not None:
            raise SchemaError("failed package 嘅 selected_attempt_id 必須係 null")
        if transcript is not None:
            raise SchemaError("failed package 嘅 transcript 必須係 null")
    return payload


def load_package(path: Path, *, require_completed: bool = False) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except FileNotFoundError as exc:
        raise SchemaError(f"搵唔到 transcript JSON：{path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaError(f"無法讀取 transcript JSON：{exc}") from exc
    return validate_package(payload, require_completed=require_completed)


def _atomic_write(path: Path, content: str, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise OutputExistsError(f"輸出已存在：{path}；如要取代請加 --overwrite")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise OutputExistsError(
                    f"輸出已存在：{path}；如要取代請加 --overwrite"
                ) from exc
            temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_package(path: Path, payload: dict[str, Any], *, overwrite: bool = False) -> None:
    validate_package(payload)
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    _atomic_write(path, content, overwrite=overwrite)


def atomic_write_text(path: Path, content: str, *, overwrite: bool = False) -> None:
    _atomic_write(path, content, overwrite=overwrite)
