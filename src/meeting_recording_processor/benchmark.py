"""Offline, sequential benchmarking around the canonical extraction pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from importlib import util
import json
from pathlib import Path
import platform
from time import perf_counter
from typing import Callable

from .config import AsrMode, BenchmarkConfig, ExtractConfig, SUPPORTED_EXTENSIONS
from .errors import ConfigurationError, OutputExistsError, ProcessorError, TranscriptionFailed
from .manifest import utc_now_iso
from .media.probe import MediaMetadata, probe_media
from .models import (
    AsrModelSet,
    ModelSetStatus,
    build_model_inventory,
    inspect_model_sets,
)
from .pipeline import ExtractResult, extract
from .schema_io import atomic_write_text, load_package
from .progress import format_duration


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    input_path: Path
    input_duration_seconds: float | None
    report_path: Path
    results: tuple[dict[str, object], ...]

    @property
    def passed_count(self) -> int:
        return sum(result["status"] == "passed" for result in self.results)


ModelStatusInspector = Callable[
    [Path, tuple[AsrModelSet, ...]], tuple[ModelSetStatus, ...]
]
RuntimeChecker = Callable[[AsrModelSet], str | None]
Clock = Callable[[], float]


def backend_runtime_reason(model_set: AsrModelSet) -> str | None:
    """Return a skip reason without loading any ASR model."""

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return "requires Apple Silicon macOS (arm64)"

    missing_modules: list[str] = []
    for module_name in model_set.runtime_modules:
        try:
            available = util.find_spec(module_name) is not None
        except (ImportError, AttributeError, ValueError):
            available = False
        if not available:
            missing_modules.append(module_name)
    if missing_modules:
        return f"runtime package(s) unavailable: {', '.join(missing_modules)}"

    if model_set.requires_mps:
        try:
            torch_module = importlib.import_module("torch")
            available = bool(torch_module.backends.mps.is_available())
        except (AttributeError, ImportError, RuntimeError) as exc:
            return f"MPS runtime unavailable: {exc}"
        if not available:
            return "torch.backends.mps.is_available() returned false"
    return None


def _missing_model_reason(status: ModelSetStatus) -> str:
    details = []
    for asset_status in status.assets:
        if asset_status.installed:
            continue
        state = (
            "cached but incomplete or not offline-resolvable"
            if asset_status.cached
            else "missing"
        )
        details.append(f"{asset_status.asset.model_id} ({state})")
    return "local model asset(s) unavailable: " + ", ".join(details)


def _relative_output_path(path: Path, benchmark_dir: Path) -> str:
    return path.relative_to(benchmark_dir).as_posix()


def _diagnostic_output_path(
    diagnostic_path: object,
    *,
    benchmark_dir: Path,
    backend_output_dir: Path,
) -> str | None:
    if diagnostic_path is None:
        return None
    try:
        candidate = Path(diagnostic_path).expanduser().resolve()
        benchmark_root = benchmark_dir.resolve()
        backend_root = backend_output_dir.resolve()
    except (OSError, RuntimeError, TypeError):
        return None
    if (
        not candidate.is_file()
        or not candidate.is_relative_to(benchmark_root)
        or not candidate.is_relative_to(backend_root)
    ):
        return None
    return _relative_output_path(candidate, benchmark_root)


def _output_path(benchmark_dir: Path, model_set: AsrModelSet, input_path: Path) -> Path:
    return benchmark_dir / model_set.backend / f"{input_path.stem}.transcript.json"


def _preflight_outputs(
    report_path: Path,
    attempted_sets: tuple[AsrModelSet, ...],
    benchmark_dir: Path,
    input_path: Path,
    *,
    overwrite: bool,
) -> None:
    if overwrite:
        return
    destinations = [report_path]
    destinations.extend(
        _output_path(benchmark_dir, model_set, input_path)
        for model_set in attempted_sets
    )
    existing = [path for path in destinations if path.exists()]
    if existing:
        details = "\n".join(f"  {path}" for path in existing)
        raise OutputExistsError(
            "benchmark output already exists; use --overwrite to replace exact destinations:\n"
            + details
        )


def _result(
    *,
    model_set: AsrModelSet,
    status: str,
    elapsed_seconds: float | None,
    input_duration_seconds: float | None,
    output: str | None = None,
    character_count: int | None = None,
    reason: str | None = None,
) -> dict[str, object]:
    realtime_factor = None
    if (
        elapsed_seconds is not None
        and input_duration_seconds is not None
        and input_duration_seconds > 0
    ):
        realtime_factor = round(elapsed_seconds / input_duration_seconds, 6)
    return {
        "backend": model_set.backend,
        "status": status,
        "models": list(model_set.model_ids),
        "elapsed_seconds": (
            round(elapsed_seconds, 3) if elapsed_seconds is not None else None
        ),
        "realtime_factor": realtime_factor,
        "character_count": character_count,
        "output": output,
        "reason": reason,
    }


def run_benchmark(
    config: BenchmarkConfig,
    *,
    extractor: Callable[[ExtractConfig], ExtractResult] = extract,
    media_probe: Callable[[Path], MediaMetadata] = probe_media,
    model_status_inspector: ModelStatusInspector = inspect_model_sets,
    runtime_checker: RuntimeChecker = backend_runtime_reason,
    clock: Clock = perf_counter,
) -> BenchmarkResult:
    input_path = config.input_path.expanduser().resolve()
    if not input_path.is_file():
        raise ConfigurationError(f"搵唔到 input file：{input_path}")
    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ConfigurationError(f"不支援 {input_path.suffix}；支援格式：{supported}")
    context_file = config.context_file.expanduser().resolve() if config.context_file else None
    if context_file is not None and not context_file.is_file():
        raise ConfigurationError(f"搵唔到 context file：{context_file}")

    media = media_probe(input_path)
    input_duration_seconds = media.duration_seconds
    inventory = build_model_inventory(
        qwen_model=config.qwen_model,
        qwen_aligner_model=config.qwen_aligner_model,
        sensevoice_model=config.sensevoice_model,
        vibevoice_model=config.vibevoice_model,
    )
    statuses = model_status_inspector(config.cache_dir, inventory)
    status_by_backend = {status.model_set.backend: status for status in statuses}
    benchmark_dir = config.benchmark_dir.expanduser().resolve()
    report_path = benchmark_dir / "benchmark.json"

    attempted_sets: list[AsrModelSet] = []
    skipped_results: dict[str, dict[str, object]] = {}
    for model_set in inventory:
        if model_set.experimental and not config.include_experimental:
            skipped_results[model_set.backend] = _result(
                model_set=model_set,
                status="skipped",
                elapsed_seconds=None,
                input_duration_seconds=input_duration_seconds,
                reason="experimental backend not included; use --include-experimental",
            )
            continue
        status = status_by_backend[model_set.backend]
        if not status.installed:
            skipped_results[model_set.backend] = _result(
                model_set=model_set,
                status="skipped",
                elapsed_seconds=None,
                input_duration_seconds=input_duration_seconds,
                reason=_missing_model_reason(status),
            )
            continue
        runtime_reason = runtime_checker(model_set)
        if runtime_reason is not None:
            skipped_results[model_set.backend] = _result(
                model_set=model_set,
                status="skipped",
                elapsed_seconds=None,
                input_duration_seconds=input_duration_seconds,
                reason=runtime_reason,
            )
            continue
        attempted_sets.append(model_set)

    _preflight_outputs(
        report_path,
        tuple(attempted_sets),
        benchmark_dir,
        input_path,
        overwrite=config.overwrite,
    )

    results_by_backend = dict(skipped_results)
    for model_set in inventory:
        if model_set.backend not in {item.backend for item in attempted_sets}:
            continue
        output_path = _output_path(benchmark_dir, model_set, input_path)
        started = clock()
        try:
            extracted = extractor(
                ExtractConfig(
                    input_path=input_path,
                    output_dir=output_path.parent,
                    work_dir=config.work_dir,
                    cache_dir=config.cache_dir,
                    asr_mode=AsrMode(model_set.backend),
                    language=config.language,
                    context_file=context_file,
                    qwen_model=config.qwen_model,
                    qwen_aligner_model=config.qwen_aligner_model,
                    sensevoice_model=config.sensevoice_model,
                    vibevoice_model=config.vibevoice_model,
                    vibevoice_acoustic_chunk_size=config.vibevoice_acoustic_chunk_size,
                    overwrite=config.overwrite,
                    verbose=config.verbose,
                    progress_mode=config.progress_mode,
                )
            )
            package = load_package(extracted.output_path, require_completed=True)
            transcript = package["transcript"]
            if not isinstance(transcript, dict):
                raise ConfigurationError("completed transcript package has no transcript object")
            text = transcript.get("text")
            if not isinstance(text, str):
                raise ConfigurationError("completed transcript package has no transcript text")
            elapsed_seconds = max(clock() - started, 0.0)
            results_by_backend[model_set.backend] = _result(
                model_set=model_set,
                status="passed",
                elapsed_seconds=elapsed_seconds,
                input_duration_seconds=input_duration_seconds,
                output=_relative_output_path(extracted.output_path, benchmark_dir),
                character_count=len(text),
            )
        except TranscriptionFailed as exc:
            elapsed_seconds = max(clock() - started, 0.0)
            results_by_backend[model_set.backend] = _result(
                model_set=model_set,
                status="failed",
                elapsed_seconds=elapsed_seconds,
                input_duration_seconds=input_duration_seconds,
                output=_diagnostic_output_path(
                    exc.diagnostic_path,
                    benchmark_dir=benchmark_dir,
                    backend_output_dir=output_path.parent,
                ),
                reason=str(exc),
            )
        except ProcessorError as exc:
            elapsed_seconds = max(clock() - started, 0.0)
            results_by_backend[model_set.backend] = _result(
                model_set=model_set,
                status="failed",
                elapsed_seconds=elapsed_seconds,
                input_duration_seconds=input_duration_seconds,
                reason=str(exc),
            )

    ordered_results = tuple(
        results_by_backend[model_set.backend] for model_set in inventory
    )
    report = {
        "input": str(input_path),
        "input_duration_seconds": input_duration_seconds,
        "created_at": utc_now_iso(),
        "results": list(ordered_results),
    }
    atomic_write_text(
        report_path,
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        overwrite=config.overwrite,
    )
    return BenchmarkResult(
        input_path=input_path,
        input_duration_seconds=input_duration_seconds,
        report_path=report_path,
        results=ordered_results,
    )


def render_summary(result: BenchmarkResult) -> str:
    lines = [
        "ASR Benchmark",
        f"Input: {result.input_path.name}",
        f"Duration: {format_duration(result.input_duration_seconds)}",
        "",
        "Backend       Status    Runtime      RTF       Characters",
    ]
    for item in result.results:
        elapsed = item["elapsed_seconds"]
        runtime = format_duration(elapsed) if isinstance(elapsed, (int, float)) else "-"
        rtf = (
            f"{item['realtime_factor']:.3f}"
            if isinstance(item["realtime_factor"], (int, float))
            else "-"
        )
        characters = (
            str(item["character_count"])
            if isinstance(item["character_count"], int)
            else "-"
        )
        lines.append(
            f"{str(item['backend']):13} "
            f"{str(item['status']).upper():9} "
            f"{runtime:11} "
            f"{rtf:8} "
            f"{characters}"
        )
        if item["reason"]:
            lines.append(f"  reason: {item['reason']}")
    lines.extend(["", "Report:", f"  {result.report_path}"])
    return "\n".join(lines)
