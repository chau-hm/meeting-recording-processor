"""Command-line interface for the two-step extract/export workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import (
    AsrMode,
    DEFAULT_QWEN_ALIGNER_MODEL,
    DEFAULT_QWEN_MODEL,
    DEFAULT_SENSEVOICE_MODEL,
    DEFAULT_VIBEVOICE_ACOUSTIC_CHUNK_SIZE,
    DEFAULT_VIBEVOICE_MODEL,
    ExportConfig,
    ExtractConfig,
    SUPPORTED_EXTENSIONS,
    project_root,
)
from .diagnostics import doctor_report
from .errors import ConfigurationError, ProcessorError, TranscriptionFailed
from .models import directory_size, download_model, human_size
from .outputs import export_transcript
from .pipeline import extract
from .progress import ProgressMode, resolve_progress_mode
from .runtime import require_apple_silicon


def _add_transcription_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--asr",
        choices=[mode.value for mode in AsrMode],
        default=AsrMode.AUTO.value,
        help=(
            "auto 只係 Qwen3 → SenseVoice；"
            "vibevoice 係 explicit evaluation backend"
        ),
    )
    parser.add_argument("--language", default="Cantonese")
    parser.add_argument("--context-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--qwen-aligner-model", default=DEFAULT_QWEN_ALIGNER_MODEL)
    parser.add_argument("--sensevoice-model", default=DEFAULT_SENSEVOICE_MODEL)
    parser.add_argument("--vibevoice-model", default=DEFAULT_VIBEVOICE_MODEL)
    parser.add_argument(
        "--vibevoice-acoustic-chunk-size",
        type=int,
        default=DEFAULT_VIBEVOICE_ACOUSTIC_CHUNK_SIZE,
        help="VibeVoice acoustic tokenizer chunk size in 24 kHz samples",
    )
    parser.add_argument("--keep-work-files", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--progress",
        dest="progress_mode",
        choices=[mode.value for mode in ProgressMode],
        help="進度輸出：auto（預設）、on 或 off；亦可用 ASR_PROGRESS",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meeting-recording-processor",
        description="本機廣東話會議轉錄：extract 產生 JSON，export 產生 TXT/SRT。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser(
        "extract", help="由 audio／video 抽取並轉錄成 .transcript.json"
    )
    extract_parser.add_argument(
        "input", type=Path, help="輸入 .m4a/.mp3/.wav/.flac/.mp4/.mov"
    )
    _add_transcription_options(extract_parser)

    batch_parser = subparsers.add_parser(
        "batch", help="逐一轉錄目錄內支援嘅 audio／video 檔案"
    )
    batch_parser.add_argument("input", type=Path, help="輸入目錄")
    _add_transcription_options(batch_parser)

    export_parser = subparsers.add_parser(
        "export", help="由 .transcript.json 產生 .txt 同 .srt"
    )
    export_parser.add_argument("transcript", type=Path)
    export_parser.add_argument("--output-dir", type=Path)
    export_parser.add_argument("--overwrite", action="store_true")

    doctor_parser = subparsers.add_parser("doctor", help="檢查平台、依賴同 model cache")
    doctor_parser.add_argument("--cache-dir", type=Path)
    doctor_parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    doctor_parser.add_argument("--qwen-aligner-model", default=DEFAULT_QWEN_ALIGNER_MODEL)
    doctor_parser.add_argument("--json", action="store_true", dest="as_json")

    download_parser = subparsers.add_parser(
        "download-model", help="明確下載本機 ASR model；extract 本身永遠 offline"
    )
    download_parser.add_argument(
        "--asr",
        choices=["qwen3", "sensevoice", "vibevoice", "all"],
        default="all",
    )
    download_parser.add_argument("--cache-dir", type=Path)
    download_parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    download_parser.add_argument("--qwen-aligner-model", default=DEFAULT_QWEN_ALIGNER_MODEL)
    download_parser.add_argument("--sensevoice-model", default=DEFAULT_SENSEVOICE_MODEL)
    download_parser.add_argument("--vibevoice-model", default=DEFAULT_VIBEVOICE_MODEL)

    cache_parser = subparsers.add_parser("cache-size", help="顯示 project-local model cache 大小")
    cache_parser.add_argument("--cache-dir", type=Path)
    return parser


def _paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path | None]:
    root = project_root()
    output_dir = (args.output_dir or root / "output").expanduser()
    work_dir = (getattr(args, "work_dir", None) or root / "work").expanduser()
    cache_dir = (getattr(args, "cache_dir", None) or root / ".cache/huggingface").expanduser()
    context_file = getattr(args, "context_file", None)
    if context_file is None:
        generic = root / "profiles/generic.txt"
        context_file = generic if generic.is_file() else None
    return output_dir, work_dir, cache_dir, context_file


def _preflight_batch_output_collisions(
    files: list[Path],
    output_dir: Path,
) -> None:
    destinations: dict[str, tuple[Path, list[Path]]] = {}
    for input_path in files:
        destination = (
            output_dir / f"{input_path.stem}.transcript.json"
        ).expanduser().resolve()
        key = str(destination).casefold()
        entry = destinations.setdefault(key, (destination, []))
        entry[1].append(input_path)

    collisions = [entry for entry in destinations.values() if len(entry[1]) > 1]
    if not collisions:
        return

    details = [
        f"  {destination}: {', '.join(str(input_path) for input_path in inputs)}"
        for destination, inputs in collisions
    ]
    raise ConfigurationError(
        "batch output collision(s)；同一批次內唔可以共用 transcript destination：\n"
        + "\n".join(details)
    )


def _run_extract(args: argparse.Namespace) -> int:
    output_dir, work_dir, cache_dir, context_file = _paths(args)
    result = extract(
        ExtractConfig(
            input_path=args.input,
            output_dir=output_dir,
            work_dir=work_dir,
            cache_dir=cache_dir,
            asr_mode=AsrMode(args.asr),
            language=args.language,
            context_file=context_file,
            qwen_model=args.qwen_model,
            qwen_aligner_model=args.qwen_aligner_model,
            sensevoice_model=args.sensevoice_model,
            vibevoice_model=args.vibevoice_model,
            vibevoice_acoustic_chunk_size=args.vibevoice_acoustic_chunk_size,
            keep_work_files=args.keep_work_files,
            overwrite=args.overwrite,
            verbose=args.verbose,
            progress_mode=args.progress_mode,
        )
    )
    print(f"本機轉錄已完成：{result.output_path}")
    print(f"採用 backend：{result.selected_backend}")
    if result.fallback_used:
        print("Qwen3 出現客觀 hard failure，已保留原 attempt 並改用 SenseVoice。")
    for warning in result.warnings:
        print(f"注意：{warning}")
    print("流程已停喺 extract；未有自動 export、cleanup 或生成會議記錄。")
    return 0


def _run_batch(args: argparse.Namespace) -> int:
    input_dir = args.input.expanduser().resolve()
    if not input_dir.is_dir():
        raise ConfigurationError(f"搵唔到 input directory：{input_dir}")
    try:
        files = sorted(
            (
                path
                for path in input_dir.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            ),
            key=lambda path: (path.name.casefold(), path.name),
        )
    except OSError as exc:
        raise ConfigurationError(f"無法讀取 input directory：{exc}") from exc
    if not files:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ConfigurationError(f"目錄內搵唔到支援檔案（{supported}）：{input_dir}")

    output_dir, work_dir, cache_dir, context_file = _paths(args)
    _preflight_batch_output_collisions(files, output_dir)
    show_progress = resolve_progress_mode(args.progress_mode) is not ProgressMode.OFF
    failures = 0
    for index, input_path in enumerate(files, start=1):
        if show_progress:
            print(f"File {index} of {len(files)}: {input_path.name}", file=sys.stderr)
        try:
            result = extract(
                ExtractConfig(
                    input_path=input_path,
                    output_dir=output_dir,
                    work_dir=work_dir,
                    cache_dir=cache_dir,
                    asr_mode=AsrMode(args.asr),
                    language=args.language,
                    context_file=context_file,
                    qwen_model=args.qwen_model,
                    qwen_aligner_model=args.qwen_aligner_model,
                    sensevoice_model=args.sensevoice_model,
                    vibevoice_model=args.vibevoice_model,
                    vibevoice_acoustic_chunk_size=args.vibevoice_acoustic_chunk_size,
                    keep_work_files=args.keep_work_files,
                    overwrite=args.overwrite,
                    verbose=args.verbose,
                    progress_mode=args.progress_mode,
                )
            )
        except TranscriptionFailed as exc:
            failures += 1
            print(f"錯誤：{exc}", file=sys.stderr)
            if exc.diagnostic_path:
                print(f"診斷 JSON：{exc.diagnostic_path}", file=sys.stderr)
            continue
        except ProcessorError as exc:
            failures += 1
            print(f"錯誤：{exc}", file=sys.stderr)
            continue
        if show_progress:
            print(f"Completed: {result.output_path}", file=sys.stderr)

    if failures:
        print(f"Batch transcription failed for {failures} of {len(files)} file(s).", file=sys.stderr)
        return 1
    print(f"Batch transcription completed: {len(files)} file(s).")
    return 0


def _run_export(args: argparse.Namespace) -> int:
    txt_path, srt_path = export_transcript(
        ExportConfig(
            transcript_path=args.transcript,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    )
    print(f"TXT：{txt_path}")
    print(f"SRT：{srt_path}")
    print("流程已完成並停止；未有自動生成會議記錄或跟進事項。")
    return 0


def _run_doctor(args: argparse.Namespace) -> int:
    root = project_root()
    cache_dir = (args.cache_dir or root / ".cache/huggingface").expanduser()
    report = doctor_report(
        cache_dir,
        qwen_model=args.qwen_model,
        qwen_aligner_model=args.qwen_aligner_model,
    )
    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Platform：{report['platform']['system']} {report['platform']['machine']}")
        print(f"Python：{report['python']['version']}")
        for name, detail in report["commands"].items():
            state = "OK" if detail["available"] else "MISSING"
            print(f"{state:7} command:{name}")
        for name, detail in report["packages"].items():
            state = "OK" if detail["available"] else "MISSING"
            version = f" ({detail['version']})" if detail.get("version") else ""
            print(f"{state:7} package:{name}{version}")
        for name, detail in report["runtime"].items():
            state = "OK" if detail["available"] else "MISSING"
            print(f"{state:7} runtime:{name}")
        for name, detail in report["models"].items():
            state = "OK" if detail["available"] else "MISSING"
            suffix = (
                f" snapshot={detail['snapshot']}"
                if detail.get("snapshot")
                else f" detail={detail.get('detail', 'not available')}"
            )
            print(f"{state:7} model:{name}{suffix}")
        print(f"Cache：{report['cache_dir']}")
    return 0 if report["healthy"] else 1


def _run_download(args: argparse.Namespace) -> int:
    require_apple_silicon()
    root = project_root()
    cache_dir = (args.cache_dir or root / ".cache/huggingface").expanduser()
    targets: list[tuple[str, str]] = []
    if args.asr in {"qwen3", "all"}:
        targets.append(("qwen3", args.qwen_model))
        targets.append(("qwen3-aligner", args.qwen_aligner_model))
    if args.asr in {"sensevoice", "all"}:
        targets.append(("sensevoice", args.sensevoice_model))
    if args.asr in {"vibevoice", "all"}:
        targets.append(("vibevoice", args.vibevoice_model))
    for backend, model_id in targets:
        print(f"下載 {backend}：{model_id}")
        resolved = download_model(model_id, cache_dir)
        print(f"完成；snapshot：{resolved.snapshot or 'unknown'}")
    print(f"Cache 大小：{human_size(directory_size(cache_dir))}")
    return 0


def _run_cache_size(args: argparse.Namespace) -> int:
    root = project_root()
    cache_dir = (args.cache_dir or root / ".cache/huggingface").expanduser()
    print(f"{cache_dir.resolve()}\t{human_size(directory_size(cache_dir))}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "extract":
            return _run_extract(args)
        if args.command == "batch":
            return _run_batch(args)
        if args.command == "export":
            return _run_export(args)
        if args.command == "doctor":
            return _run_doctor(args)
        if args.command == "download-model":
            return _run_download(args)
        if args.command == "cache-size":
            return _run_cache_size(args)
        parser.error(f"未知 command：{args.command}")
    except TranscriptionFailed as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        if exc.diagnostic_path:
            print(f"診斷 JSON：{exc.diagnostic_path}", file=sys.stderr)
        return 1
    except ProcessorError as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("已中止。", file=sys.stderr)
        return 130
    return 1


def entrypoint() -> None:
    raise SystemExit(main())
