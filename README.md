# Meeting Recording Processor

Apple Silicon 上完全本機運行嘅廣東話會議轉錄工具。輸入錄音或會議錄影後，以兩個獨立 command 完成工作：

1. `extract`：抽取／normalize 音訊、離線 ASR、確定性後處理，然後只寫出 `<input-stem>.transcript.json`。
2. `export`：讀取成功嘅 transcript JSON，另外寫出 `<input-stem>.txt` 同 `<input-stem>.srt`。

兩個 command 都唔會自動清理逐字稿、摘要、生成會議紀錄或 action items。

## 支援範圍

- macOS Apple Silicon arm64（M1/M2/M3/M4）
- Python 3.11–3.13、`uv`、system `ffmpeg`／`ffprobe`
- `.m4a`、`.mp3`、`.wav`、`.mp4`、`.mov`
- 影片只讀 audio stream；唔分析畫面、唔做 OCR
- `qwen3`：`mlx-qwen3-asr==0.3.5` + `Qwen/Qwen3-ASR-1.7B`
- `sensevoice`：`mlx-audio==0.4.1` + `mlx-community/SenseVoiceSmall`
- runtime 一律 offline；只有明確執行 `download-model` 先會連網
- 保留廣東話／中英夾雜，canonical output 以 OpenCC `s2hk` 轉為香港繁體，唔翻譯或改寫內容

v1 明確不包括 speaker diarization、Whisper、cloud ASR、live captions、影片 OCR／frame analysis、自動會議紀錄。

## 安裝

```bash
brew install uv ffmpeg
chmod +x setup.sh scripts/*.sh
./setup.sh
./scripts/download-models.sh
./scripts/doctor.sh
```

`setup.sh` 建立 `.venv`；model 下載到 project-local `.cache/huggingface/`。模型下載完成後，`extract` 只會讀 local cache，缺少 asset 時會 fail closed。

## 使用方法

先轉錄，流程會停喺 JSON：

```bash
uv run mrp extract /path/to/meeting.mp4 --asr auto
# output/meeting.transcript.json
```

確認 JSON 後，先另外 export：

```bash
uv run mrp export output/meeting.transcript.json
# output/meeting.txt
# output/meeting.srt
```

Shell wrappers 提供同一功能：

```bash
./scripts/extract.sh /path/to/meeting.m4a --asr auto
./scripts/export.sh output/meeting.transcript.json
```

常用選項：

```bash
# 明確只用 Qwen3；失敗時絕不靜默 fallback
uv run mrp extract meeting.wav --asr qwen3

# 人手指定 SenseVoice 重試，寫入另一個 output directory
uv run mrp extract meeting.wav --asr sensevoice --output-dir output/sensevoice-retry

# opt-in technical vocabulary/context
uv run mrp extract meeting.m4a \
  --context-file profiles/examples/loq-technical-meeting.txt

# 檢查 runtime、models 同 cache
uv run mrp doctor
uv run mrp cache-size
```

預設 context 係 `profiles/generic.txt`。LOQ vocabulary 只係 opt-in example，核心程式冇 hard-code domain data。

## `auto` fallback 規則

`--asr auto` 固定先試 Qwen3，只有以下客觀 hard failure 先會再試 SenseVoice：

- backend exception；
- 空白輸出；
- 非空白字元超過 90% 係 Unicode 標點或符號；
- active audio 至少 10 秒，但 substantive 字元少過 3 個。

一般專有名詞、accuracy、標點或分段質素問題唔會自動 fallback。每個 attempt 嘅 raw text、raw segments、model snapshot、quality report、錯誤同 runtime 都保留喺同一 transcript JSON，唔會融合或覆蓋。

## Output contract

`extract` 成功：

```text
output/
└── meeting.transcript.json
```

`export` 後：

```text
output/
├── meeting.transcript.json
├── meeting.txt
└── meeting.srt
```

Qwen3 原生 model timestamps 會標記為 `timing_source: "model"`。SenseVoiceSmall 冇 word-level timestamps，本程式會先保留 30 秒 chunk timing，再按文字長度建立 cue；JSON 會標記 `timing_source: "estimated_from_chunk"` 並加入 warning。

如所有 attempts 都失敗，`extract` 仍會原子寫出 `status: "failed"` 嘅 diagnostic JSON，再以非零 exit code 停止。`export` 拒絕處理 failed JSON。

完整 schema 見 [`schemas/transcript-v1.schema.json`](schemas/transcript-v1.schema.json)。

## Project structure

```text
meeting-recording-processor/
├── src/meeting_recording_processor/   # CLI、pipeline、adapters、media、writers
├── tests/                             # 無需 MLX/model 嘅 unit + integration tests
├── schemas/                           # transcript JSON Schema
├── scripts/                           # setup／extract／export wrappers + legacy baseline
├── profiles/                          # generic 同 opt-in vocabulary
├── README.md
├── SPEC.md
├── ARCHITECTURE.md
├── SKILL.md
├── pyproject.toml
└── setup.sh
```

`scripts/transcribe-qwen3-*-baseline.sh` 只為追溯早期 Qwen shell baseline 而保留；正式 workflow 請使用 `mrp extract`／`mrp export`。

## 開發與驗證

```bash
uv sync --extra dev
uv run pytest
uv run python -m compileall -q src tests
bash -n setup.sh scripts/*.sh
```

Backend-independent tests 用 fake services 驗證 fallback、attempt preservation、schema、post-processing、export、media command 同 CLI contract。真正 MLX inference 同 model download 必須喺 Apple Silicon Mac 完成 smoke test。

## 文件

- [SPEC.md](SPEC.md)：產品／技術 contract、acceptance criteria。
- [ARCHITECTURE.md](ARCHITECTURE.md)：module boundary、offline routing、資料生命週期。
- [SKILL.md](SKILL.md)：俾 Codex／agent 執行呢個工具時遵守嘅操作規則。
- [mlx-qwen3-asr](https://github.com/moona3k/mlx-qwen3-asr/)／[MLX-Audio](https://github.com/Blaizzy/mlx-audio)／[SenseVoiceSmall](https://huggingface.co/mlx-community/SenseVoiceSmall)：upstream runtime/model documentation。
