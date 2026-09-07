# Meeting Recording Processor

Apple Silicon 上完全本機運行嘅廣東話會議轉錄工具。輸入錄音或會議錄影後，以兩個獨立 command 完成工作：

1. `extract`：抽取／normalize 音訊、離線 ASR、確定性後處理，然後只寫出 `<input-stem>.transcript.json`。
2. `export`：讀取成功嘅 transcript JSON，另外寫出 `<input-stem>.txt` 同 `<input-stem>.srt`。

兩個 command 都唔會自動清理逐字稿、摘要、生成會議紀錄或 action items。

## 支援範圍

- macOS Apple Silicon arm64（M1/M2/M3/M4）
- Python 3.11–3.13、`uv`、system `ffmpeg`／`ffprobe`
- `.m4a`、`.mp3`、`.wav`、`.flac`、`.mp4`、`.mov`
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

批次轉錄會逐一處理目錄內嘅支援檔案：

```bash
./scripts/transcribe-batch.sh recordings/
# 或：uv run mrp batch recordings/
```

Batch 會按檔名排序，並喺第一個檔案開始前預先檢查全部 stem-based transcript destinations。相同 stem（例如 `meeting.m4a` 同 `meeting.mp3`）或 macOS case-insensitive 等價 destination 會直接拒絕，並列出 conflicting inputs；即使加 `--overwrite` 都唔會容許一個 input 覆蓋同一批次另一個 input。

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

## 轉錄進度

`extract` 同 `batch` 預設用 `ASR_PROGRESS=auto`：TTY 會更新單一進度行，redirect／CI output 會寫普通 log 行。亦可明確設定：

```bash
export ASR_PROGRESS=auto  # 預設；TTY compact display，非 TTY plain logs
export ASR_PROGRESS=on    # 強制輸出（非 TTY 仍然唔會用 cursor escape）
export ASR_PROGRESS=off   # 關閉進度輸出
```

等價嘅單次 command option 係 `--progress auto|on|off`。

進度會顯示目前 phase、elapsed time，同可用嘅 media duration。Qwen3 透過 pinned runtime 嘅 structured `on_progress` callback，以實際已處理 audio seconds／總 duration 計算百分比；SenseVoice 以已完成 chunk 嘅實際 audio duration 報告。若 backend 沒有可靠 total，會顯示 `Transcribing...` 而唔會估算百分比。`auto` 因 objective hard failure fallback 時，會先顯示 `fallback` transition／下一個 model loading；SenseVoice 開始後，percentage 會由自己嘅 0% 重新計，唔會沿用 Qwen3 嘅進度。

單檔 TTY output 例子：

```text
Preparing audio...
Loading qwen3 model...
Transcribing [#############-------] 68%  43:18 / 1:03:42  Elapsed: 18:27
Writing transcript files...
Completed in 27:11
```

batch output 例子：

```text
File 3 of 8: meeting-03.m4a
Transcribing [##########----------] 51%  32:28 / 1:03:42  Elapsed: 14:02
```

非互動 output 會保留 phase、percentage（如有）、processed／total duration 同 elapsed，方便 redirect 到 log。
Progress output 係 observability side channel；如果 stderr 或 backend callback stream 失效，會停用後續 progress，但 transcription 會繼續，亦唔會因此觸發 fallback。

Fallback 例子：

```text
[transcribe] Transcribing progress=100% processed=00:10/00:10 elapsed=00:08
[fallback] qwen3 failed objective quality gate; falling back to sensevoice and loading model... elapsed=00:08
[transcribe] Transcribing progress=0% processed=00:00/00:10 elapsed=00:09
```

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
├── scripts/                           # setup／extract／export／batch wrappers + legacy baseline
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
