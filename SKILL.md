---
name: meeting-recording-processor
description: Locally extract Cantonese-heavy meeting audio or video to a canonical transcript JSON on Apple Silicon, then export TXT/SRT only when explicitly requested. Never auto-generate meeting notes.
---

# Meeting Recording Processor

## Contract

使用呢個 project 處理本機廣東話／中英夾雜 meeting recording。流程有兩個獨立步驟；唔可以自動由第一步跳到第二步，更唔可以自動生成 meeting notes。

## Preconditions

1. 只接受 `.m4a`、`.mp3`、`.wav`、`.flac`、`.mp4`、`.mov` regular file。
2. 必須係 macOS Apple Silicon arm64，並已安裝 `uv`、`ffmpeg`、`ffprobe`。
3. 先執行 `uv run mrp doctor`。缺少 model 時，向使用者說明並明確執行 `uv run mrp download-model --asr all`；Qwen3 timestamp path 需要 ASR model 同 forced aligner，`download-model --asr qwen3` 會下載完整 set。下載係唯一可連網步驟。VibeVoice 係大型 model，並要求 Apple Silicon MPS。
`doctor` 會分開檢查 `model:qwen3`、`model:qwen3-aligner`、package presence 同 `runtime:vibevoice-mps` readiness；MPS unavailable 或任一 Qwen asset 缺失時唔好開始相應 extract。

4. 如要釋放 model cache 空間，只執行明確指定嘅 model set：

```bash
uv run mrp clear-model --asr vibevoice
uv run mrp clear-model --asr qwen3 --dry-run
```

`clear-model` 只會刪 configured Hugging Face cache 內所選 repository 嘅 revisions，
唔會刪 input、output、work、`.venv` 或其他 models；Qwen3 會同時處理 ASR 同 forced
aligner。之後可用 `uv run mrp download-model --asr <model-set>` 還原；model
download 仍然係唯一可連網步驟，例如 `uv run mrp download-model --asr vibevoice`。
5. 唔可以 upload media、transcript 或 context，亦唔可以改用 cloud ASR。

## Step 1 — Extract

執行：

```bash
uv run mrp extract <input> --asr auto
```

必要時可以加入 `--context-file`、`--output-dir`、`--work-dir`、`--cache-dir` 或 `--progress auto|on|off`。亦可以用 `ASR_PROGRESS=auto|on|off` 控制進度輸出。預設 `auto` 先 Qwen3，只喺客觀 hard failure fallback SenseVoice：backend exception、空白、Unicode 標點／符號比例超過 90%，或至少 10 秒 active audio 但少過 3 個 substantive 字元。
如要明確評估 VibeVoice，使用 `uv run mrp extract <input> --asr vibevoice`。VibeVoice 輸入係 24 kHz、現時 MPS path 使用 FP32、最多單次 60 分鐘；預設 `acoustic_tokenizer_chunk_size` 係 64000 samples，可用 `--vibevoice-acoustic-chunk-size` 調整（正整數、3200 倍數）。較細 chunk 只降低 tokenizer peak memory，唔保證長錄音一定 fit unified memory；project 唔會停用 PyTorch MPS high-watermark protection。`--language` 只保留喺 request／attempt metadata，canonical transcript language 係 `und`，因為 backend 未提供 verified language detection。完整 speaker/timestamp structured output 會保留喺 attempt metadata，但 canonical schema 暫時唔啟用 diarization。

一般專有名詞、accuracy、punctuation 或 segmentation 問題唔可以觸發自動 fallback。Qwen raw
backend timing 同 canonical timing 分開保存：zero-duration word 只會 deterministic repair
canonical copy；negative、non-finite、backwards 或 malformed timing 會整組放棄 word-level
timing，改用 valid chunk timing 或總時長估算，唔會因 timing defect 觸發 SenseVoice fallback。
SenseVoice 固定用 30 秒 chunk、Cantonese → `yue`，只提供 coarse chunk timing；較後 chunk
失敗時 diagnostic 會保留 completed chunk metadata，但 partial output 唔會當成功。使用者如要求
人工重試，另行執行 `--asr sensevoice` 並使用另一 output directory，避免覆蓋第一次 JSON。

如果 `auto` 因客觀 hard failure fallback，進度會先顯示 `fallback` transition 同下一個 model loading，之後由新 backend 重新顯示自己嘅 `transcribing` progress；唔會將 model loading 假裝成 transcription，亦唔會沿用上一個 backend 嘅 percentage。

Extract 成功只會產生：

```text
<output-dir>/<input-stem>.transcript.json
```

檢查 `status`、`selected_attempt_id`、`attempts`、`warnings` 同 `transcript`。Raw attempt 必須保留；唔融合、刪除或以 postprocessed text 覆蓋。

完成後必須停止並回報：

```text
本機轉錄已完成並寫出 transcript JSON。
流程已停喺 extract；未有 export、整理逐字稿或生成會議記錄。
```

### Batch（只係逐一重用 extract）

如要處理一個 directory，執行：

```bash
uv run mrp batch <input-directory> --asr auto
```

Batch 會按檔名排序，逐一產生 `<input-stem>.transcript.json`，並喺進度輸出顯示 `File N of M`。開始第一個 transcription 前，程式會先計算全部 destinations；同 stem 或 macOS case-insensitive 等價 destination 嘅 input 會 fail closed。`--overwrite` 只授權取代已存在嘅精確 destination，唔會放寬同一 batch 內嘅 collision。

如要比較本機已安裝嘅 ASR backend，執行：

```bash
uv run mrp benchmark meeting.mp4
uv run mrp benchmark meeting.mp4 --include-experimental
```

Benchmark 預設只會 sequentially 嘗試完整 installed 嘅 Qwen3 同 SenseVoice，
每個 backend 都係 explicit extraction，唔使用 `auto`，亦永遠唔下載 model。Qwen3
缺少 ASR 或 forced aligner、SenseVoice 缺少 model、或 runtime prerequisite
不可用時會 `skipped`；VibeVoice 係 experimental，只有 `--include-experimental`
先 eligible。輸出會隔離喺 `output/benchmark/<input-stem>/<backend>/`，並寫
`benchmark.json`。Runtime／RTF 只代表 performance，唔係 accuracy；冇 reference
transcript 時唔會推算 WER/CER。至少一個 backend 成功時 return 0，否則 return 1。

## Step 2 — Export（只在明確要求時）

執行：

```bash
uv run mrp export <input-stem>.transcript.json
```

只輸出同名 `.txt` 同 `.srt`。`status: failed` 嘅 JSON 唔可以 export。SenseVoice/estimated timing warnings 要原樣告知使用者，唔聲稱係 word-level timestamp。VibeVoice 只有完整 structured result 全部通過 validation 先使用 model timing；任何 malformed／incomplete record 都會放棄全部 model timing，保留完整 text 並估算字幕時間。

完成後停止；唔生成 summary、notes、clean transcript 或 action items。

## Failure handling

- 指定 `--asr qwen3`、`--asr sensevoice` 或 `--asr vibevoice` 時，唔可以靜默換 backend；VibeVoice 失敗時唔會 fallback 到其他 backend。
- 所有 attempt 失敗時，保留 `status: failed` diagnostic JSON，報告非零狀態並停止；人類可讀
  錯誤會包括最具體嘅 backend/model/chunk reason，machine-readable `error` 仍可為
  `all_asr_attempts_failed`。
- output 已存在時，預設拒絕覆蓋。只喺使用者明確授權取代該精確檔案先用 `--overwrite`。
- 唔刪 input。Work cleanup 只可由程式清理本次 run 建立嘅 scoped directory。

## Text policy

- Canonical output 使用香港繁體，但保留 attempt raw text。
- 保留廣東話口語、語氣、中英夾雜同原意；唔翻譯或改寫成普通話書面語。
- Context/profile 只作 ASR hint；唔可以加入錄音冇講過嘅內容。

## Out of scope

Canonical speaker diarization promotion、video OCR/frame analysis、Whisper、cloud ASR、live captions、自動 transcript cleanup、meeting notes、summary、action-item extraction。

詳細 contract：`SPEC.md`。Implementation/data flow：`ARCHITECTURE.md`。Machine-readable schema：`schemas/transcript-v1.schema.json`。
