# Meeting Recording Processor — Product & Technical Spec

**版本：** 0.1.0  
**狀態：** Phase 1 source complete；待 Apple Silicon 實機 model smoke test  
**平台：** Apple Silicon macOS arm64  
**語言：** 香港廣東話／中英夾雜；canonical output 使用香港繁體

## 1. 目標與工作流程

工具將本機錄音或錄影轉成可追溯 transcript，並刻意分開兩個 command：

| Command | Input | Output | Stop condition |
|---|---|---|---|
| `extract` | `.m4a/.mp3/.wav/.flac/.mp4/.mov` | `<stem>.transcript.json` | 寫出 JSON 後停止 |
| `batch` | 支援格式嘅 input directory | 每個 input 一個 `.transcript.json` | 所有檔案完成或記錄 failure 後停止 |
| `export` | 成功嘅 `.transcript.json` | `<stem>.txt` + `<stem>.srt` | 寫出兩個格式後停止 |

任何 command 都唔會自動產生 clean transcript、meeting notes、summary 或 action items。上述衍生內容只可由使用者另行要求，並唔屬於本工具 v1 pipeline。

## 2. Functional requirements

### FR-01 Platform and input

- 只支援 Darwin arm64；其他平台 fail closed 並顯示原因。
- Python `>=3.11,<3.14`，以 `uv` 管理 environment。
- 接受 `.m4a`、`.mp3`、`.wav`、`.flac`、`.mp4`、`.mov`；extension 比對不分大小寫。
- 原始 input 永不修改或刪除。

### FR-02 Media handling

- 以 `ffprobe` 讀 metadata、duration 並選擇第一條 audio stream；duration 缺失或無效時唔阻止 ASR，進度會退回 indeterminate。
- 冇 audio stream 時失敗；影片 frame 永遠唔會傳入 ASR。
- Qwen3/SenseVoice 以明確 stream map 將 audio normalize 成 16 kHz、mono、16-bit PCM WAV；VibeVoice 明確 mode 使用 24 kHz、mono、16-bit PCM WAV。
- 記錄 container、duration、stream、signal stats、ffprobe version。

### FR-03 ASR backends

| Mode | Runtime/model | 行為 |
|---|---|---|
| `qwen3` | `mlx-qwen3-asr==0.3.5`, `Qwen/Qwen3-ASR-1.7B` | 單一 attempt；失敗不 fallback |
| `sensevoice` | `mlx-audio==0.4.1`, `mlx-community/SenseVoiceSmall` | 單一 attempt；30 秒 chunk inference |
| `vibevoice` | `torch`, `transformers>=5.3.0,<5.4.0`, `microsoft/VibeVoice-ASR-HF` | explicit evaluation；MPS、24 kHz、最多 60 分鐘；失敗不 fallback |
| `auto` | Qwen3 + SenseVoice | 先 Qwen3；只因 hard failure fallback；不包括 VibeVoice |

Backend imports 同 model load 必須 lazy；CLI help、schema、export、unit tests 唔應觸發 MLX/Transformers import 或下載。

### FR-04 Objective fallback

Qwen3 attempt 符合任一條件時，`auto` 先執行 SenseVoice：

1. backend exception；
2. strip 後輸出為空；
3. 非空白字元中，Unicode punctuation/symbol 比例 `> 0.90`；
4. WAV signal detector 測到 active audio `>= 10.0s`，但 Unicode letter/number 少過 3 個。

專有名詞錯誤、一般 accuracy、標點或分段差異屬主觀品質，唔觸發 fallback。系統唔自動比較、融合或刪除 attempts。明確 `--asr qwen3`／`--asr sensevoice`／`--asr vibevoice` 絕不暗中改用另一 backend。

### FR-05 Offline and models

- `extract` 固定設定 Hugging Face offline mode，並以 `local_files_only=True` resolve model。
- 缺少 snapshot 時提示先執行 `download-model`，不使用 cloud API 或其他 model。
- 只有 `download-model` command 可啟用 network 下載。
- 預設 cache 為 `<project>/.cache/huggingface/`；cache path 同 snapshot id 寫入 provenance。
- VibeVoice 使用 native Transformers API，extract 只會由已 resolve 嘅 local snapshot 建立 processor/model。

### FR-06 Text policy

- 預設語言 `Cantonese`；Qwen 可指定 `auto`，SenseVoice 對應 `Cantonese → yue`。
- raw backend output 原封不動保留喺 attempt。
- canonical text 只做 Unicode NFC、control/whitespace normalization、OpenCC `s2hk` 同 subtitle cue grouping。
- 唔翻譯、唔摘要、唔將廣東話改寫成普通話書面語。
- context file 係 opt-in；path、SHA-256 同 backend limitation 寫入 JSON。
- VibeVoice 嘅 `--language` 不作 model conditioning，只記錄為 metadata；`profile_text` 只經 verified `prompt` interface 傳入。

### FR-07 Timing

- Qwen word/model timestamps 轉成 canonical segments，`timing_source` 為 `model`。
- VibeVoice 有效 structured `Start`／`End` records 轉成 canonical segments，`timing_source` 為 `model`；speaker id 只保留喺 attempt metadata。
- Qwen 如只得 chunk timing，或 SenseVoice chunk output，cue timing必須標成估算來源。
- 完全冇 timestamp 時，按 audio duration 同文字長度估算，並加入 warning。
- SRT 只由 canonical segments 產生，唔聲稱 estimated timing 係 word-level timestamp。

### FR-07a Progress

- Progress events 使用 backend-neutral `phase/current/total/unit/elapsed/message/determinate` contract。
- Qwen3 以 runtime structured progress callback 嘅 processed audio seconds／total duration 報告實際 transcription progress；無可靠 total 時唔顯示百分比。
- TTY 使用 compact updating display；非 TTY 只輸出 plain log lines，唔包含 cursor-control escape sequences。
- `ASR_PROGRESS=auto|on|off` 控制 progress output，亦可用 `--progress`；預設係 `auto`。
- VibeVoice 暫時冇 verified processed-audio progress source，transcribing phase 使用 indeterminate progress，唔由 elapsed time 或 token count 估算百分比。
- `batch` 顯示 current file index、total files、filename，同 current file 嘅 progress；唔以檔案數量冒充 duration-weighted aggregate。
- Progress renderer 或 backend callback 出現普通 I/O／reporting exception 時必須 fail open：停用後續 progress output，但唔可以改變 ASR、quality gate 或 fallback；`KeyboardInterrupt` 仍然要傳出。
- Rendered lifecycle phases 只可以向前行；成功而無 fallback 嘅 run 順序係 `transcribing → writing-output → completed`。`auto` objective hard failure 會先輸出 backend-neutral `fallback` transition（顯示下一個 model loading），再由下一個 backend 以自己嘅 `transcribing` progress 重新由 0% 開始；failed run 絕不輸出 `completed`。

### FR-08 Output and collision safety

`extract` 只寫一個 JSON：

```text
<output-dir>/<input-stem>.transcript.json
```

JSON 包含 source metadata/hash、request、immutable attempts、selected attempt、canonical transcript、processing、warnings、error 同 tool provenance。所有文字檔以 temp file + `fsync` + atomic replace 寫入。

預設不覆蓋任何已存在 output；只有明確 `--overwrite` 可以取代。所有 attempts 都嵌入 JSON，避免 fallback 證據散失。

`batch` 必須喺第一個 `extract` 前預先計算全部 `<input-stem>.transcript.json` destinations，並拒絕同 stem 或 macOS case-insensitive 等價 destination 嘅 input collision。`--overwrite` 唔可以繞過 intra-batch collision；collision error 必須列出 conflicting inputs 同 shared destination。

`export` 同時 preflight TXT/SRT collision；failed 或無效 JSON 一律拒絕：

```text
<output-dir>/<input-stem>.txt
<output-dir>/<input-stem>.srt
```

### FR-09 Failure behavior

- 所有 ASR attempts 失敗：原子寫出 `status: "failed"` diagnostic JSON，CLI return code 1。
- 成功 JSON：`status: "completed"`、非空 `selected_attempt_id`、transcript text 同至少一個 segment。
- work file 預設只清理本 run directory；`--keep-work-files` 先保留 normalized WAV。
- Ctrl-C return code 130；configuration/media/model/schema errors return code 1。

## 3. CLI contract

```text
mrp extract INPUT
  [--asr auto|qwen3|sensevoice|vibevoice]
  [--language VALUE]
  [--context-file PATH]
  [--output-dir PATH]
  [--work-dir PATH]
  [--cache-dir PATH]
  [--qwen-model MODEL_ID]
  [--sensevoice-model MODEL_ID]
  [--vibevoice-model MODEL_ID]
  [--keep-work-files]
  [--overwrite]
  [--verbose]
  [--progress auto|on|off]

mrp batch INPUT_DIRECTORY
  [same ASR, path, overwrite, verbose and progress options as extract]

mrp export TRANSCRIPT_JSON
  [--output-dir PATH]
  [--overwrite]

mrp doctor [--cache-dir PATH] [--json]
mrp download-model [--asr qwen3|sensevoice|vibevoice|all] [model/cache overrides]
mrp cache-size [--cache-dir PATH]
```

## 4. Canonical package

簡化例子：

```json
{
  "schema_version": "1.0",
  "status": "completed",
  "run_id": "20260820T120000Z-a1b2c3d4",
  "source": {"name": "meeting.m4a", "sha256": "..."},
  "request": {"asr": "auto", "offline": true},
  "attempts": [
    {
      "attempt_id": "attempt-1-qwen3",
      "backend": "qwen3",
      "raw_text": "...",
      "quality": {"hard_failure": false, "reasons": []}
    }
  ],
  "selected_attempt_id": "attempt-1-qwen3",
  "transcript": {
    "language": "Cantonese",
    "text": "今日我哋討論部署。",
    "segments": [
      {"start": 0.0, "end": 2.4, "text": "今日我哋討論部署。", "timing_source": "model"}
    ]
  },
  "warnings": [],
  "error": null
}
```

Normative machine-readable schema：`schemas/transcript-v1.schema.json`。

## 5. Non-functional requirements

- **Privacy：** runtime 不 upload media、transcript、context 或 metadata。
- **Auditability：** model id/snapshot、attempt raw output、fallback reason、版本、hash 可追溯。
- **Determinism：** media parameters、post-processing、quality thresholds 固定。
- **Safety：** input immutable、output fail-on-collision、atomic write、scoped work cleanup。
- **Testability：** platform/media/model/backend 全部可注入 fake；unit tests 無需大型 model。
- **Dependency policy：** runtime pins 保留喺 `pyproject.toml`；正式發佈前需於支援平台產生並提交 `uv.lock`。

## 6. Acceptance status

- [x] 完整 canonical CLI：`extract`／`export`／`doctor`／model/cache commands。
- [x] input validation、ffprobe、explicit stream selection、Qwen3/SenseVoice 16 kHz mono normalization。
- [x] Qwen3 同 SenseVoice adapter；backend lazy load。
- [x] `auto` 客觀 hard-failure routing；明確 mode 無 silent fallback。
- [x] raw attempts、quality、snapshot、provenance 同 failed diagnostic preservation。
- [x] 香港繁體 deterministic post-processing，無內容改寫。
- [x] JSON → TXT/SRT export；failed JSON fail closed。
- [x] unit/integration tests 無需真 model。
- [x] TTY/plain progress、indeterminate fallback、batch file reporting 同 failure cleanup。
- [x] shell wrappers、baseline preservation、完整文件同 JSON Schema。
- [x] VibeVoice explicit backend、native Transformers model management、24 kHz media path、metadata provenance 同 model-free tests。
- [ ] Apple Silicon 實機下載兩個 models 並完成真實 M4A/MP3/WAV/FLAC/MP4/MOV smoke test。
- [ ] Apple Silicon 實機下載 VibeVoice、MPS offline inference 同 structured speaker/timestamp smoke test。
- [ ] Apple Silicon 上產生／驗證 `uv.lock` 同量度 cache/runtime disk usage。

## 7. Deferred／out of scope

Canonical speaker/diarization schema promotion、VibeVoice streaming、>60 分鐘 segmentation/reconciliation、live captions、Whisper、cloud ASR、video OCR/frame analysis、automatic transcript cleanup、meeting-note/action-item generation。任何新增功能必須另行更新 privacy、schema 同 acceptance contract。
