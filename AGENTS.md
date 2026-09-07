# AGENTS.md

## 1. Purpose

This file defines the engineering rules for AI coding agents working on `meeting-recording-processor`.

The project is a **local-first meeting transcription processor for Apple Silicon macOS**, primarily targeting Cantonese and Cantonese-English mixed meetings.

Agents must optimize for:

1. transcription correctness;
2. provenance and reproducibility;
3. privacy and local execution;
4. failure isolation;
5. backend independence;
6. small, reviewable changes.

Do not trade these properties for convenience.

---

## 2. Read before changing code

Before implementing any non-trivial change, read the relevant project contracts.

At minimum inspect:

```text
AGENTS.md
SPEC.md
ARCHITECTURE.md
SKILL.md
README.md
pyproject.toml
```

When touching transcript serialization, also read:

```text
schemas/transcript-v1.schema.json
```

When touching an ASR adapter, inspect both:

```text
src/meeting_recording_processor/asr/
```

and the exact pinned upstream runtime version used by `pyproject.toml`.

Do not assume the latest upstream API matches the pinned runtime.

---

## 3. Repository contract hierarchy

Use the following responsibilities when documents overlap:

### `SPEC.md`

Normative product and technical behaviour.

It defines:

- supported workflows;
- failure behaviour;
- CLI contracts;
- fallback rules;
- output semantics;
- acceptance criteria.

### `schemas/transcript-v1.schema.json`

Normative machine-readable transcript structure.

Do not change serialized output incompatibly without explicitly updating the schema and contract.

### `ARCHITECTURE.md`

Implementation boundaries and data flow.

Use it to decide which module owns a behaviour.

### `SKILL.md`

Operational instructions for agents using the finished tool.

It must remain consistent with the actual CLI and supported formats.

### `README.md`

User-facing setup and usage documentation.

### `AGENTS.md`

Engineering and change-management rules for coding agents.

If these files disagree, do not silently invent a reconciliation.

Determine whether the implementation or documentation is stale, make the smallest contract-preserving correction, and report the discrepancy in the completion summary.

---

## 4. Core system invariants

These are hard constraints unless the user explicitly changes the project requirements.

### 4.1 Local-first processing

Transcription runtime must remain local.

Do not:

- upload recordings;
- upload extracted audio;
- upload transcript contents;
- upload OCR frames;
- send context/profile text to cloud services;
- silently replace a local backend with a cloud ASR API.

Network access is allowed only for explicit model/dependency acquisition workflows such as `download-model`.

Normal `extract` processing must remain offline after required assets have been downloaded.

---

### 4.2 Input media is immutable

Never modify or delete source media.

Temporary work must be scoped to the current run.

Cleanup must only remove work artifacts created by that run.

---

### 4.3 Extract and export remain separate

The canonical workflow is intentionally staged:

```text
media
  ↓
extract
  ↓
canonical transcript JSON
  ↓
STOP / human review
  ↓
explicit export
  ↓
TXT / SRT
```

`extract` must not automatically:

- export TXT;
- export SRT;
- clean or rewrite transcript content;
- generate meeting notes;
- generate summaries;
- extract action items.

`export` must remain an explicit operation.

Future transcript cleanup or meeting-document generation must remain a separate derived stage unless the product contract is deliberately revised.

---

### 4.4 Preserve provenance

Raw ASR output and canonical/postprocessed output are different artifacts.

Never silently replace raw attempts with corrected text.

Every ASR attempt must preserve the available provenance, including where applicable:

- backend;
- model;
- model snapshot;
- raw text;
- raw segments;
- timestamps;
- runtime;
- warnings;
- errors;
- quality result;
- timing source.

Post-processing may normalize representation but must not pretend derived timing or corrected text came directly from the model.

---

### 4.5 Cantonese content must not be rewritten into Mandarin prose

Canonical Chinese output uses Hong Kong Traditional Chinese representation where deterministic normalization applies.

Preserve:

- Cantonese wording;
- spoken register;
- English code-switching;
- technical identifiers;
- meaning and uncertainty.

Do not translate or stylistically rewrite speech during transcription post-processing.

Context/hotword files are hints only.

They must never cause content not present in the recording to be inserted deliberately.

---

## 5. Supported platform and runtime assumptions

Current target platform:

```text
macOS
Apple Silicon arm64
Python >=3.11,<3.14
uv
ffmpeg
ffprobe
```

Current primary ASR architecture includes:

```text
Qwen3-ASR
SenseVoice fallback
```

The project may add further backends such as VibeVoice.

New backends must integrate through the existing backend abstraction rather than creating a parallel pipeline.

Do not hard-code orchestration around a single ASR implementation.

---

## 6. ASR backend rules

Every backend adapter must convert backend-specific behaviour into the project-owned canonical interface.

Conceptually:

```text
normalized audio
      ↓
ASR adapter
      ↓
BackendResult
      ↓
quality / fallback routing
      ↓
canonical transcript
```

Backend adapters may know:

- backend-specific model APIs;
- backend-specific progress callbacks;
- backend-specific timestamps;
- backend-specific speaker information;
- backend-specific metadata.

Backend adapters must not decide:

- whether another backend should run;
- whether subjective accuracy is acceptable;
- final output filenames;
- transcript JSON structure outside `BackendResult`;
- whether cloud fallback should occur.

---

## 7. Fallback rules

`auto` fallback must remain objective and deterministic.

Current behaviour is conceptually:

```text
Qwen3
  ↓
objective hard-failure check
  ├── pass → use Qwen3
  └── fail → SenseVoice
```

Examples of objective hard failure include the conditions defined in `SPEC.md`, such as:

- backend exception;
- empty output;
- punctuation/symbol collapse;
- clearly insufficient substantive text for meaningful active audio.

Do not trigger automatic fallback merely because:

- a technical term may be wrong;
- punctuation is imperfect;
- segmentation looks poor;
- the transcript is subjectively less fluent.

When the user explicitly selects one backend, never silently switch to another.

Future backends such as VibeVoice must follow the same orchestration principle.

---

## 8. Progress reporting is observability, not business logic

Progress reporting is a side channel.

It must never change transcription semantics.

The required dependency direction is:

```text
ASR / pipeline
      │
      └── emits progress
              │
              ▼
          renderer
```

Never allow:

```text
renderer failure
      ↓
ASR failure
      ↓
fallback changes
```

A progress-rendering failure must fail open:

```text
progress output disabled
transcription continues unchanged
```

This applies to:

- broken stderr pipes;
- closed output streams;
- terminal rendering failures;
- non-critical progress callback exceptions.

Do not swallow `KeyboardInterrupt` or other intentional process-control signals.

---

## 9. Determinate progress must be real

Never fabricate percentages.

Valid progress sources include:

1. actual processed audio seconds / actual total audio duration;
2. backend structured chunk index / total chunks;
3. another backend-native measure that directly corresponds to completed transcription work.

Never calculate progress from:

```text
elapsed wall time / guessed total runtime
```

Never implement a timer that gradually approaches 99%.

If reliable progress is unavailable, display an indeterminate state such as:

```text
Transcribing... Elapsed: 12:43
```

This is preferable to a misleading percentage.

---

## 10. Progress phases must remain monotonic

Generic lifecycle:

```text
preparing
  ↓
loading-model
  ↓
transcribing
  ↓
writing-output
  ↓
completed
```

Failure may terminate from an appropriate phase.

Rendered events must not visually move backwards because of:

- delayed callbacks;
- heartbeat threads;
- stale events;
- deferred rendering;
- backend fallback.

For successful runs, output ordering must remain logically consistent.

For failed runs, do not print a successful pipeline completion state.

Background heartbeat mechanisms must not replay an older phase after a newer phase has already been emitted.

Add deterministic tests for lifecycle ordering when changing progress code.

---

## 11. Progress callback isolation

Backend-native progress callbacks may run inside inference code.

Therefore callback exceptions require special handling.

Do not place progress rendering inside the same broad exception boundary that translates inference errors into `BackendError`.

A callback/rendering problem must not be reported as:

```text
Qwen3-ASR execution failed
```

or equivalent.

Protect the progress boundary explicitly.

This rule applies equally to future VibeVoice and other backends.

---

## 12. TTY and non-TTY output

Interactive terminal output may use compact in-place updates.

Redirected output, CI logs, pipes, and files must use plain text.

Non-TTY output must not contain:

- cursor movement escape sequences;
- carriage-return animation;
- terminal-specific control codes.

Do not flood logs with one line per tiny progress event.

Use sensible throttling.

`ASR_PROGRESS=off` must disable progress reporting without altering transcription behaviour.

---

## 13. Batch processing rules

Batch mode reuses the canonical single-file extraction pipeline.

Do not create a second transcription implementation for batch mode.

Before processing the first file, batch mode must perform preflight validation.

At minimum preflight:

- input directory;
- supported files;
- output destinations;
- intra-batch output collisions.

### Output collision rule

Current output naming is stem based:

```text
meeting.m4a → meeting.transcript.json
```

Therefore:

```text
meeting.m4a
meeting.mp3
```

cannot safely be processed into the same output directory without disambiguation.

Batch mode must detect this before transcription starts.

`--overwrite` may authorize replacing an already-existing output belonging to that exact requested destination.

It must **not** authorize one input in the same batch to overwrite another input's output.

Fail closed on intra-batch collisions unless the product contract explicitly introduces deterministic disambiguated naming.

On the macOS target, collision detection must also consider case-insensitive filename conflicts where relevant.

---

## 14. Output safety

Default behaviour is fail-on-collision.

Do not silently overwrite transcript JSON.

Failed runs may write the defined diagnostic JSON when required by the pipeline contract.

Atomic output semantics must be preserved.

Do not leave a partially written canonical transcript pretending to be completed.

---

## 15. Media handling

Use `ffprobe` for metadata and audio-stream discovery.

Do not decode an entire recording merely to determine duration.

Normalize audio through the existing media layer.

Video inputs remain audio-only unless a future explicitly scoped video/OCR change updates the architecture.

Do not silently begin analysing frames.

Supported input formats in all of the following must remain synchronized:

- implementation;
- `SPEC.md`;
- `README.md`;
- `SKILL.md`;
- CLI help;
- tests.

---

## 16. Future VibeVoice integration

VibeVoice is a candidate future ASR backend.

Do not implement it as a special pipeline.

It must conform to the same project-owned backend contract.

Keep these concerns distinct:

```text
ASR backend
speaker attribution / diarization capability
timestamps
streaming/live capability
progress reporting
canonical transcript schema
```

A backend providing several of these capabilities does not justify bypassing the canonical layers.

Prerecorded long-form processing and live/streaming transcription may eventually use different backend configurations, but shared output/provenance rules should remain centralized.

Do not implement VibeVoice opportunistically inside unrelated changes.

---

## 17. Scope discipline

Prefer one focused concern per PR.

Before modifying code, state the intended scope internally and compare the final diff against it.

Do not add unrelated functionality merely because it is easy while editing adjacent code.

Examples:

- a progress fix should not redesign transcript schema;
- a backend addition should not redesign batch naming unless required;
- a documentation correction should not upgrade model versions;
- a UI/progress change must not change fallback policy.

If a discovered problem is important but unrelated, report it separately.

---

## 18. Dependency discipline

Do not add a dependency when the standard library or an existing dependency is sufficient.

Do not upgrade pinned ASR runtimes incidentally.

The exact runtime version matters because backend APIs and model behaviour may drift.

When an upstream feature is required:

1. verify it exists in the pinned version;
2. inspect its actual API;
3. add tests against the expected contract;
4. only then integrate it.

Model/runtime upgrades require their own compatibility assessment.

---

## 19. Documentation discipline

Whenever behaviour changes, inspect whether these files need synchronization:

```text
SPEC.md
ARCHITECTURE.md
SKILL.md
README.md
schemas/
```

Do not update every file mechanically.

Update only documents whose contract actually changed.

However, do not leave known contradictions.

Examples that must remain synchronized include:

- supported media extensions;
- CLI commands;
- fallback rules;
- progress controls;
- output formats;
- out-of-scope features.

Do not mark an acceptance criterion `[x]` merely because code was written.

Only mark it complete when the stated verification has actually been performed.

Platform-specific smoke-test items must remain incomplete until tested on the required Apple Silicon environment.

---

## 20. Testing expectations

Normal backend-independent verification:

```bash
uv sync --extra dev
uv run pytest
uv run python -m compileall -q src tests
bash -n setup.sh scripts/*.sh
```

Unit and integration tests must not require downloading multi-GB models unless they are explicitly platform smoke tests.

Use fakes/mocks for:

- ASR backends;
- model resolution;
- ffprobe/ffmpeg;
- progress output;
- clocks;
- pipeline failure conditions.

Avoid flaky timing tests.

For concurrency/progress behaviour, expose deterministic units that can be tested without `sleep()` whenever practical.

---

## 21. Required regression coverage for progress changes

When modifying progress reporting, cover at least:

- duration formatting;
- percentage clamping;
- indeterminate progress;
- TTY rendering;
- non-TTY rendering;
- progress disabled mode;
- lifecycle ordering;
- failure lifecycle;
- renderer I/O failure isolation;
- callback failure isolation;
- no false backend fallback caused by progress;
- heartbeat/stale-event protection;
- Ctrl-C terminal cleanup where practical.

Assertions must match the actual renderer being exercised.

For example, do not assert only that TTY-specific text is absent when the test is using a non-TTY stream.

---

## 22. Required regression coverage for batch changes

When modifying batch behaviour, cover at least:

- deterministic file ordering;
- empty directory;
- unsupported files;
- current-file index/total reporting;
- individual file failure handling;
- mixed success/failure exit status;
- same-stem output collisions;
- case-related collisions on the target platform semantics where applicable;
- collision rejection even with `--overwrite`;
- preflight failure before the first transcription starts.

---

## 23. Real-model smoke tests

Real ASR inference remains platform-specific.

Before claiming a backend integration is production-ready, validate on Apple Silicon with representative media.

Important fixtures should include:

- Cantonese;
- Cantonese-English code switching;
- technical terms;
- long recordings;
- at least one video container;
- failure/degenerated transcript cases.

Do not commit private meeting recordings to a public repository.

Private regression fixtures must remain outside public source control.

---

## 24. Completion report

At the end of a coding task, report:

1. files changed;
2. behaviour changed;
3. behaviour intentionally unchanged;
4. tests run;
5. test results;
6. platform-specific checks not run;
7. known limitations;
8. any documentation or contract discrepancy discovered.

Do not claim tests passed unless they were actually executed.

Do not claim Apple Silicon/model smoke testing if only mocked unit tests were run.

---

## 25. Git and diff hygiene

Before completing work:

```bash
git diff --check
git status --short
```

Review the complete diff.

Do not include:

- model files;
- Hugging Face cache;
- `.venv`;
- generated transcripts;
- work directories;
- private recordings;
- unrelated formatting churn.

Keep the implementation reviewable.

---

## 26. Default decision rule

When uncertain between a clever solution and a conservative solution, prefer the one that preserves:

```text
privacy
provenance
determinism
failure isolation
backend independence
existing contracts
```

Partial functionality with an explicit limitation is better than silently pretending unsupported behaviour is reliable.