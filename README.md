# spark-tts-open-tts

**English** · [中文](./README.zh.md)

OpenAI-compatible HTTP TTS service built on top of
[Spark-TTS](https://github.com/SparkAudio/Spark-TTS). Ships as a single CUDA
container image on GHCR.

Implements the [Open TTS spec](https://github.com/OpenTTSGroup/open-tts-spec):

- `POST /v1/audio/speech` — OpenAI-compatible synthesis (clone only)
- `POST /v1/audio/clone` — one-shot zero-shot cloning (multipart upload)
- `POST /v1/audio/design` — controllable voice design (gender / pitch / speed)
- `GET  /v1/audio/voices` — list file-based voices
- `GET  /v1/audio/voices/preview?id=...` — download a reference WAV
- `GET  /healthz` — engine status, capabilities, concurrency snapshot

Six output formats (`mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`); mono
`float32` encoded server-side. File voices live on disk as
`${VOICES_DIR}/<id>.{wav,txt,yml}` triples.

## Quick start

```bash
mkdir -p voices cache

# Drop a 5–15 s reference WAV plus its transcript:
cp ~/my-ref.wav voices/alice.wav
echo "This is the transcript of the reference clip." > voices/alice.txt

docker run --rm --gpus all -p 8000:8000 \
  -v "$PWD/voices:/voices:ro" \
  -v "$PWD/cache:/root/.cache" \
  ghcr.io/openttsgroup/spark-tts-open-tts:latest
```

First boot downloads the model weights (~5 GB) to `/root/.cache`. Mount the
cache directory to avoid repeat downloads. `/healthz` reports
`status="loading"` until the engine is ready.

```bash
curl -s localhost:8000/healthz | jq

# Zero-shot clone (file voice)
curl -X POST localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello from Spark-TTS.","voice":"file://alice","response_format":"mp3"}' \
  -o out-clone.mp3

# Controllable voice design — no reference audio needed
curl -X POST localhost:8000/v1/audio/design \
  -H 'Content-Type: application/json' \
  -d '{"input":"Designed voice demo.","gender":"female","pitch":"moderate","speed":"moderate"}' \
  -o out-design.mp3

# One-shot clone with an uploaded reference clip
curl -X POST localhost:8000/v1/audio/clone \
  -F "audio=@reference.wav" \
  -F "prompt_text=Transcript of the reference clip." \
  -F "input=Generated speech using the uploaded reference." \
  -o out-oneshot.mp3
```

## Capabilities

| capability | value | notes |
|---|---|---|
| `clone` | `true` | zero-shot via `voice="file://..."` or `POST /v1/audio/clone` |
| `streaming` | `false` | Spark-TTS generates semantic tokens in one shot; `/v1/audio/realtime` is not exposed |
| `design` | `true` | controllable generation via gender / pitch / speed on `POST /v1/audio/design` |
| `languages` | `false` | Chinese + English work inline without a language field |
| `builtin_voices` | `false` | Spark-TTS ships no SFT voice bank; `/speech` requires `voice="file://..."` |

## Environment variables

### Engine (prefixed `SPARKTTS_`)

| variable | default | description |
|---|---|---|
| `SPARKTTS_MODEL` | `SparkAudio/Spark-TTS-0.5B` | HuggingFace repo id, or a local directory |
| `SPARKTTS_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `SPARKTTS_CUDA_INDEX` | `0` | GPU index when multiple are visible |
| `SPARKTTS_DTYPE` | `bfloat16` | `float16` / `bfloat16` / `float32`; the LLM is cast to this dtype on CUDA, CPU is always `float32`. `float16` can produce `inf`/`nan` logits during sampling on Qwen2.5-0.5B — switch to `float32` if you must run on Turing/Volta (T4/V100) where bf16 isn't supported |
| `SPARKTTS_TEMPERATURE` | `0.8` | LLM sampling default |
| `SPARKTTS_TOP_K` | `50` | LLM sampling default |
| `SPARKTTS_TOP_P` | `0.95` | LLM sampling default |
| `SPARKTTS_MAX_NEW_TOKENS` | `3000` | LLM generation ceiling on the clone path |
| `SPARKTTS_PROMPT_CACHE_SIZE` | `16` | LRU size for cached `(global, semantic)` BiCodec tokens of file voices |
| `SPARKTTS_DEFAULT_GENDER` | `female` | fallback for `/v1/audio/design` when the field is omitted |
| `SPARKTTS_DEFAULT_PITCH` | `moderate` | fallback for `/v1/audio/design` when the field is omitted |
| `SPARKTTS_DEFAULT_SPEED` | `moderate` | fallback for `/v1/audio/design` when the field is omitted |

### Service-level (no prefix)

| variable | default | description |
|---|---|---|
| `HOST` | `0.0.0.0` | |
| `PORT` | `8000` | |
| `LOG_LEVEL` | `info` | uvicorn log level |
| `VOICES_DIR` | `/voices` | scan root for file-based voices |
| `MAX_INPUT_CHARS` | `8000` | 413 above this |
| `DEFAULT_RESPONSE_FORMAT` | `mp3` | |
| `MAX_CONCURRENCY` | `1` | in-flight synthesis ceiling |
| `MAX_QUEUE_SIZE` | `0` | 0 = unbounded queue |
| `QUEUE_TIMEOUT` | `0` | seconds; 0 = unbounded wait |
| `MAX_AUDIO_BYTES` | `20971520` | upload limit for `/v1/audio/clone` |
| `CORS_ENABLED` | `false` | `true` mounts a `CORSMiddleware` that allows any origin / method / header on every endpoint (no credentials — see the [spec](https://github.com/OpenTTSGroup/open-tts-spec/blob/main/http-api-spec.md#37-cors)). Keep `false` when the service is fronted by a reverse proxy or called same-origin. |

## Compose

See [`docker/docker-compose.example.yml`](docker/docker-compose.example.yml).

## API request parameters

GET endpoints (`/healthz`, `/v1/audio/voices`, `/v1/audio/voices/preview`)
take no body and at most a single `id` query parameter — see the
[Open TTS spec](https://github.com/OpenTTSGroup/open-tts-spec/blob/main/http-api-spec.md)
for their response shape.

The tables below describe the POST endpoints that accept a request body. The
**Status** column uses a fixed vocabulary:

- **required** — rejected with 422 if missing.
- **supported** — accepted and consumed by Spark-TTS.
- **ignored** — accepted for OpenAI compatibility; has no effect.
- **conditional** — behaviour depends on other fields or the loaded model;
  see the notes column.
- **extension** — Spark-TTS-specific field, not part of the Open TTS spec.

### `POST /v1/audio/speech` (application/json)

| Field | Type | Default | Status | Notes |
|---|---|---|---|---|
| `model` | string | `null` | ignored | OpenAI compatibility only. |
| `input` | string | — | required | 1..`MAX_INPUT_CHARS` chars. Empty ⇒ 422, over limit ⇒ 413. |
| `voice` | string | — | required | **Must use `file://<id>` prefix** — Spark-TTS has no built-in voices. Bare names return 422. |
| `response_format` | enum | `mp3` | supported | One of `mp3`/`opus`/`aac`/`flac`/`wav`/`pcm`. |
| `speed` | float | `1.0` | conditional | Accepted (range `[0.25, 4.0]`) but **ignored with a warning** — Spark-TTS' zero-shot path has no speed knob. Use `/v1/audio/design` (5-level enum) for speed control. |
| `instructions` | string \| null | `null` | conditional | Accepted but **ignored with a warning** — Spark-TTS does not expose instruct-style prompting on the zero-shot path. |
| `temperature` | float | `SPARKTTS_TEMPERATURE` | extension | LLM sampling. Range `[0.0, 2.0]`. |
| `top_p` | float | `SPARKTTS_TOP_P` | extension | LLM sampling. Range `[0.0, 1.0]`. |
| `top_k` | int | `SPARKTTS_TOP_K` | extension | LLM sampling. `>= 0`. |

### `POST /v1/audio/clone` (multipart/form-data)

| Field | Type | Default | Status | Notes |
|---|---|---|---|---|
| `audio` | file | — | required | Extension must be one of `.wav/.mp3/.flac/.ogg/.opus/.m4a/.aac/.webm`. Over `MAX_AUDIO_BYTES` ⇒ 413. The upload is never persisted to `${VOICES_DIR}`. |
| `prompt_text` | string | — | required | Reference-clip transcript. Empty ⇒ 422. |
| `input` | string | — | required | Same semantics as `/speech.input`. |
| `response_format` | string | `mp3` | supported | Same as `/speech`. |
| `speed` | float | `1.0` | conditional | Accepted but ignored (same reason as `/speech.speed`). |
| `instructions` | string \| null | `null` | conditional | Accepted but ignored (same reason as `/speech.instructions`). |
| `model` | string | `null` | ignored | OpenAI compatibility only. |

### `POST /v1/audio/design` (application/json)

Text-only voice design. Spark-TTS does not accept free-form `instruct` text —
instead the controllable path consumes three discrete knobs. The request
field `instruct` is accepted for spec compatibility and **logged only**.

| Field | Type | Default | Status | Notes |
|---|---|---|---|---|
| `input` | string | — | required | 1..`MAX_INPUT_CHARS` chars. |
| `instruct` | string \| null | `null` | conditional | **Accepted and logged but not parsed.** Use the structured fields below for control. |
| `response_format` | enum | `mp3` | supported | Same set as `/speech`. |
| `gender` | `female` \| `male` | `SPARKTTS_DEFAULT_GENDER` | extension | |
| `pitch` | 5-level enum | `SPARKTTS_DEFAULT_PITCH` | extension | One of `very_low` / `low` / `moderate` / `high` / `very_high`. |
| `speed` | 5-level enum | `SPARKTTS_DEFAULT_SPEED` | extension | Same 5 levels. **Note:** this is an enum, **not** the 0.25–4.0 multiplier used on `/speech.speed`. |
| `temperature` | float | `SPARKTTS_TEMPERATURE` | extension | |
| `top_p` | float | `SPARKTTS_TOP_P` | extension | |
| `top_k` | int | `SPARKTTS_TOP_K` | extension | |

## Known limitations

- **No streaming.** Spark-TTS produces semantic tokens in one `model.generate()`
  call; `/v1/audio/realtime` is not exposed. `/healthz.capabilities.streaming`
  is `false`.
- **No built-in voices.** `/speech` requires `voice="file://..."`. Unprefixed
  voice names return 422. Use `/v1/audio/design` if you want synthesis
  without a reference clip.
- **`speed` and `instructions` are no-ops on clone paths.** Spark-TTS' zero-shot
  inference has no speed or instruct parameter. The fields are accepted for
  OpenAI compatibility and logged as warnings. For speed control, route to
  `/v1/audio/design` and pass `speed=very_low|low|moderate|high|very_high`.
- **`/v1/audio/design.instruct` is logged but not parsed.** The field exists
  for open-tts spec compatibility. All real control goes through the
  extension fields `gender` / `pitch` / `speed`.
- **CPU mode works but is slow** — several minutes per short utterance; the
  dtype is forced to `float32`. Recommended for debugging only.
- **Sample rate is fixed at 16 kHz** (Spark-TTS' native output).
- **`SPARKTTS_MAX_NEW_TOKENS` only affects the clone path.** The design path
  reuses upstream's `SparkTTS.inference()`, which hard-codes 3000 internally.
