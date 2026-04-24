# spark-tts-open-tts

[English](./README.md) · **中文**

基于 [Spark-TTS](https://github.com/SparkAudio/Spark-TTS) 构建的 OpenAI 兼容
HTTP TTS 服务。作为单一 CUDA 镜像发布到 GHCR。

实现 [Open TTS 规范](https://github.com/OpenTTSGroup/open-tts-spec)：

- `POST /v1/audio/speech` — OpenAI 兼容合成（仅克隆）
- `POST /v1/audio/clone` — 一次性上传零样本克隆（multipart）
- `POST /v1/audio/design` — 可控声音设计（gender / pitch / speed）
- `GET  /v1/audio/voices` — 列出文件型音色
- `GET  /v1/audio/voices/preview?id=...` — 下载参考音频
- `GET  /healthz` — 引擎状态、能力矩阵、并发快照

支持六种输出格式（`mp3` / `opus` / `aac` / `flac` / `wav` / `pcm`），服务端
以单声道 `float32` 编码。文件音色以 `${VOICES_DIR}/<id>.{wav,txt,yml}` 三件套
存放。

## 快速开始

```bash
mkdir -p voices cache

# 准备一段 5~15 秒参考音频 + 转录文本:
cp ~/my-ref.wav voices/alice.wav
echo "这是参考音频对应的转录文本。" > voices/alice.txt

docker run --rm --gpus all -p 8000:8000 \
  -v "$PWD/voices:/voices:ro" \
  -v "$PWD/cache:/root/.cache" \
  ghcr.io/openttsgroup/spark-tts-open-tts:latest
```

首次启动会下载模型权重（约 5 GB）到 `/root/.cache`。挂载 cache 目录可避免
重复下载。引擎加载期间 `/healthz` 返回 `status="loading"`。

```bash
curl -s localhost:8000/healthz | jq

# 零样本克隆（文件音色）
curl -X POST localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"你好，来自 Spark-TTS。","voice":"file://alice","response_format":"mp3"}' \
  -o out-clone.mp3

# 可控声音设计 — 无需参考音频
curl -X POST localhost:8000/v1/audio/design \
  -H 'Content-Type: application/json' \
  -d '{"input":"这是声音设计示例。","gender":"female","pitch":"moderate","speed":"moderate"}' \
  -o out-design.mp3

# 一次性上传参考音频做克隆
curl -X POST localhost:8000/v1/audio/clone \
  -F "audio=@reference.wav" \
  -F "prompt_text=参考音频的转录文本。" \
  -F "input=使用上传的参考音频合成这段话。" \
  -o out-oneshot.mp3
```

## 能力矩阵

| 能力 | 值 | 说明 |
|---|---|---|
| `clone` | `true` | 通过 `voice="file://..."` 或 `POST /v1/audio/clone` 做零样本克隆 |
| `streaming` | `false` | Spark-TTS 一次生成 semantic token，不提供流式；`/v1/audio/realtime` 未注册 |
| `design` | `true` | 通过 `POST /v1/audio/design` 的 gender / pitch / speed 控制生成声音 |
| `languages` | `false` | 中英文混排无需显式 language 字段 |
| `builtin_voices` | `false` | Spark-TTS 不附带 SFT 音色库；`/speech` 必须使用 `voice="file://..."` |

## 环境变量

### 引擎变量（前缀 `SPARKTTS_`）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SPARKTTS_MODEL` | `SparkAudio/Spark-TTS-0.5B` | HuggingFace repo id，或本地目录 |
| `SPARKTTS_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `SPARKTTS_CUDA_INDEX` | `0` | 多 GPU 时指定卡号 |
| `SPARKTTS_DTYPE` | `float16` | `float16` / `bfloat16` / `float32`；CUDA 下只对 LLM 做 cast，CPU 强制 `float32` |
| `SPARKTTS_TEMPERATURE` | `0.8` | LLM 采样默认值 |
| `SPARKTTS_TOP_K` | `50` | LLM 采样默认值 |
| `SPARKTTS_TOP_P` | `0.95` | LLM 采样默认值 |
| `SPARKTTS_MAX_NEW_TOKENS` | `3000` | 克隆路径 LLM 生成上限 |
| `SPARKTTS_PROMPT_CACHE_SIZE` | `16` | 文件音色 BiCodec tokens 的 LRU 缓存容量 |
| `SPARKTTS_DEFAULT_GENDER` | `female` | `/v1/audio/design` 字段缺省兜底 |
| `SPARKTTS_DEFAULT_PITCH` | `moderate` | `/v1/audio/design` 字段缺省兜底 |
| `SPARKTTS_DEFAULT_SPEED` | `moderate` | `/v1/audio/design` 字段缺省兜底 |

### 服务级变量（无前缀）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `HOST` | `0.0.0.0` | |
| `PORT` | `8000` | |
| `LOG_LEVEL` | `info` | uvicorn 日志级别 |
| `VOICES_DIR` | `/voices` | 文件音色扫描根目录 |
| `MAX_INPUT_CHARS` | `8000` | 超出返回 413 |
| `DEFAULT_RESPONSE_FORMAT` | `mp3` | |
| `MAX_CONCURRENCY` | `1` | 同时推理上限 |
| `MAX_QUEUE_SIZE` | `0` | `0` = 不限 |
| `QUEUE_TIMEOUT` | `0` | 秒；`0` = 不限 |
| `MAX_AUDIO_BYTES` | `20971520` | `/v1/audio/clone` 上传上限 |
| `CORS_ENABLED` | `false` | `true` 时挂 `CORSMiddleware`，对所有端点放通任意 origin / method / header（不开 credentials — 见[规范](https://github.com/OpenTTSGroup/open-tts-spec/blob/main/http-api-spec.md#37-cors)）。服务被反向代理或同源调用时建议保持 `false`。 |

## Compose

见 [`docker/docker-compose.example.yml`](docker/docker-compose.example.yml)。

## API 请求参数

GET 端点（`/healthz`、`/v1/audio/voices`、`/v1/audio/voices/preview`）不接收
请求体，最多一个 `id` 查询参数，响应结构参见
[Open TTS 规范](https://github.com/OpenTTSGroup/open-tts-spec/blob/main/http-api-spec.md)。

以下表格描述带请求体的 POST 端点。**Status** 列使用固定词汇：

- **required**：缺失时返回 422。
- **supported**：接受且被 Spark-TTS 消费。
- **ignored**：接受以兼容 OpenAI，无实际作用。
- **conditional**：行为取决于其他字段或加载的模型，参见说明列。
- **extension**：Spark-TTS 扩展字段，不属于 Open TTS 规范。

### `POST /v1/audio/speech` (application/json)

| 字段 | 类型 | 默认值 | Status | 说明 |
|---|---|---|---|---|
| `model` | string | `null` | ignored | 仅用于 OpenAI 兼容。 |
| `input` | string | — | required | 长度 1..`MAX_INPUT_CHARS`。空 ⇒ 422，超长 ⇒ 413。 |
| `voice` | string | — | required | **必须使用 `file://<id>` 前缀** — Spark-TTS 无内置音色。裸名返回 422。 |
| `response_format` | enum | `mp3` | supported | `mp3`/`opus`/`aac`/`flac`/`wav`/`pcm` 之一。 |
| `speed` | float | `1.0` | conditional | 接受（范围 `[0.25, 4.0]`）但 **带 warning 忽略** — Spark-TTS 零样本路径没有速度控制。请改走 `/v1/audio/design`（5 级枚举）。 |
| `instructions` | string \| null | `null` | conditional | 接受但 **带 warning 忽略** — Spark-TTS 零样本路径不提供 instruct 风格提示。 |
| `temperature` | float | `SPARKTTS_TEMPERATURE` | extension | LLM 采样。范围 `[0.0, 2.0]`。 |
| `top_p` | float | `SPARKTTS_TOP_P` | extension | LLM 采样。范围 `[0.0, 1.0]`。 |
| `top_k` | int | `SPARKTTS_TOP_K` | extension | LLM 采样。`>= 0`。 |

### `POST /v1/audio/clone` (multipart/form-data)

| 字段 | 类型 | 默认值 | Status | 说明 |
|---|---|---|---|---|
| `audio` | file | — | required | 扩展名须为 `.wav/.mp3/.flac/.ogg/.opus/.m4a/.aac/.webm` 之一。超过 `MAX_AUDIO_BYTES` ⇒ 413。上传文件不会写入 `${VOICES_DIR}`。 |
| `prompt_text` | string | — | required | 参考音频转录。空 ⇒ 422。 |
| `input` | string | — | required | 与 `/speech.input` 同语义。 |
| `response_format` | string | `mp3` | supported | 与 `/speech` 相同。 |
| `speed` | float | `1.0` | conditional | 同 `/speech.speed`，带 warning 忽略。 |
| `instructions` | string \| null | `null` | conditional | 同 `/speech.instructions`，带 warning 忽略。 |
| `model` | string | `null` | ignored | 仅用于 OpenAI 兼容。 |

### `POST /v1/audio/design` (application/json)

纯文本声音设计。Spark-TTS 不接受自然语言 `instruct`，可控路径通过三个离散
参数驱动。请求中的 `instruct` 字段接受以满足规范，**仅作日志，不解析**。

| 字段 | 类型 | 默认值 | Status | 说明 |
|---|---|---|---|---|
| `input` | string | — | required | 长度 1..`MAX_INPUT_CHARS`。 |
| `instruct` | string \| null | `null` | conditional | **接受并记入日志，但不解析**。请使用下方结构化字段精确控制。 |
| `response_format` | enum | `mp3` | supported | 同 `/speech`。 |
| `gender` | `female` \| `male` | `SPARKTTS_DEFAULT_GENDER` | extension | |
| `pitch` | 5 级枚举 | `SPARKTTS_DEFAULT_PITCH` | extension | `very_low` / `low` / `moderate` / `high` / `very_high` 之一。 |
| `speed` | 5 级枚举 | `SPARKTTS_DEFAULT_SPEED` | extension | 同 5 级枚举。**注意：此处是枚举，不是 `/speech.speed` 的 0.25–4.0 倍数。** |
| `temperature` | float | `SPARKTTS_TEMPERATURE` | extension | |
| `top_p` | float | `SPARKTTS_TOP_P` | extension | |
| `top_k` | int | `SPARKTTS_TOP_K` | extension | |

## 已知限制

- **不支持流式。** Spark-TTS 用一次 `model.generate()` 输出 semantic tokens，
  `/v1/audio/realtime` 未注册，`/healthz.capabilities.streaming` 固定为
  `false`。
- **无内置音色。** `/speech` 必须传 `voice="file://..."`，裸名返回 422。
  若需无参考音频合成，请走 `/v1/audio/design`。
- **克隆路径不消费 `speed` 和 `instructions`。** Spark-TTS 零样本路径无
  速度/指令参数。字段接受以兼容 OpenAI，调用时会写入 warning 日志。需要
  速度控制请走 `/v1/audio/design` 的 `speed=very_low|low|moderate|high|very_high`。
- **`/v1/audio/design.instruct` 仅作日志，不解析。** 该字段保留以满足
  open-tts 规范；真正的控制通过扩展字段 `gender` / `pitch` / `speed`。
- **CPU 模式可跑但慢** — 短句生成可能几分钟，dtype 自动强制为 `float32`。
  建议仅用于调试。
- **采样率固定为 16 kHz**（Spark-TTS 原生输出）。
- **`SPARKTTS_MAX_NEW_TOKENS` 仅作用于克隆路径。** design 路径复用上游
  `SparkTTS.inference()`，其内部硬编码为 3000。
