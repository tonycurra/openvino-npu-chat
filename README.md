# OpenVINO NPU Chat

A local chat app for an Intel NPU. You talk to a quantized language model in the browser, hear the reply with Piper or Kokoro, and can point [Zed](https://zed.dev) at the same machine through an OpenAI-compatible API. Inference stays on the computer that runs the server. The weights are downloaded separately and are not in this repository.

[![test](https://github.com/tonycurra/openvino-npu-chat/actions/workflows/test.yml/badge.svg)](https://github.com/tonycurra/openvino-npu-chat/actions/workflows/test.yml)

[MIT](LICENSE) license.

## What this is

OpenVINO GenAI loads an INT4 (or NF4) model and runs it on the NPU. This repo is the small layer around that:

- a browser page for conversation, speech, and a microphone
- a second process that speaks the OpenAI `/v1` protocol so Zed can call tools
- a downloader for the models Intel already exported for the NPU

There is no database and no account. The page remembers the last four exchanges in the browser and sends them with the next message. **clear**, or a reload, forgets them. The server does not write the transcript to disk.

Both servers listen on `127.0.0.1` only and do not check an API key. Leave them off a public interface. The NPU holds one language model, so stop one server before starting the other.

## Requirements

- Linux on an Intel Core Ultra whose OpenVINO device list includes `NPU`. Another machine with that same device can run the same weights. The chat server falls back to GPU, then CPU, when `NPU` is missing. The Zed server asks for `NPU` directly.
- Python 3.11 or newer. This tree is developed on CPython 3.14 with OpenVINO 2026.3.
- Disk for the weights you choose. One 7B INT4 model is about 2–4.5 GB, plus the Hugging Face cache of anything you convert yourself.

## Install

```bash
git clone git@github.com:tonycurra/openvino-npu-chat.git
cd openvino-npu-chat
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/download.py
```

`scripts/download.py` takes one argument:

| Argument | Downloads |
|---|---|
| `all` (default) | NPU language models, Whisper, Piper voices, Kokoro, dictionaries |
| `models` | The NPU collection listed in `config.py` |
| `whisper` | Whisper small INT8 |
| `voices` | Piper en_GB and it_IT voices |
| `kokoro` | Kokoro 82M OpenVINO FP16 |
| `dicts` | Pronunciation lists. The app does not read them |

Gemma 3 is gated. Accept the license on the [model card](https://huggingface.co/OpenVINO/gemma-3-4b-it-int4-cw-ov), run `hf auth login`, then `.venv/bin/python scripts/download.py models` again.

## Start

Chat, in the browser:

```bash
.venv/bin/python talk.py
```

Open <http://127.0.0.1:7860>.

Zed, instead of the chat server:

```bash
.venv/bin/python zed_server.py
```

Tests, with no model loaded:

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

`make run`, `make zed`, and `make test` are the same three commands.

## Two programs

| | Chat | Zed |
|---|---|---|
| Command | `talk.py` | `zed_server.py` |
| Address | <http://127.0.0.1:7860> | <http://127.0.0.1:8000/v1> |
| Model | Whichever folder is on disk. The default is Phi-3.5 mini | Qwen3 8B only |
| Prompt budget | 1024 tokens | 4096 tokens, up to 1024 new |
| Memory | Last four exchanges, held by the page | Zed resends the thread; this server trims it |
| Speech | Whisper, Piper, Kokoro | None |

In Zed, add an OpenAI-compatible provider:

| Setting | Value |
|---|---|
| Base URL | `http://127.0.0.1:8000/v1` |
| Model | `Qwen3-8B-int4-cw-ov` |
| API key | any non-empty string |

Qwen3 is the coding model because its chat template emits tool calls. The Phi and Mistral weights in this list do not. The first start for the 4096-token budget writes `.npucache/zed-Qwen3-8B-int4-cw-ov/` and can take several minutes. When a request includes tools, the stream holds tokens until the JSON tool call is complete, then emits the OpenAI `tool_calls` chunk.

## The chat page

The header lists models already on disk. The number beside the model is the maximum count of new tokens (16–2048). **speak** reads the reply with the engine and voice selected next to it: Kokoro (54 voices in the full download, default `bf_emma`) or Piper (English and Italian; a speaker number appears for multi-speaker voices such as VCTK). **mic** records, and a second click sends that recording as the turn. It appears when Whisper is on disk.

The first NPU compile for a model is slow. Later starts reuse `.npucache/`. Kokoro's first compile is slow too; later replies reuse `.npucache/kokoro/`. Kokoro prefers the GPU, so the NPU stays on the language model. Piper runs on CPU.

The same actions exist as form posts on port 7860:

- `GET /api/state` — devices, downloaded models, and voices
- `POST /api/talk` — `agent`, `text`, `tokens`, `history`. Optional: `audio` (WAV), `speak=1`, `tts` (`piper` or `kokoro`), `voice`, `speaker`
- `POST /api/speak` — `text`, `voice`, `speaker`, `tts`. Returns base64 WAV

## How it works

`talk.py` asks OpenVINO which devices exist and prefers `NPU`, then `GPU`, then `CPU`. A language model is loaded with `LLMPipeline`. Gemma 3 is a vision-language model: text chat on the NPU came out as word salad, so that one model is loaded with `VLMPipeline` on CPU.

The compiled NPU graph has a fixed prompt length, 1024 tokens for chat and 4096 for Zed. The first generate for a model and that length writes a cache under `.npucache/`. The page sends the recent turns as JSON. The server drops the oldest ones until the text fits about 1400 characters, then builds an OpenVINO `ChatHistory`. Nothing else is stored.

Whisper uses `WhisperPipeline`, on the NPU when that works and on CPU otherwise. Kokoro uses `Text2SpeechPipeline`. Piper is ONNX through `piper-tts`, outside OpenVINO.

`zed_server.py` is the same `LLMPipeline` behind `/v1/chat/completions` and `/v1/models`. It accepts streaming. Tool calls are parsed out of Qwen's `<tool_call>` JSON.

There is no LangChain and no retrieval index. An early terminal script tried a LangChain wrapper around a plain text prompt. The current path uses the model's own chat template instead. That script is not in the tree.

## Models

`scripts/download.py` fetches the entries in `config.py` from Intel's [LLMs optimized for NPU](https://huggingface.co/collections/OpenVINO/llms-optimized-for-npu-686e7f0bf7bc184bd71f8ba0) collection:

| Folder | Weights |
|---|---|
| Phi-3.5 mini | INT4 group-quant and INT4 channel-wise |
| Phi-3 mini 4k | INT4 group-quant and INT4 channel-wise |
| Qwen3 8B | INT4 channel-wise. This is the Zed model |
| Mistral 7B Instruct v0.2 and v0.3 | INT4 channel-wise |
| DeepSeek-R1 distill Qwen 7B | NF4. Needs a Series 2 NPU |
| DeepSeek-R1 distill Qwen 1.5B | INT4 group-quant |
| Gemma 3 4B | INT4 channel-wise vision-language weights, run as text on CPU |

Speech weights:

- Whisper: [OpenVINO/whisper-small-int8-ov](https://huggingface.co/OpenVINO/whisper-small-int8-ov)
- Kokoro: [luis-castillo/Kokoro-82M-OpenVINO-FP16-OVMS](https://huggingface.co/luis-castillo/Kokoro-82M-OpenVINO-FP16-OVMS)
- Piper voices: [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) at tag `v1.0.0`

To export a decoder model Intel has not packaged, follow [docs/npu-export.md](docs/npu-export.md). Put the result in `models/npu/<name>/`. The page lists any folder there with a real `openvino_model.bin` or `openvino_language_model.bin`.

Those weights, Piper voices, and dictionaries keep their own licenses. A dictionary download writes `dicts/SOURCES.txt` with the upstream links.

## What is in the repository

```
talk.py              chat page and speech API, port 7860
zed_server.py        OpenAI /v1 bridge for Zed, port 8000
config.py            paths, model list, token budget
web/index.html       the browser page
scripts/download.py  Hugging Face and Piper downloads
docs/npu-export.md   convert another Hugging Face model for the NPU
archive/             earlier Phi tool-call adapter, not started by anything else
tests/               pytest suite, no NPU required
```

Created on your machine, and listed in `.gitignore`:

```
.venv/       Python environment
models/      language models, Whisper, Kokoro
voices/      Piper ONNX voices
dicts/       optional pronunciation lists
.npucache/   compiled NPU graphs
```

## Tests

The suite covers history trimming, tool-call parsing, WAV encoding, the download skip check, and the HTTP surface. It does not load a model onto the NPU.
