# NPU Talk

Local chat on an Intel NPU, plus an OpenAI-compatible API for [Zed](https://zed.dev). Speech recognition and speech synthesis sit on the same chat server. Model weights are downloaded onto the machine and are not part of this repository.

[![test](https://github.com/tonycurra/openvino-npu-chat/actions/workflows/test.yml/badge.svg)](https://github.com/tonycurra/openvino-npu-chat/actions/workflows/test.yml)

Inference stays on the computer that runs the server. Both processes bind to `127.0.0.1` and do not check an API key, so leave them off a public interface.

The NPU holds one language model. Stop the chat server before starting the Zed server, and the other way around.

## What you get

- A browser UI at <http://127.0.0.1:7860> for chat against any OpenVINO GenAI model in `models/npu/`. The page keeps the last four exchanges and sends them with the next message.
- An OpenAI-compatible API at <http://127.0.0.1:8000/v1> that serves Qwen3 8B, including tool calls, for Zed.
- Whisper speech-to-text, plus Piper and Kokoro voices on the same page. The speaker control speaks the reply. The microphone transcribes a recording into that turn.

## Requirements

- Linux on an Intel Core Ultra with an NPU. The chat server falls back to CPU or GPU when a model cannot run on the NPU. Gemma 3 answers text on CPU.
- Python 3.11 or newer. This tree is developed on CPython 3.14 with OpenVINO 2026.3.
- Disk for the weights you choose. One 7B INT4 model is about 2–4.5 GB, plus the Hugging Face cache of anything you convert yourself.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/download.py
```

`scripts/download.py` takes one argument:

| Argument | Downloads |
|---|---|
| `all` (default) | NPU LLMs, Whisper, Piper voices, Kokoro, dictionaries |
| `models` | The NPU collection listed in `config.py` |
| `whisper` | Whisper small INT8 |
| `voices` | Piper en_GB and it_IT voices |
| `kokoro` | Kokoro 82M OpenVINO FP16 |
| `dicts` | Optional pronunciation lists (the app does not read them) |

Gemma 3 is gated. Accept the license on the [model card](https://huggingface.co/OpenVINO/gemma-3-4b-it-int4-cw-ov), run `hf auth login`, then `python scripts/download.py models` again.

## Chat UI

```bash
.venv/bin/python talk.py
```

Open <http://127.0.0.1:7860>.

The header lists models already on disk. The number beside the model is the maximum count of new tokens (16–2048). The page remembers the last four exchanges in the browser, trimmed to fit the 1024-token NPU prompt. **clear** forgets them. A reload does too. The server does not write the transcript to disk. The first NPU compile for a model is slow. Later starts reuse `.npucache/`.

### Speech

The header has a Piper / Kokoro menu and a voice menu. Turn on **speak** to hear the reply. Kokoro's first compile is slow; later replies reuse `.npucache/kokoro/`. The microphone appears when Whisper is on disk. Click it to record, click again to send that recording.

The same routes are available without the page:

- `GET /api/state` — devices, downloaded models, and voices.
- `POST /api/talk` — form fields `agent`, `text`, `tokens`. Optional: `audio` (WAV), `speak=1`, `tts` (`piper` or `kokoro`), `voice`, `speaker`.
- `POST /api/speak` — form fields `text`, `voice`, `speaker`, `tts`. Returns base64 WAV.

Kokoro compiles onto the GPU when one is present, so the NPU stays free for the language model. Piper runs on CPU.

## Zed

Qwen3 8B is the coding model because its chat template emits tool calls. Phi and Mistral weights in this list do not.

```bash
.venv/bin/python zed_server.py
```

Point Zed at an OpenAI-compatible provider named however you like:

| Setting | Value |
|---|---|
| Base URL | `http://127.0.0.1:8000/v1` |
| Model | `Qwen3-8B-int4-cw-ov` |
| API key | any non-empty string |

The compiled graph accepts 4096 prompt tokens and up to 1024 new tokens. The first start for that budget writes `.npucache/zed-Qwen3-8B-int4-cw-ov/` and can take several minutes. A long agent thread is trimmed to the newest turns that still fit, and a tool call leaves together with its results.

`GET /v1/models` lists the one loaded model. `POST /v1/chat/completions` accepts `stream: true`. When the request includes tools, the stream holds tokens until the JSON tool call is complete, then emits the OpenAI `tool_calls` chunk.

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

The suite covers prompt trimming, tool-call parsing, WAV encoding, the download skip check, and the HTTP surface. It does not load a model onto the NPU. `make test` runs the same command.

## Layout

```
talk.py              chat UI and speech API, port 7860
zed_server.py        OpenAI /v1 bridge for Zed, port 8000
config.py            paths, model list, NPU token budget
web/index.html       browser UI
scripts/download.py  Hugging Face and Piper downloads
docs/npu-export.md   convert another Hugging Face LLM for the NPU
archive/             earlier Phi tool-call adapter
tests/               pytest suite
```

`models/`, `voices/`, `dicts/`, `.venv/`, and `.npucache/` stay on disk. `.gitignore` already excludes them.

## Models

The default NPU LLMs come from Intel's [LLMs optimized for NPU](https://huggingface.co/collections/OpenVINO/llms-optimized-for-npu-686e7f0bf7bc184bd71f8ba0) collection. Speech weights:

- Whisper: [OpenVINO/whisper-small-int8-ov](https://huggingface.co/OpenVINO/whisper-small-int8-ov)
- Kokoro: [luis-castillo/Kokoro-82M-OpenVINO-FP16-OVMS](https://huggingface.co/luis-castillo/Kokoro-82M-OpenVINO-FP16-OVMS)
- Piper voices: [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) at tag `v1.0.0`

To export a decoder LLM that is not in that collection, follow [docs/npu-export.md](docs/npu-export.md). Drop the result in `models/npu/<name>/`. The UI lists any folder there that contains a real `openvino_model.bin` or `openvino_language_model.bin`.

Those weights, Piper voices, and dictionaries keep their own licenses. A dictionary download writes `dicts/SOURCES.txt` with the upstream links.

## License

[MIT](LICENSE).
