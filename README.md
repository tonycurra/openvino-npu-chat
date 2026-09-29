# Local NPU talk

Personal laptop chat on Intel NPU. Inference stays on this machine — no cloud API.

`talk.py` is a thin FastAPI + uvicorn process on http://127.0.0.1:7860 that serves the vanilla `web/index.html` UI. FastAPI is here because it is one file and one port. Django would add settings, apps, and an ORM. This app has no deploy plan and no database, so that extra structure would not help.

## Run

```bash
.venv/bin/python talk.py
```

Open http://127.0.0.1:7860

First NPU load (and first Kokoro compile) is slow. Later turns reuse `.npucache/`.

The NPU holds one LLM. Stop `zed_server.py` before starting talk, and the other way around.

## Zed

Coding in Zed uses **Qwen3 8B** (the only local model whose chat template emits tool calls):

```bash
.venv/bin/python zed_server.py
```

That serves an OpenAI-compatible API at http://127.0.0.1:8000/v1. In Zed, provider `openvino-npu`, model `Qwen3-8B-int4-cw-ov`. API key can be any non-empty string.

The compiled prompt budget is 4096 tokens, with up to 1024 new tokens. First start for that budget recompiles into `.npucache/zed-Qwen3-8B-int4-cw-ov/` and can take several minutes. Use it for a short coding turn (one file, one edit). A long agent thread is trimmed to fit.

## UI

- **Model** — any OpenVINO GenAI LLM in `models/npu/`. One loads at a time; the next message switches.
- **Tokens** — max new tokens (16–2048). That is the only length control.

Each send is one user turn. No system prompt, no history. Gemma 3 is text-only on CPU; the rest prefer NPU.

## Downloads (once)

Weights are not in git. `scripts/download.py` fetches them:

```bash
.venv/bin/python scripts/download.py            # NPU LLMs, Whisper, Piper, Kokoro
.venv/bin/python scripts/download.py kokoro     # just Kokoro if TTS is missing
.venv/bin/python scripts/download.py models     # just the NPU collection
```

- NPU LLMs: https://huggingface.co/collections/OpenVINO/llms-optimized-for-npu-686e7f0bf7bc184bd71f8ba0
- Whisper: https://huggingface.co/OpenVINO/whisper-small-int8-ov
- Kokoro: https://huggingface.co/luis-castillo/Kokoro-82M-OpenVINO-FP16-OVMS
- Piper voices: https://huggingface.co/rhasspy/piper-voices

The model ids `download.py` uses are in `config.py`. Convert a new Hugging Face LLM for NPU: [docs/npu-export.md](docs/npu-export.md). Talk already lists any extra folder you drop in `models/npu/`.

## Layout

```
talk.py                 FastAPI chat UI (port 7860)
zed_server.py           OpenAI /v1 bridge for Zed (Qwen3 on NPU, port 8000)
config.py               paths, NPU models, personas, token budget
web/index.html          vanilla UI
scripts/download.py     one-time local downloads
docs/npu-export.md      converting models for NPU
archive/                old Phi OpenAI bridge; Zed uses zed_server.py instead
```

Weights stay in `models/` and `voices/` (gitignored). Do not commit `.venv/` or `.npucache/`.

## Privacy

Chat, mic audio, and TTS stay on this laptop. No account, no remote inference.
