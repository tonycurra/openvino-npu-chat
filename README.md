# Local talk

Pick a model and a personality. Last four turns stay in the browser; plus starts a new chat. Nothing is written to disk.

Audio off by default. Speaker icon turns on TTS. Switch **Piper** vs **Kokoro** to compare.

```bash
.venv/bin/python talk.py
```

Open http://127.0.0.1:7860

Kokoro (OpenVINO GenAI) if missing:

```bash
.venv/bin/python scripts/download.py kokoro
```

Convert a *new* Hugging Face LLM for NPU (not any random file): see [docs/npu-export.md](docs/npu-export.md).

First NPU load is slow (compile). Later turns use `.npucache/`.
