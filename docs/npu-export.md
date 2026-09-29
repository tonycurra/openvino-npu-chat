# Converting models for Intel NPU

Short version: **you cannot take an arbitrary file and magically run it on the NPU.** You *can* convert many Hugging Face **decoder LLMs** (and some VLMs / Whisper) into OpenVINO IR with a specific INT4 recipe, then load them with GenAI on `NPU`. That is a one-evening job for a supported 3B–8B instruct model. Exotic architectures, huge MoE, and “draw me a picture” diffusion models are a different, harder path.

Official reference: [OpenVINO GenAI on NPU](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html). Ready-made weights: [LLMs optimized for NPU](https://huggingface.co/collections/OpenVINO/llms-optimized-for-npu-686e7f0bf7bc184bd71f8ba0).

This machine already has GenAI **2026.3.1**. Export is **offline after the download**: Hugging Face is only used to fetch the original weights (and optionally a calibration set). Runtime stays local.

---

## How possible is “any model”?

| Source | Convert for NPU chat? | Effort |
|---|---|---|
| Already in [OpenVINO NPU collection](https://huggingface.co/collections/OpenVINO/llms-optimized-for-npu-686e7f0bf7bc184bd71f8ba0) | Download only. No convert. | Minutes. **Do this first.** |
| Common HF instruct LLM (Llama, Mistral, Phi, Qwen, Gemma) ≤ ~8B | Yes, `optimum-cli` + NPU flags. | 30–90 min first time (tools + export). |
| Same, but 14B–32B | Maybe. RAM and first NPU **compile** hurt. | Hours, may OOM. |
| GPTQ / already-INT4 HF repo | Often `optimum-cli export openvino -m …` with no extra flags. | Easier than FP16. |
| Whisper | Yes, different flags (INT8, not INT4-CW). | Easy. We already use `whisper-small-int8-ov`. |
| VLM (Gemma 3, etc.) | Same family of export; GenAI `VLMPipeline`. | Medium. Gemma 3 4B is on disk here (gated HF repo). |
| Diffusion / SD / Flux (image **generation**) | `Text2ImagePipeline`, usually **CPU/GPU**. NPU can take text-encoder; UNet often stays CPU. | Separate project. Not the NPU LLM recipe. |
| Random `.gguf` / `.safetensors` custom arch / huge MoE | Often no. GenAI must know the topology. | Hard or impossible. |
| Piper / Kokoro TTS | Not NPU LLMs. Piper is ONNX on CPU. Kokoro is GenAI speech on GPU/CPU so the NPU stays on chat. | Already local. |

NPU is picky: **symmetric INT4 (or NF4)**, **`--sym`**, **`--ratio 1.0`**, and either **channel-wise** (`--group-size -1`) or **group-128**. A default CPU INT8/FP16 export often **will not** load on NPU.

NF4 needs Intel Core Ultra **Series 2** (Lunar Lake) NPU and up.

---

## Two different “waits”

1. **Export** (once): Hugging Face model → `openvino_model.xml` + `.bin` + tokenizer. CPU/RAM bound. You can do this without the NPU.
2. **NPU compile** (once per model + config, then cached in `.npucache/`): first `LLMPipeline(..., "NPU")` can take **minutes**. Later loads reuse the blob.

Talk already sets `CACHE_DIR` under `.npucache/`.

---

## Resources (ballpark)

| Piece | Typical |
|---|---|
| RAM for export of a 7B–8B | **16 GB+** comfortable. 32 GB nicer. Docs: Series 2, **>16 GB** if you push long prompts on 7B+. |
| Disk for one 7B INT4 | **~2–4.5 GB** IR (see `models/npu/` here). Plus Hugging Face cache of the original. |
| CPU time, data-free INT4-CW 7B | **~10–40 min** depending on CPU. |
| Extra: `--awq --scale-estimation --dataset wikitext2` | **+30–90 min**, downloads WikiText, better accuracy. |
| First NPU compile 7B–8B | **several minutes** (once). |
| Tools | Separate venv recommended (Optimum pins Transformers; talk’s venv should stay GenAI-runtime). |

You do **not** need a GPU for the NPU recipe. A GPU only helps if you also export/run diffusion on iGPU.

---

## Easiest path (no convert)

```bash
# already how this repo got Phi / Mistral / Qwen / DeepSeek
.venv/bin/python scripts/download.py models
```

Or pull one extra card from the collection and drop it in `models/npu/<name>/`. Talk lists any folder there that has a real `openvino_model.bin`.

---

## Convert a Hugging Face LLM yourself

Use a **throwaway venv** so Optimum does not fight the talk runtime. Pins change every OpenVINO release; check the [NPU GenAI page](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html) for the current `nncf` / `optimum-intel` / `transformers` trio.

```bash
python3 -m venv ~/venvs/ov-export
source ~/venvs/ov-export/bin/activate
pip install -U pip
# pins move every OpenVINO minor — this laptop is GenAI 2026.3.1
pip install nncf onnx optimum-intel transformers
pip install openvino openvino-tokenizers openvino-genai
```

**NPU-friendly INT4 channel-wise** (best starting point, especially ≥5B):

```bash
optimum-cli export openvino \
  -m Qwen/Qwen2.5-7B-Instruct \
  --weight-format int4 --sym --ratio 1.0 --group-size -1 \
  /home/t/code/openvino/models/npu/Qwen2.5-7B-Instruct-int4-cw-ov
```

**Smaller models (≤ ~4–5B)** can use group-128:

```bash
optimum-cli export openvino \
  -m microsoft/Phi-3.5-mini-instruct \
  --weight-format int4 --sym --ratio 1.0 --group-size 128 \
  /home/t/code/openvino/models/npu/Phi-3.5-mini-instruct-int4-gq-ov
```

`--group-size -1` means channel-wise. **Not** `1`.

**Better accuracy** (slower, needs dataset):

```bash
optimum-cli export openvino \
  -m meta-llama/Meta-Llama-3.1-8B-Instruct \
  --weight-format int4 --sym --ratio 1.0 --group-size -1 \
  --awq --scale-estimation --dataset wikitext2 \
  Llama-3.1-8B-Instruct-int4-cw-ov
```

Gated repos: accept the license on Hugging Face, then `huggingface-cli login`.

Already INT4 GPTQ:

```bash
optimum-cli export openvino -m TheBloke/Llama-2-7B-Chat-GPTQ
```

### Smoke test (still local)

```python
import openvino_genai as ov_genai
pipe = ov_genai.LLMPipeline("models/npu/Your-Export-int4-cw-ov", "NPU")
print(pipe.generate("Say hello in one sentence.", max_new_tokens=32))
```

If that works, refresh talk — it already scans `models/npu/*`.

---

## Whisper (STT)

Not the LLM INT4 recipe. INT8 (or FP16) is the documented NPU path:

```bash
optimum-cli export openvino --trust-remote-code \
  --model openai/whisper-small whisper-small-int8 --weight-format int8
```

This repo already uses `models/whisper/whisper-small-int8-ov` via `WhisperPipeline` (NPU, CPU fallback). Mic audio stays on the machine.

---

## Image generation

That is **not** converting a chat LLM. You export a **diffuser** (`--library diffusers` / SD, LCM, SDXL) and run `Text2ImagePipeline`. Intel’s NPU story here is **heterogeneous**: text encoder on NPU, heavy UNet often CPU/GPU. Treat it as a second app, not a dropdown on the existing NPU chat models.

---

## What usually fails

- FP16/INT8 LLM export loaded with `"NPU"` → compile error or glacial load.
- New architecture GenAI does not implement yet.
- Mixture-of-experts / very wide models → RAM or compiler limits.
- First compile with huge `MAX_PROMPT_LEN` → more SRAM/DRAM, longer compile.
- Mixing Optimum’s Transformers pin inside the **talk** venv until chat import breaks.

---

## Suggested order if you want a new chat model

1. Search the [NPU collection](https://huggingface.co/collections/OpenVINO/llms-optimized-for-npu-686e7f0bf7bc184bd71f8ba0) — download if it exists.
2. If not, data-free INT4-CW export (`--group-size -1`).
3. If quality is mushy, redo with `--awq --scale-estimation --dataset wikitext2`.
4. First NPU generate once; keep `.npucache/`.
5. Point talk at `models/npu/<folder>` (automatic if the bin is there).

For this laptop, **download-first is the low-effort path**. Convert yourself when you want a model Intel has not packaged — a supported 7B instruct is doable in one sitting; “literally any weights” is not.
