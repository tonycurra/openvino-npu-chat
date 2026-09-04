"""Shared paths and model registry for the local NPU talk stack."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

MODELS_NPU = ROOT / "models" / "npu"
MODELS_WHISPER = ROOT / "models" / "whisper"
MODELS_KOKORO = ROOT / "models" / "kokoro"
VOICES_PIPER = ROOT / "voices" / "piper"
DICTS = ROOT / "dicts"

# GenAI Text2Speech (Kokoro). Pre-exported IR + speaker bins.
# https://github.com/openvinotoolkit/openvino.genai/tree/master/samples/python/speech_generation
KOKORO_HF_ID = "luis-castillo/Kokoro-82M-OpenVINO-FP16-OVMS"
KOKORO_DIR = MODELS_KOKORO / "Kokoro-82M"
DEFAULT_KOKORO_VOICE = "bf_emma"

KOKORO_LANG = {
    "a": "en-us",
    "b": "en-gb",
    "e": "es",
    "f": "fr-fr",
    "h": "hi",
    "i": "it",
    "p": "pt-br",
}

# STT: OpenVINO Whisper INT8. Docs (GenAI on NPU): Whisper has no extra NPU
# flags — INT8 or FP16, then WhisperPipeline(..., "NPU"). Not the INT4 LLM recipe.
WHISPER_HF_ID = "OpenVINO/whisper-small-int8-ov"
WHISPER_DIR = MODELS_WHISPER / "whisper-small-int8-ov"

# LLM on NPU. Load one at a time — the NPU will not hold several of these.
# kind: "llm" uses LLMPipeline, "vlm" uses VLMPipeline (text still works).
NPU_MODELS = [
    {
        "hf_id": "OpenVINO/Phi-3.5-mini-instruct-int4-gq-ov",
        "kind": "llm",
        "label": "Phi-3.5 mini · INT4 group-quant",
    },
    {
        "hf_id": "OpenVINO/Phi-3.5-mini-instruct-int4-cw-ov",
        "kind": "llm",
        "label": "Phi-3.5 mini · INT4 channel-wise",
    },
    {
        "hf_id": "OpenVINO/Phi-3-mini-4k-instruct-int4-gq-ov",
        "kind": "llm",
        "label": "Phi-3 mini 4k · INT4 group-quant",
    },
    {
        "hf_id": "OpenVINO/Phi-3-mini-4k-instruct-int4-cw-ov",
        "kind": "llm",
        "label": "Phi-3 mini 4k · INT4 channel-wise",
    },
    {
        "hf_id": "OpenVINO/Qwen3-8B-int4-cw-ov",
        "kind": "llm",
        "label": "Qwen3 8B · INT4 channel-wise",
    },
    {
        "hf_id": "OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov",
        "kind": "llm",
        "label": "Mistral 7B Instruct v0.3 · INT4 CW",
    },
    {
        "hf_id": "OpenVINO/Mistral-7B-Instruct-v0.2-int4-cw-ov",
        "kind": "llm",
        "label": "Mistral 7B Instruct v0.2 · INT4 CW",
    },
    {
        "hf_id": "OpenVINO/DeepSeek-R1-Distill-Qwen-7B-nf4-ov",
        "kind": "llm",
        "label": "DeepSeek-R1 Distill Qwen 7B · NF4",
    },
    {
        "hf_id": "OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-gq-ov",
        "kind": "llm",
        "label": "DeepSeek-R1 Distill Qwen 1.5B · INT4 GQ",
    },
    {
        "hf_id": "OpenVINO/gemma-3-4b-it-int4-cw-ov",
        "kind": "vlm",
        "label": "Gemma 3 4B IT · INT4 CW (vision+text)",
        "gated": True,
    },
]

# Official Piper v1.0.0 voices: all en_GB + all it_IT.
# Nested HF path → flat local filename (already used by the voices you had).
PIPER_VOICES = [
    "en/en_GB/alan/low/en_GB-alan-low",
    "en/en_GB/alan/medium/en_GB-alan-medium",
    "en/en_GB/alba/medium/en_GB-alba-medium",
    "en/en_GB/aru/medium/en_GB-aru-medium",
    "en/en_GB/cori/medium/en_GB-cori-medium",
    "en/en_GB/cori/high/en_GB-cori-high",
    "en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium",
    "en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium",
    "en/en_GB/semaine/medium/en_GB-semaine-medium",
    "en/en_GB/southern_english_female/low/en_GB-southern_english_female-low",
    "en/en_GB/vctk/medium/en_GB-vctk-medium",
    "it/it_IT/paola/medium/it_IT-paola-medium",
    "it/it_IT/riccardo/x_low/it_IT-riccardo-x_low",
]
PIPER_REPO = "rhasspy/piper-voices"
PIPER_REVISION = "v1.0.0"

DEFAULT_VOICE = "en_GB-alba-medium"
DEFAULT_AGENT = "Phi-3.5-mini-instruct-int4-gq-ov"

NPU_PIPELINE_CONFIG = {
    "MAX_PROMPT_LEN": 1024,
    "GENERATE_HINT": "BEST_PERF",
}

MAX_NEW_TOKENS = 48

# Rolling chat: last N user/assistant pairs. NPU MAX_PROMPT_LEN is 1024, so
# keep this tiny — no vector store, no disk, no summaries.
HISTORY_TURNS = 4
HISTORY_CHARS = 2200  # ~550 tokens; leave room for system + this user turn
PERSONA_CHARS = 280

PERSONAS = [
    {"id": "plain", "label": "plain", "text": ""},
    {"id": "dry", "label": "dry", "text": "You are dry, brief, slightly sarcastic. No fluff."},
    {"id": "warm", "label": "warm", "text": "You are a calm, warm friend. Plain language, no cheerleading."},
    {"id": "it", "label": "italiano", "text": "Parla italiano, tono da amico, naturale, non formale."},
]


def budget_prompt(max_new_tokens: int) -> str:
    """Ask the model to finish a complete answer inside the token cap."""
    words = max(8, int(max_new_tokens * 0.65))
    if max_new_tokens <= 24:
        shape = "one short sentence"
    elif max_new_tokens <= 64:
        shape = "one or two short sentences"
    elif max_new_tokens <= 128:
        shape = "a few short sentences"
    else:
        shape = "a short paragraph"
    return (
        f"Answer in {shape}, at most {words} words. "
        "Plan a complete thought that fits this budget, then write only that. "
        "Finish every sentence. Never stop mid-sentence. "
        "No preamble, no lists. Same language as the user."
    )


def model_dir(hf_id: str) -> Path:
    return MODELS_NPU / hf_id.split("/")[-1]


def local_agents() -> list[dict]:
    """Agents whose OpenVINO IR weights are actually on disk."""
    found = []
    for spec in NPU_MODELS:
        path = model_dir(spec["hf_id"])
        blobs = (
            path / "openvino_model.bin",
            path / "openvino_language_model.bin",
        )
        if any(b.exists() and b.stat().st_size > 10_000_000 for b in blobs):
            found.append({**spec, "path": path})
    return found


def local_voices() -> list[Path]:
    return sorted(VOICES_PIPER.glob("*.onnx"))


def kokoro_ready() -> bool:
    return (KOKORO_DIR / "openvino_model.xml").exists() and any(
        (KOKORO_DIR / "voices").glob("*.bin")
    )


def local_kokoro_voices() -> list[Path]:
    voices = KOKORO_DIR / "voices"
    if not voices.is_dir():
        return []
    return sorted(voices.glob("*.bin"))


def kokoro_lang(voice_id: str) -> str:
    return KOKORO_LANG.get((voice_id or "b")[:1], "en-gb")
