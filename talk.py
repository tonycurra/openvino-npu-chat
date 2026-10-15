#!/usr/bin/env python3
"""Local NPU chat. One prompt, pick a model.

    .venv/bin/python talk.py

Then open http://127.0.0.1:7860
"""

from __future__ import annotations

import base64
import io
import json
import re
import threading
import time
import wave
from pathlib import Path

import numpy as np
import openvino as ov
import openvino_genai as ov_genai
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from piper import PiperVoice, SynthesisConfig

from config import (
    DEFAULT_AGENT,
    DEFAULT_VOICE,
    KOKORO_DIR,
    MAX_NEW_TOKENS,
    MODELS_NPU,
    NPU_MODELS,
    NPU_PIPELINE_CONFIG,
    ROOT,
    VOICES_PIPER,
    WHISPER_DIR,
    kokoro_lang,
    kokoro_ready,
    local_agents,
    local_kokoro_voices,
    local_voices,
    model_dir,
)

WEB = ROOT / "web" / "index.html"
_AVAILABLE = ov.Core().available_devices


def _pick(preferred, fallback="CPU"):
    for name in preferred:
        if name in _AVAILABLE:
            return name
    return fallback


LLM_DEVICE = _pick(["NPU", "GPU", "CPU"])
STT_DEVICE = _pick(["NPU", "CPU"])
# Kokoro is FP16 (~190 MB IR). Keep NPU for the LLM; iGPU is much faster than CPU.
TTS_DEVICE = _pick(["GPU", "CPU"])

_lock = threading.Lock()
_llm = _llm_id = _llm_kind = _llm_device = _whisper = _voice = _voice_name = _kokoro = None
_kokoro_emb = {}

app = FastAPI()


def _load_whisper():
    global _whisper, STT_DEVICE
    if _whisper is not None:
        return _whisper
    if not (WHISPER_DIR / "openvino_encoder_model.xml").exists():
        raise FileNotFoundError("run: python scripts/download.py whisper")
    order = [STT_DEVICE] + (["CPU"] if STT_DEVICE != "CPU" else [])
    last = None
    for dev in order:
        try:
            print(f"Whisper on {dev}")
            kwargs = {}
            if dev == "NPU":
                cache = ROOT / ".npucache" / "whisper"
                cache.mkdir(parents=True, exist_ok=True)
                kwargs["CACHE_DIR"] = str(cache)
            _whisper = ov_genai.WhisperPipeline(str(WHISPER_DIR), dev, **kwargs)
            STT_DEVICE = dev
            return _whisper
        except Exception as exc:
            last = exc
            print(f"Whisper {dev} failed: {exc}")
            _whisper = None
    raise RuntimeError(f"Whisper failed on {order}: {last}")


def _load_llm(agent_id):
    global _llm, _llm_id, _llm_kind, _llm_device
    if _llm is not None and _llm_id == agent_id:
        return _llm, _llm_kind
    spec = next((s for s in NPU_MODELS if s["hf_id"].endswith(agent_id)), None) or {
        "hf_id": f"OpenVINO/{agent_id}",
        "kind": "llm",
    }
    path = model_dir(spec["hf_id"])
    kind = spec.get("kind", "llm")
    cache = ROOT / ".npucache" / agent_id
    cache.mkdir(parents=True, exist_ok=True)
    _llm = None
    attempts = []
    if kind == "vlm":
        # NPU compiles this Gemma VLM (~1.7 GB blobs) but text-only decode is
        # word salad. CPU produces real Italian/English chat. Keep NPU for LLMs.
        attempts.append(("CPU", {}))
        if "GPU" in _AVAILABLE:
            attempts.append(("GPU", {}))
    else:
        npu_cfg = {**NPU_PIPELINE_CONFIG, "CACHE_DIR": str(cache)}
        attempts.append((LLM_DEVICE, npu_cfg if LLM_DEVICE == "NPU" else {}))
        if LLM_DEVICE == "NPU":
            attempts.append(("NPU", {}))
    last = None
    for dev, extra in attempts:
        try:
            print(f"Loading {agent_id} on {dev} ({kind})")
            if kind == "vlm":
                _llm = ov_genai.VLMPipeline(str(path), dev, **extra)
            elif extra:
                _llm = ov_genai.LLMPipeline(str(path), dev, extra)
            else:
                _llm = ov_genai.LLMPipeline(str(path), dev)
            _llm_id, _llm_kind, _llm_device = agent_id, kind, dev
            return _llm, kind
        except Exception as exc:
            last = exc
            print(f"{agent_id} {dev} failed: {exc}")
            _llm = None
    raise RuntimeError(f"{agent_id} failed: {last}")


def _load_voice(name):
    global _voice, _voice_name
    if _voice is None or _voice_name != name:
        _voice = PiperVoice.load(VOICES_PIPER / f"{name}.onnx")
        _voice_name = name
    return _voice


def _resample(audio, orig, target=16000):
    audio = audio.astype(np.float32)
    if orig == target or audio.size == 0:
        return audio
    n = int(round(audio.size * target / orig))
    x0 = np.linspace(0, 1, audio.size, endpoint=False)
    x1 = np.linspace(0, 1, n, endpoint=False)
    return np.interp(x1, x0, audio).astype(np.float32)


def _read_wav(data: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(data), "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        raw = w.readframes(w.getnframes())
        sw = w.getsampwidth()
    if sw == 2:
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        pcm = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    if nch > 1:
        pcm = pcm.reshape(-1, nch).mean(axis=1)
    return _resample(pcm, sr, 16000)


def transcribe(wav_bytes: bytes) -> str:
    samples = _read_wav(wav_bytes)
    if samples.size < 1600:
        return ""
    result = _load_whisper().generate(samples.tolist(), task="transcribe")
    text = getattr(result, "texts", None) or getattr(result, "text", None) or str(result)
    if isinstance(text, (list, tuple)):
        text = " ".join(map(str, text))
    return str(text).strip()


def _short(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.split(r"<end_of_turn>|<start_of_turn>", text, maxsplit=1)[0].strip()
    return text


def _from_result(result) -> tuple[str, bool]:
    text = getattr(result, "texts", None) or getattr(result, "text", None) or str(result)
    if isinstance(text, (list, tuple)):
        text = text[0] if text else ""
    reasons = getattr(result, "finish_reasons", None) or []
    truncated = any(r == ov_genai.GenerationFinishReason.LENGTH for r in reasons)
    return str(text), truncated


def _gen_cfg(pipe, kind: str, max_new_tokens: int) -> ov_genai.GenerationConfig:
    if kind == "vlm":
        try:
            cfg = pipe.get_generation_config()
        except Exception:
            cfg = ov_genai.GenerationConfig()
        cfg.max_new_tokens = max_new_tokens
        cfg.do_sample = True
        cfg.top_k = 64
        cfg.top_p = 0.95
        try:
            cfg.stop_strings = {"<end_of_turn>", "<start_of_turn>"}
        except Exception:
            pass
        return cfg
    return ov_genai.GenerationConfig(max_new_tokens=max_new_tokens)


def generate_text(agent_id: str, user_text: str, max_new_tokens: int) -> str:
    """One user turn. No system prompt, no history. The token box is the only cap."""
    pipe, kind = _load_llm(agent_id)
    cfg = _gen_cfg(pipe, kind, max_new_tokens)
    if kind == "vlm":
        text, _cut = _from_result(pipe.generate(user_text, generation_config=cfg))
        return _short(text)
    history = ov_genai.ChatHistory([{"role": "user", "content": user_text}])
    try:
        history.set_extra_context({"enable_thinking": False})
    except Exception:
        pass
    text, _cut = _from_result(pipe.generate(history, generation_config=cfg))
    return _short(text)


def synthesize(engine: str, voice_name: str, text: str, speaker_id: int | None) -> tuple[int, np.ndarray]:
    if engine == "kokoro":
        return synthesize_kokoro(voice_name, text)
    return synthesize_piper(voice_name, text, speaker_id)


def synthesize_piper(voice_name: str, text: str, speaker_id: int | None) -> tuple[int, np.ndarray]:
    voice = _load_voice(voice_name)
    cfg = SynthesisConfig()
    if speaker_id is not None and voice.config.num_speakers > 1:
        cfg.speaker_id = int(speaker_id)
    chunks = [c.audio_float_array for c in voice.synthesize(text, syn_config=cfg)]
    rate = voice.config.sample_rate
    return rate, np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)


def _load_kokoro():
    global _kokoro
    if _kokoro is None:
        if not kokoro_ready():
            raise FileNotFoundError("run: python scripts/download.py kokoro")
        cache = ROOT / ".npucache" / "kokoro"
        cache.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        print(f"Kokoro compile on {TTS_DEVICE} (first load is slow)…")
        try:
            _kokoro = ov_genai.Text2SpeechPipeline(
                str(KOKORO_DIR), TTS_DEVICE, {"CACHE_DIR": str(cache)}
            )
        except Exception:
            _kokoro = ov_genai.Text2SpeechPipeline(str(KOKORO_DIR), TTS_DEVICE)
        print(f"Kokoro ready on {TTS_DEVICE} in {time.perf_counter() - t0:.1f}s")
    return _kokoro


def _kokoro_embed(voice_name: str) -> ov.Tensor:
    if voice_name in _kokoro_emb:
        return _kokoro_emb[voice_name]
    path = KOKORO_DIR / "voices" / f"{voice_name}.bin"
    data = np.fromfile(path, dtype=np.float32)
    pipe = _load_kokoro()
    if hasattr(pipe, "get_speaker_embedding_shape"):
        shape = tuple(pipe.get_speaker_embedding_shape())
        data = data.reshape(shape)
    else:
        data = data.reshape(1, -1)
    tensor = ov.Tensor(data)
    _kokoro_emb[voice_name] = tensor
    return tensor


def synthesize_kokoro(voice_name: str, text: str) -> tuple[int, np.ndarray]:
    pipe = _load_kokoro()
    t0 = time.perf_counter()
    result = pipe.generate(
        text,
        _kokoro_embed(voice_name),
        language=kokoro_lang(voice_name),
    )
    speech = result.speeches[0]
    wav = np.array(speech.data).reshape(-1).astype(np.float32)
    rate = int(getattr(result, "output_sample_rate", 24000) or 24000)
    print(f"Kokoro generate {len(text)} chars → {wav.size / max(rate, 1):.1f}s audio in {time.perf_counter() - t0:.1f}s")
    return rate, wav


def _wav_b64(rate: int, audio: np.ndarray) -> str:
    pcm = np.clip(audio, -1, 1)
    pcm = (pcm * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def _voice_meta(path: Path) -> dict:
    n = 1
    cfg = path.with_suffix(".onnx.json")
    if not cfg.exists():
        cfg = Path(str(path) + ".json")
    if cfg.exists():
        n = int(json.loads(cfg.read_text()).get("num_speakers") or 1)
    return {"id": path.stem, "speakers": n}


@app.get("/")
def index():
    return FileResponse(WEB)


@app.get("/api/state")
def state():
    ready = {a["path"].name for a in local_agents()}
    agents = []
    for spec in NPU_MODELS:
        name = spec["hf_id"].split("/")[-1]
        agents.append({
            "id": name,
            "label": spec["label"],
            "ready": name in ready,
            "note": spec.get("note", ""),
            "kind": spec.get("kind", "llm"),
        })
    for p in sorted(MODELS_NPU.glob("*")):
        if p.is_dir() and p.name not in {s["hf_id"].split("/")[-1] for s in NPU_MODELS}:
            agents.append({"id": p.name, "label": p.name, "ready": True})
    return {
        "llm_device": LLM_DEVICE,
        "stt_device": STT_DEVICE,
        "whisper": (WHISPER_DIR / "openvino_encoder_model.xml").exists(),
        "default_agent": DEFAULT_AGENT if DEFAULT_AGENT in ready else (next(iter(ready), DEFAULT_AGENT)),
        "default_voice": DEFAULT_VOICE,
        "default_tokens": MAX_NEW_TOKENS,
        "loaded": _llm_id or "",
        "agents": agents,
        "tts": [
            {
                "id": "piper",
                "label": "Piper",
                "ready": bool(local_voices()),
                "voices": [_voice_meta(p) for p in local_voices()],
            },
            {
                "id": "kokoro",
                "label": "Kokoro",
                "ready": kokoro_ready(),
                "voices": [{"id": p.stem, "speakers": 1, "lang": kokoro_lang(p.stem)} for p in local_kokoro_voices()],
            },
        ],
    }


@app.post("/api/talk")
async def talk(
    agent: str = Form(...),
    voice: str = Form(""),
    speaker: int = Form(0),
    speak: int = Form(0),
    tts: str = Form("piper"),
    tokens: int = Form(MAX_NEW_TOKENS),
    text: str = Form(""),
    audio: UploadFile | None = File(None),
):
    tokens = max(16, min(2048, int(tokens or MAX_NEW_TOKENS)))
    with _lock:
        heard = ""
        if audio is not None:
            blob = await audio.read()
            if blob:
                try:
                    heard = transcribe(blob)
                except Exception as exc:
                    return JSONResponse({"error": f"STT: {exc}"}, 500)
        user = (text or "").strip() or heard
        if not user:
            return JSONResponse({"error": "type or record something"}, 400)
        t_llm = time.perf_counter()
        try:
            reply = generate_text(agent, user, tokens)
        except Exception as exc:
            return JSONResponse({"heard": user, "error": f"LLM: {exc}"}, 500)
        llm_s = time.perf_counter() - t_llm
        b64 = ""
        tts_s = 0.0
        engine = ""
        if speak and voice:
            try:
                engine = tts if tts in ("piper", "kokoro") else "piper"
                t_tts = time.perf_counter()
                rate, wav = synthesize(engine, voice, reply, speaker)
                tts_s = time.perf_counter() - t_tts
                if wav.size:
                    b64 = _wav_b64(rate, wav)
            except Exception as exc:
                return {
                    "heard": user,
                    "reply": reply,
                    "audio": "",
                    "status": f"text ok, tts failed: {exc}",
                }
        bits = [f"ok · llm {_llm_device or LLM_DEVICE} {llm_s:.1f}s"]
        if heard:
            bits.append(f"stt {STT_DEVICE}")
        if engine:
            bits.append(f"{engine} {TTS_DEVICE if engine == 'kokoro' else 'cpu'} {tts_s:.1f}s")
        return {
            "heard": user,
            "reply": reply,
            "agent": _llm_id or agent,
            "device": _llm_device or LLM_DEVICE,
            "audio": b64,
            "status": " · ".join(bits),
        }


@app.post("/api/speak")
async def speak(
    text: str = Form(""),
    voice: str = Form(""),
    speaker: int = Form(0),
    tts: str = Form("piper"),
):
    phrase = (text or "").strip()
    if not phrase:
        return JSONResponse({"error": "empty text"}, 400)
    with _lock:
        try:
            engine = tts if tts in ("piper", "kokoro") else "piper"
            t0 = time.perf_counter()
            rate, wav = synthesize(engine, voice, phrase, speaker)
            tts_s = time.perf_counter() - t0
            if not wav.size:
                return JSONResponse({"error": "tts produced no audio"}, 500)
            dev = TTS_DEVICE if engine == "kokoro" else "cpu"
            return {
                "audio": _wav_b64(rate, wav),
                "status": f"ok · {engine} {dev} {tts_s:.1f}s",
            }
        except Exception as exc:
            return JSONResponse({"error": f"tts failed: {exc}"}, 500)


if __name__ == "__main__":
    print(f"LLM={LLM_DEVICE} STT={STT_DEVICE} TTS={TTS_DEVICE} http://127.0.0.1:7860")
    uvicorn.run(app, host="127.0.0.1", port=7860)
