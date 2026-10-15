import base64
import json

import numpy as np
from fastapi.testclient import TestClient

import talk


def test_short_strips_think_blocks_and_turn_markers():
    assert talk._short("<think>hidden</think>\nhello") == "hello"
    assert talk._short("answer<end_of_turn>tail") == "answer"
    assert talk._short("answer<start_of_turn>tail") == "answer"


def test_from_result_reads_text_and_length_stop():
    class Result:
        texts = ["hello"]
        finish_reasons = []

    text, truncated = talk._from_result(Result())
    assert text == "hello"
    assert truncated is False


def test_resample_changes_length_with_the_rate():
    audio = np.linspace(-1, 1, 8000, dtype=np.float32)
    up = talk._resample(audio, 8000, 16000)
    assert up.dtype == np.float32
    assert up.size == 16000
    same = talk._resample(audio, 16000, 16000)
    assert np.shares_memory(same, audio) or np.array_equal(same, audio)


def test_wav_roundtrip():
    rate = 16000
    audio = np.linspace(-0.4, 0.4, 3200, dtype=np.float32)
    raw = base64.b64decode(talk._wav_b64(rate, audio))
    back = talk._read_wav(raw)
    assert back.shape == audio.shape
    assert np.allclose(back, audio, atol=2 / 32768)


def test_read_wav_downmixes_stereo():
    import io
    import wave

    left = np.full(1600, 0.5, dtype=np.float32)
    right = np.full(1600, -0.5, dtype=np.float32)
    stereo = np.stack([left, right], axis=1)
    pcm = (np.clip(stereo, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(pcm.tobytes())
    mixed = talk._read_wav(buf.getvalue())
    assert mixed.shape == (1600,)
    assert np.allclose(mixed, 0, atol=2 / 32768)


def test_voice_meta_reads_speaker_count(tmp_path):
    voice = tmp_path / "en_GB-alba-medium.onnx"
    voice.write_bytes(b"onnx")
    (tmp_path / "en_GB-alba-medium.onnx.json").write_text(json.dumps({"num_speakers": 3}))
    assert talk._voice_meta(voice) == {"id": "en_GB-alba-medium", "speakers": 3}


def test_trim_history_keeps_the_newest_pairs_that_fit():
    long = "x" * 800
    raw = [{"role": "note", "content": "skip"}]
    for i in range(6):
        raw.append({"role": "user", "content": f"q{i} {long}"})
        raw.append({"role": "assistant", "content": f"a{i} {long}"})
    kept = talk._trim_history(raw)
    assert kept
    assert kept[0]["role"] == "user"
    assert all(m["role"] in {"user", "assistant"} for m in kept)
    assert len(kept) <= talk.HISTORY_TURNS * 2
    assert sum(len(m["content"]) for m in kept) <= talk.HISTORY_CHARS
    assert "q5" in kept[-2]["content"]


def test_trim_history_rejects_junk():
    assert talk._trim_history(None) == []
    assert talk._trim_history(["nope", {"role": "tool", "content": "x"}]) == []


def test_transcribe_skips_clips_shorter_than_a_tenth_of_a_second():
    short = np.zeros(100, dtype=np.float32)
    raw = base64.b64decode(talk._wav_b64(16000, short))
    assert talk.transcribe(raw) == ""


client = TestClient(talk.app)


def test_index_is_the_chat_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "NPU Talk" in response.text
    assert 'id="agent"' in response.text
    assert 'id="tts"' in response.text
    assert 'id="voice"' in response.text
    assert 'id="mic"' in response.text
    assert 'id="clear"' in response.text


def test_state_lists_devices_and_engines():
    body = client.get("/api/state").json()
    assert body["llm_device"] in {"NPU", "GPU", "CPU"}
    assert isinstance(body["agents"], list)
    assert {engine["id"] for engine in body["tts"]} == {"piper", "kokoro"}
    assert body["default_tts"] in {"piper", "kokoro"}
    assert body["default_kokoro"]
    for agent in body["agents"]:
        assert {"id", "label", "ready"} <= set(agent)


def test_talk_rejects_an_empty_turn():
    response = client.post("/api/talk", data={"agent": "missing", "text": "  "})
    assert response.status_code == 400
    assert "error" in response.json()


def test_speak_rejects_empty_text():
    response = client.post("/api/speak", data={"text": "", "voice": "bf_emma", "tts": "kokoro"})
    assert response.status_code == 400
