import os

import config


def test_kokoro_lang_uses_the_voice_prefix():
    assert config.kokoro_lang("bf_emma") == "en-gb"
    assert config.kokoro_lang("if_sara") == "it"
    assert config.kokoro_lang("") == "en-gb"
    assert config.kokoro_lang("zz_unknown") == "en-gb"


def test_model_dir_is_the_repo_folder_name():
    path = config.model_dir("OpenVINO/Qwen3-8B-int4-cw-ov")
    assert path == config.MODELS_NPU / "Qwen3-8B-int4-cw-ov"


def test_local_agents_ignore_tiny_blobs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_NPU", tmp_path)
    assert config.local_agents() == []

    spec = config.NPU_MODELS[0]
    dest = tmp_path / spec["hf_id"].split("/")[-1]
    dest.mkdir()
    blob = dest / "openvino_model.bin"
    blob.write_bytes(b"x" * 1000)
    assert config.local_agents() == []

    os.truncate(blob, 10_000_001)
    found = config.local_agents()
    assert [item["hf_id"] for item in found] == [spec["hf_id"]]
    assert found[0]["path"] == dest


def test_local_agents_accept_vlm_language_blob(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_NPU", tmp_path)
    spec = next(item for item in config.NPU_MODELS if item["kind"] == "vlm")
    dest = tmp_path / spec["hf_id"].split("/")[-1]
    dest.mkdir()
    blob = dest / "openvino_language_model.bin"
    blob.touch()
    os.truncate(blob, 10_000_001)
    found = config.local_agents()
    assert found[0]["kind"] == "vlm"


def test_kokoro_ready_needs_ir_and_a_voice(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "KOKORO_DIR", tmp_path)
    assert config.kokoro_ready() is False
    (tmp_path / "openvino_model.xml").write_text("<net/>")
    assert config.kokoro_ready() is False
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "bf_emma.bin").write_bytes(b"\x00\x00")
    assert config.kokoro_ready() is True
    assert [path.name for path in config.local_kokoro_voices()] == ["bf_emma.bin"]
