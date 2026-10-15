import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "download",
    Path(__file__).resolve().parents[1] / "scripts" / "download.py",
)
download = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(download)


def test_already_have_ov_skips_a_real_blob(tmp_path):
    assert download._already_have_ov(tmp_path) is False
    blob = tmp_path / "openvino_encoder_model.bin"
    blob.write_bytes(b"x" * 1000)
    assert download._already_have_ov(tmp_path) is False
    blob.write_bytes(b"x" * 1_000_001)
    assert download._already_have_ov(tmp_path) is True
