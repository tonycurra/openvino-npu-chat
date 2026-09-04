#!/usr/bin/env python3
"""Download NPU LLMs, Whisper STT, Piper voices, and phonetic dictionaries."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    DICTS,
    KOKORO_DIR,
    KOKORO_HF_ID,
    MODELS_KOKORO,
    MODELS_NPU,
    MODELS_WHISPER,
    NPU_MODELS,
    PIPER_REPO,
    PIPER_REVISION,
    PIPER_VOICES,
    VOICES_PIPER,
    WHISPER_DIR,
    WHISPER_HF_ID,
    kokoro_ready,
    model_dir,
)


def _hf_token():
    from huggingface_hub import get_token

    return get_token()


def _hf_download(hf_id: str, dest: Path) -> None:
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    token = _hf_token()
    print(f"\n==> {hf_id}\n    -> {dest}")
    if not token:
        print("    no Hugging Face login (gated repos will 401)")
    snapshot_download(repo_id=hf_id, local_dir=str(dest), token=token or True)
    print("    done")


def _already_have_ov(dest: Path) -> bool:
    for name in ("openvino_model.bin", "openvino_language_model.bin", "openvino_encoder_model.bin"):
        blob = dest / name
        if blob.exists() and blob.stat().st_size > 1_000_000:
            print(f"skip (present): {dest.name} ({blob.stat().st_size / 1e6:.0f} MB)")
            return True
    return False


def download_models() -> None:
    MODELS_NPU.mkdir(parents=True, exist_ok=True)
    failed = []
    for spec in NPU_MODELS:
        dest = model_dir(spec["hf_id"])
        if _already_have_ov(dest):
            continue
        try:
            _hf_download(spec["hf_id"], dest)
        except Exception as exc:
            print(f"FAILED {spec['hf_id']}: {exc}")
            if spec.get("gated"):
                print(
                    "  This repo is gated. Accept the license on Hugging Face\n"
                    "  and run:  hf auth login\n"
                    f"  then re-run this script. Card: https://huggingface.co/{spec['hf_id']}"
                )
            failed.append(spec["hf_id"])
    if failed:
        print("\nFailed:", *failed, sep="\n  ")
        print("Continuing with other downloads. Re-run: python scripts/download.py models")


def download_whisper() -> None:
    MODELS_WHISPER.mkdir(parents=True, exist_ok=True)
    if _already_have_ov(WHISPER_DIR):
        return
    _hf_download(WHISPER_HF_ID, WHISPER_DIR)


def download_kokoro() -> None:
    MODELS_KOKORO.mkdir(parents=True, exist_ok=True)
    if kokoro_ready():
        print(f"skip (present): {KOKORO_DIR}")
        return
    _hf_download(KOKORO_HF_ID, KOKORO_DIR)


def download_voices() -> None:
    from huggingface_hub import hf_hub_download

    VOICES_PIPER.mkdir(parents=True, exist_ok=True)
    for rel in PIPER_VOICES:
        name = rel.rsplit("/", 1)[-1]
        onnx = VOICES_PIPER / f"{name}.onnx"
        cfg = VOICES_PIPER / f"{name}.onnx.json"
        for suffix, dest in ((".onnx", onnx), (".onnx.json", cfg)):
            if dest.exists() and dest.stat().st_size > 0:
                print(f"skip (present): {dest.name}")
                continue
            hf_path = f"{rel}{suffix}"
            print(f"==> {hf_path}")
            downloaded = hf_hub_download(
                repo_id=PIPER_REPO,
                filename=hf_path,
                revision=PIPER_REVISION,
            )
            _copy(downloaded, dest)
        print(f"    {name}")


def _copy(src: str, dest: Path) -> None:
    import shutil

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"skip (present): {dest.name} ({dest.stat().st_size} bytes)")
        return
    print(f"==> {url}\n    -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"    {dest.stat().st_size} bytes")


def download_dicts() -> None:
    DICTS.mkdir(parents=True, exist_ok=True)
    _fetch(
        "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict",
        DICTS / "en_cmudict.txt",
    )
    _fetch(
        "https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/en_UK.txt",
        DICTS / "en_UK_ipa.txt",
    )
    _fetch(
        "https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/en_US.txt",
        DICTS / "en_US_ipa.txt",
    )
    _fetch(
        "https://raw.githubusercontent.com/dohliam/cmu-ipa/master/it/it_ipa.txt",
        DICTS / "it_ipa.txt",
    )
    sources = DICTS / "SOURCES.txt"
    sources.write_text(
        "English ARPAbet: CMU Pronouncing Dictionary\n"
        "  https://github.com/cmusphinx/cmudict  (BSD-style)\n"
        "English IPA (UK/US): open-dict-data/ipa-dict\n"
        "  https://github.com/open-dict-data/ipa-dict  (MIT)\n"
        "Italian IPA: CMUSphinx Italian dict converted by dohliam/cmu-ipa\n"
        "  https://github.com/dohliam/cmu-ipa  (~7k words)\n"
        "\n"
        "Any word missing from these files can still be phonemized live via\n"
        "espeak-ng (bundled with piper-tts) from the Talk UI dictionary tab.\n"
    )
    print(f"wrote {sources}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "what",
        nargs="?",
        default="all",
        choices=["all", "models", "whisper", "voices", "dicts", "kokoro"],
    )
    args = parser.parse_args()
    steps = {
        "models": download_models,
        "whisper": download_whisper,
        "voices": download_voices,
        "dicts": download_dicts,
        "kokoro": download_kokoro,
    }
    order = list(steps) if args.what == "all" else [args.what]
    for name in order:
        print(f"\n######## {name} ########")
        steps[name]()
    print("\nall requested downloads finished")


if __name__ == "__main__":
    main()
