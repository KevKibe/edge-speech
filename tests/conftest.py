"""Shared paths and fixtures. Every test uses the real crispasr build, real models and real data."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from edge_speech.config import DEFAULT_REGISTRY_DIR, Registry, load_registry

REPO = Path(__file__).resolve().parents[1]
CRISP_BIN = REPO / "bin" / "crispasr"
LIB_PATH = REPO / "CrispASR" / "build" / "src" / ("libcrispasr.dylib" if sys.platform == "darwin" else "libcrispasr.so")
AUDIO_FILE = Path(__file__).parent / "data" / "jfk.wav"
AUDIO_TEXT = "and so my fellow americans ask not what your country can do for you ask what you can do for your country"

WHISPER_REPO = "ggerganov/whisper.cpp"
WHISPER_FILE = "ggml-tiny.en.bin"
WHISPER_HF_MODEL = f"hf:{WHISPER_REPO}:{WHISPER_FILE}"
MAX_WER = 0.2


@pytest.fixture(scope="session")
def registry() -> Registry:
    return load_registry(DEFAULT_REGISTRY_DIR)


@pytest.fixture(scope="session", autouse=True)
def _crispasr_build() -> None:
    assert CRISP_BIN.is_file(), f"{CRISP_BIN} missing: run ./build_binary.sh"
    assert LIB_PATH.is_file(), f"{LIB_PATH} missing: run ./build_binary.sh"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDGE_SPEECH_NO_DOTENV", "1")  # don't load the developer's .env
