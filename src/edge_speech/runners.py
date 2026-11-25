"""Transcribers: run CrispASR on a batch of utterances.

* ``CliTranscriber`` runs one ``crispasr`` process per batch (``-f`` repeated).
  Each process loads the model, so timings include one model load per batch;
  the warm-up only keeps downloads out of them. Per-utterance latency is the
  batch wall time split by audio duration (``latency_method = "batch_apportioned"``).
* ``SessionTranscriber`` uses the ``crispasr`` Python bindings: the model loads
  once per job, before timing, and each utterance is timed individually, so
  timings measure inference only.
"""

from __future__ import annotations

import errno
import json
import os
import re
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import soundfile as sf

from .config import ModelSpec, RunOptions

# Speech, not silence: with --vad, crispasr writes no output for audio without speech.
WARMUP_AUDIO = Path(__file__).parent / "assets" / "warmup.wav"
STDERR_TAIL = 2000
VERSION_PATTERN = re.compile(r"version\s*:\s*(\S+)")
GIT_SHA_PATTERN = re.compile(r"git sha\s*:\s*(\S+)")
CRISPASR_WARNING = "crispasr: warning:"
IGNORED_OPTION_MARKER = "does not support"


@dataclass
class Utterance:
    audio: np.ndarray
    sampling_rate: int

    @property
    def duration(self) -> float:
        if not self.sampling_rate:
            return 0.0
        return len(self.audio) / self.sampling_rate


@dataclass
class Transcript:
    text: str | None  # None = failed
    latency: float
    latency_method: str
    error: str | None = None
    model_file: str | None = None


class TranscriberError(RuntimeError):
    pass


class MalformedOutput(ValueError):
    pass


class Transcriber(ABC):
    def __enter__(self) -> Transcriber:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        pass

    def warmup(self) -> str:
        audio, sampling_rate = sf.read(WARMUP_AUDIO, dtype="float32")
        result = self.transcribe_batch([Utterance(audio, sampling_rate)])[0]
        if result.error is not None:
            raise TranscriberError(f"warm-up failed: {result.error}")
        if not result.model_file:
            raise TranscriberError("warm-up failed: crispasr did not report which model file it loaded")
        return result.model_file

    @abstractmethod
    def transcribe_batch(self, batch: Sequence[Utterance]) -> list[Transcript]: ...

    @abstractmethod
    def version(self) -> str: ...


def _local_model_path(model: ModelSpec) -> str:
    path = Path(model.path or "").expanduser()
    if not path.is_file():
        raise TranscriberError(f"model file not found: {path}")
    return str(path)


def _download_hf_model(model: ModelSpec, token: str | None, downloader: Callable[..., str] | None) -> str:
    if downloader is None:
        from huggingface_hub import hf_hub_download

        downloader = hf_hub_download
    try:
        return str(downloader(repo_id=model.repo, filename=model.file, token=token))
    except Exception as exc:  # network/auth/404 all surface the same way to the user
        raise TranscriberError(f"could not download {model.source()}: {exc}") from exc


def resolve_model_path(
    model: ModelSpec,
    *,
    token: str | None = None,
    downloader: Callable[..., str] | None = None,
) -> str | None:
    if model.kind == "path":
        return _local_model_path(model)
    if model.kind == "hf":
        return _download_hf_model(model, token, downloader)
    return None  # "auto": CrispASR resolves and caches it


def model_cli_args(model: ModelSpec, local_path: str | None) -> list[str]:
    if local_path is not None:
        return ["-m", local_path]
    if model.quant:
        return ["-m", f"auto:{model.quant}"]
    return ["-m", "auto"]


def _segment_text(segment: Any) -> str:
    if isinstance(segment, dict) and "text" in segment:
        text = segment["text"]
    elif hasattr(segment, "text"):
        text = segment.text
    else:
        raise MalformedOutput(f"segment without text: {segment!r}")
    return str(text or "").strip()


def _segments_text(segments: Any) -> str:
    if not isinstance(segments, list):
        raise MalformedOutput(f"expected a list of segments, got {type(segments).__name__}")
    texts = []
    for segment in segments:
        text = _segment_text(segment)
        if text:
            texts.append(text)
    return " ".join(texts)


def _check_executable(path: Path) -> None:
    if not path.is_file():
        raise TranscriberError(f"crispasr binary not found: {path} (run ./build_binary.sh or pass --crisp-bin)")
    if not os.access(path, os.X_OK):
        raise TranscriberError(f"crispasr binary is not executable: {path}")


def _parse_version(text: str) -> str:
    version = VERSION_PATTERN.search(text)
    if version is None:
        raise TranscriberError(f"could not read a version from crispasr --version output: {text.strip()[:200]!r}")
    sha = GIT_SHA_PATTERN.search(text)
    if sha is None:
        return version.group(1)
    return f"{version.group(1)}+{sha.group(1)}"


def _write_wavs(batch: Sequence[Utterance], directory: Path) -> tuple[list[Path], list[Path]]:
    wav_paths = []
    output_prefixes = []
    for number, utterance in enumerate(batch):
        wav_path = directory / f"{number:05d}.wav"
        sf.write(wav_path, utterance.audio, utterance.sampling_rate, subtype="PCM_16")
        wav_paths.append(wav_path)
        output_prefixes.append(directory / f"{number:05d}")
    return wav_paths, output_prefixes


def _apportioned_latencies(batch: Sequence[Utterance], wall_time: float) -> list[float]:
    total_audio = sum(utterance.duration for utterance in batch)
    latencies = []
    for utterance in batch:
        if total_audio > 0:
            share = utterance.duration / total_audio
        else:
            share = 1.0 / len(batch)
        latencies.append(wall_time * share)
    return latencies


def _read_transcript(
    output_prefix: Path,
    latency: float,
    latency_method: str,
    process: subprocess.CompletedProcess[str],
) -> Transcript:
    output_json = output_prefix.with_suffix(".json")
    if not output_json.is_file():
        reason = f"no output (exit code {process.returncode})"
        stderr_tail = (process.stderr or "")[-STDERR_TAIL:].strip()
        if stderr_tail:
            reason = f"{reason}: {stderr_tail}"
        return Transcript(None, latency, latency_method, reason)
    try:
        with open(output_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return Transcript(None, latency, latency_method, f"unreadable output JSON: {exc}")
    try:
        text = _segments_text(payload.get("transcription"))
    except MalformedOutput as exc:
        return Transcript(None, latency, latency_method, f"malformed output JSON {output_json.name}: {exc}")
    model_file = (payload.get("crispasr") or {}).get("model")
    return Transcript(text, latency, latency_method, model_file=model_file)


def _check_crispasr_warnings(stderr: str) -> None:
    ignored_options = []
    for line in stderr.splitlines():
        if CRISPASR_WARNING not in line:
            continue
        if IGNORED_OPTION_MARKER in line:
            ignored_options.append(line.strip())
        else:
            print(f"[crispasr] {line.strip()}")
    if ignored_options:
        # crispasr exits 0 and silently drops options the backend lacks; results would be
        # labelled with settings that were never applied.
        raise TranscriberError("crispasr ignored requested options: " + "; ".join(ignored_options))


class CliTranscriber(Transcriber):
    def __init__(
        self,
        *,
        crisp_bin: Path,
        backend: str,
        model_args: list[str],
        language: str | None,
        options: RunOptions,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> None:
        self.crisp_bin = Path(crisp_bin)
        _check_executable(self.crisp_bin)
        self.backend = backend
        self.model_args = model_args
        self.language = language
        self.options = options
        self.env = {**os.environ, **(env or {})}
        self.timeout = timeout
        self._version: str | None = None

    def build_command(self, inputs: Sequence[Path], out_prefixes: Sequence[Path]) -> list[str]:
        command = [str(self.crisp_bin), "--backend", self.backend, *self.model_args]
        if self.language is not None:
            command += ["-l", self.language]
        command += ["-oj", "-np"]
        if self.options.threads:
            command += ["-t", str(self.options.threads)]
        if self.options.vad:
            command.append("--vad")
        if self.options.chunk_seconds is not None:
            command += ["--chunk-seconds", f"{self.options.chunk_seconds:g}"]
        command += list(self.options.extra_args)
        for input_path, output_prefix in zip(inputs, out_prefixes):
            command += ["-f", str(input_path), "-of", str(output_prefix)]
        return command

    def _run(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                command, capture_output=True, text=True, env=self.env, timeout=self.timeout, check=False
            )
        except OSError as exc:
            if exc.errno == errno.ENOEXEC:
                raise TranscriberError(
                    f"{self.crisp_bin} was built for a different OS/architecture (Exec format error); "
                    "rebuild it on this machine with ./build_binary.sh"
                ) from exc
            raise TranscriberError(f"could not execute {self.crisp_bin}: {exc}") from exc

    def transcribe_batch(self, batch: Sequence[Utterance]) -> list[Transcript]:
        if not batch:
            return []
        with tempfile.TemporaryDirectory(prefix="edge-speech-") as tmp:
            wav_paths, output_prefixes = _write_wavs(batch, Path(tmp))
            command = self.build_command(wav_paths, output_prefixes)
            start = time.perf_counter()
            try:
                process = self._run(command)
            except subprocess.TimeoutExpired:
                return [Transcript(None, 0.0, "process", f"timed out after {self.timeout}s") for _ in batch]
            wall_time = time.perf_counter() - start
            _check_crispasr_warnings(process.stderr or "")

            latency_method = "process" if len(batch) == 1 else "batch_apportioned"
            latencies = _apportioned_latencies(batch, wall_time)
            transcripts = []
            for output_prefix, latency in zip(output_prefixes, latencies):
                transcripts.append(_read_transcript(output_prefix, latency, latency_method, process))
            return transcripts

    def version(self) -> str:
        if self._version is None:
            process = self._run([str(self.crisp_bin), "--version"])
            self._version = _parse_version(f"{process.stdout}\n{process.stderr}")
        return self._version


def _import_session_class() -> tuple[Callable[..., Any], str | None]:
    try:
        import crispasr  # type: ignore[import-not-found]
    except ImportError as exc:
        raise TranscriberError(
            "the session runner needs the crispasr Python package: "
            "uv pip install -e CrispASR/python and set CRISPASR_LIB_PATH to libcrispasr"
        ) from exc
    return crispasr.Session, getattr(crispasr, "__version__", None)


class SessionTranscriber(Transcriber):
    def __init__(
        self,
        *,
        model_path: str,
        backend: str,
        language: str | None,
        options: RunOptions,
        lib_path: str | None = None,
        session_factory: Callable[..., Any] | None = None,
    ) -> None:
        if options.vad or options.chunk_seconds is not None or options.extra_args:
            raise TranscriberError(
                "the session runner doesn't support vad / chunk_seconds / extra_args; use runner: cli"
            )
        self._module_version: str | None = None
        if session_factory is None:
            session_factory, self._module_version = _import_session_class()
        self.language = language
        self.model_path = model_path

        session_kwargs: dict[str, Any] = {"backend": backend}
        if lib_path:
            session_kwargs["lib_path"] = lib_path
        if options.threads:
            session_kwargs["n_threads"] = options.threads
        try:
            self._session = session_factory(model_path, **session_kwargs)
        except Exception as exc:
            raise TranscriberError(f"could not open session for {model_path}: {exc}") from exc

    def _transcribe_one(self, utterance: Utterance) -> Transcript:
        start = time.perf_counter()
        try:
            segments = self._session.transcribe(utterance.audio, utterance.sampling_rate, language=self.language)
        except Exception as exc:
            return Transcript(None, time.perf_counter() - start, "session", str(exc))
        latency = time.perf_counter() - start
        try:
            text = _segments_text(list(segments))
        except MalformedOutput as exc:
            return Transcript(None, latency, "session", f"malformed session output: {exc}")
        return Transcript(text, latency, "session", model_file=self.model_path)

    def transcribe_batch(self, batch: Sequence[Utterance]) -> list[Transcript]:
        transcripts = []
        for utterance in batch:
            transcripts.append(self._transcribe_one(utterance))
        return transcripts

    def close(self) -> None:
        close = getattr(self._session, "close", None)
        if callable(close):
            close()

    def version(self) -> str:
        if not self._module_version:
            raise TranscriberError("the crispasr Python package does not report a version")
        return self._module_version
