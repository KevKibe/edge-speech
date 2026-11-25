"""Transcribe tests/data/jfk.wav and FLEURS samples with real models through both runners."""

from pathlib import Path

import pytest
import soundfile as sf
import yaml
from huggingface_hub import hf_hub_download

from conftest import AUDIO_FILE, AUDIO_TEXT, CRISP_BIN, LIB_PATH, MAX_WER, WHISPER_FILE, WHISPER_HF_MODEL, WHISPER_REPO
from edge_speech import analysis, evaluate
from edge_speech.config import expand_runs
from edge_speech.evaluate import EvalContext, make_transcriber, read_records
from edge_speech.metrics import utterance_scores
from edge_speech.runners import Utterance

MODEL_SOURCES = {
    "local-model-file": ("whisper", lambda: hf_hub_download(repo_id=WHISPER_REPO, filename=WHISPER_FILE)),
    "hf-model": ("whisper", lambda: WHISPER_HF_MODEL),
    "crispasr-auto-model": ("moonshine", lambda: "auto"),
}
RUNNERS = ["cli", "session"]


def _transcriber(registry, backend: str, model: str, runner: str = "cli", options: dict | None = None):
    run = {"backend": backend, "model": model, "languages": ["en_us"], "runner": runner, "options": options or {}}
    (job,) = expand_runs([run], registry).jobs
    return make_transcriber(job, EvalContext(crisp_bin=CRISP_BIN, lib_path=str(LIB_PATH)))


def _local_audio() -> Utterance:
    audio, sampling_rate = sf.read(AUDIO_FILE, dtype="float32")
    return Utterance(audio, sampling_rate)


def _check_transcript(transcript, label: str) -> None:
    assert transcript.error is None, transcript.error
    wer, _ = utterance_scores(AUDIO_TEXT, transcript.text)
    print(f"\n{label}: {transcript.text!r} (WER {wer:.3f}, {transcript.latency:.3f}s, {transcript.latency_method})")
    assert wer < MAX_WER, transcript.text
    assert transcript.latency > 0


@pytest.mark.parametrize("runner", RUNNERS)
@pytest.mark.parametrize("source", MODEL_SOURCES)
def test_transcribe_local_audio_file(registry, source: str, runner: str) -> None:
    backend, model = MODEL_SOURCES[source]
    with _transcriber(registry, backend, model(), runner) as transcriber:
        model_file = transcriber.warmup()
        assert transcriber.version()
        (transcript,) = transcriber.transcribe_batch([_local_audio()])
    assert Path(model_file).is_file()
    assert transcript.latency_method == ("session" if runner == "session" else "process")
    _check_transcript(transcript, f"{source}/{runner}")


def test_transcribe_batch_of_local_audio_files(registry) -> None:
    with _transcriber(registry, "whisper", WHISPER_HF_MODEL) as transcriber:
        transcripts = transcriber.transcribe_batch([_local_audio(), _local_audio()])
    assert [t.latency_method for t in transcripts] == ["batch_apportioned"] * 2
    for transcript in transcripts:
        _check_transcript(transcript, "batch")


def test_transcribe_local_audio_file_with_vad_and_chunking(registry) -> None:
    options = {"vad": True, "chunk_seconds": 5}
    with _transcriber(registry, "whisper", WHISPER_HF_MODEL, options=options) as transcriber:
        transcriber.warmup()
        (transcript,) = transcriber.transcribe_batch([_local_audio()])
    _check_transcript(transcript, "vad+chunking")


def _fleurs_sweep(path: Path) -> Path:
    sweep = {
        "defaults": {"dataset": "fleurs", "languages": ["en_us"], "backend": "whisper", "max_samples": 2},
        "runs": [
            {"runner": "cli", "batch_size": 2, "models": [{"hf": WHISPER_HF_MODEL.removeprefix("hf:"), "label": "cli"}]},
            {"runner": "session", "models": [{"hf": WHISPER_HF_MODEL.removeprefix("hf:"), "label": "session"}]},
        ],
    }
    path.write_text(yaml.safe_dump(sweep))
    return path


def test_transcribe_hf_dataset_samples_with_sweep_resume_and_analysis(tmp_path) -> None:
    results_dir = tmp_path / "results"
    argv = [
        "--sweep", str(_fleurs_sweep(tmp_path / "sweep.yaml")), "--output-dir", str(results_dir),
        "--crisp-bin", str(CRISP_BIN), "--lib-path", str(LIB_PATH),
    ]
    assert evaluate.main(argv) == 0

    job_dir = results_dir / "fleurs" / "en_us" / "whisper"
    for runner, latency_method in [("cli", "batch_apportioned"), ("session", "session")]:
        records = read_records(job_dir / f"{runner}.jsonl")
        assert len(records) == 2
        for record in records:
            print(f"\nFLEURS {runner} #{record['idx']}: {record['prediction']!r} (WER {record['wer']:.3f})")
            assert record["hf_dataset"] == "google/fleurs"
            assert record["latency_method"] == latency_method
            assert record["duration"] > 0 and record["latency"] > 0
        summary = (job_dir / f"{runner}.summary.json")
        assert summary.is_file()
        assert not (job_dir / f"{runner}.errors.jsonl").exists()

    assert evaluate.main(argv) == 0  # resume: nothing left to do
    assert len(read_records(job_dir / "cli.jsonl")) == 2

    assert analysis.main(["--results-dir", str(results_dir)]) == 0
    assert (results_dir / "summary.csv").is_file()
    assert any((results_dir / "plots").iterdir())
