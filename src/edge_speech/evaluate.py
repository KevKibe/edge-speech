"""Run evaluation jobs: dataset × language × backend × model.

    edge-speech-eval --sweep configs/sweeps/fleurs_parakeet.yaml
    edge-speech-eval --backend parakeet --model auto:q4_k --languages en_us fr_fr --max-samples 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from dotenv import load_dotenv

from .config import (
    DEFAULT_CRISP_BIN,
    DEFAULT_REGISTRY_DIR,
    ConfigError,
    Expansion,
    Job,
    ModelSpec,
    detect_quant,
    expand_runs,
    load_registry,
    load_sweep,
)
from .data import Sample, iter_samples
from .metrics import corpus_scores, utterance_scores
from .runners import (
    CliTranscriber,
    SessionTranscriber,
    Transcriber,
    TranscriberError,
    Transcript,
    Utterance,
    model_cli_args,
    resolve_model_path,
)

SUMMARY_HEADER = f"{'status':<7} {'job':<60} {'n':>5} {'WER':>7} {'CER':>7} {'RTFx':>7} {'err':>4}"


@dataclass
class EvalContext:
    crisp_bin: Path = DEFAULT_CRISP_BIN
    lib_path: str | None = None
    hf_token: str | None = None
    overwrite: bool = False
    timeout: float | None = None


@dataclass
class JobResult:
    job: Job
    status: str  # done | failed
    new_records: int = 0
    errors: int = 0
    summary: dict[str, Any] | None = None
    message: str = ""


@dataclass
class BatchOutcome:
    new_records: int
    errors: int
    first_error: str


def _cli_transcriber(job: Job, ctx: EvalContext, local_path: str | None) -> CliTranscriber:
    env = {"HF_TOKEN": ctx.hf_token} if ctx.hf_token else None
    return CliTranscriber(
        crisp_bin=ctx.crisp_bin,
        backend=job.backend.name,
        model_args=model_cli_args(job.model, local_path),
        language=job.asr_language,
        options=job.options,
        env=env,
        timeout=ctx.timeout,
    )


def _session_model_path(job: Job, ctx: EvalContext, local_path: str | None) -> str:
    if local_path is not None:
        return local_path
    # The session API takes a file path; crispasr's CLI resolves (and downloads) auto models.
    with _cli_transcriber(job, ctx, None) as cli:
        return cli.warmup()


def make_transcriber(job: Job, ctx: EvalContext) -> Transcriber:
    local_path = resolve_model_path(job.model, token=ctx.hf_token)
    if job.runner == "session":
        return SessionTranscriber(
            model_path=_session_model_path(job, ctx, local_path),
            backend=job.backend.name,
            language=job.asr_language,
            options=job.options,
            lib_path=ctx.lib_path,
        )
    return _cli_transcriber(job, ctx, local_path)


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    while lines and not lines[-1]:
        lines.pop()

    records = []
    for number, line in enumerate(lines, start=1):
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if number == len(lines):
                continue  # a run killed mid-write leaves a partial last line
            raise ValueError(f"{path}:{number}: corrupt result record: {exc}") from exc
    return records


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _latency_methods(records: Sequence[dict[str, Any]]) -> list[str]:
    methods = set()
    for record in records:
        if record.get("latency_method"):
            methods.add(str(record["latency_method"]))
    return sorted(methods)


def summarize_job(job: Job, records: Sequence[dict[str, Any]], errors: int) -> dict[str, Any]:
    summary = job.metadata()
    for key, value in corpus_scores(records).as_dict().items():
        summary[key] = _json_safe(value)
    summary["primary_metric"] = "wer" if job.language.space_delimited else "cer"
    summary["latency_methods"] = _latency_methods(records)
    summary["errors"] = errors
    summary["output"] = str(job.output_path)
    return summary


def _write_summary(job: Job, records: Sequence[dict[str, Any]], errors: int) -> dict[str, Any]:
    summary = summarize_job(job, records, errors)
    with open(job.summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def _settings_mismatch(existing: Sequence[dict[str, Any]], expected: dict[str, Any]) -> str | None:
    for record in existing:
        for key, value in expected.items():
            if record.get(key) != value:
                return f"{key}: existing {record.get(key)!r}, now {value!r}"
    return None


def _mismatch_failure(job: Job, mismatch: str) -> JobResult:
    message = (
        f"existing results in {job.output_path} were produced with different settings ({mismatch}); "
        "pass --overwrite or use another --output-dir"
    )
    return JobResult(job, "failed", message=message)


def _job_identity(job: Job) -> dict[str, Any]:
    # quant is excluded: for `auto` models it is only known once crispasr reports the file it loaded.
    identity = job.metadata()
    del identity["quant"]
    return identity


def _clear_outputs(job: Job) -> None:
    for path in (job.output_path, job.errors_path, job.summary_path):
        path.unlink(missing_ok=True)


def _done_indices(records: Sequence[dict[str, Any]]) -> frozenset[int]:
    indices = set()
    for record in records:
        if "idx" in record:
            indices.add(int(record["idx"]))
    return frozenset(indices)


def _batched(items: Iterable[Sample], size: int) -> Iterator[list[Sample]]:
    batch: list[Sample] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _error_record(sample: Sample, transcript: Transcript) -> dict[str, Any]:
    return {"idx": sample.idx, "id": sample.sample_id, "error": transcript.error, "time": time.time()}


def _result_record(metadata: dict[str, Any], sample: Sample, transcript: Transcript) -> dict[str, Any]:
    wer, cer = utterance_scores(sample.reference, transcript.text or "")
    return {
        **metadata,
        "idx": sample.idx,
        "id": sample.sample_id,
        "reference": sample.reference,
        "prediction": transcript.text,
        "duration": round(sample.duration, 4),
        "latency": round(transcript.latency, 6),
        "latency_method": transcript.latency_method,
        "wer": round(wer, 6),
        "cer": round(cer, 6),
    }


def _transcribe_batch(
    job: Job,
    metadata: dict[str, Any],
    transcriber: Transcriber,
    batch: list[Sample],
) -> BatchOutcome:
    utterances = [Utterance(sample.audio, sample.sampling_rate) for sample in batch]
    transcripts = transcriber.transcribe_batch(utterances)
    outcome = BatchOutcome(new_records=0, errors=0, first_error="unknown error")
    first_error_found = False
    for sample, transcript in zip(batch, transcripts):
        if transcript.error and not first_error_found:
            outcome.first_error = transcript.error
            first_error_found = True
        if transcript.text is None:
            outcome.errors += 1
            _append_jsonl(job.errors_path, _error_record(sample, transcript))
            continue
        _append_jsonl(job.output_path, _result_record(metadata, sample, transcript))
        outcome.new_records += 1
    return outcome


def _resolved_metadata(job: Job, transcriber: Transcriber) -> dict[str, Any]:
    model_file = transcriber.warmup()
    metadata = job.metadata()
    metadata["crispasr_version"] = transcriber.version()
    metadata["model_file"] = model_file
    if metadata["quant"] is None:
        metadata["quant"] = detect_quant(Path(model_file).stem)
    return metadata


def _finished_result(job: Job, new_records: int, errors: int) -> JobResult:
    summary = _write_summary(job, read_records(job.output_path), errors)
    if errors:
        message = f"{errors} sample(s) failed; see {job.errors_path}"
        return JobResult(job, "failed", new_records, errors, summary, message=message)
    return JobResult(job, "done", new_records, errors, summary)


def _run_with_transcriber(
    job: Job,
    ctx: EvalContext,
    transcriber: Transcriber,
    existing: Sequence[dict[str, Any]],
    done: frozenset[int],
) -> JobResult:
    try:
        metadata = _resolved_metadata(job, transcriber)
    except TranscriberError as exc:
        return JobResult(job, "failed", message=str(exc))
    mismatch = _settings_mismatch(existing, metadata)
    if mismatch:
        return _mismatch_failure(job, mismatch)

    samples = iter_samples(
        job.dataset, job.dataset_lang, token=ctx.hf_token, max_samples=job.max_samples, skip_idx=done
    )
    new_records = 0
    errors = 0
    started = time.perf_counter()
    for batch in _batched(samples, job.batch_size):
        try:
            outcome = _transcribe_batch(job, metadata, transcriber, batch)
        except TranscriberError as exc:
            return JobResult(job, "failed", new_records, errors, message=str(exc))
        new_records += outcome.new_records
        errors += outcome.errors
        if new_records == 0 and outcome.errors == len(batch):
            # Every sample of the batch failed and nothing has worked yet: the setup is
            # broken, don't grind through the remaining samples.
            return JobResult(job, "failed", 0, errors, message=f"every sample in a batch failed: {outcome.first_error}")
        elapsed = time.perf_counter() - started
        print(f"[{job.describe()}] {len(done) + new_records} done, {errors} errors ({elapsed:.0f}s)")

    return _finished_result(job, new_records, errors)


def run_job(job: Job, ctx: EvalContext) -> JobResult:
    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    if ctx.overwrite:
        _clear_outputs(job)

    existing = read_records(job.output_path)
    mismatch = _settings_mismatch(existing, _job_identity(job))
    if mismatch:
        return _mismatch_failure(job, mismatch)
    done = _done_indices(existing)
    if done:
        print(f"[resume] {job.describe()}: {len(done)} samples already done")
    if job.max_samples is not None and len(done) >= job.max_samples:
        # Nothing left to do: skip model loading and dataset streaming entirely.
        summary = _write_summary(job, existing, 0)
        return JobResult(job, "done", 0, 0, summary, message="already complete")

    try:
        transcriber = make_transcriber(job, ctx)
    except TranscriberError as exc:
        return JobResult(job, "failed", message=str(exc))
    with transcriber:
        return _run_with_transcriber(job, ctx, transcriber, existing, done)


def run_jobs(jobs: Sequence[Job], ctx: EvalContext) -> list[JobResult]:
    results = []
    for number, job in enumerate(jobs, start=1):
        print(f"\n=== [{number}/{len(jobs)}] {job.describe()} -> {job.output_path}")
        result = run_job(job, ctx)
        if result.status == "failed":
            print(f"[failed] {job.describe()}: {result.message}")
        results.append(result)
    return results


def _format_metric(summary: dict[str, Any], key: str) -> str:
    value = summary.get(key)
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return "-"


def _summary_line(result: JobResult) -> str:
    summary = result.summary or {}
    return (
        f"{result.status:<7} {result.job.describe()[:60]:<60} {summary.get('samples', 0):>5} "
        f"{_format_metric(summary, 'wer'):>7} {_format_metric(summary, 'cer'):>7} "
        f"{_format_metric(summary, 'rtfx'):>7} {result.errors:>4}"
    )


def format_summary(results: Sequence[JobResult]) -> str:
    lines = ["", "=== Summary ===", SUMMARY_HEADER]
    for result in results:
        lines.append(_summary_line(result))
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("what to run (a sweep file, or one ad-hoc run)")
    src.add_argument("--sweep", type=Path, help="YAML sweep file (see configs/sweeps/)")
    src.add_argument("--backend", help="CrispASR backend, e.g. parakeet, omniasr, whisper")
    src.add_argument("--model", action="append", help="auto[:quant] | hf:owner/repo:file | path (repeatable)")
    src.add_argument("--dataset", default="fleurs", help="dataset key in configs/datasets.yaml")
    src.add_argument("--languages", nargs="+", help="dataset language keys, or 'all'")

    ov = p.add_argument_group("overrides (win over the sweep file)")
    ov.add_argument("--max-samples", type=int, help="samples per job")
    ov.add_argument("--runner", choices=["cli", "session"])
    ov.add_argument("--batch-size", type=int)
    ov.add_argument("--output-dir", type=Path)

    env = p.add_argument_group("environment")
    env.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR)
    env.add_argument("--crisp-bin", type=Path, default=Path(os.getenv("CRISP_BIN", str(DEFAULT_CRISP_BIN))))
    env.add_argument("--lib-path", default=os.getenv("CRISPASR_LIB_PATH"), help="libcrispasr for --runner session")
    env.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    env.add_argument("--timeout", type=float, help="seconds per crispasr process")

    p.add_argument("--overwrite", action="store_true", help="discard existing results instead of resuming")
    p.add_argument("--dry-run", action="store_true", help="list jobs and skipped languages, then exit")
    return p.parse_args(argv)


def _ad_hoc_run(args: argparse.Namespace) -> dict[str, Any]:
    if not (args.backend and args.model and args.languages):
        raise ConfigError("without --sweep, --backend, --model and --languages are all required")
    for spec in args.model:
        ModelSpec.parse(spec)  # fail early on a bad spec
    languages: Any = "all" if args.languages == ["all"] else args.languages
    return {"backend": args.backend, "dataset": args.dataset, "languages": languages, "models": args.model}


def build_jobs(args: argparse.Namespace) -> Expansion:
    registry = load_registry(args.registry_dir)
    overrides = {
        "max_samples": args.max_samples,
        "runner": args.runner,
        "batch_size": args.batch_size,
        "output_dir": args.output_dir,
    }
    if not args.sweep:
        return expand_runs([_ad_hoc_run(args)], registry, overrides=overrides)
    if args.backend or args.model or args.languages:
        raise ConfigError("use either --sweep or --backend/--model/--languages, not both")
    return load_sweep(args.sweep, registry, overrides)


def _print_dry_run(expansion: Expansion) -> None:
    for job in expansion.jobs:
        print(f"{job.describe()} -> {job.output_path}")
    print(f"{len(expansion.jobs)} job(s), {len(expansion.skipped)} skipped")


def _context_from_args(args: argparse.Namespace) -> EvalContext:
    return EvalContext(
        crisp_bin=args.crisp_bin,
        lib_path=args.lib_path,
        hf_token=args.hf_token,
        overwrite=args.overwrite,
        timeout=args.timeout,
    )


def main(argv: Sequence[str] | None = None) -> int:
    if not os.environ.get("EDGE_SPEECH_NO_DOTENV"):
        load_dotenv()

    args = parse_args(argv)
    try:
        expansion = build_jobs(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    for message in expansion.skipped:
        print(f"[skip] {message}")
    if not expansion.jobs:
        print("No jobs to run.")
        return 1 if expansion.skipped else 0
    if args.dry_run:
        _print_dry_run(expansion)
        return 0

    results = run_jobs(expansion.jobs, _context_from_args(args))
    print(format_summary(results))
    return 0 if all(result.status == "done" for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
