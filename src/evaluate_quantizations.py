#!/usr/bin/env python3

from __future__ import annotations

import argparse
import errno
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import soundfile as sf
from datasets import load_dataset
from dotenv import load_dotenv
from jiwer import cer, wer

load_dotenv()

DEFAULT_CRISP_BIN = Path(os.getenv("CRISP_BIN", "./src/crispasr"))
DEFAULT_OUTPUT_DIR = Path("./src/results/jsonl")
REF_FIELDS = ("transcription", "transcript", "sentence", "text")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ASR models on a Hugging Face dataset split."
    )
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=None,
        help="Optional JSON config file describing one or more eval runs",
    )
    parser.add_argument("--lang", default="swh_Latn", help="ASR language tag for crispasr")
    parser.add_argument("--dataset", default="google/fleurs", help="Hugging Face dataset name")
    parser.add_argument("--dataset-lang", default="sw_ke", help="Dataset config/language")
    parser.add_argument(
        "--dataset-langs",
        nargs="+",
        default=None,
        help="Optional list of dataset languages/configs to run in one command",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="Backend passed to crispasr --backend (for example: omniasr)",
    )
    parser.add_argument(
        "--model-repo",
        default=None,
        help="Value passed directly to crispasr -m",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample limit per run")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for output JSONL files")
    parser.add_argument(
        "--crisp-bin",
        type=Path,
        default=DEFAULT_CRISP_BIN,
        help="Path to crispasr binary (can be outside this repo)",
    )
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face token (defaults to HF_TOKEN env var)",
    )
    parser.add_argument("--chunk-seconds", type=float, default=5, help="Optional chunk length for crispasr")
    return parser.parse_args()


def run_crispasr(
    audio_path: Path,
    model_arg: str,
    backend: str,
    lang: str,
    crisp_bin: Path,
    chunk_seconds: float | None,
) -> tuple[str, float]:
    start = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_prefix = str(Path(tmpdir) / "result")

        cmd = [
            str(crisp_bin),
            "--backend",
            backend,
            "-f",
            str(audio_path),
            "-m",
            model_arg,
            "-l",
            lang,
            "-oj",
            "-of",
            out_prefix,
            "-np",
        ]

        if chunk_seconds is not None:
            cmd.extend(["--chunk-seconds", str(chunk_seconds)])
        cmd.append("--vad")

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as exc:
            if exc.errno == errno.ENOEXEC:
                raise RuntimeError(
                    "Failed to execute crispasr binary due to Exec format error. "
                    "This usually means src/crispasr was built for a different OS/architecture. "
                    "Build/download a Linux binary for your notebook environment and place it at src/crispasr."
                ) from exc
            raise

        with open(f"{out_prefix}.json", "r", encoding="utf-8") as f:
            result = json.load(f)

    latency = time.perf_counter() - start
    text = ""
    if result.get("transcription"):
        text = " ".join(seg.get("text", "").strip() for seg in result["transcription"])

    return text, latency


def get_reference(sample: dict) -> str | None:
    for field in REF_FIELDS:
        value = sample.get(field)
        if value:
            return str(value)
    return None


def _build_job_args(base_args: argparse.Namespace, **overrides: object) -> argparse.Namespace:
    values = vars(base_args).copy()
    values.update(overrides)
    return argparse.Namespace(**values)


def _load_eval_config(path: Path) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("eval config must be a JSON object")
    return data


def _str_value(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _path_value(value: object | None, default_path: Path) -> Path:
    if value is None:
        return default_path
    return Path(str(value))


def _float_value(value: object | None, default_value: float | None) -> float | None:
    if value is None:
        return default_value
    return float(value)


def _int_value(value: object | None, default_value: int | None) -> int | None:
    if value is None:
        return default_value
    return int(value)


def _expand_lang_jobs(
    *,
    run_index: int,
    run: dict[str, object],
    lang_default: str | None,
) -> list[tuple[str, str]]:
    dataset_lang = run.get("dataset_lang")
    dataset_langs = run.get("dataset_langs")
    lang_map_raw = run.get("lang_map")
    lang_map: dict[str, str] = {}
    if lang_map_raw is not None:
        if not isinstance(lang_map_raw, dict):
            raise ValueError(f"runs[{run_index}].lang_map must be an object")
        lang_map = {str(k): str(v) for k, v in lang_map_raw.items()}

    jobs: list[tuple[str, str]] = []
    if dataset_lang is not None and dataset_langs is not None:
        raise ValueError(f"runs[{run_index}] cannot set both dataset_lang and dataset_langs")

    if dataset_lang is not None:
        if isinstance(dataset_lang, list):
            for item in dataset_lang:
                dlang = str(item)
                lang = lang_map.get(dlang, lang_default)
                if not lang:
                    raise ValueError(f"runs[{run_index}] missing lang or lang_map entry for dataset_lang={dlang}")
                jobs.append((dlang, lang))
            return jobs
        dlang = str(dataset_lang)
        lang = lang_map.get(dlang, lang_default)
        if not lang:
            raise ValueError(f"runs[{run_index}] missing lang or lang_map entry for dataset_lang={dlang}")
        jobs.append((dlang, lang))
        return jobs

    if dataset_langs is not None:
        if not isinstance(dataset_langs, list):
            raise ValueError(f"runs[{run_index}].dataset_langs must be a list")
        for item in dataset_langs:
            dlang = str(item)
            lang = lang_map.get(dlang, lang_default)
            if not lang:
                raise ValueError(f"runs[{run_index}] missing lang or lang_map entry for dataset_lang={dlang}")
            jobs.append((dlang, lang))
        return jobs

    raise ValueError(f"runs[{run_index}] must set dataset_lang or dataset_langs")


def build_eval_jobs(args: argparse.Namespace) -> list[argparse.Namespace]:
    if args.eval_config is None:
        if not args.backend:
            raise ValueError("--backend is required when --eval-config is not used")
        if not args.model_repo:
            raise ValueError("--model-repo is required when --eval-config is not used")
        dataset_langs = args.dataset_langs if args.dataset_langs else [args.dataset_lang]
        return [
            _build_job_args(args, dataset_lang=dataset_lang, lang=args.lang)
            for dataset_lang in dataset_langs
        ]

    cfg = _load_eval_config(args.eval_config)
    defaults_raw = cfg.get("defaults", {})
    if not isinstance(defaults_raw, dict):
        raise ValueError("defaults in eval config must be an object")
    defaults = dict(defaults_raw)

    runs_raw = cfg.get("runs")
    if not isinstance(runs_raw, list) or not runs_raw:
        raise ValueError("eval config must contain a non-empty runs list")

    jobs: list[argparse.Namespace] = []
    for i, run_raw in enumerate(runs_raw, start=1):
        if not isinstance(run_raw, dict):
            raise ValueError(f"runs[{i}] must be an object")
        run = dict(run_raw)

        backend = _str_value(run.get("backend", defaults.get("backend", args.backend)))
        model_repo = _str_value(run.get("model_repo", defaults.get("model_repo", args.model_repo)))
        dataset = _str_value(run.get("dataset", defaults.get("dataset", args.dataset)))
        hf_token = _str_value(run.get("hf_token", defaults.get("hf_token", args.hf_token)))
        crisp_bin = _path_value(run.get("crisp_bin", defaults.get("crisp_bin")), args.crisp_bin)
        output_dir = _path_value(run.get("output_dir", defaults.get("output_dir")), args.output_dir)
        chunk_seconds = _float_value(run.get("chunk_seconds", defaults.get("chunk_seconds")), args.chunk_seconds)
        max_samples = _int_value(run.get("max_samples", defaults.get("max_samples")), args.max_samples)
        lang_default = _str_value(run.get("lang", defaults.get("lang", args.lang)))

        if not backend:
            raise ValueError(f"runs[{i}] missing backend")
        if not model_repo:
            raise ValueError(f"runs[{i}] missing model_repo")
        if not dataset:
            raise ValueError(f"runs[{i}] missing dataset")

        lang_jobs = _expand_lang_jobs(run_index=i, run=run, lang_default=lang_default)
        for dataset_lang, lang in lang_jobs:
            jobs.append(
                _build_job_args(
                    args,
                    backend=backend,
                    model_repo=model_repo,
                    dataset=dataset,
                    dataset_lang=dataset_lang,
                    dataset_langs=None,
                    lang=lang,
                    hf_token=hf_token,
                    crisp_bin=crisp_bin,
                    output_dir=output_dir,
                    chunk_seconds=chunk_seconds,
                    max_samples=max_samples,
                )
            )

    return jobs


def evaluate_quantization(args: argparse.Namespace, dataset_lang: str) -> dict[str, object] | None:
    model_arg = args.model_repo
    quantization = Path(model_arg).stem

    dataset = load_dataset(
        args.dataset,
        dataset_lang,
        split="test",
        streaming=True,
        token=args.hf_token,
    )

    model_name = Path(model_arg).stem
    dataset_name = args.dataset.split("/")[-1]
    output_jsonl = args.output_dir / f"{model_name}_{dataset_name}_{dataset_lang}.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    wer_sum = 0.0
    cer_sum = 0.0
    rtfx_sum = 0.0

    print(f"[run:{quantization}:{dataset_lang}] writing -> {output_jsonl}")
    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for idx, sample in enumerate(dataset):
            if args.max_samples is not None and n >= args.max_samples:
                break

            audio = sample["audio"]
            ref = get_reference(sample)
            if ref is None:
                print(f"[skip:{quantization}:{dataset_lang}] sample {idx}: no reference text")
                continue

            with tempfile.NamedTemporaryFile(suffix=".wav") as wav_file:
                sf.write(wav_file.name, audio["array"], audio["sampling_rate"])
                pred, latency = run_crispasr(
                    audio_path=Path(wav_file.name),
                    model_arg=model_arg,
                    backend=args.backend,
                    lang=args.lang,
                    crisp_bin=args.crisp_bin,
                    chunk_seconds=args.chunk_seconds,
                )

            duration = len(audio["array"]) / audio["sampling_rate"]
            rtfx = latency / duration if duration > 0 else 0.0

            ref = ref.lower().strip()
            pred = pred.lower().strip()

            wer_score = wer(ref, pred)
            cer_score = cer(ref, pred)

            wer_sum += wer_score
            cer_sum += cer_score
            rtfx_sum += rtfx
            n += 1

            record = {
                "idx": idx,
                "quantization": quantization,
                "reference": ref,
                "prediction": pred,
                "wer": round(wer_score, 6),
                "cer": round(cer_score, 6),
                "rtfx": round(rtfx, 6),
                "latency": round(latency, 6),
                "duration": round(duration, 6),
            }
            f_out.write(json.dumps(record, ensure_ascii=True) + "\n")

            if n % 20 == 0:
                print(f"[{quantization}:{dataset_lang}:{n}] WER={wer_score:.3f} CER={cer_score:.3f} RTFx={rtfx:.3f}")

    if n == 0:
        print(f"[warn:{quantization}:{dataset_lang}] no evaluated samples")
        return None

    summary = {
        "quantization": quantization,
        "dataset_lang": dataset_lang,
        "model": model_arg,
        "samples": n,
        "avg_wer": wer_sum / n,
        "avg_cer": cer_sum / n,
        "avg_rtfx": rtfx_sum / n,
        "output": str(output_jsonl),
    }
    print(
        f"[done:{quantization}:{dataset_lang}] samples={n} "
        f"WER={summary['avg_wer']:.3f} CER={summary['avg_cer']:.3f} RTFx={summary['avg_rtfx']:.3f}"
    )
    return summary


def main() -> None:
    args = parse_args()

    jobs = build_eval_jobs(args)

    summaries: list[dict[str, object]] = []
    for job_args in jobs:
        if not job_args.crisp_bin.exists():
            raise FileNotFoundError(f"crispasr binary not found: {job_args.crisp_bin}")
        summary = evaluate_quantization(job_args, job_args.dataset_lang)
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        print("No quantizations were evaluated.")
        return

    print("\n=== Summary ===")
    for item in summaries:
        print(
            f"{item['dataset_lang']:>8} | {item['quantization']:>8} | samples={item['samples']:>4} | "
            f"WER={item['avg_wer']:.3f} | CER={item['avg_cer']:.3f} | RTFx={item['avg_rtfx']:.3f}"
        )


if __name__ == "__main__":
    main()
