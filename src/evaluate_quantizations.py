#!/usr/bin/env python3

from __future__ import annotations

import argparse
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

DEFAULT_MODEL_DIR = Path("./models/omniASR-CTC-300M-v2-GGUF")
DEFAULT_MODEL_REPO = "https://huggingface.co/KevinKibe/omniASR-CTC-300M-v2-GGUF"
DEFAULT_CRISP_BIN = Path("./src/crispasr")
DEFAULT_OUTPUT_DIR = Path("./src/results/jsonl")
DEFAULT_QUANTIZATIONS = (
    "fp32_gguf",
    "q8_0",
    "q6_k",
    "q5_k",
    "q4_k",
    "q3_k",
    "q2_k",
)
REF_FIELDS = ("transcription", "transcript", "sentence", "text")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate omniASR GGUF quantizations on a Hugging Face dataset split."
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
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Directory with .gguf models")
    parser.add_argument(
        "--model-repo",
        default=DEFAULT_MODEL_REPO,
        help="Repository to clone into --model-dir when it is missing",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample limit per quantization")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for output JSONL files")
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face token (defaults to HF_TOKEN env var)",
    )
    parser.add_argument("--chunk-seconds", type=float, default=5, help="Optional chunk length for crispasr")
    return parser.parse_args()


def quant_to_model_path(model_dir: Path, quantization: str) -> Path:
    base = "omniASR-CTC-300M-v2"
    if quantization == "fp32_gguf":
        model_name = f"{base}.gguf"
    else:
        model_name = f"{base}-{quantization}.gguf"
    return model_dir / model_name


def ensure_model_dir(model_dir: Path, model_repo: str) -> None:
    if model_dir.exists() and any(model_dir.iterdir()):
        return

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[setup] model directory not found, cloning {model_repo} -> {model_dir}")

    subprocess.run(["git", "clone", model_repo, str(model_dir)], check=True)

    lfs_check = subprocess.run(["git", "lfs", "version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if lfs_check.returncode == 0:
        subprocess.run(["git", "-C", str(model_dir), "lfs", "pull"], check=False)


def run_crispasr(
    audio_path: Path,
    model_path: Path,
    lang: str,
    chunk_seconds: float | None,
) -> tuple[str, float]:
    start = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_prefix = str(Path(tmpdir) / "result")

        cmd = [
            str(DEFAULT_CRISP_BIN),
            "--backend",
            "omniasr",
            "-f",
            str(audio_path),
            "-m",
            str(model_path),
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

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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


def evaluate_quantization(args: argparse.Namespace, quantization: str, dataset_lang: str) -> dict[str, object] | None:
    model_path = quant_to_model_path(args.model_dir, quantization)
    if not model_path.exists():
        print(f"[skip:{quantization}] missing model: {model_path}")
        return None

    dataset = load_dataset(
        args.dataset,
        dataset_lang,
        split="test",
        streaming=True,
        token=args.hf_token,
    )

    model_name = model_path.stem
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
                    model_path=model_path,
                    lang=args.lang,
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
        "model": model_name,
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

    ensure_model_dir(args.model_dir, args.model_repo)

    if not DEFAULT_CRISP_BIN.exists():
        raise FileNotFoundError(f"crispasr binary not found: {DEFAULT_CRISP_BIN}")

    dataset_langs = args.dataset_langs if args.dataset_langs else [args.dataset_lang]

    summaries: list[dict[str, object]] = []
    for dataset_lang in dataset_langs:
        for quantization in DEFAULT_QUANTIZATIONS:
            summary = evaluate_quantization(args, quantization, dataset_lang)
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
