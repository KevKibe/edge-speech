from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
JSONL_DIR = RESULTS_DIR / "jsonl"
SUMMARY_CSV = RESULTS_DIR / "quantization_summary.csv"
PLOTS_DIR = RESULTS_DIR / "plots"

QUANT_ORDER = ["fp32_gguf", "q8_0", "q6_k", "q5_k", "q4_k", "q3_k", "q2_k"]
QUANT_COLORS = {
    "fp32_gguf": "#1b9e77",
    "q8_0": "#4c78a8",
    "q6_k": "#f58518",
    "q5_k": "#e45756",
    "q4_k": "#72b7b2",
    "q3_k": "#b279a2",
    "q2_k": "#9d755d",
}
LANG_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]


def parse_filename(path: Path) -> tuple[str, str, str]:
    name = path.name

    quantized = re.match(r".+?-v2-(q\d+(?:_[k0])?)_([a-z0-9]+)_(.+)\.jsonl$", name)
    if quantized:
        return quantized.group(1), quantized.group(2), quantized.group(3)

    fp32_gguf = re.match(r".+?-v2_([a-z0-9]+)_(.+)\.jsonl$", name)
    if fp32_gguf:
        return "fp32_gguf", fp32_gguf.group(1), fp32_gguf.group(2)

    return "unknown", "unknown", "unknown"


def load_jsonl(path: Path) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def aggregate_records(records: list[dict[str, float]]) -> tuple[float, float, float, int]:
    if not records:
        return np.nan, np.nan, np.nan, 0

    wer_values = [float(r["wer"]) for r in records]
    cer_values = [float(r["cer"]) for r in records]

    # Stored value is latency / duration (RTF). Convert to RTFx for speed readability.
    rtfx_values = [1.0 / float(r["rtfx"]) for r in records if float(r["rtfx"]) > 0]

    avg_wer = float(np.mean(wer_values))
    avg_cer = float(np.mean(cer_values))
    avg_rtfx = float(np.mean(rtfx_values)) if rtfx_values else np.nan
    return avg_wer, avg_cer, avg_rtfx, len(records)


def load_summary_rows() -> list[dict[str, object]]:
    patterns = [str(JSONL_DIR / "*.jsonl")]

    # Backward compatibility with older outputs in src/.jsonl
    legacy_pattern = str(ROOT / "*.jsonl")
    if not any(Path(p).is_file() for p in glob.glob(patterns[0])):
        patterns.append(legacy_pattern)

    files: list[Path] = []
    for pattern in patterns:
        files.extend(Path(p) for p in glob.glob(pattern))

    files = sorted({p.resolve() for p in files})
    rows: list[dict[str, object]] = []

    for path in files:
        quant, dataset, dataset_lang = parse_filename(path)
        records = load_jsonl(path)
        avg_wer, avg_cer, avg_rtfx, n = aggregate_records(records)
        rows.append(
            {
                "quant": quant,
                "dataset": dataset,
                "dataset_lang": dataset_lang,
                "samples": n,
                "wer": avg_wer,
                "cer": avg_cer,
                "rtfx": avg_rtfx,
                "source": str(path),
            }
        )

    order = {quant: idx for idx, quant in enumerate(QUANT_ORDER)}
    rows.sort(key=lambda r: (r["dataset"], r["dataset_lang"], order.get(str(r["quant"]), 99)))
    return rows


def write_summary_csv(rows: list[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = ["quant", "dataset", "dataset_lang", "samples", "wer", "cer", "rtfx"]

    with open(output_csv, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            values = []
            for h in headers:
                value = row[h]
                if h in {"wer", "cer"} and isinstance(value, (int, float)):
                    values.append(f"{value:.2f}")
                elif h == "rtfx" and isinstance(value, (int, float)):
                    values.append(f"{value:.3f}")
                else:
                    values.append(str(value))
            escaped = [v.replace('"', '""') for v in values]
            f.write(",".join(f'"{v}"' for v in escaped) + "\n")


def _sorted_lang_rows(rows: list[dict[str, object]], dataset_lang: str) -> list[dict[str, object]]:
    order = {quant: idx for idx, quant in enumerate(QUANT_ORDER)}
    lang_rows = [r for r in rows if str(r["dataset_lang"]) == dataset_lang]
    lang_rows.sort(key=lambda r: order.get(str(r["quant"]), 99))
    return lang_rows


def render_language_metric_bars(rows: list[dict[str, object]], dataset_lang: str, output_path: Path) -> None:
    lang_rows = _sorted_lang_rows(rows, dataset_lang)
    if not lang_rows:
        return

    quants = [str(r["quant"]) for r in lang_rows]
    wer_vals = [float(r["wer"]) for r in lang_rows]
    cer_vals = [float(r["cer"]) for r in lang_rows]

    x = np.arange(len(quants))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    ax2 = ax.twinx()

    wer_bars = ax.bar(
        x - width / 2,
        wer_vals,
        width=width,
        label="WER",
        color="#4c78a8",
        edgecolor="#1f3552",
        linewidth=0.8,
        hatch="///",
    )
    cer_bars = ax2.bar(
        x + width / 2,
        cer_vals,
        width=width,
        label="CER",
        color="#f58518",
        edgecolor="#8c4d0e",
        linewidth=0.8,
        hatch="\\\\",
    )

    wer_arr = np.array(wer_vals, dtype=float)
    cer_arr = np.array(cer_vals, dtype=float)

    wer_min = float(np.min(wer_arr))
    wer_max = float(np.max(wer_arr))
    wer_span = wer_max - wer_min
    wer_pad = max(wer_span * 0.15, 0.005)
    ax.set_ylim(max(0.0, wer_min - wer_pad), wer_max + wer_pad)

    cer_min = float(np.min(cer_arr))
    cer_max = float(np.max(cer_arr))
    cer_span = cer_max - cer_min
    cer_pad = max(cer_span * 0.15, 0.003)
    ax2.set_ylim(max(0.0, cer_min - cer_pad), cer_max + cer_pad)

    ax.set_xticks(x)
    ax.set_xticklabels(quants, rotation=25, ha="right")
    ax.set_ylabel("WER ")
    ax2.set_ylabel("CER")
    ax.set_title(f"{dataset_lang}: WER and CER by Quantization")
    ax.minorticks_on()
    ax.grid(which="major", axis="both", linestyle="--", linewidth=0.8, alpha=0.45)
    ax.grid(which="minor", axis="y", linestyle=":", linewidth=0.6, alpha=0.35)
    ax.legend(handles=[wer_bars, cer_bars], loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close(fig)


def render_language_rtfx_line(rows: list[dict[str, object]], dataset_lang: str, output_path: Path) -> None:
    lang_rows = _sorted_lang_rows(rows, dataset_lang)
    if not lang_rows:
        return

    quants = [str(r["quant"]) for r in lang_rows]
    rtfx_vals = [float(r["rtfx"]) for r in lang_rows]

    x = np.arange(len(quants))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, rtfx_vals, marker="o", linewidth=2.2, color="#1b9e77")

    ax.set_xticks(x)
    ax.set_xticklabels(quants, rotation=25, ha="right")
    ax.set_ylabel("RTFx (higher is faster)")
    ax.set_title(f"{dataset_lang}: RTFx by Quantization")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    rows = load_summary_rows()
    if not rows:
        raise FileNotFoundError(
            f"No JSONL results found in {JSONL_DIR}. Run evaluate_quantizations.py first."
        )

    write_summary_csv(rows, SUMMARY_CSV)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    langs = sorted({str(r["dataset_lang"]) for r in rows})
    generated_plots: list[Path] = []
    for lang in langs:
        bars_path = PLOTS_DIR / f"{lang}_wer_cer_bars.png"
        rtfx_path = PLOTS_DIR / f"{lang}_rtfx_line.png"

        render_language_metric_bars(rows, lang, bars_path)
        render_language_rtfx_line(rows, lang, rtfx_path)

        generated_plots.extend([bars_path, rtfx_path])

    print(f"Saved summary CSV: {SUMMARY_CSV}")
    for plot_path in generated_plots:
        print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
