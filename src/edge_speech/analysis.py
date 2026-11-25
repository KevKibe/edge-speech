"""Aggregate result JSONL files into a summary CSV and per-language plots.

    edge-speech-analyze                     # reads results/, writes results/summary.csv + results/plots/
    edge-speech-analyze --results-dir path --no-plots

Metrics are recomputed from the raw reference/prediction stored in each record,
so changing the normalizer re-scores old runs too.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .config import DEFAULT_OUTPUT_DIR, DEFAULT_REGISTRY_DIR, ConfigError, LanguageSpec, load_registry
from .evaluate import read_records
from .metrics import corpus_scores

matplotlib.use("Agg")

QUANT_ORDER = ["f32", "bf16", "f16", "q8_0", "q6_k", "q5_k", "q5_1", "q5_0", "q4_k", "q4_1", "q4_0", "q3_k", "q2_k", "tq2_0"]
GROUP_KEYS = ("dataset", "dataset_lang", "backend", "model")
CSV_FIELDS = [
    "dataset", "dataset_lang", "language", "language_name", "backend", "model", "quant", "runner",
    "samples", "wer", "cer", "primary_metric", "rtf", "rtfx", "total_audio_s", "latency_methods",
    "crispasr_version", "source",
]


@dataclass
class SummaryRow:
    dataset: str
    dataset_lang: str
    backend: str
    model: str
    language: str | None
    language_name: str
    quant: str | None
    runner: str | None
    samples: int
    wer: float
    cer: float
    primary_metric: str
    rtf: float
    rtfx: float
    total_audio_s: float
    latency_methods: str
    crispasr_version: str
    source: str


@dataclass
class RecordGroup:
    records: list[dict[str, Any]] = field(default_factory=list)
    sources: set[str] = field(default_factory=set)


def result_files(results_dir: Path) -> list[Path]:
    files = []
    for path in Path(results_dir).rglob("*.jsonl"):
        if not path.name.endswith(".errors.jsonl"):
            files.append(path)
    return sorted(files)


def _quant_rank(quant: str | None) -> int:
    if quant in QUANT_ORDER:
        return QUANT_ORDER.index(quant)
    return len(QUANT_ORDER)


def sort_key(row: SummaryRow) -> tuple[Any, ...]:
    return (row.dataset, row.dataset_lang, row.backend, _quant_rank(row.quant), row.model)




def _has_group_keys(record: dict[str, Any]) -> bool:
    for key in GROUP_KEYS:
        if record.get(key) is None:
            return False
    return True


def _group_records(results_dir: Path) -> tuple[dict[tuple[str, ...], RecordGroup], list[str]]:
    groups: dict[tuple[str, ...], RecordGroup] = defaultdict(RecordGroup)
    warnings: list[str] = []
    for path in result_files(results_dir):
        records = read_records(path)
        usable = [record for record in records if _has_group_keys(record)]
        ignored = len(records) - len(usable)
        if ignored:
            warnings.append(f"{path}: {ignored} record(s) without metadata ignored")
        for record in usable:
            key = tuple(str(record[name]) for name in GROUP_KEYS)
            groups[key].records.append(record)
            groups[key].sources.add(str(path))
    return groups, warnings


def _join_unique(records: list[dict[str, Any]], key: str) -> str:
    values = set()
    for record in records:
        values.add(str(record.get(key)))
    return ";".join(sorted(values))


def _summary_row(key: tuple[str, ...], group: RecordGroup, languages: dict[str, LanguageSpec]) -> SummaryRow:
    dataset, dataset_lang, backend, model = key
    first = group.records[0]
    try:
        scores = corpus_scores(group.records)
    except ValueError as exc:
        raise ValueError(f"{';'.join(sorted(group.sources))}: {exc}") from exc
    language_code = first.get("language")
    if language_code not in languages:
        raise ConfigError(f"{sorted(group.sources)[0]}: language {language_code!r} is not in languages.yaml")
    language = languages[language_code]
    return SummaryRow(
        dataset=dataset,
        dataset_lang=dataset_lang,
        backend=backend,
        model=model,
        language=language_code,
        language_name=language.name,
        quant=first.get("quant"),
        runner=first.get("runner"),
        samples=scores.samples,
        wer=scores.wer,
        cer=scores.cer,
        primary_metric="wer" if language.space_delimited else "cer",
        rtf=scores.rtf,
        rtfx=scores.rtfx,
        total_audio_s=scores.total_audio_s,
        latency_methods=_join_unique(group.records, "latency_method"),
        crispasr_version=_join_unique(group.records, "crispasr_version"),
        source=";".join(sorted(group.sources)),
    )


def collect_rows(results_dir: Path, registry_dir: Path = DEFAULT_REGISTRY_DIR) -> tuple[list[SummaryRow], list[str]]:
    languages = load_registry(registry_dir).languages
    groups, warnings = _group_records(results_dir)
    rows = []
    for key, group in groups.items():
        rows.append(_summary_row(key, group, languages))
    rows.sort(key=sort_key)
    return rows, warnings


def _format_cell(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.{digits}f}"
    return "" if value is None else str(value)


def write_csv(rows: Sequence[SummaryRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            values = asdict(row)
            writer.writerow({name: _format_cell(values[name]) for name in CSV_FIELDS})


def _rows_by_language(rows: Sequence[SummaryRow]) -> dict[tuple[str, str], list[SummaryRow]]:
    by_language: dict[tuple[str, str], list[SummaryRow]] = defaultdict(list)
    for row in rows:
        by_language[(row.dataset, row.dataset_lang)].append(row)
    return by_language


def _plot_accuracy(rows: list[SummaryRow], labels: list[str], title: str, width: float, path: Path) -> None:
    x = np.arange(len(rows))
    wer = np.array([row.wer for row in rows], dtype=float)
    cer = np.array([row.cer for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(width, 5))
    wer_bars = ax.bar(x - 0.2, wer, 0.4, label="WER", color="#4c78a8")
    cer_bars = ax.bar(x + 0.2, cer, 0.4, label="CER", color="#f58518")
    ax.bar_label(wer_bars, fmt="%.3f", fontsize=7, padding=2)
    ax.bar_label(cer_bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_ylim(0, max(1e-3, float(np.nanmax(np.concatenate([wer, cer])))) * 1.15)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("error rate (corpus-level, lower is better)")
    primary = rows[0].primary_metric.upper()
    ax.set_title(f"{title}: accuracy (primary metric: {primary})")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_speed(rows: list[SummaryRow], labels: list[str], title: str, width: float, path: Path) -> None:
    x = np.arange(len(rows))
    rtfx = np.array([row.rtfx for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(width, 4.5))
    bars = ax.bar(x, rtfx, 0.6, color="#1b9e77")
    ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("RTFx (audio s / compute s, higher is faster)")
    ax.set_title(f"{title}: speed")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def render_plots(rows: Sequence[SummaryRow], plots_dir: Path) -> list[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for (dataset, dataset_lang), language_rows in sorted(_rows_by_language(rows).items()):
        language_rows = sorted(language_rows, key=sort_key)
        labels = [f"{row.backend}\n{row.model}" for row in language_rows]
        width = max(6.0, 1.1 * len(language_rows) + 2)
        title = f"{dataset} {dataset_lang} ({language_rows[0].language_name})"

        accuracy_path = plots_dir / f"{dataset}_{dataset_lang}_accuracy.png"
        speed_path = plots_dir / f"{dataset}_{dataset_lang}_speed.png"
        _plot_accuracy(language_rows, labels, title, width, accuracy_path)
        _plot_speed(language_rows, labels, title, width, speed_path)
        written += [accuracy_path, speed_path]
    return written


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR)
    p.add_argument("--csv", type=Path, help="default: <results-dir>/summary.csv")
    p.add_argument("--plots-dir", type=Path, help="default: <results-dir>/plots")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows, warnings = collect_rows(args.results_dir, args.registry_dir)
    except (ConfigError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for warning in warnings:
        print(f"[warn] {warning}", file=sys.stderr)
    if not rows:
        print(f"No results found under {args.results_dir}. Run edge-speech-eval first.", file=sys.stderr)
        return 1

    csv_path = args.csv or args.results_dir / "summary.csv"
    write_csv(rows, csv_path)
    print(f"Saved summary CSV: {csv_path} ({len(rows)} rows)")
    if args.no_plots:
        return 0
    for path in render_plots(rows, args.plots_dir or args.results_dir / "plots"):
        print(f"Saved plot: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
