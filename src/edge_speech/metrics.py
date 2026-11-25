from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import jiwer

from .normalize import normalize_text


@dataclass(frozen=True)
class CorpusScores:
    samples: int
    ref_words: int
    ref_chars: int
    wer: float
    cer: float
    substitutions: int
    deletions: int
    insertions: int
    total_audio_s: float
    total_latency_s: float
    rtf: float
    rtfx: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def utterance_scores(reference: str, prediction: str) -> tuple[float, float]:
    ref = normalize_text(reference)
    hyp = normalize_text(prediction)
    if ref:
        return jiwer.wer(ref, hyp), jiwer.cer(ref, hyp)
    score_for_empty_reference = 1.0 if hyp else 0.0
    return score_for_empty_reference, score_for_empty_reference


def _required(record: Mapping[str, object], key: str) -> object:
    if key not in record or record[key] is None:
        raise ValueError(f"result record idx={record.get('idx')!r} has no {key!r}")
    return record[key]


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator > 0:
        return numerator / denominator
    return math.nan


def corpus_scores(records: Iterable[Mapping[str, object]]) -> CorpusScores:
    # Total errors / total reference words, as FLEURS results are normally reported.
    # Averaging per-utterance WER instead would give short clips too much weight.
    references: list[str] = []
    predictions: list[str] = []
    total_audio = 0.0
    total_latency = 0.0
    for record in records:
        reference = normalize_text(str(_required(record, "reference")))
        if not reference:
            continue
        references.append(reference)
        predictions.append(normalize_text(str(_required(record, "prediction"))))
        total_audio += float(_required(record, "duration"))
        total_latency += float(_required(record, "latency"))

    if not references:
        return CorpusScores(
            samples=0, ref_words=0, ref_chars=0, wer=math.nan, cer=math.nan,
            substitutions=0, deletions=0, insertions=0,
            total_audio_s=total_audio, total_latency_s=total_latency, rtf=math.nan, rtfx=math.nan,
        )

    words = jiwer.process_words(references, predictions)
    chars = jiwer.process_characters(references, predictions)
    return CorpusScores(
        samples=len(references),
        ref_words=words.hits + words.substitutions + words.deletions,
        ref_chars=chars.hits + chars.substitutions + chars.deletions,
        wer=float(words.wer),
        cer=float(chars.cer),
        substitutions=words.substitutions,
        deletions=words.deletions,
        insertions=words.insertions,
        total_audio_s=total_audio,
        total_latency_s=total_latency,
        rtf=safe_ratio(total_latency, total_audio),
        rtfx=safe_ratio(total_audio, total_latency),
    )
