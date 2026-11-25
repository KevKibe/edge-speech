from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator

import numpy as np

from .config import DatasetSpec

Loader = Callable[..., Iterable[dict[str, Any]]]


@dataclass
class Sample:
    idx: int
    sample_id: str | None
    audio: np.ndarray
    sampling_rate: int
    reference: str

    @property
    def duration(self) -> float:
        if not self.sampling_rate:
            return 0.0
        return len(self.audio) / self.sampling_rate


def _default_loader(*args: Any, **kwargs: Any) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset  # heavy import; keep it lazy

    return load_dataset(*args, **kwargs)


def _to_mono_f32(array: Any) -> np.ndarray:
    audio = np.asarray(array, dtype=np.float32)
    if audio.ndim == 2:
        # soundfile layout is (frames, channels); pick the channel axis as the smaller one
        channel_axis = 1 if audio.shape[0] >= audio.shape[1] else 0
        audio = audio.mean(axis=channel_axis)
    return np.ascontiguousarray(audio.reshape(-1))


def _open_split(spec: DatasetSpec, dataset_lang: str, token: str | None, loader: Loader) -> Iterable[dict[str, Any]]:
    kwargs: dict[str, Any] = {"split": spec.split, "streaming": True, "token": token}
    if spec.trust_remote_code:
        kwargs["trust_remote_code"] = True
    return loader(spec.hf_id, spec.hf_config(dataset_lang), **kwargs)


def _reference_text(row: dict[str, Any], spec: DatasetSpec) -> str | None:
    reference = row.get(spec.text_column)
    if reference is None or not str(reference).strip():
        return None
    return str(reference)


def _sample_from_row(idx: int, row: dict[str, Any], reference: str, spec: DatasetSpec) -> Sample:
    audio = row.get(spec.audio_column)
    if not isinstance(audio, dict) or "array" not in audio or "sampling_rate" not in audio:
        raise ValueError(
            f"{spec.name}: column {spec.audio_column!r} is not decoded audio "
            f"(expected a dict with 'array' and 'sampling_rate')"
        )
    sample_id = row.get("id")
    return Sample(
        idx=idx,
        sample_id=None if sample_id is None else str(sample_id),
        audio=_to_mono_f32(audio["array"]),
        sampling_rate=int(audio["sampling_rate"]),
        reference=reference,
    )


def iter_samples(
    spec: DatasetSpec,
    dataset_lang: str,
    *,
    token: str | None = None,
    max_samples: int | None = None,
    skip_idx: frozenset[int] = frozenset(),
    loader: Loader | None = None,
) -> Iterator[Sample]:
    rows = _open_split(spec, dataset_lang, token, loader or _default_loader)
    # Samples already done (skip_idx) still count toward max_samples, so a resumed
    # run stops at the same place a fresh one would.
    counted = 0
    for idx, row in enumerate(rows):
        if max_samples is not None and counted >= max_samples:
            return
        reference = _reference_text(row, spec)
        if reference is None:
            continue
        counted += 1
        if idx in skip_idx:
            continue
        yield _sample_from_row(idx, row, reference, spec)
