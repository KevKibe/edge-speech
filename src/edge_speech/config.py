from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_DIR = REPO_ROOT / "configs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results"
DEFAULT_CRISP_BIN = REPO_ROOT / "bin" / "crispasr"

LANG_STYLES = ("iso1", "flores", "none")
RUNNERS = ("cli", "session")
OPTION_KEYS = {"vad", "chunk_seconds", "threads", "extra_args"}
RUN_KEYS = {
    "backend", "dataset", "languages", "models", "model", "quant_sweep", "options",
    "runner", "batch_size", "max_samples", "output_dir", "lang_override",
}
QUANT_SWEEP_KEYS = {"repo", "file_template", "quants"}
QUANT_PATTERN = re.compile(
    r"(?:^|[-_.])(f32|f16|bf16|q\d_k(?:_[sml])?|q\d_[01]|iq\d_\w+?|tq\d_0)(?:$|[-_.])", re.I
)
UNSAFE_LABEL_CHARS = re.compile(r"[^A-Za-z0-9._+-]+")


class ConfigError(ValueError):
    pass


class UnsupportedLanguage(Exception):
    pass


@dataclass(frozen=True)
class LanguageSpec:
    code: str  # FLORES-200 code, the canonical key
    name: str
    iso1: str | None
    space_delimited: bool = True


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    hf_id: str
    split: str = "test"
    audio_column: str = "audio"
    text_column: str = "transcription"
    config_template: str = "{lang}"
    trust_remote_code: bool = False
    languages: dict[str, str] = field(default_factory=dict)  # dataset key -> FLORES code

    def hf_config(self, dataset_lang: str) -> str:
        return self.config_template.format(lang=dataset_lang)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


@dataclass(frozen=True)
class RunOptions:
    vad: bool = False
    chunk_seconds: float | None = None
    threads: int | None = None
    extra_args: tuple[str, ...] = ()

    def merged(self, raw: dict[str, Any] | None) -> RunOptions:
        if not raw:
            return self
        unknown = set(raw) - OPTION_KEYS
        if unknown:
            raise ConfigError(f"unknown option(s): {sorted(unknown)}")

        updates: dict[str, Any] = {}
        if "vad" in raw:
            updates["vad"] = bool(raw["vad"])
        if "chunk_seconds" in raw:
            updates["chunk_seconds"] = _optional_float(raw["chunk_seconds"])
        if "threads" in raw:
            updates["threads"] = _optional_int(raw["threads"])
        if "extra_args" in raw:
            updates["extra_args"] = _parse_extra_args(raw["extra_args"])
        return replace(self, **updates)

    def as_dict(self) -> dict[str, Any]:
        return {
            "vad": self.vad,
            "chunk_seconds": self.chunk_seconds,
            "threads": self.threads,
            "extra_args": list(self.extra_args),
        }


def _parse_extra_args(raw: Any) -> tuple[str, ...]:
    args = raw or []
    if not isinstance(args, list):
        raise ConfigError("extra_args must be a list of strings")
    return tuple(str(arg) for arg in args)


@dataclass(frozen=True)
class BackendSpec:
    name: str
    lang_style: str = "iso1"
    languages: frozenset[str] | None = None  # None = any
    lang_aliases: dict[str, str] = field(default_factory=dict)
    options: RunOptions = RunOptions()

    def language_code(self, lang: LanguageSpec) -> str | None:
        code = self._code_for(lang)
        if self.languages is None:
            return code
        code_to_check = code if code is not None else lang.iso1
        if code_to_check not in self.languages:
            raise UnsupportedLanguage(f"{self.name}: {lang.code} ({lang.name}) not in supported languages")
        return code

    def _code_for(self, lang: LanguageSpec) -> str | None:
        if self.lang_style == "none":
            return None
        if lang.code in self.lang_aliases:
            return self.lang_aliases[lang.code]
        if self.lang_style == "flores":
            return lang.code
        if lang.iso1 is None:
            raise UnsupportedLanguage(f"{self.name}: {lang.code} ({lang.name}) has no ISO 639-1 code")
        return lang.iso1


def detect_quant(label: str) -> str | None:
    match = QUANT_PATTERN.search(label)
    if match is None:
        return None
    return match.group(1).lower()


def _safe_label(text: str) -> str:
    return UNSAFE_LABEL_CHARS.sub("_", text).strip("_") or "model"


@dataclass(frozen=True)
class ModelSpec:
    kind: str  # "path" (local file), "hf" (file in a Hugging Face repo) or "auto" (CrispASR registry default)
    path: str | None = None
    repo: str | None = None
    file: str | None = None
    quant: str | None = None
    label_override: str | None = None

    @classmethod
    def parse(cls, raw: Any) -> ModelSpec:
        if isinstance(raw, str):
            return cls._parse_string(raw)
        if isinstance(raw, dict):
            return cls._parse_mapping(raw)
        raise ConfigError(f"model must be a string or mapping, got {type(raw).__name__}")

    @classmethod
    def _parse_string(cls, raw: str) -> ModelSpec:
        if raw == "auto" or raw.startswith("auto:"):
            _, _, quant = raw.partition(":")
            return cls(kind="auto", quant=quant or None)
        if raw.startswith("hf:"):
            return cls._parse_hf(raw.removeprefix("hf:"))
        return cls(kind="path", path=raw)

    @classmethod
    def _parse_mapping(cls, raw: dict[str, Any]) -> ModelSpec:
        sources = {"path", "hf", "auto"} & set(raw)
        if len(sources) != 1:
            raise ConfigError(f"model needs exactly one of path/hf/auto, got {raw!r}")

        if "path" in raw:
            spec = cls(kind="path", path=str(raw["path"]))
        elif "hf" in raw:
            spec = cls._parse_hf(str(raw["hf"]))
        else:
            spec = cls(kind="auto", quant=_auto_quant(raw["auto"]))

        label = raw.get("label")
        return replace(spec, label_override=str(label) if label else None)

    @classmethod
    def _parse_hf(cls, ref: str) -> ModelSpec:
        repo, separator, file = ref.partition(":")
        if not separator or not file or repo.count("/") != 1:
            raise ConfigError(f"hf model must be 'owner/repo:file', got {ref!r}")
        return cls(kind="hf", repo=repo, file=file)

    def label(self, backend: str) -> str:
        if self.label_override:
            return _safe_label(self.label_override)
        if self.kind == "path":
            return _safe_label(Path(self.path or "").stem)
        if self.kind == "hf":
            return _safe_label(Path(self.file or "").stem)
        if self.quant:
            return _safe_label(f"{backend}-auto-{self.quant}")
        return _safe_label(f"{backend}-auto")

    def source(self) -> str:
        if self.kind == "path":
            return str(self.path)
        if self.kind == "hf":
            return f"hf:{self.repo}:{self.file}"
        if self.quant:
            return f"auto:{self.quant}"
        return "auto"


def _auto_quant(value: Any) -> str | None:
    if value in (None, True, "default"):
        return None
    return str(value)


@dataclass(frozen=True)
class Registry:
    datasets: dict[str, DatasetSpec]
    languages: dict[str, LanguageSpec]
    backends: dict[str, BackendSpec]

    def backend(self, name: str) -> BackendSpec:
        if name not in self.backends:
            raise ConfigError(
                f"backend {name!r} is not in configs/backends.yaml; add it with its lang_style and languages"
            )
        return self.backends[name]

    def dataset(self, name: str) -> DatasetSpec:
        if name not in self.datasets:
            raise ConfigError(f"unknown dataset {name!r}; known: {sorted(self.datasets)}")
        return self.datasets[name]


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def _string_dict(raw: dict[Any, Any] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in (raw or {}).items():
        result[str(key)] = str(value)
    return result


def _load_languages(path: Path) -> dict[str, LanguageSpec]:
    languages: dict[str, LanguageSpec] = {}
    for code, raw in _read_yaml(path).items():
        if not isinstance(raw, dict) or "name" not in raw:
            raise ConfigError(f"languages.yaml: {code} needs at least a name")
        iso1 = raw.get("iso1")
        languages[code] = LanguageSpec(
            code=code,
            name=str(raw["name"]),
            iso1=None if iso1 is None else str(iso1),
            space_delimited=bool(raw.get("space_delimited", True)),
        )
    return languages


def _load_datasets(path: Path, languages: dict[str, LanguageSpec]) -> dict[str, DatasetSpec]:
    datasets: dict[str, DatasetSpec] = {}
    for name, raw in _read_yaml(path).items():
        if not isinstance(raw, dict) or "hf_id" not in raw:
            raise ConfigError(f"datasets.yaml: {name} needs hf_id")
        dataset_languages = _string_dict(raw.get("languages"))
        unknown = sorted(code for code in dataset_languages.values() if code not in languages)
        if unknown:
            raise ConfigError(f"datasets.yaml: {name} maps to unknown languages {unknown}")
        datasets[name] = DatasetSpec(
            name=name,
            hf_id=str(raw["hf_id"]),
            split=str(raw.get("split", "test")),
            audio_column=str(raw.get("audio_column", "audio")),
            text_column=str(raw.get("text_column", "transcription")),
            config_template=str(raw.get("config_template", "{lang}")),
            trust_remote_code=bool(raw.get("trust_remote_code", False)),
            languages=dataset_languages,
        )
    return datasets


def _load_backends(path: Path, languages: dict[str, LanguageSpec]) -> dict[str, BackendSpec]:
    backends: dict[str, BackendSpec] = {}
    for name, raw in _read_yaml(path).items():
        raw = raw or {}
        lang_style = str(raw.get("lang_style", "iso1"))
        if lang_style not in LANG_STYLES:
            raise ConfigError(f"backends.yaml: {name}.lang_style must be one of {LANG_STYLES}")
        aliases = _string_dict(raw.get("lang_aliases"))
        unknown = sorted(code for code in aliases if code not in languages)
        if unknown:
            raise ConfigError(f"backends.yaml: {name}.lang_aliases has unknown languages {unknown}")
        supported = raw.get("languages")
        backends[name] = BackendSpec(
            name=name,
            lang_style=lang_style,
            languages=None if supported is None else frozenset(str(code) for code in supported),
            lang_aliases=aliases,
            options=RunOptions().merged(raw.get("options")),
        )
    return backends


def load_registry(registry_dir: Path = DEFAULT_REGISTRY_DIR) -> Registry:
    registry_dir = Path(registry_dir)
    languages = _load_languages(registry_dir / "languages.yaml")
    datasets = _load_datasets(registry_dir / "datasets.yaml", languages)
    backends = _load_backends(registry_dir / "backends.yaml", languages)
    return Registry(datasets=datasets, languages=languages, backends=backends)


@dataclass(frozen=True)
class Job:
    dataset: DatasetSpec
    dataset_lang: str
    language: LanguageSpec
    backend: BackendSpec
    asr_language: str | None
    model: ModelSpec
    options: RunOptions
    runner: str = "cli"
    batch_size: int = 16
    max_samples: int | None = None
    output_dir: Path = DEFAULT_OUTPUT_DIR

    @property
    def model_label(self) -> str:
        return self.model.label(self.backend.name)

    @property
    def output_path(self) -> Path:
        job_dir = Path(self.output_dir) / self.dataset.name / self.dataset_lang / self.backend.name
        return job_dir / f"{self.model_label}.jsonl"

    @property
    def errors_path(self) -> Path:
        return self.output_path.with_suffix(".errors.jsonl")

    @property
    def summary_path(self) -> Path:
        return self.output_path.with_suffix(".summary.json")

    def describe(self) -> str:
        lang = self.asr_language if self.asr_language is not None else "-"
        return f"{self.dataset.name}/{self.dataset_lang} [{lang}] {self.backend.name}:{self.model_label}"

    def metadata(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset.name,
            "hf_dataset": self.dataset.hf_id,
            "split": self.dataset.split,
            "dataset_lang": self.dataset_lang,
            "language": self.language.code,
            "asr_language": self.asr_language,
            "backend": self.backend.name,
            "model": self.model_label,
            "model_source": self.model.source(),
            "quant": detect_quant(self.model_label),
            "runner": self.runner,
            "options": self.options.as_dict(),
        }


@dataclass
class Expansion:
    jobs: list[Job]
    skipped: list[str]


@dataclass(frozen=True)
class RunSettings:
    where: str
    backend: BackendSpec
    dataset: DatasetSpec
    runner: str
    batch_size: int
    max_samples: int | None
    output_dir: Path
    options: RunOptions
    lang_override: dict[str, str]
    dataset_langs: list[str]
    models: list[ModelSpec]


def _as_list(value: Any, what: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (str, dict)):
        return [value]
    raise ConfigError(f"{what} must be a list")


def _quant_sweep_models(sweep: Any, where: str) -> list[ModelSpec]:
    if not isinstance(sweep, dict) or not QUANT_SWEEP_KEYS <= set(sweep):
        raise ConfigError(f"{where}.quant_sweep needs repo, file_template, quants")
    file_template = str(sweep["file_template"])
    if "{quant}" not in file_template:
        raise ConfigError(f"{where}.quant_sweep.file_template must contain {{quant}}")
    models = []
    for quant in _as_list(sweep["quants"], f"{where}.quant_sweep.quants"):
        file = file_template.format(quant=quant)
        models.append(ModelSpec(kind="hf", repo=str(sweep["repo"]), file=file))
    return models


def _run_models(run: dict[str, Any], where: str) -> list[ModelSpec]:
    models = []
    for raw in _as_list(run.get("models"), f"{where}.models"):
        models.append(ModelSpec.parse(raw))
    if "model" in run:
        models.append(ModelSpec.parse(run["model"]))
    if run.get("quant_sweep") is not None:
        models.extend(_quant_sweep_models(run["quant_sweep"], where))
    if not models:
        raise ConfigError(f"{where} needs models, model, or quant_sweep")
    return models


def _check_run_keys(run: Any, where: str) -> None:
    if not isinstance(run, dict):
        raise ConfigError(f"{where} must be a mapping")
    unknown = set(run) - RUN_KEYS
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}")


def _parse_runner(value: Any, where: str) -> str:
    runner = str(value)
    if runner not in RUNNERS:
        raise ConfigError(f"{where}.runner must be one of {RUNNERS}")
    return runner


def _parse_batch_size(value: Any, where: str) -> int:
    batch_size = int(value)
    if batch_size < 1:
        raise ConfigError(f"{where}.batch_size must be >= 1")
    return batch_size


def _dataset_languages(value: Any, dataset: DatasetSpec, where: str) -> list[str]:
    if value is None:
        raise ConfigError(f"{where} missing languages (a list, or 'all')")
    if value == "all":
        return list(dataset.languages)
    return [str(lang) for lang in _as_list(value, f"{where}.languages")]


def _parse_run(
    run: dict[str, Any],
    where: str,
    registry: Registry,
    defaults: dict[str, Any],
    overrides: dict[str, Any],
) -> RunSettings:
    _check_run_keys(run, where)
    merged = {**defaults, **run, **overrides}
    if "backend" not in merged:
        raise ConfigError(f"{where} missing backend")

    backend = registry.backend(str(merged["backend"]))
    dataset = registry.dataset(str(merged.get("dataset", "fleurs")))
    runner = _parse_runner(merged.get("runner", "cli"), where)
    batch_size = _parse_batch_size(merged.get("batch_size", 16), where)
    max_samples = _optional_int(merged.get("max_samples"))
    output_dir = Path(merged.get("output_dir") or DEFAULT_OUTPUT_DIR)
    options = backend.options.merged(defaults.get("options")).merged(run.get("options"))
    lang_override = _string_dict(run.get("lang_override"))
    dataset_langs = _dataset_languages(merged.get("languages"), dataset, where)
    models = _run_models(run, where)

    return RunSettings(
        where=where,
        backend=backend,
        dataset=dataset,
        runner=runner,
        batch_size=batch_size,
        max_samples=max_samples,
        output_dir=output_dir,
        options=options,
        lang_override=lang_override,
        dataset_langs=dataset_langs,
        models=models,
    )


def _resolve_asr_language(settings: RunSettings, dataset_lang: str, lang: LanguageSpec) -> str | None:
    if dataset_lang in settings.lang_override:
        return settings.lang_override[dataset_lang]
    return settings.backend.language_code(lang)


def _expand_run(settings: RunSettings, registry: Registry) -> Expansion:
    expansion = Expansion(jobs=[], skipped=[])
    dataset = settings.dataset
    for dataset_lang in settings.dataset_langs:
        if dataset_lang not in dataset.languages:
            raise ConfigError(f"{settings.where}: {dataset_lang!r} is not a {dataset.name} language")
        lang = registry.languages[dataset.languages[dataset_lang]]
        try:
            asr_language = _resolve_asr_language(settings, dataset_lang, lang)
        except UnsupportedLanguage as exc:
            expansion.skipped.append(f"{dataset.name}/{dataset_lang}: {exc}")
            continue
        for model in settings.models:
            expansion.jobs.append(
                Job(
                    dataset=dataset,
                    dataset_lang=dataset_lang,
                    language=lang,
                    backend=settings.backend,
                    asr_language=asr_language,
                    model=model,
                    options=settings.options,
                    runner=settings.runner,
                    batch_size=settings.batch_size,
                    max_samples=settings.max_samples,
                    output_dir=settings.output_dir,
                )
            )
    return expansion


def _check_unique_outputs(jobs: list[Job]) -> None:
    seen: set[Path] = set()
    for job in jobs:
        if job.output_path in seen:
            raise ConfigError(f"two jobs write to {job.output_path}; give one of the models a distinct `label`")
        seen.add(job.output_path)


def expand_runs(
    runs: Iterable[dict[str, Any]],
    registry: Registry,
    defaults: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> Expansion:
    # Precedence: overrides (CLI flags) > run > defaults > backend registry.
    defaults = dict(defaults or {})
    set_overrides = {key: value for key, value in (overrides or {}).items() if value is not None}

    expansion = Expansion(jobs=[], skipped=[])
    for number, run in enumerate(runs, start=1):
        settings = _parse_run(run, f"runs[{number}]", registry, defaults, set_overrides)
        run_expansion = _expand_run(settings, registry)
        expansion.jobs.extend(run_expansion.jobs)
        expansion.skipped.extend(run_expansion.skipped)

    _check_unique_outputs(expansion.jobs)
    return expansion


def load_sweep(path: Path, registry: Registry, overrides: dict[str, Any] | None = None) -> Expansion:
    data = _read_yaml(Path(path))
    defaults = data.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ConfigError("defaults must be a mapping")
    runs = data.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ConfigError("sweep must contain a non-empty runs list")
    return expand_runs(runs, registry, defaults, overrides)
