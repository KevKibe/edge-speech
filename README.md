# edge-speech

Evaluate [CrispASR](https://github.com/CrispStrobe/CrispASR) backends and GGUF models on
Hugging Face speech datasets, starting with FLEURS. Each job is a
**dataset × language × backend × model** combination.

## Setup

```bash
uv sync --group dev
./build_binary.sh            # clones + builds CrispASR, installs bin/crispasr
# optional: CRISPASR_REF=v0.8.55 (pin a version), BUILD_TARGET=crispasr-cli (build only the CLI)
echo "HF_TOKEN=hf_..." >> .env   # optional, for gated repos
```

> macOS note: the scipy wheel used on Python 3.10 fails to load on recent macOS, which breaks
> FLEURS streaming. The project is pinned to Python 3.12 (`.python-version`), so `uv sync --group dev` picks it automatically.

## Running evaluations

Sweep files in [configs/sweeps/](configs/sweeps/) describe what to run:

```yaml
defaults:
  dataset: fleurs
  max_samples: 100
  runner: cli          # cli: crispasr binary, one process per batch | session: Python bindings
  batch_size: 25

runs:
  - backend: parakeet
    languages: [en_us, fr_fr, sw_ke]   # or `all`; unsupported languages are skipped
    models:
      - auto:q4_k                       # CrispASR registry default at a quant
      - hf: cstr/parakeet-tdt-0.6b-v3-GGUF:parakeet-tdt-0.6b-v3-q8_0.gguf
      - path: models/my-model.gguf
    options: {vad: false, chunk_seconds: null, extra_args: []}

  - backend: omniasr
    languages: all
    quant_sweep:
      repo: cstr/omniASR-CTC-300M-v2-GGUF
      file_template: "omniasr-ctc-300m-v2-{quant}.gguf"
      quants: [q8_0, q4_k]
```

```bash
uv run edge-speech-eval --sweep configs/sweeps/fleurs_multi_backend.yaml --dry-run   # list jobs + skips
uv run edge-speech-eval --sweep configs/sweeps/fleurs_multi_backend.yaml
uv run edge-speech-eval --backend whisper --model hf:ggerganov/whisper.cpp:ggml-small.bin \
    --languages sw_ke en_us --max-samples 50                                          # ad-hoc run
uv run edge-speech-analyze          # results/summary.csv + results/plots/
```

Runs **resume**: finished samples are skipped. Failed samples go to `*.errors.jsonl` and are
retried on the next run. Pass `--overwrite` to start over.

Runs fail loudly instead of producing results that look valid:

- Before timing starts, each job runs an untimed **warm-up** on a 3 s speech clip ([warmup.wav](src/edge_speech/assets/warmup.wav), from jfk.wav; speech, because `--vad` gives no output on silence). It checks that the
  binary, model and options work, triggers any downloads, and records the model file crispasr
  actually loaded (`model_file`; `quant` comes from it for `auto` models).
- A job with any failed sample ends as `failed` (exit code 1), even though its successful samples are kept.
- If crispasr warns that it ignored a requested option (e.g. `--vad` on a backend without VAD), the job fails.
- Resuming into results made with different settings (options, model, runner, crispasr version)
  is refused; use `--overwrite` or another `--output-dir`.
- Malformed crispasr output, corrupt result lines (other than a half-written last line), records
  missing fields, and unknown backends or languages are errors.

### Session runner

`runner: session` keeps the model loaded in-process and times each utterance exactly:

```bash
uv pip install -e CrispASR/python
export CRISPASR_LIB_PATH=$PWD/CrispASR/build/src/libcrispasr.dylib   # .so on Linux
```

It takes all three model forms. For `auto`, the warm-up runs the CLI once to resolve and download the
model file, then the session loads that file. It doesn't support `vad`, `chunk_seconds` or `extra_args`.

The CLI runner's speed numbers include one model load per batch, because each batch is a new
crispasr process. Use the session runner for RTFx without model loading.

## Configuration

| File | Purpose |
|---|---|
| [configs/datasets.yaml](configs/datasets.yaml) | HF dataset adapters: repo, split, audio/text columns, language keys → FLORES codes |
| [configs/languages.yaml](configs/languages.yaml) | FLORES-200 code → name, ISO 639-1, whether words are space-delimited |
| [configs/backends.yaml](configs/backends.yaml) | per backend: language code style (`iso1`/`flores`/`none`), supported languages, default options |

To add a dataset, add an entry to `datasets.yaml`. To restrict or alias a backend's languages,
edit `backends.yaml`. A backend must be listed there before it can be used.
The language lists come from the CrispASR README, because `crispasr --list-backends-json` doesn't include them.

## Results

```
results/{dataset}/{dataset_lang}/{backend}/{model}.jsonl          one record per utterance, full metadata
results/{dataset}/{dataset_lang}/{backend}/{model}.summary.json   corpus metrics for the job
results/{dataset}/{dataset_lang}/{backend}/{model}.errors.jsonl   failed samples
```

Scoring:

- Reference and prediction both pass through the same normalizer
  ([normalize.py](src/edge_speech/normalize.py)): NFKC, lowercase, bracketed tags removed,
  punctuation and symbols stripped, combining marks kept. Punctuating and non-punctuating
  backends are therefore comparable.
- WER and CER are **corpus-level**: total errors ÷ total reference words or characters.
- For languages written without spaces (zh, ja, th, lo, km, my, yue), CER is the primary metric.
- `rtf` = compute time ÷ audio time; `rtfx` = 1 / rtf.
- `latency_method` records how latency was measured: `session` (exact, per utterance),
  `batch_apportioned` (one CLI process per batch, split by duration), or
  `process_with_model_load` (legacy runs, which include model loading).

The committed `results/fleurs/sw_ke/omniasr/` files are migrated from earlier runs. Those ran with
`--vad --chunk-seconds 5` and a model load for every sample, so their speed numbers aren't
comparable with new runs.

## Tests

```bash
./build_binary.sh                      # bin/crispasr and libcrispasr (session runner)
uv pip install -e CrispASR/python      # crispasr Python bindings
uv run pytest -s
```

Every test uses the real crispasr build, real models and real data; nothing is faked. A missing
build fails the suite (it never skips). The tests need network on first run to download models
and to stream FLEURS.

- [tests/test_transcription.py](tests/test_transcription.py):
  - transcribes the local audio file [tests/data/jfk.wav](tests/data/jfk.wav) with a local model
    file, a Hugging Face model (`hf:`) and a crispasr `auto` model (moonshine), each through both
    runners (`cli`, `session`);
  - transcribes a batch of files in one CLI process, and with `vad` + `chunk_seconds`;
  - runs a sweep on two real FLEURS `en_us` samples with both runners, resumes it, and analyzes
    the results (CSV and plots).
- [tests/test_configs_and_results.py](tests/test_configs_and_results.py): every sweep in
  `configs/sweeps/` and an ad-hoc run expand (`--dry-run`), and the committed results analyze.

GitHub Actions ([.github/workflows/tests.yml](.github/workflows/tests.yml)) builds crispasr with
`build_binary.sh` (pinned via `CRISPASR_REF`; the CLI and the shared library only; cached),
installs the bindings and runs the whole suite on Python 3.12.
