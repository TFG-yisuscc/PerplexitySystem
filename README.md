# PerplexitySystem

Teacher-forcing perplexity on WikiText-2 (test split) via two inference engines:
**LLAMA** (direct llama.cpp) and **OLLAMA** (Ollama HTTP API with `top_logprobs`).

**Metric name:** `wikitext2_teacher_forcing_perplexity`  
This is not "self-perplexity" (response-level). It is standard NLP PPL computed
with the model observing the ground-truth prefix at every step.

## Convention (llama.cpp b9030)

Replicates `perplexity()` from llama.cpp b9030 (`tools/perplexity/perplexity.cpp`):

- Tokenize full corpus with `add_bos` from vocabulary; BOS replaces `tokens[chunk_start]` per chunk.
- Non-overlapping chunks of `context_size` tokens; `n_chunks = n_tokens / context_size`.
- **Second-half scoring**: positions `[context_size/2, context_size-2]` per chunk.  
  Scored tokens per chunk: `context_size - context_size/2 - 1` (255 for ctx=512).
- Log-softmax in double precision; NLL accumulated in `double`.
- `PPL = exp(total_nll / n_tokens_scored)`

## Project structure

```
perplexityScorer/
├── CMakeLists.txt          # Build definition; also wires up the embedded-corpus codegen step
├── vcpkg.json              # vcpkg manifest (llama.cpp, fmt, nlohmann_json, ...)
├── includes/                # Public headers, one per module
│   ├── config.h             # Run configuration struct + JSON parsing (engine, paths, hyperparameters)
│   ├── corpus.h              # Corpus loading/tokenisation interface
│   ├── model_loader.h         # llama.cpp model/context loading wrapper
│   ├── perplexity.h           # Teacher-forcing perplexity computation interface
│   ├── ollama_engine.h        # Ollama HTTP inference engine interface
│   ├── output.h               # Result serialization (resumen.json / .jsonl) interface
│   ├── sha256.h                # SHA-256 implementation, used to fingerprint the corpus/model
│   └── third_party/
│       └── ollama.hpp          # Vendored single-header Ollama HTTP client
├── src/                      # Implementation files matching the headers above
│   ├── main.cpp                # CLI entry point: argument/JSON parsing, engine dispatch
│   ├── model_loader.cpp
│   ├── corpus.cpp
│   ├── perplexity.cpp
│   ├── output.cpp
│   └── ollama_engine.cpp
├── cmake/
│   ├── aarch64-toolchain.cmake  # vcpkg chainload toolchain for Raspberry Pi 5 cross-compilation
│   └── gen_corpus_header.py     # Generates corpus_embedded.h from corpus/wiki.test.raw at build time
├── corpus/
│   ├── wiki.test.raw           # WikiText-2 (test split), fetched manually (not tracked in git)
│   └── credits                 # Source URL for the corpus download
├── data/                      # Placeholder for user-supplied data (empty by default)
├── doc/                       # Placeholder for additional documentation (empty by default)
├── config_example.json         # Sample run configuration
└── results/                    # Generated at runtime: one timestamped folder per run (git-ignored)
```

## Getting the corpus

```bash
wget https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
unzip wikitext-2-raw-v1.zip
cp wikitext-2-raw/wiki.test.raw corpus/
```

SHA-256 of `wiki.test.raw`: `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`

## Build (host)

```bash
cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE=$HOME/.vcpkg-clion/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build cmake-build-release -j$(nproc)
```

## Build (Raspberry Pi 5 cross-compilation)

```bash
cmake -S . -B cmake-build-cross -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE=$HOME/.vcpkg-clion/vcpkg/scripts/buildsystems/vcpkg.cmake \
      -DVCPKG_CHAINLOAD_TOOLCHAIN_FILE=$(pwd)/cmake/aarch64-toolchain.cmake \
      -DVCPKG_TARGET_TRIPLET=arm64-linux
cmake --build cmake-build-cross -j$(nproc)
```

## Run

```bash
# From a JSON file
./PerplexitySystem config.json

# Inline JSON
./PerplexitySystem --json '{"model_path":"/path/to/model.gguf","corpus_path":"/path/to/wiki.test.raw"}'
```

## Configuration

### engine=LLAMA (default)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `corpus_path` | string | — | Path to `wiki.test.raw` (required) |
| `model_path` | string | — | Path to GGUF model (required) |
| `inference_engine` | string | `"LLAMA"` | `"LLAMA"` or `"OLLAMA"` |
| `context_size` | int | `512` | Evaluation window in tokens |
| `batch_size` | int | `context_size` | `llama_decode` batch size (must be ≥ context_size) |
| `n_threads` | int | all cores | CPU decode threads |
| `seed` | int | `0` | Stored in metadata only |
| `output_dir` | string | `"results"` | Root directory for output files |
| `annotations` | any | `null` | Free-form JSON metadata passed through to output |

### engine=OLLAMA

Requires Ollama ≥ 0.12.11 running locally with a model pulled.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `corpus_path` | string | — | Path to `wiki.test.raw` (required) |
| `inference_engine` | string | — | `"OLLAMA"` (required) |
| `ollama_model_name` | string | — | Model name in Ollama, e.g. `"llama3.2:1b"` (required) |
| `model_path` | string | — | GGUF path for tokenisation/detokenisation (required in practice) |
| `ollama_url` | string | `"http://localhost:11434"` | Ollama server URL |
| `top_logprobs_k` | int [1–20] | `20` | Candidates per position |
| `max_tokens_scored` | int | `0` (unlimited) | Stop early after N tokens (useful for quick tests) |
| `ollama_raw` | bool | `true` | Bypass Ollama chat template |
| `context_size` | int | `512` | Chunk size in tokens (uses GGUF tokeniser) |
| `seed` | int | `0` | Stored in metadata only |
| `output_dir` | string | `"results"` | Root directory for output files |
| `annotations` | any | `null` | Free-form JSON metadata |

> **Token matching note:** The OLLAMA engine matches target tokens with three strategies:
> exact string → normalize `▁`→space → byte-array comparison.
> `coverage_pct` in the output shows what fraction of positions were found in the
> top-K candidates. Coverage < 90% usually means the `model_path` tokeniser
> does not match the Ollama model's tokeniser — use the GGUF of the same model.

## Output

```
results/
└── <timestamp_ns>/
    ├── resumen.json                                # full metadata + result
    └── <ts>_perplexity_<model>.jsonl              # compact single-line record
```

`resumen.json` keys: `run_id`, `ts_start_ns`, `ts_end_ns`, `ts_start_iso`, `ts_end_iso`,
`duration_sec`, `metric`, `convention`, `engine`, `corpus`, `model`, `ollama_meta`,
`result` (perplexity, nll_mean, ppl_stderr, n_tokens_scored, n_chunks, add_bos,
n_tokens_missing, coverage_pct, per_chunk_ppl), `annotations`, `config`.

`ollama_meta` is `null` for the LLAMA engine.

## Experiment E4 — llama.cpp vs Ollama quality

To compare the two engines on the same model:

```bash
# LLAMA engine — direct via GGUF
./PerplexitySystem --json '{
  "corpus_path": "corpus/wiki.test.raw",
  "model_path": "/path/to/llama3.2-1b-q8_0.gguf",
  "context_size": 512,
  "annotations": {"engine_variant": "llama_cpp", "experiment": "E4"}
}'

# OLLAMA engine — via HTTP API
./PerplexitySystem --json '{
  "inference_engine": "OLLAMA",
  "corpus_path": "corpus/wiki.test.raw",
  "ollama_model_name": "llama3.2:1b",
  "model_path": "/path/to/llama3.2-1b-q8_0.gguf",
  "context_size": 512,
  "annotations": {"engine_variant": "ollama", "experiment": "E4"}
}'
```

Use the **same** GGUF for `model_path` in both runs to ensure identical tokenisation.
