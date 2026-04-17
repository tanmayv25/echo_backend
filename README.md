<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Echo Backend [Experimental]

An illustrative [Dynamo](https://github.com/ai-dynamo/dynamo) backend
that echoes the request tokens back to the caller. It exists to show
backend developers **the minimum surface required** to wire a new
inference engine into Dynamo using the
[`dynamo.common.backend`](https://github.com/ai-dynamo/dynamo/tree/main/components/src/dynamo/common/backend)
module, and to give each backend a home for its own CI and container
image instead of bundling everything into the main `dynamo` repo.

## Why a separate repo?

Writing a backend against `dynamo.common.backend` is the easy part --
the friction is everywhere *around* it: GitHub Actions wiring,
Dockerfile templates, release pipelines. Keeping each backend in its
own repo:

- Isolates per-engine CI (no need to pull in every framework's tests).
- Lets each backend pick its own container base image and dependency
  pins.
- Keeps the core `dynamo` repo lean.

The long-term plan `vllm`, `trtllm`, and `sglang` can follow the same pattern.

## Repository layout

```
echo_backend/
├── src/echo_backend/
│   ├── engine.py           # EchoLLMEngine -- the LLMEngine subclass
│   ├── main.py             # Entry point -- run(EchoLLMEngine)
│   └── __main__.py         # `python -m echo_backend`
├── tests/
│   └── test_engine.py      # Unit tests (no distributed runtime needed)
├── container/
│   └── Dockerfile          # Multi-stage build on top of ai-dynamo
├── scripts/
│   └── install-dynamo.sh   # Shared installer; classifies DYNAMO_REF
├── .github/
│   ├── dynamo-ref.txt      # ai-dynamo ref CI tests against
│   └── workflows/ci.yml    # Pre-commit + pytest
├── compose.yaml            # Local etcd + NATS + frontend + worker stack
├── pyproject.toml
└── .pre-commit-config.yaml
```

## Quickstart

### Install

```bash
python -m pip install -e ".[dev]"
```

This pulls `ai-dynamo` from PyPI. To build against the latest `dynamo`
`main` branch instead (requires a Rust toolchain), replace the
dependency in `pyproject.toml`:

```toml
dependencies = [
    "ai-dynamo @ git+https://github.com/ai-dynamo/dynamo.git@main",
]
```

### Run tests

```bash
pytest -v
```

### Run the backend

You will need running `etcd` and `nats` instances -- see the Dynamo
[quickstart](https://github.com/ai-dynamo/dynamo#quickstart).

```bash
python -m echo_backend \
    --model-name echo-model \
    --namespace dynamo \
    --repeat-count 3 \
    --delay-seconds 0.02
```

Send it a request through the Dynamo frontend; the response will be
your prompt repeated `--repeat-count` times.

### Run with docker compose

[`compose.yaml`](compose.yaml) brings up etcd, NATS (JetStream), the
Dynamo HTTP frontend on `localhost:8000`, and the echo worker -- all on
a private network:

```bash
docker compose up --build                         # dynamo main, from source
DYNAMO_REF=1.1.0 docker compose up --build        # PyPI pinned release (fast)
DYNAMO_REF= docker compose up --build             # PyPI transitive via pyproject
```

The first invocation builds `ai-dynamo` from `main` (Rust toolchain,
slow first build, fastest iteration against upstream). The PyPI variants
skip Rust and install from wheels, but need a release compatible with
this repo's `ai-dynamo>=1.1.0` pin -- `dynamo.common.backend` will land
in 1.1.0, so older PyPI releases won't import.

Send a request:

```bash
curl -s localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model": "echo-model",
         "messages": [{"role": "user", "content": "hello"}],
         "max_tokens": 32}'
```

Tear it down:

```bash
docker compose down
```

The worker borrows the `Qwen/Qwen3-0.6B` tokenizer (for its chat
template) and serves as `echo-model`; see the comment in `compose.yaml`
for context.

## Implementing a new backend

The abstract interface lives at
[`dynamo.common.backend.LLMEngine`](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/common/backend/engine.py).
A backend implements four methods:

| Method | Purpose |
|---|---|
| `from_args(argv)` | Parse CLI args, return `(engine, WorkerConfig)` |
| `start()` | Start the engine, return `EngineConfig` with registration metadata |
| `generate(request, context)` | Async-generator yielding `GenerateChunk` dicts |
| `cleanup()` | Release all engine resources |

`abort(context)` is optional -- override it when your engine needs to
release scheduler slots or KV cache blocks on cancellation.

See `src/echo_backend/engine.py` for a minimal implementation. The
entry point in `src/echo_backend/main.py` is three lines:

```python
from dynamo.common.backend.run import run
from echo_backend.engine import EchoLLMEngine

def main() -> None:
    run(EchoLLMEngine)
```

### Request / response contract

`generate()` receives a `GenerateRequest` (`TypedDict`) with:

- `token_ids` (required): prompt tokens from the Rust preprocessor.
- `sampling_options`, `stop_conditions`, `output_options`
  (optional dicts): engine-specific pass-through data.

and yields `GenerateChunk` dicts with:

- `token_ids` (required on every chunk).
- `finish_reason` and `completion_usage` (required on the final chunk).

Finish reason normalisation (`"abort"` → `"cancelled"`, etc.) is
handled by the Rust layer -- emit whichever string matches your
engine's native finish reason.

## CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs two jobs on
every PR:

1. **pre-commit** — isort, black, flake8, ruff, codespell, and the
   standard hygiene hooks (mirrors the configuration in
   [`ai-dynamo/dynamo`](https://github.com/ai-dynamo/dynamo/blob/main/.pre-commit-config.yaml)).
2. **unit-tests** — `pytest` on Python 3.11 / 3.12.

The tests in `tests/test_engine.py` exercise `from_args`, `start`,
`generate` (echo correctness, streaming cadence, cancellation,
`max_tokens` cap, empty prompt) and `cleanup` without spinning up a
`Worker` or the distributed runtime -- Dynamo's `Context` is replaced
by a deterministic stub.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0 -- see [LICENSE](LICENSE).
