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


## Repository layout

```
echo_backend/
├── src/echo_backend/
│   ├── __init__.py         # Re-exports EchoLLMEngine
│   ├── engine.py           # EchoLLMEngine -- the LLMEngine subclass
│   ├── main.py             # Entry point -- run(EchoLLMEngine)
│   └── __main__.py         # `python -m echo_backend`
├── tests/
│   ├── test_engine.py      # Unit tests (no distributed runtime needed)
│   └── test_integration.py # End-to-end test (opt-in, needs the stack up)
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

This pulls `ai-dynamo` from PyPI (the `>=1.3.0` pin in `pyproject.toml`
is the floor that carries the unified backend API used here). To build
against the latest `dynamo` `main` branch instead (requires a Rust
toolchain), replace the dependency in `pyproject.toml`:

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
a private network. (The worker downloads the small `Qwen/Qwen3-0.6B`
tokenizer on first start -- it has no weights to load, but Dynamo's
model registration needs a real HF repo with a chat template; see the
note in `compose.yaml`.)

```bash
docker compose up --build                         # dynamo main, from source
DYNAMO_REF=1.3.0 docker compose up --build        # PyPI pinned release (fast)
DYNAMO_REF= docker compose up --build             # PyPI transitive via pyproject
```

The first invocation builds `ai-dynamo` from `main` (Rust toolchain,
slow first build, fastest iteration against upstream). The PyPI variants
skip Rust and install from wheels, but need a release compatible with
this repo's `ai-dynamo>=1.3.0` pin -- the unified backend API used here
(`EngineConfig.llm` / `LlmRegistration`, `start(worker_id)`, the
required `GenerateChunk["index"]` field) is current as of 1.3.x, so
older PyPI releases won't import.

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

## Implementing a new backend

The abstract interface lives at
[`dynamo.common.backend.LLMEngine`](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/common/backend/engine.py)
(which subclasses the modality-agnostic `BaseEngine`). A backend
implements four methods:

| Method | Purpose |
|---|---|
| `from_args(argv)` | Parse CLI args, return `(engine, WorkerConfig)` |
| `start(worker_id)` | Start the engine, return `EngineConfig` with registration metadata. `worker_id` is a runtime-allocated, cluster-unique id (ignore it unless you need a per-worker key) |
| `generate(request, context)` | Async-generator yielding `GenerateChunk` dicts |
| `cleanup()` | Release all engine resources |

`abort(context)` is optional -- override it when your engine needs to
release scheduler slots or KV cache blocks on cancellation. `BaseEngine`
exposes other optional hooks (`health_check_payload()`,
`kv_event_sources()`, `register_prometheus()`, ...) that all default to
no-ops; a minimal backend like this one leaves them alone. See
`dynamo.common.backend.sample_engine.SampleLLMEngine` upstream for a
richer example that opts into KV events, metrics, logprobs, and
disaggregation.

`start()` returns an `EngineConfig`: the neutral fields (`model`,
`served_model_name`) apply to every modality, while token-pipeline
registration metadata (context length, KV block sizing, data-parallel
layout, disagg bootstrap) lives in the optional `llm=LlmRegistration(...)`
sub-record.

See `src/echo_backend/engine.py` for a minimal implementation. The
entry point in `src/echo_backend/main.py` is a thin wrapper -- two
imports and a one-line `run()` call:

```python
from dynamo.common.backend.run import run
from .engine import EchoLLMEngine

def main() -> None:
    run(EchoLLMEngine)
```

### Request / response contract

`generate()` receives a `GenerateRequest` (`TypedDict`) with:

- `token_ids` (required): prompt tokens from the Rust preprocessor.
- `sampling_options`, `stop_conditions`, `output_options`
  (optional dicts): engine-specific pass-through data.

and yields `GenerateChunk` dicts with:

- `token_ids` and `index` (required on every chunk; use `index=0` for
  single-choice responses).
- `finish_reason` (required on the final chunk); `completion_usage`
  (optional, aggregated by the frontend when present).

Finish reason normalisation (`"abort"` → `"cancelled"`, etc.) is
handled by the Rust layer -- emit whichever string matches your
engine's native finish reason.

## CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs two jobs on
every PR:

1. **pre-commit** — isort, black, flake8, ruff, codespell, mypy, and the
   standard hygiene hooks (mirrors the configuration in
   [`ai-dynamo/dynamo`](https://github.com/ai-dynamo/dynamo/blob/main/.pre-commit-config.yaml)).
2. **unit-tests** — `pytest` on Python 3.11 / 3.12.

The tests in `tests/test_engine.py` exercise `from_args`, `start`,
`generate` (echo correctness, `max_tokens` cap, cancellation,
stop-token filtering, empty prompt) and `cleanup` without spinning up a
`Worker` or the distributed runtime -- Dynamo's `Context` is replaced
by a deterministic stub.

`tests/test_integration.py` is an end-to-end example marked
`@pytest.mark.integration`: it sends a real HTTP request through the
frontend from `compose.yaml`. It's skipped unless `ECHO_BACKEND_E2E=1`,
so CI (which has no runtime) stays green. To run it locally, bring the
stack up (`docker compose up --build`) then:

```bash
ECHO_BACKEND_E2E=1 pytest -m integration -v
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0 -- see [LICENSE](LICENSE).
