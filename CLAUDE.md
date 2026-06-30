# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install with dev extras (required before running tests)
python -m pip install -e ".[dev]"
pre-commit install

# Run all tests
pytest -v

# Run a single test
pytest tests/test_engine.py::test_generate_echoes_prompt_repeat_count_times -v

# Run only unit or integration tests
pytest -m unit
pytest -m integration

# Lint (all hooks mirror ai-dynamo/dynamo: isort, black, flake8, ruff, codespell, mypy)
pre-commit run --all-files

# Run the backend locally (requires etcd + nats — see Dynamo quickstart)
python -m echo_backend --model-name echo-model --namespace dynamo \
    --repeat-count 3 --delay-seconds 0.02

# Or run the whole stack (etcd + NATS + Dynamo frontend + worker) in containers:
docker compose up --build                         # dynamo main, built from source
DYNAMO_REF=1.3.0 docker compose up --build        # PyPI pinned (fast, no Rust)
DYNAMO_REF= docker compose up --build             # PyPI transitive via pyproject
```

If `ai-dynamo` is not installed, `tests/conftest.py` skips collection rather than hard-failing — a clean `pytest` run with zero tests usually means you forgot `pip install -e ".[dev]"`.

CI pins the `ai-dynamo` ref it tests against in `.github/dynamo-ref.txt` (default `main`). A PR that depends on an unreleased upstream change should edit that file; revert once upstream merges. Both CI and `container/Dockerfile` dispatch to `scripts/install-dynamo.sh`, which classifies the ref (empty → PyPI transitive, version-like → PyPI pinned, else → git source with Rust toolchain).

## Architecture

This repo is an illustrative reference backend, not a production engine. The entire purpose is to demonstrate the **minimum surface** required to implement a Dynamo backend on top of [`dynamo.common.backend`](https://github.com/ai-dynamo/dynamo/tree/main/components/src/dynamo/common/backend). Real backends (vllm, trtllm, sglang) are expected to follow the same repo pattern.

The contract is a single class, `EchoLLMEngine` (`src/echo_backend/engine.py`), subclassing `dynamo.common.backend.LLMEngine`. It implements exactly four methods:

- `from_args(argv)` — parse CLI args, return `(engine, WorkerConfig)`.
- `start(worker_id)` — return an `EngineConfig` with registration metadata. Neutral fields (`model`, `served_model_name`) sit on `EngineConfig`; token-pipeline metadata (context length, KV block sizing, DP layout, disagg bootstrap) goes in the optional `llm=LlmRegistration(...)` sub-record. `worker_id` is a runtime-allocated, cluster-unique id; ignore it unless the engine needs a per-worker key.
- `generate(request, context)` — async generator yielding `GenerateChunk` dicts. Every chunk must carry `token_ids` and `index` (0 for single-choice); the final chunk must additionally carry `finish_reason` (and optionally `completion_usage`). Check `context.is_stopped()` inside the loop to honour cancellation; emit `finish_reason="cancelled"` on early termination. `finish_reason` normalisation (e.g. `"abort"` → `"cancelled"`) is handled downstream in the Rust layer, so emit whatever matches your engine's native reason.
- `cleanup()` — release resources.

`abort(context)` is optional; override only when the engine needs to release scheduler slots / KV blocks on cancellation. `BaseEngine` (the modality-agnostic parent of `LLMEngine`) exposes more optional hooks — `health_check_payload()`, `kv_event_sources()`, `register_prometheus()`, `is_quiescent()`, etc. — all defaulting to no-ops. The echo engine leaves them alone; `dynamo.common.backend.sample_engine.SampleLLMEngine` upstream is the richer reference that opts into KV events, metrics, logprobs, and disaggregation.

The entry point (`src/echo_backend/main.py`) is a thin wrapper: it hands the engine class to `dynamo.common.backend.run.run()`, which owns worker registration and the distributed runtime. Backend authors should not re-implement any of that glue here.

### Testing approach

`tests/test_engine.py` exercises the engine **without** spinning up a `Worker` or the distributed runtime. The Dynamo `Context` is replaced with `_StubContext` so cancellation is deterministic.

`tests/test_integration.py` is the end-to-end counterpart: it's marked `@pytest.mark.integration` and sends a real HTTP request through the Dynamo frontend stood up by `compose.yaml`. It's **skipped unless `ECHO_BACKEND_E2E=1`** (so plain `pytest` and CI stay green with no runtime). Bring the stack up first (`docker compose up --build`), then `ECHO_BACKEND_E2E=1 pytest -m integration -v`. Use it as the template when adding integration tests that boot a `Worker`.

### Keeping the engine thin

Logic shared across backends belongs in `dynamo.common` upstream, not in this repo. If you find yourself adding generic glue here, push it upstream instead.

## Conventions

- Every new file needs an SPDX header:
  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0
  ```
- Sign off commits (`git commit -s`) — DCO, matching `ai-dynamo/dynamo`.
- Linter / formatter config in `pyproject.toml` intentionally mirrors `ai-dynamo/dynamo`; keep it aligned when upstream changes.
- Supported Python: 3.11 / 3.12 (CI matrix). 3.10 support will return once `ai-dynamo` stops importing `typing.Required` unguarded.
