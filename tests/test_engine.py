# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`echo_backend.EchoLLMEngine`.

These tests exercise the engine in isolation -- no :class:`Worker`,
no distributed runtime. Dynamo's :class:`Context` is replaced with a
stub so cancellation is deterministic.
"""

from __future__ import annotations

import pytest

from echo_backend import EchoLLMEngine


class _StubContext:
    def __init__(self, stopped: bool = False) -> None:
        self._stopped = stopped

    def is_stopped(self) -> bool:
        return self._stopped

    def stop(self) -> None:
        self._stopped = True


pytestmark = pytest.mark.unit


async def _collect(engine, request, context):
    return [chunk async for chunk in engine.generate(request, context)]


# --- from_args -----------------------------------------------------------


async def test_from_args_defaults():
    engine, cfg = await EchoLLMEngine.from_args([])
    assert engine.model_name == "echo-model"
    assert engine.repeat_count == 3
    assert cfg.namespace == "dynamo"
    assert cfg.served_model_name == "echo-model"


async def test_from_args_overrides():
    engine, cfg = await EchoLLMEngine.from_args(
        ["--repeat-count", "5", "--served-model-name", "public"]
    )
    assert engine.repeat_count == 5
    # --served-model-name reaches both the engine (for EngineConfig) and
    # the WorkerConfig, so registration metadata and advertised name agree.
    assert engine.served_model_name == "public"
    assert cfg.served_model_name == "public"


@pytest.mark.parametrize("bad_repeat", [0, -1])
def test_invalid_repeat_count_rejected(bad_repeat):
    with pytest.raises(ValueError, match="repeat_count"):
        EchoLLMEngine(repeat_count=bad_repeat)


def test_invalid_delay_rejected():
    with pytest.raises(ValueError, match="delay_seconds"):
        EchoLLMEngine(delay_seconds=-0.1)


def test_invalid_model_name_rejected():
    with pytest.raises(ValueError, match="model_name"):
        EchoLLMEngine(model_name="")


def test_invalid_context_length_rejected():
    with pytest.raises(ValueError, match="context_length"):
        EchoLLMEngine(context_length=0)


# --- start ---------------------------------------------------------------


async def test_start_returns_engine_config():
    # `served_model_name` flows from __init__ into EngineConfig so registration
    # metadata and the worker's advertised name stay in sync.
    engine = EchoLLMEngine(
        model_name="internal", served_model_name="public", context_length=1024
    )
    cfg = await engine.start()
    assert cfg.model == "internal"
    assert cfg.served_model_name == "public"
    assert cfg.context_length == 1024


# --- generate ------------------------------------------------------------


async def test_generate_echoes_prompt_repeat_count_times():
    engine = EchoLLMEngine(repeat_count=3, delay_seconds=0)
    chunks = await _collect(engine, {"token_ids": [10, 20, 30]}, _StubContext())

    emitted = [tok for chunk in chunks for tok in chunk["token_ids"]]
    assert emitted == [10, 20, 30] * 3
    assert chunks[-1]["finish_reason"] == "stop"
    assert chunks[-1]["completion_usage"]["completion_tokens"] == 9


async def test_generate_respects_max_tokens_cap():
    engine = EchoLLMEngine(repeat_count=4, delay_seconds=0)
    chunks = await _collect(
        engine,
        {"token_ids": [1, 2, 3], "stop_conditions": {"max_tokens": 5}},
        _StubContext(),
    )

    emitted = [tok for chunk in chunks for tok in chunk["token_ids"]]
    assert emitted == [1, 2, 3, 1, 2]
    assert chunks[-1]["finish_reason"] == "length"


async def test_generate_stops_on_cancellation():
    engine = EchoLLMEngine(repeat_count=5, delay_seconds=0)
    ctx = _StubContext()

    collected = []
    async for chunk in engine.generate({"token_ids": [7, 8]}, ctx):
        collected.append(chunk)
        if len(collected) == 2:
            ctx.stop()

    assert collected[-1]["finish_reason"] == "cancelled"
    assert collected[-1]["completion_usage"]["completion_tokens"] == 2


async def test_generate_filters_stop_tokens_from_echo():
    # Chat-template control tokens (e.g. Qwen's <|im_end|>) reach the
    # worker via `stop_conditions.stop_token_ids_hidden` and must not
    # be echoed back, or the frontend's stop-token detector will
    # truncate the stream on the first one.
    engine = EchoLLMEngine(repeat_count=2, delay_seconds=0)
    chunks = await _collect(
        engine,
        {
            "token_ids": [1, 2, 99, 3],
            "stop_conditions": {"stop_token_ids_hidden": [99]},
        },
        _StubContext(),
    )

    emitted = [tok for chunk in chunks for tok in chunk["token_ids"]]
    assert emitted == [1, 2, 3, 1, 2, 3]
    assert chunks[-1]["completion_usage"]["prompt_tokens"] == 4


async def test_generate_all_stop_tokens_yields_terminal_chunk():
    # If the prompt is entirely stop tokens, the filter leaves nothing
    # to echo and generate() must still emit a well-formed terminal chunk.
    engine = EchoLLMEngine(repeat_count=3, delay_seconds=0)
    chunks = await _collect(
        engine,
        {
            "token_ids": [99, 99, 99],
            "stop_conditions": {"stop_token_ids_hidden": [99]},
        },
        _StubContext(),
    )

    assert len(chunks) == 1
    assert chunks[0]["token_ids"] == []
    assert chunks[0]["finish_reason"] == "stop"
    assert chunks[0]["completion_usage"]["prompt_tokens"] == 3
    assert chunks[0]["completion_usage"]["completion_tokens"] == 0


async def test_generate_empty_prompt_yields_terminal_chunk():
    engine = EchoLLMEngine(repeat_count=3, delay_seconds=0)
    chunks = await _collect(engine, {"token_ids": []}, _StubContext())

    assert len(chunks) == 1
    assert chunks[0] == {
        "token_ids": [],
        "finish_reason": "stop",
        "completion_usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


async def test_cleanup_is_noop():
    await EchoLLMEngine().cleanup()
