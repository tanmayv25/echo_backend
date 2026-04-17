# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Echo backend engine.

An illustrative :class:`LLMEngine` subclass that echoes the request
``token_ids`` back to the caller ``repeat_count`` times. The output is
streamed token-by-token with a configurable per-token delay to emulate
a real inference engine.

This module is intentionally small -- it exists to demonstrate the
minimal surface required to implement a Dynamo backend using
``dynamo.common.backend``. Real backends replace the echo loop with
calls into an actual inference engine.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import AsyncGenerator

# `Context` is re-exported from dynamo._core today. If a public
# re-export lands in dynamo.common.backend, prefer that.
from dynamo._core import Context
from dynamo.common.backend import (
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
    WorkerConfig,
)

logger = logging.getLogger(__name__)

DEFAULT_REPEAT_COUNT = 3
DEFAULT_DELAY_SECONDS = 0.01
DEFAULT_CONTEXT_LENGTH = 2048


class EchoLLMEngine(LLMEngine):
    """Echoes the request ``token_ids`` ``repeat_count`` times.

    Sampling options are ignored; ``stop_conditions`` is read for
    ``max_tokens`` and ``stop_token_ids`` / ``stop_token_ids_hidden``
    (stop tokens are filtered out of the echo source so the frontend's
    stop-token detector does not truncate the stream on echoed
    chat-template control tokens).
    """

    def __init__(
        self,
        model_name: str = "echo-model",
        served_model_name: str | None = None,
        repeat_count: int = DEFAULT_REPEAT_COUNT,
        delay_seconds: float = DEFAULT_DELAY_SECONDS,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
    ):
        if not model_name:
            raise ValueError("model_name must be non-empty")
        if repeat_count < 1:
            raise ValueError("repeat_count must be >= 1")
        if delay_seconds < 0:
            raise ValueError("delay_seconds must be >= 0")
        if context_length < 1:
            raise ValueError("context_length must be >= 1")
        self.model_name = model_name
        self.served_model_name = served_model_name or model_name
        self.repeat_count = repeat_count
        self.delay_seconds = delay_seconds
        self.context_length = context_length

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple[EchoLLMEngine, WorkerConfig]:
        # Async to match LLMEngine.from_args; no awaits needed for argv parsing.
        parser = argparse.ArgumentParser(
            prog="echo-backend",
            description="Echo backend -- illustrative Dynamo worker",
        )
        # --- Engine knobs (consumed by EchoLLMEngine.__init__) ---
        parser.add_argument(
            "--model-name",
            default="echo-model",
            help="Model name advertised to the Dynamo frontend.",
        )
        parser.add_argument(
            "--served-model-name",
            default=None,
            help="Optional override for the served model name (defaults to --model-name).",
        )
        parser.add_argument(
            "--repeat-count",
            type=int,
            default=DEFAULT_REPEAT_COUNT,
            help="How many times to echo the input tokens.",
        )
        parser.add_argument(
            "--delay-seconds",
            type=float,
            default=DEFAULT_DELAY_SECONDS,
            help="Per-token streaming delay (seconds).",
        )
        parser.add_argument(
            "--context-length",
            type=int,
            default=DEFAULT_CONTEXT_LENGTH,
            help="Advertised context length.",
        )
        # --- Worker config (plumbed into WorkerConfig) ---
        parser.add_argument(
            "--namespace",
            default="dynamo",
            help="Dynamo namespace to register under.",
        )
        parser.add_argument(
            "--component",
            default="echo",
            help="Dynamo component name.",
        )
        parser.add_argument(
            "--endpoint",
            default="generate",
            help="Dynamo endpoint name.",
        )
        parser.add_argument(
            "--endpoint-types",
            default="chat,completions",
            help="Comma-separated endpoint types.",
        )
        # Mechanism selectors. Addresses come from ETCD_ENDPOINTS / NATS_SERVER.
        parser.add_argument(
            "--discovery-backend",
            default="etcd",
            help="KV-store backend for model discovery.",
        )
        parser.add_argument(
            "--request-plane",
            default="tcp",
            help="Transport for worker <-> frontend request streaming.",
        )
        parser.add_argument(
            "--event-plane",
            default="nats",
            help="Transport for Dynamo runtime events.",
        )
        args = parser.parse_args(argv)

        served_model_name = args.served_model_name or args.model_name
        engine = cls(
            model_name=args.model_name,
            served_model_name=served_model_name,
            repeat_count=args.repeat_count,
            delay_seconds=args.delay_seconds,
            context_length=args.context_length,
        )
        worker_config = WorkerConfig(
            namespace=args.namespace,
            component=args.component,
            endpoint=args.endpoint,
            model_name=args.model_name,
            served_model_name=served_model_name,
            endpoint_types=args.endpoint_types,
            discovery_backend=args.discovery_backend,
            request_plane=args.request_plane,
            event_plane=args.event_plane,
        )
        return engine, worker_config

    async def start(self) -> EngineConfig:
        logger.info(
            "Echo engine starting: model=%s repeat_count=%d delay=%.3fs",
            self.model_name,
            self.repeat_count,
            self.delay_seconds,
        )
        # KV / batching values below are placeholders. A real engine derives
        # them from GPU memory, model config, and scheduler policy.
        return EngineConfig(
            model=self.model_name,
            served_model_name=self.served_model_name,
            context_length=self.context_length,
            kv_cache_block_size=16,
            total_kv_blocks=1024,
            max_num_seqs=64,
            max_num_batched_tokens=self.context_length,
        )

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        prompt_tokens = list(request.get("token_ids", []))
        prompt_len = len(prompt_tokens)

        stop_conditions = request.get("stop_conditions") or {}
        requested_max = stop_conditions.get("max_tokens")
        # Strip stop tokens (e.g. Qwen's `<|im_end|>`) from the echo source
        # so the frontend's stop-token detector doesn't truncate the stream
        # on the first echoed control token. Dynamo currently passes these
        # under `stop_token_ids_hidden`.
        stop_token_ids = set(
            (stop_conditions.get("stop_token_ids") or [])
            + (stop_conditions.get("stop_token_ids_hidden") or [])
        )
        echo_source = [t for t in prompt_tokens if t not in stop_token_ids]

        echoed = echo_source * self.repeat_count
        total_available = len(echoed)
        if requested_max is not None and requested_max >= 0:
            max_new = min(int(requested_max), total_available)
        else:
            max_new = total_available

        def _usage(completion_tokens: int) -> dict[str, int]:
            return {
                "prompt_tokens": prompt_len,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_len + completion_tokens,
            }

        if max_new == 0:
            yield {
                "token_ids": [],
                "finish_reason": "stop",
                "completion_usage": _usage(0),
            }
            return

        finish_reason = "length" if max_new < total_available else "stop"

        for i in range(max_new):
            if context.is_stopped():
                yield {
                    "token_ids": [],
                    "finish_reason": "cancelled",
                    "completion_usage": _usage(i),
                }
                return

            if self.delay_seconds > 0:
                await asyncio.sleep(self.delay_seconds)

            chunk: GenerateChunk = {"token_ids": [echoed[i]]}
            if i == max_new - 1:
                chunk["finish_reason"] = finish_reason
                chunk["completion_usage"] = _usage(max_new)
            yield chunk

    async def cleanup(self) -> None:
        # Real engines release KV allocators, join worker threads, close
        # CUDA streams, etc. The echo engine holds no resources.
        logger.info("Echo engine shutdown complete")
