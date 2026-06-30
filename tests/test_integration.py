# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration test for the echo backend.

Unlike ``test_engine.py`` (which drives :class:`EchoLLMEngine` in
isolation), this test exercises the *full* path: a running Dynamo
frontend, the distributed runtime (etcd + NATS), and a registered echo
worker. It sends a real OpenAI-style chat request over HTTP and checks
the worker echoed the prompt back.

It needs the stack from ``compose.yaml`` already up::

    docker compose up --build

then::

    ECHO_BACKEND_E2E=1 pytest -m integration -v

The test is **skipped by default** -- it only runs when ``ECHO_BACKEND_E2E``
is set, so a plain ``pytest`` run (and CI, which has no runtime) stays
green. Point ``ECHO_BACKEND_URL`` / ``ECHO_BACKEND_MODEL`` elsewhere if
your frontend isn't on the compose defaults.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest

pytestmark = pytest.mark.integration

FRONTEND_URL = os.environ.get("ECHO_BACKEND_URL", "http://localhost:8000")
MODEL = os.environ.get("ECHO_BACKEND_MODEL", "echo-model")

_E2E_ENABLED = os.environ.get("ECHO_BACKEND_E2E") not in (None, "", "0")

skip_unless_e2e = pytest.mark.skipif(
    not _E2E_ENABLED,
    reason="set ECHO_BACKEND_E2E=1 (and bring up `docker compose up`) to run",
)


def _post_chat(prompt: str, *, max_tokens: int) -> dict:
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"{FRONTEND_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as exc:  # frontend not reachable
        pytest.fail(
            f"could not reach Dynamo frontend at {FRONTEND_URL}: {exc}. "
            "Is `docker compose up` running?"
        )


@skip_unless_e2e
def test_chat_completion_echoes_prompt():
    body = _post_chat("hello echo backend", max_tokens=64)

    assert body["model"]
    choices = body["choices"]
    assert len(choices) == 1

    choice = choices[0]
    content = choice["message"]["content"]
    # The worker echoes the (tokenized) prompt back, so the response must be
    # non-empty and report completion tokens. We assert structure rather than
    # exact text, since the surface form depends on the tokenizer round-trip.
    assert content, "expected a non-empty echoed completion"
    assert choice["finish_reason"] in {"stop", "length"}

    usage = body["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


@skip_unless_e2e
def test_chat_completion_respects_max_tokens():
    body = _post_chat("count the tokens please", max_tokens=4)
    usage = body["usage"]
    assert usage["completion_tokens"] <= 4
