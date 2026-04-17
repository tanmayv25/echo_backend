# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test fixtures.

If ``ai-dynamo`` is not installed (e.g. someone runs ``pytest`` before
``pip install -e .[dev]``), skip collecting tests that would otherwise
fail at import time, and emit a clear warning.
"""

import importlib.util

_HAS_DYNAMO = importlib.util.find_spec("dynamo") is not None


def pytest_ignore_collect(collection_path, config):
    """Skip collecting test files that import echo_backend.engine."""
    if _HAS_DYNAMO:
        return None
    if collection_path.suffix == ".py" and collection_path.name.startswith("test_"):
        return True
    return None


def pytest_configure(config):
    if not _HAS_DYNAMO:
        config.issue_config_time_warning(
            UserWarning(
                "ai-dynamo is not installed; echo_backend tests were skipped. "
                "Run `pip install -e .[dev]` first.",
            ),
            stacklevel=2,
        )
