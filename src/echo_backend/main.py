# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point for the echo backend.

Run with::

    python -m echo_backend --model-name echo-model --namespace dynamo

Or via the installed console script::

    echo-backend --model-name echo-model
"""

from dynamo.common.backend.run import run

from .engine import EchoLLMEngine


def main() -> None:
    run(EchoLLMEngine)


if __name__ == "__main__":
    main()
