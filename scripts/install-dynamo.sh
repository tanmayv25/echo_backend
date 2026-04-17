#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install ai-dynamo into the current Python environment based on a ref.
# Used by both container/Dockerfile and .github/workflows/ci.yml so the
# classification rules stay in one place.
#
#   scripts/install-dynamo.sh                # empty ref: no-op (PyPI via pyproject)
#   scripts/install-dynamo.sh 1.1.0          # PyPI pinned release
#   scripts/install-dynamo.sh main           # git source build (rustup + runtime compile)
#   scripts/install-dynamo.sh <branch|sha>   # git source build
#
# Classification:
#   empty                -> no-op; caller relies on pyproject.toml resolving from PyPI.
#   ^v?[0-9]+\.[0-9]+.*  -> PyPI `ai-dynamo==$REF` (leading v stripped).
#   anything else        -> git (installs ai-dynamo-runtime + ai-dynamo pinned to $REF).
#
# Prerequisites the caller provides before invoking with a git ref:
#   apt packages: build-essential clang cmake curl git libclang-dev
#                 libssl-dev pkg-config protobuf-compiler
# This script installs rustup itself when a git ref is given and cargo
# isn't already on PATH.
#
# Override `DYNAMO_REPO` to point at a fork (defaults to the upstream
# `ai-dynamo/dynamo` repo on GitHub).

set -euo pipefail

REF="${1:-}"
REPO="${DYNAMO_REPO:-https://github.com/ai-dynamo/dynamo.git}"

if [ -z "$REF" ]; then
    echo "ai-dynamo: PyPI transitive (pyproject.toml)"
    exit 0
fi

if echo "$REF" | grep -Eq '^v?[0-9]+\.[0-9]+'; then
    VERSION="${REF#v}"
    echo "ai-dynamo: PyPI ==${VERSION}"
    # `ai-dynamo` on PyPI declares `ai-dynamo-runtime` as a dep, so pip
    # pulls the matching runtime wheel transitively -- no explicit pin
    # needed here (unlike the git branch below, where runtime dev
    # versions aren't on PyPI).
    pip install "ai-dynamo==${VERSION}"
    exit 0
fi

echo "ai-dynamo: git ${REF}"
if ! command -v cargo >/dev/null 2>&1; then
    echo "Installing Rust toolchain via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal
    export PATH="${HOME}/.cargo/bin:${PATH}"
fi

pip install \
    "ai-dynamo-runtime @ git+${REPO}@${REF}#subdirectory=lib/bindings/python" \
    "ai-dynamo @ git+${REPO}@${REF}"
