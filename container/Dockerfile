# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build the echo_backend image.
#
#   docker build -t echo-backend:dev -f container/Dockerfile .
#   docker build --build-arg DYNAMO_REF=main  -t echo-backend:dev -f container/Dockerfile .
#   docker build --build-arg DYNAMO_REF=1.1.0 -t echo-backend:dev -f container/Dockerfile .
#
# DYNAMO_REF selects how ai-dynamo is installed:
#   empty / unset             -> PyPI transitive (resolve from pyproject.toml). Fast.
#   v?[0-9]+.[0-9]+...        -> PyPI pinned release. Fast.
#   anything else (branch/SHA)-> git ref, compiled from source. Needs Rust.
# DYNAMO_REPO overrides the upstream repo URL for forks (default is the
# `ai-dynamo/dynamo` repo on GitHub).

ARG BASE_IMAGE=python
ARG BASE_IMAGE_TAG=3.11-slim

# ========== Builder ==========
FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} AS builder

ARG DYNAMO_REF=""
ARG DYNAMO_REPO=https://github.com/ai-dynamo/dynamo.git

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:/root/.cargo/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates clang cmake curl git \
        libclang-dev libssl-dev pkg-config protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv && pip install --upgrade pip

WORKDIR /build
COPY pyproject.toml README.md LICENSE /build/
COPY src /build/src
COPY scripts/install-dynamo.sh /build/scripts/install-dynamo.sh

# Install ai-dynamo per DYNAMO_REF (classification + git/PyPI dispatch is
# in the shared script, reused by CI), then install echo_backend itself.
RUN DYNAMO_REPO="${DYNAMO_REPO}" /build/scripts/install-dynamo.sh "${DYNAMO_REF}" \
    && pip install /build

# ========== Runtime ==========
FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

RUN useradd -m -u 1000 -g 0 dynamo
USER dynamo
WORKDIR /workspace

ENTRYPOINT ["echo-backend"]
CMD ["--help"]
