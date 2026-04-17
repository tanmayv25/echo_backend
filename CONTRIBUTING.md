<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Contributing

Thank you for your interest in `echo_backend`. This repository is an
illustrative example, not a production backend, so the bar for
contributions is simple: keep it small, keep it readable, and keep it
aligned with the coding style of
[`ai-dynamo/dynamo`](https://github.com/ai-dynamo/dynamo).

## Developer setup

```bash
python -m pip install -e ".[dev]"
pre-commit install
```

`pre-commit install` wires the hooks so they run automatically on
`git commit`. You can also run them on demand:

```bash
pre-commit run --all-files
```

## Running tests

```bash
pytest -v
```

The unit tests do not require a running Dynamo runtime. If you add
integration tests that spin up `Worker`, mark them with
`@pytest.mark.integration` so CI can gate them appropriately.

## Testing against a specific `ai-dynamo` ref

CI builds `echo_backend` against the ref pinned in
[`.github/dynamo-ref.txt`](.github/dynamo-ref.txt) -- `main` by default.
If your PR depends on an unreleased change in `ai-dynamo`, edit that
file to point at the needed branch, commit, or PyPI version (the same
classification rules as [`scripts/install-dynamo.sh`](scripts/install-dynamo.sh)
apply: empty → PyPI transitive, `1.2.3` → PyPI pinned, anything else →
git ref). Revert it in the same PR once the upstream change merges.

To smoke-test against a different ref ad-hoc, trigger the `CI` workflow
via `Actions → Run workflow` and supply `dynamo_ref`.

## Pull requests

- Sign off your commits (`git commit -s`) to comply with the
  [Developer Certificate of Origin](https://developercertificate.org/),
  matching the `ai-dynamo/dynamo` convention.
- Include SPDX headers on new files:
  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0
  ```
- Keep the engine implementation thin — if a piece of logic is shared
  across backends, it likely belongs in `dynamo.common`, not here.

## Reporting issues

Use GitHub issues on this repository. For issues with the core
`dynamo.common.backend` interface itself, please file against
[`ai-dynamo/dynamo`](https://github.com/ai-dynamo/dynamo/issues).
