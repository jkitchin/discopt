# Contributing to discopt

Thank you for your interest in contributing to discopt. This document explains how to set up your development environment, run tests, and submit changes.

## Development Setup

Prerequisites:

- Rust 1.84+
- Python 3.10+
- Ipopt (`brew install ipopt` on macOS)

```bash
# Clone the repo
git clone https://github.com/jkitchin/discopt.git
cd discopt

# Clone ripopt (Rust IPM dependency, path dependency at ../ripopt)
git clone <ripopt-repo-url> ../ripopt

# Create a Python virtual environment
python -m venv .venv && source .venv/bin/activate

# Install Python dependencies
pip install -e ".[dev,ipopt,highs]"

# Build Rust-Python bindings
cd crates/discopt-python && maturin develop && cd ../..
```

## Running Tests

The Python suite is tiered so you can pick the right cost/coverage point.
Prefer the Make targets — they pin flags consistent with CI.

```bash
# Rust tests
cargo test -p discopt-core

# PR-fast (matches the python-fast CI job; ~5 min). What CI runs.
make test

# Dev inner loop: unit + smoke markers only (~60 s).
make test-quick

# Subject-area slices (PR-fast filter applied within the slice).
make test-modeling   make test-solvers   make test-amp
make test-nn         make test-convexity make test-jax    make test-llm

# Long tail: only slow-marked tests.
make test-slow

# Known-optima validation (heavy).
make test-correctness

# Everything (slow + correctness + every marker).
make test-all

# Coverage (>=85% required); add to any pytest invocation.
pytest python/tests/ --cov=discopt
```

### Marker conventions

| Marker | Meaning | Runs in PR? |
|---|---|---|
| *(unmarked)* | Default solver/feature tests; <3 s each | yes |
| `unit` | Pure logic; no solver, no JAX trace beyond cached; <0.1 s | yes |
| `smoke` | One solve per code path on a tiny instance; <1 s | yes |
| `slow` | Backend cross-product, mid-size instances, ML training | nightly |
| `correctness` | Known-optima validation; usually also `slow` | nightly / pre-release |
| `pr_correctness` | Curated 5-instance correctness subset; <30 s total | yes |
| `integration` | End-to-end workflows (DOE, discrimination, CUTEst) | nightly / manual |

When adding a test, default to no marker for normal feature tests; add
`slow` if it routinely costs more than ~3 s, and `unit` or `smoke` if it
fits the budgets above. Never weaken `correctness` checks.

## Code Style

- Python: ruff v0.14.6 (pinned), line-length 100, target Python 3.10+
- Rust: standard rustfmt
- Type checking: mypy

```bash
ruff check python/
ruff format --check python/
mypy python/discopt/
cargo fmt --check
```

Pre-commit hooks are configured. Install with:

```bash
pip install pre-commit
pre-commit install
```

## Pull Request Process

1. Create a feature branch from `main`.
2. Write tests for new functionality.
3. Ensure all tests pass and coverage stays >= 85%.
4. Run `ruff check` and `ruff format` before committing.
5. Keep commits focused; use descriptive commit messages.
6. Open a PR against `main` with a clear description.
7. Add a one-line entry to the `## [Unreleased]` section of `CHANGELOG.md` under the appropriate group (Added / Changed / Fixed / etc.).

## Releasing

Releases are cut by following [`RELEASE.md`](RELEASE.md), which is the
authoritative checklist for tests, documentation, manuscript, changelog,
version bump, tagging, and PyPI publication. Tagging `vX.Y.Z` triggers
`.github/workflows/release.yml`, which builds wheels and publishes to PyPI.

## Project Structure

- `crates/discopt-core/` -- Rust solver engine (expression IR, B&B tree, presolve)
- `crates/discopt-python/` -- PyO3 bindings
- `python/discopt/` -- Python package (modeling API, JAX layer, solver orchestrator)
- `python/tests/` -- Python test suite
- `docs/` -- Jupyter Book documentation (notebooks live in `docs/notebooks/`)

## Reporting Issues

Use the GitHub issue tracker: https://github.com/jkitchin/discopt/issues
