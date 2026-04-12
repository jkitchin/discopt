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

```bash
# Rust tests
cargo test -p discopt-core

# Python tests (JAX requires these env vars on macOS)
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 pytest python/tests/ -v

# Quick smoke tests
pytest python/tests/ -m smoke

# Skip slow tests
pytest python/tests/ -m "not slow"

# With coverage (>=85% required)
pytest python/tests/ --cov=discopt
```

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
