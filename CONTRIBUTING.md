# Contributing to discopt

Thank you for your interest in contributing to discopt. This document explains how contributions are scoped and reviewed, how to set up your development environment, run tests, and submit changes.

All participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Before You Start: Propose First

Before taking on a contribution, please file a feature request / proposal
issue on the [issue tracker](https://github.com/jkitchin/discopt/issues) so
we can track and discuss it. This avoids duplicated or misdirected effort:
it lets us agree on whether the idea is a good fit, where it belongs (see
[Scope](#scope-core-vs-plugins) below), and what the best way to build it is
— before you invest time in code.

Small, well-formed projects are quite easy to implement, and if a proposal
is a good fit, the maintainer is happy to collaborate on it as time and
resources allow.

## Scope: Core vs. Plugins

There is a dividing line between core contributions and plugin / contrib
contributions:

- **Core contributions** are expected to be of general utility to MINLP
  modeling and solving — for example, the GDP, DAE, and NN embedding type
  modules.
- **Plugin / contrib contributions** are more domain-specific tools built on
  top of discopt — for example, the DOE package
  ([discopt-doe](https://github.com/jkitchin/discopt-doe)) or a
  microkinetic modeling language. These should be externally maintained as
  their own packages that depend on discopt: publish a namespace subpackage
  (`discopt.<name>`, see discopt-doe/discopt-mkm for the pattern) and, for CLI
  surface, register a subcommand in the `"discopt.cli"` entry-point group —
  the entry-point name is the subcommand and the value is a module exposing
  `add_subparser(subparsers)` and `run(args)` (protocol documented in
  `python/discopt/cli.py`).

The line is not always obvious, and we are open to discussion on where a
contribution might fall and what the best way to build it is. Raise this in
your proposal issue.

## LLM-Assisted Development

The maintainer almost exclusively uses Claude for development and for review
of issues and PRs. The project has a [CLAUDE.md](CLAUDE.md) that is used for
development; if you use Claude Code (or a similar tool), it will pick up the
project conventions from there.

LLM-generated code is welcome and will be considered, with the same bar as
any other contribution:

- It will be reviewed for consistency with the codebase and for correctness.
- It will be adversarially tested.
- Tests are expected for all contributions — LLM-generated or not. See
  [Running Tests](#running-tests) for the marker conventions and coverage
  requirements.

You remain responsible for the code you submit: review it yourself before
opening a PR, and be prepared to discuss and revise it.

## Development Setup

Prerequisites:

- Rust 1.84+
- Python 3.10+
- Ipopt (optional cyipopt fallback only; `brew install ipopt` on macOS)

```bash
# Clone the repo
git clone https://github.com/jkitchin/discopt.git
cd discopt

# Create a Python virtual environment
python -m venv .venv && source .venv/bin/activate

# Install Python dependencies. The `pounce` extra pulls in POUNCE
# (pure-Rust Ipopt port, the default single-solve NLP backend); the
# `ipopt` extra adds the optional cyipopt fallback.
pip install -e ".[dev,pounce,ipopt,highs]"

# Build Rust-Python bindings
cd crates/discopt-python && maturin develop && cd ../..
```

## Running Tests

The Python suite is tiered so you can pick the right cost/coverage point.
Prefer the Make targets — they pin flags consistent with CI.

```bash
# Rust tests
cargo test -p discopt-core

# PR-fast (matches CI; excludes slow/correctness/integration/manual/cyipopt
# markers, includes pr_correctness; target <10 min).
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
| `integration` | End-to-end workflows (CUTEst, GAMS) | nightly / manual |
| `amp_benchmark` | AMP Alpine, MINLPTests, or incidence benchmark coverage | manual |
| `requires_cyipopt` | Requires the optional cyipopt/Ipopt solver stack | manual |
| `requires_pounce` | Requires the optional POUNCE (pure-Rust Ipopt port) solver | manual |

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

1. File (or link to) a proposal issue describing the change (see
   [Before You Start](#before-you-start-propose-first)).
2. Create a feature branch from `main`.
3. Write tests for new functionality.
4. Ensure all tests pass and coverage stays >= 85%.
5. Run `ruff check` and `ruff format` before committing.
6. Keep commits focused; use descriptive commit messages.
7. Open a PR against `main` with a clear description that links the proposal issue.
8. Add a one-line entry to the `## [Unreleased]` section of `CHANGELOG.md` under the appropriate group (Added / Changed / Fixed / etc.).

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
