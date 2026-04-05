# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

discopt is a hybrid Mixed-Integer Nonlinear Programming (MINLP) solver combining a Rust backend (LP solving, B&B tree management), JAX (automatic differentiation, NLP relaxations, GPU acceleration), and Python orchestration. This repository contains the testing and benchmarking framework that validates correctness and measures performance against solvers like BARON, Couenne, SCIP, and HiGHS.

## Commands

### Install
```bash
cd discopt_benchmarks && pip install -e ".[dev]"
```

### Tests
```bash
pytest python/tests/ -v                                  # discopt tests
pytest discopt_benchmarks/tests/ -v                      # Benchmark suite
pytest discopt_benchmarks/tests/ -m smoke                # Quick CI smoke tests
pytest discopt_benchmarks/tests/ -m "not slow"           # Skip long tests
pytest discopt_benchmarks/tests/ -k test_correctness     # Single test file
pytest discopt_benchmarks/tests/ --cov=benchmarks --cov=utils  # With coverage (≥85% required)
```

### Benchmarking
```bash
python discopt_benchmarks/run_benchmarks.py --suite smoke     # Quick sanity check
python discopt_benchmarks/run_benchmarks.py --suite phase1    # Phase 1 validation
python discopt_benchmarks/run_benchmarks.py --gate phase1     # Check phase gate criteria
python discopt_benchmarks/run_benchmarks.py --suite comparison --solvers discopt,baron
```

### Linting & Type Checking
```bash
ruff check python/
ruff format --check python/
mypy python/discopt/
```

## Architecture

- **`python/discopt/modeling/`** — Python modeling API with expression DAG system for MINLP formulation, supporting continuous/binary/integer variables and operator overloading that maps to Rust AST. Imported as `from discopt import Model` or `import discopt.modeling as dm`.
- **`python/discopt/_jax/`** — JAX DAG compiler, McCormick relaxations, NLP evaluator, relaxation compiler.
- **`python/discopt/solvers/`** — HiGHS LP wrapper, cyipopt NLP wrapper.
- **`python/discopt/nn/`** — Neural network embedding module. Embeds trained feedforward NNs as algebraic constraints in MINLP models (inspired by OMLT). `network.py` defines `NetworkDefinition`/`DenseLayer`/`Activation`; `bounds.py` does interval arithmetic propagation; `formulations/` has `FullSpaceFormulation` (smooth activations), `ReluBigMFormulation` (big-M MILP), `ReducedSpaceFormulation` (nested expressions); `readers/onnx_reader.py` loads ONNX models. Optional dep: `pip install discopt[nn]`.
- **`python/discopt/solver.py`** — Solver orchestrator: end-to-end `Model.solve()` via B&B.
- **`crates/discopt-core/`** — Rust: Expression IR, B&B tree, .nl parser, FBBT/presolve.
- **`crates/discopt-python/`** — Rust: PyO3 bindings with zero-copy numpy.
- **`discopt_benchmarks/`** — Benchmark orchestration, phase gate criteria, performance testing.
  - **`benchmarks/`** — `runner.py` loads instances, `metrics.py` computes metrics.
  - **`tests/`** — Pytest suite with markers: `smoke`, `correctness`, `regression`, etc.
  - **`config/benchmarks.toml`** — Single source of truth for suites, gates, solver configs.
  - **`utils/`** — Statistical utilities, profiles, report generation.

## Documentation (Jupyter Book)

The `docs/` directory contains a Jupyter Book site built with `jupyter-book build docs/`.

- **Config**: `docs/_config.yml`, `docs/_toc.yml`
- **Notebooks**: `docs/notebooks/` (single source of truth for all notebooks)
- **Bibliography**: `docs/references.bib` (BibTeX entries), `docs/references.md` (rendered bibliography page)
- **Landing page**: `docs/intro.md`

All notebooks live in `docs/notebooks/` and should always include relevant `{cite:p}` / `{cite:t}` MyST citations (keys from `docs/references.bib`). There is no separate `notebooks/` directory.

**When adding a new notebook**, you must:
1. Create the notebook in `docs/notebooks/`
2. Add `{cite:p}` / `{cite:t}` MyST citations to relevant markdown cells
3. Add any new BibTeX entries to `docs/references.bib`
4. Add the notebook to `docs/_toc.yml` under the appropriate `parts` section
5. Rebuild with `jupyter-book build docs/` and verify zero warnings

## LLM Integration (`python/discopt/llm/`)

Optional LLM-powered features using litellm as a universal adapter (100+ providers). Install with `pip install discopt[llm]`.

- **`llm/__init__.py`** — `is_available()`, `get_completion()` convenience wrapper
- **`llm/provider.py`** — Thin litellm wrapper; model resolution: explicit `model=` > `DISCOPT_LLM_MODEL` env var > default `anthropic/claude-sonnet-4-20250514`
- **`llm/serializer.py`** — Serialize Model/SolveResult to structured text for LLM context
- **`llm/prompts.py`** — All prompt templates (explain, formulate, diagnose, teach, debug)
- **`llm/safety.py`** — Output validation, bounds clamping, name sanitization
- **`llm/tools.py`** — OpenAI-format tool definitions + `ModelBuilder` for structured `from_description()`
- **`llm/advisor.py`** — Rule-based + LLM-augmented solver parameter suggestions, pre-solve analysis
- **`llm/commentary.py`** — `SolveCommentator` for streaming B&B commentary
- **`llm/diagnosis.py`** — Infeasibility diagnosis, convergence analysis, limit diagnosis
- **`llm/chat.py`** — `ChatSession` for conversational model building (`discopt.chat()`)
- **`llm/reformulation.py`** — Auto-reformulation detection (big-M, weak bounds, symmetry, bilinear)

**Safety invariant**: LLM outputs never affect solver math. Formulations pass `validate()`. Explanations are sanitized. Graceful degradation when litellm is unavailable.

**Claude Code skill files** in `.claude/commands/`: `/formulate`, `/diagnose`, `/reformulate`, `/explain-model`, `/convert`, `/benchmark-report`.

## Key Constraints

- **Correctness is non-negotiable**: Every phase gate enforces `incorrect_count ≤ 0`. Never weaken this check.
- **Numerical tolerances**: abs=1e-6, rel=1e-4, integrality=1e-5, factorization=1e-12 (defined in `conftest.py`).
- **ruff** line-length is 100 chars, targeting Python 3.10+. Pinned to v0.14.6 across pre-commit and CI.
- **Coverage** must stay ≥85%.
- Tests have a 300-second default timeout (configurable in `pyproject.toml`).

<!-- crucible-project -->
## Crucible Knowledge Base

This project has a Crucible knowledge base in `.crucible/`.
Use the `crucible` CLI to ingest sources, search, and maintain the wiki.

Layout: `.crucible/sources/` (primary sources), `.crucible/wiki/` (distilled articles),
`.crucible/crucible.db` (graph database).

Conventions: org-mode with scimax, org-ref citations, narrative prose.
The LLM maintains the wiki; manual edits are the exception.
Run `crucible help all` for the full CLI reference.
<!-- crucible-project -->
