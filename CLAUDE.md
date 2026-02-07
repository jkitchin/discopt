# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JaxMINLP is a hybrid Mixed-Integer Nonlinear Programming (MINLP) solver combining a Rust backend (LP solving, B&B tree management), JAX (automatic differentiation, NLP relaxations, GPU acceleration), and Python orchestration. This repository contains the testing and benchmarking framework that validates correctness and measures performance against solvers like BARON, Couenne, SCIP, and HiGHS.

## Commands

### Install
```bash
cd jaxminlp_benchmarks && pip install -e ".[dev]"
```

### Tests
```bash
pytest jaxminlp_benchmarks/tests/ -v                    # Full suite
pytest jaxminlp_benchmarks/tests/ -m smoke               # Quick CI smoke tests
pytest jaxminlp_benchmarks/tests/ -m "not slow"          # Skip long tests
pytest jaxminlp_benchmarks/tests/ -k test_correctness     # Single test file
pytest jaxminlp_benchmarks/tests/ --cov=benchmarks --cov=utils  # With coverage (≥85% required)
```

### Benchmarking
```bash
python jaxminlp_benchmarks/run_benchmarks.py --suite smoke     # Quick sanity check
python jaxminlp_benchmarks/run_benchmarks.py --suite phase1    # Phase 1 validation
python jaxminlp_benchmarks/run_benchmarks.py --gate phase1     # Check phase gate criteria
python jaxminlp_benchmarks/run_benchmarks.py --suite comparison --solvers jaxminlp,baron
```

### Linting & Type Checking
```bash
ruff check jaxminlp_benchmarks/benchmarks jaxminlp_benchmarks/utils jaxminlp_benchmarks/jaxminlp_api
mypy jaxminlp_benchmarks/benchmarks jaxminlp_benchmarks/utils jaxminlp_benchmarks/jaxminlp_api
```

## Architecture

All framework code lives under `jaxminlp_benchmarks/`:

- **`benchmarks/`** — Benchmark orchestration: `runner.py` loads instances (MINLPLib, Netlib, CUTEst, SuiteSparse), launches solvers with time/memory limits, `metrics.py` computes shifted geometric means, Dolan-Moré performance profiles, root gap analysis, GPU speedup, and regression detection.
- **`tests/`** — Pytest suite with markers: `smoke`, `unit`, `integration`, `correctness`, `regression`, `property`, `adversarial`, `slow`, `gpu`. GPU tests auto-skip when unavailable. `test_correctness.py` validates against 25+ known MINLPLib optima (zero incorrect results is a hard requirement at every phase gate).
- **`config/benchmarks.toml`** — Single source of truth for benchmark suites, phase gate criteria, solver configurations, and performance targets. Phase gates (1–4) define automated go/no-go criteria.
- **`jaxminlp_api/`** — Python modeling API with expression DAG system for MINLP formulation, supporting continuous/binary/integer variables and operator overloading that maps to Rust AST.
- **`utils/`** — Statistical utilities (`statistics.py`), Dolan-Moré profiles (`profiles.py`), and markdown report generation (`reporting.py`).
- **`agents/`** — Automated literature review agent monitoring arXiv and optimization journals.

## Key Constraints

- **Correctness is non-negotiable**: Every phase gate enforces `incorrect_count ≤ 0`. Never weaken this check.
- **Numerical tolerances**: abs=1e-6, rel=1e-4, integrality=1e-5, factorization=1e-12 (defined in `conftest.py`).
- **mypy strict mode** is enabled; all code must have type annotations.
- **ruff** line-length is 100 chars, targeting Python 3.10+.
- **Coverage** must stay ≥85%.
- Tests have a 300-second default timeout (configurable in `pyproject.toml`).

## Implementation Status

**DO NOT begin implementation or ask about starting implementation until the user explicitly says it is time to start.** The development plan for `discopt` has been organized into tasks (see task list) but no code should be written, no files created, and no build systems set up until given explicit go-ahead. This applies to all phases and all weeks of the plan.
