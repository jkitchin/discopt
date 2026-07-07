# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

discopt is a hybrid Mixed-Integer Nonlinear Programming (MINLP) solver combining a Rust backend (in-house primal/dual simplex, B&B tree management, FBBT/presolve), JAX (automatic differentiation, NLP relaxations, GPU acceleration), and Python orchestration. This repository contains the testing and benchmarking framework that validates correctness and measures performance against solvers like BARON, Couenne, SCIP, and HiGHS.

## Development Philosophy (read this first)

1. **Correctness before performance, always.** A global solver's product is its
   *certificate*. A change that makes anything faster but risks a false
   optimal/infeasible/bound is a regression, full stop. `incorrect_count ≤ 0` is a
   hard gate with zero slack; the certificate invariant (`bound ≤ incumbent` for
   min sense, dual bound never crossing the known oracle) must hold on every panel.
   Never weaken a validation, fallback, or safety guard to make a test or gate pass
   — if a goal can only be met that way, the goal loses; stop and surface it.
2. **General solutions, not single-problem solutions.** Fix the *class*, not the
   instance. Named instances (gear4, nvs17, casctanks, …) are gate probes only; a
   change whose benefit is confined to a named instance or benchmark is rejected.
   No hardcoded special cases keyed to problem names or shapes.
3. **Prefer the hard, right fix over the band-aid.** No silent approximations, no
   swallowed exceptions, no tolerance-tweaks to mask a bug, no dead flags. If the
   correct fix is a refusal (loud error) rather than a cheap approximation, refuse
   loudly. Root-cause first; a workaround ships only with an issue tracking the
   real fix and a comment linking it.
4. **Data-driven, evidence-based, testable hypotheses.** No fix ships on a
   hypothesis. Before building: state the hypothesis, cite the evidence (a
   measurement in this repo or the literature — see `docs/references.bib` and the
   Crucible knowledge base), and name the experiment + kill criterion that would
   falsify it. Run the entry experiment *before* writing the implementation. If a
   measurement contradicts a plan or an assumption, the measurement wins — record
   the falsification in the relevant plan doc (see `docs/dev/performance-plan.md`
   §6 for the house style) and re-scope before continuing.
5. **Two verification regimes for solver changes:**
   - *Bound-neutral* (refactors, caching, marshaling): assert `node_count` and
     certified `objective` are **exactly unchanged** on a certifying panel. Any
     drift — even an apparent improvement — means the change is wrong.
   - *Bound-changing* (relaxations, cuts, reductions): differential bound test
     (new bound ≥ old bound AND ≤ true box optimum on fixed boxes) plus
     feasible-point sampling (no valid point cut), behind a feature flag,
     default-off until green on consecutive nightly runs.

## Workflow

- **Feature branches + PRs, always.** Work happens on a feature branch
  (`git checkout -b <topic>` from `main`); open a PR so CI runs and the change is
  reviewable. Do not commit directly to `main`. Keep PRs scoped (one task/issue
  per PR) and name the task/issue ID in the title (e.g. `fix(correctness): C-16 …`,
  `cert:T1.2 …`).
- Every PR: `pytest -m smoke`, the adversarial suite
  (`pytest -m slow python/tests/test_adversarial_recent_fixes.py`), and
  `cargo test -p discopt-core` when Rust was touched. State in the PR description
  what was run and the result. New behavior requires a regression test that fails
  before the change and passes after.
- Benchmark/perf claims in a PR must include the measurement (suite, baseline,
  numbers), not adjectives.

## Canonical planning documents

- `docs/dev/certification-gap-plan.md` — the performance roadmap (per-node engine,
  branch-and-reduce, cuts, structure). Its §0 is a binding implementation contract;
  its §14 is the executable task list.
- `docs/dev/correctness-issues.md` — the prioritized correctness backlog (tracking
  issue #396). Loop-executable per its §0 protocol; fix P0/P1 items before
  performance work that touches the same layer.
- `docs/dev/performance-plan.md` — measured cost model (CC1–CC5) and the record of
  falsified hypotheses; treat its negative results as binding.
- `docs/design/relaxation-catalog.md` — what the relaxation layer has and its
  soundness rules; do not rebuild what it lists as done.

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
pytest discopt_benchmarks/tests/ --cov=benchmarks --cov=utils  # With coverage (≥65% floor; #87 tracks restoring 85%)
```

### Benchmarking
```bash
python discopt_benchmarks/run_benchmarks.py --suite smoke     # Quick sanity check
python discopt_benchmarks/run_benchmarks.py --suite phase1    # Phase 1 validation
python discopt_benchmarks/run_benchmarks.py --gate phase1     # Check phase gate criteria
python discopt_benchmarks/run_benchmarks.py --suite comparison --solvers discopt,baron
```

**Benchmark instance corpus**: `~/Dropbox/projects/discopt-minlp-benchmark/` holds
the full MINLPLib snapshot — ~4,800 `.nl` instances (`minlplib/nl/`), reference
optima/dual bounds (`minlplib.solu`), problem-type/size metadata
(`minlplib_types.csv`, `problem_sizes.csv`), curated problem lists by runtime
(`problems_{small,short,medium,long}.txt`), SCIP reference results
(`scip_join.csv`), and a standalone `benchmark.py`/`Makefile` harness with prior
results in `results/`. Use it to draw instances beyond the in-repo test corpus
(e.g. when a fix targets operators/structures the 61-file
`python/tests/data/minlplib_nl/` corpus doesn't exercise), and use `minlplib.solu`
as the oracle for correctness checks.

### Linting & Type Checking
```bash
ruff check python/
ruff format --check python/
mypy python/discopt/
```

## Architecture

- **`python/discopt/modeling/`** — Python modeling API with expression DAG system for MINLP formulation, supporting continuous/binary/integer variables and operator overloading that maps to Rust AST. Imported as `from discopt import Model` or `import discopt.modeling as dm`.
- **`python/discopt/_jax/`** — JAX DAG compiler, McCormick relaxations, NLP evaluator, relaxation compiler.
- **`python/discopt/solvers/`** — optional external-solver wrappers (highspy is used only on the OA/GDP paths; cyipopt NLP wrapper). NOTE: the default per-node LP engine is the **in-house Rust simplex** (`MccormickLPRelaxer(backend="simplex")` → `crates/discopt-core/src/lp/simplex/`), not HiGHS — do not plan work against a "HiGHS backend".
- **`python/discopt/nn/`** — Neural network / tree embedding module. Embeds trained feedforward NNs and decision-tree ensembles as algebraic constraints in MINLP models (inspired by OMLT). `network.py` defines `NetworkDefinition`/`DenseLayer`/`Activation`; `bounds.py` does interval-arithmetic bound propagation; `scaling.py` defines `OffsetScaling` (input/output affine scaling); `tree.py` + `formulations/tree_ensemble.py` do the decision-tree/ensemble MILP embedding (Mišić-style per-leaf encoding). `formulations/` has `FullSpaceFormulation` (smooth activations), `ReluBigMFormulation` (big-M MILP), and `ReducedSpaceFormulation` (lean full-space: one var per layer, affine+activation fused). `predictor.py`'s `add_predictor(model, inputs, predictor, ...)` is the convenience dispatcher that auto-detects the predictor type, formulates it, and links it to your input variables. `presolve.py` has `NNPresolvePass` (informational v0). `readers/` loads ONNX (`onnx_reader`), sklearn MLPs + trees + ensembles (`sklearn_reader`), and torch `Sequential` (`torch_reader`). Optional dep: `pip install discopt[nn]`.
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
- **Coverage** must stay ≥65% (temporary floor after AMP merge; #87 tracks restoring to 85%).
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
