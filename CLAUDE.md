# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

discopt is a hybrid Mixed-Integer Nonlinear Programming (MINLP) solver combining a Rust backend (in-house primal/dual simplex, B&B tree management, FBBT/presolve, the POUNCE AD tape), a numpy relaxation layer, and Python orchestration. **JAX is not on the default solve path**, but it is a **core dependency** (`jax`/`jaxlib` in `[project] dependencies`, not an extra) and it is not confined to the optional subsystems. An ordinary nonlinear solve imports **zero** `jax` modules (measured) — *unless the model contains an expression with no POUNCE tape opcode*, in which case `build_evaluator` deliberately falls back to the legacy JAX evaluator and loads it. That fallback is on the **default** path, so "a default solve never imports JAX" is false as an unconditional claim: measured on a plain `Model.solve()` of `dm.norm(X, 2)` with a 3x3 `X` and no user `import jax`, `sys.modules` goes from 0 to 219 `jax` entries. `estimate.py` also uses `jax.jacobian` for exact sensitivity Jacobians. Do not plan work against "the JAX layer"; see the `_relax/` note under Architecture. This repository contains the testing and benchmarking framework that validates correctness and measures performance against solvers like BARON, Couenne, SCIP, and HiGHS.

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
     default-off until a corpus-wide differential panel passes — flag ON vs OFF
     over the in-repo corpus, requiring BOTH (1) *cert-clean*: `incorrect_count
     = 0`, no bound above its reference optimum, no certification regression (no
     `gap_certified=True` instance regresses to uncertified), objective drift
     within tolerance, and incumbents independently feasibility-verified
     (soundness is non-negotiable; never flip a flag whose run shows any false
     or looser bound), AND (2) *net-positive*: measurably helpful broadly (node
     count / wall / bound), not merely sound — the `DISCOPT_CUT_INHERIT` lesson
     (sound ≠ helpful; a cert-clean but neutral-or-harmful flag stays OFF, with
     the measurement recorded). A passing panel graduates the flag default-ON
     (keep the `=0` opt-out and the legacy path intact); the nightly panel
     remains the ongoing regression watch, not a graduation gate.
     (Policy updated 2026-07-17 by the owner — consecutive-nightly graduation
     was dropped in favor of the panel gate; first applied to the #309 flags.
     One passing graduation-gate run meeting both bars suffices — consecutive
     nightly runs are no longer required.)

   **A graduation attempt has three outcomes, not two (added 2026-09-19, #1345).**
   "A cert-clean but neutral-or-harmful flag stays OFF, with the measurement
   recorded" is where flags accumulate: it records an answer and gives it nowhere
   to go. Measured on `7b3a1e0a` by `python/tests/flag_audit_scan.py`: **138**
   distinct `DISCOPT_*` flags across 145 read sites — 47 default-OFF, 68
   default-ON, 19 selectors, 4 undecidable — and **45 of the default-OFF gates
   are solver math**, nearly every one already carrying its measurement in a
   docstring. (An earlier figure here, "86 flags / 19 defaulting to `\"0\"` / 14
   gating solver math", was **retracted 2026-09-22 by #1421**: it came from a scan
   that recognised only `environ.get(F, "0")` and was blind to the 46 `_env_flag(F,
   default=...)` sites, 23 of them default-OFF bound-changing math. Two instruments
   shared the blind spot and confirmed each other. Re-derive from the scan script;
   do not copy a flag count forward.) What §5 lacked was an exit. Every
   default-OFF gate over solver math is therefore in exactly one of three states,
   and a gate in none of them is a defect:
   - **Graduated** — the panel passed both bars. Flip the default, keep the `=0`
     opt-out and the legacy path intact.
   - **Retired** — the panel ran and did not pass, or nobody is willing to run it.
     Delete the flag **and the implementation behind it**, in one PR, recording the
     measurement that killed it (or that none was ever taken). Keep reusable
     pieces, regression fixtures and benchmark cases; the entry point goes.
     Retiring is not a failure — it is the measurement being acted on, which is the
     point of taking it.
   - **Kept as a documented opt-out** — the flag guards a route deliberately not
     the default (an alternative backend, an escape hatch, a debugging lever). Its
     docstring must say *why it is not the default and what would change that*, so
     the next reader cannot mistake it for a stalled graduation.

   A gate whose docstring promises a panel, with no panel and no owner, is the
   defect this clause names. The standing audit is
   `docs/dev/flag-retirement-audit.md`; a new default-OFF gate over solver math
   adds a row to it in the same PR that introduces the flag.

   **Out of scope** — these are not graduation candidates and the rule does not
   apply to them: a numeric tuning knob whose `0` is a *value* rather than an
   off-switch (`DISCOPT_HEUR_OFFSET`, `DISCOPT_ROOT_CUT_ROUNDS`); an opt-*out* for
   a shipped default (`DISCOPT_NATIVE_SPATIAL_KERNEL=0`,
   `DISCOPT_LP_MILP_BACKEND=rust`, `DISCOPT_NS_MARGIN=0`), which exists so a
   default can be A/B'd and is exactly right; and a switch over non-solver
   behaviour (`DISCOPT_EAGER_IMPORTS`, `DISCOPT_DISABLE_JAX_CACHE`,
   `DISCOPT_GAMS_NO_DAEMON`). Classify by how the flag is *consumed* and what it
   gates, not by its default being `"0"`.

## Measurement & instrumentation discipline

§4 says no fix ships on a hypothesis. In practice the failure is rarely a *wrong*
measurement — it is a measurement that **never happened** and reported success
anyway. Every rule here is from an instrument that silently measured nothing and
was believed.

6. **Prove the probe fired.** Every experimental script ends by printing an
   *executed-assertion count* (or comparison count) and exits non-zero when it is
   zero. A probe with `if x is None: continue` and no counter degrades to a no-op
   that prints "0 violations" and reads as a pass. Incidents: a probe traversed
   nothing because it used `Constraint.lhs`/`.expr` (the real attribute is
   `.body`); an attribution probe gated only the *root* dive, making its "neither"
   arm meaningless; a sparse-parity test that never asserted
   `scipy.sparse.issparse()` on the forced-sparse arm compared dense against dense
   and was believed.
7. **Never swallow an exception in an instrument.** A bare `except` turns "this
   path is broken" into "this path is fine". `copy.deepcopy(Model)` raises
   `TypeError: cannot pickle 'builtins.PyModelRepr'`; a bare `except` hid it and an
   entire fallback was an invisible no-op while reported as working. Let probes
   crash. This is §3 applied to the thing you are using to judge §3.
8. **Verify which code you actually loaded.** Before any measurement in a worktree
   or against a branch, assert both `module.__file__` *and* a marker string unique
   to the version under test (and, for a baseline run, assert that marker
   **absent**). A `pytest` run silently imported `discopt` from the main tree
   instead of the worktree and produced 19 bogus failures.
9. **Timing claims require a load gate, an interleaved control, and a spread.**
   Check `uptime`; run A/B interleaved, not sequentially; report a standard
   deviation. Two claims were published and retracted this way: "leftover model
   state slows relaxation ~40%" (true value +2%, pooled sd 4.55 — an unrelated
   pytest held ~87% CPU) and a `gastrans040` regression that was a timing artifact.
   **Check for stray load you created yourself**: one round was invalidated by
   three zombie probes at 99% CPU that survived a `pkill`.
10. **A long job needs incremental output, and "no output" is not "dead".** Print
    per-item progress with `python -u`/`flush=True` and never send stderr to
    `/dev/null`. A background sweep that only prints a summary at the end was twice
    declared dead while still running, and the second time a competing job was
    launched that then timed out on the contention. Confirm with `pgrep` and a
    growing log before concluding anything. For a step that may never return,
    `faulthandler.dump_traceback_later` is the tool — a timing wrapper that prints
    on return never fires for a call that does not return.
11. **When a measurement contradicts a claim you already published, retract it in
    writing** — in the PR, the issue, or the plan doc — before continuing. §4
    already requires recording falsifications; this extends it to your *own* prior
    statements in the same session.

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
- **Never write an issue-closing keyword near a negation.** GitHub matches the
  substring: "Does **NOT** close #863" auto-closed #863 from PR #868. Write
  "supersedes #863" / "leaves #863 open", and re-check issue state after merging
  any PR whose body names an issue number.
- **Look up an API before calling it; do not guess attribute names.** One `grep`
  costs seconds; guessing cost four failed runs in a single probe (`Model.variables`,
  `Model.from_nl`, `m._sense`, `m.objective_sense` — the real names are
  `model._variables` and the module-level `from_nl`).
- **Re-derive any order-of-magnitude figure before stating it.** A dense `A_eq` was
  reported as 91.5 **TB**; it is 91.5 **GB**. Numbers in a PR or issue are read as
  measurements.

### Working on an issue

Default to **finishing**. An issue is done when the work it asks for is complete —
not when a first increment lands. The lifecycle:

1. **Confirm it's still relevant; reframe if needed.** Many issues go stale or were
   opened mid-fix against assumptions that no longer hold. Before writing code,
   re-read the issue against the current tree; if its premise has shifted, restate
   the goal (a comment on the issue, or edit it) before proceeding. A wrong or stale
   framing wastes the whole effort.
2. **Plan the implementation.** State what "done" means for *this* issue and the
   steps to get there before building. Honor the entry-experiment discipline (§4):
   if the work rests on a hypothesis, run the falsifying experiment on **real corpus
   instances** first — a mechanism validated only on a synthetic proxy can be a
   no-op on the real class (the #727 RLT lesson: synthetic root-gain 0.68, real
   gain 0.0).
3. **Work to completion; avoid issue proliferation.** The goal is to complete the
   work the issue asks for, not to advance it and hand off. Prefer doing the whole
   job over splitting it. Splitting — filing a follow-up and closing the original —
   is the *exception*, taken only when the scope genuinely must expand beyond the
   issue; when you do split, the follow-up must be detailed enough to pick up
   independently (e.g. #114→#741, #572→#713).
4. **Work in a branch or worktree; use a PR to check CI.** Never on `main` (see the
   Workflow rules above); the PR is how CI validates the change. Use `Closes #N`
   only when the PR actually finishes the issue; use `Contributes to #N` for a
   partial or default-off increment.
5. **Close the loop with a succinct summary.** When the work is done, give a short
   summary of what changed and how it was verified, ending with an explicit
   statement of **whether the issue can be closed or not** — and if not, exactly
   what remains.
6. **Finish rather than document.** Writing an issue, a plan doc, or a follow-up is
   not progress on the fix and must never be substituted for it. If an
   investigation has produced three artifacts and no code change, that is the
   signal to stop analyzing and start building. Filing a follow-up is the §3
   exception, not the default deliverable.

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

`discopt` is a maturin/Rust build, and `discopt_benchmarks` does **not** depend on it —
installing only the benchmark package leaves you without a working solver. The canonical
source install (matches `CONTRIBUTING.md`):

```bash
pip install -e ".[dev,pounce,ipopt,highs]"        # discopt itself
cd crates/discopt-python && maturin develop && cd ../..   # build the Rust bindings
cd discopt_benchmarks && pip install -e ".[dev]" && cd ..  # benchmark harness (separate pkg)
```

Re-run `maturin develop` after any change under `crates/`; the Python tests import the
compiled extension, not the Rust source.

### Tests
```bash
pytest python/tests/ -v                                  # discopt tests
pytest discopt_benchmarks/tests/ -v                      # Benchmark suite
pytest discopt_benchmarks/tests/ -m smoke                # Quick CI smoke tests
pytest discopt_benchmarks/tests/ -m "not slow"           # Skip long tests
pytest discopt_benchmarks/tests/ -k test_correctness     # Single test file
pytest discopt_benchmarks/tests/ --cov=benchmarks --cov=utils  # With coverage (≥85% floor, restored by #87)
```

### Benchmarking
```bash
python discopt_benchmarks/run_benchmarks.py --suite smoke     # Quick sanity check
python discopt_benchmarks/run_benchmarks.py --suite phase1    # Phase 1 validation
python discopt_benchmarks/run_benchmarks.py --gate phase1     # Check phase gate criteria
python discopt_benchmarks/run_benchmarks.py --suite comparison --solvers discopt,baron
```

#### The 3-way head-to-head (discopt vs BARON vs SCIP) — run it THIS way

This is the standard pre-release comparison. It has been re-derived from scratch
several times and gotten wrong the same way each time, so the procedure is fixed
here. **BARON and SCIP do not go through the same harness.** There is no single
command that produces a valid three-way table.

**Step 1 — discopt and SCIP, via the benchmark runner.** SCIP is fully licensed
and reads `.nl` directly, so it runs in-harness:

```bash
python -u discopt_benchmarks/run_benchmarks.py --suite global50 \
    --solvers discopt,scip --report --output <out>.json
```

**Step 2 — BARON, via GAMS only.** `/Applications/AMPL/baron` is **DEMO-LICENSED
(10 variables, 10 constraints)** and silently refuses anything larger. The full
CMU floating-network license lives inside GAMS, which reads `.gms`, not `.nl`:

```bash
python -u -m discopt_benchmarks.scripts.global_opt_baron_vs_discopt \
    --time-limit 60 --instances "<comma-separated names>" --out-dir <dir>
```

That script fetches `minlplib.org/gms/<name>.gms` (the canonical GAMS source, so
there is no `.nl → .gms` conversion-fidelity risk), runs
`gams <name>.gms minlp=baron optcr=0 optca=1e-9 reslim=<T>`, parses the `.lst`,
and runs **discopt interleaved on the same instances** — so its discopt column is
the one to compare BARON against, not the column from step 1.

**The failure signature to watch for.** Demo-licensed BARON returns in ~0.03 s
with `nodes=0` and no objective. A panel run against it reports something like
"BARON 31/50" — which reads as a solver result but is just "19 instances exceeded
10 variables". The 31 successes are a size-biased subset, not a comparison. Before
believing any BARON number, confirm the license is not demo:

```bash
grep "Floating Network License" <run>.lst   # full license
# vs. "Sorry, a demo license is limited to 10 variables and 10 constraints"
```

Measured 2026-08-16 on the global50 panel at 60 s/instance: demo BARON solved
31/50; the same binary under GAMS with the full license solved `flay03m`
(26 vars, 24 cons) to optimality in 0.32 s after the demo had refused it outright.

Report **median time over solved instances** and **total wall over all
instances** alongside SGM — a solved-only statistic flatters whichever solver
times out most, since its slowest instances leave the population. Both columns
are in the runner's summary table.

**Benchmark instance corpus**: `~/Dropbox/projects/discopt-minlp-benchmark/` holds
the full MINLPLib snapshot — 1,610 `.nl` instances (`minlplib/nl/`), reference
optima/dual bounds (`minlplib.solu`), problem-type/size metadata
(`minlplib_types.csv`, `problem_sizes.csv`), curated problem lists by runtime
(`problems_{small,short,medium,long}.txt`), SCIP reference results
(`scip_join.csv`), and a standalone `benchmark.py`/`Makefile` harness with prior
results in `results/`. Use it to draw instances beyond the in-repo test corpus
(e.g. when a fix targets operators/structures the 66-file
`python/tests/data/minlplib_nl/` corpus doesn't exercise), and use `minlplib.solu`
as the oracle for correctness checks.

The same directory also holds `qplib/` — the full **QPLIB** library (453
quadratic instances, 390 nonconvex, no overlap with MINLPLib), read natively via
`discopt.interfaces.qplib` with no AMPL/GAMS conversion. Unlike `.solu`, QPLIB
ships reference solution *vectors*, so an incumbent can be feasibility-verified
directly. Select instances by filtering `qplib/qplib_manifest.csv` (gate on
`usable_oracle`; 421 of 453 qualify) — never by hardcoding names. See
`docs/dev/qplib-corpus.md`; the `.qplib` layout is conditional on the instance's
`probtype` code and misreading it produces a wrong model *without raising*, so
treat that doc's format notes as binding before touching the reader.

### Linting & Type Checking
```bash
ruff check python/
ruff format --check python/
mypy python/discopt/
```

## Architecture

- **`python/discopt/modeling/`** — Python modeling API with expression DAG system for MINLP formulation, supporting continuous/binary/integer variables and operator overloading that maps to Rust AST. Imported as `from discopt import Model` or `import discopt.modeling as dm`.
- **`python/discopt/_relax/`** — the relaxation layer: DAG compiler, McCormick/alphaBB envelopes, cutting planes, factorable reformulation, convexity detection, NLP evaluator, relaxation compiler. Named `_jax` until the JAX removal; it is **numpy**. Do not reason about "the JAX layer" from this directory's contents. (This bullet used to add "and JAX does not enter `sys.modules` during a solve" — that is the unconditional claim the Project Overview above explicitly labels **false**. An *ordinary* solve imports zero `jax` modules; a model with an expression lacking a POUNCE opcode hits the legacy fallback and loads it. Believe the Overview.)
- **`python/discopt/solvers/`** — external-solver wrappers (`highspy` is a core dependency: the pure LP/MILP route `lp_milp_highs.py` and the OA/GDP paths; cyipopt NLP wrapper). NOTE: the default MINLP per-node LP engine is the **in-house Rust simplex** (`MccormickLPRelaxer(backend="simplex")` → `crates/discopt-core/src/lp/simplex/`), not HiGHS — do not plan MINLP node work against a "HiGHS backend". Pure LP/MILP models classified at entry are routed to HiGHS with discopt-verified certificates (`docs/dev/lp-milp-highs-routing-plan.md`; opt-out `DISCOPT_LP_MILP_BACKEND=rust`).
- **`python/discopt/ml/`** — ML predictor embedding + trainable surrogates (inspired by
  OMLT). Optional dep: `pip install discopt[nn]`. `ls` the directory for the file map; what
  you cannot derive from it:
  - **Two regimes, not one.** *Frozen* — embed an already-trained NN/tree/ensemble as
    algebraic constraints (`network.py`, `formulations/`, `readers/` for ONNX, sklearn,
    torch `Sequential`); entry point `predictor.py:add_predictor()`, which auto-detects the
    predictor type. *Trainable* (`trainable.py`) — weights become decision `Variable`s
    emitting symbolic expressions, so a surrogate trains *simultaneously* with a physics
    model (a neural rate law inside a collocation DAE, à la Lueg et al. 2025).
    `TrainableNetwork.freeze()`/`from_definition()` bridge the two; `train()` is a thin
    local-NLP solve, not a deep-learning trainer.
  - **Only two of the four encodings are MIP.** `relu_bigm.py` and
    `formulations/tree_ensemble.py` emit binaries and big-M; `full_space.py` (smooth
    activation equalities) and `reduced_space.py` (nested expressions, one var per layer)
    emit **no binaries**. Do not assume embedding a network makes the model a MIP.
  - **`surrogate.py`'s `Surrogate` protocol is `runtime_checkable` and duck-typed**
    (`__call__(x)->expression`, `parameters`, `n_parameters`, `l2_penalty`,
    `initial_values`), so a GP mean, a soft tree or a fixed-structure symbolic formula
    plugs in with no framework change.
  - Named `discopt.nn` until #1219; `discopt.nn` remains a deprecation shim forwarding by
    object identity. `presolve.py`'s `NNPresolvePass` is informational v0 — it does not
    tighten anything.
- **`python/discopt/dae/`** — DAE/ODE discretization for dynamic optimization (no solver/DAG-compiler changes). `collocation.py` (`DAEBuilder` + `ContinuousSet`) transcribes ODEs/index-1 DAEs/2nd-order ODEs via orthogonal collocation on finite elements (Radau/Legendre); `finite_difference.py` (`FDBuilder`) and `mol.py` (`MOLBuilder`, method of lines for PDEs) are alternatives; `polynomials.py` holds the collocation matrices/roots. `fit.py` adds multi-experiment fitting glue (`Trajectory`, `fit_trajectories`, `TrajectoryFit`): one collocation block per trajectory on a shared model wired to one RHS, so a trainable surrogate's weights are shared and trained jointly.
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

## Claude Code skills (`python/discopt/skills/`)

The shareable slash commands and agent personas live **inside the Python
package**, so they ship with the wheel. They are not in `.claude/` in this repo —
that directory is a *destination*, written by the installer, not a source:

```bash
discopt install-skills                  # into ~/.claude/ (every project)
discopt install-skills --project-scope  # into ./.claude/
discopt install-skills --dev            # symlink, for `pip install -e`
discopt install-skills --force          # overwrite
```

- **`skills/commands/`** (8) — `/formulate`, `/debug`, `/diagnose`, `/reformulate`,
  `/explain-model`, `/convert`, `/estimate`, `/benchmark-report`
- **`skills/agents/`** (17) — `minlp-solver-expert`, `presolve-expert`,
  `convex-relaxation-expert`, `convexity-detection-expert`, `differentiability-expert`,
  `ipopt-expert`, `highs-expert`, `scip-expert`, `amp-expert`, `modeling-expert`,
  `heuristics-expert`, `estimation-expert`, `ml-embedding-expert`,
  `multiobjective-expert`, `robust-opt-expert`, `benchmarking-expert`,
  `llm-feature-expert`
- **`skills/__init__.py`** — `commands_dir()`, `agents_dir()`, `iter_commands()`,
  `iter_agents()` for programmatic discovery.

Editing a command or agent means editing the file under `python/discopt/skills/`;
editing a copy under `.claude/` changes only that one machine's install and is
lost on the next `install-skills` (use `--dev` to symlink instead).

Not in the bundle, and not slash commands: `discopt-dev` ships an **`adversary`
CLI verb** (`discopt-dev adversary` — the agent that files the adversarial issues),
alongside `search-arxiv`, `search-openalex`, `lit-scan` and `write-report`.
`claude-skills/README.md` also mentions a `/discoptbot` command and a
`/discopt-doe` skill from the external
[discopt-doe](https://github.com/jkitchin/discopt-doe) plugin; neither is present
in this repo, so treat that README's framing of them as unverified here.

## Key Constraints

- **Correctness is non-negotiable**: Every phase gate enforces `incorrect_count ≤ 0`. Never weaken this check.
- **Numerical tolerances**: abs=1e-6, rel=1e-4, integrality=1e-5, factorization=1e-12
  (the `numerical_tolerance` fixture in `discopt_benchmarks/tests/conftest.py` — there are
  several `conftest.py` files; this is the one).
- **Python is 3.12+** (`requires-python = ">=3.12"`); **ruff** line-length is 100 chars,
  `target-version = "py312"`, pinned to v0.14.6 across pre-commit and CI.
- **Coverage** must stay ≥85% (restored by #87 after the post-AMP-merge lowering).
- Tests have a 300-second default timeout (configurable in `pyproject.toml`).
- **`np.asarray()` on a scipy sparse matrix does not raise** — it returns a 0-d
  object array, so a missed call site feeds garbage onward silently. Use the
  `dense_Q()` / `dense_A()` helpers in `_relax/problem_classifier.py`, which raise on
  object/non-2-D input. One such call site was found inside
  `_quadratic_rows_solution_feasible` — a *feasibility check*, which it would have
  rendered meaningless. Related: `.size` on a sparse matrix is `nnz`, not rows×cols.
- **`INF` in the Rust LP layer is the sentinel `1e20`, not `f64::INFINITY`.** Test
  unboundedness on the *bound* (`hi >= INF`), never on a product like `a_ij * hi`:
  for `|a_ij| < 1` the product of an unbounded bound is an ordinary finite number,
  which both defeats the infinity check and destroys smaller terms in a running sum
  by cancellation. This produced invalid FBBT tightenings that cut the optimum out
  of the box and returned a certified-`optimal` false bound.

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
