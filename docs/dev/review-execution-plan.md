# Review Execution Plan — Priorities & Opus Loop Protocol

**Date:** 2026-07-03
**Status:** master plan for resolving the findings in the `docs/dev/*-review.md`
series (19 module reviews). This document is the entry point for the fix loop; it
defines **what order to fix in**, **how to fix each finding**, and **how to record
completion** so a later reader knows exactly what is done.

> **How to use this in a loop.** Each iteration picks the next unresolved finding
> (respecting the tier order and the shared-root dependencies in §1), runs the
> five-step protocol in §3, and updates the status markers in §4. When a module's
> findings are all resolved, check its box in GitHub issue #413.

---

## 0. The reviews and where the findings live

Nineteen review documents under `docs/dev/`, each ending in an "Implementation
plan" section with per-finding IDs, repros, and acceptance criteria:

`dae`, `modeling`, `mo`, `gp`, `ro`, `tutor`, `export`, `llm`, `validation`,
`solver-core-extraction`, `nn`, `mpec-estimate-pooling-opf`,
`tightening-conflict-warmstart-iis`, `doe`, `cli-daemon-gamslink-benchmarks`,
`infra-interop`, `decomposition`, `_numpy` McCormick (this file references its ID
prefix `NM-` once landed).

Every finding has a stable ID (e.g. `CORE-1`, `NN-1`, `TG-1`, `RO-2`, `EX-1`,
`LLM-1`). Fix and track by ID.

---

## 1. Cross-cutting shared roots — FIX THESE FIRST

Three root causes each produce findings in multiple modules. **Fixing the root once
closes or de-risks many findings; patching each consumer separately invites the next
instance.** These come before the per-module P0s because they *are* several of the
per-module P0s.

### X-1 — Builder-resident-rows blind spot (5 layers, 2 of them P0)
Fast-API constraint rows (`Model.constraint(...)` linear fast path,
`add_linear_constraints`) live in `model._builder_linear_blocks` / the Rust builder,
**not** in `model._constraints`. Consumers that read only `_constraints` silently
see a *subset* of the model:
- **gp GP-1** (P0) — auto-GP path solves without the fast-path rows → wrong optimum.
- **validation VAL-1** (P0) — examiner certifies a point that violates a fast-path row.
- **export EX-2** (P0) — MPS/LP/GAMS export an empty/zero model.
- **modeling M12** — `num_constraints`/`summary` under-report.
- **solver-core** — `classify_problem`/`extract_*` read only `_constraints`.

**Root fix:** add `Model._has_builder_only_rows()` and a single
`iter_all_rows(model)` (yielding expression-path *and* builder-block rows uniformly,
already implemented once in `export/nl.py:_decompose_builder_blocks`). Route the
classifier, extractor, all exporters, and the examiner through it. `export/_common.py`
is the natural home. After this, GP-1/VAL-1/EX-2/M12 are closed by construction.

### X-2 — "Variable block treated as a scalar" collapse (2 P0s + export)
Code that assumes an array-variable block is a single scalar:
- **solver-core CORE-1** (P0) — `_extract_linear_coefficients` "array variable
  treated as sum"; vector constraint → one summed row → infeasible point certified.
- **tightening TG-1** (P0) — FBBT seeds each block from element-0's bounds and stamps
  them on every element → cuts feasible points, false-infeasible (Rust
  `fbbt.rs:1204-1208` + `tightening.py:116-120`).
- **export / GAMS** — `.flat[0]` collapse of heterogeneous array bounds.

**Root fix:** a codebase sweep for `.first()` / `.flat[0]` / block-as-scalar on
array variables, plus the two specific fixes (extractor raises on array-in-scalar
position and routes to the row-correct autodiff path; FBBT carries per-scalar
intervals). Add an array-variable-with-heterogeneous-bounds fixture to the shared
test utilities so every such site is exercised.

### X-3 — `from_nl` binary bounds drop (corpus + Pyomo bridge)
**modeling M2 = infra INT-2**: `from_nl` rebuilds binary columns as free `[0,1]`,
discarding parsed `lb/ub`. Hits `from_pyomo`, the `SolverFactory('discopt')` bridge,
and any MINLPLib `.nl` with a fixed binary → wrong optimum. One fix in `from_nl`
(route bound-narrowed binaries through `m.integer(name, lb, ub)`).

---

## 2. Priority tiers (after the shared roots)

Ordering rule: **security and wrong-certified-answer-on-the-default-path first;
opt-in-path correctness next; display/example/advisory after; robustness/perf/SOTA
last.** "Default path" = a plain `m.solve()` on an idiomatic model.

### Tier 0 — Security
- **LLM-1** (llm) — RCE via `eval()` sandbox escape. Optional (`discopt[llm]`) path,
  but arbitrary code execution reachable through `chat()`/`from_description()`.
  Fix: AST-allowlist evaluator (no `Attribute` nodes) or structured operand trees.
  *Independent of everything else — can proceed in parallel.*

### Tier 1 — Wrong certified answer on the DEFAULT path (idiomatic models)
- **CORE-1** / **CORE-2** (solver-core = modeling M1, ro ADJ-1) — vector-body row
  collapse; maximize-sense loss on `sum(c·x)`. *Part of X-2.*
- **TG-1** (tightening) + **CF-1** (conflict) — FBBT array collapse; invalid conflict
  cuts. *Part of X-2.*
- **NN-1** (nn) — scaling ignored in bound propagation → unsound big-M → wrong optimum.
- **NM-1** (`_numpy`/`_jax` McCormick) — `relax_asin`/`relax_acos` inverted curvature
  → **unsound convex envelope in the LIVE JAX relaxation layer** (cv sits above the
  function) → invalid dual bound. Fix both `mccormick.py` files + docstrings; extend
  the soundness harness to sample off-diagonal (diagonal-only masking is what hid it).
- **NM-2** (`_numpy` McCormick) — compiler leaf drops the box → numpy relaxation
  collapses to the exact nonconvex function (latent: numpy backend compiled-but-
  unused). Keep the numpy backend disabled until NM-1+NM-2 are green.
- **GP-1** (gp), **VAL-1** (validation) — *closed by X-1.*

### Tier 2 — Wrong answer on OPT-IN paths / specialized builders
- **RO-1/RO-2/RO-3** (ro) — robust counterpart silently not robust (box sign-tracking,
  ellipsoidal no-op, budget-set superset). Add the "no uncertain parameters remain"
  post-assertion first (converts the whole class to loud errors).
- **DAE C1/C2/C3** (dae) — left-Neumann sign, `integral()` 2× at ncp=1, silent
  no-dynamics models.
- **EX-1** (export) — `.nl` Jacobian non-conformant with ASL (breaks BARON/Couenne/
  SCIP comparisons).
- **INT-1** (infra/solver.py) — `nlp_bb=True` + `lazy_constraints` swallow-and-accept.
- **MP-1** (mpec) — `tighten_complementarity_bounds` overwrites a positive lb → hides
  infeasibility.
- **DC-S1** (decomposition) — unbounded recourse populates an invalid `bound`.
- **M3** (modeling) — cross-model expression aliasing.

### Tier 3 — Display / examples / method-fidelity (visible-wrong, not certificate)
- **MO1** (mo) — NBI quasi-normal wrong axis (k≥3); **MO2/MO3/MO4** fidelity/dedup.
- **E1/E2/E3** (modeling examples) — Haverly wrong (1390 vs 400), logical-constraints
  crashes, reactor design infeasible.
- **L1–L8** (latex) — `to_latex` crashes on GDP models; fast-path constraints invisible.
- **M4/M5/M6/M8** (modeling) — `__eq__`/`__bool__` traps, `subject_to` mutation,
  swallowed kwargs, no shape checking.

### Tier 4 — Robustness, hygiene, honesty, perf, SOTA
- Everything marked P2/P3 in the reviews: doe doc/robustness nits, gams-link edges
  (CD-1..CD-5), llm advisor validation (LLM-3), no-op stubs (LLM-4, reformulation),
  perf items (O(n²) `_check_name`, scalar-loop assembly), and the SOTA-direction
  design work (mesh adaptation, cutting-set RO, exact bi-objective MILP, signomial
  global, notebook CI, deterministic grading).

---

## 3. The Opus fix loop — five steps per finding

For each finding ID, in tier order (shared roots X-1..X-3 first):

**(1) VERIFY.** Reproduce the failing case from the review doc's repro against the
current HEAD. Confirm it fails *now*. If it does **not** reproduce, do not "fix" it —
record `NOT REPRODUCED` with the evidence in the review doc and move on. A finding
that cannot be reproduced is not a bug.

**(2) FIX — root, not band-aid.** Follow CLAUDE.md's development philosophy:
correctness before performance; fix the *class*, not the instance; prefer the hard
right fix (a loud refusal) over a silent approximation; never weaken a validation,
gate, or fallback to make a test pass. For a shared-root finding, fix the shared
primitive (§1) and let the dependent findings close.

**(3) VALIDATE — the fix must be proven, not asserted.**
- Write a **regression test that fails before the fix and passes after.** Confirm it
  fails on the pre-fix code (stash the fix, run the test, see red) — a test that was
  never red proves nothing.
- Run the module's suite + `pytest -m smoke` + the adversarial suite
  (`test_adversarial_recent_fixes.py`); `cargo test -p discopt-core` if Rust changed.
- **Bound-changing fixes** (relaxations, cuts, tightening, big-M, extraction that
  alters what the solver sees): the CLAUDE.md §5 *differential* regime — new bound ≥
  old bound AND ≤ the true box optimum on fixed boxes, plus feasible-point sampling
  (no valid point cut), behind a feature flag if it can't be proven green immediately.
- **Bound-neutral fixes** (refactors, marshaling, the shared-primitive plumbing):
  assert `node_count` and certified `objective` are **exactly unchanged** on a
  certifying panel. Any drift means the change is wrong.
- Correctness gate is zero-slack: `incorrect_count ≤ 0` on every panel touched.

**(4) UPDATE THE DEV REPORT — mark the finding done, visibly.** In the finding's
review doc:
- Prefix the finding's row/heading with **`✅ RESOLVED`** and append the commit SHA
  and the regression-test name (e.g. `✅ RESOLVED a1b2c3d — test_nn_scaling_bigM`).
- If verification found it non-reproducible or already-fixed, mark
  **`◻︎ NOT REPRODUCED`** / **`✅ ALREADY FIXED`** with the evidence.
- Keep the original finding text intact below the marker (so the history is legible)
  — do not delete findings.

**(5) MARK COMPLETE at the module level.** When *every* finding in a review doc is
resolved (or not-reproduced), add a status banner at the **top** of that doc:
`> STATUS: COMPLETE — all N findings resolved as of <SHA>, <date>.` Then check that
module's box in **GitHub issue #413**. A partially-done module gets
`> STATUS: IN PROGRESS — k/N resolved` so the loop can resume precisely.

**PR discipline (CLAUDE.md Workflow):** one finding or one shared-root cluster per
PR; task ID in the title (`fix(correctness): CORE-1 …`, `fix(nn): NN-1 …`); the PR
body states what was run and the result; benchmark/perf claims carry the measurement.
Feature branch off `main`; do not commit fixes directly to the review branch's docs
without the accompanying code change.

---

## 4. Status ledger (update as findings resolve)

Legend: ⬜ open · �her in-progress · ✅ resolved · ◻︎ not-reproduced.

| Tier | ID(s) | Module | Status |
|------|-------|--------|--------|
| root X-1 | GP-1, VAL-1, EX-2, M12, classifier | gp/validation/export/modeling/core | ⬜ |
| root X-2 | CORE-1, TG-1, CF-1, GAMS bounds | solver-core/tightening/conflict/export | ⬜ |
| root X-3 | M2 = INT-2 | modeling/infra | ⬜ |
| 0 | LLM-1 | llm | ⬜ |
| 1 | CORE-2 | solver-core | ⬜ |
| 1 | NN-1, NN-2 | nn | ⬜ |
| 1 | NM-1 (live JAX + numpy), NM-2 | _numpy/_jax McCormick | ⬜ |
| 2 | RO-1, RO-2, RO-3 | ro | ⬜ |
| 2 | C1, C2, C3 | dae | ⬜ |
| 2 | EX-1 | export | ⬜ |
| 2 | INT-1 | infra/solver | ⬜ |
| 2 | MP-1 | mpec | ⬜ |
| 2 | DC-S1 | decomposition | ⬜ |
| 2 | M3 | modeling | ⬜ |
| 3 | MO1–MO4 | mo | ⬜ |
| 3 | E1–E3, L1–L8, M4–M8 | modeling/examples/latex | ⬜ |
| 4 | (P2/P3 across all) | all | ⬜ |

*(Expand per-finding as the loop proceeds; the authoritative per-finding detail
stays in each module's review doc.)*

---

## 5. Pending inputs

- **`_numpy` McCormick soundness-fuzz review** — LANDED (`numpy-mccormick-review.md`).
  Verdict: `NM-1` (`relax_asin`/`relax_acos` inverted curvature) is an **unsound
  envelope in the live JAX layer** → Tier 1; `NM-2` (numpy compiler leaf drops the
  box) is Tier 1 for activation-gating (keep the numpy backend disabled until fixed).
  As part of NM-1, **check whether any test/benchmark instance uses `asin`/`acos`**;
  if so the risk is not merely latent.
- **Solver core proper** (`solver.py` B&B loop, `_jax/` relaxation layer,
  `solvers/`, Rust `crates/`) — **not yet reviewed**; ~75k lines. This is where
  global-optimality soundness ultimately lives and needs its own review regime
  (differential bound tests, `cargo test`), not a read-through. Schedule as a
  dedicated multi-session effort after the Tier 0–1 fixes land. The extraction
  review (`solver-core-extraction-review.md`) covers only the classify/extract path.

---

## 6. Guiding principles (from CLAUDE.md, restated for the loop)

1. **Correctness before performance, always.** A fix that makes something faster but
   risks a false optimal/infeasible/bound is a regression. `incorrect_count ≤ 0` has
   zero slack.
2. **Never weaken a validation, gate, or fallback to make a test pass.** If a goal
   can only be met that way, the goal loses — stop and surface it.
3. **Fix the class, not the instance.** No fix keyed to a named problem/shape.
4. **Prefer the loud refusal over the silent approximation.** Many findings here are
   *silent* failures; the minimum bar for several (RO, GAMS parser, GP recognition)
   is to make them raise before making them handle more cases.
5. **No fix ships on a hypothesis.** Reproduce first (step 1); prove with a
   fails-before/passes-after test (step 3).
6. **Shared roots before consumers.** Don't patch five layers when one primitive is
   wrong.
