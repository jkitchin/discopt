# Review Execution Plan — Priorities & Opus Loop Protocol

**Date:** 2026-07-03
**Status:** master plan for resolving the findings in the `docs/dev/*-review.md`
series (20 module reviews). This document is the entry point for the fix loop; it
defines **what order to fix in**, **how to fix each finding**, and **how to record
completion** so a later reader knows exactly what is done.

> **How to use this in a loop.** Each iteration picks the next unresolved finding
> (respecting the tier order and the shared-root dependencies in §1), runs the
> five-step protocol in §3, and updates the status markers in §4. When a module's
> findings are all resolved, check its box in GitHub issue #413.

---

## 0.5 Reconciliation with `correctness-issues.md` (the core backlog, #396)

`docs/dev/correctness-issues.md` is the solver-core soundness audit (relaxation math,
presolve/FBBT, B&B status, cuts, `.nl` parser, LP/dual). The two documents are
complementary and, as of 2026-07-03, reconciled with **stable C-IDs**:

- The four core-soundness P0s these submodule reviews surfaced (that the core audit
  missed) are carded there as **C-29 (=CORE-1), C-30 (=CORE-2), C-31 (=TG-1),
  C-32 (=NM-1)**. The solver-core review adds **C-33 (=SC-1), C-34 (=FR-1),
  C-35 (=OA-1)**, confirms **C-17**, broadens **C-31**, and escalates **C-23** (=DIV-1).
  *(The C-25..C-28 range belongs to the NN backlog — embedded-NN scaling, tree big-M,
  ONNX, sklearn — do not confuse with the CORE findings.)*
- Overlaps cross-linked to avoid double-work: C-6 (integer clamp) ↔ gams CD-1;
  C-11 (`__ne__`) ↔ modeling M4; C-19 (`relax_tan` pole) = NM-4.
- Same protocol both boards (§0 there = §3 here); one loop can run both. Fix each
  finding once, by its C-ID; do not double-track.

## 0. The reviews and where the findings live

Twenty review documents under `docs/dev/`, each ending in an "Implementation
plan" section with per-finding IDs, repros, and acceptance criteria:

`dae`, `modeling`, `mo`, `gp`, `ro`, `tutor`, `export`, `llm`, `validation`,
`solver-core-extraction`, `nn`, `mpec-estimate-pooling-opf`,
`tightening-conflict-warmstart-iis`, `doe`, `cli-daemon-gamslink-benchmarks`,
`infra-interop`, `decomposition`, `numpy-mccormick` (ID prefix `NM-`), and
`solver-core` (the B&B core proper — convexity certifier, node bounds, cutting
planes, reformulation, OA; IDs `SC-1`=C-33, `FR-1`=C-34, `FR-2`, `OA-1`=C-35,
`DIV-1`=C-23, plus confirmations of `C-17`/`C-31`/`C-23`).

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

**✅ RESOLVED — residual sweep + two fixes.** The two P0s (CORE-1=C-29 extractor,
TG-1=C-31 FBBT `seed_block_interval`) landed on main. The residual sweep for
`.flat[0]`/`.first()`/`lb[0]`/block-as-scalar reads of array-variable **bounds**:

| Site | Reads | Verdict |
|------|-------|---------|
| `export/gams.py:176-191` | array bounds emitted **only when uniform** (`np.all(arr==arr.flat[0])`) | **BUG (EX-4)** — heterogeneous per-element bounds silently dropped *entirely*; repro `x[0]∈[0,1],x[1]∈[2,5]` → 0 bound lines. **FIXED** (`_write_array_bound`: per-element `.lo`/`.up` at 1-based labels, uniform still compacted). `from_gams` taught to parse concrete-label `x.lo('1')` bounds (round-trip green). |
| `_jax/gdp_reformulate.py:267-268` | `_compute_big_m_lp` seeds LP box from element 0's bounds, stamps onto all `v.size` slots | **BUG** — collapses heterogeneous block; repro `x1∈[0,10]` with `x0∈[0,1]` → big-M **1.01** (true 10). Too-small M cuts feasible points of inactive disjunct (opt-in `mbigm`/auto path). **FIXED** (per-flat-element bounds; now 10.1 == sound interval path). |
| `_extract.py:42-55` `_flatten_variables` | scalar branch `.flat[0]` for `shape==()/(1,)`; array branch already `.flat[k]` | benign — size-1 has one element; array path per-element. |
| `_extract.py:153,287,475` | `Constant.value.flat[0]` in scalar position | out of scope — array-**Constant** collapse (EX-7, P2), not array-**variable bounds**. Filed separately. |
| `estimate.py:367` | `result.value(var).flat[0]` extracting an estimated parameter | out of scope — solution-value read; the whole FIM/estimate machinery assumes *scalar* parameters (one FIM row per name), so an array param is mishandled structurally, not merely at `.flat[0]`. Not a bounds collapse. |
| `symmetry.rs:110-111` | `vinfo.lb.first()/ub.first()` | benign — guarded by `if vinfo.size != 1 { continue; }`; only scalars reach it. |
| `fbbt.rs` | `seed_block_interval` (was `v.lb.first()`) | already **FIXED** (C-31): element-wise union seed. |

Shared fixture `heterogeneous_array_bounds` added to `python/tests/conftest.py`;
regression tests in `python/tests/test_x2_residual_array_bounds.py` (fails-before /
passes-after verified by stash). Cert-baseline neutrality: EXACT (nodes X->X,
|Δobj|=0 on all 41), `incorrect_count==0`. **Pre-existing, out of scope:**
`from_gams` cannot parse a concrete quoted label in an *equation body* (`x('1')`
as an expression atom) — independent of bounds, present on main; filed for a
future `from_gams` PR, not an X-2 bounds finding.

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
- **CORE-1=C-29** / **CORE-2=C-30** (solver-core = modeling M1, ro ADJ-1) — vector-body
  row collapse; maximize-sense loss on `sum(c·x)`. *Part of X-2.*
- **TG-1=C-31** (tightening) + **CF-1** (conflict) — FBBT array collapse; invalid
  conflict cuts; broadened to reach the certified LP dual bound via
  `_fbbt_argument_box`. *Part of X-2.*
- **SC-1=C-33** (solver-core) — pure-continuous **fallback** certifies a nonconvex
  model's local optimum with `gap_certified=True` (`solver.py:3716`). Verified:
  double-well bound −37, cert True; true −64. Withhold the certificate unless
  convexity is rigorously known.
- **FR-1=C-34** (solver-core) — even-power bound over a zero-straddling base is
  endpoint-only (`gdp_reformulate.py:493`) → `x**4` on `[−2,2]` → `[16,16]` (true
  `[0,16]`) → invalid aux box → false optimal. Verified e2e (`x**4·y ≤ 1` → 0.25,
  true 0).
- **C-17** (solver-core, was open now **confirmed**) — alphaBB node bound uses sampled
  α + center-only PSD check; deterministic spike repro (width ≤ 0.006 → α=0, bound
  0.0, true min −3.5). Route through `rigorous_alpha`.
- **DIV-1=C-23 (escalated P3→P1)** (solver-core) — `relax_div` invalid convex
  underestimator for **nonlinear** denominators (`1/(x*y)` cv=1.334 > 1.0). Opt-in
  `mccormick_bounds="nlp"` only, but the relaxation math is unsound and must be fixed
  at the math level. Extend the containment harness to nonlinear denominators.
- **NN-1** (nn) — scaling ignored in bound propagation → unsound big-M → wrong optimum.
- **NM-1=C-32** (`_jax` McCormick) — `relax_asin`/`relax_acos` inverted
  curvature → **unsound convex envelope in the LIVE JAX relaxation layer** (cv sits
  above the function) → invalid dual bound. Fix `_jax/mccormick.py` + docstrings;
  extend the soundness harness to sample off-diagonal (diagonal-only masking hid it).
- **NM-2** (`_numpy` McCormick) — ✅ **RESOLVED by deletion** (#413). The compiler
  leaf dropped the box → numpy relaxation collapsed to the exact nonconvex function,
  but the backend was compiled-but-unused (params accepted, never consumed). Deleted
  the whole dead backend rather than porting it — behavior-preserving; see
  `numpy-mccormick-review.md` §4.
- **GP-1** (gp), **VAL-1** (validation) — *closed by X-1.*
- *The convexity certifier + convex fast-path were audited and verified **SOUND**
  (901-certificate fuzz, 0 false) — see `solver-core-review.md`. Not a finding.*

### Tier 2 — Wrong answer on OPT-IN paths / specialized builders
- **RO-1/RO-2/RO-3** (ro) — robust counterpart silently not robust (box sign-tracking,
  ellipsoidal no-op, budget-set superset). Add the "no uncertain parameters remain"
  post-assertion first (converts the whole class to loud errors).
- **DAE C1/C2/C3** (dae) — left-Neumann sign, `integral()` 2× at ncp=1, silent
  no-dynamics models.
- **EX-1** (export) — `.nl` Jacobian non-conformant with ASL (breaks BARON/Couenne/
  SCIP comparisons).
- **INT-1** (infra/solver.py) — `nlp_bb=True` + `lazy_constraints` swallow-and-accept.
  ✅ RESOLVED (#413): the NLP-BB path cannot honor a rejecting callback (no per-node
  cut application; primal heuristics inject incumbents without consulting it), so it
  now refuses `lazy_constraints`/`incumbent_callback` loudly; auto-select routes them
  to spatial B&B (which honors them). `_invoke_pre_import_callbacks` reordered
  reject-before-add with a narrowed `except`.
- **MP-1** (mpec) — `tighten_complementarity_bounds` overwrites a positive lb → hides
  infeasibility.
- **DC-S1** (decomposition) — unbounded recourse populates an invalid `bound`.
- **OA-1=C-35** (solver-core) — OA/LOA emits an unconditional no-good cut on
  non-rigorous NLP failure → false infeasible/optimal (opt-in OA/ECP/LOA paths).
- **FR-2** (solver-core) — `nlp_evaluator` built from `model._constraints` only, blind
  to builder-resident rows (X-1 extension; live for `nlp_ipopt.py:254` + examiner).
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
- Write a **fast regression test that fails before the fix and passes after — this is
  mandatory for every fix, no exceptions.** Requirements:
  - **Fails-before / passes-after:** confirm it goes red on the pre-fix code (stash
    the fix, run the test, see red) — a test that was never red proves nothing.
    Prefer landing the test first (red), then the fix (green).
  - **Fast:** a `@pytest.mark.smoke` unit test (or a Rust `#[test]` unit test) that
    runs **sub-second** and therefore runs in CI on *every* PR, not just nightly.
    Call the buggy primitive/relaxation/bound/extractor **directly** on the repro
    input wherever a direct call reproduces the defect — do not spin up a full
    `Model.solve()` when a one-line call to `_compute_alphabb_bound`,
    `_bound_expression`, `relax_div`, `fbbt_box`, etc. exhibits it. (A short
    end-to-end `solve()` assertion is fine *in addition* when the defect is only
    visible end-to-end, e.g. C-33's `gap_certified` flag.)
  - **Tests the class, not the instance** (CLAUDE.md §2): encode the
    false-certificate *class* so any future reintroduction trips it — off-diagonal
    soundness harness (C-32/NM-1), nonlinear-denominator containment sweep (C-23),
    whole-box α validity (C-17), heterogeneous-block FBBT (C-31/TG-1) — not just the
    one named repro value.
  - **Named in the Log and the ✅ RESOLVED marker** (step 4).
  - Where the review doc already carries a repro (C-17 spike table, C-23 `1/(x*y)`,
    C-31 Rust characterization tests, C-33 double-well, C-34 `x**4`), port it verbatim
    into the fast test — the verification work is already done.
  - **Do not close a finding without this test committed.** "Fixed but no fast
    regression test" is not done.
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

Legend: ⬜ open · ◧ in-progress · ✅ resolved · ◻︎ not-reproduced.

**Reconciled 2026-07-05 (#413):** 14 previously-⬜ rows confirmed-fixed and ticked
✅ against the #396 C-board (`correctness-issues.md`, all `fixed`) plus a present
test on `origin/main` — LLM-1, CORE-2=C-30, SC-1=C-33, FR-1=C-34, C-17, DIV-1=C-23,
NM-1=C-32, OA-1=C-35, dae C1/C2/C3, MP-1, E1–E3, and the L1/L5/L6/L8 latex subset.
Remaining open: **NN-1/NN-2 (=C-25/C-26; C-27/C-28)** — `in progress` on the
C-board, no fix or test on `origin/main` yet (PRs in flight: sibling worktrees
`fix-nn-c25-c26-soundness`, `fix-nn-c27-c28-readers`); **MO5–MO10** (mo
perf/robustness); the **L2/L3/L4/L7** latex tail and **M4–M8** modeling P1/P2 items
(no RESOLVED marker in `modeling-module-review.md`); the **P2/P3 remainder** across
modules; and the **gdp focused review** (not started).

| Tier | ID(s) | Module | Status |
|------|-------|--------|--------|
| root X-1 | GP-1, VAL-1, EX-2, M12 ✅ · classifier/extractor ◻︎ not-reproduced (already builder-aware via Rust repr) · FR-2 (Tier-2, out of scope) | gp/validation/export/modeling/solver-core | ✅ 1dc3278 — test_x1_builder_resident_rows |
| root X-2 | CORE-1=C-29 ✅, TG-1=C-31 ✅, CF-1 ✅ (on main) · **residual**: GAMS/export per-element bounds ✅ + gdp big-M LP element-0 collapse ✅ · sweep otherwise clean | solver-core/tightening/conflict/export/gdp | ✅ — sweep + 2 residual fixes; `test_x2_residual_array_bounds.py`, shared `heterogeneous_array_bounds` fixture |
| root X-3 | M2 = INT-2 | modeling/infra | ✅ — `from_nl` clamps+preserves parsed binary lb/ub; test `TestFromNlBinaryBoundsX3` in `test_nl_reconstruction.py` |
| 0 | LLM-1 | llm | ✅ — RCE-via-`eval()` sandbox escape closed; `ModelBuilder._eval_expression` now uses a whitelisted-AST evaluator (no `eval`). Confirmed on `origin/main`: `test_llm_eval_safety.py`; `llm-module-review.md` STATUS marks LLM-1 RESOLVED. #413 |
| 1 | CORE-2=C-30 | solver-core | ✅ — C-30 `fixed` on the C-board (maximize sense lost on `sum(const·var)` bodies). Confirmed on `origin/main`: `test_c29_c30_linear_body_classify.py`. #413 |
| 1 | SC-1=C-33, FR-1=C-34 (DEFAULT-path P0s) | solver-core | ✅ — both `fixed` on the C-board. **SC-1=C-33** (pure-continuous fallback certifying a nonconvex local optimum): `test_c33_nonconvex_fallback_cert.py` on `origin/main`. **FR-1=C-34** (even-power bound over a zero-straddling base): PR #425, `test_power_certification.py` on `origin/main`. #413 |
| 1 | C-17 (confirmed), DIV-1=C-23 | solver-core | ✅ — both `fixed` on the C-board. **C-17** (alphaBB sampled-α / center-only PSD → false optimal): `test_alphabb_bound_soundness.py` on `origin/main`. **DIV-1=C-23** (`relax_div` invalid underestimator for nonlinear denominators): `test_div_nonlinear_denominator_c23.py` on `origin/main`. #413 |
| 1 | NN-1, NN-2, NN-3, NN-4 | nn | ✅ — **NN-1=C-25 (scaled-NN bound-prop → valid big-M) ✅, NN-2=C-26 (tree out-of-box big-M) ✅** (code fixed by PR #411, differential-soundness re-verified + doc-closed #489: 0/30 & 0/40 feasible pts cut). NN-3=C-27 (ONNX reader Gemm/Add/branch) ✅ and NN-4=C-28 (sklearn logit/base_score) ✅** — code fixed by PR #411 (2026-07-03), verified vs onnxruntime/sklearn oracles + doc-closed (#488), `test_nn_reader_fixes.py`. NN-1=C-25 (scaling→unsound big-M) / NN-2=C-26 (tree out-of-box big-M) — PR in flight (`fix-nn-c25-c26-soundness`). |
| 1 | NM-1=C-32 (live JAX) | _jax McCormick | ✅ — C-32 `fixed` on the C-board (`relax_asin`/`relax_acos` inverted curvature → unsound convex envelope in the live JAX layer). Confirmed on `origin/main`: `test_asin_acos_envelope_c32.py`. #413 |
| 1 | NM-2 | _numpy McCormick (deleted) | ✅ — deleted the dead, compiled-but-unused `discopt._numpy` backend (behavior-preserving: params accepted but never consumed; POUNCE builds the bound from the JAX relaxation alone). Removed `_numpy/` pkg, `solver.py` compile block + 2 call sites, dead `obj/con_relax_fns_numpy` params in `_jax/mccormick_nlp.py`, numpy `_BACKENDS` id in C19/C23/C32 envelope tests, stale mypy override; #413 |
| 2 | RO-1, RO-2 ✅ · RO-3 ✅ | ro | ✅ — **RO-1/RO-2** prior PR (`test_ro_soundness.py`). **RO-3** (#413): `budget_uncertainty_set` stored only the all-plus/all-minus budget facets (2 of the 2^k) — a strict superset of the true Bertsimas–Sim set → SOUND but up to 2× over-conservative in mixed-sign directions (support of `[1,-1]` at k=2,δ=1,Γ=1 was **2.0 vs the true 1.0**). Now stores the exact **compact lifted `(ξ,u)` polytope** (`5k+1` rows, `2k` vars — polynomial, NOT `2^k`); the polyhedral LP-dualization dualizes it exactly (formulation pads `coeff(x)` with zeros over the `u`-block via the new `n_param`). Differential test **both directions**: (a) LOAD-BEARING robustness — the fixed Γ=1 solution satisfies the constraint at **every** in-set realization across **10,927** exhaustive samples (vertices + 41² grid + 20k random interior); worst in-set slack **4e-8 ≤ tol** → no under-protection; (b) less conservative — support `[1,-1]`=**1.0** (was 2.0), end-to-end price-of-robustness Γ=0/1/2 → obj **3.0/2.0/1.5** matching closed-form B–S (Γ=1 was over-conservative **1.5** pre-fix). Fails-before verified on origin/main (support 2.0, obj 1.5). `test_ro_soundness.py::test_ro3_*` + `test_robust_uncertainty.py`. **Cert-baseline NEUTRAL**: `discopt.ro` is never imported on the `Model.solve()` path (verified) → RO-3 provably off the cert path, `incorrect_count == 0`. #413 |
| 2 | C1, C2, C3 | dae | ✅ — all three RESOLVED per `dae-module-review.md` STATUS. C1 (left Neumann BC sign error), C2 (Radau `ncp=1` `integral()` 2× error), C3 (silent no-dynamics paths). Confirmed on `origin/main`: `test_dae_c1c2c3.py`. #413 |
| 2 | EX-1 | export | ✅ — union-based J/G/k/header sparsity; nonlinear-only vars get a 0-coeff `J` entry; pure-constant bodies move to r-section rhs (`n0` body, no longer counted nonlinear). Byte-level structural parity with Pyomo's writer on a 4-case corpus; `TestNLWriterJacobianConformance` in `test_nl_writer.py`; #413 |
| 4 | EX-7 | export | ✅ — array-valued `Constant` no longer collapses to `value.flat[0]`: reduction contexts sum the full array (`sum([1,2,3,4]) → 10.0`, was `1.0`), scalar-required positions accept 0-d/size-1 and raise loudly on multi-element (no silent element-0 drop). `_eval_constant_array` + `_scalar_constant` in `_extract.py`; `TestArrayConstantCollapse` (6 cases) in `test_export.py`; #413 |
| 2 | INT-1 | infra/solver | ✅ — NLP-BB path now REFUSES `lazy_constraints`/`incumbent_callback` loudly (it cannot enforce a rejection: no per-node cut application + primal heuristics inject incumbents without consulting the callback), and auto-select routes them to spatial B&B which honors them; `_invoke_pre_import_callbacks` reordered to reject-before-add with a narrowed `except` so a programming error can no longer swallow a lost rejection. Tests `TestInt1NlpBbLazyRejection` in `test_callbacks.py`; #413 |
| 2 | MP-1 | mpec | ✅ — RESOLVED per `mpec-estimate-pooling-opf-review.md` STATUS (`tighten_complementarity_bounds` no longer overwrites a positive lower bound with 0, so it can't hide infeasibility). Confirmed on `origin/main`: `test_mpec_mp1_mp2.py`. #413 |
| 2 | DC-S1 | decomposition | ◻︎ not-reproduced — **DC-S1 was SUSPECTED against the pre-#409 tree and is already fixed on `main`.** PR #409 ("Decomposition module remediation", 2026-07-03, same day as the review) added BOTH halves: (1) distinct `unbounded`-recourse detection (`benders/solver.py:489-490,574-575` → `status="unbounded"`, `bound=None`), and (2) the T0.5 eta-floor-withholding guard (`:632-639` → withholds `bound` when any η rests on the `_ETA_FLOOR`). Confirmed on the exact review repro (`min y−x, x∈[0,1e30], x≥y` → `status=unbounded, bound=None, gap_certified=False`) AND the bounded-below-floor sub-case (`min y+x, x∈[−5e12,0]` → `bound=None` withheld, incumbent −5e12 correct, `gap_certified=False`) — no invalid populated bound in either. No solver code changed; added regression tests `test_benders_soundness.py::test_dcs1_*` (+ existing `test_c3_unbounded_recourse_reported`) to pin the fix. #413 |
| 2 | OA-1=C-35 | solver-core | ✅ — C-35 `fixed` on the C-board (OA/LOA emitting an unconditional no-good cut on non-rigorous NLP failure → possible false infeasible/optimal on the opt-in OA/LOA path). Confirmed on `origin/main`: `test_c35_oa_nogood_nlp_failure.py`. #413 |
| 2 | M3 | modeling | ✅ `test_m3_cross_model_ownership` — validate() rejects index-incompatible cross-model leaves |
| 3 | MO1–MO4 ✅ · MO5–MO10 ⬜ | mo | ◧ — MO1 (NBI axis, prior PR) ✅; **MO2** (AUGMECON2: lexicographic payoff + bypass) ✅, **MO3** (Pareto dedup) ✅, **MO4** (stable HV reference `common_reference`) ✅ — `test_mo_augmecon2.py`, `test_mo_pareto.py::TestFilteredDedup`, `test_mo_indicators.py::TestCommonReference`; cert-baseline NEUTRAL (mo off the solve path). MO5–MO10 (perf/robustness) open. |
| 1/2 | GP-2 (P1), GP-3 (P2), GP-4 (P2) | gp | ✅ — **GP-2** (certified GP optima mislabeled `gap_certified=False`): `solve_gp` now builds `SolveResult` in one shot with the log-space bound mapped back to x-space, so `__post_init__` no longer downgrades; certified ONLY when the model classified as an exact GP AND the convex solve returned `optimal` with a finite recovered objective (non-optimal → `bound=None`, `gap=None`, no over-claim). **GP-3** (product-over-sum): `_flatten_sum_terms` distributes `const/monomial * (sum)` with a ≤64-term budget (posynomial-preserving; over-budget → refuse). **GP-4** (`SumOverExpression`): flattener recurses into indexed-`dm.sum` terms; top-level rejection dropped. Tests in `test_gp.py` (`TestGP2Certification`, `TestGP3Distribution`, `TestGP4IndexedSum`), all fail-before/pass-after; negative controls (signomial-in-product, oversized product, indexed-sin/signomial) refuse. **Cert-baseline NEUTRAL**: zero of the 41 baseline instances classify as GP before OR after (all carry integer/non-positive-lb vars → `classify_gp` early-bails), so the changed recognizer is never reached — routing, `node_count`, `objective`, `status`, `gap_certified` all provably unchanged; `incorrect_count == 0`. #413 |
| 3 | E1–E3 ✅ · L1/L5/L6/L8 ✅ · L2/L3/L4/L7 ⬜ · M4–M8 ⬜ | modeling/examples/latex | ◧ — **E1–E3** RESOLVED (`modeling-module-review.md` STATUS; `test_examples_gallery.py` on `origin/main`: Haverly pooling certifies 400, logical-constraints example builds, reactor feasible). **L1/L5/L6/L8** RESOLVED (`modeling-module-review.md` STATUS; `test_display_cluster.py` on `origin/main`). **Still open:** L2/L3/L4 (slice rendering) + L7 (HTML-vs-LaTeX escaping), and M4–M8 (M4 `__eq__`-returns-Constraint, M5 `subject_to` in-place mutation, M6 misspelled-kwarg swallow, M7 O(n²) name check, M8) — no RESOLVED marker in `modeling-module-review.md`. #413 |
| 4 | **Rust-1** (batch LP dual/ray unscale), **Rust-2** (numpy panic) | lp/simplex/batch, discopt-python | ✅ — **Rust-1** (option a): `solve_lp_batch`'s shared-scaling branch now calls `unscale_dual`/`unscale_ray` after `unscale_x`, so a batch solve returns the dual/ray in the *original* space — bit-identical to the single-solve `solve_lp` (`primal.rs`). Confirmed on the ill-scaled matrix: batch dual `−0.671` vs true unscaled `−0.00131` (factor-512 scaling error) → now equal to <1e-9. **Rust-2**: `evaluate_objective`/`evaluate_constraint` bindings did `as_slice().unwrap()`, which returns `None` for a non-contiguous numpy view → a Rust panic surfaced as an opaque `pyo3_runtime.PanicException` across FFI. Now materialize a contiguous `Vec` via `as_array().iter().copied().collect()` (matches the file's established pattern), accepting strided/transposed input. Confirmed: `np.ones(2n)[::2]` panicked at `expr_bindings.rs:372` before, returns the correct value after. Tests fail-before/pass-after: `batch_unscales_{dual,ray}_like_single_solve` (Rust), `test_evaluate_{objective,constraint}_noncontiguous_input` (`test_rust_ir.py`). Cert-baseline NEUTRAL (both are on discarded/error paths — no solve-result change). #413 · P2/P3 remainder across other modules ⬜ |

*(Expand per-finding as the loop proceeds; the authoritative per-finding detail
stays in each module's review doc. Solver-core findings: `solver-core-review.md`;
the C-IDs are also carded in `correctness-issues.md`.)*

**X-1 resolution notes (builder-resident-rows blind spot).** All four confirmed
findings reproduced on `origin/main` and were closed by routing the affected
consumers through the shared `discopt.export._common` row iterator plus
`Model._has_builder_only_rows()`:
- **GP-1** (P0, confirmed) — `classify_gp` recognised a fast-API GP-shaped model and
  solved the log-space reformulation WITHOUT the builder rows (`m.solve()` returned
  `0.1`; true optimum `0.5`). Fix: `classify_gp` returns `None` when
  `model._has_builder_only_rows()` (refuse → fall back to spatial B&B, which sees the
  builder rows). Both the `m.constraint(...)` fast path and `add_linear_constraints`
  variants now solve `0.5`.
- **VAL-1** (P0, confirmed) — the examiner (`examine`) certified `x=[0,0]` against a
  builder-resident `x[0] >= 3` floor as `passed=True` (0 constraints seen). Fix: a new
  `primal_con_feas (builder rows)` check evaluates the builder linear rows directly in
  the examiner's flat-variable frame; the violating point is now `passed=False`. The
  fix only *adds* coverage — no existing check was relaxed. (The dual-side KKT machinery
  still reads expression rows only — full dual coverage of builder rows is FR-2, Tier-2.)
- **EX-2** (P0, confirmed) — MPS/LP/GAMS export of a fast-API-only model emitted an
  empty model (ROWS = `N OBJ` only, LP `Subject To` empty, GAMS `obj_var =e= 0`). Fix:
  MPS/LP/GAMS now emit the builder rows via `iter_builder_linear_rows`; the exported MPS
  round-trips through HiGHS to the true optimum (3.0).
- **M12** (confirmed) — `num_constraints` / `summary` reported `0` for a fast-API-only
  model. Fix: `num_constraints` adds `_num_builder_constraint_rows()`.
- **classifier / extractor** (◻︎ NOT REPRODUCED) — `classify_problem` and the public
  `extract_lp_data`/`extract_qp_data`/`extract_qcp_data` already route fast-API models
  through the Rust `model_to_repr` path (`_extract_*_from_repr`), which *is* builder-aware
  (`repr.n_constraints == 1`, `A_eq` carries the row). The `*_algebraic` variants are
  blind but are only reached as a fallback when the repr path raises; fast-API models
  always carry a `_builder`, so the repr path wins. No wrong answer reproduced. (The
  shared iterator is nonetheless the canonical home should a future consumer need it.)
- **FR-2** — a Tier-2 X-1 extension (nlp_evaluator / `nlp_ipopt.py:254` blind to builder
  rows in the *dual*/relaxation path); tracked separately, out of scope for this PR.

**X-3 resolution notes (`from_nl` binary-bounds drop = M2 = INT-2).** Reproduced on
this branch's base (`origin/main`): `from_nl` reads each column's parsed `lb/ub` but
the `binary` branch called `m.binary(name, shape=shape)`, which hardcodes `[0, 1]` and
ignores them — so a `.nl` binary fixed to 1 (`lb == ub == 1`, routine Pyomo/presolve
output) re-imported as free `[0, 1]`. Repro: `to_nl`→`from_nl` of `min x + 10·y`
(x∈[0,5], y binary fixed to 1) reconstructed `y` as `lb=0, ub=1` and solved to `~0.0`
(y un-fixed to 0) instead of the true `10.0`. **Fix** (`modeling/core.py`, `from_nl`):
after `m.binary(...)`, stamp the parsed bounds onto the variable, clamped into `[0, 1]`
for soundness (`np.clip(..., 0, 1)`, broadcast to shape) — general to all binary
columns and all shapes, not keyed to any instance; continuous/integer branches already
preserved their bounds. After the fix the reconstructed `y` carries `lb=ub=1` and the
round-trip solves to `10.0`. Regression: `TestFromNlBinaryBoundsX3` in
`python/tests/test_nl_reconstruction.py` (fails-before verified by stashing the core
fix: both cases assert the dropped bound, `0.0 == 1.0`). This closes both **M2**
(modeling review) and **INT-2** (infra/interop review, which adds the Pyomo-bridge and
MINLPLib-corpus impact surface). Cert-baseline neutrality: exact (no baseline instance
carried a bound-binding fixed binary that was being dropped).

**Verified SOUND (not findings, do not re-audit without new evidence):** the
convexity certifier + convex fast-path (901-certificate fuzz, 0 false), the JAX
relaxation-compiler Variable leaf, primal/dual simplex, tree manager, PyO3 boundary,
Gurobi/AMP nonconvex gating, and the RLT/PSD/SOC/edge-concave cut families — all
verified in the solver-core review.

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
