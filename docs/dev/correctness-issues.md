# Correctness issue backlog — audit findings, loop-executable

**Date:** 2026-07-02 (updated 2026-07-03: +NN C-25..C-28, +Fable-review core P0s
C-29..C-32, +solver-core review C-33..C-35; C-17 confirmed, C-23 escalated)
**Status:** complete backlog — 35 issues. C-1..C-24 from the six-agent core audit;
C-25..C-28 from the NN module remediation (`docs/dev/nn-module-plan.md`); C-29..C-32
are the four core-soundness P0s the Fable submodule reviews surfaced that the core
audit missed (CORE-1/CORE-2/TG-1/NM-1); C-33..C-35 from the solver-core review.
**Source:** six-agent soundness audit of the relaxation math, presolve/FBBT, B&B
status logic, cut validity, `.nl`/IR ingestion, and LP/dual-bound layers; plus the
Fable per-submodule reviews (`docs/dev/review-execution-plan.md`) and the solver-core
review (`docs/dev/solver-core-review.md`).
**Companion:** `docs/dev/certification-gap-plan.md` (performance). This file is
correctness only.

> **2026-07-03 reconciliation.** The Fable submodule reviews independently re-found
> some issues here and surfaced core-soundness P0s the six-agent audit missed. Folded
> in as **C-29** (=CORE-1, vector-body collapse), **C-30** (=CORE-2, maximize-sense
> loss), **C-31** (=TG-1, FBBT array collapse), **C-32** (=NM-1, `asin`/`acos`
> inverted curvature). The solver-core review added **C-33** (=SC-1, pure-continuous
> fallback certifies a nonconvex model), **C-34** (=FR-1, even-power bound over a
> zero-straddling base), **C-35** (=OA-1, unconditional no-good cut); it **confirmed
> C-17** with a deterministic repro, **broadened C-31** (now reaches the certified LP
> dual bound via `_fbbt_argument_box`, not just conflict cuts), and **escalated C-23**
> P3→P1 (`relax_div` is unsound for *nonlinear* denominators). It also verified the
> **convexity certifier and convex fast-path SOUND** under 901-certificate fuzz — the
> highest-stakes gate holds; do not re-audit without new evidence. Pre-existing
> overlaps cross-linked in-line: C-6 (integer clamp) ↔ gams CD-1; C-11 (`__ne__`) ↔
> modeling M4; C-19 (`relax_tan` pole) confirmed on the numpy backend too (= NM-4).
> **Note the C-25..C-28 range is the NN backlog** — do not confuse with the CORE
> findings (C-29..C-32).

---

## 0. Loop protocol (binding on the implementing agent)

This document is designed to be looped over, one issue per iteration, in status-board
order (§1). For EACH issue:

1. **CONFIRM first.** Run the issue's *Reproduce/confirm* step before writing any
   fix. Audit findings are evidence-based but not all were independently verified.
   - If confirmation fails, set the issue's status to `false-positive`, record what
     you observed in the issue's **Log**, and move to the next issue. Do NOT fix
     unconfirmed issues.
2. **Fix minimally.** Implement the *Fix* as scoped. If the true fix is materially
   different from the sketch, that's fine — the *Done criteria* govern, not the
   sketch. Record the deviation in the **Log**.
3. **Verify.** All *Done criteria* must pass, PLUS the standing gates:
   - `pytest -m smoke` — 0 failures
   - `pytest -m slow python/tests/test_adversarial_recent_fixes.py` — 0 failures
   - `cargo test -p discopt-core` when Rust was touched — 0 failures
   - `incorrect_count ≤ 0` on the affected suite (never weaken this)
   - **every fix ships with a NEW *fast* regression test — mandatory, not optional.**
     It must (a) **fail before** the fix and **pass after** (prove it red on the
     pre-fix code — ideally land the test first, red, then the fix, green); (b) be
     **fast**: a `@pytest.mark.smoke` unit test (or a `cargo test` unit test for Rust)
     that runs **sub-second** by calling the buggy function/relaxation/bound
     **directly** on the repro input rather than a full `Model.solve()` where a direct
     call suffices — so it runs in CI on every PR and locks the class against future
     regression; (c) encode the **specific false-certificate class**, not just the
     named instance (§2 of CLAUDE.md) — e.g. the off-diagonal soundness harness for
     C-32, the nonlinear-denominator containment for C-23, the whole-box α check for
     C-17, the heterogeneous-block FBBT for C-31 — so a future refactor that
     reintroduces the bug in any guise trips it; (d) be **named in the Log**. Where a
     repro already exists (C-17 spike, C-23 `1/(x*y)`, C-31 characterization tests,
     C-33 double-well, C-34 `x**4`), port it verbatim. Do not close an issue without
     this test committed.
4. **Update this file in the same PR:** status (`open` → `confirmed` → `fixed` /
   `false-positive` / `wontfix+reason`), Log entry, PR number.
5. One issue per PR, titled `fix(correctness): <ID> <summary>`.
6. **Never weaken a safety mechanism to make a test pass** (validation fallbacks,
   `gap_certified` downgrade, trusted-incumbent gate, `incorrect_count` check,
   cut-margin/fractional-bound guards).
7. Stop and escalate to the maintainer if: a fix requires touching a safety
   mechanism; confirmation reveals the bug is WORSE than described (e.g. a false
   "optimal", not just a false "infeasible"); or two consecutive issues from the
   same audit section are false-positives (re-verify that section before continuing).

Severity scale: **P0** = can report a wrong certified answer (false optimal /
false infeasible / wrong optimum) in a reachable configuration. **P1** = wrong
status/answer in a realistic edge case, or silently solves a different problem than
the user posed. **P2** = robustness/weakness; wrong only under unusual numerics or
in non-default configs; loud ingestion gaps. **P3** = hygiene.

## 1. Status board (work top to bottom)

| ID | Priority | Area | Summary | Status |
|---|---|---|---|---|
| C-16 | P0 | presolve/aggregate | positional bound resync after variable-removing pass fuses unrelated variables' bounds → false "infeasible" / silent optimum cut, DEFAULT path | fixed |
| C-17 | P0 | alphaBB bound | node bound uses sampled (non-rigorous) α + center-only PSD check → false "optimal", default path for small nonconvex; deterministic repro confirms (spike width ≤ 0.006 → α=0, bound 0.0, true min −3.5) | fixed |
| C-13 | P0 | solver.py bounds | serial convex path trusts under-converged NLP objective as node lower bound → false "optimal" | fixed |
| C-1 | P0 | solver.py status | false "infeasible" from non-rigorous NLP fathoms (solve_model path) | fixed |
| C-4 | P1 | mir.rs cuts | integer-MIR applied with fractional integer lower bound → invalid cut | fixed |
| C-2 | P1 | milp_driver status | false "Infeasible" when deadline orphans deferred nodes | fixed |
| C-19 | P1 | relax_tan | pole-straddling interval classified as one branch → secant across a pole, invalid envelope | fixed |
| C-5 | P1 | .nl parser | floor/ceil/round/trunc→identity, intdiv→div, all silent | fixed |
| C-6 | P1 | modeling API | integer vars silently clamped to [0, 1e6] | fixed |
| C-18 | P1 | midpoint bound | `mccormick_bounds="midpoint"` returns u(mid), not a lower bound (opt-in mode) | fixed |
| C-20 | P2 | fbbt_fp.rs | watch-list FBBT declares infeasibility with zero tolerance (opt-in engine) | fixed |
| C-15 | P2 | obbt.py | `run_obbt` variant tightens to raw LP vertex, no NS safe-bound clamp | fixed |
| C-14 | P2 | milp_driver | LP-infeasible fathom trusts status alone; Farkas ray never verified | fixed |
| C-21 | P2 | incremental MC | soundness-gate validation boxes never exercise negative/zero-spanning bounds | fixed |
| C-7 | P2 | .nl parser | defined variables (V segments) discarded → hard parse error | fixed |
| C-8 | P2 | .nl parser | common opcodes unhandled (o76/o77/o78/o48/o11/o12/o35) | fixed |
| C-9 | P2 | .nl parser | nlvo>nlvc integer-block classification unverified | fixed |
| C-10 | P2 | lp_spatial cuts | GMI cuts appended without rhs safety margin (opt-in path) | fixed |
| C-3 | P2 | solver.py incumbent | unrounded integer incumbent survives if terminal polish throws | fixed |
| C-11 | P2 | modeling API | missing `__ne__` → `x != y` silently evaluates False | fixed |
| C-22 | P3 | fbbt.rs interval | `interval_mul` NaN endpoints on 0·∞ (lost tightening, not unsound) | fixed |
| C-23 | P1 | mccormick.py | ESCALATED (was P3): `relax_div` produces an invalid convex underestimator (cv > f) for **nonlinear** denominators (`1/(x*y)` cv=1.334 > true 1.0) — the "harmless" label held only for variable/affine denominators; `_relax_reciprocal` also mislabels concavity for negative denominators (= DIV-1) | fixed |
| C-24 | P3 | mccormick.py | secants produce NaN on infinite bounds; soundness leans on downstream filters | fixed |
| C-12 | P3 | .nl parser | range-split renumbers constraints vs source indices | fixed (documented) |
| C-25 | P1 | nn/formulations | scaling + bound-propagation domain mismatch cuts the true optimum on scaled embedded NNs | in progress (nn-module-plan T-N0.2) |
| C-26 | P1 | nn/tree_ensemble | tree big-M invalid for out-of-box thresholds → cuts feasible points | in progress (nn-module-plan T-N0.3) |
| C-27 | P2 | nn/readers/onnx | ONNX reader silently mis-reads Gemm attrs / residual Add / branched graphs | in progress (nn-module-plan T-N1.1) |
| C-28 | P2 | nn/readers/sklearn | sklearn classifier semantics silently embed logits / wrong base_score | in progress (nn-module-plan T-N1.2) |
| C-29 | P0 | classify/extract | vector-body constraint collapses to one summed row ("array var treated as sum") → infeasible point certified optimal, DEFAULT path (= CORE-1, modeling M1) | fixed |
| C-30 | P0 | classify/extract | maximize sense lost on `sum(const·var)` bodies (raises `ValueError` not `_NotLinearError`, mis-routes to sense-dropping fallback) → returns 0 instead of true max (= CORE-2, ro ADJ-1) | fixed |
| C-31 | P0 | presolve/FBBT | FBBT collapses an array-variable block to element-0's bounds and stamps them on every element → cuts feasible points AND false "infeasible"; chains into invalid conflict cuts AND (broadened) into the certified LP dual bound via `_fbbt_argument_box` (= TG-1) | fixed |
| C-32 | P0 | relaxation/mccormick | `relax_asin`/`relax_acos` inverted curvature regime → unsound convex envelope (cv > f) in the LIVE JAX layer → invalid dual bound (= NM-1) | fixed |
| C-33 | P0 | solver.py fallback | pure-continuous fallback certifies a nonconvex model's local optimum with `gap_certified=True` (= SC-1), DEFAULT path | fixed |
| C-34 | P0 | gdp_reformulate | even-power bound over a zero-straddling base uses endpoint-only bounds (omits interior min at 0) → invalid aux box → false optimal (= FR-1), DEFAULT path | fixed |
| C-35 | P1 | oa.py / gdpopt_loa | non-rigorous NLP failure → unconditional no-good cut → possible false infeasible/optimal (= OA-1, opt-in OA/LOA path) | fixed |
| C-36 | P3 | convexity/interval.py | Python `interval_mul` yields NaN on 0·∞ (`RuntimeWarning: invalid value encountered in multiply`), same 0·∞ class as C-22 but a SEPARATE Python code path (`python/discopt/_jax/convexity/interval.py:171`); lost tightening, not unsound | open |

---

## C-16 (P0) — Presolve bound resync after a variable-removing pass remaps positionally, fusing unrelated variables' bounds (DEFAULT path)

**Area:** `crates/discopt-core/src/presolve/pass.rs:104-124`
(`resync_bounds_after_rewrite`), called from `orchestrator.rs:105-107`; the
variable-removing pass is `aggregate.rs:229-232` (removes the eliminated block and
renumbers later `Variable.index` down by one; the survivor's correct bounds are
written into the new model's `VarInfo` at `:249-250`); emptiness latched at
`orchestrator.rs:158-159` (`any_empty` → `TerminationReason::Infeasible`).
**Reachability:** `run_root_presolve(..., aggregate=True)` is the default
(`python/discopt/_jax/presolve_pipeline.py:45,95-96`) — this runs on every solve.

**Mechanism:** after aggregation shrinks the model, `resync_bounds_after_rewrite`
intersects `ctx.bounds[i]` (OLD variable i's interval) with new variable i's
declared bounds — but for every `i >= elim_block`, new var i is OLD var i+1. The
intersection fuses two different variables' intervals. If the eliminated block's
neighbor had a non-overlapping range → empty interval → **feasible model reported
infeasible**. If overlapping but tighter → **silent wrong tightening** of the
survivor, which can cut the true optimum (false optimal downstream).

**Worked failure (from the audit; use as the regression test):**
`x0∈[0,10]`, `x1∈[100,200]` appearing only in the equality `x1 − 10·x2 = 100`,
`x2∈[0,10]`. Feasible (x2=8, x1=180). Aggregation eliminates x1; resync at new
index 1 intersects old x1's `[100,200]` with x2's `[0,10]` → empty → Infeasible.
The determinism test (`presolve_determinism.rs`) only asserts run-to-run byte
identity, so it cannot catch this.

**Reproduce/confirm:** Rust unit test constructing the worked model above; run the
orchestrator with `AggregatePass` enabled; buggy behavior:
`TerminationReason::Infeasible`. Also assert the survivor's bounds after a
non-empty variant (overlapping ranges) equal the correct values, catching the
silent-tightening mode.

**Fix:** variable-removing passes must return an explicit old→new index mapping and
`resync_bounds_after_rewrite` must remap `ctx.bounds` through it — or (simpler,
since aggregate already writes correct bounds into the new `VarInfo`) when the
model shrank, rebuild `ctx.bounds` directly from `new_model.variables[i].{lb,ub}`
instead of intersecting with stale positional entries.

**Done criteria:**
- The worked model solves feasible/optimal through root presolve; the
  silent-tightening variant keeps correct survivor bounds.
- A property test: for random aggregation-eliminable models, post-presolve bounds
  of every surviving variable contain the true optimum's values (oracle by dense
  solve of the original).
- Standing gates pass (`cargo test -p discopt-core`, smoke, adversarial,
  incorrect_count).

**Log:**
- 2026-07-02 — **CONFIRMED then FIXED.** Confirmed both failure modes with new
  orchestrator-level tests running `AggregatePass` on the worked model: with the
  pre-fix `resync_bounds_after_rewrite`, a feasible model terminates
  `Infeasible` (empty fusion) and an unrelated survivor is silently tightened to
  `[3, 7]` instead of its true `[0, 10]` (non-empty fusion). Both reproduce
  exactly as the card predicts.
- **Fix (per the card's simpler option).** `presolve/pass.rs:resync_bounds_after_rewrite`
  now detects a *shrinking* rewrite (`n < self.bounds.len()`) and rebuilds
  `ctx.bounds` directly from the new model's `VarInfo` bounds instead of
  intersecting the mis-aligned positional vector. A removing pass (aggregate)
  already writes each survivor's correct bounds into the new model; the stale
  positional tightening that is dropped is re-derived by the fixpoint loop, so
  no bound strength is lost across sweeps. Growing rewrites (aux vars appended at
  the end) keep the original intersect-with-prior behavior — indices are
  preserved there. No explicit old→new index map was needed.
- **Regression tests** (`presolve::orchestrator::tests`, fail-before/pass-after
  verified by stashing the fix): `c16_shrink_does_not_report_feasible_model_infeasible`
  (empty-fusion / false infeasible), `c16_shrink_does_not_silently_tighten_unrelated_survivor`
  (non-empty wrong tightening), and `c16_property_survivor_bounds_contain_feasible_point`
  (300 random aggregation-eliminable models built around a known feasible point;
  asserts every survivor's post-presolve bounds still contain that point — the
  sound "never cut a feasible point" oracle, no dense solver needed).
- **Gates:** `cargo test -p discopt-core` 371 lib + 4 determinism + 1 doctest
  green, no warnings; `pytest -m smoke` 211 passed / 1 skipped (extension rebuilt
  via `maturin develop --release`); adversarial suite 10 passed. PR: #399.

---

## C-17 (P0) — alphaBB node bound uses a sampled α and checks convexity only at the box center → false "optimal"

**Area:** `python/discopt/solver.py:477` (`_compute_alphabb_bound`; tangent-min
computed at `:584-598`); the PSD gate at `:551,556` evaluates the Hessian at the
box CENTER only (`eigvalsh(Hf + 2·diag(α)).min() ≥ −1e-7`); α from
`alphabb.py:56` (`estimate_alpha` — samples the Hessian at ~100 points and
inflates ×1.5; the module itself admits it is non-rigorous at `alphabb.py:261-268`).
Consumed at `solver.py:4920-4923`, `:5174-5179` → `result_lbs[i]` (`:5270,:5282`).
**Reachability:** default path for nonconvex models with `n_vars ≤ 50` whenever the
LP relaxer is not the bound source (`:4412`).
**Note:** a rigorous alternative already exists and is verified correct —
`alphabb.py:271 rigorous_alpha` (interval-Hessian Gershgorin, abstains `+inf` on
unbounded Hessians).

**Mechanism:** `tangent_min` of `L(x) = f(x) − Σαᵢ(xᵢ−lbᵢ)(ubᵢ−xᵢ)` is a valid
lower bound only if L is convex over the WHOLE box. With α under-estimated by
sampling and PSD checked only at the center, a Hessian whose most-negative
curvature lives at an unsampled interior point passes the gate while L is
nonconvex → the "lower bound" can exceed `min_box f` → fathoms the node holding
the true optimum → **wrong answer certified optimal**.

**Reproduce/confirm:** CONFIRMED 2026-07-03 (solver-core review) with a
deterministic repro. `f(x) = ½x² − B·exp(−(x−a)²/2s²)` on `[−2,2]`, B=4, a=1;
sweep the spike width `s`:

| s | sampled α | needed α (true) | H(center) | alphaBB "bound" | true box min | invalid? |
|---|---|---|---|---|---|---|
| 0.006  | 0.000 | 24,791  | 1.0 (passes gate) | 0.000 | −3.500 | +3.5 |
| 0.003  | 0.000 | 99,168  | 1.0 | 0.000 | −3.500 | yes |
| 0.0015 | 0.000 | 396,674 | 1.0 | 0.000 | −3.500 | yes |

The narrow negative-curvature band falls between all deterministic sample points →
α=0; the center Hessian (=1.0) passes the gate; the function returns 0.0 as a
"valid lower bound" while the true minimum is −3.5. In B&B this fathoms the node
holding the true optimum → wrong answer certified optimal. (Repro script was saved
under the review scratchpad; re-derive it as the regression fixture — see below.)

**Fix:** replace `estimate_alpha` with `rigorous_alpha` in the node-bound path,
and drop the center-only PSD gate (rigorous α guarantees convexity by
construction; when it abstains — unbounded interval Hessian — return no bound
rather than a guessed one). Keep `estimate_alpha` only for non-certifying
heuristic uses, renamed/flagged accordingly.

**Regression test (required — fast, add in the fix PR):** port the `s=0.006` spike
repro into a `@pytest.mark.smoke` unit test that (a) asserts the OLD path returns a
bound > true box minimum (documents the bug, xfail-until-fixed), and (b) asserts the
NEW `rigorous_alpha` path returns a bound ≤ true box minimum (or abstains). Sub-second;
no full solve needed — call `_compute_alphabb_bound` directly on the root box. This
locks the class so a future revert to sampled α fails CI immediately.

**Done criteria:**
- The adversarial construction returns a bound ≤ true box minimum (or no bound).
- Differential bound test on a panel of nonconvex ≤50-var instances: alphaBB bound
  never exceeds the oracle box minimum across 100 random boxes each.
- The fast regression test above is committed and passes.
- Benchmarks: node counts may increase (weaker but sound bounds) — acceptable;
  incorrect_count must not.
- Standing gates pass.

**Regression test (committed):** `python/tests/test_alphabb_bound_soundness.py`
— rewritten to build REAL models so alpha is derived rigorously as in
production. `test_c17_spike_bound_is_sound` ports the `s ∈ {0.006, 0.003,
0.0015}` spike and asserts the node bound is `≤` the dense-grid box minimum;
`test_c17_sampled_alpha_would_have_been_unsound` pins that the OLD sampled α
collapses to 0 while the rigorous α is finite-and-large, so the fix is
load-bearing (not vacuous); `test_random_box_panel_never_exceeds_true_min`
sweeps 100 random sub-boxes (differential-bound). All `@pytest.mark.smoke`,
sub-second, calling `_compute_alphabb_bound` directly. Red-before verified: the
pre-fix path returns `−1e-9` on the `s=0.006` spike while the true min is
`−3.5` (assertion fails on old code, passes on new).

**Fix (this PR):** routed the alphaBB node bound through
`alphabb.rigorous_alpha` (sound interval Hessian + per-row interval-Gershgorin
eigenvalue bound), computed **per node box** (not a sampled root α reused at
every node). `_compute_alphabb_bound(evaluator, model, alphabb_expr, node_lb,
node_ub)` now derives α internally and **abstains** (`−inf`, no bound emitted)
whenever `rigorous_alpha` returns a `+inf` entry (unbounded/indefinite interval
Hessian → convexity uncertifiable). The center-only PSD gate (the unsound gate)
is removed — rigorous α makes `L` provably convex over the whole box by
construction, so no fast path can fathom on an unsound value; the center Hessian
is retained ONLY to pick a FISTA step size (tightness, never validity). Setup
now stashes the internally-minimized objective EXPRESSION (`-obj` for maximize)
instead of a root α. Sites: `solver.py` `_compute_alphabb_bound`,
`_alphabb_node_box` (new), the setup block, and both call sites (batch + serial).
Also moved the `inf−inf` row-radius subtraction inside `rigorous_alpha`'s
`errstate` block (benign RuntimeWarning hygiene, now hit per-node).

**Node-count impact (measured):** on the representative alphaBB e2e (the spike,
s=0.2 — wide enough that the old sampled α was *accidentally* sound there) the
fix converged in **5 nodes vs 21 old** — rigorous per-node α tightens as boxes
shrink, so it was *faster* here, not slower. On the narrow spike (s=0.006) the
old path was a false optimal (bound 0.0 > true −3.5); the new path is sound. On
the default MINLPLib corpus the alphaBB path is not engaged at all (the LP/MILP
relaxer is the bound source; `alphabb_calls=0` across the tested nonconvex
instances), so blast radius is confined to small nonconvex models with a
transcendental objective the MILP cannot linearize.

**Log:**
- 2026-07-03 — CONFIRMED open P0 via deterministic spike repro
  (solver-core review, `docs/dev/solver-core-review.md` §1). Status open→confirmed.
- 2026-07-03 — FIXED. Routed through `rigorous_alpha` per node box + abstain;
  dropped center-only PSD gate. Repro red-before confirmed; standing gates green
  (smoke 203p/1s, adversarial 10p, alphabb/convex/bound 967p, ruff+mypy clean).
  Status confirmed→fixed. PR #426.

---

## C-13 (P0) — Serial convex path seeds a node lower bound from an under-converged NLP objective (false "optimal")

**Area:** `python/discopt/solver.py:5164-5169` (seed: `nlp_lb = nlp_obj` when
`nlp_result.status in (OPTIMAL, ITERATION_LIMIT)`), `:5282` (used as the node's
rigorous lower bound `result_lbs[i] = nlp_lb` for convex models), `:5361` (the
serial decertify block runs only `if not _model_is_convex`). Reference correct
implementation: the batch path's trust guard at `:4868`
(`if _model_is_convex and not np.all(_batch_trusted): _gap_certified = False`,
`_batch_trusted` from the dual-infeasibility check at `:8101-8143`).

**Mechanism:** for a convex model the node NLP's objective is used as a valid lower
bound — legitimate only if the NLP actually converged (an interior-point iterate
stopped at ITERATION_LIMIT can sit strictly *above* the true node minimum, with
unconverged duals). The serial path accepts ITERATION_LIMIT objectives as bounds
with no KKT/dual-feasibility trust check and never decertifies for convex models.
An inflated "lower bound" can prune the subtree containing the true optimum while
`gap_certified` stays True → **wrong answer reported as certified optimal**. The
batch path already implements the fix pattern.

**Why P0:** false optimality certificate on the default serial path for convex
MINLPs; the worst failure class in this backlog.

**Reproduce/confirm:**
1. Static: read the three sites; confirm ITERATION_LIMIT flows into `nlp_lb` and
   that no `_batch_trusted`-equivalent exists on the serial path.
2. Dynamic: unit-level — monkeypatch the serial node NLP solve to return
   `status=ITERATION_LIMIT` with `objective = true_node_min + 0.1` on a small
   convex MINLP with known optimum; assert the solve returns the wrong optimum
   with `gap_certified=True` (buggy) vs decertified/correct (fixed).

**Fix:** on the serial path, seed `nlp_lb` from the objective only on
`SolveStatus.OPTIMAL`; for ITERATION_LIMIT either discard the bound (treat as
failed/non-rigorous fathom — interacts with C-1's flag) or subject it to the same
dual-feasibility trust check the batch path uses, decertifying when untrusted.
Remove the `not _model_is_convex` restriction on the serial decertify block so
convex models can decertify too.

**Done criteria:**
- New regression test (the monkeypatched scenario): result is never "optimal with
  gap_certified=True" when the bound came from an untrusted NLP; either the true
  optimum is found or the result is decertified.
- Batch-path behavior unchanged (its guard already exists — add a test pinning it
  if none exists).
- Standing gates pass.

**Log:** 2026-07-03 — **CONFIRMED then FIXED** (branch `fix-c13-convex-nlp-bound`).
*Confirm (static):* the serial `solve_model` node loop seeds `nlp_lb = nlp_obj`
whenever the node NLP status is `OPTIMAL` **or** `ITERATION_LIMIT`
(`solver.py:5312-5317`), with no `_batch_trusted`-equivalent trust check; for a
convex model that value flows into `result_lbs[i]` and thence `tree.import_results`
as the rigorous node lower bound, and the only serial node-loop decertify block was
gated `if not _model_is_convex` (`:5528`), so it never fired for convex models. The
serial `_solve_node_nlp` was not even passed `convex=`, so the convex polish-retry
in `_solve_node_nlp_pounce` never ran. The batch path (`:5016`) and `_solve_nlp_bb`
(`:7411`) both already decertify this exact case — the serial `solve_model` path was
the gap. *Confirm (dynamic):* forcing every convex serial node NLP to report
`ITERATION_LIMIT` (non-KKT) on a convex MINLP solved with `nlp_bb=False, batch_size=1`
returned `status="optimal", gap_certified=True` (false certificate) on pre-fix code.
Reachability: a *nonlinear* convex MINLP normally auto-routes to `_solve_nlp_bb`
(`:3843`), but `nlp_bb=False` (or `lazy_constraints`) forces the `solve_model` serial
loop — user-reachable. *Fix:* (1) pass `convex=_model_is_convex` to the serial
`_solve_node_nlp` so the existing convex polish-retry gets its chance to reach KKT;
(2) for a convex node compute `_serial_nlp_trusted = (not convex) or status==OPTIMAL`
— when untrusted, ABSTAIN from the NLP bound (`nlp_lb=-inf`, so the node imports at
its inherited parent bound and fathoms nothing — no unsound prune) and decertify the
gap after every bound source (mirrors the batch `_batch_trusted` guard and
`_solve_nlp_bb`). Discards the bound rather than trusting it (option 1 of the Fix
sketch); OPTIMAL convex nodes and all nonconvex nodes are untouched.
*Regression test:* `python/tests/test_c13_serial_convex_nlp_bound.py`
(`@pytest.mark.smoke`, sub-second): `test_serial_convex_iteration_limit_does_not_certify`
is RED before the fix (asserts NOT `optimal`+`gap_certified`) / GREEN after;
`test_serial_convex_inflated_bound_never_crosses_primal` pins the dual-≤-primal and
dual-≤-true-optimum invariants under an above-optimum inflated objective;
`test_serial_convex_converged_still_certifies` is the bound-neutral control (all-OPTIMAL
convex solve still certifies, node_count unchanged 3→3). Existing
`test_p03_trust_gate.py` (batch guard + `_solve_nlp_bb` serial) still green (10/10).
Status: `open` → `fixed`. PR: #443.

---

## C-1 (P0) — `solve_model` can report false "infeasible" when every fathom was a non-rigorous NLP failure

**Area:** `python/discopt/solver.py` finalize branch `:6478-6494`; sentinel
handling `:5070-5078` (batch) and `:5361-5367` (serial); contrast with the correct
implementation in `_solve_nlp_bb` (`_unconverged_fathom`, set at `:7145/:7223/:7250`,
consumed at `:7689-7715`).

**Mechanism:** when a node's NLP solve fails or is only locally infeasible (not a
proof), the node loop already *knows* the fathom is non-rigorous — it decertifies
the gap (`_gap_certified=False`). But the finalize else-branch checks only
`max_nodes` and `time_limit`; if the tree exhausts with no incumbent it returns
`status="infeasible"` regardless of whether the exhausting fathoms were rigorous.
`_unconverged_fathom` exists only on the `_solve_nlp_bb` path (which correctly
reports `"unknown"`). `SolveResult.__post_init__` exempts `status=="infeasible"`
from its bound-finiteness downgrade, so nothing catches it downstream.

**Why P0:** "infeasible" is a *certificate claim* about the model. A user with a
feasible model gets told it has no solution — silently wrong, no gap/bound hint.

**Reproduce/confirm:**
1. Static: read `solver.py:6478-6494` and confirm no rigor flag is consulted; grep
   `_unconverged_fathom` — confirm it appears only in the `_solve_nlp_bb` path.
2. Dynamic: construct a small feasible nonconvex MINLP whose node NLPs fail (e.g.
   monkeypatch the node-NLP solve to return the failure sentinel in a test). Solve
   via the default `solve_model` path with `mccormick_bounds="none"` so no rigorous
   relaxation bound exists. Expected buggy behavior: `status == "infeasible"` on a
   model with a known feasible point.

**Fix:** track a non-rigorous-fathom flag in `solve_model` set at exactly the
points that already do `_gap_certified=False` (`:5078`, `:5367`); in the finalize
else-branch return `status="unknown"` (bound=None, gap_certified=False) instead of
`"infeasible"` when the flag is set. Mirror the naming/semantics of
`_unconverged_fathom` on the NLP-BB path.

**Done criteria:**
- New regression test: the reproduction model returns `"unknown"` (or
  time_limit/feasible as appropriate), never `"infeasible"`.
- Rigorous-infeasible still works: a model whose ROOT McCormick LP is infeasible
  still returns `"infeasible"` (add/keep a test asserting this).
- Standing gates (§0.3) pass.

**Log:**
- 2026-07-03 — **CONFIRMED then FIXED.** Static confirm: `solve_model`'s finalize
  else-branch (now `solver.py:~6656-6672`) declared `status="infeasible"` whenever
  the tree exhausted with no incumbent, checking only `max_nodes`/`time_limit` — it
  consulted no rigor flag. `grep _unconverged_fathom` confirms that flag lives
  ONLY in `_solve_nlp_bb` (5 occurrences, all in `7006..7960`), never in the
  `solve_model` batch loop.
- **Dynamic confirm (decisive).** A genuinely feasible nonconvex MINLP
  (`x*y>=1`, feasible x=y=1; integer `z` to keep it on the spatial B&B batch path)
  in which every node NLP fails non-rigorously (constraint-violating "optimal"
  iterate → `_INFEASIBILITY_SENTINEL`) and no rigorous bound exists
  (`mccormick_bounds="none"` + interval bound stubbed to −inf), with the tree
  driven to `is_finished()` + no incumbent — the exact state the Rust
  fathom-at-tight-box path (`tree_manager.rs:511/537`) produces when every leaf
  carries a non-rigorous sentinel. Pre-fix: `status="infeasible"` on the feasible
  model (the false certificate). This is the worst-class error. Note: reaching this
  tree-state through the *full* end-to-end path is guarded in practice — a sentinel
  node is not pruned without an incumbent (`tree_manager.rs:318`), so it keeps
  branching → `node_limit`/`time_limit`, not `infeasible` — but the finalize LOGIC
  itself was unsound and the state IS reachable via the Rust tight-box fathom, so
  the fix hardens the logic regardless.
- **Fix** (`solver.py`, per the card). Added a `_nonrigorous_fathom` flag
  (initialized False alongside `_gap_certified`) with a single authoritative,
  path-agnostic sweep after each batch's node loop, before `import_results`: any
  node entering the tree with the failure sentinel but WITHOUT the rigorous
  `node_infeasible_mask` (empty McCormick/LP relaxation over the finite box) sets
  the flag. The finalize else-branch gains an `elif _nonrigorous_fathom:` arm that
  returns `status="unknown"` (`_gap_certified=False`) instead of `"infeasible"`,
  mirroring `_solve_nlp_bb`'s `_unconverged_fathom` semantics exactly. Covers
  convex + nonconvex, batch + serial in one place. No rigorous check was weakened:
  a node fathomed via `node_infeasible_mask` or a real `SolveStatus.INFEASIBLE`
  still yields `"infeasible"`. `SolveResult.__post_init__` already handles
  `"unknown"` (its `status != "infeasible"` finite-bound downgrade clears the
  meaningless bound/gap).
- **Regression tests** (`python/tests/test_c1_nlp_fathom_infeasible.py`,
  `@pytest.mark.smoke`, sub-second, fail-before/pass-after verified by stashing the
  fix): `test_nonrigorous_fathom_is_not_reported_infeasible` (the decisive repro —
  pre-fix returns `"infeasible"`, post-fix `"unknown"`; asserts the class, not a
  named instance) and `test_rigorous_infeasibility_is_preserved` (empty-box
  `x>=5 ∧ x<=1` still returns `"infeasible"` — the fix must not downgrade a genuine
  certificate).
- **Gates:** ruff check + format clean; mypy (pre-commit) passed; `pytest -m smoke`
  195 passed / 1 skipped / 0 failed (includes the 2 new tests); adversarial
  `test_adversarial_recent_fixes.py -m slow` 10 passed. No Rust touched. No
  correctness assertion weakened (`incorrect_count` unaffected; the change only
  turns an unsound `"infeasible"` into a sound `"unknown"`). PR: #444.

**Status:** open → fixed.

---

## C-4 (P1) — `mir.rs` applies the integer-MIR formula without the fractional-lower-bound guard that `gomory.rs` has

**Area:** `crates/discopt-core/src/lp/mir.rs:45-52` (`mir_row`), reached via
`separate_mir` → `mir_cuts_py` → `solver.py:10223` (`_separate_mir_cuts`) →
`_root_cover_cut_loop` (`solver.py:10371`). Reference implementation of the guard:
`gomory.rs:287-295` (`use_integer = integrality[j] && (pinned - pinned.round()).abs()
<= tol`) and its test `gmi_cut_valid_when_integer_var_has_fractional_bound`.

**Mechanism:** the MIR derivation shifts `x'_j = x_j − l_j` and assumes `x'_j` is a
non-negative *integer*, which requires `l_j` integral. Presolve
(implied bounds / coefficient strengthening) can leave a fractional lower bound on
an integer column — the GMI separator was explicitly hardened against exactly this
(its test comment says so), but `mir_row` uses the integer rounding coefficient
whenever `integrality[j]` is true, with no integrality check on `l_j`. A fractional
`l_j` yields a cut that can exclude a feasible integer point → false certificate.
Both separators are fed the *same* `lp_data` in the root cover-cut loop, so the
scenario the GMI guard defends against reaches MIR unguarded.

**Why P1 (borderline P0):** reachable in the shipping root cut loop, but requires
presolve to actually produce the fractional integer bound; the maintainers'own GMI
test asserts that state is producible.

**Reproduce/confirm:**
1. Static: read `mir_row` and confirm no `l_j` integrality check; read the GMI
   guard and its test for the intended semantics.
2. Dynamic (decisive): port `gmi_cut_valid_when_integer_var_has_fractional_bound`
   to the MIR separator — same LP, integer variable with fractional lower bound,
   enumerate the feasible integer points, assert every one satisfies each returned
   cut. Expected buggy behavior: at least one feasible point violated.

**Fix:** thread per-column lower bounds into `mir_row`; take the integer branch
only when `(l_j − l_j.round()).abs() <= tol`, else use the continuous coefficient
`min(a_j, 0)/(1 − f)` (mirror the GMI fallback).

**Done criteria:**
- The ported validity test passes (no feasible integer point cut) with integral
  AND fractional integer bounds.
- Existing MIR tests still pass; root cover-cut loop still produces cuts on a
  known-good instance (no silent disable).
- Standing gates pass.

**Status:** open → fixed.

**Log:**
- 2026-07-03 — **CONFIRMED then FIXED.** Confirmed with a decisive repro ported to
  the MIR separator: row `−x0 − 2·x1 ≤ −2`, x0 integer in **[0.5, 4]** (fractional
  lower bound, exactly the presolve-produced state the GMI guard defends against),
  x1 integer in [0, 2]; at LP point (0.8, 0.1) the pre-fix separator emits
  `−x0 − 2·x1 ≤ −2.5`, which **excludes the feasible integer point (2, 0)**
  (`−2 > −2.5`) — a cut that removes a feasible integer point of the row. Test
  fails on the pre-fix code with exactly that message.
- **Root cause.** `mir_row` applied the integer-MIR rounding
  `γ_j = ⌊ã_j⌋ + max(0, f_j − f)/(1 − f)` whenever `integrality[j]`, with no check
  on the *active substitution bound*. The bound substitution `y_j = x_j − l_j`
  (lower shift) / `y_j = u_j − x_j` (upper complement) makes `y_j` integer-valued
  only when that bound is integral; a fractional bound breaks the premise and the
  integer γ can cut feasible integer points.
- **Fix (mirrors `gomory.rs:294-295`'s `use_integer` guard).** `mir_under_substitution`
  now computes `int_shift[j] = integrality[j] && (pinned − pinned.round()).abs() ≤
  INT_BOUND_TOL`, where `pinned` is the active bound (`u[j]` when complemented, else
  `l[j]`), and passes `int_shift` (not raw `integrality`) into `mir_row`. When the
  premise fails the column falls back to the always-valid continuous coefficient
  `min(ã_j, 0)/(1 − f)`. Reused/renamed the existing `INT_UB_TOL` (1e-6) as
  `INT_BOUND_TOL` since it is the same integral-bound premise the near-bound
  complementation eligibility check already used for `u_j`. Module + soundness
  docstrings updated. Fix is confined to `crates/discopt-core/src/lp/mir.rs`
  (`mir_row` signature + `mir_under_substitution` + one constant) — no wider than
  the card; no other file touched.
- **Regression tests** (`lp::mir::tests`, both **fail-before / pass-after**, verified
  by temporarily reverting the guard — the 6 pre-existing MIR tests still pass on the
  buggy code, so the new tests isolate specifically the fractional-bound class):
  `c4_mir_valid_when_integer_var_has_fractional_lower_bound` (the deterministic repro
  above; enumerates the integer box and asserts no feasible point is cut) and
  `c4_mir_validity_random_fractional_int_bounds` (1000 random rows with integer
  columns carrying fractional lower AND upper bounds, mixed integer/continuous,
  LP points near bounds so complementation fires; every emitted cut validated over
  the integer box, ≥20 cuts exercised). The pre-existing
  `mir_validity_random_complemented_rows` / `mir_cut_*` tests still pass (no silent
  disable; the root cut path still produces cuts on integral-bound rows).
- **Gates:** `cargo test -p discopt-core` 383 lib + 4 determinism + 1 doctest green,
  no warnings; `cargo clippy -p discopt-core --lib` clean; extension rebuilt via
  `maturin develop --release`; `pytest -m smoke` 339 passed / 1 skipped / 0 failed;
  adversarial `test_adversarial_recent_fixes.py -m slow` 10 passed. `incorrect_count`
  unchanged (0 failures; no correctness assertion weakened). PR: #TBD.

---

## C-2 (P1) — Rust MILP driver reports false "Infeasible" when the deadline orphans in-flight nodes

**Area:** `crates/discopt-core/src/bnb/milp_driver.rs:512-523` (deferred-node
handling), `:598-612` (final status); `tree_manager.rs:233` (`export_batch` marks
nodes `Evaluated`), `:588-596` (`is_finished`); `pool.rs:181-186` (`open_count`
counts only `Pending`).

**Mechanism:** nodes dispatched in a batch are marked `Evaluated` and their heap
entries popped. A node whose LP solve is deferred past the wall-clock deadline is
skipped (`gap_certified=false; continue`) and never re-imported — orphaned in
`Evaluated` state, invisible to `open_count()`. If the rest of the last batch
fathoms rigorously and no incumbent exists, `is_finished()` is true and the driver
returns `MilpStatus::Infeasible` — but the orphaned subtrees were never searched
and may contain the optimum. (`gap_certified` is correctly false, so no false
"Optimal"; the damage is the Infeasible label on a time-limit termination.)

**Reproduce/confirm:**
1. Static: read the three code sites; confirm a deferred node has no path back to
   `Pending`/the heap and that the no-incumbent status branch doesn't consult
   deferral.
2. Dynamic (Rust unit test): build a small MILP tree where one batch contains (a)
   nodes that fathom LP-infeasible and (b) one node forced to defer (inject an
   already-expired deadline before that node's solve — the deferral path is
   `milp_driver.rs:751-769`). Assert the returned status. Expected buggy behavior:
   `Infeasible`; expected fixed: a limit-style status.

**Fix:** add a `search_incomplete: bool` on the driver, set whenever a node is
deferred (or otherwise dropped un-solved); gate the final status:
`!has_inc && tm.is_finished() => if search_incomplete { TimeLimit/NodeLimit } else
{ Infeasible }`. (Do NOT reset the node to `Pending` — its heap entry is already
popped, so `open_count` would still miss it.)

**Done criteria:**
- New Rust regression test as above: status is not `Infeasible` when any node was
  deferred un-solved; genuine infeasibility (all nodes rigorously fathomed, no
  deferral) still returns `Infeasible`.
- `gap_certified` remains false in the deferred case.
- Standing gates pass.

**Status:** fixed.

**Log:** 2026-07-03 — CONFIRMED statically at all three code sites: `export_batch`
(`tree_manager.rs:233`) marks a dispatched node `Evaluated` and pops its heap
entry; the deferral path (`milp_driver.rs`, `out.deferred`) drops the node's
result without re-import, so it is stranded `Evaluated`; `open_count()`
(`pool.rs`) counts only `Pending`, so the orphan is invisible and `is_finished()`
reads `true`; the `!has_inc` status branch returned `Infeasible` unconditionally —
a false certificate on a time-limit termination. (Confirmed the *real* damage is
scoped to the Infeasible label, not a false Optimal: `gap_certified` is already
correctly false on the deferral path.) A deferred-node time-out is inconclusive,
not proven-infeasible; `python/discopt/infeasibility.py` treats `status ==
"infeasible"` as a *proof*, so the false label propagates.

FIX (minimal, per the card sketch): added `search_incomplete: bool` on the driver,
set alongside the existing `gap_certified = false` whenever a node is deferred
un-solved. The terminal-status decision was extracted into a pure `decide_status`
helper so it is unit-testable in isolation; its no-incumbent branch now returns
`Infeasible` **only** on `tree_finished && !search_incomplete` (a rigorous
empty-tree proof), else a limit status (`NodeLimit`, mapped to `"node_limit"` at
`lp_bindings.rs`, which the Python layer treats as inconclusive). Reused the
existing `NodeLimit` variant rather than adding a `TimeLimit` arm — no new enum
arm, no Python-mapping churn; the card permits either. The rigorous-infeasible
path (`search_incomplete == false`) is untouched. `gap_certified` still goes false
on defer.

Regression tests (`crates/discopt-core/src/bnb/milp_driver.rs`, module `tests`,
sub-second, direct calls into `decide_status`): `c2_deferred_node_orphaned_by_
deadline_is_not_infeasible` (RED before fix: returned `Infeasible`; GREEN after:
`NodeLimit` — verified by temporarily reverting the `&& !search_incomplete` gate),
`c2_genuine_infeasible_still_reported_when_search_complete` (the rigorous path is
not weakened), `c2_deferred_with_incumbent_reports_feasible_not_infeasible`
(orthogonal guard), and end-to-end `c2_end_to_end_genuine_infeasible_unaffected`
via `solve_milp`. Gates: `cargo test -p discopt-core` 390 passed / 0 failed;
`cargo clippy -p discopt-core --lib` 0 warnings; `maturin develop --release` built;
`pytest -m smoke` 193 passed / 1 skipped; adversarial suite 10 passed;
`incorrect_count = 0`. PR: (fix-c2-milp-driver-deadline-infeasible).

---

## C-19 (P1, FIXED) — `relax_tan` classifies pole-straddling intervals as a single convex/concave branch → secant across a pole

**Area:** `python/discopt/_jax/mccormick.py:393-436`. Centers on the nearest
inflection `center = round(mid/π)·π` and classifies via `lb ≥ center` /
`ub ≤ center`, never checking that `[lb,ub]` lies within one continuous branch
`(kπ−π/2, kπ+π/2)`. Wired into the compiler at `relaxation_compiler.py:981` —
default path for any model containing `tan`.

**Mechanism:** for `[1.4, 1.8]` (straddles the pole at π/2≈1.5708): `center=π`,
`ub ≤ π` ⇒ "concave half", so cv = secant through `tan(1.4)=+5.8` and
`tan(1.8)=−4.3` — a line across a pole, not a valid underestimator. Invalid
envelope → invalid bound → potential false certificate on tan-containing models.

**Reproduce/confirm:** unit test: `cv, cc = relax_tan(...)` on `[1.4, 1.8]`;
evaluate at x=1.5 (`tan(1.5)≈14.1`): buggy cv exceeds the function on part of the
branch / cc falls below it — assert the envelope property `cv ≤ tan ≤ cc` fails on
a grid restricted to each continuous branch.

**Fix:** detect any pole in the open interval (`∃k: lb < kπ+π/2 < ub`) and abstain
(return `(−inf, +inf)` envelopes, matching how other relaxations abstain), leaving
FBBT/branching to shrink the box below pole width.

**Done criteria:** grid property test — for random intervals (pole-straddling and
not), `cv ≤ tan ≤ cc` holds wherever tan is defined and finite envelopes are
returned; a tan-containing MINLP with pole-straddling initial bounds solves to the
known optimum; standing gates pass.

**Log:** 2026-07-03 — CONFIRMED then FIXED (status open→fixed). Direct-call repro
on the pre-fix code: `relax_tan` on `[1.4,1.8]` (straddles π/2≈1.5708) draws a
secant through `tan(1.4)=+5.8` and `tan(1.8)=−4.3` across the pole → `max(cv−f) ≈
+2.7e5` on the branch (invalid underestimator); the mirror `[−1.8,−1.4]` gives
`max(f−cc) ≈ +2.7e5`. **Fix:** detect the branch spanning the box (nearest
inflection `center=k·π`, bounding poles `center±π/2`) and abstain — return the
no-information envelope `(−inf,+inf)` — whenever an endpoint reaches or crosses a
pole (`pole_free = (lb > center−π/2) & (ub < center+π/2)`). Pole-free boxes keep
the tight branch envelope. Also hardened `_secant` so the degenerate `lb==ub`
branch never evaluates `0/0` (was only guarded *after* the division). Applied to
BOTH backends — the JAX `_jax/mccormick.py` (in scope) and the ported
`_numpy/mccormick.py` (the NM-4 twin noted in the 2026-07-03 reconciliation) — so
the class is closed on both. **After:** every pole-straddling box abstains (no
finite crossing envelope); all pole-free branches (incl. k=±1 at `[3.3,4.6]`,
`[−4.6,−3.3]`) remain sound to 1e-7. **Regression test:**
`python/tests/test_tan_pole_envelope_c19.py` (`@pytest.mark.smoke`, sub-second,
both backends): literal `[1.4,1.8]` repro, pole-straddling never-crosses,
pole-free stays-tight, and a random-sub-box property test over `[−5,5]` — 14 red
before, all green after. **Gates:** targeted `-k "tan or div or envelope or relax
or mccormick or convex"` 1366 passed / 0 failed; ruff + format clean; mypy clean
on the changed modules. PR: (pending).

---

## C-5 (P1) — `.nl` parser silently rewrites floor/ceil/round/trunc to identity and intdiv to real division

**Status:** fixed (PR #447).

**Area:** `crates/discopt-core/src/nl_parser.rs:578-611` (`parse_opcode`):
`o13` floor → arg unchanged (`:602-606`); `o14` ceil → arg unchanged (`:607-611`);
`o57` round → left operand (`:588-594`); `o58` trunc → left operand (`:595-601`);
`o55` intdiv → `BinOp::Div` (`:578-587`). No warning or log anywhere in the file.

**Mechanism:** `y = floor(x)` parses as `y = x`; `7 intdiv 2` becomes `3.5`. The
solver then correctly solves the *wrong* problem and labels the result optimal.
The current 61-file test corpus (`python/tests/data/minlplib_nl/`) uses none of
these opcodes — verified — so the phase gate cannot catch this; it fires the moment
a broader MINLPLib slice is ingested.

**Why P1:** silent problem substitution; latent (zero current-corpus exposure).

**Reproduce/confirm:** write a 3-line AMPL model using `floor(x)` (or hand-craft a
tiny text `.nl` with opcode `o13`), parse it, and evaluate the resulting expression
at x=1.7: buggy result 1.7, correct behavior = parse error (until floor/ceil get
real IR support).

**Fix:** replace each silent branch with a hard error (`NlParseError::
UnsupportedOpcode(name)`) carrying the opcode name and a message that the operator
is not representable. Loud refusal, never silent misparse. (Real floor/ceil support
would need IR + relaxation work — out of scope here; the fix is the refusal.)

**Done criteria:**
- New parser tests: each of o13/o14/o55/o57/o58 produces the error, message names
  the operator.
- The 61-file corpus still parses (confirms zero regression).
- Standing gates pass.

**Log:**
- 2026-07-03 — **Confirmed** then **fixed**. Added five Rust unit tests
  (`test_{floor,ceil,round,trunc,intdiv}_opcode_refused*` in `nl_parser.rs`) that
  feed a hand-built text `.nl` whose objective is the bare opcode over `v0`/`v1`.
  Fail-before: on the pre-fix silent-rewrite code all five return `Ok` with the wrong
  expression (floor/ceil/round/trunc → identity, intdiv → real `Div`); proven red by
  temporarily reverting only the five `parse_opcode` branches (5 failed, "got Ok").
  Fix: added `NlParseError::UnsupportedOpcode { name, opcode }` with a Display message
  that names the operator and states the parse is refused rather than substituting a
  different operator; replaced the five silent branches (`o13`/`o14`/`o55`/`o57`/`o58`)
  with that error (the error aborts the parse, so operands are deliberately not
  consumed). No IR/relaxation support was added — the sound fix per CLAUDE.md §3 is the
  loud refusal. Pass-after: all five tests green.
- Verified end-to-end via the Python binding path (`discopt.modeling.core.from_nl` →
  `_rust.parse_nl_file` → `parse_nl`): a floor `.nl` now raises
  `ValueError: unsupported .nl operator 'floor' (opcode o13) …` instead of parsing
  silently. (Note: the shared dev venv's `discopt.pth` resolves the package to a
  different worktree; the Python gates were re-run with `PYTHONPATH` pinned to this
  worktree so they exercise this fix's freshly-built `_rust.so`.)
- Gates: `cargo test -p discopt-core` 386 lib + 4 integration + 1 doctest pass,
  no warnings; `cargo clippy -p discopt-core --lib` clean; `maturin develop --release`
  builds; `pytest -m smoke` 193 passed / 1 skipped; adversarial suite
  (`test_adversarial_recent_fixes.py`) 10 passed; all 61 corpus files still parse.
- Regression tests named above are committed and fast (sub-millisecond, direct
  `parse_nl` calls; no `Model.solve()`).

---

## C-6 (P1) — Modeling API silently clamps integer variables to [0, 1e6]

**Area:** `python/discopt/modeling/core.py:1712-1713` (`integer(..., lb=0,
ub=1e6)` defaults); indexed path `:1743`; documented at `:67`. The `.nl` path is
NOT affected (defaults ±inf, clamps only Binary to [0,1] — `nl_parser.rs:1168-1171`).

**Mechanism:** `m.integer("n")` yields `0 ≤ n ≤ 1_000_000` with no diagnostic. A
model whose optimum needs a negative or >1e6 integer is silently cut off and the
truncated-box optimum is reported as certified optimal.

**Reproduce/confirm:** `m.integer("n"); m.minimize(n); m.solve()` → reports n=0
optimal instead of unbounded; and a model with true optimum at n=-3 (e.g.
`minimize (n+3)**2`) reports n=0.

**Fix (decision for maintainer, pick one and record):** (a) default `lb=None,
ub=None` → ±inf and require the B&B layer to reject/branch unbounded integers
explicitly, or (b) keep the finite default but emit a loud warning whenever a
default-bounded integer variable's optimum lands ON the default bound, or (c) keep
defaults, require explicit bounds (deprecation path). Minimum acceptable fix: (b) —
never silently report a default-box-active optimum as certified.

**Done criteria:** the two reproduction models either solve correctly (option a),
warn visibly (option b), or raise (option c); a test locks the chosen behavior;
docs updated; standing gates pass.

**Log:** 2026-07-03, status open→fixed (branch `fix-c6-integer-var-silent-clamp`,
PR #448). **Confirmed** the mechanism precisely: user-provided `lb`/`ub` were
*already* honored exactly (a repro with `integer("n", lb=-5, ub=10)` → stored
`[-5, 10]`; `ub=5e6` → stored `5e6`) — the bug was the *silent* substitution of
the finite default `[0, 1e6]` whenever a bound was left unspecified, so a model
needing a negative or >1e6 integer was truncated with no diagnostic. Two of the
six regression assertions were **red before / green after**
(`test_unspecified_integer_bounds_warn_loudly`,
`test_partial_bounds_default_only_missing_side`); the four
bounds-honored assertions were green on both sides (they lock in that the fix
does not regress the already-correct explicit-bound path).

**Fix — chose option (b), refined per the loud-default principle:** changed the
`integer(...)` signature defaults from `lb=0, ub=1e6` to `lb=None, ub=None`, and
added `Model._resolve_integer_defaults`, which substitutes the finite fallbacks
(`_INTEGER_DEFAULT_LB=0.0`, `_INTEGER_DEFAULT_UB=1e6`) **only for a `None`
(unspecified) side** and emits a `UserWarning` naming the variable and the imposed
range. A user-provided bound (any explicit value, including `0` / `1e6`) is passed
through unchanged and never warns — the default can never override or narrow a
declared bound. The finite fallback is kept (not ±inf per option (a)) so B&B still
receives a bounded integer domain; making the substitution loud rather than
removing it avoids touching any downstream solver assumption about integer
boundedness. The indexed (`over=`) path already resolved `None`→default via
`resolve_indexed_values`, so it inherits the same behavior; `_make_indexed_var`
now receives the shared `_INTEGER_DEFAULT_*` constants. Overload signatures updated
to `Optional[...]`. Docstrings and the `VarType.INTEGER` doc (`core.py:67`) updated
to describe the honored-exactly / loud-default contract. Internal `.integer(...)`
callers (`llm/tools.py`, `solvers/milp_pounce.py`) all pass explicit bounds, so
none newly warn.

**Regression test (named):** `python/tests/test_c6_integer_default_bounds.py` —
six `@pytest.mark.smoke` unit tests calling the modeling API directly (sub-second,
no `Model.solve()`): negative bounds honored exactly, large ub not clamped,
explicit bounds do NOT warn, unspecified bounds WARN loudly (with finite fallback),
partial (lb-only) warns for the defaulted side while honoring lb exactly, and the
indexed path honors negative bounds. Encodes the class (silent default box), not a
named instance.

**Gates:** `pytest -m smoke` 345 passed/1 skipped; adversarial
`test_adversarial_recent_fixes.py -m slow` 10 passed; `pytest -k "integer or
modeling or bound or variable"` 617 passed/1 skipped/2 xfailed; C-6 file 6 passed.
`ruff check` + `ruff format --check` clean on both files. No Rust touched.
`incorrect_count=0` (no false certificate; nothing weakened — the fix strictly adds
a diagnostic and honors input). mypy blocked only by a **pre-existing** env mismatch
(installed numpy ships Python-3.12 `type` statement syntax in `__init__.pyi:737`
while `[tool.mypy] python_version = "3.10"`) — reproduces identically on the
unmodified tree, unrelated to this change.

---

## C-7 (P2) — `.nl` defined variables (V segments) are parsed then discarded → hard error on any reference

**Area:** `nl_parser.rs:981-995` (V-segment handler parses the defining expression
into `let _expr = …` and drops it; the defined var is never pushed to `var_nodes`);
`:345-354` (`parse_expr` errors on `v<idx>` with `idx >= n_vars`).

**Mechanism:** loud, not silent (verified `var_nodes` never grows → no silent
misalignment). But AMPL emits common subexpressions constantly, so this rejects a
large slice of real MINLPLib. Also the subject of certification-gap-plan Phase 4
(preserving the sharing); this issue is the correctness-adjacent minimum: *inline*
them so instances parse.

**Reproduce/confirm:** hand-craft a tiny `.nl` with one V segment referenced by a
constraint (or export one from AMPL/Pyomo); `parse_nl` currently errors.

**Fix:** build defined-var expr = linear part + nonlinear body, push to
`var_nodes` so later `v<idx>` references resolve to the inlined subexpression.
(Sharing/CSE is the Phase 4 upgrade; inlining is correct now.)

**Done criteria:** the crafted V-segment instance parses and evaluates to the same
values as an independent evaluation (e.g. Pyomo) at 3 random points; corpus still
parses; standing gates pass.

**Log:**
- 2026-07-03 — **CONFIRMED then FIXED.** Confirmed with a hand-crafted `.nl`
  carrying one defined variable `V2 = 2·v0 + v1²` referenced by the objective as
  `min v2`: pre-fix `parse_nl` errored `"variable index 2 out of range"` (the V
  handler read the defining expression into `let _expr = …` and dropped it, never
  growing `var_nodes`). No MINLPLib instance (in-repo 61-file corpus nor the full
  ~4,800-file snapshot) emits V segments — MINLPLib is exported without common
  subexpressions — but AMPL/Pyomo direct exports emit them routinely, so this
  rejected a large real slice.
- **Fix** (`nl_parser.rs`, `b'V'` segment handler): the defined var is now INLINED.
  Read the `n_linear` `(var_index, coeff)` pairs and the nonlinear body, build
  `linear_part + body` (same +1/−1 coeff shortcuts as `build_linear`, dropping a
  zero-constant body), and push the resulting `ExprId` onto `var_nodes` at the
  defined var's index. `.nl` numbers defined vars sequentially from `n_vars`, so
  the push lands at the next `var_nodes` slot; a later `v<idx>` (idx ≥ n_vars)
  reference resolves to the inlined subexpression. Chained references (a defined
  var referencing an earlier defined var) work because the body is parsed against
  the already-grown `var_nodes`. Out-of-sequence / redefining indices and linear
  terms referencing not-yet-defined indices are refused with a `Parse` error
  rather than guessed (never a silent mis-map). CSE/sharing of the inlined nodes
  is handled for free by the arena's Phase-4 interning; the correctness fix is the
  inlining itself.
- **Regression tests** (`nl_parser::tests`, fail-before/pass-after verified):
  `test_defined_variable_inlined_and_evaluated` (`2x + y²` checked at 3 points:
  (3,4)→22, (0,5)→25, (−1,2)→2) and `test_defined_variable_chained_reference`
  (`V1 = v0+1`, `V2 = v1·v1`, objective `v2 = (v0+1)²`, at v0=3→16). Both error
  on the pre-fix parser and pass after. Corpus lock `test_minlplib_corpus_all_parse`
  (all 61 in-repo `.nl` still parse) added in the same PR.
- **Gates:** `cargo test -p discopt-core` 408 lib + 4 determinism + 1 doctest
  green; clippy clean; `rustfmt --check` clean on `nl_parser.rs`; `maturin develop
  --release` OK; smoke 478 passed / 1 skipped; adversarial 10 passed;
  `incorrect_count` unchanged (0). PR: (this PR).

---

## C-8 (P2) — Common `.nl` opcodes unhandled → hard `UnknownOpcode` error

**Area:** `nl_parser.rs:620` (fallthrough arm). Missing: `o76` (1POW `x^n`), `o77`
(2POW `x^2` — AMPL emits this routinely), `o78` (CPOW `c^x`), `o48` (atan2),
`o11`/`o12` (min/max lists), `o35` (if). Loud failure; ingestion gap only.

**Reproduce/confirm:** craft or export a `.nl` using `x^2` emitted as `o77`;
parse → `UnknownOpcode`.

**Fix:** add arms — o76/o77/o78 → `BinOp::Pow` with the constant on the right/left
as appropriate; o11/o12 → Min/Max. atan2 requires a `MathFunc::Atan2` IR addition
(the JAX layer already has atan2, `dag_compiler.py:229` — close the asymmetry) —
if that's too large, error with a *named* UnsupportedOpcode like C-5. `o35` (if):
error with named message unless trivially constant-guarded.

**Done criteria:** parser tests for each added opcode (value-checked against
direct evaluation); named errors for any still-unsupported ones; standing gates.

**Log:**
- 2026-07-03 — **CONFIRMED then FIXED.** Confirmed every listed opcode hit the
  `_ => UnknownOpcode` fallthrough and aborted the parse (fail-before tests red).
  Of the whole ~4,800-file MINLPLib snapshot only `o11` occurs (once, `fuzzy.nl`);
  the rest are emitted by AMPL/Pyomo direct exports (o77 for `x**2` routinely).
  Authoritative opcode numbers cross-checked against MathOptInterface.jl's
  `opcode.jl` and Couenne's `readnl/nl2e.cpp` (OP1POW=`pow(L,R.const)`,
  OP2POW=`pow(L,2)`, OPCPOW=`pow(L.const,R)`; MINLIST=11/MAXLIST=12 use the o54
  count-line wire format).
- **Fix** (`nl_parser.rs::parse_opcode`):
  - `o76`/`o77`/`o78` → `BinOp::Pow` with the correct operand order (o76 base then
    constant power; o77 base with a synthesized `Constant(2.0)`; o78 constant base
    then exponent — constants arrive as ordinary `n<val>` leaves).
  - `o11`/`o12` → folded into a **left-nested tree of binary** `MathFunc::Min`/`Max`
    rather than a single n-ary `FunctionCall`. Rationale (soundness): every IR
    consumer (`evaluate`, `evaluate_node`, FBBT) treats `Min`/`Max` as strictly
    binary (`args[0]`,`args[1]`) — an n-ary call would silently drop `args[2..]`, a
    silent misparse. The nested-binary fold is exactly equivalent and uses only the
    sound binary semantics; no evaluator/FBBT/relaxation change needed (keeps the
    fix inside the parser card's scope). Zero-arg lists are refused.
  - `o48` atan2 and `o35` if-then-else → **loud** `UnsupportedOpcode` (no sound IR
    representation; Atan2 is not in `MathFunc` and single-arg `atan` would be a
    silent misparse), mirroring the C-5 refusal policy. Operands are deliberately
    not consumed (the error aborts the parse).
- **Regression tests** (`nl_parser::tests`, fail-before/pass-after verified):
  `test_o77_2pow_is_square_not_unknown` (3²=9), `test_o76_1pow_is_power_with_constant_exponent`
  (2³=8), `test_o78_cpow_is_constant_base_to_var_power` (2³=8),
  `test_o11_minlist_is_minimum` (min{5,2,8}=2), `test_o12_maxlist_is_maximum`
  (max{5,2,8}=8 — this is the one that caught the n-ary-vs-binary trap and forced
  the nested-fold), `test_o48_atan2_refused_not_misparsed`,
  `test_o35_if_refused_not_misparsed` (both assert the named `UnsupportedOpcode`).
- **Gates:** same run as C-7 (shared PR): core 408+4+1 green, clippy clean, fmt
  clean on the file, smoke 478/1-skip, adversarial 10, `incorrect_count`=0.
  PR: (this PR).

---

## C-9 (P2) — Integer-block classification for the `nlvo > nlvc` header case is unverified

**Area:** `nl_parser.rs:766-780`. The objective-only nonlinear-integer block is
placed at `[nlvc, nlvo)` with size `nlvo − nlvc`, but the AMPL convention sizes
the objective-only group as `nlvo − nlvb`; these differ when `nlvb < nlvc < nlvo`.
The branch was added for `ex1252a`, which is not in the local corpus — unexercised.
Misclassification = an integer variable treated continuous (wrong answer) or vice
versa (over-constrained).

**Reproduce/confirm:** obtain/craft an instance with `nlvb < nlvc < nlvo` and
nonlinear integer vars in objectives only (ex1252a from MINLPLib); parse with
discopt and with a reference reader (Pyomo/ASL); diff the per-variable
integrality vectors.

**Fix:** whatever the diff says — align to the reference reader; add the instance
(or a minimized version) to the corpus.

**Done criteria:** integrality vector matches the reference on the new corpus
instance + 2 more header-shape variants; standing gates pass.

**Log:**
- 2026-07-03 — **ALREADY FIXED (by PR #310), now CONFIRMED + LOCKED.** The
  `nlvo > nlvc` objective-only integer block is already correctly typed in the
  code: `nl_parser.rs` sizes the objective-only nonlinear group as `[nlvc, nlvo)`
  (not the wrong `(nlvo − nlvb)`-sized formula) when `nlvo > nlvc`, and stamps its
  last `nlvoi` entries `Integer`. This landed in `fix(nl-parser): type
  objective-only integer vars when nlvo > nlvc (false-feasible) (#310)` on
  2026-06-23 — for `ex1252a`, whose 3 binaries (b22..b24) the old formula left
  Continuous, relaxing integrality and admitting a false-feasible incumbent. The
  card was never flipped from `open`.
- **Confirmation.** The card's real bug is the *missing test*, not a live misparse:
  a fresh regression test (below) that encodes the `nlvo>nlvc` header shape passes
  on the current code and — verified by reverting the `#310` branch to the old
  `(nlvo − nlvb)` sizing — fails (the objective-only integers slip to Continuous).
- **Regression test** (`nl_parser::tests::test_nlvo_gt_nlvc_objective_only_integer_block_typed`):
  header shape `nlvb=1 < nlvc=2 < nlvo=4`, `nlvoi=2`; asserts vars 2,3 are
  `Integer` (the obj-only nl integer tail) and 0,1 `Continuous`. This is the
  specific false-feasible class (integrality silently dropped), not the named
  `ex1252a` instance, so a future refactor reintroducing the mis-sizing trips it.
- **Gates:** shared PR run with C-7/C-8 (core 408+4+1, clippy/fmt clean, smoke
  478/1-skip, adversarial 10, `incorrect_count`=0). Status open→fixed. PR: (this PR).

---

## C-10 (P2) — LP-spatial GMI cuts appended without the rhs safety margin every other path uses

**Area:** `python/discopt/_jax/lp_spatial_bb.py:131-134` (`_separate_node_cuts`
appends raw Rust GMI coeffs/rhs). Contract in `gomory.rs:31`: validity holds "up
to machine precision"; a caller-side rhs margin absorbs the remainder. All other
consumers add `~1e-7·(1+|row|₁)` (`solver.py:10099`, `cmir_cuts.py:91`); this path
adds none. Off by default (`root_cut_rounds=0`, per-node `_rounds` hard-set 0 at
`lp_spatial_bb.py:401`) — live only when cut rounds are enabled.

**Reproduce/confirm:** static read of the four sites (the margin discipline is the
documented contract; no dynamic repro needed).

**Fix:** apply the same margin formula when appending (`rhs += 1e-7*(1+‖row‖₁)`
for the ≤ form), matching `_augment_lpdata_with_gomory_cuts`.

**Done criteria:** margin applied; a validity test on the nvs17-class engine with
cut rounds enabled (enumerate feasible integer points in a small box; none
violated); standing gates pass.

**Log:**
- 2026-07-03 — **CONFIRMED then FIXED.** Confirmed dynamically (not just by static
  read): drove the incremental-McCormick node LP of a small all-integer bilinear
  model (`a*b + c >= 10`, box `[0,6]×[0,5]×[0,4]`) through the actual
  `_separate_node_cuts` GMI branch and reproduced the raw crossover-vertex GMI cut
  independently. The emitted GMI cut's `<=` rhs was **byte-identical** to the raw
  Rust GMI rhs (`-6.000000000000001`) — no safety margin, exactly as the card
  predicts. Every other GMI consumer relaxes by `1e-7·(1+‖row‖₁)`; this path did
  not, so a cut whose boundary passes through a feasible integer point could shave
  it under the ~1e-12 float error the crossover vertex carries.
- **Fix (per the card sketch).** `lp_spatial_bb.py:_separate_node_cuts` GMI branch
  now adds `margin = 1e-7*(1+‖row‖₁)` to each cut's `<=` rhs before appending —
  matching `solver.py:_augment_lpdata_with_gomory_cuts` / `cmir_cuts.py`. Sound by
  construction: the margin only ever moves the cut *outward* (relaxes the
  constraint), so it can never remove a feasible point; it only forgoes cutting an
  ~1e-7-thin sliver. The cMIR/aggregation branches already carried their own
  margins and were left untouched. No validity check weakened; no other path
  changed.
- **Regression tests** (`python/tests/test_lp_spatial_bb.py`, `@pytest.mark.smoke`,
  sub-second, direct calls into `_separate_node_cuts` — no full solve; verified
  RED-before/GREEN-after by stashing the fix):
  `test_c10_lp_spatial_gmi_cut_carries_safety_margin` (reconstructs the raw GMI
  cut and asserts the emitted rhs is relaxed outward by *exactly* the margin — RED
  pre-fix: emitted == raw, no margin) and `test_c10_no_feasible_integer_point_is_cut`
  (encodes the class invariant: enumerates every integer-feasible point of the
  model and asserts no emitted node cut violates it). The first test pins the
  emission contract so any future revert to a raw-rhs append trips CI immediately.
- **Gates:** C-10 tests 2 passed (RED-before confirmed via `git stash`); cut suite
  (`test_{cutting_planes,cover_cuts,conflict_cuts,clique_cuts,constraint_cuts,rlt_cuts,auto_cut_policy,cut_recognizer,lp_spatial_bb}`)
  135 passed; `-k "mccormick or relax or spatial" -m "not slow"` 693 passed / 1
  skipped; `pytest -m smoke` 480 passed / 1 skipped; adversarial
  `test_adversarial_recent_fixes.py -m slow` 10 passed; `ruff check` +
  `ruff format --check` clean; pre-commit `mypy` passed. No Rust touched (no
  `cargo test`). `incorrect_count` unchanged (0 failures across all suites; no
  correctness assertion weakened). PR: (pending).

**Status:** open → fixed.

---

## C-18 (P1) — `mccormick_bounds="midpoint"` returns the underestimator's *value at the midpoint*, which is not a lower bound

**Area:** `python/discopt/_jax/mccormick_nlp.py:57-64` (`evaluate_midpoint_bound`:
`cv, cc = obj_relax_fn(mid, mid, lb, ub); return float(cv)`); the compiled leaf
returns point values (`relaxation_compiler.py:409-410`), so the result is
`u(mid)`, not `min_box u`. Consumed at `solver.py:5213-5224` → `convex_lb` → for
nonconvex models `nlp_lb = convex_lb` (`:5270`) → `result_lbs[i]` (`:5282`).
**Reachability:** opt-in only — `"auto"` never selects midpoint
(`solver.py:4095-4145`) — but the mode is documented as a valid weak bound
(`mccormick_nlp.py:11-13`).

**Mechanism:** `u(mid) ≤ f(mid)` does not imply `u(mid) ≤ min_box f`. Worked
counterexample: objective `x²` on `[1,3]`: envelope cv is `x²` itself,
`u(2) = 4`, but `min_box x² = 1`. The returned "lower bound" of 4 can fathom a
node whose true minimum beats the incumbent → wrong answer for any user who
selects the documented mode.

**Reproduce/confirm:** unit test on the worked example — assert
`evaluate_midpoint_bound(...) == 4 > 1`; then an end-to-end solve of a small
nonconvex model with `mccormick_bounds="midpoint"` that certifies a wrong optimum.

**Fix (pick one):** (a) compute an interval lower bound of the underestimator over
the box (evaluate the relaxation in interval mode / at the box that minimizes cv),
(b) actually minimize cv (it is convex — one cheap convex solve), or (c) remove
the mode and error on selection. Minimum acceptable: (c) — a documented bound mode
must never return a non-bound.

**Done criteria:** the worked example returns ≤ 1 (or the mode is gone); the
end-to-end model certifies the true optimum or refuses the mode; docs updated;
standing gates pass.

**Log:**
- 2026-07-03 — **CONFIRMED then FIXED.** Confirmed the non-bound directly: for
  objective `x²` on `[1,3]`, `evaluate_midpoint_bound(...)` returned **3.0**
  (the compiled McCormick secant envelope's `cv` at the midpoint x=2), while the
  true box minimum is **1.0** (at x=1). 3.0 > 1.0, so the returned value is NOT a
  valid lower bound — it can fathom the node holding the true optimum → false
  "optimal" for any user who selected the documented mode. (The card predicted 4.0
  under the assumption `cv = x²`; the compiler's actual secant envelope gives 3.0,
  but the class is identical: `u(mid) > min_box f`.) The value was consumed as a
  bound at the two McCormick objective-bound seams (`solver.py` batch ~5136 and
  serial ~5361) via `max()` into `result_lbs`/`convex_lb`/`nlp_lb`.
- **Fix — option (c), remove the mode and refuse loudly.** The only sound cheap
  way to turn the convex underestimator into a bound is to MINIMIZE it over the
  box, which is exactly what `mccormick_bounds="nlp"` already does
  (`solve_mccormick_relaxation_nlp`). A sound "midpoint" mode would just duplicate
  "nlp", so the mode was removed rather than reimplemented:
  - `solver.py` now validates `mccormick_bounds ∈ {auto, nlp, lp, none}` at the
    top of `solve_model` (next to the `nlp_solver` check); `"midpoint"` raises a
    `ValueError` naming C-18 and the worked counterexample, and any other unknown
    value is also rejected (previously an unknown value silently fell through).
  - `evaluate_midpoint_bound` / `evaluate_midpoint_bound_batch` (the non-bound
    helpers) and the `_midpoint_batch_cache` were **deleted** from
    `mccormick_nlp.py`; the two now-dead consumer branches in `solver.py` and the
    `_mc_mode in ("midpoint","nlp")` guard were collapsed to `"nlp"`-only.
  - Docstrings updated (`solve_model` `mccormick_bounds` param, `mccormick_nlp`
    module header, `convex-relaxation-expert` skill doc).
- **Regression tests** (`python/tests/test_mccormick_bounds.py::TestC18MidpointNotABound`,
  fail-before/pass-after verified by stashing the fix — the rejection and
  helpers-gone tests fail on pre-fix source): `test_midpoint_value_exceeds_true_box_minimum`
  (reconstructs `u(mid)=3.0 > 1.0` for x² on [1,3] straight from the compiled
  relaxation — the mechanism, fix-agnostic), `test_midpoint_mode_is_rejected_loudly`
  (`solve(mccormick_bounds="midpoint")` raises `ValueError` matching "C-18"),
  `test_unknown_mccormick_bounds_rejected` (typos rejected), and
  `test_evaluate_midpoint_helpers_are_gone` (the unsound helpers are deleted, not
  just unwired). The pre-existing `TestMidpointBounds` class (which exercised the
  removed non-bound and even asserted an end-to-end "optimal") was removed; the
  three integration tests that passed `"midpoint"` were repointed to the sound
  `"nlp"`/`"none"` modes; `test_nlp_bound_finds_minimum_of_underestimator` now
  derives its `cv(mid)` reference directly instead of via the deleted helper.
- **Gates:** `test_mccormick_bounds.py` 15 passed; `pytest -m smoke` 193 passed /
  1 skipped; adversarial `test_adversarial_recent_fixes.py -m slow` 10 passed; `ruff check` +
  `ruff format --check` clean on all touched files; `pre-commit run mypy` passed.
  No Rust touched (no `cargo test`). The broad `-k "midpoint or mccormick or bound
  or nlp"` filter showed 39 failures that are **pre-existing and unrelated** —
  `TestSchurPassthrough` (POUNCE Schur-block) fails identically with this fix
  stashed, and the `test_relaxation_coverage`/`test_qp_pounce` cases pass in
  isolation (test-ordering artifacts); `incorrect_count` unaffected (no correctness
  assertion weakened — the fix strictly REMOVES a false-bound path). PR: #449.

**Status:** open → fixed.

---

## C-20 (P2) — Watch-list FBBT (`fbbt_fixed_point`) declares infeasibility with zero tolerance

**Area:** `crates/discopt-core/src/presolve/fbbt_fp.rs:148` (and `:181`):
`if body_bound.intersect(&output_bound).is_empty()` — strict `lo > hi`. The main
engine deliberately uses `is_empty_beyond(FEAS_TOL)` for the identical check
(`fbbt.rs:1138-1140, 1229-1231`) and documents why the strict check is unsound
(`fbbt.rs:74-87`: GDP hull perspective forms leave ~1e-8 residuals at integer
faces). **Off by default** (`run_root_presolve(..., fbbt_fixed_point=False)`) —
fires only when explicitly enabled, hence P2 despite being certificate-class.

**Mechanism:** an eps-residual equality (body forward-propagates to
`[c+5e-7, c+5e-7]` vs rhs `c`) is declared infeasible by the watch-list engine
while the main engine correctly treats it as feasible → false "infeasible" when
the opt-in engine is on.

**Reproduce/confirm:** Rust unit test with an eps-residual equality (mirror the
`fbbt.rs:74-87` scenario): `fbbt` reports feasible, `fbbt_fixed_point` reports
infeasible (buggy).

**Fix:** use `.is_empty_beyond(FEAS_TOL)` at `fbbt_fp.rs:148` (import already
present), matching `fbbt.rs`.

**Done criteria:** the differential test passes (both engines agree feasible);
existing fbbt_fp tests green; standing gates pass.

**Log:**
- 2026-07-03 — **CONFIRMED then FIXED.** Confirmed via a new differential test:
  an eps-residual equality (body forward-props to a point 5e-7 < FEAS_TOL off the
  rhs) is feasible in the main engine (`fbbt`) but the watch-list engine reported
  it infeasible (strict `lo > hi`). Verified fail-before by reverting the guards.
- **Fix.** Both infeasibility-declaring sites in `presolve/fbbt_fp.rs` now use
  `.is_empty_beyond(FEAS_TOL)` instead of `.is_empty()`: the constraint-body check
  (`:148`) — matching `fbbt.rs:1138-1140/:1229-1231` exactly — and the per-variable
  post-backprop check (`:181`), which the main engine doesn't even have (so it was
  strictly harsher). Added `FEAS_TOL` to the `super::fbbt` import.
- **Regression tests** (`presolve::fbbt_fp::tests`):
  `c20_eps_residual_equality_matches_main_engine_feasible` (differential vs `fbbt`;
  fails before the fix) and `c20_real_infeasibility_still_detected` (a residual
  4.0 ≫ FEAS_TOL is still caught — guards against over-loosening).
- **Gates:** `cargo test -p discopt-core` 373 lib + 4 + 1 doctest green, no
  warnings. Python default path is unaffected (`fbbt_fixed_point` is off by
  default); smoke 211 + adversarial 10 green locally. PR: #403.

---

## C-21 (P2) — Incremental-McCormick soundness gate validates only on nonnegative boxes

**Area:** `python/discopt/_jax/incremental_mccormick.py:171-186` (`_validate`):
every validation box has `lb ≥ 0` (`lb=0` boxes at `:173`, `lb=arange(n)` at
`:176`), while real nodes carry negative/zero-spanning bounds. The audit verified
the current closed forms ARE sound on negative and zero-spanning boxes and that
support-based row discovery cannot cross-map (aux columns unique) — so **not
currently unsound**; the issue is that the gate whose docstring promises "can
never change a result" (`:13-18`) would not catch a future divergence in exactly
the sign regimes that dominate real nodes. This gate is also the safety mechanism
certification-gap-plan Phase 1 (T1.2) leans on — harden it before that work.

**Reproduce/confirm:** static read of `_validate`'s box construction.

**Fix:** add negative-lb, zero-spanning, and mixed-sign boxes (and a degenerate
lb==ub box) to the validation set; keep the count small (construction-time cost).

**Done criteria:** `_validate` exercises ≥ 4 sign regimes; a deliberately-broken
`_bilinear_rows` sign flip is caught by the gate (mutation test); standing gates
pass.

**Log:**
- 2026-07-03 — **CONFIRMED (not unsound), then HARDENED.** Static confirm of
  `_validate`'s box construction: for every variable with `_root_sign >= 0`
  (positive *and* zero-spanning), each validation box used `lb >= 0` (`lo = 0.0`
  or `lo = 0.5+0.3*i`); negative-definite vars used `ub <= 0`. So a variable whose
  root box **strictly spans zero** — which is what dominates real bilinear nodes —
  was only ever probed on its nonnegative sub-boxes. No box was zero-spanning
  (`lb<0<ub`), negative-lb-for-a-spanning-var, or degenerate (`lb==ub`). The
  closed forms themselves ARE sound on those regimes (verified: the hardened gate
  passes on all of them), so this was a **gate-coverage** gap, not a live
  soundness bug — no P0 escalation.
- **Fix** (`incremental_mccormick.py`): split the old `>= 0` branch so
  *sign-definite* vars still stay in their reachable root regime, but **spanning**
  vars (`_root_sign==0`, which carry no monomial — only bilinear rows) are now
  driven through six per-trial profiles: `shift_pos`, `zero_lb` (`lb==0<ub`),
  `span` (`lb<0<ub`), `neg` (`ub<0`), `span_wide`, and `degen` (`lb==ub`). Box
  count unchanged at 6 (construction-time cost neutral). Added `_box_sign_regime`
  + `_validated_regimes` so the gate records which regimes it exercised; no solver
  math changed (row builders, patch, aux bounds untouched). On a two-spanning-var
  bilinear model the set now covers 5 regimes {pos, zero_lb, span, neg, degen}.
- **Regression tests** (`python/tests/test_incremental_mccormick_node.py`):
  `test_validate_exercises_at_least_four_sign_regimes` (asserts span/neg/degen/
  zero_lb all present, ≥4 total) and the **mutation test**
  `test_validate_catches_negative_box_sign_flip_mutation` — monkeypatches
  `_bilinear_rows` to clip negative lower bounds to 0 (identity on `lb>=0`,
  wrong on negative/spanning boxes); the hardened gate rejects it (`_inc is None`).
  Verified FAIL-before/PASS-after by stashing the `_validate` change: both new
  tests fail (mutation slips through the old `lb>=0`-only set), pass with the fix.
- **Gates:** `test_incremental_mccormick_node.py` 9 passed; `-m smoke` 211 passed /
  1 skipped / 0 failed; adversarial `test_adversarial_recent_fixes.py -m slow` 10
  passed; `ruff check` + `ruff format --check` clean; `pre-commit run mypy` passed.
  No Rust touched (no `cargo test`). `incorrect_count` unchanged (0 failures across
  all suites; no correctness assertion weakened). PR #404.

**Status:** open → fixed.

---

## C-15 (P2) — `run_obbt` (model-linear variant) tightens bounds to the raw LP vertex without the safe-bound clamp

**Area:** `python/discopt/_jax/obbt.py:667-701` (`run_obbt`: `lb[var_idx] =
result.objective` gated only by `+1e-8`). Contrast with the hardened
`run_obbt_on_relaxation` (`:895-950`), which clamps the vertex to the
Neumaier–Shcherbina safe bound `g(y)` whenever the vertex is optimistic, with a
conditioning guard (`require_ns = _cond > _OBBT_COND_LIMIT`).

**Mechanism:** on an ill-conditioned constraint matrix the reported LP optimum can
be slightly optimistic; tightening a variable bound past the true feasible minimum
cuts off feasible (possibly optimal) points. An unsound reduction is a false
certificate seed even though this variant is not the main OBBT path.

**Reproduce/confirm:** static — read both functions and confirm the NS clamp and
conditioning guard exist only in the relaxation variant; identify `run_obbt`'s
callers to establish reachability (if unreachable/dead, downgrade to P3 and note).

**Fix:** route the tightening through `_ns_safe_lp_lower_bound` with the same
conditioning guard used by `run_obbt_on_relaxation` (or delete `run_obbt` if dead
code — deletion is the better fix if nothing calls it).

**Done criteria:** either the variant uses the NS clamp (test: with a mocked
optimistic LP objective, the applied bound never exceeds the NS bound) or the dead
function is removed; standing gates pass.

**Log:**
- 2026-07-02 — **CONFIRMED (not dead) then FIXED.** Static confirm: `run_obbt`
  applied `result.objective` (raw LP vertex) with only a `+1e-8` gate; the NS
  clamp + conditioning guard lived only in `run_obbt_on_relaxation`. Reachability:
  `run_obbt` is **live** — called from the AMP path (`solvers/amp.py:1897,1941`)
  and heavily tested — so deletion is off the table; routed the tightening
  through the NS-safe bound instead.
- **Fix.** `run_obbt` now (a) resolves its oracle via `get_exact_dual_lp_solver`
  (vertex duals), (b) applies the `_cond > _OBBT_COND_LIMIT` → `require_ns`
  conditioning guard mirroring `run_obbt_on_relaxation`, and (c) clamps every
  min/max tightening to the Neumaier–Shcherbina safe bound `g` via a new
  `_ns_clamp` closure. `_ns_safe_lp_lower_bound` gained an `n_eq` parameter so
  the trailing **equality** rows (`run_obbt` sees `A_eq`, unlike the relaxation
  variant) keep a *free-sign* multiplier while inequality rows stay clamped
  `≥ 0`; `n_eq=0` default is byte-identical for the existing callers (DBBT,
  `run_obbt_on_relaxation`). The local `_max_abs` in `run_obbt_on_relaxation`
  was hoisted to module scope and reused. Equilibration was deliberately **not**
  added here — the NS bound `g(y)` is a rigorous under-estimate for *any* duals
  regardless of conditioning, so the clamp restores soundness on its own.
- **Regression tests** (`python/tests/test_obbt.py::TestC15NsSafeClamp`,
  fail-before/pass-after verified): `test_optimistic_vertex_is_clamped` (fake
  oracle reports an optimistic vertex 5.0 with inactive duals → clamped to the
  NS bound ~0, feasible region not cut; patches both oracle seams so it
  discriminates the pre-fix path), and `test_ns_safe_lower_bound_free_equality_multiplier`
  (equality `x==2`: `n_eq=1` recovers the tight bound 2.0 vs 0.0 when clamped as
  `<=`, both rigorous under-estimates of the true min). Updated
  `test_total_time_limit_stops_before_all_variables` to patch the new preferred
  seam.
- **Gates:** `test_obbt.py` 48 passed; `test_amp` 135, `test_amp_integration` +
  `test_lp_backend_select` 17; `pytest -m smoke` 211 passed / 1 skipped;
  adversarial suite 10 passed; ruff clean. PR: #402 (→ main; supersedes #401).

---

## C-14 (P2) — LP-infeasible fathom trusts the simplex status; the documented Farkas verification is never performed

**Area:** `crates/discopt-core/src/bnb/milp_driver.rs:1061-1068`
(`LpStatus::Infeasible => lower_bound: INFEAS_SENTINEL`, node pruned). The simplex
contract (`lp/simplex/mod.rs:104-112`) documents the Infeasible result as a Farkas
ray **candidate** — "Verification is the caller's job" — and no caller verifies.
The phase-1 infeasibility test is an absolute `1e-6` threshold in *equilibrated*
space (`primal.rs:462`).

**Mechanism:** a feasible-but-numerically-tight node whose equilibrated phase-1
artificial sum sits just above the threshold is declared Infeasible and fathomed —
a region that may contain the optimum is dropped. Low probability (phase-1 is
reliable), but the module's own contract promises a check that doesn't exist.

**Reproduce/confirm:** static — read the contract comment and the fathom site;
confirm no `yᵀb`/`yᵀA` validation anywhere between them (grep for farkas usage).

**Fix:** validate the exported Farkas ray (`yᵀb < 0`, `yᵀA ≥ 0` within tolerance)
before fathoming; on validation failure, treat the node as unsolved (re-solve with
numeric-focus mode or defer — never fathom). Alternatively/additionally make the
phase-1 threshold relative to ‖b‖.

**Done criteria:** ray validation in place with a unit test (valid ray → fathom;
corrupted ray → no fathom, node re-solved); no measurable slowdown on the MILP
smoke suite (the check is one mat-vec); standing gates pass.

**Log:**
- 2026-07-03 — **CONFIRMED then FIXED** (PR TBD, branch `fix-c14-milp-driver`).
  *Confirmed statically per the card:* `solve_node`'s `LpStatus::Infeasible` arm
  fathomed to `INFEAS_SENTINEL` on the status alone; `grep -ri farkas` over the
  core found the contract comment (`lp/simplex/mod.rs`) and the exported ray but
  **no caller-side verification** anywhere between them. Confirmed dynamically via a
  fail-before probe: hard-wiring the new verifier to `true` (= pre-fix
  "trust-status") makes the regression test `c14_non_certifying_ray_is_refused` fail
  (it wrongly fathoms a zero/non-certifying ray); with the verifier live it passes.
  *Fix:* added `verify_farkas_infeasible` — checks the exported dual ray `y` with the
  objective-free safe bound `g0(±y) = bᵀy + Σⱼ min_box((−Aᵀy)ⱼ zⱼ) > margin` (a
  weak-duality certificate of emptiness; free-sign, so both `±y` tried). Runs on the
  *scaled solve-space* data (`ctx.sa`/`ctx.sb`/scaled `l`/`u`) where the warm-simplex
  ray lives — the safe-bound identity is invariant under equilibration
  (`scaling.rs`), so the verdict matches the original space with no unscaling. On
  verification **failure** the node is NOT fathomed: it is handed back uncertified
  (non-pruning `−∞` bound, midpoint) exactly like `IterLimit`/`Numerical`, so a
  numerically-tight feasible box can never be silently cut and optimality is never
  falsely claimed. Deviation from the sketch: no separate "re-solve in numeric-focus
  mode" step — the existing uncertified/branch path already re-solves the node's
  children, and the driver's `decide_status` (C-2) refuses to certify a search that
  dropped a node un-fathomed, so deferral is sufficient and simpler.
  *Subtlety found during VERIFY (regression):* the warm dual-simplex ray carries
  rounding noise on ∞-bounded columns (reduced costs down to `1e-38`); a naive check
  let a `1e-18` dribble send `g0` to `−∞` and reject **valid** certificates,
  regressing 6 AMP piecewise-relaxation tests to `iteration_limit` (never a wrong
  certificate — conservative — but a real tree-explosion regression). Fixed by a
  ray-magnitude-scaled reduced-cost zero tolerance (`1e-7·‖y‖∞`): only a reduced cost
  genuinely past that toward an infinite bound blocks certification. This is the
  *class* fix (noise floor tracks the ray), not an instance patch.
  *Regression tests (Rust, sub-second, `milp_driver::tests`):*
  `c14_valid_farkas_ray_certifies_emptiness` (a real infeasible LP's ray verifies →
  fathom allowed), `c14_non_certifying_ray_is_refused` (zero ray / absent ray /
  feasible box → refused; the fail-before lock), `c14_certificate_is_scale_invariant`
  (row-equilibrated A/b/ray still verifies — the scaled-space soundness the fathom
  relies on), `c14_infinite_column_noise_does_not_reject_valid_ray` (∞-column
  noise-level reduced cost must not reject a valid ray — the regression class).
  *Gates:* `cargo test -p discopt-core` 417 passed / 0 failed; `cargo clippy --lib`
  clean; `cargo fmt --check` clean; `maturin develop --release` ok; `pytest -m smoke`
  482 passed / 1 skipped / 0 failed (the 6 AMP fails resolved once the noise
  tolerance landed); adversarial suite 10 passed; benchmark `--suite smoke` discopt
  10/10 solved & proved, **incorrect_count = 0**.

---

## C-3 (P2) — Fractional integer values can survive in the reported incumbent if the terminal polish throws

**Area:** `python/discopt/solver.py:5265-5269` (incumbent injection without integer
rounding; `_check_constraint_feasibility` default tol 1e-4 at `:847`), `:6404-6427`
(terminal KKT polish rounds + re-solves), `:6463` (exception path keeps the
unpolished incumbent).

**Mechanism:** `nlp_result.x` is injected with integer coordinates near-integral
but not exact; feasibility is verified on the *unrounded* point. If the terminal
polish raises, the fractional incumbent is reported as-is (e.g. `x_int = 2.999997`).

**Reproduce/confirm:** monkeypatch the polish to raise; assert the returned `x`
has non-integral integer coordinates.

**Fix:** round integer coordinates (to integrality tol) at injection and verify
feasibility of the ROUNDED point (reject if rounding breaks feasibility); or at
minimum round-and-verify on the polish exception path before reporting.

**Done criteria:** with polish forced to raise, reported solution has
exactly-integral integer variables and passes feasibility at standard tol;
standing gates pass.

**Log:**
- 2026-07-03 — **CONFIRMED then FIXED.** Static confirm: `tree.inject_incumbent`
  (`tree_manager.rs:724`) stores the solution vector verbatim after only an
  objective-improvement check — no integrality snap — and the warm-start /
  node-injection gates accept any coordinate within `1e-5` of an integer. The
  reported `x_dict` is built from that raw vector at each finalizer; the terminal
  KKT polish rounds integers only *inside* its own accept branch, so if the polish
  raises / returns non-OPTIMAL / is not adopted, the fractional coordinate is
  certified. **Dynamic repro** (fails-before): a MIQP warm-started with
  `n=2.999997`, terminal polish monkeypatched to raise → reported `n =
  2.999999999997271` (residual 2.7e-12, NOT integral); with the fix → `n = 3.0`
  exactly. Verified fail-before/pass-after by stashing the fix.
- **Fix** (`solver.py`): new `_round_incumbent_integers(sol_flat, int_offsets,
  int_sizes, evaluator=None, cl_list=None, cu_list=None)` helper snaps every
  discrete coordinate *within the `1e-5` integrality tol* of an integer to that
  integer (a perturbation no larger than the tol the point already satisfied),
  leaves genuinely-fractional coordinates untouched (snapping them would fabricate
  an unproven point), and — when an evaluator + constraint bounds are supplied —
  re-verifies the rounded point at `1e-4`; it returns `(rounded, feasible)` and the
  caller adopts the rounded point only when `feasible` (else keeps the
  already-verified unrounded point, so no infeasible "integral" point is ever
  certified). Wired into **all four** incumbent finalizers: `solve_model`,
  `_solve_nlp_bb` (both with the constraint re-check), and `_solve_milp_bb` /
  `_solve_miqp_bb` (integers are branch-fixed to `[k,k]` before each node solve, so
  a stored `k±ε` is a numeric artifact and rounding to `k` cannot move a linear row
  by more than the integrality tol — documented at each site). No safety mechanism
  weakened; the round-and-verify is strictly additive over the existing polish.
- **Regression test** (`python/tests/test_c3_incumbent_rounding.py`, all
  `@pytest.mark.smoke`, fail-before/pass-after verified by stashing the fix):
  `test_c3_round_incumbent_snaps_near_integral` (in-tol snap, input not mutated,
  continuous untouched), `test_c3_round_incumbent_leaves_genuinely_fractional_untouched`
  (a 2.4 is left alone — never fabricate an unproven point),
  `test_c3_round_incumbent_rejects_when_rounding_breaks_feasibility` (rounding that
  violates a constraint returns `feasible=False` so the caller must not adopt), and
  the end-to-end `test_c3_fractional_integer_does_not_survive_polish_failure`
  (polish forced to raise + near-integral incumbent injected → reported integer is
  exactly integral). The first three encode the *class* (round-and-verify contract)
  by calling the helper directly, sub-second.
- **Gates:** `pytest -m smoke` 482 passed / 1 skipped / 0 failed; adversarial
  `test_adversarial_recent_fixes.py -m slow` 10 passed; the incumbent/solver/
  certif/gap/bound selection 727 passed / 2 xfailed / 0 failed;
  `test_c3_incumbent_rounding.py` 4 passed. `ruff check` + `ruff format --check`
  clean; `pre-commit run mypy` passed. No Rust touched. `incorrect_count`
  unchanged (0 failures; no correctness assertion weakened). PR: #456.
**Status:** open → fixed.

---

## C-11 (P2) — No `__ne__` guard: `x != y` on expressions silently evaluates to False

**Area:** `python/discopt/modeling/core.py:157-158` (`__eq__` returns a
Constraint); no `__ne__` anywhere in `modeling/`.

**Mechanism:** Python's default `__ne__` is `not (x == y)` → `not <truthy
Constraint>` → `False`, silently. User-side `if x != 0:` logic misbehaves with no
error.

**Reproduce/confirm:** `bool(x != 0)` → False today.

**Fix:** add `__ne__` raising `TypeError("!= is not a valid constraint operator")`.

**Done criteria:** test asserting the TypeError; standing gates pass (check no
internal code relies on expression `!=`).

**Log:** 2026-07-03 — **CONFIRMED then FIXED** (PR: fix-c11-modeling-api).
Repro before: `bool(x != 0)` returned `False` (a plain bool, no Constraint, no
error) — verified `x != 0`, `x != y`, `(x+1) != 2`, `v[0] != 0` all silently
yielded `False`. Root cause exactly as carded: `Expression.__eq__`
(`core.py:157`) builds a Constraint, and with no `__ne__` Python's default
`__ne__` computes `not <truthy Constraint>` → `False`. Fix: added
`Expression.__ne__` raising `TypeError("'!=' is not a valid constraint
operator …")` (loud refusal, per CLAUDE.md §3 — mirrors the C-6 "no silent
transformation" pattern). Placed on the `Expression` base so every node type
(Variable/BinaryOp/UnaryOp/IndexExpression/…) inherits it. Confirmed no internal
code depends on `Expression`-level `!=` (all `!=` hits across `modeling/` and
`_jax/` operate on scalars/str/enums/ndarrays, none on Expression objects).
`__hash__` unaffected — defining `__ne__` (unlike `__eq__`) does not null
`__hash__`; `Variable.__hash__` still works and a Variable remains dict/set-usable.
Repro after: all four `!=` forms raise `TypeError`; `==` still returns a
Constraint. Regression test: `python/tests/test_c11_ne_guard.py` (6 smoke tests,
fail-before 4/6 → pass-after 6/6, sub-second, direct API calls). Gates: `pytest
-m smoke` 484 passed / 1 skipped / 0 failed; adversarial suite 10 passed / 0
failed; `pytest -k "modeling or api or variable or constraint or model"` 629
passed / 1 skipped / 0 failed; ruff + ruff-format clean; mypy clean on core.py
(pre-existing numpy-stub / unrelated-module errors only); incorrect_count 0.

---

## C-22 (P3) — `interval_mul` yields NaN endpoints on 0·∞

**Area:** `crates/discopt-core/src/presolve/fbbt.rs:118-124`. For `[0,0]·[−∞,∞]`
all four corner products are NaN; `f64::min/max` discard NaN inconsistently.
Downstream comparisons vs NaN are all false, so this degrades to *lost tightening*
(a variable intersected with a NaN interval keeps its bound) — the audit found no
unsound path. Robustness fix: guard non-finite products (0·±∞ → 0 by interval
convention).

**Done criteria:** unit test `interval_mul([0,0], entire()) == [0,0]`; property
test over random intervals incl. infinite endpoints: result contains the true
product range and is never NaN; standing gates pass.

**Log:**
- 2026-07-03 — **CONFIRMED then FIXED** (`open`→`fixed`). Confirmed via a new
  Rust unit test: `interval_mul([0,0], entire())` returned `[NaN, NaN]` on the
  pre-fix code (each corner is `0 * ±∞ = NaN`; `f64::min/max` then propagate NaN),
  and a grid property test over infinite-endpoint intervals hit NaN endpoints.
  Verified fail-before (both tests red on pre-fix) / pass-after by reverting the
  guard. As the card predicted the defect is *lost tightening* (a variable
  intersected with a `[NaN,NaN]` interval keeps its stale bound — sound but weak),
  not an unsound bound; the fix restores the tightening.
- **Fix** (`presolve/fbbt.rs`, `interval_mul`): map NaN corner products to `0`.
  NaN here can arise *only* from `0 * ±∞`, whose interval-convention value is `0`;
  every other operand pair is finite×finite or a genuine ±∞ product, so the
  substitution never masks a real value. Result: `[0,0]·[−∞,∞] = [0,0]`. Minimal,
  local; no other operator touched (`interval_div`'s `inv_b` path routes through
  `interval_mul`, so it inherits the fix). Comment documents the convention.
- **Regression tests** (`presolve::fbbt::tests`, sub-second, call `interval_mul`
  directly): `c22_interval_mul_zero_times_entire_is_zero` (the exact `[0,0]·entire`
  repro) and `c22_interval_mul_never_nan_and_encloses_true_product` (grid over
  intervals incl. ±∞ endpoints and zero-width factors — asserts never-NaN,
  `lo ≤ hi`, and rigorous containment of the true product at finite witness
  points). Both fail before / pass after.
- **Gates:** `cargo test -p discopt-core` 383 lib + 4 integration + 1 doctest
  green, no warnings; `cargo clippy -p discopt-core --lib` clean; my added code is
  `cargo fmt`-clean (the single pre-existing fmt drift at `fbbt.rs:2597` is in the
  C-31 test block, on `main`, untouched — out of scope). `maturin develop
  --release` OK; `pytest -m smoke` 193 passed/1 skipped; adversarial suite 10
  passed; `incorrect_count = 0`. PR: #460.

---

## C-23 (P1, FIXED) — `relax_div` produces an invalid convex underestimator for nonlinear denominators (ESCALATED from P3)

**Area:** `python/discopt/_jax/mccormick.py:119-135` (`relax_div`), wired at
`relaxation_compiler.py:719-726`. Also `:102-116` (`_relax_reciprocal` sets
`cv=1/y, cc=secant` on the negative branch — inverted concavity label).

**Escalation (2026-07-03, solver-core review = DIV-1):** the "latent/harmless"
verdict was **falsified**. `relax_div` composes `x/y = x·(1/y)` and evaluates the
reciprocal at `mid_r = ½(cv_r + cc_r)` — the midpoint of the *denominator's
relaxation interval*. When the denominator is a bare variable or affine its
relaxation point-collapses (`cv_r = cc_r`), so `1/mid_r` is exact → sound (the only
case C-23/NM originally tested). When the denominator is **nonlinear** (`x*y`,
`x*x`, `sqrt(x*y)`, `x*y+1`, …) `cv_r ≠ cc_r` even at a point, so `mid_r ≠` the true
denominator and `1/mid_r` sits **above** the true `1/(·)` → the "convex
underestimator" `cv > f` → invalid dual bound.

**Reproduce/confirm (VERIFIED):** `1/(x*y)` on `[0.3,2]×[0.4,1.8]` at (1,1):
compiler returns `cv = 1.334`, true `f = 1.0` (worst over box +0.80); `cv > f` at
**3000/3000** sampled points. Reproduced on `x/(y*z)` (+1.52), `(x+1)/(x*y)` (+1.24),
`1/(x*x)` (+1.15), `1/sqrt(x*y)` (+0.23), `1/(x*y+1)` (+0.07). `1/x`, `1/(x+y)`,
`x/y`, `1/exp(x)` stay sound. Instrumented root cause: for `1/(x*y)` at (1,1) the
denominator relax returns `[cv_r=0.499, cc_r=1.0]`, `mid_r=0.75`,
`_relax_reciprocal → cv=1.334`.

**Reachability:** writes the invalid value into `result_lbs[i]` (`solver.py:5054`)
under opt-in `mccormick_bounds="nlp"` only; default `auto` mode routes to the Rust
`MccormickLPRelaxer`, which does not use JAX `relax_div`. The #120 decertification
is currently the *only* backstop preventing a wrong pruned answer — the relaxation
itself is unsound and must be fixed at the math level, not left to a downstream
guard (CLAUDE.md §3).

**Fix:** `relax_div`/`_relax_reciprocal` must apply the monotone-composition rule
over the inner interval rather than reciprocating its midpoint — for `1/y` with
`y∈[cv_r,cc_r]>0` (convex, decreasing), the valid convex underestimator of the
composite is `1/cc_r` (evaluate the decreasing outer envelope at the concave
over-estimator of the inner), not `1/mid_r`. Also branch on `y_ub < 0` and swap
cv/cc for the negative branch (match `envelopes.py:578`).

**Regression test (required — fast):** add the `1/(x*y)` containment repro to the
relaxation soundness harness (`relaxation_harness.py`), which currently omits
**reciprocal-/division-of-nonlinear-inner** — that omission is why CI was blind.
Sub-second: sample cv/cc over the box and assert `cv ≤ f ≤ cc` at every grid point,
on `1/(x*y)`, `x/(y*z)`, `1/(x*x)`, `1/sqrt(x*y)`, plus the sound `1/x`/`x/y` cases
as controls. Fails before, passes after.

**Done criteria:** the containment harness (incl. the new nonlinear-denominator
operators) is green — `cv ≤ f ≤ cc` at every sampled point; the `1/(x*y)` repro
flips from `cv=1.334>1.0` to sound; existing div/fractional relaxation tests
unchanged; standing gates pass.

**Log:** 2026-07-03 — ESCALATED P3→P1 and status open→confirmed. The
variable/affine-only test coverage masked the nonlinear-denominator defect; the
diagonal-vs-composite blind spot is the same one that hid C-32/NM-1
(solver-core review, `docs/dev/solver-core-review.md` §1).

2026-07-03 — FIXED (status confirmed→fixed). **Confirmed** via the full compiler
(the `relaxation_harness`): pre-fix worst `cv−f` = **+0.80** on `1/(x*y)`, **+1.15**
on `1/(x*x)`, **+0.71** on `x/(y*z)`, plus `1/sqrt(x*y)`, `(x+1)/(x*y)`,
`1/(x*y+1)` all `cv>f`; controls `1/x`, `x/y`, `1/(x+y)` sound. **Fix chosen:** the
tight bilinear composition `x·(1/y)` is retained **only where the denominator
relaxation collapses to a point** (`|y_ub−y_lb|<1e-12` — constant / variable /
affine denominator, where `1/y` is exact and the composition is sound & tight).
For a **non-degenerate (nonlinear) denominator interval** the midpoint-reciprocal
crosses the function, so we abstain to the **sound interval enclosure**
`[x_lb,x_ub] · [1/y_ub, 1/y_lb]` (a constant `[cv,cc]` that brackets `x/y` at every
point, both denominator signs) — the "sound construction / fall back to interval
bounds" the card sanctions. This is looser than a perfect composite envelope but
never emits `cv>f`; spatial branching shrinks the denominator interval → the
enclosure tightens. (The by-hand bivariate-McCormick composite was prototyped and
**rejected** — brute-force truth sampling showed it still crossed by up to +11, so
it was not shipped; the enclosure is provably sound to machine-epsilon.) Applied to
BOTH `_jax/mccormick.py` (in scope) and the ported `_numpy/mccormick.py`. Also
hardened `_secant` against `0/0` on the degenerate branch. **After:** every listed
nonlinear-denominator case is sound to ≤3.6e-15 (`cv≤f≤cc`); the `1/(x*y)` repro
flips from `cv=1.334>1.0` to sound; controls stay exactly tight; point-denominator
`3/2` stays `cv=cc=1.5`. **Regression test:**
`python/tests/test_div_nonlinear_denominator_c23.py` (`@pytest.mark.smoke`,
sub-second, both backends): literal repro, non-degenerate never-crosses (pos & neg
denom), point-denominator tightness, random-sub-box property test, plus a
full-compiler containment layer over `1/(x*y)`, `x/(y*z)`, `1/(x*x)`,
`1/sqrt(x*y)`, `(x+1)/(x*y)`, `1/(x*y+1)` + the `1/x`/`x/y`/`1/(x+y)` controls —
the reciprocal-/division-of-nonlinear-inner case the harness previously omitted.
36 assertions red before, all green after. **Gates:** targeted `-k` suite 1366
passed / 0 failed; ruff + format clean; mypy clean on the changed modules. PR:
(pending).

---

## C-24 (P3) — Secant construction produces NaN on infinite bounds; soundness leans on downstream filters

**Area:** `python/discopt/_jax/mccormick.py:23-33` (`_secant` divides by
`ub − lb`; `ub=+∞` → `inf/inf = NaN`); bilinear cv/cc hit `inf − inf`. Today the
NaN/inf filters in `mccormick_nlp.py:96-136, 276-283` and the LP relaxer's
finiteness checks absorb it; any future consumer using `cc` unguarded gets an
unsafe value.

**Fix:** return explicit ±∞ (no-information) envelopes when either bound is
non-finite, at the `_secant`/envelope level rather than relying on callers.

**Done criteria:** unit tests — envelopes on half-infinite boxes return ±∞, never
NaN; grep-audit that no caller special-cases NaN envelopes anymore; standing gates
pass.

**Log:** 2026-07-03 — CONFIRMED and FIXED (status open→fixed, PR #462).
Repro (pre-fix): `_secant(x²,x=0,lb=−2,ub=+∞)=NaN`; `relax_square`/`relax_exp`/
`relax_cosh` return `cc=NaN` on any half-infinite box; `relax_bilinear` with an
∞ factor bound returns `cv=NaN`; `relax_pow` odd on `[−2,+∞)` returns NaN. A NaN
is not a valid envelope — every `cv≤f` / `f≤cc` soundness comparison is False for
NaN, so an unguarded consumer silently gets an unsafe bound.
Fix (envelope level, per the sketch): `_secant` gained a `fallback` arg and
returns it (not NaN) whenever either bound is non-finite; every call site passes
the sign matching its role (`+∞` for the concave-overestimator `cc`, `−∞` for the
convex-underestimator `cv`), so the envelope degrades to the sound no-information
bracket `−∞ ≤ f ≤ +∞`. `relax_bilinear` replaces its whole envelope with
`(−∞,+∞)` when any factor bound is non-finite (its NaN came from `inf−inf` in the
affine terms, not from `_secant`). `relax_div`/`_relax_reciprocal` inherit the
fix through `relax_bilinear` and the reciprocal's finite `1/±∞→0` bounds; the
sin/cos wide-interval path already returned `[−1,1]` and is unchanged. Curvature
regimes (C-32) and division math (C-23) left untouched to avoid collision — only
the ±∞-vs-NaN behaviour changed. Verified: finite random sub-box soundness for
all univariate + bilinear + odd-power relaxations unchanged (0 crossings, 0 NaN);
half-infinite boxes now bracket at all sampled interior points with 0 NaN.
On the downstream filters: the `isfinite` guards in `mccormick_nlp.py`/
`mccormick_lp.py` are NOT dead after this fix — they still (correctly) drop the
distinct *singularity* NaN class (`1/(x³·sin x)` at wide bounds) that has nothing
to do with infinite bounds; removing them would weaken a real safety guard
(CLAUDE.md §1/§3), so they stay. Regression test:
`python/tests/test_c24_infinite_bound_envelope.py` (5 `@pytest.mark.smoke` cases,
sub-second, direct primitive calls) — fails-before (5 red on pre-fix) /
passes-after. Encodes the class (every secant-using primitive + bilinear + power),
not a named instance.

---

## C-12 (P3) — Range-constraint split renumbers constraints vs source `.nl` indices

**Area:** `nl_parser.rs:1115-1129`. Range `l ≤ body ≤ u` becomes two rows
(senses/RHS verified correct, body shared) — shifting every subsequent constraint
index vs the source numbering. Internally self-consistent; breaks any external
mapping by original index (duals, suffixes) if one is ever added.

**Fix (when touched next):** carry a `source_row` index on `ConstraintRepr` so
external mappings survive. No urgency; document until then.

**Done criteria:** `source_row` populated + one test, OR a documented note in the
parser module header; standing gates pass.

**Log:**
- 2026-07-03 — **FIXED via the documented-note option** (the card's explicit
  alternative to `source_row`). Confirmed the behavior: range constraints (and any
  two-finite-bound constraint not flagged as an explicit range) are split into two
  `ConstraintRepr` rows, so a split shifts all subsequent constraints' positions
  vs the source `.nl` row index. It is internally self-consistent (bodies/senses/
  RHS correct) and no current solver path maps results back to source rows by
  index, so there is no live mis-certification — hence P3.
- **Why the note, not `source_row`:** threading a mandatory `source_row` field
  through `ConstraintRepr` would touch **132 construction sites across 24 files**
  (presolve passes, AMP, bindings, tests) — far wider than a P3 hygiene item and
  risky to soundness-critical presolve, for zero current benefit. The card
  sanctions the note as sufficient; the note names the exact prerequisite
  (`source_row` at each split site) for any *future* feature that needs source-row
  alignment (AMPL duals, `.sol` suffixes, round-trip writer).
- **Fix:** added a `# Constraint numbering (C-12)` section to the `nl_parser.rs`
  module header documenting the split-renumbering, its self-consistency, and the
  future-work prerequisite. No code/behavior change; standing gates unaffected
  (shared PR run with C-7/C-8/C-9). Status open→fixed (documented). PR: (this PR).

---

## C-25 (P1) — Embedded-NN scaling propagates bounds in the wrong (unscaled) domain → infeasible or true optimum cut

**Area:** `python/discopt/nn/formulations/full_space.py:72`,
`python/discopt/nn/formulations/relu_bigm.py:73`. `propagate_bounds(net)` runs on
`net.input_bounds` (the user/unscaled domain — those bounds are applied to the
unscaled `inputs` var) while layer 1 actually consumes
`scaled_in = (inputs − x_offset)/x_factor`. Every `zhat`/`z` variable bound and
every ReLU big-M constant is therefore computed from the wrong box.
**Reachability:** default path for any embedded NN carrying a non-identity
`OffsetScaling` (nonzero `x_offset` or `x_factor ≠ 1`) — `add_predictor` /
`NNFormulation(..., scaling=…)`. Every existing scaling test uses offset 0 /
factor 1, so the module's own suite cannot catch it.

**Mechanism:** with `x_offset`/`x_factor` nontrivial, the propagated interval on
each hidden pre-activation is derived from the wrong input box. A big-M sized
from the wrong box can be too small, making a valid ReLU state infeasible
(spuriously infeasible model), or the `zhat`/`z` variable bounds can exclude the
scaled activation's true range, cutting the true optimum → a wrong surrogate
optimum reported as certified. This is certificate-class for the modeling layer:
"the global optimum over the embedded surrogate" can be wrong.

**Why P1:** wrong answer / infeasible on a reachable, realistic configuration
(scaled inputs are the norm for trained models), but confined to the NN modeling
layer (no solver-core certificate is implicated).

**Reproduce/confirm:** build a small net with
`OffsetScaling(x_offset=[100,…], x_factor=[0.5,…])` (and a negative-`x_factor`
variant); run the T-N0.1 equivalence harness
(`python/tests/test_nn_equivalence.py`,
`assert_embedding_matches` + `assert_optimum_matches`) on `full_space` (sigmoid
net) and `relu_bigm` (relu net). Before the fix these are infeasible or certify a
wrong optimum.

**Fix:** implemented under `docs/dev/nn-module-plan.md` **T-N0.2**. Add an
optional `input_bounds` override to `propagate_bounds(network, input_bounds=None)`
and, when `self._scaling is not None`, compute the scaled box `(s_lo, s_hi)`
*before* propagation and call `propagate_bounds(net, input_bounds=(s_lo, s_hi))`.
No behavior change when scaling is None/identity.

**Done criteria:**
- `assert_embedding_matches` and `assert_optimum_matches` pass with nontrivial
  positive and negative `x_factor` for both `full_space` and `relu_bigm`
  (fail-before / pass-after).
- Identity-scaling equivalence unchanged (T-N0.1 green baseline).
- Standing gates pass.

**Log:**
- 2026-07-03 — Filed from the 2026-07-03 nn-module review (finding F1). Fix owned
  by nn-module-plan T-N0.2 (same effort); status tracked there.

---

## C-26 (P1) — Tree-ensemble big-M is invalid for thresholds outside the declared feature box → cuts feasible points

**Area:** `python/discopt/nn/formulations/tree_ensemble.py:79,98-111`. The per-leaf
constraints use `M_j = ub_j − lb_j`, which keeps a non-selected leaf's constraint
inert only when `lb_j ≤ thr ≤ ub_j − eps`.
**Reachability:** default path for any embedded tree ensemble whose optimization
bounds are tighter than the training-data range — a common case (a threshold then
falls outside the declared feature box).

**Mechanism:** for a split threshold outside `[lb_j, ub_j]`, `M_j = ub_j − lb_j`
is too small to relax the leaf's split constraint when the leaf is *not* selected
(`z=0`), so the constraint stays active and cuts otherwise-feasible input points.
The embedded MILP then optimizes over a strict subset of the true feasible region
→ can report a wrong surrogate optimum as certified. Certificate-class for the
modeling layer.

**Why P1:** wrong answer on a reachable, realistic configuration (tightened
optimization bounds vs training range), confined to the NN/tree modeling layer.

**Reproduce/confirm:** an ensemble with one threshold `> ub_j − eps` and one
`< lb_j`; run `assert_embedding_matches` + `assert_optimum_matches` vs dense
enumeration (T-N0.1 harness). Before the fix, feasible points are cut / the
certified optimum disagrees with enumeration.

**Fix:** implemented under `docs/dev/nn-module-plan.md` **T-N0.3**. Replace the
per-feature `M_j` with per-constraint coefficients: left split →
`x_j ≤ thr + max(ub_j − thr, 0)·(1 − z)`; right split →
`x_j ≥ (thr + eps) − max(thr + eps − lb_j, 0)·(1 − z)`. These are inert for
`z=0` at any threshold position, correctly make box-unreachable leaves infeasible
when selected, and are strictly tighter LP relaxations. Drop the unused
`feat_range`.

**Done criteria:**
- Out-of-box-threshold ensemble: `assert_embedding_matches` +
  `assert_optimum_matches` pass (fail-before / pass-after).
- No-behavior-change check on the in-box `_make_tree_ensemble` fixture.
- Standing gates pass.

**Log:**
- 2026-07-03 — Filed from the 2026-07-03 nn-module review (finding F2). Fix owned
  by nn-module-plan T-N0.3 (same effort); status tracked there.

---

## C-27 (P2) — ONNX reader silently mis-reads Gemm attributes, residual Adds, and branched graphs

**Area:** `python/discopt/nn/readers/onnx_reader.py`. (a) Gemm `alpha`/`beta`/
`transA` ignored (`:104-121`); (b) a `MatMul → Add` where the `Add` is not an
initializer (e.g. a residual connection) is consumed and zero biases substituted
(`:74-79`); (c) no dataflow verification — a branched graph of individually
supported ops parses into a wrong sequential net.
**Reachability:** any model loaded via `add_predictor(model, inputs, "model.onnx")`
/ `load_onnx` that uses `alpha≠1`/`beta≠1`/`transA=1` Gemm, a residual Add, or a
non-sequential topology. Silent substitution class (like C-5): the solver then
faithfully embeds the *wrong* network and labels the result optimal.

**Mechanism:** each path drops or mis-orients structure without raising — scaled
Gemm coefficients ignored, a residual Add mistaken for a bias node then replaced
by zeros, or a branch/join topology flattened into a sequence — so the embedded
constraints encode a different function than the ONNX file. No user-visible error.

**Why P2:** silent problem substitution requiring a non-default ONNX structure
(scaled/residual/branched graphs); loud-refusal class, no certificate of the
*solver* is implicated (mirrors the C-5 `.nl`-opcode silent-rewrite severity).

**Reproduce/confirm:** load a Gemm graph with `alpha=2.0, beta=0.5` and diff the
reconstructed `forward` against `onnxruntime` inference on random points; load a
two-branch graph with a joining Add. Before the fix the values disagree / the
branched graph parses without error.

**Fix:** implemented under `docs/dev/nn-module-plan.md` **T-N1.1**. Apply Gemm
`alpha`/`beta`, raise `ValueError` on `transA=1`; verify the MatMul initializer is
`input[1]`; add dataflow tracking from `graph.input[0]` so any residual/branch
topology raises `ValueError("non-sequential graph")` (which also removes the
residual-Add-as-zero-bias path).

**Done criteria:**
- Gemm `alpha`/`beta` graph loads and matches `onnxruntime`; `transA=1` and a
  two-branch joining-Add graph both raise; `transB=1` still round-trips.
- Coverage omit for `onnx_reader.py` removed once these tests exist (T-N4.1).
- Standing gates pass.

**Log:**
- 2026-07-03 — Filed from the 2026-07-03 nn-module review (finding F3). Fix owned
  by nn-module-plan T-N1.1 (same effort); status tracked there.

---

## C-28 (P2) — sklearn classifier readers silently embed logits / wrong base_score

**Area:** `python/discopt/nn/readers/sklearn_reader.py`. `load_sklearn_mlp`
ignores `model.out_activation_`, so an `MLPClassifier` embeds pre-activation
logits while the docstring claims classifier support; `load_sklearn_ensemble` on a
`GradientBoostingClassifier` reads `base_score` wrong (a classifier's `init_` has
no `constant_`).
**Reachability:** any classifier passed to `add_predictor` / the sklearn reader.
Silent substitution class (like C-5): the embedded surrogate is not the trained
classifier's decision function, but no error is raised.

**Mechanism:** the readers assume regressor semantics. For `MLPClassifier` the
final `out_activation_` (logistic/softmax) is dropped, so the embedding predicts
logits, not probabilities; for `GradientBoostingClassifier` the log-odds `init_`
offset is silently mis-read → the additive intercept is wrong. Both produce a
formulation whose optimum answers a different question than the user posed.

**Why P2:** silent problem substitution on a non-default (classifier) input;
loud-refusal class, no solver certificate implicated (same severity rationale as
C-5 / C-27).

**Reproduce/confirm:** load a binary `MLPClassifier` and compare the embedded
output to `predict_proba[:, 1]` through the T-N0.1 harness; load a
`GradientBoostingClassifier` and check the intercept. Before the fix they
disagree with no error.

**Fix:** implemented under `docs/dev/nn-module-plan.md` **T-N1.2**. In
`load_sklearn_mlp` read `out_activation_` (`identity`→LINEAR, `logistic`→SIGMOID
final layer, `softmax`→`ValueError`); in `load_sklearn_tree`/`load_sklearn_ensemble`
raise `TypeError` on classifiers (`sklearn.base.is_classifier`) with a
regressor-only message; update docstrings to regressors-only. (Also fixes the
single-leaf 0-d `squeeze()` crash via `value.reshape(len(feature), -1)`.)

**Done criteria:**
- Binary `MLPClassifier` loads with a SIGMOID final layer and matches
  `predict_proba[:, 1]` through the harness; `GradientBoostingClassifier` and
  `DecisionTreeClassifier` raise `TypeError`; a constant-target
  `DecisionTreeRegressor` loads and predicts.
- Standing gates pass.

**Log:**
- 2026-07-03 — Filed from the 2026-07-03 nn-module review (finding F4). Fix owned
  by nn-module-plan T-N1.2 (same effort); status tracked there.

---

## C-29 (P0) — Vector-body constraint collapses to one summed row (DEFAULT path)

**Origin:** Fable solver-core-extraction review (CORE-1) and modeling review (M1).
**Area:** `python/discopt/_jax/problem_classifier.py:224-226`
(`_extract_linear_coefficients`, the "Array variable treated as sum" branch) and
`:614` (`_extract_constraints_algebraic` appends exactly one row per `Constraint`
object). The quadratic walker `_extract_quadratic_coefficients` (`:372` region) has
the identical branch.
**Reachability:** DEFAULT `m.solve()` on any LP/MILP/QP written with numpy-vectorized
bodies — the idiom the modeling API's own docstrings teach.

**Mechanism:** a vector-valued constraint body (`a + b <= 1` with `a,b` shape `(2,)`)
is one `Constraint` whose body is array-valued; the extractor sums every element
into a single coefficient and appends one row, so `a + b <= 1` is extracted as
`Σa + Σb <= 1`. The correct autodiff extractor (`_extract_lp_data_autodiff`) is
row-per-component but is bypassed because the algebraic path *succeeds* (wrongly).

**Reproduce/confirm (VERIFIED):**
- LP `min Σa+Σb s.t. a+b>=1` (shape (2,), box [0,1]) → objective **1.0**, true
  **2.0**; the returned point is infeasible by the model's own `NLPEvaluator`.
- MILP set-cover `min Σy+Σz s.t. y+z>=1` (shape (3,) binaries) → **1.0**, true **3.0**.
- The scalar-loop form of the identical model solves correctly on the same path.

**Fix:** Stage 1 (minimal, safe) — in both walkers, raise `_NotLinearError` on any
array-shaped node in scalar position (delete the "treated as sum" branches); make
`_eval_const` raise `_NotLinearError` (not `ValueError`) on a non-scalar. Affected
models then route to the row-correct autodiff extractor. Stage 2 (restore speed) —
teach the algebraic extractor to expand array bodies into per-element rows, verified
bound-neutral against the autodiff extractor. Details: `solver-core-extraction-review.md` §3.

**Done criteria:** the three repros return the true optima with elementwise-feasible
points; algebraic-vs-autodiff extraction agree (rows, matrix, objective, sense) on a
property-test panel of vectorized affine models; `_eval_const` non-scalar → `_NotLinearError`;
standing gates; `incorrect_count ≤ 0`.

**Log:** 2026-07-03 **fixed** (branch `fix-c29-c30-linear-body-classify`, with C-30).
Confirmed first: LP `min Σa+Σb s.t. a+b>=1` (shape (2,), box [0,1]) certified **1.0**
at the elementwise-infeasible point `a=b=[0.25,0.25]` (`a+b=0.5<1`), true **2.0**; MILP
set-cover `y+z>=1` (shape (3,) binaries) certified **1.0**, true **3.0**. Root cause:
the algebraic walkers `_extract_linear_coefficients`/`_extract_quadratic_coefficients`
collapsed a size>1 array variable in *scalar* position into one summed row, and the
constraint loop appends exactly one row per `Constraint`, so a vector body `a+b>=1`
became `Σa+Σb>=1`. **Fix (Stage 1, sound-minimal):** both `_walk`s now carry an
`allow_array` flag — True only under a `SumExpression` (where element-collapse with a
*uniform* scale is a legitimate reduction), False elsewhere; an array node in scalar
position raises `_NotLinearError`/`_NotQuadraticError` so `extract_lp_data`'s dispatcher
routes the body to the per-component autodiff extractor (one LP row per element). Also
`_eval_const` now raises `_NotLinearError` (not `ValueError`) on a non-scalar array
constant. Legit `sum(array_var)` bodies still take the fast algebraic single-row path
(bound-neutral guard test). Post-fix: both repros certify 2.0 / 3.0 at elementwise-
feasible points. Regression: `python/tests/test_c29_c30_linear_body_classify.py`
(`test_c29_vector_body_not_collapsed_to_single_row`,
`test_c29_solve_certifies_feasible_point`, `test_c29_milp_set_cover_vector_body`,
`test_legit_sum_stays_on_fast_algebraic_path`), all `@pytest.mark.smoke`, red-before/
green-after. Gates: smoke 297 passed (only the 3 pre-existing C-31 tests fail — separate
Rust FBBT bug, out of scope), adversarial 10 passed, ruff clean. PR #___.

---

## C-30 (P0) — Maximize sense lost on `sum(const·var)` bodies

**Origin:** Fable solver-core-extraction review (CORE-2) and ro review (ADJ-1).
**Area:** `python/discopt/_jax/problem_classifier.py:259,569`
(`_extract_linear_coefficients` → `_eval_const` calls `float(v.item())` on a size-2
array and raises `ValueError`, not `_NotLinearError`), plus the fallback that catches
it and drops the objective sense.

**Mechanism:** `dm.sum(c * x)` recurses `SumExpression → BinaryOp("*") → _eval_const(Const([1,1]))`
which raises a raw `ValueError`; the algebraic path aborts and falls over to a
fallback that does **not** apply the maximize negation. `extract_lp_data` returns
`c = [1,1,0]` (un-negated) for a *maximize* model, so the solver minimizes and returns 0.

**Reproduce/confirm (VERIFIED):** `maximize dm.sum([1,1]·x) s.t. dm.sum([1,1]·x) <= 4`,
`x∈[0,10]²` → objective **1.5e-08** at `x≈[0,0]`, true **4.0**. `extract_lp_data(m).c`
is `[1,1,0]` un-negated. Scalar / `x[0]+x[1]` / `A@x` / `dm.sum(x)` forms all correct.

**Fix:** raising `_NotLinearError` on the non-scalar `_eval_const` (C-29 Stage 1)
routes this to the autodiff extractor, which handles sense — so C-29's fix largely
subsumes this. ALSO audit every `except _NotLinearError`/`except Exception` around
the extractors to confirm the fallback applies objective sense and row expansion;
add a regression asserting `extract_lp_data(maximize_model).c` is negated.

**Done criteria:** the maximize repro returns 4.0 and negated `c`; the sense-negation
audit lands with a regression test; standing gates.

**Log:** 2026-07-03 **fixed** (branch `fix-c29-c30-linear-body-classify`, with C-29).
Confirmed first: `maximize dm.sum([1,1]·x) s.t. dm.sum([1,1]·x)<=4`, `x∈[0,10]²`
certified **1.5e-08** at `x≈[0,0]`, true **4.0**; `extract_lp_data(m).c == [1,1,0]`
un-negated. Two-part root cause: (1) `_eval_const` raised a raw `ValueError` on the
size-2 array constant, aborting the algebraic walk; (2) the fallback it landed in —
`_extract_lp_data_autodiff` — was the ONLY extractor that never applied the maximize
negation (`extract_lp_data_algebraic:743`, `_extract_lp_data_from_repr:927`,
`_extract_qp_data_autodiff:1375` all do), so it emitted `c=[1,1,0]` and the solver
minimized a maximize model. **Fix:** `_eval_const` now raises `_NotLinearError` on a
non-scalar array (C-29 change), AND `_extract_lp_data_autodiff` now negates `c`/
`obj_const` for `ObjectiveSense.MAXIMIZE` — closing the sense-dropping gap for *every*
body that routes to the autodiff LP fallback, not just this one. Audited all six
sense-negation sites: the autodiff LP extractor was the sole omission. Post-fix: repro
certifies **4.0** with `c=[-1,-1,0]`. Regression:
`test_c30_eval_const_refuses_array_with_notlinear`,
`test_c30_maximize_sense_preserved_in_extracted_c`,
`test_c30_autodiff_extractor_applies_maximize_sense`,
`test_c30_solve_maximize_returns_true_max` in
`python/tests/test_c29_c30_linear_body_classify.py`, all `@pytest.mark.smoke`,
red-before/green-after. Gates as for C-29. PR #___.

---

## C-31 (P0) — FBBT collapses an array-variable block to element-0's bounds

**Origin:** Fable tightening/conflict review (TG-1, CF-1).
**Area:** Rust `crates/discopt-core/src/presolve/fbbt.rs:1204-1208` (seeds each
variable's interval from element-0's bounds, `v.lb.first()/v.ub.first()`, returns one
`Interval` per block; the identical seeding is also at `fbbt_with_cutoff`
`fbbt.rs:1105-1111`, and `eval_node_interval` `fbbt.rs:513-516` ignores the column) +
`python/discopt/tightening.py:114-121` (stamps that single interval onto every scalar
slot of the block).
**Consumers (broadened 2026-07-03, solver-core review):**
(1) `conflict.py:81-85` — uses `fbbt_box(...).infeasible` as its infeasibility oracle
→ invalid no-good cuts (documented reach);
(2) **NEW — the certified LP dual bound.** `_fbbt_argument_box`
(`milp_relaxation.py:4338-4352`) intersects the FBBT box with the node box and builds
a McCormick univariate-rescue envelope over it. `_fbbt_argument_box` is sound on its
own logic (`fbbt_box`/`tightening.py:65` uses no objective cutoff — verified), so an
envelope over it is valid **provided `fbbt_box` is correct** — but `fbbt_box` *is* the
collapse bug, so on heterogeneous per-element array bounds it omits feasible arguments
→ **invalid envelope feeding the certified LP relaxation bound**, not merely invalid
conflict cuts. This makes fixing `fbbt_box` more urgent than the original card stated.
(3) Reaches `probing.rs:78,90` via the same seeding.
**Reachability:** any array variable with heterogeneous per-element bounds (DAE
initial conditions, per-index box) through the presolve path.

**Mechanism:** element 0's (tighter) bound is illegally propagated onto the other
elements. The claim "a valid outer bound for every element of the block" is false.

**Reproduce/confirm (VERIFIED):**
- Over-tighten: `x=continuous(shape=(2,), lb=[8,0], ub=[10,10])` → `fbbt_box` returns
  `lb=[8,8]`, deleting the feasible region `x[1]∈[0,8)`.
- **False infeasible:** `x=continuous(shape=(2,), lb=[5,0], ub=[5,3])` → `fbbt_box`
  returns `infeasible=True` (element 1 gets element-0's `lb=5` vs its own `ub=3`),
  though `x=[5, 0..3]` is feasible — violates "FBBT never reports feasible as infeasible."
- Chained: `find_conflict_cuts` on a model with the above `x` and a free binary
  returns two cuts that together make the binary infeasible.

**Fix (LANDED):** the FBBT engine carries **one interval per variable block** and,
by design, resolves every `Index{base,col}` node — forward *and* backward — to that
single shared block interval (the column is ignored). Backward tightening onto an
array block (`size > 1`) is already a no-op (`backward_propagate`'s `Variable` arm
only writes when `size == 1`, and the `Index` arm recurses into the `size > 1` base),
so the block interval is never *narrowed* below its seed for arrays. The **only**
unsoundness was the **seed**: it used element 0's bounds (`v.lb.first()/v.ub.first()`)
as if they bounded every element. The fix seeds each block from the element-wise
**UNION** `[min lb, max ub]` (`seed_block_interval`, `fbbt.rs`), a valid *outer* bound
for every element — so a forward `Index` on element `k` evaluates against a superset
of element `k`'s true interval: FBBT can only **lose** tightening for the block, never
cut a feasible point. Where an array element's rescue box then remains out of domain
(the union straddles 0 / is non-finite), `_collect_univariate_relaxations` correctly
**abstains** (drops the op) rather than emitting an envelope over a box that excludes
feasible arguments. The same element-0 seed in the presolve driver
(`pass.rs::from_model`, `resync_bounds_after_rewrite`) was switched to the union too.
For a **homogeneous** block the union equals element 0, so this is a no-op there (no
regression / no lost tightening for the common case). This is an *abstain*-class fix
per CLAUDE.md §3: array-block FBBT is now sound-but-conservative rather than tight; a
future per-scalar FBBT (real per-element intervals + `Index`-column-aware
forward/backward) would recover the lost tightness — tracked separately, not required
for soundness.

**Regression tests (LANDED):** the two Rust characterization tests were **flipped**
from asserting the collapse to asserting the fixed behavior:
`presolve::fbbt::tests::c31_array_block_seeds_from_element_union_not_element0`
(feasible region `x[1]∈[0,8)` preserved; block covers every element) and
`c31_heterogeneous_block_no_false_infeasible` (feasible `x=[5,0..3]` not declared
infeasible). Three fast Python `@pytest.mark.smoke` tests added in
`python/tests/test_tightening.py`:
`test_c31_heterogeneous_array_block_not_overtightened`,
`test_c31_heterogeneous_array_block_no_false_infeasible`, and — for the broadened
certified-LP reach — `test_c31_fbbt_argument_box_envelope_contains_feasible_arg`
(calls `_collect_univariate_relaxations` directly and asserts every emitted rescue
envelope's arg box contains a feasible argument). All five FAIL on the pre-fix
element-0 seed and PASS after; all sub-second.

**Done criteria:** ✅ both repros fixed (no feasible cut; no false infeasible); ✅ the
`_fbbt_argument_box` envelope contains every feasible argument on a heterogeneous
block (or the op is abstained — never an envelope over an excluding box); ✅ the two
Rust characterization tests flipped to assert correct behavior; ✅ homogeneous-bounds
tests still pass; ✅ `cargo test -p discopt-core` (386 tests, 0 failures, clippy clean);
✅ full `python/tests -k "fbbt or envelope or relax or array or tighten"` green (639
passed). Certified-path tightening only ever *loosens* the box (union ⊇ element-0
seed), so it cannot raise the dual bound above the truth — no bound can cross the
oracle.

**Log:**
- 2026-07-03 — Rust characterization tests committed (pin current behavior,
  `fbbt.rs` tests). Blast radius broadened to the certified LP dual bound via
  `_fbbt_argument_box` (solver-core review, `docs/dev/solver-core-review.md` §1).
- 2026-07-03 — **FIXED** on `fix-c31-fbbt-array-block-collapse` (PR #424). Root cause
  is the block-granular FBBT *seed*, not per-scalar backward propagation (that is
  already a no-op for arrays). Seed switched to the element-wise union in `fbbt`,
  `fbbt_with_cutoff`, and the presolve driver (`pass.rs`); characterization tests
  flipped; Python smoke regressions added. Demonstrated pre-fix invalid envelope:
  `log(x[1])` on `x=continuous(shape=(2,), lb=[3,-1], ub=[3,5])` produced a rescue
  envelope over arg box `[3,3]`, **excluding** the feasible argument `4.0`; post-fix
  the union seed `[-1,5]` straddles 0 → the op is soundly abstained (0 relaxations).

---

## C-32 (P0) — `relax_asin`/`relax_acos` inverted curvature → unsound envelope in the LIVE JAX layer

**Origin:** Fable numpy-mccormick review (NM-1). **This audit's McCormick pass
(C-18/19/23/24) missed it.**
**Area:** `python/discopt/_jax/mccormick.py:474-524` (LIVE) and the numpy port
`python/discopt/_numpy/mccormick.py:210-241`. `arcsin''(x)=x(1−x²)^{−3/2}`, so arcsin
is **convex on [0,1]** and concave on [−1,0] (acos mirrored); the code sets
`is_concave = lb≥0` — treating the convex region as concave — and swaps cv/cc. The
docstrings ("asin is convex on [-1,0]") are themselves wrong.

**Mechanism:** the convex under-estimator `cv` is returned *above* the function →
cuts feasible points below → invalid dual bound → risk of certifying a wrong optimum.

**Reproduce/confirm (VERIFIED, live JAX):** `relax_asin(0.5)` on `[0.1,0.9]` returns
`cv=0.609968 > true=0.523599` (gap 0.086) on **both** the JAX and numpy primitives.
Off-diagonal fuzz over `[-0.99,0.99]` sub-boxes: **7,493 crossings**, worst 0.216.

**Why both audits missed it:** the relaxation soundness suite evaluates on the
diagonal `relax_fn(x, x, lb, ub)`, where a univariate-of-a-bare-variable relaxation
collapses to a degenerate interval (`_secant → f(x)`), so the buggy branch is never
exercised; and `asin(x*y)` routes via the sound multivariate bilinear path.

**Fix:** swap the regime flags so `lb≥0` is the *convex* case for asin (mirror acos)
in **both** `mccormick.py` files; correct the docstrings. THEN extend the soundness
harness to sample **off-diagonal** (`x_cv ≠ x_cc`, univariate over non-degenerate
boxes) so this class can never hide again. Also fix the related C-19 (`relax_tan`
pole-straddling) — confirmed live on the numpy backend during this reconciliation —
while in the same file. Details: `numpy-mccormick-review.md`.

**Done criteria:** off-diagonal fuzz shows `cv ≤ arcsin ≤ cc` with zero crossings for
asin/acos (the repro flips to sound); the harness now samples off-diagonal; check
whether any test/benchmark instance uses asin/acos (if so the risk is not latent);
differential-bound checks per §0. NM-2 (numpy compiler leaf drops the box) is tracked
separately in `numpy-mccormick-review.md` and gates activation of the numpy backend.

**Log:** 2026-07-03 — FIXED (branch `fix-c32-asin-acos-curvature`, PR #<TBD>).
Confirmed the inverted regime on **both** backends first: off-diagonal fuzz over
`[-0.99,0.99]` sub-boxes gave **3997/4000 crossing boxes, worst 0.235** for each of
asin/acos; the literal repro `relax_asin(0.5)` on `[0.1,0.9]` returned `cv=0.609968 >
true=0.523599`. Root cause: `relax_asin` was a copy of the `tanh` layout
(concave-on-positive) and `relax_acos` a copy of the `sinh` layout
(convex-on-positive) — exactly swapped, since `asin''(x)=x(1−x²)^{−3/2}` makes asin
**convex on [0,1]** / concave on [−1,0] and acos the mirror. Fix (both
`_jax/mccormick.py` and `_numpy/mccormick.py`): set `is_convex = lb>=0` for asin
(mirror `sinh`) and `is_concave = lb>=0` for acos (mirror `tanh`), with the
straddling case3 split at the inflection x=0 (positive convex/concave branch uses
f(x)/sec, negative branch reversed); corrected the wrong docstrings. No safety guard
weakened — this is a pure regime correction. After the fix: **0 crossings** on both
backends over the same fuzz; repro `cv=0.523599 == true`. Regression test:
`python/tests/test_asin_acos_envelope_c32.py` (`@pytest.mark.unit`+`smoke`, sub-second,
both backends) — calls the primitives DIRECTLY on **off-diagonal** boxes (one-signed,
zero-straddling) plus a 500-sub-box property test asserting no crossing; proven RED on
pre-fix (34 failed) and GREEN after (34 passed). Scope check: `envelopes.py` has no
asin/acos (only asinh/acosh, separately defined and sound); `atan`/`tanh`/`sinh`/
`sigmoid` regimes verified correct — the inversion was confined to these two
functions, no wider class. No in-repo `.nl` corpus instance uses the asin/acos opcodes
(`o51`/`o53`), so the false-optimum risk was **latent** (never tripped by a benchmark),
consistent with why the diagonal-only soundness harness missed it. Gates: smoke green
(the only 3 failures — `test_tightening.py::test_c31_*` — are the still-open C-31 issue
and fail identically on the pristine base, i.e. pre-existing, not a regression);
adversarial suite 10/10; ruff clean; mypy error is a pre-existing env stub mismatch
present on the base. C-19 (`relax_tan` pole) left to its own issue — not touched here
to keep the PR scoped to one issue.

---

## C-33 (P0) — Pure-continuous fallback certifies a nonconvex model's local optimum (DEFAULT path)

**Origin:** solver-core review (SC-1).
**Area:** `python/discopt/solver.py:3716-3738` (the pure-continuous fallback that
routes to a single NLP solve); certificate emitted at `:6889-6896`.
**Reachability:** DEFAULT path — fires when `skip_convex_check or not
_pure_continuous_convexity_known`. No opt-in flag required.

**Mechanism:** when a pure-continuous model's convexity is **unknown** (classifier
abstained, or `skip_convex_check` set), the fallback solves one NLP and emits its
local optimum **with `gap_certified=True`** — treating "convexity not established" as
if it were "convex." For a nonconvex model the local optimum is not the global
optimum, so a wrong answer is certified. This is distinct from the convex fast path
(`:3651`, which correctly gates on a *rigorous* verdict and is verified sound); the
bug is that this *fallback* path certifies without that gate.

**Reproduce/confirm (VERIFIED):** nonconvex double-well objective → returns
`objective=-37, bound=-37, gap_certified=True`; the true global minimum is **−64**.
The certificate is false.

**Fix:** the fallback must **withhold** `gap_certified` unless
`_pure_continuous_convexity_known and _pure_continuous_is_convex`. When convexity is
unknown or `skip_convex_check` is set, fall through to spatial B&B (or return
`gap_certified=False`), never a certified single NLP. Do not weaken the certificate
to "trust the NLP" — refuse to certify (CLAUDE.md §1, §3).

**Regression test (required — fast):** a `@pytest.mark.smoke` test that solves the
double-well model and asserts `gap_certified is False` (or the true −64 via B&B) —
and, as a control, that a genuinely convex pure-continuous model still returns
`gap_certified=True` (guards against over-correcting into never certifying). One
short `Model.solve()` each; add an xfail-until-fixed variant asserting the current
false `gap_certified=True` to pin the bug.

**Done criteria:** the double-well returns `gap_certified=False` or −64; the convex
control still certifies; no node-count/objective drift on the certifying panel for
models that were already convex; standing gates pass.

**Log:**
- 2026-07-03 — VERIFIED new default-path P0 (solver-core review §1).
- 2026-07-03 — **CONFIRMED then FIXED.** Confirmed the false certificate with the
  nonconvex asymmetric quartic double-well `f(x)=x**4-16x**2+5x` on `[-4, 6]`
  (midpoint start x=1 sits in the shallow well's basin): the pure-continuous
  fallback returned `objective=-50.06, bound=-50.06, gap_certified=True` while the
  true global minimum is `-78.33` at x≈-2.90 — a false optimality certificate. The
  trigger used was `skip_convex_check=True` (one of the two documented triggers,
  the other being classifier abstention → `not _pure_continuous_convexity_known`);
  both reach the same fallback with convexity **not established**. Verified
  fail-before/pass-after by stashing the fix.
- **Fix** (`solver.py`, pure-continuous fallback at the `if _pure_continuous and
  not _pure_continuous_force_spatial and (skip_convex_check or not
  _pure_continuous_convexity_known)` block): reaching this branch means convexity
  was never established — the KNOWN-convex case already returned via the convex
  fast path above, so **no convex certificate can be lost here**. On any
  non-error, non-infeasible fallback result the code now **withholds the
  certificate**: keeps the feasible incumbent (`objective`, `x`, `status`) but
  sets `gap_certified=False` and drops the fabricated dual bound/gap
  (`bound=None, root_bound=None, gap=None, root_gap=None`). The local NLP
  objective is no longer emitted as a proven bound. No safety mechanism was
  weakened — the gate was made stricter (refuse to certify per CLAUDE.md §1/§3).
  Rigorous `status="infeasible"` (from nonlinear tightening / NLP infeasibility)
  is left untouched. The DEFAULT non-skip path on a *known*-nonconvex model is
  unaffected (it routes to spatial B&B, which finds the true global -78.33 with a
  valid bound and legitimately certifies).
- **Regression test** (`python/tests/test_c33_nonconvex_fallback_cert.py`, all
  `@pytest.mark.smoke`, fail-before/pass-after verified):
  `test_c33_nonconvex_fallback_not_certified_optimal` (the double-well repro —
  asserts `gap_certified is False` and `bound is None`; fails before the fix with
  the -50.06-certified-as-optimal assertion),
  `test_c33_convex_control_still_certifies` (convex `exp(x)+x**2` still certifies
  via the convex fast path — guards against over-correction), and
  `test_c33_default_path_nonconvex_uses_spatial_bb` (default path on the same
  nonconvex model finds the true global via spatial B&B — guards the sound path).
- **Gates:** new test 3 passed; `pytest -k "convex or certif or gap"` 635 passed /
  1 skipped / 2 xfailed; `pytest -m smoke` 246 passed / 1 skipped; adversarial
  `test_adversarial_recent_fixes.py -m slow` green; `ruff check` + `ruff
  format --check` clean; `pre-commit run mypy` passed. No Rust touched. PR: #422.

**Status:** confirmed → fixed.

---

## C-34 (P0) — Even-power bound over a zero-straddling base uses endpoint-only bounds (DEFAULT path)

**Origin:** solver-core review (FR-1).
**Area:** `python/discopt/_jax/gdp_reformulate.py:493-506` (the `**` arm of
`_bound_expression`). Feeds the aux box at `factorable_reform.py:254` and the
denominator-clearing path at `:707`.
**Reachability:** DEFAULT reformulation path for any model with an even power `p≥4`
of a subexpression whose interval straddles 0.

**Mechanism:** only `p==2` gets the straddle-aware `[0, max(lb², ub²)]`. For `p≥4`
over a base whose box straddles 0 the code returns `[min(lb**p, ub**p),
max(lb**p, ub**p)]` — endpoint-only — which **omits the interior minimum at 0**. So
`_bound_expression(x**4)` on `[−2,2]` returns `[16,16]` when the true range is
`[0,16]`. The bogus lower bound 16 propagates into the aux variable's box and into
denominator clearing → an invalid relaxation box → the true optimum can be cut.

**Reproduce/confirm (VERIFIED e2e):** `min (x−0.5)²+(y−1)² s.t. x**4·y ≤ 1` returns
`objective=0.25` at x=1; the true optimum is **0** at x=0.5 (feasible: `0.5**4·1 =
0.0625 ≤ 1`). The reformulation's bad `x**4` box cut off x=0.5.

**Fix:** for even `p` over a zero-straddling base, return `[0, max(lb**p, ub**p)]`
(the interior min at 0), generalizing the `p==2` branch to all even powers. For a
base that does not straddle 0, endpoint bounds are correct (monotone on the branch).

**Regression test (required — fast):** a unit test asserting
`_bound_expression(x**4)` on `[−2,2]` → `[0,16]` (and `x**6` → `[0,64]`), plus the
end-to-end repro asserting the `x**4·y ≤ 1` model returns 0 at x=0.5. Both
sub-second; add an xfail-until-fixed variant pinning the current `[16,16]`.

**Done criteria:** `_bound_expression` returns the interior-inclusive bound for all
even powers over zero-straddling bases; the e2e repro returns 0 at x=0.5;
non-straddling and odd-power cases unchanged; standing gates + differential-bound
checks per §0.

**Log:**
- 2026-07-03 — VERIFIED new default-path P0 (solver-core review §1).
- 2026-07-03 — **FIXED** (PR #425). Confirmed fails-before: direct
  `_bound_expression(x**4)` on `[−2,2]` returned `(16.0, 16.0)` (true `(0.0,
  16.0)`); `x**6` → `(64.0, 64.0)`. E2e `min (x−0.5)²+(y−1)² s.t. x**4·y ≤ 1`
  with `x,y ∈ [−2,2]` returned a false certified `objective≈3.13` at
  `x=2, y=0.0625` (aux `x**4` pinned to the bogus `[16,16]` box), true optimum
  is `0` at `x=0.5, y=1`. (Observed 3.13 rather than the review's 0.25 because
  y's box differs; either way it is a false certified optimum vs the true 0.)
  **Fix:** in the `**` arm of `_bound_expression`
  (`gdp_reformulate.py`), generalized the `p==2` straddle handling to ALL even
  integer powers: for any even `p` with base interval straddling 0
  (`lb < 0 < ub`), return `[0, max(lb**p, ub**p)]` (interior min at 0);
  otherwise (one-signed even, or odd) keep the monotone endpoint bounds. No
  `p==2` vs `p>=4` special-casing remains. After: `x**4→(0,16)`,
  `x**6→(0,64)`, `x**8→(0,256)`; one-signed even (`[1,3]**4→(1,81)`,
  `[−3,−1]**4→(1,81)`) and odd (`[−2,2]**3→(−8,8)`) unchanged; e2e returns
  `≈0` at `x=0.5, y=1`.
  **Regression tests** (`python/tests/test_power_certification.py`):
  `test_even_power_straddle_bound_includes_interior_min` (p=2,4,6,8),
  `test_power_bound_non_straddling_and_odd_unchanged`,
  `test_c34_even_power_straddle_no_false_optimum` (e2e). Fail-before /
  pass-after verified by stashing the fix.
  **Gates:** smoke 193 passed/1 skipped; adversarial suite 10 passed; targeted
  power/reform/mccormick/monomial/gdp sweep 210 passed (only pre-existing
  `st_e11` tolerance flake, proven independent by reverting the fix); ruff
  clean; mypy env-stub error only (pre-existing, hits any file).

---

## C-35 (P1) — OA/LOA emits an unconditional no-good cut on non-rigorous NLP failure (opt-in) — **FIXED**

**Status:** fixed (PR #451).
**Origin:** solver-core review (OA-1).
**Area:** `python/discopt/oa.py:209-236, 893-931, 916` and
`python/discopt/gdpopt_loa.py:314-370, 225`.
**Reachability:** opt-in OA/ECP/LOA solve paths only (not default B&B).

**Mechanism:** when the per-configuration NLP subproblem fails to converge, the code
adds an **unconditional** integer no-good cut excluding that configuration — treating
"NLP did not converge" as "this configuration is infeasible." A local NLP failure is
not a proof of infeasibility, so a feasible (possibly optimal) configuration can be
permanently excluded → false infeasible, or a worse configuration certified optimal.

**Reproduce/confirm:** construct a GDP/OA instance where the optimal disjunct's NLP
is hard to converge from the default start (tight/ill-conditioned) while a suboptimal
disjunct converges easily; assert the OA/LOA path returns the suboptimal objective or
"infeasible" while `Model.solve()` (B&B) returns the true optimum. (Not yet built —
build during the fix.)

**Fix:** make the no-good cut **conditional on a rigorous infeasibility proof** (e.g.
an infeasibility certificate / feasibility-restoration failure), not on NLP
non-convergence. On mere non-convergence, either retry from alternative starts or
**withhold** the cut and downgrade certification, never exclude the configuration.

**Regression test (required — fast):** the constructed instance above as a unit test
asserting the OA/LOA path matches the B&B optimum (no spurious exclusion); a
sub-second GDP. Fails before, passes after.

**Done criteria:** the instance returns the true optimum on the OA/LOA path; no
no-good cut is emitted without a rigorous infeasibility proof; existing OA/LOA tests
unchanged; standing gates pass.

**Log:**
- 2026-07-03 — new opt-in P1 (solver-core review §2).
- 2026-07-03 — **CONFIRMED then FIXED.** Confirmed both false-certificate classes
  on pre-fix code with a direct `solve_oa` repro (monkeypatching the fixed-integer
  NLP subproblem to fail *non-rigorously*, as a diverged/iteration-limited local
  NLP does):
  * **False infeasible** — a feasible model `min (x-0.5)^2+3y s.t. x+y>=0.3` (true
    opt x=0.5,y=0,obj=0) returned `status="infeasible", gap_certified=True` when
    every NLP subproblem merely failed.
  * **False optimal** — same model with only the *optimal* config's (y=0) NLP
    failing returned `status="optimal", objective=3.0` (the suboptimal y=1),
    certified, because the no-good cut excluded the optimal config.

  Root cause was structural: `_solve_nlp_subproblem` collapses *every* non-optimal
  outcome (diverged / iteration-limit-without-feasible / error / exception) to
  `(None, None)`, and the loop's `else` branch treated that as "infeasible" and
  emitted an **unconditional** no-good cut. The NLP layer never even produces a
  rigorous infeasible verdict (`nlp_ipopt._IPOPT_STATUS_MAP` deliberately maps
  IPOPT status 2 `Infeasible_Problem_Detected` → `ERROR`), so *every* no-good cut
  in this branch was unsound.

  **Fix** (`python/discopt/solvers/oa.py`, `python/discopt/solvers/gdpopt_loa.py`):
  added `_fixed_subproblem_rigorously_infeasible(...)` — a *sufficient, rigorous*
  infeasibility test (currently: no free continuous variables remaining ⇒ the
  fixed point is the sole candidate, so a constraint violation there is a complete
  proof). The no-good cut is now emitted **only** when that test passes;
  certification is preserved for genuinely-provable exclusions (e.g. all-integer
  models). On a non-rigorous failure the cut is **withheld**, the configuration is
  recorded as unresolved (OA gradient cuts at the master point are still added —
  those never cut a feasible point), the loop breaks soundly if the master
  re-proposes an unresolved config (anti-cycling without exclusion), and the result
  is downgraded: `gap_certified=False`, never a certified `optimal`, and never a
  certified `infeasible` (a new `status="unknown"` is returned instead when there
  is no incumbent). No rigorous check was weakened.

  **Note (scope, per §0 rule 7):** the fix showed the bug is slightly *broader*
  than the card framing — OA's `optimal` certification on ordinary MINLPs with an
  infeasible integer config was also resting on the unsound no-good logic. The
  rigorous no-continuous-freedom check preserves certification for those (the one
  existing test that momentarily flipped `optimal`→`feasible`,
  `test_oa_maximize_linear_objective_is_not_the_minimum`, is green again). Blast
  radius was a single test; the OA convergence guarantee is not touched broadly, so
  no escalation was needed. A future, more general rigorous infeasibility prover
  (interval/FBBT over the fixed box; feasibility-restoration certificate) would let
  the no-good cut fire in more cases — tracked as a follow-on, not required for
  soundness (withholding is always sound).

  **Regression test:** `python/tests/test_c35_oa_nogood_nlp_failure.py` (4 tests,
  `@pytest.mark.smoke`, sub-second, calls `solve_oa`/`solve_gdpopt_loa` directly):
  no false infeasible (OA + LOA), no false optimal (OA), AND certification IS
  preserved when the infeasible config is rigorously provable (all-integer case).
  Red before the fix (3 fail), green after (4 pass).

  **Gates:** smoke 197 passed / 1 skipped; adversarial 10 passed; OA/LOA/gdpopt/
  nogood/benders 165 passed / 2 skipped; ruff clean; mypy clean on changed files;
  no Rust touched. PR: #451.

---

## Verified-correct (do not re-audit)

From the solver-core review (2026-07-03) — the highest-stakes gate:
- **The convexity certifier and convex fast-path are SOUND.** `certify_convex` /
  `classify_model` never certified a nonconvex function convex across **901
  non-abstaining certificates** in a 4,000-trial fuzz (each cross-checked against the
  sampled Hessian spectrum, 0 false). The interval-Gershgorin eigenvalue layer is
  rigorous (Higham inflation, outward rounding, ±∞ on non-finite Hessian); the
  interval-Hessian AD correctly seeds the Variable leaf from the box. The convex fast
  path (`solver.py:3651`) and MIQP path (`:3619`) fire only on a rigorous verdict;
  `skip_convex_check` only downgrades certification. Pattern recognizers
  (quad-over-linear, geo-mean, perspective, norm2, softplus, 1/x) are Jensen-clean.
  Do not re-audit this surface without new evidence. *(C-33's pure-continuous
  fallback is a separate, non-gated path — that is the bug, not this gate.)*
- The JAX relaxation-compiler Variable leaf is correct (returns `(x_cv, x_cc)`; box
  enters via the call convention) — 0 containment violations on broad composite fuzz.
  (The **numpy** compiler drops the box — NM-2 — but is compiled-but-unused.)

From the B&B audit:
- Global-LB soundness: `update_global_lower_bound` scans by node status, not the
  heap (`tree_manager.rs:557-582`); capped at incumbent; gap can't go negative.
- Maximize-sense sign handling is consistent end-to-end (`solver.py:6470-6605`);
  everything internal is minimize.
- Integrality tolerance uniform (1e-5) across accept/branch/hint gates; untrusted
  (-inf-bound) nodes cannot become incumbents (`tree_manager.rs:332`).

From the cuts audit:
- Root cut-pool inheritance is globally valid: separated once at the ROOT box
  (`solver.py:4359`), consumed read-only per node; node-local box-dependent cuts
  (RLT/edge-concave/PSD) cannot leak into the pool or across subtrees.
- Rust GMI (`gomory.rs`) is hardened: refined tableau rows (residual ~1e-15),
  integer snapping, fractional-bound fallback, coefficient/dynamism caps, margins,
  four validity tests. (GMI does not require an optimal crossover basis — any
  basis's tableau row is a valid equation.)
- OA/tangent gating: `certify_convex` abstains on indefinite; `refresh_convex_mask`
  only flips False→True; α-BB fires only with finite bounds + detected negative
  curvature; c-MIR guards infinite bounds and carries margins; edge-concave fires
  only on detected sign-definite blocks.

From the parser audit:
- Binary `.nl` rejection is clean and tested (both header and first-byte checks).
- Range/eq/ineq bound decoding (codes 0-4) correct, split senses/RHS correct.
- Rust vs JAX edge-case semantics agree (pow neg-base, div-by-zero, sign(0),
  abs(0)); PyO3 MathFunc mapping symmetric both directions including norm{p}.
- The vector-reduction NaN stub fix covers ALL vector-arg funcs (Prod, Norm1/2/
  Inf/P), regression-tested.

From the relaxation-math audit:
- `multilinear_separation.py` (Rikun) is rigorously sound: it never trusts the LP
  dual — the intercept is recomputed as `min/max over box vertices`, so the
  hyperplane is valid for ANY slope; degenerate boxes stay valid.
- `alphabb.py:271 rigorous_alpha` is a correct interval-Gershgorin sufficient
  condition and abstains (+∞) on unbounded Hessians (this is the fix vehicle for
  C-17); the `relax_bilinear` both-nonneg clamp is valid.
- The #120 gate and `"auto"` routing are sound: nlp-mode objective bounds disabled
  for nonconvex (`solver.py:4432-4439`); auto routes nonconvex-continuous to the
  rigorous LP outer approximation; `milp_relaxation.py` consistently gates on
  `_objective_bound_valid`.

From the presolve/FBBT audit:
- Even-power backward with zero-straddling base soundly relaxes to the hull;
  odd-power inversion preserves sign; negative even-power output → empty.
- Multiplicative backward refuses zero-spanning divisors; backward narrows only
  via leaf intersection, so intermediate widening can't loosen bounds.
- Integrality snapping and probing both carry FEAS_TOL (outward snap before
  ceil/floor; probe-infeasible only beyond tolerance; both-feasible children
  union-enclosed — eps-residual regression-tested). Coefficient strengthening
  abstains on unbounded rest-activity; reduced-cost fixing is sound for
  minimization (documented caller contract: maximize duals pre-negated).

From the LP/dual-bound audit:
- The Neumaier–Shcherbina safe dual bound (`obbt.py:55-142`) is rigorous at any
  conditioning (weak duality; returns None on non-finite input);
  `run_obbt_on_relaxation` clamps optimistic vertices to it; DBBT no-ops soundly
  on non-OPTIMAL/missing reduced costs and widens by a margin. Iteration-limit
  duals never reach a bound through the LP bridge.
- Equilibration round-trip is exact (power-of-two factors; objective invariant;
  duals unscaled sign-consistently) — reported bounds are in true units.
- Warm-start `PreparedDual` cannot certify a wrong Optimal: rejects wrong-size/
  singular/dual-infeasible starts → cold fallback; `run_dual` recomputes exact
  x_B and reduced costs and re-audits before returning Optimal (tested).
- EXPAND anti-degeneracy only grows the ratio-test tolerance — actual bounds are
  never shifted, and `assemble` recomputes the objective from true c, x with a
  feasibility audit before certifying. Crossover moves only in the null space of
  A and cᵀ, so cut derivation never inherits an objective drift.


---

## C-36 (P3) — Python `interval_mul` yields NaN on 0·∞ (separate path from C-22)

**Area:** `python/discopt/_jax/convexity/interval.py` (≈ line 171).
**Class:** same `0·∞ → NaN` interval-arithmetic defect as C-22, but a **distinct
code path**: C-22 fixed the *Rust* FBBT `interval_mul`
(`crates/discopt-core/src/presolve/fbbt.rs`); this is the independent *Python*
interval module used by the convexity/relaxation layer. Fixing C-22 does **not**
fix this one.

**Symptom / evidence:** surfaced by the C-22 agent's adversarial run as
`RuntimeWarning: invalid value encountered in multiply` originating at
`convexity/interval.py:171`. A product corner `0 * ±∞` evaluates to IEEE-754 NaN
and `min`/`max` over the four corners propagate it, so an interval that touches
zero times an unbounded interval becomes `[NaN, NaN]`.

**Severity (P3 — sound but weak, pending confirmation):** as with C-22, a
`[NaN, NaN]` interval is *lost information*, not an unsound bound — downstream
comparisons against NaN are all false, so a consumer keeps its prior (looser)
interval rather than adopting a wrong tighter one. **This must be re-confirmed on
this path**, though: if any convexity/relaxation consumer treats a NaN endpoint as
`0`/finite (rather than ignoring it), the classification could flip — verify the
consumer semantics before downgrading confidence. No unsound path is *known*
today.

**Fix (mirror C-22):** in `interval_mul`, map NaN corner products to `0` — NaN can
arise there only from `0 · ±∞`, whose interval-convention value is `0`; every other
operand pair is finite×finite or a genuine ±∞ product. The result is a strictly
tighter (never looser) enclosure and cannot exclude a feasible point. Add a
property test (grid over ±∞ endpoints / zero-width factors: never-NaN, `lo ≤ hi`,
rigorous product containment), mirroring
`c22_interval_mul_never_nan_and_encloses_true_product`.

**Provenance:** discovered by the C-22 fix (#460); filed as a separate item because
it is a separate module with its own tests and consumers.

**Log:**
- (filed) — surfaced during C-22 (#460) verification; carded P3 as the same 0·∞
  class on the Python interval path. Status `open`.
