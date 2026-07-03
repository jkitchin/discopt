# Correctness issue backlog — audit findings, loop-executable

**Date:** 2026-07-02
**Status:** complete backlog — all six audit reports incorporated (24 issues)
**Source:** six-agent soundness audit of the relaxation math, presolve/FBBT, B&B
status logic, cut validity, `.nl`/IR ingestion, and LP/dual-bound layers.
**Companion:** `docs/dev/certification-gap-plan.md` (performance). This file is
correctness only.

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
   - every fix includes a NEW regression test that fails before the fix and passes
     after (name it in the Log).
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
| C-17 | P0 | alphaBB bound | node bound uses sampled (non-rigorous) α + center-only PSD check → false "optimal", default path for small nonconvex | open |
| C-13 | P0 | solver.py bounds | serial convex path trusts under-converged NLP objective as node lower bound → false "optimal" | open |
| C-1 | P0 | solver.py status | false "infeasible" from non-rigorous NLP fathoms (solve_model path) | open |
| C-4 | P1 | mir.rs cuts | integer-MIR applied with fractional integer lower bound → invalid cut | open |
| C-2 | P1 | milp_driver status | false "Infeasible" when deadline orphans deferred nodes | open |
| C-19 | P1 | relax_tan | pole-straddling interval classified as one branch → secant across a pole, invalid envelope | open |
| C-5 | P1 | .nl parser | floor/ceil/round/trunc→identity, intdiv→div, all silent | open |
| C-6 | P1 | modeling API | integer vars silently clamped to [0, 1e6] | open |
| C-18 | P1 | midpoint bound | `mccormick_bounds="midpoint"` returns u(mid), not a lower bound (opt-in mode) | open |
| C-20 | P2 | fbbt_fp.rs | watch-list FBBT declares infeasibility with zero tolerance (opt-in engine) | open |
| C-15 | P2 | obbt.py | `run_obbt` variant tightens to raw LP vertex, no NS safe-bound clamp | fixed |
| C-14 | P2 | milp_driver | LP-infeasible fathom trusts status alone; Farkas ray never verified | open |
| C-21 | P2 | incremental MC | soundness-gate validation boxes never exercise negative/zero-spanning bounds | open |
| C-7 | P2 | .nl parser | defined variables (V segments) discarded → hard parse error | open |
| C-8 | P2 | .nl parser | common opcodes unhandled (o76/o77/o78/o48/o11/o12/o35) | open |
| C-9 | P2 | .nl parser | nlvo>nlvc integer-block classification unverified | open |
| C-10 | P2 | lp_spatial cuts | GMI cuts appended without rhs safety margin (opt-in path) | open |
| C-3 | P2 | solver.py incumbent | unrounded integer incumbent survives if terminal polish throws | open |
| C-11 | P2 | modeling API | missing `__ne__` → `x != y` silently evaluates False | open |
| C-22 | P3 | fbbt.rs interval | `interval_mul` NaN endpoints on 0·∞ (lost tightening, not unsound) | open |
| C-23 | P3 | mccormick.py | `_relax_reciprocal` wrong concavity label for negative denominators (latent) | open |
| C-24 | P3 | mccormick.py | secants produce NaN on infinite bounds; soundness leans on downstream filters | open |
| C-12 | P3 | .nl parser | range-split renumbers constraints vs source indices | open |

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

**Reproduce/confirm:**
1. Static: read `_compute_alphabb_bound` and confirm the α source is
   `estimate_alpha` and the PSD check is center-only.
2. Dynamic: construct a low-dim objective with a sharp negative-curvature spike
   off-center (e.g. `f(x)=x⁴ − c·exp(−k(x−a)²)` tuned so center Hessian is PSD
   with the sampled α); assert `_compute_alphabb_bound > min_box f` (direct grid
   check). Deterministic seed for the sampler.

**Fix:** replace `estimate_alpha` with `rigorous_alpha` in the node-bound path,
and drop the center-only PSD gate (rigorous α guarantees convexity by
construction; when it abstains — unbounded interval Hessian — return no bound
rather than a guessed one). Keep `estimate_alpha` only for non-certifying
heuristic uses, renamed/flagged accordingly.

**Done criteria:**
- The adversarial construction returns a bound ≤ true box minimum (or no bound).
- Differential bound test on a panel of nonconvex ≤50-var instances: alphaBB bound
  never exceeds the oracle box minimum across 100 random boxes each.
- Benchmarks: node counts may increase (weaker but sound bounds) — acceptable;
  incorrect_count must not.
- Standing gates pass.

**Log:** —

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

**Log:** —

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

**Log:** —

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

**Log:** —

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

**Log:** —

---

## C-19 (P1) — `relax_tan` classifies pole-straddling intervals as a single convex/concave branch → secant across a pole

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

**Log:** —

---

## C-5 (P1) — `.nl` parser silently rewrites floor/ceil/round/trunc to identity and intdiv to real division

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

**Log:** —

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

**Log:** —

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

**Log:** —

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

**Log:** —

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

**Log:** —

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

**Log:** —

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

**Log:** —

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

**Log:** —

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

**Log:** —

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
  adversarial suite 10 passed; ruff clean. PR: #401 (stacked on #399).

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

**Log:** —

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

**Log:** —

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

**Log:** —

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

**Log:** —

---

## C-23 (P3) — `_relax_reciprocal` labels `1/y` convex on (−∞,0); it is concave there (latent)

**Area:** `python/discopt/_jax/mccormick.py:102-116` (sets `cv=1/y, cc=secant` on
the negative branch — inverted). Currently benign: `relax_div` (`:135`) uses only
the exact-value side with correct bounds, and `relax_fractional`
(`envelopes.py:156-161`) min/max-combines away the bad branch. Latent trap for any
future caller trusting `recip_cc`. The correct implementation already exists at
`envelopes.py:578` (`relax_reciprocal`).

**Fix:** branch on `y_ub < 0` and swap cv/cc (match `envelopes.py:578`); add the
envelope property test `cv ≤ 1/y ≤ cc` on negative intervals.

**Done criteria:** property test green on positive AND negative branches; existing
div/fractional relaxation tests unchanged; standing gates pass.

**Log:** —

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

**Log:** —

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

**Log:** —

---

## Verified-correct (do not re-audit)

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

