# MAiNGO Capability-Parity Plan — staged, executable

**Status:** proposed · **Created:** 2026-07-11 · **Owner issue:** #572 · **Prereq
reading (every executor, every stage):** this file top-to-bottom, then
`.crucible/wiki/comparisons/maingo-vs-discopt.org`, the #572 issue thread, and
CLAUDE.md (§Development Philosophy, §5 verification regimes).

This document is written to be executed **stage by stage by a fresh Opus session**
with no other context. Each stage has: orientation reading, verified codebase facts,
deliverables with signatures, an implementation checklist, a test spec, a gate with a
kill criterion, and a PR boundary. Execute stages in order; do not start a stage
whose predecessor's gate has not passed. After each stage, update the **State
ledger** (§7) in the same PR.

---

## 0. Strategic framing (binding)

1. **Capability plan, not throughput plan.** The #557 density-route fix (PR #573)
   removed the FT-refactor storm from the lifted path; reduced-space McCormick is
   pursued for *generality/robustness/expressiveness parity with MAiNGO*, not as the
   primary speed lever. Do not justify a stage by speed alone; do not block a stage
   on beating the lifted path.
2. **Keep discopt's structural advantages.** Every deliverable must remain
   `jax.jit`/`jax.vmap`-compatible (GPU-batchable) and differentiable. A parity
   feature that forces scalar CPU evaluation is a regression.
3. **Sound-or-refuse.** No stage may ever emit a partial or invalid bound. Anything
   out of scope raises (`UnsupportedRelaxation` or a documented equivalent) and the
   caller falls back to a sound alternative (α-BB, lifted). Never weaken a guard to
   make a gate pass — if a gate can only pass that way, the stage FAILS and the
   failure is recorded here.
4. **CLAUDE.md §4/§5 per stage:** entry experiment before implementation where one is
   named; bound-changing changes (anything that can alter a dual bound on an existing
   path) ship behind a default-off flag with the differential-bound + feasible-point
   regime. Library-only stages (new modules, nothing on the default path) need the
   stage's own test gate only.
5. **Workflow:** one stage = one PR (feature branch from current `origin/main`,
   task/stage ID in the title, e.g. `feat(mcbox): P0.2 — …`). Every PR: the stage's
   test spec green + `pytest -m smoke` + adversarial suite
   (`pytest -m slow python/tests/test_adversarial_recent_fixes.py`) + `cargo test
   -p discopt-core` if Rust touched. State what was run in the PR body.

## 1. Verified codebase facts (do not re-derive; re-verify only if stale)

Facts below were verified 2026-07-09..11 on `origin/main` ≈ `7cbe25de`+. If the tree
has moved far, spot-check before relying on them.

- **The collapse bug (why the existing compiler can't be reused for subgradients):**
  `relaxation_compiler.py::compile_objective_relaxation/_compile_relax_node` compile
  fns of signature `(x_cv, x_cc, lb, ub) -> (cv, cc)`. Evaluated at `x_cv == x_cc`
  they **collapse to the exact function value** (verified: `x*y` on `[0,4]²` at
  `(1,1)` returns 1.0, not the envelope 0.0), because composition passes the
  operand's `(cv_l, cc_l)` as the *bounds* of the child envelope (e.g.
  `relaxation_compiler.py:561`). Hence `jax.grad` of these fns is the **true
  (nonconvex) gradient — not a valid subgradient**. Both `mode="standard"` and
  `mode="nlp"` collapse. Any reduced-space work must build envelopes from **box
  bounds**, never operand cv/cc pairs at coincident points.
- **The v1 reduced-space evaluator exists** on branch `feat/572-mccormick-subgrad`
  (worktree `/Users/jkitchin/projects/discopt-572`, 3 commits `c6b89351`, `7a4916f8`,
  `e0a56204`): `python/discopt/_jax/mccormick_subgradient.py` with
  `_RNode(cv, cc, lo, hi, affine)`, `build_reduced_relaxation(model, lb, ub) ->
  ReducedRelaxation(obj_under, con_feas, negate, n)`,
  `reduced_mccormick_lp_bound(model, lb, ub, max_rounds, tol) -> ReducedBound`
  (Kelley loop, **scipy/HiGHS scaffold LP**), `UnsupportedRelaxation`. Scope: affine,
  products/squares of affine, integer powers `affine**n` with a **soundness guard
  refusing odd powers over sign-spanning bases** (there the McCormick cv is piecewise
  non-convex, so `jax.grad` would be invalid). Tests:
  `python/tests/test_mccormick_subgradient.py`, 14 passing (envelope-correctness,
  validity, convexity, subgradient-support, Kelley convergence/infeasible/unsupported,
  maximize sense). Known limitation: **eager JAX, ~12–23 s per 15 Kelley rounds** on
  nvs19/24 — unusable per-node without jit.
- **Envelope kernels to reuse:** `python/discopt/_jax/mccormick.py` —
  `relax_bilinear(x, y, x_lb, x_ub, y_lb, y_ub)` (exact hull; C-24 non-finite-bounds
  guard returns `(-inf, +inf)`), `relax_pow(x, lb, ub, n)` (even convex; odd 3-regime
  with `jnp.where`), `relax_square`, `relax_exp/log/...`, `_secant`.
  **Tsoukalas–Mitsos composition rules exist:**
  `python/discopt/_jax/multivariate_mccormick.py::_COMPOSITION_RULES` — 11 univariate
  ops (`exp log log2 log10 sqrt softplus abs tanh atan sigmoid sinh`), each a
  `compose_*(cv_g, cc_g, g_lb, g_ub) -> (cv, cc)` mid-rule implementation.
- **AST:** `discopt.modeling.core` — `Constant`, `Parameter`, `Variable`,
  `IndexExpression(base, index)`, `BinaryOp(op, left, right)` with ops
  `+ - * / **`, `UnaryOp(op, operand)`, `FunctionCall` (named intrinsics),
  `CustomCall(fn, *args)` (see below). Constraints are normalized `body sense 0`
  (`Constraint.body/.sense/.rhs==0`); objective is `model._objective.expression`,
  sense via `ObjectiveSense.MAXIMIZE`. Scalar leaf → flat column:
  `relaxation_compiler._resolve_scalar_var_offset(expr, model) -> int | None`
  (handles `Variable` and `IndexExpression`). Model construction for tests:
  `m = dm.Model()`, `x = m.continuous("x", n, lb=…, ub=…)`, `m.minimize(expr)`,
  `m.subject_to(expr <= 0)`.
- **`CustomCall` is the P3 hook and it already exists** (`modeling/core.py:433`):
  an opaque **jax-traceable** callable; today documented/enforced as **local-NLP
  only, no global certificate, raises for integers, relaxation compilation and
  `.nl` export raise**. Consumers to touch in P3: `relaxation_compiler.py`,
  `dag_compiler.py`, `nonlinear_bound_tightening.py`, `monotonicity.py`,
  `sparsity.py`, `objective_epigraph.py`, `solver.py`.
- **Solver mode switch:** `solver.py` `_mc_mode ∈ {"auto","lp","nlp","none"}`
  (≈ line 4591+); the default global path is `MccormickLPRelaxer`
  (`_jax/mccormick_lp.py`, lifted aux columns, in-house Rust simplex backend,
  incremental fast path `IncrementalMcCormickLP`, root cut pool). Tuning-knob house
  pattern: `solver_tuning.py` dataclass fields via `_env_flag("DISCOPT_…", default=…)`
  with a docstring naming the env var.
- **In-house LP warm start:** `MilpRelaxationModel.solve(backend="simplex")`
  (`_jax/milp_relaxation.py`) warm-starts the Rust dual simplex when **columns are
  unchanged and rows only grow** — exactly the Kelley-loop shape (`_solve_lp_warm`,
  equilibrated retry, then cold fallback). P2 should ride this, not scipy.
- **DBBT is NOT a gap** — `python/discopt/tightening.py` implements FBBT/DBBT/OBBT;
  DBBT is wired in `solver.py` ≈ 7072. Do not rebuild.
- **Probes/corpus:** `~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/*.nl`
  (`dm.from_nl(path)`; needs the built Rust ext `discopt._rust`), oracle
  `minlplib.solu`. Dense-QP probes used throughout #557/#572: nvs19, nvs24, st_e36
  (opt −1098.19, −1033.20, −246.0). In-repo corpus:
  `python/tests/data/minlplib_nl/` (85 files).
- **Env quirk:** the installed `discopt` package is a namespace shim; run repo code
  with `PYTHONPATH=<repo>/python python …`. A worktree needs its own maturin build
  for `discopt._rust` (`from_nl`); pure-Python model tests need no Rust build.

## 2. P0 — `MCBox`: a propagating McCormick type (the MC++ analogue, the JAX way)

**Goal.** A registered JAX pytree `MCBox` carrying `(cv, cc, lo, hi, sub_cv, sub_cc)`
with operator overloads, propagating the McCormick relaxation **and its subgradients
by per-operator rule** (Mitsos–Chachuat–Barton 2009) through arbitrary jax-traceable
code. This subsumes the v1 DAG walk, removes its two weaknesses (operator-table
walls; `jax.grad`-over-construction fragility that forced the odd-power refusal), and
stays `jit`/`vmap`/GPU-able.

**Design invariants (all stages):**
- `cv`, `cc` are scalars (or arrays elementwise); `sub_cv`, `sub_cc` are length-`n`
  vectors (n = number of independent variables of the enclosing relaxation). Leaf
  variable `i` seeds `cv=cc=x_i`, `lo=lb[i]`, `hi=ub[i]`, `sub_cv=sub_cc=e_i`.
- Envelope bounds always come from the **static box intervals** (`lo`,`hi` fields),
  propagated by interval arithmetic alongside — never from cv/cc values (the
  collapse-bug lesson).
- Piecewise selections (`max` of affine underestimators, 3-regime odd powers,
  sign-splits) select the **subgradient of the active piece** with the same
  `jnp.where` predicate that selects the value. This is what makes the v1
  odd-power-spanning-zero refusal unnecessary: a valid subgradient of a convex
  `max(f1,f2)` is the active branch's gradient, chosen *by rule*, not by autodiff
  of a non-convex construction. (The *envelope itself* must still be convex — the
  odd-power spanning case's cv from `relax_pow` is genuinely non-convex, so P1
  replaces it with the convex hull construction for that regime rather than
  refusing; see P1.3.)
- NaN discipline: follow the codebase pattern (cf. C-24 in `relax_bilinear`,
  the xlogx `jnp.where` guard at `relaxation_compiler.py:537-563`) — discarded
  `jnp.where` branches must not evaluate to NaN and poison gradients; non-finite
  bounds produce explicit `(-inf, +inf)` no-information brackets, and any consumer
  of a no-information bracket refuses.

### P0.1 — entry experiment (run BEFORE building the real module)

Prototype in a scratch file (not committed to the package): `MCBox` as a
`jax.tree_util.register_pytree_node_class` dataclass with only `{+, -, neg,
scalar-mul, *, exp}`; `*` via `relax_bilinear` values + rule-based subgradients
(affine pieces: `cv = max(cv1,cv2)` ⇒ `sub = where(cv1>=cv2, ∇cv1, ∇cv2)` where
`∇cv1 = x_lb·sub_y + y_lb·sub_x` etc.); `exp` via the Tsoukalas–Mitsos
`compose_exp` mid-rule with the mid-point's branch selection reused for the
subgradient.

Test on `f(x,y) = x*exp(y) − x*y` over `[0,2]×[0,1.5]`, all checks sampled ≥20k
points, plus a 1k-box `vmap`:
1. **Bracket:** `cv(x) ≤ f(x) ≤ cc(x)` everywhere (tol 1e-9 rel).
2. **Subgradient support:** `cv(x1) ≥ cv(x0) + sub_cv(x0)·(x1−x0)` for 200 random
   pairs (and the mirrored check for `cc`). No convexity guard involved.
3. **jit/vmap:** `jax.jit` compiles; `vmap` over 1000 boxes runs; per-evaluation
   cost under jit ≤ **3×** a plain jitted `f` evaluation (measure with
   `%timeit`-style min-of-N on CPU).

**Gate:** all three pass. **Kill criterion:** if (3) exceeds **10×** after honest
optimization effort (pytree flattening overhead, branch bloat), P0 is dead: record
the numbers here, fall back to extending the v1 DAG walk per-phase instead (P1
still proceeds, on `_RNode`), and re-scope P3 to DAG-walk-over-`CustomCall.fn`
tracing. Record the entry-experiment numbers in §7 either way.

### P0.2 — the real module

**Deliverable:** `python/discopt/_jax/mcbox.py` —

```python
@jax.tree_util.register_pytree_node_class
@dataclass
class MCBox:
    cv: jnp.ndarray; cc: jnp.ndarray          # relaxation values at the point
    lo: jnp.ndarray; hi: jnp.ndarray          # static interval over the box (aux data, not leaves)
    sub_cv: jnp.ndarray; sub_cc: jnp.ndarray  # subgradients, shape (n,)
    # dunders: __add__ __radd__ __sub__ __rsub__ __neg__ __mul__ __rmul__
    #          __truediv__ (const divisor + sign-definite reciprocal) __pow__ (int)

def mcbox_leaves(x: jnp.ndarray, lb: jnp.ndarray, ub: jnp.ndarray) -> list[MCBox]:
    """Seed one MCBox per variable: cv=cc=x[i], lo/hi=lb/ub[i], sub=e_i."""

def relax_through(fn: Callable, x, lb, ub) -> MCBox:
    """Trace fn(*mcbox_leaves(...)) and return the composite MCBox.
    fn is any jax-expressible callable written against MCBox-compatible ops."""

class UnsupportedMcboxOp(Exception): ...   # sound-or-refuse, same contract as v1
```

plus a function namespace `discopt._jax.mcbox.ops` (or methods) for `exp`, `log`,
`sqrt`, `abs`, `tanh`, `atan`, `sigmoid`, `softplus`, `sinh`, `log2`, `log10` —
each implemented by calling the existing `_COMPOSITION_RULES` kernel for values and
adding the rule-based subgradient (the mid-rule's median selection is a `clip`;
its subgradient is the clipped-argument's chain rule — implement once as a helper
`_compose_with_subgrad(compose_kernel, envelope_derivs, g: MCBox) -> MCBox`).

Interval propagation: `lo/hi` per node via the existing interval logic (reuse
`_interval_prod`-style helpers from v1; intrinsics take monotone-interval images or
the kernels' interval variants). Mark `lo/hi` as **aux_data** in the pytree (they
are box constants, not traced values), so `vmap` batches over `x` cleanly; document
that a new box ⇒ rebuild leaves (same as MC++).

**Checklist:** module + ops table; docstring stating the collapse-bug rationale and
the piece-selection subgradient rule; every op has a NaN-discipline note; no import
cycles (`mcbox.py` imports `mccormick.py`/`multivariate_mccormick.py`, never the
compiler).

**Test spec:** `python/tests/test_mcbox.py`
- per-op property tests (each op, 3 boxes incl. sign-spanning and non-finite-guard):
  bracket (5k samples), subgradient-support (200 pairs), interval containment
  (`lo ≤ f ≤ hi` sampled);
- composite: the P0.1 function + `x*log(x+y)+tanh(x*y)` bracket/support;
- pytree: `jit`, `vmap` over boxes, `grad` *through* `cv` w.r.t. `x` runs (AD
  compatibility — value only, no soundness claim);
- refusal: an unsupported op (e.g. `__mod__`) raises `UnsupportedMcboxOp`.

**Gate:** test spec green; P0.1's ≤3× cost bound re-confirmed on the real module.
**PR:** `feat(mcbox): P0.2 — propagating McCormick pytree with rule-based
subgradients`. Library-only (nothing on any solve path) → no flag needed.

### P0.3 — subsume the v1 evaluator

Rebase/merge branch `feat/572-mccormick-subgrad` (the v1 module + its 14 tests) into
the P0 line; reimplement `build_reduced_relaxation` on `MCBox` (walk the Model AST,
build a jax-traceable closure, call `relax_through`), keep `UnsupportedRelaxation`
for *model-level* refusals (non-scalar leaves, unbounded boxes). **All 14 v1 tests
must pass unchanged** (they are the regression floor), except
`test_odd_power_spanning_zero_refused`, which flips to a *soundness* test once P1.3
lands (until then it remains a refusal). Delete none of the tests.

**Gate:** v1 suite green on the MCBox implementation; the Kelley loop
(`reduced_mccormick_lp_bound`) runs on it unchanged.
**PR:** `feat(mcbox): P0.3 — port the #572 reduced-space evaluator onto MCBox`.

## 3. P1 — full-library reduced-space coverage

**Goal.** Retire the QP-only scope: every operator the modeling API exposes (the
§1/§2 tables of `docs/design/relaxation-catalog.md`) relaxes through `MCBox`, with
refusal only where mathematically undefined over the box.

Stages (one PR each, same test template as P0.2 per operator):
- **P1.1 univariate intrinsics** already in `_COMPOSITION_RULES` (11 ops) — mostly
  done in P0.2; add the stragglers the catalog lists (`sin`, `cos`, `tan`, `asin`,
  `acos`, `asinh`, `acosh`, `atanh`, `erf`, `log1p`, `cosh`, `sign`) by wrapping
  their `envelopes.py`/`mccormick.py` kernels with the `_compose_with_subgrad`
  helper. Regime-based kernels (trig) need per-regime subgradient selection.
  **P0.2 pushed tanh/atan/sigmoid/sinh here** (their cv is non-convex over a
  sign-spanning box, so the P0.2 kernel-chain subgradient is invalid): implement
  per-regime by splitting the box at the inflection and selecting the active
  regime's subgradient with the same predicate that selects the value.
- **P1.2 products/divisions of non-affine operands** — general bilinear composition
  `relax_bilinear(cv_a-or-cc_a, …)` per McCormick's product rule with the four-case
  `mid` selections and their subgradients (this is the real MCB-2009 product rule —
  implement from the paper, property-test heavily; it is the single hardest kernel).
  Division via sign-definite reciprocal composition; refuse `0 ∈ [y_lb, y_ub]`.
- **P1.3 odd powers over sign-spanning bases, soundly** — replace the v1 refusal
  with the convex-hull cv of `x^{2k+1}` on `[lb,ub] ∋ 0` (tangent-line construction:
  cv is the max of the function's convex part and the tangent from `(lb, lb^n)`;
  standard construction, cite Liberti–Pantelides in the docstring), with rule-based
  subgradients. Flip the v1 refusal test to a soundness test.
- **P1.4 fractional powers / signomials** (`x**a`, `a` non-integer, `lb>0`) via the
  existing signomial kernels.

**Stage gate (each):** per-op property tests green; **P1 exit gate:** st_e36
(`x**3`, positive base), one sign-spanning-odd-power synthetic, and ≥5
transcendental MINLPLib instances (pick from `python/tests/data/minlplib_nl/` so no
external corpus is needed — e.g. `alan`, `gbd`, `ex4_1_1`, plus two with
`sin`/`exp`) produce **valid** reduced-space bounds (≤ sampled min, 20k samples)
with zero refusals among them. **Kill criterion:** none (coverage work) — but any
operator whose subgradient rule cannot be property-test-validated ships as a
refusal, never as a "probably fine" rule.

## 4. P2 — `DISCOPT_RELAX_SPACE` solver mode (task #69)

**Goal.** Reduced-space Kelley bounding as a selectable per-node engine on the
**in-house simplex**, jitted, flag-gated, default-off.

### P2.1 — jit the Kelley evaluator (entry experiment)

The v1 loop is eager (~12–23 s / 15 rounds on nvs19/24 — measured, unusable).
Restructure so each round is one jitted call: `eval_and_subgrad(x) ->
(u0, g, h0s, gg s)` compiled once per (model, box-shape); rounds reuse it.
**Gate:** ≥50 Kelley rounds on nvs24's root in **<1 s total** after compile
(compile time excluded, reported separately). **Kill criterion:** if jit cannot get
a round under ~20 ms on these probes, the reduced mode cannot be a per-node engine;
re-scope P2 to "root + OBBT-probe bounding only" and record it here.

### P2.2 — inner LP on the in-house simplex

Replace the scipy scaffold: assemble the Kelley LP (`n_orig+1` columns, rows grow
monotonically) as a `MilpRelaxationModel` (or the thinnest direct binding available)
and solve with `backend="simplex"` so the **rows-only-grow warm start** engages
between rounds. Keep scipy as a `DISCOPT_REDUCED_LP_BACKEND=scipy` debug fallback.
**Gate:** identical bounds (≤1e-8 rel) to scipy on the P0/P1 test battery;
`DISCOPT_PROFILE=1` on the Kelley LP sequence for st_e36 shows **no
`RefacFtTinyPivot` storm** (the n_orig-column bases must be clean; this was the
de-risk experiment promised in #572 — record the counter values here).

### P2.3 — wire the mode

- `solver_tuning.py`: `relax_space: str` field, env `DISCOPT_RELAX_SPACE`,
  values `{auto, lifted, reduced, hybrid}`, default `lifted` (byte-identical);
  `hybrid` reserved/rejected loudly until P2.5.
- `solver.py`: where `_mc_mode` resolves (≈4591–4750), a `reduced` selection builds
  the P0/P1 relaxation once per model, then per node calls the Kelley bound with the
  node box; `UnsupportedRelaxation` at build time ⇒ log once + fall back to the
  lifted path (sound fallback, never an error to the user).
- Node interface parity: return the same fields the lifted node result carries
  (bound, status, dual info for DBBT if cheaply available; else document DBBT skip
  on reduced nodes).
- Branching: unchanged (branch on original variables — reduced space makes every
  branching variable original by construction).

**Verification (bound-changing regime):** default `lifted` byte-identical on the
cert panel (node_count + objective, the 10-instance deterministic panel from #573 is
a good template); `reduced` mode on the P1 exit-gate instances: differential bound
(≥ v1 bounds where comparable), feasible-point sampling (no valid point cut —
reuse the adversarial-suite pattern), full smoke + adversarial with the flag ON.

**Gate:** all the above + nodes/s measured on {nvs19, nvs24, st_e36} reduced vs
(post-#573) lifted, reported in the PR — win or lose. **Kill criterion:** if
reduced loses nodes/s on *every* class (dense-QP, transcendental, and the P3
flowsheet probe when it exists), it ships as opt-in capability and `auto` never
selects it; parity kept, no default claim. That is an acceptable outcome — record
it, don't fight it.

### P2.4 — `auto` policy (only if P2.3 found a winning class)

A structural gate (never instance names): e.g. "model has CustomCall" (P3) or
"lifting ratio n_aux/n_orig > τ with storm-prone m-band". Graduation per CLAUDE.md
§5: default-off flag → green streak → default-on PR with the panel evidence.

### P2.5 — `hybrid` (optional, after P3)

Najman-2021-style MC↔AVM hybrid: lift the subexpressions where AVM tightness pays
(dense-integer QP per #554/#567 evidence), reduced-space elsewhere. Per-term
structural gate. Defer; file its own entry experiment when reached.

## 5. P3 — hidden intermediates + DOF-only trees (`CustomCall` goes global)

**Goal.** MAiNGO's signature capability: models written as opaque jax code
(`y = F(x)`) get **certified global** solves, branching only on the true degrees of
freedom. The hook **already exists**: `modeling/core.py::CustomCall` — currently
documented as local-NLP-only, `gap_certified=False`, raises for integers, and
relaxation compilation raises. P3 upgrades it: a `CustomCall` whose callable traces
through `MCBox` becomes globally relaxable.

### P3.1 — relax `CustomCall` through MCBox

In `relaxation_compiler.py` (and the reduced-space builder), a `CustomCall` node
attempts `relax_through(node.fn, …)` on MCBox leaves built from its argument
relaxations' boxes; `UnsupportedMcboxOp` inside the trace ⇒ the *existing* refusal
path (local-NLP only), unchanged behavior. Success ⇒ the node yields a genuine
`(cv, cc, subgradients)` and the model is eligible for the global path. Update the
`CustomCall` docstring's "consequences" list; keep `.nl` export refusal (a hidden
fn has no `.nl` form). FBBT/interval: propagate `lo/hi` through the same trace for
`nonlinear_bound_tightening.py`/Rust-presolve consumers, or document that hidden
models skip Rust FBBT (Python interval only) — decide by reading how
`dag_compiler.py` handles `CustomCall` today, and record the decision here.

**Test spec:** a `CustomCall` wrapping `lambda x, y: x*jnp.exp(y) - x*y` solves
globally with certificate on a box; bracket/support property tests through the
wrapper; an MCBox-unsupported callable still routes to local-NLP exactly as today
(regression: existing CustomCall tests unchanged).

### P3.2 — integer support for MCBox-relaxable CustomCall models

Lift the "raises if integers present" guard **only** when every `CustomCall` in the
model proved MCBox-relaxable (the guard exists because there was no valid node
relaxation; now there is one). B&B over integer + continuous DOF with reduced-space
Kelley node bounds. Regression: non-relaxable CustomCall + integers still raises.

### P3.3 — the flowsheet gate (entry experiment for the whole of P3, run FIRST
as a prototype before P3.1 hardening)

Build a small flowsheet-style model: 2–3 sequential unit functions as `CustomCall`s
(e.g. flash + mixer algebra, ~10–20 internal intermediates each), ≤6 DOF, 1–2
integer choices. Solve (a) reduced-space via P3, (b) the flattened/lifted
formulation of the same model. **Gate:** the reduced tree is dramatically smaller
(MAiNGO's published regime — expect ≥5× fewer nodes) and both certify the same
optimum. **Kill criterion:** if the reduced tree is not smaller even on this
favorable case, STOP P3, record the falsification here, and re-scope (the
capability may still ship for expressiveness — CustomCall-global — but the
DOF-tree-size claim is dead).

**PR boundary:** P3.1 and P3.2+P3.3 as separate PRs; P3.3's model becomes a
permanent test + a docs notebook (`docs/notebooks/`, with `{cite:p}` citations —
Bongartz & Mitsos 2017 at minimum — per the CLAUDE.md notebook checklist).

## 6. P4 — implicit-function / algorithm relaxation (unscheduled)

MC++/MAiNGO relax implicit solutions (`h(z,x)=0`) and iterative algorithms
(Stuber–Scott–Barton; MCB 2009 §algorithms). Deepest remaining exclusive, least
common need. **Do not schedule** until P3 ships and a concrete user model demands
it; then file a dedicated issue with its own entry experiment (candidate approach:
propagate MCBox through K fixed-point iterations of a contractive map + rigorous
remainder enclosure; kill if the enclosure blows up on the motivating model).

## 7. State ledger (update in every stage PR)

| Stage | Status | Evidence / notes |
|---|---|---|
| P0.1 entry experiment | **PASS (2026-07-11)** | `f=x·exp(y)−x·y` on `[0,2]×[0,1.5]`: bracket valid (cv−f ≤ −2e-5, f−cc ≤ −1.6e-4); subgradient support **0/200** cv & cc violations (rule-based, no convexity guard); jit + vmap-over-1000-boxes at **2.61×** plain-f cost (≤3× gate). **Design refinement adopted:** `lo/hi` are DYNAMIC traced leaves (not aux_data) → jit compiles once, vmaps over per-node boxes; interval-sign selections via `jnp.where`. Scratch: `scratchpad/p0_1_mcbox.py`. GO. |
| P0.2 MCBox module | **DONE (2026-07-11)** | `feat/mcbox-p0` commit `5d7fdf4f`: `_jax/mcbox.py` (MCBox pytree, arithmetic + bilinear + sign-agnostic repeated-mult powers + exp/log/log2/log10/sqrt/softplus/abs), `test_mcbox.py` **22/22**. **Finding (binding for P1.1):** the kernel-chain subgradient is valid **exactly for provably-convex envelopes** — the S-shaped ops (tanh/atan/sigmoid/sinh) have a non-convex cv over a sign-spanning box, so they *refuse* in P0.2 and move to P1.1 (per-regime subgradient selection). Powers are sound via repeated bilinear multiplication for all n≥1 and all signs (looser than the tight monomial hull → P1.3). Not merged (library-only; PR pending). |
| P0.3 v1 port | **DONE — PR #575 (2026-07-11)** | `mccormick_subgradient.py` reimplemented on MCBox (`_to_mcbox` AST interpreter); 14/14. **Strictly more capable:** 12 core v1 tests unchanged (regression floor); 2 v1 refusal tests flipped to soundness (x**3 spanning, exp(x)*x now relax soundly). P0.2 (`5d7fdf4f`) merged in **#574**; density-route #557 fix merged in **#573**. |
| P1.1 S-shaped ops | **DONE — PR #576 (2026-07-11)** | tanh/atan/sigmoid/sinh in MCBox: kernel-chain (tight) on non-spanning boxes, sound constant-envelope fallback (`jnp.where`, loose) on spanning boxes — jit/vmap-able. Diagnostic confirmed the sigmoidal cv is convex non-spanning, non-convex spanning (issue #51). test_mcbox 27/27. **P1.1b** follow-up: tight tangent envelope for the spanning case. P0.3 merged in **#575**. |
| P1.2 division | **PARTIAL — reciprocal-of-affine only (2026-07-10)** | PR #577 added `x/y` via sign-definite reciprocal (mid-rule clamp; no-info bracket when denom crosses 0); sound for `1/(affine)`. **Correction (P2.3, nvs22):** the general **non-affine-denominator** case (`(A·x6)/((x2·x3)·(Σ-sq))`) produces an INVALID `cc` subgradient (numeric witness in §7 P2.3). `_to_mcbox` now REFUSES division by a non-affine denominator (sound-or-refuse); reciprocal-of-affine and non-affine *products* remain sound. FOLLOW-UP: a validated non-affine reciprocal subgradient to lift the refusal. test_mcbox 30/30, `min x/y`=exact 1/3 still hold. |
| P1.3–P1.4 coverage | not started | P1.3 tight monomial hull (`x**n` via relax_pow, tighter than repeated-mult); P1.4 fractional powers/signomials. P1.1b: tight sigmoidal spanning envelope. |
| P2.1 jit Kelley | not started | eager baseline: 12–23 s / 15 rounds (nvs19/24) |
| P2.2 in-house LP | not started | de-risk: check `RefacFtTinyPivot` ≈ 0 on Kelley bases |
| P2.3 mode wiring | **SOUND, default-OFF (2026-07-10)** | Branch `feat/relax-space-p2.3`: `solver_tuning.relax_space` + env `DISCOPT_RELAX_SPACE` (default `lifted`), reduced bound fed into both the batched and serial node loops, sound-or-refuse fallback. **Default byte-identical: PASS** (11-instance panel unchanged). **Corpus soundness sweep: PASS, incorrect_count=0** across 21 MINLPLib instances (curated small + nvs19/24/st_e36/nvs22 + QPs) cross-checked vs `minlplib.solu` — no false-optimal, no invalid bound, no false-infeasible. **Feasible-point soundness: PASS** (74,015 samples, 0 cuts). **Adversarial suite with the flag ON: 10/10** (incl. nvs22). **nvs22 false-optimal FIXED** — three root causes: (1) unbounded leaves (#582 `_require_finite_box`); (2) the wiring fed the reduced evaluator the LIFTED node box (15 cols incl. ~1e8-bound factorable-reformulation aux vars) instead of the original DOF — fixed by building on `_prereform_model` and slicing node boxes to `_prereform_nvars`; (3) the deep defect — an **INVALID `cc` subgradient of the non-affine division** in nvs22 con2 (`(A·x6)/((x2·x3)·(Σ-of-squares))`), whose Kelley cut excluded the true optimum by ~1.7e5 (numeric witness: cut idx 20 at iterate `[1,1,1,1,42.2,1,1.12,0.064]`). Fixed sound-or-refuse: `_to_mcbox` now **refuses division by a non-affine denominator** (`UnsupportedRelaxation` → status `unsupported` → whole-solve lifted fallback); reciprocal-of-affine is still accepted. Defense-in-depth: an `infeasible` from the in-house simplex is cross-checked against scipy/HiGHS before it is trusted to fathom (never a single mis-solve). nvs22 now certifies 6.0582 (35 nodes, = lifted). nodes/s: reduced ~5× slower/node on st\_e36 where it applies — capability, not speed (§0.1); `auto` still never selects reduced (P2.4). NOT merged; wiring stays default-OFF pending the P2.4 graduation gate. |
| P2.4 auto policy | blocked on P2.3 | |
| P2.5 hybrid | deferred | |
| P3.1 CustomCall relax | not started | prototype P3.3 first |
| P3.2 integers | blocked on P3.1 | |
| P3.3 flowsheet gate | not started | THE go/no-go for P3's tree-size claim |
| P4 | unscheduled | |

Prior falsifications binding on this plan (§4 of CLAUDE.md — measurements win):
- Linearizing `compile_objective_relaxation` fns is **unsound** (collapse bug) —
  never reuse them for subgradients (#572 entry experiment, 2026-07-09).
- Root-box fixed-grid linearization is hopeless; Kelley-at-the-LP-optimum +
  branching is the correct realization (the "needs a convex-QP engine" caveat was
  a measurement artifact — #572 correction, 2026-07-10).
- Reduced-space tightness ≈ lifted McCormick on the dense-QP probes at the root
  (the win there is structural, not bound tightness) — do not promise tighter
  root bounds on QPs.
- The FT-storm motivation for reduced space is **gone** (#573 density route);
  §0.1's capability framing is the binding rationale.
- **The reduced-space evaluator's non-affine-division subgradient was unsound**
  (P2.3, found+fixed 2026-07-10): the general `x/y = x·(1/y)` composition produced
  an INVALID `cc` subgradient on `nvs22` con2, cutting off the true optimum → false
  `infeasible` → false optimal. RESOLUTION (sound-or-refuse, CLAUDE.md §3):
  `_to_mcbox` refuses division by a **non-affine denominator** (reciprocal-of-affine
  stays sound); the caller falls back to lifted. Two adjacent bugs fixed with it:
  the solver must feed the reduced evaluator the **original-DOF box**
  (`_prereform_model` / `_prereform_nvars` slice), NOT the factorable-lifted node box
  (whose ~1e8 aux bounds also made the Kelley LP mis-solve); and the unbounded-leaf
  refusal (#582). Corpus sweep now clean (incorrect_count=0, 21 instances) + adversarial
  10/10 with the flag ON. FOLLOW-UP (P1.2): implement a *validated* non-affine
  division/reciprocal subgradient to widen scope beyond reciprocal-of-affine; until
  then the refusal is the correct, sound behavior.

## 8. What "parity" means at the end

A user can (1) write any jax-expressible model — including hidden `CustomCall` code
and embedded surrogates — and get a sound reduced-space McCormick relaxation with
valid subgradients; (2) select `relax_space=reduced` (or `auto`/`hybrid` where
graduated) and get a certified global solve branching only on their degrees of
freedom, integers included; (3) keep discopt's lifted/RLT tightness, GPU batching,
and differentiability wherever those win. MAiNGO's remaining exclusives: implicit/
algorithm relaxation (P4, tracked) and MPI scale-out (explicit non-goal).
