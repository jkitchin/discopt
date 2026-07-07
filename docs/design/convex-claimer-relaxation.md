## Goal

Make discopt's LP relaxation **keep convex subexpressions exact** instead of McCormick-decomposing them. When a multivariate polynomial subexpression (e.g. the convex quadratic `3x²+2y²+xy`) is certified convex/concave over the node box, lift it to a single aux `d` and relax it with supporting-hyperplane (gradient) cuts — exactly what SCIP's convex nonlinear handler does — rather than relaxing each `x²`/`xy` term separately and accumulating McCormick slack.

This is the first handler of the hybrid architecture (a complete base + tight pattern recognizers, which is what SCIP/BARON are): discopt already has the tight pattern recognizers; this adds the convex-exact one.

## What the spikes proved (so this is NOT a from-scratch build)

Phase-0 spikes (on branch `feat/convex-claimer-relaxation`) established:

1. **The detector is sound and sufficient.** `convexity.classify_expr(expr, model)` is conservative (never false-certifies a non-convex expression — verified on `x·y`, `sin`, sign-straddling `x³`), is **box-dependent** (`x³` convex on [0,2], UNKNOWN on [-2,2]), and certifies every target: `x⁴`, `x⁶`, `3x²+2y²+xy`, sums of even powers. Only miss: nested `(convex)²` (a recall gap, not a soundness issue).
2. **The evaluator/gradient exists.** `dag_compiler.compile_expression(expr, model)` + `jax.grad` gives `f(x)` and `∇f(x)` for any subexpression (validated vs analytic + finite-diff).
3. **The relaxation machinery already exists.** `CompositeMultivarRelaxation` (`milp_relaxation.py:3401`) + `_collect_composite_multivar_relaxations` (`:4069`) already lift a convex/concave node to an aux and emit gradient cuts `d ≥ g(xₖ)+∇g(xₖ)·(x−xₖ)`, certified by DCP **or** the *general* interval-Hessian Gershgorin PSD certificate `_multivar_box_curvature` (`:3942`, "covers every twice-differentiable multivariate node, not one problem class"). It is used today for `tspn*` Euclidean-distance objectives.

**So the only thing blocking convex sums is the claim *gate*** `_should_claim_composite_multivar` (`:3908`), which returns `False` for a `BinaryOp('+')` (it claims only single call/power nodes, and excludes integer powers).

## The one real obstacle (found by the builder-ordering investigation)

The linearizer is shape-agnostic — it resolves a claimed node by `id(expr)` at the top of every visit (`milp_relaxation.py:4496-4501`) and never descends into its children, so a claimed sum is relaxed as one column with **no double-relaxation** (soundness is not threatened). BUT:

- The composite collectors run on the **raw** `model._objective.expression` / `constraint.body`.
- The linearizer runs on the **distributed** tree (`distribute_products` is called once at `:6424` before row emission).
- `distribute_products` **rebuilds** `+`/`*`/integer-power nodes (new `id()`), so a claim keyed on the raw sum's `id()` is **silently lost** — the linearizer falls through to the loose separable McCormick path and the convex aux becomes a dead column.

**The fix is an existing pattern, not new machinery:** extend the `affine_square_protected_ids → protected_squares` set (`:5740, :5791, :6425`; consumed `term_classifier.py:355-361`) to cover the newly-claimed convex nodes, so `distribute_products` returns them intact and the `id()` short-circuit fires. No new dedup set is required; the classifier's now-internalized bilinear/monomial columns become harmless dead columns.

## Design decisions (settle before Phase 2)

- **D1 — claim unit.** Claim the **convex nonlinear part only**, keeping the affine remainder linear (a convex quadratic = PSD form + linear; lift `3x²+2y²+xy`, leave `-40x-30y` to the additive linearizer). Needs a small splitter built from `_flatten_additive_terms` (`:3259`) + `_linearize_affine_expr` (`:2125`) — there is no existing `_split_affine`. Start: claim the maximal convex `+`-subtree of nonlinear terms.
- **D2 — box-dependence.** `build_milp_relaxation` is invoked **per node** (with `bound_override`), so curvature is re-certified per node against the node box — box-dependence is handled for free (no stale root verdict). Cost: per-node `classify_expr` calls; mitigate by gating on cheap structural pre-checks before the certificate, and reuse the existing deadline guard.
- **D3 — cut placement.** v1 emits the existing fixed reference-point gradient cuts (center/edges), matching today's `CompositeMultivarRelaxation`. A later phase may add an LP-optimum separation round (a `_separate_convex` mirroring `_separate_univariate_square`) to match the bolt-on's per-node tightness. Decision: ship v1 with fixed cuts; LP-optimum separation is a measured follow-up.
- **D4 — coexist with the `solver.py` convex-objective bolt-on.** Keep it. It is applied with `max()` (`solver.py:6190`, "only tightens, always sound") and is **complementary**: the bolt-on is tighter for the *objective* (exact at the box-argmin via FISTA) but objective-only and gated to pure convex quadratics; the LP lift is more composable (tightens the same LP the integer/constraint structure lives in, and handles convex nodes inside **constraints and products**, which the bolt-on never touches). They are safe together.

## Phased plan with verification gates

**Soundness is the hard gate at every phase. Tightness is measured, never assumed.**

### Phase 0 — Spikes ✅ (done; see "What the spikes proved")
Outcome: GO, with the plan reshaped to "widen gate + protect identity."

### Phase 1 — Vertical slice (one convex-QP case, end-to-end)
- Widen `_should_claim_composite_multivar` to also claim a maximal convex `BinaryOp('+')` nonlinear subtree (≥2 vars), gated on `classify_expr ∈ {CONVEX, CONCAVE}` (or `_multivar_box_curvature`), affine remainder peeled (D1).
- Add the claimed node's `id()` to `affine_square_protected_ids` so it survives `distribute_products` (the obstacle fix).
- Restrict to a single target form first: a standalone convex quadratic objective term (e.g. a synthetic `min 3x²+2y²+xy-40x-30y, x,y∈[0,20]`).
- **Verify (routing):** the node resolves to one column in `_linearize_expr` (`id()` survives distribution); the convex aux is *not* a dead column; the bilinear/monomial columns for `x²/y²/xy` are present-but-unreferenced (dead, harmless).
- **Verify (no double-relax):** the objective row references only `col(d)` + linear terms, not the McCormick `xy` aux.
- **Verify (tightness):** root LP bound on the synthetic case beats the term-by-term McCormick bound and matches the convex-hull/supporting-hyperplane value.
- **Verify (soundness):** sample 1000 box points; the gradient cut underestimates `f` everywhere (debug assertion, kept as opt-in runtime check).

### Phase 2 — Generalize + coordinate
- Extend to: concave subtrees (symmetric, `upper_lines`); convex nodes **inside constraints** (`g(x)≤b` → linear row on `d` + gradient cuts); convex nodes **inside products** (`g(x)·z` via the existing `composite_var_map → _decompose_product` tspn path, `:1577-1587`).
- Mirror the gate widening into the **univariate** path `_should_claim_composite` (`:3495`) only where integer powers of non-bare bases aren't already convex-relaxed by the monomial tangent/secant path (avoid overlap — bare `x⁴` is already convex via the monomial path; do not re-claim it).
- **claimed_ids coordination:** ensure every newly-claimed node is added to `composite_var_map`/`_multivar_claimed_ids` (`:5408-5419`) so later builders (affine-square, affine-power, ratio) skip it; confirm no column-conflict.
- **id()-namespace keep-alive:** widening the `id()`-keyed claim space enlarges exposure to the ex7_2_3 freed-id hazard — ensure claimed expression objects are pinned for the build lifetime (extend the `_nested_div_keepalive` pattern, `:5838/:6194`).
- **Verify:** nvs21 (`x⁴` product structure), the convex-QP cohort (nvs17/19/23), tspn* unchanged, cvxnonsep_* — bounds tighten or hold; tspn* must not regress.

### Phase 3 — Soundness gauntlet + cohort measurement (the hard gate)
- **Guard chain unmodified-or-strengthened (never weakened):** the curvature gate (`:4120`), `_multivar_box_curvature` interval-Hessian PSD certificate (`:3942`), abstain-on-UNKNOWN→drop (`:4125-4132`), finite-box requirement (`:4115-4118`), finite column bounds (`:4141-4144`), per-point finiteness (`:4171-4176`). The cut direction must stay bound to the verdict (`:4189-4194`).
- **Lift-specific tests (must pass):** `test_cross_term_sqrt_soundness`, `test_nested_division_soundness`, `test_soc_cuts` (gradient-cut validity), `test_convexity_minlplib_suspect` (tspn* pinned NEGATIVE, cvxnonsep_* pinned CONVEX), `test_convexity_suspect_parity`.
- **Broad soundness battery (must stay 100% green):** `test_alphabb_bound_soundness`, `test_unbounded_relaxation_soundness`, `test_gear4_false_infeasible`, `test_hda_false_infeasible`, `test_bucket2_sound_bounds`, `test_bucket2_composition_soundness`, `test_nvs16_soundness`, `test_adversarial_recent_fixes`, `test_incumbent_injection_soundness`, `test_convex_objective_bound`, plus the convexity suites (`test_convexity_soundness/_certificate/_wide_box/_pathological/_interval_ad`), `test_relaxation_theorems`, `test_obbt_soundness`.
- **Known-bug regression guards (explicit):**
  1. **ex7_2_3 id()-cache** — `test_lifter_expression_dedups_by_structure_not_identity`, `test_nested_ratio_solve_is_sound`; ensure claimed-node keep-alive (no freed-id false hit).
  2. **himmel16 unbounded box** — `test_himmel16_no_false_certification`; the finite-box requirement must reject lifting on unbounded/half-open boxes.
  3. **cross-term conditioning (nvs05/22)** — `test_cross_term_guard_keeps_minlplib_sound`, `test_bucket2_composition_soundness`; preserve abstain-on-large-magnitude + cross-backend agreement (the lift raises aux magnitudes).
  4. **double-relax/column-conflict** — verify via the routing tests in Phase 1/2.
- **Adversarial gate test (new):** construct expressions designed to fool the detector (sign-straddling cubics, products dressed as sums) and assert the gate returns UNKNOWN → falls back to McCormick (no cut).
- **0-WRONG corpus gate:** `discopt_benchmarks` `incorrect_count == 0` and `relaxation_validity_rate == 1.0` (`benchmarks.toml:172-211`) on the phase1/full suites covering nvs17/19/23, tspn*, cvxnonsep_*, nvs05/22. Measure bound-tightness deltas and any frozen-bound improvements.
- **Rollback criteria (absolute):** any false certificate, any `incorrect_count > 0`, any soundness-test failure, any `relaxation_validity_rate < 1.0` → revert. Tightness regressions are investigable; correctness regressions are not.

### Phase 4 — Land
- Tests: routing/no-double-relax unit test; a generality test (multivariate convex sum past the old catalog); the adversarial gate test; a tightness test (convex lift beats term-by-term on a convex QP).
- Flag opt-in until one full nightly corpus run is clean, then default-on in a follow-up.
- Branch → CI green → PR, with the spike note + Phase-3 measurements in the PR body.

## HiGHS boundary (owned by the parallel pure-Rust effort, issue #356)

This work **inherits** HiGHS touchpoints but must not fix or depend on them:
- The convexity detector's constraint-aware sign refinement (`convexity/linear_context.py`) uses `scipy.linprog(method="highs")`.
- The per-node relaxation LP re-solve uses the HiGHS-first cross-checks (`mccormick_lp.py`).

Requirements: (a) add **no new** HiGHS dependency; (b) the cross-backend soundness tests (`test_bucket2_composition_soundness`) must pass under both backends so the change composes cleanly when #356 removes HiGHS; (c) flag each touchpoint in the PR so the two efforts don't collide.

## Risk register

| risk | severity | mitigation |
|---|---|---|
| Convexity false-positive → invalid cut → false cert | **critical** | detector is conservative (spike-verified); guard chain unmodified; adversarial gate test; runtime sampling assertion |
| Claim silently inert (id lost in distribution) | medium (tightness only) | the `protected_squares` identity fix + Phase-1 routing verification |
| Double-relax / column conflict | medium | id() short-circuit already prevents it; claimed_ids coordination; routing tests |
| ex7_2_3 freed-id false cache hit | high | keep-alive pinning of claimed nodes; regression test |
| Unbounded-box lift (himmel16) | high | finite-box requirement (existing guard); regression test |
| Aux-magnitude blow-up (nvs05/22 conditioning) | high | abstain-on-large-magnitude + cross-backend agreement (existing guards) |
| Per-node `classify_expr` cost | low/perf | cheap structural pre-checks before the certificate; deadline guard |

## Out of scope / follow-ups

- **Detector recall** for nested `(convex)²` compositions (the one spike-A miss) — a power-of-convex-nonneg-base rule. Complementary; raises #2's reach.
- **LP-optimum separation** (`_separate_convex`) to match the bolt-on's per-node tightness (D3) — measured follow-up.
- **The "no-drop systematic base"** (every operator yields *some* estimator, never an omitted constraint) — the second half of the hybrid; removes the st_e40 4.41-vs-30.41 constraint-drop failure. Separate, larger effort.
- HiGHS removal in this path — issue #356.
