# Audit report — `discopt.bilevel` (2026-07-13)

**Scope.** Verify the existing (Phases 0–3) bilevel module: does it run green, are
its soundness gates correct, and does the design doc match the code. Environment:
`pip install -e ".[dev]"` + pounce, Rust extension built; JAX + scipy present.

## Summary

The reformulation *math* is correct and the convexity gate is sound, but the
module's headline capability — a **certified global bilevel solve** — was
**untested and, on its default path, unsound**. A P0 false-optimal was found and
fixed; coverage and doc drift were closed.

| Workstream | Result |
|---|---|
| A. Baseline suite | ✅ 64 bilevel/mpec tests passed; ruff clean |
| B. `symbolic_diff` engine | ✅ correct vs `jax.grad`; **coverage gap closed** (structural nodes were untested) |
| C. Convexity gate | ✅ sound & conservative on every edge case (incl. max-follower sign-flip); **edge cases now tested** |
| D. KKT / strong-duality emission | ✅ signs, free multipliers, follower-bound fold all correct |
| E. End-to-end certified solve | ❌→✅ **P0 false-optimal found & fixed**; first end-to-end solve tests added |
| F. Doc/code drift | ✅ reconciled (`convexity_gate.py`, false "certified → CI" claims) |

## E — the P0 false optimal (headline)

No test anywhere called `model.solve()` on a `BilevelProblem`. Running one:

- **`kkt` + `gdp`** (the module default, used by `example_bilevel_toll`) on a linear
  Bard-style follower returned `status=optimal, gap_certified=True, x=5, y=10`,
  while the independent scipy follower oracle gives `y=0`. Complementarity was
  violated by `μ·(−g)=953` (must be 0) — **a certified follower-infeasible point.**
- **`kkt` + `sos1`** refused loudly (correct).
- **`strong_duality`** returned the correct optimum (`x=1, y=2, obj=−7`).

**Root cause.** KKT multipliers have no a-priori upper bound, so `kkt.build_kkt`
creates them at discopt's `ub≈9.999e19` unbounded sentinel. `_compute_big_m`
(`_jax/gdp_reformulate.py`, the GDP-1/#413 contract) treated that sentinel as a
*finite, valid* big-M (`np.isfinite(9.999e19)` is True) → `M≈1e20`. That `M` is
numerically vacuous: `M · integrality_tol = 1e20 · 1e-5 = 1e15`, so the disjunction
selector binary rounds to "integer" while the big-M term stays huge, and
complementarity is never enforced. #413 guarded "`M` too *small* cuts feasible
points → false infeasible"; it missed "`M` too *large* → disjunction vacuous →
false optimal." `_compute_big_m_lp` and SOS1 already rejected `≥1e15`; only the
`_compute_big_m` fallback re-admitted the sentinel.

**Fix (shared GDP layer + bilevel front-end).**
1. `_compute_big_m` now refuses a bound `|b| ≥ _BIGM_SENTINEL (1e15)` as it already
   did for `±inf` — a sentinel is not a usable big-M. Real finite bounds below the
   sentinel still use the true bound (the #413 fix for genuine bounds is preserved).
2. `BilevelProblem.formulate(method="kkt", mpec_method="gdp"|"sos1")` refuses loudly
   when follower multipliers are unbounded, pointing at `strong_duality`.
3. `example_bilevel_toll` switched to `strong_duality` (the sound path).
4. Blast-radius check: the change reverses one #413 sub-decision (sentinel → refuse,
   not `M=1e20`); the two tests encoding it were updated to the corrected behavior
   (real finite bound → use it; sentinel/inf → refuse). All other GDP / indicator /
   SOS / MPEC / GDPopt / pattern / gallery suites stay green.

## What is and isn't certified now

`strong_duality` solves linear, convex-QP, and convex-NLP bilevel to the true
optimistic optimum, but its solve is **not gap-certified** (its strong-duality
equality is nonconvex/bilinear) — an honest state. Integer/pessimistic lower levels
remain **refused loudly**, per CLAUDE.md §3.

## Follow-on capabilities (2026-07-14)

- **Certified path via user-supplied multiplier bounds.** Auto-deriving valid
  multiplier bounds was investigated and **falsified** (entry experiment: `run_obbt`
  leaves every KKT multiplier at the sentinel — boundedness comes from
  complementarity, not the linear KKT relaxation; valid bilevel big-Ms are NP-hard,
  Kleinert et al. 2021). Instead `BilevelProblem(..., multiplier_ub=M)` lets the user
  assert a valid finite bound on the follower duals (like any big-M), making
  `kkt`+`gdp` gap-certified (verified on Bard: `optimal, obj=−7, gap_certified=True`).
  `solve()` best-effort-warns if a multiplier sits at its bound. `strong_duality`
  stays the default for users who cannot bound their duals.
- **Convex-NLP followers.** The convexity gate now certifies convexity *in `y`* for a
  nonlinear body via a symbolic follower Hessian + interval-Gershgorin PSD test over
  the box (`problem.py::_y_convex_on_box`) — tight enough to accept the natural
  linearly-coupled form (`exp(y) − x·y`), refusing nonconvex/concave bodies. This
  removes the "convex-NLP deferred" limitation.

## Tests added / changed

- `python/tests/test_bilevel_phase3.py` (new): the missing end-to-end coverage —
  `kkt`+`gdp`/`sos1` refuse unbounded multipliers; `strong_duality` solves the
  linear bilevel to `x=1,y=2,obj=−7` with the returned `y` confirmed as the
  follower argmin (scipy oracle).
- `test_bilevel_symbolic_diff.py`: added differential-test cases for the structural
  nodes (`UnaryOp neg`, `SumExpression`, `SumOverExpression`, `IndexExpression`,
  `MatMulExpression`) — documented-supported but previously unexercised.
- `test_bilevel_phase2.py`: added convexity-gate edge cases (non-diagonal PSD QP
  accepted, indefinite sibling refused, max-follower sign-flip both directions,
  concave inequality body refused).
- `test_bilevel_phase1.py` / `phase2.py`: KKT-math checks now build via the new
  `BilevelProblem.build_kkt_system()` (sound KKT construction, independent of the
  now-refused big-M encoding).
- `test_gdp.py`, `test_gdp1_gdp2_bigm_soundness.py`: updated the #413 sentinel tests
  to the corrected behavior.

## Known limitations (out of scope, documented)

- Scalar-only lower variables (arrays passed component-wise).
- No `.crucible/` wiki concept page on bilevel programming.
- No gap-certified KKT path (needs valid multiplier bounds).
