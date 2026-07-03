# mpec / estimate / pooling / opf Review

**Date:** 2026-07-03
**Scope:** `python/discopt/mpec.py` (275), `estimate.py` (482), `pooling.py` (287),
`opf.py` (161).
**Method:** Delegated verification with numerical repros against the installed
package; the one soundness finding re-confirmed here. All 34 relevant tests pass.

Headline: **three of the four are correct and validated to known optima/closed
forms.** The exception is one verified edge-case soundness gap in `mpec.py`.

---

## 1. Summary

| # | Severity | Module | Finding |
|---|----------|--------|---------|
| MP-1 | **P1 soundness** | `mpec.py:168-175` | `tighten_complementarity_bounds` overwrites a variable's **positive lower bound** with 0 when fixing the complementary partner, silently **hiding infeasibility** [CONFIRMED] |
| MP-2 | P2 | `mpec.py:254` | `solve_mpec` Scholtes loop `except BaseException: break` swallows every solver error with no diagnostic [CONFIRMED by inspection] |
| ES-1 | P3 stat | `estimate.py:199-208` | `confidence_intervals` uses Student-t with `dof = n_obs − n_params` while σ is *known/fixed* (cov not rescaled by reduced χ²) — with known σ the consistent interval is normal(z)-based; the t/z mix is statistically inconsistent (slightly conservative) [SUSPECTED] |

Verified **correct** (with the evidence):

- **`pooling.py` — does NOT share the `examples.py` Haverly bug.** `haverly_hpp1()`
  solves to **objective 400.0** (the textbook Haverly-I optimum), node_count 3; the
  second known case gives 700. Proportion (`Σq=1`), pq/RLT cuts (`Σw=y`), bilinear
  defs `w=q·y`, mass balances, and quality blending (`Σλ·w ≤ spec·flow`) are all
  mathematically correct with finite McCormick bounds. (Contrast: the separate
  modeling-review E1 found `modeling/examples.py`'s Haverly wrong at 1390 — this
  independent `pooling.py` implementation is right.)
- **`opf.py` — correct.** Power-injection formulas match `Re/Im[V·conj(YV)]` over
  20 random points; `two_bus_example()` solves to 0.50306260, matching an
  independent scipy `fsolve` power-flow reference to 1.1e-16. Injection signs
  (`P = Pg − Pd`), P/Q box limits, voltage-magnitude limits (`Vmin² ≤ e²+f² ≤
  Vmax²`), and slack pinning are correct. (Scope, not a bug: no line/thermal flow
  limits, no shunts — matches its docstring.)
- **`estimate.py` — numerically correct.** Weighted LSQ `Σ((y_obs−y_model)/σ)²`,
  FIM `= JᵀWJ` with `W = diag(n_reps/σ²)`, `cov = FIM⁻¹` — the correct
  Jacobian/Fisher covariance for known-σ nonlinear LSQ. Reproduced against closed
  forms: `y=k·x` gives FIM 1400 (analytic 1400), cov 7.14e-4 (exact), k=2.0
  recovered; replication `Var ∝ 1/n` and the 2-parameter `XᵀX/σ²` case match.
- **`mpec.py` reformulations are sound.** On `min (x−1)²+(y−1)²` with `0≤x⊥y≥0`,
  Scholtes homotopy (`t→1e-8`, final `x·y≈1.7e-8`), SOS1 (`x·y≈9.9e-9`), and the
  GDP disjunction `(f==0)∨(g==0)` (`x·y≈1e-7`) all reach the true optimum 1.0 with
  correct nonnegativity handling; the disjunction is exactly equivalent to
  `f·g=0`. Scholtes is a documented *local* NLP path, correctly labeled.

---

## 2. MP-1 in detail

`tighten_complementarity_bounds` implements the sound inference "if one side of
`0 ≤ f ⊥ g ≥ 0` is bounded away from 0, the other must be 0." But when it fixes
the partner it sets **both** `lb` and `ub` to 0, discarding a pre-existing positive
lower bound instead of intersecting with it.

**Reproduced:** `a, b` both with `lb=0.5, ub=5`. The condition `0≤a⊥b≥0` with both
`≥0.5` is genuinely **infeasible** (`a·b ≥ 0.25 > 0`). The correct propagation is
`b.ub = 0` intersected with `b.lb = 0.5` → `lb > ub` → infeasible. Instead the
function returns `n_fixed=1` and sets `b.lb = b.ub = 0.0`, so a subsequent solve
reports the infeasible model as feasible/optimal — a **false certificate**. The
docstring calls this "sound and exact," which is false in this corner.

Severity is bounded by the trigger being unusual (a complementarity variable
carrying a strictly positive declared lower bound) and by the caller having to
invoke this helper — but per the repo's zero-slack correctness gate it is a real
soundness gap. **Fix:** set only `ub = 0` (intersect, never overwrite `lb`); when
the intersection gives `lb > ub`, surface infeasibility rather than silently
producing an empty-but-nonempty-looking box. Regression test: the repro must be
detected infeasible (or left for the solver to prove infeasible), not fixed to 0.

**MP-2:** replace `except BaseException: break` with a narrow catch that records
and re-raises (or returns a diagnostic status), per the "no swallowed exceptions"
rule — a first-iteration solver failure currently returns `None`/stale silently.

---

## 3. Plan (for Opus)

Small single PR `fix(mpec): MP-1..MP-2` — intersect (don't overwrite) bounds in
`tighten_complementarity_bounds` with infeasibility surfacing; narrow the
`solve_mpec` exception handling. Optional `fix(estimate): ES-1` — use z-quantiles
for known-σ intervals (or rescale cov by reduced χ² if σ is to be treated as
estimated) and document which regime is intended. `pooling.py` and `opf.py` need no
changes; `opf.py` could note line-flow limits as future scope.
