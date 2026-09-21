# #1397 — the absolute-tolerance audit

**Status:** the classification is complete; the fixes landed over three PRs. Four
scale-exposed unsound sites were fixed in the original sweep, a fifth in PR #1400,
and the remaining six in the follow-up PR recorded in **§6**, which also
**retracts two prescriptions made here** (§0's `max` shape and §4.5's
`* max(1, |new_lb|, |new_ub|)`). Read §6 before acting on §0 or §4.5.
**Date:** 2026-09-20. **Branch:** `audit/1397-absolute-tolerance-sweep`.

## 0. The defect class, and nothing else

One class, stated as narrowly as it can be:

> an absolute numeric constant compared against a quantity that carries the
> problem's scale.

A tolerance of `1e-6` is a statement about magnitude. Comparing it against a
quantity whose natural size is `‖H‖`, `|f(x)|`, a row norm, or an objective value
is dimensionally incoherent: the comparison means what its author intended at
unit scale and silently stops meaning anything once the problem is scaled up. The
canonical instance, and the one that opened the audit: `np.linalg.eigvalsh`
returns a minimum eigenvalue carrying error `O(u·‖H‖)`, so a fixed `1e-6` margin
on it is informative while `‖H‖ ≈ 1` and pure noise once `‖H‖ ≳ 1e10`.

**This was not a tolerance-tuning exercise.** No constant's value changed
anywhere in this audit. The question asked of each site was only whether the
comparison is dimensionally coherent; where it was not, the *yardstick* changed
and the constant stayed exactly where it was, as a floor. That is the standing
rule from #1384/#1392 and it held for all four fixes here.

### The fix pattern

From #1384/#1392, and applied unchanged at all four sites: the yardstick is the
magnitude of the terms the quantity is a **difference of** — Ipopt's `s_d` — and
**not** their sum, floored at 1:

```python
margin = max(ABSOLUTE_FLOOR, K * eps * scale_of_the_compared_quantity)
```

Three properties make this the right shape, and each one was load-bearing:

1. **`max` is never looser than the bare absolute gate**, so the change can only
   make a soundness gate stricter.
2. **Floored at 1, an O(1) problem is byte-identical.** Every existing test that
   pins a number on a unit-scale problem keeps passing unchanged, which is how
   these fixes were kept bound-neutral where they had to be.
3. **The floor still does real work**: it handles the near-zero case, where the
   scale-carrying term vanishes and a gate with no floor would decide on dust.

> **Correction (§6, CLAUDE.md §11).** The `max` in that shape is wrong for the
> round-off subclass and was retracted after this section was written. `max` is
> right when the absolute constant and the scaled term answer the *same* question
> at different scales. It is wrong when they answer *different* questions — a
> feasibility tolerance is a statement about the model, a round-off bound is a
> statement about the arithmetic, and a residual can be inside neither while
> exceeding each alone. Those two **add** (#1392's lesson, restated):
> `margin = ABSOLUTE_FLOOR + K*u*sum|terms|`. The additive form is also never
> looser than the bare gate, so property 1 above still holds, and at O(1) scale the
> added slack is O(1e-15), so property 2 still holds.

## 1. Method

Enumerated every module-level numeric constant in `python/discopt/` whose name
carries a tolerance word, then every site where one is compared against a
computed quantity. Measured at `56a2744`, the commit #1397 cites, with the
issue's own name pattern:

```bash
RX='^_?[A-Z][A-Z0-9_]*(TOL|TOLERANCE|FLOOR|EPS|EPSILON|THRESHOLD|MARGIN)[A-Z0-9_]*[[:space:]]*(:[^=]*)?='
git grep -hoE "$RX" 56a2744 -- python/discopt | wc -l                    # 117
git grep -hoE "$RX" 56a2744 -- python/discopt |
    sed -E 's/[[:space:]]*(:.*)?=.*$//' | sort -u | wc -l                # 105
git grep -hoE '^[[:space:]]*(pub[[:space:]]+)?(const|static)[[:space:]]+_?[A-Z][A-Z0-9_]*(TOL|TOLERANCE|FLOOR|EPS|EPSILON|THRESHOLD|MARGIN)[A-Z0-9_]*[[:space:]]*:' \
    56a2744 -- crates | wc -l                                            # 21
```

So **117 constant assignments / 105 distinct names** in `python/discopt/` plus
**21** in `crates/` = **138**, with **185** reference sites for those names
inside `python/discopt/`. Narrowing to *comparison* sites (a constant merely
passed as an argument, logged, or used as a value rather than a threshold is not
a comparison) is what produces the candidate list this audit classifies.

**Correction to #1397's headline figure, and to an earlier draft of this
document (CLAUDE.md §11).** The issue states "134 module-level constants in
`python/discopt/`, 20 more in `crates/`, ~154 sites total"; an earlier draft of
this file stated "128 constants, 346 use sites". Neither reproduces. The
commands above are the figures this document uses, and the closest variants tried
were 120 (any indentation, so module- *and* class-level) and 155 (all of
`python/`, i.e. including the test suite) — the latter being a numerical
coincidence with the issue's 154, not the same measurement. The gap does not
change any conclusion below, since every fixed and shortlisted site was reached
by reading code rather than by counting it, but a scope number in an issue reads
as a measurement and this one was not reproducible.

Each candidate was then classified into exactly one of four buckets by
**tracing the verdict to its consumer**.

The two rules that decided most of the calls:

- **Classify by role, not by name.** `cutting_planes.py` has two constants three
  lines apart, same prefix, opposite verdicts: `ALPHABB_EPS` gates only *whether
  a cut is attempted*, and skipping a cut is unconditionally safe → benign.
  `ALPHABB_SAFETY`, three lines later, is what makes the emitted cut *valid* →
  unsound. The name tells you nothing.
- **The relaxation layer is asymmetric.** Too **loose** is sound; too **tight**
  is not. A tolerance that can only cause the code to decline to relax, widen an
  interval, or skip a tightening is benign almost by construction. One that
  nudges a bound inward, or decides an interval is empty / a bound redundant / a
  row negligible, can be unsound.

### The buckets

| Bucket | Meaning |
| --- | --- |
| **Sound** | The comparison is dimensionally coherent — the quantity is genuinely absolute or already dimensionless (a ratio, fraction, probability, relative gap, count, fractional part), or the constant is a *value* (a step size, a perturbation, a divisor) and not a threshold. |
| **Scale-exposed, benign** | The quantity carries scale, but a wrong verdict at large scale costs only performance, tightness, a skipped opportunity, or a log line. Nothing unsound follows. |
| **Scale-exposed, unsound** | The quantity carries scale **and** a wrong verdict breaks a correctness claim: a dual bound above the true optimum, a cut excluding feasible points, a tightening removing the optimum from the box, a false feasibility/optimality/integrality verdict, a convexity claim licensing an invalid relaxation. |
| **Needs a measurement** | Undecidable by reading. The experiment that would settle it is named. |

## 2. The five unsound sites, fixed

All five are the same defect wearing five hats: a margin that had to cover an
`O(u·‖H‖)` error was an absolute constant, or absent.

| # | Site | Constant | What broke | Commit |
| --- | --- | --- | --- | --- |
| 1 | `solver.py:2709` `_hessian_is_psd_with_margin` | `_CONVEX_OBJ_PSD_TOL` | A false PSD verdict licenses the convex-objective supporting hyperplane on a nonconvex objective, which does not underestimate it → false dual bound. | `90fe46dc` |
| 2 | `_relax/convexity/certificate.py` | `_PSD_TOL` | Same verdict, different consumer: a false convexity certificate licenses the whole convex relaxation path. | `d2430cec` |
| 3 | `_relax/cutting_planes.py` αBB cut | `ALPHABB_SAFETY` | α below `−λ_min/2` leaves `q_under` nonconvex, so its tangent can exclude points satisfying `q(x) ≤ 0` — a cut that is not a relaxation. | `277c6b3c` |
| 4 | `_alphabb_rigorous.py` `rigorous_alpha` | *(none — no margin at all)* | Feeds the per-node αBB **dual bound**. Non-outward-rounded Gershgorin raised the bound above its true value, so the αBB body is nonconvex and its box minimum can exceed the true one → false dual bound. | `277c6b3c` |
| 5 | `_relax/convexity/g_convexity.py` `certify_g_convex` | `_PSD_TOL` | The module's own comment claimed this matched "the ordinary convexity certificate" — it stopped matching the moment site 2 became scale-aware *in this same audit*. A false `λ_min ≥ 0` verdict certifies a non-G-convex form. **Flag-gated, not default path** (see the REACH note below). | `8e7ab2f8` |

### The measurements

Each figure below is from a probe that **calls the shipped function** and grades
its output against an oracle independent of the code under test, with a load gate
asserting both `module.__file__` and a marker string unique to the version (and
the marker *absent* on the baseline run).

| Site | Oracle | Before | After |
| --- | --- | --- | --- |
| 1 | `H = V diag(ev) Vᵀ` (spectrum by construction, so `eigvalsh` never grades itself) | **194 of 2190** indefinite Hessians declared PSD — each a licence to emit an invalid dual bound | **0**, same 2190 assertions |
| 2 | same, plus an end-to-end `certify_convex` arm | 10 convex certificates **lost**, exit 1; and see the materiality table below | 0 lost, 0 unsound, exit 0 |
| 3 | same, α recovered from the cut the generator **returns** | **84 of 504 cuts invalid** — 0/72 at `‖H‖_F` = 1e0/1e3/1e6/1e9, 12/72 at 1e10, 35/72 at 1e12, 37/72 at 1e14 | 0/504, worst shortfall 0.0 |
| 4 | the same Gershgorin formula in **exact rational arithmetic** over the same float entries | **513 of 1120 rows** with α provably below `−λ_min/2`, worst shortfall 5.76e-4, at **every** scale 1e0…1e12 | 0/1120, worst shortfall 0.0 |
| 5 | a **diagonal** interval matrix, so the Gershgorin row bound is exact and `λ_min` is the least entry with no enclosure slop; graded figure is admitted *relative* nonconvexity `δ/‖aug‖_F` | **21 unsound admissions** (worst relative nonconvexity **5.774e-04** at `‖aug‖`~1e-8, i.e. 2.6e+12·u) and **9 genuinely-PSD matrices refused** at `‖aug‖` ∈ {1e6, 1e9, 1e12}; exit 1 | **0** unsound, **0** lost, 176 comparisons per arm, exit 0 |

#### Site 2's raw admission count goes UP, and why that is the wrong statistic

Reported in full so nobody later finds the number and concludes a regression
shipped. `verify_qform_psd.py`, arm A (indefinite `Q`, exactly one negative
eigenvalue, certified PSD), 2000 executed assertions on each tree:

| | admissions | worst relative nonconvexity admitted | **material** (> `100u`) | valid PSD refused (arm B) |
| --- | --- | --- | --- | --- |
| before | 240 | `1.967e-12` = **8858·u** | **25** | 149 |
| after | **452** | `3.490e-15` = **15.7·u** | **0** | **0** |

The two gates admit *different sets*, so the raw count is not comparable between
them. What is comparable is the **relative** nonconvexity `|λ_min|/‖Q‖` of what
gets admitted, because that is the quantity bounding the relative bound error: the
error from treating a form with `λ_min = −δ` as convex is `≤ δ·diam²/2`, and the
form's own magnitude over the same box is `~‖Q‖·diam²/2`, so `δ/‖Q‖` bounds the
relative error **independently of the box**.

By that measure the fix is strictly better on every axis:

- admissions that are **materially** nonconvex — the only ones that can produce a
  non-trivial relative error — went **25 → 0**;
- the worst relative nonconvexity admitted fell from `8858·u` to `15.7·u`
  (median `4.0e-19` → `5.0e-19`), i.e. from a thousandfold above the arithmetic's
  own resolution to within it, consistent with the predicted uniform ceiling of
  `(32 + 5.42)·u`;
- the extra 212 admissions are all below that resolution, where the verdict is not
  physically decidable in binary64 by any gate;
- and **149 → 0** genuinely PSD matrices are no longer refused — the absolute
  `1e-10` had become *stricter than the eigensolver's own roundoff* at large `‖Q‖`
  and was silently discarding valid certificates.

Arm C (parity on well-scaled `Q`) reports **0** verdict differences on both trees,
which is the check that this was a yardstick change and not a tolerance change —
#1397's stated non-goal.

The direction of the absolute gate's failure is worth stating plainly, because it
is the opposite of the intuition: a fixed absolute tolerance on an eigenvalue is
relatively **loosest at small `‖Q‖`** (at `‖Q‖ = 1e-8`, admitting `δ = 1e-10` is a
relative nonconvexity of `1e-2`) and relatively **over-strict at large `‖Q‖`**,
where it falls below the roundoff it was meant to absorb. Both failures are fixed
by the same substitution, which is why the count moves in one direction while
soundness moves in the other.

Site 3's onset matched the prediction almost exactly: an absolute `1e-6` covers
`5.42·u·‖H‖/2` only while `‖H‖ ≲ 1.7e9`, and the first invalid cuts appear at
1e10. Site 4 fails at *every* scale because it carried no margin whatsoever —
its error is `O(u·‖A‖)` with nothing to absorb it.

`fractions.Fraction` is the oracle for site 4 because every binary64 is a
rational: evaluating the formula in `Fraction` over the *same* float entries has
no error of its own, so a violation is a proof rather than an estimate.

### Site 5's reach — a retracted claim (CLAUDE.md §11)

Commit `8e7ab2f8`'s own message asserted that `certify_g_convex` "is on the
default path". **That was wrong, and it is retracted here.** A peer review
contradicted it; the peer was right. Verified caller by caller:

* `_relax/mccormick_lp.py:2828` sits inside `_separate_g_convex`, called at
  `:1792` only under `out_cuts is None and self._g_convex_enabled()` — that is
  `DISCOPT_G_CONVEX_CUTS`, **default OFF** (`:2733`);
* `_relax/convexity/g_convex_inject.py` is gated on the same flag (`:10`, `:93`);
* `g_convex_cut.py:150`/`:248` and `g_products_ratios.py:94` have **no callers at
  all** outside the `convexity/__init__.py` re-export.

How the error was made, because the mechanism generalizes: I checked that no
caller passes `tol`, and mistook that for checking that a caller *runs*. Those
are different questions and only the second one establishes reach.

So site 5's bucket is **scale-exposed, unsound, but flag-gated** — it cannot
emit a certificate on a default solve today. It was still worth fixing, and is
still filed under §4.5 rather than §4.4, because the flag's whole purpose is to
be graduated: a gate that is sound only because nobody switches it on is not
sound, it is unexercised. Commit `5d8bcc03` records the verified reach in the
module comment so the next reader does not have to re-derive it.

### Site 4 was also a duplicate

`rigorous_alpha` had **reimplemented** interval Gershgorin with a plain
round-to-nearest `np.sum` and no outward rounding, while the correct,
outward-rounded computation already existed one module away in
`gershgorin_lambda_min`. Per CLAUDE.md §3 the fix was to delete the duplicate
rather than patch it: `convexity/eigenvalue.py` grew
`gershgorin_row_lower_bounds`, the per-row bounds extracted from the already
rigorous routine, and `gershgorin_lambda_min` became `rows.min()`. A bound-neutral
guard pins that equality (60 comparisons). The per-row form is also what αBB
wants — `α_i` scaled to each variable's own row is tighter than applying the
global minimum everywhere — so the shared helper is strictly better than either
copy.

## 3. Why this class cannot be caught end-to-end — a measured negative result

The obvious probe is to solve a model, scale it up, and check the certificate.
**It does not work, and the arithmetic says it cannot.** Measured: the
end-to-end sweep in `test_1397_certificate_scale_invariance.py` (5 nonconvex
models with closed-form optima × scales 1e0…1e12) passes **26/26 on the pre-fix
tree**, with the load gate asserting the fixes absent.

The reason is that this class's footprint is *relatively* invisible:

- a false-PSD verdict licenses a bound whose error is at most `|λ_min|·diam²/2`
  with `|λ_min| ≲ 5·u·‖H‖`, while the objective's own magnitude over the same box
  is `~‖H‖·diam²/2` — a **relative** error of `~5u ≈ 1.1e-15`;
- `rigorous_alpha`'s shortfall is `O(u·‖A‖)` for the same reason, hence `O(u)`
  relative at every scale.

So a defect in this class yields a certificate that is **provably** invalid but
invalid by a relative roundoff, which no relative end-to-end check can resolve —
and an *absolute* end-to-end check would itself be the audited defect.

This is not a reason to tolerate it (§1: a certificate invalid by 1e-15 relative
is still not a certificate; the error compounds over thousands of nodes and flips
a prune decision at a tie). It is the reason the real probes are **per-site
sweeps over the internal quantity**, each of which fails before its fix and
passes after:

| Probe | Sites | Fails before → passes after |
| --- | --- | --- |
| `test_1397_scale_aware_psd_gate.py` | 1 | **not directly comparable** — it imports the helper the fix introduced (`_hessian_is_psd_with_margin`), so on `main` it errors at *collection*, not on an assertion. The behavioural before/after for site 1 is the probe's 194 → 0 above, run on both trees with the marker asserted present/absent. |
| `test_1397_scale_aware_convexity_slack.py` | 2 | yes — 77 tests, including a 400-check parity arm against the literal removed gate |
| `test_1397_alphabb_alpha_dominates_nonconvexity.py` | 3, 4 | yes — 17 failed / 15 passed on `main`, 36 passed after |
| `test_1397_certificate_scale_invariance.py` | *class-level guard* | **no** — passes on both trees, by construction (above) |

The distinction in the first row is worth keeping rather than tidying away: a
regression test that can only fail on the baseline by failing to *import* pins the
fix's API, not its behaviour. The behavioural evidence for site 1 is the probe,
and the probe is the artifact to re-run if that site is ever touched again.

The last row is in CI anyway, documented as what it is: a forward-looking guard
on the coarser failure the per-site sweeps cannot see — a scale bug whose
footprint *is* visible in the answer, anywhere on the default solve path,
including code written after this audit.

## 4. Classification

The sweep found **105 distinct constants, 116 declarations, 447 non-declaration
reference lines**. Of the 116 declarations, **41 are never compared against
anything** — they are divisors, step sizes, format widths and re-exports, so
there is no comparison to be dimensionally wrong about. The remaining **75
comparison-bearing declarations, carrying 300 comparison lines**, are classified
below. Line numbers are as of `640cfdc1`.

Every row's justification answers the same question: *does the constant stand
opposite a quantity that carries the problem's scale?* Where it does not, the
comparison is coherent and no change is warranted.

### 4.1 The distinction that decides most rows

Two things look alike in a grep and are not the same:

* **A declared absolute tolerance** — `ABS_TOL = 1e-6`, `INT_TOL = 1e-5`,
  `GAP_ABS_TOL`, `_DEFAULT_ABS_GAP_TOL`, `FEAS_TOL`. These are the solver's
  *specification* (CLAUDE.md "Key Constraints": `abs=1e-6, rel=1e-4,
  integrality=1e-5`). An absolute floor is what the caller asked for, and every
  one of them is paired with a relative term (`ABS_TOL + BOUND_REL_TOL*|x|`,
  `FEAS_TOL + FEAS_RTOL*|row|`) exactly as #1397 check (5) prescribes. **Sound.**
  Changing one would be the tolerance-tuning the issue rules out.
* **An absolute constant standing in for a relative judgment** — is this
  eigenvalue negative, is this coefficient zero, is this box empty, is this
  roundoff. The quantity has units; the constant does not. **This is the defect
  class,** and it is the whole of §4.4 and §4.5.

Integrality tolerances are Sound for a third reason: `|x - round(x)|` is measured
in *lattice units*, which are dimensionless by construction whatever `x`'s
magnitude. So `INT_TOL`, `_INTEGER_BOX_TOL`, `_INTEGRALITY_TOL` and
`_relax/lp_spatial_bb.py`'s `_INT_TOL` all pass. (`implied_integer.py`'s `_INT_TOL`
is a *different* constant on a different quantity; see §4.5.)

### 4.2 Sound — the yardstick is already there (21)

Each of these already multiplies by a magnitude floored at 1, which is #1397's
prescription. Recorded so the next audit does not re-derive them.

| site | constant | the yardstick it already carries |
|---|---|---|
| `solver.py:2199` | `_LAZY_RESEP_GLB_EPS` | `* max(1.0, abs(glb))` |
| `solver.py:3260` | `_REFINE_DEGRADE_EPS` | `* (1.0 + abs(obj_val))` |
| `solver.py:4806` | `_DECLARED_DUAL_STAT_TOL` | `* (1.0 + max abs(grad))` |
| `solver.py:17905` | `_KKT_STATIONARITY_REL_TOL` | compares an already-relative residual |
| `solver.py:22564` | `_FEAS_SUM_FLOOR` | `* abs(x)`; is itself `2·u` |
| `validation/feasibility.py:137` | `BOUND_REL_TOL` | is the relative term |
| `validation/feasibility.py:147` | `CANCELLATION_RTOL` | `* abs(terms)` |
| `solvers/lp_milp_highs.py:53` | `FEAS_RTOL` | is the relative term |
| `solvers/lp_pounce.py:195` | `_RAY_COST_TOL` | `* max(1.0, max abs(cost))` |
| `solvers/lp_pounce.py:206` | `_RAY_DIRT_TOL` | `* d_scale` |
| `solvers/milp_simplex.py:41` | `_NS_MARGIN_REL` | `* (1.0 + abs(by) + ...)` |
| `solvers/oa.py:1571` | `_NLP_DEFAULT_TOL` | `* max(f0, _NLP_TOL_SCALE_FLOOR)` |
| `solvers/surrogate.py:372` | `_RBF_RESIDUAL_TOL` | `* (1.0 + norm)` |
| `_relax/convexity/linear_context.py:49` | `_LP_ENCLOSURE_MARGIN` | `* (1.0 + abs(f))` |
| `_relax/convexity/signomial_global.py:790` | `_OBBT_MARGIN` | `* max(1.0, abs(incumbent))` |
| `_relax/integer_ratio.py:69` | `_ACH_TOL` | `* max(1.0, abs(q_star))` |
| `_relax/mccormick_lp.py:240` | `_EMPTY_BOX_TOL` | `* maximum(1.0, abs(bounds))` |
| `_relax/mccormick_lp.py:422` | `_ST_BINDING_TOL` | `* maximum(1.0, abs(b))` |
| `modeling/argmin.py:121` | `KKT_RESIDUAL_TOL` | `* s` |
| `modeling/argmin.py:128` | `CURVATURE_TOL` | `* scale` |
| `mpec_report.py:486` | `_SOURCE_TOL` | compares `complementarity.scaled_value` |

### 4.3 Sound — the compared quantity carries no problem scale (22)

| site | constant | why the comparison is coherent |
|---|---|---|
| `solver.py:521` | `_DEADLINE_NODE_FLOOR_S` | seconds of wall clock; a second is a second |
| `solver.py:6487` | `_CONVEX_ROUTE_FALLBACK_FLOOR_S` | seconds |
| `solver.py:6529` | `_CONVEX_ROUTE_DECISION_POINT_FLOOR_S` | seconds |
| `solvers/oa.py:1567` | `_NLP_WALL_FLOOR_S` | seconds |
| `_relax/mccormick_lp.py:232` | `_SOLVE_DEADLINE_FLOOR_S` | seconds |
| `_relax/primal_heuristics.py:145` | `_DEADLINE_NLP_FLOOR_S` | seconds |
| `solvers/surrogate.py:377` | `_DIMENSION_WARN_THRESHOLD` | a variable count |
| `mo/scalarization.py:37` | `_GRID_WARN_THRESHOLD` | a subproblem count |
| `decomposition/advisor/selection.py:34` | `_TIE_EPSILON` | a fraction of a normalized score |
| `stochastic/scenario.py:21` | `_PROB_TOL` | probabilities sum to 1 |
| `validation/feasibility.py:134` | `INT_TOL` | lattice units (§4.1) |
| `mpec_report.py:487` | `_INTEGRALITY_TOL` | lattice units |
| `_relax/nonlinear_bound_tightening.py:60` | `_INTEGER_BOX_TOL` | lattice units |
| `_relax/lp_spatial_bb.py:72` | `_INT_TOL` | lattice units |
| `_relax/convexity/signomial.py:69` | `_EXP_TOL` | a signomial *exponent*, dimensionless |
| `solver.py:5162` | `_BOUND_WARN_THRESHOLD` | effective-infinity sentinel (below) |
| `solvers/lp_pounce.py:75` | `_LEGACY_BOUND_THRESHOLD` | effective-infinity sentinel |
| `ml/predictor.py:21` | `_BOUND_INF_THRESHOLD` | effective-infinity sentinel |
| `constants.py:20` | `SENTINEL_THRESHOLD` | effective-infinity sentinel |
| `debug/context.py:24` | `_SENTINEL_THRESHOLD` | effective-infinity sentinel (89 refs) |
| `_relax/milp_relaxation.py:147` | `_SUBNORMAL_FLOOR` | `float64.tiny`, an arithmetic constant |
| `_relax/ellipsoidal_arith.py:81` | `_PSD_FLOOR = 0.0` | zero has no scale |

The five **effective-infinity sentinels** are absolute *by definition* — they
define what "unbounded" means in binary64, so scaling them would be incoherent.
They are a real hazard class, but a different one: it is the `INF = 1e20`
discipline already recorded in CLAUDE.md, and it belongs to a separate sweep.

Also Sound, and worth recording because it looks exactly like the defect:
`constants.py:60`'s `ALPHABB_EPS` at `_relax/cutting_planes.py:592`. `min_eig >=
-ALPHABB_EPS` gates only *whether an alphaBB cut is attempted*; skipping a cut is
unconditionally safe, so an absolute constant is appropriate. The comment there
now says so. The **margin that makes the emitted cut valid** is a different
constant in the same function, and that one is §2's site 3.

### 4.4 Scale-exposed, benign — the failure is a refusal (6)

Per the issue: *fix, priority by reach*. None of these can emit a wrong
certificate; each can only decline to do something helpful, or over-warn.

| site | constant | the refusal |
|---|---|---|
| `validation/examiner.py:37` | `SHOW_TOL` | display filter on a violator list; no math |
| `validation/examiner.py:38` | `ACTIVE_TOL` | already `max(ACTIVE_TOL, abs(lam))` at the argmin call sites; a mis-set active set loses a dual, not a bound |
| `solvers/_root_cuts.py:69` | `OA_TOL` | skips an OA cut |
| `solvers/_root_cuts.py:71` | `CUT_VIOL_TOL` | skips a cut judged insufficiently violated → weaker bound, never a wrong one |
| `_relax/uniform_relax.py:600` | `_OBJ_BOX_FLOOR_GARBAGE_CAP` | rejects a legitimately huge objective box as garbage → conservative |
| `_relax/ellipsoidal_arith.py:78` | `_ROUND_FLOOR` | added to an enclosure *radius*; an absolute term only ever widens, and widening an enclosure is always sound |

### 4.5 Scale-exposed, unsound — on a certifying path (7 + 5 fixed)

The five fixed by this PR are §2 and §2's site 5. The seven below are the
remainder of the class. Every one sits on a path that can emit a certificate, so
by the issue's own ordering they are "fix first"; they are **not** fixed here,
for the reason in §4.6.

| site | constant | the unsound direction | status |
|---|---|---|---|
| `validation/feasibility.py:140` | `FEASIBLE_DISTANCE_TOL` | **measured, confirmed.** A relative roundoff allowance `16·u·(\|ub\|+\|x\|)` is added into `room`, then divided by this absolute constant and multiplied by `\|J_ij\|`. A column pinned *on* the bound that blocks its improving direction — true room exactly 0 — is credited with phantom room ∝ \|bound\|. Measured on a hand-oracle row: the returned improving-gradient norm goes 1.0 → 1000.0 (its plain sup-norm ceiling) as the bound sweeps 1e0 → 1e11, inflating the acceptance cap from 1e-4 to 1e-1. At that point #1284's tightening is a silent no-op — and #1284 exists because the untightened cap certified a point 0.87 away in `y` against a true optimum of −6.699. Default path, two call sites (`solver.py:3107`, `_relax/primal_heuristics.py:639`), and it gates whether a point becomes the incumbent. | **fixed, PR #1400** |
| `_relax/nonlinear_bound_tightening.py:58` | `_EMPTY_INTERVAL_FEAS_TOL` | 17 comparison lines. `new_lb - new_ub <= tol` snaps a sub-tolerance crossover to a midpoint instead of declaring the node infeasible. A *larger* tolerance is therefore the safe direction; the unsound direction is this absolute 1e-6 being too small relative to scale — on a box with bounds ~1e10, ordinary rounding crossover exceeds it, falls through, and emits `status="infeasible"`. ~~Needs `* max(1, \|new_lb\|, \|new_ub\|)`.~~ **That prescription was falsified before the fix landed — see §6.1.** The scale lives in the row's *inputs*, not in the crossover it produces. | **fixed, §6** |
| `_relax/node_reduce.py:48` | `_RC_TOL` | reduced-cost fixing. `dj` carries objective-over-variable units; a genuine zero reduced cost reading above an absolute 1e-7 at large objective scale makes `cand = lb + gap/dj` spuriously small and **fixes the optimum out of the box**. Same shape at `solver.py:24017` (`_RCF_RC_TOL`; the line cited when this table was written has moved). | **fixed, §6** |
| `_relax/perspective.py:74` | `_ZERO_TOL` | The unsound direction is **not** the one guessed here (see §6.3): the reach is the *separability* gate, which computed the off-diagonal mass as `\|Q[j,:]\|.sum() - \|Q[j,j]\|` — a difference of two large numbers — and `find_candidates`' own docstring calls that gate a soundness condition. At `Q[j,j] = 1e12` a genuine cross-term below one ulp (1.2e-4) is absorbed and the subtraction returns exactly `0.0`, certifying a coupled row separable. Plus two accumulated-coefficient sign tests on the semicontinuity row. | **fixed, §6** |
| `_relax/convexity/posynomial.py:54` | `_POS_TOL` | **Reclassified, not scale-exposed (§6.4).** The guessed reach (`all(arr > _POS_TOL)`) is in `log_lattice.py`, not this file, and every one of its comparands is *declared model data* — a variable's `lb` or a constant leaf — never an accumulated quantity. In `posynomial.py` itself the comparands are a monomial exponent (a model literal, O(1) by construction) and a monomial coefficient, which is a **product** chain: products preserve sign exactly and carry relative error only, so a truly non-positive coefficient cannot read positive. Both remaining uses refuse on failure. | **not a defect** |
| `_relax/factorable_reform.py:69` | `_ZERO_MARGIN` | `lo > _ZERO_MARGIN` decides a denominator is strictly positive before clearing it. Confirmed unsound and **worse than the row above**: clearing a sign-indefinite denominator does not weaken the bound, it *flips the inequality* wherever the denominator is negative, so the rewritten model has a different feasible set than the one the user wrote. Measured: 5 unsound clears at `M = 1e16…1e18` (§6.2). | **fixed, §6** |
| `solvers/lp_simplex.py:46` | `_BOUND_SNAP_TOL = 1e-3` | **Reclassified as a deliberate guard (§6.5).** Its own comment states the intent: 1e-3 is "far below any meaningful constraint scale, so a genuine solver defect (a large off-box value) is left intact to surface in tests." Making it scale-relative would *widen* the snap on large models and mask exactly the defects it exists to expose. Weakening a fail-loud guard to make a dimensional argument tidy is what CLAUDE.md §1/§3 forbid. The `lp_pounce.py:403` use is already relative. | **kept, documented** |

Two further sites from the same sweep are **flag-gated**, so they are not on a
default certifying path and are recorded here rather than fixed:
`decomposition/benders/solver.py:67` / `gbd.py:125` (`_ETA_FLOOR = -1e12`, an
objective floor compared against an objective) and
`decomposition/benders/gbd.py:141` (`_STATIONARY_TOL`).

### 4.6 Why the seven are not fixed in this PR

They are real and they are §1 defects. They are also in five subsystems this PR
does not otherwise touch, and three of them (`FEASIBLE_DISTANCE_TOL`,
`_EMPTY_INTERVAL_FEAS_TOL`, `_RC_TOL`) have named regression instances whose
behaviour a fix must preserve exactly — `portfol_roundlot` and `clay0303hfsg` for
the first, `hda` at ~68930 for the second. Landing them alongside five convexity
fixes would make a bound-changing PR that no single differential panel can
attribute.

**Update (PR #1400): `FEASIBLE_DISTANCE_TOL` is fixed**, in its own PR for exactly
that attribution reason, so item 2 stands at 6 of 12 and the remaining list is six,
not seven. The cap on the round-off allowance leaves all four #1284-pinned
allowances byte-identical and `portfol_roundlot`'s tie still breaking with 24x
margin; the measurement is in that PR.

So: **#1397 cannot be closed by this PR.** Item 1 (the table) is complete here;
item 3 (the CI probe) is complete here; item 2 (every unsound site fixed) has
5 of 12 done in this PR and a 6th in PR #1400. The remaining six are listed above
in the issue's own priority order, `_EMPTY_INTERVAL_FEAS_TOL` now first.

**Update (§6): the remaining six are resolved** — four fixed, two reclassified
with the reasoning recorded — so item 2 is complete and #1397 can be closed.

## 5. What this audit does not cover

- **Rust.** `crates/discopt-core/` was not swept. The `INF = 1e20` sentinel
  discipline documented in CLAUDE.md is the same class of hazard and has its own
  history of producing false certified bounds; a Rust-side sweep is separate work.
- **Constants that are values, not thresholds.** Step sizes, perturbations,
  divisors and numeric knobs whose `0` is a value rather than an off-switch were
  enumerated but not classified — there is no comparison to be dimensionally
  wrong about.
- **Whether any fixed site was ever hit in practice.** The fixes are justified by
  provable invalidity of the margin, not by a corpus instance that failed. No
  claim is made that a released solve returned a wrong answer because of these.
- **Bare numeric literals compared inline** — added 2026-09-21, see §7.1. §1's sweep
  matches constants by **name**, so a threshold written as a literal at its comparison
  site was never enumerated. This exclusion was discovered, not designed: it is how
  `edge_concave.py`'s `1e-12` / `1e-9` (§7, a §4.5 site) escaped all 103 rows of §4
  while this document read as complete over `_relax/`. Anything below about coverage of
  the Python relaxation layer means *named* constants in it.

## 6. Update — the remaining six, and three retractions

**Date:** 2026-09-20. **Branch:** `fix/1397-remaining-scale-yardsticks`.

This section closes item 2 of the issue. Of the six sites left open by §4.5, four
were confirmed unsound and fixed, and two were reclassified with the reasoning
recorded rather than "fixed" to make a table tidy. It also **retracts two
prescriptions made earlier in this document** (CLAUDE.md §11), because both were
falsified by measurement before any code landed, and in §6.13 **retracts a result I
published from this PR's own panel** — a `tls2` certification gain that the spread
test showed to be a wall-clock artifact.

### 6.1 `_EMPTY_INTERVAL_FEAS_TOL` — and the retraction of `max(1, |result|)`

§4.5 prescribed `* max(1, |new_lb|, |new_ub|)`. **Retracted.** A cancellation
residual is *small by construction*: terms of magnitude 1e10 that cancel to
`+3e-5` give `max(1, 3e-5) = 1`, so the prescription changes nothing at exactly the
site it was written for. Worse, at `M = 1e14` the multiplicative form yields a
threshold of 1e8, which would snap away a genuinely empty interval 1.4e6 wide —
turning a soundness fix into a soundness defect in the other direction.

**The scale lives in the inputs, not in the result.** The fix is an additive
round-off bound over the terms the crossover was differenced from,
`ROUNDOFF_OPS * u * sum|terms|`, now shared from `_relax/_numeric.py`
(`roundoff_slack`, `roundoff_slack_arr`) so the other layers reuse one definition.
`_EMPTY_INTERVAL_FEAS_TOL` still reads `1e-6`, and is pinned as unchanged by a test.

Two-arm verification: with the marker asserted **absent**, 81 of 117 pinning tests
fail on the baseline; all 117 pass on the fix. The suite sweeps four row shapes ×
five magnitudes (1e11 … 1e18) and carries a matching *no-weakening* arm at every
point — a genuine violation at the same magnitude, which must still be proved —
because the cheap way to pass the first arm is to stop proving infeasibility.

### 6.2 `factorable_reform._ZERO_MARGIN` — the denominator sign gate

`_find_clearable_denominator` licenses multiplying a whole constraint through by a
denominator once it believes the denominator is sign-definite, deciding that from
`_bound_expression`, which is plain float interval arithmetic with **no outward
rounding**. Its `lo` is therefore not a rigorous under-estimate of the true
infimum, and reading a *sign* off it against an absolute 1e-9 is the defect class.

This one is worse than a lost tightening. Clearing a sign-indefinite denominator
**flips the inequality** wherever the denominator is negative, so the rewritten
model has a different feasible set than the one the user wrote — a false-optimal
and false-infeasible generator.

*Entry experiment* (`scripts/audit_1397_denominator_sign_margin.py`): denominators
of the form `y + M - M + residue` (`dm` does not fold constants, so the four leaves
survive and the float evaluation genuinely cancels) whose exact infimum is −0.5 or
−8.0 while the folded lower bound clears 1e-9. **Baseline: 5 unsound clears,
exit 1. After the fix: 0, exit 0**, with a no-weakening arm of four genuinely
definite boxes that must still clear.

*The fix* is a rigorous per-endpoint error propagation over `_bound_expression`'s
own operator dispatch, `gdp_reformulate.bound_expression_error(expr, model)
-> (err_lo, err_hi)`, and the sign test becomes `lo > _ZERO_MARGIN + err_lo`
(and `hi < -_ZERO_MARGIN - err_hi`), with `dmin` reduced by the same slack before
the clamp. `_ZERO_MARGIN` itself does not move.

**Per-endpoint, not a scalar — measured.** A first version collapsed both endpoints
into one error and lost **92 of 303** denominator clears on the 66-instance in-repo
corpus, three `heatexch_gen*` instances to zero. The reason is that the endpoints
fail separately: `0.01 + x` with `x.ub = +inf` has an exact lower endpoint and an
unusable upper one, and the positive-sign test reads the *lower* one. Returning a
pair, plus monotone-slope handlers for `exp`/`log`/`sqrt` and the exact literal
`(-1, 1)` of `sin`/`cos`, brings it to **303 → 303: zero strength lost**
(`scripts/audit_1397_denominator_clear_strength.py`, both arms interleaved in one
process). Both CLAUDE.md §5 bars are therefore met — cert-clean and not harmful.

Anything `bound_expression_error` cannot bound returns `inf` and the clear is
refused; the McCormick-lp path bounds the division instead. Refusing is sound;
guessing is not (CLAUDE.md §3).

### 6.3 The reduced-cost deadbands — `node_reduce._RC_TOL`, `solver._RCF_RC_TOL`

§4.5 named the unsound direction correctly but not the yardstick. A reduced cost is
a **difference** — `c_j - (A^T y)_j` at the simplex site, `mult_x_L - mult_x_U`
(IPM bound multipliers) at the POUNCE site — so its own magnitude says nothing
about how much of it is the round-off of the arithmetic that produced it. Two bound
multipliers of magnitude 1e9 differ by one ulp at 2.4e-7, past the 1e-7 deadband.
The unsound direction is specific and not conservative: RC-fixing **divides** the
optimality gap by `|d_j|`, so an over-stated `|d_j|` makes `gap/|d_j|` too small and
can fix the true optimum out of the box. (Derived: the pre-existing
`gap += 1e-6*(1 + |z_inc|)` margin only covers this for `|c| <= ~500`.)

The yardstick is #1397's check 2 read literally — the magnitude of the terms the
difference was taken of — so it is computed *by the code that forms the difference*
and carried alongside it: `MccormickLPResult.rc_absum` (`|c_j| + (|A|^T|y|)_j`, one
extra matvec on the same sparse structure) and `LPResult.rc_absum` (`|mult_l| +
|mult_u|` from POUNCE, `|c_j| + (|A|^T|y|)_j` from the in-house simplex). `|A|^T|y|`
rather than `|A^T y|`, because the dot product's own cancellation is part of the
error. Both consumers then use `d_safe = |d_j| - roundoff_slack(absum_j)` for
**both** the deadband test and the divisor, and **refuse the reduction entirely**
when no scale was reported. Neither constant moves.

Two-arm verification (`scripts/audit_1397_reduced_cost_deadband.py`, one probe run
against both trees, the baseline arm reached by dropping the kwarg the pre-fix
signature does not accept): **baseline 6 unsound fixes, exit 1; fix 0, exit 0**,
with a no-weakening arm requiring an honest reduced cost to land on the *same*
`floor()` — a 1e-16-relative slack must not move it.

### 6.4 `posynomial._POS_TOL` — reclassified, not a defect

§4.5's cited comparison (`all(arr > _POS_TOL)`) is in
`_relax/convexity/log_lattice.py`, not `posynomial.py`, and all three of its uses
compare **declared model data** — a variable's `lb`, a constant leaf — not an
accumulated quantity, so there is no arithmetic whose error could invert the test.

In `posynomial.py` itself there are three uses and none is scale-exposed:

- `abs(exp) > _POS_TOL` on a monomial **exponent**. Exponents are model literals,
  O(1)–O(10) by construction; a cancellation residue reaching 1e-12 would need
  exponents of magnitude ≳1e4, which are not representable model inputs in any
  meaningful sense.
- `abs(right.coeff) <= _POS_TOL` → `return None`, and `coeff <= _POS_TOL` →
  `return None`. Both **refuse on failure**, the sound direction. The unsound
  direction would be a truly non-positive coefficient reading positive, and that
  cannot happen: `Monomial.coeff` is built by a **product** chain
  (`left.coeff * right.coeff`, `mono.coeff * scale`), never a sum, and
  floating-point multiplication preserves sign exactly and carries relative error
  only.

### 6.5 `lp_simplex._BOUND_SNAP_TOL` — kept as a documented guard

The constant's own comment states its design: 1e-3 is chosen "comfortably above
observed simplex round-off (~1e-4 on wide-range LPs) yet far below any meaningful
constraint scale, so a genuine solver defect (a large off-box value) is left intact
to surface in tests." The gate is therefore not a soundness threshold that a scale
factor would make coherent — it is a **fail-loud guard**, and scaling it by problem
magnitude would *widen* the snap on large models, silently absorbing precisely the
defects it exists to expose. CLAUDE.md §1 and §3 forbid weakening a guard to make a
gate or an argument pass, so the honest outcome is to record the classification and
leave the constant alone. A correction to §4.5's row while we are here: it says this constant is "reached from
`lp_pounce.py:403`". It is not. `lp_pounce.py:183` defines its **own**
`_BOUND_SNAP_TOL = 1e-7`, for a different quantity — an `lb > ub` *inversion*, not an
off-box `x` — and `:403` uses that one, relatively
(`(lb - ub) <= _BOUND_SNAP_TOL * (1.0 + abs(ub))`). `lp_simplex.py:46`'s `1e-3` is read
from exactly two lines, both inside `lp_simplex.py`. There is no cross-module reach and
no shared constant, so the "one is scaled, the other is not" inconsistency the row
implies does not exist: the relative form is right for a bound inversion and the
absolute form is right for a fail-loud off-box guard.

### 6.6 Two further sites found during the follow-up

Both are the same class, both found while reading the sites above, both fixed here:

- `perspective.find_candidates`' separability gate computed the off-diagonal mass as
  `|Q[j,:]|.sum() - |Q[j,j]|`. That is a catastrophic difference, so the quantity
  compared against the absolute 1e-12 was itself scale-dependent: at
  `Q[j,j] = 1e12` (ulp 1.2e-4) a genuine cross-term of 1e-5 is absorbed by the row
  sum and the subtraction returns exactly `0.0`. The docstring calls each gate "a
  soundness condition, not a heuristic", and this one feeds a perspective
  strengthening of the OA master cut that is valid only for a separable term. The
  fix sums the off-diagonals **directly**: a sum of non-negative terms has no
  cancellation, so its error is relative to itself and the absolute comparison is
  coherent again. Zero cost, tolerance unchanged. Two-arm: 4 pinning tests fail on
  the baseline, all pass on the fix, with a no-weakening arm at the same four
  diagonals and no cross-term.
- The same function's semicontinuity-row scan compared *accumulated* coefficients
  (`terms[i] = terms.get(i, 0.0) + v`) and an accumulated constant against the same
  absolute 1e-12. A coefficient that is truly zero can survive cancellation as a
  small nonzero, and that row is what licenses "`y = 0` pins `x` to 0" — a spurious
  `cx > 0` would let the perspective row forbid points the original model allows.
  The magnitude of the cancelled contributions is not observable from the surviving
  coefficients, but it is bounded below by the row's own magnitude, so the coherent
  test is relative to the row. This only ever rejects more rows than before, and the
  rows it newly rejects have `U` beyond `_U_CAP` anyway.

### 6.7 Where the shared definition lives

`python/discopt/_relax/_numeric.py` is now the single home for the round-off bound:
`FLOAT_EPS`, `ROUNDOFF_OPS = 8.0`, `roundoff_slack(*terms)`,
`roundoff_slack_arr(lo, hi)`, and `is_effectively_finite`. Every fix above imports
from it rather than re-deriving a local constant, so the justification lives in one
docstring and the operation count can only be changed in one place.

Two rules recorded there, both from measurement:

- **Filter on `math.isfinite`, never on `EFFECTIVE_INF = 1e19`.** That sentinel means
  "no scale information" for a *declared bound* (the default box is ±9.999e19). It is
  wrong for a *computed row quantity*: `b*b` at `|b| = 1e12` is a genuine 1e24 and
  must contribute its scale, not be discarded as a sentinel.
- **Force `0 * inf` to `0` in the propagation.** Plain float gives `NaN`, which
  compares `False` everywhere — so a NaN error bound would silently *pass* a sign
  test. That is the measurement-discipline failure of §7 applied to the instrument.

### 6.8 What the deflation cost, measured: one test's last float bit

`pytest -m smoke` on the fix branch failed exactly once, and the failure is worth
recording because it is the *only* behavioural change the whole §6 round of fixes
produced on the existing suite:

```
python/tests/test_amp.py::test_gas_square_difference_tightening_strengthens_root_relaxation
    assert tightened_lb[4] >= 45.0
E   assert np.float64(44.99999999999986) >= 45.0
```

Measured on the gas benchmark: `lb[4]` is declared `30.0`, the
`square_difference_lower_bound` rule still fires, and it still moves the bound to the
Weymouth-implied `45.0` — now landing **1.42e-13** short of exactly `45.0`, because the
rule deflates its result outward by the round-off slack of the arithmetic that produced
it.

That shortfall is the fix working. It is a *lower* bound, so over-stating it cuts the
optimum out of the box and under-stating it costs only tightening strength — the
asymmetry §0 is built on. The assert is what was wrong: it hard-pinned the last bit of
a float-derived tightening, while the sibling unit test of the same rule's exact output
(`test_square_difference_tightens_weymouth_like_upstream_pressure`) already used
`pytest.approx` and passed unchanged for exactly that reason. The integration assert
now states the contract in two parts — the bound moved materially off its declared
`30.0`, and it reaches `45.0` up to round-off.

No tolerance constant moved to make it pass; `_EMPTY_INTERVAL_FEAS_TOL` is still
pinned at `1e-6` by `python/tests/test_1397_roundoff_yardsticks.py`. The scale of the
cost is the point: across a 1951-test smoke run, the price of making six yardsticks
dimensionally coherent was one assertion's fourteenth decimal place.

### 6.9 A seventh site, found by following a misattribution: `signomial._ZERO_TOL`

§4.5's `perspective.py` row quotes a comparison, `coeff < -_ZERO_TOL`, that **does not
appear in `perspective.py` at all**. It is `_relax/convexity/signomial.py:92` and `:99`,
against a *different* `_ZERO_TOL` defined at `signomial.py:66`. The table conflated two
same-named constants in two modules. (§6.6 fixed the real `perspective.py` sites, which
are a different shape entirely — so both rows were right that something was wrong, and
wrong about where.)

Chasing the misattribution found the worst site in this audit.
`signomial._merge_like_terms` combined monomials sharing an exponent vector with a
**running** accumulation:

```python
buckets[key] += mono.coeff          # signomial.py:139, before
...
coeff = buckets[key]
if abs(coeff) <= _ZERO_TOL:        # 1e-12, absolute
    continue                       # "the term cancelled" -> dropped
```

so the merged coefficient carried an error of order `n · u · Σ|coeff|` and was then
(a) compared against an absolute `1e-12` to decide the term had cancelled and (b) read
for its **sign** by `is_mixed_sign` / `has_negative_term`.

**The witness needs three addends.** For two doubles the rounded sum always has the sign
of the exact sum, so nothing at this site is reachable with two terms — which is likely
why it survived. With three it falls apart immediately: `ulp(1e16) = 2.0`, so in
`1e16 + 1.0 - 1e16` the `1.0` is lost before the two large terms cancel, and the running
sum reaches **exactly 0.0** while the exact sum is **1.0**.

Measured on `1e16*x*y + 1.0*x*y - 1e16*x*y + 3.0*x`:

| | before | after |
|---|---|---|
| merged `x*y` coefficient | dropped as cancelled | `1.0` |
| `form.evaluate` at `x = y = 1` | **3.0** | **4.0** |

The parsed expression is 4.0. So `is_signomial` returned a form that **evaluates
differently from the expression it was given**, directly breaking this module's stated
contract that "a non-`None` return is a genuine signomial on the strictly-positive box".
With the middle coefficient negated the lost term is the *only* negative one, so
`has_negative_term` read `False`, the form presented as a pure posynomial, and
`signomial_global` would build its DC relaxation for the wrong function — a false bound
carrying `gap_certified=True`.

Reachability is narrower than the other six: the signomial global engine is reached only
by an explicit `solver="sgo"` (#1388 retired the default-OFF auto-route flag), so this is
not on the default solve path. It is still a certifying engine, and §1 does not have a
default-path exemption.

**The fix is `math.fsum`, and it is a yardstick fix, not a tolerance fix.** Every
coefficient is a double, hence an exact binary rational, so the bucket's exact sum is
representable and `fsum` returns it correctly rounded once. The sign and the zero-ness
of the result become *exact* — which is precisely what makes the absolute `_ZERO_TOL`
comparison dimensionally coherent again. Note the difference from the other six fixes:
there the round-off is unavoidable and had to be **bounded** additively; here it can be
**removed**, so no slack term appears and no tightening strength is given up. Where that
option exists it is strictly better than a bound.

Verification: 8 parameterized arms over magnitudes `1e16 … 1e20` fail on the base tree
with `math.fsum` asserted absent and pass here, each first asserting that its own witness
actually swamps the middle term (§6); three no-weakening arms — an exactly cancelling
bucket still collapses, an ordinary mixed-sign signomial parses unchanged, and both
tolerances are pinned unmoved — pass on **both** trees. The existing corpus is unaffected:
239 passed, 1 skipped across every `signomial`/`posynomial`/`gp`/`log_lattice` test.

### 6.10 Two more §4.5 citations corrected

Recorded so the table is not trusted where it was guessing (CLAUDE.md §11):

- **`_RCF_RC_TOL` is at `solver.py:24017`**, not `:23939`. The fix in §6.3 is unaffected;
  only the citation was stale.
- **`_BOUND_SNAP_TOL` is two unrelated constants**, corrected inline in §6.5.
- **`posynomial._POS_TOL`'s `all(arr > _POS_TOL)` is `log_lattice.py:212`** against a
  *third* same-named constant at `log_lattice.py:107` — already recorded in §6.4, and the
  reclassification there stands: `log_lattice`'s comparands are variable lower bounds, and
  `posynomial.py` never accumulates a coefficient (it has no `+=` over coefficients at
  all, unlike `signomial.py:139`). That asymmetry between the two modules is exactly why
  one is a defect and the other is not.

**The lesson for the audit method.** Three of §4.5's rows cited a file that did not
contain the quoted code, in each case because two or three modules define a constant with
the same name. A grep for the *constant* found the definition; a grep for the *quoted
comparison* would have found the real site. The rows were nonetheless useful — following
the wrong pointer is what turned up §6.9 — but a table entry naming a file must be
verified against that file before anything is concluded from it, which is CLAUDE.md's
"look up an API before calling it" applied to one's own notes.

### 6.11 Site 8: the accumulated objective Hessian (`problem_classifier.py`)

§6.9's signomial defect asked an obvious follow-up: *where else does a running `+=`
build a number whose sign is later read as a certificate?* The answer is the place
with the widest blast radius in the relaxation layer.

`_extract_quadratic_terms` built each Hessian cell with

```python
q_terms[(i, j)] = q_terms.get((i, j), 0.0) + v
```

so a cell touched `k` times carries an error of order `k·u·Σ|v|`. That alone is not a
defect — an error proportional to the inputs is what floating-point arithmetic is. It
becomes one because of who reads the result. `perspective.py` gates on

```python
if Q[flat, flat] <= _ZERO_TOL:   # _ZERO_TOL = 1e-12
    continue                     # "not a positive square"
```

That is an absolute constant asked a scale-dependent question — the §4 pattern
exactly — but the consequence is worse than a missed tightening. The gate is reading
the cell's **sign** as a convexity certificate.

**Measured** (`1e16*x*x - 1.0*x*x - 1e16*x*x + 0.5*x*x`, with `x` made
semicontinuous by `x - 5y <= 0`):

```
Q[0,0] extracted      : 1.0
exact 2*fsum          : -1.0
perspective candidate on x: [(0, 0.5)]
perspective terms     : [(0, 2)]
```

The exact coefficient is negative: the term is **concave**. The extractor returned
it positive, the gate accepted it as a positive square, and
`perspective_objective_terms` emitted a perspective lift for it. A perspective
reformulation `x²/y` is a valid strengthening *only* for a convex square; applied to
a concave term it cuts off feasible points of the original model, which is a false
bound — the one category CLAUDE.md §1 gives no slack. Note the sign did not merely
get noisy, it **inverted**: `ulp(1e16) = 2.0` swallows the `-1.0` whole, so the
running sum reaches exactly `0.0` and the trailing `+0.5` sets the sign by itself.

**The fix belongs at the source, not at the gate.** This is the §3 "hard, right fix"
distinction and it is not stylistic: by the time `perspective.py` receives `Q` the
individual contributions are gone, so *no* yardstick computed from `Q` can recover
the sign. Each cell therefore keeps the list of its contributions and is summed once
by `math.fsum`:

```python
q_contrib.setdefault((i, j), []).append(v)      # in _qadd
...
q_terms = {key: math.fsum(vals) for key, vals in q_contrib.items()}
```

As in §6.9 the addends are all doubles, hence exact binary rationals, so `fsum`
returns the exact sum correctly rounded once: the cell's sign and zero-ness become
**exact** and the absolute `_ZERO_TOL` comparison is dimensionally coherent again
with no slack term and no strength given up. Dicts preserve insertion order, so the
documented "first-touch order" contract of the returned `terms` is unchanged apart
from the rounding, and an exactly cancelling cell still sums to `0.0` rather than
appearing as a phantom nonzero (pinned by a no-weakening arm).

**Why this supersedes the §6.6 patch for perspective sites E and H.** §6.6 recorded
sites E (`Q[flat, flat] <= _ZERO_TOL`) and H (`not (cand.q > _ZERO_TOL)`) as fixed by
scaling the separability gate in `perspective.py`. That patch is correct for what it
covers — a *row* magnitude — but it cannot help E and H, which read a single cell.
Those two rows are now fixed at the extractor instead, and the §6.6 change stands on
its own merits for the row-coupling gate. This is the audit's own row being
re-measured and found only partly addressed; recorded per CLAUDE.md §11.

**Blast radius, and why the panel is mandatory.** Every quadratic consumer reads this
extractor — QP classification, convexity detection, RLT, perspective — so fixing the
cell makes all of them exact at once and is bound-affecting for all of them. It is
the widest-reaching change in the PR and the reason the CLAUDE.md §5 differential
panel is run over the final tree rather than the tree as of §6.10; an earlier panel
run was killed at 3/66 precisely because continuing would have measured a tree that
was about to change. The panel result is recorded in §6.13 and on PR #1407.

Verification: 14 parameterized arms (4 magnitudes × both signs on the extractor, 4
magnitudes on the perspective consumer) fail on the base tree with the `q_contrib`
and `math.fsum` markers asserted **absent** there, and pass here; each first asserts
that its own witness actually swamps the middle contribution (§6). Four no-weakening
arms pass on **both** trees: a genuinely convex square is still lifted (`q = 1.5`),
an exactly cancelling cell stays absent rather than becoming a phantom zero, an
ordinary QP objective extracts bit-identically, and both tolerances are pinned
unmoved.

### 6.12 The linear accumulators in the same function: classified, not fixed

`_extract_quadratic_terms` also accumulates `c[idx] += scale * val` and
`const += ...`. These are the *same* class of running sum as §6.11 and the honest
thing is to say why they are not being changed rather than leave the reader to wonder
whether they were missed.

They are **out of class for this issue** because no unsound consumer has been
demonstrated for them. §4's five checks ask whether an absolute constant is compared
against a scale-dependent quantity; the defect in §6.11 is that a *sign* read from
the accumulated value gates a soundness-relevant reformulation. The linear
coefficients have no such gate: `c` is consumed as data by the LP/QP builders, where a
relative error of order `k·u` in a coefficient is ordinary floating-point
representation error and is bounded by the relaxation's own outward rounding, not by
a sign test. `const` shifts the objective uniformly and cannot flip a comparison
between two candidate solutions of the same model.

That is a statement about the consumers as they exist today, not a proof that a
running sum is fine. If a future gate reads `sign(c[j])` or `c[j] == 0.0` as a
structural certificate — the way `perspective.py` reads the Hessian diagonal — this
becomes the same defect and the same one-line `fsum` fix applies. The contributions
are already collected per cell for `q`; doing the same for `c` is cheap. It is being
left alone because a change with no demonstrated defect behind it is exactly the
hypothesis-driven work CLAUDE.md §4 forbids, and because it would widen the
bound-affecting surface of a PR whose differential panel covers the changes that do
have a demonstrated defect behind them.

### 6.13 The §5 differential panel: result, and a retraction

The panel required by CLAUDE.md §5 for a bound-changing change was run over the
**final** tree of this PR, interleaved base/fix, one instance at a time:

```
python -u panel_1397.py <base-worktree> <fix-worktree> panel_1397_final.json --reps 5
```

Launched 22:18:32 at load average 3.55 on 14 cores, with no process of my own
competing (CLAUDE.md §9's load gate; an earlier round in this series was invalidated
by three zombie probes of mine at 99% CPU, so this was checked explicitly rather than
assumed). Both arms were load-gated identically by construction: the panel alternates
arms per instance rather than running base-then-fix, so any drift in machine load hits
both columns.

Final line of the log, which is CLAUDE.md §6's executed-assertion count and the reason
this panel can be believed at all:

```
EXECUTED COMPARISONS: 66 of 66
bound differences surviving the spread test: 1 ['nvs05']
```

**First pass: 62 of 66 instances bit-identical in the dual bound.** Four differed and
were re-measured 5 reps per arm:

| instance | base range (5 reps) | sd | fix range (5 reps) | sd | verdict |
|---|---|---|---|---|---|
| `contvar` | `[171259.27256139443, 180801.31613405075]` | 4267.33 | `[180801.31613405063, 180801.31613405063]` | 0 | overlapping → wall-clock artifact |
| `nvs05` | `[4.046044721969811, 4.046044721969811]` | 0 | `[5.470716109909194, 5.470716109909194]` | 0 | **disjoint → REAL** |
| `tanksize` | `[1.2662152044076513, 1.266412848540781]` | 8.03e-5 | `[1.2662262839434653, 1.266270424402608]` | 1.83e-5 | overlapping → wall-clock artifact |
| `tls2` | `[3.1819001304431507, 5.29999842045334]` | 0.947242 | `[3.1819001304431507, 5.29999842045334]` | 0.947242 | overlapping → wall-clock artifact |

Exactly one instance moved deterministically. `nvs05` is reproducible to the last bit
in *both* arms (sd 0 on each side), which is what distinguishes a real bound change
from the three instances whose apparent movement is the node-limit/time-limit lottery.

**Bar 1, cert-clean.** Bounds checked against `minlplib.solu` as the oracle:

| instance | reference | fix bound | slack |
|---|---|---|---|
| `nvs05` | `=opt= 5.4709341080` | 5.470716109909194 | below by 2.18e-4 — **valid** |
| `tls2` | `=opt= 5.3000000000` | 5.29999842045334 | below — valid |
| `tanksize` | `=opt= 1.2686437540` | 1.266391942225201 | below — valid |
| `contvar` | `=bestdual= 560271.2759` | 180801.31613405063 | far below — valid |

No bound anywhere on the panel exceeds its reference optimum. Certification
transitions over all 66: **49 `True→True`, 15 `False→False`, 2 `False→True`, and zero
`True→False`** — no instance that was certified lost certification. `incorrect_count`
is 0. The one real change is a *tightening* that closes nearly the whole remaining
root gap on `nvs05` (4.046 → 5.4707 against an optimum of 5.4709) while staying
strictly below the optimum, which is exactly the shape a sound bound improvement has.

**Bar 2, net-positive.** One instance substantially and deterministically improved, 65
unchanged, none harmed. Node counts moved on the four disputed instances only, in both
directions (`nvs05` 71→111, `tls2` 205→153), consistent with the lottery on three of
them and with a different — tighter — relaxation on `nvs05`.

Both bars pass, so the fix ships default-ON with no flag, as a correctness fix rather
than a graduation candidate.

**Retraction (CLAUDE.md §11).** On first pass I wrote that `tls2` "gained
certification (uncertified → certified) … Big improvement, sound", from the single
first-pass sample where base read 3.1819 and fix read 5.29999. **That claim was
wrong and is withdrawn.** The 5-rep re-measurement gives base and fix *identical*
ranges `[3.1819001304431507, 5.29999842045334]` and *identical* standard deviation
0.947242: `tls2` reaches the certifying bound in some runs and stalls in others, in
both arms equally, and the first-pass pairing caught base on a low draw and fix on a
high one. The `False→True` in the transition count above is real as a log entry and
an artifact as a claim about this change. Only `nvs05` survives as a genuine
improvement. This is the second time in this series that a single-sample bound
comparison read as a solver result and was a timing artifact; the spread test is what
caught it both times.

**Mid-run edits, and why the measurement still holds.** Three commits landed in the
fix worktree while the panel was running (22:21 docs, 22:23 a docstring, 22:31 tests).
That would normally void the run, so it was checked rather than waved through:
`git diff 00434501..HEAD -- python/discopt/` is one file, `problem_classifier.py`,
11 insertions and 4 deletions, and **every changed line is `_materialise_Q` docstring
text** — no executable line changed. The panel therefore measured the behaviour of the
tree as it stands, and its numbers are the final tree's numbers.

**What the panel does *not* establish: reachability of the reduced-cost sites.**
`_tuning().phase2_dbbt` was measured `False`, so the `node_reduce` half of the §6.3
reduced-cost fix sits behind a default-OFF Regime-2 flag with no production reach
today; and `_root_reduced_cost_fixing` was not reached on a 5-binary knapsack under
either MILP backend (including `DISCOPT_LP_MILP_BACKEND=rust`) nor on `alan` or
`clay0303hfsg`. The `audit_1397_reduced_cost_deadband.py` result — 6 unsound
tightenings under base, 0 under fix — is therefore a **function-level** measurement of
a latent defect, not a demonstration of live impact, and should not be read as one.
That is the honest scope: the §6.3 fix removes a defect from a path that is correct
to fix and currently cold. (Separately: `phase2_dbbt` is not listed in
`docs/dev/flag-retirement-audit.md`, which the three-outcome rule in CLAUDE.md §5
requires of a default-OFF gate over solver math. Noted here rather than fixed, to
keep this PR scoped.)

### 6.14 Two more `node_reduce` constants: strength knobs, not soundness gates

An audit helper flagged `node_reduce._EPS = 1e-7` as a sibling of the reduced-cost
deadband fixed in §6.3, on the grounds that it is an absolute constant compared
against `ub[j] - cand`, which is scale-dependent. The observation about the units is
right; the classification is not, and the distinction is the one §4's check list
exists to make.

```python
cand = lb[j] + gap / d_safe        # the bound; _EPS does not appear
if cand < ub[j] - _EPS:            # ... only whether to record it
    ub[j] = max(lb[j], cand)
```

`_EPS` does not participate in computing `cand`. It decides only whether an
*already sound* bound is a large enough improvement to be worth writing down. Both
error directions are therefore benign: too large and a valid tightening is skipped
(strength lost, bound still correct); zero and a 1e-20 tightening is recorded
(pointless, bound still correct). On a variable with `ub[j] = 1e12` the absolute
1e-7 threshold does mean sub-1e-7 tightenings are never recorded — a real
scale-dependence, and a real strength question, but not a soundness one. Contrast
§6.3's `_RC_TOL`, which gates a *division* (`gap / d_safe`) and so sets the bound's
value directly: over-stating `|d_j|` there shrinks the step and can fix the optimum
out of the box. Same shape, opposite consequence.

The neighbouring `np.floor(cand + 1e-9)` / `np.ceil(cand - 1e-9)` integrality nudges
are also scale-exposed — at `cand = 1e10` the 1e-9 is far below the ulp (1.9e-6) and
the nudge silently becomes a no-op — but they are sound in *both* regimes, because
the nudge direction only ever weakens. For an upper bound, `floor(cand + 1e-9)` can
return 3 where the exact floor is 2; that is a looser upper bound, hence sound. When
the nudge vanishes at large magnitude, the plain floor is returned, which is also
sound. A constant whose failure mode in every regime is "weaker bound" is not a
#1397 defect.

Both are recorded here rather than fixed, so the next reader can see they were
examined and why they were left. This is the audit's own standard applied to its
helpers' findings: a site is a defect when an absolute constant is compared against
a scale-dependent quantity **and** an error in that comparison can make a bound
wrong. The second half of that conjunction is doing most of the work.

### 6.15 Site 8, second pass: two more consumers, and a docstring the fix invalidated

An audit helper re-read the same chain independently and arrived at `_qadd` from the
other direction (from §6.9's signomial find, reasoning that the same pattern would
recur). Two of its findings change what §6.11 should say; both were verified here
before being recorded.

**The blast radius is wider than `perspective.py`.** §6.11 named the perspective
candidate gate as the unsound consumer. There are two more, on the OA/GDP path, and
they are **default ON**:

- `solvers/oa.py:2949-2962` (aggregate perspective objective cut) shifts the `y=0`
  branch by `q·x̄²`. Validity needs `q_detected ≤ q_true`; with `q_true < 0` read as
  positive, the shift is invalid outright — a cut excluding feasible points.
- `solvers/oa.py:3105` (`_disaggregate_objective_cut`) refuses on
  `not np.isfinite(q) or q <= 0.0` — the same sign test, and the same fooling. Its
  own docstring explains that a term removed from one row and not another leaves the
  master double-counting, "an over-estimate of ``f``, i.e. an invalid bound".

So three independent consumers gate on the sign of a number the extractor was
computing with a running `+=`, and two of them are on a default-ON path. This does
not change the fix — `math.fsum` at the source makes the cell exact for all three at
once, which is precisely the §3 argument for fixing the extractor rather than each
gate. It does change the *severity*: §6.11 should be read as "three consumers, two
default-ON", not one.

**The fix invalidated a docstring, and that was caught by a reader rather than by
me.** `_materialise_Q`'s docstring justified #863's bit-identity claim with "the dict
performs the same ``+=`` additions in the same order starting from the same 0.0".
That was accurate when written and the §6.11 change makes it false: cells are now
exact `fsum` results, which differ from the dense predecessor's in the last ulp by
construction. Left alone it would have told the next reader that a bound-neutral
guarantee still holds when it no longer does. Corrected in place, stating what
changed and which direction the difference goes. Recorded here because "my change
falsified a neighbouring comment" is the same class of error as §11's "retract a
published claim" — a stale justification is a false claim that happens to live in a
docstring.

**`_EPS`, one detail sharper than §6.14 put it.** The helper's independent reading
agreed `_EPS` is out of class and added the mechanism: at `|ub[j]| = 1e12` the ulp is
~1.2e-4, so `ub[j] - 1e-7 == ub[j]` **exactly, by absorption** — the test does not
merely become relatively weak, it degenerates to `cand < ub[j]` and accepts any
one-ulp change. The cost is per-node staging churn in `_reduce_node_and_stage`
(`solver.py:3680`), which rewrites the batch box and pending entry for a 1-ulp move.
That makes the coherence complaint real and its consequence a *performance* one, not
a §1 one — the classification in §6.14 stands, with a better reason than it gave.

## 7. Site 9 — and a retraction of this audit's own coverage claim

**Date:** 2026-09-21. **Same branch.** Reported as a candidate by a peer session,
verified here before any code was written.

### 7.1 The retraction (CLAUDE.md §11)

I stated, in this conversation and on the strength of §6, that *"all three items of
#1397's definition of done are complete; #1397 can be closed once this PR merges —
nothing else remains in it."* **That was wrong, and this section is the retraction.**

The error was not in any individual row. It was in treating §1's *name-pattern* sweep
as if it had been a sweep of *comparison sites*. The `RX` regex in §1 matches constants
whose **name** carries a tolerance word. A bare numeric literal compared inline carries
no name, so it was never enumerated — and §4's 103 rows inherited that blind spot
without saying so. `edge_concave.py` compares against the literals `1e-12` and `1e-9`
with no named constant anywhere in the file:

```bash
$ grep -c 'edge_concave\|hessian_tol' docs/dev/1397-absolute-tolerance-audit-2026-09-20.md
0
```

§5 lists three exclusions — Rust, value-constants, practice-hit claims — and **none of
them covers `_relax/`**. So the document read as complete over the Python relaxation
layer while in fact covering only the *named* constants in it. §5 now records this as a
fourth exclusion, and it is the one that matters most, because the inline-literal sites
are exactly the ones no one gave a name to and therefore no one defended.

### 7.2 The site

`_relax/edge_concave.py` accumulated the `x_i^2` coefficients of each candidate block
with a running `+=`, then read the **sign** of the result to choose `sense`:

```python
sq[i] = sq.get(i, 0.0) + float(coeff)          # :146, running sum
...
diag = [sq.get(i, 0.0) for i in varset]        # :156
if all(v <= 1e-12 for v in diag) and any(v < -1e-9 for v in diag):
    sense = "under"
elif all(v >= -1e-12 for v in diag) and any(v > 1e-9 for v in diag):
    sense = "over"
```

`sense` is not a tightness knob. At `mccormick_lp.py:3159` it selects the **direction**
of the inequality appended to the node LP, and the two branches emit opposite rows
(`A.x - q <= -B` versus `q - A.x <= B`). For an edge-concave block the *minimum* is at a
box vertex, so vertex data gives a valid under-estimator; for an edge-convex block the
*maximum* is at a vertex. Flip the sign and the vertex-derived hyperplane is emitted on
the side where it is not a bound at all.

**There is no downstream check.** `_separate_edge_concave`'s `_append(rows, rhs)` stacks
the row straight into `milp._A_ub` / `milp._b_ub` — no feasible-point sampling, no
incumbent verification — and the whole function sits inside
`except Exception: logger.debug(...)`. The path is default-ON
(`DISCOPT_EDGE_CONCAVE` defaults true) inside the default MINLP node relaxer.

A purely bilinear variable makes the block reachable with a **single** contaminated
entry: `sq.get(i, 0.0)` returns `0.0`, which satisfies both `all(v <= 1e-12)` and
`all(v >= -1e-12)`, so it blocks neither branch. That `.get` default also erased the
distinction the fix needs — a *structurally absent* coefficient is an exact zero, while
a *computed* one near zero is not.

### 7.3 Entry experiment, and the severity

Reached through the real `Model` → `distribute_products` → `_expr_to_polynomial` → 
`collect_edge_concave_quadratics` path, not a synthetic proxy (#727 lesson). The
pattern is addends `[M, t, -M, -u]` with `t < u`, so the true sum is negative
(edge-concave), but `t` exceeds half an ulp of `M` while `u` does not:

| quantity | value |
| --- | --- |
| addends | `[51903611.79275842, 5.5700e-09, -51903611.79275842, -5.9653e-09]` |
| true sum (`math.fsum`) | `-3.952954e-10` — negative, edge-**concave** |
| running sum | `+1.485329e-09` — clears the `1e-9` edge-**convex** threshold |
| collected `sense` | `"over"` — the wrong side |

63 of 800,000 random trials flip this way. The residue scales with the magnitude that
cancelled while `1e-9` does not, so severity is unbounded. Invalidity of the emitted
cut over a width-10 box:

| cancelling magnitude `M` | worst true curvature still read as positive | cut invalid by |
| --- | --- | --- |
| 1e6 | 0 | 0 |
| 1e8 | 6.41e-09 | 6.41e-07 |
| 1e10 | 9.34e-07 | **9.34e-05** |
| 1e12 | 5.59e-05 | **5.59e-03** |
| 1e14 | 6.89e-03 | **6.89e-01** |
| 1e16 | 9.29e-01 | **9.29e+01** |

Everything from `M = 1e10` up exceeds the 1e-6 feasibility tolerance. This is a §4.5
site — scale-exposed, unsound, on a certifying path — not a round-off curiosity.

**The same threshold also loses valid cuts.** The `all(v <= 1e-12)` / `any(v < -1e-9)`
pair is internally incoherent: a diagonal in `(-1e-9, -1e-12]` satisfies the `all` test
but fails the `any` test, so a genuinely edge-concave block whose only curvature sits in
that window is silently dropped. Measured: the same reproducer under two other addend
orders yields no block at all, and an honest single-addend curvature of `-1e-30` was
discarded outright. The fix recovers both.

### 7.4 The fix

Per #1397's non-goal, **no constant's value changed**. The yardstick changed:

1. Coefficients accumulate into **lists**, summed once with `math.fsum` — correctly
   rounded, so the sum no longer depends on term order (the same `fsum`-at-the-source
   move as §6.11).
2. A new `_definite_sign(total, addends)` returns `+1` / `-1` only when
   `|total| > 64 * (n+1) * eps * Σ|addend|`, and `0` otherwise. The bound carries the
   units of the addends, because the error inherited from the upstream expansion is set
   by the magnitudes that went *in*, not by the magnitude that came *out*.
3. A variable absent from the accumulator is an exact zero and is compatible with either
   sense. A variable whose coefficient does not clear its own round-off bound has **no
   determined sign** and disqualifies the block. Refusing is the sound choice: a cut
   whose side cannot be justified is not emitted (CLAUDE.md §3).

`64` is a safety factor on a round-off bound, not a tuned tolerance: an uncancelled
coefficient clears it by ~14 orders of magnitude, and the cancelling case falls ~1e6
*below* it, so no nearby value behaves differently.

The module docstring's claim that detection "is exact for a quadratic (constant
Hessian)" was corrected in the same commit. It was true symbolically and false of the
floating-point accumulation the code actually performs — the same class of error as
§6.15's stale `_materialise_Q` docstring.

### 7.5 Verification

`python/tests/test_1397_edge_concave_sign.py`, 15 tests, no `slow` marker (so it
collects under CI's default `-m "not slow"`), including a 6-point sweep over the
cancelling magnitude. One test guards the premise — that the reproducer really does
invert the sign — so the rest cannot pass vacuously (§6).

Before/after was measured by running the behavioural assertions against both trees
with a load gate asserting `_definite_sign` **absent** on the baseline and **present**
on the fix (§8). Baseline: 11 checks, **7 failures** — the wrong `sense` at every
magnitude from 1e8 to 1e16, plus the lost `-1e-30` block. Fixed: 11 checks, **0
failures**. A separate positive control confirms five well-conditioned blocks
(edge-concave, edge-convex, purely-bilinear partner, accumulated-but-uncancelled,
tiny-but-honest) are all still collected — a soundness fix that quietly disables the
feature is not a fix.

### 7.6 Site 10 is deferred, not cleared

The same peer report named `_relax/cutting_planes.py`'s `hessian_tol` (default `1e-8`,
consumed as `np.abs(hess) > hessian_tol` to pick the "curved" variable set) as a
candidate. It arrived truncated and is **not** verified here. It is recorded as an open
candidate rather than classified, so this document does not repeat §7.1's mistake in
the other direction by implying it was checked.
