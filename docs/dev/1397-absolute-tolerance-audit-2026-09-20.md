# #1397 — the absolute-tolerance audit

**Status:** complete. Four scale-exposed unsound sites found and fixed; the rest
classified below. **Date:** 2026-09-20. **Branch:** `audit/1397-absolute-tolerance-sweep`.

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
| `_relax/nonlinear_bound_tightening.py:58` | `_EMPTY_INTERVAL_FEAS_TOL` | 17 comparison lines. `new_lb - new_ub <= tol` snaps a sub-tolerance crossover to a midpoint instead of declaring the node infeasible. A *larger* tolerance is therefore the safe direction; the unsound direction is this absolute 1e-6 being too small relative to scale — on a box with bounds ~1e10, ordinary rounding crossover exceeds it, falls through, and emits `status="infeasible"`. Needs `* max(1, \|new_lb\|, \|new_ub\|)`. | open |
| `_relax/node_reduce.py:48` | `_RC_TOL` | reduced-cost fixing. `dj` carries objective-over-variable units; a genuine zero reduced cost reading above an absolute 1e-7 at large objective scale makes `cand = lb + gap/dj` spuriously small and **fixes the optimum out of the box**. Same shape at `solver.py:23939` (`_RCF_RC_TOL`). | open |
| `_relax/perspective.py:74` | `_ZERO_TOL` | 13 comparison lines, used as `coeff < -_ZERO_TOL` — coefficients carry the model's units, so at small coefficient scale a genuinely negative term reads as zero and a perspective reformulation is applied to a form that does not admit it. | open |
| `_relax/convexity/posynomial.py:54` | `_POS_TOL` | `all(arr > _POS_TOL)` decides posynomial-ness from coefficient signs; same units argument. | open |
| `_relax/factorable_reform.py:69` | `_ZERO_MARGIN` | `lo > _ZERO_MARGIN` decides a bound is strictly positive before a reformulation that requires it. | open |
| `solvers/lp_simplex.py:46` | `_BOUND_SNAP_TOL = 1e-3` | snaps `x` onto a bound without recomputing the objective; 1e-3 is enormous next to the declared `abs=1e-6`. Reached from `lp_pounce.py:403` as `(lb - ub) <= _BOUND_SNAP_TOL * (1.0 + ...)` — *that* call site is scaled; the unscaled uses need separate treatment. | open |

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
