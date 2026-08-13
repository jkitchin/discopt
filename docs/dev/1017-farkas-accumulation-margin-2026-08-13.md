# The Farkas certificate's margin must bound its accumulations (#1017, 2026-08-13)

**Status.** Fixed. `farkas_ray_certifies_cols`
(`crates/discopt-core/src/lp/simplex/primal.rs`) now subtracts a forward-error bound
assembled from the magnitudes that were actually summed, instead of a flat
`1e-9`-relative constant scaled off the *results*. The engine-side twin of the
Python boundary's #309 sharp NS margin (`docs/dev/ns-sharp-margin-2026-07-16.md`),
which graduated default-ON on 2026-07-16 and which this path never received.

## 1. The defect

A node-LP `LpStatus::Infeasible` is a fathoming proof: the B&B deletes the node. The
engine emits it only when a candidate dual ray `y` is *verified* — when the `c = 0`
Neumaier–Shcherbina quantity

    g₀(y) = bᵀy − max_{l ≤ x ≤ u} (Aᵀy)ᵀx

is strictly positive. Strictly positive **in exact arithmetic**, which is why a
margin is subtracted. The margin was

```rust
let margin = 1e-9 * (1.0 + by.abs() + abs_sum);
```

where `abs_sum = Σ|boxmax_j|`. Both terms are sizes of *results*. Neither bounds the
rounding of the accumulations that produced them:

* `by` is a length-`m` dot product — error `≤ γ_m · Σ|b_i·y_i|`, and `Σ|b_i·y_i|` can
  be arbitrarily larger than `|by|`;
* each `aty_j = Σ_i a_ij·y_i` is another cancelling dot product, and only its result
  reaches `abs_sum`;
* the implied slack upper bounds from `slack_upper_bounds` are `(b_i − min_other)/a`
  with `min_other` a floating-point row sum, whose rounding was not represented at
  all.

Reported on a captured `QPLIB_3814` relaxation LP: `bᵀy = 3.0000052e-8` out of terms
whose absolute sum is `600` — a relative cancellation of **5e-11**, tested against a
margin of `1e-9`. The LP is feasible (SciPy/HiGHS `optimal` 0.238394628; three
perturbed pivot paths of our own engine agree; an elastic `min t` LP gives `t* = 0`),
and the verdict flips to `optimal` under any perturbation of the pivot path. So the
same LP fathomed or did not depending on rounding — CLAUDE.md §1, not a performance
concern.

## 2. The fix

`margin = max(legacy, rigorous)`, where the rigorous part is (Higham 2002 §3.1,
`γ_k = k·u/(1 − k·u)`, `u = f64::EPSILON` = 2× the true unit roundoff so the
first-order bound dominates the dropped `O(u²)` terms and the margin's own
evaluation):

1. **Row side** — `γ_{m+1} · Σ|b_i·y_i|`, accumulated alongside `by` itself.
2. **Column side** — `aty_j` is known only to lie in `[aty_j − e_j, aty_j + e_j]`
   with `e_j = γ_{nnz_j+2}·Σ_i|a_ij·y_i|`. Since `sup_{x_j∈[l_j,u_j]} t·x_j` is convex
   in `t`, its maximum over that interval is attained at an endpoint, so the column's
   box-max is upper-bounded by evaluating both endpoints and taking the larger. An
   endpoint that selects a genuinely open side yields `+∞` and bails the sign — the
   same conservative exit the pre-existing code took, now also taken when the *sign*
   of `aty_j` is uncertain. (This mirrors the interval-corner treatment of
   sign-uncertain reduced costs in `_safe_lp_lower_bound_sharp`.)
3. **Implied slack bounds** — `slack_upper_bounds` now carries `Σ|a_ik·bnd_k|` and the
   term count per row and per side, and rounds each recovered bound **up** by
   `γ_{k+1}·Σ|terms| / |a|` plus a few ulps. A larger upper bound is free soundness
   here: it can only enlarge the box, hence only enlarge the box-maximum, hence only
   weaken a certificate.
4. **Accumulation of the column terms** — `γ_{n+1}·Σ|boxmax_j|` (the `+1` covering one
   product rounding per term), plus the final subtraction's own rounding.

Keeping the pre-#1017 flat floor as a lower bound on the margin makes the change
**monotone**: it can only ever withdraw a certificate, never issue one. Its entire
risk is therefore lost fathoming, which is measurable — and it cannot introduce a new
false `Infeasible` by construction.

`profile::Ctr::FarkasRejectCancellation` counts the rejections that clear the legacy
floor but not the rigorous bound, i.e. exactly the certificates this change removes,
so the cost is visible in any profiled run without an A/B rebuild.

## 3. Why this is sound by construction, not by instance

The soundness claim is a theorem, not an observation: if every error source is
bounded, then `computed g₀ > margin ⇒ exact g₀ > 0`, so a **feasible** LP can never
certify — for `QPLIB_3814` or anything else. The reported LP could not be replayed
here (`scratchpad/i1013/` is not in the repository and the MINLPLib/QPLIB corpus is
not present in this container), so the regression coverage below reproduces the
*mechanism* in arithmetic small enough to verify by hand, rather than pinning the one
instance.

## 4. Regression coverage

`c39_farkas_margin_rejects_a_cancellation_only_certificate_1017` (primal.rs) and
`python/tests/test_1017_farkas_cancellation_margin.py` (through the shipped
extension) both build an LP that is **feasible by construction**: `A_j = e_j − e_{j+1}`
(so `Aᵀ1 = 0` exactly and `range(A) = {v : Σv_i = 0}`) with

    b = [1e16, 1.5×8, −1e16, −12]

which sums to exactly zero as a real number — hence `b ∈ range(A)` and `Ax = b` has a
real solution well inside the box. The naive left-to-right sum the check performs
returns `+4`, because each `1.5` added onto `1e16` (where the ulp is `2`) rounds up by
`0.5`. Pre-fix that cleared the `5e-9` margin and the engine returned `Infeasible`
(verified on the branch point, `ab0cc54`); the rigorous margin is `γ_12 · 2e16 ≈ 53`
and rejects it.

`c39_farkas_never_certifies_a_feasible_cancellation_family_1017` sweeps that
construction over the big-term magnitude, small-term count and step (120 LPs × 2
signs): **114 of the 240 checks false-certify on the pre-#1017 margin**, none after.

## 5. The cost side, measured

`c39_genuine_infeasibility_keeps_wide_margin_headroom_1017` reports the certificate
quantities on the two captured *genuinely-infeasible* node LPs already in
`testdata/`:

| fixture | m×n | g₀ | rigorous margin | legacy margin | headroom |
|---|---|---:|---:|---:|---:|
| `c39_surplus_slack` | 159×186 | 3.810e-2 | 7.650e-14 | 2.513e-9 | 5.0e11× |
| `c39_scaled_surplus_slack` | 531×685 | 1.004e0 | 3.141e-10 | 1.098e-7 | 3.2e9× |

On both, the rigorous term is *below* the legacy floor, so `max(legacy, rigorous)` is
the legacy value and the check is unchanged. The new term binds only when the
cancellation ratio `Σ|terms| / |result|` exceeds ~1e4; the reported LP's was 2e10.

## 6. Corpus panel: exactly bound-neutral

`scratchpad/issue1017_panel.py`, baseline (`ab0cc54`, wheel-installed) vs fix
(editable), both marker-checked in **both** directions — the worker runs the crafted
cancellation LP through `solve_lp_py` first and refuses to measure if the loaded
extension is not the arm it was told to expect (CLAUDE.md §8).

The panel is 13 instances selected by `scratchpad/issue1017_scout.py` for actually
*reaching* the certificate (22 of 48 candidates do), plus 3 Farkas-inactive controls.
The first cut of this panel was a no-op that read as a pass — every counter zero,
because `profile::incr` is compiled to nothing unless `DISCOPT_PROFILE` is set and
because the instances chosen never produced an infeasible node LP. That is why the
panel now prints the certification count and **fails** when it is zero (§6 of the
measurement discipline).

| quantity | baseline | fix |
|---|---:|---:|
| instances certifying (`gap_certified`) | 16/16 | 16/16 |
| successful Farkas certifications (`LpVerdict/WarmVerdictInfeasible`) | 335 | **335** |
| margin rejections (`FarkasRejectMargin`) | 5764 | **5764** |
| open rejections (`FarkasRejectOpen`) | 0 | 0 |
| certificates withdrawn (`FarkasRejectCancellation`) | — | **0** |

`status`, `gap_certified`, `node_count`, `objective` and `bound` are identical on
every instance — the bound-neutral regime of CLAUDE.md §5, exactly. Not one of the
335 real certificates on this corpus depended on cancelled accumulation; the tighter
margin costs nothing here, and the counter is in place to say so on any future run.
