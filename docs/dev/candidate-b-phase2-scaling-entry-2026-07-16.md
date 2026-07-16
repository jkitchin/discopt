# Candidate (B) entry experiment — hda phase-2 blocked by post-scaling residual range (#664)

Follow-up to #517 / #662 (candidate A, merged). Goal (B): make the in-house
simplex converge on hda's ill-conditioned node LP so hda gets a **tight** dual
bound, not the loose (A) floor. Per Dev-Philosophy #4, this is the entry
experiment run **before** any implementation.

## Diagnosis (built-in profiler, `DISCOPT_PROFILE`)

hda's node LP (2974 rows × 4112 cols lifted):

| metric | value |
|---|---|
| Phase1Pivots | 8 540 – 16 504 (degenerate grind) |
| Phase2Pivots | 0 – 2 156 (starved) |
| Refactorizations | 188 – 398 (mostly `RefacCap`, the 48-update cap) |
| `RefacFtFail` cause | **`Growth` 18–26**, `TinyPivot` 2–6, **`Singular` 0** |

At `time_limit=150 s` hda explores only **7 nodes** and never certifies a bound —
a conditioning/speed wall, not a time budget. The FT failures are **numerical
growth** in feral's eta factors (not singular bases, not tiny pivots).

## Killed lever — feral iterative refinement (`with_numeric_focus`)

Enabling feral's 2-step iterative refinement in the *main pivoting* factorization
(`primal.rs` / `dual.rs`, env-gated `DISCOPT_LP_NUMERIC_FOCUS`) left the profile
**byte-identical** (8540 / 0 / 20 / 18) and hda still `bound=None`. Refinement
polishes solve *accuracy* but does not change the FT-update growth that forces the
grind. **KILLED.**

## Confirmed root cause — the scaler leaves a ~1e20 residual range

Measured the coefficient spread feral actually factorizes, *after* the
geometric-mean equilibration (`scaling.rs`, env-gated `DISCOPT_SCALE_DEBUG`):

```
m=2974 n=4112 raw_spread=2.837e26  scaled_residual_spread=1.845e20
m=2974 n=4112 raw_spread=5.765e17  scaled_residual_spread=2.951e20   (scaling made it WORSE)
```

The equilibration barely dents the spread — and in one case *increases* it. The
mechanism: the scaler's noise floor (`MAX_LINE_RANGE = 1e-10`) **excludes** tiny
coefficients (cancellation residue ~1e-16, envelope slopes ~1e-9) from the scale
computation but leaves them **in the matrix**. Scaling the structural entries
toward 1 makes those excluded tiny entries relatively even smaller, so the
residual dynamic range balloons to ~1e20 — which feral's LU cannot factorize
without runaway growth.

## Tested fix hypotheses (both refined/falsified) — the conditioning is harder than a scaling tweak

**H1 — drop/fold sub-tolerance coefficients: WRONG.** Capturing the real node LP
and tracing its entries showed the residual-driving small coefficients are **not**
droppable cancellation residue. They are **crushed real coefficients**: hda's
Arrhenius rows pair a huge pre-exponential (~6.3e10) with normal (~1) entries.
Geometric-mean row scaling crushes the ~1 entries to ~1e-11 (they are below the
`MAX_LINE_RANGE=1e-10` noise floor relative to 6.3e10, so the floor *excludes* them
from the scale computation). Dropping them would corrupt real constraints — not a
sound fix.

**H2 — stronger scaling (lower noise floor + more passes): FALSIFIED.** Env-gated
`DISCOPT_SCALE_PASSES=12 DISCOPT_SCALE_LINE_RANGE=1e-14` made the in-engine residual
**worse** (1.8e20 → 2.25e21), and hda still `bound=None`. Root cause of the limit:
the Rust scaler runs on the **standard-form matrix `[A | I]` with slack columns**
(n=4112 = 1138 structural + 2974 slacks). The identity slacks **pin** the row
scales, so no diagonal scaling of the full standard form can balance the Arrhenius
rows.

**The two irreducible facts:**

1. Scaling `A_ub` *alone* (no slacks; 8-pass geo-mean, no floor) reaches ~**2.5e9**
   — vs ~2e20 for the slack-included standard form. So scaling *before* the
   standard-form conversion is worth ~1e11.
2. Even 2.5e9 exceeds feral's reliable range (~1e7, per this module's own header):
   the 6.3e10 Arrhenius/normal coupling is an **irreducible ~1e9 conditioning** that
   diagonal scaling cannot remove.

## L1 — slack-aware scaling: BUILT + MEASURED (necessary, effective, insufficient alone)

Implemented (env-gated `DISCOPT_LP_SLACK_AWARE_SCALE`, since reverted): a
single-entry column is a slack (standard-form `I`) or bound row; exclude such
columns from the row equilibration and normalize each to 1 via its own column
scale, so slacks no longer pin the row factors.

L1 alone barely moved the residual (1.8e20 → ~1e20) because the *structural*
scaling is itself capped by the 4-pass / `MAX_LINE_RANGE=1e-10` config. **L1 +
stronger structural scaling** (`DISCOPT_SCALE_PASSES=12 DISCOPT_SCALE_LINE_RANGE=1e-16`)
collapses it as theory predicts:

```
raw=2.837e26  →  scaled_residual = 4.096e9    (vs 1.845e20 default — a ~1e10 win)
```

But **hda end-to-end still `bound=None`**: **4e9 exceeds feral's ~1e7 reliable
range** at m=2974. The dense LU route can't rescue it (m=2974 ≫ `FORCE_DENSE_M=256`;
a dense O(m³) factorization is infeasible at this size), so the basis stays on the
sparse route. Diagonal scaling provably cannot go below ~4e9 (the 6.3e10 Arrhenius
coupling), so **no combination of L1 + scaling params + L2 factorization route
solves hda** — the conditioning floor is above feral's ceiling.

## Verdict — hda needs L3 (model-level rescale); L1 is a separable general win

- **hda specifically:** blocked. Even the theoretical-best diagonal scaling (4e9)
  is past feral's factorization limit at this size. The only lever that helps is
  **L3** — normalize the `6.3e10·exp(−E/RT)` product (aux value ~1e-13, product
  ~6e-3) at *relaxation build*, removing the 6.3e10 from the matrix so conditioning
  drops below ~1e7. Most fundamental fix, largest surface; not yet built.
- **L1 as a general improvement:** the slack-aware scaling (+ tuned stronger
  structural params) is a real, separable conditioning win that likely helps
  *other* ill-conditioned instances whose residual currently sits between feral's
  ~1e7 ceiling and the ~1e20 the slacks impose. Worth productionizing on its own —
  but it changes scaling factors globally, so it must ship bound-neutral-verified
  (panel node_count/objective unchanged + `cargo test`) with carefully tuned
  params (the `MAX_LINE_RANGE` floor exists to guard spurious residue). That is its
  own task, separate from the hda goal.

Candidate A's loose floor remains hda's shipped fallback until L3 lands.

## L3 entry experiment — the conditioning is irreducible and structural (measured)

Before building any reformulation, solved the **true optimal diagonal scaling** of
hda's captured node matrix (the classic log-space min-max LP:
`min (M−m)` s.t. `m ≤ log|a_ij| + r_i + c_j ≤ M`, via HiGHS):

```
OPTIMAL diagonal-scaling residual spread = 1.771e9
```

So the heuristic ~2–4e9 was near-optimal; **~1.8e9 is the rigorous floor** no
scaling algorithm can beat. And it is **not** caused by residue: recomputing the
optimum after dropping every entry below 1e-14 … 1e-8 leaves it **exactly 1.771e9**
at all thresholds. Tracing the binding entries: they are **normal-magnitude
coefficients** (|a|≈1) — e.g. column 722 (6 rows) is forced to *both* extremes
(scaled `e^0` in some rows, `e^-21.3` in others) because its rows demand scales
differing by 1.77e9. It is a genuine **non-diagonal structural coupling** among real
coefficients (the Arrhenius `6.3e10` / `3.27e6` rate rows vs the ~1 mass-balance
rows sharing variables), which diagonal scaling provably cannot resolve.

**Conclusion — candidate B is blocked at the relaxation-conditioning level, not the
engine level.** hda's LP relaxation is inherently ~1.8e9-conditioned; float64
simplex (feral ~1e7; HiGHS also false-infeasibles on the unscaled form) cannot
solve it cleanly regardless of scaling/factorization. The only genuine fixes are
both major and uncertain:

- **L3-structural** — a *general* relaxation-build reformulation that changes the
  coefficient structure (variable substitution/elimination to break the coupling,
  or a rate-term reformulation). A constant-scale variable split is **not** enough
  (it is diagonal, already in the 1.77e9). Must be general (Dev-Philosophy #2 — no
  hda-keyed special case) and sound, which is a substantial relaxation-layer
  research effort with no guarantee it drops below ~1e7.
- **Higher-precision LP arithmetic** (f128 / rational residuals in the factorization)
  — a different engine capability.

**Disposition:** candidate (A)'s sound loose floor is the practical answer for hda.
Candidate (B)'s tight bound requires L3-structural or higher-precision arithmetic —
tracked here as the honest scope; not attempted speculatively, since the entry
experiment shows no scaling/factorization path exists.

## CORRECTION (the "~1e7 ceiling" reasoning above was wrong)

The "feral reliable to ~1e7" figure is `scaling.rs`'s empirical **coefficient
dynamic-range** heuristic (`SCALE_TRIGGER`-adjacent), **not** a float64 limit —
float64 LU with partial pivoting is accurate to condition numbers ~1e12–1e14. And
the "1.77e9" repeatedly cited above is a **coefficient range**, not the basis
**condition number** that governs factorization. Comparing them was an error.

The real measurement (SVD of the captured hda node matrix, raw and optimally
scaled):

| | σ_max | σ_min (nz) | numerical rank (σ<σmax·1e-14) | #σ < σmax·1e-7 |
|---|---|---|---|---|
| raw A (2974×1138) | 8.91e10 | 4.47e-18 | 1123 / 1138 | — |
| optimally-scaled A | 3.55 | 9.4e-26 | **1095 / 1138** | **409** |

hda's relaxation matrix is **rank-deficient by ~15–43** (genuine near-linear
dependence among columns) with a true SVD condition number **~1e14+** — and this is
**scale-invariant**, which is why L1 cut the coefficient spread to 4e9 yet hda still
failed. feral's FT `Growth` is it choking on **near-singular bases** (tiny pivots),
exactly what a rank-deficient matrix produces.

**Cross-check — every float64 route fails, presolve included (measured):**

| approach on hda's captured node LP | result |
|---|---|
| HiGHS simplex, unscaled | INFEASIBLE (false) |
| HiGHS simplex, optimally scaled (coef range 1.77e9) | `numerical` |
| HiGHS simplex, **rank-reduced** to 1063 full-rank rows + scaled | `numerical` |
| HiGHS **interior-point**, scaled | `numerical` |

So the presolve / rank-reduction hypothesis is **falsified**: dropping the
redundant rows does not make HiGHS solve it. And it is **not** a feral-specific
weakness — HiGHS's simplex *and* IPM both fail, with scaling and presolve. hda's
McCormick relaxation LP is **genuinely intractable at float64 precision**: the
rank-deficiency (~43) plus the cluster of ~400 tiny singular values makes the
KKT/basis systems near-singular for *any* float64 factorization.

**Final disposition (verified).** The tight dual bound for hda is **not obtainable
at float64 precision** with this relaxation, by any engine/scaling/presolve. The
only genuine fixes are:

1. **Higher-precision LP arithmetic** (f128 / rational residual refinement in the
   factorization) — now the *primary* lever, not optional.
2. **A reformulated, well-conditioned relaxation** of the hda-class flowsheet
   (change the constraint structure so the McCormick LP is not near-singular) — a
   relaxation-layer research problem.

Candidate (A)'s sound loose floor (merged, #662) stands as the practical answer:
it is the best a float64 engine can certify on this relaxation. My earlier
"~1e7 ceiling" framing was wrong (it is feral's coefficient-range heuristic, not a
float64 limit); the *verified* obstacle is genuine near-singularity, confirmed by
HiGHS failing identically.

Acceptance / regime unchanged from #664: bound-neutral verification
(`node_count` + certified `objective` exactly unchanged on the certifying panel,
`cargo test -p discopt-core`), and a **tight** hda bound materially closer to opt
`−5964.534084` than (A)'s `−1.80e10`.
