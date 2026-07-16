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

## Real levers (harder; not yet built)

- **L1 — scale `A_ub` pre-slack:** equilibrate the structural matrix before slacks
  are appended (append slacks with coefficient = row scale, not 1), so feral sees
  ~2.5e9 instead of ~2e20. Necessary but likely not sufficient alone.
- **L2 — robust factorization for ~2.5e9:** pair L1 with the dense LU route
  (`#557`) and/or a higher feral pivot threshold, since ~2.5e9 is past the sparse
  route's reliable range.
- **L3 — model/relaxation-level rescale of the Arrhenius term:** normalize the
  `6.3e10 · exp(-E/RT)` product (the aux value is ~1e-13, so the product is ~6e-3)
  at relaxation build, removing the 6.3e10 from the matrix entirely. The most
  fundamental fix; largest surface.

Kill note: candidate B is **not** a scaling-parameter tweak. It needs L1+L2
(and possibly L3) together, each a real core-engine/relaxation change under the
bound-neutral regime. Candidate A's loose floor remains the shipped fallback until
one lands.

Acceptance / regime unchanged from #664: bound-neutral verification
(`node_count` + certified `objective` exactly unchanged on the certifying panel,
`cargo test -p discopt-core`), and a **tight** hda bound materially closer to opt
`−5964.534084` than (A)'s `−1.80e10`.
