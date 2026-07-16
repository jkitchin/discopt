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

## Fix hypothesis (to test next)

**Drop / fold the sub-tolerance coefficients before scaling.** They are provably
near-zero (cancellation residue) or fold soundly into the RHS via their interval
worst case (the candidate-A "fold", here repurposed as a *conditioning* fix). With
them gone, the equilibration can bring the residual to ~O(1), feral factorizes
cleanly, phase-2 converges, and hda gets a **tight** bound.

Kill criterion: if removing sub-tolerance coefficients does **not** collapse the
scaled residual range (or if a clean residual still leaves phase-1 grinding on
degeneracy), scaling is not the whole story and the degeneracy (redundant envelope
rows, rank ~1136 of 2974) is the next target.

Acceptance / regime unchanged from #664: bound-neutral verification
(`node_count` + certified `objective` exactly unchanged on the certifying panel,
`cargo test -p discopt-core`), and a **tight** hda bound materially closer to opt
`−5964.534084` than (A)'s `−1.80e10`.
