# G5 — family-D relaxation-strength diagnoses (baron-gap-plan §7 / TX6)

Status: **BOTH DIAGNOSES DONE (2026-07-15).** Diagnosis-first, no build (baron-gap
§0.3: any build that follows gets the full bound-changing gate). Written in the
`performance-plan.md` §6 / `pf4-rootgap-spike.md` §6 falsification house style
(hypothesis → the measurement that settled it → verdict vs kill criterion →
scoped follow-up, if any).

**Environment (both diagnoses, load-immune — every number is a bound value, not a
wall time).** Isolated worktree of `discopt` @ `origin/main` `0f3ebd7d`; Rust ext
`_rust.cpython-312-darwin.so` copied from the shared build (same commit);
`JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=$PWD/python`. Instances from the
in-repo corpus `python/tests/data/minlplib_nl/` (`bchoco06.nl`,
`heatexch_gen1.nl`). Probes committed under `discopt_benchmarks/scripts/`:
`g5_bchoco06_hole_probe.py`, `g5_heatexch_pole_children_probe.py`. The root
relaxation is discopt's own uniform factorable engine
(`build_uniform_relaxation`); the ground-truth LP status is cross-checked with
HiGHS (scipy `linprog`, all three methods) on the byte-identical assembled matrix.

---

## Diagnosis 1 — bchoco06 "unbounded-relaxation hole"

**Hypothesis (baron-gap §7 task 1).** bchoco06 reports **no finite dual bound at
7 nodes** (EP0 record: `avm-canonicalization-plan.md`; EP3 classes it
"unbounded-root"). The hypothesis to test: some atom class of some constraint
leaves the **root LP genuinely unbounded** — a structural hole à la the heatexch
LMTD ε-pole (PF4), for which no sound finite bound exists. **Kill criterion:** the
hole is structural (no sound finite bound exists) → document and close.

**The measurement that settled it.** Rebuild the uniform root relaxation with
`track_aux_exprs=True` (probe mirrors `build_uniform_relaxation`'s own assembly),
then solve the byte-identical LP three ways:

| solver | status | dual bound (max x0) |
|---|---|---|
| discopt in-house Rust simplex (`backend="simplex"`, the default node engine) | `iteration_limit` | **None** |
| discopt `backend="auto"` (highspy present) | `iteration_limit` | **None** |
| HiGHS via scipy `linprog` — `highs-ds` / `highs-ipm` / `highs` | **optimal** (all 3) | **0.99998** |

The root LP is **342 rows-ish** (833 `≤` rows × 349 cols; 118 originals + 231
aux). It is **not unbounded**: independent HiGHS solves it to a finite optimum
≈ 1.0, and `objective_bound_valid` is already **True** (the objective is the bare
variable `x0`, box `[0.95, 1]`, so the `uniform_relax.py` free-cost-column refusal
never fires). **A sound finite bound provably exists** — discopt's shipped LP
layer simply fails to compute it. The failure is **conditioning, not a free ray**:
clamping all 96 free/semi-infinite aux columns to a symmetric big-M box (M up to
1e6) leaves the in-house simplex at `iteration_limit` still. The constraint matrix
has an effectively infinite nonzero-coefficient spread (max ≈ 1e10, min ≈ 5e-324
subnormal), from McCormick rows over products/powers of the **14 unbounded-above
originals** (`x8..x14`, `x64..x70`, all `[0, +∞)`).

Second measurement — **even the correct bound is useless.** The dual bound ≈ 1.0
is exactly the a-priori box top of the single objective variable `x0 ∈ [0.95,1]`.
Tightening the 14 unbounded originals to a finite box does **not** move it:

| original upper bound | HiGHS dual bound (max x0) |
|---|---|
| +∞ (raw) | 0.99998 |
| ≤ 100 | 0.99998 |
| ≤ 10 | 0.99998 |
| ≤ 1000 / ≤ 1e4 | HiGHS itself returns **false-infeasible** (numerical, the coefficient spread) |

`x0` is coupled to the polynomial body through **28 relaxed rows**, but that body
is a sum of high-degree monomials — bilinear through **degree-5 multilinear**
(e.g. `((((3·x8)·x11)·x13)·x14)·x9`) plus odd powers (`x9³`, `x11³`, `x13³`) — and
the recursive-McCormick envelope of those products over the wide box is too loose
to bind `x0` below its box top even when the inputs are bounded to `[0,10]`. The
MINLPLib oracle is **`known=null`** for bchoco06/07/08 (`v-baron-remeasure`
§; no primal oracle), so no relative gap can be quantified.

**Verdict — kill criterion NOT met; the hypothesis is FALSIFIED.** The root LP is
**not** structurally unbounded and is **not** a hole à la LMTD: a sound finite
dual bound exists (HiGHS, 3 methods agree, ≈ 1.0). "No finite dual bound at 7
nodes" has two distinct, **non-structural** causes:

- **Cause A — LP-backend non-convergence.** discopt's shipped relaxation solve
  (both `simplex` and `auto`) returns `iteration_limit`/`None` on this
  ill-conditioned LP, so the solver has no bound to report. This is a solver-
  robustness failure (issue #15-adjacent), not a relaxation property — it
  suppresses even the trivial-but-sound bound that HiGHS computes.
- **Cause B — vacuous relaxation on the objective variable.** Even solved
  exactly, the bound is the box top of the lone objective variable `x0`; the
  high-degree-monomial relaxation over the (unbounded-above) originals never binds
  it. This is a genuine bound-strength weakness, but it is *masked* by Cause A and,
  with `known=null`, cannot even be measured until Cause A is fixed.

**Scoped follow-up (NOT built; each gets the full bound-changing gate per §0.3).**

1. *Cause A (prerequisite).* A concrete, testable LP-robustness bug: the shipped
   backend reports no bound where HiGHS on the identical matrix reports `optimal`.
   *Entry experiment (already reproducible):* `g5_bchoco06_hole_probe.py`
   demonstrates the discrepancy. *Direction:* on `iteration_limit` (or when the
   nonzero-coefficient spread exceeds a threshold) route the relaxation LP through
   a cleaned/presolved HiGHS solve, or drop structural near-zeros (|coef| ≲ 1e-300)
   from the row matrix before the in-house simplex. *Gate:* bound-changing (it
   introduces a previously-absent bound) → report a bound only where an independent
   HiGHS solve agrees; `bound ≤ incumbent` on every node; feasible-point sampler
   0 cuts. No oracle here, so cross-solver agreement is the soundness check.
2. *Cause B (blocked on A).* OBBT to derive finite bounds on the 14 unbounded
   originals (FBBT alone does not — PF1 recorded bchoco stuck ≤ 3 nodes, "upstream
   of range reduction") plus a stronger high-degree-monomial relaxation
   (RLT / recursive-McCormick tightening). Unmeasurable until Cause A reports a
   bound and while `known=null`; lowest priority.

---

## Diagnosis 2 — heatexch_gen1 pole-excluded sub-boxes

**Hypothesis (baron-gap §7 task 2).** PF4 §3/§6 found the LMTD atoms
`w = (a−b)/log(a/(ε+b))` (ε=1e-6, a,b ∈ [10,+∞)) have a pole on `a = ε+b` **inside**
the raw box, so the `GM ≤ LMTD ≤ AM` envelope is unsound over the whole box. The
only sound route PF4 §6 left open: relax over **pole-EXCLUDED sub-boxes**. Test it
— branch once by hand on the `a=ε+b` pole line, build the two children, and measure
their root bounds with the AM over-estimator `w ≤ (a+b)/2` (the decisive
direction: area cost ∝ 1/LMTD is minimised by driving `w` large, so an *upper* cut
on `w` should raise the dual bound). **Kill criterion:** children improve gen1's
**38,184** root bound by **< 10%**.

**Setup (measured).** gen1 has **8 LMTD terms**, chained temperature pairs
`(x20,x21),(x21,x22),(x23,x24),(x24,x25),(x26,x27),(x27,x28),(x29,x30),(x30,x31)`,
every input `[10,+∞)`; each lifts an output aux `w` with box `[-∞,+∞]`, feeding the
objective *indirectly* (`w ∉ objective` directly — it enters via the downstream
area = Q/(U·w) ratio). Baseline uniform root bound = **38,183.53** (matches PF4
§1's 38,184; HiGHS agrees to 38,183.53; the in-house simplex **converges** here,
unlike bchoco06). Oracle: `=bestdual= 100552.19`, `=best= 154895.93` — the baseline
sits 62% below the dual bound.

**AM soundness on the pole-excluded children (feasible-point sampled, 4×10⁵ pts /
term, a,b ∈ [10,700]).** A single clean split at `a=ε+b` still leaves each child's
closure touching the pole, so AM needs a *margin* δ (PF4 §6's point). Measured
worst AM violation `LMTD − (a+b)/2` (>0 = unsound cut):

| δ (margin) | child `a ≥ b+δ` | child `a ≤ b−δ` |
|---|---|---|
| 1e-3 | **+0.696** (unsound) | −1.8e-4 (sound) |
| 0.1 | +0.007 (unsound) | −1.8e-4 (sound) |
| 1.0 | +5.8e-4 (unsound) | −1.8e-4 (sound) |
| **5.0** | **−0.0028 (sound)** | −1.8e-4 (sound) |

The `a<b` child is sound at any δ (the pole is approached from the `a>b` side, where
the denominator → 0⁺); the `a>b` child needs δ ≈ 5 to be sound — corroborating PF4
§6 that the AM margin is quantitative and non-trivial.

**The measurement that settled it.** With a sound δ=5, add the AM cut and re-solve
the children's root LP (HiGHS on the assembled matrix):

| configuration | AM sound? | root bound | improvement vs 38,184 |
|---|---|---|---|
| baseline (raw box, no AM) | — | 38,183.53 | — |
| **Exp A:** branch once on term-0 pole, child `a≥b+5`, AM on term-0 | yes | 38,183.53 | **0.00%** |
| **Exp A:** child `a≤b−5`, AM on term-0 | yes | 38,183.53 | **0.00%** |
| **Exp B:** pole-excluded ORTHANT `a_k ≥ b_k+5 ∀k`, AM on **all 8** terms | yes (worst −0.003) | 38,183.53 | **0.00%** |
| control: orthant branch only, **no AM** | — | 38,183.53 | 0.00% |

**Why (mechanism, measured at the LP optimum).** At the baseline LP optimum **all
8 temperature pairs park at the floor `a=b=10`**, the LMTD aux **`w = 0`**, and the
AM cut `w ≤ (a+b)/2 = 10` is therefore **fully slack (slack = 10)**. The LMTD
*upper* envelope is not the binding looseness. The relaxation reaches 38,184 by
setting every approach to zero (LMTD → 0), and the dual bound is capped by the
**downstream area = Q/(U·w) ratio** relaxation near `w → 0` (loose reciprocal),
*not* by the value of `w` from above. Bounding `w` above (AM) cannot move a bound
that is already achieved at `w = 0`. This holds in the best case — all 8 poles
excluded, AM sound on every term.

**Verdict — kill criterion MET (0.00% ≪ 10%).** Pole-excluded sub-box branching +
the AM/GM envelope does **not** tighten gen1's root bound, even in its strongest
form (full pole-excluded orthant, AM sound on all 8 LMTD terms). This **confirms
and extends PF4 §6** from a new angle: the LMTD envelope is not merely unsound on
the raw box — where it is *made* sound (pole excluded), it is **inert at the root**,
because the LMTD *upper* bound is not the constraint that caps the dual. Close the
"pole-excluded LMTD envelope" as a **non-lever** for gen1's root gap.

**Scoped follow-up (a different lever, NOT built).** The measured binding looseness
is the **downstream area = Q/(U·LMTD) reciprocal** near LMTD→0, together with the
[10,+∞) temperature domains that let the relaxation drive every approach to zero.
Any future family-D root-strength work should target (a) OBBT/FBBT tightening of the
temperature bounds so LMTD cannot collapse to 0, and (b) the reciprocal/ratio
envelope on area, *not* the LMTD AM/GM envelope — with the full bound-changing gate
(differential per-box bound never lower/crossed + feasible-point 0-cuts). This is a
distinct, unmeasured direction, explicitly out of this diagnosis's scope.

---

## Summary

| diagnosis | kill criterion | verdict | key numbers |
|---|---|---|---|
| bchoco06 hole | structural à la LMTD (no sound finite bound) → close | **NOT met — FALSIFIED**: a sound finite bound exists (HiGHS 3-way ≈ 1.0); "no bound" = in-house LP non-convergence (Cause A) over a vacuous high-degree-monomial relaxation (Cause B); `known=null` | discopt LP: `iteration_limit`/None; HiGHS: optimal 0.99998; coeff spread 1e10..5e-324; 14 unbounded-above originals |
| heatexch pole children | children improve 38,184 by < 10% | **MET — 0.00%** | baseline 38,183.53; AM sound at δ=5; Exp A & B both 38,183.53; at optimum a=b=10, w=0, AM slack=10 |

Neither diagnosis authorises a build. bchoco06's actionable item is an
LP-conditioning/solver-robustness fix (Cause A), not a relaxation envelope;
heatexch's LMTD AM/GM envelope is closed as a non-lever even on pole-excluded
children. Both follow-ups, if taken, ship only through the CLAUDE.md §5
bound-changing gate.

---

## G6 update — subnormal-flush hypothesis FALSIFIED (2026-07-15)

Follow-up on Diagnosis 1 / Cause A (baron-gap-plan.md §10 "G6"). The G6 entry
experiment tested a specific, cheap hypothesis for the bchoco06 LP
non-convergence before committing to a Rust-layer numerics fix. It is falsified;
recording per the performance-plan.md §6 house style (hypothesis → the
measurement that settled it → verdict vs kill criterion → scoped follow-up).

**Environment.** Worktree of `discopt` @ `b7ed54c` (post-#647, on `main`'s line);
`JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`; `bchoco06.nl` from the in-repo corpus. Root
uniform relaxation assembled exactly as `g5_bchoco06_hole_probe.py`, byte-identical
matrix solved by the shipped in-house Rust simplex (`backend="simplex"`, which
already retries via `_solve_lp_warm_equilibrated` / geometric-mean Ruiz
equilibration) and cross-checked with HiGHS (scipy `linprog`).

**Hypothesis (G6).** The `-4.941e-324` subnormal lower bounds on `x^2`/`x*y` aux
columns (interval-arithmetic underflow of a true `0`) plus any subnormal matrix
coefficients are the conditioning poison. Flushing structural noise `|v| < 1e-300`
to `0` in `(A_ub, b_ub, bounds)` before the solve lets the existing equilibration
compress the *real* 1e10 range, and the in-house simplex then converges to the
HiGHS bound (≈ 1.0). **Kill criterion:** if the flush does NOT make the simplex
converge (the 1e10 side alone still stalls it), STOP — it is a deeper Rust
linear-algebra/refactorization problem, not a cleanup pass.

**The measurement that settled it.** bchoco06 root LP, one stage at a time:

| stage of the matrix | A nonzero spread (max / min) | in-house simplex | HiGHS |
|---|---|---|---|
| raw (as shipped) | 1.0e10 / 4.9e-324 = **inf** | `iteration_limit`, None | optimal, **0.99998** |
| subnormals flushed (`|v|<1e-300 → 0`) | 1.0e10 / 1.0e-10 = **1e20** | **`iteration_limit`, None** | optimal, 0.99998 |
| flushed + Ruiz equilibrated (20/50/100 sweeps) | 1.95 / 1.56e-12 = **1.25e12** | **`iteration_limit`, None** | optimal, ~1.0 |

- Subnormal count: **20** nonzeros in `A.data`, **29** in `col_lb`, 0 in `b`/`col_ub`.
  Flushing *all* of them to 0 leaves the simplex at `iteration_limit`, bound None.
- The subnormals are **not** the poison. After the flush the smallest nonzero is a
  *genuine* McCormick coefficient `1e-10` (not a subnormal), and the real spread is
  `1e20`. Geometric-mean (Ruiz, power-of-two) equilibration compresses this only to
  a **residual `1.25e12`** and cannot go lower — 50 and 100 sweeps give the identical
  `1.25e12`. That residual is the intrinsic ill-conditioning of the recursive-
  McCormick envelope over degree-5 multilinear products on the wide/unbounded box,
  which a diagonal rescaling cannot remove.
- The raw Rust warm simplex returns `None` (iter-limit at its 100k-pivot cap / stall
  guard, `crates/discopt-core/src/lp/simplex/`) on the equilibrated matrix, flushed
  or not; the Neumaier–Shcherbina `safe_bound` side channel is also `None` — no
  salvageable bound. HiGHS solves the identical raw / flushed / equilibrated matrix
  to optimal ≈ 1.0.

**Verdict — kill criterion MET; hypothesis FALSIFIED.** Flushing subnormals does
not make the in-house simplex converge. The `-4.941e-324` bounds are a real
interval-underflow artifact, but they are **not** the conditioning poison — the
poison is the genuine `≥1e12` ill-conditioning that *survives* full geometric-mean
equilibration, a matrix HiGHS's dual simplex handles (LU refactorization + Harris
ratio test + pivoting tolerances) and the in-house Rust simplex does not. This is a
**Rust linear-algebra / refactorization robustness defect** in
`crates/discopt-core/src/lp/simplex/`, exactly the escalation branch baron-gap §10
G6 named ("scaling does not recover a finite bound → escalate to a linear-algebra
(refactorization/tolerance) fix with its own plan"). No band-aid shipped: a subnormal
flush is (a) a proven no-op for convergence on the target class and (b) would add a
threshold with no measured beneficiary — rejected per CLAUDE.md §3. Tracked in
**issue #649** for a Rust-layer numerics fix (refactorization cadence / Markowitz
pivoting tolerances / Harris ratio test), which will carry the full bound-changing
gate + `cargo test -p discopt-core`. Reproduce with
`discopt_benchmarks/scripts/g6_bchoco06_conditioning_probe.py`.

## G6 resolution — the defect was counterproductive re-scaling, not the simplex core (2026-07-15, #649)

The issue-#649 fix work overturned its own hypothesis (the measurement wins). The
Rust simplex **core** is not the problem: fed bchoco06's equilibrated root LP
directly, `solve_lp_cols` (the cold primal, **no** Rust scaling) returns
**Optimal, obj −1.0, 0 iters** — clean, matching HiGHS. The `iteration_limit`/`None`
comes from the **warm/binding path** the shipped `backend="simplex"` uses:
`dual::solve_lp_warm_csc` applies its **own** pow2 `Scaling::from_sparse` on top of
the already-Ruiz-equilibrated matrix, and on this class that re-scaling is
counterproductive — the scaled optimum, unscaled, lands just past the feasibility
audit and the solve returns **`Numerical`** (0 iters, obj already ≈ −1.0 but
refused). Isolation (Rust diagnostic on the captured std-form fixture):

| path on the identical bchoco06 std-form LP | result |
|---|---|
| `solve_lp_cols` (no Rust scaling) | **Optimal**, obj −0.99999 |
| `Scaling::from_sparse` | re-scales range 1.25e12 → 3.33e8 (looks *better*) … |
| `solve_lp_warm_csc` (scaling ON), pre-fix | **`Numerical`**, no bound |
| `solve_lp_warm_csc` (scaling ON), post-fix | **Optimal**, obj −0.99999 |

**Fix (shipped, #649).** Neither scaling nor no-scaling dominates (scaling is
*needed* on the issue-#170 degenerate LPs), so keep scaling-first and, **only when
the scaled solve yields `Numerical`/`IterLimit`, retry on the exactly-reconstructed
unscaled matrix** (`Scaling::unscale_cols`, exact powers of two, zero allocation on
the success path) and prefer that resolved verdict. Sound: the unscaled cold solve
carries its own feasibility audit, so a returned `Optimal`/`Infeasible`/`Unbounded`
is genuine — a scaled point the audit rejected is never accepted, only a bound the
scaling lost is recovered. Plus a Python `isfinite` guard in `milp_relaxation.py`
so a non-finite objective/bound (seen on the raw un-presolved probe path once the
solve started returning `optimal`) is reported as no-bound, never a NaN that would
silently pass `bound ≤ incumbent`.

**Result.** bchoco06 `Model.solve()` now reports a finite dual **bound ≈ 0.99999**
(was none at 7 nodes — the original EP0 symptom), agreeing with HiGHS. Gate: 446
`discopt-core` tests (byte-neutral on all existing solves + the new regression
`bchoco06_illcond_scaled_path_recovers_bound_649`), smoke 638, adversarial 10,
clean-instance optima (alan/ex1221/ex1225/nvs09/st_miqp2) unchanged. **Cause B**
(vacuous high-degree-monomial relaxation — the bound ≈ 1.0 is `x0`'s box top) is now
*measurable* and remains open/lowest-priority (still `known=null`).
