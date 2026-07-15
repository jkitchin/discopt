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
