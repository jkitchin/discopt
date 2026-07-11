# P1-A2 (task #98) — joint Gibbs/log-sum OA relaxation for ex6_2 — ENTRY EXPERIMENT

**Date:** 2026-07-11 · **Base:** `origin/main` @ `c2493bad` (centropy RELAX-1 #597
default-ON) · **Verdict: KILL** (premise falsified — the objective is nonconvex).

## 0. TL;DR

The lever proposed by task #98 was: the MINLPLib `ex6_2_*` Gibbs/log-sum objective
is a **sum of convex** centropy / `x·log` atoms, so the *whole objective is convex*;
discopt loses ~300× of root bound by summing loose per-atom tangent underestimators
instead of taking **joint outer-approximation (OA) cuts** (gradient of the full
objective at the LP point, Kelley rounds), which for a convex function are rigorous
underestimators.

**The premise is false.** The ex6_2 objective is **not convex** on the box — the
Gibbs free-energy form carries Wilson activity-coefficient terms of the shape
`−x_i·log(a·x + b·y + …)` (24 of them in ex6_2_5, the `neg(...)` nodes), and
`−x·log(affine)` is nonconvex at **100 %** of box points in isolation. Over the whole
box, a negative Hessian eigenvalue appears at **78 % (ex6_2_5) / 92 % (ex6_2_9) /
99.5 % (ex6_2_10)** of random points. Gradient/OA cuts of a nonconvex function are
**not valid underestimators** — the joint-OA lever is therefore *unsound*, not merely
loose, and cannot be built.

The sound joint alternative — αBB over the whole objective — is **catastrophically
worse**, not tighter: the rigorous α (interval-Gershgorin) is ~1e40–1e49 because the
`x·log(x)` / `log(x)` terms have Hessian entries `~1/x` that blow up at the `x→1e-7`
box edge, driving the joint-αBB box-minimum to ~1e40–1e52 below zero (vs the per-atom
root bound of −89 397). Per-atom relaxation is used *precisely because* it is the only
tractable sound handling of this singular structure; a joint convex underestimator of
the whole objective is dominated by the worst single-variable curvature and loses.

**Where the residual looseness actually is:** 100 % in the **objective**. The
constraints are all **linear equality mass-balances** (`x0+x1+x2 = 40.3071`, …) — zero
nonlinear content. But the objective's looseness is *not* the per-atom-vs-joint split
the task hypothesized; it is the intrinsic difficulty of underestimating a **nonconvex,
`1/x`-singular** `x·log` sum, which no convex OA/hull addresses. The real lever is a
tighter **per-atom** convex hull for `x·log(affine)` / bilinear-of-log (spatial B&B on
those terms), not a joint objective cut.

## 1. Method

- Instances: ex6_2_5 (9 vars, 3 cons), ex6_2_9 (4, 2), ex6_2_10 (6, 3) from
  `~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/`, oracle `minlplib.solu`.
- Root bounds: `Model.solve(time_limit=5)`, read `result.root_bound` / `root_gap`.
- Convexity: JAX Hessian of the compiled objective sampled at 1200–1500 random box
  points; count points with `λ_min(∇²f) < −1e-9`.
- Objective's own box-minimum (constraints ignored): 40–60-start L-BFGS-B on the
  compiled objective + gradient over the box → the tightest a *perfect* objective
  underestimator could reach.
- Joint αBB: `rigorous_alpha(obj, model)` (the solver's own interval-Gershgorin α)
  → minimize `f(x) − Σ α_i (x−lb_i)(ub_i−x)` over the box (the tightest *sound convex*
  joint underestimator).
- Harnesses: `scratchpad/root_probe.py`, `box_min.py`, `joint_alphabb.py`.

## 2. Results

| instance | oracle | root_bound (per-atom, main) | root_gap | obj box-min (constraints off) | joint-αBB box-min | α_max |
|---|---:|---:|---:|---:|---:|---:|
| ex6_2_5  | −70.75  | **−89 397**  | 1263× | −207.6  | −4.9e52 | 4.2e49 |
| ex6_2_9  | −0.0341 | **−332.1**   |  332× | −0.068  | −8.8e40 | 5.4e41 |
| ex6_2_10 | −3.052  | **−257.0**   |   83× | −5.321  | −8.0e39 | 6.0e40 |

Convexity (fraction of box points with a negative Hessian eigenvalue; min eig):
ex6_2_5 **0.78** (−11.3), ex6_2_9 **0.92** (−181), ex6_2_10 **0.995** (−76).
Isolated Wilson term `−x0·log(3.9235·x0 + 6.0909·x1)`: nonconvex at **100 %** of box
points. ex6_2_5 objective has **no `centropy` token** — it is raw `log(x/Σ)·x`,
`x·log(x)`, and `neg(x·log(affine))` (49 `log(`, 24 `neg(`); RELAX-1's centropy atom
never even matches this `.nl` form).

## 3. Verdict — KILL (two independent reasons)

1. **Joint-OA is unsound.** The objective is nonconvex (78–99.5 % of box), so a
   supporting-hyperplane/gradient cut of the whole objective is not a valid lower
   bound. Building it would violate the certificate invariant (CLAUDE.md §1). It is
   not a tightness question — it cannot be built soundly.
2. **The sound joint alternative loses by 40+ orders of magnitude.** Joint αBB's α is
   ~1e40 from the `1/x` Hessian singularity at the box edge; its box-min is ~1e40–1e52
   below the per-atom −89 397. Per-atom relaxation is *better*, not worse, here.

Kill criterion (task #98): "if the joint hull does NOT materially reduce the ex6_2
root_gap (<2×), the looseness is elsewhere / KILL." Met with margin: the joint
approach is either unsound (OA) or ~1e40× *worse* (αBB).

## 4. Where the looseness actually is (for the follow-on)

- **Objective, not constraints** (constraints are linear mass-balances — 0 nonlinear).
- **Not** per-atom-vs-joint. It is the intrinsic difficulty of underestimating the
  **nonconvex, `1/x`-singular** `x·log(affine)` (bilinear-of-log) terms. The lever is a
  **tighter per-atom spatial relaxation** of `x·log(affine)` / `log(x)·x` — e.g. a
  tighter convex/concave envelope for the composite `x·log(g(x))` where `g` is affine,
  branched spatially — not a joint objective cut. This is Lever A (relaxation
  strength) on the specific `x·log` composite, consistent with DECOMP-1 §5(ii) which
  already flagged "#597's tangent planes are demonstrably not yet enough" for this
  family. #597 (centropy tangent) is a *univariate-convex* handling; the binding term
  is the *nonconvex* bilinear-of-log, a different and harder object.
- No code shipped (a joint-OA cut would be unsound; a joint-αBB flag would be a
  strictly-worse dead flag). No default changed.

## 5. Reproduction

```bash
D=discopt_benchmarks/scripts/p1a2_gibbs_entry
for i in ex6_2_5 ex6_2_9 ex6_2_10; do python $D/root_probe.py   $i 5; done   # root bounds
for i in ex6_2_5 ex6_2_9 ex6_2_10; do python $D/box_min.py      $i;   done   # obj box-min
for i in ex6_2_5 ex6_2_9 ex6_2_10; do python $D/joint_alphabb.py $i;   done   # joint αBB
```

macOS arm64, sequential, 2026-07-11, base `c2493bad`.
