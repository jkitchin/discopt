# LR-nvs05 — division/ratio-lever entry experiment (GO/KILL)

**Task:** the re-scoped nvs05 follow-up from `docs/dev/lever-a-root-tightness-plan.md`
§7 (F16 re-scope item 2) and `docs/dev/lr0-logspace-entry-2026-07-11.md` §Re-scope.
Gates whether a *division/ratio relaxation + OBBT* lever certifies nvs05 at (or
near) the root. **Date:** 2026-07-11. **Regime:** entry experiment, probe only —
no `python/discopt/` solver code touched. **Env:** `JAX_PLATFORMS=cpu
JAX_ENABLE_X64=1`, `PYTHONPATH=<worktree>/python:<worktree>/docs/dev/lr0_probe`,
prebuilt `_rust.so` (crates hash `f919f41e`).

## Verdict (headline): **KILL**

A tighter treatment of nvs05's division/ratio constraints — *including* keeping
C2/C6 (which the LR-0 probe dropped), a purpose-built reciprocal envelope for
every `a/(x·y)`, an algebraically-simplified C6, and OBBT — moves the **root** LP
bound from discopt's 0.674 only to **0.78** (best sound variant, 32 tangents per
envelope). That closes **~2 %** of the root gap; **>50 % of the root gap remains
on the only target of this experiment**, which is the plan's explicit KILL
criterion (§4 LR-0 / §1.1). No sound variant reaches a root certificate, and the
gap is **not** closable by a ≤10-node probe B&B (a 10-node probe reaches ~0.95).

**OBBT does not bite:** after up to 10 rounds of optimality-based bound tightening
against the (sound) relaxation, the ratio-feeding boxes **x0, x1 ∈ [0.01, 200]**
and **x3 ∈ [1, 200]** are **byte-unchanged** — the relaxation is too loose to
prove any primal variable is bounded away from its box edge, so OBBT tightens
nothing on x0..x3 (it only trims the aux vars x4..x7 slightly).

**Were nvs05's ratio boxes actually wide?** **Yes** — this is the one clean
positive finding. FBBT leaves x0, x1 at the full `[0.01, 200]` and x3 at
`[1, 200]`; only x2 tightens (`[1,200] → [1,35]`). So the "boxes are wide"
premise is confirmed. But — contra the OBBT hypothesis — **shrinking them is the
hard part, not a matter of running OBBT**: OBBT over a loose relaxation cannot
shrink a wide box; you need a tighter relaxation *first*, which is exactly what
we could not build soundly at the root.

nvs05 is **research-grade** for a root/near-root certificate: it needs genuine
spatial branch-and-bound on the primal variables (thousands of nodes here), not
a root-relaxation lever. Recorded as falsification **F17** below.

## What was built (probe: `docs/dev/lr0_probe/nvs05_division.py`)

A standalone, from-scratch **sound** relaxation of the full nvs05 welded-beam
model, transcribed directly from the validated `.nl` parse
(`nl_parse.py`, cross-checked to the Rust `_nl_repr` oracle to machine precision
in LR-0). Every row is a **proven** over/under-estimator; the LP optimum is a
valid lower bound (checked `≤ opt 5.4709` and against feasible points). Key
differences from the LR-0 general probe — all aimed at the division/ratio lever:

- **C2 and C6 are KEPT, not dropped.** The LR-0 probe dropped both for
  "explosive lifted magnitude > 1e10". Root causes, both fixable:
  - **C6**'s `1e15` lives *inside a sqrt*:
    `0.0204745·sqrt(1e15·x3³·x2²·x3³) = 647460.54·x2·x3³·(1−0.0282346·x2)` for
    x2,x3 > 0 — a **modest signomial**, handled here in log space as two positive
    monomials `x2·x3³` and `x2²·x3³`. Dropping it was avoidable.
  - **C2/C4**'s magnitudes are *genuine* (x5 ≈ 12 400 at the optimum, near the
    13 600 shear limit), not artifacts. Kept them numerically sane by clamping
    the ratio outputs x4, x5 to their (tighter) FBBT interval enclosure — sound,
    a valid interval never removes a feasible point — and by scaling the C4 row
    by 1/13600².
- **Every `q = a/(x·y…)` gets an exact 1-D reciprocal envelope**: build the
  denominator product `p` by McCormick, then `q = a/p` via the exact convex
  reciprocal (tangent under-estimators + secant over-estimator) over
  `p ∈ [p_lo, p_hi]`, p > 0. This is the "tighter ratio envelope" the re-scope
  asked for, applied to C0 (x4), C2 (x5), C3 (x7).
- **Log-space (H-LOG) for the objective monomials and the positive-signomial
  constraints C5/C6/C7** (all factors strictly positive on the FBBT box).
- **OBBT** driver (`obbt_round`): min/max each base var over the sound
  relaxation, intersect with the box, iterate. Runs both against a
  numerically-safe subset (C5/C6/C7/C8 + objective) and against the full model.

## Measured results

Optimum 5.4709; discopt root bound (plan §1) 0.674; root-certificate tolerance
`1e-4·(1+|opt|) = 6.5e-4`. `%root-gap-closed = (bound − 0.674)/(5.4709 − 0.674)`.

| variant | root LP bound | gap-to-opt | %root-gap-closed | root-cert? |
|---|---|---|---|---|
| discopt (plan baseline) | 0.674 | 4.797 | 0 % | no |
| **V0** full relaxation, C2/C6 KEPT, ratio envelopes, no OBBT (n_tan=6) | **0.725** | 4.746 | **1.1 %** | no |
| V0 with n_tan=16 | 0.741 | 4.730 | 1.4 % | no |
| **V0 with n_tan=32** (densest) | **0.777** | 4.694 | **2.1 %** | no |
| **V1** = V0 after 6 OBBT rounds (safe subset, x0..x3) | 0.725 | 4.746 | 1.1 % | no |
| **V2** = V0 after 4 full-model OBBT rounds (warm-started) | 0.725 | 4.746 | 1.1 % | no |

**OBBT box, before vs after (10 full-model rounds, n_tan=16):**

| var | FBBT box | after OBBT | shrunk? |
|---|---|---|---|
| x0 | [0.01, 200] | [0.01, 200] | **no** |
| x1 | [0.01, 200] | [0.01, 200] | **no** |
| x2 | [1, 35] | [1, 35] | **no** |
| x3 | [1, 200] | [1, 200] | **no** |
| x4 | [0.106, 13600] | [1.06, 13600] | trivially |
| x6 | [0.505, 154.3] | [0.505, 142.4] | trivially |
| x7 | [3.2e-5, 198] | [1.4e-4, 189] | trivially |

Soundness (validated): (i) every reported bound `≤ 5.4709`; (ii) fixing the base
vars to a near-optimal feasible point (x0=0.645, x1=3.73, x2=4.10, x3=1.0; true
obj 5.211) the relaxation **admits a feasible completion — no row cuts it**;
(iii) 1000 structured feasible points sampled (aux vars solved exactly from the
defining equalities C0–C3), no row cut. *Note:* random sampling over the wide box
gives min sampled obj ≈ 657 — it misses the tiny optimum basin, so the binding
soundness check is (i)+(ii), not the sampled minimum.

## Root cause — why the ratio lever + OBBT cannot certify at the root

The objective decomposes at the optimum (near-opt point, true obj 5.211):
`1.10471·x0²·x1 = 1.71` and `0.04811·x2·x3·(14+x1) = 3.50`. Two independent
loosenesses, both driven by the **wide box**, cap the root bound:

1. **The signomial constraints C5/C6/C7 are loose over the wide `x2∈[1,35],
   x3∈[1,200]` box.** At the V0 LP optimum x2=1.01, x3=3.80 — where the *true*
   `x2²·x3 = 3.88 < 16.8` **violates C5** — yet the ln-over-envelope of C5 admits
   it. So the constraints do not force x2,x3 up, and the objective's `x2·x3`
   term stays small. Independently, the objective's own log-space `exp`
   under-estimator lets the lifted `x2·x3 = 1.08` sit far below the true 3.84.
   **Diagnostic:** the *exact* minimum of the objective over C5∩C6∩C7∩C8 alone
   (dropping the shear constraints) is **2.76** — so even a perfect relaxation of
   the pure-(x2,x3) signomials plateaus at 2.76, not 5.47.
2. **The remaining 2.76 → 5.47 is carried by the shear-stress constraint C4**
   (`sqrt(x4²+2·x4·x5·x7+x5²) ≤ 13600`), which is what forces x0,x1 *up* off
   0.01 (giving the 1.71 term). C4's relaxation is loose over the wide box:
   at x0=x1=0.01 the true x4 = 4243.28/(0.01·0.01) = 4.2e7 massively violates
   C4, but the reciprocal + squared-McCormick relaxation admits x0=x1=0.01.

**OBBT cannot break this:** to shrink x2's box, OBBT maximizes x2 over the
relaxation; but the loose C5/C6/C7 admit x2 = 35, so the box does not move. The
looseness that caps the bound is the same looseness that neuters OBBT. This is
the general failure mode of OBBT-over-a-loose-relaxation, and it is why the
"apply OBBT to the wide boxes" hypothesis fails here even though the boxes *are*
wide.

## The only thing that closes it: spatial branch-and-bound (thousands of nodes)

A probe best-first spatial B&B branching (geometric-midpoint) on the four primal
vars x0,x1,x2,x3, node bound = this sound relaxation:

(reproducible: `python3 docs/dev/lr0_probe/nvs05_division.py --bnb`)

| nodes | global lower bound |
|---|---|
| 3 (≈ root) | 0.74 |
| ~11 (10-node probe) | 0.81 |
| 28 | 1.02 (branching x2,x3 alone plateaus at 2.76 — the pure-subset ceiling) |
| 208 | 4.42 |
| 501 | 5.08 |
| 4001 | **5.208**, still short of the 5.4703 certificate target; **wall 8.8 s** |

(The bound asymptotes near 5.21 — the near-optimal local value — indicating that
even with unlimited branching this relaxation is too loose to certify to
`6.5e-4`; certifying the true 5.4709 needs a materially stronger per-node
relaxation, not merely more nodes.)

So the gap *is* closable, but only by genuine spatial B&B on the continuous
primal variables — **thousands of nodes**, and it did not reach the certificate
tolerance within the probe budget. This is exactly the outcome the plan's §1.1
bar excludes: "the bound improved but the tree still needs [thousands of nodes /
seconds-to-not-converging] is a KILL." nvs05's residual is spatial-B&B effort,
not a missing root relaxation.

## Falsification statement (for plan §7 + gap-closing-plan §6)

> **F17 — A division/ratio relaxation lever + OBBT is NOT the root-tightness lever
> for nvs05.** On a sound probe that keeps C2/C6 (algebraically simplifying C6's
> in-sqrt 1e15 to a modest signomial), gives every `a/(x·y)` an exact 1-D
> reciprocal envelope, uses log-space for all positive monomials/signomials, and
> runs OBBT: the **root** bound reaches only **0.78** (best of 32 tangents/
> envelope), closing ~2 % of the root gap — **>50 % of the root gap remains**
> (KILL). **OBBT shrinks none of the wide ratio-feeding boxes** (x0,x1∈[0.01,200],
> x3∈[1,200] unchanged after 10 rounds) because the relaxation is too loose to
> pin any primal variable off its bound. The nvs05 gap decomposes as: 0.674
> (objective box-min) → 2.76 (exact minimum over the pure-(x2,x3) signomials
> C5/C6/C7) → 5.47 (the shear-stress C4 forcing x0,x1 up). Only the second and
> third steps are gained, and only by **spatial B&B on the continuous primal
> vars** (4001 nodes → 5.208, not certified to tolerance; 8.8 s). nvs05 is
> research-grade for a near-root certificate; the lever is branch-and-reduce
> throughput on continuous vars, not any root relaxation (H-LOG per F16, nor
> ratio/OBBT per F17). **The one confirmed premise:** nvs05's ratio boxes *are*
> genuinely wide (FBBT does not tighten x0,x1,x3) — but wide boxes plus a loose
> relaxation make OBBT inert, so "wide box" is not by itself an actionable lever.

## Consequences / recommendation

- **Do not build an nvs05-specific root division/ratio pass.** It is inert where
  the gap lives, and OBBT does not engage. (Consistent with the P2 top-k OBBT
  KILL referenced in the task brief.)
- nvs05's honest path is **continuous-variable spatial branch-and-bound
  throughput** (the branch-and-reduce front already on the roadmap,
  `docs/dev/certification-gap-plan.md`), not a relaxation lever. Its 3803-node /
  9 s probe trajectory suggests it is a *tree-efficiency* instance, and even then
  the relaxation is loose enough that certifying to 6.5e-4 needs either a much
  stronger per-node relaxation or many more nodes — i.e. it is genuinely hard,
  matching BARON's own 0.5 s only via BARON's mature branch-and-reduce + range
  reduction, not a single transform.
- **Lever-A campaign status after F16 + F17:** the one banked win remains
  **LR-2/H-UNI on nvs09** (LR-0 variant b). nvs05 and tanksize are both KILLs for
  root-relaxation levers.

## Reproduce

```bash
cd <worktree>
export PYTHONPATH=$PWD/python:$PWD/docs/dev/lr0_probe
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
python3 docs/dev/lr0_probe/nvs05_division.py          # V0/V1/V2 + OBBT box + validation
python3 docs/dev/lr0_probe/nvs05_division.py --bnb    # + the spatial-B&B trajectory
```
