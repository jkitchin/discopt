# #764 — Direct SCIP comparison on `tanksize` (2026-07-19)

Ran SCIP 10.0 (pyscipopt 6.2.1) on `tanksize.nl` directly. This settles what discopt is
missing to reach SCIP performance — and **corrects an earlier over-emphasis in this campaign
on "root relaxation quality"**: the root is NOT the gap.

## The numbers

| | root dual bound | nodes | time | node rate |
|---|---|---|---|---|
| **SCIP** | **0.949** | 1351 | 1.46 s | **~925 nodes/s** |
| **discopt** | ~0.92 (frontier) | ~450 to close | >300 s (times out) | **~2–10 nodes/s** |

**SCIP's root bound (0.949) ≈ discopt's (0.92).** SCIP does *not* start from a tighter
relaxation. Both climb the dual bound from ~0.95 to the optimum 1.2686 by **branching**
(SCIP: 0.95 @root → 1.13 @100 nodes → 1.27 @1351). The entire difference is **node
throughput: ~100–450×**.

## What SCIP does per node that discopt doesn't

SCIP printStatistics on `tanksize` — cuts *applied* over the whole solve:
`aggregation/c-MIR 47 (cmir 40, flowcover 4, knapsackcover 3)`, `gomory 6`, `rlt 6`,
`impliedbounds 11`, `mixing 3` — cheap **integer** cuts on the LP, generated throughout the
tree, pooled and reused. Per node SCIP does ~1 warm dual-simplex LP + domain propagation +
occasional cut separation, all in native C ⇒ **~1 ms/node**.

discopt per node (~100–500 ms): **~95 OBBT LPs** (2n probes, transient — recomputed every
node) + McCormick LP + Python/JAX orchestration + NLP heuristics. discopt does per-node OBBT
because **its bound STALLS without it** (measured: OBBT-off → 0.89, never climbs), whereas
SCIP's branching + cheap cuts climb without any per-node OBBT.

## Why discopt can't just adopt SCIP's cheap-node recipe (measured)

Tried the SCIP recipe in discopt — per-node OBBT OFF, cuts ON (`DISCOPT_CMIR_AGGREGATION=1`,
`CUT_INHERIT=1`, `ROOT_CUT_ROUNDS=15`): bound **0.8915, stalled** (same as OBBT-off alone).
`DISCOPT_CMIR_AGGREGATION=1` with OBBT on was net-*negative* (0.884 vs 0.911 default — cut
separation overhead, no bound gain). discopt's integer cuts are **inert on this bilinear
class** (the CUT-1/CUTS-1 NO-GO, reconfirmed here): they cannot drive the bound climb, so
discopt is forced onto expensive per-node OBBT, which is what makes its nodes 100–450× slower.

Branching probe: discopt branches on x0/x1/x2 and x3/x4/x5 but **never on x6–x14** (the wide
vars in the loose products x0·x6 …) — it "pins" those functionally-dependent intermediates with
OBBT instead of branching, another reason it leans on OBBT.

## The honest conclusion — what reaching SCIP performance actually requires

It is **throughput**, and it is two coupled architectural gaps (both = `certification-gap-plan`
core, not a missed flag — every relevant flag was tested here and none helps):

1. **Cut-engine quality (C3):** make discopt's integer/bilinear cuts *effective* on this class
   so **cuts drive the bound climb (SCIP's mechanism), replacing per-node OBBT**. This is the
   higher-leverage first step: it removes the 95-LP/node cost, and the separators already exist
   (they just underperform). Today they are inert-to-harmful here.
2. **Native per-node engine (C1):** the ~100× Python/JAX per-node overhead vs SCIP's native C.

**Correction to the earlier campaign framing:** the RLT/SDP "root relaxation" research was a
detour — SCIP's own root is 0.95, so a tighter root was never the blocker. The blocker is
per-node *throughput*, and specifically that discopt substitutes expensive OBBT for the cheap
cuts SCIP uses. There is no single flag being skipped; matching SCIP is the cut-engine +
native-node work.

## Deep dive: why discopt's branching stalls but SCIP's climbs (2026-07-19)

Investigated whether discopt's spatial branching is the gap. Findings:

- **discopt already uses pseudocost + strong (reliability) branching** (`tree.score_candidates` +
  `_strong_branch_lp`), like SCIP — the Rust `select_spatial_branch_variable` widest-box rule is
  only the fallback. So branching is not naively broken.
- **The looseness is distributed, decisively.** At the root LP solution, branching *any* single
  continuous variable at its LP value `x*` gives **0.0000 bound gain** (all 26 candidates,
  measured). Tightening one variable never helps because the other products' McCormick slack
  compensates. Only *coordinated* tightening moves the bound: narrowing `x16` to a ±0.05 window
  (not a single branch — a deep contraction) jumps 0.838→1.05.
- This is exactly why discopt uses per-node OBBT: OBBT tightens *all* boxes at once (the coordinated
  move), where single-variable branching cannot. SCIP climbs because it does ~925 cheap nodes/s —
  each node's branch contributes a little and the volume compounds; discopt at 2–10 nodes/s cannot
  afford that volume, so it substitutes the (expensive, coordinated) OBBT.

## Final synthesis — the one thing to fix

There is **no missed flag or heuristic** — relaxation (RLT/SDP), cheap DBBT, cuts, selective OBBT,
and branching were each measured and none is the shortcut. The gap is singular and architectural:

> **SCIP does a node in ~1 ms (native C: warm dual-simplex LP + propagation + cheap pooled cuts).
> discopt does a node in ~100–500 ms (Python/JAX orchestration + ~95 OBBT LP solves it cannot drop,
> because its cheap tightening — cuts, reduced-cost DBBT — is inert on this bilinear class).**

To reach SCIP performance discopt must move the per-node loop — bound patch → **warm** LP → OBBT
probes (warm dual-simplex, in-kernel, no Python marshaling between probes) → propagation → branch —
into `discopt-core` (the C1 "per-node language cost — architecture" gap). The earlier "Phase B
ceiling ~2.3×" underestimated this: it assumed the OBBT probe LPs stay at their current ~2.4 ms
cold cost, but in-kernel warm dual-simplex probes (no marshaling) target the ~0.1–0.9 ms floor —
so the real ceiling on the OBBT-bound class is well above 2.3×. That native node kernel is the
work; it is weeks of Rust, not a configuration, and every cheaper alternative is now measured out.
