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
