# #764 — Direct SCIP comparison on `tanksize` (2026-07-19)

Ran SCIP 10.0 (pyscipopt 6.2.1) on `tanksize.nl` directly. This settles what discopt is
missing to reach SCIP performance — and **corrects an earlier over-emphasis in this campaign
on "root relaxation quality"**: the root is NOT the gap.

## MECHANISM TRACE (2026-07-19) — the decisive finding: propagation, not OBBT/cuts/throughput

Ablating SCIP on tanksize (60 s cap) pins exactly what climbs the dual bound:

| config | nodes | time | closes? |
|---|---|---|---|
| default | 1351 | 1.6 s | ✓ |
| OBBT propagator off | 1771 | 1.9 s | ✓ |
| separation (cuts) off | 1559 | 1.6 s | ✓ |
| OBBT + cuts off | 1391 | 1.0 s | ✓ (faster!) |
| **propagation off** | **183 018** | **60 s** | **✗ (dual stuck 1.253)** |

**Domain propagation is the sole lever.** OBBT and cuts are near-irrelevant (SCIP is *faster*
without them). Per-propagator domain-reduction counts (OBBT+cuts off) localize it to the
**nonlinear constraint handler: 38 258 DomReds** (linear 12 766; every standalone propagator
combined ~48). SCIP propagates *bidirectionally through the product constraints*
(`w = x·y` ⟹ tighten `x,y` from `w` and `w` from `x,y`), cutoff-coupled — cheap (no LP solves),
which is why it does **1391 fast nodes**. Bound trajectory (pure propagation+branching, OBBT+cuts
off): 0.833 @1 → 1.053 @50 → 1.193 @200 → 1.269 @1391. SCIP has the incumbent (1.2686) from node 1.

**Why this reframes #764.** discopt uses the OPPOSITE recipe: its FBBT is too weak to climb this
bound (recorded: OBBT-off → 0.89, stalls), so it substitutes **expensive per-node OBBT** (~95 LP
solves/node) to get the coordinated tightening SCIP gets for free from strong nonlinear
propagation. So:
* The **native-kernel throughput work (C1)** — and the in-kernel **OBBT sweep** it centers on —
  optimizes the WRONG axis for tanksize. Fast OBBT nodes still climb the bound slowly; the native
  kernel is bound-neutral on tanksize (root 0.838 = trusted) but does not certify it (bound frozen
  at 0.838 across DFS/best-bound to 20 000 nodes — reproduced).
* The **real lever is per-node nonlinear-propagation QUALITY** (reverse-McCormick / product FBBT,
  cutoff-coupled, iterated) — the branch-and-reduce line (C2). SCIP has it; discopt does not.
  Building SCIP-strength nonlinear propagation (so the bound climbs via cheap propagation instead
  of expensive OBBT) is the actual path to certifying tanksize, and is distinct from everything
  the native kernel delivers.

Repro: `pyscipopt` ablation + `writeStatistics` (Propagators / Constraints sections).

## C2 ENTRY EXPERIMENT (2026-07-19) — GO: propagation climbs the bound on the real instance

Ran the falsifying experiment before any C2 build
(`discopt_benchmarks/scripts/issue764_c2_propagation_entry.py`; raw trajectory in
`results/issue764_c2_propagation_entry_20260719.txt`). Setup: best-bound B&B over tanksize's
presolved root box, incumbent seeded (1.2686437615, mirroring SCIP having it at node 1), each node =
**FBBT fixpoint** (linear rows + bidirectional affine-form-product + sqrt + integer rounding +
objective cutoff — **zero LP probes**) then ONE trusted `MccormickLPRelaxer.solve_at_node` LP for
the node bound, then spatial branching. Control arm: identical loop, propagation off.

Dose-response (global dual bound):

| propagation strength | bound trajectory |
|---|---|
| **off** (control) | **0.83824 — perfectly flat, 300 nodes, zero movement** |
| v1: strict-sign reverse division, widest-var midpoint branch | 0.891 @300, **hard stall** (= discopt's recorded OBBT-off 0.89 stall) |
| v2: + one-sided extended division (`B∈[0,bhi], w>0 ⇒ A ≥ w_lo/b_hi`) + LP-point spatial branch | 0.904 @51 → 0.920 @600 → 0.930 @1000 → **0.956 @1800–3000, still stepping up, no stall** (1306 propagation-infeasibility prunes) |
| SCIP reference | 1.053 @50 → 1.193 @200 → closed @1391 |

**Verdict: GO.** The kill criterion (stuck ≤0.86 @300) is decisively not hit. The mechanism is
confirmed on the real instance with a clean three-level dose-response — the bound moves exactly as
much as the propagation is strong, and the control proves nothing else moves it. Two calibrations
that bound the build:

1. **Naive FBBT is NOT enough** — v1's 0.891 stall reproduces discopt's existing stall exactly.
   The load-bearing ingredients (each broke a stall when added): the **extended zero-touching
   reverse division** (tanksize's variables sit at 0, so strict-sign reverse propagation is mostly
   blocked) and **LP-point spatial branching**. Closing at SCIP's node-rate needs SCIP-grade
   propagation strength + violation/reliability branching on top.
2. **C1 × C2 is the synthesis.** The prototype runs ~32 ms/node (≥30 ms of it the Python LP). The
   native kernel (built this session) does that node LP in ~1–2 ms — so even at the prototype's
   node-rate (slower than SCIP's), tens of thousands of propagation-driven nodes run in tens of
   seconds natively. Neither axis alone suffices (throughput alone: bound frozen at 0.838;
   propagation alone at Python speed: climbing but slow); together they are the credible path.

**Next (the C2 build):** port the propagation pass into the native kernel — linear + product
(with extended division) + sqrt fixpoint, cutoff-coupled, run per node BEFORE the LP — replacing
the ~95-probe OBBT sweep as the default tightening; branch at the LP point; then the bound-neutral
and net-positive graduation panels per the CLAUDE.md regime-2 protocol.

---


## The numbers

> **CORRECTED (2026-07-19, direct measurement).** An earlier version of this table listed SCIP
> root 0.949 / discopt ~0.92. Those figures were wrong. Direct pyscipopt root solves
> (`limits/nodes=1`) give SCIP root **0.8508** (cuts on) / **0.8383** (separation off); discopt
> root is **0.8402**. The qualitative conclusion (throughput, not root) is unchanged and in fact
> *strengthened* — SCIP's root advantage over discopt is ~0.01, essentially zero, not 0.03.

| root dual bound | value | nodes to close | time | node rate |
|---|---|---|---|---|
| **SCIP**, cut loop on (`nodes=1`) | **0.8508** | 1351 | 1.46 s | **~925 nodes/s** |
| **SCIP**, separation off | 0.8383 | — | — | — |
| **discopt** root | **0.8402** | ~450 to close | >300 s (times out) | **~2–10 nodes/s** |
| oracle optimum | 1.2686 | | | |

**discopt's root (0.8402) already MATCHES SCIP's fully cut-loaded root (0.8508).** SCIP does *not*
start from a tighter relaxation — its entire cut loop closes only **2.91 % of the root-to-opt gap**
(0.8383 → 0.8508). Both solvers climb from ~0.84 to the optimum 1.2686 by **branching**. The entire
difference is **node throughput: ~100–450×**.

### Cut-reachability sub-finding (2026-07-19) — real gap, but irrelevant

Instrumented tanksize's default solve: **zero cut separators are invoked** across the spatial
McCormick B&B (Gomory/MIR/c-MIR/aggregation/RLT/root-cover loop all called 0×, even with
`GOMORY_CUTS_ENABLED=True` + `DISCOPT_CMIR_AGGREGATION=1` forced). The integer-cut machinery lives
entirely in `_solve_milp_bb` (the MILP path); tanksize takes the spatial path, which has **no cut
seam**. So there *is* a genuine cut-reachability (plumbing) gap the prior P3 NO-GO work never tested
on the spatial path — **but building the seam would not help**: SCIP's own cut loop, the ceiling any
seam could reach, gains only 2.91 % of the root gap, and discopt's root already sits at SCIP's
cut-loaded level. Direct kill-criterion measurement; the cut line is NO-GO for tanksize, now on the
real instance (not a proxy). The lever is per-node throughput (C1), not cuts (C3).

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

It is **throughput**, and — corrected by the 2026-07-19 direct measurement — it is **one**
architectural gap, not two:

1. **Native per-node engine (C1):** the ~100× Python/JAX per-node overhead vs SCIP's native C.
   This is the sole lever. Move the per-node loop into `discopt-core` so its ~95 warm OBBT probes
   cost SCIP-like time.

2. ~~**Cut-engine quality (C3):**~~ **RULED OUT for tanksize (measured).** The earlier version of
   this doc called cuts "the higher-leverage first step." That was wrong: SCIP's *own* cut loop
   closes only **2.91 %** of tanksize's root gap, and discopt's root (0.8402) already matches
   SCIP's cut-loaded root (0.8508). A perfect cut engine — matching SCIP's separators exactly —
   could therefore gain **at most ~3 %** of the root gap here. Cuts do **not** drive tanksize's
   bound climb for *either* solver; branching throughput does. Do NOT build a spatial cut seam for
   this class. (The reachability gap is real — 0 separators reached — but its ceiling is ~3 %.)

**Correction to the earlier campaign framing:** the RLT/SDP "root relaxation" research *and* the
cut-engine hypothesis were both detours — SCIP's own cut-loaded root (0.8508) ≈ discopt's (0.8402),
so neither a tighter relaxation nor better cuts is the blocker. The blocker is per-node
*throughput*: discopt substitutes expensive per-node OBBT for cheap fast branching, and does that
OBBT in Python/JAX at ~100× SCIP's per-node cost. There is no single flag being skipped, and no cut
family to add; matching SCIP is the native-node (C1) work alone.

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
