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

## C2 BUILD RESULT (2026-07-19) — propagation in the native kernel; honest measurement

Built `bnb/spatial_propagate.rs` (the validated propagator: linear rows + affine-form products
with **extended zero-touching division** + sqrt/monomial/affine-square forward+reverse + integer
rounding + objective cutoff, outward-guarded, fixpoint) and wired it into `spatial_tree` per node
BEFORE the LP, default ON with OBBT default OFF (the validated recipe); `initial_incumbent` seeds
the cutoff coupling. Node profile is now SCIP-like: **~1 LP solve per node, zero OBBT probes,
~2.6 ms/node**.

**Soundness bug found & fixed during validation:** the first native run reported `optimal` at
20 547 nodes with bound 0.937 vs incumbent 1.2686 — a **false certificate**. Root cause: a feasible
leaf closed its region unconditionally after accepting the incumbent even when the region's
rigorous bound did not certify (the region could hold better points), and exhaustion then labeled
itself `Optimal`. Fixed: a feasible leaf closes only when `bound >= incumbent - gap_tol` (else it
keeps branching, with a widest-column fallback), and exhaustion returns `Optimal` only when the
final bound genuinely closes the gap — a new honest `Exhausted` status otherwise. Regression test
`optimal_status_implies_certified_gap` locks it.

**Honest tanksize measurement post-fix** (`results/issue764_native_propagation_20260719.txt`):
bound climbs **0.838 → 0.927 @3k → 0.937 @20k, then plateaus flat through 100k nodes**. Without
propagation the bound is frozen at 0.838 forever, so the mechanism works and the node economics are
right — but the climb stalls at 0.937, well short of certification (1.2686). The prototype with the
*trusted Python relaxer* per node reached 0.956 @3k — richer per-node relaxation (separable
objective floor, convex-lift OA, separation) than the kernel's bare McCormick+BLF rows — and SCIP
reaches 1.269 @1391 with far stronger propagation + enforcement linearization (8 946 cuts) +
reliability branching. **Remaining gap to certify tanksize natively: per-node relaxation/propagation
QUALITY** — (a) richer node LP rows (the trusted build's extra families), (b) SCIP-strength
propagation (more atoms, better ordering), (c) violation/reliability branching. The infrastructure
(cheap propagated nodes at ~2.6 ms) is in place; the quality ratchet is the follow-on.

## QUALITY RATCHET RESULT (2026-07-19) — **tanksize CERTIFIED natively**

Diagnosed the 0.937 plateau before building (§4 discipline): new `n_uncertified` telemetry showed
**51.6 % of nodes (10 315 / 20 000) had uncertifiable safe bounds** — the plateau was a
**certification artifact**, not relaxation looseness. Root cause: `ns_safe_bound_csc` returns
`None` whenever a nonzero reduced cost meets a `>= 1e20` bound, and the kernel's **slack columns
carried `u = 1e20`** — so a roundoff-level slack reduced cost (`rc = −y_r ≈ −1e−13`) killed the
whole certificate, and stuck subtrees froze at ancestor bounds. Equilibrated node solves alone did
NOT fix it (51.4 % after switching to `solve_lp_cols_scaled` — kept anyway for conditioning);
**finite min-activity slack upper bounds** (`s_r = b_r − min_activity_r`, finite because every
structural bound is) fixed it outright: **0 / 10 653 nodes uncertified**.

| run | result |
|---|---|
| seeded incumbent | **optimal**, gap 9.97e-05, 10 653 nodes, **26.9 s** (2.53 ms/node, 1 LP/node) |
| unseeded (self-contained) | **optimal**, gap 1.00e-04, 78 667 nodes, 191.8 s |
| end-to-end `m.solve()` (flag ON) | **optimal**, obj 1.2686457, bound 1.2685458, 190.8 s |

Every certificate brackets the oracle (`bound <= 1.2686437540 <= incumbent`), `Optimal` is the
honest post-fix semantics (`bound >= incumbent − gap_tol` enforced), and the run is locked by
`test_tanksize_certifies_natively`. The "richer node rows" turned out unnecessary for tanksize —
the rows were adequate; half the certificates were being discarded. **The #764 definition of done
(certify tanksize at default tolerances) is met** — 27 s seeded / 192 s self-contained vs the
Python path's >300 s timeout at a 32.8 % gap. Remaining follow-ons for parity with SCIP's 1.46 s:
seed the driver with a cheap NLP-heuristic incumbent (27 s path), then per-node throughput +
propagation-strength polish. Raw numbers: `results/issue764_tanksize_certified_20260719.txt`.

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

## RESOLUTION (2026-07-19)

The pessimism above ("the native kernel does not certify tanksize") was **superseded**
by the C2 nonlinear-propagation pass + finite-slack safe bounds landed on this branch:
the native kernel now **certifies tanksize** (0 uncertified nodes, certificate brackets
the oracle 1.2686437540).

**Seeded wall time (Task 1).** `_try_native_spatial_kernel` now seeds the kernel's
`initial_incumbent` from a *rigorously-verified* feasible point: solve the continuous
NLP relaxation once, pin the presolve-fixed integers, enumerate every 0/1 assignment of
the free binaries (tanksize: 5 free → 32 sub-NLPs), keep the best point that passes
`_native_kernel_verify_point` (bounds + constraints to abs=1e-6/rel=1e-4, integrality
1e-5) and use its TRUE objective. End-to-end `m.solve()` with the flag ON:

| run | status | objective | bound | nodes | wall |
|---|---|---|---|---|---|
| unseeded | optimal | 1.2686457461 | 1.2685457806 | 78 667 | ~255 s |
| **seeded** | **optimal** | **1.2686437526** | 1.2685440897 | **11 379** | **~50 s** |

A ~5× speedup; the seeded objective is within 1.4e-9 of the oracle and the certificate
brackets it (`bound ≤ 1.2686437540 ≤ incumbent`).

**Graduation panel (Task 2).** `discopt_benchmarks/scripts/issue764_native_kernel_graduation_panel.py`
ran ON-vs-OFF over the 66-instance in-repo corpus (60 s budget, subprocess-isolated).
Artifact: `discopt_benchmarks/results/issue764_native_kernel_graduation_panel_20260719T155819Z.{json,txt}`.

- **cert-clean: PASS — 0 violations.** Every ON-optimal objective matches OFF to
  abs=1e-6/rel=1e-4; no ON dual bound past a reference optimum; no optimal→non-optimal
  regression; all 4 engaged incumbents (dispatch, nvs13, st_e13, tanksize)
  independently feasibility-verified against the original model.
- **net-positive: PASS (by the median bar).** Median non-engaged wall Δ = **−0.146 s**
  (ON slightly *faster*); tanksize moves **feasible→optimal** (headline win); st_e13,
  nvs13, dispatch engage and are ≤ OFF. Producer-probe overhead on cleanly-completing
  decliners is < 0.5 s; the large ± deltas (bchoco07 +40 s, bchoco08 −39 s) are
  timeout-instance wrap-up noise.

**Default decision: KEEP OPT-IN (default OFF) — do NOT flip to default-ON, despite both
panel bars passing.** Two safety blockers, and CLAUDE.md puts safety/gate-integrity
before performance:

1. **Blast radius on the smoke gate.** With the flag forced ON, **20 of the `-m smoke`
   tests fail** (807 pass) — all exercising Python-engine machinery the native kernel
   short-circuits (incumbent/node callbacks, RENS/SubNLP heuristic paths, solution
   pools, warm-start incumbents, `mccormick_bounds` modes, deadline handling, batched
   node processing, lazy constraints). Not native-kernel *correctness* failures, but a
   default-ON would silently disable a large body of validated Python-engine behavior
   and break a hard PR gate. Greening it by editing 20+ tests would be weakening
   validations to pass a gate (forbidden).
2. **No wall budget.** The kernel runs to `max_nodes` with no time limit, so on a
   covered-but-hard instance it can run away — panel `contvar`: OFF 65 s → ON >200 s
   (an instance neither flag certifies). A default must be runaway-safe.

The panel PASS is the evidence the engine is sound and net-helpful; graduation to
default-ON is deferred to two follow-ups: (a) give the native kernel a wall-time budget
(no runaways), and (b) native-kernel feature parity / pass-through (or re-scope the
Python-engine smoke tests to the engine they validate) so default-ON does not silently
disable callbacks/heuristics/pools. Until then the flag stays opt-in
(`DISCOPT_NATIVE_SPATIAL_KERNEL=1`) — which already certifies tanksize, the issue's
definition of done.
