# #2172: batch_processing's dual gap under big-M / hull (2026-11-05)

Where the dual bound stalls on GDPlib `batch_processing` (reference: BARON 679365.33),
measured on the #2144 branch plus the #2170 commit. Host: 4 cores. Load 1-min was 3-5
throughout, because the #2170 panel ran two solves beside these. Walls are indicative
only; bounds at a fixed node count are deterministic and are the evidence.

Arms: `DISCOPT_GDP_CONFIG_FEAS_FALLBACK=0` gives no incumbent. `=1` (default) gives the
#2144 fallback's first point.

## 1. The bound at a fixed node count does not depend on the incumbent

`bp_probe.py <method> 900` with `MAXN=300` (`max_nodes`; the tree stops at the batch
that crosses it):

| lowering | fallback | incumbent | nodes | dual bound | wall |
|---|---|---|---|---|---|
| hull | OFF | none | 317 | 592238.9435633742 | 89.5 s |
| hull | ON | 1623421.03 | 317 | 592238.9435633742 | 165.3 s |
| big-M | OFF | none | 311 | 552271.2989113937 | 63.4 s |
| big-M | ON | 1034233.32 | 311 | 552271.2989113943 | 100.9 s |

The bounds agree to 10+ digits. The tree is the same tree, and an incumbent at
1.5-2.4x the bound prunes nothing. The node loss #2172 observed at 120 s
(big-M 1163 -> 693, hull 699 -> 349) is therefore pure wall cost: +76 s per 317
hull nodes and +38 s per 311 big-M nodes.

## 2. Where the incumbent's wall goes: RINS, priced far below its cost

cProfile of hull, fallback ON, 159 nodes (171.6 s profiled):

| function | calls | cumulative |
|---|---|---|
| `rins` (one restricted `diving` per call) | 6 | 54.8 s |
| root `fractional_diving` | 1 | 7.7 s |
| root `feasibility_pump` | 1 | 7.0 s |
| batched node NLPs (`solve_problem_batch`) | 20 batches | 11.0 s |

No improver call improved the incumbent (1623421.03 at the start and at the end).
NLP-BB's #1344 contingent (`_improver_within_contingent`) charges RINS a flat 5 units,
against a budget of `0.5 x nodes x weight`. One RINS call here costs about 9 s. That is
more than 100 nodes' worth of node NLPs, so the governor admits it at roughly one call
per 20 nodes. This is a defect of the governor's cost model (abstract units, not wall),
not of this instance. It is general to every NLP-BB solve, and fixing it needs its own
corpus panel. It does **not** decide #2172: the OFF arm runs no improvers and still
stalls (section 3).

## 3. The limiting factor: the relaxation

Hull, fallback OFF (no improvers at all):

| nodes | dual bound | gap to 679365.33 | source |
|---|---|---|---|
| 0 (root NLP) | 546849.80 | 19.5 % | #2172 |
| 317 | 592238.94 | 12.8 % | table 1 |
| 699 | 608539 | 10.4 % | #2144 panel, 120 s |
| 3015 | 635222.16 | 6.5 % | `n3000_fb0` (profiled run, 1051 s wall) |

The bound grows about logarithmically in nodes. A tenfold increase in nodes (317 to 3015)
bought 43 k, and 44 k remain. At that rate certification needs on the order of 3e4 nodes.
At the 0.28 s/node of the unprofiled 317-node run, that is ~8e3 s; even at the node-NLP
cost alone (~0.07 s/node), it is ~2e3 s. Neither fits the 120 s gdp_parity budget. Node
time in the profiled run is dominated by per-node Python FBBT
(`tighten_nonlinear_bounds`, 673 s cumulative vs 200 s of node NLPs). That figure is
inflated by cProfile on ~1.6e8 calls, so the split is indicative.

## 4. Directions from the issue, measured

- **LOA's bound** (`gdp_method="loa"`, 120 s): master bounds 598109.85 and 610758.89,
  final bound 598109.85, which is no better than NLP-BB hull at 120 s. Each LOA
  subproblem is itself a convex MINLP (routed to LP/NLP-BB); the first took ~75 s. LOA's
  **incumbent**, 681615.14 (0.33 % above BARON), is far better than NLP-BB's
  1.0-1.6 M. It is primal only and does not close the dual side. The root LOA proposer
  (`DISCOPT_GDP_LOA_PROPOSER`) never runs here: batch_processing routes to NLP-BB,
  whose root has no proposer call site.
- **LP/NLP-BB on the lowered model.** The convex-MINLP router judges the *source* GDP
  ("not routed: model is not convex"). Both lowered forms qualify for `lp_nlp_bb`
  (`route_check.py`). Solving the lowered hull directly (`lowered_route.py hull 120`)
  routes there and stops at bound **569660.62 after 89 s**, below NLP-BB's. Only 1 of
  the 1735 rows is nonlinear, so the gap is in the discrete structure, not the
  curvature. Routing the lowered model there would not help; not pursued.
- **Throughput.** Section 2 recovers about half the node rate with an incumbent, which is
  worth having. Section 3 shows that even the incumbent-free node rate is 1-2 orders of
  magnitude short.

## Verdict

batch_processing cannot certify under big-M or hull within the gdp_parity budget with
the current relaxations. The measured limiting factor is the strength of the lowered
relaxation and its bound-per-node growth (root 19.5 % below the optimum, 6.5 % after
3015 nodes), not the incumbent and not the node throughput. Closing it needs a stronger
relaxation of the lowered model, as BARON's certificate suggests, which is outside
#2172's three directions. The RINS cost-model defect in section 2 is real and general,
and worth its own issue.

## Files

- `bp_probe.py`: one solve (`MAXN` caps nodes; the solve is cProfiled to `PROF_OUT`).
  The four 300-node rows ran before the profiler was added; `n3000_fb0` ran with it.
- `route_check.py`: `_convex_minlp_auto_route` on the source and on both lowerings.
- `lowered_route.py`: solve the lowered model directly (router sees it).
- `*.result.txt`: the `RESULT` / `STATS` lines of each run.
