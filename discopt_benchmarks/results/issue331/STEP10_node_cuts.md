# Issue #331 — Step 10: node cuts — the node-count lever, made wall-safe

The node-count investigation (after Step 9) found the genuine lever for the sparse
frontier is **cuts at fractional nodes** (node cuts reach ~SCIP node counts;
strong-branch tuning does not — it's a fragile, regime-conflicting knob, so it was
*not* shipped). Step 10 makes node cuts a clean win instead of the 2× wall
regression they were.

## Why node cuts regressed before, and the two guardrails

`node_cuts` already separated globally-valid cover cuts at fractional nodes into a
shared pool. But the pool was uncapped-in-practice and never aged, so it grew and
bloated every node's LP. Two grounded guardrails (from the #331 sweep) fix it:

1. **Tight pool cap ≈ 2× the original row count.** The win lives at a *small*
   active set; loose caps drive per-node LP cost up and erase it.

   | smdk80x25 (SCIP 753n/1.9s) | nodes | wall |
   |---|---|---|
   | node cuts off | 28 907 | 3.2s |
   | cap = 2m (50) | **6 453** | **2.4s** |
   | cap = 4m (100) | 4 619 | 5.3s |
   | cap = 8m (200) | 2 569 | 10.6s |

   `2m` is the wall-positive sweet spot; tighter loses nodes, looser loses wall.
   (A tight static cap is the cheap stand-in for SCIP-style cut aging — removing a
   cut row mid-solve would mean repairing the warm basis/factorization, a much
   larger change that the cap sidesteps.)

2. **Density gate.** A cover cut spans its source row's support, so on *dense*-row
   models it is itself dense and bloats every node LP for **no** node benefit
   (measured: dense knapsacks +2× wall, 0 node change). Only separate when the
   structural rows are sparse (`density < 0.5`), where cover cuts are row-local and
   cheap. Set-covering ≥-rows yield no knapsack covers, so it is a no-op there too.

## Result (density gate + tight cap, sound — objectives match SCIP everywhere)

| regime | instance | nodes off→on | wall off→on |
|---|---|---|---|
| **sparse (hard)** | smdk80x25 | 28 907 → **6 453** (−78%) | 3.16s → **2.46s** (−22%) |
| **sparse (hard)** | smdk70x20 | 6 841 → **2 221** (−68%) | 0.55s → **0.49s** (−11%) |
| sparse (easy) | smdk60x20 | 1 127 → 565 (−50%) | 0.13s → 0.21s (+80ms) |
| sparse (easy) | smdk50x15 | 815 → 475 (−42%) | 0.06s → 0.09s (+30ms) |
| **dense** (all) | mdk30x5…200x25 | **unchanged** (gated off) | ±noise |

Dense is fully protected by the density gate (identical trees). On sparse, the
*hard* instances — the ones that matter for the frontier — win big on both nodes
and wall; the *easy* ones cut nodes too but the fixed cut overhead costs a small
absolute wall (+30–80 ms on sub-0.15 s solves).

## The remaining tension, and the shipped decision

There is **no clean static easy/hard discriminator**: node cuts must separate from
the first fractional node to compound (a tree-size *difficulty* gate backfires —
deferring cuts lets the hard tree explode uncut first: smdk80x25 went −22% → +50%
wall). So default-on would help the hard instances but regress the easy ones by
those tens of milliseconds — violating #331's "net wall parity on **every**
instance" rule.

**Shipped:** node cuts are now correct and wall-positive *when enabled* (density
gate + `2m` cap, internal — no tuning needed), and left **default-off** so no
instance regresses. Enabling `node_cuts=True` is the node-count lever for
hard sparse-row models (−78% nodes / −22% wall on smdk80x25, ~SCIP territory).
Closing the last gap to a default-on win needs cheaper cut-nodes still (the
sparse cut-augmented LP data layer) so the easy-instance overhead disappears —
the same frontier STEP8 named.

### Reproducing

`solve_milp_py(..., node_cuts=True)` on `gen_sparse_mdk` instances; the cap and
density gate are automatic. Soundness + gating locked by
`test_node_cuts_sound_gated_and_reduce_sparse_nodes`.
