# Replacing the MILP driver's search core with HiGHS's design

Status: plan, 2026-11-10. Owner: highs-parity D2. Supersedes the lever-by-lever approach of
`highs-parity-d2-gap-plan.md` §12.20-§12.39 for the search core.

## 1. Decision

discopt keeps **one** MILP engine. Its search core (everything between "root LP solved" and
"tree closed") is replaced, in place and in stages, by a faithful translation of HiGHS's MIP
solver design. No second solver, no parallel engine behind a flag: each stage deletes the
code it replaces in the same change.

**Why.** The ported HiGHS components are exact: on HiGHS's own inputs the separators reproduce
all 16,117 of its cuts over 270 root and node rounds of mik-250-20-75-1 (§12.39). But they run
inside a driver built differently from HiGHS (parallel node batches, a shared LP folded at
batch boundaries, a global box fixed after the root, its own pools, queue and branching), so
they are fed discopt's state. Every lever that emulated one HiGHS behaviour inside that loop
was sound and failed to transfer (§12.31-§12.39). The fix is the surrounding system, not
another lever.

**What stays discopt's** (the product, and the reason the engine is ours):

- the API: `solve_milp*` entry points, `MilpOptions` (38 fields), `MilpResult`, and the hooks
  `MilpLazyHook` (#1060 single tree), `MilpNodeHook` (#1141 fractional-node cuts),
  `MilpBoundHook`, `MilpDebugHook`, the node-callback observer, `initial_incumbent`;
- the safety layer behind certificates: Neumaier-Shcherbina safe node bounds, verified Farkas
  rays before pruning, independent incumbent verification on the original rows, the
  unresolved-floor handling of failed node LPs, deadline truncation that never certifies,
  the no-incumbent extension and stall guard; none of these exist in HiGHS as such;
- the in-house simplex as the LP engine; the MILP presolve (`milp_presolve.rs`);
- determinism.

**What is replaced:** node selection and the tree, node evaluation order, LP row management,
the global/local domains, cut pools and node separation, branching, the heuristic schedule.

**Cost accepted.** HiGHS at one thread processes nodes sequentially; stage 1 makes discopt's
node processing sequential too (equal-thread comparison unaffected). Multi-thread node
processing returns later as HiGHS's worker design (`HighsMipWorker`), not as batches.

**Limit accepted.** The LP engine is ours, so where an LP has alternative optima our simplex can
return a different vertex than HiGHS's and whole-solve traces diverge from there. Correctness
of each stage is therefore checked per node on HiGHS's dumped state (bit-exact where the LP is
not involved), and parity of the whole is checked statistically on the panel.

## 2. HiGHS MIP inventory (HiGHS 1.15.1 sources, lines `.cpp` + `.h`)

| HiGHS class | lines | discopt today | where | verified |
|---|---:|---|---|---|
| `HighsCDouble`, `HighsRandom`, hash helpers | (util) | ported | `highs_cutgen/cdouble.rs`, `random.rs` | unit + replay |
| `HighsCutGeneration` | 1,717 | ported | `core.rs` | `generateCut` dump replay, bit-exact |
| `HighsTransformedLp` | 840 | ported | `transformed_lp.rs` | TL dump replay, bit-exact |
| `HighsLpAggregator`, `HighsSeparator` | 211 | ported | `separators.rs` | replay |
| `HighsPathSeparator`, `HighsTableauSeparator` | 907 | ported | `separators.rs` | replay incl. node rounds (16,117 cuts) |
| `HighsModkSeparator`, `HighsGFkSolve` | 872 | ported | `modk.rs` | GF(k) brute-force test; **no HiGHS replay** |
| `HighsCutPool`, `HighsDynamicRowMatrix` | 1,235 | ported | `pool.rs` | pool replay |
| `HighsSeparation` | 293 | partial | `bridge.rs` (root rounds) | root only; node path uses a fresh pool per round |
| `HighsDomain` | 4,645 | partial | `domain.rs` (1,224) | row propagation; missing objective, cut-pool and conflict propagation, conflict analysis, clique/implication fixings |
| `HighsImplications` | 1,118 | partial | `vbound.rs` | variable-bound discovery/storage; missing probing implications |
| `HighsCliqueTable` | 2,582 | missing | - | - |
| `HighsRedcostFixing` | 380 | missing (discopt's own root-only RC fixing) | `milp_driver.rs` `reduced_cost_fix` | - |
| `HighsConflictPool` | 409 | missing | - | - |
| `HighsObjectiveFunction` | 195 | missing | - | - |
| `HighsPseudocost` | 644 | discopt's own | `branching.rs` | - |
| `HighsNodeQueue` | 756 | missing (discopt `TreeManager`) | `tree_manager.rs` | - |
| `HighsSearch` | 2,288 | emulated inside the batched driver | `milp_driver.rs` (plunging, `DISCOPT_HIGHS_SEARCH`) | - |
| `HighsLpRelaxation` | 2,040 | missing (discopt cut-LP fold + ageing) | `milp_driver.rs` | - |
| `HighsMipSolver`, `HighsMipSolverData` | 4,874 | missing (discopt driver) | `milp_driver.rs` | - |
| `HighsMipWorker` | 392 | missing | - | later (threads) |
| `HighsPrimalHeuristics` | 1,953 | partial | `milp_driver.rs` (RENS/RINS, randomized rounding) | - |
| `HighsFeasibilityJump` | 139 (+extern) | missing | - | - |
| `HPresolve` (+ postsolve) | 8,594+ | partial, own design | `milp_presolve.rs` (2,071) | 93 % / 89 % of HiGHS's row/column reductions (§12.37) |
| `HighsSymmetry` | 1,936 | missing | - | off in all HiGHS comparisons so far |

Ported and replay-verified: about 5,900 HiGHS lines (cut generation, transformed LP,
separators, pool). Partial: about 8,000 (separation, domain, implications, heuristics).
Missing: about 14,700, of which the search core
proper (`HighsMipSolver(+Data)`, `HighsSearch`, `HighsNodeQueue`, `HighsLpRelaxation`,
`HighsRedcostFixing`, `HighsPseudocost`, `HighsObjectiveFunction`) is about 11,200.

## 3. The driver today, by subsystem (`milp_driver.rs` at `6753be4a`, 15,745 lines)

`solve_milp_node_search` (lines 4,259-7,646, one function):

| lines | subsystem | fate |
|---|---|---|
| 4,259-4,394 | setup, options, working matrix | kept (API) |
| 4,395-4,467 | lazy / node-hook separation state | kept, re-hosted on the new loop |
| 4,468-4,554 | caller-seeded incumbent validation | kept |
| 4,555-4,667 | sub-MIP base snapshot | stage 6 |
| 4,668-5,602 | root cut loop (GMI/c-MIR, ported HiGHS separation, root RENS, propagation) | stage 0 boundary: root stays, its end state hands over to the new search |
| 5,603-5,985 | root cut cleanup, pools seeding, search setup | stages 2-4 |
| 5,986-6,233 | search loop head: limits, batch export, plunge slots | stage 1 |
| 6,234-6,521 | parallel per-node evaluation (`solve_node`, 7,800-8,650) | stage 1 (sequential `evaluateNode`), safety layer kept |
| 6,522-6,999 | sequential reduce: tree mutations, hooks, requeues | stage 1 |
| 7,000-7,188 | node pool ageing, cut selection and fold into the shared LP | stages 2 and 4 |
| 7,189-7,646 | sub-MIP / rounding heuristics at batch boundaries, termination | stage 6; termination kept |

Plus `tree_manager.rs` (4,282: queue, node pool, pseudocost hand-off; stage 1),
`branching.rs` (1,705: stage 5).

**Flags.** 43 `DISCOPT_*` variables are read by the driver and the cut-generation port. On
this plan: test/replay diagnostics stay; switches emulating a HiGHS behaviour inside the
batched loop (`HIGHS_SEARCH`, `HIGHS_SEARCH_DOMAIN`, `HIGHS_DIVE(_BATCH)`, `HIGHS_SB_TREE`,
`HIGHS_SEP_FIRST`, `HIGHS_TREE_RCFIX`, `HIGHS_ROOT_POOL`, `HIGHS_SEPA_NODE`,
`MILP_CUT_LP(_READMIT)`, `MILP_NODE_GMI(_LAZY)`) are deleted by the stage that replaces their
subsystem; the rest are classified when their stage lands.

**Tests pinning current behaviour.** Rust: 100 in `milp_driver.rs`, 53 in `tree_manager.rs`,
29 in `branching.rs`, 21 in `highs_cutgen`. Python: 156 test files reach the MILP engine.
A stage must keep all of them green, or change a test only with a recorded reason (a test
that pins a batch-era artefact rather than behaviour).

## 4. Stages and gates

Every stage: (a) **per-node replay** against a HiGHS dump of the node state the stage
consumes, bit-exact where no LP solve is involved; (b) the **soundness panel** (62 rows,
sense-aware scoring, `p3_score_ab.py`): zero violations; (c) the old code for that subsystem
**deleted in the same change**; (d) the certified count on the 62-row panel recorded (it may
dip mid-way; the target is the end state, but a dip is explained before the next stage).

| stage | builds | replaces | needs from HiGHS dumps |
|---|---|---|---|
| 0 | baseline: 62-row panel of the current default and bundle, one thread; HiGHS node-state dump patch (search, domain, LP rows, pool) | - | new dump format |
| 1 | `HighsNodeQueue` + `HighsSearch` (install, evaluate, branch, dive, backtrack, plunge), sequential | batches, `TreeManager` selection, plunge emulation | node queue order, branching decisions |
| 2 | `HighsLpRelaxation` (rows, ageing, removal, basis handling) over our simplex | cut-LP fold, relaxed-row readmission | LP row set per node |
| 3 | global + local `HighsDomain` (complete: objective, cut-pool, conflict propagation), `HighsRedcostFixing`, node-queue global pruning | fixed global box, per-node FBBT | domain change stacks |
| 4 | `HighsSeparation` at nodes with the one global `HighsCutPool` | node GMI pool, per-round fresh pool | cutsets per separation |
| 5 | `HighsPseudocost` + HiGHS branching scores | `branching.rs` scoring | branching choice per node |
| 6 | `HighsPrimalHeuristics` schedule (RENS/RINS/rounding ported; feasibility jump, shifting, ziRound new) | batch-boundary heuristics | heuristic calls and results |
| 7 | `HighsCliqueTable`, `HighsImplications` (probing), `HighsConflictPool` | - | - |
| 8 | `HighsMipWorker` threads | - | - |

Order rationale: stages 1-4 are the system the separators need (§12.39); 5-7 follow HiGHS's
own dependency order (pseudocost feeds search, cliques feed domain and heuristics).

## 5. First steps

1. Stage 0: record the baseline panel (default, bundle, bundle + presolve; one thread) and
   write the HiGHS node-state dump patch, with a reader and a replay test skeleton.
2. Retire the three levers that did not pay (`HIGHS_SEP_FIRST`, `HIGHS_ROOT_POOL=all`,
   `HIGHS_TREE_RCFIX`) with their implementations, as the flag audit requires; their logic
   returns inside stages 2-4.
3. Stage 1 design note: the sequential loop's interface to the kept safety layer and hooks,
   before code.

## 6. Stage 1 design: the search loop before its components

Stage 1 replaces the *control* of the search (which node, when to dive, branch, backtrack,
plunge, separate, run heuristics) with HiGHS's, sequentially. The components it calls are
replaced by later stages, so stage 1 calls them through narrow interfaces, each first backed
by discopt's current implementation and later swapped for the HiGHS translation:

| interface | HiGHS call sites | stage-1 backing (today's code) | replaced by |
|---|---|---|---|
| `NodeLp::evaluate(box) -> NodeEval` (status, safe bound, point, basis, fractional columns) | `HighsSearch::evaluateNode` (LP part) | `solve_node`'s LP solve with the safety layer (NS bound, Farkas check, unresolved floor) | stage 2 (`HighsLpRelaxation`); the safety layer stays |
| `Domain` (install a node's bound changes, propagate, backtrack) | `installNode`, `localdom.propagate`, `backtrack` | node box vectors + FBBT | stage 3 (`HighsDomain` local/global) |
| `Branching::select(point) -> (col, value)` and the child estimate | `selectBranchingCandidate`, `branch` | current scorer and strong branching | stage 5 (`HighsPseudocost`) |
| `Separation::at_node()` | `separateAndStoreBasis` in `processNodes` | node separation as today | stage 4 |
| `Heuristics::before_dive()` | `runHeuristics` | sub-MIP/rounding as today | stage 6 |

**Ported faithfully in stage 1:** `HighsNodeQueue` (done, `bnb/highs_search/node_queue.rs`,
replay against `HS_DUMP_QUEUE`), `HighsSearch`'s node stack (`NodeData`: bounds, estimate,
branching decision, open subtrees, domain-stack positions), `installNode`, `evaluateNode`'s
control part, `branch`, `backtrack`, `backtrackPlunge`, `dive`, `currentNodeToQueue`,
`openNodesToQueue`, and `HighsMipSolver::run`'s node-processing loop
(`processNodes`/`evaluateNode`/`pruneNode`/`separateAndStoreBasis`/`backtrackPlunge` with its
plunge limits and node selection policy).

**Kept discopt's, re-hosted on the new loop:** the safety layer (inside `NodeLp`), the
certified dual bound (the minimum over the queue, the search's open subtrees and the
unresolved floor), the incumbent with its independent verification, lazy-constraint vetoes
(#1060: the node is re-evaluated, as HiGHS re-evaluates after `addIncumbent` rejection),
fractional-node hooks (#1141: a separation call at the node), bound and debugger hooks,
deadline handling.

**Gates.** (a) Queue replay bit-exact (`full_queue_replay` on a mik dump). (b) Search-control
replay: a `HS_DUMP_SEARCH` dump records, per node, what HiGHS's control decided given its node
evaluations (install, result, branching column and direction, backtrack target, plunge
choice); the port, fed the dumped evaluations through mock `NodeLp`/`Branching`, must take the
same decisions. (c) Soundness panel clean. (d) Deleted in the same change: batched node
evaluation, the reduce, plunge emulation (`DISCOPT_HIGHS_SEARCH`, `_SEARCH_DOMAIN`, `_DIVE`,
`_DIVE_BATCH`), and the MILP driver's use of `TreeManager` (which the spatial B&B keeps).

## 7. Progress log

**Stage 0 (2026-11-10), done.**
- Baseline (`data/highs-parity/search/stage0_baseline.{py,jsonl}`, 62 rows, 20 s,
  `RAYON_NUM_THREADS=1`, three arms interleaved, idle machine): **default 40 certified**
  (total wall 547.1 s), **bundle + search 48** (398.1 s), **bundle + search + presolve 50**
  (376.0 s); sense-aware soundness (`score_ab.py`) 505 checks, 0 violations. HiGHS at one
  thread: 59. (The "default 47" quoted in earlier notes was the base settings with randomized
  rounding and run-1 continuation on, not the shipped default.)
- Retired with their implementations: `DISCOPT_HIGHS_SEP_FIRST`, `DISCOPT_HIGHS_TREE_RCFIX`,
  `DISCOPT_HIGHS_ROOT_POOL=all` (`500f03f4`).
- `HighsNodeQueue` translated (`bnb/highs_search/node_queue.rs`). HiGHS queue dump
  (`data/highs-parity/search/hs_dump_queue.patch`, `HS_DUMP_QUEUE`) over mik-250-20-75-1,
  p0201, flugpl, fiber, sp150x300d and dcmulti: 145 queues (main search and sub-MIPs), 665
  operations, 226 pops, 776 recorded states, bounding that removes nodes (4 calls, 27 nodes),
  global pruning (3), suboptimal nodes: **0 mismatches**. Committed as the fixture
  `tests/data/highs_nodequeue_replay.txt` with `node_queue_fixture_replays_exactly`.

**Next: stage 1** (§6): the `HS_DUMP_SEARCH` dump of HiGHS's search control, the
`HighsSearch` node stack and control translation with its replay, then the switch-over of the
driver's search loop.

**Stage 1, sub-phases** (from reading `HighsSearch.cpp`: `evaluateNode`, `backtrack`,
`backtrackPlunge`, `currentNodeToQueue`, `openNodesToQueue`, `installNode`). The control is
inseparable from the local domain (`setDomainChangeStack`, `propagate`, `backtrack`,
`backtrackToGlobal`, `getReducedDomainChangeStack`, `getObjectiveLowerBound`) and from
pseudocost scores (`backtrackPlunge` queues the sibling when an ancestor's inactive-minus-active
score exceeds the sibling's score). So stage 1 uses the translated `HighsDomain`
(`highs_cutgen/domain.rs`) as its local domain from the start, extended as needed (that pulls
the local-domain half of stage 3 forward), and reads scores through a pseudocost interface.

| sub-phase | content | gate |
|---|---|---|
| 1a | `HighsNodeQueue` | queue replay, 0 mismatches (**done**) |
| 1b | `HS_DUMP_SEARCH`: per node, the inputs and outcome of each control decision (install, evaluate result and bounds, branching decision, each backtrack step with prune outcome, `backtrackPlunge` scores and plunge/queue choice, dive steps, main-loop plunge/queue selection) | dump covers all decision kinds on mik and the queue fixture's instances |
| 1c | `HighsSearch` node stack and control (`NodeData`, `installNode`, `evaluateNode` control, `branch` control, `backtrack`, `backtrackPlunge`, `backtrackUntilDepth`, `dive`, `currentNodeToQueue`, `openNodesToQueue`) with `NodeLp`, `Pseudocost` and `Domain` behind traits | replay with mocks fed from 1b: identical decisions |
| 1d | `HighsMipSolver::run`'s node processing over 1c; the driver's search loop switched to it; the safety layer and hooks re-hosted; batches, reduce and plunge emulation deleted | Rust and Python suites green; soundness panel clean; certified count recorded against stage 0 |
