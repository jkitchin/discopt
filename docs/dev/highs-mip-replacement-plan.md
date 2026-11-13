# Replacing the MILP driver's search core with HiGHS's design

Status: **closed 2026-11-13** (§8: final state, population result, stop rule). Opened as a plan 2026-11-10. Owner: highs-parity D2. Supersedes the lever-by-lever approach of
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
| `HighsDomain` | 4,645 | ported (wiring pending) | `domain.rs`, `objective.rs`, `conflict_pool.rs` | row, cut-pool, objective and conflict-pool propagation; conflict analysis (`ConflictSet`, both entry points, reconvergence cuts); clique-table reasons explained; missing `tightenCoefficients`, `fixCol`, redundant-row recording |
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

**Stage 1b (2026-11-11), done.** `HS_DUMP_SEARCH` (`data/highs-parity/search/hs_dump_search.patch`,
on top of the queue patch; HiGHS 1.15.1) writes the queue events and the search's control
events to one file, in order, top-level search only:

| event | when | fields |
|---|---|---|
| `MI R\|B\|E` | main loop installs a node: root, best bound, best estimate | - |
| `MS`, `ML` | separation at the installed node; plunge-limit break | - |
| `MG` | after each main-loop round | global lower and upper bound, nodes |
| `SI` | `installNode` | depth, lower bound, estimate, bound changes |
| `SE` + `SEF` | each `evaluateNode` (recursive calls too) | result, depth, node lb/estimate/LP objective, open subtrees, branching decision, cutoff, optimality limit; fractional columns |
| `SB`, `SD0`/`SD`, `SK`, `SP` | `branch`, `dive`, `backtrack`, `backtrackPlunge` | result plus the same node fields after the call |
| `SKS` | each sibling tried while backtracking | depth, decision, node lb, pruned |
| `SPQ` | `backtrackPlunge`'s plunge-or-queue choice | depth, decision, node lb, node score, sent to queue |
| `SQ C\|O` | `currentNodeToQueue` / `openNodesToQueue` per node | depth, lb, estimate, pruned |

The dump does not change the search (mik-250-20-75-1: 718 nodes with and without). Over the 58
distinct instances of the population at 20 s (`hs_dump_search_coverage.md`): 99,279 open and
32,618 pruned evaluations, 54,425 branchings, 29,981 dives, 27,646 plunges (5,912 siblings sent
to the queue), 54,000 sibling decisions (11,480 pruned), 4,297 installs from the queue. Every
decision kind and outcome is observed except a node pruned inside `openNodesToQueue`
(`SQ O` with prune = 1: 0 of 8,926), which 1c covers with a unit test instead of a replay.

**Stage 1c (2026-11-11), done.** `bnb/highs_search/search.rs` translates the control of
`HighsSearch` (`installNode`, `evaluateNode`'s bookkeeping, `branch`'s child creation with
`branchDownwards`/`branchUpwards`, `backtrack`, `backtrackPlunge`, `dive`,
`currentNodeToQueue`, `openNodesToQueue`) and of `HighsMipSolver::run`'s node processing
(`installNodes`, `processNodes`, `pruneNode`, `separateAndStoreBasis`, `runHeuristics`, the
dive and plunge loop, the round's end). Every component call goes through `SearchEnv`,
including the node queue.

Replay (`replay_search.rs`): the control runs against a mock that answers each component
call with HiGHS's recorded outcome and requires HiGHS's decision wherever the dump records
one; the queue is the translated queue. Fixture `tests/data/highs_search_replay/` (six complete
solves: bell5, bppc8-02, flugpl, gesa2, mik-250-20-75-1, p0201): **0 mismatches, no event
left unconsumed**: 272 rounds, 1,580 evaluations, 902 branchings, 819 siblings tried, 495
plunge-or-queue choices (125 to the queue), 623 strong-branching commitments, 1,043 queue
states, plus the rarer paths (current-node hand-off, plunge limit, ancestor closed by
propagation, search ended by `checkLimits`). A node pruned inside `openNodesToQueue` (never
observed) is covered by `open_nodes_to_queue_prunes_without_queueing`.

What the replay taught (each found as a mismatch, then fixed in the translation or the
dump): HiGHS's root processing drives the search object before the main loop (the replay
starts at the first install); heuristic dives run a second search object in heuristic mode
(excluded from the dump); a recursive `evaluateNode` prints one identical record per level;
the child's bound is the chosen side's strong-branching bound, not the parent's; strong
branching commits to children inside selection (`branchDownwards`/`branchUpwards`, with
`skipDepthCount` and `depthoffset -= 1`), so selection returns steps for the control to
apply; and a round can end the search through `checkLimits` before its bound update.

**Not translated in stage 1** (components or unused at one thread): LP basis storage and
recovery, symmetry, conflict analysis, worker parallelism, restarts (`mip_allow_restart` was
off in every recorded solve; HiGHS's default restarts are a later addition).

**Next: stage 1d** (§6): the production `SearchEnv` over discopt's components with the
safety layer, the driver's search loop switched to `MainLoop`, and the batched search
deleted.

**Stage 1d, broken down** (the switch-over; each piece tested before the next):

| piece | content | gate |
|---|---|---|
| 1d-1 | local domain for the control: `HighsDomain::setDomainChangeStack` and `getReducedDomainChangeStack` added to the domain translation (`highs_cutgen/domain.rs`) | install/reduce round trip on random IPs (**done**) |
| 1d-1b | the rest of the domain's propagation that a node needs: `CutpoolPropagation` (cut activities, thresholds, `markPropagateCut`, the cut section of `propagate`, `getMinCutActivity`; the pool notifies its domains through `PoolObserver`) and `ObjectivePropagation` with `HighsObjectiveFunction` (objective lower bound, clique-partition contribution trees, the cutoff row for conflict analysis). The clique partition is taken as input until the clique table is ported (stage 7); conflict pools and conflict analysis follow with stage 7 | randomized dive tests: no feasible point (cuts) and no point under the cutoff (objective) removed, infeasibility only without such points, incremental activities and objective lower bound equal to recomputation through dives and backtracks; `std::partition` order checked against libstdc++ output (**done**) |
| 1d-2 | node evaluation behind `SearchEnv::evaluate`: `bnb/milp_driver/highs_eval.rs` translates `evaluateNodeImpl`'s component part over the local domain (propagate; flush the domain's box -- structurals plus activity-implied slack ranges, `domain_lp_box` -- to the working LP; LP solve; objective-propagation lower bound; integral vertex -> incumbent and `BoundExceeding`; cutoff; reduced-cost fixing + propagation; re-evaluation when columns changed). The LP and its safety layer are one function, `node_lp`, extracted from `solve_node` (NS bound, Farkas check, point and snap verification -> a `NodeLpVerdict`), so both searches trust the same outcomes; untrusted LPs leave the node open on its inherited bound. Estimate = LP objective until stage 5 | (a) the extraction is bound-neutral: status, nodes, objective and bound bits, LP iterations identical before/after on 400 random MILPs (8,800 nodes, largest tree 931); (b) `highs_eval_agrees_with_solve_node_and_never_cuts_the_optimum`, run on every node of real searches over 160 random pure-integer and mixed MILPs: same box -> verdict and LP bound bit-identical to `solve_node` (4,197 nodes, 0 mismatches); full evaluation with a cutoff just above the reference optimum -> never prunes a box holding it, never bounds above it, never removes it from the domain (2,988 evaluations, 569 with the optimum in the box, 0 violations; 1,222 re-evaluations, 4,515 reduced-cost fixings) (**done**) |
| 1d-3 | branching behind `SearchEnv::branch`: `bnb/milp_driver/highs_branch.rs` translates `branchImpl` + `selectBranchingCandidate` (reliability strong branching with HiGHS's iteration budget, `analyzeSolution`, the child-selection rules, the all-fixed fallback) and `computeLPDegneracy` on the local domain, against a `BranchPseudocost` interface (the stage-5 `HighsPseudocost`); probes solve through `node_lp`, so a child is pruned only on a verified ray or a safe bound, and an integral probe vertex becomes an incumbent only once verified. Commitments and re-evaluations are reported as `BranchStep`s; the final branching changes the local domain. A node that can be neither branched nor resolved keeps its valid bound in an unresolved floor (HiGHS declares it infeasible). `SearchEnv::branch` now takes the parent node; node evaluation records HiGHS's pseudocost observations | `highs_search_reaches_the_certified_optimum`: a whole HiGHS-design search (translated control, queue, local domain, evaluation, branching; no separation or heuristics yet) run on the batched search's root working LP reaches its certified optimum, closes its own gap and returns a point feasible on the original rows, bounds and integrality, on 120 random integer and mixed MILPs (4,083 nodes, 2,162 branchings, 88 strong-branching commitments, 1,876 strong-branching LP iterations); search-control replay still exact (**done**) |
| 5, 7 (parallel) | `HighsPseudocost` (`bnb/highs_search/pseudocost.rs`, used by branching and evaluation); `HighsCliqueTable` (`lp/highs_cutgen/clique.rs`: storage, `addClique` with fixings and substitutions, queries, propagation, `cliquePartition` feeding `ObjectiveFunction`, extraction from rows and cuts, subsumption, merging, separation, rebuild); `HighsConflictPool` and conflict analysis in the domain. Deviations recorded in each header: clique iteration in id order (HiGHS hashes); no re-entrant domain-to-table calls (possibly fewer fixings, never wrong ones). Upstream quirk reproduced and recorded: `ConflictPoolPropagation::linkWatchedLiteral` never links into an empty watch list (`head = linkPos` sits inside `if (head != -1)`), so a conflict propagates only when queued on addition or re-marked on backtrack -- sound, misses propagation; a candidate for a deliberate fix once replay is no longer the goal | pseudocost: hand-computed formulas plus a randomized master/worker test (37,380 checks); clique table: two randomized enumeration tests (400 binary programs each: no clique, fixing, implied change, partition or separated cut removes a feasible point; ~57k implied changes checked; a deliberately loosened threshold makes them fail); conflicts: randomized enumeration test (5,147 conflicts valid on every feasible point, 1.2M point checks, propagation never removes a point) (**done**, not yet wired into the search) |
| 1d-4 | `bnb/milp_driver/highs_env.rs`: the production `SearchEnv` and `HighsMipSolver::run`'s main loop -- queue, local domain, evaluation, branching with `HighsPseudocost`; time/node/open-node limits; HiGHS's optimality limit with suboptimal nodes kept in the bound; dual bound `min(incumbent, queue, unresolved floor)`; discopt's rounding heuristic at nodes; node separation in `separateAndStoreBasis`'s slot (node-local Gomory rounds + a search-local pool, lifting the node in the dive's re-evaluation); root node seeded with the driver's root-evaluation bound. Lazy, node, bound and debug hooks are not re-hosted: those models keep the batched search | route test through the entry point (150 random MILPs: optimum, bound, infeasibility, node limits) (**done**) |
| 1d-5 | the driver runs the HiGHS-design search once from its root working LP (`DISCOPT_HIGHS_MIP_SEARCH`, default OFF until r4) | **panel 2026-11-11** (`docs/dev/data/highs-parity/search/stage1d5_panel_r{1,2}.jsonl`, 62 rows, 20 s, 1 thread, sense-aware soundness 485 + 172 checks clean): default 40; new search on the default root 34; same HiGHS root (root CMIR/RENS/pool + presolve): batched 50 vs new 46, and 48 after node separation and the root-bound seed (lost: the three `neos-36*` rows, which the batched search certifies through cuts folded into the shared LP; gained `bppc8-02`). **Sound and complete, not yet better: the flag stays OFF and the batched search stays.** Next (improvements): cuts into the search's LP (HiGHS's separation persists in the LP -- stage 2/4), sub-MIP heuristics (stage 6), per-node basis storage |
| 1d-5 fold (falsified) | fold node cuts into the search's own copy of the working LP between rounds (surplus rows, rebuilt `ScaledProblem`/domain, basis extended; `fold_candidates >= 1`, up to 20 per fold, `max(50, rows/2)` in all, no deletion) | **falsified 2026-11-11, reverted**: on the six rows where the searches disagree it certified 0/6 (3/6 without); `nexp-50-20-4-2`: 13 cuts folded doubled the per-LP cost (26 vs 12 ms per node LP, 69 vs 413 nodes in 20 s, open vs certified). The batched search's fold survives on selection by HiGHS score, ageing and deletion; a fold without them costs more than it gains |
| 1d-5 r3 | four fixes after the r2 panel, each from a measurement: (1) `fit_basis_to_box` -- a basis kept across nodes refitted by reduced-cost sign (HiGHS's bound flipping; phase-1 pivots on `jijia` 230k -> 21k, ms per node LP 2.7 -> 0.85); (2) one-pass `getReducedDomainChangeStack` (same output; a quadratic chain walk held every stack sample of a 29.9 s `neos-1425699` solve); (3) the driver's objective lattice (`TreeManager::cutoff_for`, `round_bound_up`); (4) the driver's RENS/RINS in the `runHeuristics` slot. Entry experiment that redirected the work: the batched search certifies the `neos-36*` rows with `DISCOPT_MILP_CUT_LP=0` too, so folded cuts were not the difference | **panel 2026-11-11 r3** (`stage1d5_panel_r3.jsonl`, 62 rows, 20 s, 1 thread, 665 sense-aware checks clean): HiGHS root -- **new search 51 (367 s) vs batched 50 (393 s)**; default root -- new search 35 (609 s) vs default 40 (573 s): on the default root the new search's node bounds are weaker (`jijia` bound 102 vs optimum 107), where the batched search adds node FBBT and Gomory rounds on each batch's leading nodes. The HiGHS stack (HiGHS root + presolve + new search) vs today's default: 51 vs 40 certified, 367 vs 573 s |
| 1d-5 r4 (graduation) | the stack goes default-ON; the smoke suite under the new defaults exposed defects, each root-caused on a captured instance: (1) **false cuts** -- the ported `HighsTransformedLp::untransform`, its post-substitution cleanup and `HighsLpAggregator::getCurrentAggregation` drop `|v| <= small_matrix_value` with no rhs change; on the fac2 OA master a 6e-11 coefficient on the epigraph column (3.3e8, no upper bound) made 8 of 17 root cuts cut a verified feasible point (certified 376262250 vs 331837498). Now relaxed by the column's bound, or kept when that bound is infinite (`transformed_lp::drop_with_bound`; replays keep HiGHS's semantics under a test-only switch); (2) **false presolve bound** -- `aggregate`/`doubleton_eq` deleted cancellation leftovers `<= 1e-12` without relaxing the row; on a portfol_roundlot OA master (`n_j = a_j x_j`, integers to 1e5) the bound sat 4.6e-8 above a verified point (`cancel_entry`; parallel rows/columns made sound the same way); (3) a node-limited presolved incumbent returned with `x = 0`; (4) options the route silently ignored (`node_propagation = false`, `DISCOPT_MILP_DIVE_STRIDE`) now route to the batched search; (5) two profile counters double-counted nested sub-MIPs | **panel 2026-11-11 r4** (`stage1d5_panel_r4.jsonl`, 62 rows, 20 s, 1 thread, 248 sense-aware checks clean): **graduated 51 vs legacy 40 certified, 390 vs 567 s** (+11, -b-ball). **Graduated.** Still on the batched search: the debugger (re-hosted in 1d-6), lazy/node/bound hooks, non-default branching/selection, node propagation off, dive stride. Measured regression kept on record: OA on portfol_roundlot at 12 iterations takes 108 s (two masters ~49 s each to find an exactly feasible lot vector: the presolved master leaves no equality residual to exploit, and the HiGHS search finds such points slower than the batched one) |
| 1d-6 (debugger) | the debugger (`MilpDebugHook`, which also serves `node_callback`) runs on the HiGHS-design search instead of forcing the batched one. discopt's node identity (`NodeIdent`: id, parent, branching change) is carried on `NodeData` and the queue's `OpenNode`; the translated control hands out ids where it creates children, settles both children's identities when a node branches (the sibling's decision is HiGHS's deterministic flip, so it is reported with the parent), and reports `node_created` / `node_selected` / `node_closed` through `SearchEnv` (no-ops for the replays, which stay bit-identical). `HighsEnv`'s `Debugger` makes every evaluated node a batch of one: `IterStart` + `AfterSelect` (box = local domain intersected with the parent's reported box: a node reinstalled from the queue carries only HiGHS's reduced domain-change stack), `AfterProcess` when it branches or closes (lineage, children created since, its LP bound/point), `IncumbentFound`, `Terminated`; `Stop` ends the search uncertified. A committed strong-branching child is reported selected after its `Update` step. MILP presolve still does not run with a hook (the debugger's boxes are in the model's columns) | `debug_hook_runs_on_the_highs_search_with_a_consistent_tree` (route ran, tree consistent, nodes/objective identical to the unhooked solve); Python: #2043/#2049 lineage contracts, debugger neutrality and trace invariants, `node_callback` payloads -- 184 passed on the HiGHS route (a debugged market split: 6 search runs, 131 nodes selected and processed one for one). The bound hook (`MilpBoundHook`, `lagrangian_bound=True`) followed: called once per node on an open (fractional) evaluation over the node's local box, every `bound_hook_every`-th node, `max(LP, hook)` kept, past the cutoff pruned (`bound_hook_runs_on_the_highs_search`: silent hook neutral, a valid tight hook raises bounds and explores no more; Python `test_lagrangian_bb_hook.py` 17 passed with both arms on this search). Still on the batched search: lazy and node hooks (they add rows, which needs cuts in the search's LP -- step 2) |
| 2 entry (cuts in the search's LP) | hypothesis: the rows the default stack leaves open are lost in the tree, not at the root, and HiGHS's lever there is cuts that persist in the search's LP (added at nodes, aged out when non-binding, re-added from the pool). Evidence and kill criterion before building | **measured 2026-11-11** (`data/highs-parity/search/step2_entry.txt` and scripts): HiGHS certifies 8 of our 11 open rows, in 71-2547 nodes; its root gap equals ours (mik-1 1.60 vs 1.46 %, b-ball 4.74 vs 4.46, gsvm2rl3 23.9 vs 24.7), so the difference is the tree. HiGHS with cuts aged out immediately: **b-ball 102 nodes/0.2 s -> 92k nodes/8.2 % open, gsvm2rl3 4.9 -> 23.0 %** -- exactly our regime, so persistence is the lever there; mik/neos17 still close without it (534-2974 nodes; persistence cuts nodes 4-15x), so a second factor separates us on those. The batched search's existing fold (node cuts into a shared LP) gives no edge on these rows (r3: within 0.1-0.3 points either way). **Kill criterion**: with persistent aged cuts in the search's LP, b-ball must close or fall well below 4.5 % and gsvm2rl3 must drop > 5 points at 20 s; otherwise the implementation does not capture HiGHS's lever |
| 2.1 (persistent cuts, discopt's node GMI) | the search owns its LP once a node separates (`SearchLp`: the root working LP plus managed rows `a·x - s = rhs`, every row over the base columns by substituting cut slacks, so deletion never cascades); node separation's new GMI cuts and the violated pool cuts join it as persistent rows (<= 20 per node, cap `max(100, base rows)`); rows age on the LP's row duals (HiGHS `HighsLpRelaxation.cpp:660-673`) and are deleted between rounds past `CUT_LP_AGE_LIMIT` when their slack is basic, back to the pool; basis extended/remapped, LP caches cleared. `DISCOPT_HIGHS_SEARCH_CUT_LP`, default OFF | **measured 2026-11-11, kill criterion not met** (`step21_ab.txt`, 20 s, 1 thread): b-ball 4.46 -> 3.19 % (2789 rows added, 2628 aged out; HiGHS closes it in 102 nodes), gsvm2rl3 24.69 -> 22.63 % (needed > 5 points), mik-1 1.45 -> 1.34, mik-2 0.99 -> 0.92, neos17 0.51 -> 0.59 (2149 -> 432 nodes); no bound past an oracle; the full Rust suite passes with the flag forced on. Right direction on 4 of 5 rows, but this implementation does not capture HiGHS's lever. The obvious difference: HiGHS persists the cuts of its own separators (the ported tableau/path/mod-k, `lp/highs_cutgen`, through `HighsCutPool`) over several LP rounds per node; this persists only discopt's node-local GMI. Next experiment (2.1b): the ported separators (`bridge::separate_node`) at nodes into the persistent LP, same kill criterion |
| 2.1b (persistent cuts, ported HiGHS separators) | the same persistent LP fed by the ported tableau/path/mod-k separators at nodes (`DISCOPT_HIGHS_SEPA_NODE=1`, through `node_local_gmi_bound`'s HiGHS branch: several LP rounds per node, a fresh ported pool) -- no new code | **A/B** (`step21b_ab.txt`, 20 s): b-ball 4.461 -> **0.314 %** (kill criterion met), mik-1 1.369 -> 1.077, mik-2 0.960 -> 0.755, neos17 0.426 -> 0.369; gsvm2rl3 unchanged because the ported separators return **0 cuts** at its nodes. **Correction (§11):** gsvm2rl3 was a wrong probe for this mechanism -- HiGHS's own log shows its bound move at a restart (0.151 -> 0.283 at 9.3-11.9 s, 1485 conflicts and the pool carried over), and the entry experiment's no_persist arm also emptied what a restart carries; the criterion as written is reported not met there. **§5 panel** (`step21b_panel.jsonl`, 62 rows, 20 s, 248 checks clean): graduated 51 vs cut-LP 51 certified, 397 vs 398 s, nothing gained or lost; bound better on 6 of the 11 open rows (b-ball 4.46 -> 0.31, neos17 0.49 -> 0.30, mik-1/2/3, neos-911970 3.66 -> 3.27), worse on 2 by <= 0.16 points; certified-row wall geomean x1.013. **60 s** (`step21b_60.txt`): bounds better on 7 of 11 (b-ball 0.126, mik-2 0.72 -> 0.18, mik-1 0.97 -> 0.45, mik-5 0.70 -> 0.35, neos-911970 3.62 -> 2.11), still no new certificate; enlight_hard certifies in both (23 vs 30 s). Cert-clean; bound-positive; certification-neutral. **Not graduated**: §5's bar 2 is met on bound alone, and the combination needs `DISCOPT_HIGHS_SEPA_NODE`, whose default also changes the batched search -- a decision for the owner. Open: why the ported separators find nothing at gsvm2rl3's nodes; why b-ball at 0.13 % does not close where HiGHS closes in 102 nodes |
| 2.1c (H1: ported cut pool in the search; falsified, reverted) | hypothesis: the root-pool seeding (2ef4b8d5: b-ball 0.31 -> 0.69 % at 20 s) hurts because of the ad-hoc pick (top 20 by violation/norm), and HiGHS's own `HighsCutPool` selection would make it help. Built (`91f939a4`, `DISCOPT_HIGHS_SEARCH_POOL=highs`): the root's ported `CutPool` handed to the search with its ages and LP membership (root cleanup reports dropped cuts), aged 5x after the root (`HighsMipSolver.cpp:229-235`), `separate` per node (efficacy, `minScoreFactor`, parallelism 0.1, ageing), node cuts via `addCut`, `performAging` when a node does not separate, `lpCutRemoved` on every LP row deletion | **falsified 2026-11-11** (`step21c_h1.txt`, 20000 nodes, deterministic panel mode, 1 thread). Entry experiment first: at a fixed node budget seeding hurts only b-ball (0.925 -> 1.509 %; mik-1 0.470 % -> certified, mik-2 8097 -> 6827 nodes to certify), so the 20 s loss on mik/neos17 was throughput, not bound. H1: b-ball **2.226 %** (worse than both 0.925 unseeded and 1.509 seeded; with a fresh pool 1.689), mik-1 0.484 % (seeded certifies), mik-2 certifies in 10361 nodes (seeded 6827); neos17 certifies in every arm, its root and search are not deterministic at a node budget (seeded pool 2970/2532/2470, noseed 1530-6580 nodes across repeats). The path fired (b-ball 246 pool cuts selected, 752 LP removals reported; mik-1 14537/14163). The ported pool is far more selective (b-ball 807 cut rows added vs 4730 seeded / 3422 unseeded) and the bound follows the rows: what b-ball lacks is not a better pick among pool cuts. Kill criterion (b-ball beats unseeded, mik-2/neos17 not worse) not met; code reverted. Flag off: b-ball and mik-2 bounds identical between builds (mik-2 showed one 0.745 vs 0.628 outlier on the *unchanged* default path, not reproduced in 4 reruns, so the default MILP path is not bit-deterministic under `DISCOPT_PANEL_DETERMINISTIC=1`). **P0 candidate, unreproduced:** one neos17 `noseed` run (CUT_LP=1, SEPA_NODE=1, 2ef4b8d5 + seed toggle) reported `optimal` with objective 0.1500692 and a bound above the oracle 0.1500026 (dual-gap 0.0067 %, past tolerance 1.15e-5); 7 reruns of that arm were clean |
| 2.2 (H2: cut-pool propagation in the search's domain; falsified, not merged) | hypothesis: b-ball's decisive pool lever (HiGHS `mip_pool_soft_limit=1` -> 8.3 %) is HiGHS's `CutpoolPropagation` over every pool cut in the local domain, which the search lacks. Built and measured on `worktree-agent-a5fa88ccecedd5435` @ `e2701ee4` (ported `CutpoolPropagation` attached to the search's domain, pool cuts slack-substituted with an outward rounding margin; a `shadow` instrument propagating on a synchronised copy; 360-MILP feasible-point regression test) | **falsified 2026-11-11** (`step22_h2_pool_prop.txt`, b-ball 20000/99999 nodes): no pool cut ever passes the capacity test -- the pool arm is bit-identical to off (1.509 / 0.448 %). **Lead:** the ported objective propagation (`HighsDomain::ObjectivePropagation`), also not wired into the search, is the only arm that moves b-ball: 1.509 -> 1.148 % at 20k nodes, 0.448 -> 0.261 % at 99999 (24 / 42 % of the gap; kill line was half), no bound past the optimum, 0 nodes made infeasible. The pool-propagation code is not merged (§5: a falsified gate is retired, not kept); the objective-propagation arm needs its own hypothesis and panel. Side finding: node counts on certifying instances are not reproducible under `DISCOPT_PANEL_DETERMINISTIC=1` (suspected sub-MIP wall-clock caps) |
| 2.3 (H3: objective propagation in the search's domain; falsified) | hypothesis (pre-registered in `step23_obj_prop.txt`): HiGHS's `ObjectivePropagation` on the search's local domain, against the search's lattice-aware cutoff, cuts b-ball's gap at 20000 nodes by >= 25 % with no panel row (b-ball, mik-1/2/3/5, neos17, neos-911970, gsvm2rl3) worse beyond the off arm's 2-rep noise. Built as `DISCOPT_HIGHS_SEARCH_OBJ_PROP` (default OFF) | **falsified 2026-11-11** on the default search: b-ball 4.461 -> 4.581 % (deterministic), gsvm2rl3 67.7 -> 90.7 % (node LPs ~100x slower: 150 nodes 3.96 s -> 396 s, NodeLpSolve 1.8 -> 394 s); mik-5 -45 %, mik-3 -14 %, neos17 -33 % / neos-911970 -19 % (median, within noise). No bound past an oracle. Post-hoc (not pre-registered): in the 2.1b configuration (`CUT_LP=1`, `SEPA_NODE=1`) b-ball 1.509 -> 1.148 %, reproducing e2701ee4 -- the b-ball effect depends on persistent node cuts. Open: why objective propagation's continuous tightenings make gsvm2rl3's node LPs pathological  Code not merged (§5: falsified gate retired); it lives at `worktree-agent-adac17e31404b9441` @ `a42964be` for reproduction. |
| P0 fix (suboptimal nodes in the search's bound) | **false certificate on the default route** (the HiGHS-design search is default ON since 1d-5 r4): `run()`'s final bound and `after_round`'s gap test used `if q.is_empty() { inc }`, and `NodeQueue::is_empty` counts active nodes only, so once the active queue emptied the incumbent itself was reported as the bound, ignoring the queue's suboptimal set (nodes above the optimality limit, kept for the bound). HiGHS takes `min(upper_bound, getBestLowerBound())`, suboptimal set included. Fix: `HighsEnv::dual_bound(inc) = inc.min(lift(q.best_lower_bound())).min(lift(unresolved_floor))` at both sites (`worktree-agent-a2e4aa87d2f8ea02e` @ `8f1e8112`, cherry-picked) | Found from one unreproduced neos17 run (2.1c) and reproduced in 3 of ~60 runs (objective 0.150069 certified, oracle 0.1500026: the optimum's node sat in the suboptimal set at 0.149966, below the optimality limit 0.14995 + ...); deterministic repro with a seeded incumbent at 0.15005 and sub-MIPs off. An env-gated reference-point instrument (`DISCOPT_HIGHS_SEARCH_CHECK_POINT`, `highs_env/refcheck.rs`) checked 7100 cuts, every LP change, node/queue/branching bound and domain reduction in the failing runs: only the final bound was wrong. Regression test `highs_search_bound_counts_suboptimal_nodes` (fails before: optimum -231, bound -229). **Correction (§11):** the r4 graduation panel's "248 sense-aware checks clean" was true at its tolerance (`1e-6 + 1e-4·|opt|`) but blind to this defect: the engine closes its gap with denominator `max(|inc|, 1)`, so the false bound only crosses the panel's tolerance on `|obj| < 1` rows, and none of r4's did by enough. Expect some small-objective rows to drop from "certified" to "feasible" now; the default is re-measured below |
| Step 2 combination panel (after the P0 fix) | arms: the default (with the suboptimal-set fix) vs the step-2 combination `DISCOPT_HIGHS_SEARCH_CUT_LP=1` + `DISCOPT_HIGHS_SEPA_NODE=1` + `DISCOPT_HIGHS_SEPA_CUTOFF_BOX=1` | **panel 2026-11-11** (`step2_combo_panel.jsonl` + `.summary.log`, 62 rows, 20 s, 1 thread, 248 sense-aware checks clean): **default 51 certified, the same set as r4** (the P0 fix cost no certificate); combination **50** (-bppc8-02, which it leaves at 4.04 %), 391 vs 399 s. On the 11 rows neither certifies the combination's bound is better on 7 -- gsvm2rl3 24.7 -> **7.9 %** (HiGHS 4.9), b-ball 4.46 -> **0.11 %**, mik-1/2/3/5 (1.40 -> 0.98 best), neos-911970 3.86 -> 3.49 -- and worse on neos17 (0.31 -> 0.37). Bound-positive, one certificate short: **not graduated** (§5 bar 2); the flags stay OFF. Next: why bppc8-02 loses its certificate under the combination (throughput of the larger LP, or a separator effect), then re-panel |
| Step 2 bppc8-02 + HiGHS cut-row ageing | why the combination loses bppc8-02; hypothesis (§4): the per-node cost is my cut LP's churn, and HiGHS's own `performAging` (age only rows whose slack is basic, per-node ageing only of rows already ageing, an epoch per separation round's resolve, deletion every 5th epoch above `min(epochs, 10)`) removes it | **measured 2026-11-12.** Profiles (bppc8-02, 60 s, deterministic): the combination does *fewer* pivots than the default (75k vs 82k) but each costs more -- LU 254 vs 152 us/call, pivot loop 12.8 vs 8.1 s -- because the LP carries up to 100 cut rows on a 60-row model; the `SearchLp` rebuild is not the cost. HiGHS solves bppc8-02 at the root (0.36 s, 51 LP cut rows): the instance is a root-cut gap, not a search gap. Ageing port (`b6c4bc39`; unit test `search_lp_rows_age_as_highs_performaging`): churn falls (rows added 2747 -> 1981 at 60 s) but rows stay longer, LU 254 -> ~360 us, 2594 -> 5567 nodes -- **the churn hypothesis is falsified on bppc8-02**. Interleaved panel (`step2_ageing_panel.jsonl` + `.summary.log`, 62 rows, 20 s, deterministic, 186 solves, 0 violations): default 50, combination with HiGHS ageing **48**, with the old ageing **48** (they split 2122/blend2; bounds trade places on the open rows). The rule is kept as the faithful port (no measured difference); the combination stays OFF (-2 vs default). **Mode note:** under `deterministic=True` run 1 gets the caller's limit minus the 35 % #844 reserve (12.97 of 20 s) and no wall-clock re-entry (#1116), so this panel's solves had ~13 s; the earlier combination panel ran without the deterministic flag (full 20 s). Counts across the two panels are not comparable (p0201 certifies at 17 s, so it flips with the mode, not the code). Next: bppc8-02 is root cut strength (HiGHS closes it at the root) |
| Root cut strength (entry experiment; implied integers falsified, not merged) | question: is root cut strength what keeps the open rows open? Measured per row, before building (`data/highs-parity/root-strength/`, `root_now.py` + `compare.py`): discopt's root bound after cuts on today's default stack (root only, `max_nodes=2`) against HiGHS's root bound (`p3_gap_diagnostic.jsonl`, unchanged HiGHS build) | **2026-11-12: root strength is at parity, not the gap.** discopt's root is equal or better on 47 of 62 rows (`root_vs_highs.summary.log`). Of the 8 rows HiGHS certifies and we do not, ours is better on b-ball (4.58 vs 4.74 %), neos-911970 (3.28 vs 3.69) and enlight_hard (8.1 vs 58.1), and ~0.4 points behind on mik-1/2/3/5 and 0.14 on neos17 -- the mik gap is HiGHS's restarts (with `mip_allow_restart=false` HiGHS still certifies all four, 487-1,551 nodes), and §12.27 of the gap plan already put stripped HiGHS's mik root level within reach. The big root deficits are bppc8-02 (0.16 vs 4.04 %), gsvm2rl3, issue-2446: we certify two of them and HiGHS does not certify gsvm2rl3. **bppc8-02's root, traced:** HiGHS's 495 needs presolve AND its tableau separator (knock-outs: tableau off 487, clique/path off 495; presolve off 486.5); on HiGHS's written presolved model discopt's root is also 486.5; marking the one implied-integer column `h` (bin load, dual-implied) integer there takes **discopt to 504** and HiGHS to 494; on the original model with `h` integer both reach only 487, and on HiGHS's presolved model *without sparsify* (`presolve_rule_off=16384`) with `h` integer ours is 487 -- the gain needs sparsify and implied integrality together. HiGHS with sparsify off certifies the same set (`highs_sparsify_off.jsonl`; gsvm2rl3 at the 20 s edge 19.95/20.0 s), so sparsify is not a population lever and the one row it helps we already certify: not ported. **Implied integers, built and falsified:** HiGHS's `isImpliedInteger` (primal + dual detection, with dual detection counting only original/primal-implied integers and 1e-12 integrality) in MILP presolve behind `DISCOPT_MILP_IMPLIED_INT`; exact-oracle test over 20,000 random problems (347 checked, 0 violations). Entry result: root bound 486.5 -> 487 on bppc8-02 (objective rounding only), roots elsewhere move both ways by <= 0.6 points (`root_implied_int.jsonl`), and the full bppc8-02 solve **regresses** (certified 13 s -> bound stuck at 487 after ~1,650 nodes). Retired before merge (§5); the patch with its test is `implied_int_retired.patch`. **Found on the way (not root strength; next candidate):** with `h` integer and an infinite upper bound our node LPs cost 7.9 ms vs 1.6 ms with `h <= 6614` (`bppc8-02` + `h` integer: stuck at 487 vs certified in 7,203 nodes); dual degenerate pivots 135 vs 35 per warm solve, Bland activations 92 vs 12. HiGHS solves the infinite-bound model in 470 nodes. An integer column without a finite upper bound makes our dual simplex grind; unbounded general integers are common, so this is a class, to be measured on the corpus before any fix |
| Infeasible node LPs under dual degeneracy (the "unbounded integer" slowdown, root-caused) | the side finding of the root-strength row: bppc8-02 with its load column `h` integer and unbounded spends 7.9 ms per node LP (1.6 ms with `h <= 6614`) and stalls at 487 | **2026-11-12, fixed (bound-neutral).** Traced (`DISCOPT_LP_DUAL_TRACE`, now with entering/leaving indices and a `DUALPERTURB` line): 77 % of the dual pivots sat in 15-111 warm solves that ran to the warm stall cap (7,680 pivots, every one zero-length, Bland engaged for ~7,100) and then went cold. They are **infeasible** node LPs (the down-branches `h <= 486` below the LP value 486.5, hence more of them when `h` is integer), whose objective is `h` alone, so every reduced cost is 0 and the dual simplex wanders. The #1013 cost-perturbed attempt proved each one infeasible in 29-64 pivots, but `cost_perturbed_warm_solve` discarded every non-`Optimal` perturbed status -- right for unboundedness, wrong for infeasibility, which does not depend on `c`. Fix: a perturbed `Infeasible` whose Farkas ray re-certifies on the true `(A, b, l, u)` is returned (counter `DualCostPerturbInfeasible`); anything else is rejected as before, and the B&B caller re-verifies the ray before fathoming. The infinite bound was not the defect, only the trigger. Result on the reproducer: certified in 16.1 s / 5,965 nodes (was open at 487), 1.9 ms per node LP. Tests: the two #1935/#1945 audit tests now exercise their audits with the perturbation off (with it on, the warm solve proves those fixtures infeasible before any false `Optimal` forms) and assert the default path returns a certified `Infeasible`. **Bound-neutral panel** (`data/highs-parity/dual-perturb-infeasible/`, 62 rows, deterministic, 3,000 nodes, old path twice via a temporary build switch not in the tree): 0 violations; on all 59 rows that reproduce between the two old runs the new path is identical in status, nodes, objective and bound (neos-911970, 23588, neos17 differ between old runs). Fired on 4 rows (79 verdicts); single-run walls amur 18.2/18.5 -> 12.5 s, neos17 25.1/25.3 -> 20.8 s (directional, not a timing claim) |
| Time-boxed parity bet (stopped: kill criterion not met) | the 8 rows HiGHS alone certifies (mik-250-20-75-{1,2,3,5}, neos17, neos-911970, b-ball, enlight_hard). Kill criterion, set before measuring: one concrete difference, measured to account for most of the gap on >= 3 rows, or stop the parity push at 51/59 | **2026-11-12, stopped.** (`data/highs-parity/parity-bet/`.) (1) Matched node budgets (`matched.py`): the gap is per-node search quality, not throughput -- HiGHS closes b-ball in 102 nodes, mik-2/5 in 89/71, neos17 in 428, neos-911970 in 1,370; discopt's default is still 0.45-4.46 % open after 5,000 nodes. (2) HiGHS one-at-a-time knock-outs (`knock.py`, 11 arms): none stops HiGHS on any row; node separation is the only component that costs >= 7x nodes on 5 rows (mik-1 11x, mik-3 7x, mik-5 14x, neos-911970 16x, b-ball 11x); symmetry is irrelevant even on b-ball. (3) discopt's ported node separation (the step-2 combination) at the same budgets (`matched_combo.py`) moves the per-node bound most of the way on b-ball (4.46 -> 0.13 %), neos-911970 (2.52 -> 0.29), neos17 (0.45 -> 0.03), mik-2 (1.06 -> 0.05), but barely on mik-1/3/5 (-0.2 to -0.3 points), and costs 216-370 s per 5,000 nodes on the neos rows. (4) The cost has a different cause per row (`*_combo5k.summary.log`): neos17 -- infeasible cut-LP node LPs whose Farkas ray fails certification (109 exits) go cold and the primal phase 1 runs to the iteration limit (52 LPs, 2.36M phase-1 pivots, ~90 % of the time); neos-911970 -- the warm dual itself (111 s, 6 ms per solve); mik-2 -- no pathology. b-ball's HiGHS bound snaps -1.571 -> -1.5 at node ~101, consistent with objective-integrality rounding (one row, not followed). **No single build target spans 3 rows; per the criterion the HiGHS-parity push stops at 51/59.** Open, for whoever resumes it: node separation is HiGHS's dominant per-node lever and ours is sound; what blocks it is LP-engine robustness and speed on cut-augmented LPs (neos17's uncertified rays, neos-911970's dual), and mik-1/3/5's per-node gap with node cuts on |
| Uncertified rays on the search cut LP (parity push resumed, item 1) | the owner overrode the bet's stop ("parity before we move on"); item 1 of the resumed plan is neos17's uncertified-ray grind | **2026-11-12.** 8 of 8 uncertified exits captured (`DISCOPT_LP_DUMP_UNCERTIFIED`, `data/highs-parity/noise-rows/analyze_uncertified.py`) are on **feasible** LPs (HiGHS optimal, primal infeasibility 6e-17): the leaving row is a basic column 1.0e-9 / 1.96e-9 above a bound of 0, against the loop's 1e-9 tolerance; no eligible pivot repairs it, so the ray (g0 = -3e-9..0) cannot certify and the cold primal then ran to the iteration limit. `DISCOPT_LP_DUAL_NOISE_ROWS` (default OFF; `fad2eb7c`) tolerates such a row (<= 1e-7, HiGHS's primal feasibility tolerance); the bound stays the NS bound on the duals. neos17 combo, 5,000 nodes: 369 -> 78.5 s, uncertified exits 109 -> 0. Panel (`noise_panel.jsonl`, 20 s): default 51 = 51, combination 49 -> 50 (+blend2), 0 violations. **Parity unchanged at 20 s**: neos17 under the combination is still 0.42 % open (default 0.25 %): its per-node cost (78 s / 5,000 nodes vs ~28 s default) and node count (HiGHS 428) both remain. Next on the list: b-ball (combination at 0.05 %, HiGHS closes by objective-lattice rounding), neos-911970's dual speed on the cut LP, mik-1/3/5 |
| b-ball: objective lattice through inequalities (parity push, item 2; **graduated**) | b-ball maximises `x12` with `x12 <= x_k` and `2 x_k = sum of 8 binaries`: the optimum is a half-integer, but the lattice detector resolved only columns pinned by equalities. Generality first (§2): a static census of the 240-instance MIPLIB 2017 benchmark set (`data/highs-parity/obj-lattice-interval/latdetect.py`) finds the inequality-bounded pattern on 7 instances (assign1-5-8, CMS750_4, csched007, csched008, neos-4532248-waihi, neos-911970, ns1830653), beside 8 equality-only ones | **2026-11-12.** `interval_lattice` (exact rationals only; neighbours integral, fixed or equality-pinned, plus the row's own singleton slack as its sides; the model's own bounds, not root-propagated ones, which are rounded outward and would wrongly refuse) and the `optimum_only` cutoff (largest lattice point below an off-lattice incumbent). Exact-oracle test: 2,490 lattices hold at the true optimum; the plain cutoff would have fathomed the optimum in 105 of 73,077 checks, the corrected one in none. b-ball: certified at the root in 0.3 s (default and combination; was open at 20 s). Panel: 51 -> 52, 0 violations, graduated default-ON (`DISCOPT_MILP_OBJ_LATTICE_INTERVAL=0` opts out). **Parity: 52 / 59.** |
| Dominant per-node cost (parity push, items 1-3; sub-tolerance ladder falsified, not merged) | owner: "find and fix the dominant cost". Matched budgets (`data/highs-parity/subtol-ladder/`, deterministic): the base engine, not the step-2 combination, is the slow part -- neos-911970 27 ms/node (HiGHS 7.7), enlight_hard 14 ms/node (HiGHS 5); the combination adds 30-45 % | **2026-11-12.** neos-911970 (2,000 nodes): 491k primal phase-2 pivots vs 164k dual; 2,561 of 5,695 warm solves go to the #1953 sub-tolerance repair, ~130-190 primal pivots each, and a trace shows the repaired objective equal to the dual's to 1e-12 in 2,380 of 2,730 -- every repair triggered by noise-level "suspects" (open slacks, `d_j` ~ -1e-15 times infinite room makes the plain NS bound `-inf`). Built `DISCOPT_LP_SUBTOL_LADDER`: the check also tries the cheap rigorous rungs of the driver's node-bound ladder (implied slack upper bounds, open-column dual repair). neos-911970: repairs 2,729 -> 248, primal pivots 544k -> 57k, node-LP time 58.5 -> 26.4 s. **Panel** (`ladder_panel.jsonl`, 62 rows, 20 s): cert-clean, **52 -> 51** (-bppc8-02), blend2 14.9 -> 6.9 s, neos-911970's 20 s gap 3.2 -> 4.0 %. bppc8-02 explains it: with the ladder its repairs fall 3,580 -> 456 but it needs **52 % more nodes** (4,979 -> 7,586) and each warm dual costs more (0.83 -> 1.08 ms) -- the repaired vertex is a better warm start and branching point, so the repair's pivots are an investment, not waste. Not net-positive: retired before merge (§5), patch `subtol_ladder_retired.patch`. enlight_hard's dominant cost is different: the warm dual itself (48k of 64k node-LP pivots dual-degenerate, `DualPivotLoop` 9.0 of 13.8 s) |
| Dual degeneracy: perturb every warm dual from its first pivot (falsified, not merged) | the other dominant cost (enlight_hard: 48k of 64k node-LP pivots dual-degenerate). HiGHS perturbs costs up front; ours only after a 64-pivot degenerate run. `DISCOPT_LP_DUAL_PERTURB_FIRST` routes every warm dual through the #1013 perturbed solve plus true-cost clean-up (verdict-neutral by construction) | **2026-11-12.** Fixed budgets: neos-911970 node-LP time 63.1 -> 32.6 s (2,000 nodes), enlight_hard 7.4 -> 8.9 s (iterations 123k -> 72k, but two solves each). **Panel** (`perturb_first_panel.jsonl`, 62 rows, 20 s): cert-clean, **52 -> 51** (-p0201), wall on rows both certify 143 -> 159 s, neos-911970's 20 s gap 3.2 -> 4.1 %. Retired before merge (`perturb_first_retired.patch`). **Finding across this row and the ladder row:** two independent, sound ways of halving neos-911970's node-LP cost both leave its 20 s bound *worse* (3.2 -> 4.0 / 4.1 %), because each changes which optimal vertex the tree receives and the default's vertices branch better here. Per-node LP speed is not what binds the remaining rows; search quality per node is (as the matched-node measurement said: HiGHS closes them in 71-1,370 nodes). |
| **All-in HiGHS configuration (owner decision 2026-11-12)** | owner: "I am skeptical you are retiring things that HiGHS has and maybe that work for HiGHS" / "go all in on making this work". The per-component "net-positive alone" test was biased against HiGHS components, which compound (HiGHS survives every single knock-out on the hard rows; §12.29 of the gap plan said the same). **Method from here:** the HiGHS-ported components are judged together as one configuration; none is removed for being neutral or losing alone; when the configuration underperforms on a row, its behaviour is compared with HiGHS's on that row and the deviation is fixed. Removal only for unsoundness. | **Correction (§11):** the implied-integer and perturb-first rows above were retired on a per-component test that does not apply to HiGHS components; both are **restored** as members of `DISCOPT_HIGHS_CONFIG=1` (`crates/discopt-core/src/highs_config.rs`): `DISCOPT_HIGHS_SEARCH_CUT_LP`, `DISCOPT_HIGHS_SEPA_NODE` (replace), `DISCOPT_HIGHS_SEPA_CUTOFF_BOX`, `DISCOPT_LP_DUAL_NOISE_ROWS`, `DISCOPT_MILP_IMPLIED_INT`, `DISCOPT_LP_DUAL_PERTURB_FIRST`; HiGHS's sparsify is being ported to join them. Each member's own flag still overrides. The sub-tolerance ladder stays retired: it is not a HiGHS component (HiGHS has no such repair). The default is unchanged until the configuration beats it (§5 not overridden). |
| **Presolve noise-coefficient defect (P0, default path)** | found by the sparsify port's census (its rows carry fractional coefficients): the default-ON MILP presolve divided by, pivoted on and derived bounds through coefficients at the round-off level (~1e-16) that cancellation leaves on equality rows (where `cancel_entry` must keep them). discopt certifies under tolerance semantics; those rules reason in exact arithmetic and cut off tolerance-feasible points | **2026-11-12, fixed.** Reproduced without sparsify (`presolve_never_divides_by_a_noise_coefficient`: one 1e-16-scale entry on an equality row of random integer programs): before the fix **260 unsafe outcomes in 11,695 cases** -- reduced optima above the true optimum (e.g. 0 vs -1: a false bound nothing downstream catches) and postsolved points that pass the driver's check with a misreported objective -- after: 0 (11,693 exact, 2 caught by the driver). Fix: `Work::is_noise` (`|a| <= 1e-9 x` the row's largest): `doubleton_eq` and `aggregate` refuse such a pivot, both bound propagations and the implied-free test skip such an entry, a forcing row leaves such a column free, a singleton row whose coefficient is <= 1e-9 is left alone; and `solve_presolved` now also requires the postsolved point's objective on the original costs to equal the reduced solve's, else it solves the original. **Bound-neutral panel** (`data/highs-parity/presolve-noise/bound_neutral_panel.jsonl`, 62 rows, deterministic 3,000 nodes, old behaviour twice via a temporary switch): 0 violations, identical status/nodes/objective/bound on all 60 reproducible rows, the new check never fired. |

- **2026-11-12 -- config deviation on p0201 (warm flip + noise-row Farkas cost).** HiGHS closes p0201 in 5 nodes / 0.7 s; the configuration timed out (20 s) where the default certified at 17 s. Two measured costs: (a) 162 warm-start rejections, every one dual infeasibility on boxed columns only, each sending the node LP to primal phase 1 (62k pivots). HiGHS repairs these in `correctDual` by flipping the column; `DISCOPT_LP_WARM_FLIP_BOXED` joins the configuration: rejections 171 -> 0. (b) The noise-row branch (`dual_noise_rows`) ran 2,518 times per 100 nodes at 4.2 ms each -- 10.7 of 11.5 s of `DualPivotLoop` -- because an uncertified ray first went through the full Farkas ladder including the exact-rational rung. A noise-level row whose ray does not certify is tolerated anyway, so it now uses the floating-point rungs only (`farkas_ray_certifies_cols_float`; certifying through raw/cleaned is as rigorous as before, only the exact rung is skipped): `DualPivotLoop` 11.5 s -> 1.6 s per 100 nodes. p0201 under the configuration: certified 7615 in 6.1 s / 225 nodes (flip off: 10.3 s / 103 nodes; default in the same session: not certified at run 1's 13 s, 17 s in the last panel).

- **2026-11-12 -- config deviation on blend2 (factorizations per node LP).** blend2 under the configuration ran 24,320 LU factorizations for 7,841 node LPs (3.1 per LP) against 1.05 on the default, and `LuNumeric` was 10.9 of 20 s. Perturb-first built two LUs per LP that HiGHS does not: (1) it threw away the start factor already prepared on the true costs and re-prepared the same basis on the perturbed costs, and (2) it re-prepared the perturbed optimum's basis to clean up on the true costs. HiGHS perturbs and removes the perturbation on one factor (`HEkk::cleanup` recomputes the duals). Now `PreparedDual::repriced` re-verifies the start's preconditions on the perturbed costs with the existing factor, and `with_clean_costs` re-prices the verified-optimal final basis on the true costs inside the dual loop (x_B is cost independent; a dual-infeasible re-price falls through to the old clean-up). Test `in_loop_clean_up_matches_the_two_factor_clean_up` (200 random LPs: in-loop clean fired 191, all equal to the fresh-factor clean-up at 0 pivots; re-pricing equal to a fresh prepare 200/200). Only perturb-first (a config member) uses either path, so the default is untouched. blend2: 1.19 LUs per node LP, 9,121 nodes in run 1 vs 5,454 -- still not certified (incumbent 17.73 vs the default's 7.599; next).

- **2026-11-12 -- config deviation on blend2 (implied integers in heuristics).** One member off at a time on blend2: only `DISCOPT_MILP_IMPLIED_INT=0` certifies (562 nodes, 4 s; HiGHS 67 nodes, 2.2 s). Presolve marks 24 implied integers (the `PresolveImpliedInt` counter read 0 because it was incremented before the nested search reset the counters -- §11: an earlier "0 detected" in this investigation was that broken counter; fixed). We made them full integers everywhere; HiGHS keeps them in `integral_cols` (branching, cuts, propagation, reduced-cost fixing) but not `integer_cols`, which every primal heuristic and `checkSolution` use (`HighsPrimalHeuristics.cpp:44`, `HighsMipSolverData.cpp:165`). With them rounded/fixed by the heuristics, RENS/sub-MIP/randomized rounding found nothing and 9-11 incumbent snaps were refused. Now `MilpOptions::implied_int_cols` carries presolve's list and the heuristics (RENS/RINS fixings, randomized rounding, stall re-dive, rounding, dive repair, snap, polish, seed validation) use a true-integer mask; branching, node integer feasibility, cuts, FBBT, rc-fixing, the lattice and sub-MIPs' own types keep the full mask. Soundness: implied columns are continuous in the original model, and the postsolve feasibility/objective checks still run on every result. One real bug found and fixed on the way: with implied columns the lattice holds only at an optimum (an incumbent may be off it), so the lattice is forced `optimum_only` -- without that the new test `implied_integer_columns_are_continuous_for_heuristics_only` certifies 4.5 when the optimum is 4. Measured (config, 20 s, interleaved): blend2 timeout (18.35) -> optimal 7.598985 in 466 nodes, 3.3-3.7 s; bppc8-02 14.1 s -> 7.3-7.7 s; iskar identical (0 implied); **23588 15.3 s -> 17.9-24.6 s** (137 implied columns; heuristics now succeed but sub-MIP time 0.75 -> 4.9 s) -- the next deviation to chase. Under `DISCOPT_HIGHS_CONFIG=1` the lib suite has 7 failing tests, the same 7 on the unchanged tree (golden/basis-path tests written against the default); not yet triaged.

- **2026-11-12 -- config panel after the p0201, blend2 and implied-int fixes** (62 rows, 20 s, 1 thread, `data/highs-parity/config-panel/panel_2026-11-12_r3.jsonl`): `DISCOPT_HIGHS_CONFIG=1` **52** vs default **52** (was 49 vs 52, then 50 vs 52), total wall 348.0 vs 391.2 s, 0 violations. The configuration now certifies gsvm2rl3 (15.5 s), which the default does not, and loses only 23588 (default 14.2 s; config 20 s limit). Not yet ahead of the default, so it stays opt-in (§5); 23588 is the next deviation (sub-MIP time 0.75 -> 4.9 s since implied ints stopped breaking the heuristics).

- **2026-11-12 -- 23588 under the configuration (open; one hypothesis falsified).** HiGHS: 623 nodes, 13.0 s, root with one restart ~3.9 s, finds 8090 by node evaluation at 10.4 s, then closes at the restart on the integral objective. Default: 1,551 nodes, 13.8 s. Configuration: 1,272 nodes, 18.3 s once, then four runs in a row at the 20 s limit with incumbents 8090-8113 (sub-MIP budgets are wall-clock, so this row is not node-deterministic under the config). Phase deltas config - default: node separation +1.0 s (the ported tableau separator: 17 ms per call, 665 of 789 ms), LU +0.9 s (larger node LPs from the persistent search cut rows), dual prepare +0.9 s, c-MIR +0.3 s, sub-MIP +0.3 s. **Hypothesis falsified:** `separate_node` builds a fresh `TableauSeparator` every call, while HiGHS keeps one per solve whose `numTries` throttles it (`maxTries = 5000 + 50 numCalls + lpIters/10`). Persisting the counters (temporary global, A/B interleaved) changed nothing: ~42 node calls x ~100 tries never reach the 5,000 cap within 20 s, and separation time was 784 vs 874 ms. Not built. The gap on this row is spread across phases; the deciding difference is when 8090 is found (default earlier, HiGHS at 10.4 s).

- **2026-11-12 -- re-measured: uncertified Farkas rays on cut-augmented node LPs (neos17, 5,000 nodes, deterministic, 1 thread; `data/highs-parity/noise-rows/neos17_5k_remeasure_*.summary.log`, script alongside).** The parity bet recorded the combination (`SEARCH_CUT_LP` + `SEPA_NODE` + `SEPA_CUTOFF_BOX`) at ~370 s for 5,000 nodes with 109 uncertified rays, 2.36M phase-1 pivots and 1.99M EXPAND steps. Now: **combination 35.5 s**, gap closed at 2,332 nodes (`Phase1Pivots` 42k, `ExpandMinSteps` 26k, `DualExitFarkasUncertified` 91, `LpVerdictIterLimit` 50); **configuration 46.3 s**, gap closed at 4,347 nodes, **0 uncertified rays** (3,116 noise rows tolerated); default 37.9 s, node limit (38 uncertified, 30 iteration-limit verdicts). Explanation (noise-rows row above): the "infeasible" node LPs were not infeasible -- HiGHS solves the captured LP to optimality (`testdata/neos17_cut_lp_noise_row_lp.json`) -- the warm dual simplex stalled on a leaving row infeasible only at the noise level (<= 1e-7), whose ray cannot certify, and fell back cold into phase 1. The fixes since (certified perturbed-infeasible verdicts, noise-row toleration after an uncertified ray, the cheap-Farkas rung for such rows) remove the grind; no verdict is ever returned on an uncertified ray. The remaining 30-50 iteration-limit verdicts occur in the default as well (not cut-LP specific).

- **2026-11-12 -- 23588: restarts falsified, heuristic effort ported.** (1) HiGHS without restarts is *faster* on 23588 (`mip_allow_restart=false`: 8.75 s, 1,081 nodes, vs 11.3 s with them): restarts are not the lever. (2) Entry experiment: the configuration with sub-MIPs off certifies in 12.6 s, deterministic (1,258 nodes twice), and its top-level root cut loop is 1.1 s -- the "5.8 s root" was the sub-MIPs' own root loops. (3) HiGHS bounds node heuristics by LP-iteration effort (`moreHeuristicsAllowed`, 5 %); we had only a node stride. Ported as `DISCOPT_HIGHS_HEUR_EFFORT` (config member, `highs_search/heur_effort.rs`, sub-MIPs charged as `solveSubMip` charges them). (4) Per-call trace: our sub-MIPs on 23588 are the root RENS (2.0 s, with a nested RENS of 0.64 s) and one in-tree RINS (1.3 s, nested 0.24 s) -- the same schedule and similar cost as HiGHS's two root sub-MIPs (`L` at 1.5 and 2.5 s). So the remaining 23588 gap is not heuristic effort: the config's search takes ~1,630 nodes where HiGHS takes 1,081 at a similar cost per node, and the sub-MIP incumbents do not shorten it. With the gate: 23588 17.9-18.7 s (still above the default's 13-14 s, under 20 s), blend2 3.1 s, bppc8-02 6.3 s (was 7.5).

- **2026-11-12 -- config panel r4 (after `DISCOPT_HIGHS_HEUR_EFFORT`; `data/highs-parity/config-panel/panel_2026-11-12_r4.jsonl`): `DISCOPT_HIGHS_CONFIG=1` 53 vs default 52 certified, 348.7 vs 380.2 s total wall, 0 violations.** First panel where the configuration certifies more than the default: everything the default certifies (23588 at 19.9 s, at the edge) plus gsvm2rl3 (12.9 s). Before graduation, per §5: the 23588 margin (node count, next) and a repeat run.

- **2026-11-12 -- 23588 node count: no systematic deviation (measured, nothing built).** With heuristics out of both (ours `DISCOPT_MILP_SUBMIP=0`, HiGHS `mip_heuristic_effort=0`, no restarts) the counts are 1,258 vs 1,081. Root after cuts: ours 7738.8 (42 rounds) vs HiGHS 7747.05 (30 rounds, its `2 sqrt(maxTreeSizeLog2)` cap); presolve is identical (137 x 368, 137 implied integers); HiGHS's root does not depend on clique or implied-bound cuts (`HS_NO_CLIQUE`/`HS_NO_IMPLB`: 7747.05 either way). The root LP is massively dual degenerate: at the same objective 7649.866 our vertex and HiGHS's differ in 210 of 368 columns. **The root bound and the tree are chaotic in that vertex**, same code, only the root's starting basis changed (temporary loader, reverted): root 7738.8 (cold) / 7756.7 (our own basis reloaded warm) / 7737.2 (HiGHS's LP basis); full solves 1,258 nodes 12.7 s / 1,457 nodes 13.3 s / **4,311 nodes 30.2 s**. A stronger root did not shrink the tree. With sub-MIPs on, the count ranged 1,630-2,071 across runs (wall-clock sub-MIP budgets). HiGHS's 1,081 is one draw from its own equally sensitive process. Matching it would mean matching HiGHS's simplex vertex selection (its dual simplex pricing and perturbation), which is a port of the LP engine, not a fix of the search; recorded, not attempted.

- **2026-11-12 -- GRADUATED: `DISCOPT_HIGHS_CONFIG` default-ON in the MILP engine.** Repeat panel r5 (`data/highs-parity/config-panel/panel_2026-11-12_r5.jsonl`): configuration **54 vs 52** certified, 336.5 vs 374.8 s, 0 violations (r4: 53 vs 52, 0 violations); it certifies every row the default does plus gsvm2rl3 and enlight_hard. **Scope:** the graduation was measured on MILP solves, so it is scoped to the MILP engine. `highs_config::enabled()` (the engine's members) defaults ON; the four LP-level members in `SimplexOptions::default()` -- every LP outside the engine (spatial B&B, NLP-BB, Python LP bindings) -- stay off unless `DISCOPT_HIGHS_CONFIG=1` is explicit (`highs_config::lp_default`), and the MILP driver sets them on its own options (`dual::milp_lp_members`, at `solve_milp_node_search`'s entry). **Bound-neutral check of that scoping** (`data/highs-parity/graduation/`, 58 rows, deterministic 3,000 nodes, unset vs `=1` twice): identical status/nodes/objective/bound on all 55 reproducible rows, 0 violations (3 rows wall-clock noisy between the two `=1` arms). **Suites:** `cargo test -p discopt-core --lib` 1,136 passed with the switch unset and with `=0`; `pytest -m smoke` 4,718 passed; adversarial suite 19 passed. Tests that pin the pre-graduation engine name it: `cmir::LEGACY_DEFAULTS` now also reads the switch as off (golden pivot counts, the #2163 refusal probe, which also checks its bound property under the graduated engine); the dual-simplex fixture tests use `legacy_lp_opts`/`config_lp_opts` and the #2142/#2212 fixtures are additionally pinned to HiGHS's optimum under the configuration (2142: 0.2397017272 vs 0.2397017202); the OA exit-gate control (`test_oa_exit_gate_verifies_incumbent`) runs the previous engine in a child process, because under the configuration no budget 1-13 hands the gate a verifier-rejected point (measured). Opt-out: `DISCOPT_HIGHS_CONFIG=0`, or any member's own `=0`.

- **2026-11-12 -- after graduation: the remaining HiGHS-only rows (mik-250-20-75-{1,2,3,5}, neos17, neos-911970) are one class: bound progress per node.** On the 58 rows of `p3_gap_diagnostic.jsonl`: HiGHS 55, discopt 50 (+ gsvm2rl3, which only discopt certifies). Measured on mik-250-20-75-5 (HiGHS no restarts: 588 nodes, 4.3 s), all with temporary probes, reverted:
  - **Not the primal side.** Our root bound is *stronger* than HiGHS's (-52,363 vs -59,080), and seeded with the optimal point (`DISCOPT_MILP_SEED_POINT`, presolve off, sub-MIPs off) we still need 3,267 nodes (unseeded 4,071).
  - **Node separation is HiGHS's lever and ours is weak.** HiGHS with tableau/path/mod-k off: 94,930 nodes, unsolved at 60 s; each alone off: 4,374 / 3,484 / 2,694 nodes. Ours with node separation off: 5,087 vs 3,267 (1.6x, not >160x).
  - **Falsified:** HiGHS's `computeBasicDegenerateDuals` (knock-out `HS_NO_BDD`: 588 -> 622 nodes); more node rounds (`NODE_GMI_ROUNDS` 3 -> 20: identical); more cuts per node (200 instead of 20: 5,551 nodes, worse); a greedy 0.1-parallelism pick from the search pool (6,695 nodes, worse).
  - **Found (structural deviation):** our node rounds generate as many cuts as HiGHS's (tableau 26 / path 16 per call vs HiGHS 46 / 24; `HS_TRACE` vs a per-separator trace) but **select 1 per call** (259 calls, mean 1.0) from a fresh per-node `CutPool`, and the other ~40 are discarded; HiGHS keeps every generated cut in its global pool and selects ~10 per node round (157 rounds, mean 10.2) from everything accumulated. Retaining all generated cuts in the search's pool (probe): mik-5 seeded 3,267 -> **2,560 nodes**; under the real configuration at 20 s the bound improves on mik-2/3/5, neos17, neos-911970 and worsens on mik-1, no new certificate. The faithful form is HiGHS's single global `CutPool` serving both the root and the search (node cuts added to it, `CutPool::separate` over it at every node); the earlier "2.1c" attempt seeded such a pool from the root but discarded node-generated cuts, so it did not test this. Next build target.


- **2026-11-12 -- step 2.1d, one global pool for the HiGHS search (`DISCOPT_HIGHS_SEARCH_GLOBAL_POOL`; **retired before merge**, §5: the implementation is `data/highs-parity/search/step21d_global_pool_retired.patch`, the measurement `step21d_global_pool.txt`).** Built as the previous entry named it: the search takes the root's ported `CutPool` (ages and LP membership; the root cleanup now reports dropped cuts with `lp_cut_removed`), ages it five times (`HighsMipSolver.cpp:229-235`), and each node round runs tableau/path/mod-k **into** it over the global box (`separate_node_into`), then `CutPool::separate` over the whole pool at the node point and node bounds (`select_node_cuts`, `HighsSeparation.cpp:139-158`); selected cuts become search-LP rows, every row removal is reported, a node that does not separate ages the pool. The batched path keeps per-node pools. Sound (reference-point checker with HiGHS's optimum: 266,982 separator cuts and 15,644 search cuts/rows checked, 0 violations; 16,754 node bounds, 0 bad). **Not helpful as is:** mik-5 seeded 3,267 -> 16,935 nodes (uncertified at 60 s); selection stays at **1.33 cuts per round** (HiGHS 8.85) although 41.6 cuts are generated into the pool per round. Retention was not the binding constraint; **the cut population is**: in the first root round (same LP point, same 150 cuts accepted, same best score) 3 of 11,175 pairs of our cuts are <= 0.1 parallel against HiGHS's 8,325 (median parallelism 0.385 vs 5.8e-05), so HiGHS's 0.1 filter admits ~1. The separation boxes differ on 120 of 270 upper bounds (column 0: ours 3.0, HiGHS 239.8); `DISCOPT_HIGHS_SEPA_CUTOFF_BOX=0` does not change the rate. (The earlier probe's 2,560 nodes came from its ad-hoc top-20 pick, which has no parallelism filter.) 20 s panel, 9 rows interleaved: bound better on mik-1/2/3 and neos-911970, worse on neos17, nothing certified or lost. Next: which separator input makes our root cuts parallel (box, vbounds, aggregation choice) -- the same defect likely throttles the root: on mik-5 seeded 117 of its 118 rounds selected 1 cut (temporary trace) where HiGHS's seeded root selects 29, 49, 53, 40, 49, 41, ... (`HS_DUMP_POOL`).

- **2026-11-12 -- next lever: why our cuts are parallel.** Step 2.1d shows the selection, not the retention, is the limit: `CutPool::separate` admits ~1.3 of our cuts per round (HiGHS 8.85) because ours are nearly all mutually parallel (> 0.1). At the first root round on mik-5 seeded -- same LP point as HiGHS, same 150 cuts accepted, same best score -- only 3 of 11,175 pairs of our cuts are <= 0.1 parallel against HiGHS's 8,325 (median 0.385 vs 5.8e-5), and our root takes one cut in 117 of 118 rounds. The separation box differs on 120 of 270 upper bounds (column 0: ours 3.0, HiGHS 239.8). To find: which separator input (box, variable bounds, aggregation) makes the cuts parallel (`step21d_root_pool_{parallelism,box}.py`).

- **2026-11-12 -- the parallel-cuts lead traced, and its premise qualified.** In the seeded first root round, HiGHS's cuts are sparse (first cut: columns 0-20 and 195, 22 nonzeros) and ours carry ~44 more columns (99-191). Those 44 columns are **fixed at 0 in HiGHS's global domain** in that run and open (ub 1-4) in ours: HiGHS had applied root reduced-cost fixing against the seeded incumbent before separating, while our seeded diagnostics ran with `DISCOPT_MILP_SUBMIP=0`, which also disables our root reduced-cost block (it runs when `DISCOPT_HIGHS_SEPA_PROPAGATE` or root RENS is on). So the parallelism comparison was partly a setup artifact. Checks: (a) ours seeded with `DISCOPT_HIGHS_SEPA_PROPAGATE=1` (root FBBT + reduced-cost fixing each round, 27 rounds propagated): 4,958 nodes vs 3,267 -- worse; (b) HiGHS unseeded knock-outs on mik-5: root reduced-cost fixing off 334 nodes (vs 588), node reduced-cost fixing off 1,589, both off 734 -- reduced-cost fixing is not HiGHS's lever here (separation is: >160x). **State of the six rows:** the deviation is in what our node separation produces per round under real (unseeded) conditions; the instruments built for it (per-separator counts, pool selection counts, the seeded comparison) are recorded above; the unseeded per-round cut comparison is the next measurement.

- **2026-11-12 -- mik-250-20-75-5 unseeded, graduated default (20 s; temporary per-separator trace, reverted).** Two gaps, the first the larger: (1) **separation frequency** -- 44 node separation rounds in 3,348 nodes (1.3 %) against HiGHS's 157 main-loop installs, each separated, in 588 nodes (27 %); our search plunges far longer between queue installs (seeded: 247 rounds in 3,267 nodes, 7.6 %); (2) **cuts per round** -- 2.9 selected on average (24 of 44 rounds select 1; tableau 33.9 and path 7.3 generated per round) against HiGHS's 10.2. The plunge-or-queue decisions (`nodeToQueue = nodelb > optimality limit`, `backtrackPlunge`'s ancestor-score comparison) are replayed bit-exact against HiGHS given the same inputs (`replay_search.rs`), so (1) comes from the inputs: the incumbent's arrival (HiGHS has the optimum at node 129) and the pseudocost scores. Next: per-node trace of `backtrackPlunge` decisions (SKS/SKP records) on both sides on this row to see which input makes ours keep plunging.

- **2026-11-12 -- mik-5 plunge decisions (temporary trace, reverted).** HiGHS (`HS_DUMP_SEARCH`): 169 main-loop installs (164 best-estimate), 175 `backtrackPlunge` decisions of which 62 send the node to the queue (`nodelb > optimality limit`), plunge limit hit once. Ours with sub-MIPs off (600 nodes): 212 decisions, 1 to the queue by the optimality limit, 55 by the ancestor-score switch, 6 separation rounds. The cause there is the incumbent: ours is -27,365 (optimality limit ~-27,237) while HiGHS holds the optimum -51,532 from its second root sub-MIP at 1.0 s, so almost no node clears the limit and the search plunges. With sub-MIPs on, our 20 s incumbent is -50,684 (HiGHS's optimum at 1.0 s). So on mik the primal side (how fast the optimum is found) and the per-node side (seeded: 3,267 vs 588) compound. Next, in order: (1) why our root sub-MIPs do not find -51,532 at the root as HiGHS's second `L` does (same RENS/RINS schedule, different result); (2) the per-node gap with the optimum known.

- **2026-11-12 -- in-tree sub-MIP schedule (falsified as the primal lever).** Our in-tree RENS/RINS runs at most every `SUB_MIP_BATCH_STRIDE * 64` = 1,024 nodes (backing off to 4,096); HiGHS calls RINS at every heuristic slot its effort budget allows (mik-5: RINS at nodes 108 and 129 find -50,872 and the optimum -51,532). With the stride removed under the graduated configuration (temporary switch; the HiGHS effort gate still applies), 20 s, the six rows run 2-5x more sub-MIPs but: mik-5 incumbent unchanged (-50,684), mik-3 -49,788 -> -51,906 (opt -52,242), mik-1/2 equal, neos-911970 worse (58.09 -> 60.69), no certificate; bounds worse on several rows. So the sub-MIPs run often enough; what differs is what they find. Next: run our RINS on HiGHS's node-129 neighbourhood (its incumbent and LP point from `HS_DUMP_SEARCH`) to compare the sub-MIP outcome directly.

- **2026-11-13 -- sub-MIP presolve (`DISCOPT_HIGHS_SUBMIP_PRESOLVE`, config member).** Root profiles of neos17 / neos-911970: 5.8 s of each root is 4 sub-MIP calls (HiGHS's root is done by ~1.3 s), and our sub-MIPs skipped the MILP presolve (gated to the top level) while HiGHS presolves every sub-MIP (`solveSubMip`). With it: mik-250-20-75-5's root finds the optimum -51,532 (was -39,884), its root sub-MIP time 3.0 -> 1.7 s, its 20 s bound -52,146 -> -51,961 (oracle -51,532); neos rows unchanged within noise. Panel: 53 = 53, 348.7 vs 356.9 s, 0 violations. Kept as a member (HiGHS component, sound, neutral-to-positive). Not yet ported from `solveSubMip`: the stall limit (`mip_max_stall_nodes` 12) and leaf limit (500), and presolve for seeded RINS (HiGHS passes no seed; it uses `objective_bound`).

- **2026-11-13 -- sub-MIP work limits (`DISCOPT_HIGHS_SUBMIP_LIMITS`, config member).** Ported `solveSubMip`'s 500-leaf and 12-stall-node limits (the stall check at install, `HighsMipSolver.cpp:641-650`). On neos17, neos-911970, mik-1, mik-5 (20 s) the stall rule fired 0 times and results are within noise: our sub-MIPs' ~1.45 s each is spent in their roots (5.8 s of `RootCutLoop` on neos17 is almost all inside the 4 sub-MIPs), not their trees. HiGHS's sub-MIPs also run root RENS (nesting is faithful), so the open question is the per-sub-MIP root cost. Kept as a member (faithful, sound, neutral).

- **2026-11-13 -- root time on neos17 / neos-911970 (temporary trace, reverted).** Top-level root 3.3 s / 4.4 s (HiGHS ~1.3 s / ~1.7 s), of which ~2 s is two sub-MIPs of ~1 s each (presolved RENS 331x621 / 380x418; seeded RINS over the cut LP 511x684 / 888x266, not presolved), each with a nested RENS of ~0.1 s. HiGHS's root sub-MIP on neos17 takes ~0.8 s (`L` at 1.3 s after ~0.5 s of separation), so per sub-MIP the cost is comparable; we run two at the root where HiGHS's log shows one before the tree. The larger gap on both rows is again the tree: ~1,800 / ~500 nodes in the remaining 17 s without closing, against HiGHS's 428 / 1,370 nodes to certify. All six open rows are therefore one class -- bound progress per node -- with the inputs measured so far (separation frequency 1.3 % vs 27 % of nodes, 2.9 vs 10 cuts per round on mik-5).

- **2026-11-13 -- branching inputs on mik-5 (temporary traces on both sides, reverted).** (1) LP degeneracy factor (`computeLPDegneracy`; >= 10 disables strong branching): 1.0 at every branching on both sides (HiGHS 670, ours 629) -- not the difference. (2) `minreliable` is 8 throughout ours and the strong-branch iteration budget never binds (2,341 of ~106,000). (3) **Retracted (§11):** an intermediate comparison of "strong-branch probes" counted HiGHS's `SBD` dump records (364) against our probe LPs (186) and read it as "half the strong branching"; `SBD` is `branchDownwards`/`branchUpwards` (dive/branch steps), not a probe, so that comparison was void. HiGHS's probe count is not in the dump. Our probes cluster in the first 50 branchings (158 of 186) and then nearly stop as pseudocosts become reliable, which is the expected reliability behaviour; whether HiGHS's differs needs a probe counter on its side.

- **2026-11-13 -- strong branching on mik-5, counted properly (temporary counters in HiGHS's `selectBranchingCandidate`, reverted).** HiGHS: 1,651 probe LPs over 670 selections in 588 nodes; ours: 186 over 397 branchings at the same node count. The difference is trajectory, not the reliability rule (minrel 8 on both; the port's `isReliable` matches): HiGHS's candidates are 94-100 % reliable in its first 100 selections (238 probes, then ~0.5 per selection), then reliability collapses to 3-30 % between selections 100 and 325 -- right after its in-tree RINS finds -50,872 and the optimum -51,532 (nodes 108 / 129) and the search moves to a region whose fractional columns were never sampled -- and it probes ~2.6 per selection there. Ours stays reliable after its first 50 branchings (158 probes) because its search stays in the first region, consistent with its weaker incumbent. So the strong-branching gap is downstream of the primal gap (when the optimum is found), not a separate deviation.

- **2026-11-13 -- mik-5 under the graduated default: presolve, seeds and the per-node gap.** 60 s, graduated default vs ablations: default 9,722 nodes, uncertified (bound -51,719); presolve off 6,209 uncertified; sub-MIPs off 17,380 uncertified; **presolve and sub-MIPs off: certified in 42.9 s, 4,071 nodes**; `DISCOPT_HIGHS_CONFIG=0` 20,201 uncertified. Dominated column pairs are not the cause (skipping them: 17,163 / 10,370 nodes). The presolved root has the *better* bound (-53,083 vs -53,624) but its root incumbent is -26,594 against -51,206 on the original model, with no heuristic counter differing: the root runs 150 vs 119 cut rounds, its LP points differ, and so do the rounding heuristics' results -- the same root-path sensitivity as 23588. HiGHS seeded with the optimum (`--read_solution_file`): 482 nodes, 5.5 s. **Correction (§11):** a "seeded" run of our default reported earlier in this session (12,698 nodes at 90 s) was not seeded -- `DISCOPT_MILP_SEED_POINT` is read at driver depth 0 and a presolved solve searches at depth 1, so the seed was ignored; the valid seeded comparison is presolve off: 3,267 nodes vs HiGHS's 482 (6.8x), the per-node gap. Minor finding on the way: sub-MIPs launched from a presolved top-level solve read depth 2 and are budgeted as nested (half the time left instead of a quarter, `try_rens_highs` `nested`).

- **2026-11-13 -- fix: root sub-MIPs of a presolved solve were budgeted as nested.** `try_rens_highs` decided "nested" by `MILP_DEPTH > 1` ("the top-level driver reads 1"); since the MILP presolve the top-level search runs at depth 2, so every root RENS/RINS of a presolved solve got the nested share (half the time left instead of a quarter). Now `in_sub_mip()` (launched from inside a heuristic sub-MIP), the documented intent. Panel (`data/highs-parity/submip-presolve/nested_share_panel.jsonl`, 62 rows, 20 s): 53 = 53, 329.0 vs 330.2 s, 0 violations -- neutral, kept as the intended behaviour.

- **2026-11-13 -- mik-5 bound vs nodes, both seeded with the optimum (sub-MIPs off on ours, presolve on; HiGHS `--read_solution_file`, `mip_max_nodes`).** Global dual bound at 50 / 150 / 300 / 480 nodes: HiGHS -60,336 / -59,983 / -56,169 / **-51,742** (closes at 482); ours -53,590 / -53,516 / -53,464 / -53,203. HiGHS keeps a few weak open nodes (its global bound is the weakest of them) while it plunges, mostly from best-estimate installs, and its subtrees prune; at ~480 nodes the remaining weak nodes close at once. Ours advances the global bound steadily from a stronger start but prunes subtrees far less. This again points at node-LP strength (node bounds that clear the cutoff), observed only through the trajectory so far. **Next experiment (matched boxes):** take node boxes from HiGHS's `HS_DUMP_SEARCH` (`QE`/`SI` records carry the node's bound changes), solve our node LP plus node separation on the identical box, and compare the resulting bound with HiGHS's recorded node bound -- this isolates per-node LP and separation strength from the search trajectory.

- **2026-11-13 -- matched boxes on mik-5: our node bound is weaker on HiGHS's own boxes, and in the seeded setup the gap is the root cut LP (`data/highs-parity/matched-box/`).** Instrument: HiGHS dumps every top-level node-LP solve (`NBE`, box before `resolveLp`) and every node separation (`NBS`, box, LP objective before and after `separate`) with the full local column box in `%a` (`HS_DUMP_NODEBOX`, `highs_nodebox.patch`; with the dump, 397 nodes and 47,399 LP iterations, as without). Ours: `DISCOPT_NODEBOX_FILE` (`highs_env/nodebox.rs`) evaluates each box at the start of the top-level search through the search's own `HighsEnv` (built as `run` builds it: root cut rows, seeded pool, incumbent): (a) `evaluate` (propagation, node LP, NS-safe bound, reduced-cost fixing), and for `NBS` boxes (b) `separate` then the re-`evaluate` the dive does. Both seeded with the optimum, presolve off (ours also sub-MIPs off); 270 columns in MPS order on both sides (asserted); 1,237 `NBE` + 37 `NBS` records (HiGHS separates at 37 of 397 nodes). Comparison `compare.py` (1,311 comparisons), rows `mik5_*.jsonl`, summaries `summary_mik5_*.txt`, `run.sh` regenerates all.
  - **Matching.** No HiGHS box is empty or infeasible for us; no untrusted LP. Every box has 20 columns outside our global box, all on the HiGHS-looser side: the continuous columns 0-19, upper 220-354 in HiGHS, 1-20 in ours (ours is tighter, so that only helps us). HiGHS's 102 root fixings are in its boxes, so they are in ours too; our root LP gives -53,631.7 on HiGHS's root box and on our own global box alike. HiGHS's LP stops at its objective limit (-51,532.000001, ours -51,532.0): 130 `NBE` values are only known to be at or above it. HiGHS's values are its LP objective, ours the certified bound (they differ at 1e-6 scale here). Different cut sets are part of what is measured: HiGHS's LP is the one its trajectory built (273 rows at the root, median 375); ours starts from the root's 274 rows (195 + 79 cuts).
  - **(a) vs HiGHS's node LP, our root LP on each box (`fresh`):** ours weaker on **1,189 of 1,237** boxes, equal on 48 (both at the limit), stronger on none; median shortfall **-821** (q1 -1,074, q3 -486). Root box: HiGHS -52,422.7, ours -53,631.7 (-1,209). The LP prunes 130 boxes for HiGHS and 49 for us (48 of HiGHS's 130; on its other 82 ours is a median 341 below the limit).
  - **Before vs after separation (`NBS`, `fresh`):** (a) vs HiGHS pre-separation 37 of 37 weaker, median -987; (b) vs HiGHS post 36 of 37 weaker, median -1,042. Separation gains median 155 for HiGHS, 145 for ours. **So in this setup the gap is before separation, and it is the root cut LP:** our node separation adds about as much per call.
  - **`cumulative`** (one environment, our cuts separated at HiGHS's 37 separation boxes and kept as aged rows, in HiGHS's order): `NBE` median -329, `NBS` pre -119, post -227; our separation gains 82 against HiGHS's 155 once rows accumulate (ours median 354 rows, max 489).
  - **The root gap is a setup artifact of the seeded baseline.** `DISCOPT_MILP_SUBMIP=0` also disables our root reduced-cost block (the 2026-11-12 parallel-cuts entry), so our root cuts are separated over a box without the reduced-cost fixings HiGHS's root applies against the seeded incumbent before separating. Control with `DISCOPT_HIGHS_SEPA_PROPAGATE=1` (root FBBT and reduced-cost fixing each round, 275 root rows): root box -52,368.2, **54 stronger than HiGHS**; `NBE` median shortfall **-95** (1,072 weaker, 81 equal, 84 stronger; LP prunes 93 vs 130); `NBS` pre -95, post -180; separation gain 81 vs 155. With the root matched, the node gap is about 95 before separation (the LP HiGHS carries into the node: 375 rows vs our 275) plus about 85 that our node separation adds less than HiGHS's.
  - **Open contradiction (to resolve before building):** the same `SEPA_PROPAGATE=1` seeded run took 4,958 nodes against 3,267 without it (2026-11-12), although its node LPs are much closer to HiGHS's here. So the 3,267-vs-397 node gap is not explained by node-LP strength alone. Next: the matched-box run under `SEPA_PROPAGATE=1` with the search's own trajectory (which boxes our search visits and prunes), and the separation shortfall (81 vs 155 per call) on matched boxes.
  - **Harness caveats.** At `max_nodes` 1 the driver stops before the search, so a first run evaluated nothing from the MILP: the harness fired in a later top-level driver call that solves the integrality-free relaxation (every LP point "integral", 195 rows). It now skips a call with no integer column (`NODEBOX_SKIP`) and prints `call=` and `n_int`; `compare.py` refuses two `NODEBOX_BEGIN` blocks. The harness runs before the search on environments of its own: with it active (cumulative) the seeded full solve still takes 3,267 nodes, bound -51,536.906, as without; the `run` refactor it needed (`HighsEnv::new`, `seed_pool`) without the harness gives 3,267 nodes, the count recorded before it.

- **2026-11-13 -- node separation shortfall: per-node caps measured, step 2.1d re-run in the real default and retired again.**
  - **Matched boxes (root RC fixing on, `fresh` mode):** our separation gains a median 81 vs HiGHS 155 because two per-node caps bind -- `SEARCH_CUTS_PER_NODE` (20 rows added) and `NODE_GMI_POOL_PER_NODE` (20 pool picks); every separation added exactly 20 rows (275 -> 295) where HiGHS added 59-76. Rounds (`NODE_GMI_ROUNDS` 3 -> 10) and the new-cut cap alone change nothing. Both caps lifted: gain **184** (HiGHS 155), post-separation gap -180 -> -85.
  - **In the search the lifted caps lose:** 20 s, six rows, all worse (nodes down 1.5-3.5x). At equal nodes (1,500) worse too while the total search-row cap (`search_cut_cap`, `max(100, base rows)`) stays -- later nodes find no room; with it lifted as well, better at equal nodes (mik-2 -51,066 vs -51,236) but the LP swells (mik-2: 6,188 rows added / 5,850 deleted over 19 ageing epochs, `LuNumeric` 188 of 264 s for 1,500 nodes). HiGHS keeps ~200-330 cut rows via `CutPool::separate` (~10 per round, 0.1 parallelism) and ageing per separation round.
  - **Step 2.1d (global pool) re-run in the real default** (sub-MIPs on, so root RC fixing on; patch rebased onto the current tree, `data/highs-parity/search/step21d_global_pool_retired.patch`, logs `step21d_rerun/`): the retirement reason is falsified -- the pool now selects 12.5 (mik-2) / 9.2 (mik-5) cuts per round (HiGHS 8.85; was 1.33 under `SUBMIP=0`) and the LP does not swell (`LuNumeric` flat / -23 %). But it is not net-positive: 20 s, 4 of 6 bounds better (mik-1/2/3, neos-911970), mik-5 and neos17 worse, no certificate; equal nodes (1,500) 3 better (mik-2 +115) / 3 worse (mik-5 -175, with a worse incumbent); 0 soundness failures in 28 solves. Named follow-up (global pool + total row cap lifted, since 45 % of mik-2's selections were refused by it): equal nodes mik-2 -51,099 (better), mik-5 -52,205 and neos17 0.1426 (worse), the rest within noise -- kill criterion met. **Retired again (§5 bar 2), now on real-default measurements.** What remains: separation runs at 2-4 % of nodes (HiGHS separates at every queue install, 27 %), which is the search-trajectory gap recorded above, not a separator deficit.

## 8. Close-out (2026-11-13)

The parity push stops here. This section is its final state; §7 is the record that led to it.

### 8.1 What shipped

- **One engine, HiGHS-design search core**: `DISCOPT_HIGHS_MIP_SEARCH`, `DISCOPT_MILP_PRESOLVE`
  and `DISCOPT_MILP_ROOT_CMIR=highs` graduated 2026-11-11 (stage 1d-5 r4 panel). The ported
  separators, cut pool, aged search-LP rows, node queue/plunge, heuristic-effort budget and
  sub-MIP presolve/limits run on discopt's own simplex and safety layer.
- **`DISCOPT_HIGHS_CONFIG` graduated default-ON for the MILP engine** 2026-11-12 (two panels,
  53 and 54 vs 52 certified); every member's audit row in `flag-retirement-audit.md` now leads
  with its final state (graduated; `DISCOPT_HIGHS_SEPA_NODE=add` kept as a documented
  alternative). Retired with their implementations on the way: `HIGHS_SEP_FIRST`,
  `HIGHS_TREE_RCFIX`, `HIGHS_ROOT_POOL=all`, step 2.1d's global pool (twice).
- **Final-tree validation** (`800a3b45` + this commit's docs; `data/highs-parity/closeout/`):
  62-row panel (20 s, 1 thread) graduated **53** vs `DISCOPT_HIGHS_CONFIG=0` **52** certified,
  327.3 vs 369.8 s total wall, 124 solves, **0 violations**; `pytest -m smoke` 4717 passed,
  15 skipped; adversarial suite 19 passed; `cargo test --release -p discopt-core --lib` 1141
  passed, 0 failed.

### 8.2 The 62-row panel: 53-54 vs HiGHS 59

The six HiGHS-only rows (mik-250-20-75-{1,2,3,5}, neos17, neos-911970) are one class --
bound progress per node -- driven by a path-dependent chain (degenerate LP vertex -> which cuts
-> when the incumbent arrives -> plunge length -> separation frequency). About a dozen
experiments after graduation (§7, 2026-11-12/13: restarts, heuristic effort, in-tree sub-MIP
schedule, strong-branching counts, branching inputs, plunge decisions, matched-box node LPs,
per-node separation caps, the global pool re-run) each removed a measured difference and gained
**zero** rows. No single deviation explains the rows; they are not closable by another lever.

### 8.3 Population check: the panel understates the gap

To test whether 54 vs 59 is representative, a second, independent population was fixed
**before any solve** (`data/highs-parity/population/select.py`): MIPLIB 2017 `easy-v2.test`,
minus the 62-row panel, with an `=opt=` value in `miplib2017-v31.solu`, ordered by
sha1(name), the first 120 whose `.mps.gz` is at most 1 MB. Ours (graduated default) vs highspy
1.15.1 (`threads=1`, `time_limit=20`, `mip_rel_gap=1e-4`, everything else default), 20 s, two
solves in parallel (`run.py`; `population_panel.jsonl`, `.summary.log`).

| | certified (of 112 readable) | no incumbent | total wall |
|---|---|---|---|
| discopt | **9** | 53 | 2446.8 s |
| HiGHS | **26** | 19 | 1880.1 s |

0 violations over 224 solves (certificates and bounds against the MIPLIB optima, our
incumbents' max row violation <= 1e-5). 8 rows were unreadable by **both** readers (SOS/indicator
sections, coefficients >= 1e15) and are excluded. Both certify 8; HiGHS-only 18; ours-only 1
(decomp2). On rows neither certifies HiGHS has the better incumbent on 34 vs 7 and the better
bound on 32 vs 8.

So on a population that was not used for tuning the engine certifies about a third of what
HiGHS does. The 62-row panel has been the tuning set for months and is not representative.
The 18 HiGHS-only rows fall into classes different from §8.2's:

- **Root never finishes (6 rows: neos-827175, neos-913984, drayage-25-27, neos-932721,
  neos-960392, nursesched-sprint-hidden09):** we report no dual bound after 20 s; HiGHS
  certifies five of them at the root (1 node) and drayage-25-27 in 755 nodes.
- **Bound at the optimum, no incumbent (3: 10teams, acc-tight2, neos-498623 with a 943 %
  incumbent):** HiGHS closes each at the root -- a root primal-heuristics gap.
- **Root closes in HiGHS, not in ours (3: drayage-25-32, neos-4971100-bodrog, neos-555001):**
  HiGHS needs 1 node; we have an incumbent or a bound but not both closed -- presolve/root
  strength.
- **Small HiGHS trees (2: nu25-pr12, neos-595904, 19 nodes each).**
- **Tree rows (4: swath1, aflow30a, rlp1, mik-250-20-75-4):** §8.2's class.

Two non-certificate defects surfaced too: **the time limit is not honoured** (17 of 112 solves ran
past 25 s on a 20 s limit, worst neos-4971100-bodrog 118 s, n2seq36q 68 s), and 16 solves stopped
**early** (8.6-17.7 s) with status `time_limit`.

### 8.4 Stop rule

1. **No more per-row probing of the 62-row panel.** Its six rows reopen only inside a scoped
   project with its own entry experiment and kill criterion (CLAUDE.md §4); "find the next
   deviation on mik-5" does not qualify.
2. **The 62-row panel stops being the parity measure.** Any future parity claim is scored on
   the §8.3 population (or a larger pre-registered draw by the same rule), with the panel kept
   only as the soundness and regression gate.
3. **What would reopen parity work** is one of the §8.3 classes as a project of its own, entered
   by measuring that class across the population first: root-LP throughput on the six
   no-bound rows; root primal heuristics on the bound-at-optimum rows; root presolve/strength on
   the 1-node rows. Its kill criterion: no gain in certified count on the population at 20 s.
4. **The time-limit defect is an API-contract bug, not parity work,** and is fixed on its own:
   a 6x overrun of `time_limit` breaks what the caller was promised.

### 8.5 The time-limit defect, fixed (2026-11-13)

Four causes, each measured on a population row before it was changed (after:
`data/highs-parity/population/population_panel_timefix.jsonl`, the same 112 rows, HiGHS rows
reused from the close-out run):

1. **MILP presolve ignored the deadline, and the search's clock started after it.**
   `solve_presolved` handed the search `opts` unchanged, so a 50 s presolve was followed by
   a full 20 s search (neos-4971100-bodrog: 118 s). Now one clock starts at the entry of
   `solve_milp_node_hooked` (`clock_from`): presolve stops at the caller's deadline
   (`presolve_until`: between rows, columns and rules, so every applied reduction is
   complete), and the reduced search -- or the original search when presolve declines --
   gets what is left.
2. **Presolve was slow for a reason that was not the rules' work.** 54 of 58 s on bodrog
   was the aggregator re-testing the same columns (750k implied-free tests, 615k on columns
   of > 50 rows, 6 accepted). The test reads only the problem's state, so a rejection holds
   until the next change: an epoch cache answers the repeats. Exact -- byte-identical
   reduced problems on all 88 instances tried (38 panel + 50 population, HEAD build vs this
   one) and a unit test against the uncached rule on 300 random IPs. bodrog's presolve
   1.5 s (was ~50 s); the row now **certifies** (it was HiGHS-only).
   `dominated_pairs` (28 s of one pass on n2seq36q) is still slow; with the deadline it
   no longer overruns, and its speed is presolve work, not this fix.
3. **No-incumbent solves stopped at half the budget.** Two graduated changes conflicted:
   `RUN1_CONTINUE` passes the #291 stall slice (`stall_time_s`), the no-stall-tree
   retirement passes `extend_no_incumbent_s` ("keep searching without an incumbent"), and
   the guard fired first, under a route label claiming the whole budget was searched
   (16 rows, 8.6-17.7 s). The guard now yields when the extension is offered
   (`stall_stops`); its with-incumbent role (the stall re-solve) is unchanged. And the
   #844 LP-spatial reserve (35 % of the budget) no longer applies to **linear** models:
   22 linear pure-integer solves had ended with no incumbent and the fallback supplied
   none (on reblock420 it raised `RecursionError`).
4. **A reporting-only root-bound LP ran after the search with its own 5 s cap**
   (neos-1445738: search ended at 20.1 s, the LP at 25.2 s). It now gets what is left,
   or is skipped.

| ours, 112 readable rows, 20 s | before | after |
|---|---|---|
| certified | 9 | **11** (+bodrog, +acc-tight2; none lost) |
| no incumbent | 53 | 50 |
| max wall | 118.3 s | 27.1 s |
| rows > 21 s / > 25 s | 44 / 17 | 15 / 1 |
| stopped early (< 19 s, uncertified) | 16 | 0 |
| total wall | 2447 s | 2180 s |

62-row panel on the same build (`data/highs-parity/timelimit/panel_62_after.jsonl`): 53
certified, the same 53 as the close-out run (none gained or lost), 0 violations, 308.6 s
total (was 327.3), max wall 20.2 s (was 21.4). Presolve equivalence evidence and the
probes: `data/highs-parity/timelimit/`.

0 violations over 224 solves. Incumbents gained: rococoB10-011000, 10teams, neos-932721,
neos-686190, acc-tight2. Lost: fiball (found at 39 s, past the limit, before) and
neos-555001 (an incumbent at the deadline: 2 of 3 standalone repeats find it).

**Remaining overrun, recorded not fixed:** the #1537 E incumbent repair
(`_repair_published_incumbent` -> `repair_point`) runs dense minimum-norm least squares
after the search with no budget: mkc1 ended at 22-33 s in 2 of 4 loaded runs, all of the
excess in `numpy.linalg.lstsq`. It is a certificate guard (it can withdraw a certificate),
so it is not skipped for time; the fix is a cheaper exact solve, verified against
`test_1537_invariance`.

### 8.6 Root-gap entry measurement (2026-11-13)

The §8.4 rule-3 entry experiment, on the 22 population rows where discopt reported no
dual bound at 20 s (after the §8.5 fixes; HiGHS: 2 such rows). Script and data:
`data/highs-parity/rootgap/` (`measure.py`, `rootgap.jsonl`, `rootgap.summary.log`).
Per row: the LP relaxation by HiGHS 1.15.1 (threads=1, presolve on and off) and by
discopt's LP route (Rust simplex), each capped at 60 s; and the 20 s MILP solve split into
the Python set-up before the first engine call and the engine's root phases.

Two causes, both large; neither is the cut loop or the heuristics:

1. **The simplex.** Our LP relaxation is >= 10x HiGHS's on 11 of 22 rows (12-127x) and
   does not finish in 60 s on 9 where HiGHS takes 0.7-5 s (reblock420 5.0 s, piperout-d27
   0.66 s, neos-827175 0.43 s, neos-738098 2.4 s ...). HiGHS's presolve is not the reason:
   presolve off, its times are the same. In the MILP the root LP is where the engine's
   time goes (`RootSolve` ~ `RootCutLoop` on 15 rows, one cut round, the LP stopped by the
   deadline). Sub-MIPs and MILP presolve: 0 s on 21 of 22 rows.
2. **Python set-up before the engine starts:** median 9.7 s of the 20 s budget, > 5 s on
   20 of 22 rows (piperout-d27 18.4 s, leaving the engine 1.8 s). HiGHS has no such cost.

Counterfactual from the same numbers: with HiGHS-speed LPs and today's set-up, the root
LP fits in the remaining budget on 17 of 22 rows; with no set-up and today's simplex, on
about 7. So the simplex is the larger lever and the set-up the cheaper one; both are
needed for the rows where they compound (piperout-d27, tbfp-network).

Next, per the stop rule: (a) set-up -- profile the 9.7 s on these rows and remove what is
not needed for a linear model (kill: no certified gain on the population at 20 s);
(b) simplex -- split our LP time into iteration count vs per-iteration cost against
HiGHS's `simplex_iteration_count` (recorded), before choosing what to port.

