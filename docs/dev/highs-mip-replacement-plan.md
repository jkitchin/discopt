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
