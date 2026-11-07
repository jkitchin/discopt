## A. The HiGHS mechanisms, as read from the source (HiGHS 1.15.1, `highs/`)

Paths are relative to `scratchpad/highs_src/highspy-1.15.1/highs/`. Nothing here was executed; the
source was read as data. Option names are the ones `lp_data/HighsOptions.h` registers (lines
1095–1310) and the mip code reads (`grep options_mip_->` over `mip/*.cpp`).

### A.1 Root flow (`mip/HighsMipSolver.cpp:120-260`, `mip/HighsMipSolverData.cpp:1987-2534`)

1. **Presolve** (`HighsMipSolverData::runMipPresolve`, 811-829 → `presolve/HPresolve.cpp:5780-6050`):
   a full dimension-reducing presolve with MIP-only rules — clique extraction, implications,
   `strengthenInequalities` (coefficient strengthening, 7023), `dominatedColumns` (1045),
   `detectParallelRowsAndCols` (7267), `sparsify` (8301), `aggregator` (6688), `enumerateSolutions`,
   and `runProbing` (1601-1873: binaries ordered by implication count, each probed both ways with
   the clique table and implication graph, early-abort on a deletion/clique/time budget).
   `presolve=off` disables all of it; `presolve_rule_off` is a bitmask over
   `lp_data/HConst.h:260-288` (bit 15 = probing).
2. **Setup** (`runSetup`, 831-1157): row integrality + RHS rounding, up/down locks, clique
   partition of the objective, domain propagation of every row once, clique extraction.
3. **Trivial heuristics + Feasibility Jump** (`HighsMipSolver.cpp:189-211`;
   `mip/HighsFeasibilityJump.cpp`, `feasibilityjump.hh`, 939 lines): a weighted local-search
   (Luteberget & Sartor 2022) run *before the root LP*. Switch `mip_heuristic_run_feasibility_jump`.
   Log source letter `J`.
4. **Root LP**, then rounding heuristics on the first LP point (`randomizedRounding`,
   `HighsPrimalHeuristics.cpp:1117-1202`: round each integer by its locks or a random threshold,
   propagate the domain after every fix, re-solve the continuous part; `R`), optional ZI-round and
   shifting (`Z`, `I`, both default off).
5. **Restart check #1** (`evaluateRootNode` 2127-2147): if ≥ 10 % of integer columns are
   already inactive (`percentageInactiveIntegers`, 1310-1315: fixed or clique-substituted),
   **restart** — `performRestart` (1317-1467) folds the cut rows into the model, re-runs the full
   presolve on the fixed model with the root basis carried over, and re-enters the root. Requires
   `mip_allow_restart` and `presolve != off`.
6. **Root separation loop** (2150-2290): rounds until 3 stall rounds (progress measured as the
   smoothed direction change of the LP point, 2238-2262) or `maxSepaRounds`. Each round
   (`HighsSeparation::separationRound`, `mip/HighsSeparation.cpp:43-180`): propagate the domain,
   implied-bound cuts (`HighsImplications::separateImpliedBounds`, 583), clique cuts
   (`HighsCliqueTable::separateCliques`, 1605), then the three separators registered at
   `HighsSeparation.cpp:38-40` — **tableau** (`HighsTableauSeparator.cpp:41-244`), **path**
   (`HighsPathSeparator.cpp`: "single-row relaxations from the current LP relaxation by
   substituting bounds and aggregating rows", `HighsPathSeparator.h:9-11` — the analogue of
   discopt's `cmir.rs`, fed into the same `generateCut`), **mod-k** (`HighsModkSeparator.cpp`,
   mod-k cuts over integral rows)
   — then the local and global cut pools re-separated. After every round the root reduced costs
   are recorded (`redcostfixing.addRootRedcost`, `HighsSeparation.cpp:82-86, 171-175`) and the
   **analytic centre** + `centralRounding` runs once (`C`).
   - Tableau separator: for every fractional basic integer (and fractional *integral row*),
     score by `f(1−f)/‖row of B⁻¹‖²` (75-160), take the aggregated tableau row through
     `HighsLpAggregator` (160-240) and call `HighsCutGeneration::generateCut`
     (`HighsCutGeneration.cpp:1058-1197`) on it **and on its negation**. `generateCut` transforms the
     row to the bound-substituted space (`HighsTransformedLp::transform`, variable bounds and
     complementation), then `tryGenerateCut` (1382-1455): **lifted cover** first (`determineCover`
     25-130 → `separateLiftedKnapsackCover` 131-215 / `separateLiftedMixedBinaryCover` 216-279 /
     `separateLiftedMixedIntegerCover` 280-500), then the **c-MIR heuristic** (504-740: deltas =
     {1, |a_j| of integer columns at nonzero value, integral scale}, f0 ∈ [0.005, 0.995],
     complement integers above half their range, keep the most efficacious), and the better of
     the two; `postprocessCut` (760-900) scales, drops tiny coefficients, requires violation
     > 10·feastol, then `tightenCoefficients`. There is **no GMI** anywhere.
   - No option switches root separation off. The tightest throttle the options permit (the
     `cuts_throttle` arm) is `mip_pool_soft_limit=1` (tableau separator refuses when the pool
     holds > 1 available cut, `HighsTableauSeparator.cpp:47`), `mip_lp_age_limit=0` (a cut
     row basic in two consecutive epochs leaves the LP, `HighsLpRelaxation.cpp:629-690`) and
     `mip_pool_age_limit=0` (a pool row not re-added is deleted at the next aging,
     `HighsCutPool.cpp:162-220`). Path, mod-k, clique and implied-bound separation still run.
7. **Root heuristics after separation** (2351-2460), each followed by an LP re-solve and one
   more separation round when the global domain changed: `rootReducedCost`
   (`HighsPrimalHeuristics.cpp:314-392`; sub-MIP `L`), `RENS` (394-659; `L`), and, only while
   there is **no** incumbent, `feasibilityPump` (1578-1695; `F`). Gated by
   `moreHeuristicsAllowed` (`HighsMipSolverData.cpp:669-731`: heuristic LP iterations ≤
   `mip_heuristic_effort` × total + 10,000 at the root).
8. **Restart check #2** (2487-2512): any inactive integer column on the first pass
   (`fixingRate > 0 && numRestarts == 0`), or ≥ 2.5 % later, restarts. Then symmetry detection
   finishes (orbitopes / orbital fixing, `presolve/HighsSymmetry.cpp`).
9. **Tree** (`HighsMipSolver.cpp:246-1120`): best-bound queue with plunging
   (`HighsSearch::backtrackPlunge`, 1623-1803); at each node taken from the queue
   `evaluateNode` → `separateAndStoreBasis` (684-720: a **node separation round** when
   `mip_allow_cut_separation_at_nodes`, otherwise only aging) → `runHeuristics` (836-848, gated
   by `moreHeuristicsAllowed`: `randomizedRounding` + `RENS` while no incumbent, else `RINS`,
   760-800) → `dive`. In-tree **restart votes** (`checkRestart`, 474-540: tree-size estimate
   ≥ 20× nodes so far × active-integer-ratio² with no gap progress). **Reduced-cost fixing** at
   every node from the node LP duals (`HighsRedcostFixing::propagateRedCost`, 74-193, called
   at `HighsSearch.cpp:1008`), binaries' fixings turned into **conflict constraints** through a
   dual proof (`computeDualProof`). **Conflict analysis** on every infeasible propagation
   (`HighsDomain::conflictAnalysis`, pool `HighsConflictPool.cpp`). **Reliability branching**
   (`HighsSearch::selectBranchingCandidate`, 247-717: strong-branch the candidates with fewer
   than `mip_pscost_minreliable` (8) pseudo-cost samples, LP-iteration-capped; `0` means every
   candidate counts as reliable, `HighsPseudocost.h:243-245`, so no strong branching).

### A.2 Root reduced-cost "lurking bounds" (`mip/HighsRedcostFixing.cpp`)

`addRootRedcost` (195-324) is called after every root separation round with the root duals: for
each integer column with nonzero reduced cost `d_j` it records, for a geometric ladder of
candidate bounds `b` (up to 2^10 steps, 230-300), the objective cutoff `z_root + d_j·(b − l_j)` at
which that bound becomes valid — a multimap `cutoff → bound` per column. `propagateRootRedcost`
(37-72) applies every lurking bound whose cutoff lies at or above the current `upper_limit`
to the **global** domain, and prunes those below the lower bound. So the root reduced costs keep
fixing columns globally every time the incumbent improves, long after the root — that is what
feeds `percentageInactiveIntegers` and the restarts. `rootReducedCost` (heuristic) sorts the
lurking bounds by cutoff, applies them with propagation until 30–50 % of the integers are fixed,
and solves the sub-MIP of what is left (314-392).

### A.3 Sub-MIP heuristics (`mip/HighsPrimalHeuristics.cpp`)

`solveSubMip` (77-247): a *full* HiGHS MIP on the LP's current rows (cuts included) with the
neighbourhood bounds, `mip_max_leaves` 500, `mip_max_nodes` 200 + nodes/20, stall 12,
`mip_pscost_minreliable=0`, presolve on, symmetry off, `mip_heuristic_effort` 0.8, objective bound
= the incumbent cutoff, abs gap `feastol·max(gap, 1000)`; the clique table, implications and
pseudo-costs are inherited. The sub-MIP's LP iterations count against the heuristic budget
scaled by the unfixed fraction. Observed fixing rates (success / infeasible) feed
`determineTargetFixingRate` (249-275): the target fixing rate is drawn uniformly between 0.9×
the mean successful rate and max(0.6, 1.1× successful, 0.9× infeasible).
- **RENS** (394-659): a **propagating dive** on a copy of the LP relaxation (`HighsSearch` in
  heuristic mode): fix every integer that is integral in the LP point to its value (branch
  upwards/downwards + propagate after each, conflict analysis on infeasibility), then fix
  fractional ones one at a time towards the rounding chosen by `calcFixVal` (direction away from
  the root LP solution if it moved ≥ 0.4, else the objective-worsening direction), re-solving the
  LP, until the target fixing rate is hit; if the dive reaches an integral LP it is the solution
  (no sub-MIP); if the fixing rate ends < 0.1 it runs a 10-backtrack depth-first search instead;
  otherwise `solveSubMip`. On an infeasible sub-MIP it halves the fixing rate and retries from
  half the dive depth (`goto retry`, 633-656).
- **RINS** (661-969): the same dive, but the first fixing candidates are the integers where the
  LP point **agrees with the incumbent** (`std::partition`, 735-741); when none remain, it
  continues fixing towards the LP value with the same machinery; minimum fixing rate 0.25.
- Both run at **every plunge start** in the tree while `moreHeuristicsAllowed()` (10 k LP
  iterations free, then `mip_heuristic_effort` = 5 % of the tree's LP iterations, front-loaded
  into the first 30–80 % of the tree by the pruned-weight estimate, 669-731).

### A.4 Node cuts and aging

`separateAndStoreBasis` (684-720) runs a separation round on the node LP of every node taken
from the queue (`skip_separation` false): same separators as the root, cuts go to the worker's
pool and the global pool. `HighsLpRelaxation::performAging` (629-690) removes a cut row from the
LP once it has been basic (non-binding) for `mip_lp_age_limit` (10) consecutive epochs and
resets the age when the cut's dual is nonzero; `HighsCutPool::performAging` (162-220) deletes pool
rows older than `mip_pool_age_limit` (30), with the age limit lowered dynamically while the pool
is above `mip_pool_soft_limit` (10 000). `separate` (222-...) re-adds pool cuts violated by the
current point (efficacy-sorted, parallelism-filtered). Conflict constraints live in a parallel
pool (5× the age limit) and are propagated, not added to the LP.

## B. What discopt's Rust MILP engine has today (`crates/discopt-core/src/bnb/milp_driver.rs`, 9 860 lines)

- **Presolve on the engine's rows**: `root_box` (1595-1640) = integer bound rounding + one
  FBBT contraction (`tighten_bounds_csc`); no dimension reduction, no cliques, no probing, no
  coefficient strengthening on the marshaled rows. The Python route runs the model-level
  orchestrator first (`solver.py:18230-18275`: `run_root_presolve(eliminate=True, fbbt=True,
  probing=True)`, cliques default off), whose output is a *bounds* delta plus singleton
  elimination; the engine then never re-presolves. **No restart** of any kind.
- **Root cuts** (2103-2340): `cut_rounds` rounds of cover (`lp/cover.rs`, CJP greedy covers from
  the original rows), aggregated c-MIR from the original rows (`lp/cmir.rs`, `separate_cmir_csc`
  477; Marchand–Wolsey with variable-bound substitution, `CMIR_MAX_AGGR` aggregation steps), and
  GMI from the basis (`lp/gomory.rs`, iterative-refined tableau rows; P4.4's completed-basis
  repair, `gmi_repair_enabled`), every cut re-expressed over structural columns
  (`substitute_slacks_to_structural`, 4710), selected by efficacy/parallelism (`lp/cut_select.rs`),
  capped by `root_cuts` (500) and a time sub-budget; **the kept rows stay in every node LP for
  the whole search — no aging, no pool re-separation** (the analysis doc §5 measured the
  per-node cost of that on `enlight_hard`). The root cut rounds stop on a stalled objective
  (`prev_obj` test, 2185), not on a direction-change measure.
- **Node cuts**: cover separation only (`separate_cover_csc` at every fractional node, 3783-3795,
  behind the `row_density` gate and `node_cut_cap`), folded in at batch boundaries. No tableau
  cuts, no c-MIR, no clique/implied-bound cuts in the tree.
- **Primal heuristics**: nearest/floor rounding of the node point with a feasibility check
  (`try_rounding_csc`, 5156-5232: no propagation, no LP re-solve of the continuous part);
  the continuous-repair dive (`try_dive_repair`, 5342-5472: most-fractional fixing with one
  alternative rounding, warm LP re-solve per step, unbudgeted) at the root and, with
  `DISCOPT_MILP_DIVE_STRIDE`, on batches **while no incumbent exists** (`dive_batch_eligible`
  371-389); RINS (`try_rins` 912-996 → `run_sub_mip` 1042-1158: a recursive call of the same
  driver on the agreement neighbourhood, node budget `sub_mip_node_budget`, 25 % of the remaining
  time ≤ 2 s, strict improvement, re-validated) **default OFF** (`rins_enabled` 883-891,
  `Err(_) => false`). No RENS (retired P4.2), no FJ (retired P4.6), no feasibility pump, no
  reduced-cost sub-MIP, no randomized/propagating rounding.
- **Reduced-cost fixing** at every node from the node duals against the incumbent
  (`reduced_cost_fix`, 4357-4480: `floor(gap/|d_j|)` with the #1409 deflated divisor), merged into
  the children's bounds — but only for the subtree (there is no global "lurking bound" ladder
  from the root duals, and no conflict constraint is derived from the fixing).
- **Strong branching** (`strong_branch`, 4122-4264): `sb_max_cands` candidates on the first
  `sb_node_budget` nodes, warm dual solves without an LP-iteration cap (analysis doc §1.1 B.3).
- **Propagation**: `tighten_bounds_csc_with_row` FBBT at every node (all rows plus the #1953
  objective-cutoff row, ≤ 8 rounds), no clique table, no implications, no conflict analysis.
- **Search**: `TreeManager` best-bound with depth tie-break; the route runs the engine for a
  first slice (`min(0.5·remaining, 10 s)`) and re-enters with a fresh tree seeded with the
  incumbent (`solver.py:39812-39830`).
