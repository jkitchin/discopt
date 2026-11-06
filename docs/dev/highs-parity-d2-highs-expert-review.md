# D2 design review — discopt's Rust MILP search against HiGHS 1.15.1

Date 2026-11-06. Reviewer: HiGHS-internals persona (`skills/agents/highs-expert.md`, with its
"discopt context" paragraph corrected per CLAUDE.md: the in-house Rust simplex is the MINLP LP
engine, JAX is off the default path). Read-only review: nothing in the tree was edited, no
HiGHS code was built or run. Host load was 7.7 on 4 cores throughout, so no new timing was
taken; every number below is from checked-in data (`docs/dev/data/highs-parity/`) or from
counters in those files (`d2-analysis/b1_gt2.jsonl`, `b2_cost.jsonl`, `b5_rins.jsonl`,
`p6.6/miplib.json`, `p6.6/lmi.json`).

Paths: `D` = `/home/user/discopt/crates/discopt-core/src`, `H` =
`/tmp/claude-0/-home-user-discopt/38303ad9-c9f4-5a5b-9f75-f33b92707a61/scratchpad/highs_src/highspy-1.15.1/highs/mip`.

## 0. The one-paragraph answer

The gap is mostly two things, and neither is a missing cut family or a missing heuristic.

1. **The shape of the search.** discopt pops 64 best-bound nodes at a time, solves each as an
   isolated LP (its own LU factorisation, its own full-row FBBT sweep), folds the results back,
   and repeats (`D/bnb/milp_driver.rs:2534-3132`). Its only incumbent sources are one
   continuous-repair dive at the root and nearest-integer rounding at nodes
   (`milp_driver.rs:3731-3777`); once a point exists nothing improves it (RINS default OFF,
   `:883`). HiGHS does the opposite: it pops ONE node by hybrid best-estimate, then **plunges**
   — evaluates the chosen child immediately on the hot LP (bound change only, factorisation
   kept), chooses the direction by pseudocost/inference/root-solution, stops on prune or
   integrality, and decides at every backtrack whether to continue the plunge or return to the
   queue by comparing pseudocost scores (`H/HighsMipSolver.cpp:803-868`, `H/HighsSearch.cpp:1870-1885`,
   `:1623-1803`). Every plunge is simultaneously a diving heuristic, a pseudocost generator and
   a warm-start chain; it runs `randomizedRounding` (no incumbent) or `RINS` (incumbent) at the
   node it just installed (`HighsMipSolver.cpp:766-795`) under an LP-iteration budget
   (`H/HighsMipSolverData.cpp:670-731`), and every node LP carries the incumbent as an
   objective bound so a dominated subtree costs a handful of pivots (`H/HighsSearch.cpp:924`,
   `H/HighsLpRelaxation.cpp:1114-1123`). This is why HiGHS is robust to the starting vertex and
   discopt is not: discopt's primal side is one deterministic trajectory; HiGHS's is hundreds
   of cheap, partly randomised ones, pruned by a cutoff.
2. **What each node does with its LP.** On top of the LP, HiGHS runs *incremental* activity
   propagation over cliques/implications/objective (`H/HighsDomain.cpp:1936-1958`, `:1124`),
   reduced-cost fixing with conflict reconvergence (`H/HighsRedcostFixing.cpp:74-193`) plus
   root "lurking" reduced-cost bounds re-applied each time the incumbent improves (`:37`,
   `:195-260`), in-tree separation from the cut pool with aging on every queue-popped node
   (`HighsMipSolver.cpp:684-723`, `H/HighsSeparation.cpp:181-228`,
   `H/HighsLpRelaxation.cpp:629-662`), and iteration-budgeted reliability branching whose
   probes themselves fix columns and feed observations for *other* candidates
   (`H/HighsSearch.cpp:1112-1132`, `:412-531`, `:553-569`). discopt runs a full 8-round sweep
   of all rows (`D/lp/simplex/presolve.rs:110-165`, CSR rebuilt per call), per-node reduced-cost
   fixing only (`milp_driver.rs:4357-4483`), zero in-tree separation on the MILP route
   (`node_cuts=false` in `crates/discopt-python/src/lp_bindings.rs:1069`), and strong
   branching on the first 48 nodes only (`:1072`, `milp_driver.rs:2617`). That is the 31×
   geomean node ratio of the D2 analysis: not one lever, a stack of them, each compounding
   through the plunge.

Presolve and cut families (tableau c-MIR, implied-bound, clique, path, mod-k; probing, cliques,
coefficient tightening in HPresolve) are real and third. Everything measured as "≤ 1.43×" in
A4–A11 was measured *inside* a batched best-first engine with no cutoff and no improvement
heuristic, where no single lever can show its value — the A-series negatives say "this lever
alone does not rescue that architecture", not "HiGHS does not need it".

## 1. Ranked differences

Ranking is by plausible share of the 14/38 → 35/38 gap, using the D2 populations (A1 pure
primal: gt2, iskar, 2122, b-ball; A2 primal-dominated: kaihu, jijia, istra, obra, gesa2,
bppc8-02; A3 pure dual: mik×4; A4 both: nexp-50-20-1-1, gsvm2rl3, enlight_hard, p0548;
B per-node cost: fiber, neos-911970, neos17, beavma, sp150x300d, 2122, issue-2446).

### R1. Plunging search with hot-LP child evaluation and an objective cutoff (vs batched best-first with cold per-node factorisation)

**HiGHS.** Main loop `H/HighsMipSolver.cpp:891-979`: `installNodes` (`:615-648`) pops by hybrid
estimate `0.5·lb + 0.5·estimate` (`H/HighsNodeQueue.cpp:83-90`, estimate from
`computeBestEstimate`, `H/HighsLpRelaxation.cpp:428-450`), every 10th leaf the best-bound node
(`:618-622`). `processNode` (`:825-872`) then: `evaluateNode` → separate (cut pool, aging) →
`dive(i, nodeLim)` (`:803-819`) which is `HighsSearch::dive` (`H/HighsSearch.cpp:1870-1885`):
`evaluateNode; branch; repeat` down to a prune, with `plungeLimit=100` nodes (`:926`), and
`backtrackPlunge` (`:1623-1803`) that flips the last branching, propagates, and continues the
plunge unless an ancestor's inactive child scores better by pseudocost (`:1744-1773`), in which
case the node goes to the queue. Child evaluation is `lp->resolveLp(&localdom)`
(`H/HighsLpRelaxation.cpp:1441-1448`): bound changes on the live `Highs` object, no
refactorisation, dual simplex continues from the parent's invert. `lp->setObjectiveLimit(getUpperLimit())`
(`H/HighsSearch.cpp:924`) makes the dual simplex exit as soon as its (dual-feasible) objective
exceeds the incumbent (`kObjectiveBound`, `H/HighsLpRelaxation.cpp:1263`), which also prices
strong-branching probes (`H/HighsSearch.cpp:405`). Node-LP iteration limit
`10·max(avg node iters, avg root iters)`, ≥ 10 000 (`HighsMipSolver.cpp:840-843`).

**discopt.** `'search` loop `D/bnb/milp_driver.rs:2534-3132`: `tm.export_batch(64)` (`:2562`),
pure best-first (`D/bnb/pool.rs:50-65`, `SelectionStrategy::BestFirst` fixed at
`milp_driver.rs:1803`; `BestEstimate` exists in `pool.rs:82-96` and is never selected), 64
nodes solved in parallel through `solve_node` (`:3354-3937`), each with
`ctx.problem.solve_warm(.., &basis, ..)` → `solve_lp_warm_scaled_csc` →
`PreparedDual::prepare` = a fresh LU of the inherited basis (`D/lp/simplex/dual.rs:1354-1367`,
`:1436-1445`; `scaled_problem.rs:131-144`). Measured on gt2 (`b1_gt2.jsonl`, RINS=0 row = the
shipped configuration): 22,197 node LPs, 22,332 `DualPrepare` and 22,419 `LuNumeric` calls — one
factorisation per node — and a second run that re-explores 73,897 nodes from the root. No
objective limit exists in `SimplexOptions` (`D/lp/simplex/mod.rs:62-100`: `tol`, `max_iter`,
`deadline`, `warm_stall_guard`, …; no `obj_limit`/cutoff hit in `dual.rs`/`mod.rs`/`batch.rs`),
so a node whose dual objective crosses the incumbent after 3 pivots is still solved to
optimality and only then pruned by `process_evaluated` (`D/bnb/tree_manager.rs:896`).

**Why it matters.** (a) *Trajectory fragility*: in discopt, after the root the incumbent can
change only if a best-first node is integral or nearest rounding happens to be feasible
(`:3731-3744`, no LP repair, no propagation). On A1/A2 rows neither fires: `b4_60s.jsonl` holds
the same incumbent at 20 s and 60 s on 12/12 rows. So the whole outcome of a run is "where did
the single root dive land", and that dive (most-fractional, nearest-first,
`milp_driver.rs:5342-5472`) is a chaotic function of the root vertex — the gt2 1-ulp and the
khb05250 equality-vs-pair flips of the D2 follow-ups are exactly this. HiGHS's plunges *are*
dives, taken from every queue node, with the direction chosen by inference/pseudocost/root
solution (`H/HighsSearch.cpp:1151-1328`), so the incumbent is the best of hundreds of
trajectories; `randomizedRounding` adds a random offset in `[0.1,0.9]` per column
(`H/HighsPrimalHeuristics.cpp:1132`) precisely to decorrelate from the vertex. (b) *Cost*: the
hot-LP child costs a few pivots; discopt pays prepare + LU + full FBBT per node (gt2: 0.27 ms
LP, 0.46 ms FBBT, 0.17 ms prepare+LU). (c) *Breadth*: 64 best-bound nodes on a bound plateau
(mik, b-ball, bppc8-02: bound frozen at 20 s and 60 s) is breadth-first across the plateau;
a plunge goes through it and finds the integral points that let reduced-cost fixing bite.

**Not the A-series "node selection" negative.** That lever (plan §1.5) changed the heap order
of a batched engine; it did not evaluate children on a hot basis, did not run heuristics inside
the plunge, did not carry a cutoff into the LP. Those three are what make plunging pay.

**Fix (design).** Keep the batch architecture for the *queue* and add a plunge inside
`solve_node`: after the node LP is optimal and fractional, instead of returning, branch on the
chosen column, pick a direction (`kRootSol`-style: inference count × distance to root LP
value, `H/HighsSearch.cpp:1166-1207`), re-optimise the child from the *same* `PreparedDual`
(`reoptimize`, `dual.rs:1579` — the mechanism strong branching already uses), propagate the
child's box, and continue down until prune/infeasible/integral or a depth cap; push every
untaken sibling (with the parent's bound and basis) to the tree. Each plunge node is a full
node (bound recorded, pseudocost observed) so soundness is unchanged — it is a different cover
order, not a different cover. Add `obj_limit` to `SimplexOptions`: the dual loop returns
`BoundExceeded` once the NS safe bound of the current dual-feasible iterate (the same
`node_ns_bound` already computed at `:3587`) is ≥ cutoff; the node is imported with
`lower_bound = that safe bound` (valid) and pruned. Batch size then matters less; the pool's
hybrid estimate (`pool.rs:82`) should be the queue order with a best-bound pop every k leaves.

**Entry experiment.** No code: run HiGHS with `output_flag=true` on the 10 A1/A2 rows and
tabulate the *source letter* of every incumbent it logs (`T` evaluateNode, `B` branching, `R`
randomised rounding, `S` sub-MIP, `F` pump, …). Hypothesis: on ≥ 7/10 the final incumbent
comes from a dive/branching node (`T`/`B`) or RINS (`S`), i.e. from inside the plunge, not from
a root heuristic. Then a 2-day prototype (plunge depth ≤ 100, no cutoff yet) on the same rows:
count incumbent improvements per 10k nodes and nodes/s vs HEAD. **Kill:** < 4/10 rows improve
their incumbent within 20 s *and* nodes/s < 1.5× HEAD → the plunge is not the lever; fall back
to R2 and R4 only. Verification regime: bound-changing (search order changes the certified set),
P4-standard panel with the "no incumbent lost / per-node cost on the open population" bars P4.4
lacked.

### R2. The improvement heuristic is missing by policy, and the rounding has no repair

**HiGHS.** At every installed node while `moreHeuristicsAllowed()` (`H/HighsMipSolverData.cpp:670-731`:
heuristic LP iterations ≤ 5 % of total + 10 000 initial, front-loaded into the first 30 % of
the tree): `randomizedRounding` if no incumbent, else `RINS` (`H/HighsMipSolver.cpp:766-795`).
RINS (`H/HighsPrimalHeuristics.cpp:661-969`) is a *dive* toward the incumbent on a copy of the
LP with propagation after each fix (`:751-758`, `HeuristicNeighbourhood::getFixingRate`
`:296-306` counts propagation-implied fixings too), then a sub-MIP with presolve, the parent's
pseudocosts, cliques and implications (`solveSubMip`, `:77-247`), whose fixing-rate target
adapts from infeasible/success history (`determineTargetFixingRate`, `:249-274`). Rounding
goes through `tryRoundedPoint` (`:971-1059`): fix integers with propagation, then re-solve the
continuous part by LP with a 2×root-iteration cap, then `ziRound` on what is left. Root:
`ziRound`/`randomizedRounding`/`shifting` (`H/HighsMipSolverData.cpp:2109-2117`),
`centralRounding` from the analytic centre (`:2213`, `:2320`), `rootReducedCost` sub-MIP
(`:2367`), RENS (`:2398`), feasibility pump (`:2433`), each re-evaluating the root LP and
separating once more if bounds changed.

**discopt.** `try_rounding_csc` (`milp_driver.rs:5156-5229`): round/floor, clamp, check rows
against slack ranges — no propagation, no LP repair, so it fails on anything with equality
or big-M structure. The dive is root-only unless `DISCOPT_MILP_DIVE_STRIDE>0` and
`!has_incumbent` (`:371-389`, `:3767`). RINS exists (`:912-993`, batch-boundary, candidate =
best-bound fractional node) but `rins_enabled()` defaults to false (`:883-893`), with a flat
0.30 fixing rate (`:614`) and `SUB_MIP_NODE_CAP=500` (`:618`).

**Why it matters.** Entire A1/A2 population. The D2 `b5_rins.jsonl` single-rep entry: `RINS=1`
alone certifies gt2 (5.1 s) and iskar (7.4 s) and puts kaihu, jijia, bppc8-02 at the optimum
incumbent. That is the strongest single measurement in the whole track and it was never
decided on.

**Fix (design).** (i) Run the A12 RINS panel and graduate it (D2 L3a; zero code). (ii) Replace
the node-level rounding with a propagate-fix-then-LP-repair rounding (one warm re-solve from
the node basis with integers fixed; the dive machinery at `:5342` already does this per step —
it needs the all-at-once variant and a propagation pass). (iii) Make the RINS neighbourhood
adaptive (fixing-rate target from infeasible/success observations) and run it from *plunge*
nodes (R1), where the LP solution is deep and agreement with the incumbent is high. (iv) Budget
all heuristics by LP iterations (5 % + offset) rather than batch strides so cost is bounded on
B-population instances.

**Entry experiment.** (i) is its own panel. For (ii): on the 12 primal rows, count
`try_rounding_csc` calls vs hits at HEAD (new counter), then the same with LP repair.
Hypothesis: hit rate goes from ~0 to ≥ 5 % on ≥ 6 rows. **Kill:** < 3 rows → rounding is not
the mechanism; keep only RINS.

### R3. Node propagation: full-sweep FBBT vs incremental activity propagation with cliques, implications, objective and root reduced-cost bounds

**HiGHS.** `HighsDomain::propagate()` works a queue of rows whose activity bounds changed
(`markPropagate`, `H/HighsDomain.cpp:1936-1958`: a row is queued only if its min/max activity
is within `capacityThreshold_` of its side), with clique-table and implication propagation and
an objective-propagation structure (`:1124`, `getObjectiveLowerBound` used at
`H/HighsSearch.cpp:940-941`). Reduced-cost fixing in-node (`H/HighsRedcostFixing.cpp:74-193`)
re-propagates after fixing and, for binaries, runs conflict reconvergence analysis to learn a
constraint from the dual proof (`:149-173`). Root reduced costs are stored as "lurking bounds"
and re-applied globally whenever the incumbent improves (`addRootRedcost` `:195-260`,
`propagateRootRedcost` `:37`). The LP's basic-degenerate duals are used to strengthen the
proof (`H/HighsSearch.cpp:1003-1006`). Strong-branching probes that come back infeasible or
bound-exceeding *fix the column at the node* (`:553-569`, `:619-639`) — branching doubles as
propagation.

**discopt.** `tighten_bounds_csc_with_row` (`D/lp/simplex/presolve.rs:110-165`) rebuilds
`row_nz` from the CSC on every call and sweeps all `m` rows up to 8 rounds; it is 1.7× the
node-LP time on gt2 (`Fbbt` 10,277 ms vs `NodeLpSolve` 5,954 ms thread-summed, 22,197 calls)
and 37 % of node time on neos17 (`b2_cost.jsonl`). No cliques, no implications, no conflict
constraints (P4.7 retired at entry, correctly for its stated class). `reduced_cost_fix`
(`milp_driver.rs:4357-4483`) acts on the node only, with the batch-start incumbent snapshot
(`:3710`); nothing stores root reduced costs for later incumbents, so the fixings available at
the root once gt2's incumbent reaches 24106 (optimum 21166, bound at the optimum: the gap is
14 % of the objective) are never applied globally. `RootIntFixed 7` of 188 in gt2 run 2.

**Why it matters.** Throughput ×1.5–2 on every A-row (L6 already names the bit-identical
dirty-row rewrite) and *strength* on A3/A2: HiGHS closes mik in 91–453 nodes because each node
is smaller after propagation and RC fixing, not because its LP is tighter per se (its root is
1.5–1.8 % vs 3.8–4.2 %; the rest is the tree). Root lurking bounds are the cheapest unbuilt
piece: globally valid, re-usable, and exactly what a RINS-found incumbent needs to pay off in
the tree.

**Fix (design).** (i) L6 as specified (dirty-row queue, same order, same 8-round cap,
exact-identity gate). (ii) Store the root's reduced costs and basis status after the root cut
loop; on every incumbent improvement compute `floor(gap/|d_j|)` tightenings for nonbasic
integer columns and apply them to the *global* box (`tm` root bounds), re-propagating once.
Flagged, bound-changing. (iii) Let strong-branching infeasible/cutoff probes tighten the node
box (already have the Farkas-verified status in the probe; apply the bound on the opposite
side). (iv) A clique table from set-packing rows is the next step and is also what makes
HiGHS's rounding/propagation strong on binaries; out of scope for a first PR.

**Entry experiment (ii).** New counter on the 12 primal rows: at each incumbent improvement,
the number of root columns with `|d_j|·(u_j-l_j) > U - z_root`. Hypothesis: ≥ 10 % of integer
columns become globally fixable on ≥ 6 rows once the RINS incumbent is in hand (P4.5's seeded
measurement already found 15/57 roots ≥ 10 % *with the optimum*; this asks the same with the
incumbents RINS actually produces). **Kill:** < 3 rows → drop (ii), keep (i).

### R4. In-tree separation with a cut pool, LP cut aging, and cut selection by parallelism

**HiGHS.** Every node popped from the queue runs `separateAndStoreBasis`
(`H/HighsMipSolver.cpp:684-723`) → `HighsSeparation::separate` (`H/HighsSeparation.cpp:181-228`):
separation rounds continue while the objective keeps improving by > 1 % of the gain so far;
otherwise only aging. The LP drops cuts that have been slack for `mip_lp_age_limit` (default
10) consecutive solves (`H/HighsLpRelaxation.cpp:629-662`, `removeObsoleteRows` `:536`) and
the pool keeps them (soft limit 10 000, pool age limit 30, `H/HighsCutPool.cpp:162-227`) for
re-separation at any later node (`separate`, `:229-340`, with efficacy and parallelism
filtering). Families: tableau c-MIR and lifted cover via `HighsTableauSeparator`/`HighsCutGeneration`,
path/flow (`HighsPathSeparator`), mod-k, clique, implied-bound — all in `H/`. Root rounds are
capped at `2·sqrt(maxTreeSizeLog2)` and stopped by a direction-based stall test
(`H/HighsMipSolverData.cpp:1990-1993`, `:2231-2272`), not by a generated-cut count.

**discopt.** Root loop only (`milp_driver.rs:2103-2350`): GMI + cover (+ c-MIR, but the pure
MILP route does not pass `cmir_cuts`, per P4.4 finding 1), 500 generated cuts over 50 rounds,
tail-off at 1e-7 relative (`:2220`), slack cleanup once after the loop (`:2352-2461`). In the
tree: `node_cuts=false` at the binding default (`lp_bindings.rs:1069`), and even when on it is
cover-only, capped at `2·m`, never aged, never removed (`:2032-2052`). Every surviving root cut
rides in every node LP for the whole search (`enlight_hard`: 5,943 vs 28,637 nodes in the
slice with root cuts on/off, D2 §5).

**Why it matters.** A3 (mik×4, bound stalls at 0.033 from 20 s to 60 s), b-ball (0.00024
stalled), bppc8-02 (0.0404 stalled): the tree *cannot move the bound* because nothing tightens
a node beyond its box. A2's "in-tree cut aging" negative measured aging of a root-only set on
node counts; it did not score per-node cost, and it did not test *separating* in the tree.

**Fix (design).** A pool with age: cut rows carry an age that increments when slack at a
node's optimum and resets when tight; rows past the limit leave the working matrix (the
cleanup at `:2352` is the mechanism, applied at batch boundaries instead of once) and stay in
the pool with their signature; at every k-th batch, for the best-bound node of the batch,
separate GMI off the node basis + cover + c-MIR + pool re-separation, select by efficacy and
parallelism, fold at the batch boundary (the plumbing at `:3111-3131` exists). Soundness:
every cut is globally valid (GMI is derived on the node LP over the node box — it is **only
locally valid**; it must be tagged with the node's box or restricted to root-box-valid
derivations; HiGHS generates tableau cuts from the *global* domain with local bounds
substituted and lifted, `HighsTransformedLp`). Simplest sound first cut: c-MIR/cover from
original rows (globally valid) at nodes, GMI only at the root.

**Entry experiment.** On mik-1/-2 and b-ball at a fixed 5,000-node budget, dual gap with
node-level c-MIR+cover separation every 8th batch vs HEAD. Hypothesis: ≥ 1 pp on ≥ 2 of the 3.
**Kill:** < 1 pp everywhere → the mik tree needs the root (L5 tableau c-MIR) and in-tree
separation is retired.

### R5. Strong branching: node-prefix budget with uncapped full probes vs an iteration budget with capped probes, inside-probe fixing and implied observations

**HiGHS.** `branch()` (`H/HighsSearch.cpp:1099-1137`): SB iteration budget
`sbmaxiters = 100000 + (node LP iters)/2`; past half of it the reliability threshold decays
linearly from 8 to 1, past it to 0; `minreliable=0` if the LP is 10× degenerate (`:1130-1132`).
`selectBranchingCandidate` (`:247-800`): candidates whose score is already decided skip
probing; probes run on a `Playground` (bound change + resolve on the hot LP, `H/HighsLpRelaxation.h:110-155`)
with the node-LP iteration limit and the objective cutoff (`:405`); an infeasible or
cutoff probe fixes the column at the node and continues (`:553-569`, `:619-639`); every
optimal probe's solution is scanned for *other* candidates that landed on an integer
(`analyzeSolution`, `:412-531`) and credited as observations; the score blends pseudocost,
conflict, cutoff and inference statistics (`H/HighsPseudocost.h:310-350`), and unobserved
columns start at the global average pseudocost (`:296-299`).

**discopt.** SB only while `total_nodes < sb_node_budget = 48` (`lp_bindings.rs:1072`,
`milp_driver.rs:2617`), 6 candidates ordered most-fractional (`:4141-4147`), each probe a
full warm dual solve with `max_iter=100 000` and no cutoff (`:4185-4199`), score = product of
degradations (`:4246`), observations recorded (`:4224`, `:4238`). After node 48, pure
pseudocost with `default_cost = 1.0` for unobserved columns (`D/bnb/branching.rs:74`,
`:140-145`) — i.e. most-fractional until a column has been branched on. No inference, cutoff
or conflict component.

**Why it matters.** Population B's cost (nexp-50-20-4-2: 31 SB calls = 21 of ~30 thread-s;
neos-911970: 14 calls = 14.1 s; beavma: 24 calls = 14.1 s, `b2_cost.jsonl`) and the A-rows'
tree size (branching on `default_cost` after node 48 is most-fractional branching, known to be
near-random on bound quality). A10 varied only the node-prefix count; the per-probe cost and
the post-prefix rule were never changed.

**Fix (design).** Replace the node prefix with HiGHS's budget shape: SB allowed anywhere on an
unreliable candidate while `sb_iters < 1e5 + node_iters/2`, reliability decaying with the
budget; per-probe `max_iter = max(2× average node-LP iterations, 100)` (keep the full-solve
fallback off: a probe hitting its cap scores 0 and reports no observation); probes carry
`obj_limit` (R1); infeasible/cutoff probes tighten the node box and are not re-probed;
unobserved pseudocost = running average over all observations (`cost_total`); candidate order
by current score, not fractionality. Bound-changing (branching order), one flag.

**Entry experiment.** `b2_cost.jsonl` method on the 8 B rows with the per-probe cap alone:
SB share of thread time ≤ 20 % with node count in the slice ≥ 5× HEAD on ≥ 5/8. **Kill:** node
count on the 30 always-certified rows rises > 15 % (the P4 bar) → retire the cap, keep the
budget shape only if it passes separately.

### R6. Root closure: presolve + restart + separator families + heuristics, as one loop

**HiGHS.** `evaluateRootNode` (`H/HighsMipSolverData.cpp:1987-2530`) is a loop, not a cut
pass: LP → rounding/shifting → restart if ≥ 10 % integers fixed (`:2127-2147`, presolve
re-run on the reduced model, cuts kept as rows) → separation rounds with stall detection
(`:2168-2283`), central rounding after round 5 (`:2202-2229`) → root reduced-cost sub-MIP,
RENS, feasibility pump, each followed by one more separation round if bounds changed
(`:2361-2443`) → restart again if ≥ 2.5 % fixed or any fixing on the first pass
(`:2494-2513`). In-tree restarts when the estimated tree is huge relative to gap progress
(`H/HighsMipSolver.cpp:475-538`). HPresolve (`/presolve/`) does probing, cliques, implied
integers, coefficient tightening, dominated columns, parallel rows — the P4.8 entry measured it
removing ≥ 20 % of rows on 29/58 panel instances (mik 62 %).

**discopt.** FBBT-only root presolve (`root_box`, `milp_driver.rs:1595`), one cut loop, one
dive, no restart (P4.5 killed at entry — correctly for *bound* gain, but a restart's value in
HiGHS is also a smaller LP for every node and a second chance for the root heuristics).

**Why it matters.** The rows HiGHS closes at the root (gt2, khb05250, nexp-50-20-1-1,
nexp-50-20-4-2, sp150x300d, beavma, gesa2, neos-1425699, p0548, 2122, issue-2446 — 11 of the
24 flips, all `H nodes ≤ 2` in `p6.6`). For gt2 the dual bound is already exact at discopt's
root; what HiGHS has at its root and discopt does not is the *primal* half of that loop
(R2) — the measurement says the gap on gt2 is 100 % primal.

**Fix (design).** This is the largest item and is deliberately ranked after R1–R5 because
R1/R2 deliver most of the root-closed rows (the bound is already there on A1; the primal is
not). Order within R6: (a) wire the existing c-MIR separator into the pure-MILP route (P4.4
finding 1 — a one-line omission); (b) tableau c-MIR + lifted cover (L5) behind its entry; (c)
a restart on fixing rate ≥ 10 % *after* R2 gives the root an incumbent (P4.5's unseeded
measurement found 0/57 roots fixing ≥ 10 % *because no incumbent existed at the root* — with
RINS/rounding landing one, re-run the P4.5 counter before deciding); (d) HPresolve-class
reductions last (P4.8).

**Entry experiment.** For (c): `p45_restart_probe.py` with the root incumbent coming from R2's
rounding/RINS rather than the HiGHS optimum; hypothesis ≥ 8 roots ≥ 10 % fixed. **Kill:** the
P4.5 bar again.

### R7. LP engine per node: pricing and hot start

**HiGHS.** HEkkDual with dual steepest-edge weights, BFRT and cost perturbation, on a single
`Highs` instance whose invert persists across node bound changes.

**discopt.** Bound-flipping ratio test (`D/lp/simplex/dual.rs:1909`, `:2548`), cost
perturbation (`:195-258`) and dual Devex pricing (`:1684`, `select_leaving` `:2493-2500`) are
all present — the pricing *rule* is not the gap. What differs is that the Devex reference
weights, the invert and the perturbation state are all created fresh per node
(`PreparedDual::prepare`, `:1436`), because the batched search never continues a solve from
the parent's live state; HiGHS's weights and invert persist along a plunge. neos-911970
(42/125 cold fallbacks, 6,254 refactorisations in 125 solves, `b2_cost.jsonl`) and
issue-2446 (120 µs/pivot, dense `PriceSweep`) are the two LP-engine rows; the first looks like
a warm-start robustness problem (dual-infeasible inherited basis → cold two-phase), the second
like missing hypersparsity on a 4,163-column LP (P2.3 killed hypersparse FTRAN/BTRAN on the L
panel, which did not contain it).

**Fix.** Nothing to design here before R1: with a plunge the parent's `PreparedDual` (weights,
invert, etas) is reused for the child as strong branching already does (`reoptimize`,
`:1579`), which removes the per-node `DualPrepare`+`LuNumeric` cost and keeps Devex weights
meaningful. For neos-911970 the entry question is why 34 % of warm starts fall back cold.

**Entry experiment.** New counters on neos-911970: which `DualPrepReject*` arm fires
(`:1445+`) and the pivots of the subsequent cold solve. Hypothesis: one rejection arm accounts
for ≥ 80 % of the fallbacks and is repairable (e.g. a dual-infeasible start that a primal
phase on the few infeasible columns would fix). **Kill:** fallbacks are spread across arms →
accept the row as LP-hard, no lever.

### R8. Budget shape (Python route): two trees per budget

`_solve_milp_simplex` (`python/discopt/solver.py:39820-39829`, `:39982-40060`) runs a
`min(0.5·rem, 10 s)` first slice then a *fresh* seeded tree. gt2 at 20 s: 22,197 nodes then
73,897 from the root again (`b1_gt2.jsonl`). HiGHS runs one tree and restarts only on a
tree-size estimate. L4 (continue through the P3.1 handle) is the fix; it is independent of
R1–R7 and bound-neutral in result. Entry/kill as in D2 L4.

## 2. Structural issues not tied to one lever

- **Stale state within a batch.** `inc_snapshot`, `reliability` and pseudocosts are frozen for
  64 nodes (`milp_driver.rs:2622-2682`); an incumbent found by node 3 of a batch does not prune
  nodes 4–64, and nothing in a batch can learn from a sibling's probe. A plunge (R1) makes the
  batch the *queue* refill unit, not the evaluation unit.
- **The LP never sees the incumbent.** Every node LP, every SB probe, every dive step is solved
  to optimality regardless of the cutoff (R1). With a 14 % primal gap this is cheap work; with
  a 1e-4 gap (b-ball, 2122, iskar after RINS) it is the whole proving phase.
- **Every heuristic is gated by `!has_incumbent` or is root-only** (`:371-389`, `:3767`,
  `:883`). HiGHS gates heuristics by *effort spent* (`moreHeuristicsAllowed`), never by
  "already have one". This single policy difference is the "frozen primal" of A14/P4.1/D2.
- **GMI cuts are the only tableau family and they are dense after slack rewriting**
  (P4.4 retraction: 297 nnz/cut on fiber); HiGHS's tableau separator produces c-MIR from
  aggregated tableau rows with bound substitution (sparser, stronger) and never raw GMI.
- **Cut budget counts generated cuts; HiGHS budgets active rows.** The 500-cut cap stops mik at
  3.8–4.2 % (P4.4 finding 3) while HiGHS keeps separating but ages rows out of the LP.
- **The per-node work is paid for by the only thing HiGHS does not have: 4 rayon threads.**
  The 2× wall bar already credits Rust with the cores; a plunging single-tree engine would
  lose that parallelism unless the batch becomes "k parallel plunges", which is exactly
  HiGHS's worker model (`HighsMipWorker`, `HighsMipSolver.cpp:596-613`).

## 3. What I would do, in order, and why this order

1. **R2(i): decide `DISCOPT_RINS`** (panel, no code). It is the measured +2 and the cheapest
   test of the "primal policy" diagnosis.
2. **R1 objective cutoff in the dual simplex** (`obj_limit` via the NS safe bound). Small,
   nearly bound-neutral (prunes with a valid bound), and it makes every later lever cheaper.
3. **R1 plunge prototype** behind a flag, with the HiGHS-log source-letter entry first.
4. **R3(i)+(ii)** incremental FBBT (exact-identity gate) and root lurking reduced costs.
5. **R5** SB budget shape and per-probe cap (the B-population unblocker).
6. **R4** pool + aging + node separation; then **R6(a,b)** c-MIR on the route and tableau
   c-MIR; **R6(c)** restart re-measured with a real root incumbent.
7. **R7, R8** throughput and budget plumbing.

If after 1–5 the MIPLIB count is not ≥ 24/38 with the A1/A2 rows certified, the diagnosis
"search shape + primal policy" is falsified and the remaining gap is root closure (R6) — at
that point R6 moves to the front and the plan's P4.8 deferral should be reversed.

## 4. Things in the plan I would re-read in light of the HiGHS source

- "node selection" (§1.5 negative) ≠ plunging (R1). Not re-proposing the heap order; proposing
  depth-first child evaluation on a hot basis with a cutoff and in-plunge heuristics.
- "in-tree cut aging" (A2 negative) measured aging of root cuts by node count; R4 is separation
  in the tree with aging as the cost control, measured on bound and per-node cost.
- "strong-branching budget" (A10 negative) varied the node prefix; R5 changes probe cost and
  the post-prefix rule.
- P4.5 "root restart killed": its unseeded arm had no incumbent, which is the precondition the
  restart needs; re-run after R2.
- P4.7 conflict analysis: correctly retired for the class named; note HiGHS's main use of the
  conflict machinery on these instances is reduced-cost *reconvergence* (R3), not infeasible
  nodes.
