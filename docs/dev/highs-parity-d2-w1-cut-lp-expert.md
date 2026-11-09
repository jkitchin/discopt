# D2 W1 — how HiGHS keeps cut rows in the tree LP (HiGHS-expert consult, 2026-11-07)

This consult was asked after the W1 entry measurement (`highs-parity-d2-plan.md` §2 W1).
The question was what HiGHS does with in-tree cuts.

The measurement:

- The node-local re-solve with pool rows is the node-GMI cost. On `istra`, `NodeGmiLp`
  took 14.2 s against 7.1 s for the plain node LP. Scan, augment, NS verification and
  separation together took < 10 %.
- The scratch experiment `DISCOPT_MILP_NODE_GMI_FOLD=cap:uses` moved hot pool cuts into
  the shared LP. The rows were append-only and never removed. With `200:3`:
  - Gains:
    - `b-ball` certified in 2.8 s (open at 20 s before);
    - `p0548` certified;
    - `istra`, `jijia` and `iskar` certified in about half the wall;
    - the `mik` bounds rose by 60-250.
  - Losses:
    - `bppc8-02` lost its certification;
    - `fiber` took 2x the wall;
    - `gsvm2rl3`, `beavma` and `obra` lost about 10x node throughput.
- A global nonzero cap on the folded rows (`:0.1`, `:0.3`) undid the gains. It did not
  restore `bppc8-02`. That lever is killed: it is not a HiGHS mechanism (Q4 below).

The expert answered from the source; every path is under `ref/HiGHS/highs/`. The report
is kept nearly verbatim.

## Q1. How cut rows enter and leave the LP

**One persistent LP per worker.** Each worker gets its own copy of the root LP, cut rows
included (`mip/HighsMipSolver.cpp:275-301`). The LP is reused node after node; nodes do
not rebuild it.

**Separation runs only at subtree roots, not at every node.**

- Each node installed from the queue goes through `evaluateNode` and then
  `separateAndStoreBasis` (`HighsMipSolver.cpp:826-837, 684-693`).
- `separate()` (`HighsSeparation.cpp:180-226`) loops `separationRound`. Each round runs
  these steps:
  1. the implied-bound and clique separators;
  2. the tableau, path and mod-k separators (`:37-39, 93-147`);
  3. re-separation of the worker pool and the global pool against the LP point
     (`:150-158`);
  4. `addCuts`, `resolveLp` and `performAging(true)` (`:162-166`).
- Rounds continue only while the objective improves by more than 1 % of the previous
  gain (`:211-213`).
- The dive that follows (up to 100 nodes, `HighsMipSolver.cpp:928`) does **no**
  separation. Each dive node only re-solves the same LP and calls `performAging()`
  without deletion (`HighsSearch.cpp:957-958`).

A dive node's LP therefore holds the model rows plus every cut alive in that worker's LP.
The pool is not re-scanned per node.

**Leaving the LP** (`HighsLpRelaxation.cpp:629-676`):

- The age limit is `mip_lp_age_limit`, default 10 (`lp_data/HighsOptions.h:1176-1180`).
- Only rows whose **slack is basic** age. A nonbasic row with |dual| > the dual
  tolerance has its age reset to 0 (`:660-673`).
- Rows are deleted only when `deleteRows=true`, which happens inside separation rounds.
  Deletion runs only every `max(agelimit/2, 2)` = 5 epochs, with limit
  `min(10, epochs)` (`:639-646`). Otherwise the limit is infinite (`:647-649`).
- `removeObsoleteRows` (`:536-554`) drops every cut with a basic slack, after the root
  (`HighsMipSolverData.cpp:2481`) and after a restart (`:2098`).

**The pool (`HighsCutPool`):**

- Limits: `mip_pool_age_limit` = 30 and `mip_pool_soft_limit` = 10000
  (`HighsOptions.h:1183-1193`).
- A new cut starts at age `agelim-5` = 25 (`HighsCutPool.cpp:609`). Unless it is
  violated within about 5 separations, it is deleted.
- In `separate()` (`:236-458`):
  - an unviolated cut ages and is deleted at the age limit (`:279-310`);
  - a violated cut resets to age 0 (`:337-341`);
  - above the soft limit, the effective age limit shrinks (`:242-246`; `performAging`
    `:166-170`).
- Selection:
  - **Score** = viol / (numActiveNzs · ‖a_active‖) (`:342`). "Active" means not at its
    bound (`:319-335`), so dense cuts are penalised roughly quadratically.
  - The threshold is adaptive: `minScoreFactor · bestObservedScore`, starting at 0.9
    (`:364-391`, `.h:97`).
  - **Parallelism**: a candidate is discarded when its parallelism with an
    already-selected cut in the same cutset exceeds 0.1 (`:397-418`).
- Cuts already in an LP are skipped (`ages_ = -1`, `:252`). A cut removed from the LP
  returns to the pool at age 1 (`:150-160`).
- Tableau separation is skipped while the pool holds more than 10000 available cuts
  (`HighsTableauSeparator.cpp:47-48`). It also skips aggregations longer than
  `(10000+n)/10` (`:205-209`).

**Density limits:**

- At generation, a cut longer than **100 + 0.15·n** nonzeros is rejected, unless the
  excess can be cancelled using columns at their bounds
  (`HighsCutGeneration.cpp:974-985`).
- A cut needs violation > 10·feastol (`:1336`).
- Coefficients below `100·feastol·max(maxAbs, 1e-3)` are dropped (`:784`).
- The pool's `2·avgRowNnz` / `0.1·n` density rule (`HighsCutPool.cpp:540-566`) applies
  to **propagation only**, not to the LP.
- There is **no explicit cap on LP rows or nonzeros** in `HighsLpRelaxation`. The LP
  size is bounded only indirectly, by basic-slack aging and the selection filters.

## Q2. Warm starts when rows are removed

- Only rows with a **basic** slack are ever deleted (`HighsLpRelaxation.cpp:542, 660`).
  Dropping a basic slack with its row leaves a valid basis. `removeCuts` compacts
  `row_status` and re-solves (`:584-610`).
- Per node, HiGHS stores a `shared_ptr<const HighsBasis> nodeBasis` in `NodeData`
  (`HighsSearch.h:91`). It is set after the node LP (`HighsSearch.cpp:957-960`) and
  inherited by the children (`:155, :178`).
- Queue nodes (`OpenNode`) store **no basis**, only domain changes, bound and estimate
  (`HighsNodeQueue.h:142-160`). Bases exist only along the active dive.
- A stored basis is used only if its row count matches the current LP
  (`HighsSearch.cpp:788-792`). Otherwise HiGHS falls back to the first root basis with
  every extra row basic (as at `HighsMipSolver.cpp:711-719`).

## Q3. HiGHS's policy mapped onto discopt (batch LP = worker LP)

- (a) Cuts live in the shared LP, not in node-local copies. Per-node re-adding of pool
  rows has no HiGHS counterpart.
- (b) Cuts are added at "subtree-root" events: the batch boundary.
- (c) Rows are removed by basic-slack aging: limit 10, applied every 5 epochs.
- (d) Pool cuts age out fast: about 5 rounds for a new cut.
- (e) Selection uses the density-penalised score, the adaptive threshold and the 0.1
  parallelism filter.
- (f) No cut is longer than 100 + 0.15·n.

## Q4. Does the loss pattern match a HiGHS filter?

Partly; the source cannot settle it.

- The clear structural difference is that the fold admits rows on **usage count** and
  never removes them. In HiGHS, a row survives only while it is nonbasic with a nonzero
  dual (`HighsLpRelaxation.cpp:660-673`). On a deep tree the folded set goes stale away
  from the region that used it, and every node pays for it.
- The density filters that would bite are the 100 + 0.15·n length cap
  (`HighsCutGeneration.cpp:974`) and the 1/numActiveNzs score factor
  (`HighsCutPool.cpp:342`).
- A global nonzero-ratio cap is not a HiGHS mechanism.
- Which mechanism dominates on `bppc8-02`, `gsvm2rl3`, `beavma` and `obra` is a
  hypothesis. Measure it: folded rows, folded nonzeros / nnz(A), the share of folded
  rows that are basic at node optima, and iterations per node.

## Q5. Departures that could explain the near rows

1. Pool rows are re-added in a node-local LP that is solved twice; HiGHS keeps the
   persistent LP and separates once per subtree root (`HighsMipSolver.cpp:684-693`,
   `HighsSearch.cpp:957`).
2. Rows are never removed from the shared LP (`HighsLpRelaxation.cpp:629-676`).
3. Efficacy has no density penalty, and there is no parallelism filter
   (`HighsCutPool.cpp:342, 397-418`).
4. There is no cut-length cap (`HighsCutGeneration.cpp:974`).
5. The pool holds GMI only. HiGHS also separates path/c-MIR, lifted covers, mod-k,
   cliques and implied bounds at nodes (`HighsSeparation.cpp:37-39, 93-118`). Path/c-MIR
   is the likely missing strength on `mik-*` (inferred, not measured).
6. Pool aging is too slow: 30 *batches* unused, against about 5 separation calls for a
   new cut (`HighsCutPool.cpp:609`).
7. Separation runs on up to 8 nodes per batch, against once per installed subtree root.

## Recommended policy (expert)

1. **Cuts live in the shared LP only.**
   - Drop the node-local augmented re-solve for pool cuts.
   - Re-solve locally only for fresh node separation.
   - Fold the binding cuts into the next batch's shared LP.
   - Source: `HighsSeparation.cpp:150-166`, `HighsSearch.cpp:957`.
2. **Fold on activity, not on usage.**
   - At each batch boundary, take the pool cuts violated at the batch's node LP points.
   - Score each by viol / (numActiveNzs·‖a_active‖) and keep those ≥
     `minScoreFactor·bestScore`, with the factor starting at 0.9 and adaptive.
   - Source: `HighsCutPool.cpp:319-391`.
3. **Parallelism.** Within a fold set, reject a cut whose parallelism with an
   already-selected cut exceeds 0.1. Source: `HighsCutPool.cpp:400-418`.
4. **Length.** Reject, or cancel down, any cut with more than 100 + 0.15·n nonzeros;
   require violation > 10·feastol. Source: `HighsCutGeneration.cpp:974-985, 1336`.
5. **Removable rows with basic-slack aging.**
   - Track an age per shared-LP cut row. Increment it when the slack is basic in the
     node optima; reset it on a nonbasic row with |dual| > tol.
   - Every 5 batches, delete rows with age > 10.
   - Removing a basic-slack row keeps a basis valid. A stored basis where a deleted row
     is nonbasic falls back to the root basis with the new rows basic.
   - Source: `HighsLpRelaxation.cpp:629-676`; `HighsMipSolver.cpp:711-719`;
     `HighsSearch.cpp:788-792`.
6. **Removed rows** return to the pool at age 1. Source: `HighsCutPool.cpp:150-160`.
7. **Pool aging.**
   - New cuts start at `agelim-5` (limit 30); unviolated cuts age per scan.
   - Above a soft limit of 10000, shrink the effective limit and stop tableau
     separation.
   - Source: `HighsCutPool.cpp:166-170, 242-246, 279-310, 609`;
     `HighsTableauSeparator.cpp:47`.
8. **Separate only at "subtree-root" nodes.** The other nodes get one plain warm solve
   on the shared LP. Source: `HighsMipSolver.cpp:826-837`.
9. **Bases.** Children inherit the parent's basis; queue nodes need not keep one, so a
   basis invalidated by a deletion can be discarded. Source: `HighsSearch.cpp:155, 178`;
   `HighsNodeQueue.h:142-160`.
10. **Instrument before tuning** on `bppc8-02`, `gsvm2rl3`, `beavma` and `obra`. Any
    nonzero cap is discopt-specific and needs its own evidence.
