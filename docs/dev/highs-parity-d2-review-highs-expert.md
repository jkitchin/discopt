# HiGHS-expert critique of `highs-parity-d2-review.md` (2026-11-08)

Role: HiGHS expert (`python/discopt/skills/agents/highs-expert.md`). Every HiGHS claim is
checked against `ref/HiGHS/highs/` (1.15.1); paths below are relative to `ref/HiGHS/highs/mip/`
unless they start with `crates/`, `python/` or `docs/`. Read-only: no solver was run. The only
numbers computed here come from checked-in profiles
(`docs/dev/data/highs-parity/d2-analysis/w5d_cut_lp_exchange_entry.jsonl.gz`, default arm =
today's CUT_LP path, and `tol_robust_v2_entry.jsonl.gz`, `w5_expel_entry.jsonl.gz`).

**Retraction first (CLAUDE.md §11).** The W1 consult (`highs-parity-d2-w1-cut-lp-expert.md`
Q3(a)/(b), recommended policy items 1 and 8) said "per-node re-adding of pool rows has no HiGHS
counterpart" and mapped HiGHS's subtree-root separation to discopt's *batch boundary*. That
mapping is wrong. discopt does not plunge: every node it evaluates is popped from the queue.
HiGHS runs `separate()` at **every** queue-popped node, after its LP and **before** branching
(`HighsMipSolver.cpp:826-837` → `separateAndStoreBasis` `:684-693`). In that `separate()`, pool
re-separation, `addCuts`, `resolveLp` and the round loop all act on that node's LP
(`HighsSeparation.cpp:154-166`, `:181-213`). The node-local augmented re-solve that W1 removed
*was* the HiGHS mechanism. What it lacked was HiGHS's selection rule and the persistence of the
binding rows. The R1 regression follows from that consult.

---

## 1. Item by item

| item | verdict | one line (key source) |
|---|---|---|
| R1 | **agree, amend the mechanism** | Root cause is two concrete departures (§2), not only "no per-node lift". (a) 86–89 % of folded rows end lazily relaxed, and a relaxed row is never returned to the pool; HiGHS's `lpCutRemoved` returns it at age 1 (`HighsCutPool.cpp:150-160`). (b) Separation was moved off the popped node (`HighsMipSolver.cpp:826-837`). |
| R2 | agree, add two items | Add a `deterministic=True` arm. Add a check that run 1/run 2 split on the wall clock (`python/discopt/solver.py:40052-40056`). The split is a candidate source of the `p0548` bimodality (§4). |
| R3 | **agree, amend** | HiGHS's "c-MIR" is `tryGenerateCut`: lifted knapsack, mixed-binary or mixed-integer cover first, c-MIR only if more efficacious (`HighsCutGeneration.cpp:1382-1480`), over a VUB/VLB-transformed row (`HighsTransformedLp.cpp:37-65`). The tableau separator is the *same* generator on tableau-row aggregations (`HighsTableauSeparator.cpp:199-237`). The cuts must enter through the per-node loop of the R1 fix, not only the batch fold. Order: R1 → R3. |
| R4 | agree, amend | Path start rows must be **tight** (both slacks > feastol ⇒ `kUnusuable`, `HighsPathSeparator.cpp:49-52`), not "within 10 %" (`cmir.rs` `MAX_BASE_SLACK = 0.1`). The next column is the continuous one with the largest VUB-aware bound distance; the partner row is random among tight arc rows; the path stops at the first cut; both orientations; a path-mixing cut when the path is ≥ 2 rows (`HighsPathSeparator.cpp:170-420`). "Depth 6" alone is not the design. The `nexp`/`sp150x300d` primal half is root-heuristic redundancy, not the sub-MIP (Table 3 `heur_all_off`). |
| R5 | **disagree with the target list; demote** | Rust has no MIP presolve, so its HiGHS counterfactual is `presolve_off`, not `restart_off`. Under `presolve_off`, `beavma` takes 9 nodes and `mik-1/2/3` take 718/87/682 (≈ base) (Table 1). The root `mik`/`beavma` gaps close with cuts on the unpresolved model (Table 3 `root+presolve_off` 1.5–2.0 % / 0.82 %, 0 restarts). Restart only undoes what presolve does to these formulations. Realistic MIPLIB target: `bppc8-02` only. Build it at the existing re-entry point (§3, R0). |
| R6 | amend | HiGHS node LPs are dual simplex. Primal is a last-resort fallback (`HighsSearch.cpp:1450-1465`; `HighsLpRelaxation.cpp:1232-1250`). The HiGHS-faithful lever is to stop node solves landing in primal repair, not to price primal faster. Measure why `bppc8-02` makes 1,959 repairs before reviving incremental primal pricing. |
| R7 | agree (last) | Clique implications fire on every binary bound change (`HighsDomain.cpp:2098`). Clique *substitutions* also count toward the restart trigger (`HighsMipSolverData.cpp:1309-1313`), which R5 as written omits. |
| §3 gate | agree, four objections (§4) | Select on a recorded external filter, not on HiGHS's runtime. Align gap tolerances (`mip_rel_gap` default 1e-4, `lp_data/HighsOptions.h:1267-1270`). Record threads (HiGHS MIP is single-worker unless `parallel=on`, `HighsMipSolver.h:161-167`). The ≤ 2 s harm bar is mostly a presolve/root-closure bar. |

### Notes on the review's HiGHS descriptions (verified)

- **Correct.** Per-worker LP copy (`HighsMipSolver.cpp:287-301`). Separation per queue-popped
  node (`:684-693`, `:826-837`). Add-resolve-age (`HighsSeparation.cpp:162-166`). Path length 6
  (`HighsPathSeparator.cpp:170`). The tableau soft-limit gate (`HighsTableauSeparator.cpp:47`).
  `percentageInactiveIntegers` (`HighsMipSolverData.cpp:1309`). `performRestart` with
  `appendCutsToModel` (`:1316-1335`). The ≥ 10 % and ≥ 2.5 %/first-pass triggers
  (`:2127-2134`, `:2486-2497`).
- **Missing from the restart description.**
  - The trigger counts clique substitutions (`:1311`).
  - A restart keeps the cut pool and re-separates it (`separateLpCutsAfterRestart`, `:2084`;
    `HighsCutPool.cpp:462-500`).
  - It keeps pseudocosts (`HighsPseudocostInitialization`, `:1318-1320`).
  - There is also an in-tree huge-tree restart (`HighsMipSolver.cpp:475-540`).
- **`pool_soft_1` is not "tableau off, everything else on".** `mip_pool_soft_limit=1` also
  collapses the pool's effective age limit: to ≥ 5 in `performAging` and to ≥ 1 in `separate`
  (`HighsCutPool.cpp:166-170`, `:242-246`).
  - So it is tableau off **plus** pool cuts that die after about one unviolated scan.
  - `mik-5` is 209 vs 71 under it (3×), so "harmless" holds for `mik-1` only.
  - The inference that the tableau separator is not the `mik` lever stands. The secondary
    inference, that pool persistence does not matter for `mik`, also follows. That supports
    the cuts in the LP rows being what counts.
- **The ablation's own "the mik root is a presolve/restart product" (doc lines 77-79) is
  contradicted by its `root+presolve_off` column.**
  - `mik` roots reach 1.53–2.03 % with no presolve and no restart.
  - `root+restart_off` (16.4 %) is "presolve on, restart off". Presolve on these models removes
    what the cuts need, and the restart gives it back.
  - That is the opposite of what R5 assumes for the Rust engine.

---

## 2. R1: what the port left out, and the fix

### Evidence (checked-in profiles, CUT_LP default arm, `enlight8`, m = 64, n = 128)

From `w5d_cut_lp_exchange_entry.jsonl.gz` (`env={}`), run 1 / run 2:

| counter | run 1 | run 2 |
|---|---|---|
| `CutLpFolded` | 547 | 975 |
| `CutLpRelaxed` | **468 (86 %)** | **870 (89 %)** |
| `CutLpDeleted` | 83 | 165 |
| `CutLpDeleteBlocked` | 4,328 | 13,559 |
| fold rows basic at the node optimum (`NodeGmiFoldRowsBasic / Seen`) | 95.1 % | 96.7 % |
| mean basis nnz per factorization (`LuBasisNnz / LuSparseFactorizations`) | 1,528 | 2,850 |

The basis nnz is for a 64-row model. The panel shows the same pattern: nodes 47–50k OFF vs
9.6–11.4k ON, and `bppc8-02` 17–26k vs 1.8–2.1k (`w1_cut_lp_panel_r3.json.gz`).

### What HiGHS does that the port does not

1. **A row that leaves the LP goes back to the pool at once.**
   - `HighsLpRelaxation::performAging`/`removeObsoleteRows` call `lpCutRemoved`
     (`HighsLpRelaxation.cpp:549, 667`). That sets age 1 and makes the cut separable again at
     the next node that violates it (`HighsCutPool.cpp:150-160`, `separate` skips only
     `ages_ < 0`, `:253`).
   - discopt's *lazy deletion* frees the slack and leaves the row dead in the matrix
     (`crates/discopt-core/src/bnb/milp_driver.rs:4453-4462`). It never un-relaxes it and never
     returns the cut to `gmi_pool`.
   - The row keeps its signature in `gmi_pool_sigs` (`:4216` comment "their signature stays";
     rebuilt from all `cut_lp_rows`, relaxed included, `:4527-4531`). So a re-separated copy is
     refused at `:4203`.
   - **That is the "pool cuts violated at a node stop tightening it" mechanism, and it hits
     86–89 % of folded rows on `enlight8`.** A relaxed cut is in neither the LP nor the pool.
   - Second-order effect: the dead rows stay in the basis dimension (mean factor nnz 1.5k → 2.9k
     on a 64-row model). That fits the 5× node-rate collapse.
2. **Separation, pool cuts included, happens at the popped node before it is branched, in
   rounds.**
   - `HighsSeparation::separationRound` → `cutpool.separate(sol, propdomain, …)` on *this
     node's* LP point → `addCuts` → `resolveLp(&propdomain)` → `performAging(true)`
     (`HighsSeparation.cpp:154-166`). Rounds repeat while the objective gains more than 1 %
     (`:199-213`).
   - The node's bound and the point it branches on are post-cut.
   - discopt CUT_LP scores violated pool cuts at the *pre-cut* point, returns the unlifted bound
     (`milp_driver.rs:5424-5435`; `pool_pick` stays empty under `cut_lp`), and selects over the
     **union of the batch** (max score per cut, `:3818-3828`) at the boundary.
   - The rows then enter an LP shared by unrelated subtrees, where 95–97 % of them are basic at
     the next optima. They are selected for points that no longer exist.
   - In HiGHS, the rows in a worker LP are, by construction, the ones that were binding at that
     worker's last separation.
3. **No stored bases on queue nodes** (`HighsNodeQueue.h:162-171`, `OpenNode` has no basis).
   - A basis whose row count no longer matches is dropped (`HighsSearch.cpp:788-792`,
     `:1861-1865`), so HiGHS can always delete physically.
   - This is the root of discopt's "basic in every stored frontier basis" condition
     (`milp_driver.rs:4449-4452`), and so of lazy relaxation.
   - It is known: W5(d) and the recent-basis fallback were both measured worse
     (`highs-parity-d2-plan.md` W1 table and W5(d)). So I do **not** propose changing it.
     Fix 1 makes it harmless for cut supply.
4. **Not the cause on `enlight8`: pool-cut propagation.**
   - HiGHS propagates sparse pool cuts in every node domain, whether or not they are in the LP
     (`HighsDomain.cpp:448-507`, `:2500-2530`; admission `HighsCutPool.cpp:540-566`, length
     < max(2·avg row nnz, 0.1·n)).
   - On `enlight8` that limit is about 12.8 nonzeros. The Rust GMI cuts average 52
     (`SepGomoryCutNnz / SepGomoryCutsEmitted` = 592,074 / 11,454), so HiGHS would not propagate
     them either.
   - Worth a later item for sparse c-MIR/cover cuts (R3), but not part of R1.

### Faithful fix (behind the existing flag, and a new sub-flag for step C)

- **A. Relaxed = removed** (`HighsCutPool.cpp:150-160`).
  - When a folded row is lazily relaxed, push its cut back into `gmi_pool` at age 1 and remove
    its signature from `gmi_pool_sigs`.
  - A later violation then re-selects it.
  - Optional, measured separately: re-admit into the *same dead slot* by restoring the slack
    bounds, instead of appending a row. That gives zero matrix growth. Caveat: a slack that is
    nonbasic in a stored basis becomes a binding equality in that basis. If the warm start is
    dual infeasible there, re-admit as a new row and leave the slot dead.
  - Bound-neutral per LP: the set of valid cuts is unchanged. It is bound-changing for the tree
    (more cuts stay available).
- **B. Fold on post-cut activity, not on pre-cut violation.**
  - Fold into the shared LP the pool cuts that are **nonbasic with |dual| > 1e-7** at the
    node-local post-cut optima (the same activity test as the aging,
    `HighsLpRelaxation.cpp:660-673`).
  - This is HiGHS's "rows in the worker LP are binding at the last separation".
  - Kill-check: the 95–97 % basic share must fall.
- **C. Restore per-node pool separation where HiGHS would do it.**
  - On the `NODE_GMI_PER_BATCH` best-bound nodes (the analogue of HiGHS's best-bound pops;
    dives get none, `HighsMipSolver.cpp:826-860`), and on any node whose top pool score clears
    the adaptive threshold, run HiGHS's per-node selection *at that node's point*. That means
    `cut_pool_score`, `minScoreFactor·best`, and the 0.1 parallelism within this node's cutset
    (`HighsCutPool.cpp:229-458`).
  - Add the selected cuts to the node-local LP and re-solve. Loop while the objective gains more
    than 1 % of the previous gain (`HighsSeparation.cpp:199-213`).
  - Return the **lifted bound and the post-cut point** to the tree, so the node branches where
    HiGHS would. The pre-W1 path returned the lifted bound but branched on the pre-cut `xs`.
  - The pre-W1 cost (`NodeGmiLp` 2× node LP on `istra`) came from doing this at *every* node
    with any violated cut. HiGHS pays it once per subtree root, so the slot gate is the faithful
    frequency.

**Entry** (6 solves plus counters, before building B/C):
- Add the counter `CutLpRelaxedViolated`: relaxed rows whose cut is violated by more than
  feastol at a later node LP point.
- Run `enlight8` and `bppc8-02`, `CUT_LP=1`, 3 reps.
- Kill hypothesis 1 if `CutLpRelaxedViolated` ≈ 0. Then the loss is all in item 2 and A is a
  no-op.

**Then:**
- Run A alone, then A+B+C, ON vs `CUT_LP=0`, on `enlight8`, `bppc8-02`, `istra`, `b-ball`,
  `kaihu` and `fiber`.
- Kill if `enlight8` is not 3/3 or any W1 winner loses a rep.

---

## 3. What HiGHS does that the review omits and the data says matters

**R0 (new): the re-entry restart throws away the tree, the pool and the root cuts.**
- `_solve_milp_simplex` stops run 1 at `max(0.5, min(0.5·remaining, 10 s))`
  (`python/discopt/solver.py:40052-40056`, `_SIMPLEX_MILP_BUDGET_CAP_S = 10.0` at `:38590`).
- On an uncertified `feasible` it restarts the Rust tree from scratch, seeded with the incumbent
  (`:40196-40290`). Run 2 has an empty GMI pool and, with `DISCOPT_MILP_REENTRY_CUTS` off, a
  single-pass root cut budget (`:39723-39739`).
- In every uncertified two-run row in the three entry files above (23 rows, 20 s):
  - run 2 never beat run 1's bound;
  - `enlight8`: 6.5 s → bound 20, then 13.5 s → 19;
  - `beavma`: 352,347 vs 341,758;
  - `nexp-50-20-1-1`: 25 vs 24;
  - `bppc8-02`, `sp150x300d`, `neos-911970`, `obra`: equal.
- So **on the bound-limited target rows, half or more of a 20 s budget re-proves a weaker
  bound.**
- HiGHS's only restart (`performRestart`, `HighsMipSolverData.cpp:1316-1420`) keeps:
  - the cuts, as model rows (`appendCutsToModel`, `:1333`);
  - the pool (`separateLpCutsAfterRestart`, `:2084`);
  - the pseudocosts (`:1318`);
  - the incumbent.
- It also re-presolves with the new fixings.
- Faithful options, in order of cost:
  - (i) when run 1 holds an incumbent, do not stop it: move the #291 stall guard into the driver
    as a "no incumbent by T" deadline. The `#280` swap-reseed measurement (`:40160-40163`) must
    be re-checked;
  - (ii) make run 2 a HiGHS restart: run 1's cut rows appended as model rows, its pool, its
    pseudocosts, and the root reduced-cost fixings under run 1's incumbent. **This is where R5
    should be built.**
- At the shipped 3600 s budget the loss is only run 1's ≤ 10 s. At the review's proposed
  20 s/120 s gate it is large, so it must be settled **before** the gate is run.
- Entry: zero code. Read the `calls[]` arrays already in the entry files, as above. Then run
  `enlight8`, `mik-1` and `bppc8-02` with run 1 given the full 20 s (`deterministic=True` does
  this, `:40052-40054`) vs default.

**Root heuristics on `nexp`/`sp150x300d`.**
- Table 4: no single sub-heuristic matters (`fj`/`rins`/`rens`/`rootredcost` all 1 node), but
  `heur_all_off` is 119×/20×/12×.
- Table 3 `root+heur_all_off`: dual 0.00 but primal 20.7/53.5/2.9 %.
- HiGHS's root runs ziRound, randomized rounding and shifting on the first and the post-cut root
  points, plus central rounding (`HighsMipSolverData.cpp:2110-2115`, `:2213`, `:2304-2320`),
  besides RENS/FJ. Each is redundant with
  the others, and all are cheap once the root is closed.
- The review's "primal half rides on the sub-MIP" is not what Rust shows: `nexp-1-1` is at a
  41 % primal gap after 20 s.
- Pair R4 with a root rounding pass on the post-cut root LP point.

**Tableau-row bases for the R3 generator.**
- `b-ball` and `neos-911970` need HiGHS's *tableau* separator: `cuts_throttle` and `pool_soft_1`
  both FAIL.
- That separator is not GMI. It runs lifted-cover/c-MIR on `B⁻¹A` row aggregations with VUB
  substitution and global bounds (`HighsTableauSeparator.cpp:199-237`, `HighsSeparation.cpp:128`).
- Rust's in-tree GMI on the same rows is a different cut. Feeding R3's generator the tableau
  rows Rust already computes for GMI is cheap, and is HiGHS's design.

---

## 4. The gate (§3 of the review): HiGHS-side objections

I agree with the shape: held-out, two budgets, harm-bounded. I have four objections.

1. **Selection.**
   - "HiGHS single-thread ≤ 60 s" selects on the comparator's outcome. It regresses toward
     instances that are HiGHS-easy by luck of trajectory.
   - Use MIPLIB 2017's own `easy` status plus recorded n/nnz caps, and report HiGHS failures in
     the sample as well.
2. **Same certificate.**
   - HiGHS stops at `mip_rel_gap` 1e-4 by default (`lp_data/HighsOptions.h:1267-1270`) and
     `mip_abs_gap` 1e-6.
   - "Certified" must mean the same gap in both arms, with the tolerance written in the gate.
3. **Threads.**
   - HiGHS 1.15 MIP is one worker unless `parallel=on` (`HighsMipSolver.h:161-167`). Rust's batch
     runs on `threads_available()`.
   - Either pin both to one core or state the core count. The ≤ 2× SGM bar is otherwise
     comparing different hardware budgets.
4. **What the "≤ 2 s" harm bar measures.**
   - The instances HiGHS certifies in ≤ 2 s are dominated by 1-node solves, closed by presolve
     plus root cuts (Table 1 `base`: 12 of 25 instances take 0–1 nodes).
   - That bar is in effect a presolve/root-closure bar. Under it, presolve/restart (R5) is
     *gate-critical* even though it is worth ≤ 1 row on MIPLIB-38.
   - The plan should say so rather than discover it.

**On variance (supports R2).**
- HiGHS at `threads=1` is deterministic: Table 5 reproduced node counts exactly. Three HiGHS
  reps add nothing.
- The Rust coin flips are not intrinsic. The run 1/run 2 boundary is wall-clock-decided
  (`solver.py:40052-40056`; #1116 already notes this as a role-2 clock), so the seed and state
  run 2 starts from move with host load.
- Add a `deterministic=True` + `max_nodes` arm to R2. If `p0548` stops being bimodal there,
  the instrument noise is partly a discopt orchestration artifact, removable by R0.

---

## 5. Recommended re-ranking

1. **R1 (amended: A, then B+C)**, the only item with a known-good endpoint. It is also the
   prerequisite through which R3's cuts must flow.
2. **R0 (new)**: no stop and restart from scratch on a bound-limited row. Either run 1
   continues, or run 2 becomes a HiGHS restart (cuts as rows, pool, pseudocosts, root
   reduced-cost fixings). Do this before any gate at 20 s or 120 s. It absorbs R5's restart half.
3. **R2**, plus the deterministic arm and the run-split check.
4. **R3** (lifted-cover-first generator, path and tableau-row bases, through R1's per-node loop).
5. **R4** (tight start rows, VUB-aware bound distance, path mixing), plus a root rounding pass.
6. **R6** (re-scoped: why node solves go primal).
7. **R5's presolve half** (only `bppc8-02` on MIPLIB-38, but gate-critical under the ≤ 2 s
   bar).
8. **R7**.

Expected gain on the review's own accounting: R1 +1 (`enlight8`), R3 +2–4 (`mik`), and R4 + root
heuristics +0–3 (`nexp`×2, `sp150x300d`; `beavma` belongs here, not in R5). R0's gain is not
estimable from the files; its entry is zero-code.
