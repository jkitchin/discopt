# Phase B design consult: cut machinery that moves the dual bound (HiGHS-expert)

Date 2026-11-06. Read-only consult; nothing in the repo was edited. Tree `8792bff9`. HiGHS
source: `scratchpad/highs_src/highspy-1.15.1/highs/` (`H/` below). discopt Rust:
`crates/discopt-core/src/` (`D/` below). Data re-read from
`docs/dev/data/highs-parity/d2-ablation/ablation_phase{1,2}.jsonl.gz` (the per-row
`log.root_end_bound`, `log.last_cuts` = pool size, `log.last_inlp` = cut rows in the LP,
`log.report.iters_separation`). These fields are in the committed data but are not tabulated
in the ablation doc. Section 1 tabulates them, and they change the build order.

## 0. Answer in five lines

1. On the dual-limited target rows, HiGHS's cut lever is **in-tree separation with a pool and
   aging. It is not a stronger root.** In the no-presolve world (`all_off` → `keep_cuts`),
   the root gap barely moves on iskar/istra/kaihu/jijia: 7.23→6.64, 5.94→6.13, 6.17→6.18,
   5.92→6.03 %. Over the same rows the pool grows from ~70–100 cuts to **1.9–3.5 k**, the LP
   carries only **100–260 cut rows**, separation costs **5–13 % of LP iterations**, and every
   one of them certifies. The mik family shows the same pattern: the root gains ~1 pp
   (5.7–8.3 → 4.5–7.1 %), and the tree closes from a root that is *worse* than Rust's
   3.8–4.2 %. On sp150x300d the root gap is 92.9 % in both arms, and the instance certifies
   in 114 nodes with 39 % of LP iterations spent on separation.
2. Rust has no in-tree separator that can move these bounds. The only node separator is
   pure-binary knapsack covers. It is off on the route, capped at `2m ≤ 128` rows, and those
   rows are never removed. Root cuts are dense GMI: 190–295 nnz/cut on iskar/istra/jijia,
   versus 8 nnz per matrix row. They are frozen into every node LP. The route's run 2 even
   drops the root cut budget.
3. So the build order is: (B0) zero-code entry runs; (B1) node separation with the
   globally valid separators Rust already has; (B2) cut-row aging plus a sparse pool on the
   shared matrix; (B3) tableau-row c-MIR, restored from the retired `tableau_cmir.rs`, with
   a global-bound transform so it is valid at nodes; (B4–B6) cover complementation,
   aggregation depth, and implied-bound/clique cuts, each only if its residual entry says so.
4. GMI never goes to nodes. It is node-local by construction because it substitutes the
   node's nonbasic bounds. HiGHS has no GMI at all. HiGHS keeps the per-node cost low with
   about seven filters (§2); discopt has the row-length cap and an efficacy/parallelism
   selector with `max_parallel = 0.99` (HiGHS's pool re-entry uses **0.1**).
5. Every step has a falsifying entry run on the real rows before any graduation work. Two
   of them need no code: (a) new HiGHS option arms; (b) the existing `cmir_cuts` and
   `node_cuts` keywords on the direct driver.

## 1. Evidence: what HiGHS's cuts do on the 25 rows (no presolve, no heuristics, no SB)

`all_off` = plain B&B with separators throttled (tableau off, everything aged out at once;
path, mod-k, clique and implied-bound cuts still run each root round). `keep_cuts` =
`all_off` with cuts restored (tableau separator, node separation, aging). Gap % =
|opt − bound| / |opt|.

| row | all_off root | keep_cuts root | all_off end (20 s) | keep_cuts end | pool all_off→keep_cuts | cut rows in LP (keep_cuts) | sep share of LP iters |
|---|---|---|---|---|---|---|---|
| neos-3610040-iskar | 7.23 | 6.64 | 5.29 FAIL | 0.01 ✓ 6.0 k n | 69→2473 | 103 | 5 % |
| neos-3610051-istra | 5.94 | 6.13 | 3.98 FAIL | 0.01 ✓ 4.8 k n | 96→3490 | 258 | 5 % |
| neos-3611689-kaihu | 6.17 | 6.18 | 3.24 FAIL | 0.00 ✓ 1.5 k n | 79→3114 | 141 | 13 % |
| neos-3611447-jijia | 5.92 | 6.03 | 0.01 ✓ | 0.00 ✓ | 64→1942 | 137 | 9 % |
| mik-250-20-75-1/2/3/5 | 5.7–8.3 | 4.5–7.1 | 1.0–2.0 FAIL | 0.01 ✓ 0.9–4.6 k n | ~85→830–1210 | 149–188 | 11–15 % |
| sp150x300d | 92.91 | 92.91 | 10.14 FAIL | 0.00 ✓ 114 n | 132→2431 | 156 | 39 % |
| b-ball | 16.28 | **6.55** | 14.89 FAIL | 0.00 ✓ 7.6 k n | 16→683 | 19 | 2 % |
| neos-911970 | 7.35 | 4.82 | 6.07 FAIL | 0.75 FAIL | 162→5535 | 141 | 20 % |
| fiber | 6.18 | 2.18 | ✓ | ✓ | 46→3770 | 81 | 18 % |
| nexp-50-20-1-1 | 2.84 | 3.82 | ✓ | ✓ | 78→3120 | 98 | 26 % |

Readings:

- **Only b-ball (and neos-911970 in part) is a root-cut row.**
- **neos-361\*, mik and sp150x300d are in-tree-separation rows.**
- **The "all_off" roots are already far above the LP bound.** fiber goes 61.5 → 6.2 %,
  nexp-1-1 58.5 → 2.8 %, neos-911970 57.5 → 7.35 %, mik ~LP → 5.7 %. These come from
  HiGHS's *non-tableau* root separators: path c-MIR with variable bounds, implied-bound,
  clique and mod-k. They run even in `all_off`.
- Rust's root on fiber is still at the LP gap (61.55 %, zero root cuts on the route,
  `p4.4/root_head.json`). Rust's 20-s dual gap on nexp-1-1 is 41 % (improve-dive panel).
  This is the second, separate root gap: the path-class c-MIR, which Rust has (`cmir.rs`)
  but **does not run on the pure-MILP route**.

**Rust today on the same rows.** Two sources: `d2-analysis/r1_improve_dive_panel.json.gz`
(3 reps, 20 s) and `b4_60s.jsonl` (60 s):

- **Bounds are frozen at an integer in every rep**, the objective-lattice floor:
  istra 48 (opt 49), iskar 36 (37), kaihu 115 (119), jijia 104 (107).
  Mik dual gap is 3.3–3.4 %.
- **60 s does not move istra** (still 48.0). Run 2 of the route is called with no cut
  options: `b4_60s.jsonl` `calls[1].kw` = `abs_gap_tol`, `max_open_nodes`, `time_limit_s`.
  So run 2 falls back to `root_cuts=16, cut_rounds=1` and no node cuts.
- **Root GMI density** (`SepGomoryCutNnz` / `SepGomoryCutsEmitted`, b4 profiles):

  | row | nnz/cut | n |
  |---|---|---|
  | istra | 295 | 805 |
  | jijia | 211 | 472 |
  | iskar | 193 | 430 |
  | mik-1 | 109 | 270 |
  | b-ball | 11 | 100 |

  The kept set is 21–91 rows.
- **b-ball is variance-dominated.** The dual gap is 0.024 % in `b4_60s` but 20 % in all
  3 improve-dive reps. Judge it on reps, never on one run.

## 2. How HiGHS keeps per-node separation cheap (the GMI-failure question)

The discopt GMI-repair failure was dense rows in every node LP (fiber ~297 nnz/cut; node
throughput 2058 → 13). HiGHS stacks the following controls, in pipeline order:

1. **Choice of base rows** (tableau, `H/mip/HighsTableauSeparator.cpp`):
   - only fractional basic integers, or fractional *integral-row* slacks (:60-80);
   - tries capped at `200 + 0.1·min(m, n_int)` per call, with a lifetime budget of
     `5000 + 50·calls + LPiters/10` (:85-96);
   - candidates ranked by `f(1−f)/‖row_ep‖²` (:171-173); a row is rejected if its weight
     spread exceeds 1e4 (:171), its aggregated row is too long (:205-206), or its
     structural coefficient range exceeds 1e6 (:221);
   - the loop stops at score < 0.0025·best, or 0.01·best after 50 cuts (:190-197).
2. **Base-row preprocessing** (`H/mip/HighsCutGeneration.cpp:901-1037`):
   - positive continuous terms and negligible terms are relaxed out (:944-959);
   - row length is capped at `100 + 0.15·ncols`, but only by cancelling terms that sit at
     a bound (:974-1003). discopt's `CUT_LEN_BASE/FRAC` are the same constants
     (`D/bnb/milp_driver.rs:62-63`), applied to the cut after slack expansion instead.
3. **c-MIR itself drops every continuous term with a positive coefficient** (`:700-708`).
   GMI keeps both signs (`ψ_j = ᾱ/f0` or `−ᾱ/(1−f0)`). This is the structural density
   difference between HiGHS's tableau cuts and discopt's GMI.
4. **Postprocessing**: coefficients below `100·feastol·max` are moved into the rhs (:784-808),
   the cut is scaled to integers if possible (:821-870), it must be violated by more than
   10·feastol (:1184), and its coefficients are tightened (:1186).
5. **Cuts go to the pool, not the LP.** Pool re-entry selection (`H/mip/HighsCutPool.cpp:
   222-458`):
   - **score = viol / (numActiveNzs · ‖a_active‖)** (:342), an explicit density penalty;
   - a candidate is dropped if its cosine with an already-selected cut, in any pool,
     exceeds **0.1** (:400-417);
   - the score threshold adapts (:383-397).
6. **Aging.** An LP cut row whose slack is basic gains age, the age resets on a nonzero
   dual, and rows past `mip_lp_age_limit` (10) are deleted, checked every 5th epoch
   (`H/mip/HighsLpRelaxation.cpp:629-676`). Pool rows age per separation call (:271-305) and
   per `performAging` (:162-220). New cuts start at `agelim−5` (:614), so a cut not reused
   within 5 calls dies. The soft limit of 10 k lowers the age limit dynamically.
7. **Propagation density gate**: a cut longer than `max(2·avg row nnz, minDensityLim)` is
   not propagated, and total propagated nnz is capped at 2× the model's (:548-590).

The measured result on the target rows (§1) is pools of 1–5 k cuts, but only 100–260 cut
rows in the LP and 5–15 % of LP iterations in separation.

**Placement in the tree** (`H/mip/HighsMipSolver.cpp:684-694, 823-838`):
- separation runs once per node *popped from the queue*. Dive and plunge children only
  age (`H/mip/HighsSearch.cpp:204, 581, 958`; `performAging(false)` never deletes);
- separation rounds repeat on the hot LP only while the objective gains more than 1 % of
  the gain so far (`H/mip/HighsSeparation.cpp:189-217`);
- tableau and path cuts at nodes are transformed with **global** bounds
  (`HighsSeparation.cpp:132-133`; `H/mip/HighsTransformedLp.cpp:155-165`), so node cuts are
  globally valid.

**Parallel workers in 1.15.1** (the closest analogue of discopt's 64-node batch):
- each worker has a local pool plus the global pool (`H/mip/HighsMipSolver.cpp:311-330`);
- duplicates are checked against both (`HighsCutPool.cpp:531-540`);
- after each parallel round, `syncPools` merges the worker pools into the global one and
  ages the global pool (`HighsMipSolver.cpp:336-347`).

## 3. discopt today (what Phase B builds on)

- **Root loop** (`D/bnb/milp_driver.rs:1770-2017`):
  - separators: cover (1897), c-MIR **only if `opts.cmir_cuts`** (1911-1930), GMI (1932-1946);
  - slack rewrite (1950-1980), then selection (1983-1990: efficacy ≥ 1e-4, parallelism
    ≤ 0.99, consts :44/:54);
  - the tail-off test is objective-only (1887);
  - one slack-row cleanup after the loop (2019-2128).
- **Route** (`python/discopt/solver.py:39398-39467, 39862-39884`): 500 cuts / 50 rounds /
  select, with no `cmir_cuts` and no `node_cuts`. Binding defaults are `cmir_cuts=false`,
  `node_cuts=false`, `max_pool_cuts=128` (`crates/discopt-python/src/lp_bindings.rs:
  1068-1074`). Run 2 passes none of these.
- **Node cuts**:
  - `separate_cover_csc` at fractional nodes only (3426-3444), using the global
    `ctx.l_w/u_w`, so the cuts are globally valid;
  - gated by row density < 0.5 and capped at `2·m ≤ max_pool_cuts` (1716-1719);
  - deduped in node order and folded at the batch boundary (2519-2523, 2733-2753);
  - **never removed**.
- **Requeue-after-separation plumbing exists** for the `MilpNodeHook` path (2567-2620):
  the node is re-queued with its raised bound and re-solved against the folded rows.
- **Separators**:
  - `cover.rs:125-165` refuses any negative binary weight and any row without a `+1`
    slack, so bin-packing capacity rows (`Σw x − C y ≤ 0`) never yield covers;
  - `cmir.rs` aggregates original rows only, with max depth `CMIR_MAX_AGGR = 2`
    (driver :49; HiGHS path: 6, `H/mip/HighsPathSeparator.cpp:170`);
  - `cmir.rs` uses only start rows within 10 % slack (`cmir.rs:79`) and the delta set
    {1, 1/|a_k|}×{1,2,4,8}, with no complementation-flip pass (HiGHS
    `HighsCutGeneration.cpp:646-695`);
  - slacks are always continuous (`milp_driver.rs:1474-1475`); HiGHS treats
    integral-row slacks as integer.
- **Cut storage is dense**: `GomoryCut.coeffs: Vec<f64>` of length n
  (`D/lp/gomory.rs:262-266`). Parallelism and pool scans are O(n) per pair.
- **No basis factor is exported** from an LP solve (`D/lp/simplex/mod.rs:328-353`). Tableau
  rows at a node need a fresh LU of the node basis (`gomory.rs:470-480` pattern) plus one
  refined btran per row (`gomory.rs:355`).
- **Retired, recoverable**: `lp/tableau_cmir.rs` (1167 lines; HiGHS-style tableau c-MIR,
  both signs, variable-bound substitution, delta search, enumeration soundness tests).
  Retired 2026-11-04 (`docs/dev/flag-retirement-audit.md:490`) on the #1917 **MINLP
  OA-master** replay panel, root only: cert-clean, not net-positive. Recoverable from
  commit `3eba9bf` (branch `archive/g32-coef-tighten-cmir`). It was never measured on
  MIPLIB rows or at nodes.

## 4. Build order (each step its own default-OFF flag + audit row)

Bound-changing throughout except where marked. Graduation follows the owner's decision
(analysis follow-up 8): re-panel together with `DISCOPT_MILP_IMPROVE_DIVE` under the same
pre-registered bars, **plus** the P4.4 lesson's bars on the open population: no incumbent
lost in any rep, and median nodes/s on the open rows ≥ 0.5× OFF. All entries run on the
**direct driver** (`d2_direct_driver_panel.py` style), because the route's run 2 drops cut
options. The rows are variance-dominated (follow-up 4), so use ≥ 2 reps.

### B0. Zero-code entries that fix the order (≈ 1–1.5 h of compute)

- **E0a: HiGHS arms** (add to `d2_highs_ablation.py`; 25 rows, threads=1, 20 s, 2 reps):
  - `keep_cuts_rootonly` = `keep_cuts` + `mip_allow_cut_separation_at_nodes=False`;
  - `keep_cuts_noage` = `keep_cuts` + `mip_lp_age_limit=10**6`, `mip_pool_age_limit=10**6`;
  - `keep_cuts_notableau` = `keep_cuts` + `mip_pool_soft_limit=1` (tableau off everywhere;
    this confounds with faster pool aging, see `HighsCutPool.cpp:168-171`, so read it as
    an upper bound).

  It answers three things: whether node separation is necessary (predicted yes on
  neos-361\*/mik/sp150x300d from §1), whether aging matters for certification or only for
  cost (read LP iterations per node), and whether tableau rows matter at nodes or
  path/clique/implied cuts suffice. Also log the root pool via `kCallbackMipGetCutPool`
  (fires after root separation, `H/mip/HighsMipSolverData.cpp:2353-2356`; presolve off, so
  original space) to get HiGHS's **nnz/cut** next to Rust's GMI 190–295.
  - **Decision rule**: if `keep_cuts_rootonly` certifies ≥ 8 of the 10 rows `keep_cuts`
    rescues, invert the order and put B3 at the root before B1.
- **E0b: Rust arms with no code** (`solve_milp_csc_py` keywords already exist): {`cmir_cuts`
  F/T} × {`node_cuts` F/T} at the route's cut budget, `max_pool_cuts` raised to 2000.
  Record the root bound (`max_nodes=2`, the p44 probe's convention) and the 20-s outcome on
  all 25 rows. Today the Rust root is known on only 6 of them (`p4.4/root_head.json`).
  Compare against the `all_off root` column in §1.

### B1. Node separation with the globally valid separators Rust already has (`DISCOPT_MILP_NODE_SEP`)

- **HiGHS.** One separation round per queue-popped node, repeated while the objective gain
  exceeds 1 % (`H/mip/HighsSeparation.cpp:181-217`). The pool is re-separated every round
  (:157-162). Global-bound transform.
- **discopt.** Node cover only, off on the route, permanent rows.
- **Change.**
  - In `solve_node`, after a fractional Optimal LP, run `separate_cover_csc` and
    `separate_cmir_csc` with `ctx.l_w/u_w` (global, so the cuts are valid everywhere).
    The second call is about 20 lines next to 3432.
  - Separate only at the **k nodes whose bound is within ε of the batch's lowest bound**
    (start with k = 8). The global dual bound is the frontier minimum, and these nodes are
    what moves it. Separating all 64 is cost without bound.
  - At the boundary: dedup in node order (existing), select across the whole batch (cosine
    ≤ 0.1 between batch-mates, HiGHS's cross-pool rule), fold, and re-queue the
    separated nodes with `raise_node_bound` (reuse 2590-2610).
  - Budget: separation LP pivots ≤ 20 % of tree pivots. Measured HiGHS share on the
    targets: 5–15 %, sp150x300d 39 %.
  - Lift the `2m ≤ 128` cap; the pool cap becomes B2's job.
- **Entry.** Rows: iskar, istra, kaihu, jijia, mik ×4, sp150x300d, b-ball, nexp-1-1. Direct
  driver, 20 s, 2 reps, flag ON/OFF interleaved. Hypothesis: the dual bound moves ≥ 0.5 pp
  on ≥ 4 of 10 rows (for istra that means crossing 48). Report `NodeSepCalls`/`NodeSepCuts`
  (§6 anti-vacuity) and nodes/s.
- **Kill.** Fewer than 2 of 10 move, or nodes/s drops more than 2× with no bound gain. Then
  original-row families are not the node lever: keep the plumbing for B3 and retire the
  cmir-at-node call.

### B2. Cut-row aging and a sparse pool on the shared matrix (`DISCOPT_MILP_CUT_AGING`; cost control)

- **HiGHS.** §2 items 5–7 (aging every 5th epoch, pool re-entry by density-penalised score,
  new cuts start old).
- **discopt.** Rows are permanent after the one-shot root cleanup; cuts are dense vectors.
- **Change.**
  - A `CutPool` holding sparse `(idx, val)` rows, rhs, age and hash. Dedup moves off
    `cut_signature` over dense vectors.
  - Per matrix cut row, age += 1 at each batch boundary where its slack was basic at
    **every** Optimal node of the batch. Reset the age if the slack is nonbasic, or the
    dual is nonzero, at **any** node. The matrix is shared by 64 nodes, so the active set
    must be the union of what any node uses.
  - Every K = 5 boundaries, compact: delete rows past age 10 into the pool, and remap
    stored node bases by dropping deleted rows and slacks. A basis whose deleted slack was
    *nonbasic* is one column too many. Demote the lowest-index structural basic to its
    nearest bound and let `solve_warm` repair, the HiGHS `isBasisConsistent` fallback
    (`H/mip/HighsMipSolver.cpp:704-714`). Count this as `CutAgeBasisRepairs`.
  - Pool re-separation at B1's k nodes: violation > tol, score =
    viol / (nActive·‖a_active‖), cosine ≤ 0.1. Delete pool rows unused for 30 calls.
  - Apply the same to root cuts, which makes `root_cut_prune` a special case.
- **Entries.**
  1. *Identity* (bound-neutral arm): with age limit = ∞, nodes and objective are
     bit-identical on the P6.6-certified rows.
  2. *Value*: the dense-cut case, `DISCOPT_MILP_ROOT_GMI_REPAIR=1` with aging ON vs OFF, on
     fiber, beavma, neos17, sp150x300d, nexp-4-2 in a 10-s slice. Hypothesis: nodes/s
     recovers ≥ 5× toward repair-OFF while the root bound keeps the repair's gain (fiber
     19.76 %), and no incumbent is lost (the three rows that lost theirs under P4.4).
- **Kill.** nodes/s < 2× repair-ON, or `CutAgeBasisRepairs` > 20 % of warm node solves.
- **Binding negative respected.** "In-tree cut aging" (A2) was aging of a root set judged
  on node counts. This step is judged on per-node cost and on whether it lets denser
  families run at all, never on nodes alone.

### B3. Tableau-row c-MIR at the root and at nodes, global transform (`DISCOPT_MILP_TABLEAU_CMIR`; restore, do not rebuild)

- **HiGHS.** `H/mip/HighsTableauSeparator.cpp:41-244`. The row is `w = B⁻ᵀe_i`, aggregated
  over the LP rows including cut rows, which is valid for any `w`. `generateCut` is called
  on both signs (:232, :237), followed by `transform` on global bounds, lifted cover
  vs c-MIR (`HighsCutGeneration.cpp:1382-1480`), postprocess, and pool.
- **discopt.** The archived `tableau_cmir.rs` already does row × both signs, variable-bound
  substitution and the delta search, with 45 k-point enumeration tests. Three things are
  missing, and all three must be added:
  - (a) HiGHS's base-row filters (§2 item 1);
  - (b) the global box at nodes. Pass `ctx.l_w/u_w`, never the node's
    `full_l/full_u`. Columns fixed in the global box (completion artificials,
    `ub − lb < eps`) go to the rhs (`HighsTransformedLp.cpp:170-174`) instead of the slack
    rewrite that densified P4.4's GMI;
  - (c) integral-row slacks treated as integer.
  GMI stays root-only; it is node-local, because it substitutes the node's nonbasic bounds.
- **Re-proposal check.** This is not the retired experiment as-is. The population changes
  (MIPLIB dual rows, not OA masters), the placement changes (nodes), and the bars change.
  It must clear its own entry.
- **Entry (root, after restore).**
  - On the 25 rows, at the same root basis: nnz/cut and root bound, tableau-c-MIR vs GMI.
  - Bars: nnz/cut ≤ ⅓ of GMI's on istra/iskar/jijia (GMI 190–295); root bound within
    0.5 pp of GMI's on the mik rows; b-ball root ≤ 9 % (HiGHS `keep_cuts` 6.55 %, `all_off`
    16.28 %). Soundness via the archived enumeration tests plus the panel incumbents
    (feasible-point sampling).
  - Node entry: B1's protocol with tableau rows added at the k nodes. Count the cost of one
    LU per separated node; no factor is exported (§3).
- **Kill.** Not sparser than GMI and not stronger on b-ball → stays retired. At nodes, B1's
  kill.

### B4. Cover with complementation and the c-MIR flip pass (`DISCOPT_MILP_CUT_LIFT`)

- **HiGHS.**
  - The transformed row is complemented so integer coefficients are positive
    (`HighsCutGeneration.cpp:1128-1136`);
  - then `determineCover` (25-130) → knapsack, mixed-binary or mixed-integer lifted cover
    (131-500), versus c-MIR, keeping the better;
  - in c-MIR, the integral-scale delta (548-560), f0 ∈ [0.005, 0.995] (524-526), and the
    complementation-flip pass (646-695).
- **discopt.** `cover.rs` handles pure binary, nonnegative-weight, `≤`-slack rows only.
  `cmir.rs` has neither the flip pass nor the integral scale.
- **Change.** Put the cover branch and the flip pass inside the shared `cmir_row` core, so
  that B1 (original rows) and B3 (tableau rows) both get them.
- **Entry.**
  - Count rows where a complemented cover is violated at the Rust root LP point on
    bppc8-02 and b-ball. This is offline: read the root x and the rows.
  - Then the root bound on bppc8-02. Rust is at 486.5 = LP for 60 s; HiGHS's
    `cuts_throttle` root is 493.6 and its full root 506.2, both *with* presolve.
- **Kill.** bppc8-02 root < 490 and b-ball root not better than B3 alone.
- **Caveat.** The evidence is indirect. In the no-presolve world HiGHS's bppc8-02 root is
  also 486.5 (`keep_cuts` logged 0 root cuts; unknown why) and certifies in the tree. Last
  in priority.

### B5. Path-aggregation depth and start rule for `cmir.rs` (`DISCOPT_MILP_CMIR_PATH`)

- **HiGHS.** Path length up to 6, any start-row type, both scales, a cut tried at every
  step (`HighsPathSeparator.cpp:170-362`).
- **discopt.** Depth 2, only start rows tight within 10 %.
- **Entry.** E0b with `CMIR_MAX_AGGR` = 6 and `MAX_BASE_SLACK` = ∞ (const rebuild) on
  fiber, sp150x300d, nexp-1-1, nexp-4-2, neos-911970. P4.4 already saw sp150x300d c-MIR
  saturate at 38.57 % for any round count. Target: HiGHS's `all_off` roots (6.2 / 92.9 /
  2.8 / 1.3 / 7.4 %).
- **Kill.** < 2 pp root gain on ≥ 3 of 5.

### B6. Implied-bound and clique cuts (only if B0/B5 leave the `all_off` root gap)

- **HiGHS.**
  - `HighsImplications::separateImpliedBounds` (`H/mip/HighsImplications.cpp:583`);
  - `HighsCliqueTable::separateCliques` (`H/mip/HighsCliqueTable.cpp:1605`);
  - both run every round, ahead of tableau/path/mod-k (`H/mip/HighsSeparation.cpp:94-128`).
- **Dependency.** The implication/clique tables (expert review R3; `presolve/cliques.rs`
  exists, default off).
- **Entry.** The residual from B5 on fiber/nexp/neos-911970/p0548 after c-MIR. If Rust's
  root with c-MIR is within 2 pp of HiGHS `all_off`, do not build.
- **Unknown.** Which non-tableau family closes fiber and nexp in `all_off`. 1.15.1 has no
  per-separator options.

Not in Phase B: `neos-911970`'s Rust failure is an LP-engine stall (7 nodes in 20 s), and no
cut fixes that. `gsvm2rl3` needs about 20 s of HiGHS however it is configured.

## 5. Interaction with the batched best-first parallel tree (binding design constraints)

- **Validity.** Only cuts derived with the global box (`ctx.l_w/u_w`) may enter the shared
  matrix: cover, c-MIR, tableau c-MIR with the global transform. Rows that aggregate
  earlier cut rows are fine, because those rows are global. **No GMI off a node basis.**
- **Determinism.** Cuts are found inside rayon tasks but deduped and selected in node order
  at the boundary, as today (2519-2523). Batch-level selection replaces per-node selection,
  because 64 nodes find near-duplicates (HiGHS checks across worker pools,
  `HighsCutPool.cpp:401-417`).
- **One matrix, one active set.** HiGHS's 100–260 active rows are *per worker LP*. In
  discopt every folded row is in all 64 LPs, so aging must use union-of-use across the
  batch (B2). Without it, each node's row count grows with the whole tree's separation
  history, which is the P4.4 failure mode.
- **Staleness and rounds.** Rows found in batch t act from batch t+1. A separation round
  costs one requeue and one warm re-solve (the hook path's pattern), not HiGHS's
  immediate hot-LP resolve. So cap rounds per node at 2–3 and separate only at
  near-frontier-minimum nodes (B1).
- **Stored bases.** Additions are handled by `extend_basis` (`D/bnb/milp_driver.rs:4700`).
  Deletions need the remap and repair counter in B2. Do not delete rows on a non-compaction
  boundary.
- **Route.** Run 2 must receive the same cut options and, ideally, run 1's pool as initial
  rows. Otherwise every gain is limited to the first slice (b4: istra 48 → 48 over 50 s).
  Plumb this in each step's flag. Until then, measure on the direct driver.
- **Objective lattice.** On the integral-objective rows the bound only has to cross the next
  integer (istra > 48, iskar > 36). Pre-register the bars as "lattice step crossed", not
  "x pp".

## 6. Unknowns (the source and data do not settle them)

- Which HiGHS separator family supplies the in-tree pool growth on neos-361\*. There are no
  per-separator options, and the callback exposes only the root LP cuts. E0a's `notableau`
  arm bounds it from one side.
- Why `keep_cuts` logged 0 root cuts on bppc8-02, neos17 and sp150x300d. The log shows the
  root line, then tree lines with the pool already at 2.4 k.
- The Rust root bound on 19 of the 25 rows (E0b).
- The per-node cost of an LU plus k btrans for node tableau rows in the Rust simplex. No
  factor is exported.
- b-ball's true Rust dual gap (0.024 % vs 20 % across runs).
