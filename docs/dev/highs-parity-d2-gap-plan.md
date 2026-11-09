# Why the Rust MILP engine misses 15 instances HiGHS certifies — diagnosis and plan

Date 2026-11-09. Tree `/home/user/discopt` @ `68f672f1` (branch `claude/happy-hamilton-0vvo3b`).
Read-only: no repository file was edited; nothing was built. Host: 4 cores, **1-min load 9.1**
(another job's `pytest -n 4` was running the whole time), so per CLAUDE.md §9 **no new timing or
20 s solve was run**. Everything below is (a) a checked-in measurement (path given), (b) a code
reading (file:line), or (c) one of two **load-insensitive, solve-free probes run here** (§7: a row
census and an implied-integer rule check over the 15 instances; commands and outputs recorded).
Each diagnosis is tagged **measured** (a number in a data file or a probe) or **inferred** (a
reading of structure + HiGHS source + the ablation, not a Rust measurement).

Baseline for "the gap": `docs/dev/data/highs-parity/d2-analysis/rust_vs_highs_2026-11-09.json`
(58 instances, 3 reps × 20 s): Rust 37/58 all-rep certified, HiGHS 55/58. The 15 rows HiGHS
certifies and Rust never does are the subject.

---

## 0. Three facts about the engine that frame every row (measured)

**F1 — half of every 20 s panel solve is spent in a weaker second tree.** `_solve_milp_simplex`
runs the driver for `max(0.5, min(0.5·remaining, 10 s))` (`solver.py:38829`), then restarts a
**fresh** tree seeded with the incumbent for the remaining ~10 s (`:39040-39260`). Run 2 is
called with `kw = [abs_gap_tol, max_open_nodes, time_limit_s]` only (every profile in
`d2-analysis/*_entry.jsonl.gz`, `calls[1].kw`), so it falls back to the binding defaults
`root_cuts=16, cut_rounds=1, cut_select=false` (`lp_bindings.rs:1113-1115`) and an **empty**
GMI pool. On the bound-limited target rows run 2 re-proves a weaker bound: `enlight8` 20 then
19, `beavma` 352,347 vs 341,758, `nexp-50-20-1-1` 25 vs 24 (`highs-parity-d2-review-highs-expert.md`
§3, 23 two-run rows; `r0_entry.json`: run 2 beat run 1 on 12 of 138 uncertified two-run rows).
HiGHS uses one tree for the whole budget; its only restart keeps cuts as rows, the pool, the
pseudocosts and the incumbent (`HighsMipSolverData.cpp:1317-1335`, `appendCutsToModel`). At the
shipped 3600 s default this costs ≤ 10 s; **on the 20 s panel it halves the effective bound
budget of every row below.** `DISCOPT_MILP_RUN1_CONTINUE` exists (default OFF, `solver.py:38372`);
its entry (`r0_entry.json`, 12 rows): cert-clean, certified 6 = 6, bound better on `enlight8`
(20 → 21, 3/3), `beavma`, `gsvm2rl3`, `nexp-1-1`, `mik-1/2`; worse on `neos17`; **median primal
gap 0.126 → 0.240** because run 2's seeded restart (plus the #280 swap-reseed) was what produced
the improving incumbents.

**F2 — the pure-MILP route's root loop runs GMI + binary knapsack covers only.** The route passes
`root_cuts=500, cut_rounds=50, cut_select=True, root_cut_time_s` (`solver.py:38212-38218`) and
never `cmir_cuts` (grep: the only `cmir_cuts=True` sites are `solvers/milp_simplex.py:134,208`,
the MINLP relaxer's path). No profile of any target row has a `SepCmir*` counter; every root
cut is `SepGomory*` or `SepCover*`. `cover.rs:32-34` refuses any row without a `+1` slack and
any negative binary weight, so VUB/capacity rows (`x − u·y ≤ 0`) never yield covers. **The
engine has no flow-cover / c-MIR-with-variable-bounds family on this route at all.** The W4
port of HiGHS's c-MIR core (`a9bc8d7d`, `d31ad1d2`) was retired 2026-11-08 on route LP numerics
(sub-tolerance primal repair pivots on `dcmulti`/`bell5`), not on cut quality; its root bounds
on `mik-1`: GMI-only −51,757 → −51,493 (SCIP aggregation-only −51,173; HiGHS whole root
−50,587; `d2-plan.md` W4 table).

**F3 — the graduated node-GMI/CUT_LP machinery leaves most folded rows dead in the matrix.**
On every target row 85–95 % of folded cut rows are basic at node optima (`NodeGmiFoldRowsBasic /
Seen`: `enlight8` 95 %, `sp150x300d` 93 %, `beavma` 79 %, `mik-*` 67–80 %), 30–90 % of folded
rows end *lazily relaxed* (slack freed, row left in the LP, cut returned to neither pool nor LP:
`milp_driver.rs:4769-4783`), and deletion is blocked by stored frontier bases (`CutLpDeleteBlocked`
3,936 on `enlight8`, 1,802 on `sp150x300d`). `DISCOPT_MILP_CUT_LP_READMIT` (fix A) and
`_DROP_RELAXED` (fix D) are built, default OFF, and did not recover `enlight8` (§2).

Default states today (read from the parse tables, `milp_driver.rs`): ON — `CUT_LP`, `NODE_GMI`,
`NODE_GMI_LAZY`, `SUBMIP`, `TOL_ROBUST`, `SB_ARMED`, `ALIGNED_SLACKS` (route), `OBJ_INTEGRALITY`,
`OBJ_LATTICE_SUBST`, `NS_PROJECT`, `NS_NODE_BOUND`, `NOISE_RELAX`, `SWAP_RESEED` (route). OFF —
`CUT_LP_READMIT`, `CUT_LP_DROP_RELAXED`, `RAND_ROUND`, `IMPROVE_DIVE` (stride 0), `DIVE_STRIDE`
(0), `ROOT_FULL_BASIS`, `ROOT_GMI_REPAIR`, `RUN1_CONTINUE`, `REENTRY_CUTS`. Strong branching:
`sb_max_cands` on the first `sb_node_budget` nodes, probes through `reoptimize_armed`, **no
per-probe pivot cap** (`strong_branch`, `milp_driver.rs:6190-6320`; HiGHS caps SB by LP
iterations, `HighsSearch.cpp:247,394`).

---

## 1. Per-instance diagnosis (all 15)

Columns: Rust at 20 s (brief's table / `d2-review/attrib.out`, 3 reps); HiGHS from
`d2-ablation/ablation_phase{1,2}.jsonl.gz` (`base`, `all_off` = plain B&B, `presolve_off`,
`keep_cuts` arms; `rootEnd` = bound after root separation). Structure from the census (§7.1).

### 1.1 "Root closes in HiGHS" group

| instance | Rust (20 s) | HiGHS | structure (census) | dominant reason Rust fails | tag |
|---|---|---|---|---|---|
| **beavma** | bound 350,913 / inc 544,942 / opt 383,285; 4–6k nodes; root gap after aligned slacks **17.9 %** (`d2-analysis` follow-up 19) | base: 1 node, 5 restarts, root = opt; `presolve_off`: root 379,804 (**0.9 %**) with 1,773 cuts, 9 nodes; `restart_off`: 327,149 (14.7 %) | 195 VUB rows (`x − u y ≤ 0`), 88 set-packing, 89 flow equalities; obj on 152 binaries + 192 continuous (fixed-charge network / lot sizing) | **Root cut family.** HiGHS closes the root *without presolve* using path/tableau c-MIR with VUB substitution and lifted mixed-binary covers (`HighsPathSeparator.cpp`, `HighsTransformedLp.cpp:37-99`, `HighsCutGeneration.cpp:216-279`). Rust's root loop has no VUB-aware separator (F2); covers refuse these rows. Secondary: primal 42 % off (sub-MIP 2 RINS/solve, `SubMipRinsRun` 2). | root gap: **measured**; cut family attribution: **inferred** (structure + HiGHS arms) |
| **nexp-50-20-1-1** | bound 26 / inc 38 / opt 29; root gap 20 % (follow-up 19); route profile: `RootCutsKept` 141 (all GMI), `DualDegenerateStallBails` 6 | base: 1 node (0.1 s, 754 cuts); `all_off` (cuts throttled, no presolve) root **2.8 %**; `presolve_off` root 3.8 % with 4,553 cuts | 245 VUB + 245 set-packing + 50 flow eq; obj on the 245 binaries only (fixed-charge network design) | **Same root cut family** (flow covers on VUBs). HiGHS's *throttled* non-tableau separators (path, implied-bound, clique) already reach 2.8 %. Then primal: HiGHS `heur_all_off` primal 20.7 % at the root → root rounding/sub-MIP needed once the bound is right. | root gap **measured**; attribution **inferred** |
| **nexp-50-20-4-2** | bound 68 / inc 76 / opt 71; root gap 9.7 % (follow-up 19); `RootCutLoop` 3.7 s of a 9.9 s slice | base: 1 node, 5,196 cuts; `presolve_off` root 70.59 (**0.6 %**), 1 node | 245 multi-choice capacity rows (`x ≤ Σ_k u_k y_k`, 980 binaries), 50 flow eq | **Root cut family**, mixed-integer-cover variant (`separateLiftedMixedIntegerCover`, `HighsCutGeneration.cpp:280-500`) on multi-capacity rows. | **inferred** (measured root gap) |
| **sp150x300d** | bound 62 / inc 89 / opt 69 (6 reps, 0/6); root gap 10 %; `SepGomoryGlobalRefused` 4,136 (sentinel bounds), 93 % folded rows basic | base: 1 node; `all_off` root = LP (**92.9 %**, 27k nodes FAIL); `presolve_off` root 67.18 (**2.6 %**) with 2,597 cuts; `keep_cuts` 114 nodes; `keep_presolve` 59 nodes | 300 VUB + 150 flow eq (fixed-charge transportation 150×300); obj on 300 binaries, integral | **Root cut family is decisive here** (tableau lifted flow covers take HiGHS from 92.9 % to 2.6 % with no presolve). Objective is integral (census), so a 69 bound certifies as soon as a 69 incumbent exists; Rust's sub-MIP reached 81. | LP gap/HiGHS arms **measured**; attribution **inferred** |
| **gesa2** (M) | bound 25,768,009 / inc 26,228,445 / opt 25,779,856: **dual gap 0.046 %, primal 1.7 %** (brief; attrib.out 1.1 %); 2.5–3.5k nodes; `RootCutLoop` 1.1 s, `RootCutsKept` 201 | base: 1 node, 2,869 cuts, incumbent from sub-MIP (`L`); `presolve_off` root 0.058 % gap, 20 nodes; `heur_all_off` root primal 3.5 % | 432 VUB, 720 mixed rows, 168 general ints (ub 4), 816 continuous; obj on 312 ints + 624 continuous (312 of them implied-integer by HiGHS's rule, §7.2) | **Primal.** The bound is within 5× tolerance of the optimum; the incumbent is the gap. Rust's sub-MIP fires 2× per solve (`SubMipRinsRun` 2, ≤ 2 s each, 736 sub-nodes) on the full-dimension problem; HiGHS's sub-MIP is a full presolved MIP (`solveSubMip`, `HighsPrimalHeuristics.cpp:77-247`). The W4 root c-MIR bundle certified `gesa2` 3/3 in both of its panels (`history.out`), so a stronger root also flips it. | **measured** (gaps, sub-MIP counts); sub-MIP strength attribution **inferred** (W3 record) |
| **issue-2446** (M) | **1 node in 20 s**, no incumbent, bound = LP bound 18,930 (maximize; opt −785). Root LP: 111k pivots, 77k phase 1, 2,049 refactorizations, ~120 µs/pivot dense `PriceSweep` (W5 record); `StrongBranch` 11 probes 7.75 s in the `w5a` profile | base: 1 node 0.1 s after presolve removes **1,106 of 1,963 rows and 1,344 of 2,200 cols**, 1 restart; `presolve_off`: root 4,513 (still 675 % gap), 567 nodes, 12.3k SB iterations; `all_off` FAIL; HiGHS pure LP: 973 pivots at 9 µs | 985 equalities, 1,095 pure-continuous rows, 498 VUB, 214 binary knapsacks, 472 binaries, 1,718 continuous | **LP engine first** (the root LP does not finish), **then dimension-reducing presolve** (HiGHS needs it: without presolve even HiGHS needs 567 nodes and 12k SB iterations). Follow-up 20: the Rust route on HiGHS's presolved model goes 1 → 169 nodes, bound 18,930 → −9,284 — better, still open. | **measured** |

### 1.2 `mik-250-20-75-{1,2,3,5}` (MIR-structured)

| instance | Rust (20 s) | HiGHS |
|---|---|---|
| mik-1 | inc = opt; bound −51,236 (**3.1 %**); 16.9k nodes | base 453–1,099 nodes, root −50,518 (1.6 %) after 3 restarts; `presolve_off` root −50,587 (1.75 %) with 5,000 cuts, 718 nodes; `all_off` root −52,571 (5.7 %) → **1.7 % after 26k nodes**, FAIL; `keep_cuts` root −52,241 (5.1 %) → certified in 4,368 nodes with 9,865 conflicts |
| mik-2 | inc = opt; bound −52,414 (3.2 %); 16.3k | base 89–91 nodes; `keep_cuts` 2,072 nodes |
| mik-3 | inc −52,106 (opt −52,242, 0.26 %); bound −53,788 (3.0 %); 16.5k | base 433–455; `keep_cuts` 2,529 |
| mik-5 | inc −51,506 (opt −51,532, 0.05 %); bound −53,117 (3.1 %); 4.2k (brief) / 17k (attrib) | base 71–159; `keep_cuts` 4,590 |

Structure: 75 mixed knapsack rows (big-M-like binary coefficients), 100 pure general-integer rows,
20 continuous; obj on 250 ints + 20 continuous (not lattice-eligible). Rust root (direct driver,
GMI only) −51,757 = **4.1 %**, i.e. *better* than HiGHS's cuts-only root (`keep_cuts` 5.1 %), and
Rust's tree then moves ~1 pp in 17k nodes while HiGHS's `keep_cuts` tree closes 5.1 % in 2–4.6k
nodes and even HiGHS's plain B&B (`all_off`: no cuts, no SB, no presolve, 1 conflict) moves
5.7 % → 1.7 % in 26k nodes. Per-node profile (`w4_d2_cmir_entry.jsonl.gz`, default arm): **`Fbbt`
17.4 s of ~26 thread-s vs `NodeLpSolve` 5.2 s** — FBBT sweeps all 195 dense rows (47 nnz/row) up to
8 rounds at every node; `NodeGmiScan` 5.4 s; root GMI emitted 51k cuts (`SepGomoryCutNnz` 5.6M, ~110
nnz/cut on 47-nnz rows).

**Dominant reason (measured + inferred):** the *tree* is weak per node, not the root. Two
mechanisms HiGHS has and Rust lacks are both shown necessary/sufficient by the ablations:
(i) in-tree aggregated c-MIR / lifted-cover cuts into a shared pool (`node_cuts_off` costs
`mik-1`/`mik-5` 10–14×; `keep_cuts` alone certifies all four; SCIP aggregation-only certifies
`mik-1` in 8.2 s, `w4_scip_mik_ablation`); (ii) whatever carries `all_off`'s tree from 5.7 % to
1.7 % with no cuts — propagation + reduced-cost fixing + pseudocost/plunging — which Rust's
17k-node tree does not reproduce (node GMI pool moved the `mik` bounds 35–70 units of a 1,800-unit
gap, follow-up 13). The per-node `Fbbt` cost (3.3× the LP) is the measured throughput brake.
Measured non-levers (do not repeat): original-row c-MIR at nodes, depth 2 (`NODE_SEP`, retired);
node GMI alone; re-separating root cuts in run 2 (`REENTRY_CUTS` costs the `mik` incumbents);
presolve of the original model (HiGHS's presolve removes only the 120 singleton bound rows Rust's
FBBT already absorbs, follow-up 20).

### 1.3 "Genuine trees"

| instance | Rust (20 s) | HiGHS | dominant reason | tag |
|---|---|---|---|---|
| **bppc8-02** | bound **486.5 = root LP** after 1.8–2k nodes; inc 513 (opt 507); 100 nodes/s. Profile (`w5d`, default): `NodeLpSolve` 27.6 s thread-time for 1,573 solves (17 ms each), `PriceSweep` 394k calls 6.1 s, `Refactorize` 5,747, `DualPrepRejectShape` 888, `DualDegenerateStallBails` 524 (527 cost perturbations accepted, none ends the stall), `StrongBranch` 89 probes 4.3 s; W5 record: 1,959 sub-tolerance primal repairs / 312k primal pivots | base: **1 node** (root 486.5 → 506.17 after 1 restart + 2,138 cuts, then certified by *objective integrality*); `presolve_off`: root stays 486.5, 247 nodes; `all_off`: root 486.5, **1,753 nodes in 0.8 s**, incumbents from node evaluation (`T`); `restart_off` 78×, doubleton-eq rule off 152× | 19 set-partitioning rows, 39 "−z + Σ w x ≤ 0" bin-capacity rows over 229 binaries, obj = z (continuous) + a fixed column | **Three stacked causes.** (a) LP engine per node: warm starts rejected (888) and dual stalls (524) push node LPs into cold/primal repair — Rust is ~20× slower per node than HiGHS `all_off`. (b) **Objective integrality not detected**: `z` is implied integer by HiGHS's dual rule (`HPresolve.cpp:359-416`; verified on this instance, §7.2) and the other objective column is fixed at 0, so HiGHS's objective is integral (`HighsObjectiveFunction::checkIntegrality`, `HighsMipSolverData::computeNewUpperLimit` `:625-652`): cutoff 506, bound ceil. Rust's `obj_integral.rs` refuses any continuous cost column → bound 486.4999999995, cutoff 513. (c) The tree itself: HiGHS plain B&B closes from the same root in 1,753 nodes with `T` incumbents; Rust's 2k nodes move the bound 0 and the incumbent to 513. | (a),(b): **measured**; (c): **inferred** |
| **enlight8** | bound 19–20 / inc 27 = opt; 8–9.5k nodes; `CutLpFolded` 537–1,253, `CutLpRelaxed` 86–89 % of them, `CutLpDeleteBlocked` 3.9–10.6k, 95 % of fold rows basic, mean factor nnz 1.5k → 2.9k on a 64-row model | 412 nodes, 3.0 s | 64 equalities over 64 binaries + 64 unbounded general ints (`Σ x − 2 y = rhs`, mod-2 structure), obj on the binaries | **A graduated regression** (`DISCOPT_MILP_CUT_LP`): `CUT_LP=0` certifies 3/3 in 14.5 s / 46–49k nodes (`r1_entry.json`, `r1_rows.json`); ON holds the bound at 20 with 5× fewer nodes. Mechanism: the shared-LP fold replaced the per-node pool lift that certified it, relaxed rows go back to neither LP nor pool while 193k later node points violate them (`CutLpRelaxedViolated`), and dead rows bloat every basis. Fix A (readmit) 20 → 21 only; fix C killed (`fiber` 3/3 → 0/3); fix D/D' failed (`fiber`/`istra`/`kaihu` lose or `enlight8` 18); expert's fix E (lazy basis repair on physical deletion, `HFactor.cpp:395-425` primitive) not yet built. | **measured** |
| **neos17** | bound 0.1006 / inc 0.363 / opt 0.150 (dual 33 %, primal 142 %); 5.4–5.7k nodes; `NodeLpSolve` 23 s / 2,912 solves (7.9 ms), `DualRefactorizations` 5,649 (≈ 1 per warm solve), `SepGomoryRejectDynamism` 33k | base 428 nodes (root 0.1438 = 4.2 % after 2 restarts); `all_off`: **8,746 nodes, certified from the LP root** (0.00068, 99.5 % gap) with dozens of `S`/`T` incumbents; `presolve_off` 2,099 nodes with 26.8k SB iterations | 300 mixed rows (binary ∧ continuous), 186 pure continuous; obj on 300 binaries + 185 continuous | **Tree quality on both sides**: HiGHS's plain B&B — propagation, reduced-cost fixing, node-evaluation incumbents, pseudocost branching — closes it from the LP bound; Rust's tree of comparable size finds no improving incumbent after the root and moves the bound a third of the way. Plus per-node cost: a refactorization per warm solve. | **measured** (counts); mechanism **inferred** |
| **gsvm2rl3** | bound 0.2286 / inc 0.586 / opt 0.3365; 6–7.8k nodes; `NodeNsCutoff` 2,319, `CutLpLenRejected` 1,084 | base **fails at 20 s under load, certifies at 21.5 s** (5,817 nodes) with 12.5k SB iters, 6.8k conflicts and 75k heuristic LP iters; `all_off`, `keep_cuts`, `keep_presolve` all FAIL at 40 s; only `cuts + SB` certifies | 60 binaries, 60 mixed rows (SVM with ramp loss), 120 pure continuous | **Not a 20 s target.** HiGHS needs every component and all of its budget. Rust's gap is primal (74 %) and dual (32 %) at once. | **measured** |
| **neos-911970** | bound **23.26 = LP relaxation** (root cuts gained nothing: `RootCutsKept` 43, 191 cuts dropped as `SubstDropUnbounded`), inc 103.9 (opt 54.76); 973 nodes; `NodeLpSolve` 23 s / 574 solves (40 ms each; HiGHS ≈ 72 LP iters/node vs Rust ≈ 285), `PriceSweep` 335k calls 8.9 s, `DualPrepRejectShape` 463, `DualDegenerateStallBails` 355, `StrongBranch` 57 probes 6.6 s | base 1,370 nodes 11.6 s, 2 restarts, **9,706 conflicts**, 47k SB iters; `presolve_off` root 52.15 (**4.8 %**) with 3,122 cuts; `cuts_throttle` and `pool_soft_1` (tableau separator off) **FAIL**, `node_cuts_off` FAIL; `all_off` FAIL | 35 set-partitioning + 24 set-covering rows over 840 binaries; 48 rows `−z_i + Σ w x ≤ 65` (24 of the 48 `z_i` implied integer); obj = Σ z_i (54.76 not integral) | **Root and node tableau cuts** (lifted cover/c-MIR on `B⁻¹A` rows mixing set-partition binaries with the continuous `z_i`; HiGHS needs the tableau separator *and* node separation *and* one more component), then **LP per-node cost** (stalls, shape rejections, 4× the iterations per node). Rust's GMI-only root loop drops 191 of its cuts because they touch unbounded slacks. | **measured** (root bound = LP, counters); cut family **inferred** from the HiGHS arms |

---

## 2. Failure classes → the HiGHS mechanism → what discopt has, and why it is not enough

**Class A — fixed-charge / VUB root gap (beavma, nexp×2, sp150x300d; root half of neos-911970,
gesa2, mik).** HiGHS: `HighsSeparation::separationRound` (`HighsSeparation.cpp:43-180`) runs
implied-bound and clique cuts, then the **tableau** (`HighsTableauSeparator.cpp:41-244`: fractional
basic integers, ranked `f(1−f)/‖B⁻¹row‖²`, aggregated row and its negation) and **path**
(`HighsPathSeparator.cpp`: start rows must be *tight* — both slacks > feastol ⇒ `kUnusuable`
`:49-53`; next column by VUB-aware bound distance; `maxPathLen = 6` `:170`; path-mixing cut)
separators; every row goes through `HighsTransformedLp::transform` (`HighsTransformedLp.cpp:132`:
substitute the best VUB/VLB `:30-99`, complement, global bounds) and `generateCut` →
`tryGenerateCut` (`HighsCutGeneration.cpp:1058, 1382-1455`): **lifted cover first** (`determineCover`
`:25`, knapsack `:131`, mixed-binary `:216` — the flow-cover analogue — mixed-integer `:280`), c-MIR
(`:504-740`) only if more efficacious; `postprocessCut` `:760` (integral scaling, tiny coefficients
into the rhs, violation > 10·feastol, coefficient tightening). Measured: with presolve **off** these
alone bring beavma 59 % → 0.9 %, nexp-1-1 58 % → 3.8 %, nexp-4-2 31 % → 0.6 %, sp150x300d 93 % →
2.6 %, neos-911970 58 % → 4.8 %, mik 19 % → 1.75 % (Table 3 / phase-2 logs). *discopt*: `cmir.rs`
(aggregated c-MIR with VUB substitution, depth `CMIR_MAX_AGGR = 2`, start rows within 10 % slack
`MAX_BASE_SLACK`, delta set `{1, 1/|a_k|}×{1,2,4,8}`, no flip, no lifted cover) exists but is **not
called on the pure-MILP route** (F2); `cover.rs` is pure-binary knapsack with a `+1` slack only;
`gomory.rs` is GMI (keeps both signs of continuous terms, hence 110–300 nnz/cut on 3–47 nnz rows —
HiGHS's c-MIR drops positive continuous terms, `HighsCutGeneration.cpp:700-708`). The W4 port
(HiGHS delta rule, flip, lifted mixed-integer cover, small-coefficient removal, power-of-two
scaling) reached −51,493 on `mik-1` and certified `gesa2` 3/3, but was retired on **root-LP
numerics** of the route (thousands of sub-tolerance primal repair pivots on `dcmulti`), with its
last two planned steps (validated scaling + small-coefficient removal; deeper/tighter path
aggregation; tableau rows through the same core) never run. Nothing in it was falsified as a cut
mechanism.

**Class B — in-tree separation into a shared, aged pool (mik×4; enlight8's lost lift; bound
side of bppc8-02, neos-911970).** HiGHS separates at every queue-popped node *into that node's
LP* before branching (`HighsMipSolver.cpp:684-693, 826-837`; `HighsSeparation.cpp:154-166`:
pool re-separation, `addCuts`, `resolveLp`, aging), rows leave the LP after 10 basic epochs and go
**back to the pool at age 1** (`HighsLpRelaxation.cpp:629-690`, `HighsCutPool.cpp:150-160`), queue
nodes store no basis (`HighsNodeQueue.h:142-171`) so deletion is always physical. *discopt*:
`DISCOPT_MILP_CUT_LP` folds pool cuts into the **shared batch** LP by the batch-union score at
boundaries, ages per node solve, and deletes only when the slack is basic in **every stored
frontier basis** — otherwise it *lazily relaxes* the row (`milp_driver.rs:4769-4783`), which is
then in neither LP nor pool and never returns (`READMIT` fixes the return, OFF). The only in-tree
family is GMI (`separate_gomory_cols_global`, globally complemented). W5(d) (physical removal by
exchange pivots) and the recent-basis fallback were measured worse. **Why not enough:** the rows a
node needs are not the rows the batch majority admits (`enlight8` 95 % basic), and the family that
moves `mik` (aggregated c-MIR / lifted cover) is not separated in the tree at all.

**Class C — primal strength (gesa2, sp150x300d, beavma, nexp×2, neos17; bppc8-02's 513).**
HiGHS: sub-MIPs (`solveSubMip`, `HighsPrimalHeuristics.cpp:77-247`: a *full* presolved MIP on the
neighbourhood, 500 leaves, `200 + nodes/20` nodes, cutoff = incumbent) from RENS/RINS/root-reduced-
cost neighbourhoods built by a **propagating fixing dive**, at every plunge start under an
LP-iteration budget (`moreHeuristicsAllowed`, `HighsMipSolverData.cpp:669`); randomized/central
rounding and node-evaluation incumbents (`T`) in the tree. Final incumbent source on 20/25 of the
Rust failures is `L` (sub-MIP); `heur_all_off` still certifies 25/25 from `T`/`S` incumbents
(ablation §C.5). *discopt*: `DISCOPT_MILP_SUBMIP` (RENS/RINS, `try_sub_mip`
`milp_driver.rs:1764-1930`): FBBT-propagated neighbourhood, a recursive call of the **same driver
on the full-dimension problem** (fixed columns kept), `≤ 25 % of remaining ≤ 2 s`, strided (2
runs per 20 s solve on every target profile), incumbents re-validated by `validate_seed_incumbent`.
`IMPROVE_DIVE` (primal gap halved on 13 rows, 0 certifications) and `RAND_ROUND` (2/4 G2 rows)
are OFF; `SUBMIP_SCHED` (HiGHS's schedule) was retired: "HiGHS's sub-MIPs win through the
sub-solver (presolve, cliques, propagation), not the schedule" (`d2-plan.md` W3). **Why not
enough:** the neighbourhood sub-MIP is the same weak engine on the same large LP.

**Class D — the base tree machinery (neos17, bppc8-02, mik's `all_off` closure).** What runs in
HiGHS `all_off`: domain propagation with the clique table and implications
(`HighsDomain.cpp:2098-2101`), conflict analysis (`HighsDomain.cpp:2585`), reduced-cost fixing at
every node (`HighsSearch.cpp:1008`) plus the root reduced-cost ladder re-applied at every
incumbent (`HighsRedcostFixing.cpp:37, 195`), plunging with pseudocosts, node-evaluation
incumbents. *discopt*: `tighten_bounds_csc_with_row` FBBT at every node over **all** rows, ≤ 8
rounds (`Fbbt` = 3.3× node-LP time on `mik`, 37 % on `neos17`); subtree-local reduced-cost fixing
with the #1409-deflated divisor (`reduced_cost_fix`, safe bound `node_lp_bound`, `:5640`); no
clique table or implications in the engine; **no conflict analysis in the MILP driver**
(`bnb/dual_proof.rs` implements SCIP-style Farkas dual-proof rows, propagate-only and globally
valid, for the *spatial* tree only); no root ladder. Node selection / pseudocost scaling / SB
budgets are the A-series binding negatives and are not re-proposed.

**Class E — LP engine per node (issue-2446, bppc8-02, neos-911970; neos17 refactorizations).**
HiGHS node LPs are dual simplex with cost perturbation and bound flipping, primal only as a last
resort (`HighsLpRelaxation.cpp:1232-1250`); hyper-sparse row-wise PRICE; 9 µs/pivot on
issue-2446's LP vs 50–120 µs here. *discopt*: measured and **retired** on their panels —
`NODE_EXPEL_ARTIFICIALS` (neutral), `LP_DUAL_COLD` (throughput collapse), `CUT_LP_EXCHANGE`,
`SUBTOL_INF_SKIP` (worse bounds), `LP_PRIMAL_INCR_PRICING` steps 1+2 (−16 % nodes, retired on
`p0548` alone — a row bimodal in both arms of 9 consecutive panels, `history.out`). The open
measured fact: on `bppc8-02` **524 dual-degeneracy stall bails after 527 accepted cost
perturbations** and 888 warm-start shape rejections per 10 s slice — the perturbation does not end
the stall, and each bail is a cold solve.

**Class F — presolve + restart (issue-2446 decisively; bppc8-02 via doubleton equations;
beavma/sp150x300d only as an alternative to cuts).** `HPresolve` (`presolve/HPresolve.cpp`:
`runProbing` 1601, `dominatedColumns` 1045, `doubletonEq` 3036, `sparsify` 5921,
`detectImpliedIntegers` 7261) and `performRestart` with the ≥ 10 % / ≥ 2.5 %-or-first-pass triggers
(`HighsMipSolverData.cpp:2127-2134, 2494-2497`). *discopt*: the Python orchestrator presolves the
model once (bounds + singleton elimination); the engine's `root_box` is FBBT only; no restart.
Follow-up 20 measured HiGHS's presolved model through the Rust route: only `p0548` flips;
`issue-2446` improves but stays open. The expert's demotion of R5 stands: Rust's counterfactual is
`presolve_off`, under which HiGHS still closes beavma/mik/nexp/sp150x300d with cuts.

**Class G — objective integrality on implied-integer columns (bppc8-02 certainly; a general
mechanism).** HiGHS marks continuous columns implied integer (`HPresolve::isImpliedInteger`
`:359-416`: every row of the column has only integer other columns, integral scaled coefficients
and rhs) and `checkIntegrality` then treats them as integral; `computeNewUpperLimit` (`:625-652`)
sets the cutoff to `floor(scale·ub − 0.5)/scale`. *discopt*: `obj_integral.rs` refuses on any
continuous cost column (its soundness guard). On `bppc8-02` the rule fires (§7.2) and HiGHS's
1-node certificate at bound 506.17 is exactly this cutoff.

---

## 3. Prioritised plan

Every item: default-OFF flag, audit row in `flag-retirement-audit.md`, regression test, entry on
the named rows first (direct driver or route, as stated), graduation on `d2_flag_panel.py`
(MIPLIB-38 + M, 3 reps, 20 s, quiet machine) **plus** `flag_onoff_corpus_panel.py`. Ordered by
expected instances closed per unit effort; "expected" counts only the 15 rows above.

| # | item | hypothesis | evidence | entry experiment (real rows) | kill | expected | effort | soundness | §5 path |
|---|---|---|---|---|---|---|---|---|---|
| **P0** | **Harden the instrument before any graduation** (no solver code): add bar "no instance 3/3 OFF → 0/3 ON" (CLAUDE.md §5 wording — bar 2a still *sums*, `d2_flag_panel.py:19`, which is how `CUT_LP` graduated while losing `enlight8`); score rows bimodal in OFF across ≥ 2 panels (`p0548`) separately or at 6 reps; add a `deterministic=True`+`max_nodes` arm; record the run-1/run-2 split per solve | the instrument has manufactured one false graduation and three kills on a coin-flip row (`history.out`) | §1.3 enlight8; `history.out` p0548 0/1, 0/2, 3/3, 3/2, 2/1, 1/3, 2/0, 0/3, 2/3, 3/2 | re-read the `INCR_PRICING` and `TOL_ROBUST` verdicts under the new bar | n/a | 0 directly; prerequisite | ½ day | n/a | n/a |
| **P1** | **One tree for the whole budget** (R0 re-scoped): `RUN1_CONTINUE` semantics by default, with run 2's job moved *inside* run 1 — at the #291 stall point (incumbent in hand) run the sub-MIP/swap-reseed heuristics in-tree instead of restarting; if a restart is kept at all, it inherits run 1's cut rows, pool and incumbent (HiGHS `performRestart` semantics), never the 16×1 root pass | half the 20 s budget currently re-proves a weaker bound (F1); every bound lever below is measured over 10 s today | `r0_entry.json`: bounds better on enlight8/beavma/gsvm2rl3/nexp-1-1/mik-1/2 with run 1 continued; primal worse because the seeded restart was the incumbent source; `b3_primal.jsonl` run 2 ≈ run 1 node counts | route flag, 12 rows of `r0_entry` + `gesa2`, 3 reps: bound ≥ OFF on ≥ 9/12 **and** median primal gap ≤ OFF (the in-tree primal must replace what run 2 gave) | primal bar fails after the in-tree sub-MIP is scheduled at the stall point → keep the restart but make it inherit cuts + pool (the `REENTRY_CUTS` re-scope) | +0–1 alone (enlight8 at 21 is still short); multiplies every later item | 1–2 days | search order only; bounds from one tree; incumbents re-validated | route flag + §5 panel |
| **P2** | **Finish the CUT_LP repair** (R1): fix E — physical deletion of aged/relaxed rows with *lazy* basis repair when a stored basis holds the slack nonbasic (rank-revealing elimination leaving k columns unpivoted, HiGHS `HFactor.cpp:395-425` primitive), with A (relaxed → pool at age 1) on. Plan B if E costs > 3× pivots: a measured policy — when the relaxed share of folded rows exceeds a threshold *and* node rate has collapsed vs the pre-fold rate, stop folding on that solve and use the node-local lift (the `CUT_LP=0` path) | the fold removed the per-node lift that certified enlight8 and left dead rows in every basis | `r1_entry.json` (CUT_LP=0 3/3, ON 0/3), `r1_rows.json` (10.6k blocked deletions, 95 % basic), fix A/C/D/D' records in `d2-review.md` | `enlight8`, `bppc8-02`, `fiber`, `istra`, `kaihu`, `b-ball`, 3 reps, route 20 s: repaired-solve pivots ≤ 3× ordinary warm solves; `enlight8` 3/3; no W1 winner loses a rep | `enlight8` not 3/3 or any W1 winner loses a rep → retire E, ship plan B or revert to per-node lift for small-`m` models | **+1** (enlight8, known-good endpoint); bppc8-02 bound 486.5 → ~495–504 | 2–3 days | pool cuts are globally valid and NS-verified; deletion only relaxes; the `delete_cut_rows` slack substitution is an exact identity (regression test exists since the `p0548` false certificate) | existing flag + §5 panel with P0's bar |
| **P3** | **Root c-MIR/lifted-cover separator to HiGHS's design, on this route**: recover the W4 core (`a9bc8d7d`, `d31ad1d2`), add HiGHS's path rules (tight start rows, VUB-aware next column, depth 6, both orientations, path-mixing cut), the lifted *mixed-binary* cover (`HighsCutGeneration.cpp:216`), `postprocessCut` (integral scaling, small-coefficient-to-rhs, coefficient tightening) — the two W4 steps that targeted the LP-numerics failure — and feed it the GMI tableau rows Rust already computes (tableau separator) | the fixed-charge root gap is a path/flow-cover gap: HiGHS closes it without presolve, Rust has no such family on the route | §1.1 rows; Table 3 `presolve_off` roots (0.9 / 3.8 / 0.6 / 2.6 / 4.8 / 1.75 %); W4 `mik-1` root −51,757 → −51,493; W4 certified `gesa2` 3/3 twice; retirement cause was route LP numerics, not cuts | **root only** (`max_nodes=2`, direct driver *and* route with aligned slacks), 1 rep: beavma / nexp-1-1 / nexp-4-2 / sp150x300d / neos-911970 / mik-1 root gap vs today's 17.9 / 20 / 9.7 / 10 / 57.5 / 4.1 %. Bar: ≥ 2 pp gain on ≥ 4 of 6 and within 2 pp of HiGHS's `presolve_off` root on ≥ 2; no root bound above the oracle; panel incumbents never cut; no `dcmulti`/`bell5` repair-pivot signature | < 2 pp on ≥ 3 rows → the family is not the lever and the root half of class A is dead; or the repair-pivot signature returns after scaling → engine item first | **+2–4** (gesa2 likely; nexp-4-2 and sp150x300d need the primal half too; beavma/nexp-1-1 root + P5) | 1–2 weeks | every cut is a nonnegative combination of original rows + MIR/lifting over global bounds: `cmir.rs` soundness argument carries over; feasible-point sampling + brute-force lifted-cover tests (W4 had them) are the guards | new flag + root entry + §5 panel + corpus differential |
| **P4** | **In-tree c-MIR/lifted cover into the global pool via the per-node path** (R3): run P3's separator at the best-bound nodes of each batch (the HiGHS "queue-pop" analogue) over original + folded rows with **global** bounds, output to the existing aging pool, enter the LP through P2's per-node selection | `mik` is certified by in-tree aggregated cuts in both HiGHS and SCIP; Rust's only in-tree family is GMI | ablation Table 2 (`keep_cuts` 2–4.6k nodes vs `all_off` FAIL), `node_cuts_off` 10–14×, SCIP aggregation-only 8.2 s; node GMI moved mik 35–70 units | direct driver, `max_nodes=5000`, `deterministic`, ON vs OFF on mik-1/2/3/5, gesa2, bppc8-02: dual gap at the budget falls ≥ 1 pp on ≥ 3 of 4 mik | otherwise → mik's lever is class D, not cuts; retire | **+2–4** (the four mik) | 3–5 days after P2+P3 | as P3; node cuts complemented at global bounds (the `separate_gomory_cols_global` pattern) | new flag; §5 panel; re-panel `IMPROVE_DIVE` with it (owner's standing instruction) |
| **P5** | **Implied-integer objective detection** feeding `obj_integral.rs`: HiGHS's dual rule (`HPresolve.cpp:395-416`) over continuous cost columns, fixed columns dropped first; lattice applies only when *every* cost column is integer or implied-integer and the scale is integral | bppc8-02 is certified by HiGHS at a 0.16 % root gap through objective integrality; Rust holds 486.4999999995 and a 513 cutoff | §7.2 (rule verified on bppc8-02: 1 of 2 cost columns implied, the other fixed at 0); `computeNewUpperLimit` | zero code: run §7.2's script over all 62 panel instances and count objectives that become integral; then the route on `bppc8-02` with the lattice: bound ceil ≥ 487 and `NodeNsCutoff` rising | fires on bppc8-02 only and bppc8-02 does not move (its LP cost dominates) → keep as a cheap general soundness-preserving mechanism anyway, or retire if it changes nothing on the corpus either | **+0–1** (bppc8-02 only with P7) | 1 day | **argument:** given the integer columns' values, an implied-integer column's feasible set is an interval with integral endpoints and the objective is linear in it, so an optimal value lies on the lattice *inside every node* (branching and reduced-cost fixing touch integer columns only; FBBT tightenings on continuous columns are implied). Same refusal discipline as `obj_integral.rs` (`MAX_DENOM`, exact rationality) | new flag + §5 |
| **P6** | **Sub-MIP on a reduced sub-problem**: drop fixed columns and empty/redundant rows, substitute fixings into the rhs, give the sub-driver its own root cuts and the incumbent cutoff; keep `validate_seed_incumbent` on the full rows; schedule by an LP-pivot budget (HiGHS `mip_heuristic_effort`) instead of the batch stride | HiGHS's sub-MIPs win through the sub-solver, not the schedule (W3's own verdict); a sub-problem a tenth the size gets a real tree in 2 s | §1 gesa2 (bound 0.046 % off, incumbent 1.7 %), sp150x300d (bound 62→69 needs a 69 point), `SubMipRinsRun` = 2 per solve everywhere; W3 retired schedule-only | route, 20 s, 2 reps: gesa2, sp150x300d, beavma, nexp-1-1, neos17, bppc8-02: median primal gap ≤ 0.5× OFF and ≥ 3 rows strictly better; `dcmulti` certified (the P4.2/RINS failure mode) as an explicit bar | < 3 better or dcmulti lost → retire the reducer | **+1–2** (gesa2; sp150x300d/nexp-4-2 once P3 lands) | 2–3 days | primal only; every point re-verified on the original rows before injection | new flag (replaces the stride) + §5 |
| **P7** | **Why node LPs stall into cold/primal repair** (R6 re-scoped, diagnostic first): instrument the 524 `DualDegenerateStallBails` after 527 accepted perturbations and 888 `DualPrepRejectShape` on bppc8-02 (463/355 on neos-911970); HiGHS's answers are bound-flipping ratio test + perturbation kept until optimality + no primal fallback | the per-node cost on these rows is the stall-bail → cold-primal cascade, which every retired lever attacked downstream of the stall | profiles §1.3; W5 record; EXPEL/DUAL_COLD/INCR_PRICING/SUBTOL_SKIP retirements | profile-only entry on bppc8-02/neos-911970/issue-2446 (counts per node, where the bail fires, whether the stalled basis is dual-degenerate or cycling); no lever chosen in advance | a named cause whose fix is not bound-neutral per LP → stop and surface | +0–1 (bppc8-02), throughput on every repair-heavy row | 2–4 days diagnostic | bound-neutral per LP by construction (same optimum, audited) | identity test then §5 |
| **P8** | **Conflict / dual-proof rows in the MILP driver**: port `bnb/dual_proof.rs` (Farkas dual proofs, propagate-only, globally valid) to the MILP node loop; later reduced-cost-fixing dual proofs | HiGHS uses 4–10k conflicts on mik/neos17/neos-911970/gsvm2rl3; SCIP certifies istra/iskar with conflicts alone when the pool is off (`node-gmi-expert.md`) | HiGHS logs (`confl` column, §1); SCIP arms | **zero code first:** SCIP ablation on mik-1, neos17, bppc8-02 with `conflict/enable` on/off and the pool off, 30 s — does conflict analysis alone move them? | SCIP shows < 10 % node change on all three → not the lever for these rows | +0–3, uncertain (mik, neos17, bppc8-02) | 1–2 weeks | rows are nonnegative combinations of global rows with directed rounding (the spatial module's argument) | new flag + §5 |
| P9 | **Engine presolve + restart** (issue-2446; R5): only after P7, and only as dimension reduction + HiGHS restart semantics | issue-2446 needs presolve even in HiGHS (`presolve_off` 567 nodes) | follow-up 20 (presolved model 1 → 169 nodes, still open) | the route on the HiGHS-presolved model after P7 | still open → issue-2446 is not a 20 s target | +0–1 | weeks | postsolve must be exact; incumbent verified on original rows | §5 |

Honest sum on these 15 if every entry holds: P2 +1, P3 +2–4, P4 +2–4, P5/P6 +1–2, P8 +0–3 →
**roughly 6–12 of the 15**, i.e. 43–49 of 58 against HiGHS's 55. `gsvm2rl3` and `neos-911970`
sit at HiGHS's own 20 s limit (11–21 s with every component) and `issue-2446` needs an LP engine
plus presolve; those three are not credible 20 s targets.

### What is NOT worth doing (binding negatives honoured, with the record)
- Presolve of the *original* model, FJ, RINS/RENS as built, `SUBMIP_SCHED`, `NODE_SEP`
  (original-row c-MIR depth 2 at nodes), `REENTRY_CUTS` as built (re-separating root cuts in run
  2), `EXPEL_ARTIFICIALS`, `LP_DUAL_COLD`, `CUT_LP_EXCHANGE`, `SUBTOL_INF_SKIP`, the global nnz
  fold cap, `ROOT_GMI_REPAIR` (dense completed-basis cuts), "more GMI" — all measured, recorded.
- Node selection, pseudocost scaling, SB node-prefix budgets, in-tree cut aging alone,
  reliability gating, symmetry, probing alone (A-series and ablation negatives).
- W6 plunging with a hot LP: HiGHS's `all_off` *is* a plunging B&B and still fails the mik,
  sp150x300d, neos-911970 rows without cuts.
- `gsvm2rl3` / `neos-911970` / `issue-2446` as 20 s targets (above).
- Any per-instance predicate: §3.1 of the D2 analysis shows none exists; the only general
  behavioural class is "certified inside the first slice".
- Judging anything on `p0548` before P0.

---

## 4. Correctness risks (reported separately; none is a confirmed false certificate today)

Read for this review: `delete_cut_rows` / `substitute_deleted_slacks` / `remap_basis_after_delete`
(`milp_driver.rs:843-1130`), the lazy relaxation (`:4769-4783`), `node_local_gmi_bound` (`:1310`),
`separate_gomory_cols_global` and the β guard (`gomory.rs:55-110, 495-720`), `reduced_cost_fix` and
its call site (`:6429`, `:5640`), `try_sub_mip` / `validate_seed_incumbent` (`:1764-1930`,
`:7325`), `decide_status` (`:2329`), the route's run-2 adoption (`solver.py:39040-39260`),
`_milp_aligned_lp` (`solver.py:38472-38520`), `obj_integral.rs` header.

1. **Measurement-instrument risk (highest, actionable now).** `d2_flag_panel.py` bar 2a sums
   certifications (`:19`): a flag can graduate while turning a 3/3 instance into 0/3 — this is how
   `CUT_LP` shipped with the `enlight8` loss. CLAUDE.md §5's cert-clean bar is per instance. Until
   P0 lands, every graduation on this panel is at risk of the same error, and the `p0548`
   coin-flip row has decided three verdicts. The per-instance no-regression bar is **not** in the
   script today (grep).
2. **History of false certificates in exactly this region, now fixed and pinned — treat the
   region as high-risk for any further change:** CUT_LP's per-node deletion dropped a slack column
   other folded rows used → `p0548` certified 9,154 > optimum 8,691 (fixed by substitution,
   regression test); GMI's integer snap on continuous columns cut the `fac2` optimum (#1236);
   the reduced-cost divisor (#1409); FBBT on the `1e20` sentinel. Any P2/P3/P4 change to row
   deletion, slack rewriting or cut complementation must keep those tests and add the
   feasible-point sampling test over the panel's incumbents.
3. **Lazily relaxed rows (default ON).** A relaxed folded row keeps its equality `a·x − s = rhs`
   with `s` freed to `(−INF, INF)`. Sound (a pure relaxation; node bounds stay valid), and the
   global GMI separator refuses cuts whose complementing bound is the sentinel
   (`SepGomoryGlobalRefused`), so a free slack cannot enter a cut. The risk is indirect: a free
   nonbasic column in the dual simplex relies on the "two-sided free encoding" the comment
   asserts; there is no regression test that a relaxed row's slack is never nonbasic at a nonzero
   value in a node optimum. Recommend adding one (profile counter + test) before P2 touches it.
4. **Incumbent acceptance is tolerance-based while emptiness proofs are exact-then-relaxed**
   (`TOL_ROBUST`, graduated as a correctness fix 2026-11-08): the known residual, recorded in the
   D3 handoff — cut-row validity "over the tolerance set", reduced-cost fixing's gap and the NS
   bound are still taken over the exact rows. A model feasible only within 1e-6 can be answered
   differently by trajectory. Not a false *optimal*, but a certificate-consistency gap; P3/P4 add
   more cut rows whose validity is only exact-row validity.
5. **`GLOBAL_BETA_MAG_MAX = 1e6`** (`gomory.rs:71`) is the only guard between a globally
   complemented GMI β's rounding error and `FRAC_MIN`; measured 0 refusals on MIPLIB-38+M. The
   argument is sound (a refusal only drops a cut) but the guard was sized by one probe; wide
   general-integer boxes (`enlight8` has unbounded general integers → cuts refused, 11,212 of 11,401
   emitted) are exactly where it matters. Keep the probe in the regression set.
6. **P5's soundness argument** (implied-integer objective lattice) holds only while branching and
   reduced-cost fixing never restrict the implied-integer continuous columns and while every cost
   column is covered; `reduced_cost_fix` skips `!is_int[j]` (`:6497`) and branching is on integer
   columns, so it holds today — it must be asserted in the detector (refuse if any mechanism can
   tighten such a column's bounds other than by implication).
7. **No new soundness defect was found in the paths read.** The route's final bound is
   `max(bound₁, bound₂)` of two valid bounds on the same problem, the run-2 point passes
   `_point_feasible` on the original rows, `reduced_cost_fix` uses the safe node bound (not the
   vertex value), `try_sub_mip` re-validates every point on the original rows, and `decide_status`
   returns `Infeasible` only on a finished, resolved tree.

---

## 5. What was measured here vs. taken from the record

- **New (solve-free, load-insensitive):** §7.1 row census of the 15 instances; §7.2 HiGHS
  implied-integer rule over their continuous cost columns. Executed counts printed (15 / 1,448).
- **Taken from checked-in data:** every Rust number (brief's JSON, `attrib.out`, `history.out`,
  `r0/r1_*.json`, `*_entry.jsonl.gz` profiles — the file is named at each use) and every HiGHS
  number (`ablation_phase{1,2}.jsonl.gz`, `d2_lp_bounds.json`).
- **Not run, and why:** any 20 s or node-rate probe — 1-min load 9.1 on 4 cores from another
  job's test suite (CLAUDE.md §9); the record already holds the quantities the plan rests on.

## 6. Open items for the owner
- P0 before any further panel; re-read `INCR_PRICING`'s kill under it (nodes −16 %, 0 dual
  losses, lost on `p0548` only).
- Whether the W4 core may be recovered from git (`a9bc8d7d`, `d31ad1d2`) for P3 — it was
  retired on route numerics with two planned steps unrun.
- Whether the 20 s panel should keep the 10 s slice (P1): at the shipped 3600 s default the slice
  costs ≤ 10 s; on the panel it is half of every solve.

---

## 7. Probes run here (commands and outputs)

### 7.1 Row census (`struct15.py`, no solve)
`nice -n 10 python -I /tmp/claude-0/-home-user-discopt/622858fd-8692-539a-a137-f41f6cd43f5f/scratchpad/struct15.py`
(reads `ref/miplib2017/*.mps.gz` and `ref/HiGHS/check/instances/{gesa2,issue-2446}.mps` through
`lp_milp_loader.read_instance`; asserts `discopt.__file__` under the repo). Output (EXECUTED 15):

```
beavma         m=372  n=390  nnz=975  bin=195 cont=195 | eq 89 setpack 88 vub 195 pure_cont 89
nexp-50-20-1-1 m=540  n=490  nnz=1225 bin=245 cont=245 | eq 18 setpack 245 vub 245 pure_cont 50
nexp-50-20-4-2 m=540  n=1225 nnz=2695 bin=980 cont=245 | eq 17 setpack 245 mixed 245 pure_cont 50
sp150x300d     m=450  n=600  nnz=1200 bin=300 cont=300 | eq 150 vub 300 pure_cont 150
mik-250-20-75-* m=195 n=270  nnz=9270 bin=75 genint=175(ub 100) cont=20 | mixed 75 (bigM-like 75) pure_int 100 pure_cont 20; max row nnz 122
bppc8-02       m=59   n=232  nnz=4387 bin=229 genint=1 cont=2 | eq 20 setpart 19 mixed 39; obj on 2 continuous cols
enlight8       m=64   n=128  nnz=352  bin=64 genint=64(ub inf) | eq 64 pure_int 64
neos17         m=486  n=535  nnz=4931 bin=300 cont=235 | eq 1 mixed 300 pure_cont 186
gsvm2rl3       m=180  n=241  nnz=4020 bin=60 cont=181 | mixed 60 pure_cont 120
neos-911970    m=107  n=888  nnz=3408 bin=840 cont=48 | eq 35 setpart 35 setcover 24 mixed 48 (bigM-like 9); obj on 48 continuous
gesa2          m=1392 n=1224 nnz=5064 bin=240 genint=168(ub 4) cont=816 | eq 48 setpack 48 vub 432 mixed 720 pure_int 144 pure_cont 48
issue-2446     m=1963 n=2200 nnz=5206 bin=472 genint=10 cont=1718 | eq 985 setpack 63 knap 214 vub 498 mixed 90 (bigM 80) pure_cont 1095
```

### 7.2 HiGHS implied-integer rule on the continuous cost columns (no solve)
Rule as in `HPresolve::isImpliedInteger` dual detection (`HPresolve.cpp:395-416`): every row of
the column has only integer other columns, coefficients and finite rhs integral after scaling by
`1/a_ij`, integral column bounds. Output (CHECKED 1448):

```
bppc8-02      cont obj cols 2  implied 1  (the other is fixed [0,0] with no rows -> integral after fixed-column removal)  => objective integral in HiGHS's sense
neos-911970   cont obj cols 48 implied 24                                                                                 => not integral (opt 54.76)
gesa2         cont obj cols 624 implied 312; coefficients fractional                                                      => not integral
neos17, gsvm2rl3, beavma, issue-2446: 0 implied; mik: 0 of 20
sp150x300d, nexp-1-1, nexp-4-2: no continuous cost columns, integral coefficients                                         => integral (Rust's detector already covers these)
```

### 7.3 Data extractions (read-only, scripts inline in the session)
- HiGHS per-arm logs for the 15 rows from `d2-ablation/ablation_phase{1,2}.jsonl.gz` (root LP /
  root-end bound, cuts, pool, conflicts, restarts, presolve reductions, incumbent sources, LP-iteration
  split) — the numbers in §1.
- Latest default-arm Rust profiles per target from `d2-analysis/*_entry.jsonl.gz` (phases and
  counters quoted in §1; file named per row: `w5d_cut_lp_exchange_entry`, `tol_robust_v2_entry`,
  `w4_d2_cmir_entry`, `w4_cmir_bundle_entry`, `w3_entry`, `w5a_dual_cold_entry`).

## 8. Test-level certification gaps exposed by the default flip (2026-11-09)

When the pure-MILP default became the Rust engine on every install (owner decision,
PR #2196), these tests stopped certifying on the default route. In every case the Rust
engine returns an honest uncertified `feasible` with a valid bound and a correct point —
none is a wrong answer. They are concrete, small targets alongside the 15 MIPLIB
instances above; each test now pins HiGHS as its certifying oracle (or, for the two
soundness tests, checks the Rust arm for a correct uncertified value), so the gap stays
visible rather than hidden:

| Test | Model class | Rust today |
|---|---|---|
| `test_1494_piecewise_large_offset` (7 cases) | piecewise lowerings at breakpoint offsets 1e4–1e7 | `feasible`, correct value (both-engine test) |
| `test_1654_bigm_coefficient_tightening::test_sequencing_certifies_the_true_makespan` (M = 1e7, 1e9) | big-M sequencing | `feasible` within 120 s |
| `test_1655_lost_certificates::test_pure_binary_scheduling_milp_certifies_fast` | pure-binary STN scheduling | not certified |
| `test_2134_relu_partition_ideal::test_milp_optimum_identical_across_strategies[min-0-sizes1]` | trained-ReLU big-M | `relu_bigm` arm `feasible` |
| `test_2156_relu_negligible_partitions::test_tiny_weight_net_certifies_bigm_optimum` (3) | tiny-weight ReLU nets | `feasible` |
| `test_issue_1617::test_tiny_coefficient_is_not_proved_infeasible_by_bound_tightening` | `1e-13·y >= 5`, `y <= 1e14` integer | `feasible`, correct value (soundness test) |

A fix that closes a class above should flip the corresponding test back to the default
route.

## 9. P0 results (2026-11-09): the instrument, corrected and re-applied

`d2_flag_panel.py` now has bar 1b (no instance certified in every OFF rep and in no ON
rep), sets bimodal rows aside (OFF certified in some but not all reps, or listed in
`KNOWN_BIMODAL` with its cross-panel record: `p0548`), takes `--deterministic`
(`deterministic=True`, bounded by `--max-nodes`) and `--rescore FILE`. The route now records
the run-1 / run-2 split per solve (`milp/run1_*`, `milp/run2_*` in `solver_stats`), which P1
needs. Regression tests: `discopt_benchmarks/tests/test_d2_flag_panel_bars.py`.

Re-scoring the recorded panels with the corrected bars:

| Panel | Stored verdict | Corrected | Deciding row |
|---|---|---|---|
| `CUT_LP` (W1) | graduated | **fails bar 1b** | `enlight8` 3/3 OFF -> 0/3 ON |
| `CUT_LP` + `SB_ARMED` bundle (default ON) | graduated | **fails bar 1b** | `enlight8` |
| `LP_PRIMAL_INCR_PRICING` (steps 1+2, retired) | failed | **passes every bar** | its only failure was `p0548` (2a, 2d); stable rows: certified 123 = 123, nodes 47,723 -> 40,050 (-16 %), median primal gap 0.160 -> 0.160 |
| `LP_PRIMAL_INCR_PRICING` + row-wise pricing (retired) | failed | fails 2e | median primal gap 0.160 -> 0.168 |
| `TOL_ROBUST` v2 | failed | fails 2b | (independent of `p0548`) |
| `MILP_RUST_COEF_ENTRY` (graduated today) | graduated | graduates | no instance lost; stable rows 123 -> 126 |

Two owner decisions follow from this, neither taken here: `CUT_LP` is default ON on a panel
that fails the corrected bar (the audit already retracted its cert-clean claim and kept it ON
pending the P2 fix); and the incremental-pricing engine (steps 1+2) was retired on a coin
flip, so it is a candidate to restore from history and re-panel.

## 10. P1 results (2026-11-09): one tree, with run 2's primal job moved inside it

**Entry 1 — `RUN1_CONTINUE` as it stood** (13 rows, 3 reps, 20 s;
`docs/dev/data/highs-parity/d2-analysis/p1_entry_run1_continue.json.gz`). Certified 6 = 6,
bar 1b clean. Bound better on 7 of 11 open rows (`beavma`, `enlight8` 20 -> 21, `gesa2`,
`gsvm2rl3`, `mik-1/2`, `nexp-4-2`), worse on 2 (`neos17` 0.101 -> 0.065, `nexp-1-1` 26 -> 25).
Primal half **failed**: median primal gap 0.069 -> 0.286 (`mik-1` -49,716 -> -5,529, `mik-2`
-50,768 -> -29,921, `nexp-4-2` 76 -> 92). The OFF arm's run split (`milp/run2_better_primal`)
named run 2 as the source of the better incumbent on `beavma`, `fiber`, `gesa2`, `mik-1/2`
and `nexp-4-2`.

**Where run 2's incumbents come from (measured, not assumed).** A per-batch incumbent
timeline (`node_callback`) on `mik-1`, `mik-2`, `nexp-4-2`, `beavma`, `gesa2`: in every
case the improvement arrives in run 2's **first batch** (node count 3, i.e. its root),
before any tree search. It is not the #280 swap reseed (`DISCOPT_MILP_SWAP_RESEED=0`:
same -49,716 at the same point) and not RINS (`SubMipImproved` = 0 over the whole OFF
solve). Run 2's root LP differs from run 1's: it gets the binding's single 16-cut pass,
not the 500-cut / 50-round root budget, so its root heuristics start from a different
vertex. The existing improving dive (`DISCOPT_MILP_IMPROVE_DIVE` 4 / 16) under the flag
recovers part of it (`mik-1` -40,523 / -47,410; `nexp-4-2` 77 / 82), not all.

**The change.** Under `stall_time_s` (only set by `DISCOPT_MILP_RUN1_CONTINUE`), at the
first batch boundary past the #291 slice with an incumbent in hand, the driver runs
`try_stall_redive` **once**: a root-batch-only solve (`max_nodes = 3`) of the base problem
with run 2's root cut options, seeded with the incumbent, capped at min(2 s, 25 % of the
time left). Primal only: its bound is discarded, its point must be strictly better and is
re-validated on the base rows (`StallRediveRejected` must stay 0). Counters
`StallRediveRun` / `StallRediveImproved` / `StallRediveRejected`; Rust regression test
`stall_redive_runs_once_and_keeps_the_optimum`.

**Entry 2 — with the re-solve** (same 13 rows, 3 reps, 20 s;
`.../p1_entry_redive.json.gz`). Certified 6 = 6; bar 1b clean; no incumbent lost.
Median primal gap 0.069 = 0.069 (`mik-1/2`, `nexp-4-2`, `beavma` back to OFF's values,
`gesa2` better: 25,902,720 -> 25,788,343). Bound ≥ OFF on 9 of 11 open rows (better on
`beavma`, `enlight8`, `gesa2`, `gsvm2rl3`, `mik-1/2`, `nexp-4-2`), worse on `neos17` and
`nexp-1-1` as before (run 2's fresh root lifted those two). **The pre-registered entry bar
passes** (bound ≥ OFF on ≥ 9 of the open rows and median primal not higher). The panel's
own bars 2b (nodes) and 2c (dual losses 2 > cap 1) fail on 13 rows, as expected of an
entry sample; they are for the §5 panel.

Next: the §5 panel (`d2_flag_panel.py --flag DISCOPT_MILP_RUN1_CONTINUE --fired
milp/run1_continue`, MIPLIB-38 + M, 3 reps) plus `flag_onoff_corpus_panel.py`. The two
dual losses are the open question: both are rows where a second root pass tightened the
bound, which the one-tree design gives up.

## 11. P2 results (2026-11-09): fix E fails its entry; `DROP_RELAXED` retired with it

**Fix E as built** (`DISCOPT_MILP_CUT_LP_REPAIR`, not committed): an aged-out relaxed folded
row is deleted as fix D does, and every stored basis that held its slack nonbasic is
*kept* with `k` excess basic columns, then repaired when its node is popped. The repair is
exact: the basic block `B'` (`m x (m+k)`) keeps full row rank through the slack
substitution and row deletion, `null(B')` is read off `[B'; R] V = [0; I]` with `k`
pseudo-random rows, and the `k` columns to drop come from threshold partial pivoting over
`V` (dropping `D` leaves a nonsingular block iff `V_D` is nonsingular), preferring boxed
columns and never a free one. Unit-tested; on `enlight8` 5,419 of 6,447 popped bases were
repaired, and relaxed-cut violations at node points fell 355k -> 13k.

**Why it does not pay.** Roughly half the repaired bases are dual-infeasible
(`DualPrepRejectDualInf` 1,079 on one `enlight8` solve): a row's slack is nonbasic in a
stored basis because the row was *binding* at that node when the basis was stored, so
deleting it genuinely changes that node's LP optimum, and no choice of dropped columns
restores dual feasibility in general (the bound flip covers boxed columns only). Those
nodes go to the primal warm or cold path, at ~3.4 ms per node LP against ~2 ms for the
default `CUT_LP` and ~0.18 ms for `CUT_LP=0`.

**Entry** (`enlight8`, `bppc8-02`, `fiber`, `kaihu`, `b-ball`; `istra` is not in the
panel population; `CUT_LP_READMIT=1` in both arms; 3 reps, 20 s;
`docs/dev/data/highs-parity/d2-analysis/p2_entry_fixE.json.gz`):

| row | A only | A + E |
|---|---|---|
| `enlight8` | 0/3, bound 21 | 0/3, bound 21 |
| `fiber` | 3/3 | **0/3** (bound ~399.0k) |
| `neos-3611689-kaihu` | 3/3 | **0/3** (bound 118) |
| `b-ball` | 3/3, 1.6 s | 3/3, 6.3 s |
| `bppc8-02` | bound 491-493 | bound 499.8 |

The pre-registered kill ("any W1 winner loses a rep") fires. E is retired, and with it
`DISCOPT_MILP_CUT_LP_DROP_RELAXED`, which the audit kept only as E's base (flag,
deletion arm and depth-based root test deleted; the default path is bit-identical, since
`is_root` reduced to `no_basis` there).

**What this says about the class.** Physical deletion cannot be made cheap on this
driver: every frontier node carries a basis in which the rows it needs are binding, so
removing those rows always costs that node a real re-solve. HiGHS escapes this only
because its queued nodes hold no basis. The remaining route is plan B (§3 P2): keep
folding where it pays and fall back to the node-local lift (the `CUT_LP=0` path) when
the folded LP stops paying, decided by a measured per-solve signal, not by instance.
