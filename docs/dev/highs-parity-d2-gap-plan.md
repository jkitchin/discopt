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

### 11.1 Plan B: cap the fold per solve, unfold on the trip (`DISCOPT_MILP_CUT_LP_CAP`)

**Signal (measured, one 20 s solve each; live folded rows `CutLpFolded - CutLpDeleted`
over the model's rows m):** the fold helps where it stays small — `fiber` 0.25, `kaihu`
0.23, `gesa2` 0.40 — and hurts where it grows past the model — `bppc8-02` 4.7, `enlight8`
21; `b-ball` 1.8 certifies either way. Rule: past 2·m live rows, the solve stops folding,
unfolds every row the deletion pass can rewrite (cuts back to the pool; a stored basis
holding a deleted slack nonbasic is dropped once, and the root is decided by depth so a
basis-less node does not run the root-only dive), and returns to the node-local lift.

**A cap without the unfold does nothing** (`enlight8` bound 22 at 20k nodes, 273 rows
already folded when it trips and still in every LP).

**Entry** (3 reps, 20 s, 2 jobs; `p2_entry_cap.json.gz`, `p2_entry_cap_e8.json.gz`):
`fiber`, `kaihu`, `b-ball`, `gesa2` never trip and are unchanged. `enlight8`: OFF
bound 19-21 at ~9k nodes; ON bound 23, 23, 26 at 36k, 36k, 125k nodes, 0/3 certified.
Quiet machine, one solve: ON certifies 27 at 125k nodes; `CUT_LP=0` certifies at 36k.
The split is structural: when run 1 stops at the frontier cap at ~13 s, run 2 has 7 s
and must re-fold, trip and unfold from its own root. `bppc8-02` does not move (120 of
308 rows unfoldable; the rest are used by non-folded rows).

**Factor 1.0 falsified** (`p2_entry_cap1.json.gz`): `b-ball` 3/3 -> 0/3 (a trip
discards a fold that was working), `enlight8` 22. The constant stays 2.0.

The pre-registered "`enlight8` 3/3" bar is not met at 20 s under load; the §5 panel over
MIPLIB-38 + M decides plan B (bars 1, 1b, 2a-2e). Open: the unfold cannot rewrite rows
referenced by non-folded rows (`bppc8-02`), and the two-run route doubles the fold-trip
cost (`RUN1_CONTINUE` would remove it; the combination is not yet measured on a panel).

**§5 panel and retirement (2026-11-09).** `d2_flag_panel.py --flag DISCOPT_MILP_CUT_LP_CAP`
(MIPLIB-38 + M, 3 reps, 20 s, 372 solves; `p2_cut_lp_cap_panel.json.gz`): cert-clean,
certified 126 = 126, no incumbent lost, median primal gap unchanged. The cap trips on 2
of 62 instances: `enlight8` (dual gap 0.25 -> 0.11, still uncertified) and
`enlight_hard` (bound 27 -> 26, the one dual loss). Neutral, so bar 2b fails, and the
flag and its implementation are retired (last version `d85480bc`).

**Where P2 stands.** Both repairs of the fold were measured and neither restores
`enlight8`: deleting rows costs the nodes whose bases hold them (E), and capping the fold
only reaches the two instances whose LP grows past the model, one better and one worse
(plan B). `CUT_LP` stays default ON (owner decision 2026-11-09) with the `enlight8`
regression recorded against it. The plan's remaining P2 option, the per-node lift for
"small-m models", is a size-keyed special case of the plan B rule this panel just
measured as neutral, so it is not pursued. Next is P3 (root c-MIR / lifted cover).

### 8.1 Fast-lane time budget after the default flips (2026-11-09, first full CI matrix)

The full test matrix first ran when PR #2196 left draft; 10 tests broke the 15 s fast-lane
budget (none failed an assertion). Measured locally, branch head vs the PR's base
`976b583f` (a worktree build, loaded module verified), and on the branch with HiGHS named:

| test | `main` | branch (Rust default) | branch, HiGHS named | cause | remedy |
|---|---|---|---|---|---|
| `test_1654::test_refused_incumbent_is_repaired_not_an_error` | — | 120.5 s (its whole `time_limit`) | 6.8 s | MILP default | HiGHS-route fixture: the module tests `lp_milp_highs` mechanisms |
| `test_2156::...[relu_extended-kw3]` | — | 62.2 s | 4.2 s | MILP default | names HiGHS: the subject is the formulation |
| `test_2156::...[relu_partition-kw2]` | 2.9 s | 13.4 s | — | MILP default | as above |
| `test_1595::...psig40r...[2]` | 2.2 s | 25.5 s | 24.6 s (MILP knob only) | OA master default | allowlisted (default path) |
| `test_1595::psig40r_still_certifies_through_the_default_path` | 1.8 s | 21.9 s | 19.9 s (MILP knob only) | OA master default | allowlisted (default path) |
| `test_1624::routed_kernel_eligible_model_goes_to_the_route` | 5.2 s | 27.7 s | 28.7 s (MILP knob only) | route defaults | allowlisted |
| `test_issue1729::...[shape1]` | 1.8 s | 12.4 s | — | route defaults | allowlisted |
| `test_1917::...cstr...[True]` | 6.8 s | 12.6 s | — | route defaults | allowlisted |
| `test_1940::...[ex1264-False]` | 2.1 s | 5.0 s | — | route defaults | allowlisted |
| `test_milp_simplex_dive::...matches_highs` | 1.2 s | 5.4 s | — | route defaults | allowlisted |

Root cause of the `psig40r` pair, traced: on `main` the model goes LP/NLP-BB with
`master=highs` and certifies in 8 nodes (1.4 s); on the branch the master is the Rust
lazy-cut driver (D3 default), whose first call ends at `node_limit` after 1,321 nodes
(16.3 s) and a second call certifies at 491 nodes. None of the five driver flags this
branch graduated is responsible (`NODE_GMI`, `CUT_LP`, `SUBMIP`, `SB_ARMED`,
`ALIGNED_SLACKS` each `=0`: 19.4-21.2 s). It is the Rust engine's tree-size gap, the
subject of this plan. Default-path tests stay on the default (allowlisted at their CI
times, so the default keeps its coverage); the allowlist entries leave when the engine
closes the gap.

## 12. P3 entry, step 1 (2026-11-09): the family is the lever; our separator is not

**Rust root today** (`discopt_benchmarks/scripts/p3_root_gap.py`, route, `max_nodes=2`;
`docs/dev/data/highs-parity/gap-plan-p3/p3_root_base.json`): beavma 18.4 %, nexp-1-1
27.6 %, nexp-4-2 9.9 %, sp150x300d 15.9 %, neos-911970 57.5 %, mik-1 4.1 %.

**The W4 core, restored for the entry only** (not committed; `cmir.rs` from
`e5eac131^`, routed by a root-loop `cmir_cuts` switch): arm a (c-MIR alone) beavma
14.6 %, sp150x300d 18.8 % (worse), the rest within 1.5 pp; arm b (+ HiGHS delta
selection) mik-1 3.8 %, the rest unchanged or worse; arm c (+ flip, lifted MIC)
sp150x300d 10.1 %, nexp-4-2 12.7 % (worse). Kill-level on its own (< 2 pp on 4 of 6).

**SCIP root ablation, presolve off, node limit 1**
(`gap-plan-p3/p3_scip_root_ablation.{py,jsonl}`): root gap %

| row | none | all | aggregation only | gomory | implied bounds | mcf | zerohalf | all but aggregation |
|---|---|---|---|---|---|---|---|---|
| beavma | 58.66 | 0.02 | **1.07** | 28.53 | 58.66 | 11.11 | 58.66 | 2.74 |
| nexp-50-20-1-1 | 55.49 | 0.00 | **0.00** | 16.48 | 55.49 | 0.00 | 55.49 | 0.00 |
| nexp-50-20-4-2 | 30.86 | 0.91 | **0.78** | 13.02 | 30.86 | 4.84 | 30.86 | 2.11 |
| sp150x300d | 51.89 | 3.18 | **4.63** | 21.69 | 51.89 | 41.91 | 51.89 | 18.81 |
| neos-911970 | 57.52 | 2.75 | **5.96** | 57.52 | 57.52 | 57.52 | 57.52 | 57.52 |
| mik-250-20-75-1 | 18.62 | 2.54 | **2.73** | 4.54 | 18.62 | 18.62 | 18.62 | 5.33 |

(In SCIP 6 `aggregation` emits nothing unless the `cmir` / `flowcover` / `knapsackcover`
dummy separators are on; a first run that missed this measured `agg` = `none` and is
superseded.) SCIP's path-aggregated c-MIR / flow cover alone closes every row to within
6 %, and is the only family that moves `neos-911970` at all. So P3's hypothesis
survives the zero-code test: the root gap on these rows is an aggregation-separator gap.

**Where ours falls short (one row, `nexp-50-20-1-1`, route + `cmir_cuts`).** The
separator runs: 1,058 c-MIR cuts over 18 root rounds, 1,000 generated, 282 kept, and
the root bound stays at 21 against SCIP's 29 from the same family. The defect is cut
strength, not missing plumbing. Next: compare one round's cuts against SCIP's on a
single flow-conservation row of this instance (aggregation path, bound substitution
choice, delta / complementation), then fix the construction, then re-run this entry.

### 12.1 P3 entry, step 2: isolating the separator (`gap-plan-p3/p3_root_arms.py`)

The route's root with the cut families switched per arm (driver kwargs patched, root
only; W4 core restored for the entry, not committed; `p3_root_arms_w4.jsonl`):

| row | no cuts | GMI (default) | c-MIR only | GMI + c-MIR | SCIP none -> aggregation |
|---|---|---|---|---|---|
| nexp-50-20-1-1 | 14 | 21 | 23 | 21 | 12.9 -> 29 |
| beavma | 172,135 | 312,687 | 309,683 | 327,186 | 158,460 -> 379,184 |
| mik-250-20-75-1 | -57,326 | -51,757 | -52,544 | -51,762 | -58,972 -> -51,076 |

Two defects, separately: our c-MIR alone is far weaker than SCIP's aggregation alone,
and GMI + c-MIR can be worse than c-MIR alone (`nexp-1-1` 21 vs 23: the shared budget
and selection crowd c-MIR cuts out). With W4's HiGHS delta selection + flip + lifted MIC
on the c-MIR-only arm: `beavma` 10.2 %, `mik-1` 3.6 %; `nexp-1-1` stays 23 and
`neos-911970` 57.5 %.

**Ruled out:**
- the base construction: the textbook flow cover (`x1 + x2 >= 7`, `x_i <= 5 y_i`, LP
  `y = 0.7`) comes out exactly, `y1 + y2 >= 2`.
- aggregation depth: `CMIR_MAX_AGGR` 2 -> 6 leaves `nexp-1-1` at 22, `neos-911970`
  unchanged, `beavma` 20.8 % and `sp150x300d` 29.0 % (worse).
- walking into variable-bound rows: skipping them as aggregation candidates changes
  nothing measurable.

`neos-911970` has no variable bounds at all (48 free continuous columns carrying the
whole objective, 48 mixed rows, 59 integer-only rows), and SCIP's aggregation still
takes it from 57.5 % to 6 %. So the missing piece is not variable-bound handling either.

**Next (not yet run):** read SCIP's applied aggregation cuts at the root of
`nexp-50-20-1-1` and `neos-911970` (pyscipopt, presolve off) and reconstruct which rows
each one aggregates and which bound substitutions and delta it used. That turns "our
separator is weaker" into a named difference, before any further code.

### 12.2 P3 entry, step 3: SCIP's root cuts read, and the named differences

**What SCIP applies** (`gap-plan-p3/p3_scip_root_cuts.{py,json.gz}`; aggregation
separators only, presolve off, root): on `nexp-50-20-1-1` 440 cuts over 37 rounds (216
c-MIR, 202 flow cover, 22 knapsack cover); of the 41 binding at the final root LP, 18 are
lifted covers that are pure binary (`y_a + y_b >= 1`: "one of these arcs opens"), 17
flow covers, 6 c-MIR. On `neos-911970` (an assignment model with knapsack-type cost rows
`sum a_j y_j - x_k <= 6.5`, one continuous `x_k` per row carrying the objective) all
97 binding cuts are single-row c-MIR over one cost row's binaries and its `x_k`.

**Our separator on `neos-911970`, first root round** (temporary dump): 80 c-MIR cuts,
all violated at the LP point (up to 0.93), 40 of them on an objective column, the same
shape as SCIP's. The bound still did not move, because of the root loop's stop rule.

**Named differences, each measured on the route's root (c-MIR-only arm unless stated;
`p3_root_arms_stall.jsonl`, `p3_root_arms_3x.jsonl`):**

1. **The root loop stops at the first stalled round.** `milp_driver.rs` breaks as soon as
   one round fails to raise the bound by 1e-7 relative; a dual-degenerate LP (the
   assignment rows) needs several rounds before the bound moves. SCIP allows stalled
   rounds at the root. With 10 allowed: `neos-911970` 57.5 % -> 31.9 %.
2. **The start-row filter requires a nearly tight row** (`MAX_BASE_SLACK` = 0.1 of the
   row's largest coefficient). A knapsack-type row's cover / MIR cut is violated
   exactly when the row is slack. Relaxing it, with W4's HiGHS delta selection, flip and
   lifted MIC and the stall allowance: `neos-911970` 21.9 %, `beavma` 8.1 %, `nexp-1-1` 24.
3. **The root cut budget.** With 3x the route's cut and round budget (1,500 / 150), c-MIR
   alone reaches **`nexp-50-20-1-1` 0.0 % (29 = optimum)** and **`neos-911970` 5.4 %**,
   matching SCIP's aggregation-only root (0.0 % and 6.0 %); `mik-1` 3.1 %, `beavma` 8.1 %.
4. **GMI + c-MIR together lose to the wall-clock cap on the root loop, not to the cuts.**
   On `nexp-1-1` the c-MIR-only arm runs 125 rounds to 29; the combined arm stops at 23
   rounds and 19 (below GMI alone, 49 rounds, 21): dense GMI rows slow every re-solve and
   the loop's time ceiling ends it. Which arm wins differs by row: combined is better on
   `sp150x300d` (11.6 % vs 23.2 %), `nexp-50-20-4-2` (12.7 % vs 28.2 %) and `mik-1` (2.7 %
   vs 3.1 %).

Root gap %, combined table (route, root only):

| row | today (GMI) | c-MIR 3x | GMI + c-MIR 3x | SCIP aggregation | HiGHS presolve off |
|---|---|---|---|---|---|
| beavma | 18.4 | 8.1 | 8.1 | 1.1 | 0.9 |
| nexp-50-20-1-1 | 27.6 | **0.0** | 34.5 | 0.0 | 3.8 |
| nexp-50-20-4-2 | 9.9 | 28.2 | 12.7 | 0.8 | 0.6 |
| sp150x300d | 15.9 | 23.2 | 11.6 | 4.6 | 2.6 |
| neos-911970 | 57.5 | **5.4** | 30.9 | 6.0 | 4.8 |
| mik-250-20-75-1 | 4.1 | 3.1 | 2.7 | 2.7 | 1.75 |

No arm put a bound past its optimum (every comparison checked by the instrument).

**Not yet explained:** the flow-cover and lifted-cover share of SCIP's binding cuts on
`nexp` (we reach its optimum by c-MIR alone, so not needed there), and the rows where c-MIR
alone is worse than today (`nexp-4-2`, `sp150x300d`: the GMI cuts it displaces were doing
the work). All of the above ran with temporary environment knobs; nothing of it is
committed code.

**Design this points to (to build next, default OFF, one flag):** the W4 c-MIR core with
HiGHS's delta / flip / lifted MIC folded in; root stall rounds; no tight-row start filter
(rank start rows instead); and a root loop that budgets families by measured yield per
LP second instead of one shared cap (so GMI stays where it pays and stops where its dense
rows only slow the loop). Then the pre-registered entry on these six rows at the route's
real budget, then the §5 panel.

### 12.3 P3: the consolidated flag and its entry (2026-11-09)

`DISCOPT_MILP_ROOT_CMIR` (default OFF) replaces the experiment knobs: the W4 c-MIR core
with HiGHS's delta selection, flip and lifted MIC, no tight-row start filter, 10 stalled
root rounds, and a 1,500 / 150 root budget; `1` keeps GMI, `only` drops it. Rust tests:
`textbook_flow_cover_is_separated_under_both_designs`,
`design_cuts_hold_at_every_sampled_feasible_point` (40 seeds, every cut checked at every
sampled integer-feasible point), `root_cmir_parse_refuses_garbage`. The aggregation-depth
and variable-bound-row knobs were falsified (§12.1) and are gone.

**Entry, as pre-registered** (route, root only, the panel's 20 s; `p3_entry20_*.json`):

| row | today | `=1` | `=only` | HiGHS presolve off |
|---|---|---|---|---|
| beavma | 18.4 | **8.1** | **8.1** | 0.9 |
| nexp-50-20-1-1 | 27.6 | 34.5 | **0.0** | 3.8 |
| nexp-50-20-4-2 | 9.9 | 12.7 | 28.2 | 0.6 |
| sp150x300d | 15.9 | **11.6** | 23.2 | 2.6 |
| neos-911970 | 57.5 | **30.9** | **5.4** | 4.8 |
| mik-250-20-75-1 | 4.1 | 2.7 | 3.1 | 1.75 |

No bound past an optimum (6 / 6 comparisons per arm). **Neither arm meets the bar**
(>= 2 pp on >= 4 of 6; each reaches 3). `only` also meets the kill clause as written
(< 2 pp on >= 3 rows), but that clause encoded "the family is not the lever", which the
SCIP ablation (§12) refutes directly, so the result is recorded as *not passed*, not as
a kill. The best of the two arms per row would pass (>= 2 pp on 4; within 2 pp of
HiGHS on 3), so what is missing is (a) budgeting the families per solve and (b) the
separator strength on `nexp-50-20-4-2` and `sp150x300d`, where SCIP's aggregation alone
reaches 0.8 % / 4.6 % and ours is worse than GMI. Half of SCIP's binding root cuts on
`nexp-1-1` come from its flow-cover and lifted-cover generators (§12.2), which `cmir.rs`
has no counterpart for. Also measured: the `=1` arm spends the whole 20 s at the root on
`nexp-1-1` and `neos-911970`; the loop's time ceiling (checked at the top of a round)
does not hold when a round's GMI re-solves are slow, which a full-solve arm would pay.

**Next (as first written, retracted below):** a flow-cover generator over the same
aggregated rows, then per-solve family budgeting, then this entry again.

**Retracted before building (same day).** SCIP's aggregation separator with exactly one
generator on (`p3_scip_root_ablation.py` arms `agg_cmir` / `agg_flow` / `agg_knap`;
`p3_scip_generator_ablation.jsonl`), root gap %:

| row | aggregation | c-MIR only | flow cover only | knapsack cover only |
|---|---|---|---|---|
| nexp-50-20-4-2 | 0.78 | **0.94** | 7.25 | 30.86 |
| sp150x300d | 4.63 | **5.23** | 44.22 | 51.89 |
| nexp-50-20-1-1 | 0.00 | **0.00** | 30.79 | 55.49 |
| beavma | 1.07 | **0.84** | 48.17 | 55.52 |
| neos-911970 | 5.96 | **5.96** | 57.52 | 57.52 |
| mik-250-20-75-1 | 2.73 | **2.42** | 6.47 | 18.62 |

SCIP's c-MIR generator alone does the work on every row; its flow-cover and lifted-cover
rows in the final root LP (§12.2) were incidental. So the "half of SCIP's binding cuts
are flow / lifted covers" reading of §12.2 does not identify the missing piece, and a
flow-cover generator is not the next step. What remains is c-MIR strength on
`nexp-50-20-4-2` (ours 28.2 % alone vs SCIP's c-MIR 0.94 %) and `sp150x300d` (23.2 % vs
5.23 %): SCIP's aggregation path (row scoring, aggregation depth, which rows it walks
to), its bound-substitution choice, or its delta / complementation. **Next:** the §12.2
procedure on `nexp-50-20-4-2` with the `agg_cmir` arm (SCIP's c-MIR cuts against ours
on the same rows), then per-solve family budgeting, then this entry again.

### 12.4 SCIP's c-MIR on `nexp-50-20-4-2`, and aggregation depth re-measured

SCIP with only `aggregation` + `cmir` on (`p3_scip_nexp42_cmir_cuts.json.gz`): root bound
70.34 (optimum 71) from 283 cuts in the final root LP, 157 binding, cut names up to
round 48. Binding cuts are large: median 16 integer columns (max 48), median 1
continuous column, 47 of them pure-integer; the top ones read as cut-set inequalities
over several arcs' capacity modules (`t_x.<arc>.<module>`), i.e. aggregations of several
flow-conservation rows.

Aggregation depth 2 -> 6 under the full P3 design (temporary constant change, reverted;
`p3_entry20_aggr6_*.json`), root gap %, 20 s:

| row | `only`, depth 2 | `only`, depth 6 | `1`, depth 2 | `1`, depth 6 |
|---|---|---|---|---|
| nexp-50-20-4-2 | 28.2 | 23.9 | 12.7 | 11.3 |
| sp150x300d | 23.2 | 26.1 | 11.6 | 7.2 |
| nexp-50-20-1-1 | 0.0 | 3.4 | 34.5 | 17.2 |
| neos-911970 | 5.4 | 5.4 | 30.9 | 30.9 |
| beavma | 8.1 | 6.9 | 8.1 | 6.4 |
| mik-250-20-75-1 | 3.1 | 3.1 | 2.7 | 2.7 |

Mixed, and `nexp-50-20-4-2` stays near 24 % against SCIP's c-MIR 0.94 %: depth is not
the missing piece either. What SCIP builds there (16-48-column cut-set rows from many
rounds) points at *which* rows the path walks: SCIP scores aggregation candidates by
the continuous column's bound distance and the row's density and slack, and walks along
flow rows; ours eliminates the column farthest from its bound through the tightest row
containing it, so on a network it can keep re-selecting the same few rows. Next: log
the rows our aggregation uses from one `nexp-4-2` start row against the support of a
binding SCIP cut, and change the path rule only if that shows the divergence.

### 12.5 The aggregation path, traced (2026-11-09)

Temporary trace in `cmir.rs` (knobs saved as `gap-plan-p3/p3_path_trace_knobs.patch`, not
on the solve path), `nexp-50-20-4-2`, `DISCOPT_MILP_ROOT_CMIR=only`, root:

* **Our cuts are tiny.** 240 of 13,580 start-row attempts yield a cut; cut support is a
  median 2 integer columns (max 8), against SCIP's binding c-MIR cuts at median 16
  (max 48).
* **Every refusal is "no violated MIR", on single-arc rows.** 15,112 of 15,461 refused
  substituted rows carry exactly 4 integer columns: one arc's 4 capacity-module
  binaries. The separator works almost only on capacity rows `flow <= sum_k cap_k y_ak`,
  which alone yield nothing violated.
* **Why it never builds cut sets.** (1) `has_int_structure` skips start rows with no
  integer column and no detected variable bound, i.e. every flow-conservation row;
  (2) the variable-bound reader takes 2-3-nonzero rows only, so a 4-module capacity row
  is no variable bound and the flows have none; (3) when a flow is eliminated, the
  candidate rows (the next node's conservation row, slack 0, and the flow's capacity
  row, tight at the LP) tie on slack, so the path often walks into the next
  conservation row, which adds flows and no integers. Skipping variable-bound rows
  (§12.1's knob) changes nothing at all: they are never candidates.

Measured with the three removed (continuous start rows, candidate rows ranked by their
continuous count before slack, depth 6/10; `p3_entry20_path_*.json`, 24 comparisons, no
bound past an optimum), root gap %:

| row | `only` before | `only` + path, depth 10 | `1` + path, depth 10 |
|---|---|---|---|
| nexp-50-20-4-2 | 28.2 | **7.0** | 7.0 |
| nexp-50-20-1-1 | 0.0 | 0.0 | 20.7 |
| neos-911970 | 5.4 | 5.4 | 30.9 |
| mik-250-20-75-1 | 3.1 | 3.7 | 3.2 |
| sp150x300d | 23.2 | 29.0 | 14.5 |
| beavma | 8.1 | **37.4** | 12.5 |

The path is a real lever on the network row (`nexp-4-2` 28.2 -> 7.0 %, cut support
median 7, max 31), but a lexicographic "fewest continuous first" rule breaks `beavma`
(8.1 -> 37.4 %). SCIP's `aggregateNextRow` blends density, slack and integer content
into one score per candidate row; that blend, not a lexicographic rule, is the next
thing to port, then this table again. The `1` arm's 20 s root on `nexp-4-2` and
`neos-911970` (the time-ceiling overrun of §12.3) is unchanged.

### 12.6 HiGHS's path rules, ported under the flag (2026-11-09)

Under `DISCOPT_MILP_ROOT_CMIR`, `cmir.rs` now aggregates the way `HighsPathSeparator.cpp`
does (lines 30-420): every row starts a path except a substitution row; an *equality* row
in which exactly one continuous column is off its bound is that column's substitution
row and is applied whenever the column appears; the next row for the eliminated column
is drawn pseudo-randomly (deterministic in start row, level and column) among the tight
rows containing it whose multiplier lies in [1e-6, 1e6], rotating from the draw
(`findRow`); path length 6 (`maxPathLen`). Off the flag the separator is unchanged.
Soundness does not depend on these choices (every row of the standard form is an
equality); the design sampling test covers the new path (40 seeds).

First cut of the port blocked inequality rows with one continuous column as start rows,
which HiGHS does not (`kEq` only): `neos-911970` 5.4 -> 44.6 %. With substitution rows
restricted to rows without a slack column (`p3_entry20_hpath2_*.json`; the first cut is
`p3_entry20_hpath_*.json`), root gap %, 20 s, no bound past an optimum (12 / 12):

| row | today | `only` §12.3 | `only` + path | `1` §12.3 | `1` + path | HiGHS presolve off |
|---|---|---|---|---|---|---|
| beavma | 18.4 | 8.1 | **4.8** | 8.1 | **4.2** | 0.9 |
| nexp-50-20-1-1 | 27.6 | 0.0 | 0.0 | 34.5 | 24.1 | 3.8 |
| nexp-50-20-4-2 | 9.9 | 28.2 | **8.5** | 12.7 | **7.0** | 0.6 |
| sp150x300d | 15.9 | 23.2 | 21.7 | 11.6 | 14.5 | 2.6 |
| neos-911970 | 57.5 | 5.4 | 5.4 | 30.9 | 30.9 | 4.8 |
| mik-250-20-75-1 | 4.1 | 3.1 | 3.1 | 2.7 | 2.5 | 1.75 |

Against the pre-registered bar: `1` gains >= 2 pp on **4 of 6** (beavma, nexp-1-1,
nexp-4-2, neos-911970) but ends within 2 pp of HiGHS on 1 (mik-1; 2 needed); `only`
gains on 3 and is within 2 pp of HiGHS on 3. Not passed; `1` is one row short on the
second clause. Still open: `sp150x300d` (SCIP's c-MIR alone 5.2 %), the `1` arm's root
time overrun (21 s / 20 s on beavma / neos-911970), and the GMI / c-MIR split (`only`
wins nexp-1-1 and neos-911970, `1` wins sp150x300d).

### 12.7 The `=1` arm's time overrun, root-caused (2026-11-09)

`neos-911970`, `DISCOPT_MILP_ROOT_CMIR=1`, root-only probe at 20 s returned at 39.5 s.
Profile (one driver call): root cut loop 10.2 s (inside its `root_cut_time_s`), root dive
9.4 s, then with the dive capped, a RENS sub-MIP 19.8 s and strong branching 33.7 s over
8 calls. Three defects, fixed in that order:

1. **The root dive is unbounded.** Over the cut-augmented LP (215 dense rows) each of
   its up-to-840 fixings re-solves the LP. Under the flag a dive now stops at
   `ROOT_DIVE_SHARE` = 10 % of the time left, at most 2 s (`DiveCapped`). HiGHS budgets
   heuristics as a share of the search.
2. **Strong branching checked the deadline once per node, before probing.** It now
   stops probing past the deadline and branches on the best candidate so far
   (`StrongBranchDeadlineStop`; only the branching variable can change). 33.7 -> 19.9 s.
3. **A heuristic sub-solve inherited the parent's `extend_no_incumbent_s`** (the
   NO_STALL_TREE rule: with no incumbent at the time limit, move the deadline to the
   whole remaining budget). RENS runs exactly when there is no incumbent, so its 2 s
   sub-budget became the parent's full budget, counted from the sub-solve's start:
   19.8 s. `sub_solve_options` now drops the parent's deadline policies
   (`extend_no_incumbent_s`, `stall_time_s`) for every sub-solve (RENS/RINS and the P1
   stall re-solve); regression test `sub_solve_options_drop_the_parents_deadline_policies`.
   **This one is a default-path bug** (it lives in #2196's code); on the last §5 panel's
   default arm it shows as 1 overrun in 372 solves (`sp150x300d` 21.1 s at 20 s), because
   the default root dive usually supplies an incumbent first. Ported to #2196.

After all three: 19.2 s, inside the limit. The `=1` arm's root bound on `neos-911970`
(30.9 %) is unchanged: that is the GMI / c-MIR split, still open.

### 12.8 c-MIR first, GMI on stall: the entry bar is met (2026-11-09)

HiGHS's length cap (100 + 0.15 n) already applies to every root cut (`CUT_LEN_BASE`,
`CUT_LEN_FRAC`), so GMI density is not the issue. Under `DISCOPT_MILP_ROOT_CMIR=1` the root
loop now separates c-MIR alone until the bound stalls (or c-MIR finds nothing), then GMI
joins with a fresh stall allowance (`RootCmirGmiHandover`). A per-solve, measured rule:
the rows `only` won (nexp-1-1, neos-911970) are closed by c-MIR before it stalls; GMI
still runs where c-MIR runs dry. `p3_entry20_cmirfirst_1.json`, root gap %, 20 s:

| row | today | c-MIR first, then GMI | HiGHS presolve off |
|---|---|---|---|
| beavma | 18.4 | **3.9** | 0.9 |
| nexp-50-20-1-1 | 27.6 | **0.0** | 3.8 |
| nexp-50-20-4-2 | 9.9 | **7.0** | 0.6 |
| sp150x300d | 15.9 | 21.7 | 2.6 |
| neos-911970 | 57.5 | **6.4** | 4.8 |
| mik-250-20-75-1 | 4.1 | 2.5 | 1.75 |

**Pre-registered entry bar: met.** >= 2 pp on 4 of 6 (beavma, nexp-1-1, nexp-4-2,
neos-911970); within 2 pp of HiGHS on 3 (nexp-1-1, neos-911970, mik-1); no root bound
past an optimum (6 / 6). Caveats for the panel: `sp150x300d` is worse than today
(GMI now arrives only after c-MIR stalls), and the root takes 17-20 s of 20 on three rows
(nexp-1-1, nexp-4-2, neos-911970), which a full solve pays in tree time. Next: the §5
panel (`d2_flag_panel.py --flag DISCOPT_MILP_ROOT_CMIR --fired RootCmirGmiHandover`,
MIPLIB-38 + M, 3 reps, 20 s) and the corpus differential.

### 12.9 §5 panel (2026-11-09): not graduated; the root starves the tree

`d2_flag_panel.py --flag DISCOPT_MILP_ROOT_CMIR --fired RootCmirGmiHandover` (MIPLIB-38 + M,
3 reps, 20 s, 372 solves; `gap-plan-p3/p3_root_cmir_panel.json.gz`): bar 1 passes (no bound
past an optimum); certified 126 -> 127 (`gesa2` 0 -> 3, `nexp-50-20-1-1` 0 -> 1,
`neos-3610051-istra` 3 -> 0); nodes on the always-certified set 60,244 -> 34,685 (-42 %).
Fails: bar 1b (`istra`), 2c (6 dual losses, cap 1), 2d (incumbent lost on 7: `enlight8`,
`neos-911970`, `nexp-50-20-1-1`, `nexp-50-20-4-2`, `sp150x300d`, `istra`,
`neos-3118745-obra`), 2e (median primal gap 0.137 -> 0.212). The bound side works; the
cost is the §12.8 caveat: the root takes most of the 20 s, so the tree and its primal
heuristics starve. Default OFF. One rework is justified by the measurement (bound the
root's share of the budget under the flag, then re-panel); if that panel fails too, the
flag and its implementation are retired.

### 12.10 Primal starvation fixed from HiGHS evidence; presolve hypothesis falsified (2026-11-09)

The §12.9 starvation was chased in quick 20 s target solves (not panels), with every step
taken from HiGHS's source or its logs rather than guessed:

- **HiGHS's incumbent sources** on the rows the flag lost (`p3_highs_solution_sources.*`):
  sub-MIP at the root (`L`) — sp150x300d 0.1 s, nexp-50-20-4-2 1.6 s, neos-911970 2.1 s.
- **Ported** (`P3_*` experiment knobs, flag-on path only): per-round cut ageing as
  `performAging` (`HighsLpRelaxation.cpp:629-675`), randomized rounding after every root
  round with no incumbent (`HighsMipSolverData.cpp:1858`), RENS/RINS as
  `HighsPrimalHeuristics.cpp` (fixing dive to 0.6 / RINS 0.25, sub-MIP on the cut LP with
  ≤ 5 rounds, no strong branching, `200 + nodes/20`, one recursion level). Tree LPs expel
  zero artificials (short bases had refused every warm start). Incumbents came back on every
  row that had lost one; certified on the targets 0 -> 3 (istra guard, beavma, gesa2).
- **Falsified: "presolve is why HiGHS solves our RENS neighbourhood".** HiGHS solves the
  neighbourhood discopt builds (dumped, `P3_RENS_DUMP`) at 1 node with presolve OFF and
  heuristics OFF alike (`p3_highs_on_rens_dump.py`). What does it is the root cut loop:
  LP 64.0 -> 68.998 (sp150, opt 69) and 16.2 -> 29 (nexp-50-20-1-1, opt 29) in ≤ 0.3 s. On
  the same MPS, discopt's root reaches 67 (1.4 s) and 26 (12 s with c-MIR on). Presolve was
  not built.
- **Bug found:** `lifted_mic_knapsack`'s cover sort used a tolerance comparator that is not a
  total order; Rust's sort panicked on it (mik-250-20-75-1 in a sub-MIP). Fixed with buckets.

Next is the separator itself — strength and speed on these neighbourhoods, where HiGHS's
root is the measured reference (`nbhd_*.mps` can be regenerated with `P3_RENS_DUMP`).

### 12.13 §5 bundle panel (2026-11-10): not graduated

Bundle `DISCOPT_MILP_ROOT_CMIR,DISCOPT_MILP_RAND_ROUND,DISCOPT_MILP_RUN1_CONTINUE` after the
fold (`62e83dab`), `d2_flag_panel.py --fired SepCmirCuts`, MIPLIB-38 + M, 3 reps, 20 s, 372
solves (`gap-plan-p3/p3_bundle_panel.json.gz`). Bar 1 passes (372 checks, no bound past an
optimum). Certified 129 -> 123 (incl. bimodal p0548): gained `gesa2` 0 -> 3,
`nexp-50-20-4-2` 0 -> 3; lost `2122` 3 -> 0, `b-ball` 3 -> 0, `dcmulti` 6 -> 0 (bar 1b).
Nodes on the 37 always-certified rows 38,370 -> 33,357 (-13 %). Dual losses 4 (cap 2),
incumbent lost on `enlight8`, median primal gap 0.012 -> 0.045. Fails 1b, 2a-2e.

The losses are tree-side, not root time: `b-ball` runs 20,135 nodes (off 5,753) and ends at
a WEAKER bound (-1.5499 vs -1.50015, certified off in 2.3 s); `dcmulti` (off 4 s) ends with a
worse bound and incumbent. Under the flag the tree differs in its root cut set (all-row
c-MIR + tableau, pruned/remapped), full bases (expelled artificials) and the RENS/RINS
port; which one weakens the node bound on these rows is the next measurement. Default OFF.

### 12.14 Where the HiGHS gap is: a population-wide ablation of HiGHS itself (2026-11-10)

The instance-by-instance P3 work (§12.10-12.13) netted **zero**: on MIPLIB-38 + M at 20 s
(1 rep, `p3_ab_main_vs_branch.*`) main certifies 43/62, this branch 43 (flag off) and 41
(flag bundle on), HiGHS 59. So the question was re-asked structurally: which HiGHS stages
account for its lead? HiGHS (ref source, ablation build `p3_highs_*.patch`, 1 thread, 20 s)
with one stage off per arm (`p3_highs_ablate_population.*`):

| arm | certified | lost vs default |
|---|---|---|
| default | 60 | - |
| presolve off / restarts off / symmetry off / heuristics off / strong branching off | 60 each | 0 |
| conflict analysis off | 61 | 0 |
| presolve + restarts + symmetry + conflicts all off | 60 | 0 |
| **cuts off** (all separators, root and node) | **47** | **13** |

At this budget on this population HiGHS's lead is its **cut separation**: without cuts it
certifies 47 (discopt: 43); without presolve, restarts, symmetry and conflict analysis
together it still certifies 60. The 13 rows lost without cuts are the P3 target list
(beavma, sp150x300d x2, nexp-50-20-1-1/-4-2, neos-911970, neos17, gsvm2rl3, enlight_hard,
fiber, issue-2446, b-ball, neos-3611689-kaihu). This retires the "missing presolve /
restarts / conflicts" hypotheses for this panel and makes one general component the work.

Per family (`p3_highs_ablate_separators.*`): no single family is decisive. Tableau off 58,
path off 58, mod-k off 60, implied-bound + clique off 59, node cuts off 58; **path only 58,
tableau only 57**. The families back each other up, so HiGHS with *either* its tableau or
its path separator alone keeps almost all of its lead. Discopt's flag ports both and its
certified count did not move, so the gap is in how the same families perform on full
models, or downstream of them.

Root bound, HiGHS (presolve off, root only) vs discopt (flag bundle, root only), on the
cut-dependent rows (`p3_root_cmp_highs.py`): HiGHS tighter on beavma 0.82 vs 5.81 %,
sp150x300d 0.00 vs 2.90, nexp-4-2 0.00 vs 1.41, neos-911970 4.77 vs 13.70, neos17 2.60 vs
9.97, b-ball 6.55 vs 21.21, enlight_hard 29.7 vs 46.0, enlight8 25.9 vs 59.3, fiber 2.11 vs
3.07, mik-1 1.75 vs 2.40; comparable or better for discopt on nexp-1-1 (0/0),
gsvm2rl3 (23.9/22.2), neos-3611689-kaihu (4.2/2.5). And HiGHS's root takes **0.2-1.8 s**
where discopt's takes **2-20 s**. Two general gaps follow: root cut quality on full models
(10 of 14 rows) and root cost (5-20x), which leaves the tree little time; on the rows with
a comparable root, HiGHS still certifies via its tree.

### 12.15 Separator differential on full models (2026-11-10)

Method (`p3_cut_eval_same_lp.py`, `p3_cut_single_gain.py`, dumps from
`p3_highs_separation_trace_dump.patch` / `p3_separator_differential_instruments.patch`): each
solver's root cuts, round by round, are added to the **same** LP relaxation (HiGHS's LP,
presolve off; discopt's structural columns verified equal to the MPS columns via the
objective). beavma:

| cuts from rounds | HiGHS cuts -> bound | discopt cuts -> bound |
|---|---|---|
| 1 | 11 -> 171,642 | 13 -> 156,315 |
| 2 | 33 -> 223,139 | 22 -> 159,736 |
| 20 | 191 -> 348,816 | 148 -> 291,371 |

(LP without cuts 155,778.) The strong first round is HiGHS's path separator (path-only: 22
cuts -> 176,761). Per-cut single gains show discopt's **generator** is not the round-1
problem: its 82 round-1 candidates include cuts worth 4,535 / 2,131 / 2,096 alone (sum 47k
vs HiGHS's 22 path cuts 23.7k; all 82 together 168,688). Its **selection** kept the 13
weakest (single gains 0-139). Cause: a porting bug in the score's active-entry rule — HiGHS
counts an entry active unless its column sits at the activity-minimising bound
(`HighsCutPool.cpp:309-330`); the port counted only columns strictly inside both bounds, so a
zero-gain cut with an active norm of 3.7e-4 scored highest and set the adaptive threshold.

Fixing the rule makes round 1 stronger (8 cuts -> 163,953) but the root WORSE overall on
the 14 cut-dependent rows (beavma 5.8 -> 47.2 %, sp150x300d 2.9 -> 26.1 %): with a correct
score, the 0.1 parallelism filter leaves 2-8 cuts per round from discopt's 80-160
candidates and the loop stalls; HiGHS keeps 10-22 per round under the same filter
(parallelism measures verified identical). Keeping unselected candidates in a persistent
pool, as HiGHS does, recovers some rows (nexp-50-20-4-2 0.0 %, gsvm2rl3 6.9 %) but not the
big ones (beavma 33.8 %). Both changes are reverted (patch kept); the measured lesson is that
HiGHS's selection only works on HiGHS's candidate population, which is far less redundant
than ours after round 1. Porting pieces in isolation does not transfer.
