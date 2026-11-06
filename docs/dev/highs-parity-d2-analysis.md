# D2 analysis — can the Rust MILP route close the gap to the HiGHS route, and if not, what is the narrowest acceptable end state?

Date 2026-11-06. Tree `/home/user/discopt` @ `ebe4a955` (branch `claude/happy-hamilton-0vvo3b`),
extension `python/discopt/_rust.abi3.so` built 2026-11-06 02:34 (P4.9's build `d8d40da0` plus
P6.1/P6.2 docs/tests; no engine change since the gate run). Host: 4 cores. Every number below is
either copied from a checked-in data file (path given) or from the probe
`scratchpad/d2_probe.py` (outputs `scratchpad/b{1,2,3,4,5}_*.jsonl`, fresh subprocess per solve,
`DISCOPT_LP_MILP_BACKEND=rust`, `DISCOPT_PROFILE=1`, `discopt.__file__` asserted, executed count
printed; at most 2 concurrent solves, so **wall figures from the probe are indicative only** —
phase *shares* and node/bound/incumbent values are what is used). Nothing in the tree was edited.

## 0. The gate as measured (P4.9, `docs/dev/data/highs-parity/p4.9/`)

| bar | required | Rust (R) | HiGHS route (H) |
|---|---|---|---|
| MIPLIB-38 certified | ≥ 34 | **16** | 36 |
| MIPLIB total wall | ≤ 2× H | 2.75× (477.5 / 173.7 s) | |
| M certified | ≥ 23/24 | **18** | 24 |
| incorrect | 0 | 0 | 0 |

R fails 22 MIPLIB instances; H certifies 20 of those (it also fails `neos-2624317-amur` and
`neos-3118745-obra`). R fails 6 M instances (`2122`, `gesa2`, `gt2`, `issue-2446`, `p0548`,
`sp150x300d`; `gt2`/`sp150x300d` are also in MIPLIB), all of which H certifies. **Distinct
instances to flip: 24** (20 + 4 M-only). Reaching 34/38 means flipping 18 of the 20 MIPLIB rows
H certifies.

## 1. Per-instance diagnosis

### 1.1 Per-node cost vs per-node strength — the two populations

From `p49_miplib.json` / `p49_m_inf.json` (median of 3 rounds; nodes/s = nodes ÷ wall):

| instance | R nodes | H nodes | R n/s | H n/s | nodes R/H | (n/s) R/H | R primal gap | R dual gap | H wall |
|---|---|---|---|---|---|---|---|---|---|
| mik-250-20-75-1 | 64,476 | 453 | 3,200 | 40 | 142× | 80× | 0 | 0.032 | 11.3 s |
| mik-250-20-75-2 | 58,910 | 91 | 2,925 | 20 | 647× | 149× | 1e-16 | 0.032 | 4.6 s |
| mik-250-20-75-3 | 59,518 | 433 | 2,955 | 41 | 138× | 71× | 0.0026 | 0.027 | 10.5 s |
| mik-250-20-75-5 | 61,732 | 159 | 3,066 | 20 | 388× | 156× | 0.0005 | 0.030 | 8.1 s |
| b-ball | 208,207 | 101 | 10,357 | 361 | 2,062× | 29× | 4e-15 | **0.00024** | 0.28 s |
| gt2 | 79,777 | 1 | 3,961 | 4 | 79,777× | 964× | 0.139 | 1e-12 | 0.24 s |
| neos-3610040-iskar | 119,376 | 380 | 5,935 | 49 | 314× | 122× | 0.105 | 1e-15 | 7.8 s |
| neos-3611689-kaihu | 96,554 | 283 | 4,802 | 91 | 341× | 53× | 0.183 | 0.0083 | 3.1 s |
| neos-3611447-jijia | 104,810 | 384 | 5,213 | 137 | 273× | 38× | 0.157 | 0.0093 | 2.8 s |
| neos-3610051-istra | 24,383 | 288 | 1,209 | 29 | 85× | 42× | 0.08 | 0.02 | 10.0 s |
| gsvm2rl3 | 100,998 | 2,547 | 5,019 | 286 | 40× | 18× | 0.187 | 0.11 | 8.9 s |
| bppc8-02 | 11,577 | 152 | 574 | 230 | 76× | 2.5× | 0.018 | 0.040 | 0.66 s |
| enlight_hard | 5,721 | 2,303 | 286 | 187 | 2.5× | 1.5× | none | 0.263 | 12.3 s |
| nexp-50-20-1-1 | 9,051 | 1 | 449 | 3 | — | 165× | 0.30 | 0.233 | 0.37 s |
| nexp-50-20-4-2 | 18,096 | 1 | 895 | 1 | — | 1,773× | 0.208 | 0.153 | 2.0 s |
| gesa2 (M) | 11,527 | 1 | 548 | 0.4 | — | 1,368× | 0.027 | 0.0046 | 2.5 s |
| p0548 (M) | 1,245 | 0 | 95 | — | — | — | none | 0.063 | 0.52 s |
| **fiber** | **27** | 7 | **1.3** | 4 | 3.9× | **0.3×** | 0.313 | 0.191 | 1.7 s |
| **neos-911970** | **29** | 1,077 | **1.4** | 87 | 0.03× | **0.02×** | 1.05 | 0.565 | 12.4 s |
| **neos17** | **249** | 1,524 | **12** | 156 | 0.16× | **0.08×** | none | 0.115 | 9.8 s |
| **beavma** | **441** | 1 | **22** | 1.5 | — | 15× | 0.254 | 0.141 | 0.68 s |
| **sp150x300d** | **1,987** | 1 | **99** | 1 | — | 103× | none | 0.30 | 1.0 s |
| **2122 (M)** | **3** | 2 | **0.15** | 1.3 | 1.5× | **0.1×** | 1.05 | **4.7e-5** | 1.6 s |
| **issue-2446 (M)** | **1** | 1 | **0.05** | 0.2 | 1× | **0.2×** | none | 25.1 | 4.5 s |
| neos-3118745-obra (H fails) | 25,681 | 3,340 | 1,277 | 166 | 7.7× | 7.7× | 0.28 | 0.0039 | 20 s |
| neos-2624317-amur (H fails) | 609 | 2,474 | 30 | 189 | 0.25× | 0.16× | none | 0.779 | 13.1 s |

Over the 29 instances both arms certify, geomean nodes R/H = **31.5×** and geomean nodes/s R/H =
**27×** (same files). This reproduces `milp-competitiveness-plan.md` A3 (105× / 80× in
September) with a smaller spread: R is still an order of magnitude *faster per node* and pays for it
with trees 1–3 orders of magnitude larger. (Caveat on every wall comparison in this plan: the Rust
node loop is rayon-parallel — `discopt-python/Cargo.toml:21` `default = ["parallel"]`; in the
probe the summed `NodeLpSolve`/`Fbbt` phase times exceed the `SearchLoop` wall by up to 2.7×, i.e.
≥ 3 threads were active. The HiGHS route's B&B runs with `threads` at HiGHS's default. The 2× wall
bar therefore already credits Rust with the host's cores.)

So the failures split into **two populations with opposite diagnoses**:

**(A) Per-node strength — 17 instances.** R explores 40–80,000× more nodes than H needs and still
does not close. Sub-classes by what is short (P4.9 values, P4.1 `attribution.json` for the 60 s
behaviour):

- *A1, bound at (or within 1e-4 of) the optimum, incumbent frozen far away — pure primal:* `gt2`
  (dual 1e-12, primal 0.139), `neos-3610040-iskar` (1e-15 / 0.105), `2122` (4.7e-5 / 1.05, and in
  population B too), `b-ball` (2.4e-4 / 0 — the bound is 2.4× the 1e-4 tolerance short; the
  incumbent *is* the optimum). HiGHS closes `gt2` at the root (1 node, 0.24 s) and `iskar` in 380
  nodes.
- *A2, bound within 1–2 % and incumbent 8–28 % off — primal-dominated:* `neos-3611689-kaihu`,
  `neos-3611447-jijia`, `neos-3610051-istra`, `neos-3118745-obra` (H fails too), `gesa2` (0.46 % /
  2.7 %), `bppc8-02` (4 % / 1.8 %). P4.1: on every A1/A2 row the incumbent at 60 s is **the same
  point as at 20 s** (A14's "frozen primal", re-confirmed after c-MIR and again by the P4.1 table).
- *A3, incumbent at the optimum, bound 2.7–3.2 % short — pure dual:* the four `mik-250-20-75-*`.
  Root gap 3.8–4.2 % after P4.4 vs HiGHS 1.5–1.8 % (`p4.4/` table in the plan); the tree moves the
  bound ~1 pp and then **stalls**: `mik-1` dual gap 0.0331 at 20 s (64k nodes) and 0.0331 at 60 s
  (120k nodes, `b4_60s.jsonl`); the same stall on `b-ball` (0.00024 at 20 s and at 60 s after
  347k nodes) and `bppc8-02` (0.0404 at both, incumbent unchanged). HiGHS needs 91–453 nodes from
  its 1.7 % root. Not a cost problem (3,000 n/s): on these rows the tree contributes nothing after
  the first seconds, so only the root/cut side (L5) or node-level separation can move them.
- *A4, weak bound and poor/no incumbent — both:* `nexp-50-20-1-1` (root 26.7 % after repair vs
  H 0 %), `gsvm2rl3` (H itself needs 2,547 nodes), `enlight_hard` (no incumbent; P4.6's FJ found
  nothing in 20 s), `p0548` (no incumbent after 6k Rust nodes; root bound 8.8 % short; 451 of 950
  root GMI cuts rejected by the dynamism gate, `p4.3/entry_head.json`).

**(B) Per-node cost — 7 instances, a population P4.1 did not name.** R is *slower per node than
HiGHS* by 5–50× and explores 1–441 nodes in 20 s: `fiber`, `neos-911970`, `neos17`, `beavma`,
`sp150x300d`, `2122`, `issue-2446` (+ `nexp-50-20-4-2` after P4.4). The probe's phase tables
(`b2_cost.jsonl`, run 1 = the route's first driver call, 9.8 s slice) say where the time goes.
Figures are thread-summed ms from `DISCOPT_PROFILE`:

| instance (HEAD) | nodes in slice | dominant phases (run 1) | mechanism |
|---|---|---|---|
| fiber | 13 | StrongBranch 6 calls 5.9 s; DiveRepair 3.7 s; NodeLpSolve 9 calls 1.1 s (127 ms/node); 3,029 refactorizations | P4.4 completed-basis GMI cuts: 377 emitted with 112,155 nnz = **297 nnz/cut** on a matrix averaging 8 nnz/row; 42 kept ≈ 4× the matrix's nnz. With `DISCOPT_MILP_ROOT_GMI_REPAIR=0`: 1,051 nodes, 31 ms/node, SB 0.35 s/call |
| neos-911970 | 29 | StrongBranch 14 calls 14.1 s; DualPivotLoop 125 solves 12.5 s; `DualColdFallbacks` 42/125, `DualExitDeadline` 27, `DualRefactorizations` 6,254 (50 per solve) | the warm dual simplex stalls on this 888×107 / 840-integer LP (cold fallbacks, deadline exits); each node LP ≈ 100 ms vs HiGHS ≈ 11 ms/node. LP-engine robustness, not search |
| 2122 | 3 | **DiveRepair 9.49 s of 9.66 s** (93 dive steps ≈ 100 ms each, 2,698 refactorizations) | the root continuous-repair dive is unbudgeted and consumes the whole slice; it then lands a point 105 % off (bound is 4.7e-5 from the optimum) |
| issue-2446 | 7 | DiveRepair 4.9 s (abandoned after 9 steps ≈ 0.5 s each); StrongBranch 1 call 2.9 s; RootCutLoop 0.9 s; 1,591 refactorizations | 2200×1963 LP at ≈ 0.3–0.5 s per warm re-solve (120 µs/pivot, dense `PriceSweep`); no incumbent → run 1 returns `node_limit` → the route **defers to the Python MILP tree** for the remaining 10 s |
| neos17 | 3 | DiveRepair 6.8 s (abandoned at step 0); RootSolve 2 calls 2.45 s | the completed-basis cuts make the root re-solve and the dive pathological; no incumbent → Python-tree fallback. With repair OFF: 9,771 nodes in the slice, incumbent found (primal 0.214), but `Fbbt` 8.6 s = 37 % of node time and `DualPrepare` (per-node refactorization) 12 s |
| beavma | 49 | StrongBranch 24 calls 14.1 s; Refactorize 6,310 4.4 s; DiveRepair 3.5 s | repair ON: node LP ≈ 10× slower than OFF (3,501 nodes/slice, 14 ms/node) |
| sp150x300d | 3 | DiveRepair 6.8 s (abandoned after 8 steps ≈ 0.85 s each); RootCutLoop 3.0 s (19 rounds) | no incumbent → Python-tree fallback. Repair OFF: 4,263 nodes, dive hits, primal 0.343 |
| nexp-50-20-4-2 | 3 | DiveRepair 9.5 s (abandoned after 217 steps) | no incumbent → Python-tree fallback. Repair OFF: 255 nodes, but **StrongBranch 31 calls = 21 s of ~30 thread-s** (0.68 s per SB call, 56 ms/node LP) |

Three general mechanisms, none of them "search quality":

1. **P4.4's completed-basis GMI cuts are dense** (297 nnz/cut on `fiber`, 319 on `nexp-50-20-4-2`,
   31 on `sp150x300d` vs 2.7/row) and every node LP, the root re-solve, strong branching and the
   dive pay for them. The graduation panel (`p4.4/p44_panel.json`) shows it — `fiber` 2,484 → 17
   nodes, `beavma` 10,266 → 35, `neos17` 11,609 → 187 and **its incumbent lost**, `sp150x300d`
   5,931 → 1,115 and incumbent lost, `nexp-50-20-4-2` incumbent lost, `obra` incumbent 0.215 →
   0.281, `bppc8-02` 0.0118 → 0.0177 — but the bars only scored certified count (+3 = `neos-1425699`
   × 3 reps), nodes on both-certified rows and dual-bound losses; per-node cost and incumbent losses
   on the open population were not bars. Net on P4.9: +2 certifications (`neos-1425699`,
   `khb05250`), `b-ball` dual 0.12 → 0.00024, better bounds on 6 more; and 5 instances pushed off the
   Rust engine (no incumbent → Python tree) or down to tens of nodes. The repair is a real bound win
   whose cut *density* was never gated.
2. **The root continuous-repair dive is unbudgeted** (`try_dive_repair`, up to `n_int + 1` warm
   re-solves, no iteration/time cap other than the solve deadline): 97 % of the slice on `2122` and
   `nexp-50-20-4-2`, 70 % on `sp150x300d`/`neos17`, 50 % on `issue-2446`, 36–38 % on `fiber`/`beavma`.
   It is also the **only** mechanism that ever produces an improving point (see 1.2).
3. **Strong branching has no per-probe LP-iteration cap**: 6 candidates × 2 probes on the first 48
   nodes (`lp_bindings.rs:1072`), each probe a full warm dual solve that may fall back cold. On
   `nexp-50-20-4-2` (repair OFF) that is 70 % of all node time, on `neos-911970` 100× the node-LP
   time. A10 varied the *node-prefix count* (6/48 vs 64/5000), never the per-probe iteration cap; the
   cost it found ("the per-node strong-branching cost eats the entire node gain") is this.

Plus one LP-engine case (`neos-911970`: warm dual stalls, 34 % cold fallbacks; HiGHS 11 ms/node) and
one pure LP-size case (`issue-2446`, 120 µs/pivot with a dense pricing sweep on 4,163 columns —
P2.3 killed hypersparse FTRAN/BTRAN at entry on the *L panel*; this instance was not in that
measurement).

### 1.2 Why the primal side is frozen — read from the code, not inferred

- Off the root the only heuristic is nearest rounding (`try_rounding_csc`, every fractional node).
- The dive (`try_dive_repair`) runs at the root, and at non-root batches only when
  `DISCOPT_MILP_DIVE_STRIDE > 0` **and `!has_incumbent`** (`dive_batch_eligible`,
  `milp_driver.rs:371-389`, default stride 0).
- RINS is **default OFF** (`rins_enabled`: `Err(_) => false`, `milp_driver.rs:883`; A12 measured it
  neutral-or-harmful). RENS was retired by P4.2. FJ was retired by P4.6.
- So after the first incumbent the search has **no improvement mechanism** — exactly A14's finding
  ("given 3× the budget, the incumbent improved on 0 of 14"), re-measured by P4.1 (12/12 primal rows
  hold the same incumbent at 20 s and 60 s).

The P4.0 re-entry restarts a fresh tree seeded with run 1's incumbent; on the A-population the
seeded run 2 re-explores about the same number of nodes as run 1 to the same gap (`b3_primal.jsonl`:
`iskar` 30,513 + 30,443; `kaihu` 26,983 + 28,023; `jijia` 24,625 + 26,363) — the slice/re-entry
design spends half the budget rebuilding the tree the first half built.

### 1.3 `gt2`: a flip that is not a lever (and a measurement that contradicts the plan's framing)

`gt2` certifies in **1.1 s / 11,931 nodes** with `DISCOPT_MILP_ROOT_CUTS=0` (the binding's legacy
16×1 root pass) and fails at 20 s / 250k nodes with the shipped 500×50 budget (`b1_gt2.jsonl`). The
dual bound is 21166 = the optimum either way; with the big cut set the root dive lands on 24106 and
nothing afterwards moves it. None of the other graduated flags matters (`ROOT_GMI_REPAIR=0`,
`ENGINE_FULL_BUDGET=0`, `NS_PROJECT=0`, `NS_NODE_BOUND=0`, `OBJ_LATTICE_SUBST=0`, `RINS=0`,
`SWAP_RESEED=0`: all 20 s, primal 0.139).

It is **not** general: the same switch on the rest of the primal class (`b3_primal.jsonl`) leaves
the incumbent identical on `iskar`/`kaihu`/`jijia`/`gsvm2rl3`/`b-ball`/`mik-1`, makes it *worse* on
`istra` (0.08 → 0.14), `obra` (0.28 → 0.36), `bppc8-02` (0.018 → 0.031), `gesa2` (0.027 → 0.057),
`fiber` (0.31 → 0.92), and loses `khb05250`'s certification; dual gaps are worse everywhere. `gt2`'s
sensitivity is the dive's starting vertex, which is 1.2's point again: one dive, once.

### 1.4 The September → November regression (contradicts §7's "superseded, not wrong")

`scratchpad/a14/rens_panel.json` (checked in; 2026-09-06, the shipped arm "OFF", 2 reps, 20 s,
gap 1e-4 — the same panel and tolerance as P4.9) certified **21/38**. P0.3 (2026-11-04, `baseline-
dcdac02/miplib.json`) certified 14/38; P4.9 16/38. Instances certified in September and not now:

| instance | Sept-6 shipped | P0.3 (Nov-4) | P4.9 (Nov-6) |
|---|---|---|---|
| fiber | optimal, 23,537 nodes, 9.5–12 s | feasible, 1,205 nodes | feasible, 27 nodes |
| khb05250 | optimal, 101 nodes, 0.12 s | feasible, 715 nodes, 12.6 s | optimal, 419 nodes, 9.4 s |
| gt2 | optimal, 2,643 nodes, 0.07 s | feasible, 49,589 nodes | feasible, 79,777 nodes |
| neos-3611689-kaihu | optimal, 97,269 nodes, 11–13 s | feasible, 47,349 | feasible, 96,554 |
| bppc8-02 | optimal, 11,883 nodes, 0.7–2.5 s | feasible, 8,923 | feasible, 11,577 |
| neos-3611447-jijia | optimal 1/2 reps, 100k nodes, 4.9 s | feasible | feasible |
| neos-1425699 | optimal, 3 nodes | time_limit | optimal (P4.4) |

and the node throughput on the open rows fell far more than the 14-core → 4-core host change
explains (≈ 3.5× for the rayon loop): `sp150x300d` 266k → 5.3k nodes at P0.3 (50×), `beavma` 148k →
3.9k (38×), `enlight_hard` 128–163k → 4.0k (35×), `neos17` 69k → 12k, `b-ball` 758k → 215k, `mik-*`
270–330k → 9–14k (P0.3 ran a 10 s slice, so halve those ratios; still 10–25×). §7's P0.3 entry
attributes the drop to the host; the per-instance data say the per-node cost of the *shipped*
engine rose 10–25× on these instances between 2026-09-06 and 2026-11-04 (the plan's own A3 number,
80× faster per node than HiGHS, is now 27×). The commits in that window that add per-node work are
all correctness fixes and are not candidates for removal — #1945 "verify every MILP-driver Optimal
against the original LP" (2026-10-25), #1953 cutoff-row FBBT and rigorous NS gate (10-25), #2115
NS projection (11-03; `NS_PROJECT=0` / `NS_NODE_BOUND=0` give +20 % / +25 % nodes on `gt2` in the
probe), #1510, #1409 — but **which of them costs what has never been measured**, and `fiber`/`gt2`/
`kaihu`/`bppc8-02`/`jijia` certifying in September at the same tolerance is 5 of the 18 flips the
gate needs. This is the cheapest open question in the whole track (§2, lever L1).

## 2. Levers not already falsified, ranked

Binding negatives honoured (not re-proposed): symmetry, SB node-prefix budget, node selection,
pseudocost scaling, in-tree cut aging, reliability gating, relaxing cut-cleanup gates, "more GMI",
RENS as measured, FJ on the no-incumbent trio, root restart, conflict analysis, cut repair by
dynamism/length, single-row implied bounds and probing (A9), `cut_select`/stall/orthogonality
variants (A1). Yields are stated per instance against P4.9; "plausible" means the mechanism is the
one the data names, not that it is measured to flip.

### L1 — Bisect the September→November per-node-cost regression on `gt2`/`fiber`/`kaihu`/`bppc8-02` (bound-neutral or a documented correctness cost)
- *Hypothesis:* one or two of the 2026-09-06…11-04 engine commits multiplied per-node cost on
  medium instances (≥ 300 rows) by 10–25× beyond the core-count change; part of it is recoverable
  without weakening any check (e.g. a per-node verification that re-factorizes, or a per-node
  allocation).
- *Evidence:* §1.4; the probe's `DualPrepare` 1.7 ms/node + `LuNumeric` 1.6 ms/node on `neos17`
  (a full LU per node), `Fbbt` 37 % of node time on `neos17` and 1.7× the LP time on `gt2`
  (`tighten_bounds_csc_with_row` rebuilds the CSR and sweeps **all** rows up to 8 rounds at every
  node, `presolve.rs:110-165`).
- *Instances:* `fiber`, `gt2`, `kaihu`, `bppc8-02`, `jijia` (certified in Sept); throughput on every
  A-row.
- *Entry experiment (≈ 1 h machine time, no new code):* `git bisect` between the Sept-6 commit of
  `rens_panel.json` and `dcdac02` with `maturin develop --release` per step and the test "`gt2` on
  the Rust route certifies in < 5 s" (`d2_probe.py gt2`, 7–8 builds). Then the same for `fiber`
  (23k nodes in 10 s in Sept). Report the commit and the phase table before/after.
- *Kill:* the bisect lands on a correctness fix whose cost is intrinsic (e.g. the #1945 re-audit
  needs the factorization) and no bound-neutral reformulation recovers ≥ 2× → record it, keep the
  fix, L1 contributes throughput only.
- *Expected yield:* 2–5 flips if the September behaviour is recoverable; otherwise 0 flips and a
  1.5–3× throughput gain that helps L3/L4.

### L2 — Density-gate the P4.4 completed-basis cuts; budget the root dive; cap strong-branching probes by LP iterations (bound-changing but mechanical; three flags, one panel)
- *Hypothesis:* population B is cost, not strength: with (a) completed-basis cuts rejected or kept
  root-only when `nnz > k·(matrix nnz/m)`, (b) the dive capped at a fraction of the slice (SCIP
  `maxlpiterquot` style, e.g. ≤ 20 % of root LP iterations × a factor) and (c) SB probes capped at
  2× the average node-LP iteration count (SCIP `maxlpiter`), the B instances run thousands of nodes
  in the slice instead of 3–49, keep their incumbent in the Rust engine instead of deferring to the
  Python tree, and the bound/incumbent values of P4.4 OFF are recovered or beaten.
- *Evidence:* §1.1 table B; repair OFF restores `fiber` 13 → 1,051, `beavma` 49 → 3,501, `neos17`
  3 → 9,771 (+ incumbent), `sp150x300d` 3 → 4,263 (+ incumbent), `nexp-50-20-4-2` 3 → 255
  (+ incumbent) in the same slice (`b2_cost.jsonl`); cut density 297/319/31 nnz per cut vs 8/2.7
  per matrix row; dive 50–97 % of the slice on 5 instances; SB 70 % on `nexp-50-20-4-2`.
- *Instances:* `fiber`, `beavma`, `neos17`, `sp150x300d`, `nexp-50-20-4-2`, `2122`, `issue-2446`,
  `neos-911970` (partly).
- *Entry experiment:* `p44_root_gap.py --all --profile` with the density gate: root gap must stay
  within 1 pp of HEAD on `fiber`/`beavma`/`b-ball`/`khb05250`/`neos-1425699` (the repair's wins)
  while `SepGomoryCutNnz`/cut drops ≥ 5×; dive and SB caps: node count in the 10 s slice on the 8 B
  instances ≥ 10× HEAD with incumbent present on ≥ 6 of 8.
- *Kill:* a density gate that keeps the bound wins cannot be found (the wins *are* the dense
  cuts) → keep the repair but move dense cuts out of the node LP (root bound only); dive/SB caps
  that lose an incumbent on any P4.9-certified row → retire.
- *Expected yield:* honest — **0–2 flips** (`2122` if an incumbent within 1e-4 appears; `neos17`,
  `sp150x300d` and `beavma` need HiGHS's root closure, which R does not have); the real value is
  that population B stops masking population A's levers and the Python-tree fallback stops eating
  half the budget on 5 instances.

### L3 — Switch the in-tree improvement heuristic on: re-panel `DISCOPT_RINS` at HEAD first, then improvement diving (bound-changing)
- **L3a, zero code: `DISCOPT_RINS=1`** (the A12 implementation, built September, cert-clean on
  four panels, "flipping the default is an owner call" — a call never taken; the flag has **no row
  in `flag-retirement-audit.md`**, which is the CLAUDE.md §5 defect). On the primal class today
  (`b5_rins.jsonl`, 20 s, 2 concurrent): **`gt2` certifies in 5.1 s (27,983 nodes), `iskar`
  certifies in 7.4 s (9,775 nodes)**, `kaihu` and `jijia` reach the optimum as incumbent (primal
  1e-15 / 1e-16; dual 0.017 / 0.019 still open), `bppc8-02` reaches the optimum (dual 0.040
  open); `istra` (0.08) and `obra` (0.28) unchanged. A12's September verdict (4–7 better
  incumbents, +6–46 % both-solved wall) predates P4.4 (which put `iskar`'s and `b-ball`'s bounds
  at the optimum) and the re-entry; **its measurement is stale and must be re-run as a P4-standard
  panel before anything else is built**. The cost side A12 measured (+6 % wall at cap 2) is the
  bar to pre-register.
- *L3b hypothesis:* the frozen primal is a design gate, not a missing heuristic family: scheduled
  dives (guided / pseudocost / fractional, SCIP's `heur_*diving` at freq 10) from node LP vertices,
  budgeted by LP iterations, with the `!has_incumbent` gate removed and the objective cutoff in the
  dive, find improving incumbents on the A1/A2 rows RINS does not reach (`istra`, `obra`, `2122`);
  where the bound is already at the optimum the gap then closes immediately.
- *Evidence:* A5 (median primal share 89.6 %), A14 (0 of 14 incumbents improved with 3× budget;
  root cause named as this gate), P4.1 (12/12 frozen), §1.2 (code), §1.3 (`gt2`'s incumbent is
  decided by the one root dive), A14's RENS entry (7/15 hit rate, 80 % median primal-gap reduction —
  sub-MIP neighbourhoods *are* cheap on this panel; the panel loss was a single-instance search
  interaction on `dcmulti`), A12's precondition (98.6 % of incumbent-present nodes in the RINS
  fixing band). Dives are the mechanism HiGHS's own node count is made of (§2.6 of the
  competitiveness plan).
- *Instances:* `gt2`, `iskar`, `2122`, `b-ball` (bound ready), `kaihu`, `jijia`, `istra`, `gesa2`,
  `bppc8-02`, `obra` (bound within 2 %), `mik-*` (already optimal incumbent — no effect).
- *Entry experiment (one Rust change, one flag):* `DISCOPT_MILP_DIVE_STRIDE=k` with the incumbent
  gate removed and a dive iteration cap; on the 10 instances above at 20 s, 2 reps: ≥ 4 instances
  strictly improve the incumbent, ≥ 2 reach the optimum. Pre-register the §5 bars; add the
  `dcmulti` certification as an explicit bar so the P4.2 failure mode is scored.
- *Kill:* < 4 improved or any certified row lost → retire the flag and its code.
- *Expected yield:* **3–6 flips** if the entry holds (A1 rows plus whichever A2 rows the tree then
  closes; `kaihu`/`jijia` certified in September with a *good* incumbent, which is the evidence that
  their trees close once the incumbent is right).

### L4 — Continue run 1's tree on re-entry instead of restarting it (bound-neutral in result, changes search order)
- *Hypothesis:* the slice/re-entry design (`_SIMPLEX_MILP_BUDGET_CAP_S` stall guard, then a
  **fresh** seeded tree) spends the first half of every budget on a tree the second half rebuilds;
  continuing the frontier through the P3.1 persistent `MilpHandle` doubles the effective tree.
- *Evidence:* `b3_primal.jsonl` run-2 ≈ run-1 node counts at 20 s (§1.2); at 60 s
  (`b4_60s.jsonl`) run 1 stops at 10.0 s and run 2 re-explores from the root for 49.7 s. I first
  read P4.1's `termination = node_limit` at 60 s on `jijia`/`istra`/`iskar` (with node counts equal
  to the 20 s ones) as the #1874 frontier cap firing; **retracted**: the equal counts are the
  adoption rule (`nodes` reports run 1's count whenever run 2 is not adopted), and in my own 60 s
  runs every run 2 ended `feasible` at its deadline, none `node_limit`. No frontier-cap exit was
  observed; the memory-bounded-search half of this lever has no evidence and is dropped.
- *Instances:* every A-row (`iskar` certified at 46.6 s in the probe's 60 s run: run 2 found the
  optimum and closed on a bound already at 1e-15 — so a continued tree with an incumbent source
  is what the 60 s budget buys).
- *Entry experiment:* re-entry through the handle on the 10 A1/A2 rows at 20 s and 60 s: nodes
  explored after the slice ≥ 1.8× HEAD's run 2, bound/incumbent never worse.
- *Kill:* < 1.5× or any row worse → not worth the plumbing.
- *Expected yield:* 0–1 flips at 20 s; it is what makes longer budgets count, which the owner's
  real users have.

### L5 — Tableau c-MIR / lifted cover for the `mik` family (deferred by P4.4, not falsified)
- *Hypothesis:* P4.4's own table: existing separators plateau at 3.8–4.2 % on `mik` (500-cut cap;
  2000 helps 0–0.4 pp) vs HiGHS 1.5–1.8 % with tableau c-MIR + lifted cover; the `mik` trees need
  that last 2 pp (HiGHS: 91–453 nodes; R: 60k nodes/20 s moving 1 pp).
- *Evidence:* `p4.4/root_*.json`; competitiveness plan §2.6a (HiGHS's tableau separator is c-MIR
  on tableau rows, no GMI at all); §2.5 (own cuts close 72–75 % of the `mik` root gap).
- *Instances:* the four `mik-250-20-75-*` (+ `gesa2`, `bppc8-02` plausibly).
- *Entry experiment:* P4.4's original — root residual on `mik` ≤ 2.5 % with tableau-row c-MIR
  (kill > 3.5 %); **plus** the tree question the plan never asked: give R a 1.7 % root (seed the
  HiGHS cuts or run R at 60–120 s and read the bound curve, `b4_60s.jsonl` `mik-1` row) and see
  whether its tree closes in the budget at all. If a 1.7 % root still needs > 10k R nodes, the lever
  is the tree, not the root, and L5 is dead.
- *Kill:* residual > 3.5 % or the tree does not close from a HiGHS-strength root.
- *Expected yield:* 0–4 flips, binary on the entry.

### L6 — Incremental node propagation + factor reuse (bound-neutral, throughput only)
- `tighten_bounds_csc_with_row` builds `row_nz` from the CSC and sweeps every row ≤ 8 rounds at
  every node; a dirty-row queue in the same row order with the same 8-round cap is bit-identical
  (an untouched row cannot change `lo`/`hi`) and skips most of the work; the CSR is constant within
  a batch. `DualPrepare` re-factorizes the inherited basis at every node (`LuNumeric` once per node).
- *Evidence:* `Fbbt` 0.53 ms vs `NodeLpSolve` 0.30 ms per node on `gt2`; 37 % on `neos17`.
- *Gate:* exact node_count/objective identity on the certifying panel (CLAUDE.md §5 bound-neutral).
- *Yield:* 1.3–1.7× nodes/s on small-row instances; **0 flips** by itself.

### L7 — Dimension-reducing presolve (P4.8, deferred) — stays last
HiGHS's own ablation (`milp-competitiveness-plan.md` §1c): ≤ 2 instances / 1.7×; root gap ~0–3 pp
(§2.5a). Largest effort in the track. Only after L1–L5.

### Does the sum reach 34/38?

No, not on the evidence. Measured today: L3a (RINS) +2 certified (`gt2`, `iskar`) and 3 rows
(`kaihu`, `jijia`, `bppc8-02`) left with the optimum in hand and only the bound open — those
three close only if their trees close (`kaihu`/`jijia` did certify in September at 11–13 s / 5 s,
so L1's throughput recovery plus the optimum incumbent plausibly buys them; `bppc8-02`'s bound is
stalled at 60 s). Plausible per-lever yields beyond that: L1 1–3 more (`fiber`, `khb05250`-class
throughput), L3b 1–3 (`istra`, `obra`, `2122`), L5 0–4 (`mik`), L2 0–2, L4 0–1. The honest range
is **+4 to +12 MIPLIB flips → 20–28/38**, against 34 required. What stays out of reach on this
evidence: the four `mik` without a HiGHS-strength root (L5's entry decides), `enlight_hard` and
`amur` (no incumbent; FJ and the dive both find nothing), `nexp-50-20-1-1`/`sp150x300d`/`beavma`
(HiGHS closes them at the root with cuts + restart + integral-objective rounding; R's root is
27–30 % short after repair), `gsvm2rl3` (HiGHS needs 2,547 nodes with 15k SB iterations and
conflicts), `neos-911970` and `issue-2446` (LP-engine cost: dual-simplex stalls, 120 µs/pivot).
The per-node *strength* gap on the A-population (31× geomean nodes, 100–2,000× on the failing
rows) is the product of many HiGHS mechanisms that A4–A11/B1 each measured individually at
≤ 1.43×. On the M panel L3a + L2 put `gt2`, `2122`, `gesa2` in reach (≤ 21–22/24; 23 required).

## 3. The narrowest acceptable end state if full closure is out of reach

### 3.1 A static class predicate does not exist in this data

Certified-vs-failed is not separable by size or integrality: `gt2` (188×29, 100 % integer) and
`b-ball` (100×30) fail while `app2-2` (1226×335, 100 % integer), `f2gap40400` (400×40) and
`p0201` (201×133) certify at the root; `p0548` (548×176) fails while `dcmulti` (548×290) certifies;
`khb05250` (1350×101) certifies while `nexp-50-20-1-1` (490×540) fails. What separates them is
whether the root (bound + first incumbent) closes the instance or the tree is small (≤ ~10k R
nodes) — a property of the instance's relaxation, visible only by running it. A predicate on
`(n, m, nnz, n_int, n_int/n)` keyed to this panel would be exactly the named-instance special case
CLAUDE.md §2 forbids.

### 3.2 What can be defined from measurable behaviour: the engine's own first slice

The route already runs the Rust engine for a bounded slice (`_SIMPLEX_MILP_BUDGET_CAP_S`, ≤ 10 s
or half the remainder) and only then decides. On P4.9 every one of R's 16 + 18 certifications
completes inside that first slice except `khb05250` (9.4 s, inside a 10 s slice on an idle host) —
and on those rows R's total wall (34.2 s over the 16 MIPLIB both-certified rows) is **1.07×** H's
(32.1 s). So the measurable class is "**certified by the Rust engine within
`min(0.5·budget, 10 s)`**", and the narrowest end state with HiGHS still installed is:

- `DISCOPT_LP_MILP_BACKEND=auto` (new default): Rust first slice as today; on `optimal` → done
  (Rust certificate, no `highspy` touched); on anything else → hand the *remaining* budget to the
  HiGHS route when `highspy` imports, else continue with Rust re-entry as today. Cost: on the 24
  rows H certifies and R does not, H starts ≤ 10 s late. At 20 s that costs certifications (H needs
  8–13.6 s on `mik-5`, `mik-3`, `mik-1`, `23588`, `enlight_hard`, `neos17`, `istra`, `iskar`,
  `gsvm2rl3`, `issue-2446`); at the shipped default budget it costs ≤ 10 s of wall and nothing
  else. The slice can be made proportional (e.g. 10 % of the budget, floor 2 s) so the 20 s panel
  loses ≤ 2 s per H row; whether that still keeps the 34 Rust certifications is the entry
  experiment (`lp_milp_panel.py run --arms auto,H`, 3 rounds, bars: certified ≥ H's count, total
  wall ≤ 1.3× H, 0 incorrect).
- `pyproject.toml`: `highspy` moves to the `highs` extra; the no-highspy lane (P6.2) runs Rust-only
  and **documents** the measured loss (16/38 at 20 s on this panel, no incorrect). This meets the
  *packaging* goal (HiGHS optional) and M-1671's close condition on the default path only if the
  HiGHS leg keeps the sparsify-off workaround — it does not remove HiGHS from the default behaviour
  of an install that has the extra.

This is CLAUDE.md §5 state 3 for the HiGHS route ("kept as a documented opt-out", with the
docstring saying what would change it: the P4.9 gate). It is the end state the plan's D2 row calls
"HiGHS optional for R1 by class", with the class defined by the engine's own certification inside
its first slice rather than by a model feature — the only definition the data support.

### 3.3 What dropping HiGHS outright costs today

With `highspy` absent the default MILP route is Rust-only: 16/38 MIPLIB and 18/24 M at 20 s
(P4.9), 0 incorrect, INF 16/16. At 60 s the picture barely moves (P4.1: one more, `khb05250`),
because the losses are frozen incumbents and 2–4 % residual root gaps, not budget. Any user with a
`gt2`- or `mik`-shaped MILP gets an uncertified `feasible` where HiGHS returned `optimal` in
under a second. That is the number the owner is accepting if D2 is closed as "drop".

## 4. Recommended sequence and effort

0. **L3a: P4-standard panel for `DISCOPT_RINS` at HEAD (half a day, no code).** MIPLIB-38 + M,
   ON vs OFF, 3 reps, 20 s, bars pre-registered: cert-clean; certified ON ≥ OFF + 2; no incumbent
   worse; both-certified wall ≤ 1.10× (A12's own bar). Graduate on pass and add the missing audit
   row either way. Today's single-rep entry: +2 certified, +3 optimum incumbents, 0 worse.
1. **L1 bisect (1–2 days, mostly machine time).** It is the only lever with a known-good endpoint
   (the September tree) and it decides whether population B's cost and the A-population's
   throughput are regressions or intrinsic. Deliverable: the commit(s), the phase table before/after
   on `gt2`/`fiber`/`neos17`, and either a bound-neutral recovery PR or a §7 entry stating the cost
   is the price of a correctness fix (with the number).
2. **L2 density gate + dive/SB iteration caps (2–3 days).** Three flags, one P4-standard panel
   (MIPLIB-38 + M, 3 reps, 20 s) with two *added* bars: node count in the first slice on population
   B ≥ 10× HEAD, and no incumbent lost on any row (the bar P4.4's panel lacked). Graduate or retire
   each separately.
3. **L3b improvement diving (1 week).** For the A1/A2 rows RINS leaves (`istra`, `obra`, `2122`
   after L2). Entry on those rows first; the `dcmulti`-style interaction is a pre-registered bar.
   If it fails its entry, the primal side is closed for this engine and the track should stop at
   §3.2.
4. **L4 memory-bounded search + handle continuation (2–3 days).** Needed for any budget above 20 s
   to mean anything; measure after L3 so the frozen-primal rows do not mask it.
5. **L5 tableau c-MIR, entry-first (3–5 days if the entry holds).** Run the "tree from a 1.7 %
   root" half of the entry *before* writing a separator.
6. **Re-run P4.9.** If < 34/38, close D2 as §3.2 (`auto` backend, `highspy` to an extra, documented
   loss), not as "keep HiGHS the default and stop" — the packaging goal is reachable now, the
   performance goal is not, and conflating them is what has kept Phase 6 blocked.

L6 (incremental propagation) is a bound-neutral PR that can go in at any point under the exact-
identity gate; L7 stays deferred.

## 5. Remaining probe results (`b3_primal.jsonl`, `b4_60s.jsonl`, `b5_rins.jsonl`)

- **Off-root dives while no incumbent** (`DISCOPT_MILP_DIVE_STRIDE=20`, the shipped opt-in, 20 s):
  `enlight_hard`, `neos-2624317-amur`, `neos17`, `sp150x300d`, `issue-2446` still find nothing;
  `p0548` finds a point 219 % off and `nexp-50-20-4-2` one 21 % off (both then stay in the Rust
  engine instead of deferring). Find-any-point on these rows is hard for the dive as it is for FJ
  (P4.6); L3 is about *improvement* dives on rows that already hold a point, not this.
- **Root cuts slow the tree too**: `enlight_hard` with `DISCOPT_MILP_ROOT_CUTS=0` runs 28,637
  nodes in its 6.5 s slice vs 5,943 with the 500-cut root (same bound class, 0.37 vs 0.29) —
  the cut rows stay in every node LP for the whole search (no pool/aging; A2 measured aging as
  not a lever on nodes, but did not score per-node cost).
- **60 s** (2 concurrent): `iskar` **certifies** (run 2 finds the optimum, 46.6 s); `jijia`
  0.157 / 0.0093 unchanged with 143k nodes; `istra` 0.08 / 0.02 unchanged; `mik-1`, `b-ball`,
  `bppc8-02` bounds and incumbents unchanged (§1.1 A3). The budget does not buy the A-rows.
- **RINS ON** (`DISCOPT_RINS=1`, the A12 implementation, 20 s, 2 concurrent, 12 rows,
  `b5_rins.jsonl`), against P4.9 HEAD:

| instance | HEAD (P4.9) primal / dual | RINS=1 primal / dual | status |
|---|---|---|---|
| gt2 | 0.139 / 1e-12 | 1e-14 / 1e-12 | **optimal, 5.1 s, 27,983 nodes** |
| neos-3610040-iskar | 0.105 / 1e-15 | 0 / 3e-15 | **optimal, 7.4 s, 9,775 nodes** |
| neos-3611689-kaihu | 0.183 / 0.0083 | **1e-15** / 0.0167 | feasible (bound open) |
| neos-3611447-jijia | 0.157 / 0.0093 | **1e-16** / 0.0185 | feasible (bound open) |
| bppc8-02 | 0.018 / 0.040 | **3e-16** / 0.040 | feasible (bound stalled) |
| gesa2 (M) | 0.027 / 0.0046 | 0.027 / 0.0048 | unchanged |
| neos-3610051-istra | 0.08 / 0.02 | 0.08 / 0.02 | unchanged |
| neos-3118745-obra | 0.28 / 0.0039 | 0.28 / 0.0039 | unchanged |
| gsvm2rl3 | 0.187 / 0.11 | 0.187 / 0.12 | unchanged |
| mik-250-20-75-1 | 0 / 0.032 | 0 / 0.033 | unchanged (already optimal incumbent) |
| 2122 (M) | 1.05 / 4.7e-5 | 1.05 / 4.7e-5 | unchanged (3 nodes: population B) |
| beavma | 0.254 / 0.141 | 0.254 / 0.161 | unchanged (27 nodes: population B) |

  Two certifications and three optimum-incumbents from a flag that exists, was cert-clean on four
  September panels, and was never decided on. Single rep, loaded host (2 concurrent), so this is an
  *entry* result, not a graduation; the P4-standard panel (3 reps, interleaved, cert-clean +
  net-positive with the A12 wall bar) is the next step and needs no code.


---

## Follow-up measurements (2026-11-06, after the analysis; data `docs/dev/data/highs-parity/d2-analysis/`)

**1. The "September 21/38 vs November 14/38" comparison mixed two entry points -- retracted
as an engine regression.** The September A14 panel (`scratchpad/a14/rens_panel.py`) called
the Rust driver DIRECTLY (`solve_milp_py`, whole 20 s budget, `root_cuts=500,
cut_rounds=50, cut_select, root_cut_time_s=10`); P0.3/P4.9/P6.6 measured the shipped ROUTE
(`Model.solve` -> `_solve_milp_simplex`: a `min(0.5 rem, 10 s)` first slice, then re-entry or
deferral). Today's driver called the September way
(`discopt_benchmarks/scripts/d2_direct_driver_panel.py`, `direct_driver_head.json`):
**17/38, 0 incorrect.** So:
- the route wrapper costs ~3 certifications at 20 s (`neos-1425699`, `khb05250`, `jijia`
  certify direct, not via the route);
- the remaining September -> HEAD difference on direct calls is ~4 (`gt2`, `fiber`, `kaihu`,
  `bppc8-02`); on this 4-core host `kaihu` and `fiber` certify in NEITHER build, so only
  `gt2` and `bppc8-02` reproduce.

**2. Per-node cost did not regress.** `Model.solve` at a 20,000-node cap, September build
(`c55bb170`, own worktree + venv, `discopt.__file__` asserted) vs HEAD: nodes/s HEAD/Sept =
`gt2` 3.8x, `fiber` 1.7x, `beavma` 2.0x, `sp150x300d` 1.9x -- HEAD is faster per node.

**3. The `gt2` "regression" bisects to a 1-ulp butterfly, not a lost capability.**
`git bisect` over the 140 `crates/` commits (predicate: `gt2` certifies within 20 s, direct
call) converges on `2ee4fc7d` ("Rust LP presolve FBBT encloses the whole activity-sum
error", a #1955-class soundness fix). With presolve OFF both sides fail identically; with it
ON the two root boxes differ only by ONE ULP on 9 continuous columns (outward rounding,
e.g. `3.000000000000009` vs `3.0000000000000093`), and that alone moves `gt2` from
certified-in-13,893-nodes to open-after-200,000 with the incumbent frozen at 31,661
(optimum 21,166). Nothing to revert: the fix is correct, and the search is chaotic in the
root vertex. This **falsifies lever L1 (bisect) for `gt2`** and sharpens the diagnosis: the
engine's primal side cannot recover from an unlucky trajectory (no improvement heuristic
runs once an incumbent exists; RINS is off). Lever L3a/L3b (RINS, improvement diving) is
the one that addresses this class.

**4. RETRACTION of follow-up 1's "the route wrapper costs ~3 certifications".** On
`khb05250` (65 nodes / 0.44 s direct, open via the route) the route's own arrays, replayed
through the same driver with the direct options and the full 20 s
(`d2-analysis/route_spy.py`), still do not certify (2107 nodes, open). The difference is the
formulation, not the wrapper: the direct path (`_ranged_to_ub_eq` +
`_marshal_std_form`) writes each of the 77 equality rows as two inequalities (m = 178,
1528 columns), the route keeps them as equalities (m = 101, 1374 columns). Same problem,
different LP bases and vertices, opposite outcomes -- the `gt2` 1-ulp effect again, from an
equivalent reformulation. **Consequence for D2:** the engine's certification count on this
panel is variance-dominated (trajectory-sensitive); levers have to be judged on several
equivalent perturbations or reps, and the levers that can move it robustly are primal
recovery (RINS, improvement heuristics) and per-node bound strength, not reverts or
re-tuning one path.

**5. Lever L3a (`DISCOPT_RINS=1`) panel: bars missed, decision deferred to the HiGHS-expert
review.** `d2_flag_panel.py --flag DISCOPT_RINS --fired RinsRun`, MIPLIB-38 + M, 3 reps,
20 s, 372 solves, fired ON 632 / OFF 0 (`d2-analysis/rins_panel.json`). Cert-clean (372
checks); **no incumbent lost**; median primal gap over the open population 0.214 -> 0.187
(`gt2`, `iskar` to the optimum; `jijia`, `neos-911970`, `nexp-50-20-1-1`, `fiber`, `bppc8-02`
better); certified **96 = 96** -- `gt2` 0/6 -> 6/6 but `dcmulti` 6/6 -> 0/6 (ON ends with a
WORSE incumbent, primal gap 0 -> 0.12, the RENS P4.2 pattern again); bar 2b (nodes -15 %
or +2 certified) fails. Median 1-min load 7.2 on 4 cores (two analysis agents ran
alongside): arms were interleaved under the same load, but `dcmulti` certifies near the
20 s line, so the loss may be partly budget. Not retired yet: the RINS scheduling/budget
question goes to the HiGHS-expert review first.

**6. Expert R1 entry checks (2026-11-06).**

*(a) Where HiGHS's incumbents come from.* Read from the ablation logs; see
`highs-parity-d2-highs-ablation.md` §C.5. With HiGHS defaults the final incumbent comes from
a sub-MIP (`L`) on 20/25 of the rows Rust fails. With every heuristic off, HiGHS still
certifies 25/25, with final incumbents from tree-node LPs (`T`/`B` 17, `S` 7). The expert
review's letter key read `S` as RINS. The source says `S` is "Solve LP" and `L` is sub-MIP;
this is corrected in the review. **Consequence for RINS:** a sub-MIP is not what certifies
these rows, so L3a cannot close D2. The primal lever to test is dives from tree nodes.

*(b) What an objective cutoff in the dual simplex could save.* Probe
`d2-analysis/r1_cutoff_entry.{py,jsonl}`, profile counters `NodeLp*`, direct driver, 20 s,
20 of the 25 rows. It measures the share of node-LP pivots spent on `Optimal` nodes whose bound
reached the dispatch-time incumbent; a cutoff can save at most a fraction of those.

- Pooled share: **8.2 %**.
- Zero on 13/20 rows: all four `mik`, `b-ball`, `iskar`, `kaihu`, `nexp-*`, `gsvm2rl3`,
  `sp150x300d`, `beavma`, `neos-911970`. On these the incumbent sits far from every node
  bound.
- High only where the incumbent is already close: `bppc8-02` 40 %, `fiber` 22 %, `jijia`
  13 %, `istra` 2 %.

**Verdict:** the cutoff is not a standalone lever on the failure set. It pays only after
the primal side delivers good incumbents, so it moves behind the plunge/dive prototype in
the build order. The `NodeLp*` counters stay: they are profile-gated and bound-neutral.

*(c) Improving-dive prototype entry (`DISCOPT_MILP_IMPROVE_DIVE=1`, no effort cap).* Setup:
`d2_flag_panel.py`, the 10 A1/A2 rows, 1 rep, 20 s, load 5-7, data
`d2-analysis/r1_improve_dive_entry.json`. Cert-clean (22 checks); no incumbent lost.

- **Better incumbent on 7/10 rows:**
  - `istra` 0.120 -> 0, `iskar` 0.132 -> 0 (the optimum; the dual gap of 2-3 % remains);
  - `gt2` 0.139 -> 0.028, and its second panel copy certifies;
  - `jijia` 0.241 -> 0.065, `kaihu` 0.217 -> 0.075, `obra` 0.215 -> 0.137,
    `gesa2` 0.058 -> 0.046.
- **Worse:** `bppc8-02` 0.012 -> 0.022.
- **Unchanged:** `b-ball`; `2122` has no incumbent in either arm.
- Median primal gap 0.135 -> 0.037.

**The expert's kill criterion (< 4/10 improve) is not met: the primal half of R1 is
confirmed.** The cost is throughput where dives are long: `gesa2` 10,975 -> 2,431 nodes,
`obra` 15,231 -> 7,188. So the graduation candidate carries an effort cap,
`IMPROVE_DIVE_EFFORT = 0.1` of the tree's LP pivots (HiGHS budgets its heuristics as a
share of LP iterations). That panel follows.

**7. `DISCOPT_MILP_IMPROVE_DIVE=1` §5 panel (effort cap 0.1): cert-clean, primal gap halved,
certified count unchanged -- bar 2b missed, does not graduate.**

Setup: `d2_flag_panel.py --flag DISCOPT_MILP_IMPROVE_DIVE --fired ImproveDiveRuns`,
MIPLIB-38 + M, 3 reps, 20 s, 372 solves, median 1-min load 6.4 on 4 cores, arms
interleaved. Data: `d2-analysis/r1_improve_dive_panel.json.gz`. Code under test: the Rust
of `0facf021`. The panel header names `e9e5d1b9` because the build preceded the commit;
the extension was rebuilt from the effort-cap source and checked for the flag string
before launch.

Results:

- **BAR 1:** cert-clean, 372 checks.
- **2a:** certified 96 = 96, with no instance changing its certified count.
- **2b fails:** nodes on the 30 always-certified instances are 148,395 OFF vs 151,167 ON
  (+1.9 %), against a bar of -15 % or +2 certified.
- **2c:** 0 dual losses (cap 2).
- **2d:** no incumbent lost.
- **2e:** median primal gap over the 28 open instances **0.214 -> 0.100**.

Primal per instance (median of 3 reps):

- **Better on 13 rows:**
  - `istra` 0.120 -> 0 and `iskar` 0.132 -> 0 (the optimum);
  - `kaihu` 0.217 -> 0.025, `jijia` 0.241 -> 0.065, `obra` 0.215 -> 0.086;
  - `gt2` 0.139 -> 0.056, `neos17` 0.214 -> 0.114, `nexp-50-20-1-1` 0.633 -> 0.333;
  - `fiber` 0.548 -> 0.301, `neos-911970` 3.08 -> 1.31, `khb05250` 0.044 -> 0.013;
  - `gesa2` 0.058 -> 0.027, `sp150x300d` 0.343 -> 0.314.
- **Worse on 1:** `bppc8-02` 0.012 -> 0.014.

**Reading.** The primal half of the expert's R1 diagnosis is confirmed at panel scale:
tree-node dives with propagation and a cutoff recover incumbents the batched best-first
search never finds. But on the target rows the remaining gap is now **dual**.

- `istra` and `iskar` hold the optimum and stay open at a 2-3 % dual gap: the bound is one
  objective unit short, e.g. 48 vs 49.
- `kaihu` and `jijia` are within 2-7 % primal with a 3 % dual gap.
- With the effort cap, `gt2` no longer certifies (the uncapped entry certified one copy).

What closes the dual gap on these rows is what the HiGHS ablation names: cuts (C.1, 23/25
alone) and presolve/restart (C.2). So the dive is necessary for D2 but not sufficient. Its
certification value can only show once the dual side moves.

**Decision pending (owner):** either retire the flag now, or keep it default-OFF owned by
D2 and re-panel it together with Phase B (tableau cuts with node separation), with the
same pre-registered bars. The bars are not changed after the fact.

**8. Owner decisions (2026-11-06).**

- `DISCOPT_MILP_IMPROVE_DIVE` stays default-OFF, owned by D2. It is re-panelled together
  with Phase B (tableau cuts with node separation and aging), with the same pre-registered
  bars.
- **`DISCOPT_RINS` is retired** (flag and implementation; see the audit row).
- Next build is Phase B, with a HiGHS-expert design consult first.

**9. `DISCOPT_MILP_REENTRY_CUTS=1` §5 panel: does not graduate (bar 2e).**

Setup: `d2_flag_panel.py --flag DISCOPT_MILP_REENTRY_CUTS --fired RootCutRounds`,
MIPLIB-38 + M, 3 reps, 20 s, 372 solves, load ~6.5. Data:
`d2-analysis/reentry_cuts_panel.json.gz`. The fired counter runs in both arms, because
run 1 cuts in both, so firing reads as ON > OFF: 1389 vs 1143. The script's exit check
(OFF must be 0) does not apply to this flag.

Results:

- **BAR 1:** cert-clean, 372 checks.
- **2a, 2b:** certified **96 -> 100**: `gt2` 0/6 -> 6/6, `enlight8` 3/3 -> 1/3.
- **2c:** one dual loss, `enlight8`, within the cap of 2.
- **2d:** no incumbent lost.
- **2e fails:** median primal gap 0.214 -> 0.248. The four `mik-250-20-75-*` rows go from
  a primal gap of 0.0000-0.0026 to 0.25-0.89.

**Reading.** P4.0 measured that the `mik` incumbents come from run 2's search. ON spends up
to half of run 2's time (`root_cut_time_s = 0.5 * remaining`) re-deriving the root cuts run
1 already found, and that displaces the search. The `gt2` gain says a strong root matters
in run 2. The `mik` loss says re-separating is the wrong way to get it. The fix the
evidence points to is that **run 2 inherits run 1's root cut pool** (rows appended to its
LP, no re-separation), as the Phase B review suggested. That is a re-scope, not a tuning
of this flag. Owner decision: retire `DISCOPT_MILP_REENTRY_CUTS` now and build pool
inheritance under a new flag, or keep it pending that build.

**10. Phase B entry B0 (no code): in-tree separation is the HiGHS lever; the Rust root is
not the gap.**

*HiGHS* (`d2-ablation/ablation_phase4_b0.jsonl`, the 10 rows `keep_cuts` rescues over
plain B&B, 20 s, threads=1):

- `keep_cuts` minus node separation (`mip_allow_cut_separation_at_nodes=false`) certifies
  **3/10**, against 10/10 for `keep_cuts`.
  - Lost: all four `mik`, `iskar`, `istra`, `b-ball`.
  - `kaihu` still certifies, but with 21x the nodes.
  - Root bounds are identical in both arms, so what certifies these rows is cuts found
    in the tree.
- `keep_cuts` minus aging (LP/pool age limits at their maximum) certifies 7/10. It loses
  `b-ball`, `mik-1` and `mik-3`, with dual gaps < 0.8 %.
- The Phase B review's kill criterion for "node separation first" (no-nodesep certifies
  >= 8/10) is not met. **B1 goes first.**

*Rust* (`d2-analysis/b0_rust_cuts_entry.{py,jsonl}`, direct driver, A14 options,
`{cmir_cuts, node_cuts}` 2x2, root-only and 20 s):

- **The Rust root matches or beats HiGHS's root with cuts.**

  | row | Rust root | HiGHS root |
  |---|---|---|
  | `istra` | 46.57 | 45.99 |
  | `iskar` | 34.65 | 34.54 |
  | `mik-1` | -51,757 | -52,241 |
  | `sp150x300d` | 55 | 4.89 |

- **The existing node separator (`node_cuts`, knapsack cover) is identical to `base` on
  all 10 rows.** It finds nothing there.
- **Root c-MIR (`cmir_cuts`)** moves the root a little (`kaihu` 110.4 -> 112.2,
  `issue-2446` -339 -> -415), but the 20 s outcome is mixed: `istra` gets worse, `mik-1`
  gets a better incumbent.

*Lattice check.* On the four `neos-361*` rows the objective is one free continuous column
defined by a single equality, `z = 24 + sum(binaries)` on `istra`. The default-ON
`DISCOPT_OBJ_LATTICE_SUBST` resolves it on the route: the panel bounds sit at exactly 48,
36, 115 and 104. So the route needs the bound to cross to the next integer (`istra` 48 ->
49), which HiGHS gets in the tree.

The direct probe writes equality rows as inequality pairs (follow-up 4), so the
substitution declines there (`ObjLatticeSubst` = 0; `istra` bound 48.31 unrounded). This
is a probe artifact, not a route defect. For lattice rows, panels must use the route.

**11. B1 entry (`DISCOPT_MILP_NODE_SEP`, original-row c-MIR at nodes): at the kill line.
SCIP shows the lever is in-tree Gomory (tableau) cuts.**

*B1 entry* (`d2-analysis/b1_node_sep_entry.json`; the 10 B0 rows, 2 reps, route, 20 s,
load ~7). Cert-clean, 44 solves.

- `kaihu` crosses one integer in both reps (115 -> 116; optimum 119).
- `sp150x300d` 39 -> 40 on two of three runs.
- `istra` and `iskar` stay at 48 and 36; `mik` moves at noise level; `b-ball` finds no
  cuts.
- Node throughput is -25 % to -75 %.

The route never passes `max_pool_cuts`, so node cuts are capped at **128 for the whole
solve** (`node_cut_cap = min(2m, 128)`). A scratch build without that cap folded 538-1,148
cuts (`b1_node_sep_cap8_scratch.json`, new counter `NodeCutsFolded`). Result: `istra` and
`iskar` unchanged, `kaihu` the same +1, `sp150x300d` 39 -> 42, `mik-1` slightly better,
and 2-6x fewer nodes. **The cap is not what limits the target rows: original-row c-MIR is
the wrong family for them.**

*Which family certifies them?* SCIP 10 via pyscipopt 6.3, presolve off, 30 s, on `iskar`
and `istra`. Data: `b1_scip_separators.jsonl` (default run, applied cuts per separator) and
`b1_scip_gomory_ablation.log` (from `b1_scip_loo.py` and `b1_scip_pairs.py`).

- **Leave one out** from all-on: dropping **`gomory`** is the only change that stops
  certification on both rows (35.97 / 47.98). Dropping `aggregation` costs ~3x the nodes
  but still certifies. Dropping `zerohalf`, `impliedbounds`, `clique`, `intobj`,
  `rapidlearning`, `cmir`, `flowcover`, `mcf`, `knapsackcover`, `strongcg`, `oddcycle` or
  conflict analysis changes nothing.
- **Where Gomory runs:**
  - root only: `iskar` stalls at 36; `istra` needs 11,440 nodes;
  - tree only: both certify in ~1,600 nodes;
  - `gomory`+`gomorymi` alone: both certify;
  - `gomory` with any one of `aggregation`, `intobj`, `zerohalf` or `strongcg`+`cmir`
    (all else off): stalls at the no-cut bound.
- **Aggregation-only with presolve off is the Rust situation**, and it stalls where Rust
  stalls (36 / 48).

This agrees with HiGHS, whose node separation includes its tableau separator: without it,
HiGHS loses both rows (follow-up 10).

**Consequence for Phase B.** The lever on the `neos-361*` class is **Gomory cuts from the
node's own tableau**, applied in rounds at the node. SCIP marks them local (node bounds),
so they do not belong in the shared global pool. The Phase B review's "GMI never goes to
nodes" holds only for global cuts.

The batched tree can take them as **node-local separation rounds**: separate GMI from the
node basis with the node's bounds, append the rows to a node-local copy of the LP,
re-solve warm, and repeat for 2-3 rounds. The node keeps the stronger (NS-verified) bound,
the children inherit it as their floor, and the rows are dropped with the node. The shared
matrix and the other nodes of the batch are untouched. That is the next build (B1').

**12. Owner decisions (2026-11-06, second round).**

- **Next build:** node-local GMI rounds (B1').
- **`DISCOPT_MILP_REENTRY_CUTS`:** kept default-OFF, re-scoped so that run 2 inherits run
  1's root cuts instead of re-separating.
- **`DISCOPT_MILP_NODE_SEP`:** retired, flag and implementation.
