# D2 — which HiGHS MIP mechanisms certify the MIPLIB/M instances discopt's Rust MILP engine fails (option-level ablation of HiGHS 1.15.1)

Date 2026-11-06. Tree `/home/user/discopt` @ `c41854e3` (nothing in the tree was edited). HiGHS
`highspy 1.15.1` (`Highs.version()` = `1.15.1`, git hash `04024d7`, recorded on every row). Host: 4
cores; a 2-core panel (`d2_flag_panel.py --jobs 2`) was running until ~16:26, so **phase 1 ran at
1-min load 6.5–8.7 with one solve at a time, phases 2–3 at load 2–3 with two solves at a time**.
Walls are therefore indicative; node counts, bounds, incumbents and certification are what the
analysis uses, with one exception called out below (`gsvm2rl3`). Single rep per cell.

Deliverables in this directory (`/tmp/claude-0/-home-user-discopt/38303ad9-c9f4-5a5b-9f75-f33b92707a61/scratchpad/`):
`d2_highs_ablation.py` (the experiment; fresh `python -I` process per solve, executed-solve count
printed, exit 2 on zero, no exception swallowed — a worker crash would be an `error` row and is
counted separately; 0 occurred), raw rows `d2_highs_ablation_phase{1,2,3}.jsonl` (689 solves) and
`d2_highs_ablation_base40.jsonl` (10), `d2_lp_bounds.py` / `.json` (25 LP relaxations),
`d2_highs_ablation_summary.py` / `d2_highs_ablation_tables.py` (this document's tables),
`d2_highs_ablation_p12.summary.txt` (the full per-instance matrix with LP-iteration splits and
incumbent-source letters), `run_phase3.sh`.

## 0. Summary (the answer)

**Instances.** From `p6.6/miplib.json` (R `gap_certified` false, H true): 21 MIPLIB instances; from
`p6.6/lmi.json` panel M: 6, of which `gt2` and `sp150x300d` are also in MIPLIB → **25 distinct
instances** (Rust certifies 14/38 MIPLIB and 18/24 M; HiGHS 35/38 and 24/24). Same arrays as
discopt's routes (`lp_milp_loader.read_instance`), `mip_rel_gap 1e-4`, 20 s.

**`threads=1` changes nothing** (Table 1, `base_tdef` vs `base`: identical status and nodes on
24/25; `gsvm2rl3` fails both ways at 20 s under load 7.5 and certifies at 21.5 s / 5 817 nodes
in the 40 s re-run at load 2.2 — Table 5 — so it is a *budget-at-load* miss, not an ablation
result; it is excluded from the critical counts below and marked ‡ where it appears).

**Leave-one-out (Table 1): HiGHS is robust to losing any single component.** Removing one
component makes HiGHS *fail* on only two instances — the **tableau cut separator**
(`cuts_throttle`, and in phase 3 the single switch `pool_soft_1` that gates
`HighsTableauSeparator`) on `b-ball` and `neos-911970`, and **node separation**
(`node_cuts_off`) on `neos-911970`. Everything else shows up as node-count blow-ups (≥ 10×):

| component removed | instances that FAIL | instances with ≥ 10× nodes (still certified) |
|---|---|---|
| primal heuristics (FJ + RINS + RENS + root-reduced-cost + effort 0) | 0 | **11**: 2122 (447×), bppc8-02 (219×), gesa2 (81×), gt2 (109×), khb05250 (32×), mik-2 (16×), mik-5 (35×), nexp-50-20-1-1 (119×), nexp-50-20-4-2 (20×), p0548 (62×), sp150x300d (12×) |
| presolve (all MIP presolve incl. probing/cliques, and with it restarts) | 0 | **6**: b-ball (65×), bppc8-02 (247×), gesa2 (20×), issue-2446 (567×), mik-5 (15×), neos-3610040-iskar (13×) |
| node cut separation | 1 (neos-911970) | 3: b-ball (11×), mik-1 (10×), mik-5 (14×) |
| root cuts throttled (tableau separator off, aging 0) | 2 (b-ball, neos-911970) | 1: bppc8-02 (27×) |
| restart | 0 | 2: beavma (120×), bppc8-02 (78×) |
| strong branching (`mip_pscost_minreliable=0`) | 0 | 1: fiber (15×) |
| symmetry, probing alone | 0 | 0 |

**Leave-one-in (Table 2) is the decisive measurement.** Plain HiGHS B&B with every switchable
component off (`all_off`) certifies **13/25** of the Rust failures — and on the 12 it fails it
looks like the Rust engine (the four `mik` stuck at a 1.0–2.0 % dual gap after 26–28 k nodes,
`iskar` 5.3 %, `istra` 4.0 %, `kaihu` 3.2 %, `b-ball` 14.9 %, `sp150x300d` 10 %). Adding ONE
component back:

| plain B&B + | certified / 25 | rescues (of the 12 `all_off` failures) |
|---|---|---|
| **cuts** (tableau separator + node separation + aging) | **23** | b-ball, issue-2446, mik ×4, iskar, istra, kaihu, sp150x300d (10 of 12; not neos-911970‡, gsvm2rl3‡) |
| **presolve + restart** | **22** | issue-2446, mik ×4, iskar, istra, kaihu, sp150x300d (9 of 12; not b-ball, neos-911970, gsvm2rl3‡) |
| strong branching | 15 | mik-2, kaihu |
| heuristics | 14 | kaihu |
| cuts + heuristics | 24 | all but gsvm2rl3‡ |
| cuts + presolve | 24 | all but gsvm2rl3‡ |
| cuts + strong branching | 23 | all but mik-1, istra |
| everything (`base`) | 24 (25 at 40 s) | |

**Ranking of what discopt has to build, by the Rust failures each component accounts for:**

1. **Cut separation of HiGHS's kind — lifted-cover / c-MIR on aggregated *tableau* rows
   (`HighsTableauSeparator` → `HighsCutGeneration`), separated again at tree nodes, with LP/pool
   aging.** Sufficient on its own for 23/25 (10 of the 12 plain-B&B failures); necessary for
   `b-ball` and `neos-911970` outright and for 10–27× node counts on `bppc8-02`, `mik-1`, `mik-5`.
   Phase 3 pins it to the tableau separator itself: `pool_soft_1` (the only switch that stops
   `HighsTableauSeparator`, line 47) reproduces the failures; `lp_age_0` (aging) does not
   (`neos-911970` certifies with 2 508 nodes under `lp_age_0`, fails under `pool_soft_1`).
   This is the analysis doc's L5, measured to be the first-order item rather than a `mik`-only one.
2. **Presolve + restart (+ the root reduced-cost ladder that feeds it).** Sufficient on its own
   for 22/25 (9 of 12); necessary for 13–567× node counts on 6 instances and the restart alone
   for 78–120× on `beavma`/`bppc8-02`. Root-only decomposition (Table 3): on the `mik` family the
   root bound after HiGHS's root is 1.0–1.6 % *because of restarts* (`root+restart_off`:
   13.9–16.4 %, i.e. the pure LP gap; `root+cuts_throttle`: 1.3–1.6 %) — the mik root is a
   presolve/restart product, not a cut product; on `b-ball` it is the opposite (`root+cuts_throttle`
   14.9 % vs 4.7 %). Phase 3 (Table 4): no single presolve rule carries it (`probing_off`,
   `parallel_rc_off`, `sparsify_off`, `dualfix_off`, `presolve_reduce_only` all ≤ 2×) except
   doubleton-equation elimination on `bppc8-02` (152×) — the effect is the composition of the
   reductions plus re-presolving after fixings.
3. **Primal heuristics — one budgeted sub-MIP neighbourhood mechanism with a propagating fixing
   dive.** Never a certification lever for HiGHS at 20 s (0 fails) but the largest *node* lever
   (11 instances ≥ 10×, up to 447×), and the Rust engine is already 2–180× HiGHS-`all_off`'s node
   count (Table 2, first column). Phase 3 shows the four heuristics are **mutually redundant**:
   switching off any one of FJ / RINS / RENS / root-reduced-cost leaves every instance at its
   base node count, except RENS on `bppc8-02` (104×) and `mik-5` (18×). Build one, not four.
4. **The un-ablatable base machinery** (no option switches it): HiGHS with everything off still
   certifies 13 instances Rust does not, mostly from a *worse* root than Rust's (`gt2`: root gap
   36.4 % = the pure LP gap, 386 nodes, incumbent found by node evaluation (`T` sources) — Rust
   holds the bound at 0.0 % and never finds the incumbent in 70 k nodes; `nexp-50-20-1-1` 350 vs
   13 935 nodes; `fiber` 520 vs 7 026; `jijia` 5 060 vs 60 971; `khb05250` 210 vs 1 839). What
   is still running in `all_off`, from the source: domain propagation with the clique table and
   implications (`HighsDomain`, `runSetup` 1000-1157), conflict analysis and the conflict pool,
   reduced-cost fixing at every node plus the root reduced-cost *ladder* (A.2), randomized
   rounding with propagation and an LP re-solve (`R`), central rounding (`C`), the feasibility pump
   while no incumbent, plunging node selection with pseudo-costs, and the implied-bound / clique /
   path / mod-k separators at the root (throttled, not off). This is the part the analysis doc's
   "population A per-node strength" names, and it has to be built alongside 1–3; the ablation
   cannot rank its pieces, C.4 below says which to take first on the evidence available.

Per-instance minimal critical sets, the 40 s confirmation and the caveats are in §2–§4; the
HiGHS mechanisms as read from the source are in §A, discopt's current state in §B, and the port
designs with entry experiments and kill criteria in §C.

## 1. Method

- Arms are HiGHS options (names from `lp_data/HighsOptions.h:1095-1310`, read sites verified in
  `mip/*.cpp`); every option is read back after setting and the worker raises on a mismatch;
  `presolve_rule_off` bits are confirmed by the "Rule nn … not allowed" log line. One fresh
  `python -I` process per solve; the HiGHS log is parsed for the root bound before/after cuts, the
  incumbent-source letters, restarts, cut counts and the LP-iteration split (strong branching /
  separation / heuristics).
- *Certified* = HiGHS status `Optimal` at `mip_rel_gap 1e-4`. Every row is checked against the
  panel oracle: a certified objective outside `1e-6(1+|o|) + 1e-4|o|`, a dual bound beyond the
  oracle, or an incumbent with dense max violation > 1e-5 would be flagged `incorrect` —
  **0 of 699 rows** were.
- Phase 1 (250 solves): leave-one-out, 10 arms × 25. Phase 2 (291): leave-one-in combos and the
  root-only (`mip_max_nodes=1`) decomposition. Phase 3 (114): sub-switches on the instances where
  the family mattered, plus 40 s re-runs of the five slowest baselines.
- No option disables root cut separation. `cuts_throttle` is the tightest the options allow
  (`mip_pool_soft_limit=1`, `mip_lp_age_limit=0`, `mip_pool_age_limit=0`): it stops the tableau
  separator (`HighsTableauSeparator.cpp:47`) and ages every other cut out at once; the implied-
  bound, clique, path and mod-k separators still run each round (visible as 11–162 cuts in the
  pool at the end). `mip_pscost_minreliable=0` makes every candidate "reliable" so no strong
  branching runs (`HighsPseudocost.h:243-245`; the `strong br.` LP-iteration line reads 0).
  `presolve=off` also disables restarts (`HighsMipSolverData.cpp:2127-2128, 2487-2489`).

## Tables

### Table 1 — leave-one-out (threads=1 except `base_tdef`; 20 s; nodes to certify, or FAIL with dual/primal gap % at the limit; bold = >= 10x the `base` nodes)

| instance | base_tdef | base | presolve_off | restart_off | symmetry_off | heur_all_off | node_cuts_off | cuts_throttle | sb_off | probing_off |
|---|---|---|---|---|---|---|---|---|---|---|
| 2122 | 1 | 1 | 1 | 1 | 1 | **447 (447x)** | 1 | 1 | 1 | 1 |
| b-ball | 102 | 102 | **6621 (65x)** | 102 | 100 | 105 | **1116 (11x)** | **FAIL** 100550 d8.2/p0.0 | 102 | 102 |
| beavma | 0 | 0 | 9 | **120 (120x)** | 0 | 1 | 0 | 1 | 0 | 0 |
| bppc8-02 | 1 | 1 | **247 (247x)** | **78 (78x)** | 1 | **219 (219x)** | 1 | **27 (27x)** | 1 | 1 |
| fiber | 9 | 9 | 7 | 7 | 9 | 11 | 13 | 7 | **134 (15x)** | 11 |
| gesa2 | 1 | 1 | **20 (20x)** | 1 | 1 | **81 (81x)** | 1 | 1 | 1 | 1 |
| gsvm2rl3 | **FAIL** 5450 d3.5/p0.0 | **FAIL** 5433 d3.5/p0.0 | **FAIL** 5422 d3.9/p0.0 | **FAIL** 4249 d11.0/p1.6 | **FAIL** 5437 d3.5/p0.0 | 8023 | **FAIL** 8566 d12.0/p0.2 | **FAIL** 12901 d23.0/p0.0 | **FAIL** 7528 d6.0/p1.3 | **FAIL** 5293 d4.2/p0.0 |
| gt2 | 1 | 1 | 1 | 1 | 1 | **109 (109x)** | 1 | 1 | 1 | 1 |
| issue-2446 | 1 | 1 | **567 (567x)** | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| khb05250 | 1 | 1 | 1 | 1 | 1 | **32 (32x)** | 1 | 1 | 1 | 1 |
| mik-250-20-75-1 | 1099 | 1099 | 718 | 1551 | 1099 | 1996 | **11187 (10x)** | 2974 | 2056 | 1099 |
| mik-250-20-75-2 | 89 | 89 | 87 | 769 | 89 | **1456 (16x)** | 180 | 618 | 381 | 89 |
| mik-250-20-75-3 | 455 | 455 | 682 | 487 | 455 | 2442 | 3213 | 2593 | 1572 | 455 |
| mik-250-20-75-5 | 71 | 71 | **1035 (15x)** | 588 | 71 | **2472 (35x)** | **1013 (14x)** | 534 | 691 | 71 |
| neos-1425699 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| neos-3610040-iskar | 380 | 380 | **4899 (13x)** | 380 | 380 | 651 | 2737 | 2791 | 899 | 508 |
| neos-3610051-istra | 288 | 288 | 2310 | 288 | 288 | 1648 | 757 | 473 | 404 | 370 |
| neos-3611447-jijia | 384 | 384 | 933 | 115 | 384 | 435 | 1080 | 940 | 354 | 73 |
| neos-3611689-kaihu | 283 | 283 | 392 | 283 | 283 | 551 | 1141 | 613 | 303 | 163 |
| neos-911970 | 1370 | 1370 | 1114 | 294 | 2290 | 1367 | **FAIL** 7335 d0.8/p0.0 | **FAIL** 9333 d1.8/p0.0 | 2311 | 3304 |
| neos17 | 428 | 428 | 2099 | 443 | 428 | 1873 | 976 | 1907 | 604 | 246 |
| nexp-50-20-1-1 | 1 | 1 | 1 | 1 | 1 | **119 (119x)** | 1 | 1 | 1 | 1 |
| nexp-50-20-4-2 | 1 | 1 | 1 | 1 | 1 | **20 (20x)** | 1 | 1 | 1 | 1 |
| p0548 | 0 | 0 | 1 | 1 | 0 | **62 (62x)** | 0 | 0 | 0 | 0 |
| sp150x300d | 1 | 1 | 1 | 1 | 1 | **12 (12x)** | 1 | 1 | 1 | 1 |

### Table 2 — leave-one-in: plain HiGHS B&B (`all_off` = presolve/restart/symmetry off, heuristics off, cuts throttled, no node cuts, no strong branching) plus ONE component (or two); same cell format, x = vs `base`

| instance | Rust p6.6 nodes (pg/dg %) | base | all_off | keep_cuts | keep_presolve | keep_heur | keep_sb | keep_cuts_heur | keep_cuts_presolve | keep_cuts_sb |
|---|---|---|---|---|---|---|---|---|---|---|
| 2122 | 243 (-/0.0) | 1 | **1916 (1916x)** | **517 (517x)** | **2477 (2477x)** | 1 | **494 (494x)** | 1 | **545 (545x)** | **396 (396x)** |
| b-ball | 233529 (0.0/20.0) | 102 | **FAIL** 163735 d14.9/p0.0 | **7609 (75x)** | **FAIL** 191711 d14.9/p0.0 | **FAIL** 94270 d14.9/p0.0 | **FAIL** 157737 d14.9/p0.0 | **30957 (304x)** | 101 | **13969 (137x)** |
| beavma | 7897 (41.2/39.2) | 0 | **1119 (1119x)** | **39 (39x)** | **116 (116x)** | **312 (312x)** | **249 (249x)** | **26 (26x)** | 1 | **30 (30x)** |
| bppc8-02 | 10513 (1.2/4.0) | 1 | **1753 (1753x)** | **948 (948x)** | **511 (511x)** | **1665 (1665x)** | **782 (782x)** | **407 (407x)** | **380 (380x)** | **281 (281x)** |
| fiber | 7026 (54.8/40.1) | 9 | **520 (58x)** | **461 (51x)** | **1132 (126x)** | **179 (20x)** | 75 | 54 | **225 (25x)** | 19 |
| gesa2 | 34866 (5.8/0.8) | 1 | **8525 (8525x)** | **213 (213x)** | **217 (217x)** | **3103 (3103x)** | **6644 (6644x)** | **83 (83x)** | **10 (10x)** | **752 (752x)** |
| gsvm2rl3 | 42383 (25.0/15.0) | **FAIL** 5433 d3.5/p0.0 | **FAIL** 43144 d22.5/p25.3 | **FAIL** 18028 d7.1/p7.2 | **FAIL** 33307 d19.9/p16.7 | **FAIL** 7977 d24.7/p0.0 | **FAIL** 41413 d21.8/p22.2 | **FAIL** 7462 d6.0/p1.3 | **FAIL** 14365 d2.4/p0.0 | 10898 |
| gt2 | 69841 (13.9/0.0) | 1 | **386 (386x)** | **1046 (1046x)** | **232 (232x)** | 1 | **99 (99x)** | 1 | **66 (66x)** | **105 (105x)** |
| issue-2446 | 1 (-/2510.4) | 1 | **FAIL** 15041 d1859.0/p- | **256 (256x)** | 1 | **FAIL** 7570 d2036.9/p0.0 | **FAIL** 6508 d1621.8/p0.0 | **483 (483x)** | 1 | **240 (240x)** |
| khb05250 | 1839 (4.4/0.8) | 1 | **210 (210x)** | **77 (77x)** | **96 (96x)** | **15 (15x)** | **69 (69x)** | 9 | **114 (114x)** | **22 (22x)** |
| mik-250-20-75-1 | 58006 (0.0/3.2) | 1099 | **FAIL** 26073 d1.7/p0.0 | 4368 | **17225 (16x)** | **FAIL** 19642 d1.3/p0.0 | **FAIL** 24403 d1.6/p0.0 | 1420 | 3346 | **FAIL** 4714 d1.1/p0.6 |
| mik-250-20-75-2 | 52160 (0.0/3.3) | 89 | **FAIL** 27569 d1.0/p0.0 | **2072 (23x)** | **12922 (145x)** | **FAIL** 19391 d1.5/p0.0 | **17678 (199x)** | 625 | 588 | **2343 (26x)** |
| mik-250-20-75-3 | 52756 (0.3/2.8) | 455 | **FAIL** 26155 d2.0/p2.3 | 2529 | **13261 (29x)** | **FAIL** 22491 d1.4/p0.0 | **FAIL** 23927 d1.8/p0.0 | 1594 | 1910 | 2315 |
| mik-250-20-75-5 | 55986 (0.1/3.0) | 71 | **FAIL** 28298 d1.9/p0.0 | **4590 (65x)** | **32512 (458x)** | **FAIL** 22344 d1.3/p0.0 | **FAIL** 25215 d1.3/p0.0 | **1714 (24x)** | **3703 (52x)** | **3110 (44x)** |
| neos-1425699 | 4587 (-/0.0) | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| neos-3610040-iskar | 71401 (13.5/2.7) | 380 | **FAIL** 32968 d5.3/p2.7 | **6031 (16x)** | **6506 (17x)** | **FAIL** 24250 d5.3/p0.0 | **FAIL** 32809 d5.0/p2.7 | **4783 (13x)** | 591 | **5145 (14x)** |
| neos-3610051-istra | 27423 (12.2/2.0) | 288 | **FAIL** 13338 d4.0/p0.0 | **4757 (17x)** | **3534 (12x)** | **FAIL** 10707 d3.7/p0.0 | **FAIL** 12179 d3.9/p2.0 | **3510 (12x)** | 2084 | **FAIL** 7389 d2.0/p0.0 |
| neos-3611447-jijia | 60971 (24.3/1.9) | 384 | **5060 (13x)** | 1596 | 2954 | **5350 (14x)** | **10496 (27x)** | 800 | 449 | 1291 |
| neos-3611689-kaihu | 63125 (21.8/2.5) | 283 | **FAIL** 24489 d3.2/p0.8 | 1463 | **13478 (48x)** | **12779 (45x)** | **15753 (56x)** | 581 | 1323 | 1116 |
| neos-911970 | 7 (313.6/57.5) | 1370 | **FAIL** 15081 d6.1/p6.1 | **FAIL** 8229 d0.8/p0.0 | **FAIL** 15841 d5.1/p0.1 | **FAIL** 5203 d6.8/p1.0 | **FAIL** 8396 d3.5/p11.4 | 2603 | 5716 | 3229 |
| neos17 | 17467 (24.6/9.9) | 428 | **8746 (20x)** | 2553 | **7185 (17x)** | **11456 (27x)** | **7912 (18x)** | 2341 | 2338 | 1865 |
| nexp-50-20-1-1 | 13935 (65.5/41.4) | 1 | **350 (350x)** | **229 (229x)** | **293 (293x)** | 1 | **118 (118x)** | 1 | **149 (149x)** | **101 (101x)** |
| nexp-50-20-4-2 | 6873 (12.7/23.9) | 1 | **1897 (1897x)** | 7 | **1694 (1694x)** | 1 | **1479 (1479x)** | 1 | **25 (25x)** | **86 (86x)** |
| p0548 | 983 (-/7.3) | 0 | **1668 (1668x)** | **303 (303x)** | **177 (177x)** | **60 (60x)** | **754 (754x)** | 1 | **67 (67x)** | **78 (78x)** |
| sp150x300d | 9181 (34.8/42.0) | 1 | **FAIL** 27047 d10.1/p0.0 | **114 (114x)** | **59 (59x)** | **FAIL** 18944 d10.1/p0.0 | **FAIL** 11271 d8.7/p0.0 | 1 | 6 | **34 (34x)** |
| **certified / 25** | 0 | 24 | 13 | 23 | 22 | 14 | 15 | 24 | 24 | 23 |

### Table 3 — root-only decomposition (`mip_max_nodes=1`): dual gap % | primal gap % after the root (restarts in parentheses); first column = pure LP relaxation gap (integrality dropped, `d2_lp_bounds.json`)

| instance | LP gap % | root | root+cuts_throttle | root+presolve_off | root+restart_off | root+heur_all_off |
|---|---|---|---|---|---|---|
| 2122 | 0.00 | 0.00 \| 0.00 (0) | 0.00 \| 0.00 (1) | 0.00 \| 0.00 (0) | 0.00 \| 0.00 (0) | 0.00 \| 62.46 (0) |
| b-ball | 21.21 | 4.74 \| 0.00 (0) | 14.89 \| 0.00 (0) | 6.55 \| 0.00 (0) | 4.74 \| 0.00 (0) | 4.74 \| 0.00 (0) |
| beavma | 59.36 | 0.00 \| 0.00 (5) | 0.00 \| 0.00 (4) | 0.82 \| 0.00 (0) | 14.65 \| 0.29 (0) | 0.00 \| 0.00 (3) |
| bppc8-02 | 4.04 | 0.00 \| 0.00 (1) | 2.56 \| 0.00 (2) | 4.04 \| 1.38 (0) | 2.37 \| 0.00 (0) | 2.37 \| 6.31 (0) |
| fiber | 61.55 | 2.21 \| 0.00 (2) | 3.53 \| 0.00 (4) | 2.11 \| 0.00 (0) | 2.47 \| 0.00 (0) | 2.32 \| 54.65 (1) |
| gesa2 | 1.18 | 0.00 \| 0.00 (0) | 0.01 \| 0.00 (2) | 0.05 \| 0.00 (0) | 0.00 \| 0.00 (0) | 0.00 \| 3.52 (0) |
| gsvm2rl3 | 32.45 | 23.91 \| 26.14 (0) | 30.56 \| 26.35 (0) | 23.91 \| 26.14 (0) | 23.91 \| 26.14 (0) | 23.91 \| 26.35 (0) |
| gt2 | 36.41 | 0.00 \| 0.00 (0) | 0.00 \| 0.00 (0) | 0.00 \| 0.00 (0) | 0.00 \| 0.00 (0) | 0.00 \| 76.78 (0) |
| issue-2446 | 2510.37 | 0.00 \| 0.00 (1) | 0.00 \| 0.00 (1) | 676.54 \| - (0) | 34.76 \| 0.00 (0) | 0.00 \| 0.00 (1) |
| khb05250 | 10.31 | 0.01 \| 0.00 (4) | 0.01 \| 0.00 (2) | 0.17 \| 0.00 (0) | 0.21 \| 0.00 (0) | 0.30 \| 7.38 (0) |
| mik-250-20-75-1 | 18.99 | 1.60 \| 0.00 (3) | 1.56 \| 0.00 (3) | 1.75 \| 0.00 (0) | 16.38 \| 46.80 (0) | 2.13 \| 4.20 (3) |
| mik-250-20-75-2 | 18.16 | 1.05 \| 0.00 (4) | 1.27 \| 0.00 (4) | 1.53 \| 0.00 (0) | 15.72 \| 41.41 (0) | 1.07 \| 2.81 (3) |
| mik-250-20-75-3 | 16.13 | 1.44 \| 0.00 (3) | 1.54 \| 0.00 (3) | 2.03 \| 0.00 (0) | 13.93 \| 34.73 (0) | 1.38 \| 2.02 (3) |
| mik-250-20-75-5 | 17.46 | 1.15 \| 0.00 (3) | 1.42 \| 0.00 (4) | 1.63 \| 0.00 (0) | 15.02 \| 37.13 (0) | 1.89 \| 3.68 (3) |
| neos-1425699 | 0.14 | 0.00 \| 0.00 (0) | 0.00 \| 0.00 (0) | 0.00 \| 0.00 (0) | 0.00 \| 0.00 (0) | 0.00 \| 0.00 (0) |
| neos-3610040-iskar | 8.53 | 5.41 \| 0.00 (0) | 5.41 \| 0.00 (0) | 5.52 \| 0.00 (0) | 5.41 \| 0.00 (0) | 5.41 \| 10.81 (0) |
| neos-3610051-istra | 9.07 | 4.08 \| 0.00 (0) | 4.08 \| 0.00 (0) | 4.02 \| 0.00 (0) | 4.08 \| 0.00 (0) | 4.08 \| 16.33 (0) |
| neos-3611447-jijia | 11.95 | 2.80 \| 0.93 (1) | 3.74 \| 0.93 (1) | 3.54 \| 0.93 (0) | 2.80 \| 0.93 (0) | 2.80 \| 11.21 (0) |
| neos-3611689-kaihu | 15.23 | 4.20 \| 0.00 (0) | 4.20 \| 0.00 (0) | 4.24 \| 0.00 (0) | 4.20 \| 0.00 (0) | 4.20 \| 20.17 (0) |
| neos-911970 | 57.52 | 3.69 \| 3.94 (2) | 5.82 \| 2.01 (2) | 4.77 \| 0.82 (0) | 4.47 \| 4.00 (0) | 4.81 \| 213.86 (0) |
| neos17 | 14.93 | 0.63 \| 0.00 (1) | 0.48 \| 0.02 (3) | 2.60 \| 0.93 (0) | 1.29 \| 0.00 (0) | 1.15 \| 6.57 (1) |
| nexp-50-20-1-1 | 58.52 | 0.00 \| 0.00 (0) | 0.00 \| 0.00 (0) | 0.00 \| 0.00 (0) | 0.00 \| 0.00 (0) | 0.00 \| 20.69 (0) |
| nexp-50-20-4-2 | 30.86 | 0.00 \| 0.00 (0) | 0.00 \| 0.00 (0) | 0.00 \| 0.00 (0) | 0.00 \| 0.00 (0) | 0.00 \| 53.52 (0) |
| p0548 | 96.37 | 0.00 \| 0.00 (1) | 0.00 \| 0.00 (3) | 0.00 \| 0.00 (0) | 0.03 \| 0.00 (0) | 0.03 \| 31.33 (1) |
| sp150x300d | 92.91 | 0.00 \| 0.00 (0) | 0.00 \| 0.00 (0) | 0.00 \| 0.00 (0) | 0.00 \| 0.00 (0) | 0.00 \| 2.90 (0) |

### Table 4 — sub-switches (phase 3) on the instances where the family mattered: nodes (x base), FAIL = not certified at 20 s

**heuristics**

| instance | base | heur_all_off | fj_off | rins_off | rens_off | rootredcost_off | heur_effort_0 |
|---|---|---|---|---|---|---|---|
| 2122 | 1 | **447 (447x)** | 1 | 1 | 1 | 1 | 1 |
| bppc8-02 | 1 | **219 (219x)** | 1 | 1 | **104 (104x)** | 1 | 1 |
| gesa2 | 1 | **81 (81x)** | 1 | 1 | 1 | 1 | 1 |
| gsvm2rl3 | 5433 | 8023 | 6072 | 5018 | 4726 | 4645 | 5055 |
| gt2 | 1 | **109 (109x)** | 1 | 1 | 1 | 1 | 1 |
| khb05250 | 1 | **32 (32x)** | 1 | 1 | 1 | 1 | 1 |
| mik-250-20-75-2 | 89 | **1456 (16x)** | 89 | 33 | 97 | 39 | 89 |
| mik-250-20-75-5 | 71 | **2472 (35x)** | 71 | 79 | **1260 (18x)** | 139 | 71 |
| nexp-50-20-1-1 | 1 | **119 (119x)** | 1 | 1 | 1 | 1 | 1 |
| nexp-50-20-4-2 | 1 | **20 (20x)** | 1 | 1 | 1 | 1 | 1 |
| p0548 | 0 | **62 (62x)** | 0 | 0 | 0 | 1 | 0 |
| sp150x300d | 1 | **12 (12x)** | 1 | 1 | 1 | 1 | 1 |

**presolve rules (`presolve_rule_off` bits)**

| instance | base | presolve_off | probing_off | parallel_rc_off | aggregator_off | doubleton_off | sparsify_off | dualfix_off | presolve_reduce_only |
|---|---|---|---|---|---|---|---|---|---|
| b-ball | 102 | **6621 (65x)** | 102 | 102 | 1 | 102 | 102 | 102 | 131 |
| bppc8-02 | 1 | **247 (247x)** | 1 | 1 | 1 | 0 | **152 (152x)** | 1 | 1 |
| gesa2 | 1 | **20 (20x)** | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| gsvm2rl3 | 5433 | **FAIL** 5422 d3.9/p0.0 | **FAIL** 5293 d4.2/p0.0 | **FAIL** 5215 d4.4/p0.0 | **FAIL** 5372 d4.0/p0.0 | **FAIL** 5346 d4.1/p0.0 | **FAIL** 5407 d3.9/p0.0 | **FAIL** 5441 d3.5/p0.0 | **FAIL** 5458 d4.4/p0.0 |
| issue-2446 | 1 | **567 (567x)** | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| mik-250-20-75-5 | 71 | **1035 (15x)** | 71 | 159 | 71 | 71 | 71 | 71 | 159 |
| neos-3610040-iskar | 380 | **4899 (13x)** | 508 | 380 | 380 | 380 | 380 | 380 | 655 |

**cuts**

| instance | base | cuts_throttle | node_cuts_off | pool_soft_1 | lp_age_0 |
|---|---|---|---|---|---|
| b-ball | 102 | **FAIL** 100550 d8.2/p0.0 | **1116 (11x)** | **FAIL** 161766 d14.9/p0.0 | 101 |
| bppc8-02 | 1 | **27 (27x)** | 1 | 1 | **53 (53x)** |
| gsvm2rl3 | 5433 | **FAIL** 12901 d23.0/p0.0 | **FAIL** 8566 d12.0/p0.2 | **FAIL** 14112 d22.9/p0.0 | **FAIL** 13634 d9.9/p0.0 |
| mik-250-20-75-1 | 1099 | 2974 | **11187 (10x)** | 1163 | 2793 |
| mik-250-20-75-5 | 71 | 534 | **1013 (14x)** | 209 | 437 |
| neos-911970 | 1370 | **FAIL** 9333 d1.8/p0.0 | **FAIL** 7335 d0.8/p0.0 | **FAIL** 6845 d0.2/p0.0 | 2508 |

### Table 5 — 40 s re-runs of the slow baselines (threads=1, host load ~2): status, nodes, wall

| instance | base @40 s | all_off @40 s |
|---|---|---|
| gsvm2rl3 | certified 5817 n 21.5 s | FAIL d20.9/p21.3 86493 n 40.0 s |
| mik-250-20-75-1 | certified 1099 n 8.4 s | certified 45495 n 34.2 s |
| mik-250-20-75-3 | certified 455 n 4.9 s | FAIL d1.6/p0.1 54554 n 40.0 s |
| neos-911970 | certified 1370 n 11.9 s | FAIL d4.8/p6.1 29665 n 40.0 s |
| neos17 | certified 428 n 8.1 s | certified 8746 n 12.0 s |


## 2. Per-instance minimal set of HiGHS components whose removal makes it fail or blow up ≥ 10× (from Tables 1 and 4; ‡ = load-confounded)

| instance | Rust (p6.6) | removal → FAIL | removal → ≥ 10× nodes | what plain B&B (`all_off`) does | single component that rescues `all_off` |
|---|---|---|---|---|---|
| 2122 | time_limit, 243 n, no incumbent within 1e-4 | – | heuristics 447× | certifies, 1 916 n (incumbents from node evaluation) | (not needed) |
| b-ball | 233 k n, dual 20 % | **tableau cuts** (`cuts_throttle`, `pool_soft_1`) | presolve 65×, node cuts 11× | FAIL, 164 k n, dual 14.9 % | **cuts only** (7 609 n); presolve/heur/SB do not |
| beavma | 7.9 k n, 41/39 % | – | restart 120× | certifies, 1 119 n | (not needed) |
| bppc8-02 | 10.5 k n, 1.2/4.0 % | – | presolve 247× (doubleton eq. 152×), heuristics 219× (RENS 104×), restart 78×, cuts 27× | certifies, 1 753 n | (not needed) |
| fiber | 7.0 k n, 55/40 % | – | strong branching 15× | certifies, 520 n | (not needed) |
| gesa2 | 34.9 k n, 5.8/0.8 % | – | heuristics 81×, presolve 20× | certifies, 8 525 n (19 s) | (not needed) |
| gsvm2rl3 ‡ | 42 k n, 25/15 % | (base itself fails at 20 s under load; certifies at 21.5 s / 40 s) | – | FAIL at 40 s, 86 k n, 20.9 % | cuts + SB (10 898 n, 16 s); heuristics *cost* it (heur_all_off certifies in 12.6 s) |
| gt2 | 69.8 k n, 13.9/0.0 % | – | heuristics 109× | certifies, 386 n from a 36.4 % root | (not needed) |
| issue-2446 | 1 n, no incumbent | – | presolve 567× | FAIL, 15 k n, no incumbent | cuts (256 n) or presolve (1 n) |
| khb05250 | 1.8 k n, 4.4/0.8 % | – | heuristics 32× | certifies, 210 n | (not needed) |
| mik-250-20-75-1 | 58 k n, 0/3.2 % | – | node cuts 10× | FAIL, 26 k n, 1.7 % (certifies at 34 s / 45 k n) | cuts (4 368 n) or presolve+restart (17 225 n) |
| mik-250-20-75-2 | 52 k n, 0/3.3 % | – | heuristics 16× | FAIL, 28 k n, 1.0 % | cuts (2 072), presolve (12 922), SB (17 678) |
| mik-250-20-75-3 | 53 k n, 0.3/2.8 % | – | none (every single removal ≤ 10×) | FAIL, 26 k n, 2.0 % (1.6 % at 40 s) | cuts (2 529) or presolve (13 261) |
| mik-250-20-75-5 | 56 k n, 0.1/3.0 % | – | heuristics 35× (RENS 18×), node cuts 14×, presolve 15× | FAIL, 28 k n, 1.9 % | cuts (4 590) or presolve (32 512, 20 s) |
| neos-1425699 | 4.6 k n, no incumbent | – | none | certifies at the root, 1 n | (not needed) |
| neos-3610040-iskar | 71 k n, 13.5/2.7 % | – | presolve 13× | FAIL, 33 k n, 2.7/5.3 % | cuts (6 031) or presolve (6 506) |
| neos-3610051-istra | 27 k n, 12.2/2.0 % | – | none | FAIL, 13 k n, 0/4.0 % | cuts (4 757) or presolve (3 534) |
| neos-3611447-jijia | 61 k n, 24/1.9 % | – | none | certifies, 5 060 n | (not needed) |
| neos-3611689-kaihu | 63 k n, 22/2.5 % | – | none | FAIL, 24 k n, 0.8/3.2 % | any one: cuts 1 463, presolve 13 478, heur 12 779, SB 15 753 |
| neos-911970 | 7 n (LP-engine stall), 314/58 % | **tableau cuts** (`cuts_throttle`, `pool_soft_1`), **node cuts** | – | FAIL, 15 k n, 6.1 % (4.8 % at 40 s) | none alone; cuts + heuristics (2 603 n), cuts + presolve (5 716), cuts + SB (3 229) |
| neos17 | 17.5 k n, 25/9.9 % | – | none | certifies, 8 746 n | (not needed) |
| nexp-50-20-1-1 | 13.9 k n, 66/41 % | – | heuristics 119× | certifies, 350 n | (not needed) |
| nexp-50-20-4-2 | 6.9 k n, 13/24 % | – | heuristics 20× | certifies, 1 897 n | (not needed) |
| p0548 | 983 n, no incumbent | – | heuristics 62× | certifies, 1 668 n | (not needed) |
| sp150x300d | 9.2 k n, 35/42 % | – | heuristics 12× | FAIL, 27 k n, 10.1 % | cuts (114 n) or presolve (59 n) |

Reading across: **13 of the 25** are certified by HiGHS's plain B&B with no switchable component
(the base machinery, item 4 of §0, is the gap on those); **10 of the other 12** are rescued by
cuts alone and 9 by presolve+restart alone, 8 by either; `b-ball` needs the tableau separator
specifically; `neos-911970` needs cuts plus one more component; `gsvm2rl3` needs ~20 s of HiGHS
however configured.

## 3. The 40 s confirmation (Table 5)

Five slow baselines re-run at 40 s on the quiet host (load ≈ 2): `gsvm2rl3` **certifies at
21.5 s** (5 817 nodes, 0 primal gap at the limit of the 20 s run too) — the 20 s result was the
budget under a load of 7.5, not a component effect; `neos-911970`, `mik-1`, `mik-3`, `neos17`
reproduce their 20 s node counts exactly (1 370 / 1 099 / 455 / 428, deterministic). `all_off` at
40 s: `mik-1` certifies (45 495 nodes, 34 s), `neos17` (8 746), `mik-3` still 1.6 % after
54 554 nodes, `neos-911970` 4.8 %, `gsvm2rl3` 20.9 % — plain B&B is *slow* on `mik-1` and *stuck*
on the other three.

## 4. Caveats (binding on how §0 may be quoted)

- **Load.** Phase 1 ran at load 6.5–8.7 (one solve at a time); the relaunched phases 2–3 at
  2–3 (two at a time). Node counts and certification are deterministic given the option set
  (confirmed: every 20 s-vs-40 s baseline pair reproduced its node count), but a *time-limited*
  arm's node count and gap depend on how far it got; the FAIL cells are "did not certify in
  20 s on this host", and `gsvm2rl3` shows the margin can be the host. Single rep per cell —
  trajectory effects of the kind the analysis doc measured on the Rust engine (1-ulp flips)
  exist in HiGHS too (`symmetry_off` on `neos-911970`: 2 290 vs 1 370 nodes with the same root).
- **`cuts_throttle` is not "cuts off".** Implied-bound, clique, path and mod-k separators still
  run; the arm removes the tableau separator and ages everything else out immediately. The
  attribution to the tableau separator rests on `pool_soft_1` (phase 3) reproducing the two
  failures while `lp_age_0` does not.
- **`presolve_off` removes restarts too** (the restart requires presolve), so "presolve" in the
  tables is presolve+restart+clique-substitution; `restart_off` alone isolates the restart.
- **Heuristics that cannot be switched off** (randomized rounding `R`, central rounding `C`,
  feasibility pump `F`, node-evaluation incumbents `T`) are in every arm including `all_off`;
  "heuristics" in the tables means FJ + RINS + RENS + root-reduced-cost + the in-tree effort
  budget.
- The raw-highspy `base` is faster than the panel's H arm on every instance (e.g. `issue-2446`
  0.1 s vs 5.1 s, `2122` 0.3 vs 1.8 s) because the H arm includes the route's `.nl` round-trip
  and certificate verification; the node counts differ on the `mik` family (`mik-1` 1 099 here vs
  453 in the panel) because the panel ran with default threads on a different host state — the
  ablation compares arms against its own `base` only.
- HiGHS source was read, never built or executed; line numbers refer to the 1.15.1 tarball under
  `scratchpad/highs_src/highspy-1.15.1/highs/`.

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

## C. Port designs for the top components (independent implementations against discopt's data structures)

Each design states what HiGHS does (§A), what discopt has (§B), the gap, the plan, and a
CLAUDE.md §4/§5 entry experiment with a kill criterion. Every one ships behind a default-OFF
`DISCOPT_*` flag with a row in `docs/dev/flag-retirement-audit.md`, and graduates only on a
P4-standard panel (MIPLIB-38 + M, 3 reps, 20 s, interleaved) that is **cert-clean** (`incorrect
= 0`, no bound above its oracle, no `gap_certified` regression, incumbents re-verified) **and
net-positive** (certified ≥ OFF, no incumbent lost on any row, nodes and primal gap not worse
on the both-certified rows, both-certified wall ≤ 1.10× OFF). Nothing below copies HiGHS code;
the behaviour is described and mapped onto discopt's CSC standard form, `TreeManager` batches
and the existing separators.

### C.1 Tableau-row cut generation: lifted cover + c-MIR on aggregated tableau rows, separated at nodes, with LP/pool aging  (accounts for 10 of the 12 plain-B&B failures; necessary on `b-ball`, `neos-911970`; 10–27× nodes on 3 more)

**HiGHS** (§A.1 step 6, §A.4). The separator that does the work is `generateCut` on the
**aggregated tableau row** of every fractional basic integer (and fractional integral row),
both signs, ranked by `f(1−f)/‖B⁻¹ row‖²`. The row is transformed to the bound-substituted,
complemented space (variable bounds included), then a *lifted cover* inequality is tried
(knapsack / mixed-binary / mixed-integer variants by row type) and a *c-MIR* over a small delta
set ({1, the |a_j| of integer columns at nonzero value, the integral scale}, f0 ∈ [0.005, 0.995]);
the more efficacious survives postprocessing (integral scaling, tiny-coefficient removal into the
RHS, violation > 10·tol, coefficient tightening against the box). There is no GMI. The same round
runs at every node taken from the queue; LP rows leave after 10 non-binding epochs, pool rows
after 30 unused agings, and violated pool rows are re-added. Measured here: `pool_soft_1` (stops
the tableau separator and nothing else) fails `b-ball` (164 k nodes, 14.9 %) and `neos-911970`;
`lp_age_0` (aging only) is harmless; `node_cuts_off` fails `neos-911970` and costs 10–14× on
`mik-1`/`mik-5`; `keep_cuts` alone turns plain B&B from 13/25 to 23/25 (`mik` 2–4.6 k nodes,
`b-ball` 7.6 k, `sp150x300d` 114, `istra` 4.8 k, `iskar` 6.0 k, `kaihu` 1.5 k).

**discopt** (§B). Root GMI from refined tableau rows (`gomory.rs`, dense after P4.4's repair),
c-MIR and covers from the *original* rows only (`cmir.rs`, `cover.rs`), node covers only, no aging,
no pool re-separation; every root cut row stays in every node LP for the whole search.

**Gap.** (i) No c-MIR / lifted cover on **tableau** rows — HiGHS's tableau separator is "GMI's
row with c-MIR's rounding and a cover lift", and it is what the analysis doc's L5 deferred;
(ii) no tableau/c-MIR separation at nodes; (iii) no aging, so every root cut is paid for at every
node (the per-node cost population B of the analysis doc).

**Plan** (`crates/discopt-core/src/lp/tableau_cuts.rs`, wired at the two existing separation
points):
1. *Row supply.* Reuse `gomory.rs`'s refined `B⁻ᵀe_i` rows (iterative refinement already there) for
   the basic integers with fractionality in `[1000·tol, 1 − 1000·tol]`; rank by `f(1−f)/‖row‖²`,
   drop rows whose scaled weights span > 1e4 (HiGHS's numerics gate), cap at `200 + 0.1·min(m,
   n_int)` rows per round, and hand each row **and its negation** on.
2. *Cut core.* Extend `cmir.rs`'s bound-substitution/complementation/MIR core (it already does
   variable-bound substitution and the delta set) with (a) HiGHS's delta set and f0 window, and
   (b) a lifted-cover branch: on a row whose integer part is binary and whose continuous part
   is bounded, build the cover from the LP point (largest `x*` first, minimal), lift the
   non-cover binaries sequence-independently (superadditive lifting function of the cover
   weights), and keep cover vs c-MIR by efficacy. Mixed-integer rows go to c-MIR only in the
   first increment.
3. *Postprocessing as HiGHS:* scale to integral coefficients when a scale ≤ 1e4 exists, move
   coefficients < 1e-9·max into the RHS using the column bound (never drop them silently —
   #1236's lesson in `gomory.rs`), require violation > 10·tol, coefficient-tighten against the box,
   then `cut_select.rs` (efficacy + parallelism). Every cut is a combination of original rows
   with real multipliers, so the soundness argument of `cmir.rs` §Soundness carries over
   unchanged; the feasible-point sampling test of §5 is the regression guard.
4. *Node separation.* Run the separator at the first fractional node of each batch, in the slot
   `separate_cover_csc` uses (`milp_driver.rs:3783`), under `node_cut_cap`; fold at the batch
   boundary like node covers.
5. *Aging.* Per cut row an epoch age: a row basic at `lp_age_limit` (10) consecutive batch
   boundaries leaves the node LP (`rebuild_csc_with_cuts` already rebuilds the CSC at boundaries)
   and is parked in the pool; the pool is re-separated at each boundary (violation > tol,
   efficacy-sorted, parallelism-filtered); rows unused for `pool_age_limit` (30) boundaries are
   deleted. The dense P4.4 GMI rows get the same treatment, which is the analysis doc's L2
   density gate without a new gate.
Flags: `DISCOPT_MILP_TABLEAU_CUTS=1` (steps 1–3, root), `DISCOPT_MILP_NODE_TABLEAU=1` (4),
`DISCOPT_MILP_CUT_AGING=1` (5), each with its audit row.

**Entry experiment** (root only, before 4–5): `p44_root_gap.py --all` with the separator ON vs
OFF on the 25 instances here. Bars, against Table 3's HiGHS root with restarts disabled
(`root+restart_off`, i.e. what cuts alone buy HiGHS): `b-ball` root gap ≤ 6 % (HiGHS 4.74 %),
`neos-911970` ≤ 6 % (4.47 %), `fiber` ≤ 3.5 % (2.47 %), `bppc8-02` ≤ 3 % (2.37 %), `neos17` ≤ 2 %
(1.29 %); no root bound above its oracle; feasible-point sampling (the panel's incumbents) cuts
off nothing. **Kill:** `b-ball` or `neos-911970` root gap not within 2 pp of HiGHS's cuts-only
root, or any invalid cut → retire `tableau_cuts.rs` (keep the lifted-cover core if it is
sound, as `cover.rs`'s lifting). Steps 4–5 graduate separately: aging must be exactly
node-count-neutral on the P6.6-certified rows with `lp_age_limit = ∞` (identity test) before
its default is measured on wall; node separation's bar is `neos-911970` and `mik-1`/`mik-5`
certifying with ≤ 0.25× HEAD's nodes.

### C.2 Presolve + restart with a root reduced-cost ladder  (accounts for 9 of 12; necessary for 13–567× on 6 instances, the restart alone for 78–120× on 2)

**HiGHS** (§A.1 steps 1, 5, 8; §A.2). A full dimension-reducing presolve with MIP-only rules
(probing into a clique table and implication graph, coefficient strengthening, dominated
columns, parallel rows/columns, doubleton equations, aggregation, sparsify, dependent
equations, clique substitution); root reduced costs kept as a per-column ladder of
(cutoff → bound) pairs after every separation round, re-applied globally at every incumbent
improvement; when the inactive-integer fraction crosses a threshold (any on the first pass,
2.5 % later, 10 % after the root heuristics, or an in-tree tree-size vote) the model is
re-presolved with the cuts folded in and the root re-evaluated. Measured: on the `mik` family the
root bound is 1.0–1.6 % only with restarts (13.9–16.4 % = the LP gap without, Table 3), 3–4
restarts per instance; `restart_off` 120× on `beavma`, 78× on `bppc8-02`; `presolve_off` 13–567×
on 6 instances; no single presolve rule carries it (Table 4) except doubleton equations on
`bppc8-02` (152×); `keep_presolve` alone rescues 9 of the 12 plain-B&B failures.

**discopt** (§B). The engine's only presolve is FBBT on the marshaled rows (`root_box`); the
Python orchestrator (elimination, FBBT, probing; cliques off) runs once on the model and hands
over bounds; reduced-cost fixing is per node and local to the subtree, no ladder; no restart in
any form (the route's re-entry re-runs the *same* model with a seeded incumbent).

**Gap.** (i) Root reduced costs are not kept, so an incumbent found later never fixes root columns
globally; (ii) nothing re-presolves after fixings; (iii) the engine has no dimension-reducing
presolve on its own rows (no parallel/dominated/doubleton/aggregation, no coefficient
strengthening, no clique substitution).

**Plan** (two increments, the first cheap):
1. *Lurking ladder* (`bnb/redcost_ladder.rs`). After each root cut round that ends `Optimal`, for
   every nonbasic integer column `j` with `|d_j|` above the #1409 deflated tolerance, store
   `(cutoff_k, bound_k)` for `k = 1, 2, 4, …, 2^10` steps from the nonbasic bound
   (`cutoff_k = z_root + |d_j|·(k − 10·tol)`), dropping entries dominated by a tighter bound at a
   lower cutoff. On every incumbent improvement (`inject_incumbent`), apply every entry with
   `cutoff ≥ incumbent − tol` to the **global** root box, FBBT once, and push the box into the
   batch's `tightened` merge, which children already consume. Sound: it is the inequality
   `reduced_cost_fix` already argues, evaluated at the root with the root's safe bound.
2. *Restart* (`bnb/restart.rs`). Count inactive integers (fixed by the ladder, root FBBT or root
   reduced-cost fixing). When ≥ 10 % after the root heuristics or ≥ 2.5 % after the cut loop (any
   on the first pass): rebuild the marshaled problem with fixed columns removed (column deletion
   with RHS adjustment — the only reduction the first increment needs), carry the kept root cuts
   as rows and the root basis, re-run FBBT and the cut loop on the smaller problem, and restart
   the tree from an empty frontier with the incumbent kept; at most 3 restarts (HiGHS's count on
   `mik`). The route's slice/re-entry already restarts trees with a seeded incumbent; the new
   part is the reduced model and the trigger.
3. *Dimension-reducing presolve on the rows* (the analysis doc's L7, last): doubleton equations
   first (the one rule phase 3 singled out), then parallel rows/columns, dominated columns and
   coefficient strengthening with a postsolve stack — port the orchestrator's
   `substitute.rs` / `aggregate.rs` / `coefficient_strengthening.rs` to the CSC standard form
   rather than writing new reductions. Only after 1–2 are measured.
Flags: `DISCOPT_MILP_RC_LADDER=1`, `DISCOPT_MILP_RESTART=1`, later `DISCOPT_MILP_ROW_PRESOLVE=1`.

**Entry experiment** for 1+2 (no presolve port): `beavma`, `bppc8-02`, the four `mik`, `gesa2`,
`iskar`, `issue-2446`, `sp150x300d` at 20 s on the direct driver (`d2_direct_driver_panel.py`),
2 reps. Bars: the ladder fixes ≥ 2.5 % of the integer columns on ≥ 6 of the 10 rows (the
restart trigger), the restarted `mik` root bound closes ≥ 2/3 of the gap between the LP root
and HiGHS's 1.0–1.6 % root (i.e. ≤ 7 % residual from the 16–19 % LP gap), and nodes to certify
on `beavma`/`bppc8-02` ≤ 0.25× HEAD. **Kill:** the ladder fixes < 2 % on every row (then the
restart has nothing to act on and the lever is C.1's cuts), any bound above its oracle (a
soundness bug — stop), or node counts not better on ≥ 4 of the 10 rows → retire both flags and
their code.

### C.3 One budgeted sub-MIP neighbourhood heuristic with a propagating fixing dive  (0 certification losses for HiGHS at 20 s, but ≥ 10× nodes on 11 instances, up to 447×; the four HiGHS heuristics are redundant with each other)

**HiGHS** (§A.3). One machinery, three neighbourhoods. A *dive* on a private copy of the LP
relaxation fixes integers one at a time with domain propagation after each fix (conflict analysis
on failure), re-solving the LP as it goes, until a target fixing rate; what remains is a
**budgeted full sub-MIP** (500 leaves, 200 + nodes/20 nodes, 12 stall nodes, objective cutoff =
incumbent, cliques/implications/pseudo-costs inherited). The fixing rate adapts from observed
successes and infeasibilities. RENS fixes the LP-integral columns first (no incumbent needed),
RINS the LP/incumbent agreement set, `rootReducedCost` the root reduced-cost ladder. They run at
the root and at every plunge start while the heuristic LP-iteration budget allows (10 k free, then
5 % of the tree's LP iterations, front-loaded). Phase 3: removing any one of FJ / RINS / RENS /
root-reduced-cost leaves every instance at its base node count except RENS on `bppc8-02` (104×)
and `mik-5` (18×); removing all of them costs 12–447× nodes on 11 instances.

**discopt** (§B). `try_rins` → `run_sub_mip` exists (agreement neighbourhood pinned by bounds,
recursive driver call with a node budget and ≤ 2 s, strict improvement, re-validated) and is
**default-OFF**; there is no propagation in the neighbourhood construction, no RENS (retired
P4.2), no root-reduced-cost neighbourhood, no adaptive fixing rate; the dive heuristic is an
LP-only most-fractional dive without propagation, cutoff or budget, run once at the root and
only while no incumbent exists. The consequence is the analysis doc's frozen primal (12/12
rows hold the same incumbent at 20 s and 60 s).

**Gap.** (i) No propagation-driven fixing dive (what makes HiGHS's neighbourhoods feasible and
small enough); (ii) no budgeted *tree* schedule; (iii) no RENS / root-reduced-cost
neighbourhoods; (iv) no adaptive fixing rate.

**Plan** (`bnb/sub_mip.rs`, replacing `try_rins`'s neighbourhood builder, keeping `run_sub_mip`):
1. *Fixing dive with FBBT.* A `Neighbourhood` builder takes the batch's node point, the incumbent
   (optional), the ladder of C.2 (optional) and a target fixing rate; it fixes columns in HiGHS's
   order (RINS: agreement set first; RENS: LP-integral first; then fractional ones towards the
   root-direction/objective rounding rule), running `tighten_bounds_csc_with_row` on the sub-box
   after each **block** of `max(1, n_int/50)` fixings (FBBT sweeps all rows, so per-fix
   propagation is too dear); on an infeasible box undo the block and halve the block size (the
   cheap stand-in for conflict analysis). Stop at the target rate, counted on integer columns.
2. *Sub-MIP call.* Add `max_leaves` and `max_stall_nodes` to `MilpOptions` (sub-solves only),
   pass the incumbent as the cutoff (the sub-driver already seeds `initial_incumbent`), keep the
   recursion guard and the re-validation through `validate_seed_incumbent`.
3. *Adaptive fixing rate.* Running means of successful and infeasible rates per kind; draw the
   target uniformly in HiGHS's band, start 0.6; an infeasible sub-MIP at ≤ 1 node halves the
   target and retries once from half the dive depth.
4. *Schedule.* Replace the `!has_incumbent` gate of `dive_batch_eligible` by an LP-pivot budget
   (heuristic pivots ≤ 0.05 × tree pivots + 10 000): RENS while no incumbent, RINS once there is
   one, at the first node of each batch inside the budget; the root-reduced-cost neighbourhood
   once at the root when C.2's ladder exists.
Flag `DISCOPT_MILP_SUBMIP=1` (retires `DISCOPT_RINS` into it, closing its missing audit row).

**Entry experiment** (root-only RENS + RINS, steps 1–3, before the schedule): the 11 instances of
Table 1's heuristics column plus `istra`, `iskar`, `2122`, 20 s direct driver, 2 reps: ≥ 6
instances strictly improve the P6.6 incumbent and ≥ 3 reach the optimum as incumbent (`gt2`,
`iskar`, `kaihu`, `jijia`, `bppc8-02` are the A12/L3a candidates). **Kill:** < 4 improved, any
P6.6-certified row loses its certification, or sub-MIP time > 30 % of the slice on the
both-certified rows → retire the builder (keep `run_sub_mip`). Graduation adds `dcmulti`
certified (the P4.2 RENS failure mode) as an explicit bar.

### C.4 What the ablation could not split: the base machinery (13 of 25 certified by plain HiGHS B&B)

Not a port design — a statement of what §0 item 4 contains, so it is not mistaken for "the LP".
Of the pieces still active in `all_off`, the evidence here points first at **(a) propagation with
implications/cliques plus conflict analysis** (on `gt2`, `fiber`, `nexp-*`, `p0548`, `beavma`
every improving `all_off` incumbent after the first came from node evaluation `T` — the plunge
lands on integral points that Rust's FBBT-only propagation does not reach; the base runs show
183 conflicts on `b-ball`, 9 706 on `neos-911970`) and **(b) the root reduced-cost ladder applied
globally at every incumbent (§A.2, part of C.2 step 1)**; then **(c) randomized rounding with
propagation and an LP re-solve of the continuous part** (`R` is the first incumbent on 6/25 base
runs and FJ `J` on 15/25 — but `fj_off` leaves every tested node count unchanged, so the first
point is not what is load-bearing; what `R` adds is a propagated, LP-repaired rounding at every
dive start while no incumbent exists). Node selection (plunging) and pseudo-costs are the analysis doc's binding
negatives and are not re-proposed. The right experiment for (a) is not an option ablation but a
Rust one: C.2 step 1 plus a clique-table propagator (`presolve/cliques.rs` edges → a
`clique_propagate` pass inside `tighten_bounds_csc_with_row`, bound-neutral by construction)
measured on `gt2`'s 70 k-node tree: bar ≤ 0.25× nodes with the same bound; kill: no change on
`gt2`, `nexp-50-20-1-1`, `fiber`.
