# D2 plan — closing the pure-MILP gap to HiGHS (highs-parity)

Status: **plan, 2026-11-07.** Branch `claude/happy-hamilton-0vvo3b`, PR jkitchin/discopt#2196.
The owner's goal is to eliminate HiGHS. D2 (the pure-MILP route default) is the last
default path that needs it. Evidence base: `highs-parity-d2-analysis.md` follow-ups 1-22,
`highs-parity-d2-highs-ablation.md`, `highs-parity-d2-highs-expert-review.md`,
`highs-parity-d2-phaseB-expert.md`, `highs-parity-d2-node-gmi-expert.md`.

## 0. Where we are

Panel: MIPLIB-38 + M (62 instances), 20 s, `gap 1e-4`.

| | certified per rep |
|---|---|
| HiGHS 1.15.1 | ~59 / 62 |
| Rust at the start of 2026-11-06 | ~32 / 62 |
| Rust before W1 (node GMI + aligned slacks + sub-MIP, all default ON) | ~38 / 62 |
| Rust after W1 (+ cut LP + armed SB, default ON) | **~41-42 / 62** (bundle panel: 125 / 186 solves) |

Rust certified 114 of 186 solves on the bundle panel. 24 instances never certify; by
cause:

| group | instances | what is missing |
|---|---|---|
| G1 near (within ~1 % on both sides) | `b-ball`, `fiber`, `istra`, `kaihu`, `gesa2`, `bppc8-02` | a little more bound or incumbent; time-sensitive |
| G2 no incumbent | `2122`, `enlight_hard`, `neos-2624317-amur`, `issue-2446`, `p0548` | a first feasible point (and, for `issue-2446`, a root LP that finishes) |
| G3 primal-limited | `neos-3118745-obra`, `beavma`, `sp150x300d`, `nexp-50-20-1-1`, `nexp-50-20-4-2`, `neos17`, `gsvm2rl3` | a better incumbent; the bound is at or near the optimum on several |
| G4 dual-limited | `mik-250-20-75-1/2/3/5` | incumbent optimal, bound ~3 % off |
| G5 LP throughput | `neos-911970` | 7 nodes in 20 s |

**Measured non-levers. Do not re-propose them without new evidence:**
- full presolve (the Rust route on HiGHS-presolved models fixes only `p0548`);
- the objective cutoff alone (8 % of node pivots);
- the improving dive (a primal-gap gain, but no certifications);
- original-row c-MIR at nodes (`DISCOPT_MILP_NODE_SEP`, retired);
- RINS as A12 built it (retired);
- re-separating root cuts in the re-entry run (`DISCOPT_MILP_REENTRY_CUTS`: costs the
  `mik` incumbents);
- the A-series heap order;
- GMI off a completed short basis (dense cuts).

## 1. Working rules (what made the last two days productive)

1. **Entry first, on the target rows, 1 rep, 20 s.** Kill the lever on the stated
   criterion. A lever that does not move its target rows does not get a full panel.
2. **Ask HiGHS or SCIP which mechanism certifies a row before building.** Use option or
   separator ablations on the exact rows; `pyscipopt` is installed. Both of the
   session's real wins (node GMI with a global pool; aligned slacks) came from such an
   ablation or a profile, not from a design guess.
3. **Graduate only on the pre-registered `d2_flag_panel.py` panel** (3 reps, bars 1 and
   2a-2e, `--resume`) **plus `flag_onoff_corpus_panel.py`.** Run on a quiet machine:
   concurrent jobs invalidated one panel (follow-up 22).
4. Every lever is a default-OFF flag with an audit row and a regression test. Commit and
   push after each step.

## 2. Workstreams, in order of expected certifications per effort

### W1 — Node-GMI per-node cost (G1, and helps everything)  *expected +3-5*

**Evidence.** The graduated node GMI costs throughput:

- `istra` falls from 13.5-16.4k nodes to 5-5.6k at 20 s, and stays open at 48 although
  the direct probe reaches 48.47 and SCIP certifies it with the same cuts shared.
- `kaihu` and `iskar` certify only because the lift outweighs the loss.

The cost is structural. Every node that uses the pool:

- clones the whole working CSC and appends up to 20 rows;
- re-prepares a dual from the plain basis;
- scans the whole pool densely (`O(pool x n_w)`) to pick violated cuts;
- abstains in thousands of NS checks (`NodeGmiNsAbstain` 3-8k per solve).

**Change.**
- (a) Keep the pool sparse (CSR rows); compute violation with a sparse dot against
  `sol.x`.
- (b) Build the node-local LP by appending rows to a shared `PreparedDual`: reuse the
  parent factor plus row updates, no CSC clone.
- (c) Re-check NS with the open-column repair (`scaled_dual_for_open_columns_csc`)
  before abstaining.
- (d) Give pool rows to the node LP only when the node is in the best `k` of its batch
  or its plain bound is within the lattice step of the next integer.

**Entry.** On `istra`, `iskar`, `kaihu`, `jijia`, `b-ball` and `mik-1`, at 20 s:

- nodes/s must be at least 2x the current node-GMI arm;
- bounds no lower than now.

**Kill:** < 1.5x nodes/s, or any target row loses its bound.

**Size:** 1-2 days. This is bound-neutral per node (same cuts, same verified bound),
so verify it as bound-neutral on fixed seeds first.

**Status 2026-11-07: done. `DISCOPT_MILP_CUT_LP` + `DISCOPT_MILP_SB_ARMED` graduated default ON.**

The entry profile overturned this section's premise. The cost was not the scan, the
CSC clone or NS abstention, which together took < 10 % of node-GMI time. It was the
second, node-local LP re-solve with pool rows: on `istra`, `NodeGmiLp` took 14.2 s
against 7.1 s for `NodeLpSolve`. Changes (a)-(d) above were therefore not built.

Following three HiGHS-expert consults (`highs-parity-d2-w1-cut-lp-expert.md`), the pool
moved into the shared LP with HiGHS's rules:
- pool score, adaptive threshold and parallelism filter;
- per-solve aging on |dual| > tol;
- deletion past age 10;
- the length cap.

Measured record:

| step | result | verdict |
|---|---|---|
| fold hot cuts, append-only (`FOLD=200:3`) | `b-ball` certified in 2.8 s, `istra` 2.2x; lost `bppc8-02`, `fiber` 2x slower, `gsvm2rl3`/`beavma`/`obra` ~10x fewer nodes | not HiGHS: no removal |
| + global nnz cap (`:0.1`, `:0.3`) | erased the gains; `bppc8-02` still lost with 39 rows | killed (not a HiGHS mechanism) |
| instrumentation | 82-96 % of folded rows basic at node optima, on winners and losers alike | removal is the missing mechanism |
| HiGHS policy, batch-level aging | ages never grew: `bppc8-02` 197 folded, 0 deleted | fixed: per-node-solve aging |
| per-node aging | **false certificate**: `p0548` 9154 > optimum 8691. Deleting a row dropped a slack column other folded rows used. | fixed: `delete_cut_rows` rewrites them by substitution; regression test fails on the old behaviour |
| aging on nonbasic only | degenerate zero-dual rows never aged | fixed: also require \|dual\| > 1e-7 (HiGHS) |
| root-basis fallback for invalidated stored bases | 3,644 fallbacks on `bppc8-02`, worse everywhere | reverted |
| recent-basis fallback (HiGHS's actual queue-node start) | churn (2,443 folded / 2,205 deleted on `enlight8`), 116k foreign starts, lost `kaihu`/`fiber` | reverted: HiGHS pays one foreign start per dive, here it lands per node |
| lazy deletion (free the slack when a stored basis blocks) | neutral to positive; `enlight8` keeps its incumbent | kept |
| §5 panel, CUT_LP alone | certified 118 -> 124, nodes -48 %, cert-clean | passed |
| corpus differential, CUT_LP alone | lost `ex1266` (OFF 3/3, ON 1/3): folded rows made strong-branch probes degenerate, and probes never reached the cost perturbation (21,856 Bland pivots vs 21 attempts) | fixed: `DISCOPT_MILP_SB_ARMED` |
| `ex1266` entry, + SB_ARMED | 5/5 in 2.1 s (both off: 11.6 s), Bland ~23k -> 812 | passed |
| **§5 panel, bundle** | **certified 117 -> 125, nodes -48 %, median primal gap 0.105 -> 0.041, 2 dual losses (cap 2), no incumbent lost** | **passed** |
| corpus differential, bundle | cert-clean; flagged `nvs24` certifies 5/5 in both arms at 17-19 s (20 s limit) | timing edge |

Still open from W1:
- `enlight8` and `bppc8-02` want node-local lifts that a shared LP does not reproduce.
- HiGHS's start-of-solve perturbation and cost-shifting anti-cycling (third consult,
  items 2-6) belong to W5.

### W2 — First incumbents (G2)  *expected +2-3*

**Evidence.**
- `2122` has bound = optimum and no incumbent: any feasible point certifies it.
- `p0548` certifies in 323 nodes once it has HiGHS's presolved model.
- The sub-MIP's RENS never fires on these rows: it needs a fractional batch candidate
  and an LP that finishes.
- HiGHS's first incumbents come from feasibility jump (`J`, 15/25 base rows) and
  randomized rounding (`R`).

**Change.**
- (a) **RENS at the root**, right after the root cut loop, from the root LP point,
  whenever there is no incumbent. The `try_sub_mip` path exists; only the call site is
  new.
- (b) **Feasibility jump** (Luteberget-Sandberg; HiGHS `HighsFeasibilityJump`): a
  weighted local search over integer assignments, run before the tree, at an effort cap
  of 5 % of LP iterations. A new module of ~400 lines.
- (c) **Randomized rounding with propagation** at the first `k` batch candidates.

**Entry.** On the 5 G2 rows, 20 s, count the rows with an incumbent: OFF has 0 of 5 on
2 of them.

**Kill:** fewer than 2 of 5 rows gain an incumbent.

**Size:** (a) half a day; (b) 2-3 days; (c) 1 day.

### W3 — Sub-MIP schedule (G3)  *expected +2-4*

**Status 2026-11-07: retired on its entry.** `DISCOPT_MILP_SUBMIP_SCHED` (HiGHS's
schedule plus fix-and-propagate neighbourhood at the adaptive target rate) raised the
number of sub-MIP attempts but almost never improved the incumbent. The median G3
primal gap stayed at 0.250 -> 0.250 (`d2-analysis/w3_entry.jsonl.gz`). HiGHS's
sub-MIPs win through the sub-solver (presolve, cliques, propagation), not the schedule.
A stronger sub-solve, i.e. presolve on the neighbourhood, is the lever to test next on
these rows.

W2 status: randomized rounding (`DISCOPT_MILP_RAND_ROUND`) passed its entry (2 of 4 G2
rows gain an incumbent). It is cert-clean and primal-positive on the panel, but fails
2b alone, so it stays OFF pending a bundle (audit row). Entry probes on the G2 rows
showed HiGHS's first incumbents there come from randomized rounding, a root sub-MIP and
the tree, not from feasibility jump, so W2b (feasibility jump) is deprioritised.

**Evidence.**
- The graduated sub-MIP fires only 2-6 times per 20 s solve (stride 16 batches, with
  backoff).
- Even so it took `istra` 56 -> 49, `beavma` 512k -> 403k and `bppc8-02` 528 -> 513.
- HiGHS runs RINS/RENS at every plunge start under an LP-iteration budget (10k free,
  then 5 % of tree LP iterations; ablation §C.3), and its final incumbent is a sub-MIP
  point on 20/25 rows.

**Change.**
- Replace the batch stride with an LP-iteration budget.
- Fire on the best-bound node of every batch while inside the budget.
- Use an adaptive fixing rate (start 0.6; halve on an infeasible sub-MIP, raise on
  success).

**Entry.** On the 7 G3 rows, 20 s, median primal gap and certified count.

**Kill:** the median primal gap is not lower, or a G3/G1 row loses certification.

**Size:** 1 day.

### W4 — `mik`: in-tree c-MIR with a global pool (G4)  *expected +2-4*

**Evidence.**
- HiGHS certifies `mik` only with node separation (`node_cuts_off` costs 10-14x).
- On `mik-1`, SCIP applies 3,865 `aggregation` and 3,428 `gomory` cuts, most of them in
  the tree.
- The retired Rust node c-MIR used the original rows only, aggregated at most 2 rows,
  and folded into a capped matrix pool.
- Node GMI moved `mik` bounds by 35-70 units, which is not enough.

**Change.** Separate aggregated c-MIR at nodes into the **same global aging pool as
node GMI** (that infrastructure exists), not the shared matrix. Use deeper aggregation
(`CMIR_MAX_AGGR` 2 -> 6) with variable-bound substitution.

**Entry, before any code.** A SCIP ablation on `mik-1` and `mik-5`, presolve off,
`gomory` vs `aggregation` vs both, in-tree only. This shows which family is necessary.
Then run the Rust probe on the 4 `mik` rows.

**Kill:** no `mik` bound improves by > 0.5 % of the gap.

**Size:** 1-2 days.

**Status 2026-11-07: entry done; the existing separator is not the lever.**

SCIP ablation (`d2-analysis/w4_scip_mik_ablation.*`, presolve off, 30 s):

| SCIP arm | `mik-1` | `mik-5` |
|---|---|---|
| default | certified, 23.6 s | certified, 15.1 s |
| **Gomory off (c-MIR/aggregation only)** | **certified, 8.2 s** | **certified, 6.6 s** |
| aggregation off | certified, 13.1 s | certified, 20.2 s |
| both off | open, 11 % gap | open, 9.6 % gap |
| both root-only | certified, 27k nodes | certified, 15k nodes |

Aggregation is the family that matters, and SCIP's root alone gets most of the way: its
c-MIR-only root bound on `mik-1` is -51,173 in 0.4 s (LP -59,157, optimum -49,716).
The Rust route's root loop runs GMI only (`SepCmirCuts` 0). Turning on the #2107
aggregated c-MIR separator there (temporary flag, removed) generated 712-966 cuts per
instance. The 20 s bounds did not move (`d2-analysis/w4_d2_cmir_entry.jsonl.gz`):
`mik-1` -51,192 -> -51,281, `mik-3` -53,609 -> -53,790, `mik-2` and `mik-5` within
0.1 %. Our whole 20 s tree ends about where SCIP's root does.

So W4 needs a separator rebuilt to SCIP's design: path aggregation over continuous
columns with variable-bound substitution, several scaling factors (deltas) per
aggregation, and complementation. A rewrite of `lp/cmir.rs`'s aggregation is planned
with a root-bound differential against SCIP as its entry, not a flag flip.

**Update 2026-11-07 (evening): port started; the separator is the root lever, but route
adoption is blocked by LP numerics.** The third consult is in the commits: HiGHS's
`HighsCutGeneration` / `HighsPathSeparator` against our `cmir.rs`.

Root bound on `mik-1` / `mik-5` (`d2-analysis/w4_root_bound.py`):

| arm | `mik-1` | `mik-5` |
|---|---|---|
| GMI only (today) | -51,757 | -53,674 |
| our c-MIR, old delta rule | -52,544 | -54,463 |
| + HiGHS delta selection (`DISCOPT_MILP_CMIR_HIGHS`) | -51,558 | -53,483 |
| + flip pass (`DISCOPT_MILP_CMIR_FLIP`) | -51,532 | -53,483 |
| + lifted mixed-integer cover (`DISCOPT_MILP_LIFTED_MIC`, brute-force tested) | -51,493 | -53,381 |
| SCIP aggregation only | -51,173 | -52,786 |
| HiGHS whole root | -50,587 | -52,372 |

The old rule tried the first 8 integer coefficients by index; HiGHS tries every
integer off its bound.

Route adoption (`DISCOPT_MILP_D2_CMIR` plus the above): the §5 panel failed
(`w4_cmir_bundle_panel_r3.json.gz`).
- Cert-clean, but certified 124 -> 120 and 4 dual losses.
- Lost `dcmulti` and `bell5`; gained `gesa2` and `p0548`.

The failure is LP numerics, not the cuts (`w4_cmir_followups.jsonl.gz`).
- With c-MIR rows, the root re-solves end in thousands of sub-tolerance-repair
  primal pivots: `dcmulti` spent the 10 s root slice there, with 6 iteration-limit
  verdicts.
- HiGHS-style power-of-two cut scaling fixed `dcmulti` (certified in 7 s) and `iskar`
  (13 -> 3 s).
- It moved other trajectories: `gesa2` open again, and `istra`/`kaihu` slower.

**Re-panel 2026-11-08, on the incremental-pricing engine (`LP_PRIMAL_INCR_PRICING=1`
in both arms; `w4_cmir_bundle_incr_panel_r3.json.gz`): fails again, the same way.**
- Cert-clean (372 checks); certified 123 -> 123 (gains `gesa2` and `p0548` 3/3; loses
  `istra` and `kaihu` 3/3); nodes -19 % where both arms certify.
- 4 dual losses (cap 2). No incumbent in any rep on `enlight8`, `nexp-50-20-1-1`,
  `beavma`, `istra`.
- The engine did not remove the root-LP numerics signature; W5 (a) and (d) were retired
  on their entries.

Remaining W4 steps (W5 (a) and (d) are retired, so these no longer wait on them):
- small-coefficient removal plus scaling validated on a panel;
- deeper path aggregation (expert step 4);
- tableau rows through the c-MIR pipeline (step 5).

### W5 — LP engine throughput (G5, plus `issue-2446`)  *expected +1-2, broad speed*

**Evidence.**
- `neos-911970`: 7 nodes in 20 s.
- `issue-2446`: the whole 20 s goes to root LP work (111k pivots, 77k in phase 1,
  2,049 refactorizations); HiGHS solves the instance in 0.8 s.

**Change.**
- Profile both LPs against HiGHS (pivots, refactorizations, per-pivot cost).
- Candidates: the bound-flipping ratio test in the dual, the refactorization trigger,
  phase-1 choice (a dual phase 1 on a bounded problem), and crash basis.
- The fix follows the profile; no lever is chosen in advance.

**Entry:** the profile itself. **Kill:** none; this is diagnostic until a cause is
named.

**Size:** 1-3 days.

**Status 2026-11-07: entry done; no quick win, the engine project is scoped.**

HiGHS 1.15.1 reference, 1 thread:

| instance | HiGHS | discopt |
|---|---|---|
| `neos-911970` | optimal 54.76 in 11 s, 1,370 nodes, about 72 LP iterations per node | 1,375 nodes in 20 s, incumbent 103.92, about 285 iterations per node |
| `issue-2446` | optimal at the root in 0.12 s (presolve plus randomized rounding plus cuts) | — |

So these two rows are not pure LP-speed rows, but the per-pivot gap is real.

1. **Cold-solve warm-start cascade (found and measured; the fix is retired).**
   - A cold two-phase node solve exports a basis short by every artificial left basic
     at zero. The next warm start refuses it (`DualPrepRejectShape`) and cold-solves
     too.
   - The cut LP's folded rows made this common: on `neos-911970`, 628 rejections and
     247k phase-1 pivots.
   - `DISCOPT_MILP_NODE_EXPEL_ARTIFICIALS` (`SimplexOptions::expel_zero_artificials`
     on node LPs) took the rejections to 0 (`d2-analysis/w5_expel_entry.jsonl.gz`).
   - Its §5 panel (`w5_expel_panel_r3.json.gz`) was cert-clean but neutral:
     certified 126 -> 125, nodes +1 %, `p0548` incumbent lost once. It was retired
     per §5.
2. **Per-pivot cost.**
   - On the pure LP relaxation of `issue-2446`, discopt takes ~2,000 pivots at
     ~50 µs/pivot; HiGHS takes 973 at ~9 µs/pivot.
   - Our cold solves are a two-phase primal that re-prices every column every pivot
     (a full BTRAN plus an O(nnz) sweep, `primal.rs` `PriceSweep`).
   - HiGHS cold-solves with the dual simplex and updates reduced costs from a
     hyper-sparse pivot row.
   - Node LPs under the cut LP are also ~9x slower per solve on `neos-911970`
     (23 ms vs 2.7 ms), because lazily deleted cut rows stay in the matrix.

The engine work, in order, each with an R1-LP-panel entry (P2.6) and a kill criterion:
- (a) cold solves through the dual simplex from a slack basis, with bound flipping;
- (b) a row-wise copy of A for hyper-sparse PRICE in the dual;
- (c) longer FT-update refactorization intervals;
- (d) physical removal of lazily deleted cut rows via one exchange pivot per stored
  basis (the expert's option b).

**Engine record 2026-11-07 (evening).**
- The primal work in the tree is mostly the #1595 sub-`tol` repair. Every repair
  measured was triggered only by infinite-room columns (open slacks, unbounded
  continuous columns), and every one pivoted: `bppc8-02` made 1,959 repairs and 312k
  primal pivots in 20 s.
- `DISCOPT_MILP_SUBTOL_INF_SKIP` skipped those repairs on node LPs, relying on the NS
  ladder for the bound. It was retired on its entry
  (`d2-analysis/eng_subtol_skip_entry.jsonl.gz`).
  - Node throughput rose (`bppc8-02` 2.4k -> 7.0k, `obra` +60 %).
  - Bounds and incumbents got worse (`beavma` 356,127 -> 353,010, `gsvm2rl3` 0.253 ->
    0.235, `bppc8-02` incumbent 513 -> 523), and `istra`/`kaihu` slowed 1.7-1.9x.
  - The repairs find the true LP optimum; HiGHS can accept the sub-`tol` vertex only
    because it takes the LP objective as its node bound, where discopt needs a
    rigorous NS bound.
- So the lever is a cheaper repair, not skipping it: the primal simplex's
  full-column re-pricing per pivot (`primal.rs` `PriceSweep`) must become an
  incremental, row-wise, hyper-sparse reduced-cost update. That is engine item (b),
  applied to the primal.

**Engine step 2 (2026-11-07, late): row-wise pivot row for the incremental pricing.**

What was built (under `DISCOPT_LP_PRIMAL_INCR_PRICING`, no new flag). The handoff's
`eng_rowprice.patch`, with the HiGHS expert's fixes:
- the CSR copy is built lazily on the first incremental update and cached on the
  `Simplex` (HiGHS keeps `ar_matrix_` persistently, `HEkk.cpp` 2276), not once per loop;
- HiGHS's `updateVerify` (`HEkkPrimal.cpp` 2539): the pivot read off the row must agree
  with the FTRANed column's within 1e-7 relative, or the basis is refactorized;
- counters for both arms (`PrimalRowPriceSparse` / `PrimalRowPriceDense` /
  `PrimalRowPriceVerifyFail`) and a `PivotRowPrice` phase timer. `PriceSweep` times only
  the entering-column scan, which this step does not change.

Rust lib tests: 1033/1033 with and without the flag.
- The #2202 test's premise (the inherited start ends `Numerical`) holds only in the
  default arm: incremental pricing decides that LP on its own (already true with step 1
  alone).
- So that arm now asserts the inherited start is decided correctly, against HiGHS.

Entry (`d2-analysis/eng_rowprice_entry.jsonl.gz`):
- Setup: 9 rows, 2 reps, 20 s, the flag ON in both arms.
- Arms: A' = the same build with the row-wise branch forced off (measurement only); B =
  step 2. The `.so` was swapped per solve, arm order alternating (`eng_rowprice_ab.sh`).
- Pivot-row update time, A' -> B (ms):

  | row | A' | B |
  |---|---|---|
  | `p0548` | 516 / 489 | 95 / 93 |
  | `2122` | 6,032 / 5,874 | 1,473 / 1,539 |
  | `enlight_hard` | 381 / 367 | 92 / 89 |
  | `obra` | 119 / 117 | 30 / 29 |
  | `sp150x300d` | 2,208 / 1,871 | 881 / 1,036 |
  | `bppc8-02` | 9,256 / 9,274 | 6,237 / 6,168 |
  | `beavma` | 687 / 752 | 451 / 447 |

- Per update that is 2-6x cheaper (`p0548` 13.8 -> 2.3 us).
- The row-wise arm took 72 % (`beavma`) to ~100 % of the updates.
- The verify fired 1-2 times, on `2122` only, in both arms.
- Statuses were unchanged and node counts within run noise; the rows are wall-limited
  elsewhere (node LP total time is level).
- §5 panel, steps 1+2 together (`d2-analysis/eng_incr_rowprice_panel_r3.json.gz`, 3 reps,
  20 s, 62 instances; fired 7.8M row-price updates ON, 0 OFF):
  - BAR 1 passes (372 checks); 2b (nodes over the 38 certified both ways 47,695 -> 40,050,
    -16 %) and 2c (0 dual losses) pass.
  - 2a (certified 125 -> 123), 2d and 2e fail, and every loss is `p0548`: 0/3 ON vs 2/3
    OFF, no incumbent within the 13 s engine share in each failing rep.
  - Run alone, both arms certify it (OFF 6.1 s, 3,309 nodes; ON 10.1 s, 7,796 nodes),
    so ON is faster per node and finds the incumbent later. OFF's own third rep ends
    the same way (775 nodes, no incumbent).
  - Verdict: does not graduate. Kept as a documented opt-out pending the owner (audit
    row). A `p0548`-robust primal lever is the precondition for a third panel.

**W5(a) cold LP solves through the dual simplex: built, entry failed, retired
(2026-11-07).** `DISCOPT_LP_DUAL_COLD` was designed with the HiGHS expert:
- the dual simplex from the all-logical basis of `A x - s = 0`;
- HiGHS's auxiliary-LP dual phase 1 when that basis is not dual feasible;
- only audited Optimals and verified Farkas rays accepted;
- basic logicals exchanged out, so the exported basis is full;
- short warm starts routed to it.

Entry (`d2-analysis/w5a_dual_cold_entry.jsonl.gz`, 10 rows, 2 reps, 20 s):
- Phase-1 pivots fell 85 % and short-basis refusals vanished (`bppc8-02` 1,640 -> 15).
- Throughput collapsed: `neos-911970` 874 -> 45 nodes, `bppc8-02` 1,760 -> 1,150.
- Controls were unchanged.

With short starts left on their old path (`..._coldonly_entry.jsonl.gz`) it never fired:
0 attempts on 4 rows. On these rows every cold primal solve is the fallback of a short
warm start, and the primal repair from that start, phase 1 and all, is far cheaper than a
dual solve from scratch.

The ceiling was already measured: removing the short bases at their source
(`NODE_EXPEL_ARTIFICIALS`) was panel-neutral. Retired, flag and implementation.

What would still pay is a *warm* dual start from the short basis (completed with
logicals for its uncovered rows), but the EXPEL ceiling says not much.

**W5(d) physical removal of lazily deleted cut rows: built, entry failed, retired
(2026-11-07).** `DISCOPT_MILP_CUT_LP_EXCHANGE` deleted every aged folded row (not only
rows whose slack is basic in every stored basis). Each frontier basis with a deleted
slack nonbasic first took the slack in by one exchange pivot (the expert's option b,
memoized per shared basis).

Entry (`d2-analysis/w5d_cut_lp_exchange_entry.jsonl.gz`, 10 rows, 2 reps, 20 s):
- The matrix did shrink: relaxed rows went to 0 and node-LP time fell.
- Throughput fell broadly:
  - `neos-3610051-istra` lost its certificate (3.8 s -> time-out);
  - `enlight8` nodes /3, `sp150x300d` /2, `beavma` -27 %;
  - `b-ball` 2.3x slower.
- Only `bppc8-02` gained (nodes x2), with `neos-911970` level.
- The cost is structural. Every deletion epoch factorizes every frontier basis that
  needs an exchange (14k factorizations on `b-ball`). Each failed exchange (up to 445
  per run) drops a basis, and that node then cold-solves (shape refusals up 5-50x).
- HiGHS avoids both because its open nodes store no basis.

So the lazily relaxed rows are cheaper to carry than to remove this way. What would
change that is a tree that keeps bases only on the dive stack, as HiGHS does. That is
W6's territory, not a W5 lever. Retired, flag and implementation.

**D3 follow-up: `DISCOPT_MILP_TOL_ROBUST` (2026-11-08).** This is the fix for the red
`test_nlpbb_unscaled_refine` under the D3 master default (handoff,
"Full-suite result").

The driver's emptiness proofs now hold for the set its incumbent check accepts:
- FBBT runs over the tolerance-relaxed rows;
- a Farkas ray fathoms only if its bound survives `sum |y_i| eps_i`;
- an LP infeasible but not robustly so is re-solved over relaxed slacks.

Design from the HiGHS-expert consult; HiGHS keeps the same consistency with feastol
slacks at each test site.

Evidence so far:
- Entry, on the fixture `portfol_roundlot_oa_master_tolerance_feasible.npz`:
  INFEASIBLE -> OPTIMAL 0.02829 (residual 5e-7) through both entry points.
- Corpus differential (`d2-analysis/tol_robust_corpus.jsonl.gz`, 132 instances,
  20 s): 0 incorrect, 244 incumbent feasibility checks, no bound past an oracle;
  certified 105 -> 105, gaining `portfol_roundlot` and losing `nvs24`.
- `nvs24` is a timing artifact. OFF certified at 18.5 s of 20 under load 7. Rerun
  alone, 3 reps (`..._nvs24_rerun.jsonl.gz`), each arm certified 2/3: OFF lost
  rep 0 and ON lost rep 2.

### W6 — Plunging with a hot LP (expert R1)  *expected: the long tail*

Only if W1-W5 leave the count below ~52/62. This is the largest build: depth-first
child evaluation on the parent's factor, an objective limit, and heuristics inside the
plunge. The improving-dive and node-GMI experience says to do it only with the hot LP
(no re-prepare per node); a heap-order change alone is a measured non-lever.

## 3. Sequence and gates

| step | work | expected count | gate |
|---|---|---|---|
| 1 | W1 (node-GMI cost) | ~41-43 | entry, then panel + corpus, graduate |
| 2 | W2a (root RENS) + W3 (sub-MIP schedule) | ~44-47 | entry per lever, one bundle panel |
| 3 | W4 entry (SCIP on `mik`), then build | ~47-50 | entry, panel |
| 4 | W2b (feasibility jump) | ~48-51 | entry, panel |
| 5 | W5 profile -> fix | ~50-53 | entry, panel |
| 6 | Re-measure against HiGHS; decide W6 or accept the residual | — | owner |

After each graduation: re-run the D2 gate (Rust vs HiGHS on MIPLIB-38 + M, 1 rep, 20 s)
and update §0. The parity bar for flipping `DISCOPT_LP_MILP_BACKEND` to `rust` is the
plan's original R2 bar: certified >= HiGHS on the panel, 0 lost, wall <= 2x. Short of
that, the owner decides whether to accept the residual (as D3 did).

## 4. Risks

- **Trajectory variance.** Single-rep counts swing by +/- 1-2 rows (`gt2` 1-ulp,
  `khb05250` formulation). Entries use the target rows; graduation uses 3 reps.
- **Load.** Never run two panels at once, and never run gates during a panel (follow-up
  22).
- **Container restarts.** They killed two panels; use `--resume` (added in `0a05d3c1`).
