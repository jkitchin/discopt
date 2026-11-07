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
