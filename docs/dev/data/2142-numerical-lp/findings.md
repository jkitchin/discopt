# #2142 numerical-verdict node LPs on cstr's hull model — findings

**Verdict: killed by the pre-registered criterion (`PREREG.md`).** On the homog hull cstr tree
no node LP ends with a numerical verdict, so a robust re-solve of such nodes is the identity on
the tree's bound. Nothing to implement.

Runs: `probe.py homog 10000` (`probe_homog_10000.log`, deterministic, `PYTHONHASHSEED=0`,
`DISCOPT_PROFILE=1`, 29 executed assertions) and the 300-node smoke (`smoke_homog_300.log`).
The 10k bound reproduces `../2142-exact-hull/arm_homog_10000.log` bit for bit:
**1.6107840610272945**.

## 1. What a numerical verdict does (code path)

`crates/discopt-core/src/bnb/spatial_tree.rs`

- `verdict_for(LpStatus::Numerical) == NodeVerdict::Undecided` (also `IterLimit`,
  `Unbounded`). Only a Farkas-certified `Infeasible` prunes; `Undecided` never does.
- An undecided node, with `undecided_unpropagated_retry` (default ON, #1995), is re-solved
  once over its *pre-propagation* box when propagation changed the box; a decided re-solve
  replaces the node (box included).
- Still undecided -> `n_undecided += 1`; the node is **branched on the widest original column
  at the midpoint**, both children inherit `parent_bound` (the bound the node itself inherited;
  it contributes nothing of its own) and the basis the node started from; when no column is
  branchable the region closes with that inherited bound (never `+inf`). So an undecided node
  costs one wasted node and a non-improving split, and its subtree keeps the ancestor's bound
  until a descendant's LP decides.

`LpVerdictNumerical` (`lp/simplex/primal.rs:3473`, `Simplex::assemble`) is **incremented once
per cold-primal attempt**, not per node. A cold node LP (`solve_spatial_node_with_cuts` ->
`solve_lp_cols_scaled` -> `solve_lp_warm_csc` -> `solve_lp_warm_csc_inner`) runs this chain:

1. scaled cold primal; on `Numerical`/`IterLimit` -> #85 dense retry (cold, dense LU);
2. still failed -> #649 unscaled retry on the exactly reconstructed matrix (own dense retry);
3. (#671 hardened retry only when its flag is on — default OFF, not active here);
4. node-level `Numerical` -> #1995 pre-propagation-box re-solve of the whole chain.

Each failed attempt counts. The dual-proof global-rows LP (`proof_with_global_lp`, one per
certified-infeasible node, 3,390 here) and strong-branching child LPs go through the same
counter.

## 2. Measurement (homog, 10k nodes)

| quantity | value |
|---|---|
| `LpVerdictNumerical` (attempts) | 1,122 |
| `LpDenseRetries` / rescues | 561 / **0** |
| node-level `Numerical` returns (`WarmVerdictNumerical`) | 19 |
| #1995 pre-propagation re-solves / decided | 19 / **19** |
| kernel `n_undecided` | **0** |
| kernel `n_uncertified` (optimal, safe bound `-inf`) | **0** |
| nodes with an LP / propagation-fathomed before any LP | 9,035 / 965 |
| node status histogram | optimal 5,640, infeasible 3,395, numerical 0 |
| observer windows holding >= 1 Numerical attempt | 293 / 9,035 (267 ended optimal with a finite certified bound, 26 certified infeasible) |
| attempts per such window | 2: 65, 4: 207, 6: 6, 8: 13, 12: 2 (always even: scaled + its dense retry fail together) |
| pivots in those windows | 27,043 of 360,396 (7.5 %) |

Frontier reconstruction (effective bound `max(own, parent)` per node, open frontier = nodes
with < 2 observed children): the 965 unseen children below the kernel bound equal the kernel's
965 propagation fathoms exactly; **one** node holds the global bound 1.6107840610272945, it is
`optimal` with its **own** finite certified bound (not inherited), and **no undecided or
uncertified node exists anywhere in its ancestry** — there are none in the tree at all.

Export fidelity: 8 decided node LPs dumped from the observer were re-solved by HiGHS and
SoPlex (pyscipopt `LP`) and reproduce the in-house objective to 1e-6 — so had any failed node
LP existed, the dump path would have judged it faithfully. None existed, so the "re-solve the
failed LPs with a robust solver" arm had an empty input set (`re-solve outcomes: {}`).

## 3. Decision

Pre-registered kill condition 1 holds: `n_undecided = 0`, no bound-holding node is undecided
or uncertified. The hypothesis "discopt loses bound to node LPs that end with a numerical
verdict where SoPlex would solve them" is **false on homog cstr**: the 1,122 verdicts are
*recovered attempts* inside the retry chain (the #649 unscaled retry and, for 19 nodes, the
#1995 re-solve), and every one of the 9,035 node LPs ends `Optimal` with a finite
Neumaier-Shcherbina bound or certified `Infeasible`. The 20k emulation was not run (the
pre-registration says not to), and no `DISCOPT_*` flag, kernel change or test is proposed.

The same holds for the big-M default and homog 20k from the existing logs (`n_undecided = 0`
in `arm_bigm_10000.log`, `arm_homog_20000.log`). Only the (already killed) aggregation arm
had undecided nodes (105 / 10k, `arm_aggr_10000.log`), 1 % of its nodes, and that arm is dead
on its own pre-registration.

Consequence for #2142: the remaining gap to SCIP on this spec is not in LP robustness. The
cost of the numerical attempts is a wall-time matter only (<= 7.5 % of pivots sit in windows
with a failed attempt; the dense retry rescued 0 of 561 here) and is out of scope for the bound
question; recorded, not pursued.
