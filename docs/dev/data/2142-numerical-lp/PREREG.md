# #2142 numerical-verdict node LPs — pre-registration

Written 2026-11-05 before any run in this directory (CLAUDE.md §4).

## Hypothesis under test

H: discopt's spatial tree on cstr's hull model loses bound to node LPs that end with a
*numerical* verdict (`LpVerdictNumerical` under `DISCOPT_PROFILE=1`), where a robust LP
(SoPlex/HiGHS) would decide them. Replacing those node LPs by a robust re-solve would raise
the deterministic 20k-node bound.

## What the code says (step 1, read before this file was written)

`crates/discopt-core/src/bnb/spatial_tree.rs`: `verdict_for(LpStatus::Numerical)` is
`NodeVerdict::Undecided`. An undecided node is never pruned: with `undecided_unpropagated_retry`
(default ON, #1995) it is re-solved once over its pre-propagation box; if still undecided it
is counted in `n_undecided`, branched on the widest original column at the midpoint, and both
children inherit `parent_bound` (the node's own inherited bound; no bound improvement). When no
column is branchable the region closes with that inherited bound.

`LpVerdictNumerical` is incremented in `Simplex::assemble` (`lp/simplex/primal.rs:3473`),
the cold primal's single exit, i.e. once per *cold primal attempt*, not once per node. A cold
node LP (`solve_spatial_node_with_cuts` → `solve_lp_cols_scaled` → `solve_lp_warm_csc_inner`)
can register up to four attempts — scaled, scaled dense-retry (#85), unscaled retry (#649),
unscaled dense-retry — and still end `Optimal`/`Infeasible`. The dual-proof global-rows LP
(`proof_with_global_lp`, one per certified-infeasible node) and OBBT probes count there too.

## Evidence already on disk (prior arms, not re-run)

| arm (det, seed 0) | nodes | `LpVerdictNumerical` | kernel `n_undecided` | node-status `numerical` |
|---|---|---|---|---|
| bigm  (`2142-exact-hull/arm_bigm_10000.log.gz`)  | 10k | 176   | **0** | absent |
| homog (`2142-exact-hull/arm_homog_10000.log.gz`) | 10k | 1,122 | **0** | absent |
| homog (`2142-exact-hull/arm_homog_20000.log.gz`) | 20k | 4,920 | **0** | absent |
| aggr  (`2142-scip-hull-ablation/arm_aggr_10000.log.gz`) | 10k | 2,430 | 105 | 105 |

So on the homog arm the counter the hypothesis rests on counts *recovered attempts*: no node
ended undecided. H predicts bound-holding nodes whose own LP decided nothing; the logs already
say there are none on homog.

## Decisive experiment (`probe.py homog 5000`)

One deterministic homog cstr solve, 5k nodes, `PYTHONHASHSEED=0`, `DISCOPT_PROFILE=1`, with
the kernel `node_observer` capturing every node LP's id/parent/status/bound/obj and the full
assembled LP of every node that ended **undecided** (`numerical`/`iter_limit`/`unbounded`)
or **uncertified** (`optimal` with safe bound `-inf` — the other way a solved node LP
contributes nothing and the children inherit the parent bound). The script reconstructs each
node's effective bound (`max(own, parent)`), the open frontier, and the nodes holding the
global bound, and asserts its reconstruction reproduces the kernel's reported bound (§6
self-check). Every dumped LP is re-solved with HiGHS (and SoPlex via pyscipopt).

## Pre-registered decision

- **Kill** if the 5k run has `n_undecided = 0` and no bound-holding node is undecided or
  uncertified: then a robust re-solve of numerical-verdict node LPs is the identity on the
  tree's bound and there is nothing to implement. The 20k emulation is not run.
- **Kill** also if undecided/uncertified nodes exist but HiGHS/SoPlex decide fewer than half
  of them, or the decided ones give bounds no higher than the inherited bound (robust solver
  gains nothing).
- Otherwise (undecided/uncertified nodes hold the frontier AND a robust solver gives them a
  better bound): emulate the robust re-solve in a scratch Rust copy (retry chain extended),
  run homog 20k deterministic; **recommend implementing** iff the 20k bound exceeds
  **2.7875** (big-M default @20k, `arm_bigm_20000.log.gz`); otherwise kill.

Secondary (reported, not a decision input): the walltime share of the recovered numerical
attempts — not a bound question and the machine is loaded (4 cores, other panels), so it is
reported from pivot counts, not seconds.
