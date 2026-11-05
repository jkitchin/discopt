# #2142 scipx: what SCIP does in-tree on cstr big-M that discopt does not

Spec: scratchpad/fable2/cstr_bigm_spec.npz (105 cols, 20 int, 235 <= rows, 5 squares, 24 BLF).
Runner: scratchpad/scipx/run.py (= fable2/e20_scip.py + stats). All SCIP arms deterministic, 20k nodes.

## Log (appended as I go)

### SCIP default statistics at 20k nodes (x_default_stats.log)
- dual 2.3927 (reproduces e20n_default20k bit-for-bit). nruns=1 (no restart).
- Tree: 20000 nodes = 10982 internal + 9018 leaves; **infeasible leaves 7182**, objective leaves 1836; nodes left 1965; max depth 42.
- Presolve: 30 vars deleted (28 aggregated by cons_linear: V1..V5 etc.), 117 conss deleted, 105 coef chg,
  55 upgrades (47 varbound, 8 setppc), probing: 22 bound chg, 310 implications, 18 cliques.
- In tree: nonlinear cons: 1.18M domreds, 910 cutoffs, 266k cuts found / 216k applied (ENFOLP, not separators);
  linear: 341k domreds, 461 cutoffs; genvbounds 10.9k domreds 131 cutoffs; pseudoobj 3.2k/8.
- Conflict analysis: infeasible-LP calls 5682, 479 dual-ray conflicts (460 local, 19 global).

### Single ablations (20k nodes)
| arm | dual |
|---|---|
| default | 2.393 |
| presolve off (== probing+presolve off, bit-identical) | 2.119 |
| conflict off | 1.854 |

### Emulation + conflict sub-ablations (20k nodes; logs scipx/a_*.log)
"emu" = presolve off, conflict off, separators off, OBBT/genvbounds/redcost/rootredcost/vbounds/probing/symmetry off
(keeps LP, cons_linear + cons_nonlinear propagation, pseudoobj, heuristics, SCIP branching).
| arm | dual @20k |
|---|---|
| emu | 1.728 |
| emu + conflict on | **2.273** (+0.545) |
| emu + presolve on | 2.091 (+0.363) |
| presolve off + conflict off (rest default) | 1.730 |
| default, conflict/useinflp=c (conflict graph only, NO dual-ray proofs) | 1.854 (== conflict off 1.8536) |
| default, conflict/useinflp=d (dual-ray proofs only) | 2.3927 (== default, bit-identical) |
| default, conflict/uselocalrows=false (proofs from GLOBAL rows only) | 2.375 |
| default, conflict/allowlocal=false | 2.3927 |
| default, conflict/separate=false | 2.3927 |
| default, conflict/repropagate=false | 2.3927 |
=> 100% of conflict analysis's value is the **dual-ray (Farkas) proof constraint** of infeasible node LPs
(conflict_dualproofanalysis.c: SCIPconflictAnalyzeDualProof -> tightenDualproof -> createAndAddProofcons;
proof built in conflict_general.c SCIPgetFarkasProof). Proof cons are created propagate-only
(initial/separate/enforce/check all FALSE, propagate TRUE), so the gain is via PROPAGATION in other nodes.
Global-row-only proofs carry ~all of it (2.375).
- Global proofs surviving at 20k (11, dumped): e.g. `QP - QFR1 - QFR2 - QFR3 - QFR4 <= 1`,
  `9 yR1 - Q2 + QP <= 9`, `-9(yR1+yR2+yR4+yR5) - QFR1 - QFR2 - QFR4 + QP <= 0`: aggregations of the
  flow-balance CHAIN (Q[n] = Q[n+1] + QFR[n], Q[NT] = 1 + QFR[NT], split rows) + big-M rows.
- Hypothesis for the §106 "geometric convergence / 500-1000 rounds" symptom: row-wise FBBT on a chain of
  linear equalities converges only geometrically, and its fixpoint is weaker than propagating an
  aggregated row. SCIP gets aggregated rows two ways: presolve aggregation (28 aggregated vars) and dual proofs.

### discopt census, 20k nodes (d1_20k.log; observer bound-neutral: 1.3431805085914783 = baseline)
- 18541 LP nodes: 7406 Farkas-infeasible, 11046 optimal, 89 numerical; ~1459 propagation-fathomed.
- So discopt has as many infeasible leaves as SCIP (7182) — it just throws each Farkas certificate away.
- incumbent 3.0620 passed in at the root (initial_incumbent), SCIP finds it at node 7162: not the gap.

### Presolve sub-ablations (base emu_pre = presolve ON, conflict off; 20k)
| arm | dual |
|---|---|
| emu_pre | 2.091 |
| + donotaggr + donotmultaggr | 1.714 |
| + donotmultaggr only | 2.107 |
| + donotaggr only | 1.785 |
| + probing maxprerounds=0 | 2.075 |
| emu_conf, uselocalrows=false | 2.203 (emu_conf 2.273) |
=> presolve's share is 2-variable AGGREGATION (V1..V5 and c1 merged into one SCIP variable; branched 6002x as one
var). Probing/implications: nothing. This looks like a branching-score effect (summed violation over all
rate*V products) rather than propagation; not pursued further (branching closed as a direction, §110).

### THE MISSING IDEA: dual (Farkas) proof constraints, reproduced in discopt (no solver change)
d2 (py/d2_proofs.py): for each of discopt's 7406 Farkas-infeasible node LPs (20k census), phase-1 LP over the
spec's FIXED (global) rows only on the node's post-propagation box. 844/7406 (11.4%) are infeasible on global rows
alone (SCIP: 479/5682 = 8.4%). Farkas duals -> aggregated row; SCIP tightenDualproof post-processing (cancel
continuous cols at their global bound). 844 verified proofs, **95 unique**, nnz median 8, max 21.
Every proof has the form  yR_n <= const + (small coef) * (flow-chain aggregate), e.g.
  node 35:   yR4 - 0.09901*QP <= 0.90099          (recycle at 4  =>  QP >= 1)
  node 278:  yR3 - 0.099(QFR1+QFR2+Q4) + 0.099 QP <= 1
i.e. learned binary->continuous implications through the chain of flow-balance equalities, which row-wise FBBT
cannot derive at any number of rounds (it needs the aggregation).

d3 (py/d3_arm.py): proofs appended to the kernel spec's fixed rows (they are nonnegative aggregations of the
fixed rows: globally valid and LP-redundant), deterministic tree:
| discopt cstr big-M | bound @20k | to certification |
|---|---|---|
| base (reproduces 1.3431805085914783) | 1.3432 | not certified at 150k (§107: 2.8697, 2946 s) |
| + 95 proofs (learned over the whole 20k tree) | **2.4980** | |
| + 40 proofs (learned in the first 5k nodes only) | **2.3360** | **certified, 53,051 nodes, 136 s** (bound 3.06171) |
| control: 95 random nonneg aggregations of fixed rows (same count, nnz med 7) | 0.1603 | |
| SCIP default | 2.3927 | ~38k nodes |

> **RETRACTED 2026-11-05 (performance-plan §114, `2142-hull-empty/`).** Every hull row in this
> file, here and in the "Hull" table below, ran SCIP at `numerics/infinity` 1e20, with bounds
> >= 1e19 passed as infinite. With a perspective binary at 0 the spec needs 1e24 (= 1e8^3), so
> SCIP treated every such configuration as infeasible and certified only the all-binaries-at-1
> restriction. At infinity 1e30 SCIP default reports a false optimum, 3.1302 against the oracle
> 3.0620. None of these numbers is a valid hull reference.

### SCIP on the HULL spec (captured the same way: py/capture.py, GDP=hull; 320 cols, 25 terms, 54 BLF, 725 rows)
- SCIP default: **optimal, 1,772 nodes, 3.7 s**, dual 3.06200. discopt hull: 0.094 at 10k nodes (§111).

### Causality checks
- SCIP emu (presolve+conflict off, 1.728) + discopt's 95 proofs as PROPAGATE-ONLY constraints (initial/separate/
  enforce/check False; LP untouched): **2.462**; + the 40 first-5k proofs: **2.321** (SCIP's own conflict: 2.273).
- discopt fixed-box (d5, 553 LP-optimal boxes of the base 20k tree, FBBT run to convergence at 100000 rounds):
  with proofs the box is proven EMPTY on 167, tightened on 385; node LP bound raised on 539/544.
- Concrete node (d9): node 152 of the base tree, bound 0.0940, binaries fixed with recycle at reactor 5,
  converged FBBT box nonempty (QP in [0, 2.211], QFR1..4 ~ 0). Minimal emptying subset = 2 proofs:
    proof 24 (learned at node 3436): yR5 - 0.099(QFR1+QFR2+QFR3+QFR4) + 0.099 QP <= 1.099   [yR5=1 => QP <= 1 + sum QFR1..4]
    proof 39 (learned at node 4985): yR5 - 0.099 PB - 0.00495 QP <= 0.90099               [yR5=1 => PB + 0.05 QP >= 1]
  each alone tightens (e.g. V1..5 [0,1.042]->[0,0.857]); together they prove the box empty.
- LP-perturbation noise in discopt's tree is large: appending LP-redundant rows moves the 20k bound
  (relaxed proofs 1.685, relaxed random 1.457, active random aggregations 0.160 [unexplained]).
  Row-permutation seeds (no extra rows), 20k nodes:
  | seed | base | +40 proofs (first 5k) |
  |---|---|---|
  | 0 | 1.343 | 2.336 |
  | 1 | 1.766 | 2.362 |
  | 2 | 1.021 | 2.328 |
  | 3 | 1.691 | 2.339 |
  base mean 1.455 (range 0.75); proofs mean 2.341 (range 0.034). Paired gain >= +0.57 on every seed.
  Contradicting point (recorded, unexplained): 95 proofs + 95 relaxed random rows -> 1.423 (62 undecided LPs);
  relaxed random rows alone 1.457. The propagate-only implementation removes the LP confound.
- Wall: proofs as LP rows cost per-node time (seed1: 58 s -> 229 s); propagate-only avoids that.

### Hull (SCIP on discopt's captured hull spec)
| arm | result |
|---|---|
| default | optimal, 1,772 nodes, 3.7 s |
| conflict off | optimal, 3,157 nodes |
| presolve off | optimal, 21,704 nodes |
| separators off | optimal, 2,289 nodes |
| emu | optimal, 58,311 nodes (212 s) |
| emu + conflict | optimal, 30,569 nodes |
| node propagation off | NOT certified at 200k, 2.636 |
| @10k: emu | 1.695 |
| @10k: emu + nonlinear propagation off | **0.0940 == discopt hull @10k (0.0940, §111)** |
| @10k: presolve off, conflict on | 2.248 |
Hull has 20 eps-perspective terms w = 1/((1-1e-8) y + 1e-8), y in [0,1], w in [1, 1e8] (kind 4, p=-1).
discopt's AffinePow propagation is FORWARD ONLY (spatial_propagate.rs, "#2142: FORWARD only ... No reverse step").
Hypothesis (not tested): hull's gap is the missing reverse step w -> y on these terms (w <= U => y >= ...).

## Proposal: dual-proof (Farkas) constraint learning in the spatial kernel (general, not cstr-specific)
SCIP source: conflict_general.c SCIPgetFarkasProof (aggregate global rows with the Farkas multipliers; local rows
only if needed, validdepth); conflict_dualproofanalysis.c tightenDualproof (cancel continuous vars at their global
bound, coefficient tightening, drop if redundant), createAndAddProofcons (fill-in limit: nnz vs 2*avg row nnz and
conf_maxvarsfac=0.15 of vars; propagate-only linear cons; conflictstore ages them out, maxstoresize 10000).
discopt pieces already present: the dual simplex returns a VERIFIED Farkas ray on LpStatus::Infeasible
(lp/simplex/dual.rs ~1937, farkas_ray_certifies_cols); node LP rows are ordered fixed rows | envelope rows | global
cuts (spatial_kernel.rs assemble_node_lp_with_cuts), so the global-row part of y is a prefix (+ the cut block).
1. On NodeVerdict::EmptyRegion: a = sum_{global rows r} y_r A_r, beta = sum y_r b_r (envelope rows dropped =
   SCIP uselocalrows=false, measured 2.375 vs 2.393). Cancel continuous cols whose activity bound == global bound.
   Keep only if min activity over the node box > beta + safe tol (directed rounding / NS-style margin, as the bound
   certificate), nnz <= max(10, 0.15*n_cols) (cstr: median 8, max 21), dedupe.
2. Store in a tree-global pool (not inherited per node: globally valid); propagate pool rows inside
   propagate_fixpoint_rounds after the fixed rows (tighten_le). Do NOT put them in the LP (LP-redundant; cost+noise).
3. Aging/cap like SCIP's conflictstore; profile counters ProofsLearned / ProofFathoms / ProofTightenings.
Soundness: each row is a nonnegative combination of globally valid rows -> valid everywhere; only rhs rounding
needs care (relax rhs outward by |y|.|b| * eps). Bound-changing => flag, default OFF, §5 panel.
Entry experiment for the implementation (before any panel): cstr big-M deterministic 20k nodes, 4 row-permutation
seeds, flag ON vs OFF. Kill: ON mean bound < 2.0 or any seed ON <= OFF. Then nodes-to-certify (expect certify <= 60k;
the offline 40-proof arm certified at 53,051). Then §5 panel (in-repo corpus + GDP parity + tanksize/nous1 probes),
which also tests generality: needs Farkas-infeasible node LPs whose proof uses only global rows; report
the per-instance count of learned proofs so an unfired instance is visible.

> **Raw logs removed (#2142 follow-up).** The `.log` / `.log.gz` run traces cited in this directory were
> removed to keep checkouts small; the results they support are recorded in these notes. The originals
> remain in git history: `git show e39975e1:docs/dev/data/2142-scip-expert/<name>.log.gz | gunzip` (or `.log`, no gunzip).
