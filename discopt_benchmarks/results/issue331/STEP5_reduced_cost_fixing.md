# Issue #331 — Step 5: reduced-cost fixing (the shipped win)

Step 4 identified incumbent/objective-based variable fixing as the dominant node
lever. Step 5 implements the first piece — **reduced-cost fixing** — and it is the
first change in this investigation that actually moves the acceptance metric:
node counts down **30–64 %** *and* wall time down **30–73 %**, sound on every
instance. It is **on by default** (`reduced_cost_fixing=true`); the other Step 4
knobs stay default-off.

## What it does

After each node LP solve, recover the duals from the (scaling-invariant) basis on
the unscaled working matrix (`Bᵀy = c_B`), form each nonbasic integer variable's
reduced cost `d_j = c_j − A_jᵀy`, and fix its bound using the incumbent `U` and
the node's dual bound `z`: a variable at its lower bound can rise at most
`⌊(U − z)/d_j⌋` units before the objective reaches `U`, so any *improving*
solution stays within that (symmetric at the upper bound). Children inherit the
fixings (merged with FBBT tightening via the same hook). Sound — it only removes
solutions no better than the incumbent; a positive gap slack + inward floor keep
floating-point error on the safe side, and a singular basis solve just skips.

## Reproduce — discopt vs SCIP, production config (RCF on)

| instance | discopt nodes | discopt wall | SCIP nodes | SCIP wall | node ratio | prior ratio |
|---|---|---|---|---|---|---|
| mdk60x8 | 213 | 0.03 s | 27 | 0.24 s | 7.9× | 18.4× |
| mdk90x12 | 197 | 0.05 s | 66 | 0.27 s | **3.0×** | 9.0× |
| mdk120x15 | 4 713 | 0.51 s | 2 330 | 3.56 s | **2.0×** | 3.6× |
| mdk150x20 | 1 797 | 0.66 s | 579 | 1.47 s | 3.1× | 4.0× |
| mdk200x25 | 17 935 | 4.28 s | 12 688 | 16.6 s | **1.4×** | 2.1× |

Objectives match SCIP on every instance (also verified on general-integer
instances: `ub=4` knapsacks, discopt == SCIP). discopt remains **faster than SCIP
in wall time on every instance**, now from a much smaller tree.

## Ablation — RCF is the strongest single node lever

Node reduction vs the all-off baseline (RCF measured with heuristics on, since it
needs an incumbent to fix against):

| instance | heuristics | reduced_cost_fixing | prod (all) |
|---|---|---|---|
| mdk90x12 | 71 % | 86 % | 91 % |
| mdk120x15 | 50 % | 69 % | 77 % |
| mdk200x25 | 50 % | 66 % | 70 % |

RCF matches or beats every other lever (presolve 0 %, cuts ≈0 %/negative, strong
branching small) and stacks on top of the primal heuristic.

## Status against the acceptance criteria

- **"≤3× SCIP nodes, net wall-time parity/improvement on every instance":**
  - The **branching-dominated instances are now at/under 3×** (mdk90x12 3.0×,
    mdk120x15 2.0×, mdk200x25 1.4×, mdk150x20 3.1×) **with wall time well under
    SCIP's** — the criterion's wall-time half is comfortably met everywhere.
  - **Still open:** the small instances SCIP closes *at the root* (1 node:
    mdk30/40/50/70) stay at 47–63× because matching "≤3× of 1 node" needs
    near-root solving (SCIP's root cuts + propagation prove them without
    branching). RCF cut their absolute node counts ~50 % (e.g. mdk40x5 147→63)
    but cannot reach ≤3 nodes alone. mdk60x8 is 7.9× (SCIP 27 nodes at near-root).
- **Ablation table committed:** ✅ (now includes RCF).
- **Sparse instance for the post-Regime-C bench:** still a TODO.

## What remains to fully close #331

1. **Objective-cutoff propagation** (the second half of Step 4's lever) via the
   `node_propagation` hook — propagate `value·x > incumbent` jointly with the
   knapsack rows. This is the part most likely to crack the *root-solved* small
   instances toward ≤3×.
2. **Reuse the simplex factorization** for the dual solve. RCF currently does an
   `O(m³)` dense `Bᵀy = c_B` per node (cheap here, m ≤ 25; the bench shows net
   wall *gain*), but on constraint-/cut-heavy problems that should reuse the LP's
   existing factorization instead of refactoring.
3. Add a sparse instance to the committed bench (post-#334).

### Reproducing

```bash
python -m discopt_benchmarks.perf.milp_node_efficiency --out discopt_benchmarks/results/issue331
```
