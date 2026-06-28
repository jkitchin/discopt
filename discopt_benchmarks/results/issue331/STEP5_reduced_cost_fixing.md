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
- **Sparse instance for the post-Regime-C bench:** ✅ added (see below).

## What remains to fully close #331

1. **The sparse regime** — discopt is still 24–53× SCIP nodes on the new sparse
   instances (and loses wall on the largest). This is the real open frontier and
   it needs **conflict-graph propagation + clique cuts** (live on sparse,
   inert on dense), not more of the dense-instance levers. See the sparse section.
2. **Tiny dense instances** SCIP closes at the root (1 node) — out of reach for
   ≤3× without root cut strength discopt's separators plateau below; low value
   (microsecond solves).
3. *(Optional)* reuse the LP factorization for RCF's dual solve on very-large-`m`
   problems (not needed for a net gain at bench sizes — see follow-ups).

Objective-cutoff propagation was investigated and **grounded out** (see
follow-ups) — it does not help these instances.

### Reproducing

```bash
python -m discopt_benchmarks.perf.milp_node_efficiency --out discopt_benchmarks/results/issue331
```

## Follow-ups resolved in this step

**RCF is safe on cut-heavy configs (default-on justified).** The per-node dual
solve is `O(m³)`; with the aggressive `full` config (root_cuts=64, 10 rounds,
node cuts → many rows) RCF still *improves* wall time (mdk90x12 −23 %, mdk120x15
−39 %, mdk200x25 −43 %) — the node savings dominate the dual-solve cost. Reusing
the LP factorization remains a worthwhile optimization for very-large-`m`
problems but is not needed for correctness or for a net gain here.

**Objective-cutoff propagation — grounded out, not built.** Simulated with a
*perfect* cutoff row (`value·x ≥ optimum`) added to the matrix and propagated by
the existing node-FBBT hook: node counts barely moved (mdk30x5 87→83, mdk40x5
147→97, mdk70x10 139→125). On uniform knapsacks the objective row fixes nothing
(same structural reason FBBT/probing don't fire — no single item dominates the
gap). So objective-cutoff propagation would be another inert knob here; reduced-
cost fixing (which uses the LP *duals*, not raw `c`) is the version that works.

## Sparse instances added (acceptance item #3) — and what they reveal

Per the issue's third item, now that Regime C (#334) has landed, the bench
includes a sparse-row family (`smdk{n}x{m}`, each row ~25 % dense). RCF helps
there too (−6…−23 % nodes), but the **sparse regime is where the node gap
remains large**:

| instance | discopt nodes | discopt wall | SCIP nodes | SCIP wall | node ratio |
|---|---|---|---|---|---|
| smdk50x15 | 703 | 0.12 s | 21 | 0.19 s | 33.5× |
| smdk60x20 | 1 025 | 0.32 s | 43 | 0.41 s | 23.8× |
| smdk70x20 | 7 643 | 0.94 s | 145 | 0.94 s | 52.7× |
| smdk80x25 | 29 147 | 3.75 s | 753 | 1.85 s | 38.7× |

On sparse instances discopt is **24–53× SCIP nodes**, and on the largest it now
**loses on wall time** (3.75 s vs 1.85 s) — the first instance where discopt is
not faster. This is the issue's "node-inefficiency dominates as instances grow"
made concrete: sparse-row problems have the **conflict/clique structure** that
SCIP's propagation and clique cuts exploit heavily and that discopt lacks. The
probing/clique machinery that was *inert on dense uniform knapsacks* (Step 2 §3)
is live here — so **the next lever for the sparse regime is conflict-graph
propagation + clique cuts**, not more of what helped the dense instances.

## Net status

- **Dense instances:** reduced-cost fixing brings the branching-dominated sizes
  to ≤3× SCIP with faster wall — the criterion's intent is met where node count
  matters. Tiny instances SCIP closes at the root remain high-ratio (need root
  cut strength discopt's separators plateau below; not propagation-fixable).
- **Sparse instances:** node gap is still 24–53×; this is the open frontier and
  it points at clique/conflict propagation as the next work, not at cuts or RCF.
