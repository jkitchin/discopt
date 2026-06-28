# Issue #331 — Step 4: branching prototype + the real node lever

Step 3 traced the node gap to the tree search (not the root bound). Step 4 scoped
discopt's branching code, prototyped reliability-pseudocost branching as planned,
and — grounding every step in an intervention test — found the dominant lever is
**incumbent/objective-based variable fixing**, which discopt lacks entirely.
Branching variable selection is only a minor part of the gap.

Two prototypes were built (sound, 335 Rust tests pass, objectives match SCIP on
every instance), both behind default-off `MilpOptions` knobs so production is
unchanged:

- `seed_pseudocosts` — true reliability branching: feed strong-branch probe gains
  into the pseudocost tracker (`strong_branch` previously discarded them).
- `node_propagation` — run FBBT (`tighten_bounds`) at every node, not just root.

## What the prototypes measured

**Reliability branching (`seed_pseudocosts`) — slightly negative.**
Seeding pseudocosts from strong-branch probes made node counts *worse* (−7% to
−82%). Reason: it marks variables "reliable" sooner, switching *off* the strong
branching that was helping. So unseeded pseudocosts were not the bottleneck.

**Why branching isn't the lever — clean isolation.** With **no cuts on either
side**, both solvers start from the *identical* LP-relaxation root bound, so node
differences are pure branching/search:

| instance | discopt strong-branch | discopt full strong-branch | SCIP |
|---|---|---|---|
| mdk60x8 | 437 | 433 | 21 |
| mdk90x12 | 555 | 583 | 49 |
| mdk120x15 | 8 891 | 8 887 | 1 588 |
| mdk200x25 | 28 887 | 28 887 | 7 667 |

Even *full* strong branching (the gold standard a good rule approximates) leaves
discopt 3.8–20× above SCIP — so the branching *rule* is not where the gap is.

**FBBT node propagation — exactly 0%.** Identical node counts with vs without
(`tighten_bounds` documents itself as "a no-op on a lone knapsack row"). FBBT
propagates *constraints only*; it can't fix these knapsack variables.

## The real lever — SCIP's node advantage is incumbent/objective-based fixing

Decomposing SCIP's pure-B&B node count (no cuts/presolve/restarts) by propagator:

| instance | full prop | no reduced-cost | no propagation at all |
|---|---|---|---|
| mdk60x8 | 21 | 45 | 135 |
| mdk90x12 | 49 | 66 | 184 |
| mdk120x15 | 1 588 | 2 360 | 6 278 |
| mdk200x25 | 7 667 | 15 058 | 20 860 |

- **Reduced-cost fixing** (full → no-redcost): 1.3–2.1× — fixes nonbasic
  variables whose LP reduced cost would push the objective past the incumbent.
- **Remaining propagation** (no-redcost → no-prop): 1.4–3.0× — knapsack
  constraint propagation, including propagation of the **objective-cutoff**
  (every improving solution must satisfy `value·x > incumbent`).

Both use the **incumbent / objective** to fix variables. That is exactly what
fires in the proving phase (~99% of discopt's nodes), and exactly what discopt's
constraint-only FBBT cannot do — which is why the FBBT prototype was 0%. The
remaining branching residual (≈1.4–3.2×) is secondary.

## Recommendation — implement objective/incumbent-based fixing

1. **Reduced-cost fixing** (highest standard payoff). After each node LP, for a
   nonbasic integer variable at a bound with reduced cost `d_j`, fix it when
   `node_obj + |d_j| ≥ incumbent` (flipping it cannot improve on the incumbent).
   Needs reduced costs `c − Aᵀy` from the node basis (`LpSolve` exposes the basis;
   duals are recoverable via the existing `PreparedDual` factorization).
2. **Objective-cutoff propagation** (no duals needed). Add the row `value·x ≥
   incumbent + ε` (min sense: `c·x ≤ incumbent − ε`) to the per-node FBBT, so it
   propagates jointly with the knapsack rows as the incumbent tightens. Reuses the
   `node_propagation` hook built here; the children inherit the fixings.

Branching-rule work (reliability seeding, better scoring) is **not** the priority
— it addresses the smaller residual and the seeding prototype showed it can
backfire. The default-off knobs are retained as opt-in / building blocks (the
`node_propagation` hook is where objective-cutoff propagation plugs in).

### Reproducing

The decomposition is via pyscipopt propagator toggles
(`propagating/redcost/freq`, `propagating/maxrounds`, `separating`/`presolving`
off) and the `solve_milp_py` knobs (`seed_pseudocosts`, `node_propagation`,
`strong_branch`, `sb_node_budget`) over the `gen_mdk` instances.
