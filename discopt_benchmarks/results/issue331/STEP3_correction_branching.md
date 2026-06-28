# Issue #331 — Step 3: correction — the node gap is BRANCHING, not bounds

**This supersedes the central claim of Steps 1–2.** Those steps concluded the
node gap was a *weak-bounds* problem (discopt closing "2–3× less root gap than
SCIP"). On re-evaluation that number was a **measurement artifact**, and the
corrected diagnosis is the opposite: discopt's root cut **strength is comparable
to SCIP's**, and the 2–18× node gap is dominated by **branching quality**.

## The artifact

The Step 1/2 "SCIP gap closed" column came from `getDualboundRoot()` on a **full**
SCIP solve. On a full solve SCIP **restarts** — it re-processes the root with
everything it has learned — so `getDualboundRoot()` reports a *post-restart*,
post-branching root bound, not the bound SCIP's root cuts reach before branching.
That made SCIP's root look ~2× stronger than it is. The bench's SCIP root-gap
measurement has been fixed (`_scip_root_bound`: one node,
`presolving/maxrestarts=0`, `getDualbound()`), so the committed
`milp_node_efficiency.{md,json}` now report the true root bound.

One clean measurement (raw knapsack-value bounds; LP relaxation is the upper
bound, optimum below it, tighter root = closer to optimum):

| instance | z_LP | z_OPT | discopt root 16/1 | discopt root 300/50 | SCIP root (1 node, no restart) |
|---|---|---|---|---|---|
| mdk60x8 | 2300.1 | 2276.0 | 2294.2 | 2292.2 | 2290.7 |
| mdk90x12 | 3351.0 | 3329.0 | 3347.9 | 3344.3 | 3343.9 |
| mdk120x15 | 4638.9 | 4614.0 | 4636.6 | 4635.5 | 4634.0 |
| mdk200x25 | 7428.1 | 7397.0 | 7426.5 | 7425.3 | 7424.8 |

SCIP's root is *slightly* tighter than discopt's even at heavy cutting (a few %
more of the gap closed) — **not** the 2–3× the artifact implied, but not equal
either. So the honest question is not "are discopt's cuts weaker" (only mildly)
but "does that residual root-bound difference explain the 2–18× node gap." It
does not — see next.

## The decisive test — closing the root-bound gap does not move the node count

If the node gap were caused by discopt's slightly looser root bound, then giving
discopt strong root cuts (closing most of that residual gap) would cut its node
count toward SCIP's. It does not:

| instance | discopt prod (16/1) | discopt strong cuts (100/8) | SCIP |
|---|---|---|---|
| mdk90x12 | 573 nodes | 487 nodes | **66** |
| mdk120x15 | 8 617 nodes | 8 631 nodes | **2 330** |
| mdk200x25 | 27 385 nodes | 28 775 nodes | **12 688** |

Strong root cutting closes nearly all the discopt↔SCIP root-bound gap but leaves
the node count essentially unchanged (and costs wall time) — while SCIP, from a
root bound only a few percent tighter, proves with 2–7× fewer nodes. The node
count is **insensitive to the residual root bound** and sensitive to something
else.

## What that something else is — decomposed

Each lever isolated on the `mdk` instances (SCIP, no-restart unless noted):

| factor | finding |
|---|---|
| **Restarts** | *Increase* SCIP nodes here (mdk120x15: 1417 → 2330). Not the source of the advantage — they only inflate the reported root bound. |
| **In-tree cuts** | **Zero** effect (mdk120x15: 1417 with vs 1443 without). SCIP does not rely on cutting below the root here. |
| **Root cut strength** | Only a few % tighter than discopt, and closing that gap doesn't move discopt's nodes (test above). |
| **Branching** | **The driver.** With restarts off and in-tree cuts off — root cuts + branching only — SCIP proves mdk120x15 in **1443** nodes; discopt with a comparable-or-better root bound needs **8 631**. |

And on the discopt side, the branching *budget* is not the issue — it is the
branching *rule*: extending strong branching to every node with 4× more
candidates changes nothing (mdk120x15: 8 387 → 8 619 nodes). The scoring and
variable selection, not the amount of probing, is the limit.

Why this is consistent with the rest: ~99% of discopt's nodes are spent *proving*
optimality (the incumbent is found in <7 nodes). Best-bound node selection means
every node with bound < optimum must be processed; the **branching variable
choice** shapes how fast children's bounds rise past the optimum and get pruned.
SCIP's reliability-pseudocost branching shapes a tree ~6–18× smaller from the
same root bound.

It is specifically **variable selection**, not the other node-processing
intelligence SCIP has, because the rest is inert on these instances: domain
propagation / FBBT discovers nothing on a knapsack row (Step 2 §3 — no fixings,
no conflict pairs), node selection cannot change the *count* of nodes-to-prove
under best-bound (every node with bound < optimum must be opened, and the
incumbent is already found in <7 nodes), and conflict analysis has little to bite
on because knapsack node LPs are essentially always feasible. What remains, and
what the data points to, is which variable to branch on.

## Conclusion → to match SCIP, match its BRANCHING

The Step 1 instruction was "confirm first whether nodes are lost to bounds or
branching." Corrected answer: **branching.** The levers, re-ranked:

1. **Reliability pseudocost branching, done properly** — the dominant lever.
   discopt's current rule is "limited reliability/strong branching"; SCIP's
   mature pseudocost branching (good pseudocost initialization, product/hybrid
   scoring combining pseudocosts with inference/conflict counts, reliability
   thresholds) is what proves in 6–18× fewer nodes. This is where to invest.
2. Node selection / search strategy — secondary (best-bound is already near
   node-optimal for the proving phase, so the upside is smaller).
3. Cuts / presolve / restarts — **not** the lever for these instances. Root cuts
   already match SCIP; in-tree cuts and restarts don't move SCIP's node count
   here. (Cuts remain relevant for *other* instance classes; see Step 2.)

Net: closing the node gap to SCIP on these MILPs is a **branching** project, not
a cuts/bounds project. discopt's cheap nodes already make it wall-competitive;
better branching is what makes the node count scale.

### Reproducing

```bash
python -m discopt_benchmarks.perf.milp_node_efficiency --out discopt_benchmarks/results/issue331
```

The decomposition (restarts/in-tree-cuts/branching isolation, matched-bound node
counts) is via pyscipopt parameter toggles (`presolving/maxrestarts`,
`separating/maxrounds`) and the `solve_milp_py` branching knobs (`sb_node_budget`,
`sb_max_cands`) over the same `gen_mdk` instances — one-off diagnostics, described
here rather than committed as scripts.
