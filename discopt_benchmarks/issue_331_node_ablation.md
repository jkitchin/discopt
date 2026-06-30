# Issue #331 — pure-MILP node-efficiency: reproduce & attribute (Step 1)

This is the **Step 1 ("Reproduce & attribute") deliverable** the issue requires
*before any engine change*: a committed micro-bench plus the ablation table that
says whether the extra B&B nodes (vs SCIP) come from weak **bounds**
(presolve/cuts) or weak **branching**. The answer, on the dense multidim-knapsack
family, is unambiguous: **branching/primal, not root bounds** — and the cheap
bound levers the issue lists for Step 2 (more cuts / more presolve) are measured
to be *flat-to-negative* here, while the one sound branching lever that could go
deeper is *wall-unsafe to change globally*. Those measurements are what should
steer (and gate) the investment.

## How to reproduce

```bash
# Lever ablation across the mdk{items}x{dims} family + the cross-regime
# strong-branch-budget safety sweep. No external solver needed.
cargo run --release --example mdk_ablation -p discopt-core           # full family
SIZES=small TL=12 cargo run --release --example mdk_ablation -p discopt-core   # quick
```

The bench calls the engine entry `discopt_core::bnb::milp_driver::solve_milp`
directly — i.e. it does **not** go through the Python `_SIMPLEX_MILP_BUDGET_CAP_S`
cap, so it measures the engine, not the cap (as the issue asks). The baseline
config is the *exact* set of defaults the Python `solve_milp_py` PyO3 binding
ships (`root_cuts=16, cut_rounds=1, gmi_cuts=true, cut_select=false,
node_cuts=false, heuristics=true, presolve=true, strong_branch=true,
node_propagation=false, reduced_cost_fixing=true, sb_max_cands=6,
sb_node_budget=48`). Each ablation row changes **one** lever vs that baseline.

Instances are deterministic (seeded LCG, no RNG/time dependency), so node counts
are bit-reproducible across machines; wall times are indicative.

> **SCIP comparison:** not run here — SCIP/pyscipopt are not a dependency of the
> Rust core and are absent from this environment/CI. The SCIP node/wall
> comparison lives in `discopt_benchmarks/bench_milp_sparse.py`. This bench
> answers the *attribution* question (which lever the engine's own nodes respond
> to), which is what decides where Step 2 should invest. The generator here is
> intentionally harder than the issue's original throwaway one (a difficulty
> cliff appears at ≥8 capacity rows), so the clean apples-to-apples ablation is
> read off the instances every config proves to optimality (mdk30x5, mdk40x5,
> mdk50x8); larger sizes hit the per-solve wall cap and are reported as such.

## Soundness tripwire

Every config's proven optimum is asserted equal to the baseline's
(`obj` column identical across all rows of every instance, e.g. `-1645.0` for
mdk50x8). No lever changed the optimum — the soundness invariant the issue marks
"sacred" holds across the whole grid. (Where a deeper-budget run shows a *worse*
`obj` it is also `feasible`, not `optimal` — it simply hit the wall cap before
proving optimality; it never reports a wrong "optimal".)

## Ablation result — node reduction attributable to each lever

Geometric-mean node ratio `config / baseline` over the optimally-solved instances
(mdk30x5, mdk40x5, mdk50x8). A ratio **> 1** means turning the lever *off* (or, for
`+` rows, *on*) **expands** the tree — i.e. the lever, in its baseline state,
*shrinks* it by that factor.

| config (vs baseline)        | node geomean | reads as |
|-----------------------------|-------------:|----------|
| `-heuristics`               | **3.00×**    | primal heuristics shrink the tree ~3× (dominant) |
| `-reduced_cost_fix`         | **1.89×**    | reduced-cost fixing shrinks ~1.9× |
| `-strong_branch`            | **1.39×**    | strong/reliability branching shrinks ~1.4× |
| `+node_prop(FBBT)`          | 0.93×        | per-node FBBT trims ~7% (small, ~wall-neutral) |
| `-gmi(cover only)`          | 0.98×        | GMI cuts ≈ no effect on knapsack |
| `-presolve`                 | 1.00×        | root presolve is a **no-op** on pure knapsack |
| `+node_cuts`                | 1.00×        | density-gated off on dense rows → no-op |
| `-root_cuts`                | 0.92×        | removing root cuts slightly *helps* (cuts cost > bound gain) |
| `+sb_budget=5k`             | 0.99×        | deeper strong branching ≈ neutral on aggregate |
| `+cut_rounds=5`             | 1.18×        | more rounds *expand* the tree once rcf is on |
| `+cutsel(r=8,c=48)`         | 1.37×        | mixed/instance-dependent; net negative here |
| `min(pure B&B)`             | 3.21×        | all primal levers off |

### Attribution

- **The extra nodes are a branching/primal problem, not a root-bound problem.**
  The three biggest node levers are all primal/branching:
  primal heuristics (3.0×) > reduced-cost fixing (1.9×) > strong branching (1.4×).
- **Root bounds are already saturated by the cheap defaults.** Single-round
  GMI + *lifted* cover (the cover separator in `lp/cover.rs` already does
  sequential up-lifting via an integer-knapsack DP) extracts essentially all the
  cheap bound: GMI off is 0.98×, root-cuts off is 0.92× (cuts even mildly
  *hurt* net), and **presolve is a structural no-op** on pure knapsack (no
  equalities/implications for FBBT/probing to exploit). Adding *more* cut rounds
  or cut selection **expands** the tree net once reduced-cost fixing is on —
  the marginal bound no longer pays for the per-node LP it bloats.

This directly answers the issue's Step-1 question ("whether nodes are lost to
weak BOUNDS or weak BRANCHING"): **branching/primal.** The known gap to SCIP on
this regime is therefore its stronger *primal heuristics* (feasibility pump,
RINS/RENS, …), *reliability-pseudocost branching* run throughout the tree, and
*conflict analysis* — the larger Step-2 items in the issue — not the cheap
cut/presolve deepening, which is measured here to not move the needle.

## Cross-regime safety check — why `sb_node_budget` can't be raised globally

Strong/reliability branching is the cleanest *sound* node lever (it only changes
the branching variable — never a bound — so it cannot affect the optimum). The
knapsack ablation hints a deeper budget could help (`+sb_budget=5k` is wall-
positive on the hardest solved instance: mdk50x8 8000-budget = 178k nodes/7.8s
vs 48-budget 191k/8.8s). **But it is not a safe global default**, because each
strong-branch probe re-solves the node LP — cheap on a few-row knapsack,
expensive on a many-row set-covering LP:

| instance     | sb_node_budget | status   | nodes | wall   |
|--------------|---------------:|----------|------:|-------:|
| mdk50x8      | 48             | optimal  | 191289 | 8.8s  |
| mdk50x8      | 2000           | optimal  | 198487 | 9.2s  |
| mdk50x8      | 8000           | optimal  | 178447 | 7.8s  |
| sc1000x500   | 48             | optimal  | 33     | 1.1s  |
| sc1000x500   | 8000           | optimal  | 33     | 1.0s  |
| **sc2000x800** | **48**       | **optimal**  | **225** | **4.8s** |
| **sc2000x800** | **2000**     | feasible | 333   | 12.1s (capped) |
| **sc2000x800** | **8000**     | feasible | 337   | 12.1s (capped) |

On `sc2000x800`, deepening the budget turns a 4.8 s **proven-optimal** solve into
a 12 s **timeout** (more nodes *and* more wall — probing makes worse choices there
*and* costs more). That would violate the issue's hard constraint ("net wall-time
improvement or at least parity on **every** instance"). So any future change to
strong-branch depth must be **conditioned on problem shape** (few rows / cheap
probes), never applied as a global default.

## Recommendation for Step 2

1. **Do not deepen cuts/presolve for this regime.** Measured ≤1.0× node effect
   and net wall-negative; the cheap bound is already saturated. (MIR — `lp/mir.rs`
   — and other separators may still help *mixed* models with continuous structure;
   they are simply not what bounds pure 0/1 knapsack, where lifted covers already
   dominate.)
2. **Invest on the primal/branching axis**, which is where the nodes actually are:
   stronger primal heuristics (the 3.0× lever), reliability-pseudocost branching
   that stays active deeper **but gated by probe cost / row count** so it cannot
   regress the covering regime (per the table above), and — the larger item —
   conflict analysis.
3. **No engine default was changed in this PR.** The data shows the only cheap,
   sound candidate (a bigger `sb_node_budget`) is wall-unsafe globally, so shipping
   it would break the issue's parity constraint. This Step-1 artifact gates that
   decision rather than guessing past it.

The bench (`examples/mdk_ablation.rs`) and this table are the committed Step-1
evidence; extend the family with a sparse instance once the Regime-C data-layer
work lands (per the issue's last acceptance bullet) to validate node-efficiency
where per-node cost is also realistic.
