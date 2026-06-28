# Issue #331 — Step 1: reproduce & attribute the MILP node gap

> ⚠️ **Superseded conclusion — read `STEP3_correction_branching.md`.** This Step 1
> writeup attributed the node gap to *weak bounds*, based on a SCIP root-gap number
> measured with `getDualboundRoot()` on a full solve — which includes SCIP's
> restarts and so overstated SCIP's root strength ~2×. Corrected, apples-to-apples:
> discopt's root bound is close to SCIP's, closing the residual difference does not
> reduce discopt's node count, and **the node gap is driven by branching, not
> bounds.** The reproduction (node counts, objectives) below is still valid; the
> *attribution* is corrected in Step 3.

This directory holds the committed output of the node-efficiency micro-bench for
the pure-MILP simplex engine (`nlp_solver="simplex"` →
`crates/discopt-core/src/bnb/milp_driver.rs`, exposed as
`discopt._rust.solve_milp_py`). It answers the question issue #331 Step 1 says
must be answered **before any solver change**: are the extra B&B nodes lost to
**weak bounds** (presolve/cuts) or **weak branching**?

## How to regenerate

```bash
# full table (writes milp_node_efficiency.{md,json} here)
python -m discopt_benchmarks.perf.milp_node_efficiency --time-limit 90 \
    --max-nodes 4000000 --out discopt_benchmarks/results/issue331
# quick smoke (4 smallest instances)
python -m discopt_benchmarks.perf.milp_node_efficiency --quick
```

The generator (`discopt_benchmarks/perf/milp_node_efficiency.py`) is the committed
replacement for the throwaway script the issue references: deterministic
`mdk{n}x{m}` 0/1 multidimensional knapsacks (uncorrelated values, half-sum
capacities, seed = a pure function of `(n, m)`), solved both by the discopt
engine (calling `solve_milp_py` **directly**, so the `_SIMPLEX_MILP_BUDGET_CAP_S`
cap in `solver.py` is bypassed and we measure the engine, not the cap) and by
SCIP (`pyscipopt`), with objectives cross-checked. The values are deliberately
**uncorrelated** with the weights — the easy regime the issue measured, where the
solve finishes sub-second and the discopt-vs-SCIP **node ratio** is the signal
(strongly correlated knapsacks are the pathological B&B case and would blow past
any sane time budget, masking it).

> Environment note: measured with the SCIP bundled in `pyscipopt` 6.2.1
> (SCIP 10.0), single machine, `parallel` feature on, feral 0.11 (crates.io) as
> the LU backend. Absolute numbers differ from the issue's original table
> (different host/SCIP build); the *attribution* is what transfers.

## Reproduce — node-inefficient but per-node fast (confirmed)

discopt explores **2–150× more nodes** than SCIP, yet wins on **wall time on
every instance** because each node is far cheaper:

| instance | discopt nodes | discopt wall | SCIP nodes | SCIP wall | node ratio |
|---|---|---|---|---|---|
| mdk60x8 | 497 | 0.040s | 27 | 0.233s | 18.4× |
| mdk90x12 | 597 | 0.064s | 66 | 0.271s | 9.0× |
| mdk120x15 | 8387 | 0.526s | 2330 | 3.555s | 3.6× |
| mdk150x20 | 2335 | 0.460s | 579 | 1.405s | 4.0× |
| mdk200x25 | 27117 | 3.865s | 12688 | 15.993s | 2.1× |

(The instances SCIP closes in presolve — 1 node — give inflated ratios like 87×;
see the full table.) Objectives match SCIP on every instance both prove optimal.
This is exactly the issue's premise: the node multiplier is masked by raw
per-node simplex speed and will dominate as instances grow.

## Headline finding — the node gap is a WEAK-BOUNDS problem

### 1. Root integrality gap closed: discopt closes 2–3× less than SCIP

This metric is **cap-independent** (root LP bound, root bound after
presolve+cuts, optimum), so it is the cleanest diagnostic. On the instances SCIP
actually branches on:

| instance | discopt gap closed | SCIP gap closed |
|---|---|---|
| mdk60x8 | 24.8% | 48.6% |
| mdk90x12 | 14.2% | 45.9% |
| mdk120x15 | 9.4% | 22.4% |
| mdk150x20 | 13.3% | 27.1% |
| mdk200x25 | 5.0% | 13.9% |

discopt leaves the root relaxation 2–3× looser than SCIP. A weaker root bound is
the proximate cause of the extra branching.

### 2. But the existing bound levers don't deliver — and a primal heuristic is doing the real work

Per-lever ablation, each config layering **one** `MilpOptions` lever onto an
all-off baseline; "node reduction attributable to each lever" = `(baseline −
config) / baseline` (negative = that lever alone *increased* nodes). All
instances solve to optimality, so these are true nodes-to-proof:

| lever | typical node reduction | reading |
|---|---|---|
| `presolve` | **0% everywhere** | root-FBBT-only presolve is inert |
| `root_cuts` (GMI+cover, 1 round) | ≈0%, often **negative** | current cuts barely move nodes |
| `cut_rounds` (multi-round) | mixed, ≤ a few % on large | multi-round doesn't rescue it |
| `node_cuts` | ≈0% on large | — |
| `strong_branch` | small / negative (≤19%) | branching is **not** the bottleneck |
| **`heuristics`** (rounding) | **42–96%** | **dominant** node lever |

The decisive cross-check (`mdk200x25`, baseline = 58 885 nodes):

```
heuristics alone .............. 29 451 nodes  (50% fewer — early incumbent prunes)
cuts+presolve+strong-branch,
   NO heuristic ............... 54 275 nodes  ( 8% fewer — the bound levers barely help)
prod (everything) ............. 27 117 nodes  (54% fewer ≈ the heuristic's contribution)
```

So discopt's current node count is propped up almost entirely by the **primal
heuristic**, not by good bounds. The cut/presolve machinery that *should* tighten
the root bound (and it is genuinely too loose — see §1) is underdelivering:
GMI+cover at one round does ~nothing, and the root-FBBT presolve does literally
nothing.

## Conclusion → what Step 2 should deepen

The attribution is **bounds, not branching** — and specifically, the *existing*
bound levers are too weak to close the gap SCIP closes:

1. **Cuts are the highest-value target, but the bar is "actually strengthen the
   bound", not "add a separator".** Today's portfolio is GMI + cover only
   (`lp/gomory.rs`, `lp/cover.rs`); `lp/mir.rs` exists but is unused, and there
   is no cut selection/management. SCIP closes 2–3× more root gap with MIR /
   flow-cover / knapsack-cover-with-lifting / clique / zerohalf + an
   efficacy+orthogonality selection pass. Add those — and measure root
   gap-closed moving toward SCIP's, since the current cut levers move neither the
   bound enough nor the node count.
2. **Presolve is dead weight as wired.** Root FBBT closes 0% extra gap. Wire more
   of the existing `presolve/orchestrator.rs` passes (probing, coefficient
   strengthening, implied bounds, cliques, dual/reduced-cost fixing) into the
   driver — sound (dimension-preserving, or carry a postsolve).
3. **Do not touch branching first.** `strong_branch` is small/negative here; the
   bound is the bottleneck, not the branching rule.

### Net-wall-time warning (the issue's hard constraint)

discopt already **beats SCIP on wall time on every instance** while losing on
nodes. Stronger cuts cost per node: the everything-on `full` config cuts nodes
vs `prod` on small instances but is roughly node-neutral on the large ones and
**much slower in wall time** (e.g. `mdk200x25`: `full` ≈ 37 s vs `prod` ≈ 3.9 s).
So "more cuts" naively applied *loses*. Step 2's cuts must close enough root gap
to remove nodes **and** carry cut selection/management to keep per-node cost
down, or they will regress the very instances discopt currently wins. Measure
wall time, not just nodes.

Every change must hold the issue's hard constraints: soundness (objectives here
match SCIP on every instance both prove optimal), the `milp_driver` Rust tests
(`root_cuts_reduce_nodes`, `presolve_matches_no_presolve`, …), and net wall time.
