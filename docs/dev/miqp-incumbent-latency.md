# MIQP / convex-MINLP incumbent latency (#281, #287) — diagnosis & plan

**Status:** diagnosis done (evidence below); implementation scoped, not started.
Companion to the #287 fix (root per-node-OBBT skip, merged in #294) and the polish
fix (#298).

## Problem

On the short-tier benchmark (5 s budget) the `smallinvDAX_*` portfolio MIQPs (and
similar convex MINLPs) return *near-optimal-but-suboptimal* incumbents or no
incumbent — #281. The root cause is **incumbent latency**, not search capability or
continuous-completion polish: they solve to the proven optimum given more time.

## Evidence (measured)

`smallinvDAXr2b010-011` (=opt= 0.39880), `time_limit` varied:

| budget | result |
|---|---|
| 5 s | feasible 0.4004 (0.4% off) — or, run-to-run, no incumbent |
| 8 s | first incumbent @5.7 s, optimal @7.0 s |
| 12 s | **optimal 0.39880** |

It is classified **Convex MINLP → NLP-BB** (NLP relaxation solved per node). Phase
breakdown of the 7 s solve (timestamped trace + cProfile):

| phase | ~time | what |
|---|---|---|
| setup + JAX compile | ~1.5 s | model finalize, convexity routing, first JAX trace/lower of the NLP evaluator |
| feasibility pump | ~2.7 s | several NLP solves; yields a 0.4%-off incumbent (wrong integers) |
| branch-and-bound | ~2.8 s | 249 nodes (NLP per node) to the optimum |

- Fixing the incumbent's integers + a continuous solve does **not** close the gap →
  the 5 s incumbent has *wrong integers* (a polish cannot help; confirms #298's
  scope note).
- cProfile: dominated by JAX `bind`/`eval_jaxpr`/MLIR lowering + one-time
  `sparsity._collect_variable_indices`. The sparsity pattern is already cached
  (`NLPEvaluator.sparsity_pattern`), so it is a one-time ~0.5 s cost, not per node.
- SCIP solves all of these in <5 s (0.3–2.9 s) via LP-based outer approximation.

There is **no single hot spot** — unlike #287 (a redundant root per-node-OBBT pass
with a one-line fix). The latency is distributed across compile + primal + search.

## Levers (ranked by breadth × impact ÷ effort)

1. **Per-solve JAX-compile / startup overhead (~1.5 s, broadest).** Hits *every*
   short-tier instance, not just MIQPs. Directions: cache compiled node-NLP
   evaluators across the tree and across solves; lazy/avoid JAX for tiny models;
   trim the convexity-routing trace. First experiment: attribute the 1.5 s precisely
   (trace vs compile vs convexity) before optimizing. *Medium effort, wide payoff.*
2. **Stronger convex-MINLP primal (~2.7 s FP → 0.4%-off).** The FP finds wrong
   integers; a RENS / NLP-relaxation-rounding / diving heuristic that exploits the
   convex relaxation would land the optimal integers earlier (within budget).
   *Medium effort; directly closes #281's incumbent quality.*
3. **Outer approximation (OA) for convex MINLP (the SCIP method).** Replace
   NLP-per-node with an LP master + NLP subproblems: orders of magnitude fewer/cheaper
   nodes (SCIP's 0.3 s vs our 249-node 2.8 s). *Largest effort; the principled fix
   for convex-MINLP search efficiency.*

## Recommendation

This is SCIP-gap-class work, not a one-liner. Sequence by payoff/risk:
- **Start with lever 1** (JAX-compile/startup) — broadest (all short-tier), and the
  first step is a measurement (attribute the 1.5 s) before any build.
- Then **lever 2** (convex-MINLP primal) to land good incumbents within budget.
- **Lever 3 (OA)** last — the deep, highest-ceiling effort.

Each lever is gated on the existing short-tier benchmark harness
(`discopt-minlp-benchmark`), tracking per-instance incumbent-latency and gap so
progress is measured and regressions caught.

## Non-goals
- Matching SCIP's raw speed on every instance; target is "return the optimum (or a
  tight incumbent) within the short-tier budget."
