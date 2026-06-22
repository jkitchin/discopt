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

1. **Per-solve startup overhead (~1.5 s, broadest). — DONE (#281, structure-cut
   gate).** Attribution (measured, warm process so JAX/SymPy import is amortized)
   found the ~1.5 s was *not* JAX compile but the **SymPy structure-cut presolve**
   (`recognize_and_inject` -> `model_to_sympy`), auto-engaged by default since #253
   to close the gas-network gap (#15). Its size gate (vars+constraints <= 100)
   wrongly assumed `model_to_sympy` cost tracks variable count; it actually tracks
   *objective expression complexity*, so a dense-objective MIQP (smallinvDAX: 31
   vars but a 465-term quadratic) paid ~1.0 s translating SymPy for a
   guaranteed-empty recognition (no equality constraints -> the square-difference
   pattern can never match). Fixed with a microsecond native-DAG necessary
   condition (`has_square_difference_candidate`): skip `model_to_sympy` unless the
   model has a nonlinear equality. Measured: recognizer 1.07 s -> 0.000 s,
   end-to-end ~7.0 s -> ~5.7 s to the same optimum; gas-network (#15) keeps its
   cuts. Any *residual* JAX-trace/compile cost (the original lever-1 hypothesis)
   is now a smaller, separate follow-up if it proves to matter.
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
- **Lever 1 (startup overhead) — DONE.** The measurement showed it was the SymPy
  structure-cut presolve, not JAX compile; gated cheaply (#281). ~1.3 s recovered
  broadly across the short tier.
- Then **lever 2** (convex-MINLP primal) to land good incumbents within budget.
- **Lever 3 (OA)** last — the deep, highest-ceiling effort.

Each lever is gated on the existing short-tier benchmark harness
(`discopt-minlp-benchmark`), tracking per-instance incumbent-latency and gap so
progress is measured and regressions caught.

## Non-goals
- Matching SCIP's raw speed on every instance; target is "return the optimum (or a
  tight incumbent) within the short-tier budget."
