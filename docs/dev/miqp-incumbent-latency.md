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
2. **Stronger convex-MINLP primal (~2.7 s FP → 0.4%-off). — DONE (#281 lever 2,
   RENS).** The feasibility pump rounds every integer at once and lands wrong
   integers (or, at a tight budget, none). RENS (Relaxation Enforced Neighborhood
   Search) instead solves the relaxation's rounding neighbourhood *exactly*: fix
   the integers already integral in the root relaxation, restrict each fractional
   one to its `{floor, ceil}` box, solve the small sub-MINLP. Measured across the
   smallinvDAX family at tight budgets: at 2 s the baseline returns no incumbent
   on 4/5 instances while RENS is near-optimal on all 5; at 3 s baseline
   incumbents are 1.4–12 % off while RENS is optimal-or-better. A 74-instance
   soundness sweep showed 0 optimal-vs-optimal mismatches. Note: lever 1 already
   recovered enough budget that the *small* smallinvDAX solve to the proven
   optimum is unchanged — RENS's win is incumbent *quality at tight budgets* and
   on larger instances. *Done.*
3. **Outer approximation (OA) for convex MINLP (the SCIP method). — engine exists;
   correctness fixed (#301); auto-routing deferred (evidence-based).** `solve_oa`
   (opt-in via `gdp_method="oa"`) already implements OA. Measured vs NLP-BB+RENS:
   - **OA wins big on transcendental convex MINLP**: batch 0.6 s vs 11.4 s (19×);
     clay0203m 1.4 s vs *NLP-BB times out with no incumbent* in 31 s.
   - **OA ties/loses on convex QPs** (smallinvDAX: ~6 s vs ~5 s) — the NLP-per-node
     is cheap there, so OA's LP master adds no leverage, and it forfeits RENS's
     tight-budget incumbent.
   - **OA loses on non-convex-objective models** (alan: returns 3.6 *feasible* in
     25 s vs NLP-BB's 2.925 *optimal* in 0.3 s — its objective is not OA-convex,
     so the master bound is disabled and it cannot certify).
   - Found and fixed an OA **soundness bug** along the way: no objective-sense
     handling meant maximize problems were optimized in the wrong direction and
     certified wrong (syn05m: −831 "optimal" vs true 837.73). Fixed in #301.

   Conclusion: a blanket "route convex MINLP to OA" would *regress* the QP and
   non-convex-objective classes. Safe auto-routing needs a structural gate (route
   to OA only for transcendental/expensive-NLP convex MINLP with an OA-valid
   master bound, i.e. `master_bound_valid` true and non-quadratic nonlinearity),
   validated on the full convex-MINLP benchmark. *That gate is the remaining
   lever-3 work; the engine and its correctness are ready.*

## Recommendation

This is SCIP-gap-class work, not a one-liner. Sequence by payoff/risk:
- **Lever 1 (startup overhead) — DONE.** The measurement showed it was the SymPy
  structure-cut presolve, not JAX compile; gated cheaply (#281). ~1.3 s recovered
  broadly across the short tier.
- **Lever 2 (convex-MINLP primal) — DONE.** RENS lands near-optimal incumbents at
  tight budgets where the pump returned none/poor ones (#281, #302).
- **Lever 3 (OA) — engine ready, correctness fixed (#301); structural auto-routing
  gate is the remaining work.** OA is a large win on transcendental convex MINLP
  but regresses QP / non-convex-objective classes, so routing must be gated and
  benchmark-validated rather than blanket-enabled.

Each lever is gated on the existing short-tier benchmark harness
(`discopt-minlp-benchmark`), tracking per-instance incumbent-latency and gap so
progress is measured and regressions caught.

## Non-goals
- Matching SCIP's raw speed on every instance; target is "return the optimum (or a
  tight incumbent) within the short-tier budget."
