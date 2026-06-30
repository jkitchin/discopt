# SHOT Port Roadmap

This roadmap tracks SHOT-parity work for the `feature/mip-nlp-solver` branch.
The target is behavioral parity in discopt-native architecture, not a direct
translation of SHOT's C++ modules. Solver-core functionality comes first;
OSiL, NEOS, CPLEX, and Cbc parity are deferred.

## Current Foundation

- `mip_nlp_profile="shot"` reserves the experimental SHOT-parity surface.
- SHOT-profile controls are validated instead of silently ignored.
- `SolveResult.mip_nlp_trace` carries a stable trace envelope for MIP-NLP runs.
- OA populates iteration-level trace records for master solves, cut counts,
  fixed-NLP calls, solution-pool candidates, node counts, bounds, and bound
  validity.
- Live GitHub issue sequence: [#138](https://github.com/bernalde/discopt/issues/138)
  through [#153](https://github.com/bernalde/discopt/issues/153).

## Issue Sequence

1. **[#138: SHOT parity audit and benchmark baseline](https://github.com/bernalde/discopt/issues/138)**
   Build a SHOT-to-discopt feature matrix and run current discopt vs local SHOT
   on selected convex and nonconvex MINLP fixtures.

2. **[#139: MIP-NLP config and trace foundation](https://github.com/bernalde/discopt/issues/139)**
   Keep expanding the SHOT profile and trace payload as new algorithmic pieces
   land. Preserve default behavior for users that do not opt in.

3. **[#140: Unified cut/provenance model](https://github.com/bernalde/discopt/issues/140)**
   Represent OA, ECP, ESH, objective, and external cuts with source, global
   validity, local validity, supporting point, violation, row id, and dedup key.

4. **[#141: Reformulation parity audit and controls](https://github.com/bernalde/discopt/issues/141)**
   Fill gaps around objective epigraph controls, nonlinear and quadratic
   partitioning, absolute-value auxiliaries, anti-epigraphs, monomial/signomial
   toggles, integer-bilinear strategy limits, and quadratic extraction/direct
   solve routing.

5. **[#142: Initial POA and bound-tightening seeding](https://github.com/bernalde/discopt/issues/142)**
   Add SHOT-style initial polyhedral approximation after bound tightening:
   solve a relaxed approximation, import hyperplanes/interior points, and
   update objective cutoffs or bounds.

6. **[#143: Interior-point and rootsearch services](https://github.com/bernalde/discopt/issues/143)**
   Add reusable interior-point storage, minimax/interior-point search, and
   rootsearch between interior points and MIP/NLP candidates.

7. **[#144: ESH and objective-rootsearch hyperplanes](https://github.com/bernalde/discopt/issues/144)**
   Implement supporting-hyperplane separation with ECP fallback, cut selection
   limits, violation ranking, known-primal safeguards, and objective-rootsearch
   cuts.

8. **[#145: SHOT-style MultiTree loop](https://github.com/bernalde/discopt/issues/145)**
   Add relaxation-first phases, adaptive MIP solution limits, MIP starts from
   incumbents, solution-pool ingestion, and per-iteration cutoff management.

9. **[#146: Fixed-NLP primal candidate manager](https://github.com/bernalde/discopt/issues/146)**
   Schedule fixed-integer NLP solves from MIP optimum, LP relaxation, solution
   pool, rootsearch, and external candidates; support original/reformulated
   NLP source selection, warm starts, dynamic frequency, and infeasibility cuts.

10. **[#147: General integer cuts and solution-pool parity](https://github.com/bernalde/discopt/issues/147)**
    Extend no-good/integer cuts beyond binary-only cases and expose solution
    pool capacity/backend parameters consistently.

11. **[#148: Master infeasibility repair and reduction cuts](https://github.com/bernalde/discopt/issues/148)**
    Add infeasible-master repair, repair-loop detection, objective cutoff reset
    logic, and nonconvex primal reduction cuts.

12. **[#149: Convex bounding for nonconvex problems](https://github.com/bernalde/discopt/issues/149)**
    Maintain a secondary master with only globally valid cuts to report valid
    dual bounds for nonconvex runs.

13. **[#150: SingleTree callback parity](https://github.com/bernalde/discopt/issues/150)**
    Upgrade the Gurobi lazy-callback path with node-relaxation cuts,
    rootsearch/fixed-NLP callbacks, integer cuts, trace updates, and callback
    termination checks.

14. **[#151: Direct NLP, MIQP, and MIQCQP routing](https://github.com/bernalde/discopt/issues/151)**
    Add SHOT-style strategy selection for continuous NLP, LP/MILP, QP/MIQP, and
    QCQP/MIQCQP when backend capabilities allow.

15. **[#152: External event hooks](https://github.com/bernalde/discopt/issues/152)**
    Add optional hooks for external primal candidates, hyperplanes, dual bounds,
    and user termination.

16. **[#153: Integration docs and release checklist](https://github.com/bernalde/discopt/issues/153)**
    Keep docs, benchmarks, optional Gurobi checks, and MIP-NLP/GOA/FP regression
    tests synchronized before merging into mainline discopt.

## Acceptance Defaults

- Default CI remains free of commercial solver requirements.
- Optional Gurobi tests cover single-tree and solution-pool behavior.
- SHOT-derived fixtures need clean EPL-2.0 attribution; otherwise recreate
  equivalent DSL models.
- Nonconvex results must distinguish heuristic incumbents from globally valid
  bounds.
