# SHOT parity audit and benchmark baseline

Issue: [#138](https://github.com/bernalde/discopt/issues/138)

This audit records the SHOT-port baseline and the final integration state for
`feature/mip-nlp-solver`. It compares current discopt behavior against the SHOT
capability surface after the roadmap series has landed. The target is
discopt-native behavioral parity in the supported surface, not a translation of
SHOT C++ internals.

## Feature matrix

| Area | SHOT capability | discopt status | Current discopt evidence and gap |
| --- | --- | --- | --- |
| Strategy selection | MultiTree and SingleTree strategies, direct NLP/MIQP/MIQCQP handling, and solver-specific master routing | Implemented for scoped backends | `solver="mip-nlp"` exposes OA, ECP, FP, GOA, and LP/NLP-BB. The SHOT profile routes direct NLP, LP/MILP, MIQP, QCP, and MIQCQP classes when convexity and backend checks pass, otherwise it records fallback details in `strategy_selection`. |
| Reformulation | Objective epigraphs, nonlinear/quadratic partitioning, signomial and monomial handling, integer-bilinear strategy choices, and optional use of reformulated primal NLPs | Implemented / scoped | Objective epigraph and anti-epigraph passes are active when structural proof conditions hold. Integer-bilinear binary expansion, quadratic extraction policy, and partitioning/signomial/monomial controls are validated and traced; fixed-NLP solves use the active discopt model rather than a separate original-vs-reformulated NLP registry. |
| Bound tightening | FBBT/OBBT and initial relaxation tightening before cut loops | Implemented / scoped | Root presolve, FBBT, OBBT, and nonlinear bound-tightening modules exist. The SHOT profile runs initial and periodic POA-style integrality-relaxed master phases, imports valid linearization cuts through provenance, stores interior points, and records fallback status. |
| Cut generation | ESH/ECP/objective/external cuts, cut selection, hyperplane provenance, validity flags, and deduplication | Implemented | OA, ECP, ESH, objective-rootsearch, feasibility, integer, reduction, and external cuts flow through provenance records with source, global/local validity, support point, violation, and source-count trace fields. Local cuts are guarded against cutting off the incumbent. |
| Primal heuristics | Fixed-integer NLP candidates from MIP optima, LP relaxations, solution pools, rootsearch, and external candidates | Implemented | `FixedNLPCandidateManager` orders and deduplicates candidates from MIP optima, relaxation phases, Gurobi solution pools, rootsearch/interior services, and external hooks. The trace records candidate sources, call statuses, warm-start sources, and scheduler state. |
| Master repair | Infeasible-master repair, cutoff reset, repair-loop detection, and nonconvex reduction cuts | Implemented | `master_repair=True` retries infeasible masters with cutoff/solution-limit controls reset and relaxable cut slacks. `reduction_cuts=True` adds local objective cutoff rows for heuristic nonconvex runs when the objective row is exact; loops and dropped reduction cuts are traced. |
| Convex bounding | Secondary globally valid master for nonconvex runs and explicit bound-validity reporting | Implemented | When local cuts make the primary master bound heuristic, the SHOT trace records a certified-bound master using only globally valid provenance rows and excluding integer/local rows. `gap_certified`, `bound_validity`, certified/heuristic bound sources, and convex-bounding update counts distinguish proof from incumbent quality. |
| Callbacks | Lazy cut callbacks, incumbent callbacks, external hyperplanes, external primal candidates, dual bounds, and termination hooks | Implemented for scoped backends | `tree_strategy="single_tree"` uses the Gurobi lazy-callback path for node and incumbent cuts. MultiTree OA exposes validated external primal-candidate, hyperplane, dual-bound, and termination hooks and records accepted/rejected/error counts. |
| Backend support | MIP backends including CPLEX/Gurobi/Cbc/HiGHS, NLP backends including Ipopt/GAMS/SHOT, AMPL/GAMS/OSiL interfaces, NEOS workflows | Partial / deferred | discopt supports HiGHS/POUNCE and optional Gurobi/cyipopt paths plus `.nl`, GAMS-link, and Pyomo interfaces. OSiL, NEOS, CPLEX, and Cbc parity are intentionally deferred by the roadmap. |
| Trace and diagnostics | Iteration reports, primal/dual bound updates, cut counts, NLP-call counts, solution-pool counts, and certification caveats | Implemented | `SolveResult.mip_nlp_trace` records normalized SHOT options, selected strategy, iteration records, summary counters, rootsearch/ESH events, repair/reduction events, external hooks, convex-bounding status, and certification fields. |

## Reformulation parity table

Issue [#141](https://github.com/bernalde/discopt/issues/141) added the
validated `mip_nlp_profile="shot"` control surface for SHOT-style
reformulation policy. Later roadmap issues attached active behavior where
discopt already has the needed native component; deferred SHOT interfaces remain
validated and trace-visible rather than silently ignored.

| SHOT reformulation mode | SHOT-profile control | discopt equivalent or status |
| --- | --- | --- |
| Objective epigraph for nonlinear minimization objectives | `objective_epigraph` | Active when the structural objective-defining-equality proof in `discopt._jax.objective_epigraph` applies; otherwise safely abstains. OA masters already use internal objective epigraph columns for convex nonlinear objectives. |
| Objective anti-epigraph for nonlinear maximization objectives | `anti_epigraph` | Active through the same objective-defining-equality pass for maximization models when the anti-epigraph inequality is convex. |
| Nonlinear expression partitioning | `nonlinear_partitioning` | Maps to the AMP/piecewise-McCormick partitioning stack (`solver="amp"` / GOA AMP path). MultiTree relaxation phases now trace when relaxed master candidates are introduced. |
| Quadratic partitioning | `quadratic_partitioning` | Current relaxation infrastructure supports quadratic and edge-concave relaxations; SHOT-style policy is validated and traced while direct quadratic routing handles supported convex quadratic classes. |
| Absolute-value auxiliaries | `absolute_value_auxiliaries` | `abs` has native convexity and relaxation support. Dedicated SHOT-style auxiliary introduction is traced and deferred until a model requires static aux variables. |
| Monomial extraction | `monomial_extraction` | Native term classification and McCormick/monomial auxiliary columns already extract integer-power monomials for relaxations. The control records whether SHOT policy wants that surface enabled. |
| Signomial extraction | `signomial_extraction` | Positive-domain monomial/posynomial recognition and signed-signomial DC envelopes exist; full SHOT signomial policy integration is deferred. |
| Integer-bilinear strategy | `integer_bilinear_strategy`, `integer_bilinear_max_bits` | Existing `integer_product_reform` can binary-expand bounded integer-factor bilinear terms into exact MILP form with a 12-bit default cap. The SHOT profile validates the strategy and limit for later routing. |
| Quadratic extraction strategy | `quadratic_extraction` | Structural LP/QP/QCP/MIQP/MIQCQP classification is used by the SHOT profile's strategy selection and recorded in trace fallback details. |
| Direct quadratic backend routing | `direct_quadratic_routing` | SHOT-profile routing now tries direct NLP, LP/MILP, MIQP, QCP, and MIQCQP classes when convexity is known and the requested backend supports the class; otherwise OA/GOA fallback is traced. |
| Reformulated primal NLP source selection | `fixed_nlp_strategy` plus reformulation controls | Fixed-NLP candidates are now scheduled from MIP optima, relaxation phases, solution pools, rootsearch points, and external providers. They use the active discopt model; a separate SHOT original-vs-reformulated NLP source registry is still out of scope. |

## Trace and certification envelope

Every MIP-NLP SHOT-profile solve returns `result.mip_nlp_trace` with
`schema_version=1`, normalized `shot_options`, selected strategy details,
iteration records, and a summary. The user-facing certification fields are:

| Field | Interpretation |
| --- | --- |
| `gap_certified` | True only when the final reported gap is backed by a finite certified bound. |
| `bound_validity` | `global` for a valid bound, `heuristic` when only local-cut or heuristic bounds are available, and `unavailable` when no finite bound is present. |
| `final_lb`, `final_ub`, `final_gap` | Certified lower bound, incumbent objective, and relative gap in discopt's minimization convention when available. |
| `heuristic_lb`, `heuristic_gap` | Best heuristic bound and gap, kept separate from certified quantities. |
| `certified_bound_source`, `heuristic_bound_source` | Last source that improved the corresponding bound, such as `primary_master`, `initial_poa`, `relaxation_phase`, `convex_bounding`, or `external`. |

Iteration and summary records cover cut provenance counts, ESH/rootsearch
events, fixed-NLP candidate/call sources, solution-pool candidates, Gurobi
master controls, relaxation phases, master repair, reduction cuts, external
hooks, single-tree callback statistics, and convex-bounding master outcomes.
These fields are the integration contract for final release testing; generated
benchmark rows must be read together with `gap_certified` and `bound_validity`.

## Benchmark command

Run the committed baseline with:

```bash
PYTHONPATH=python python scripts/shot_parity_baseline.py \
  --output docs/dev/data/shot-parity-baseline.json
```

If SHOT is built locally, provide the executable explicitly:

```bash
SHOT_EXECUTABLE=/home/bernalde/repos/SHOT/build/SHOT \
PYTHONPATH=python python scripts/shot_parity_baseline.py \
  --output docs/dev/data/shot-parity-baseline.json \
  --workdir /tmp/discopt-shot-baseline
```

When SHOT is available, the script exports each fixture to AMPL `.nl` and runs
the executable. When no executable is found, it records SHOT as `unavailable`.
That condition is part of the baseline; it should not be treated as a discopt
solver failure.

## Fixture set

| Fixture | Class | Convexity | Purpose |
| --- | --- | --- | --- |
| `convex_nlp` | NLP | Convex | Continuous nonlinear objective, direct KKT-certified solve. |
| `convex_minlp` | MINLP | Convex | OA baseline through `solver="mip-nlp"` with `mip_nlp_profile="shot"`. |
| `miqp` | MIQP-style | Convex | SHOT-profile direct MIQP routing with one binary variable. |
| `nonconvex_fp` | MINLP | Nonconvex | Feasibility-pump incumbent with no certified dual bound, exercising bound-validity reporting. |

## Current baseline notes

The committed JSON baseline is
`docs/dev/data/shot-parity-baseline.json`, regenerated on July 7, 2026. On this
workstation, the local SHOT checkout is present at `/home/bernalde/repos/SHOT`,
but no built SHOT executable was discoverable under the checkout. The SHOT rows
are therefore recorded as `unavailable` with an explicit caveat.

discopt solved the three certified fixtures as `optimal`: continuous NLP,
SHOT-profile OA on the convex MINLP fixture with direct routing disabled, and
SHOT-profile direct MIQP routing. The nonconvex feasibility-pump fixture remains
`feasible` with `gap_certified=false`; that row intentionally verifies that
heuristic incumbents are not promoted to proven optima.

A local SHOT-enabled run on June 30, 2026 matched the convex MINLP,
MIQP-style, and nonconvex incumbent rows, and exposed one parser caveat: SHOT
can report a feasible convex NLP incumbent while also stating that optimality
is not guaranteed. The benchmark script therefore records SHOT certification
from explicit global-optimality language and OSrL dual/primal bound fields
rather than treating any occurrence of the word "optimal" as proof.

## Certification caveats

- A finite objective without a finite certified bound is a heuristic incumbent,
  not a proof of optimality.
- Nonconvex rows must be read together with `gap_certified` and
  `bound_validity`; objective agreement alone is not enough.
- SHOT command-line results are parsed from stdout and OSrL best-effort fields.
  When SHOT is available, inspect the OSrL/trace files from `--workdir` for
  authoritative solver diagnostics.
