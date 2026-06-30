# SHOT parity audit and benchmark baseline

Issue: [#138](https://github.com/bernalde/discopt/issues/138)

This audit establishes the read-only baseline for the SHOT port roadmap on
`feature/mip-nlp-solver`. It compares current discopt behavior against the SHOT
capability surface before deeper solver changes land. The target is
discopt-native behavioral parity, not a translation of SHOT C++ internals.

## Feature matrix

| Area | SHOT capability | discopt status | Current discopt evidence and gap |
| --- | --- | --- | --- |
| Strategy selection | MultiTree and SingleTree strategies, direct NLP/MIQP/MIQCQP handling, and solver-specific master routing | Partial | `solver="mip-nlp"` exposes OA, ECP, FP, GOA, and LP/NLP-BB method selection. The SHOT profile validates reserved strategy controls, but automatic SHOT-style direct routing and adaptive MultiTree phases remain follow-ups. |
| Reformulation | Objective epigraphs, nonlinear/quadratic partitioning, signomial and monomial handling, integer-bilinear strategy choices, and optional use of reformulated primal NLPs | Partial | discopt has GDP reformulation, entropy canonicalization, `.nl` export, McCormick infrastructure, and selected integer-product reformulations. SHOT's full reformulation policy surface is not yet mapped into MIP-NLP options. |
| Bound tightening | FBBT/OBBT and initial relaxation tightening before cut loops | Partial | root presolve, FBBT, OBBT, and nonlinear bound-tightening modules exist. The SHOT profile now has an initial POA seed pass that solves the current OA master relaxation, imports valid linearization cuts through provenance, and falls back explicitly. Full SHOT-style OBBT scheduling and richer interior-point services remain follow-ups. |
| Cut generation | ESH/ECP/objective/external cuts, cut selection, hyperplane provenance, validity flags, and deduplication | Partial | OA/ECP cuts and selected no-good/integer cuts are present, and MIP-NLP traces count cuts. A unified cut/provenance model with ESH/rootsearch/objective hyperplanes is missing. |
| Primal heuristics | Fixed-integer NLP candidates from MIP optima, LP relaxations, solution pools, rootsearch, and external candidates | Partial | fixed-NLP calls, feasibility pump, initialization strategies, and some solution-pool plumbing exist. The full candidate manager, dynamic frequency, rootsearch, and external candidate sources remain missing. |
| Master repair | Infeasible-master repair, cutoff reset, repair-loop detection, and nonconvex reduction cuts | Missing | No dedicated SHOT-style master repair/reduction-cut loop is wired into the MIP-NLP profile. |
| Convex bounding | Secondary globally valid master for nonconvex runs and explicit bound-validity reporting | Partial | `SolveResult.gap_certified` and `mip_nlp_trace.bound_validity` distinguish certified from heuristic results. A separate SHOT-style convex-bounding master for nonconvex problems remains missing. |
| Callbacks | Lazy cut callbacks, incumbent callbacks, external hyperplanes, external primal candidates, dual bounds, and termination hooks | Partial | discopt has general solve callbacks and a Gurobi LP/NLP-BB callback path, but the SHOT external-event hook surface is not implemented. |
| Backend support | MIP backends including CPLEX/Gurobi/Cbc/HiGHS, NLP backends including Ipopt/GAMS/SHOT, AMPL/GAMS/OSiL interfaces, NEOS workflows | Partial / deferred | discopt supports HiGHS/POUNCE and optional Gurobi/cyipopt paths plus `.nl`, GAMS-link, and Pyomo interfaces. OSiL, NEOS, CPLEX, and Cbc parity are intentionally deferred by the roadmap. |
| Trace and diagnostics | Iteration reports, primal/dual bound updates, cut counts, NLP-call counts, solution-pool counts, and certification caveats | Partial | `SolveResult.mip_nlp_trace` records the current stable envelope for MIP-NLP runs. SHOT-equivalent trace fields for rootsearch, repair, external hooks, and convex bounding will be added as those features land. |

## Reformulation parity table

Issue [#141](https://github.com/bernalde/discopt/issues/141) adds the
validated `mip_nlp_profile="shot"` control surface for SHOT-style
reformulation policy. Only controls backed by an existing discopt component or
needed as stable names for later roadmap PRs are exposed.

| SHOT reformulation mode | SHOT-profile control | discopt equivalent or status |
| --- | --- | --- |
| Objective epigraph for nonlinear minimization objectives | `objective_epigraph` | Active when the structural objective-defining-equality proof in `discopt._jax.objective_epigraph` applies; otherwise safely abstains. OA masters already use internal objective epigraph columns for convex nonlinear objectives. |
| Objective anti-epigraph for nonlinear maximization objectives | `anti_epigraph` | Active through the same objective-defining-equality pass for maximization models when the anti-epigraph inequality is convex. |
| Nonlinear expression partitioning | `nonlinear_partitioning` | Maps to the AMP/piecewise-McCormick partitioning stack (`solver="amp"` / GOA AMP path). Full SHOT MultiTree phase scheduling is deferred to the POA and MultiTree issues. |
| Quadratic partitioning | `quadratic_partitioning` | Current relaxation infrastructure supports quadratic and edge-concave relaxations; SHOT-style policy scheduling is traced and deferred. |
| Absolute-value auxiliaries | `absolute_value_auxiliaries` | `abs` has native convexity and relaxation support. Dedicated SHOT-style auxiliary introduction is traced and deferred until a model requires static aux variables. |
| Monomial extraction | `monomial_extraction` | Native term classification and McCormick/monomial auxiliary columns already extract integer-power monomials for relaxations. The control records whether SHOT policy wants that surface enabled. |
| Signomial extraction | `signomial_extraction` | Positive-domain monomial/posynomial recognition and signed-signomial DC envelopes exist; full SHOT signomial policy integration is deferred. |
| Integer-bilinear strategy | `integer_bilinear_strategy`, `integer_bilinear_max_bits` | Existing `integer_product_reform` can binary-expand bounded integer-factor bilinear terms into exact MILP form with a 12-bit default cap. The SHOT profile validates the strategy and limit for later routing. |
| Quadratic extraction strategy | `quadratic_extraction` | Structural LP/QP/QCP/MIQP/MIQCQP classification exists in `problem_classifier`; MIP-NLP direct-routing policy is deferred to issue #151. |
| Direct quadratic backend routing | `direct_quadratic_routing` | Existing top-level `solver="gurobi"` and default problem classification can route LP/QP/QCP/MIQP/MIQCQP when selected safely. Automatic SHOT-style routing from `solver="mip-nlp"` is deferred to issue #151. |
| Reformulated primal NLP source selection | `fixed_nlp_strategy` plus reformulation controls | Current fixed-NLP calls use the active model passed to OA/GOA. Original-vs-reformulated NLP candidate policy is deferred to issue #146. |

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
| `miqp` | MIQP-style | Convex | Quadratic objective with one binary variable to pressure direct quadratic routing. |
| `nonconvex_fp` | MINLP | Nonconvex | Feasibility-pump incumbent with no certified dual bound, exercising bound-validity reporting. |

## Current baseline notes

The committed JSON baseline is
`docs/dev/data/shot-parity-baseline.json`. On this workstation, the local SHOT
checkout is present at `/home/bernalde/repos/SHOT`, but `SHOTpy` is not
importable and no built SHOT executable was discoverable under the checkout.
The SHOT rows are therefore recorded as `unavailable` with an explicit caveat.

discopt solved the three certified fixtures as `optimal` and the nonconvex
feasibility-pump fixture as `feasible` with `gap_certified=false`. That
uncertified row is intentional: it is the baseline that later nonconvex
convex-bounding work must improve without overstating global guarantees.

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
