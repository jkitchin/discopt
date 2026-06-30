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
| Bound tightening | FBBT/OBBT and initial relaxation tightening before cut loops | Partial | root presolve, FBBT, OBBT, and nonlinear bound-tightening modules exist. The SHOT-style initial POA plus bound-tightening seed sequence is not yet implemented for the profile. |
| Cut generation | ESH/ECP/objective/external cuts, cut selection, hyperplane provenance, validity flags, and deduplication | Partial | OA/ECP cuts and selected no-good/integer cuts are present, and MIP-NLP traces count cuts. A unified cut/provenance model with ESH/rootsearch/objective hyperplanes is missing. |
| Primal heuristics | Fixed-integer NLP candidates from MIP optima, LP relaxations, solution pools, rootsearch, and external candidates | Partial | fixed-NLP calls, feasibility pump, initialization strategies, and some solution-pool plumbing exist. The full candidate manager, dynamic frequency, rootsearch, and external candidate sources remain missing. |
| Master repair | Infeasible-master repair, cutoff reset, repair-loop detection, and nonconvex reduction cuts | Missing | No dedicated SHOT-style master repair/reduction-cut loop is wired into the MIP-NLP profile. |
| Convex bounding | Secondary globally valid master for nonconvex runs and explicit bound-validity reporting | Partial | `SolveResult.gap_certified` and `mip_nlp_trace.bound_validity` distinguish certified from heuristic results. A separate SHOT-style convex-bounding master for nonconvex problems remains missing. |
| Callbacks | Lazy cut callbacks, incumbent callbacks, external hyperplanes, external primal candidates, dual bounds, and termination hooks | Partial | discopt has general solve callbacks and a Gurobi LP/NLP-BB callback path, but the SHOT external-event hook surface is not implemented. |
| Backend support | MIP backends including CPLEX/Gurobi/Cbc/HiGHS, NLP backends including Ipopt/GAMS/SHOT, AMPL/GAMS/OSiL interfaces, NEOS workflows | Partial / deferred | discopt supports HiGHS/POUNCE and optional Gurobi/cyipopt paths plus `.nl`, GAMS-link, and Pyomo interfaces. OSiL, NEOS, CPLEX, and Cbc parity are intentionally deferred by the roadmap. |
| Trace and diagnostics | Iteration reports, primal/dual bound updates, cut counts, NLP-call counts, solution-pool counts, and certification caveats | Partial | `SolveResult.mip_nlp_trace` records the current stable envelope for MIP-NLP runs. SHOT-equivalent trace fields for rootsearch, repair, external hooks, and convex bounding will be added as those features land. |

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

## Certification caveats

- A finite objective without a finite certified bound is a heuristic incumbent,
  not a proof of optimality.
- Nonconvex rows must be read together with `gap_certified` and
  `bound_validity`; objective agreement alone is not enough.
- SHOT command-line results are parsed from stdout and OSrL best-effort fields.
  When SHOT is available, inspect the OSrL/trace files from `--workdir` for
  authoritative solver diagnostics.
