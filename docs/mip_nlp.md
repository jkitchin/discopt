# MIP-NLP decomposition

`solver="mip-nlp"` selects discopt's algebraic MINLP decomposition family. It is
the user-facing entry point for outer approximation (OA), extended cutting plane
(ECP), feasibility-pump initialization, global OA routing, and the LP/NLP
branch-and-bound variant.

Use this solver family when the model is a bounded algebraic MINLP and you want
a decomposition algorithm built around MILP masters and fixed-integer NLP
subproblems. For harder nonconvex models, compare it with the certified-global
{doc}`AMP solver <notebooks/amp_global_minlp>`.

## Selector contract

MIP-NLP solving has two independent axes:

| Axis | Option | Purpose |
| --- | --- | --- |
| Solver family | `solver="mip-nlp"` | Route the model to the MIP-NLP decomposition facade. |
| Algebraic MINLP method | `mip_nlp_method=...` | Select OA, ECP, FP, GOA, or LP/NLP BB. |
| GDP reformulation | `gdp_method=...` | Reformulate logical/disjunctive constraints before algebraic MIP-NLP solving. |

`gdp_method` is not the MIP-NLP algorithm selector. It only controls GDP
reformulation, currently `big-m`, `hull`, `mbigm`, or `auto`, before the
algebraic MINLP method runs. Native GDP logic-based OA remains on the GDP axis
and is separate from `solver="mip-nlp"`.

```python
result = model.solve(
    solver="mip-nlp",
    mip_nlp_method="oa",
    gdp_method="hull",      # optional GDP reformulation axis
)
```

Legacy calls that used `gdp_method="oa"` to request MINLP OA should migrate to:

```python
result = model.solve(solver="mip-nlp", mip_nlp_method="oa")
```

If the same model also contains GDP constructs, keep `gdp_method` for the
reformulation choice:

```python
result = model.solve(
    solver="mip-nlp",
    mip_nlp_method="oa",
    gdp_method="big-m",
)
```

## Methods

| `mip_nlp_method` | Role | Notes |
| --- | --- | --- |
| `"oa"` | Convex MINLP outer approximation {cite:p}`Duran1986` | Default. Builds MILP masters and solves fixed-integer NLP subproblems. |
| `"ecp"` | Extended cutting plane | OA cut loop without fixed-integer NLP incumbent solves. |
| `"fp"` | Standalone MIP-NLP feasibility pump {cite:p}`Fischetti2005,Bonami2009` | Heuristic incumbent search; does not by itself certify optimality. |
| `"goa"` | Global OA routing | Convexity-certified models use OA; nonconvex models route through AMP/global relaxations. |
| `"lp_nlp_bb"` | LP/NLP branch-and-bound {cite:p}`Quesada1992` | Requires `milp_solver="gurobi"` because it uses single-tree lazy callbacks. |

`"roa"` is reserved for future regularized-OA naming. Use `"oa"` with
`add_regularization=...` for the implemented regularization modes.

## Outer approximation

OA is the default MIP-NLP method. It is appropriate when the nonlinear
constraints and objective are convex in the optimization sense and the integer
variables appear in a way the MILP master can approximate with valid cuts.

```python
result = model.solve(
    solver="mip-nlp",
    mip_nlp_method="oa",
    time_limit=120,
    gap_tolerance=1e-4,
    init_strategy="rNLP",
    add_no_good_cuts=True,
)
```

Useful OA options include:

| Option | Purpose |
| --- | --- |
| `init_strategy` | Initial incumbent strategy: `"rNLP"`, `"initial_binary"`, `"max_binary"`, or `"fp"`. |
| `equality_relaxation` | Relax nonlinear equalities into paired inequalities when supported. |
| `feasibility_cuts` | Add feasibility cuts when fixed-integer NLP subproblems fail. |
| `integer_to_binary` | Opt into linked binary expansions for bounded general integers so no-good cuts can exclude those assignments safely. |
| `add_slack`, `max_slack`, `oa_penalty_factor` | Allow penalized master slacks for robustness. |
| `feasibility_norm` | Norm used in feasibility-pump style repairs. |
| `solution_pool`, `num_solution_iteration` | Iterate through a MILP solution pool; currently requires `milp_solver="gurobi"`. |

## Experimental SHOT profile

`mip_nlp_profile="shot"` enables validated SHOT-parity controls and attaches a
structured `result.mip_nlp_trace` payload. The profile is experimental: some
controls already attach SHOT-style initialization, rootsearch, and ESH cut
behavior, while others are validated and traced so later PRs can add repair,
convex-bounding, reformulation policy, and adaptive master behavior behind a
stable interface. Default MIP-NLP behavior is unchanged when the profile is not
set.

```python
result = model.solve(
    solver="mip-nlp",
    mip_nlp_method="oa",
    mip_nlp_profile="shot",
    cut_strategy="esh",
    tree_strategy="multi_tree",
    fixed_nlp_strategy="adaptive",
    master_repair=True,
)
```

Accepted SHOT-profile controls are:

| Option | Accepted values |
| --- | --- |
| `tree_strategy` | `"auto"`, `"multi_tree"`, `"single_tree"` |
| `cut_strategy` | `"auto"`, `"oa"`, `"ecp"`, `"esh"` |
| `objective_epigraph`, `anti_epigraph` | `"auto"`, `"off"`, `"on"` |
| `nonlinear_partitioning`, `quadratic_partitioning` | `"auto"`, `"off"`, `"static"`, `"adaptive"` |
| `absolute_value_auxiliaries`, `monomial_extraction`, `signomial_extraction` | `"auto"`, `"off"`, `"on"` |
| `integer_bilinear_strategy` | `"auto"`, `"off"`, `"binary_expansion"`, `"mccormick"` |
| `integer_bilinear_max_bits` | Positive integer or `None` |
| `quadratic_extraction` | `"auto"`, `"off"`, `"native"`, `"relaxation"` |
| `direct_quadratic_routing` | `"auto"`, `"off"` |
| `rootsearch_strategy` | `"auto"`, `"none"`, `"bisection"`, `"toms748"` |
| `fixed_nlp_strategy` | `"auto"`, `"none"`, `"always"`, `"adaptive"`, `"iteration"`, `"time"`, `"solution_pool"` |
| `solution_pool_capacity`, `hyperplane_max_per_iter` | Positive integer or `None` |
| `hyperplane_selection_factor` | Floating-point factor in `(0, 1]` |
| `relaxation_phase` | `"auto"`, `"off"`, `"initial"`, `"periodic"` |
| `mip_solution_limit_strategy` | `"auto"`, `"none"`, `"static"`, `"adaptive"`, `"force_optimal"` |
| `convex_bounding`, `master_repair`, `reduction_cuts` | Boolean |

Passing one of these controls without `mip_nlp_profile="shot"` raises a
`ValueError`, which prevents accidental no-op configuration.

`objective_epigraph` and `anti_epigraph` are wired to discopt's existing
objective-defining-equality reformulation when the structural proof conditions
hold. `relaxation_phase="auto"` and `"initial"` run a SHOT-style initial
polyhedral approximation pass after the normal OA/ECP initializer: discopt solves
the current OA master with integrality relaxed, imports any generated
linearization cuts through the MIP-NLP cut provenance ledger, and updates the
initial master bound only when the active convexity certificates make that bound
globally valid. `relaxation_phase="off"` disables this pass, and failures fall
back to the existing initialization path with details in
`result.mip_nlp_trace["initial_poa"]`. `relaxation_phase="periodic"` also
solves an integrality-relaxed master before each integer master iteration and
records the phase state under each iteration's `relaxation_phase` trace entry.
Failures fall back to the existing OA/ECP path with details in the iteration
trace.

`direct_quadratic_routing="auto"` enables SHOT-profile direct strategy routing
before the multi-tree OA/GOA fallback. Despite the historical option name, this
gate covers every currently implemented direct class: continuous convex NLP,
LP/MILP, convex QP/MIQP, and convex QCP/QCQP/MIQCP/MIQCQP. Convexity-sensitive
classes route directly only when the classifier proves known convexity; unknown
or nonconvex models fall back to OA/GOA and record the fallback reason in
`result.mip_nlp_trace["strategy_selection"]["direct_attempt"]`.
`direct_quadratic_routing="off"` disables these direct routes. QCP-class direct
routing currently requires an explicit `milp_solver="gurobi"` request; with
`milp_solver="auto"` those models fall back to the normal MIP-NLP path instead
of probing an optional commercial backend.

SHOT-profile runs also maintain an internal provenance-aware interior-point
store for feasible relaxation, incumbent, and initial-POA points. Reusable
rootsearch helpers in `discopt.solvers.mip_nlp_rootsearch` support bisection and
optional TOMS748-compatible segment searches between stored interior points and
candidate MIP/NLP points, including fixed-discrete compatibility checks and
structured fallback statuses.

When ECP-style separation runs with `cut_strategy="esh"` or `"auto"`, discopt
first tries extended supporting hyperplanes from a compatible stored interior
point to the violated master candidate. Missing or incompatible interiors fall
back to ordinary ECP cuts, and the trace records the rootsearch status and
fallback reason. `hyperplane_selection_factor` ranks candidate hyperplanes by
violation and keeps that top fraction, while `hyperplane_max_per_iter` caps the
selected cuts. Convex nonlinear objective epigraphs can also receive
`objective_rootsearch` hyperplanes. Non-global local cuts are marked in cut
provenance, guarded so they do not cut off the known incumbent, and make reported
master bounds heuristic rather than certified.

The SHOT multi-tree loop also traces per-master orchestration controls. With
`milp_solver="gurobi"`, incumbent points are passed back to the master as MIP
starts, incumbent objectives become objective cutoffs when the active master
bound is globally valid, and `mip_solution_limit_strategy` can request native
Gurobi `SolutionLimit` behavior. `"static"` uses the configured
`solution_pool_capacity`/candidate cap, `"adaptive"` and `"auto"` start at one
solution and increase after nonproductive iterations, `"none"` disables the
limit, and `"force_optimal"` leaves the master uncapped. Unsupported backends do
not receive silent no-op parameters; the trace records degraded features under
`master_controls` and `summary["unsupported_backend_features"]`.

When `master_repair=True`, an infeasible SHOT-profile master triggers one native
repair retry with objective-cutoff and solution-limit controls reset and shared
master cut slacks enabled. The retry outcome is recorded in each iteration's
`repair_actions`; repeated repaired integer assignments are reported as repair
loops rather than cycled indefinitely. Repair is intentionally not restricted to
nonconvex heuristic masters: in convex runs it only relaxes relaxable OA cuts
through penalized slacks, and repaired integer assignments are still validated by
fixed-NLP subproblems before they can become incumbents. With
`reduction_cuts=True`, nonconvex
heuristic runs with a known incumbent can add a strict linear objective cutoff
row when the master objective row is exact. These reduction cuts are traced as
local, non-global cuts, so reported bounds remain heuristic unless a separate
certified bounding path is active.

`fixed_nlp_strategy="solution_pool"` or an explicit `solution_pool_capacity`
uses the Gurobi solution-pool candidate ingestion path when
`milp_solver="gurobi"`. Fixed-NLP candidates are ordered deterministically from
the master optimum, relaxation candidates, solution pool points, rootsearch
points, and external providers, with duplicate integer assignments removed
before NLP solves. The per-iteration trace records each fixed-NLP call source,
status, objective, warm-start source, and whether it improved the incumbent.
Other backends keep the single incumbent candidate and record the degradation in
the trace.

`tree_strategy="single_tree"` routes SHOT-profile OA/ECP requests to the
Gurobi-backed LP/NLP-BB callback solver. In that mode, discopt keeps the MIP
tree open, separates MIPNODE relaxation cuts where Gurobi exposes an optimal
node relaxation, and runs fixed-NLP incumbent handling from MIPSOL callbacks.
Callback-generated OA/ECP/feasibility/integer rows enter the same MIP-NLP cut
provenance ledger as MultiTree runs, and the result trace records callback
events, source counts, node counts, and whether an incumbent MIP start was
applied. Non-Gurobi MILP backends are explicitly MultiTree-only for this profile
until they expose equivalent persistent callback support.

The remaining reformulation and convex-bounding controls are validated and
traced under the SHOT profile so follow-up SHOT parity PRs can attach behavior
without changing the public option names.

Top-level aliases override duplicate entries in `mip_nlp_options`:

```python
result = model.solve(
    solver="mip-nlp",
    mip_nlp_method="oa",
    mip_nlp_options={"init_strategy": "rNLP", "add_no_good_cuts": False},
    init_strategy="fp",      # top-level value wins
)
```

## ECP

ECP uses the same selector family and OA option validation, but selects the
cutting-plane path explicitly:

```python
result = model.solve(
    solver="mip-nlp",
    mip_nlp_method="ecp",
    time_limit=120,
    feasibility_cuts=True,
)
```

`ecp_mode` is now derived from `mip_nlp_method`. Passing a conflicting explicit
`ecp_mode` raises `ValueError` instead of silently choosing one method.

## Feasibility-pump initialization

For OA, `init_strategy="fp"` runs the MIP-NLP feasibility pump to seed the first
incumbent before the OA loop.

```python
result = model.solve(
    solver="mip-nlp",
    mip_nlp_method="oa",
    init_strategy="fp",
    feasibility_norm="L_infinity",
)
```

You can also run the feasibility pump as the selected method:

```python
result = model.solve(
    solver="mip-nlp",
    mip_nlp_method="fp",
    feasibility_norm="L1",
)
```

The standalone FP method is an incumbent heuristic. It is useful for finding an
integer-feasible point quickly, but it is not a proof of global optimality unless
a later certifying method closes the gap.

MindtPy-style FP controls supported by discopt are intentionally scoped:

| Option | Semantics |
| --- | --- |
| `fp_iteration_limit` | Caps FP rounds independently from the OA iteration limit. OA/GOA FP initialization still defaults to `min(max_nodes, 10)` when this is omitted. |
| `fp_main_norm` | Norm used by the projection MILP distance objective. If omitted, it follows `feasibility_norm`. |
| `feasibility_norm` | Norm used to score nonlinear constraint violation in fixed-integer feasibility repair subproblems. |
| `fp_discrete_only` | Controls whether projection distance is computed only over discrete variables (`True`, the default) or over all variables. |
| `fp_projcuts` | Controls discopt's projection-MILP path with binary no-good cuts. When `False`, FP falls back to direct integer rounding. It does not transfer projection cuts into OA masters. |
| `fp_projzerotol` | Treats projection targets near zero as zero when zero is within bounds. |
| `fp_mipgap` | Gap tolerance for FP projection MILPs; defaults to `gap_tolerance`. |

The MindtPy names `fp_transfercuts=True`, `fp_norm_constraint=True`,
non-default `fp_norm_constraint_coef`, and nonzero `fp_cutoffdecr` are not
implemented by discopt's FP path yet. Passing them raises `ValueError` instead
of silently changing neither the projection loop nor OA/GOA certificate state.
FP-generated cuts are therefore local to the pump; `init_strategy="fp"` may seed
OA/GOA incumbents and initial cuts at the pump point, but it does not import
projection no-good cuts into certified OA bounds.

## Regularized OA

Regularization is selected as an OA option, not as a separate method selector.
Implemented modes are `level_L1`, `level_L2`, `level_L_infinity`, `grad_lag`,
`hess_lag`, `hess_only_lag`, and `sqp_lag`.

```python
result = model.solve(
    solver="mip-nlp",
    mip_nlp_method="oa",
    init_strategy="rNLP",
    add_regularization="level_L1",
    level_coef=0.5,
)
```

Derivative-based regularization modes use Lagrangian information from an
initial fixed-integer NLP. They therefore need an NLP-based initialization that
returns dual information; constrained models reject derivative regularization
with `init_strategy="fp"` because FP does not provide that data.

QP-shaped regularization modes such as `level_L2`, `hess_lag`,
`hess_only_lag`, and `sqp_lag` require an available MIQP/QP-capable master
backend. If no backend can solve the regularized master, discopt raises a backend
error before claiming progress.

## GOA

`mip_nlp_method="goa"` is the global-OA facade. It first checks whether the model
is convexity-certified. Convexity-certified models run through OA. Other models
use the AMP/global-relaxation path, so AMP options may be passed through the same
call.

```python
result = model.solve(
    solver="mip-nlp",
    mip_nlp_method="goa",
    rel_gap=1e-4,
    n_init_partitions=4,
    partition_method="adaptive",
    milp_solver="highs",
)
```

AMP-only options are used only when GOA routes to the nonconvex AMP path. If GOA
routes a convexity-certified model to OA, discopt warns about AMP-only options
that do not apply.

## LP/NLP branch-and-bound

The LP/NLP BB variant keeps the master tree open and adds OA cuts lazily at
integer assignments. That single-tree design requires callback support, so the
implemented backend is currently Gurobi.

```python
result = model.solve(
    solver="mip-nlp",
    mip_nlp_method="lp_nlp_bb",
    milp_solver="gurobi",
    init_strategy="rNLP",
)
```

Using another MILP backend with `lp_nlp_bb` raises a backend error. This does
not mean general nonlinear expressions are sent to Gurobi; discopt still owns
the MIP-NLP decomposition and uses Gurobi for the callback-capable LP/MILP
master.

## Backend limitations

- `nlp_solver` controls fixed-integer NLP subproblems. POUNCE is the default
  single-solve backend when installed, with cyipopt and the pure-JAX IPM used
  where configured.
- `milp_solver` controls MILP masters where a method exposes that option.
  `highs`, `pounce`, `simplex`, `gurobi`, or `auto` may be accepted depending on
  the method and installed extras.
- `solution_pool=True` currently requires `milp_solver="gurobi"`.
- `lp_nlp_bb` requires `milp_solver="gurobi"` for lazy callbacks.
- MIP-NLP methods ignore branch-and-bound-only options such as spatial branching
  policy, learned relaxations, and sub-NLP frequency. discopt reports ignored
  solve options with a warning rather than silently applying them.

## Result interpretation

Read the returned `SolveResult` the same way as other discopt solvers:

```python
result.status
result.objective
result.bound
result.gap
result.gap_certified
```

For OA, ECP, GOA-on-OA, and LP/NLP BB, a certified result requires a valid bound
and closed gap. For standalone FP, treat the result as a heuristic incumbent
unless another method has subsequently certified it.
