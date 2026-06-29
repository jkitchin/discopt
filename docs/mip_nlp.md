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
| `add_slack`, `max_slack`, `oa_penalty_factor` | Allow penalized master slacks for robustness. |
| `feasibility_norm` | Norm used in feasibility-pump style repairs. |
| `solution_pool`, `num_solution_iteration` | Iterate through a MILP solution pool; currently requires `milp_solver="gurobi"`. |

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
