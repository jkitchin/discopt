---
name: discopt
description: Build, solve and interpret optimization models with discopt, a global MINLP solver (Python API, Rust core). Use when the user wants to formulate an LP, MILP, NLP or mixed-integer nonlinear problem in Python, solve it with discopt, read a SolveResult, or import/export .nl and GAMS models.
license: EPL-2.0
compatibility: Requires Python 3.12+ and `pip install discopt`.
---

# discopt

discopt is a mixed-integer nonlinear programming (MINLP) solver. It finds
**certified global optima** by spatial branch and bound over convex relaxations,
so a result marked `optimal` comes with a proved bound, not only a good point.

## Build a model

```python
import discopt.modeling as dm

m = dm.Model("example")
x = m.continuous("x", lb=0, ub=5)
y = m.continuous("y", lb=0, ub=5)
z = m.binary("z")
# also: m.integer("n", lb=0, ub=10), and shape=(n,) for arrays

m.minimize(x**2 + y**2 + z)        # or m.maximize(...)
m.subject_to(x + y >= 1, name="cover")
m.subject_to(x**2 + y <= 3, name="cap")

m.validate()                       # catch modeling errors before solving
result = m.solve(time_limit=60, gap_tolerance=1e-4)
```

Guidelines:

- Give every variable **finite bounds** whenever the problem allows it. Global
  solvers relax nonlinear terms over the variable box, so tight bounds give
  tighter relaxations and faster certificates.
- Name every constraint (`name=...`); diagnostics refer to them by name.
- Use `dm.exp`, `dm.log`, `dm.sqrt`, `dm.sin`, ... and `dm.sum(...)` instead of
  `numpy` or `math` functions on decision variables.
- Vectorize with `shape=` and `@` instead of Python loops over scalars.

## Read the result

| field | meaning |
|---|---|
| `status` | `optimal`, `feasible`, `infeasible`, `unbounded`, `time_limit`, `node_limit`, `iteration_limit`, `error`, or a local status (`local_optimal`, `local_infeasible`) |
| `objective` | objective value of the best point found (`None` if none) |
| `bound` | proved dual bound (lower bound for minimization) |
| `gap_certified` | `True` only when the gap between `objective` and `bound` is proved |
| `x` | dict of variable name to value (numpy arrays) |
| `node_count`, `wall_time` | search effort |

Interpretation rules:

- Only `optimal` (with `gap_certified=True`) is a global optimality claim.
  `feasible` and `time_limit` give a valid point and a valid bound, but no claim
  that the point is optimal; report the gap.
- `local_optimal` and `local_infeasible` come from a local solver and carry no
  global claim. In particular, `local_infeasible` is **not** a proof that the
  model is infeasible.
- If no bound is reported, part of the model could not be relaxed (for example an
  opaque external function). Say so rather than presenting the point as optimal.

## Files and other formats

```python
m = dm.from_nl("problem.nl")   # AMPL .nl (MINLPLib, AMPL, Pyomo exports)
```

discopt also reads GAMS `.gms` and QPLIB files and writes `.nl`, `.lp`, `.mps` and
GAMS. Pyomo users can call it via `SolverFactory("discopt")`
(`pip install discopt[pyomo]`).

## Going further

In Claude Code, `uvx knowhow add jkitchin/discopt --all` (or
`discopt install-skills`) also installs task-specific slash commands
(`/formulate`, `/diagnose`, `/reformulate`, `/convert`, `/explain-model`,
`/estimate`, `/debug`, `/benchmark-report`) and expert subagents.

Documentation: https://github.com/jkitchin/discopt
