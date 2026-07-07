# Using discopt as a Pyomo solver

discopt ships an optional [Pyomo](https://pyomo.org) plugin so existing Pyomo
models can be solved with discopt through the familiar `SolverFactory` interface.

## Installation

```bash
pip install discopt[pyomo]
```

## Usage

```python
import pyomo.environ as pyo
import discopt.pyomo  # registers the 'discopt' solver

m = pyo.ConcreteModel()
m.x = pyo.Var(bounds=(0, 10))
m.y = pyo.Var(domain=pyo.Binary)
m.obj = pyo.Objective(expr=(m.x - 3) ** 2 + 2 * m.y, sense=pyo.minimize)
m.c = pyo.Constraint(expr=m.x + 5 * m.y >= 4)

opt = pyo.SolverFactory("discopt")
results = opt.solve(m, tee=True)

print(results.solver.termination_condition)   # optimal
print(m.x.value, m.y.value)                    # 4.0  0
```

Activation: `import discopt.pyomo` (or `discopt.pyomo.register()`) registers the
solver. A `pyomo.solvers` entry point is also declared so discovery can happen
automatically after installation.

## Options

Pyomo solver options map to `Model.solve` keyword arguments:

| Pyomo option | discopt `solve()` kwarg |
|---|---|
| `timelimit` (or `time_limit`) | `time_limit` |
| `mipgap` / `gap` | `gap_tolerance` |
| `threads` | `threads` |
| `tee=True` | `stream=True` |

Pass them per solve or persistently:

```python
opt.solve(m, options={"timelimit": 60, "mipgap": 1e-4})
# or
opt.options["timelimit"] = 60
```

Unrecognised options are forwarded verbatim to `Model.solve`, so any discopt solve
keyword works without a plugin change.

## Duals and reduced costs

Declare an `import` Suffix and discopt's KKT multipliers are loaded **best-effort**:

```python
m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
m.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)
opt.solve(m)
print(m.dual[m.c], m.rc[m.x])
```

Duals follow the AMPL/Pyomo sign convention (cross-checked against ipopt). They are
populated only when discopt exposes multipliers — i.e. on the NLP/KKT path. For
problems solved by another route the Suffix is simply left empty; values are never
fabricated.

## How it works and limitations

The bridge round-trips the model through a temporary AMPL `.nl` file in-process:
Pyomo's NL writer emits the `.nl` (presolve and scaling disabled so it stays in the
original variable space), `discopt.from_nl` reads it with the **same column/row
order**, `Model.solve` runs, and the solution is mapped back by index. Consequences:

- **Variable/constraint values map by column/row order, not name** — Pyomo and
  discopt name things differently; this is handled internally.
- **Constraint names are not preserved** through `.nl` (cosmetic; solving and dual
  mapping are unaffected).
- Models Pyomo can write to `.nl` but whose operators discopt's reader does not
  support return a structured `error` result with a message rather than crashing.

## Importing a Pyomo model as a discopt `Model`

To get a native discopt `Model` from a Pyomo `ConcreteModel` (rather than just
solving it), use `discopt.modeling.from_pyomo`:

```python
import pyomo.environ as pyo
import discopt.modeling as dm

m = pyo.ConcreteModel()
m.x = pyo.Var(bounds=(0, 10))
m.obj = pyo.Objective(expr=(m.x - 3) ** 2)

dmodel = dm.from_pyomo(m)   # a discopt Model
result = dmodel.solve()
```

`from_pyomo` reuses the same temporary `.nl` round-trip as the solver plugin, so
variables/constraints come back in `.nl` column/row order (names differ from the
Pyomo model). A future direct in-memory translator may replace the `.nl`
round-trip without changing this interface.
