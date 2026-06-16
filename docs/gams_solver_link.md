# Using discopt as a GAMS solver

discopt can act as a **solver inside GAMS**: once registered, a GAMS user solves
a model with discopt the same way they would call BARON, SCIP, or CONOPT:

```gams
option minlp = discopt;
solve m using minlp minimizing z;
```

This is the inverse of the file converters (`from_gams()` / `to_gams()`): instead
of translating a `.gms` file, discopt plugs directly into the GAMS *solve* call.

## How the link works

GAMS does not pass solvers a model file. When a registered solver is invoked,
GAMS launches it with a single argument — the path to a **control file** — and
expects the solution to be written back through two libraries:

- **GEV** (GAMS Environment Object): options, logging, resource limits.
- **GMO** (GAMS Modeling Object): the model instance — columns (variables,
  bounds, types), the objective, the Jacobian, and the *nonlinear instruction
  lists* for the objective and every constraint row.

The discopt link (`discopt.gams.link`) does the following:

1. Boots GEV/GMO from the control file.
2. Reads each column into a scalar discopt variable, with bounds and
   continuous/binary/integer type.
3. Rebuilds the objective and every constraint from the GMO linear coefficients
   **and** a faithful translation of the GMO nonlinear instruction lists. The
   instruction lists are a reverse-Polish stack machine (push variable, push
   constant, add, multiply, call `exp`/`log`/`sqr`/`power`/…); the translator in
   `discopt.gams.instructions` walks them into discopt's expression DAG, mirroring
   the function mapping used by `from_gams()` so the result is identical to
   parsing the equivalent `.gms`.
4. Solves the resulting `discopt.Model`.
5. Writes the primal solution and the GAMS model/solve status back into GMO.

The model is reconstructed natively, so the full discopt stack applies:
spatial branch-and-bound, McCormick/αBB relaxations, FBBT presolve, and the
convex fast path.

## Installation and registration

The link needs the GAMS expert-level Python API (the GMO/GEV bindings). It ships
with every GAMS system and is also on PyPI:

```bash
pip install "discopt[gams]"   # pulls gamsapi[core]
```

Register discopt with your GAMS system:

```bash
discopt gams-register --directory ./discopt-gams-config
```

This writes two files:

- `gamsconfig.yaml` — a `solverConfig` block declaring the `discopt` solver and
  the model types it accepts (LP, MIP, NLP, DNLP, RMINLP, MINLP, QCP, MIQCP, …).
  Merge it into the `gamsconfig.yaml` in your GAMS system directory (or
  `$HOME/.gams`).
- `discopt-gams` — a small launcher script GAMS runs with the control file; it
  invokes `python -m discopt.gams.link`.

After registration, `option minlp = discopt;` (and the analogous options for the
other model types) dispatches to discopt.

## Low latency: the warm daemon

GAMS launches a solver as a fresh process for every solve, so a naive link
re-pays Python and JAX import plus first-JIT warmup each time, even though a
*warm* discopt solve is on the order of 10 ms. Two things keep the link fast:

- **Lazy JAX.** `import discopt` no longer imports JAX eagerly (it sets
  `JAX_ENABLE_X64` in the environment instead), so LP/MILP solves — which run
  entirely in the Rust core — never load JAX at all.
- **A warm solver daemon** (`discopt.gams.daemon`). The first solve spawns a
  detached, long-lived process that holds JAX and the JIT cache warm; every
  subsequent GAMS solve becomes a thin unix-socket round-trip to it. The daemon
  self-terminates on idle timeout, max lifetime, max solves, or a version
  change, and the client lazily respawns it — so it never needs explicit
  management. If the daemon is unreachable and cannot be started, the link
  falls back to an in-process solve, so correctness never depends on it.

The daemon is on by default. Controls:

```bash
discopt gams-daemon status      # is one running?
discopt gams-daemon stop        # ask it to exit
DISCOPT_GAMS_NO_DAEMON=1 ...     # bypass the daemon, always solve in-process
```

Tunables (env vars): `DISCOPT_GAMS_IDLE_TIMEOUT` (default 600 s),
`DISCOPT_GAMS_MAX_LIFETIME` (3600 s), `DISCOPT_GAMS_MAX_SOLVES` (500),
`DISCOPT_GAMS_SOCKET` (socket path). Concurrent solves are serialized through
the one warm interpreter.

## Supported functions

The translator supports the algebraic operators (`+ - * / **`, unary minus) and
the intrinsic functions discopt models natively: `exp`, `log`, `log2`, `log10`,
`sqrt`, `sqr`, `abs`, `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `sinh`,
`cosh`, `tanh`, `sigmoid`, `sign`, `errf`, the `power`/`rpower`/`cvpower`/
`vcpower` family, `div`, and `min`/`max`. A GAMS intrinsic outside this set
raises a clear `GamsTranslationError` naming the function rather than silently
producing a wrong model.

## Status mapping

discopt termination statuses are mapped to GAMS `(modelStat, solveStat)` pairs:
`optimal` → Optimal/Integer + Normal, `feasible` → Feasible + Normal,
`infeasible` → Infeasible + Normal, `time_limit`/`node_limit` → Resource (with
Feasible if an incumbent exists, otherwise No-Solution-Returned), and errors →
Error/Solver-Error.

## Smoke testing

A small corpus of GAMS models with known optima lives in
`python/tests/data/gams/` (LP, MIP, convex NLP, nonconvex NLP, and a convex
MINLP), described by `manifest.json`. There are two layers of checking:

- **Reader path (no GAMS needed)** — the unit tests parse each model with
  `from_gams()` and solve it with discopt, asserting the known optimum
  (`pytest python/tests/test_gams_link.py -k smoke`).
- **Solver-link path (needs a GAMS install)** — `scripts/verify_gams_link.py`
  runs every model through GAMS with the solver forced to discopt, reads back the
  objective and GAMS model/solve status, and checks them against the optimum:

  ```bash
  make gams-install   # build + register discopt with GAMS
  make gams-verify    # python scripts/verify_gams_link.py
  ```

The canonical `lp_transport.gms` (sets/tables) is included for the solver-link
path; discopt's `from_gams()` reader does not yet substitute indexed parameter
data into objective coefficients, so it is flagged GAMS-only in the manifest.

## Programmatic use

The GAMS-library-free core is importable and testable on its own — useful when
embedding GMO instances produced by other tooling:

```python
from discopt.gams import solve_view  # takes any object implementing GmoView
model, result = solve_view(my_gmo_view)
print(result.status, result.objective)
```
