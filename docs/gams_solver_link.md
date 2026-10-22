# Using discopt as a GAMS solver

discopt can act as a **solver inside GAMS**: once registered, a GAMS user solves
a model with discopt the same way they would call BARON, SCIP, or CONOPT:

```text
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

## Install discopt as a GAMS solver

Two steps — install the package, then register it with GAMS:

```bash
pip install "discopt[gams]"   # discopt (prebuilt wheel) + gamsapi[core] + PyYAML
discopt gams-register          # writes ~/.gams/gamsconfig.yaml + the launcher
```

Then, in any GAMS model:

```text
option minlp = discopt;        // and/or lp/mip/nlp/... = discopt
solve m using minlp minimizing z;
```

`discopt` ships as a prebuilt wheel (the Rust extension is compiled in CI), so
no Rust toolchain is needed. The expert-level GAMS Python API (`gamsapi[core]`,
the GMO/GEV bindings) ships with every GAMS system and is also on PyPI.

### What `gams-register` does

- **Merges** a `discopt` entry into `gamsconfig.yaml` in the target directory
  (default `$HOME/.gams`, which GAMS reads automatically), declaring the model
  types discopt accepts (LP, MIP, NLP, DNLP, RMINLP, MINLP, QCP, MIQCP, …). It
  preserves any other solvers and top-level keys already there, and re-running
  it just updates the discopt entry (no duplicates). Use `--directory` to target
  your GAMS system dir instead.
- Writes the **`discopt-gams`** launcher GAMS runs with the control file. The
  launcher hardcodes the current Python interpreter (`sys.executable`), so it
  works regardless of virtualenv activation or `PATH`.

### Two gotchas worth knowing

- **Match `gamsapi` to your GAMS version.** The GMO/GEV bindings `dlopen` the
  GAMS C libraries, so the `gamsapi` version must agree with your installed
  GAMS. The safest source is the one bundled with your GAMS
  (`<GAMS_DIR>/apifiles/Python/api/...`); otherwise pin the PyPI release to your
  GAMS version. Verify with:

  ```bash
  discopt gams-register --check    # reports gamsapi import + version, or how to fix
  ```

- **Re-register if you move the Python environment**, since the launcher
  embeds an absolute interpreter path.

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

### Isolation and memory hygiene

Reusing one process across solves is safe because each solve is self-contained:
`solve_from_control_file` creates **fresh** GMO/GEV handles and `model_from_gmo`
builds a **new** `Model`, so no mutable problem state is shared between solves
(the only reuse is the warm interpreter, imported modules, and JAX's
content-addressed JIT cache). Native handles are released on every solve via
`try/finally` (`gmoFree`/`gevFree`), so the daemon does not leak GMO memory.

Backstops bound any residual growth (e.g. JAX cache across many problem shapes).
Every guard treats **0 (or negative) as "no limit"**:

| Env var | Default | Effect |
|---|---|---|
| `DISCOPT_GAMS_IDLE_TIMEOUT` | 600 s | exit after this long with no requests |
| `DISCOPT_GAMS_MAX_LIFETIME` | 3600 s | exit after this wall-clock age |
| `DISCOPT_GAMS_MAX_SOLVES` | 500 | exit after this many solves |
| `DISCOPT_GAMS_MAX_RSS_MB` | 0 (off) | exit once resident memory exceeds this |
| `DISCOPT_GAMS_JAX_CLEAR_EVERY` | 0 (off) | `jax.clear_caches()` every N solves |
| `DISCOPT_GAMS_SOCKET` | per-user | socket path |

Recycle guards are evaluated only after an actual solve, and the client lazily
respawns, so recycling is transparent. Concurrent solves are serialized through
the one warm interpreter.

### Benchmark studies

The count/age guards would otherwise recycle the daemon mid-study and inject
re-warmup latency into your timings. Set `DISCOPT_GAMS_BENCHMARK=1` to keep one
warm daemon for the whole run: it disables the solve-count and lifetime limits
and raises the idle timeout to 30 min, leaving `DISCOPT_GAMS_MAX_RSS_MB` as the
only (opt-in) backstop. Pre-warm and pin it explicitly for clean measurements:

```bash
export DISCOPT_GAMS_BENCHMARK=1
export DISCOPT_GAMS_MAX_RSS_MB=16384   # optional safety net
discopt gams-daemon serve &            # pre-warm before timing
# ... run the GAMS benchmark sweep ...
discopt gams-daemon stop
```

## Supported functions

The translator supports the algebraic operators (`+ - * / **`, unary minus) and
the intrinsic functions discopt models natively: `exp`, `log`, `log2`, `log10`,
`sqrt`, `sqr`, `abs`, `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `sinh`,
`cosh`, `tanh`, `sigmoid`, `sign`, `errf`, the `power`/`rpower`/`cvpower`/
`vcpower` family, `div`, `min`/`max`, and `arctan2` (see below).

`errf` is GAMS's `errorf`, the standard normal CDF `(1 + erf(x/√2))/2` (not
Gauss's `erf`). `entropy` is GAMS's `-x·ln x`, translated as the negation of
discopt's convex `x·log x` atom, which has a rigorous relaxation, so it is
solved to a **certified global** optimum. `centropy(x, y[, Z])` is translated
exactly as `x·ln((x+Z)/(y+Z))` with GAMS's default `Z = 1e-20`.

The **smoothed extensions** `slexp`/`sqexp`, `sllog10`/`sqlog10` and
`slrec`/`sqrec` are *different functions* from their base function beyond the
threshold `S` (the second argument; GAMS defaults 150, 1e-150 and 1e-10): GAMS
continues them linearly (`sl`) or quadratically (`sq`) there, e.g.
`slrec(0.5, 1) = 1.5`, not `2`. Each is translated exactly through a clamp
`max(x, S)` / `min(x, S)` and the excursion past `S`, which discopt relaxes, so
the translation stays global. (Earlier versions mapped them to the base
function, which is wrong wherever the model reaches past `S`.) `div0(a, b)` is
`a/b`; at `b = 0`, where GAMS returns `1e299`, discopt raises an evaluation
error, so that point is never returned as a solution.

`arctan2` is rewritten **exactly** when the imported bounds put the whole box in
one half-plane — `atan(y/x)` for `x > 0`, `±π/2 − atan(x/y)` for sign-definite
`y` — which turns it into ordinary `atan` over a ratio and makes it solvable to
a **certified global** optimum. See `discopt.modeling._atan2`; `dm.atan2` uses
the same rewrite but *raises* when it cannot fire, whereas the link keeps the
opaque node (an imported model's bounds are not the importer's to change).

The remaining two-argument forms — `signpower`, and `atan2` on a box
that straddles its branch cut — have **no rigorous relaxation** and are solved on
the local path. Be precise about why, because an earlier version of this section
said the opposite: the αBB fallback does *not* rescue them. `rigorous_alpha`
needs an interval Hessian, and `interval_ad._function_call` abstains on **every**
call with more than one argument before it even looks at the name, so αBB returns
`+inf` and declines. Their enclosure then comes from `uniform_relax`'s
interval floor, which is sound but carries no dual bound (for a raw `atan2` node
that floor is the unconditionally valid `[-π, π]`; for the others it is
unbounded). discopt sets `gap_certified` only when it produced a valid finite
dual bound, so the link reports these as `LocallyOptimal` rather than
overstating a local solution as global (see Status mapping).

**Discontinuous** intrinsics (`ceil`, `floor`, `round`, `trunc`, `frac`, `mod`,
`ifthen`) have no valid continuous relaxation and are rejected with a clear
`GamsTranslationError`; model that behaviour with integer variables instead. Any
other unsupported intrinsic likewise raises rather than producing a wrong model.
A refused model is reported to GAMS as solve status `6 Capability Problems` /
model status `14 No Solution Returned`, with the reason written to the log and
listing.

## Variables, rows and options

Binary, integer, semicontinuous and semi-integer variables are translated
exactly. A fixed binary keeps its bounds; an integer's GAMS default upper bound
`+INF` stays infinite. A semicontinuous (or semi-integer) `x` with bounds
`[lo, up]` means `x = 0` or `lo ≤ x ≤ up` and is lowered with an on/off binary,
as the `.gms` reader does; one whose disjunction would need an infinite bound is
refused. **SOS1/SOS2** variables are refused (reading them as continuous would
drop the restriction). `=N=` rows impose nothing and are skipped; `=X=`, `=C=`
and `=B=` rows are refused.

Forwarded options: `reslim` (time limit), `optcr` (relative gap), `optca`
(absolute gap), `nodlim` (node limit; `0` = no limit, not discopt's default
100,000-node cap) and `threads`. GAMS's `optca = 0` is read at floating-point
resolution: the gap counts as closed when `|PB − DB| ≤ 1e-9·max(1, |PB|, |DB|)`
or the `optcr` test holds. A literal zero could never certify an optimum at
exactly 0, because a rigorous dual bound carries a small safety margin and the
relative gap is undefined there. `iterlim`, `domlim`, `cutoff`, `cheat` and an
option file (`optfile`) are not honoured; when set, the link says so in the log
and listing instead of ignoring them silently.

The solution written back includes every variable level, the equation levels
computed from them, the dual bound as `objest` (whenever it is valid, even with
an open gap) and the node count. discopt returns no duals here, so marginals are
reported as unavailable rather than as zero.

## Status mapping

discopt results are mapped to GAMS `(modelStat, solveStat)` pairs. The model
status follows GAMS's definitions, which distinguish problems with discrete
variables from those without:

| discopt result | continuous model | discrete model |
|---|---|---|
| `optimal`, certified | `1 Optimal` | `1 Optimal` |
| `optimal`, not certified / `local_optimal` | `2 Locally Optimal` | `8 Integer Solution` |
| `feasible` | `7 Feasible Solution` | `8 Integer Solution` |
| `infeasible` (proven) | `4 Infeasible` | `10 Integer Infeasible` |
| `local_infeasible` | `5 Locally Infeasible` | `5 Locally Infeasible` |
| `local_limit` | `6 Intermediate Infeasible` | `6 Intermediate Infeasible` |
| `unbounded` | `3 Unbounded` | `3 Unbounded` |

The solve status says why the search stopped. A time or node limit is
`3 Resource Interrupt`, an iteration limit `2 Iteration Interrupt`, a user stop
`8 User Interrupt`, and anything else `1 Normal Completion`. For a `feasible`
result this is read from `SolveResult.termination`, since `feasible` alone does
not say whether a budget stopped the search. A limit status with no incumbent is
`14 No Solution Returned`.

**Global vs local.** `1 Optimal` from discopt means a *certified global*
optimum: it is reported only when discopt's gap is mathematically certified
(`gap_certified`). When discopt falls back to heuristic NLP branch-and-bound on
a nonconvex model the result is locally optimal only, and is reported as
`2 Locally Optimal` (`8 Integer Solution` for a discrete model) rather than
overstated as global. Earlier versions reported a certified discrete optimum as
`8`, the code GAMS defines for an integer solution that is not proven optimal.

## Smoke testing

A small corpus of GAMS models with known optima ships **inside the package**
(`discopt/gams/data/`, located via `discopt.gams.data_dir()`; LP, MIP, convex
NLP, nonconvex NLP, and a convex MINLP), described by `manifest.json`. There are
two layers of checking:

- **Reader path (no GAMS needed)** — the unit tests parse each model with
  `from_gams()` and solve it with discopt, asserting the known optimum
  (`pytest python/tests/test_gams_link.py -k smoke`).
- **Solver-link path (needs a GAMS install)** — `discopt gams-verify` runs every
  model through GAMS with the solver forced to discopt, reads back the objective
  and GAMS model/solve status, and checks them against the optimum. Because the
  corpus ships in the wheel, this works from a plain `pip install` — no repo
  clone:

  ```bash
  pip install "discopt[gams]" && discopt gams-register   # install + register
  discopt gams-verify                                    # or: make gams-verify
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
