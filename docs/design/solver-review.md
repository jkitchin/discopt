# discopt Solver Review: POUNCE/discopt coverage and external-solver pluggability

**Date:** 2026-06-19
**Scope:** A cross-cutting review of (1) which engine — POUNCE or
discopt's own code — handles each problem class, and (2) what it would
take to keep plugging in external solvers (HiGHS, SCIP, Couenne, IPOPT,
and, if feasible, Gurobi and GAMS).

---

## 1. Executive summary

discopt is, by design (see `docs/design/pounce-only-roadmap.md`), a
**single-stack solver**: every production solve ultimately runs on one of
two in-house engines — **POUNCE** (the pure-Rust Ipopt port) for
continuous LP/QP/NLP relaxations, and discopt's **own Rust/JAX
branch-and-bound** for the integer/global structure. HiGHS and cyipopt
are already cleanly *swappable* for the continuous and MIP relaxation
work through two backend "seams." That is the good news for
pluggability: the abstraction exists, it is just **narrow** (LP/QP/MILP
matrix solves and NLP node solves) and **closed to a fixed enum** of
engines.

The requested external solvers fall into three very different buckets:

- **Already in-process pluggable:** HiGHS (LP/MILP; QP/MIQP routing removed, #359), IPOPT
  (NLP, via cyipopt). These return solutions directly into discopt data
  structures and participate in `Model.solve()`.
- **Reachable only for offline benchmarking:** SCIP, Couenne, BARON,
  Bonmin — invoked as **subprocesses over an exported `.nl` file** inside
  `discopt_benchmarks/`, with only the objective/status parsed back from
  stdout. They are **not** usable as a backend for `Model.solve()`.
- **Not integrated at all (as a solving backend):** Gurobi, and "GAMS as
  a solver for discopt models." (GAMS integration today runs the *other*
  direction — discopt registers itself *as* a GAMS solver.)

The single biggest gap relative to the stated goal is that there is **no
first-class, in-process "solve this `Model` with external solver X and get
a `SolveResult` back"** path. `Model.solve(solver=...)` currently accepts
only `None`, `"amp"`, `"gp"`, `"bb"`
(`python/discopt/solver.py:2034`). Section 6 proposes a small, uniform
adapter layer that closes this for all of HiGHS/SCIP/Couenne/IPOPT/Gurobi/
GAMS at once, built on machinery that already exists (the exporters).

---

## 2. Engine inventory

| Engine | Language | Role | Availability |
|---|---|---|---|
| **POUNCE** | Rust (Ipopt port) | All production LP/QP/NLP relaxation solves; node solves in the self-hosted B&B | **Core dep** (`pounce-solver>=0.3`, `pyproject.toml:31`) |
| **discopt B&B** | Rust tree + JAX McCormick/relaxations | Integer & spatial branch-and-bound, presolve/FBBT, OBBT, cuts (cover/clique/GMI/MIR), crossover | Core |
| **HiGHS** | C++ (via `highspy`) | LP/MILP matrix solves; CI correctness oracle (QP/MIQP routing removed, #359) | Optional extra `highs` (`pyproject.toml:45`) |
| **cyipopt / IPOPT** | C++ | NLP node/continuous solves | Optional extra `ipopt` (`pyproject.toml:44`) |
| **Rust warm-started simplex** | Rust | Opt-in MILP B&B node LPs (`nlp_solver="simplex"`) | Built with the Rust extension |
| **JAX** | Python/XLA | Autodiff substrate, McCormick relaxations, differentiable layers — **not** a standalone numerical solver anymore (the JAX IPM was deleted) | Core |

---

## 3. Problem-class → engine matrix

Classification: `python/discopt/_jax/problem_classifier.py` (`ProblemClass`
∈ LP, QP, MILP, MIQP, NLP, MINLP). Dispatch:
`python/discopt/solver.py:2655-2844`.

| Problem class | Default engine | POUNCE-only mode (`nlp_solver="pounce"`) | Handler |
|---|---|---|---|
| **LP** | HiGHS → POUNCE → JAX-IPM fallback | POUNCE first | `_solve_lp` (`solver.py:6398`), seam `lp_backend.get_lp_solver` |
| **QP (convex)** | POUNCE → JAX-IPM fallback (HiGHS-free, #359) | same | `_solve_qp`, `qp_pounce` |
| **QP (nonconvex)** | spatial B&B + McCormick-LP | same | falls to `_solve_nlp_bb` |
| **MILP** | HiGHS whole-problem, else self-hosted B&B | self-hosted Rust B&B, HiGHS bypassed | `_solve_milp_highs` / `_solve_milp_bb` / `_solve_milp_simplex` |
| **MIQP** | self-hosted B&B (HiGHS-free, #359), else spatial B&B | same | `_solve_miqp_bb` |
| **NLP (convex)** | POUNCE (cyipopt if requested) | POUNCE | `_solve_continuous` (`solver.py:4863`) |
| **NLP (nonconvex)** | POUNCE multistart / spatial B&B | POUNCE | `_solve_continuous` / `_solve_nlp_bb` |
| **MINLP (convex)** | NLP-BB: POUNCE NLP nodes + MILP relaxation | POUNCE | `_solve_nlp_bb` (`solver.py:4995`) |
| **MINLP (nonconvex)** | spatial B&B: POUNCE NLP nodes + McCormick-LP | POUNCE | `_solve_nlp_bb` |

**Specialized solve modes** (all still run on the engines above):

| Mode | Selector | What it is | Engine underneath |
|---|---|---|---|
| **AMP** | `solver="amp"` (`solver.py:2036`) | Adaptive multivariate-partitioning global MINLP (discopt's own) | MILP relaxation (HiGHS/POUNCE) + POUNCE NLP sub-solves |
| **GP** | `solver="gp"` or auto-detect | Geometric programming via log-space convex reformulation | single POUNCE NLP |
| **OA / GDP-LOA** | `gdp_method="oa"`/`"loa"` | Outer approximation for convex MINLP / disjunctive | MILP master (HiGHS/POUNCE) + POUNCE NLP |
| **DAE, RO, MO, NN** | modules `dae/ ro/ mo/ nn/` | Reformulate the `Model`, then call `Model.solve()` | whatever the resulting class dispatches to |
| **MPEC** | `discopt.mpec.solve_mpec` | Scholtes regularization | POUNCE NLP |

**Takeaway:** every problem class is engine-agnostic *only at the
continuous-relaxation and MIP-relaxation level*, and *only across the
fixed set {POUNCE, HiGHS, cyipopt, Rust-simplex}*. The tree search,
presolve, McCormick compiler, OBBT, and cut generation are discopt's own
and not pluggable (nor should they need to be).

---

## 4. The three ways an external solver can touch discopt today

### 4a. In-process backend seams (the real plug points)

- **`python/discopt/solvers/nlp_backend.py`** — `get_nlp_solver(backend)`
  with `backend ∈ {"auto","pounce","cyipopt"}`. Any backend exposing
  `solve_nlp(evaluator, x0, constraint_bounds, options) -> NLPResult` drops
  in.
- **`python/discopt/solvers/lp_backend.py`** —
  `get_lp_solver` / `get_qp_solver` / `get_milp_solver` over
  `{HiGHS, POUNCE, Rust-simplex}`, signature- and result-compatible
  (`LPResult`/`QPResult`/`MILPResult`). Also `get_exact_lp_solver` /
  `get_exact_dual_lp_solver` for OBBT/DBBT (which need a *vertex* oracle,
  so they deliberately exclude the POUNCE IPM).

These two seams are exactly the right shape to host more engines — but
both are **hard-coded enums with bespoke `_try_*` factories**, not an open
registry, and the NLP seam knows nothing about subprocess solvers.

### 4b. File export (write-only, no read-back)

`python/discopt/export/` — `to_nl`, `to_gams`, `to_lp`, `to_mps`
(`export/__init__.py`). `.nl` is the most complete (full MINLP, arrays,
nonlinear DAG). **Gap:** there is **no solution reader** — no `.sol`
parser, no `from_mps`/`from_lp`. So today you can *hand a problem to* an
external solver but cannot *read its answer back* into a `SolveResult`
without writing the glue yourself. Readers that do exist: `from_nl`,
`from_gams`, `from_pyomo` (`modeling/core.py:2929-3055`) — all
problem-readers, not solution-readers.

### 4c. Benchmark subprocess runner (offline comparison only)

`discopt_benchmarks/benchmarks/runner.py` runs **BARON, Couenne, SCIP,
HiGHS, Bonmin** as subprocesses over an exported `.nl`
(`benchmarks.toml:326-359`; `_build_command`, `_run_external`,
`_parse_external_output`). It parses **objective/bound/status/nodes from
stdout** — not the full solution vector, and it lives outside the
installable package. Gurobi and a standalone IPOPT CLI are **not** wired
in here. BARON additionally has a GAMS-driven path
(`scripts/global_opt_baron_vs_discopt.py`).

### 4d. GAMS — the reverse direction

`python/discopt/gams/` registers **discopt as a GAMS solver**
(`option minlp = discopt;`), translating GMO ↔ discopt `Model` and solving
on the discopt engines (`gams/link.py`, `gmo_translate.py`,
`instructions.py`, plus a warm `daemon.py`). It is a complete, tested
integration — but it does **not** let discopt call *out* to GAMS solvers.

---

## 5. Per-solver status against the request

| Solver | As a backend for `Model.solve()` | As a benchmark comparator | What's missing for "plug in & get a result" |
|---|---|---|---|
| **HiGHS** | ✅ in-process (LP/MILP; QP/MIQP removed, #359) | ✅ `.nl` subprocess | Nothing for those classes |
| **IPOPT** | ✅ in-process NLP (cyipopt) | ❌ no CLI comparator | Optional: `.nl`/CLI comparator for parity tests |
| **SCIP** | ❌ | ✅ `.nl` subprocess | In-process adapter + solution read-back (or `pyscipopt`) |
| **Couenne** | ❌ | ✅ `.nl` subprocess | Same as SCIP |
| **BARON** | ❌ | ✅ `.nl` + GAMS script | Same; license-gated |
| **Bonmin** | ❌ | ✅ `.nl` subprocess | Same as SCIP |
| **Gurobi** | ❌ | ❌ | Full adapter (`gurobipy` or `.mps`/`.lp` + reader). No `.nl` reader in Gurobi |
| **GAMS (as a backend)** | ❌ (reverse only) | partial (BARON via GAMS) | A `to_gams` + `gams.exe` subprocess + listing/solution reader |

---

## 6. Recommendation: one uniform external-solver adapter layer

The cleanest way to honor "I'd still like to plug in highs/scip/couenne/
ipopt/gurobi/gams" without disturbing the POUNCE-first core is a thin,
**registry-backed adapter** that turns `Model.solve(solver="scip")` (and
friends) into: export → subprocess (or Python API) → read solution back →
`SolveResult`. Most of the parts already exist.

### 6.1 A solution reader (the keystone, currently missing)

Add `python/discopt/export/` readers symmetric to the writers:

- **`read_sol(path, model) -> dict[var, value]`** — parse the AMPL `.sol`
  format (what Couenne/SCIP/Bonmin/BARON/IPOPT all emit beside the `.nl`).
  This single reader unlocks every ASL solver. The benchmark runner today
  throws the `.sol` away; this promotes it to a real result.
- Optional later: `from_mps`/`from_lp` (or a `.sol`-equivalent for the
  LP/MPS path) for Gurobi/CPLEX-style flows.

### 6.2 An `ExternalSolver` protocol + registry

A small module, e.g. `python/discopt/solvers/external/`:

```python
class ExternalSolver(Protocol):
    name: str
    def is_available(self) -> bool: ...
    def solve(self, model: Model, *, time_limit, gap, options) -> SolveResult: ...

register_external_solver("scip", ScipNLSolver())   # .nl + subprocess + read_sol
register_external_solver("couenne", CouenneNLSolver())
register_external_solver("bonmin", BonminNLSolver())
register_external_solver("baron", BaronNLSolver())
register_external_solver("ipopt-cli", IpoptNLSolver())
register_external_solver("gurobi", GurobiSolver())  # gurobipy or .mps/.lp
register_external_solver("gams", GamsSolver())      # to_gams + gams subprocess
```

Two adapter families cover almost everything:

1. **`AslNlSolver`** — generic: `to_nl` → run `<cmd> problem.nl` →
   `read_sol` → map status. SCIP/Couenne/Bonmin/BARON/IPOPT are *just
   different command strings* (the benchmark runner already proves the
   command shapes work — reuse `_build_command`/parsers from
   `discopt_benchmarks/benchmarks/runner.py`, lifted into the package).
2. **API/file solvers** — `gurobipy` if importable else `.mps`/`.lp`;
   `gams` via `to_gams` + subprocess + listing reader.

### 6.3 Wiring into dispatch

Extend the `solver=` validation at `solver.py:2034` from the fixed
`{None,"amp","gp","bb"}` to also accept any registered external name, and
route to the registry before classification. Keep POUNCE/HiGHS/cyipopt on
the existing in-process seams (they are faster and lose-less than a file
round-trip); the registry is for the *additional* solvers.

### 6.4 Packaging

Mirror the existing extras pattern (`pyproject.toml:40-46`): add
`scip = ["pyscipopt"]`, `gurobi = ["gurobipy"]` as optional; keep the
subprocess/`.nl` adapters dependency-free (they only need a binary on
`PATH` and a config path, exactly like the benchmark `solvers` table).
Surface availability through `available_backends()`-style helpers so
`Model.solve` can give a clear "install/`PATH` X" error.

### 6.5 What to reuse vs. build

| Need | Reuse | Build |
|---|---|---|
| Write the problem | `export.to_nl / to_mps / to_lp / to_gams` | — |
| Run the solver | benchmark `_build_command` / `_run_external` (lift into pkg) | `ExternalSolver` registry |
| Read the answer | — | **`read_sol` (AMPL .sol)** + status mapping |
| Dispatch | `solver=` selector at `solver.py:2034` | registry hook |
| Packaging | extras pattern in `pyproject.toml` | `scip`/`gurobi` extras |

---

## 7. Prioritized recommendations

1. **Build the AMPL `.sol` reader** (`export/sol.py`) — highest leverage;
   one reader makes SCIP, Couenne, Bonmin, BARON, and the IPOPT CLI all
   usable as backends.
2. **Add the `ExternalSolver` registry + `AslNlSolver`** and wire the
   `solver=` selector. This delivers SCIP/Couenne/Bonmin/BARON/IPOPT-CLI
   in one stroke, reusing the exporters and the benchmark command shapes.
3. **Promote the benchmark runner's command/parse logic into the package**
   (or share a module) so there is one source of truth for solver
   invocation, used by both `Model.solve(solver=...)` and the benchmarks.
4. **Gurobi** — add a `gurobipy` adapter (fast path) with an `.mps`/`.lp`
   fallback; Gurobi cannot read `.nl`, so it needs the matrix-export path
   and is LP/QP/MILP/MIQP only (no general MINLP).
5. **GAMS-as-backend** (optional) — `to_gams` + `gams` subprocess +
   listing/solution reader, distinct from the existing
   discopt-as-GAMS-solver link.
6. **Keep the core untouched.** POUNCE stays the default and the only hard
   dependency; HiGHS/cyipopt stay the preferred in-process alternates. The
   registry is purely additive — no change to the `incorrect_count ≤ 0`
   invariant or the POUNCE-only install story.

### 7.1 Gurobi integration update

As of issue #102, Gurobi has a narrower in-process role than the external-solver
registry proposed above. `Model.solve(solver="gurobi")` is a direct matrix
backend for LP/MILP/QP/QCP-family models only. General nonlinear expression DAGs
are not compiled to Gurobi nonlinear expressions. For global NLP/MINLP workflows,
discopt remains responsible for AMP/OA/LOA relaxations and certification, while
`milp_solver="gurobi"` selects Gurobi as the matrix MILP master subsolver. The
default spatial branch-and-bound path remains unchanged.

### Effort sketch

- `read_sol` + tests: small (~1 file).
- Registry + `AslNlSolver` + dispatch wiring + tests: medium.
- Gurobi adapter: small–medium (needs matrix export validation).
- GAMS-as-backend: medium (listing parser).

None of this touches the numerical core; it is all export/subprocess/parse
glue plus one dispatch hook.
