# discopt

[![PyPI](https://img.shields.io/pypi/v/discopt)](https://pypi.org/project/discopt/)
[![CI](https://github.com/jkitchin/discopt/actions/workflows/ci.yml/badge.svg)](https://github.com/jkitchin/discopt/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jkitchin/discopt/graph/badge.svg?token=B3Y6LAtox9)](https://codecov.io/gh/jkitchin/discopt)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19762815-blue)](https://doi.org/10.5281/zenodo.19762815)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/discopt?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/discopt)

[![discopt](https://github.com/jkitchin/discopt/blob/main/discopt.png?raw=true)](https://github.com/jkitchin/discopt/blob/main/discopt.png?raw=true)



A Mixed-Integer Nonlinear Programming (MINLP) solver built on a Rust core with
Python orchestration. Solves MINLPs by spatial Branch and Bound over rigorous
convex relaxations, with an in-house primal/dual simplex for the per-node LPs and
a Rust automatic-differentiation tape (via POUNCE) for objective, gradient,
Jacobian, and Hessian evaluation.

## Features

- **Algebraic modeling API** -- continuous, binary, and integer variables with operator overloading
- **Spatial Branch and Bound** -- Rust-powered node pool, branching, and pruning; the native Rust spatial B&B kernel is the default engine (`DISCOPT_NATIVE_SPATIAL_KERNEL=0` opts back to the Python tree)
- **Rust AD tape for NLP evaluation** -- objective, gradient, constraint Jacobian, and Lagrangian Hessian (dense and sparse) come from a POUNCE-backed tape with no JAX on the path; `DISCOPT_NLP_EVAL=jax` restores the legacy JAX evaluator
- **In-house LP/MILP engine** -- pure-Rust primal/dual simplex with warm starts and a sparse LU basis (`feral`); it drives every MINLP node LP. Models classified as *pure* LP/MILP at entry are routed to HiGHS instead, with discopt-verified certificates (`DISCOPT_LP_MILP_BACKEND=rust` opts back to the Rust simplex)
- **NLP backends** -- POUNCE (pure-Rust Ipopt port, the universal default) and cyipopt (Ipopt); `nlp_solver="simplex"` selects the pure-Rust warm-started-simplex MILP B&B. The pure-JAX IPM has been retired -- `"ipm"`/`"sparse_ipm"` remain as back-compat aliases
- **Convex relaxations** -- McCormick envelopes over 28 primitive operations (bilinear, powers, `exp`/`log` family, trig and inverse-trig, hyperbolics, `sigmoid`/`softplus`/`tanh`, `abs`/`min`/`max`/`sign`/`entropy`) plus a 22-intrinsic univariate envelope table in the uniform factorable engine (adding `erf`, `log1p`, and the inverse hyperbolics); piecewise McCormick, alphaBB underestimators, and G-convexity / convex-transformable relaxations
- **Certified global MINLP** -- Adaptive Multivariate Partitioning (`solver="amp"`) for nonconvex bilinear/trilinear/signomial/trig models, and a signomial global optimizer (`DISCOPT_SGO`) for mixed-sign signomial and integer-signomial problems
- **Decomposition solvers** -- MIP-NLP family (`solver="mip-nlp"`: OA, ECP, FP, GOA, LP/NLP-BB), Benders and Generalized Benders (GBD), Lagrangian decomposition, and an automatic structure/decomposition advisor
- **Derivative-free optimization** -- `solver="direct"` (sampling search over black-box `dm.custom` bodies) and `solver="surrogate"` (surrogate-model search); both are explicitly non-certifying, and a governed variant runs as a root heuristic
- **External functions / grey box** -- `dm.external(fn, jac=..., hess=...)` embeds a compiled simulator, subprocess or legacy kernel whose values *and* derivatives you supply, the analogue of Pyomo's `ExternalGreyBoxModel`; nested `jax.custom_jvp` rules make the block twice differentiable so it solves on the ordinary NLP path. An external block is opaque to the relaxation layer, so it is `status="feasible"` with no dual bound and is refused outright with integer/binary variables, by contract
- **Neural network & tree embedding** -- embed trained feedforward networks (ReLU, sigmoid, tanh, softplus) as MINLP constraints via big-M, full-space, and reduced-space formulations; decision trees and gradient-boosted ensembles via per-leaf MILP encoding; interval-arithmetic bound propagation; ONNX / scikit-learn / PyTorch readers. Trainable surrogates (`ml.trainable`, `ml.surrogate`) emit symbolic weights so a surrogate can be fit *simultaneously* with a physics model
- **Generalized disjunctive programming** -- `BooleanVar`, propositional logic operators (`land`, `lor`, `lnot`, `atleast`, `atmost`, `exactly`), `either_or()`, `if_then()`; reformulated via big-M, multiple big-M (LP-tightened), hull, or Logic-based Outer Approximation (`gdp_method="loa"`), with a disjunct-selection primal constructor on by default
- **Piecewise-linear functions** -- `m.piecewise(x, breakpoints, values)` (or `dm.piecewise`) declares a tabulated univariate function (a pump curve, a tariff, a property table) and returns a variable equal to it; lowered to an *exact* MILP encoding -- `"incremental"`, `"log"` (Gray-code, logarithmically many binaries), `"disaggregated"`, or `"sos2"` -- so solves stay certified. An input whose domain escapes the breakpoint span is refused rather than silently clamped
- **Nonlinear-to-PWL transformation** -- `dm.nonlinear_to_pwl(model)` replaces every univariate nonlinear term (`exp(x)`, `x**4 - 3*x**2`, a valve-point ripple) by a piecewise-linear MILP construct. `mode="outer"` (default) is a *rigorous outer approximation* -- per-segment error bands from interval arithmetic and interval-AD mean-value tapers -- so its bound is valid for the original model, and it certifies only when a verified incumbent meets that bound, refining the partition until it does; `mode="approximate"` uses the chord interpolant and never claims a bound
- **Complementarity / MPEC** -- `Model.complementarity(x, y)` (elementwise over vectors/arrays) reformulated via GDP disjunction (default), Scholtes regularization, or SOS1
- **Bilevel programming** -- KKT and strong-duality reformulations of the follower problem, including certified/convex-NLP followers
- **Stochastic programming** -- extensive form, L-shaped, progressive hedging, multistage, SAA, risk measures, and distributionally-robust variants
- **Geometric programming** -- posynomial detection with an exact log-space convex reformulation (auto-routed), plus GP-structured MINLPs solved by integer B&B over exact convex log-space node relaxations (`solver="gp-minlp"`)
- **Robust & multi-objective optimization** -- uncertainty sets with affine decision rules; scalarization (weighted-sum, ε-constraint, Tchebycheff, NBI, NNC) with Pareto-front analysis
- **Parameter estimation** -- weighted-least-squares estimation with exact Fisher-information Jacobians; model-based design of experiments (D/A/E-optimality, identifiability, model discrimination) is available via the [discopt-doe](https://github.com/jkitchin/discopt-doe) plugin
- **Presolve** -- FBBT (interval arithmetic, probing, Big-M simplification, integrality-aware snapping, periodic-variable reduction), reverse-FBBT auxiliary cascade, substitution-graph aggregation with postsolve, OBBT with LP warm-start
- **Cutting planes** -- reformulation-linearization (RLT, a first-class `rlt=True` option), PSD/SOC cuts for QCQP, GMI cuts, and outer approximation (OA); the structure-gated `rlt="auto"` policy is the default
- **Primal heuristics** -- multi-start NLP, feasibility pump, diving, RINS, local branching, QUBO/Ising local search, one-hot swap local search for graph-partition MIQPs
- **Infeasibility diagnosis** -- irreducible infeasible subsystem (`compute_iis`) and conflict analysis / no-good cuts
- **Differentiable optimization** -- parameter sensitivity via envelope theorem and KKT implicit differentiation, including differentiable MILP/MIQP (fix-and-differentiate)
- **Model import & export** -- read AMPL `.nl` (Rust parser), GAMS `.gms`, and QPLIB native format; write `.nl`, `.lp`, `.mps`, and GAMS
- **Named composites** -- `dm.register_function(name, lower)` names a composite the relaxer envelopes as ONE atom instead of term by term, so cancellations between terms survive into the bound; the model still carries the primitive lowering, so evaluation, `.nl` export and presolve are untouched
- **Batch solving** -- `dm.solve_batch(models, workers=N)` runs many small independent global solves, optionally in parallel
- **Embedded inner problems** -- `dm.argmin` places an inner NLP as a block of an outer model, with `dm.argmin_kkt` as its lowered (stationarity-constraint) arm
- **Vector reductions** -- `xs.max()` / `xs.min()` reduce a shaped operand, alongside the element-wise `dm.maximum` / `dm.minimum`
- **Model persistence with provenance** -- `Model.save("m.dopt")` / `discopt.load(...)` round-trip a model with a recorded schema id and FAIR provenance
- **Pyomo solver plugin** -- use discopt from existing Pyomo models via `SolverFactory("discopt")` (`pip install discopt[pyomo]`); see [docs/notebooks/pyomo_solver.ipynb](docs/notebooks/pyomo_solver.ipynb)
- **GAMS solver link** -- run discopt *as* a GAMS solver through the GMO/GEV API (`discopt gams-register`, `discopt gams-daemon`); see [docs/gams_solver_link.md](docs/gams_solver_link.md)
- **Warm solve daemon** -- `discopt solve model.nl` routes through a persistent daemon that keeps the process warm across solves
- **Dynamic optimization** -- DAE collocation (Radau/Legendre), finite differences, and method-of-lines for optimal control, parameter estimation, and PDE-constrained optimization, with multi-experiment trajectory fitting
- **Benchmark interfaces** -- CUTEst (NLP test set), MINLPLib `.nl`, and QPLIB (453 quadratic instances, 390 nonconvex, with reference solution vectors)
- **LLM integration** (optional) -- conversational model building, diagnostics, and reformulation suggestions
- **Extensive test suite** -- 777 Rust + 9,100+ Python test functions

## Quick Start

```python
from discopt import Model

m = Model("example")
x = m.continuous("x", lb=0, ub=5)
y = m.continuous("y", lb=0, ub=5)
z = m.binary("z")

m.minimize(x**2 + y**2 + z)
m.subject_to(x + y >= 1)
m.subject_to(x**2 + y <= 3)

result = m.solve()
print(result.status)     # "optimal"
print(result.objective)  # 0.5
print(result.x)          # {"x": array(0.5), "y": array(0.5), "z": array(0.)}
```

## Architecture

```
Model.solve()  -->  Python orchestrator  -->  Rust B&B kernel / TreeManager
                        |                          |
                  NLP evaluation:            Node pool / branching / pruning
                    POUNCE AD tape           In-house primal/dual simplex (node LPs)
                    (default, JAX-free)      Zero-copy numpy arrays (PyO3)
                  NLP backends:
                    pounce  (pure-Rust Ipopt port)  [default]
                    cyipopt (Ipopt)                 [fallback]
```

**Rust backend** (`crates/discopt-core`): Expression IR, Branch and Bound tree (node
pool, branching, pruning), the native spatial B&B kernel, in-house primal/dual
simplex with a sparse LU basis (`feral`), .nl file parser, FBBT/presolve (interval
arithmetic, probing, Big-M simplification).

**Rust-Python bindings** (`crates/discopt-python`): PyO3 bindings with zero-copy numpy
array transfer for the B&B tree manager, expression IR, batch dispatch, and .nl parser.

**NLP evaluation** (`python/discopt/_tape_nlp_evaluator.py`, `_nl_expr_compiler.py`):
objective, gradient, constraints, Jacobian, and Lagrangian Hessian (dense and sparse)
from a POUNCE Rust AD tape. This is the default; expressions with no tape opcode (an
opaque `dm.custom` body, a matrix norm) fall back to the JAX evaluator, and
`DISCOPT_NLP_EVAL=jax` selects it wholesale. A tape-representable solve does not import
JAX -- not on the LP, QP, MIQP and simplex-MILP paths, and not on the nonlinear ones
either. **That fallback is the exception, and it is on the default path**: a plain
`Model.solve()` of `dm.norm(X, 2)` with a 3x3 `X`, with no `import jax` on the caller's
side, takes `sys.modules` from 0 to 219 `jax` entries. "A default solve does not import
JAX" is true of the tape-representable majority, not of every model.

**Relaxation layer** (`python/discopt/_relax`): DAG compiler, the uniform factorable
relaxation engine, McCormick convex/concave envelopes, alphaBB, piecewise McCormick,
cutting planes, convexity detection, and the relaxation compiler. This layer is
**numpy**: measured over eight nonlinear corpus instances, a default solve loads
~50 `_relax` modules -- envelope evaluation (`uniform_relax`, `mccormick_lp`,
`incremental_mccormick`) and cut separation (`cutting_planes`,
`multilinear_separation`, `psd_cuts`) among them -- and zero `jax` modules. JAX is
imported only by the optional differentiable-solve and learned-relaxation
subsystems, which are off the default path.

**Solver wrappers** (`python/discopt/solvers`): POUNCE (pure-Rust Ipopt port) for
LP/QP/NLP, the in-house simplex LP/MILP backends, cyipopt for Ipopt, AMP, the MIP-NLP
decomposition family, GDPopt-LOA, the DFO backends (`direct`, `surrogate`), and an
optional Gurobi backend. highspy is a core dependency: it backs the pure LP/MILP
entry route (`lp_milp_highs.py`) and the OA/GDP paths.

**Interfaces** (`python/discopt/interfaces`): PyCUTEst-based evaluator for NLP
benchmarking against the CUTEst test set, and a native QPLIB reader.

**Orchestrator** (`python/discopt/solver.py`): End-to-end `Model.solve()` connecting all
components. At each B&B node: solve the relaxation with tightened bounds, prune
infeasible nodes, fathom integer-feasible solutions, branch on the selected variable.

## NLP Backends

| Backend                        | Implementation                        | Use Case                                    |
|--------------------------------|---------------------------------------|---------------------------------------------|
| `pounce` (default)             | Pure-Rust Ipopt port                  | Universal default: LP/QP/MILP/MIQP/NLP/MINLP |
| `ipopt` / `cyipopt`            | Ipopt via cyipopt                     | NLP node and continuous solves; most robust |
| `simplex`                      | Pure-Rust warm-started simplex B&B    | MILP via the in-house Rust B&B              |
| `ipm` / `sparse_ipm`           | Back-compat aliases                   | Simplex-first LP/MILP routing; resolve to POUNCE for NLP/MINLP |

The pure-JAX interior-point method has been retired. `nlp_solver="ipm"` is kept as an
alias so existing scripts keep working: it selects the simplex-first matrix routing for
LP/MILP and resolves to POUNCE for NLP/MINLP.

```python
result = model.solve()                       # default: POUNCE
result = model.solve(nlp_solver="pounce")    # POUNCE (pure-Rust Ipopt port)
result = model.solve(nlp_solver="ipopt")     # Ipopt via cyipopt
result = model.solve(nlp_solver="simplex")   # pure-Rust simplex MILP B&B
```

## Benchmarks

The numbers below are the committed outputs of
[`docs/notebooks/benchmarks_by_class.ipynb`](docs/notebooks/benchmarks_by_class.ipynb)
(Python 3.12, CPU, median of 3 runs including setup). Absolute times are
machine-dependent -- the notebook is the reproducible source, and these rows are
copied from it rather than re-timed here. All solvers agree on the objective
value. The NLP row's discopt arm is the notebook's `IPM` column, which is the
default backend the alias now resolves to (POUNCE).

| Problem Class | discopt | Comparison | Notes |
|---------------|---------|------------|-------|
| **LP** (n=100) | 0.2521s | HiGHS 0.0016s, scipy 0.0021s | Algebraic extraction, no autodiff |
| **QP** (n=100) | 0.4434s | scipy SLSQP 0.0238s | -- |
| **MILP** (n=25, 8 int) | 0.0208s | HiGHS MIP 0.0018s | B&B + LP relaxation, correct objectives |
| **MIQP** (n=10) | 0.019s | forced NLP path 0.800s | QP-specialized path: 41.2x speedup |
| **NLP** (n=20, Rosenbrock) | 0.1328s | cyipopt 0.1378s | Two implementations of the same IPM |
| **MINLP** (n=10) | 0.027s (batch=1) | 0.029s (batch=16) | These trees close in 1-5 nodes, so batching has nothing to fill |

HiGHS (C++ simplex) and scipy remain faster on the LP/MILP classes, as expected for
mature production codes; discopt's value on these classes is that they are reachable
from the same model object as the MINLP path.

See the benchmark notebooks for full scaling plots and details:
- [Benchmarks by Problem Class](docs/notebooks/benchmarks_by_class.ipynb) -- LP, QP, MILP, MIQP, NLP, MINLP
- [NLP Backend Comparison](docs/notebooks/ipm_vs_ipopt.ipynb) -- POUNCE vs Ipopt

## Installation

Requires Python 3.12+; building from source additionally needs Rust 1.84+.
POUNCE -- the default numerical engine -- is a pure-Rust Ipopt port installed as a
core dependency, with no system libraries needed. `highspy` (the pure LP/MILP entry
route) and `jax`/`jaxlib` are also core dependencies; JAX is installed but is *not*
imported by a default solve (see the NLP-evaluation note above). cyipopt is an
optional fallback that needs the Ipopt C library.

```bash
pip install discopt

# Optional cyipopt fallback (needs the Ipopt C library; macOS: brew install ipopt)
pip install "discopt[ipopt]"
```

From a source checkout:

```bash
# Build Rust-Python bindings
cd crates/discopt-python && maturin develop && cd ../..

# Run the fast default PR battery
cargo test -p discopt-core
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 make test
```

`make test` matches the PR CI gate: ordinary non-slow tests plus the
`pr_correctness` subset. Full correctness, integration, and benchmark markers
remain available through the explicit Make targets.

Optional extras: `ipopt`, `cutest`, `gams`, `llm`, `sdp`, `nn` (ONNX), `pyomo`,
`ml` (scikit-learn), `xgboost`, `lightgbm`, `gnn`, `learned`, `sympy`, `dev`, `all`.
`pounce` and `highs` also exist as no-op back-compat aliases -- both packages are
core dependencies now, so neither extra installs anything extra.

### Solving nonconvex MINLPs with AMP

For problems with nonconvex nonlinearities (bilinear, trilinear, signomial,
trig), the default branch-and-bound path only certifies optimality when the
relaxation is convex. The Adaptive Multivariate Partitioning (AMP) solver
gives discopt a **certified-global** path for these problems:

```python
import discopt.modeling as dm

m = dm.Model("bilinear")
x = m.continuous("x", lb=1.0, ub=5.0)
y = m.continuous("y", lb=1.0, ub=5.0)
m.minimize(x * y - 2 * x - 3 * y)  # nonconvex: the bilinear term is indefinite
m.subject_to(x + y <= 7.0)
m.subject_to(x - y >= -3.0)

result = m.solve(solver="amp", rel_gap=1e-4)
print(result.status, result.objective, result.gap)
# optimal -10.0 5.03e-05   (global minimum -10 at x=1, y=4)
```

`status="optimal"` is the certificate: AMP closed the gap below `rel_gap`, so
`-10.0` is proven global, not merely the best point found. When AMP cannot close
the gap within `max_iter` it returns `status="feasible"` and a `result.gap` you
can read -- an honest refusal to certify, never a false `optimal`.

AMP iterates a piecewise-McCormick / convex-hull MILP relaxation against an
NLP subproblem and refines the partition where the relaxation gap is
largest. At every iteration `LB_k <= global_opt <= UB_k`, so termination at
`gap <= rel_gap` yields a certified global optimum.

Common tuning knobs (all keyword-only on `Model.solve(solver="amp", ...)`):

| Option | Default | Effect |
| --- | --- | --- |
| `rel_gap` | `1e-4` | Relative optimality gap stop criterion |
| `max_iter` | `100` | Hard cap on partition-refinement iterations |
| `n_init_partitions` | `4` | Initial partitions per discretized variable |
| `convhull_formulation` | `"disaggregated"` | `"sos2"` or `"facet"` for tighter relaxations |
| `convhull_ebd` | `False` | Logarithmic Gray-code embedded SOS2 binaries |
| `presolve_bt` | `True` | OBBT/FBBT bound tightening before the first MILP |
| `obbt_at_root` | `True` | Strengthen variable bounds at the root |
| `milp_solver` | `"auto"` | MILP master backend: `"auto"`, `"pounce"`, `"simplex"`, or `"gurobi"` |
| `partition_method` | `"adaptive"` | How to pick which variable/interval to refine |

Gurobi can be used as AMP's MILP-master subsolver without changing the global
algorithm:

```python
result = m.solve(solver="amp", milp_solver="gurobi", rel_gap=1e-4)
```

This does not translate general nonlinear expressions into Gurobi nonlinear
constraints; discopt still builds and certifies the global MINLP relaxation.

A worked end-to-end example with a non-trivially nonconvex model and the
tuning knobs above is in `docs/notebooks/amp_global_minlp.ipynb`.

### AMP Test Suites

Routine AMP development uses a fast default regression battery. The fast
environment uses solver-independent checks plus MILP relaxations on the in-house
backends, and excludes optional cyipopt, longer Alpine, MINLPTests, and
incidence-style AMP benchmark coverage. AMP and PR-fast Make targets run pytest
through `scripts/run_memory_capped_pytest.sh`, which applies a 32 GB
address-space cap with `prlimit` when available. Override with
`PYTEST_MEMORY_LIMIT_MB=...`, or set `PYTEST_MEMORY_LIMIT_MB=0` to disable the
cap. The broad `make test-quick` dev-loop target remains uncapped and excludes
`memory_heavy` tests.

```bash
make test-amp-fast
```

Alpine-reference, MINLPTests, cyipopt, and incidence-style AMP checks are
opt-in because they can require optional solvers and longer solve budgets:

```bash
# Uses a fresh .venv and pixi-provided solver libraries rather than a local Python env.
pixi exec -s python=3.12 -s ipopt -s pkg-config -s c-compiler -s cxx-compiler -s gfortran -- \
  uv venv --allow-existing .venv
source .venv/bin/activate
uv pip install maturin pytest pytest-timeout numpy scipy jax jaxlib cyipopt
uv pip install -e ".[dev,ipopt]"
maturin develop
make test-amp-integration
```

For WSL or memory-constrained machines, keep PR-fast AMP/JAX runs capped and
use a bounded xdist worker count rather than `-n auto`. For the single-process
AMP integration suite, disable the virtual-address cap to avoid XLA
`std::bad_alloc` aborts from address-space reservations:

```bash
PYTEST_MEMORY_LIMIT_MB=32768 PYTEST_XDIST_WORKERS=2 make test
PYTEST_MEMORY_LIMIT_MB=0 make test-amp-integration
```

WSL users should also set explicit memory and swap limits in `.wslconfig` so a
single uncapped compile-heavy test cannot restart the host session. A stricter
12 GB cap is useful for reproducing memory pressure, but the JAX/XLA CPU stack
used by the relaxation layer can reserve more than 12 GB of virtual address
space during AMP runs; use the `memory_heavy` marker selection when running with
tighter caps.

The full Python test suite remains available with `make test-all`.

## Plugins

discopt keeps its core lean and ships domain-specific application builders and
teaching tools as separate **plugin packages**. Each is a PEP 420 namespace
package: once installed, its modules import under `discopt.<name>` unchanged,
and any CLI verbs it registers (through the `"discopt.cli"` entry-point group)
become available as `discopt <subcommand>`. Some are on PyPI; the rest install
directly from the repository.

| Plugin | Install | Provides |
|---|---|---|
| **[discopt-doe](https://github.com/jkitchin/discopt-doe)** | `pip install discopt-doe` | Model-based **design of experiments** — D/A/E-optimality, identifiability, model discrimination — as a `discopt doe ...` CLI loop (templates/new/status/fit/extend/gui) around an `.xlsx` workbook, with an optional Streamlit GUI. |
| **[discopt-aggregation](https://github.com/jkitchin/discopt-aggregation)** | `pip install discopt-aggregation` | **Variable aggregation** (reduced-space presolve): substitutes variables defined by equality constraints to yield a smaller reduced-space formulation, then recovers them from the solution ([Naik et al., arXiv:2502.13869](https://arxiv.org/abs/2502.13869)). Exposes `aggregate`/`solve` under `discopt.aggregation`. |
| **[discopt-apps](https://github.com/jkitchin/discopt-apps)** | `pip install "git+https://github.com/jkitchin/discopt-apps.git"` | **Application builders** for the modeling language: AC optimal power flow (`discopt.opf`) and the pooling problem in pq-formulation (`discopt.pooling`). Both moved out of the core package. |
| **[discopt-course](https://github.com/jkitchin/discopt-course)** | `pip install "git+https://github.com/jkitchin/discopt-course.git"` | An **optimization course** plus an interactive `discopt tutor ...` CLI (`discopt.course`) that walks through modeling and solving exercises. |

```bash
# Example: add the design-of-experiments plugin
pip install discopt-doe
discopt doe --help          # the plugin's verbs are now under the `discopt` CLI
```

Dependent packages are tracked in
[`.github/dependents.yml`](.github/dependents.yml); each discopt release
automatically re-runs their CI and opens a review issue so breakage surfaces
early (see [docs/dev/dependents.md](docs/dev/dependents.md)).

**Writing a plugin?** You can have discopt automatically exercise your package
against every new core release. Ask to be added to
[`.github/dependents.yml`](.github/dependents.yml), and copy
[`.github/dependent-ci-template.yml`](.github/dependent-ci-template.yml) into
your repo as `.github/workflows/discopt-integration.yml` — it listens for the
`discopt-updated` dispatch and runs your tests against discopt `main` (with a
weekly fallback), so you find out immediately if a discopt release breaks you.
Details in [docs/dev/dependents.md](docs/dev/dependents.md).

## Command-Line Interface

After installation, the `discopt` command is available on your PATH:

```bash
discopt about            # Version and installation info
discopt test             # Smoke-test the install
discopt solve model.nl   # Solve a .nl model (warm-routed through the solve daemon)
discopt convert in.gms out.nl
discopt daemon status    # Control the warm solve daemon (serve/stop/kill/status)
discopt gams-register    # Register discopt as a GAMS solver
discopt gams-daemon      # Control the warm GAMS solver daemon
discopt gams-verify      # Run the packaged .gms corpus through GAMS with solver=discopt
discopt install-skills   # Install Claude Code slash commands and agents
```

`discopt solve` accepts the usual solve controls as flags (`--profile`,
`--time-limit`, `--gap`, `--solver`, `--rlt`, `--partitions`, `--tuning`,
`--json`, `--sol`).

External packages can add subcommands through the `"discopt.cli"` entry-point
group (see the protocol notes in `python/discopt/cli.py`). For example, the
[discopt-doe](https://github.com/jkitchin/discopt-doe) plugin
(`pip install discopt-doe`) adds
`discopt doe ...` — a model-based design-of-experiments loop
(templates/new/status/fit/extend/gui) around an `.xlsx` workbook, with an
optional Streamlit GUI. See [Plugins](#plugins) above for the full list.

A separate `discopt-dev` script ships developer-only commands used from inside
a discopt source checkout (literature scanner, adversary tester, the arXiv /
OpenAlex search helpers and the report writer they call):

```bash
# Search arXiv for recent papers
discopt-dev search-arxiv 'all:"spatial branch and bound"' --max-results 10 --start-date 2026-01-01

# Search OpenAlex
discopt-dev search-openalex "McCormick relaxation" --from-date 2026-01-01 --to-date 2026-03-31

# Write a report from stdin
echo "report content" | discopt-dev write-report reports/output.md
```

All `discopt-dev` search subcommands output structured JSON. `discopt-dev
lit-scan` drives them through a `/discoptbot` Claude Code slash command to find
and summarize relevant new papers; that command and `/adversary` are dev-only and
are deliberately never shipped by `discopt install-skills`, so `lit-scan` works
only where a `.claude/commands/discoptbot.md` is present in the source tree.

## Documentation

Tutorial notebooks are available in `docs/notebooks/`:

- **Quickstart**, **Modeling Guide**, **Sets and Indexing** -- basic modeling and solving
- **Problem-class tutorials** -- LP, QP, MILP, MIQP, MINLP, GDP, DAE, robust, multi-objective, complementarity/MPEC, bilevel, stochastic, pooling, geometric programming
- **Solver backends** -- OA, MIP-NLP, Benders, GBD, Lagrangian, the decomposition advisor, AMP global MINLP, DIRECT and surrogate DFO, POUNCE, cyipopt, and solver selection
- **Advanced Features** -- relaxations, presolve, bound tightening, cutting planes, convexity detection, symbolic envelopes, primal heuristics, IIS/conflict analysis, callbacks, warm starts, export formats
- **Global Optimization** -- which problems discopt can and can't certify as global
- **Applications** -- neural network embedding, neural DAEs, AC OPF, decision-focused learning, parameter estimation
- **Appendix** -- solver comparison, the GAMS solver link, references

Full documentation is built with Jupyter Book: `jupyter-book build docs/`

## Related projects

discopt is not the only tool in any of these spaces, and for several of them it
is the newer one. This section is here so you can tell quickly whether discopt
is the right fit or whether one of these is -- and, where discopt overlaps with
an established package, what the difference actually is.

**Design of experiments and parameter estimation.** discopt does
weighted-least-squares estimation with exact Fisher-information Jacobians in the
core, and model-based DoE (D/A/E-optimality, identifiability, estimability,
model discrimination) in the [discopt-doe](https://github.com/jkitchin/discopt-doe)
plugin.

| Project | What it does |
|---|---|
| [**Pyomo.DoE**](https://pyomo.readthedocs.io/en/stable/explanation/analysis/doe/doe.html) | Model-based DoE for Pyomo models: builds the FIM from model sensitivities and optimizes A/D/E-optimality, including for dynamic (Pyomo.DAE) models. The closest analogue to `discopt.doe`; if your model is already in Pyomo, start here. |
| [**pydex**](https://github.com/salvadorgarciamunoz/pydex) | Optimal experiment design over a candidate grid: D/A/E/V-optimal, CVaR and pseudo-Bayesian criteria, continuous or apportioned to exact designs. It takes a *simulator* -- any Python function, a scipy ODE integration, or a Pyomo.DAE model -- and gets sensitivities by finite differences or the implicit function theorem, so it applies where there is no closed-form algebraic response. The OED problem itself is formulated in Pyomo, so any Pyomo-accessible solver can solve it. (This is [salvadorgarciamunoz/pydex](https://github.com/salvadorgarciamunoz/pydex), a fork of [the original](https://github.com/KennedyPutraKusumo/pydex) by Kusumo et al.) |
| [**Pyomo parmest**](https://pyomo.readthedocs.io/en/stable/explanation/analysis/parmest/index.html) | Parameter estimation for Pyomo models with bootstrap and likelihood-ratio confidence regions. |

**Global and MINLP solvers.** These are what discopt is validated and benchmarked
against; see [Benchmarks](#benchmarks) above.

| Project | Notes |
|---|---|
| [**BARON**](https://minlp.com/baron-solver) | The reference commercial global MINLP solver, and the one discopt's standard head-to-head panel compares against (run through GAMS, since the AMPL binary ships demo-licensed). |
| [**SCIP**](https://github.com/scipopt/scip) | Open-source (Apache-2.0) constraint-integer programming with spatial B&B. It reads `.nl` directly, so it is the one external global solver that runs inside discopt's own benchmark harness. |
| [**Couenne**](https://github.com/coin-or/Couenne) | COIN-OR's spatial branch-and-bound global solver for nonconvex MINLPs, built on factorable reformulation and McCormick envelopes -- the same relaxation machinery discopt's `_relax/` layer implements. |
| [**HiGHS**](https://github.com/ERGO-Code/HiGHS) | High-performance LP/MIP/QP. discopt routes models it classifies as *pure* LP/MILP to HiGHS and verifies the certificate itself; its own Rust simplex drives the MINLP node LPs. |
| [**Alpine.jl**](https://github.com/lanl-ansi/Alpine.jl) | Julia/JuMP global solver built on adaptive multivariate partitioning -- the same AMP idea behind discopt's `solver="amp"`. |
| [**EAGO.jl**](https://github.com/PSORLab/EAGO.jl) | Julia global and robust optimization with McCormick relaxations and an extensible B&B. |

**NLP and automatic differentiation.**

| Project | Notes |
|---|---|
| [**POUNCE**](https://github.com/jkitchin/pounce) | Pure-Rust port of Ipopt, and discopt's default NLP solver and AD tape. Usable standalone. |
| [**Ipopt**](https://github.com/coin-or/Ipopt) / [**cyipopt**](https://github.com/mechmotum/cyipopt) | The interior-point NLP solver POUNCE ports, and its Python bindings -- available in discopt as `nlp_solver="cyipopt"`. |
| [**CasADi**](https://github.com/casadi/casadi) | Symbolic framework for numeric optimization with forward/reverse AD and C-code generation. Overlaps discopt's modeling and DAE layers; it is a framework for building solvers rather than a global MINLP solver itself. |

**Modeling layers.** discopt has its own algebraic modeling API, but it does not
require you to switch to it.

| Project | Notes |
|---|---|
| [**Pyomo**](https://github.com/Pyomo/pyomo) | discopt registers itself as a Pyomo solver -- `SolverFactory("discopt")` after `pip install discopt[pyomo]`. See [docs/notebooks/pyomo_solver.ipynb](docs/notebooks/pyomo_solver.ipynb). |
| [**JuMP**](https://github.com/jump-dev/JuMP.jl) | The Julia modeling layer; the front end for Alpine.jl and EAGO.jl above. |
| [**CVXPY**](https://github.com/cvxpy/cvxpy) | Disciplined convex programming. If your problem is DCP-compliant and has no integers, CVXPY is the more direct route. |

**Machine-learning surrogates in optimization.** discopt's `discopt.ml` is
explicitly inspired by OMLT, and adds a *trainable* regime in which surrogate
weights are decision variables fit simultaneously with a physics model.

| Project | Notes |
|---|---|
| [**OMLT**](https://github.com/cog-imperial/OMLT) | Embeds trained neural networks and gradient-boosted trees into Pyomo models (big-M, full-space, reduced-space, and ONNX import). |
| [**gurobi-machinelearning**](https://github.com/Gurobi/gurobi-machinelearning) | The same idea against Gurobi, with scikit-learn, Keras and PyTorch readers. |

**Benchmark libraries.**

| Project | Notes |
|---|---|
| [**MINLPLib**](https://www.minlplib.org/) | The MINLP instance library and its reference optima. discopt reads its `.nl` and `.gms` files directly and uses `minlplib.solu` as the correctness oracle. |
| [**QPLIB**](https://qplib.zib.de/) | 453 quadratic instances (390 nonconvex), read natively by `discopt.interfaces.qplib`. Unlike MINLPLib it ships reference solution *vectors*, so an incumbent can be feasibility-verified directly. |

## Project Statistics

*Last updated: 2026-09-19*

| Category | Count |
|----------|-------|
| **Python source** (`python/discopt/`) | 350 files, ~204,000 lines |
| **Rust source** (`crates/`) | 93 files, ~74,600 lines |
| **Test code** (`python/tests/`) | 753 files, ~210,800 lines |
| **Total source + tests** | ~1,196 files, ~489,400 lines |
| **Python tests** | 9,100+ |
| **Rust tests** | 777 |
| **Tutorial notebooks** (`docs/notebooks/`) | 66 |

## Development History

See [ROADMAP.md](ROADMAP.md) for the full development roadmap and task history.

## License

[Eclipse Public License 2.0 (EPL-2.0)](LICENSE)
