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
- **In-house LP/MILP engine** -- pure-Rust primal/dual simplex with warm starts and a sparse LU basis (`feral`); HiGHS is no longer on the LP/MILP path
- **NLP backends** -- POUNCE (pure-Rust Ipopt port, the universal default) and cyipopt (Ipopt); `nlp_solver="simplex"` selects the pure-Rust warm-started-simplex MILP B&B. The pure-JAX IPM has been retired -- `"ipm"`/`"sparse_ipm"` remain as back-compat aliases
- **Convex relaxations** -- McCormick envelopes over 28 primitive operations (bilinear, powers, `exp`/`log` family, trig and inverse-trig, hyperbolics, `sigmoid`/`softplus`/`tanh`, `abs`/`min`/`max`/`sign`/`entropy`) plus a 19-intrinsic univariate envelope table in the uniform factorable engine (adding `erf`, `log1p`, and the inverse hyperbolics); piecewise McCormick, alphaBB underestimators, and G-convexity / convex-transformable relaxations
- **Certified global MINLP** -- Adaptive Multivariate Partitioning (`solver="amp"`) for nonconvex bilinear/trilinear/signomial/trig models, and a signomial global optimizer (`DISCOPT_SGO`) for mixed-sign signomial and integer-signomial problems
- **Decomposition solvers** -- MIP-NLP family (`solver="mip-nlp"`: OA, ECP, FP, GOA, LP/NLP-BB), Benders and Generalized Benders (GBD), Lagrangian decomposition, and an automatic structure/decomposition advisor
- **Derivative-free optimization** -- `solver="direct"` (sampling search over black-box `dm.custom` bodies) and `solver="surrogate"` (surrogate-model search); both are explicitly non-certifying, and a governed variant runs as a root heuristic
- **Neural network & tree embedding** -- embed trained feedforward networks (ReLU, sigmoid, tanh, softplus) as MINLP constraints via big-M, full-space, and reduced-space formulations; decision trees and gradient-boosted ensembles via per-leaf MILP encoding; interval-arithmetic bound propagation; ONNX / scikit-learn / PyTorch readers. Trainable surrogates (`nn.trainable`, `nn.surrogate`) emit symbolic weights so a surrogate can be fit *simultaneously* with a physics model
- **Generalized disjunctive programming** -- `BooleanVar`, propositional logic operators (`land`, `lor`, `lnot`, `atleast`, `atmost`, `exactly`), `either_or()`, `if_then()`; reformulated via big-M, multiple big-M (LP-tightened), hull, or Logic-based Outer Approximation (`gdp_method="loa"`), with a disjunct-selection primal constructor on by default
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
- **Pyomo solver plugin** -- use discopt from existing Pyomo models via `SolverFactory("discopt")` (`pip install discopt[pyomo]`); see [docs/pyomo_solver.md](docs/pyomo_solver.md)
- **GAMS solver link** -- run discopt *as* a GAMS solver through the GMO/GEV API (`discopt gams-register`, `discopt gams-daemon`); see [docs/gams_solver_link.md](docs/gams_solver_link.md)
- **Warm solve daemon** -- `discopt solve model.nl` routes through a persistent daemon that keeps the process warm across solves
- **Dynamic optimization** -- DAE collocation (Radau/Legendre), finite differences, and method-of-lines for optimal control, parameter estimation, and PDE-constrained optimization, with multi-experiment trajectory fitting
- **Benchmark interfaces** -- CUTEst (NLP test set), MINLPLib `.nl`, and QPLIB (453 quadratic instances, 390 nonconvex, with reference solution vectors)
- **LLM integration** (optional) -- conversational model building, diagnostics, and reformulation suggestions
- **Extensive test suite** -- 619 Rust + 7,100+ Python test functions

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
print(result.x)          # {"x": 0.5, "y": 0.5, "z": 0.0}
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
`DISCOPT_NLP_EVAL=jax` selects it wholesale. A default solve does not import JAX --
not on the LP, QP, MIQP and simplex-MILP paths, and not on the nonlinear ones either.

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
optional Gurobi backend. highspy is used only on the OA/GDP paths.

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
| `simplex`                      | Pure-Rust warm-started simplex B&B    | MILP; the fully JAX-free MILP path          |
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
[`docs/notebooks/benchmarks_by_class.ipynb`](docs/notebooks/benchmarks_by_class.ipynb),
re-executed on the current Rust AD tape backend (Python 3.12, CPU, median of 3 runs
including setup). Absolute times are machine-dependent -- the notebook is the
reproducible source. All solvers agree on the objective value.

| Problem Class | discopt | Comparison | Notes |
|---------------|---------|------------|-------|
| **LP** (n=100) | 0.234s | HiGHS 0.0015s, scipy 0.0019s | Algebraic extraction, no autodiff |
| **QP** (n=100) | 0.417s | scipy SLSQP 0.023s | -- |
| **MILP** (n=25, 8 int) | 0.019s | HiGHS MIP 0.0017s | B&B + LP relaxation, correct objectives |
| **MIQP** (n=10) | 0.018s | forced NLP path 0.707s | QP-specialized path: ~40x speedup |
| **NLP** (n=20, Rosenbrock) | POUNCE 0.120s | cyipopt 0.126s | Two implementations of the same IPM |
| **MINLP** (n=10) | 0.026s (batch=1) | 0.026s (batch=16) | These trees close in 1-5 nodes, so batching has nothing to fill |

HiGHS (C++ simplex) and scipy remain faster on the LP/MILP classes, as expected for
mature production codes; discopt's value on these classes is that they are reachable
from the same model object as the MINLP path.

See the benchmark notebooks for full scaling plots and details:
- [Benchmarks by Problem Class](docs/notebooks/benchmarks_by_class.ipynb) -- LP, QP, MILP, MIQP, NLP, MINLP
- [NLP Backend Comparison](docs/notebooks/ipm_vs_ipopt.ipynb) -- POUNCE vs Ipopt

## Installation

Requires Rust 1.84+ and Python 3.10+. POUNCE -- the default numerical engine -- is a
pure-Rust Ipopt port installed as a core dependency, with no system libraries needed.
cyipopt is an optional fallback that needs the Ipopt C library.

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

### Solving nonconvex MINLPs with AMP

For problems with nonconvex nonlinearities (bilinear, trilinear, signomial,
trig), the default branch-and-bound path only certifies optimality when the
relaxation is convex. The Adaptive Multivariate Partitioning (AMP) solver
gives discopt a **certified-global** path for these problems:

```python
import discopt.modeling as dm

m = dm.Model("concave_qp")
c = [-1.0, 0.5, 1.5]
xs = [m.continuous(f"x{i}", lb=-2.0, ub=2.0) for i in range(3)]
m.subject_to(sum(xs) >= -1.0)
m.subject_to(sum(xs) <= 3.0)
m.minimize(sum(-((xs[i] - c[i]) ** 2) for i in range(3)))  # concave

result = m.solve(solver="amp", rel_gap=1e-4)
print(result.status, result.objective, result.gap)
```

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

All `discopt-dev` search subcommands output structured JSON. The `/discoptbot`
literature-scanner slash command uses them to automatically find and summarize
relevant new papers from arXiv and OpenAlex.

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

## Project Statistics

*Last updated: 2026-08-14*

| Category | Count |
|----------|-------|
| **Python source** (`python/discopt/`) | 333 files, ~170,200 lines |
| **Rust source** (`crates/`) | 77 files, ~58,800 lines |
| **Test code** (`python/tests/`) | 566 files, ~161,000 lines |
| **Total source + tests** | ~976 files, ~390,000 lines |
| **Python tests** | 7,100+ |
| **Rust tests** | 619 |
| **Tutorial notebooks** (`docs/notebooks/`) | 63 |

## Development History

See [ROADMAP.md](ROADMAP.md) for the full development roadmap and task history.

## License

[Eclipse Public License 2.0 (EPL-2.0)](LICENSE)
