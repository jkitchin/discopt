# discopt

[![PyPI](https://img.shields.io/pypi/v/discopt)](https://pypi.org/project/discopt/)
[![CI](https://github.com/jkitchin/discopt/actions/workflows/ci.yml/badge.svg)](https://github.com/jkitchin/discopt/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jkitchin/discopt/graph/badge.svg?token=B3Y6LAtox9)](https://codecov.io/gh/jkitchin/discopt)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19762815-blue)](https://doi.org/10.5281/zenodo.19762815)
![PyPI Downloads](https://img.shields.io/pypi/dm/discopt.svg)

[![discopt](https://github.com/jkitchin/discopt/blob/main/discopt.png?raw=true)](https://github.com/jkitchin/discopt/blob/main/discopt.png?raw=true)



A hybrid Mixed-Integer Nonlinear Programming (MINLP) solver combining a Rust backend, JAX automatic differentiation, and Python orchestration. Solves MINLP problems via NLP-based spatial Branch and Bound with JIT-compiled objective/gradient/Hessian evaluation.

## Features

- **Algebraic modeling API** -- continuous, binary, and integer variables with operator overloading
- **Spatial Branch and Bound** -- Rust-powered node pool, branching, and pruning
- **JIT-compiled NLP evaluation** -- objective, gradient, Hessian, and constraint Jacobian via JAX
- **Three NLP backends** -- POUNCE (pure-Rust Ipopt port; default for single solves), pure-JAX interior-point method (vmap-batched B&B node engine), cyipopt (Ipopt)
- **Convex relaxations** -- McCormick envelopes (28 functions including sigmoid/softplus/tanh and the trig/inverse-trig/`erf` families), piecewise McCormick, alphaBB underestimators
- **Neural network embedding** -- embed trained feedforward networks (ReLU, sigmoid, tanh, softplus) as MINLP constraints via big-M, full-space, and reduced-space strategies; interval arithmetic bound propagation; ONNX import (`pip install discopt[nn]`)
- **Generalized disjunctive programming** -- `BooleanVar`, propositional logic operators (`land`, `lor`, `lnot`, `atleast`, `atmost`, `exactly`), `either_or()`, `if_then()`; reformulated via big-M, multiple big-M (LP-tightened), hull, or Logic-based Outer Approximation (`gdp_method="loa"`)
- **Complementarity / MPEC** -- `Model.complementarity(x, y)` reformulated via GDP disjunction (default), Scholtes regularization, or SOS1
- **Specialized problem classes** -- pooling problem (pq-formulation), geometric programming (posynomial detection + log-space convex reformulation), AC optimal power flow (rectangular QCQP)
- **Robust & multi-objective optimization** -- uncertainty sets with affine decision rules; scalarization (weighted-sum, ε-constraint, Tchebycheff, NBI, NNC) with Pareto-front analysis
- **Parameter estimation** -- weighted-least-squares estimation with exact JAX Fisher-information Jacobians; model-based design of experiments (D/A/E-optimality, identifiability, model discrimination) is available via the [discopt-doe](https://github.com/jkitchin/discopt-doe) plugin
- **Presolve** -- FBBT (interval arithmetic, probing, Big-M simplification, integrality-aware snapping, periodic-variable reduction), OBBT with LP warm-start
- **Cutting planes** -- reformulation-linearization (RLT, a first-class `rlt_cuts=True` option), PSD/SOC cuts for QCQP, and outer approximation (OA); `cuts='auto'` by default
- **Primal heuristics** -- multi-start NLP, feasibility pump, diving, RINS, local branching
- **Infeasibility diagnosis** -- irreducible infeasible subsystem (`compute_iis`) and conflict analysis / no-good cuts
- **GNN branching policy** -- bipartite graph-neural-network scaffold for learned branching (experimental; ships untrained, see #236)
- **Differentiable optimization** -- parameter sensitivity via envelope theorem and KKT implicit differentiation, including differentiable MILP/MIQP (fix-and-differentiate)
- **.nl file import** -- read AMPL-format models via Rust parser
- **Pyomo solver plugin** -- use discopt from existing Pyomo models via `SolverFactory("discopt")` (`pip install discopt[pyomo]`); see [docs/pyomo_solver.md](docs/pyomo_solver.md)
- **Dynamic optimization** -- DAE collocation (Radau/Legendre) and finite differences for optimal control, parameter estimation, and PDE-constrained optimization
- **CUTEst interface** -- NLP benchmarking against the CUTEst test set
- **LLM integration** (optional) -- conversational model building, diagnostics, and reformulation suggestions
- **Extensive test suite** -- 339 Rust + 3,700+ Python test functions

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
Model.solve()  -->  Python orchestrator  -->  Rust TreeManager (B&B engine)
                        |                          |
                  JAX NLPEvaluator           Node pool / branching / pruning
                  NLP backends:              Zero-copy numpy arrays (PyO3)
                    pounce  (pure-Rust Ipopt port)  [default single solve]
                    ipm     (pure-JAX, vmap batch)  [B&B node relaxations]
                    cyipopt (Ipopt)
```

**Rust backend** (`crates/discopt-core`): Expression IR, Branch and Bound tree (node pool, branching, pruning), .nl file parser, FBBT/presolve (interval arithmetic, probing, Big-M simplification).

**Rust-Python bindings** (`crates/discopt-python`): PyO3 bindings with zero-copy numpy array transfer for the B&B tree manager, expression IR, batch dispatch, and .nl parser.

**JAX layer** (`python/discopt/_jax`): DAG compiler mapping modeling expressions to JAX primitives, JIT-compiled NLP evaluator (objective, gradient, Hessian, constraint Jacobian), McCormick convex/concave relaxations (28 functions), and a relaxation compiler with vmap support.

**Solver wrappers** (`python/discopt/solvers`): POUNCE (pure-Rust Ipopt port), cyipopt NLP wrapper for Ipopt, HiGHS LP and MILP wrappers with warm-start support.

**CUTEst interface** (`python/discopt/interfaces/cutest.py`): PyCUTEst-based evaluator for NLP benchmarking against the CUTEst test set.

**Orchestrator** (`python/discopt/solver.py`): End-to-end `Model.solve()` connecting all components. At each B&B node: solve continuous NLP relaxation with tightened bounds, prune infeasible nodes, fathom integer-feasible solutions, branch on most fractional variable.

## NLP Backends

| Backend              | Implementation       | Use Case                                   |
|----------------------|----------------------|--------------------------------------------|
| `pounce` (default)   | Pure-Rust Ipopt port | Single-problem NLP; fastest wall-clock     |
| `ipm`                | Pure-JAX IPM         | B&B inner loop; GPU-batched via `jax.vmap`  |
| `cyipopt`            | Ipopt via cyipopt    | Single-problem NLP; most robust            |

For single continuous solves the default NLP backend resolves to a KKT-valid
solver -- POUNCE when installed, falling back to cyipopt, then to the pure-JAX
IPM. The pure-JAX `ipm` remains the vmap-batched engine for B&B node relaxations.

```python
result = model.solve()                       # default: POUNCE when installed
result = model.solve(nlp_solver="pounce")    # POUNCE (pure-Rust Ipopt port)
result = model.solve(nlp_solver="ipm")       # Pure-JAX IPM
result = model.solve(nlp_solver="cyipopt")   # Ipopt
```

## Benchmarks

Performance measured on Apple M4 Pro (CPU, JAX 0.8.2). "Warm" times exclude JIT compilation. All solvers produce matching objective values.

| Problem Class | discopt | Comparison | Notes |
|---------------|---------|------------|-------|
| **LP** (n=100) | 0.015s warm | HiGHS 0.002s, scipy 0.002s | Algebraic extraction, no autodiff |
| **QP** (n=100) | 0.04s warm | scipy SLSQP 0.02s | Was 66s before algebraic extraction |
| **MILP** (n=25) | 0.002s | HiGHS MIP 0.002s | B&B + LP relaxation, correct objectives |
| **MIQP** (n=10) | 0.004s | NLP path 4.9s | QP-specialized path: 1000x+ speedup |
| **NLP** (n=20, Rosenbrock) | IPM 1.1s warm, POUNCE 0.42s, Ipopt 0.43s | -- | POUNCE fastest single-solve; IPM best for batched B&B |
| **MINLP** (n=10) | 0.9s (batch=1) | 0.9s (batch=16) | vmap batching helps with deeper B&B trees |

See the benchmark notebooks for full scaling plots and details:
- [Benchmarks by Problem Class](docs/notebooks/benchmarks_by_class.ipynb) -- LP, QP, MILP, MIQP, NLP (3 backends), MINLP
- [IPM vs POUNCE vs Ipopt](docs/notebooks/ipm_vs_ipopt.ipynb) -- detailed NLP backend comparison (incl. vmap-batched IPM for B&B inner loops)

## Installation

Requires Rust 1.84+ and Python 3.10+. POUNCE (the default single-solve NLP
backend) is a pure-Rust Ipopt port with no system dependencies; cyipopt is an
optional fallback that needs the Ipopt C library.

```bash
# Install the POUNCE NLP backend (pure-Rust Ipopt port)
pip install pounce-solver

# Optional cyipopt fallback (needs the Ipopt C library; macOS: brew install ipopt)
pip install "discopt[ipopt]"

# Build Rust-Python bindings
cd crates/discopt-python && maturin develop && cd ../..

# Run the fast default PR battery
cargo test -p discopt-core
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 make test
```

`make test` matches the PR CI gate: ordinary non-slow tests plus the
`pr_correctness` subset. Full correctness, integration, and benchmark markers
remain available through the explicit Make targets.

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
NLP subproblem (Ipopt) and refines the partition where the relaxation gap is
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
| `milp_solver` | `"auto"` | MILP master backend: `"auto"`, `"highs"`, `"pounce"`, `"simplex"`, or `"gurobi"` |
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
environment uses solver-independent checks plus HiGHS-backed MILP relaxations,
and excludes optional cyipopt, longer Alpine, MINLPTests, and incidence-style
AMP benchmark coverage. AMP and PR-fast Make targets run pytest through
`scripts/run_memory_capped_pytest.sh`, which applies a 32 GB address-space cap
with `prlimit` when available. Override with `PYTEST_MEMORY_LIMIT_MB=...`, or
set `PYTEST_MEMORY_LIMIT_MB=0` to disable the cap. The broad `make test-quick`
dev-loop target remains uncapped and excludes `memory_heavy` tests.

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
uv pip install maturin pytest pytest-timeout numpy scipy jax jaxlib highspy cyipopt
uv pip install -e ".[dev,ipopt,highs]"
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
12 GB cap is useful for reproducing memory pressure, but the current JAX/XLA
CPU stack can reserve more than 12 GB of virtual address space during AMP runs;
use the `memory_heavy` marker selection when running with tighter caps.

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
| **[discopt-apps](https://github.com/jkitchin/discopt-apps)** | `pip install "git+https://github.com/jkitchin/discopt-apps.git"` | **Application builders** for the modeling language: AC optimal power flow (`discopt.opf`) and pooling (`discopt.pooling`). |
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
discopt convert in.gms out.nl
discopt install-skills   # Install Claude Code slash commands and agents
```

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

- **Quickstart** -- basic modeling and solving
- **MINLP Examples** -- mixed-integer nonlinear programs
- **Advanced Features** -- relaxations, presolve, cutting planes, branching policies
- **Global Optimization** -- which problems discopt can and can't certify as global
- **IPM vs Ipopt** -- backend comparison (incl. vmap-batched IPM)
- **Dynamic Optimization** -- DAE collocation for optimal control, parameter estimation, and PDEs
- **Neural Network Embedding** -- optimize over trained ML surrogates as MINLP constraints
- **Decision-Focused Learning** -- differentiable optimization in ML pipelines
- **GDP Tutorial** -- disjunctive programming, logical constraints, big-M/hull/LOA reformulations

Full documentation is built with Jupyter Book: `jupyter-book build docs/`

## Project Statistics

*Last updated: 2026-06-18*

| Category | Count |
|----------|-------|
| **Python source** (`python/discopt/`) | 226 files, ~103,700 lines |
| **Rust source** (`crates/`) | 55 files, ~29,000 lines |
| **Test code** (`python/tests/`) | 222 files, ~72,100 lines |
| **Total source + tests** | ~500 files, ~205,000 lines |
| **Python tests** | 3,700+ |
| **Rust tests** | 339 |
| **Tutorial notebooks** (`docs/notebooks/`) | 63 |

## Development History

See [ROADMAP.md](ROADMAP.md) for the full development roadmap and task history.

## License

[Eclipse Public License 2.0 (EPL-2.0)](LICENSE)
