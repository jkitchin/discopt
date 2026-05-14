# discopt

[![PyPI](https://img.shields.io/pypi/v/discopt)](https://pypi.org/project/discopt/)
[![CI](https://github.com/jkitchin/discopt/actions/workflows/ci.yml/badge.svg)](https://github.com/jkitchin/discopt/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jkitchin/discopt/graph/badge.svg?token=B3Y6LAtox9)](https://codecov.io/gh/jkitchin/discopt)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19762815.svg?v=2)](https://doi.org/10.5281/zenodo.19762815)
![PyPI Downloads](https://img.shields.io/pypi/dm/discopt.svg)

[![discopt](https://github.com/jkitchin/discopt/blob/main/discopt.png?raw=true)](https://github.com/jkitchin/discopt/blob/main/discopt.png?raw=true)



A hybrid Mixed-Integer Nonlinear Programming (MINLP) solver combining a Rust backend, JAX automatic differentiation, and Python orchestration. Solves MINLP problems via NLP-based spatial Branch and Bound with JIT-compiled objective/gradient/Hessian evaluation.

## Features

- **Algebraic modeling API** -- continuous, binary, and integer variables with operator overloading
- **Spatial Branch and Bound** -- Rust-powered node pool, branching, and pruning
- **JIT-compiled NLP evaluation** -- objective, gradient, Hessian, and constraint Jacobian via JAX
- **Three NLP backends** -- pure-JAX interior-point method (default, vmap-batched), ripopt (Rust IPM via PyO3), cyipopt (Ipopt)
- **Convex relaxations** -- McCormick envelopes (21 functions including sigmoid/softplus/tanh), piecewise McCormick, alphaBB underestimators
- **Neural network embedding** -- embed trained feedforward networks (ReLU, sigmoid, tanh, softplus) as MINLP constraints via big-M, full-space, and reduced-space strategies; interval arithmetic bound propagation; ONNX import (`pip install discopt[nn]`)
- **Generalized disjunctive programming** -- `BooleanVar`, propositional logic operators (`land`, `lor`, `lnot`, `atleast`, `atmost`, `exactly`), `either_or()`, `if_then()`; reformulated via big-M, multiple big-M (LP-tightened), hull, or Logic-based Outer Approximation (`gdp_method="loa"`)
- **Presolve** -- FBBT (interval arithmetic, probing, Big-M simplification), OBBT with LP warm-start
- **Cutting planes** -- reformulation-linearization (RLT) and outer approximation (OA)
- **GNN branching policy** -- bipartite graph neural network trained on strong branching data
- **Primal heuristics** -- multi-start NLP, feasibility pump
- **Differentiable optimization** -- parameter sensitivity via envelope theorem and KKT implicit differentiation
- **.nl file import** -- read AMPL-format models via Rust parser
- **Dynamic optimization** -- DAE collocation (Radau/Legendre) and finite differences for optimal control, parameter estimation, and PDE-constrained optimization
- **CUTEst interface** -- NLP benchmarking against the CUTEst test set
- **LLM integration** (optional) -- conversational model building, diagnostics, and reformulation suggestions
- **1650+ tests** -- 141 Rust + 1510+ Python

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
                    ripopt  (Rust IPM, PyO3)
                    ipm     (pure-JAX, vmap batch)  [default]
                    cyipopt (Ipopt)
```

**Rust backend** (`crates/discopt-core`): Expression IR, Branch and Bound tree (node pool, branching, pruning), .nl file parser, FBBT/presolve (interval arithmetic, probing, Big-M simplification).

**Rust-Python bindings** (`crates/discopt-python`): PyO3 bindings with zero-copy numpy array transfer for the B&B tree manager, expression IR, batch dispatch, and .nl parser.

**JAX layer** (`python/discopt/_jax`): DAG compiler mapping modeling expressions to JAX primitives, JIT-compiled NLP evaluator (objective, gradient, Hessian, constraint Jacobian), McCormick convex/concave relaxations (21 functions), and a relaxation compiler with vmap support.

**Solver wrappers** (`python/discopt/solvers`): ripopt (Rust IPM via PyO3), cyipopt NLP wrapper for Ipopt, HiGHS LP and MILP wrappers with warm-start support.

**CUTEst interface** (`python/discopt/interfaces/cutest.py`): PyCUTEst-based evaluator for NLP benchmarking against the CUTEst test set.

**Orchestrator** (`python/discopt/solver.py`): End-to-end `Model.solve()` connecting all components. At each B&B node: solve continuous NLP relaxation with tightened bounds, prune infeasible nodes, fathom integer-feasible solutions, branch on most fractional variable.

## NLP Backends

| Backend         | Implementation    | Use Case                                   |
|-----------------|-------------------|--------------------------------------------|
| `ipm` (default) | Pure-JAX IPM      | B&B inner loop; GPU-batched via `jax.vmap` |
| `ripopt`        | Rust IPM via PyO3 | Single-problem NLP; fastest wall-clock     |
| `cyipopt`       | Ipopt via cyipopt | Single-problem NLP; most robust            |

```python
result = model.solve(nlp_solver="ipm")      # Pure-JAX (default)
result = model.solve(nlp_solver="ripopt")   # Rust IPM
result = model.solve(nlp_solver="cyipopt")  # Ipopt
```

## Benchmarks

Performance measured on Apple M4 Pro (CPU, JAX 0.8.2). "Warm" times exclude JIT compilation. All solvers produce matching objective values.

| Problem Class | discopt | Comparison | Notes |
|---------------|---------|------------|-------|
| **LP** (n=100) | 0.015s warm | HiGHS 0.002s, scipy 0.002s | Algebraic extraction, no autodiff |
| **QP** (n=100) | 0.04s warm | scipy SLSQP 0.02s | Was 66s before algebraic extraction |
| **MILP** (n=25) | 0.002s | HiGHS MIP 0.002s | B&B + LP relaxation, correct objectives |
| **MIQP** (n=10) | 0.004s | NLP path 4.9s | QP-specialized path: 1000x+ speedup |
| **NLP** (n=20, Rosenbrock) | IPM 1.1s warm, ripopt 0.42s, Ipopt 0.43s | -- | ripopt fastest single-solve; IPM best for batched B&B |
| **MINLP** (n=10) | 0.9s (batch=1) | 0.9s (batch=16) | vmap batching helps with deeper B&B trees |

See the benchmark notebooks for full scaling plots and details:
- [Benchmarks by Problem Class](docs/notebooks/benchmarks_by_class.ipynb) -- LP, QP, MILP, MIQP, NLP (3 backends), MINLP
- [IPM vs ripopt vs Ipopt](docs/notebooks/ipm_vs_ipopt.ipynb) -- detailed NLP backend comparison
- [Batch IPM vs Ipopt](docs/notebooks/batch_ipm_vs_ipopt.ipynb) -- vmap-batched IPM for B&B inner loops

## Installation

Requires Rust 1.84+, Python 3.10+, and Ipopt.

```bash
# Install Ipopt (macOS)
brew install ipopt

# Clone ripopt alongside discopt (path dependency at ../ripopt)
git clone <ripopt-repo-url> ../ripopt

# Build Rust-Python bindings (includes ripopt PyO3 bindings)
cd crates/discopt-python && maturin develop && cd ../..

# Run the fast default PR battery
cargo test -p discopt-core
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 make test
```

`make test` matches the PR CI gate: ordinary non-slow tests plus the
`pr_correctness` subset. Full correctness, integration, and benchmark markers
remain available through the explicit Make targets.

### AMP Test Suites

Routine AMP development uses a fast default regression battery. The fast
environment uses solver-independent checks plus HiGHS-backed MILP relaxations,
and excludes optional cyipopt, longer Alpine, MINLPTests, and incidence-style
AMP benchmark coverage. AMP and PR-fast Make targets run pytest through
`scripts/run_memory_capped_pytest.sh`, which applies a 16 GB address-space cap
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

For WSL or memory-constrained machines, keep broad AMP/JAX runs capped and use a
bounded xdist worker count rather than `-n auto`:

```bash
PYTEST_MEMORY_LIMIT_MB=16384 PYTEST_XDIST_WORKERS=2 make test
PYTEST_MEMORY_LIMIT_MB=16384 make test-amp-integration
```

WSL users should also set explicit memory and swap limits in `.wslconfig` so a
single uncapped compile-heavy test cannot restart the host session. A stricter
12 GB cap is useful for reproducing memory pressure, but the current JAX/XLA
CPU stack can reserve more than 12 GB of virtual address space during AMP runs;
use the `memory_heavy` marker selection when running with tighter caps.

The full Python test suite remains available with `make test-all`.

## Command-Line Interface

After installation, the `discopt` command is available on your PATH:

```bash
discopt about            # Version and installation info
discopt test             # Smoke-test the install
discopt convert in.gms out.nl
discopt install-skills   # Install Claude Code slash commands and agents
```

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
- **IPM vs Ipopt** -- backend comparison
- **Batch IPM** -- vmap-batched interior-point solving
- **Dynamic Optimization** -- DAE collocation for optimal control, parameter estimation, and PDEs
- **Neural Network Embedding** -- optimize over trained ML surrogates as MINLP constraints
- **Decision-Focused Learning** -- differentiable optimization in ML pipelines
- **GDP Tutorial** -- disjunctive programming, logical constraints, big-M/hull/LOA reformulations

Full documentation is built with Jupyter Book: `jupyter-book build docs/`

## Project Statistics

*Last updated: 2026-02-16*

| Category | Count |
|----------|-------|
| **Python source** (`python/discopt/`) | 65 files, ~27,200 lines |
| **Rust source** (`crates/`) | 19 files, ~10,700 lines |
| **Test code** (`python/tests/`) | 41 files, ~24,500 lines |
| **Total source + tests** | 125 files, ~62,400 lines |
| **Python tests** | 1,510+ |
| **Rust tests** | 141 |
| **Tutorial notebooks** (`docs/notebooks/`) | 21 |
| **Git commits** | 99 |

## Development History

See [ROADMAP.md](ROADMAP.md) for the full development roadmap and task history.

## License

[Eclipse Public License 2.0 (EPL-2.0)](LICENSE)
