# discopt

[![CI](https://github.com/jkitchin/discopt/actions/workflows/ci.yml/badge.svg)](https://github.com/jkitchin/discopt/actions/workflows/ci.yml)
[![Nightly](https://github.com/jkitchin/discopt/actions/workflows/nightly.yml/badge.svg)](https://github.com/jkitchin/discopt/actions/workflows/nightly.yml)

![img](discopt.png)

A hybrid Mixed-Integer Nonlinear Programming (MINLP) solver combining a Rust backend, JAX automatic differentiation, and Python orchestration. Solves MINLP problems via NLP-based spatial Branch and Bound with JIT-compiled objective/gradient/Hessian evaluation.

## Features

- **Algebraic modeling API** -- continuous, binary, and integer variables with operator overloading
- **Spatial Branch and Bound** -- Rust-powered node pool, branching, and pruning
- **JIT-compiled NLP evaluation** -- objective, gradient, Hessian, and constraint Jacobian via JAX
- **Three NLP backends** -- pure-JAX interior-point method (default, vmap-batched), ripopt (Rust IPM via PyO3), cyipopt (Ipopt)
- **Convex relaxations** -- McCormick envelopes (19 functions), piecewise McCormick, alphaBB underestimators
- **Presolve** -- FBBT (interval arithmetic, probing, Big-M simplification), OBBT with LP warm-start
- **Cutting planes** -- reformulation-linearization (RLT) and outer approximation (OA)
- **GNN branching policy** -- bipartite graph neural network trained on strong branching data
- **Primal heuristics** -- multi-start NLP, feasibility pump
- **Differentiable optimization** -- parameter sensitivity via envelope theorem and KKT implicit differentiation
- **.nl file import** -- read AMPL-format models via Rust parser
- **CUTEst interface** -- NLP benchmarking against the CUTEst test set
- **LLM integration** (optional) -- conversational model building, diagnostics, and reformulation suggestions
- **900+ tests** -- 140 Rust + 770+ Python

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

**JAX layer** (`python/discopt/_jax`): DAG compiler mapping modeling expressions to JAX primitives, JIT-compiled NLP evaluator (objective, gradient, Hessian, constraint Jacobian), McCormick convex/concave relaxations (19 functions), and a relaxation compiler with vmap support.

**Solver wrappers** (`python/discopt/solvers`): ripopt (Rust IPM via PyO3), cyipopt NLP wrapper for Ipopt, HiGHS LP wrapper with warm-start support.

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

## Installation

Requires Rust 1.84+, Python 3.10+, and Ipopt.

```bash
# Install Ipopt (macOS)
brew install ipopt

# Clone ripopt alongside discopt (path dependency at ../ripopt)
git clone <ripopt-repo-url> ../ripopt

# Build Rust-Python bindings (includes ripopt PyO3 bindings)
cd crates/discopt-python && maturin develop && cd ../..

# Run tests
cargo test -p discopt-core
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 pytest python/tests/ -v
```

## Documentation

Tutorial notebooks are available in the `notebooks/` directory:

- **Quickstart** -- basic modeling and solving
- **MINLP Examples** -- mixed-integer nonlinear programs
- **Advanced Features** -- relaxations, presolve, cutting planes, branching policies
- **IPM vs Ipopt** -- backend comparison
- **Batch IPM** -- vmap-batched interior-point solving
- **Decision-Focused Learning** -- differentiable optimization in ML pipelines

Full documentation is built with Jupyter Book: `jupyter-book build docs/`

## Development History

See [ROADMAP.md](ROADMAP.md) for the full development roadmap and task history.

## License

[Eclipse Public License 2.0 (EPL-2.0)](LICENSE)

## Tasks

- [ ] Performance benchmarks (timing, solution success)
  - [ ] Across LP, QP, MIP, MILP, MIQP, MINLP
    - [ ] Pure Jax version
    - [ ] ripopt version
- [ ] Add sparsity for large scale problems
- [ ] OptiLitBot to scan literature for new papers that are relevant