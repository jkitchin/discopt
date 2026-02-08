# discopt

[![CI](https://github.com/jkitchin/discopt/actions/workflows/ci.yml/badge.svg)](https://github.com/jkitchin/discopt/actions/workflows/ci.yml)
[![Nightly](https://github.com/jkitchin/discopt/actions/workflows/nightly.yml/badge.svg)](https://github.com/jkitchin/discopt/actions/workflows/nightly.yml)

![img](discopt.png)

A hybrid Mixed-Integer Nonlinear Programming (MINLP) solver combining a Rust backend, JAX automatic differentiation, and Python orchestration. Solves MINLP problems via NLP-based spatial Branch & Bound with JIT-compiled objective/gradient/Hessian evaluation.

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

**Rust backend** (`crates/discopt-core`): Expression IR, Branch & Bound tree (node pool, branching, pruning), .nl file parser, FBBT/presolve (interval arithmetic, probing, Big-M simplification).

**Rust-Python bindings** (`crates/discopt-python`): PyO3 bindings with zero-copy numpy array transfer for the B&B tree manager, expression IR, batch dispatch, and .nl parser.

**JAX layer** (`python/discopt/_jax`): DAG compiler mapping modeling expressions to JAX primitives, JIT-compiled NLP evaluator (objective, gradient, Hessian, constraint Jacobian), McCormick convex/concave relaxations (19 functions), and a relaxation compiler with vmap support.

**Solver wrappers** (`python/discopt/solvers`): ripopt (Rust IPM via PyO3), cyipopt NLP wrapper for Ipopt, HiGHS LP wrapper with warm-start support.

**CUTEst interface** (`python/discopt/interfaces/cutest.py`): PyCUTEst-based evaluator for NLP benchmarking against the CUTEst test set.

**Orchestrator** (`python/discopt/solver.py`): End-to-end `Model.solve()` connecting all components. At each B&B node: solve continuous NLP relaxation with tightened bounds, prune infeasible nodes, fathom integer-feasible solutions, branch on most fractional variable.

## Current Status

| Component                      | Status   | Tests                     |
|--------------------------------|----------|---------------------------|
| Expression IR (Rust)           | Complete | 48 Rust + 40 Python       |
| B&B Tree (Rust)                | Complete | 33 Rust                   |
| .nl Parser (Rust)              | Complete | 34 Rust + 17 Python       |
| FBBT/Presolve (Rust)           | Complete | 45 Rust                   |
| OBBT Bound Tightening          | Complete | 13 Rust + 26 Python       |
| JAX DAG Compiler               | Complete | 70 Python                 |
| McCormick Relaxations          | Complete | 88 Python                 |
| Relaxation Compiler            | Complete | 33 Python                 |
| NLP Evaluator (JAX)            | Complete | 45 Python                 |
| HiGHS LP Wrapper               | Complete | 20 Python                 |
| cyipopt NLP Wrapper            | Complete | 24 Python                 |
| Batch Dispatch                 | Complete | 34 Python                 |
| Solver Orchestrator            | Complete | 42 Python                 |
| Batch Relaxation Evaluator     | Complete | 26 Python                 |
| Differentiable Solving (L1+L3) | Complete | 46 Python                 |
| Multi-start Heuristics         | Complete | 28 Python                 |
| Benchmark Runner               | Complete | 28 Python                 |
| Piecewise McCormick            | Complete | 40 Python                 |
| alphaBB Underestimators        | Complete | 25 Python                 |
| Cutting Planes (RLT/OA)       | Complete | 40 Python                 |
| GNN Branching Policy           | Complete | 21 Python                 |
| Pure-JAX IPM                   | Complete | 36 Python                 |
| ripopt Integration (PyO3)      | Complete | (in solver tests)         |
| CUTEst Interface               | Complete | 22 Python                 |
| **Total**                      |          | **140 Rust + 772 Python** |

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

## Roadmap

The project follows a 4-phase plan. Phases 1-3 are complete; Phase 4 is in progress.

### Phase 1: Working Solver (complete)

| Task                     | Status      | Description                                        |
|--------------------------|-------------|----------------------------------------------------|
| T0 Architectural spike   | Done        | Rust-JAX GPU batch latency validation              |
| T1 Cargo workspace       | Done        | discopt-core + discopt-python, maturin builds      |
| T2 Expression IR         | Done        | Rust expression graph + PyO3 bindings              |
| T3 .nl parser            | Done        | AMPL .nl file parser in Rust                       |
| T4 JAX DAG compiler      | Done        | Expression-to-JAX compilation                      |
| T5 McCormick relaxations | Done        | 19 convex/concave relaxation functions             |
| T6 Relaxation compiler   | Done        | jit+vmap compatible relaxation compilation         |
| T7 HiGHS LP wrapper      | Done        | LP solver with warm-start support                  |
| T8 NLP evaluator         | Done        | JIT-compiled grad/Hessian/Jacobian via JAX         |
| T9 cyipopt NLP wrapper   | Done        | Ipopt interface for continuous relaxations         |
| T10 CI/CD                | Done        | GitHub Actions, ruff, mypy, cargo                  |
| T11 B&B tree             | Done        | Rust node pool, branching, pruning                 |
| T12 Batch dispatch       | Done        | Zero-copy Rust-Python array transfer               |
| T13 FBBT/presolve        | Done        | Interval arithmetic, probing, Big-M simplification |
| T14 Solver orchestrator  | Done        | End-to-end Model.solve() via B&B                   |
| T15 MINLPLib validation  | Done        | 34 solvable instances, zero incorrect              |
| T16 Phase 1 gate         | Done        | All criteria pass                                  |
| T9a Rust Ipopt (ripopt)  | Superseded  | Replaced by T17 pure-JAX IPM                       |
|                          |             |                                                    |

### Phase 2: GPU + Differentiability (complete)

| Task                                 | Status      | Description                                             |
|--------------------------------------|-------------|---------------------------------------------------------|
| T19 Batch relaxation evaluator       | Done        | jax.vmap-based batch McCormick evaluation               |
| T21 OBBT bound tightening            | Done        | LP-based bound tightening with HiGHS warm-start         |
| T22 Differentiable solving (Level 1) | Done        | custom_jvp + envelope theorem for parameter sensitivity |
| T20 Multi-start heuristics           | Done        | Multi-start NLP solving + feasibility pump              |
| T23 Differentiable solving (Level 3) | Done        | Implicit differentiation at active set via KKT          |
| T25 Benchmark runner                 | Done        | Performance metrics, batch scaling, JSON export         |
| T17 GPU-batched IPM                  | Done        | Pure-JAX IPM solver with augmented KKT, vmap batch solving |
| T24 GPU IPM in solver loop           | Done        | Batch IPM in B&B loop, ipm default backend              |

### Phase 3: Competitive Performance (complete)

| Task                                 | Status      | Description                                             |
|--------------------------------------|-------------|---------------------------------------------------------|
| Piecewise McCormick                  | Done        | k-partition domain splitting, O(1/k^2) convergence     |
| alphaBB underestimators              | Done        | Hessian-based convexification (Adjiman/Floudas 1998)    |
| Cutting planes (RLT/OA)             | Done        | Bilinear RLT cuts, gradient OA, separation oracles      |
| GNN branching policy                 | Done        | Bipartite graph GNN, strong branching data collection   |
| Solver integration                   | Done        | partitions, branching_policy, cutting_planes parameters |

### Phase 4: Polish + Release (in progress)

| Task                                 | Status      | Description                                             |
|--------------------------------------|-------------|---------------------------------------------------------|
| ripopt integration (PyO3)            | Done        | Rust IPM solver via PyO3 bindings (`nlp_solver="ripopt"`) |
| CUTEst interface                     | Done        | PyCUTEst evaluator for NLP benchmarking                 |
| Documentation + example notebooks    | In progress | Quickstart, advanced features, IPM comparison notebooks |
| Release engineering                  | In progress | `pip install discopt`, pyproject.toml, packaging        |

## Building

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

## License

[Eclipse Public License 2.0 (EPL-2.0)](LICENSE)
