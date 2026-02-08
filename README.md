# discopt

[![CI](https://github.com/jkitchin/discopt/actions/workflows/ci.yml/badge.svg)](https://github.com/jkitchin/discopt/actions/workflows/ci.yml)
[![Nightly](https://github.com/jkitchin/discopt/actions/workflows/nightly.yml/badge.svg)](https://github.com/jkitchin/discopt/actions/workflows/nightly.yml)

A hybrid Mixed-Integer Nonlinear Programming (MINLP) solver combining a Rust backend, JAX automatic differentiation, and Python orchestration. Solves MINLP problems via NLP-based spatial Branch & Bound with JIT-compiled objective/gradient/Hessian evaluation.

## Architecture

```
Model.solve()  -->  Python orchestrator  -->  Rust TreeManager (B&B engine)
                        |                          |
                  JAX NLPEvaluator           Node pool / branching / pruning
                  cyipopt (Ipopt)            Zero-copy numpy arrays (PyO3)
```

**Rust backend** (`crates/discopt-core`): Expression IR, Branch & Bound tree (node pool, branching, pruning), .nl file parser, FBBT/presolve (interval arithmetic, probing, Big-M simplification).

**Rust-Python bindings** (`crates/discopt-python`): PyO3 bindings with zero-copy numpy array transfer for the B&B tree manager, expression IR, batch dispatch, and .nl parser.

**JAX layer** (`python/discopt/_jax`): DAG compiler mapping modeling expressions to JAX primitives, JIT-compiled NLP evaluator (objective, gradient, Hessian, constraint Jacobian), McCormick convex/concave relaxations (19 functions), and a relaxation compiler with vmap support.

**Solver wrappers** (`python/discopt/solvers`): HiGHS LP wrapper with warm-start support, cyipopt NLP wrapper for Ipopt.

**Orchestrator** (`python/discopt/solver.py`): End-to-end `Model.solve()` connecting all components. At each B&B node: solve continuous NLP relaxation with tightened bounds, prune infeasible nodes, fathom integer-feasible solutions, branch on most fractional variable.

## Current Status

| Component | Status | Tests |
|-----------|--------|-------|
| Expression IR (Rust) | Complete | 48 Rust + 40 Python |
| B&B Tree (Rust) | Complete | 33 Rust |
| .nl Parser (Rust) | Complete | 34 Rust + 17 Python |
| FBBT/Presolve (Rust) | Complete | 45 Rust |
| OBBT Bound Tightening | Complete | 13 Rust + 26 Python |
| JAX DAG Compiler | Complete | 70 Python |
| McCormick Relaxations | Complete | 88 Python |
| Relaxation Compiler | Complete | 33 Python |
| NLP Evaluator (JAX) | Complete | 45 Python |
| HiGHS LP Wrapper | Complete | 20 Python |
| cyipopt NLP Wrapper | Complete | 24 Python |
| Batch Dispatch | Complete | 34 Python |
| Solver Orchestrator | Complete | 32 Python |
| Batch Relaxation Evaluator | Complete | 26 Python |
| Differentiable Solving | Complete | 25 Python |
| **Total** | | **140 Rust + 501 Python** |

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

The project follows a 4-phase plan. Phase 1 is complete; Phase 2 is in progress.

### Phase 1: Working Solver (complete)

| Task | Status | Description |
|------|--------|-------------|
| T0 Architectural spike | Done | Rust-JAX GPU batch latency validation |
| T1 Cargo workspace | Done | discopt-core + discopt-python, maturin builds |
| T2 Expression IR | Done | Rust expression graph + PyO3 bindings |
| T3 .nl parser | Done | AMPL .nl file parser in Rust |
| T4 JAX DAG compiler | Done | Expression-to-JAX compilation |
| T5 McCormick relaxations | Done | 19 convex/concave relaxation functions |
| T6 Relaxation compiler | Done | jit+vmap compatible relaxation compilation |
| T7 HiGHS LP wrapper | Done | LP solver with warm-start support |
| T8 NLP evaluator | Done | JIT-compiled grad/Hessian/Jacobian via JAX |
| T9 cyipopt NLP wrapper | Done | Ipopt interface for continuous relaxations |
| T10 CI/CD | Done | GitHub Actions, ruff, mypy, cargo |
| T11 B&B tree | Done | Rust node pool, branching, pruning |
| T12 Batch dispatch | Done | Zero-copy Rust-Python array transfer |
| T13 FBBT/presolve | Done | Interval arithmetic, probing, Big-M simplification |
| T14 Solver orchestrator | Done | End-to-end Model.solve() via B&B |
| T15 MINLPLib validation | Done | 34 solvable instances, zero incorrect |
| T16 Phase 1 gate | Done | All criteria pass |
| T9a Rust Ipopt (ripopt) | In progress | Translating Ipopt's IPM to pure Rust |

### Phase 2: GPU + Differentiability (current)

| Task | Status | Description |
|------|--------|-------------|
| T19 Batch relaxation evaluator | Done | jax.vmap-based batch McCormick evaluation |
| T21 OBBT bound tightening | Done | LP-based bound tightening with HiGHS warm-start |
| T22 Differentiable solving (Level 1) | Done | custom_jvp + envelope theorem for parameter sensitivity |
| T17 GPU-batched IPM | Not started | Dense Cholesky, vmap-compatible Rust Ipopt |
| T20 Multi-start heuristics | Not started | vmap over multiple NLP starts |
| T23 Differentiable solving (Level 3) | Not started | Implicit differentiation at active set |
| T24 GPU IPM in solver loop | Not started | Replace cyipopt with GPU-batched IPM |

### Phase 3: Competitive Performance

- Piecewise McCormick + alphaBB relaxations
- Cutting planes (RLT, outer approximation)
- GNN-based branching policy

### Phase 4: Polish + Release

- Documentation + example notebooks
- Release engineering (`pip install discopt`)

## Building

Requires Rust 1.84+, Python 3.10+, and Ipopt.

```bash
# Install Ipopt (macOS)
brew install ipopt

# Build Rust-Python bindings
cd crates/discopt-python && maturin develop && cd ../..

# Run tests
cargo test -p discopt-core
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 pytest python/tests/ -v
```

## License

MIT
