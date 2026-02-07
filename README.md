# discopt

[![CI](https://github.com/jkitchin/discopt/actions/workflows/ci.yml/badge.svg)](https://github.com/jkitchin/discopt/actions/workflows/ci.yml)

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
| JAX DAG Compiler | Complete | 70 Python |
| McCormick Relaxations | Complete | 88 Python |
| Relaxation Compiler | Complete | 33 Python |
| NLP Evaluator (JAX) | Complete | 45 Python |
| HiGHS LP Wrapper | Complete | 20 Python |
| cyipopt NLP Wrapper | Complete | 24 Python |
| Batch Dispatch | Complete | 34 Python |
| Solver Orchestrator | Complete | 32 Python |
| **Total** | | **127 Rust + 403 Python** |

## Quick Start

```python
import jaxminlp_api as jm

m = jm.Model("example")
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
