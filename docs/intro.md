# discopt

```{image} discopt-logo.png
:alt: discopt logo
:width: 300px
:align: center
```

A hybrid Mixed-Integer Nonlinear Programming (MINLP) solver combining a Rust backend, JAX automatic differentiation, and Python orchestration. Solves MINLP problems via NLP-based spatial Branch & Bound {cite:p}`Land1960,Belotti2013` with JIT-compiled objective/gradient/Hessian evaluation.

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

**JAX layer** (`python/discopt/_jax`): DAG compiler mapping modeling expressions to JAX primitives, JIT-compiled NLP evaluator (objective, gradient, Hessian, constraint Jacobian), McCormick convex/concave relaxations {cite:p}`McCormick1976` (21 functions including sigmoid, softplus, tanh), and a relaxation compiler with vmap support.

**Solver wrappers** (`python/discopt/solvers`): ripopt (Rust IPM via PyO3), cyipopt NLP wrapper for Ipopt {cite:p}`Wachter2006`, HiGHS LP and MILP wrappers with warm-start support (MILP used by the LOA decomposition solver).

**Neural network embedding** (`python/discopt/nn`): embeds trained feedforward networks as algebraic MINLP constraints {cite:p}`Ceccon2022` via full-space (smooth activations), ReLU big-M MILP {cite:p}`Anderson2020`, and reduced-space strategies; interval arithmetic bound propagation; ONNX model import.

**Generalized disjunctive programming** (`python/discopt/_jax/gdp_reformulate.py`): reformulates GDP models — `BooleanVar`, propositional logic operators, `either_or()`, `if_then()` — into standard MINLP via big-M, multiple big-M (LP-tightened), convex hull, or Logic-based Outer Approximation.

**Orchestrator** (`python/discopt/solver.py`): End-to-end `Model.solve()` connecting all components. At each B&B node: solve continuous NLP relaxation with the interior-point method {cite:p}`Nocedal2006`, prune infeasible nodes, fathom integer-feasible solutions, branch on most fractional variable.

**Certified-global MINLP via AMP** (`python/discopt/solvers/amp.py`): Adaptive Multivariate Partitioning {cite:p}`Nagarajan2019` for nonconvex MINLPs (bilinear, signomial, concave). Iterates a piecewise-McCormick / convex-hull MILP relaxation against an NLP subproblem (Ipopt), refining the partition where the relaxation gap is largest. At every iteration `LB_k <= global_opt <= UB_k`, so termination yields a certified suboptimality bound. Invoked with `Model.solve(solver="amp")`; see {doc}`notebooks/amp_global_minlp`.

**Parameter estimation & MBDoE** (`python/discopt/estimate.py`, `python/discopt/doe/`): Model-based parameter estimation via weighted least-squares NLP, and optimal design of experiments using Fisher Information Matrix analysis {cite:p}`Wang2022,Franceschini2008`. Key advantage: exact sensitivity Jacobians via JAX autodiff (no finite differences). Includes sequential DoE loop, identifiability analysis, and design space exploration with D/A/E/ME-optimality criteria {cite:p}`Atkinson2007`.

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

## Parameter Estimation & Design of Experiments

discopt includes model-based parameter estimation and optimal experimental design,
using exact JAX autodiff for Fisher Information Matrix computation.

```python
from discopt.estimate import Experiment, ExperimentModel, estimate_parameters
from discopt.doe import compute_fim, optimal_experiment, DesignCriterion
import discopt.modeling as dm
import numpy as np

# Define an experiment: y = k * x
class MyExperiment(Experiment):
    def create_model(self, **kwargs):
        m = dm.Model("exp")
        k = m.continuous("k", lb=0.01, ub=20)
        x = m.continuous("x", lb=0.1, ub=10)
        return ExperimentModel(
            model=m,
            unknown_parameters={"k": k},
            design_inputs={"x": x},
            responses={"y": k * x},
            measurement_error={"y": 0.1},
        )

# Estimate k from data
exp = MyExperiment()
data = {"y": 6.0}  # observed at some x
result = estimate_parameters(exp, data)
print(result.parameters)  # {"k": ...}
print(result.confidence_intervals)

# Find optimal measurement location (D-optimal)
design = optimal_experiment(
    exp, param_values={"k": 2.0},
    design_bounds={"x": (0.5, 10.0)},
    criterion=DesignCriterion.D_OPTIMAL,
)
print(design.summary())
```

## NLP Backend Comparison

discopt supports three NLP solver backends, each with different strengths:

| Backend         | Implementation    | Use Case                                   |
|-----------------|-------------------|--------------------------------------------|
| `ipm` (default) | Pure-JAX IPM      | B&B inner loop; GPU-batched via `jax.vmap` |
| `ripopt`        | Rust IPM via PyO3 | Single-problem NLP; fastest wall-clock     |
| `cyipopt`       | Ipopt via cyipopt | Single-problem NLP; most robust            |

```python
result = model.solve(nlp_solver="ripopt")   # Rust IPM
result = model.solve(nlp_solver="ipm")      # Pure-JAX (default)
result = model.solve(nlp_solver="cyipopt")  # Ipopt
```

## Contents

```{tableofcontents}
```
