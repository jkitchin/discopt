# discopt

```{image} discopt-logo.png
:alt: discopt logo
:width: 300px
:align: center
```

A Mixed-Integer Nonlinear Programming (MINLP) solver built on a Rust core with Python orchestration. Solves MINLPs by spatial Branch & Bound {cite:p}`Land1960,Belotti2013` over rigorous convex relaxations, with an in-house primal/dual simplex for the per-node LPs and a Rust automatic-differentiation tape for objective/gradient/Jacobian/Hessian evaluation.

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

**Rust backend** (`crates/discopt-core`): Expression IR, Branch & Bound tree (node pool, branching, pruning), the native spatial B&B kernel, in-house primal/dual simplex with a sparse LU basis (`feral`), .nl file parser, FBBT/presolve (interval arithmetic, probing, Big-M simplification).

**NLP evaluation** (`python/discopt/_tape_nlp_evaluator.py`, `_nl_expr_compiler.py`): objective, gradient, constraints, Jacobian, and Lagrangian Hessian (dense and sparse) from a POUNCE Rust AD tape, with no JAX on the path. Expressions with no tape opcode fall back to the JAX evaluator, and `DISCOPT_NLP_EVAL=jax` selects it wholesale. A **tape-representable** solve does not import JAX -- not on the LP, QP, MIQP and simplex-MILP paths, and not on the nonlinear ones either (measured over eight nonlinear corpus instances: zero `jax` modules in `sys.modules` after each solve). Those eight are all tape-representable, so they do not license the unconditional claim: the fallback lives on the **default** path, and a plain `Model.solve()` of `dm.norm(X, 2)` with a 3x3 `X` and no caller-side `import jax` takes `sys.modules` from 0 to 219 `jax` entries.

**Relaxation layer** (`python/discopt/_relax`): DAG compiler, the uniform factorable relaxation engine, McCormick convex/concave relaxations {cite:p}`McCormick1976` (28 primitive operations including sigmoid, softplus, tanh), alphaBB, piecewise McCormick, cutting planes, and a relaxation compiler. The default uniform factorable engine is **numpy**, not JAX: a nonconvex MINLP whose expressions are tape-representable does not import JAX when solved through `Model.solve()` (measured — `"jax" in sys.modules` is `False` after a spatial-B&B solve to certified optimality). **41 of the 85 modules** in this directory import JAX, 27 of them at module scope — so "a few stragglers" undersells it — but they are off the default path, with one exception that matters: `_relax/nlp_evaluator` is the legacy JAX evaluator the tape falls back to, so a model with no tape opcode reaches it from an ordinary `Model.solve()`. The directory was named `_jax` before the removal, so its name is not a guide to what it runs on. `dm.register_function(name, lower)` lets a caller declare a named univariate composite the engine envelopes as a **single atom** instead of relaxing its expansion term by term — the route a plugin uses to hand the relaxer structure that generic factorable reformulation would throw away.

**Solver wrappers** (`python/discopt/solvers`): POUNCE (pure-Rust Ipopt port) for LP/QP/NLP, the in-house simplex LP/MILP backends, cyipopt for Ipopt {cite:p}`Wachter2006`, AMP, the MIP-NLP decomposition family, GDPopt-LOA, the derivative-free backends, and an optional Gurobi backend. highspy is used only on the OA/GDP paths.

**ML model embedding** (`python/discopt/ml`): embeds trained ML predictors as algebraic MINLP constraints {cite:p}`Ceccon2022` — feedforward networks via full-space (smooth activations), ReLU big-M MILP {cite:p}`Anderson2020`, and reduced-space strategies, and decision trees/ensembles via a per-leaf MILP encoding {cite:p}`Misic2020`; interval arithmetic bound propagation; ONNX, scikit-learn and PyTorch model import. (Imported as `discopt.nn` before 0.9.0; that name still works as a deprecated alias.)

**Generalized disjunctive programming** (`python/discopt/_relax/gdp_reformulate.py`): reformulates GDP models — `BooleanVar`, propositional logic operators, `either_or()`, `if_then()` — into standard MINLP via big-M, multiple big-M (LP-tightened), convex hull, or Logic-based Outer Approximation.

**Piecewise-linear functions** (`python/discopt/modeling/_piecewise.py`): `Model.piecewise(x, breakpoints, values)` declares a tabulated univariate function and lowers it, at declaration, to an exact MILP encoding — incremental {cite:p}`MarkowitzManne1957`, logarithmic {cite:p}`Vielma2011`, disaggregated convex combination, or a native SOS2 set {cite:p}`BealeTomlin1970` — following {cite:t}`Vielma2010`. Because the table *is* the function, the model solved is the model declared and the certificate carries over; an input whose domain escapes the breakpoint span is refused, not clamped. See {doc}`notebooks/piecewise_linear`.

**Modeling surface** (`python/discopt/modeling`): the expression DAG, plus the blocks that let one model stand inside another. `dm.argmin(inner, ...)` embeds an inner NLP as a block of an outer model — its forward pass is a POUNCE solve and its derivatives are sIPOPT KKT sensitivities — with `dm.argmin_kkt` as the lowered arm of the same block. `Model.mcp(F, z, lb=..., ub=...)` declares the box-bounded mixed-complementarity relation (represented, and deliberately **refused** by `solve` rather than lowered through a two-branch encoding; see {doc}`complementarity_semantics`). `dm.solve_batch(models, workers=8)` runs many small independent global solves and returns one `SolveResult` per model in input order. Saved `.dopt` models and result files carry FAIR provenance and a validation report.

**Orchestrator** (`python/discopt/solver.py`): End-to-end `Model.solve()` connecting all components. At each B&B node: solve continuous NLP relaxation with the interior-point method {cite:p}`Nocedal2006`, prune infeasible nodes, fathom integer-feasible solutions, branch on most fractional variable. `solve(warm_start=prev)` now reuses the **dual** side as well as the primal point: on the single-NLP routes whose backend reports them, `SolveResult.kkt` carries the terminal KKT residuals (scaled and `_unscaled`) and the interior-point `barrier_parameter` that seeds the next solve; it is `None` on every branch-and-bound route.

**Certified-global MINLP via AMP** (`python/discopt/solvers/amp.py`): Adaptive Multivariate Partitioning {cite:p}`Nagarajan2019` for nonconvex MINLPs (bilinear, signomial, concave). Iterates a piecewise-McCormick / convex-hull MILP relaxation against an NLP subproblem (Ipopt), refining the partition where the relaxation gap is largest. At every iteration `LB_k <= global_opt <= UB_k`, so termination yields a certified suboptimality bound. Invoked with `Model.solve(solver="amp")`; see {doc}`notebooks/amp_global_minlp`.

**Parameter estimation** (`python/discopt/estimate.py`): Model-based parameter estimation via weighted least-squares NLP with exact sensitivity Jacobians via JAX autodiff (no finite differences), Fisher-Information-based covariance, and confidence intervals {cite:p}`Franceschini2008`. Optimal design of experiments (FIM-based D/A/E-optimal design, sequential DoE, identifiability analysis) lives in the standalone [discopt-doe](https://github.com/jkitchin/discopt-doe) plugin, which shares the same `Experiment` interface.

## Try it without installing anything

discopt runs in the browser — the whole solver, compiled to WebAssembly, with an
editor and a console. See [Run discopt in your browser](browser.md), or go
straight to the solver page:

```{raw} html
<p><a href="wasm/index.html"><strong>&rarr; Open the in-browser solver</strong></a></p>
```

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

## Parameter Estimation

discopt includes model-based parameter estimation using exact JAX autodiff for
Fisher Information Matrix computation. (Optimal experimental design on the same
`Experiment` interface is provided by the
[discopt-doe](https://github.com/jkitchin/discopt-doe) plugin:
`pip install discopt-doe`.)

```python
from discopt.estimate import Experiment, ExperimentModel, estimate_parameters
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
```

## NLP Backend Comparison

`Model.solve()` accepts these `nlp_solver` selectors:

| Backend                  | Implementation                     | Use Case                                    |
|--------------------------|------------------------------------|---------------------------------------------|
| `pounce` (default)       | Pure-Rust Ipopt port               | Universal default: LP/QP/MILP/MIQP/NLP/MINLP |
| `ipopt` / `cyipopt`      | Ipopt via cyipopt                  | NLP node and continuous solves; most robust |
| `simplex`                | Pure-Rust warm-started simplex B&B | MILP; the fully JAX-free MILP path          |
| `ipm` / `sparse_ipm`     | Back-compat aliases                | Simplex-first LP/MILP routing; resolve to POUNCE for NLP/MINLP |

The pure-JAX interior-point method has been retired. `nlp_solver="ipm"` is kept as
an alias so existing scripts keep working: it selects the simplex-first matrix
routing for LP/MILP and resolves to POUNCE for NLP/MINLP. See
{doc}`notebooks/ipm_vs_ipopt` for a measured comparison.

```python
result = model.solve()                       # default: POUNCE
result = model.solve(nlp_solver="pounce")    # POUNCE (pure-Rust Ipopt port)
result = model.solve(nlp_solver="ipopt")     # Ipopt via cyipopt
result = model.solve(nlp_solver="simplex")   # pure-Rust simplex MILP B&B
```

## Contents

```{tableofcontents}
```
