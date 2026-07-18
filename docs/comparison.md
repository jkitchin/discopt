# Comparison of Optimization Frameworks

This document compares discopt to established optimization frameworks across several dimensions: modeling API, solver architecture, differentiability, extensibility, and target use cases. The goal is to help users understand when discopt is the right tool and when an alternative is better suited.

## Overview

| Framework | Language | Type | Scope | License |
|-----------|----------|------|-------|---------|
| **discopt** | Python/Rust/JAX | Solver + modeling API | MINLP (global) | Open source |
| **Pyomo** | Python | Modeling language | LP–MINLP (via solvers) | BSD |
| **JuMP** | Julia | Modeling language | LP–MINLP (via solvers) | MPL-2.0 |
| **BARON** | GAMS/AMPL | Solver | MINLP (global) | Commercial |
| **SCIP** | C/Python | Solver + modeling API | MILP/MINLP | Apache 2.0 |
| **CVXPY** | Python | Modeling language | Convex (DCP) | Apache 2.0 |
| **Bonmin/Couenne** | C++/AMPL | Solvers | MINLP | EPL |
| **Gurobi** | C/Python/Julia | Solver | LP/QP/MILP/MIQP | Commercial |

## Modeling API Comparison

### discopt

```python
import discopt.modeling as dm

m = dm.Model("blending")
x = m.continuous("flow", shape=(3,), lb=0, ub=100)
y = m.binary("active", shape=(2,))

m.minimize(cost @ x + fixed_cost @ y)
m.subject_to(A @ x <= b, name="mass_balance")
m.subject_to(x[0] * x[1] <= 50 * y[0], name="bilinear")
result = m.solve()
print(result.value(x))
```

discopt uses a lightweight algebraic API where operator overloading builds a lazy expression DAG. Variables are created with `m.continuous()`, `m.binary()`, or `m.integer()`, and constraints are formed by comparing expressions with `<=`, `>=`, or `==`. Mathematical functions live in the `dm` namespace (`dm.exp`, `dm.log`, `dm.sin`, etc.). The DAG is compiled to both JAX-traceable functions (for autodiff) and a Rust IR (for structure detection and B&B tree management).

Indexed summation uses a functional pattern:

```python
dm.sum(lambda i: cost[i] * x[i], over=range(n))
```

The API also supports GDP (generalized disjunctive programming) constraints:

```python
m.if_then(y[0], [x[0] >= 10, x[1] <= 50])
m.either_or([[x[0] >= 5], [x[1] >= 5]])
```

### Pyomo

```python
from pyomo.environ import *

m = ConcreteModel()
m.x = Var(range(3), within=NonNegativeReals, bounds=(0, 100))
m.y = Var(range(2), within=Binary)
m.obj = Objective(expr=sum(cost[i]*m.x[i] for i in range(3))
                       + sum(fixed_cost[j]*m.y[j] for j in range(2)),
                  sense=minimize)
m.capacity = Constraint(expr=sum(A[0,i]*m.x[i] for i in range(3)) <= b[0])
solver = SolverFactory('baron')
result = solver.solve(m)
```

Pyomo provides a full algebraic modeling language in Python with `ConcreteModel`/`AbstractModel`, `Set`, `Param`, `Block`, and `RangeSet` abstractions. It is the most feature-rich Python modeling framework, supporting stochastic programming (PySP), GDP (Pyomo.GDP), DAE systems, and multi-objective optimization. The cost is verbosity and a steeper learning curve.

### JuMP (Julia)

```julia
using JuMP, BARON

m = Model(BARON.Optimizer)
@variable(m, 0 <= x[1:3] <= 100)
@variable(m, y[1:2], Bin)
@objective(m, Min, cost' * x + fixed_cost' * y)
@constraint(m, A * x .<= b)
@NLconstraint(m, x[1] * x[2] <= 50 * y[1])
optimize!(m)
value.(x)
```

JuMP uses Julia macros (`@variable`, `@constraint`, `@objective`) to provide a concise algebraic syntax that reads close to mathematical notation. It interfaces with 80+ solvers through MathOptInterface. JuMP does not include a solver itself; it is purely a modeling layer.

### CVXPY

```python
import cvxpy as cp

x = cp.Variable(3, nonneg=True)
y = cp.Variable(2, boolean=True)
prob = cp.Problem(cp.Minimize(cost @ x + fixed_cost @ y),
                  [A @ x <= b])
prob.solve(solver=cp.GUROBI)
```

CVXPY enforces disciplined convex programming (DCP) rules at model construction time. If an expression violates convexity rules, it raises an error before solving. This gives strong mathematical guarantees but excludes nonconvex problems entirely.

### API Summary

| Feature | discopt | Pyomo | JuMP | CVXPY |
|---------|---------|-------|------|-------|
| Syntax style | Operator overloading | Algebraic components | Julia macros | Operator overloading |
| Verbosity | Low | High | Low | Low |
| Nonlinear support | Full | Full | Full | Convex only (DCP) |
| Array/matrix variables | Native (numpy shapes) | Via indexed sets | Native (Julia arrays) | Native |
| GDP / logical constraints | Built-in | Pyomo.GDP extension | DisjunctiveProgramming.jl | No |
| Parameter sensitivity | Built-in (differentiable) | Limited | No | cvxpylayers (convex) |
| Learning curve | Low | Moderate–High | Low (Julia required) | Low |

## Solver Architecture

This is the fundamental difference. discopt is a **solver** with an integrated modeling API. Pyomo, JuMP, and CVXPY are **modeling languages** that dispatch to external solvers.

### discopt: Integrated Solver Stack

```
Python modeling API
    │
    ├──→ JAX DAG compiler → autodiff (grad, Hessian, Jacobian)
    │        │
    │        ├──→ McCormick / alphaBB / ICNN relaxations
    │        ├──→ Pure-JAX IPM (Mehrotra predictor-corrector)
    │        ├──→ Convexity detection → bypass / per-constraint OA
    │        └──→ vmap batching → parallel node evaluation
    │
    └──→ Rust backend (PyO3)
             ├──→ B&B tree (node pool, branching, pruning)
             ├──→ FBBT / OBBT (bound tightening)
             ├──→ .nl parser (AMPL interop)
             └──→ Expression IR (structure detection)
```

The solver owns the entire pipeline: the expression DAG, derivative computation, convex relaxations, bound tightening, branching, cutting planes, and primal heuristics. This tight integration enables features that are impossible when modeling and solving are separated (differentiable solving, learned relaxations, batch node evaluation).

### Pyomo / JuMP: Modeling Layer + External Solver

```
Modeling API
    │
    └──→ Solver interface (MathOptInterface / SolverFactory)
             │
             ├──→ BARON, SCIP, Gurobi, CPLEX, Ipopt, ...
             └──→ (solver handles everything internally)
```

The modeling layer constructs a problem representation and passes it to an external solver as a black box. The user has no control over (or visibility into) the solver's internal relaxation strategy, branching decisions, or derivative computation.

**Pros of the modeling-layer approach:**
- Access to world-class commercial solvers (Gurobi, CPLEX, BARON)
- Solver-agnostic models — switch solvers by changing one line
- Mature, battle-tested solvers with decades of engineering
- Large user communities and extensive documentation

**Pros of the integrated-solver approach (discopt):**
- Differentiable solving — gradients flow through the solve
- Batch evaluation — vmap across B&B nodes on GPU/TPU
- Composable relaxations — swap McCormick for alphaBB or learned ICNN
- Full visibility — inspect/modify any solver component from Python
- No external dependencies — pure Rust + JAX, no C/Fortran

### BARON: Monolithic Global Solver

BARON is the gold standard for deterministic global optimization of nonconvex MINLPs. It uses range reduction (FBBT/OBBT), convex relaxations (McCormick, alphaBB), and a branch-and-reduce algorithm. BARON is a closed-source commercial solver accessed through GAMS or AMPL.

Key differences from discopt:
- BARON uses Fortran/C internals; discopt uses Rust + JAX
- BARON has 25+ years of algorithmic refinement; discopt is newer
- BARON provides no differentiability; discopt provides implicit KKT differentiation
- BARON runs single-threaded on CPU; discopt can batch-evaluate on GPU via vmap
- BARON requires a commercial license; discopt is open source

### SCIP: Open-Source Constraint Integer Programming

SCIP is the strongest open-source solver for MILP and increasingly MINLP. It provides a plugin architecture for adding custom branching rules, cutting planes, and heuristics. SCIP is written in C with Python bindings (PySCIPOpt).

Key differences from discopt:
- SCIP's plugin system requires C/C++ for performance-critical components
- SCIP has no autodiff — uses finite differences or user-supplied derivatives for NLP
- SCIP has a much larger community and broader validation
- discopt's relaxation framework is more compositional (McCormick + alphaBB + piecewise + ICNN)

## Feature Comparison Matrix

| Feature | discopt | Pyomo+BARON | JuMP+BARON | SCIP | CVXPY | Gurobi |
|---------|---------|-------------|------------|------|-------|--------|
| **Problem classes** |
| LP/MILP | Yes | Yes | Yes | Yes | Yes | Yes |
| QP/MIQP | Yes | Yes | Yes | Yes | Yes (convex) | Yes |
| NLP | Yes | Yes | Yes | Yes | Yes (convex) | No |
| Nonconvex MINLP | Yes | Yes | Yes | Yes | No | No |
| GDP | Yes | Yes (Pyomo.GDP) | Yes (ext) | No | No | Indicator ctrs |
| **Solver features** |
| Global optimality | Yes | Yes | Yes | Partial | Yes (convex) | Yes (convex) |
| Differentiable solve | Yes | No | No | No | Yes (convex) | No |
| GPU/TPU acceleration | Yes (JAX) | No | No | No | No | No |
| Batch node evaluation | Yes (vmap) | No | No | No | N/A | No |
| Automatic derivatives | Yes (JAX AD) | Solver-dependent | Solver-dependent | Finite diff | N/A | N/A |
| McCormick relaxations | 19+ operations | BARON internal | BARON internal | Limited | N/A | N/A |
| alphaBB relaxations | Yes | BARON internal | BARON internal | No | N/A | N/A |
| Learned relaxations (ICNN) | Yes | No | No | No | No | No |
| Cutting planes (OA/RLT) | Yes | Solver-dependent | Solver-dependent | Yes | N/A | Yes |
| FBBT/OBBT | Yes (Rust+Python) | BARON internal | BARON internal | Yes | N/A | Yes |
| Warm-start NLP | Yes | Solver-dependent | Solver-dependent | N/A | N/A | Yes |
| **Ecosystem** |
| Solver-agnostic models | No (integrated) | Yes (80+ solvers) | Yes (80+ solvers) | No | Yes (15+ solvers) | No |
| .nl file import | Yes | Yes | Yes | Yes | No | No |
| LLM integration | Yes | No | No | No | No | No |
| Streaming solve updates | Yes | Limited | No | Callbacks | No | Callbacks |
| Sensitivity analysis | Built-in (KKT) | Via suffixes | Via duals | Via duals | Via layers | Via duals |
| **Practical** |
| License | Open source | Commercial (BARON) | Commercial (BARON) | Apache 2.0 | Apache 2.0 | Commercial |
| Language | Python | Python | Julia | C/Python | Python | Multi-language |
| External C/Fortran deps | None | Yes (BARON) | Yes (BARON) | Yes (SCIP) | Solver-dependent | Yes |
| Maturity | Research | Production | Production | Production | Production | Production |

## Sweet Spots: When to Use What

### Use discopt when:

- **You need gradients through the solve.** If the optimization problem is a layer inside a larger differentiable pipeline (decision-focused learning, predict-then-optimize, bilevel optimization), discopt is the only framework that provides exact gradients through nonconvex MINLP via implicit KKT differentiation. CVXPY + cvxpylayers handles the convex case; discopt extends this to nonconvex problems.

- **You want to batch-evaluate B&B nodes on GPU.** JAX's `vmap` lets discopt evaluate NLP relaxations at many B&B nodes simultaneously. This matters when the NLP subproblem is the bottleneck (large nonlinear models) and you have GPU/TPU hardware available.

- **You want a self-contained, extensible solver.** Every component (relaxations, branching, cutting planes, IPM) is accessible from Python. You can swap McCormick for alphaBB, add custom cutting planes, or experiment with branching rules without writing C code. This makes discopt well-suited for optimization research.

- **You want zero C/Fortran dependencies.** The solver stack is pure Rust + JAX + Python. No BLAS/LAPACK linking issues, no Fortran compiler needed. This simplifies deployment in containers, CI, and teaching environments.

- **You're solving small-to-medium nonconvex MINLPs** (up to a few hundred variables) where global optimality matters and you want an open-source solution.

### Use Pyomo when:

- **You need the broadest modeling expressiveness.** Pyomo supports stochastic programming, DAE systems, GDP, bilevel optimization, and multi-objective optimization through extensions. No other Python framework matches its breadth.

- **You need to switch between solvers.** Pyomo's `SolverFactory` lets you try BARON, SCIP, Gurobi, Ipopt, and others without changing the model. This is invaluable for benchmarking and when you need commercial solver performance.

- **You have a large team with existing Pyomo models.** Migration costs are real. Pyomo has extensive documentation, textbooks, and a large user base.

- **You're working on structured problems** (blocks, scenarios, decomposition) where Pyomo's `Block` and `Suffix` abstractions are needed.

### Use JuMP when:

- **You want the most concise algebraic syntax.** JuMP's macro-based DSL (`@variable`, `@constraint`, `@objective`) is arguably the cleanest algebraic modeling syntax available. It reads almost like LaTeX.

- **You need Julia-ecosystem integration.** If your pipeline is in Julia (DifferentialEquations.jl, Flux.jl, etc.), JuMP is the natural choice. The MathOptInterface provides type-stable, zero-overhead solver access.

- **You need access to 80+ solver backends.** JuMP has the largest solver ecosystem, including BARON, Gurobi, CPLEX, HiGHS, Ipopt, SCIP, and many specialized solvers.

- **Performance of the modeling layer matters.** Julia's compiled execution means model construction is faster than Python-based alternatives for very large models (100K+ variables).

### Use BARON when:

- **Deterministic global optimality on hard MINLPs is the priority.** BARON has 25+ years of algorithmic refinement and consistently wins on hard benchmark instances (MINLPLib). For production global optimization, it remains the gold standard.

- **You have a commercial license and need reliability.** BARON's range reduction, relaxation tightening, and branching heuristics are the result of decades of research. It handles numerical difficulties that younger solvers struggle with.

- **Your problem is medium-to-large** (hundreds to thousands of variables) and you need the tightest possible bounds and fastest convergence to global optimality.

### Use SCIP when:

- **You want the strongest open-source MILP/MINLP solver.** SCIP is the leading open-source solver for mixed-integer programming and increasingly handles nonlinear constraints.

- **You need a plugin architecture in C/C++.** SCIP's design allows custom branching rules, constraint handlers, separators, and heuristics at the C level, giving maximum performance for solver extensions.

- **You're doing academic research on solver algorithms.** SCIP's open codebase and plugin system make it the standard platform for publishing new MILP/MINLP algorithms.

### Use CVXPY when:

- **Your problem is convex.** CVXPY's DCP rules catch modeling errors at construction time, guaranteeing that the problem is solvable to global optimality. If your problem fits within DCP, CVXPY is the simplest and safest choice.

- **You need differentiable convex optimization layers.** cvxpylayers provides well-tested, efficient differentiation through convex programs. For convex problems, this is more mature than discopt's differentiable solve.

- **You want the fastest path from math to code.** CVXPY's API is minimal and intuitive. A convex problem can be formulated and solved in 5 lines.

### Use Gurobi when:

- **You're solving LP, QP, MILP, or MIQP at scale.** Gurobi is the fastest commercial solver for these problem classes, routinely handling models with millions of variables and constraints.

- **You need enterprise-grade support and reliability.** Gurobi provides commercial support, deterministic behavior, and extensive tuning parameters.

- **Your problem is linear or quadratic.** If your problem doesn't have general nonlinear constraints, Gurobi (or CPLEX) will outperform any MINLP solver by orders of magnitude.

## Summary Table: Problem Class to Recommended Tool

| Problem Class | Best Choice | Runner-Up | Notes |
|---------------|-------------|-----------|-------|
| LP/MILP (large scale) | **Gurobi** / CPLEX | HiGHS (open source) | Commercial solvers dominate |
| QP/MIQP | **Gurobi** | CPLEX, SCIP | Gurobi's barrier solver is fastest |
| Convex NLP | **CVXPY** + MOSEK | Pyomo + Ipopt | DCP guarantees + conic solvers |
| Nonconvex NLP | **Pyomo** + BARON | discopt, SCIP | BARON for hard instances |
| MINLP (global) | **BARON** | SCIP, discopt | BARON is gold standard |
| MINLP + ML pipeline | **discopt** | — | Only option for differentiable MINLP |
| Differentiable optimization | **discopt** (nonconvex), **cvxpylayers** (convex) | — | Depends on convexity |
| GPU-batched optimization | **discopt** | MPAX (LP/QP only) | JAX vmap for parallel evaluation |
| Research / extensible solver | **discopt** or **SCIP** | — | Python-accessible vs. C plugins |
| Teaching / prototyping | **CVXPY** (convex), **discopt** (nonconvex) | JuMP | Simplest APIs |
