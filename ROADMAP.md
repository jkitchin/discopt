# Development Roadmap

discopt followed a 4-phase development plan. Phases 1-4 are complete. Phases 5-7 track future extensions.

## Phase 1: Working Solver (complete)

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

## Phase 2: GPU + Differentiability (complete)

| Task                                 | Status | Description                                                |
|--------------------------------------|--------|------------------------------------------------------------|
| T19 Batch relaxation evaluator       | Done   | jax.vmap-based batch McCormick evaluation                  |
| T21 OBBT bound tightening            | Done   | LP-based bound tightening with HiGHS warm-start            |
| T22 Differentiable solving (Level 1) | Done   | custom_jvp + envelope theorem for parameter sensitivity    |
| T20 Multi-start heuristics           | Done   | Multi-start NLP solving + feasibility pump                 |
| T23 Differentiable solving (Level 3) | Done   | Implicit differentiation at active set via KKT             |
| T25 Benchmark runner                 | Done   | Performance metrics, batch scaling, JSON export            |
| T17 GPU-batched IPM                  | Done   | Pure-JAX IPM solver with augmented KKT, vmap batch solving |
| T24 GPU IPM in solver loop           | Done   | Batch IPM in B&B loop, ipm default backend                 |

## Phase 3: Competitive Performance (complete)

| Task                    | Status | Description                                             |
|-------------------------|--------|---------------------------------------------------------|
| Piecewise McCormick     | Done   | k-partition domain splitting, O(1/k^2) convergence      |
| alphaBB underestimators | Done   | Hessian-based convexification (Adjiman/Floudas 1998)    |
| Cutting planes (RLT/OA) | Done   | Bilinear RLT cuts, gradient OA, separation oracles      |
| GNN branching policy    | Done   | Bipartite graph GNN, strong branching data collection   |
| Solver integration      | Done   | partitions, branching_policy, cutting_planes parameters |
|                         |        |                                                         |

## Phase 4: Polish + Release (nearly complete)

| Task                              | Status      | Description                                               |
|-----------------------------------|-------------|-----------------------------------------------------------|
| ripopt integration (PyO3)         | Done        | Rust IPM solver via PyO3 bindings (`nlp_solver="ripopt"`) |
| CUTEst interface                  | Done        | PyCUTEst evaluator for NLP benchmarking                   |
| Documentation + example notebooks | Done        | 42 notebooks, Jupyter Book site builds with zero warnings |
| Release engineering               | Done        | Published on PyPI, maturin build, CI auto-publish on tags |

## Phase 5: Problem Class Coverage

New problem types to make discopt competitive across the full optimization landscape.

| Task                              | Status  | Description                                                                  |
|-----------------------------------|---------|------------------------------------------------------------------------------|
| SOCP support                      | Planned | Second-order cone constraints, conic solver integration (SCS/Clarabel)       |
| Semidefinite programming (SDP)    | Planned | Matrix variables, PSD cone constraint, MOSEK/SCS backend                     |
| Conic programming (general)       | Planned | Exponential cone, power cone for entropy/GP formulations                     |
| Stochastic programming            | Planned | Two-stage/multi-stage recourse, scenario trees, chance constraints, SAA/CVaR |
| Robust optimization               | Done    | Box/ellipsoidal/polyhedral uncertainty sets, adjustable robust counterparts  |
| Multi-objective optimization      | Done    | Weighted sum, AUGMECON2 ε-constraint, weighted Tchebycheff, NBI, NNC via `discopt.mo`; hypervolume / IGD / spread / ε indicators. Evolutionary/Bayesian/interactive methods remain future work. |
| Bilevel optimization              | Planned | KKT reformulation to MPEC, cutting plane methods                             |
| Complementarity problems (MPEC)   | Planned | Scholtes relaxation, penalty methods for equilibrium constraints             |
| Geometric programming             | Planned | Posynomial/signomial programs, log-transformation to convex form             |

## Phase 6: Solver and Algorithm Improvements

| Task                              | Status  | Description                                                                  |
|-----------------------------------|---------|------------------------------------------------------------------------------|
| QP-specific solver                | Planned | Dedicated active-set or IPM QP solver, or OSQP integration                   |
| Benders decomposition             | Planned | Classical and generalized Benders for block-structured MILP/MINLP            |
| Lagrangian relaxation             | Planned | Dantzig-Wolfe, column generation, subgradient/bundle methods                 |
| Global optimization beyond B&B    | Planned | DIRECT, basin-hopping multi-start, interval-based methods                    |
| Convex NLP fast path              | Planned | Detect convex NLPs and solve without B&B, guaranteeing global optimality     |

## Phase 7: Modeling API and Infrastructure

| Task                              | Status  | Description                                                                  |
|-----------------------------------|---------|------------------------------------------------------------------------------|
| Set and index abstractions        | Planned | Named sets, indexed variables/constraints, set algebra for sparse models     |
| Piecewise-linear functions        | Done    | SOS2 constraints in modeling API                                             |
| Native indicator constraints      | Done    | `_IndicatorConstraint` class in modeling API                                 |
| Warm-starting API                 | Done    | `m.solve(initial_solution=...)` with validation                              |
| Export formats (MPS/LP)           | Done    | `to_mps()`, `to_lp()` in `discopt.export`, GAMS writer                      |
| Callback and cut generation API   | Done    | Lazy constraint, incumbent, and node callbacks in B&B                        |
| Infeasibility analysis (IIS)      | Planned | Algorithmic irreducible infeasible subsystem computation                     |
| Pyomo import                      | Done    | `from_pyomo()` converter for Var, Constraint, Objective, GDP constructs      |
| GAMS import                       | Done    | `from_gams()` reader for .gms scalar models                                 |
