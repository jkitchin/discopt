# Development Roadmap

discopt followed a 4-phase development plan. Phases 1-4 are complete. Phases 5-7 track future extensions; many items in 5-7 are already shipped (see status columns).

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
| T9a Rust Ipopt (ripopt)  | Superseded  | Old ripopt crate replaced by T17 pure-JAX IPM and POUNCE |
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

## Phase 4: Polish + Release (complete)

| Task                              | Status      | Description                                               |
|-----------------------------------|-------------|-----------------------------------------------------------|
| POUNCE integration                | Done        | Pure-Rust Ipopt port via Python bindings (`nlp_solver="pounce"`); default single-solve backend, replaced the old ripopt crate |
| CUTEst interface                  | Done        | PyCUTEst evaluator for NLP benchmarking                   |
| Documentation + example notebooks | Done        | 43 notebooks, Jupyter Book site builds with zero warnings |
| Release engineering               | Done        | Published on PyPI, maturin build, CI auto-publish on tags |
| Tiered Python test suite          | Done        | Fast PR-tier + full + integration markers (#69)           |
| Examiner / KKT validator          | Done        | `Model.solve(validate=True)`, dual recovery (#55, #83)    |
| Optimization course + tutor CLI   | Done        | 30-lesson curriculum + `discopt tutor` (#85)              |

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
| QP-specific solver                | Done    | HiGHS QP wrapper + JAX IPM QP path; convex QP fast path                      |
| Benders decomposition             | Planned | Classical and generalized Benders for block-structured MILP/MINLP            |
| Lagrangian relaxation             | Planned | Dantzig-Wolfe, column generation, subgradient/bundle methods                 |
| Global optimization beyond B&B    | Done    | AMP (Adaptive Multivariate Partitioning) global MINLP solver (#23, #86)      |
| Convex NLP fast path              | Done    | SUSPECT-style convexity detector + convex-NLP fast path (#46)                |
| Structural presolve pipeline      | Done    | 22 structural passes wired into the root presolve pipeline (#53)             |
| Convexification roadmap M1-M11    | Done    | M2/M3 arithmetics, M4/M5/M9/M10 root passes, M6 eigenvalue bound (#51)       |
| Deadline-aware JAX IPM            | Done    | Wall-clock `time_limit` honored inside JAX-compiled `while_loop`s (#80)      |

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
| GAMS solver link                  | Done    | `discopt.gams`: run discopt *as* a GAMS solver via the GMO/GEV control-file link |

## Phase 8: POUNCE-Only Solver Stack

Eliminate HiGHS as a runtime dependency so discopt ships as a complete pip-installable
solver on a single in-house engine (POUNCE), with end-to-end differentiability and a
batch-first (CPU multicore; GPU opportunistic) architecture. Detailed plan, phase exit
gates, and risk register: [docs/design/pounce-only-roadmap.md](docs/design/pounce-only-roadmap.md).

| Task                                   | Status      | Description                                                              |
|----------------------------------------|-------------|--------------------------------------------------------------------------|
| P0 POUNCE universal continuous engine  | Done        | Pure-LP path, Farkas-certificate infeasibility, bound trust-gate, HiGHS demoted to CI oracle. Batch NLP node waves live on `pounce-solver` ≥0.5.0. Open: wire batch QP (`solve_qp_batch`); bump pin to `>=0.5` |
| P1 Self-hosted integer B&B             | Done        | MILP/MIQP via Rust B&B + POUNCE relaxations; incumbent purification; root reduced-cost fixing; HiGHS-free in POUNCE-only mode |
| P2 Crossover + root cuts               | Done        | Pure-Rust IPM-to-basis crossover + basis recovery; GMI/MIR cuts wired and correct. Open: c-MIR aggregation + upper-bound complementation; cross-round re-separation |
| P3 Cut and heuristic suite             | Mostly done | Cover/lifted-cover/clique cuts, diving/RINS/local-branching/feasibility-pump, conflict analysis all shipped. Open: flow-cover and implied-bound cuts |
| P4 Retire remaining HiGHS consumers    | Done        | OA/GDP masters, OBBT, McCormick-LP, partition-selection, DOE, RO → selector/POUNCE; `[pounce]`-only install fully functional. End state: HiGHS-optional at runtime (kept as default-when-installed + CI oracle) |
| P5 Parity push + differentiable MILP   | Mostly done | Differentiable MILP/MIQP shipped (KKT implicit diff through fixed-integer relaxation); warm-started Rust simplex closed the MILP gap to ~1–4× HiGHS on knapsack-class. Open: benchmark-gated parity at MIPLIB/MINLPLib scale |

Detailed status, increment-level history, and the remaining open items for each
task live in [docs/design/pounce-only-roadmap.md](docs/design/pounce-only-roadmap.md).
