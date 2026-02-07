# discopt: Reconciled Development Plan v2

**Date:** 2026-02-07
**Status:** Authoritative (supersedes all prior planning documents)
**Supersedes:**
- `archive/jaxminlp_development_plan.md` (original work stream plan)
- `archive/feasibility_assessment.md` (assessment and recommendations)
- `archive/JAX_OPTIMIZATION_ECOSYSTEM_VISION.md` (long-term vision — retained as aspirational reference only)

---

## 0. Strategic Decisions

This plan resolves the contradictions identified by the architecture, modularity, and coherence reviews. The following decisions are final:

### 0.1 Name: `discopt`

The project is named **discopt** (discrete optimization). All code, packages, imports, documentation, and configuration use this name. The previous names (`JaxMINLP`, `jaxminlp`, `jax-minlp`) are retired. The ecosystem vision names (`jax-lp`, `jax-optcore`, etc.) are aspirational only and not part of this plan.

- GitHub organization: `discopt-org`
- Python package: `discopt`
- PyPI: `discopt`
- Import: `import discopt`

**Repository structure (Option C — hybrid):**

Two repos under `github.com/discopt-org/`:

```
discopt-org/ipopt-rs          ← EPL-2.0, standalone Rust NLP solver
  ├── Cargo.toml
  ├── LICENSE-EPL-2.0
  ├── NOTICE                  (attribution to COIN-OR Ipopt project)
  ├── src/
  │   ├── lib.rs
  │   ├── ipm.rs              (filter line search IPM loop)
  │   ├── kkt.rs              (KKT system assembly + factorization via faer)
  │   ├── filter.rs           (filter acceptance criteria)
  │   ├── restoration.rs      (feasibility restoration phase)
  │   ├── problem.rs          (NlpProblem trait — the public interface)
  │   ├── options.rs          (solver configuration)
  │   └── result.rs           (solution struct: x, multipliers, status)
  └── tests/

discopt-org/discopt           ← MIT/Apache-2.0, monorepo for everything else
  ├── Cargo.toml              (workspace, depends on ipopt-rs)
  ├── LICENSE-MIT
  ├── LICENSE-APACHE
  ├── pyproject.toml           (maturin build-backend)
  ├── crates/
  │   ├── discopt-core/        (Rust: B&B engine, expression IR, presolve)
  │   └── discopt-python/      (Rust: PyO3 bindings, wraps discopt-core + ipopt-rs)
  ├── python/discopt/          (Python/JAX: modeling API, DAG compiler, McCormick,
  │   ├── __init__.py            differentiable solving, batch evaluator)
  │   ├── _jax/
  │   ├── solvers/
  │   └── ...
  ├── tests/
  ├── benchmarks/
  └── docs/
```

**Why two repos, not one:**
1. **License clarity.** `ipopt-rs` is EPL-2.0 (derivative of Ipopt). Everything else is MIT/Apache-2.0. Separate repos make this unambiguous — one license per repo, no confusion for contributors.
2. **Standalone value.** `ipopt-rs` is useful to the Rust ecosystem independent of discopt. A standalone crate on crates.io attracts contributors and users who don't need MINLP.
3. **Loose coupling.** `ipopt-rs` exposes a `NlpProblem` trait + `solve()`. `discopt-core` implements that trait. This is a clean crate dependency, not interleaved code.

**Development workflow:** During development, `discopt/Cargo.toml` uses a path dependency:
```toml
[workspace.dependencies]
ipopt-rs = { path = "../ipopt-rs" }   # local during development
# ipopt-rs = "0.1"                    # crates.io for releases
```
Clone both repos side by side. Changes to `ipopt-rs` are immediately visible in discopt without publishing. No git submodules needed — Cargo path dependencies are the right mechanism.

### 0.2 Build-vs-Buy: Translate Ipopt to Rust; Use HiGHS for LP

**Decision: Hybrid strategy.** Phase 1 uses HiGHS (LP) as external scaffolding and translates Ipopt's core IPM algorithm from C++ to Rust. This is neither "build from scratch" (too risky) nor "wrap external" (creates a Phase 2 dead end). Translation preserves Ipopt's 20+ years of numerical engineering while producing a codebase we own and can evolve for batching/GPU.

| Component | Phase 1 (early) | Phase 1 (late) | Phase 2+ |
|-----------|----------------|----------------|----------|
| LP solver | HiGHS (via highspy, MIT) | HiGHS | Custom JAX GPU LP (dense Cholesky) |
| NLP solver | cyipopt (scaffolding) | Rust Ipopt (`ipopt-rs`) | Rust Ipopt adapted for vmap/GPU |
| Sparse LA | Ipopt's MUMPS | faer (Rust, MIT) | Dense JAX Cholesky + faer |
| B&B engine | Custom Rust (core deliverable) | Same | Same, enhanced |
| DAG compiler | Custom JAX (core deliverable) | Same | Same |
| McCormick | Custom JAX (core deliverable) | Same | Same, enhanced |

**Rationale:** The original plan deferred NLP solving entirely to Phase 2 as a from-scratch JAX GPU IPM. But building a correct IPM is one of the hardest tasks in the plan — Ipopt handles hundreds of edge cases (degenerate Jacobians, rank-deficient KKT systems, cycling detection, tiny step recovery) accumulated over two decades. A from-scratch implementation would rediscover these failures one benchmark at a time, creating a Phase 2 risk cliff.

Translation is the lowest-risk path: each Rust function can be validated against C++ Ipopt's output on the same problem, providing regression tests by construction. Once the core algorithm is in Rust with explicit state (no global variables, no hidden allocation), refactoring for batch execution (struct-of-arrays layout, parameterized problem data) becomes a tractable Phase 2 task rather than a greenfield build.

**Phase 1 scaffolding**: cyipopt is used for the first end-to-end solve (T14, ~Month 5-6) while the Rust translation proceeds in parallel. By late Phase 1, the Rust Ipopt replaces cyipopt, giving us a solver with no external NLP dependency before the Phase 1 gate.

**LLM-assisted translation**: The structured, algorithmic nature of Ipopt's code (numerical linear algebra, iterative loops, well-defined interfaces) is particularly amenable to LLM-assisted C++-to-Rust translation, making this approach more feasible than it would have been even two years ago.

### 0.2a Ipopt-to-Rust Licensing: EPL-2.0 Boundary

Ipopt is licensed under the Eclipse Public License 2.0. A line-by-line translation to Rust is almost certainly a derivative work, requiring the Rust translation to also be distributed under EPL-2.0.

**Decision: Dual-repo licensing.** The Rust Ipopt translation lives in its own repository (`discopt-org/ipopt-rs`) under EPL-2.0. The main discopt repository (`discopt-org/discopt`) is MIT/Apache-2.0 and depends on `ipopt-rs` as a Cargo crate dependency. EPL-2.0 explicitly permits this — it is not a "viral" license like GPL. See Section 0.1 for the full repository structure.

**Alternative paths considered:**
- **(a) Clean-room reimplementation**: Study Ipopt's algorithms from published papers (Wächter & Biegler 2006), then reimplement from mathematical description without reference to source code. Produces fully MIT-licensable code but loses edge-case hardening. Higher risk. Could be done later as a clean-room replacement for `ipopt-rs` if EPL becomes a problem.
- **(b) Upstream relicensing**: Contact COIN-OR / Ipopt maintainers about adding MIT/Apache as a secondary license under EPL-2.0 Section 3. Worth attempting but unlikely to succeed quickly for a multi-contributor codebase.
- **(c) Functional API boundary**: If the translation produces sufficiently different internal architecture (different data structures, memory layout, linear algebra backend), it may qualify as an independent implementation. Legal gray area.

**Recommendation**: Start with option (a) dual-module licensing. Pursue (b) upstream conversation in parallel. Consider (c) clean-room only if EPL becomes a real barrier to adoption.

### 0.3 LLM Scope: Cut from Phase 1-3

**Decision: WS8 (LLM Advisory Layer) is removed as a work stream.** No LLM features (`from_description()`, `chat()`, RAG, configuration advisor, reformulation advisor) are planned until Phase 4 at the earliest. The LLM engineer role is eliminated from the team allocation.

**Rationale:** The solver does not exist yet. LLM features target a user base (non-expert modelers) the project cannot serve until the solver is mature. Cutting WS8 frees one FTE for core solver work.

### 0.4 Differentiable Solving: Add as Explicit Work Stream

**Decision: Add WS-D (Differentiable Solving) as a new Phase 2 work stream.** Level 1 differentiability (`custom_jvp` on LP relaxation sensitivity) is a core value proposition and must have implementation tasks.

### 0.5 Phase Gate Targets: Revised to Be Achievable

**Decision: Soften BARON comparison targets.** Revise Phase 1 gates for external-library strategy. Revise Phase 2 root gap target. Revise Phase 4 general BARON target.

### 0.6 Test Ownership: Each Module Owns Its Tests

**Decision: Test files are deliverables of their respective work streams, not WS9.** WS9 (CI/CD) owns pipeline configuration, coverage enforcement, and marker-based selection only.

---

## 1. Revised Work Stream Overview

| # | Stream | Phase | Critical Path? | Primary Gate Criteria |
|---|--------|-------|----------------|----------------------|
| WS0 | Architectural Spike | 0 (Month 1) | **YES** | GPU batch latency < 100μs |
| WS1 | Rust Infrastructure & Expression IR | 1 | **YES** | Crate builds, model converts, `.nl` parses |
| WS2 | JAX DAG Compiler & McCormick | 1 | **YES** | Soundness invariant holds |
| WS3a | HiGHS Integration + cyipopt Scaffolding | 1 (early) | **YES** | LP+NLP solving works end-to-end |
| WS3b | Ipopt-to-Rust Translation | 1 | **YES** | Rust Ipopt matches C++ Ipopt on all benchmarks |
| WS5 | B&B Engine & Solver Orchestration | 1 | **YES** | ≥25 solved, zero incorrect |
| WS9 | CI/CD Pipeline | 1-4 | No | CI green, ≥85% coverage |
| WS7 | Bound Tightening & Preprocessing | 1-2 | No | Root gap ≤2.0x BARON |
| WS4 | Rust Ipopt → GPU/Batch Adaptation | 2 | No | vmappable, replaces cyipopt |
| WS6 | GPU Batching & Performance | 2 | No | ≥15x GPU speedup |
| WS-D | Differentiable Solving | 2 | No | Level 1 `custom_jvp` works |
| WS10 | Advanced Algorithms & Release | 3-4 | No | ≤3.0x BARON general, ≤1.0x GPU classes |

**Removed:** WS8 (LLM Advisory Layer) — deferred indefinitely.

**Split:** WS3 into WS3a (HiGHS + cyipopt scaffolding) and WS3b (Ipopt-to-Rust translation).

**Redefined:** WS4 was "Custom JAX GPU IPM from scratch"; now "Rust Ipopt → GPU/Batch Adaptation" (evolve translated code rather than build from scratch).

**Added:** WS0 (Architectural Spike), WS-D (Differentiable Solving).

---

## 2. Revised Dependency Graph

```
PHASE 0 + FOUNDATION (Month 1, parallel):
  WS0 (Architectural Spike) ──────────────────────────────────────► GO/NO-GO
  T1  (Rust scaffold) ─► WS3b (Ipopt-to-Rust translation) ──────► Rust Ipopt working
                                                                       │
PHASE 1 (Months 1-10):                                                │
  WS1 (Rust IR + PyO3)  ────────────────────┐                         │
  WS2 (JAX compiler + McCormick) ───────────┤                         │
  WS3a (HiGHS + cyipopt scaffolding) ──────┼──► WS5 (B&B + orch.) ──┼──► Phase 1 Gate
  WS9 (CI/CD core) ────────────────────────►│                         │
  WS7-a (FBBT, basic presolve) ────────────┘                         │
                                                                       │
                                          T9c: cyipopt → Rust Ipopt ──┘
PHASE 2 (Months 10-20):
  WS4 (Adapt Rust Ipopt → GPU/batch) ────────┐
  WS6 (GPU batching) ───────────────────────┤
  WS-D (Differentiable solving, Level 1+3) ──┼──► Phase 2 Gate
  WS7-b (OBBT, advanced presolve) ───────────┘

PHASE 3-4 (Months 20-42):
  WS10 (Advanced algorithms + release) ────────► Phase 3 Gate → Phase 4 Gate
```

**Critical path:** WS0 + WS3b (parallel, Month 1) → WS1+WS2 (parallel) → WS5 → Phase 1 Gate → WS4+WS6 → Phase 2 Gate → WS10

**Starting sequence:** WS0 (architectural spike) and WS3b (Ipopt-to-Rust translation) launch simultaneously in Month 1. They have zero dependency on each other — WS0 validates Rust↔JAX GPU latency while WS3b translates the core IPM algorithm. This front-loads the two highest-value validation tasks: "can we batch on GPU?" and "do we have a correct NLP solver?"

**Key changes from original plan:**
- WS3b (Ipopt-to-Rust) is one of the first things done — starts Month 1 alongside WS0. The translation is well-defined (C++ source is the spec), self-contained, and produces the foundation everything else builds on. cyipopt is used for the first end-to-end solve (T14, ~Month 5-6) while translation completes, then replaced before the Phase 1 gate.
- WS4 is no longer "build custom JAX GPU IPM from scratch" — it adapts the Rust Ipopt for GPU batching and vmappability. This is a lower-risk evolution of known-good code rather than greenfield development.

---

## 3. Work Stream Specifications

### WS0: Architectural Spike (Month 1, 2-4 weeks)

**Purpose:** Validate the Rust↔JAX GPU batch evaluation thesis before committing resources.

**Delivers:**
- Minimal Rust crate with PyO3 binding that exports a batch of float64 arrays
- Minimal JAX function that receives the batch, applies `jax.vmap` over a simple computation (e.g., quadratic evaluation), and returns results
- Latency measurements at batch sizes 1, 32, 64, 128, 256, 512, 1024
- Array sizes: 10, 50, 100 variables (matching MINLPLib problem sizes)
- GPU vs CPU comparison for the batched computation

**Verification:**
- Round-trip latency < 100μs for batch ≥ 32 (the plan's latency budget)
- GPU batch of 512 evaluations ≥ 10x faster than serial CPU (validates GPU thesis)
- Zero-copy verified: Python array pointer matches Rust buffer pointer
- JIT recompilation count = 0 after warmup

**Test file:** `test_spike.py` (can be discarded after spike)

**Go/No-Go:** If latency exceeds 1ms or GPU provides < 3x speedup at batch 512, the architecture requires redesign before proceeding.

---

### WS1: Rust Infrastructure & Expression IR (Months 1-3)

**Delivers:**
- Two repositories: `discopt-org/ipopt-rs` (Rust crate, EPL-2.0) and `discopt-org/discopt` (monorepo, MIT/Apache-2.0)
- Cargo workspace in `discopt`: `discopt-core` (pure Rust lib) + `discopt-python` (PyO3 bindings), with `ipopt-rs` as a path dependency
- `pyproject.toml` with maturin build-backend; `maturin develop` produces `discopt._rust`
- Expression graph IR in Rust (`ExprNode` enum arena-allocated DAG) mirroring Python DAG types from `core.py` (lines 60-400)
- `impl From<PyModel> for ModelRepr` — walks Python expression tree, builds Rust arena
- Structure detection: `is_linear()`, `is_quadratic()`, `is_bilinear()`, `is_convex()`
- `.nl` file parser outputting `ModelRepr` directly
- `.pyi` type stub for mypy compatibility

**Verification:**
- `maturin develop` succeeds on macOS ARM64 and Linux x86_64
- `python -c "from discopt._rust import version"` works
- All 7 examples from `examples.py` convert to `ModelRepr` without panic
- All 24 MINLPLib instances from `KNOWN_OPTIMA` parse from `.nl` without error
- `is_linear("2*x + 3*y")` → true; `is_linear("x*y")` → false
- Round-trip: Rust evaluates parsed expression at random point, matches Python within 1e-14
- `cargo test && cargo clippy -- -D warnings` clean

**Test file:** `test_rust_ir.py` (owned by WS1, not WS9)
- Expression construction and conversion
- `.nl` parsing for all 24 known-optimum instances
- Structure detection (parametrized over expression types)
- Round-trip evaluation accuracy
- PyO3 binding availability and version

**Mock provided:** `MockModelRepr` (Python-side stub of Rust IR for WS2/WS5 development before WS1 completes)

---

### WS2: JAX DAG Compiler & McCormick Relaxations (Months 1-5)

**Delivers:**
- `discopt/_jax/dag_compiler.py`: walks `Expression` tree → pure `jax.numpy` callable (JIT-compatible)
- `discopt/_jax/mccormick.py`: relaxation primitives for all operations:
  - Bilinear products (standard McCormick envelopes)
  - Univariate convex: exp, x², sqrt (x≥0)
  - Univariate concave: log, log2, log10
  - Univariate nonconvex: sin, cos, tan, abs
  - Composite: sign, min, max, sum, prod
- `discopt/_jax/relaxation_compiler.py`: `compile_relaxation(expr, variables)` → function `(lb_vec, ub_vec) → (relax_lower, relax_upper)` that is `jax.jit` + `jax.vmap` compatible

**Verification:**
- **Soundness invariant (non-negotiable):** at 10,000 random points within bounds, `relaxation_lower ≤ true_value` and `relaxation_upper ≥ true_value` for every relaxation rule. Tolerance: 1e-10.
- `jax.make_jaxpr(compiled_fn)(x)` succeeds for all 7 example objectives (proves traceability)
- `jax.grad(compiled_fn)(x)` returns finite values at 100 random interior points
- `jax.jit(jax.vmap(compiled_relaxation))` processes batch of 128 bound vectors in one call
- As bounds tighten, relaxation gap decreases monotonically (5 progressive steps, 10 expressions)
- All 14 `FunctionCall` types handled

**Test file:** `test_relaxation.py` (owned by WS2)
- One parametrized test per operation type for McCormick soundness
- Gradient correctness vs finite differences (per operation)
- JIT traceability for all example expressions
- vmap batch correctness
- Monotonic gap convergence

**Mock provided:** `MockRelaxationEvaluator` (returns constant bounds for WS5 B&B testing before WS2 completes)

---

### WS3a: HiGHS Integration + cyipopt Scaffolding (Months 2-5)

**Purpose:** Provide LP and NLP solving capability quickly using external libraries. cyipopt is temporary scaffolding — replaced by Rust Ipopt (WS3b) before Phase 1 gate.

**Delivers:**
- `discopt/solvers/lp_highs.py`: wrapper around `highspy` providing:
  - `solve_lp(c, A_ub, b_ub, A_eq, b_eq, bounds) → LPResult`
  - `solve_lp_warm(problem, basis) → LPResult` (warm-start for B&B node re-solves)
  - Status mapping: HiGHS status → discopt status enum
- `discopt/solvers/nlp_ipopt.py`: wrapper around `cyipopt` providing:
  - `solve_nlp(evaluator, x0, bounds, constraints) → NLPResult`
  - Accepts JAX-compiled gradient/Hessian callbacks from WS2's DAG compiler
  - Handles MUMPS linear solver configuration
  - **NOTE:** This file is temporary scaffolding replaced by `nlp_ipopt_rs.py` (WS3b)
- `discopt/_jax/nlp_evaluator.py`: JAX-side NLP evaluation layer:
  - `NLPEvaluator` class providing JIT-compiled `evaluate_objective(x)`, `evaluate_gradient(x)` via `jax.grad`, `evaluate_hessian(x)` via `jax.hessian`, `evaluate_constraints(x)`, `evaluate_jacobian(x)`
  - Callbacks compatible with both cyipopt and Rust Ipopt interfaces
- Solver protocol: `SolverBackend` ABC that HiGHS wrapper, cyipopt wrapper, and Rust Ipopt all implement

**Verification:**
- HiGHS solves all 7 Netlib instances in `conftest.py:82-96` within `abs_tol=1e-6`
- Ipopt + JAX callbacks converges on all 7 example model NLP relaxations
- Warm-start LP re-solve is faster than cold-start (measured, not just asserted)
- Infeasibility and unboundedness correctly detected and reported
- Dimension mismatch → Python `ValueError`
- `NLPEvaluator` gradient matches finite differences at 50 random points (atol=1e-6)
- `NLPEvaluator` Hessian matches finite differences at 20 random points (atol=1e-4)
- Second call to `evaluate_objective` ≥ 10x faster than first (JIT warmup)

**Test files:** (owned by WS3a)
- `test_lp_highs.py`: HiGHS wrapper correctness, warm-start, error handling
- `test_nlp_ipopt.py`: Ipopt wrapper + JAX callbacks, convergence (also serves as regression baseline for WS3b)
- `test_nlp_evaluator.py`: JAX evaluation layer (gradient/Hessian accuracy, JIT behavior)

**Mock provided:** `MockLPSolver` and `MockNLPSolver` (return known solutions for WS5 B&B testing, WS7 OBBT testing)

---

### WS3b: Ipopt-to-Rust Translation (Months 2-8)

**Purpose:** Translate Ipopt's core interior point algorithm from C++ to Rust, producing a memory-safe NLP solver that we own and can evolve for GPU batching in Phase 2. This is the most strategically important early investment — it eliminates the external NLP dependency, de-risks Phase 2, and produces a community-valuable artifact.

**Why this should be one of the first things we do:**
1. **It is the critical enabler for Phase 2.** WS4 (GPU/batch adaptation) cannot proceed until we have a Rust IPM codebase to adapt. Starting the translation early means WS4 can begin immediately after Phase 1 instead of waiting for a from-scratch implementation.
2. **It runs in parallel with B&B development.** The translation is independent of WS1 (expression IR), WS2 (McCormick), and WS5 (B&B engine). It only needs to integrate once via the `SolverBackend` protocol.
3. **cyipopt is a build headache.** cyipopt depends on C/Fortran Ipopt + MUMPS + BLAS/LAPACK, creating installation friction. A Rust crate with `cargo build` is dramatically simpler.
4. **The translation provides its own regression test suite.** Every translated function is validated against C++ Ipopt's output on the same problem — correctness by construction.
5. **LLM-assisted translation is practical now.** Ipopt's code is structured, algorithmic, and well-documented — ideal for LLM-assisted C++-to-Rust translation with human review.

**Scope:** Translate Ipopt's core algorithm, not the entire C++ codebase. Specifically:

**In scope (translate):**
- Filter line search IPM algorithm (the main solver loop)
- KKT system formation and normal equation reduction
- Inertia correction (eigenvalue-based regularization)
- Filter acceptance criteria (Wächter & Biegler 2005)
- Restoration phase (feasibility restoration)
- Warm-starting from previous solution
- Convergence and termination criteria
- NLP problem interface (objective, gradient, Jacobian, Hessian evaluation callbacks)

**Out of scope (not translated):**
- MUMPS/MA27/MA57 linear solvers → use faer (Rust-native, MIT-licensed) for dense solves, or interface to system BLAS via faer
- AMPL `.nl` file reader → already handled by WS1 (T3)
- IP options parsing infrastructure → Rust config struct
- HSL linear solvers (separate proprietary license)

**Delivers:**
- `discopt-org/ipopt-rs` — standalone Rust crate in its own repository (EPL-2.0 licensed)
  - `src/ipm.rs`: Core IPM iteration (filter line search)
  - `src/kkt.rs`: KKT system assembly and factorization via faer
  - `src/filter.rs`: Filter acceptance criteria
  - `src/restoration.rs`: Feasibility restoration phase
  - `src/options.rs`: Solver options (convergence tolerances, iteration limits)
  - `src/problem.rs`: NLP problem trait (objective, gradient, Jacobian, Hessian callbacks)
  - `src/result.rs`: Solution struct (x, multipliers, status)
- In the `discopt-org/discopt` monorepo:
  - `crates/discopt-python/src/nlp_ipopt_rs.rs`: PyO3 bindings exposing `ipopt-rs` to Python
  - `python/discopt/solvers/nlp_ipopt_rs.py`: Python wrapper implementing `SolverBackend`

**Verification (translation correctness):**
- **Output matching:** Rust Ipopt matches C++ Ipopt solution on all 7 NLP benchmark problems within `rel_tol=1e-6` (objective value, variable values, multipliers)
- **Output matching:** Matches on all 7 Netlib LP instances (LP is a special case of NLP) within `abs_tol=1e-6`
- **Iteration matching:** On well-conditioned problems, Rust and C++ Ipopt take the same number of iterations (validates algorithm fidelity)
- **Failure mode handling:** Correctly detects and reports infeasibility (3 infeasible problems), unboundedness (2 problems), and iteration limit (1 problem with tight limit)
- **Warm-start:** Re-solve after parameter perturbation converges in fewer iterations than cold start
- **Drop-in replacement:** Replacing `nlp_ipopt.py` with `nlp_ipopt_rs.py` in the solver loop produces identical results on all 24 MINLPLib instances
- **Code quality:** `cargo clippy -- -D warnings` clean, zero `unsafe` blocks in core algorithm, all public functions documented

**Test files:** (owned by WS3b)
- `test_ipopt_rs.py`: Translation correctness (Rust vs C++ Ipopt output comparison on all benchmark problems)
- `test_ipopt_rs_edge_cases.py`: Degenerate Jacobians, near-singular KKT, cycling, tiny steps — the edge cases that make translation valuable
- `test_nlp_backend_swap.py`: Verify that swapping cyipopt → Rust Ipopt produces identical solver output end-to-end

**Integration milestone:** By Month 7-8, `Model.solve()` uses Rust Ipopt by default. cyipopt becomes an optional fallback (retained in CI for regression comparison but not required for installation).

---

### WS5: B&B Engine & Solver Orchestration (Months 4-10)

**Depends on:** WS1 (Rust IR), WS2 (relaxation compiler), WS3 (LP/NLP solvers)

**Delivers:**
- `discopt-core/src/bnb/`: Node pool with best-first/depth-first selection, branching (most-fractional baseline → reliability branching), pruning (bound-based, infeasibility, integrality), incumbent management, determinism guarantee
- **Batch export interface:** `TreeManager.export_batch(N) → (lb[N,n_vars], ub[N,n_vars], node_ids)` and `TreeManager.import_results(node_ids, lower_bounds, solutions, feasible)`
- Zero-copy via PyO3 numpy crate
- Layer profiling: timestamps every crossing → `rust_time_fraction`, `jax_time_fraction`, `python_time_fraction`
- **End-to-end `Model.solve()`:** implements the dispatch at `core.py:815-817`, populates `SolveResult`
- Solver orchestration loop: Rust exports batch → Python dispatches to JAX relaxation + HiGHS LP + Ipopt NLP → results imported back to Rust

**Verification:**
- All 24 `KNOWN_OPTIMA` instances solved correctly within `ABS_TOL=1e-4, REL_TOL=1e-3`
- Zero incorrect results (non-negotiable)
- `relaxation_valid = 1.0` (McCormick soundness maintained through B&B)
- `interop_overhead ≤ 0.05` — Python orchestration < 5%
- Zero-copy verified: `buf.ctypes.data` matches Rust pointer
- All arrays `dtype=float64`, batch shape `(N, n_vars)`
- Round-trip latency < 100μs for batch ≥ 32
- Batch sizes 1, 64, 128, 512, 1024 all correct
- Three identical runs with `deterministic=True` → identical node counts and objectives
- `rust_time_fraction + jax_time_fraction + python_time_fraction ≈ 1.0` (within 0.05)

**Test files:** (owned by WS5)
- `test_bnb_engine.py`: Rust-side B&B unit tests using `MockRelaxationEvaluator`
  - Node selection strategies on hand-crafted trees
  - Branching decisions on known fractional solutions
  - Pruning correctness (bound-based, integrality)
  - Determinism across runs
- `test_batch_dispatch.py`: Array transfer, shape/dtype, zero-copy, latency
- `test_correctness.py`: Activate existing stubs — replace `NotImplementedError` catch with actual `Model.solve()` calls. Incremental activation: start with `ex1221`, expand as reliability improves.
- `test_interop.py`: Activate existing stubs for overhead measurement, profiling

**Incremental activation protocol for `test_correctness.py`:**
```python
# Replace the catch-all skip:
try:
    sol = solve_instance(name)
except NotImplementedError:
    pytest.skip("Solver not yet available")

# With module-granular checks:
pytest.importorskip("discopt._rust")
pytest.importorskip("discopt._jax.dag_compiler")
sol = solve_instance(name)  # No more catch — failures are real failures
```

**Mock provided:** `MockTreeManager` (exports synthetic batches for WS6 GPU testing)

---

### WS9: CI/CD Pipeline (Months 1-4 core, continuous thereafter)

**Delivers (incrementally):**

**Months 1-3 (core CI):**
- `.github/workflows/ci.yml`: Python 3.10-3.12 × Linux/macOS, `ruff check`, `mypy --strict`, `cargo test`, `cargo clippy`, `maturin develop`, `pytest -m smoke`
- Pre-commit hooks for ruff + mypy
- Test marker configuration in `pyproject.toml`

**Months 4-8:**
- `pytest --cov --cov-fail-under=85` on PR
- Marker-based selection: smoke on PR, full on merge, correctness nightly
- Property-based testing with Hypothesis (strategies defined by each WS, integrated by WS9)

**Months 10-14:**
- Nightly regression: `run_benchmarks.py --suite nightly --ci` with `detect_regressions()`
- Phase gate workflow: `run_benchmarks.py --gate phaseN` with exit code 1 on failure

**Months 14-20:**
- GPU CI on self-hosted runner: `pytest -m gpu`, `--suite gpu_scaling`
- Maturin wheel builds for manylinux2014_x86_64, macosx_11_0_arm64

**Months 30+:**
- Tag-triggered PyPI release via `.github/workflows/release.yml`
- `pip install discopt` installs Python + Rust extension

**WS9 does NOT own:** Test file creation (each WS owns its tests), test fixture expansion, mock implementations.

---

### WS7: Bound Tightening & Preprocessing (Months 6-16)

**Phase 1 deliverables (WS7-a, Months 6-10):**
- `discopt-core/src/presolve/fbbt.rs`: Forward+backward bound propagation, fixed-point iteration
- `discopt-core/src/presolve/probing.rs`: Binary variable implications
- `discopt-core/src/presolve/simplify.rs`: Big-M strengthening, redundant constraint removal, integer bound tightening

**Phase 2 deliverables (WS7-b, Months 10-16):**
- `discopt-core/src/presolve/obbt.rs`: LP-based bound tightening with HiGHS warm-start, variable prioritization
- `discopt/_jax/obbt.py`: Gradient-based OBBT using `jax.grad` of relaxation w.r.t. bound parameters
- Pipeline: FBBT (cheap) → OBBT (expensive) → FBBT (propagate improvements)

**Verification:**
- FBBT tightens ≥ 2 bounds on `ex1221` vs original formulation
- Big-M strengthening: `x ≤ 100*y` with `x_ub=50` → coefficient tightened to 50
- Integer tightening: `x ≥ 1.3, x ≤ 4.7, x integer` → `x ∈ [2, 4]`
- OBBT + FBBT produces tighter root gap than FBBT alone on ≥ 10 of 24 known-optimum instances
- Preprocessing + solve still produces correct optimal values (zero incorrect)

**Test file:** `test_presolve.py` (owned by WS7)
- Parametrized tests for each preprocessing technique
- Specific criteria (FBBT on ex1221, big-M, integer tightening) as explicit test cases

---

### WS4: Rust Ipopt → GPU/Batch Adaptation (Months 10-18, Phase 2)

**Purpose:** Adapt the Rust Ipopt (WS3b) for GPU-accelerated batch solving via JAX. This is an *evolution* of known-good code, not a from-scratch build — the hardest part (getting a correct IPM) is already done by WS3b.

**Key refactoring steps:**
1. **Eliminate global state:** Ensure all solver state is in an explicit `SolverState` struct (WS3b should already do this, but verify)
2. **Struct-of-arrays layout:** Refactor internal arrays from AoS (one solver instance) to SoA (batch of N instances sharing problem structure but different data)
3. **Dense linear algebra:** Replace faer sparse factorization with `jax.scipy.linalg.cholesky` for problems up to ~5,000 variables (the typical MINLP relaxation subproblem size). The insight: batch throughput matters more than single-instance efficiency
4. **JAX integration:** Expose the IPM iteration as a JAX-callable function via `jax.pure_callback` or custom XLA extension, enabling `jax.vmap` over batched problem data
5. **Differentiability:** Add `custom_jvp` rule that uses KKT system solution for backward pass

**Delivers:**
- `discopt/_jax/ipm.py`: JAX-wrapped Rust Ipopt (Tier 1, dense GPU)
  - Same algorithm as `ipopt-rs`, but KKT factorization via `jax.scipy.linalg.cholesky`
  - `jax.jit` + `jax.vmap` compatible — the key requirement
  - Covers problems up to ~5,000 variables in batch mode
- `discopt/_jax/ipm_iterative.py`: PCG iterative solver (Tier 2)
  - Preconditioned Conjugate Gradient with warm-starting
  - Pure JAX — `vmap` compatible
  - Scales to ~50,000 variables for larger problems
  - Uses lineax building blocks

**Verification:**
- JAX-wrapped IPM converges on all 7 Netlib LP instances within `abs_tol=1e-6`
- Matches Rust Ipopt solution on all 7 example NLP problems within `rel_tol=1e-4`
- `jax.vmap(ipm_solve)(batch_of_64_problems)` produces correct results
- `jax.grad(lambda p: ipm_solve(p).objective)(params)` returns finite values
- KKT residual < 1e-10 at solution
- Batch of 64 GPU solves ≥ 10x faster than 64 sequential Rust Ipopt calls on CPU
- JIT recompilation count = 0 after warmup for fixed problem sizes

**Test files:** `test_ipm.py`, `test_ipm_iterative.py` (owned by WS4)

**Risk reduction vs original plan:** The original WS4 required building a correct IPM *and* making it vmappable — two hard problems stacked. Now WS3b solves correctness, and WS4 only handles the adaptation for batching/GPU. If WS4 is delayed, the Rust Ipopt still works as a single-instance solver (slower for batch, but correct).

---

### WS6: GPU Batching & Performance (Months 12-20, Phase 2)

**Depends on:** WS2 (relaxation compiler), WS5 (B&B batch dispatch)

**Delivers:**
- `discopt/_jax/batch_evaluator.py`: `BatchRelaxationEvaluator` using `jax.vmap(compiled_relaxation)` — single fused XLA kernel per batch, auto batch-size selection
- `discopt/_jax/primal_heuristics.py`: Multi-start NLP (`vmap` over 64-128 starts) + feasibility pump

**Verification:**
- GPU batch of 512 nodes ≥ 15x faster than serial CPU
- Node throughput ≥ 200 nodes/sec on 50-variable problems
- Batch sizes 1-1024 all produce correct results
- JIT recompilation count = 0 after warmup
- Memory scales linearly with batch size (no leak over 1000 consecutive batches)
- Multi-start with 64 starts finds optimal for `example_simple_minlp()` in ≥ 90% of runs

**Test files:** (owned by WS6)
- `test_batch_evaluator.py`: Functional correctness on CPU (batch vs serial), performance on GPU
- `test_primal_heuristics.py`: Multi-start success rate

---

### WS-D: Differentiable Solving (Months 14-20, Phase 2)

**Purpose:** Implement differentiable optimization — one of the three core value propositions.

**Delivers:**

**Level 1 — LP relaxation sensitivity:**
- `discopt/_jax/differentiable.py`:
  - `custom_jvp` wrapper for LP relaxation solve
  - Forward: exact solve via HiGHS or custom IPM
  - Backward: LP dual variables provide gradient of optimal value w.r.t. RHS/objective parameters
  - `result.gradient(param)` returns `d(obj*)/d(param)`
- Integration with `SolveResult` at `core.py:491-499`

**Level 3 — Implicit differentiation at optimal active set (stretch):**
- At MINLP solution, identify active constraints
- Apply implicit function theorem for exact gradients of continuous variables w.r.t. parameters
- Perturbation smoothing fallback at degenerate points

**Verification:**
- `jax.grad(lambda p: solve(model_with_param(p)).objective)(p0)` returns finite gradient
- Gradient matches finite-difference approximation within `rel_tol=1e-3` on 5 parametric problems
- Level 1 gradient is correct for LP (validate against known LP sensitivity results)
- Perturbation smoothing: `jax.vmap` over 32 perturbed solves produces stable gradient estimate

**Test files:** `test_differentiable.py` (owned by WS-D)
- Parametric LP sensitivity (analytical ground truth)
- Parametric MINLP gradient (finite-difference comparison)
- `jax.grad` composability (gradient through solve embedded in larger computation)

---

### WS10: Advanced Algorithms & Release (Months 20-42, Phase 3-4)

**Decomposed into testable sub-tasks:**

**WS10-a: Advanced Relaxations (Months 20-28)**
- Piecewise McCormick with adaptive partitioning (k=4-16)
- alphaBB relaxations using `jax.hessian` for eigenvalue estimation
- Convex envelopes for trilinear, fractional, signomial terms
- **Test:** `test_piecewise_mccormick.py`, `test_alphabb.py` — soundness invariant at 10,000 points

**WS10-b: Cutting Planes (Months 22-30)**
- RLT cuts, gradient-based outer approximation, lift-and-project
- **Test:** `test_cutting_planes.py` — cut validity on small instances

**WS10-c: Learned Branching (Months 24-32)**
- GNN branching policy (bipartite graph, Equinox + jraph + Optax)
- Imitation learning on strong branching → RL fine-tuning
- **Test:** `test_gnn_branching.py` — inference latency < 0.1ms, node reduction ≥ 20%

**WS10-d: Release Engineering (Months 36-42)**
- Documentation: API docs, tutorials, mathematical ADRs
- 3-5 example notebooks (modeling, batch solving, sensitivity analysis)
- v1.0 release: `discopt` under MIT/Apache-2.0, `ipopt-rs` under EPL-2.0

---

## 4. Revised Phase Gate Criteria

### Phase 1 Gate (Month 10) — Revised for External-Library Strategy

```toml
[gates.phase1]
description = "Phase 1 → Phase 2 gate: working MINLP solver"

  [gates.phase1.criteria]
  # Core solver correctness (validates OUR code)
  minlplib_solved_count = { min = 25, suite = "phase1", metric = "solved_count" }
  relaxation_valid = { min = 1.0, suite = "phase1", metric = "relaxation_validity_rate" }
  interop_overhead = { max = 0.05, suite = "phase1", metric = "python_orchestration_fraction" }
  zero_incorrect = { max = 0, suite = "phase1", metric = "incorrect_count" }

  # JAX evaluator integration (validates JAX+Ipopt pipeline)
  nlp_convergence_rate = { min = 0.80, suite = "nlp_cutest", metric = "convergence_rate" }

  # REMOVED: lp_netlib_pass_rate (tests HiGHS, not our code)
  # REMOVED: lp_vs_highs_geomean (meaningless when using HiGHS)
  # REMOVED: sparse_accuracy (tests MUMPS, not our code)
```

**Changes:**
- Removed 3 criteria that test external library performance rather than custom code
- Kept `nlp_convergence_rate` because it validates the JAX evaluation layer + Ipopt integration
- Timeline shortened from Month 14 to Month 10

### Phase 2 Gate (Month 20) — Revised Targets

```toml
[gates.phase2]
description = "Phase 2 → Phase 3 gate: GPU acceleration + differentiability"

  [gates.phase2.criteria]
  minlplib_30var_solved = { min = 55, suite = "phase2", metric = "solved_count_le30var" }
  minlplib_50var_solved = { min = 25, suite = "phase2", metric = "solved_count_le50var" }
  geomean_vs_couenne = { max = 3.0, suite = "comparison", metric = "geomean_ratio_vs_couenne" }
  gpu_speedup = { min = 15.0, suite = "gpu_scaling", metric = "gpu_vs_cpu_speedup" }
  root_gap_vs_baron = { max = 2.0, suite = "comparison", metric = "root_gap_ratio_vs_baron" }
  node_throughput = { min = 200, suite = "phase2", metric = "median_nodes_per_second" }
  rust_overhead = { max = 0.05, suite = "phase2", metric = "rust_tree_overhead_fraction" }
  zero_incorrect = { max = 0, suite = "phase2", metric = "incorrect_count" }

  # NEW: differentiable solving gate
  differentiable_level1 = { min = 1.0, suite = "phase2", metric = "differentiable_gradient_test_pass_rate" }
```

**Changes:**
- `root_gap_vs_baron`: relaxed from 1.3 to 2.0 (1.3 is too aggressive for 20 months)
- Added `differentiable_level1` gate (validates core value proposition)
- Timeline shortened from Month 26 to Month 20

### Phase 3 Gate (Month 32) — Unchanged Structurally

```toml
[gates.phase3]
description = "Phase 3 → Phase 4 gate: competitive performance"

  [gates.phase3.criteria]
  minlplib_30var_solved = { min = 75, suite = "phase3", metric = "solved_count_le30var" }
  minlplib_50var_solved = { min = 45, suite = "phase3", metric = "solved_count_le50var" }
  minlplib_100var_solved = { min = 20, suite = "phase3", metric = "solved_count_le100var" }
  geomean_vs_baron = { max = 2.5, suite = "comparison", metric = "geomean_ratio_vs_baron" }
  gpu_class_vs_baron = { max = 1.0, suite = "pooling", metric = "geomean_ratio_vs_baron" }
  learned_branching_improvement = { min = 0.20, suite = "phase3", metric = "node_reduction_vs_classical" }
  root_gap_vs_baron = { max = 1.3, suite = "comparison", metric = "root_gap_ratio_vs_baron" }
  zero_incorrect = { max = 0, suite = "full", metric = "incorrect_count" }
```

**Changes:**
- Added `root_gap_vs_baron ≤ 1.3` here (moved from Phase 2 where it was premature)
- Timeline shortened from Month 38 to Month 32

### Phase 4 Gate (Month 42) — Realistic BARON Target

```toml
[gates.phase4]
description = "Release gate: production quality"

  [gates.phase4.criteria]
  minlplib_30var_solved = { min = 85, suite = "full", metric = "solved_count_le30var" }
  minlplib_100var_solved = { min = 30, suite = "full", metric = "solved_count_le100var" }
  geomean_vs_baron = { max = 3.0, suite = "comparison", metric = "geomean_ratio_vs_baron" }
  gpu_classes_faster_than_baron = { min = 3, suite = "full", metric = "problem_classes_beating_baron" }
  beats_couenne = { min = 1.0, suite = "comparison", metric = "solve_count_ratio_vs_couenne" }
  beats_bonmin = { min = 1.0, suite = "comparison", metric = "solve_count_ratio_vs_bonmin" }
  zero_incorrect = { max = 0, suite = "full", metric = "incorrect_count" }

  # NEW: batch solving gate (core value proposition)
  vmap_batch_speedup = { min = 50.0, suite = "gpu_scaling", metric = "batch_512_vs_serial_speedup" }
```

**Changes:**
- `geomean_vs_baron`: relaxed from 1.5 to 3.0 (1.5x is 60-84+ months per feasibility assessment)
- `classes_faster_than_baron`: increased from 2 to 3 (focus on GPU-amenable classes where we win)
- Added `vmap_batch_speedup` gate (validates the #1 value proposition)
- Timeline shortened from Month 48 to Month 42

---

## 5. Task List with Dependencies

Tasks are organized by execution order. Each task includes its work stream, dependencies, test file, and acceptance criteria.

### Phase 0: Validation + Foundation (Month 1)

| # | Task | WS | Blocked By | Test | Acceptance |
|---|------|----|-----------|------|------------|
| T0 | Architectural spike: Rust↔JAX GPU batch latency | WS0 | — | `test_spike.py` | Latency < 100μs, GPU ≥ 10x at batch 512 |
| T1 | Create repos: `discopt-org/ipopt-rs` + `discopt-org/discopt` with Cargo workspace | WS1 | — | `test_rust_ir.py` (build) | Both repos created, `cargo build` succeeds in both, `maturin develop` produces `discopt._rust`, path dependency wired |

**T0 and T1 run in parallel.** T1 is 1-2 days of setup (create GitHub org, two repos, cargo workspaces, pyproject.toml, maturin, path dependency). T0 is a 2-4 week validation. T1 no longer waits for T0 go/no-go — the Rust workspace is needed immediately for the Ipopt translation (T9a). If T0 fails, the architecture is redesigned but the Ipopt translation work is still valuable.

### Phase 1: Working Solver (Months 1-10)

| # | Task | WS | Blocked By | Test | Acceptance |
|---|------|----|-----------|------|------------|
| T2 | Expression IR in Rust + structure detection | WS1 | T1 | `test_rust_ir.py` (expressions) | 7 examples convert, round-trip within 1e-14 |
| T3 | `.nl` file parser | WS1 | T1 | `test_rust_ir.py` (parsing) | 24 MINLPLib instances parse without error |
| T4 | JAX DAG compiler (Expression → jax.numpy) | WS2 | — | `test_relaxation.py` (compiler) | `jax.make_jaxpr` succeeds on all 7 examples, `jax.grad` finite |
| T5 | McCormick relaxation primitives | WS2 | T4 | `test_relaxation.py` (soundness) | Soundness at 10,000 points for all 14 operation types |
| T6 | Relaxation compiler (compile_relaxation) | WS2 | T4, T5 | `test_relaxation.py` (vmap) | `jax.jit(jax.vmap(...))` works on batch of 128 |
| T7 | HiGHS LP wrapper + warm-start | WS3a | — | `test_lp_highs.py` | 7 Netlib instances correct, warm-start faster than cold |
| T8 | JAX NLP evaluator (grad, Hessian, Jacobian) | WS3a | T4 | `test_nlp_evaluator.py` | Gradient atol=1e-6 vs finite diff, JIT warmup ≥ 10x |
| T9 | cyipopt NLP wrapper + JAX callbacks (scaffolding) | WS3a | T7, T8 | `test_nlp_ipopt.py` | Converges on 7 example NLP relaxations |
| T9a | Ipopt C++ core → Rust translation (IPM loop, filter, KKT) — **starts Month 1** | WS3b | T1 | `test_ipopt_rs.py` | Matches C++ Ipopt on 7 NLP + 7 LP problems |
| T9b | Rust Ipopt edge case validation | WS3b | T9a | `test_ipopt_rs_edge_cases.py` | Infeasibility, unboundedness, degenerate cases handled |
| T9c | Replace cyipopt with Rust Ipopt in solver loop | WS3b | T9a, T14 | `test_nlp_backend_swap.py` | All 24 MINLPLib identical results, cyipopt becomes optional |
| T10 | CI/CD core (GitHub Actions, ruff, mypy, cargo) | WS9 | T1 | CI green | CI passes on clean checkout |
| T11 | B&B tree manager (Rust: node pool, branching, pruning) | WS5 | T1 | `test_bnb_engine.py` | Correct on hand-crafted trees with MockRelaxationEvaluator |
| T12 | Batch dispatch interface (export/import) | WS5 | T1 | `test_batch_dispatch.py` | Zero-copy, shapes correct, latency < 100μs |
| T13 | FBBT + basic presolve | WS7-a | T2 | `test_presolve.py` | Tightens ≥ 2 bounds on ex1221, big-M strengthening works |
| T14 | Solver orchestrator: end-to-end `Model.solve()` | WS5 | T2, T3, T5, T6, T7, T9, T11, T12 | `test_correctness.py` | `ex1221` solved correctly (optimum 7.6672) |
| T15 | MINLPLib loading + full correctness validation | WS5 | T14 | `test_correctness.py` | All 24 KNOWN_OPTIMA correct, zero incorrect |
| T16 | Phase 1 gate validation | — | T15, T13 | `run_benchmarks.py --gate phase1` | All Phase 1 criteria pass |

### Phase 2: GPU + Differentiability (Months 10-20)

| # | Task | WS | Blocked By | Test | Acceptance |
|---|------|----|-----------|------|------------|
| T17 | Adapt Rust Ipopt for JAX GPU batch (dense Cholesky, vmap) | WS4 | T9a, T8 | `test_ipm.py` | Matches Rust Ipopt on 7 examples, `jax.vmap` works, ≥10x at batch 64 |
| T18 | PCG iterative solver (Tier 2, lineax) | WS4 | T17 | `test_ipm_iterative.py` | Scales to 50K vars, moderate accuracy |
| T19 | Batch relaxation evaluator (vmap) | WS6 | T6, T14 | `test_batch_evaluator.py` | GPU ≥ 15x speedup, batch correctness |
| T20 | Multi-start primal heuristics (vmap) | WS6 | T19 | `test_primal_heuristics.py` | Finds optimal ≥ 90% for simple_minlp |
| T21 | OBBT + advanced presolve | WS7-b | T7, T13 | `test_presolve.py` | Root gap ≤ 2.0x BARON on comparison set |
| T22 | Level 1 differentiable solving (custom_jvp) | WS-D | T14 | `test_differentiable.py` | `jax.grad` through solve works, matches finite diff |
| T23 | Level 3 differentiable solving (implicit diff) | WS-D | T22 | `test_differentiable.py` | Active-set implicit gradient on 3 problems |
| T24 | Replace Rust Ipopt with GPU-batched IPM in solver loop | WS4+WS5 | T17, T14 | `test_correctness.py` | All 24 still correct with GPU IPM |
| T25 | Performance measurement + benchmark runner | WS9 | T19 | `run_benchmarks.py --suite phase2` | GPU speedup, node throughput measured |
| T26 | Phase 2 gate validation | — | T19, T21, T22, T24, T25 | `run_benchmarks.py --gate phase2` | All Phase 2 criteria pass |

### Phase 3: Competitive Performance (Months 20-32)

| # | Task | WS | Blocked By | Test | Acceptance |
|---|------|----|-----------|------|------------|
| T27 | Piecewise McCormick + alphaBB | WS10-a | T6 | `test_piecewise_mccormick.py`, `test_alphabb.py` | ≥ 60% gap reduction vs standard McCormick |
| T28 | Cutting planes (RLT, OA) | WS10-b | T14 | `test_cutting_planes.py` | Cuts valid on small instances |
| T29 | GNN branching policy | WS10-c | T14 | `test_gnn_branching.py` | Inference < 0.1ms, ≥ 20% node reduction |
| T30 | Phase 3 gate validation | — | T27, T28, T29 | `run_benchmarks.py --gate phase3` | All Phase 3 criteria pass |

### Phase 4: Polish & Release (Months 32-42)

| # | Task | WS | Blocked By | Test | Acceptance |
|---|------|----|-----------|------|------------|
| T31 | Documentation + example notebooks | WS10-d | T26 | Manual review | 3-5 notebooks, API docs |
| T32 | Release engineering (`pip install discopt`) | WS9+WS10-d | T31 | `pip install` test | Wheel builds, installs, imports work |
| T33 | Phase 4 gate validation | — | T30, T31, T32 | `run_benchmarks.py --gate phase4` | All Phase 4 criteria pass |

---

## 6. Mock Interfaces for Parallel Development

Each integration seam has a defined mock to enable independent module testing:

| Mock | Defined By | Used By | Interface |
|------|-----------|---------|-----------|
| `MockModelRepr` | WS1 | WS2, WS5 | Python dict mimicking Rust expression graph |
| `MockRelaxationEvaluator` | WS2 | WS5, WS6 | Returns constant lower/upper bounds for given node bounds |
| `MockLPSolver` | WS3a | WS5, WS7 | Returns known optimal for small hand-crafted LPs |
| `MockNLPSolver` | WS3a | WS5 | Returns known optimal for small NLPs (implements `SolverBackend`) |
| `MockTreeManager` | WS5 | WS6 | Exports synthetic batches of fixed shape |
| C++ Ipopt reference | WS3a | WS3b | cyipopt solutions serve as ground truth for Rust Ipopt validation |

Each mock is created as part of its owning work stream's test deliverables and placed in `tests/mocks/`.

---

## 7. Revised Timeline

| Milestone | Original Plan | This Plan | Key Enabler |
|-----------|--------------|-----------|-------------|
| Architectural spike validated | — | Month 1 | New: validate GPU thesis first |
| Rust workspace + PyO3 working | Month 3 | Month 2-3 | Same |
| DAG compiler + McCormick basics | Month 6-10 | Month 3-5 | Narrower scope (basics only) |
| HiGHS + cyipopt scaffolding | — | Month 3-5 | New: quick-start NLP solving |
| Ipopt-to-Rust translation begun | — | Month 2 | New: parallel with all Phase 1 work |
| First MINLP solved (`ex1221`) | ~Month 10-12 | Month 5-6 | HiGHS + cyipopt as subsolvers |
| Rust Ipopt core algorithm working | — | Month 5-6 | LLM-assisted translation + faer |
| Rust Ipopt matches C++ on all benchmarks | — | Month 7-8 | Edge case validation complete |
| cyipopt replaced by Rust Ipopt | — | Month 8-9 | Drop external NLP dependency |
| 24 MINLPLib instances solved | Month 14 | Month 8-10 | Rust Ipopt + HiGHS |
| **Phase 1 gate** | **Month 14** | **Month 10** | No external NLP dependency |
| GPU-batched Rust Ipopt working | Month 20 | Month 14-16 | Adapt WS3b code, dense Cholesky |
| GPU batch evaluator working | Month 18 | Month 14-16 | After Phase 1 solver works |
| Level 1 differentiable solving | ~Month 26 | Month 14-16 | `custom_jvp` on LP relaxation |
| vmap batch solving demo | Not explicit | Month 16-18 | GPU-adapted IPM is vmappable |
| **Phase 2 gate** | **Month 26** | **Month 20** | GPU IPM + batching + differentiability |
| Competitive with Couenne | Month 38 | Month 24-30 | Advanced relaxations + presolve |
| Learned branching prototype | Month 33 | Month 24-28 | Training data from Phase 2 solves |
| **Phase 3 gate** | **Month 38** | **Month 32** | Advanced algorithms |
| v1.0 release | Month 48 | Month 42 | Narrower scope, earlier start |
| Within 3x of BARON (general) | Month 48 (1.5x) | Month 42 (3.0x) | Realistic target |
| Within 1.0x of BARON (GPU classes) | Month 48 | Month 32 | Pooling, portfolio |

---

## 8. Team Allocation (2-3 core, 1 part-time)

| Role | Phase 1 (Months 1-10) | Phase 2 (Months 10-20) | Phase 3-4 (Months 20-42) |
|------|----------------------|----------------------|--------------------------|
| **Rust + Systems Engineer** | WS0 → WS1 → WS3b (Ipopt-to-Rust, primary) → WS5 (B&B) | WS4 (adapt Rust Ipopt for GPU) → WS7-b | WS10-b (cutting planes) |
| **JAX + Numerical Engineer** | WS2 → WS3a (JAX evaluator) | WS6 → WS-D | WS10-a (relaxations) → WS10-c (GNN) |
| **Applied Math / Optimization** (from Month 3) | WS3b (Ipopt translation, algorithm review) → WS3a (HiGHS) → WS7-a | WS4 (GPU adaptation design) → WS7-b (OBBT) | WS10-a (alphaBB) → WS10-c (training) |
| **DevOps** (part-time) | WS9 (CI setup) | WS9 (benchmark CI, GPU CI) | WS9 (release automation) |

**Removed:** LLM Engineer (WS8 cut). Numerical Specialist role merged with Applied Math.

**Note on WS3b staffing:** The Ipopt-to-Rust translation benefits from both the Rust engineer (language mechanics, memory layout, PyO3 integration) and the Applied Math person (algorithm understanding, numerical edge cases, convergence theory). LLM-assisted translation handles the mechanical conversion; human expertise focuses on validating numerical correctness and identifying edge cases that need special attention.

---

## 9. Risk Mitigations

| Risk | Severity | Mitigation | Task |
|------|----------|------------|------|
| GPU-CPU transfer overhead | MEDIUM | Architectural spike validates before commitment | T0 |
| Team recruitment | CRITICAL | Start with 2; external libraries reduce specialist need | Organizational |
| Funding | HIGH | MVP by Month 10 enables grant applications (NSF CSSI, DOE ASCR) | T16 produces evidence |
| Gurobi 13.0 competition | MEDIUM-HIGH | Focus on vmap+grad differentiators, not single-instance speed | T19, T22 |
| Scope creep | HIGH | WS8 cut; strict phase gates enforce scope | This document |
| Infrastructure without solver | MEDIUM | First solve at Month 5-6 using HiGHS+cyipopt scaffolding | T14 |
| From-scratch IPM correctness | **ELIMINATED** | Ipopt-to-Rust translation preserves 20yr of edge-case handling | T9a, T9b |
| Ipopt translation takes longer than expected | MEDIUM | cyipopt remains as fallback throughout Phase 1; translation can extend into Phase 2 without blocking | T9, T9a |
| EPL-2.0 license friction | LOW | Dual-module licensing (`ipopt-rs` under EPL-2.0, discopt under MIT); pursue upstream relicensing conversation | Section 0.2a |

---

## 10. What This Plan Does NOT Include

The following are explicitly out of scope for this plan:

1. **LLM features** (`from_description()`, `chat()`, RAG, reformulation advisor) — deferred indefinitely
2. **Ecosystem packages** (`jax-lp`, `jax-qp`, `jax-milp`, `jax-miqp`, `jax-optcore`) — aspirational vision only
3. **Multi-GPU support** (`pmap`) — Phase 4+ if demand exists
4. **Sparse GPU Cholesky** (Tier 3, cuSOLVER via `jax.extend.ffi`) — Phase 4+ if problem sizes demand it
5. **`.gms` parser**, **`from_pyomo()`** — low priority, community contribution
6. **Windows support** — Linux + macOS only for Phase 1-3
7. **Matching BARON on general MINLP** — not a goal within 42 months
8. **Full Ipopt C++ translation** — only the core IPM algorithm is translated (filter line search, KKT, restoration). MUMPS/MA27 linear solvers, AMPL interface, HSL solvers, and options infrastructure are out of scope (replaced by Rust-native equivalents)

---

## 11. Documents Retained as References

| Document | Status | Use |
|----------|--------|-----|
| `archive/jaxminlp_development_plan.md` | Superseded | Historical reference for original WS1-WS10 specifications |
| `archive/feasibility_assessment.md` | Incorporated | Risk analysis and competitive positioning remain valid |
| `archive/JAX_OPTIMIZATION_ECOSYSTEM_VISION.md` | Aspirational | Long-term direction for post-v1.0 ecosystem expansion |
| `archive/review_architecture_dependencies.md` | Reference | Review findings that informed this plan |
| `archive/review_modularity_testability.md` | Reference | Review findings that informed this plan |
| `archive/review_coherence_gaps.md` | Reference | Review findings that informed this plan |

---

## 12. Existing Code to Reuse (do NOT rewrite)

| Component | File | Lines | Status | Rename Impact |
|-----------|------|-------|--------|---------------|
| Modeling API + Expression DAG | `jaxminlp_api/core.py` | 1,022 | Complete | Rename imports to `discopt` |
| Example models | `jaxminlp_api/examples.py` | 602 | Complete | Rename imports |
| Benchmark metrics + phase gates | `benchmarks/metrics.py` | 664 | Complete | Update gate criteria per Section 4 |
| Benchmark runner | `benchmarks/runner.py` | 314 | Complete (stubs to wire) | Wire to `discopt` solver |
| Statistics + profiles | `utils/statistics.py` | 212 | Complete | No change |
| Report generation | `utils/reporting.py` | 302 | Complete | No change |
| Pytest fixtures + markers | `tests/conftest.py` | 216 | Complete | Update skip conditions per Section 5 |
| Correctness test structure | `tests/test_correctness.py` | 276 | Structure ready | Incremental activation per module |
| Interop test structure | `tests/test_interop.py` | 240 | Structure ready | Split per Section 3 (WS5) |
| Phase gate config | `config/benchmarks.toml` | 190 | **Needs update** | Apply revised gates from Section 4 |
| CLI entry point | `run_benchmarks.py` | 198 | Complete | Wire to `discopt` solver |

---

*This plan is the single source of truth for discopt development. All implementation decisions, timelines, and acceptance criteria are defined here. Changes to this plan require explicit versioning with rationale.*
