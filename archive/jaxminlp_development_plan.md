# JaxMINLP Comprehensive Development Plan

## Context

JaxMINLP is a 48-month project to build a standalone, open-source MINLP solver combining Rust (tree search, LP solver, sparse linear algebra), JAX (GPU-batched relaxation evaluation, autodiff, learned heuristics), and an LLM advisory layer (formulation, explanation, reformulation). The repository currently contains a **complete benchmarking/testing framework** (`jaxminlp_benchmarks/`) with fully-implemented metrics, phase gates, test infrastructure, modeling API, and reporting — but **zero solver code**. All solver integration points raise `NotImplementedError`.

This plan consolidates the two design documents (`jaxminlp_plan_v4.docx`, `jaxminlp_rust_jax_gpu_integration.docx`) into **10 parallel work streams** with verifiable completion criteria tied to the existing phase gate definitions in `config/benchmarks.toml`.

---

## Work Stream Overview

| # | Stream | Phase | Parallel With | Primary Gate Criteria |
|---|--------|-------|---------------|----------------------|
| WS1 | Rust Infrastructure & Expression IR | 1 | WS2, WS5, WS8, WS9 | Crate builds, model converts |
| WS2 | JAX DAG Compiler & McCormick Relaxations | 1 | WS1, WS3, WS5, WS9 | Soundness invariant holds |
| WS3 | LP Solver & Sparse Linear Algebra | 1 | WS2, WS4, WS5 | ≥95% Netlib pass, ≤1e-12 residual |
| WS4 | NLP Subsolver (Hybrid IPM) | 1-2 | WS3, WS5 | ≥80% CUTEst convergence |
| WS5 | B&B Engine & Batch Dispatch | 1-2 | WS1, WS2, WS3 | ≥25 solved, zero incorrect |
| WS6 | GPU Batching & Performance | 2 | WS7, WS8 | ≥15x GPU speedup, ≥200 nodes/s |
| WS7 | Bound Tightening & Preprocessing | 1-2 | WS6, WS8 | Root gap ≤1.3x BARON |
| WS8 | LLM Advisory Layer | 2-4 | WS6, WS7 | from_description works, safety gates pass |
| WS9 | CI/CD & Testing Infrastructure | 1-4 | All | 85% coverage, all phase gates automated |
| WS10 | Advanced Algorithms & Release | 3-4 | — | ≤1.5x BARON, learned branching ≥20% |

---

## WS1: Rust Infrastructure & Expression IR

**Phase:** 1 (Months 1-6) | **Owner:** Rust developer

**Delivers:**
- Cargo workspace: `jaxminlp-core` (pure Rust lib) + `jaxminlp-python` (PyO3 bindings)
- `pyproject.toml` with maturin build-backend; `maturin develop` produces `jaxminlp._rust`
- Expression graph IR in Rust (`ExprNode` enum arena-allocated DAG) mirroring Python DAG types from `core.py` (`BinaryOp`, `UnaryOp`, `FunctionCall`, `Variable`, `Constant`, etc.)
- `impl From<PyModel> for ModelRepr` — walks Python expression tree, builds Rust arena
- Structure detection: `is_linear()`, `is_quadratic()`, `is_bilinear()`, `is_convex()`
- `.nl` file parser outputting `ModelRepr` directly — wires into `from_nl()` stub at `core.py:968`
- `.pyi` type stub for mypy compatibility

**Verification:**
- `maturin develop` succeeds on macOS ARM64 and Linux x86_64
- `python -c "from jaxminlp._rust import version"` works
- All 7 runnable examples from `examples.py` convert to `ModelRepr` without panic
- All 24 MINLPLib instances from `test_correctness.py:KNOWN_OPTIMA` parse from `.nl` without error
- `is_linear("2*x + 3*y")` → true; `is_linear("x*y")` → false
- Round-trip: Rust evaluates parsed expression at random point, matches Python within 1e-14
- `cargo test && cargo clippy -- -D warnings` clean

**Critical files:**
- `jaxminlp_benchmarks/jaxminlp_api/core.py` — expression DAG classes (lines 60-400), `from_nl()` stub (line 968)
- `jaxminlp_benchmarks/tests/test_interop.py` — defines PyO3 API surface

---

## WS2: JAX DAG Compiler & McCormick Relaxations

**Phase:** 1 (Months 1-10) | **Owner:** JAX/GPU engineer

**Delivers:**
- `jaxminlp/_jax/dag_compiler.py`: walks `Expression` tree → pure `jax.numpy` callable (JIT-compatible, no Python control flow on hot path)
- `jaxminlp/_jax/mccormick.py`: relaxation primitives for all operations (bilinear products, univariate convex/concave/nonconvex functions) — factorable programming approach
- `jaxminlp/_jax/relaxation_compiler.py`: combines DAG compiler + McCormick primitives → `compile_relaxation(expr, variables)` returning a function `(lb_vec, ub_vec) → (relax_lower, relax_upper)` that is `jax.jit` + `jax.vmap` compatible

**Verification:**
- **Soundness invariant (non-negotiable):** at 10,000 random points within bounds, `relaxation_lower ≤ true_value` and `relaxation_upper ≥ true_value` for every relaxation rule. Tolerance: 1e-10. Maps to `test_interop.py:218-239`.
- `jax.make_jaxpr(compiled_fn)(x)` succeeds for all example objectives (proves traceability)
- `jax.grad(compiled_fn)(x)` returns finite values at 100 random interior points
- `jax.jit(jax.vmap(compiled_relaxation))` processes batch of 128 bound vectors in one call
- As bounds tighten, relaxation gap decreases monotonically (5 progressive steps, 10 expressions)
- All 14 `FunctionCall` types handled: exp, log, log2, log10, sqrt, sin, cos, tan, abs, sign, min, max, sum, prod

**Critical files:**
- `jaxminlp_benchmarks/jaxminlp_api/core.py` — Expression subclasses (lines 60-400) are the input contract
- `jaxminlp_benchmarks/jaxminlp_api/examples.py` — 7 runnable examples are primary test inputs
- `jaxminlp_benchmarks/tests/test_interop.py:218-239` — soundness invariant test

---

## WS3: LP Solver & Sparse Linear Algebra

**Phase:** 1 (Months 3-12) | **Owner:** Numerical computing specialist

**Delivers:**
- `jaxminlp-core/src/lp/`: Revised simplex with primal Phase I/II, dual simplex (warm-start for B&B), steepest-edge pricing, degeneracy handling
- `jaxminlp-core/src/sparse/`: CSC matrix format, sparse LU with Markowitz pivoting + Forrest-Tomlin update, sparse Cholesky (supernodal, AMD ordering), sparse LDL^T for indefinite KKT systems
- Iterative refinement for all factorizations
- PyO3 bindings: `solve_lp(c, A_ub, b_ub, A_eq, b_eq, bounds, warm_start) -> LPResult`

**Verification:**
- **Phase 1 gate:** `lp_netlib_pass_rate ≥ 0.95` on Netlib benchmark set (`benchmarks.toml:110`)
- **Phase 1 gate:** `lp_vs_highs_geomean ≤ 3.0` (`benchmarks.toml:111`)
- **Phase 1 gate:** `sparse_accuracy ≤ 1e-12` max residual on SuiteSparse (`benchmarks.toml:112`)
- Correct optima on all 7 Netlib instances in `conftest.py:82-96` within `abs_tol=1e-6`
- Dual simplex warm-start resolves perturbed LP in <50% of cold-start iterations
- Infeasibility and unboundedness correctly detected and reported
- Dimension mismatch → Python `ValueError`, not Rust panic
- LDL^T inertia matches LAPACK reference on known indefinite matrix

**Critical files:**
- `jaxminlp_benchmarks/tests/conftest.py:82-96` — Netlib optima fixture
- `jaxminlp_benchmarks/config/benchmarks.toml:65-84` — LP/sparse benchmark suites

---

## WS4: NLP Subsolver (Hybrid IPM)

**Phase:** 1-2 (Months 6-20) | **Owner:** JAX/GPU engineer + Rust developer (shared)

**Delivers:**
- **JAX side** (`jaxminlp/_jax/nlp_evaluator.py`): `NLPEvaluator` class providing JIT-compiled `evaluate_objective(x)`, `evaluate_gradient(x)` via `jax.grad`, `evaluate_hessian(x)` via `jax.hessian`, `evaluate_constraints(x)`, `evaluate_jacobian(x)`, `evaluate_kkt_rhs(x, lam, s, mu)`
- **Rust side** (`jaxminlp-core/src/ipm/`): KKT factorization via sparse LDL^T (from WS3), inertia correction/regularization, iterative refinement to `||KKT * dir - rhs|| < 1e-10`
- **Integration:** JAX evaluates f/∇f/∇²f/g/J on GPU → KKT matrix transferred GPU→CPU → Rust factorizes → search direction transferred CPU→GPU → JAX line search. Two transfers per IPM iteration (15-50 iterations typical).

**Verification:**
- **Phase 1 gate:** `nlp_convergence_rate ≥ 0.80` on CUTEst convex instances (`benchmarks.toml:109`)
- Gradient accuracy: `jnp.allclose(grad, finite_diff_grad, atol=1e-6)` at 50 random feasible points per example
- Hessian accuracy: `jnp.allclose(hessian, finite_diff_hessian, atol=1e-4)` at 20 random points
- KKT residual < 1e-10 after iterative refinement
- Inertia correction succeeds within 3 regularization doublings on deliberately indefinite matrix
- Second call to `evaluate_objective` ≥10x faster than first (JIT warmup)

**Critical files:**
- `jaxminlp_benchmarks/config/benchmarks.toml:78-79` — CUTEst NLP suite
- `jaxminlp_benchmarks/tests/conftest.py:110-120` — numerical tolerance fixtures

---

## WS5: B&B Engine & Batch Dispatch

**Phase:** 1-2 (Months 8-18) | **Owner:** Rust developer

**Delivers:**
- `jaxminlp-core/src/bnb/`: Node pool with best-first/depth-first selection, branching (most-fractional baseline, reliability branching Phase 1, strong branching Phase 2, learned branching hook Phase 3), pruning (bound-based, infeasibility, integrality), incumbent management, determinism guarantee
- **Batch export interface** (the critical Rust↔JAX boundary): `TreeManager.export_batch(N) → (lb[N,n_vars], ub[N,n_vars], node_ids)` and `TreeManager.import_results(node_ids, lower_bounds, solutions, feasible)`
- Zero-copy via PyO3 numpy crate — `<1μs` per boundary crossing
- Double-buffer: while JAX evaluates batch K, Rust prepares batch K+1
- Layer profiling: timestamps every crossing → `rust_time_fraction`, `jax_time_fraction`, `python_time_fraction`
- **End-to-end `Model.solve()`:** implements the dispatch at `core.py:815-817`, populates `SolveResult`

**Verification:**
- **Phase 1 gate:** `minlplib_solved_count ≥ 25` (`benchmarks.toml:108`)
- **Phase 1 gate:** `zero_incorrect = 0` (`benchmarks.toml:115`)
- **Phase 1 gate:** `interop_overhead ≤ 0.05` — Python orchestration <5% (`benchmarks.toml:114`)
- **Phase 1 gate:** `relaxation_valid = 1.0` (`benchmarks.toml:113`)
- All 24 `KNOWN_OPTIMA` instances solved correctly within `ABS_TOL=1e-4, REL_TOL=1e-3`
- Zero-copy verified: `buf.ctypes.data` matches Rust pointer (`test_interop.py:73-76`)
- All arrays `dtype=float64`, batch shape `(N, n_vars)` (`test_interop.py:56-92`)
- Round-trip latency <100μs for batch ≥32 (`test_interop.py:127`)
- Batch sizes 1, 64, 128, 512, 1024 all correct
- Three identical runs with `deterministic=True` → identical node counts and objectives
- `rust_time_fraction + jax_time_fraction + python_time_fraction ≈ 1.0` (within 0.05)

**Critical files:**
- `jaxminlp_benchmarks/jaxminlp_api/core.py:814-820` — `Model.solve()` dispatch point
- `jaxminlp_benchmarks/tests/test_interop.py` — full PyO3 API surface (14 test methods)
- `jaxminlp_benchmarks/tests/test_correctness.py:52-78` — 24 known optima
- `jaxminlp_benchmarks/benchmarks/metrics.py` — `SolveResult` fields Rust must populate

---

## WS6: GPU Batching & Performance

**Phase:** 2 (Months 14-26) | **Owner:** JAX/GPU engineer

**Delivers:**
- `jaxminlp/_jax/batch_evaluator.py`: `BatchRelaxationEvaluator` using `jax.vmap(compiled_relaxation)` — single fused XLA kernel per batch, auto batch-size selection based on GPU memory, warmup method
- `jaxminlp/_jax/primal_heuristics.py`: Multi-start NLP (`vmap` over 64-128 starts) + feasibility pump (`vmap` over 32 rounding strategies)
- `jaxminlp/_jax/multi_gpu.py` (Phase 3): `pmap(vmap(relax))` across devices — no inter-GPU communication needed

**Verification:**
- **Phase 2 gate:** `gpu_speedup ≥ 15.0` — GPU batch of 512 nodes ≥15x faster than serial CPU (`benchmarks.toml:124`)
- **Phase 2 gate:** `node_throughput ≥ 200` nodes/sec on 50-variable problems (`benchmarks.toml:126`)
- Batch sizes 1-1024 all produce correct results; padded elements don't affect real results
- JIT recompilation count = 0 after warmup for any batch size up to configured max
- Memory scales linearly with batch size (1000 consecutive batches, no >5% growth)
- Multi-start with 64 starts finds optimal for `example_simple_minlp()` in ≥90% of runs
- Multi-GPU: 4 GPUs → ≥3.5x speedup over single GPU for batch ≥512

**Critical files:**
- `jaxminlp_benchmarks/config/benchmarks.toml:92-97` — GPU scaling suite
- `jaxminlp_benchmarks/benchmarks/metrics.py:390` — `gpu_vs_cpu_speedup()` function

---

## WS7: Bound Tightening & Preprocessing

**Phase:** 1-2 (Months 10-22) | **Owner:** Optimization researcher

**Delivers:**
- **Rust side** (`jaxminlp-core/src/presolve/`): FBBT (forward+backward propagation, fixed-point iteration), probing (binary variable implications), clique detection, big-M strengthening, redundant constraint removal, integer bound tightening
- **OBBT** (`jaxminlp-core/src/presolve/obbt.rs`): LP-based bound tightening with dual simplex warm-starting, variable prioritization (bilinear terms first)
- **JAX side** (`jaxminlp/_jax/obbt.py`): Gradient-based OBBT using `jax.grad` of relaxation w.r.t. bound parameters for prioritization
- Pipeline: FBBT (cheap) → OBBT (expensive) → FBBT (propagate improvements)

**Verification:**
- **Phase 2 gate:** `root_gap_vs_baron ≤ 1.3` (`benchmarks.toml:125`)
- FBBT tightens ≥2 bounds on `ex1221` vs original formulation
- Big-M strengthening: `x ≤ 100*y` with `x_ub=50` → coefficient tightened to 50
- Integer tightening: `x ≥ 1.3, x ≤ 4.7, x integer` → `x ∈ [2, 4]`
- OBBT + FBBT produces tighter root gap than FBBT alone on ≥10 of 24 known-optimum instances
- OBBT root gap improvement ≥10% on pooling problems
- Dual simplex warm-start reduces OBBT LP iterations by ≥30% vs cold-start
- Preprocessing + solve still produces correct optimal values (zero incorrect)

**Critical files:**
- `jaxminlp_benchmarks/config/benchmarks.toml:125` — root gap gate criterion
- `jaxminlp_benchmarks/tests/conftest.py:110-120` — tolerance fixtures

---

## WS8: LLM Advisory Layer

**Phase:** 2-4 (Months 10-48) | **Owner:** LLM integration engineer

**Delivers (incrementally):**

**Phase 1-2 (Months 10-20):** Safety infrastructure
- `jaxminlp_api/llm/safety.py`: `FormulationGate` (dimensional consistency, bounds, type checks, user approval), `ReformulationGate` (feasibility preservation), `ConfigurationGate` (parameter clamping), `OutputSanitizer` (no false optimality claims)
- `jaxminlp_api/llm/provider.py`: LLM provider abstraction (Claude, OpenAI, Ollama/vLLM)

**Phase 2 (Months 14-26):** Core capabilities
- `from_description()` implementation (`core.py:981-1016`) — LLM tool-calling to produce Model API calls, data ingestion, multi-turn interaction, all output passes through `FormulationGate`
- `SolveResult.explain()` enhancement (`core.py:491-499`) — LLM explains solution, binding constraints, integer decisions; sanitized output
- Configuration advisor: `suggest_config(model) → SolverConfig` with A/B testing vs defaults

**Phase 3 (Months 26-38):**
- Reformulation advisor: `reformulate(model) → list[ReformulationProposal]`
- Infeasibility diagnoser: IIS + LLM explanation

**Phase 4 (Months 38-48):**
- `jaxminlp.chat()` conversational REPL
- LLM evaluation benchmark (100 formulation tasks)
- RAG-based reformulation knowledge base
- `from_pyomo()` implementation (`core.py:938`)

**Verification:**
- Safety gates have 100% test coverage (pass + reject cases)
- `FormulationGate` rejects `lb > ub` models; `OutputSanitizer` strips "globally optimal" unless status is literally optimal
- `ConfigurationGate` clamps all parameters to valid ranges
- `from_description()` always calls `Model.validate()` before returning
- LLM benchmark (Phase 4): ≥80% formulation accuracy on 100-task set

**Critical files:**
- `jaxminlp_benchmarks/jaxminlp_api/core.py:981-1016` — `from_description()` stub with `llm_model` default
- `jaxminlp_benchmarks/jaxminlp_api/core.py:491-499` — `explain()` basic fallback

---

## WS9: CI/CD & Testing Infrastructure

**Phase:** 1-4 (continuous) | **Owner:** DevOps + all developers

**Delivers (incrementally):**

**Months 1-4:** Core CI
- `.github/workflows/ci.yml`: Python 3.10-3.12 × Linux/macOS/Windows, `ruff check`, `mypy --strict`, `cargo test`, `cargo clippy`, `maturin develop`, `pytest -m smoke`
- Extract lit review workflow from `agents/lit_review.py:905-972` to `.github/workflows/lit-review.yml`

**Months 4-8:** Test CI
- `pytest --cov --cov-fail-under=85` on PR
- Marker-based selection: smoke on PR, full on merge, correctness nightly
- Property-based testing with Hypothesis (strategies for random expression trees, LP instances, bound intervals)
- New test files: `test_api.py` (modeling API unit tests), `test_relaxation.py` (McCormick soundness), `test_lp.py` (Netlib + property), `test_sparse.py` (factorization accuracy)

**Months 10-14:** Activate stubs
- Replace `NotImplementedError` catches in `test_correctness.py` with actual solver calls
- Activate all 14 interop tests in `test_interop.py`
- Wire `runner._run_jaxminlp()` (`runner.py:194`) to actual solver

**Months 14-26:** Benchmark CI + GPU CI
- Nightly regression: `run_benchmarks.py --suite nightly --ci` with `detect_regressions()` against stored baseline
- Phase gate workflow: `run_benchmarks.py --gate phaseN` with exit code 1 on failure
- GPU CI on self-hosted runner: `pytest -m gpu`, `--suite gpu_scaling`
- Mutation testing with mutmut (monthly, ≥85% mutation score)

**Months 26+:** Release automation
- Maturin wheel builds for manylinux2014_x86_64, macosx_11_0_arm64, win_amd64
- `pip install jaxminlp` installs Python + Rust extension
- Tag-triggered PyPI release via `.github/workflows/release.yml`

**Verification:**
- CI passes on clean checkout for all platforms
- Coverage ≥85% on `benchmarks/`, `utils/` (`pyproject.toml:69`)
- All phase gate criteria from `benchmarks.toml:104-152` evaluable via automated workflow
- Regression detection fires on intentional 2x slowdown

**Critical files:**
- `jaxminlp_benchmarks/pyproject.toml` — test config, markers, coverage thresholds
- `jaxminlp_benchmarks/tests/test_correctness.py` — 24 known optima to activate
- `jaxminlp_benchmarks/tests/test_interop.py` — 14 interop test stubs to activate
- `jaxminlp_benchmarks/benchmarks/runner.py:194` — `_run_jaxminlp()` stub
- `jaxminlp_benchmarks/agents/lit_review.py:905-972` — workflow YAML to extract

---

## WS10: Advanced Algorithms & Release

**Phase:** 3-4 (Months 26-48) | **Owner:** Optimization researcher + JAX engineer

**Delivers:**
- **Piecewise McCormick** with adaptive partitioning (k=4-16 subintervals)
- **alphaBB relaxations** using `jax.hessian` for eigenvalue estimation
- **Convex envelopes** for trilinear, fractional, signomial terms
- **Quadratic convex reformulation** for MIQCQP
- **GNN branching** (`learned_branching.py`): bipartite graph representation, 2-3 message-passing layers, imitation learning on strong branching → RL on tree size
- **Learned node selection** (`learned_node_selection.py`): MLP scoring open nodes
- **Online fine-tuning** during solve
- **Symmetry detection** and orbital branching
- **Cutting planes**: RLT, gradient-based OA, lift-and-project
- **Documentation**: API docs, tutorials, mathematical ADRs
- **v1.0 release** under MIT/Apache 2.0

**Verification:**
- **Phase 3 gate:** `minlplib_30var_solved ≥ 75` (`benchmarks.toml:134`)
- **Phase 3 gate:** `geomean_vs_baron ≤ 2.5` (`benchmarks.toml:137`)
- **Phase 3 gate:** `gpu_class_vs_baron ≤ 1.0` on pooling (`benchmarks.toml:138`)
- **Phase 3 gate:** `learned_branching_improvement ≥ 0.20` node reduction (`benchmarks.toml:139`)
- **Phase 4 gate:** `minlplib_30var_solved ≥ 85` (`benchmarks.toml:146`)
- **Phase 4 gate:** `geomean_vs_baron ≤ 1.5` (`benchmarks.toml:148`)
- **Phase 4 gate:** `classes_faster_than_baron ≥ 2` (`benchmarks.toml:149`)
- **Phase 4 gate:** `zero_incorrect = 0` on full MINLPLib (`benchmarks.toml:152`)
- Piecewise McCormick (k=4) reduces relaxation gap by ≥60% vs standard on bilinear products
- GNN inference <0.1ms per branching decision
- alphaBB soundness: underestimator ≤ true value at 10,000 random points

---

## Dependency Graph & Parallelism

```
PHASE 1 (Months 1-14):
  WS1 (Rust infra + IR) ─────────────────────────┐
  WS2 (JAX compiler + McCormick) ─────────────────┤
  WS3 (LP + sparse LA) ──────────────────────┐    ├─► WS5 (B&B + batch dispatch) ─► Phase 1 Gate
  WS9 (CI/CD) ──────────────────────────────► │    │
  WS4 (Hybrid IPM) ──────────────────────────┘────┘

PHASE 2 (Months 14-26):
  WS6 (GPU batching) ─────────────────────────────┐
  WS7 (Bound tightening) ─────────────────────────┼─► Phase 2 Gate
  WS8 (LLM: from_description, explain, config) ───┘

PHASE 3-4 (Months 26-48):
  WS10 (Advanced algorithms + release) ──────────────► Phase 3 Gate → Phase 4 Gate
```

**Maximum parallelism by phase:**
- Phase 1: 5 streams active (WS1, WS2, WS3, WS4, WS9) — WS1+WS2 fully parallel, WS3 independent, WS4 starts month 6
- Phase 2: 3 streams active (WS6, WS7, WS8) — all independent after WS5 delivers batch dispatch
- Phase 3-4: WS10 subsumes multiple parallel tracks (relaxations, ML, cutting planes)

**Critical path:** WS1 + WS2 → WS5 → Phase 1 Gate → WS6 → Phase 2 Gate → WS10 → Phase 4 Gate

---

## Team Allocation (3-5 core)

| Role | Phase 1 | Phase 2 | Phase 3-4 |
|------|---------|---------|-----------|
| Rust Developer | WS1 → WS3 (LP) → WS5 (B&B) | WS7 (Rust presolve) | WS10 (cutting planes) |
| Numerical Specialist | WS3 (sparse LA) → WS4 (KKT) | WS4 (IPM complete) → WS7 (OBBT) | WS10 (advanced relaxations) |
| JAX/GPU Engineer | WS2 → WS4 (JAX eval) | WS6 → WS7 (JAX OBBT) | WS10 (learned branching, multi-GPU) |
| LLM Engineer | WS8 (safety infra) | WS8 (from_description, explain) | WS8 (chat, RAG) → WS10 (docs) |
| DevOps (part-time) | WS9 (CI setup) | WS9 (benchmark CI, GPU CI) | WS9 (release automation) |

---

## Existing Code to Reuse (do NOT rewrite)

| Component | File | Status | Lines |
|-----------|------|--------|-------|
| Modeling API + Expression DAG | `jaxminlp_api/core.py` | Complete | 1022 |
| Example models | `jaxminlp_api/examples.py` | Complete (7 runnable) | 602 |
| Benchmark metrics + phase gates | `benchmarks/metrics.py` | Complete | 664 |
| Benchmark runner | `benchmarks/runner.py` | Complete (stubs to wire) | 314 |
| Dolan-Moré profiles + statistics | `utils/statistics.py` | Complete | 212 |
| Report generation | `utils/reporting.py` | Complete | 302 |
| Pytest fixtures + markers | `tests/conftest.py` | Complete | 216 |
| Correctness test structure | `tests/test_correctness.py` | Structure ready (activate stubs) | 276 |
| Interop test structure | `tests/test_interop.py` | Structure ready (activate stubs) | 240 |
| Literature review agent | `agents/lit_review.py` | Production-ready | 1028 |
| Phase gate config | `config/benchmarks.toml` | Complete | 190 |
| CLI entry point | `run_benchmarks.py` | Complete (wire solver) | 198 |

---

## Phase Gate Checklist

### Phase 1 (Month 14) — requires WS1, WS2, WS3, WS4, WS5, WS9
- [ ] `minlplib_solved_count ≥ 25` (WS5)
- [ ] `nlp_convergence_rate ≥ 0.80` (WS4)
- [ ] `lp_netlib_pass_rate ≥ 0.95` (WS3)
- [ ] `lp_vs_highs_geomean ≤ 3.0` (WS3)
- [ ] `sparse_accuracy ≤ 1e-12` (WS3)
- [ ] `relaxation_valid = 1.0` (WS2 + WS5)
- [ ] `interop_overhead ≤ 0.05` (WS5)
- [ ] `zero_incorrect = 0` (WS5)

### Phase 2 (Month 26) — adds WS6, WS7, WS8
- [ ] `minlplib_30var_solved ≥ 55` (WS5 + WS7)
- [ ] `minlplib_50var_solved ≥ 25` (WS5 + WS7)
- [ ] `geomean_vs_couenne ≤ 3.0` (WS5 + WS7)
- [ ] `gpu_speedup ≥ 15.0` (WS6)
- [ ] `root_gap_vs_baron ≤ 1.3` (WS7)
- [ ] `node_throughput ≥ 200` (WS6)
- [ ] `rust_overhead ≤ 0.05` (WS5)
- [ ] `zero_incorrect = 0` (all)

### Phase 3 (Month 38) — adds WS10
- [ ] `minlplib_30var_solved ≥ 75` (WS10)
- [ ] `minlplib_50var_solved ≥ 45` (WS10)
- [ ] `minlplib_100var_solved ≥ 20` (WS10)
- [ ] `geomean_vs_baron ≤ 2.5` (WS10)
- [ ] `gpu_class_vs_baron ≤ 1.0` on pooling (WS10)
- [ ] `learned_branching_improvement ≥ 0.20` (WS10)
- [ ] `zero_incorrect = 0` on full MINLPLib (all)

### Phase 4 (Month 48) — release
- [ ] `minlplib_30var_solved ≥ 85`
- [ ] `minlplib_100var_solved ≥ 30`
- [ ] `geomean_vs_baron ≤ 1.5`
- [ ] `classes_faster_than_baron ≥ 2`
- [ ] `beats_couenne ≥ 1.0` (solve count ratio)
- [ ] `beats_bonmin ≥ 1.0` (solve count ratio)
- [ ] `zero_incorrect = 0` on full MINLPLib

---

## Verification Strategy

All verification is tied to existing infrastructure:
1. **Phase gates**: `python run_benchmarks.py --gate phaseN` uses `evaluate_phase_gate()` in `metrics.py:556` with criteria from `benchmarks.toml`
2. **Correctness**: 24 known optima in `test_correctness.py` + zero-incorrect invariant at every gate
3. **Interop**: 14 tests in `test_interop.py` define the Rust↔JAX contract
4. **Performance**: `gpu_vs_cpu_speedup()`, `shifted_geometric_mean()`, `performance_profile()` in `metrics.py`
5. **Regression**: `detect_regressions()` in `metrics.py:470` runs nightly against stored baseline
6. **Coverage**: `pytest --cov-fail-under=85` enforced in CI
7. **Soundness**: McCormick relaxation lower bound ≤ true value (the non-negotiable invariant tested at `test_interop.py:218-239`)
