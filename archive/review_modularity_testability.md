# Modularity & Testability Review

## JaxMINLP/discopt Development Plan

**Reviewer:** modularity-reviewer
**Date:** 2026-02-07
**Scope:** All 10 work streams (WS1-WS10), 21 implicit sub-tasks, mapping to existing test infrastructure

---

## Executive Summary

The development plan defines 10 work streams that decompose into roughly 21 deliverable units. Most work streams represent well-bounded modules with clear input/output contracts. However, the plan has several testability gaps: (1) existing test files are entirely stub-based with no mechanism to incrementally activate individual assertions, (2) several modules lack dedicated unit test files and rely solely on integration-level phase gates, (3) the WS9 testing infrastructure stream is treated as a support task rather than a prerequisite, creating a bootstrapping problem, and (4) mock/stub interfaces for cross-module development are not specified. The overall assessment is that the architecture is modular but the verification strategy is biased toward end-to-end validation rather than incremental unit-level testability.

---

## 1. Module Boundary Analysis

### WS1: Rust Infrastructure & Expression IR

**Boundary:** Cargo workspace (`jaxminlp-core` + `jaxminlp-python`) exposing PyO3 bindings to Python.

**Independently testable?** YES.

- The Rust crate is self-contained: `cargo test` and `cargo clippy` run without any JAX or Python dependency.
- PyO3 bindings can be tested via `python -c "from jaxminlp._rust import version"` without the solver existing.
- Expression IR round-trip (Rust evaluates parsed expression, matches Python within 1e-14) is a pure function test.
- `.nl` file parsing can be validated against the 24 MINLPLib instances independently.
- Structure detection (`is_linear`, `is_quadratic`, etc.) is purely deterministic on expression trees.

**Gap:** The plan lists `test_interop.py` as the primary test file, but those tests also cover WS5 (batch dispatch) and WS4 (NLP evaluation). There is no dedicated `test_rust_ir.py` for WS1's deliverables alone. The plan's WS9 mentions creating `test_api.py` but not a Rust-IR-specific test file.

**Recommendation:** Create `test_rust_ir.py` covering: expression construction, `.nl` parsing, structure detection, round-trip evaluation, and PyO3 binding availability. This can be fully activated at WS1 completion without any other work stream.

---

### WS2: JAX DAG Compiler & McCormick Relaxations

**Boundary:** Three Python modules (`dag_compiler.py`, `mccormick.py`, `relaxation_compiler.py`) that take Expression trees and produce JIT-compilable JAX functions.

**Independently testable?** YES, with partial dependency on WS1.

- The DAG compiler's input contract is the Expression class hierarchy in `core.py:60-400`, which already exists.
- McCormick relaxation primitives are pure mathematical functions; each can be tested with analytical ground truth (e.g., bilinear McCormick envelopes have closed-form expressions).
- The soundness invariant (`test_interop.py:218-239`) is the key correctness test. It only requires the relaxation compiler and the expression evaluator --- no Rust, no B&B.
- `jax.make_jaxpr()` and `jax.grad()` tests are self-contained.

**Gap:** The plan places the soundness test in `test_interop.py`, but it is not an interop test; it is a pure JAX test. The WS9 plan mentions `test_relaxation.py` but does not specify when it should be created or what it contains. The 14 `FunctionCall` types (exp, log, sin, etc.) need individual unit tests for their McCormick envelopes --- the plan does not enumerate these.

**Recommendation:** Create `test_relaxation.py` as a WS2 deliverable (not WS9), with one parametrized test per operation type for soundness, gradient correctness, and JIT traceability. Move the soundness invariant test from `test_interop.py` to `test_relaxation.py`.

---

### WS3: LP Solver & Sparse Linear Algebra

**Boundary:** Rust modules (`lp/`, `sparse/`) with PyO3 binding `solve_lp(...)`.

**Independently testable?** YES.

- LP solver correctness is validated against Netlib instances with known optima (`conftest.py:82-96`).
- Sparse factorization accuracy is validated against SuiteSparse (1e-12 residual).
- The `benchmarks.toml` suites `lp_netlib`, `lp_kennington`, and `sparse_matrices` are dedicated to WS3.
- No dependency on JAX, B&B, or NLP components.

**Gap:** The plan references WS9 creating `test_lp.py` and `test_sparse.py`, but these are logically WS3 deliverables. The Netlib optima fixture in `conftest.py` only has 7 instances; the plan requires pass_rate >= 0.95 on the full Netlib set (over 90 problems). The fixture needs expansion. Property-based testing with Hypothesis (random LP instances) is mentioned under WS9 but should be a WS3 acceptance criterion.

**Recommendation:** Make `test_lp.py` and `test_sparse.py` WS3 deliverables with WS9 responsible only for CI integration. Expand `netlib_optima` fixture to cover the full Netlib set.

---

### WS4: NLP Subsolver (Hybrid IPM)

**Boundary:** Split across JAX (`nlp_evaluator.py`) and Rust (`ipm/`), with a GPU-to-CPU-to-GPU transfer loop.

**Independently testable?** PARTIAL.

- The JAX evaluator (objective, gradient, Hessian, constraints, Jacobian) can be tested independently against finite differences.
- The Rust KKT factorization + iterative refinement can be tested independently against known indefinite matrices.
- The integration (JAX evaluates on GPU, Rust factorizes on CPU, search direction returns to GPU) cannot be tested without both WS2 (JAX expressions) and WS3 (sparse LDL^T) being available.

**Gap:** There is no `test_nlp.py` mentioned anywhere in the plan. The NLP convergence rate is only checked at the phase gate level (`nlp_convergence_rate >= 0.80` on CUTEst). This means NLP subsolver bugs would only be caught by the phase 1 gate rather than by targeted unit tests during development. The gradient/Hessian accuracy checks (finite difference comparison) are listed as verification criteria but not mapped to any test file.

**Recommendation:** Add `test_nlp_evaluator.py` (JAX side, testable with WS2) and `test_ipm.py` (Rust side, testable with WS3). These should be WS4 deliverables. The CUTEst convergence rate check remains a phase gate integration test.

---

### WS5: B&B Engine & Batch Dispatch

**Boundary:** Rust B&B engine with batch export/import interface, wired into `Model.solve()`.

**Independently testable?** PARTIAL.

- The Rust B&B engine (node pool, branching, pruning, incumbent management) can be unit-tested in Rust with a mock relaxation evaluator.
- The batch export/import interface (`export_batch()`, `import_results()`) can be tested with synthetic data via `test_interop.py`.
- The end-to-end `Model.solve()` integration requires WS1 + WS2 + WS3 + WS4 all working.

**Gap:** This is the integration point where the plan's module boundaries become blurry. WS5 is listed as requiring WS1, WS2, WS3, and optionally WS4, but the plan does not describe how to test B&B logic before all dependencies are ready. There is no mention of a mock relaxation evaluator for Rust-side B&B unit testing. The `test_interop.py` tests (14 methods) conflate array transfer (WS1), batch dispatch (WS5), and relaxation evaluation (WS2). The all-critical `test_correctness.py` requires a working end-to-end solver, meaning it cannot be incrementally activated as individual work streams complete.

**Recommendation:** (1) Add Rust-side B&B unit tests using a mock LP/NLP evaluator that returns known bounds for hand-crafted trees. (2) Split `test_interop.py` into `test_array_transfer.py` (WS1), `test_batch_dispatch.py` (WS5), and `test_relaxation_soundness.py` (WS2). (3) Define a "minimal viable solve" milestone where B&B + LP relaxation (no NLP, no McCormick) solves a pure MILP to validate the B&B engine before full NLP integration.

---

### WS6: GPU Batching & Performance

**Boundary:** JAX modules (`batch_evaluator.py`, `primal_heuristics.py`, `multi_gpu.py`).

**Independently testable?** YES, with dependency on WS2.

- Batch relaxation evaluation is testable once WS2's `compiled_relaxation` exists: apply `jax.vmap` and check shapes/values.
- Multi-start NLP is testable using the 7 example models from `examples.py`.
- GPU scaling benchmarks are isolated in `benchmarks.toml:92-97`.

**Gap:** The GPU-specific tests require actual GPU hardware. The `conftest.py` auto-skip hook handles this, but the plan does not define CPU-fallback functional tests that validate batch correctness without a GPU. The `gpu_vs_cpu_speedup()` function in `metrics.py:390` is the performance metric, but there is no correctness test that batch evaluation matches serial evaluation for all 24 known-optimum instances.

**Recommendation:** Add `test_batch_evaluator.py` that runs on CPU (JAX CPU backend) verifying functional correctness of batched evaluation vs serial. Performance tests remain GPU-only.

---

### WS7: Bound Tightening & Preprocessing

**Boundary:** Rust modules (`presolve/`, `obbt.rs`) + JAX module (`obbt.py`).

**Independently testable?** PARTIAL.

- FBBT is a purely Rust algorithm operating on variable bounds; fully testable in isolation with hand-crafted constraint sets.
- Probing, clique detection, big-M strengthening, redundant constraint removal --- all can be unit-tested on small examples.
- OBBT requires WS3 (LP solver) for dual simplex warm-starting.
- JAX-side gradient-based OBBT requires WS2 (relaxation compiler).

**Gap:** No test file mentioned for preprocessing. The root gap metric (`root_gap_vs_baron <= 1.3`) is a phase 2 gate criterion, but there are no unit tests for individual preprocessing techniques. The specific verification criteria (e.g., "FBBT tightens >= 2 bounds on ex1221", "big-M coefficient tightened to 50") are listed as prose but not mapped to test assertions.

**Recommendation:** Create `test_presolve.py` with parametrized tests for each preprocessing technique. The bound-tightening criteria from the plan (FBBT on ex1221, big-M strengthening, integer tightening) should be explicit test cases, not just verification prose.

---

### WS8: LLM Advisory Layer

**Boundary:** Python modules (`llm/safety.py`, `llm/provider.py`) + enhancements to `core.py`.

**Independently testable?** YES (for safety infrastructure); PARTIAL (for LLM-dependent features).

- Safety gates (`FormulationGate`, `ReformulationGate`, `ConfigurationGate`, `OutputSanitizer`) are pure validation logic; fully testable without an LLM.
- `from_description()` and `explain()` require LLM API access, making tests non-deterministic and API-dependent.

**Gap:** The plan specifies "100% test coverage for safety gates" but does not mention mocking the LLM provider for deterministic testing of `from_description()`. There is no strategy for testing LLM-dependent features in CI (rate limits, API keys, non-determinism). The "100-task formulation benchmark" in Phase 4 is an evaluation metric, not a test.

**Recommendation:** (1) Define a `MockLLMProvider` that returns canned responses for deterministic CI testing. (2) Create `test_safety_gates.py` as a WS8 Phase 1-2 deliverable. (3) Create `test_from_description.py` using the mock provider. (4) Keep the 100-task benchmark as a separate evaluation suite, not a pytest test.

---

### WS9: CI/CD & Testing Infrastructure

**Boundary:** GitHub Actions workflows, pytest configuration, wheel builds.

**Independently testable?** YES (infrastructure itself), but it is a dependency for all other streams.

**Gap:** This is the most significant structural issue in the plan. WS9 is treated as running in parallel with all other work streams, but it creates the test files that other streams need (`test_api.py`, `test_relaxation.py`, `test_lp.py`, `test_sparse.py`). This creates a bootstrapping problem: WS1-WS8 developers need test infrastructure from WS9, but WS9 is staffed as a part-time DevOps role. The plan lists "Months 10-14: Activate stubs" for replacing `NotImplementedError` catches, but this is too late --- WS1 and WS2 deliver in Months 1-10 and need dedicated tests immediately.

**Recommendation:** (1) Elevate the first phase of WS9 (Months 1-4) to a hard prerequisite for all other streams. (2) Move test file creation (`test_api.py`, `test_relaxation.py`, `test_lp.py`, `test_sparse.py`, etc.) into the respective work streams, making WS9 responsible only for CI pipeline configuration and the activation of existing stubs. (3) Define a "stub activation protocol" that allows incremental test activation: each WS replaces its own `pytest.skip("JaxMINLP not yet available")` calls as its module becomes available, rather than having a single big-bang activation in Months 10-14.

---

### WS10: Advanced Algorithms & Release

**Boundary:** Multiple sub-modules (piecewise McCormick, alphaBB, GNN branching, cutting planes, etc.).

**Independently testable?** PARTIAL.

- Piecewise McCormick and alphaBB are relaxation enhancements testable with the same soundness invariant as WS2.
- GNN branching requires a trained model, but inference can be tested with a randomly initialized network.
- Cutting planes (RLT, OA) can be unit-tested on small LP/QP instances.

**Gap:** WS10 is a catch-all for Phase 3-4 deliverables with no internal task decomposition in terms of testing. The plan lists 11 separate algorithms but treats them as a single stream. Each algorithm has different dependencies and different testability characteristics. There is no incremental test plan for WS10 sub-tasks.

**Recommendation:** Decompose WS10 into sub-tasks, each with its own test file and acceptance criteria. At minimum: `test_piecewise_mccormick.py`, `test_alphabb.py`, `test_gnn_branching.py`, `test_cutting_planes.py`.

---

## 2. Verification Criteria Completeness (WS1-WS10)

| Work Stream | Criteria Type | Sufficiency | Gap |
|---|---|---|---|
| WS1 | Build + round-trip + parse | Sufficient | No unit test file specified |
| WS2 | Soundness invariant + JIT + grad | Sufficient | Misplaced in `test_interop.py`; per-operation tests missing |
| WS3 | Netlib pass rate + residual | Sufficient | Netlib fixture too small (7 of 90+) |
| WS4 | CUTEst convergence | Insufficient | No unit tests for evaluator, no gradient accuracy tests |
| WS5 | Solved count + zero incorrect | Sufficient for integration | No B&B unit tests; relies on full stack |
| WS6 | GPU speedup + throughput | Sufficient for performance | No CPU-fallback correctness test for batching |
| WS7 | Root gap ratio | Insufficient | No unit tests for individual techniques |
| WS8 | Safety gate coverage | Partially sufficient | No LLM mock strategy; `from_description` untestable in CI |
| WS9 | CI passes + coverage | Sufficient | Bootstrapping problem with test file creation |
| WS10 | Phase 3-4 gate criteria | Insufficient | No per-algorithm criteria; single gate covers 11 algorithms |

---

## 3. Test Infrastructure Mapping

### Existing Test Files

| Test File | Current State | Maps To | Activation Point |
|---|---|---|---|
| `test_correctness.py` | 24 known-optimum stubs + edge cases | WS5 (end-to-end) | Requires WS1+WS2+WS3+WS5 minimum |
| `test_interop.py` | 14 interop stubs | WS1+WS2+WS5 (mixed) | Requires WS1 minimum for array tests |
| `conftest.py` | Fixtures + markers + hooks | All | Ready now |

### Missing Test Files (identified in plan but not yet created)

| Test File | Mentioned In | Needed By | Should Be Created By |
|---|---|---|---|
| `test_api.py` | WS9 | WS1 (for Expression DAG) | WS1 or WS9 Month 1 |
| `test_relaxation.py` | WS9 | WS2 | WS2 |
| `test_lp.py` | WS9 | WS3 | WS3 |
| `test_sparse.py` | WS9 | WS3 | WS3 |
| `test_nlp_evaluator.py` | Not mentioned | WS4 | WS4 |
| `test_ipm.py` | Not mentioned | WS4 | WS4 |
| `test_presolve.py` | Not mentioned | WS7 | WS7 |
| `test_safety_gates.py` | Not mentioned | WS8 | WS8 |
| `test_batch_evaluator.py` | Not mentioned | WS6 | WS6 |

---

## 4. Integration Seam Analysis

The plan defines four critical integration seams:

### Seam 1: Python Expression DAG -> Rust IR (WS1)
- **Interface:** `impl From<PyModel> for ModelRepr`
- **Mockable?** Yes. Python side can use the Expression DAG classes directly. Rust side can use hand-constructed `ModelRepr`.
- **Contract test:** Round-trip evaluation (Rust vs Python at same point). Well-defined.

### Seam 2: Expression DAG -> JAX Function (WS2)
- **Interface:** `compile_relaxation(expr, variables) -> JIT-compatible function`
- **Mockable?** Yes. Input is the existing Expression class hierarchy. Output is a JAX function.
- **Contract test:** Soundness invariant. Well-defined.

### Seam 3: Rust B&B -> JAX Batch Evaluation (WS5)
- **Interface:** `export_batch(N) -> (lb, ub, node_ids)` and `import_results(node_ids, bounds, solutions, feasible)`
- **Mockable?** Yes, but the plan does not define mocks in either direction. The Rust side could use a Python function returning constant bounds; the JAX side could receive synthetic bound arrays.
- **Contract test:** Shape/dtype assertions in `test_interop.py:56-92`. Well-defined but incomplete --- does not test correctness of the values, only shapes.

### Seam 4: JAX NLP Evaluator -> Rust IPM (WS4)
- **Interface:** GPU -> CPU transfer of Hessian/Jacobian, CPU -> GPU transfer of search direction.
- **Mockable?** Yes, with numpy arrays substituting for JAX arrays on CPU.
- **Contract test:** KKT residual < 1e-10 after refinement. Not mapped to a test file.

**Overall assessment:** The integration seams are architecturally well-defined, but mock implementations are not specified for any of them. Independent development requires mocks for each seam, and the plan should specify who creates them and when.

---

## 5. Incremental Validation Assessment

Can something new be tested after each task completes? Analysis by completion order:

| Task Completion | What Becomes Testable | Blocked On |
|---|---|---|
| WS9 (CI core, Month 1-4) | CI pipeline runs, `ruff`/`mypy` pass | Nothing |
| WS1 (Rust IR, Month 1-6) | Expression parsing, `.nl` loading, structure detection, PyO3 import | WS9 (CI) |
| WS2 (JAX compiler, Month 1-10) | Relaxation soundness, JIT tracing, gradient correctness | Nothing (uses existing Expression classes) |
| WS3 (LP, Month 3-12) | Netlib LP solving, sparse factorization accuracy | WS1 (for PyO3 bindings) |
| WS4 (NLP, Month 6-20) | Gradient/Hessian accuracy, KKT convergence | WS2 (JAX evaluator) + WS3 (sparse LDL^T) |
| WS5 (B&B, Month 8-18) | End-to-end solve on known optima, batch dispatch | WS1 + WS2 + WS3 |
| WS7 (Presolve, Month 10-22) | FBBT/probing on small examples | WS1 (expression IR) |
| WS6 (GPU, Month 14-26) | Batch evaluation speedup, multi-start heuristics | WS2 + WS5 |
| WS8 (LLM, Month 10-48) | Safety gates (immediate), `from_description` (after WS5) | Nothing (for safety), WS5 (for integration) |
| WS10 (Advanced, Month 26-48) | Per-algorithm soundness, learned branching | WS2 + WS5 + WS6 |

**Verdict:** The plan supports incremental validation, but only if dedicated test files are created per-work-stream (see Section 3). With the current test infrastructure (only `test_correctness.py` and `test_interop.py`), meaningful testing does not begin until WS5 completes, which is Month 18 --- that is 37% of the project timeline with no automated correctness validation.

---

## 6. Per-Task Compliance Matrix

| Task | Module Boundary | Dedicated Tests | Acceptance Criteria | Dependencies | Independently Validatable? |
|---|---|---|---|---|---|
| **WS1**: Rust IR + PyO3 | Clear: Cargo crate + bindings | `test_interop.py` (partial), needs `test_rust_ir.py` | Build + parse + round-trip + structure detect | None | **Yes** |
| **WS2**: DAG compiler | Clear: 3 JAX modules | `test_interop.py:218-239` (misplaced), needs `test_relaxation.py` | Soundness + JIT + grad + vmap | Expression classes (exist) | **Yes** |
| **WS3**: LP solver | Clear: Rust `lp/` + `sparse/` | Needs `test_lp.py`, `test_sparse.py` | Netlib pass rate + residual + warm-start | WS1 (for bindings) | **Yes** (Rust-only tests) / **Partial** (Python bindings) |
| **WS4**: NLP evaluator (JAX) | Clear: `nlp_evaluator.py` | None | Gradient/Hessian accuracy, JIT warmup | WS2 (expressions) | **Partial** |
| **WS4**: IPM (Rust) | Clear: `ipm/` | None | KKT residual, inertia correction | WS3 (sparse factorization) | **Partial** |
| **WS4**: Hybrid integration | Blurry: GPU-CPU-GPU loop | None | CUTEst convergence rate | WS2 + WS3 | **No** |
| **WS5**: B&B engine (Rust) | Clear: `bnb/` | None (Rust-side) | Node selection, pruning, branching | None (with mock evaluator) | **Yes** (with mock) / **No** (without) |
| **WS5**: Batch dispatch | Clear: `export_batch`/`import_results` | `test_interop.py:56-92` | Shape/dtype, zero-copy, latency | WS1 (bindings) | **Partial** |
| **WS5**: `Model.solve()` | Integration point | `test_correctness.py` | 24 known optima, zero incorrect | WS1+WS2+WS3+WS4 | **No** |
| **WS6**: Batch evaluator | Clear: `batch_evaluator.py` | None | GPU speedup, memory linearity | WS2 (relaxations) | **Yes** (correctness) / **Partial** (performance, needs GPU) |
| **WS6**: Primal heuristics | Clear: `primal_heuristics.py` | None | Multi-start finds optimum | WS2 + WS4 | **Partial** |
| **WS6**: Multi-GPU | Clear: `multi_gpu.py` | None | 4-GPU speedup | WS6 batch evaluator | **No** (needs hardware) |
| **WS7**: FBBT | Clear: `presolve/fbbt.rs` | None | Bound tightening on ex1221 | WS1 (expression IR) | **Yes** |
| **WS7**: OBBT | Clear: `presolve/obbt.rs` | None | LP-based tightening, warm-start | WS3 (LP solver) | **Partial** |
| **WS7**: JAX OBBT | Clear: `obbt.py` | None | Gradient-based prioritization | WS2 (relaxation grad) | **Partial** |
| **WS8**: Safety gates | Clear: `llm/safety.py` | None | 100% pass/reject coverage | None | **Yes** |
| **WS8**: `from_description` | Coupled: `core.py` + LLM | None | `Model.validate()` called | WS5 (working solver) | **No** (needs LLM + solver) |
| **WS8**: `explain()` | Coupled: `core.py` + LLM | None | Sanitized output | WS5 (working solver) | **Partial** (with mock LLM) |
| **WS9**: CI pipelines | Clear: `.github/workflows/` | Meta: it IS the test infra | CI green on clean checkout | All other WS deliverables | **Yes** (pipeline itself) |
| **WS10**: Relaxation improvements | Clear: per-algorithm modules | None | Soundness + gap reduction | WS2 | **Yes** (soundness tests) |
| **WS10**: Learned branching | Clear: `learned_branching.py` | None | Node reduction >= 20% | WS5 (B&B data) | **Partial** |

**Summary counts:**
- Fully independently validatable: 7 of 21 sub-tasks
- Partially independently validatable: 9 of 21 sub-tasks
- Not independently validatable: 5 of 21 sub-tasks

---

## 7. Key Findings and Recommendations

### Finding 1: Test files are assigned to the wrong work stream

The plan assigns creation of `test_api.py`, `test_relaxation.py`, `test_lp.py`, and `test_sparse.py` to WS9 (CI/CD), but these are acceptance-test deliverables for WS1, WS2, and WS3 respectively. This creates a dependency inversion: WS1-WS3 developers cannot validate their own work without WS9 creating their test files.

**Recommendation:** Each work stream should own its test files. WS9 should own CI pipeline configuration, coverage enforcement, and marker-based test selection only.

### Finding 2: No mock strategy for cross-module development

The plan does not specify mock interfaces for any of the four integration seams. This means that work streams cannot be developed and tested in true isolation. For example, WS5 (B&B) cannot be tested without a real relaxation evaluator from WS2.

**Recommendation:** Define mock interfaces as explicit deliverables:
- `MockRelaxationEvaluator` (for WS5 to test B&B without WS2)
- `MockLPSolver` (for WS4 IPM tests, WS7 OBBT tests)
- `MockLLMProvider` (for WS8 tests)
- `MockTreeManager` (for WS6 to test batch evaluation without WS5)

### Finding 3: The existing test stubs create a testing cliff

Both `test_correctness.py` and `test_interop.py` use a binary pattern: every test either raises `NotImplementedError` and skips, or runs fully. There is no mechanism for partial activation. When WS1 delivers, there is no way to activate only the array-transfer tests while keeping B&B tests skipped.

**Recommendation:** Refactor the skip conditions to be module-granular:
```python
# Instead of one catch-all:
try:
    sol = solve_instance(...)
except NotImplementedError:
    pytest.skip("JaxMINLP not yet available")

# Use module-specific availability checks:
pytest.importorskip("jaxminlp._rust")  # For WS1 tests
pytest.importorskip("jaxminlp._jax.dag_compiler")  # For WS2 tests
```

### Finding 4: WS4 (NLP Subsolver) has no dedicated test plan

WS4 is the only work stream with zero test files mentioned (existing or planned). Its only verification is the CUTEst convergence rate at the phase 1 gate. Given that NLP solvers are notoriously sensitive to numerical issues, this is a significant testability gap.

**Recommendation:** Add `test_nlp_evaluator.py` and `test_ipm.py` as WS4 deliverables, including:
- Finite-difference gradient checks at 50 points per expression
- Finite-difference Hessian checks at 20 points
- KKT residual convergence on 5 hand-crafted systems
- Inertia correction on deliberately indefinite matrices

### Finding 5: WS10 needs internal decomposition

WS10 contains 11 distinct algorithms but has no internal task breakdown for testing. The only verification criteria are Phase 3-4 gate metrics, which aggregate all algorithms into 4-5 numbers. A bug in one algorithm could be masked by another algorithm's success.

**Recommendation:** Decompose WS10 into at least 4 testable sub-tasks with dedicated test files and per-algorithm acceptance criteria.

### Finding 6: Phase gate criteria are necessary but not sufficient for incremental validation

The `benchmarks.toml` phase gates evaluate aggregated metrics (solve counts, geometric means, pass rates). These are excellent for milestone decisions but poor for developer feedback during implementation. A developer working on WS3 (LP solver) needs to know immediately if their solver returns a wrong answer on `afiro`, not wait for a phase gate evaluation.

**Recommendation:** Supplement phase gates with fine-grained per-instance assertion tests. The `test_correctness.py` structure already supports this (parametrized over instances), but needs to be activatable per-module rather than all-or-nothing.

---

## 8. Overall Assessment

The development plan's **architecture is well-modularized** --- work streams correspond to real module boundaries with clear input/output contracts. The **dependency graph is largely correct** --- Phase 1 streams are properly parallelized, and the critical path (WS1+WS2 -> WS5 -> Phase 1 Gate) reflects real technical dependencies.

However, the **testability infrastructure has significant gaps**. The plan relies too heavily on end-to-end phase gate validation and underinvests in unit-level testing per work stream. The assignment of test file creation to WS9 rather than individual work streams creates a bottleneck. The absence of mock interfaces prevents true independent development.

**If the six recommendations above are adopted**, every work stream can deliver an independently testable module with dedicated acceptance tests, enabling continuous validation from Month 1 rather than deferring meaningful testing to Month 18 when end-to-end solving first becomes possible.
