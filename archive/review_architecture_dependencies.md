# Architecture & Dependency Graph Review

**Reviewer:** arch-reviewer
**Date:** 2026-02-07
**Documents Reviewed:**
- `reports/jaxminlp_development_plan.md` (Work streams WS1-WS10, dependency graph)
- `reports/feasibility_assessment.md` (Revised timeline, external library strategy)
- `JAX_OPTIMIZATION_ECOSYSTEM_VISION.md` (Phasing strategy)
- `jaxminlp_benchmarks/config/benchmarks.toml` (Phase gate criteria)
- `jaxminlp_benchmarks/tests/test_interop.py` (PyO3 API surface)
- `jaxminlp_benchmarks/tests/test_correctness.py` (Known optima)
- `jaxminlp_benchmarks/benchmarks/metrics.py` (SolveResult fields)

---

## 1. Dependency Graph Correctness

### 1.1 Stated Dependencies

The development plan defines this dependency graph:

```
PHASE 1:
  WS1 (Rust IR) ──────────────────────┐
  WS2 (JAX compiler + McCormick) ─────┤
  WS3 (LP + sparse LA) ──────────┐    ├──> WS5 (B&B + batch dispatch)
  WS9 (CI/CD) ──────────────────> │    │
  WS4 (Hybrid IPM) ──────────────┘────┘

PHASE 2:
  WS6 (GPU batching)
  WS7 (Bound tightening)
  WS8 (LLM advisory)

PHASE 3-4:
  WS10 (Advanced algorithms + release)
```

### 1.2 Confirmed Correct Dependencies

**WS5 depends on WS1, WS2, WS3:** Correct. The B&B engine requires:
- WS1 (Rust IR): The `TreeManager.export_batch()` interface operates on `ModelRepr`, the Rust-side expression graph. Without the Rust workspace and expression IR, there is no model representation for the B&B engine to operate on.
- WS2 (JAX compiler): The batch dispatch loop calls `jax.vmap(compiled_relaxation)` on the exported bounds. This requires the DAG-to-JAX compiler and McCormick relaxation primitives.
- WS3 (LP/sparse LA): Node relaxation solving requires LP capabilities. In the original plan, this is the custom Rust simplex; in the revised plan, this is HiGHS.

**WS4 depends on WS3:** Correct. The hybrid IPM uses sparse LDL^T factorization from WS3 for KKT systems. The NLP evaluator (JAX side) is independent, but the Rust-side KKT factorization depends on sparse linear algebra.

**WS6 depends on WS5:** Correct. GPU batching is an optimization of the batch dispatch mechanism that WS5 delivers. Without a working B&B loop, there is nothing to batch.

**WS7 partially depends on WS3 and WS5:** Correct. OBBT requires LP solving (WS3 or HiGHS). FBBT is independent and could start earlier. The plan correctly notes WS7 starts at Month 10, after WS3 delivers LP capability.

**WS10 depends on WS5, WS6, WS7:** Correct. Advanced algorithms (piecewise McCormick, learned branching, cutting planes) all require a working solver with GPU batching and preprocessing.

### 1.3 Issues Found

#### ISSUE 1: WS5's dependency on WS4 is ambiguous (MEDIUM severity)

The dependency graph shows WS4 feeding into WS5, but the relationship is not clearly stated. WS5 (B&B) needs relaxation solving at each node. For *convex* MINLP, the node subproblem is an NLP, which requires WS4. For spatial B&B with McCormick relaxations, the node subproblem is an LP (after McCormick linearization), which requires WS3 but *not* WS4.

The plan text for WS5 (line 139) says "End-to-end `Model.solve()`" which implies the full MINLP solver. But the Phase 1 gate only requires 25 MINLPLib instances solved. Most small MINLPLib instances can be solved with spatial B&B + LP relaxations alone (McCormick).

**Finding:** WS5 does NOT strictly require WS4 for Phase 1. The spatial B&B with McCormick relaxations solves LP relaxations at each node (WS3 dependency), not NLP subproblems. WS4 (NLP subsolver) is needed for NLP heuristics (finding feasible solutions) and for problems without McCormick structure, but these are Phase 2 concerns.

**Recommendation:** Make the dependency explicit: WS5-basic (Phase 1 gate) depends on WS1 + WS2 + WS3. WS5-complete (with NLP heuristics) depends on WS4. This removes WS4 from the Phase 1 critical path.

#### ISSUE 2: WS2 has an undeclared soft dependency on WS1 (LOW severity)

WS2 (JAX DAG compiler) compiles the Python `Expression` tree to JAX callables. This operates on the Python-side expression DAG from `core.py`, NOT on the Rust IR. So WS2 does not depend on WS1 for compilation. However, the verification criterion "All 24 MINLPLib instances from test_correctness.py parse from .nl without error" is listed under WS1, and the `.nl` parser outputs `ModelRepr` in Rust. If WS2 also needs to handle `.nl`-parsed models, it implicitly depends on WS1's `.nl` parser providing a Python-accessible model.

**Finding:** The plan correctly identifies WS1 and WS2 as parallel. The `.nl` parsing path goes through Rust (WS1), but the DAG compiler (WS2) works on the Python expression DAG, which can be constructed programmatically via the modeling API. No circular dependency. The two connect at WS5 integration time.

#### ISSUE 3: WS7 (Bound Tightening) has stronger WS3 dependency than shown (MEDIUM severity)

WS7 OBBT requires solving many LP subproblems (one per variable per bound direction). The plan specifies dual simplex warm-starting for OBBT efficiency (line 203: "Dual simplex warm-start reduces OBBT LP iterations by >= 30%"). This creates a tight coupling to the LP solver implementation:

- If using the custom Rust LP (original WS3), warm-starting is tightly integrated.
- If using HiGHS (feasibility assessment recommendation), warm-starting is available via HiGHS's API but the interface is different.

**Finding:** The dependency is correctly identified but the integration path changes depending on whether WS3 delivers a custom LP or uses HiGHS. Under the revised plan (use HiGHS), WS7 depends on the HiGHS integration, which is simpler than building a custom LP but requires careful warm-start API usage.

#### ISSUE 4: No circular dependencies detected (CONFIRMED)

I checked all 10 work streams for circular dependencies. None exist. The dependency graph is a DAG with clear phase ordering.

---

## 2. Critical Path Analysis

### 2.1 Stated Critical Path

The plan identifies: `WS1 + WS2 -> WS5 -> Phase 1 Gate -> WS6 -> Phase 2 Gate -> WS10 -> Phase 4 Gate`

### 2.2 Analysis

**Under the original plan (build everything):** The critical path is actually longer than stated because WS3 (LP solver) is also on the critical path to WS5. The true critical path is:

```
WS3 (LP + sparse, months 3-12) -> WS4 (IPM, months 6-20) -> WS5 (B&B, months 8-18) -> Phase 1 Gate (month 14)
```

WS3 starts at month 3 and takes 9 months. WS5 cannot start until month 8 (when WS3 delivers basic LP) and WS5 itself takes 10 months. This puts Phase 1 gate at month 18, not month 14 as stated. The plan claims WS5 starts at month 8, but if WS3 delivers at month 12 (end of Phase 1), there is a contradiction.

**Under the revised plan (use HiGHS + Ipopt):** The critical path shortens dramatically:

```
WS1 (Rust IR, months 1-3) + WS2 (JAX compiler, months 2-5) -> WS5 (B&B with HiGHS, months 4-8) -> Phase 1 Gate (month 8-12)
```

WS3 (custom LP) is removed from the critical path. HiGHS provides LP capability immediately. The critical path is now WS2 (DAG compiler + McCormick, 5 months) -> WS5 (B&B integration, 4 months) = ~9 months total. This aligns with the feasibility assessment's 8-12 month estimate.

### 2.3 Hidden Bottlenecks

**Bottleneck 1: WS1 (Rust workspace) is a serial bottleneck.** Every Rust-side component (B&B tree, LP solver, presolve) lives in the same Cargo workspace. The initial maturin + PyO3 setup, CI configuration for cross-platform Rust builds, and `.pyi` type stubs must be done once by one person before any Rust development can begin. This is approximately 2-3 weeks of focused work.

**Recommendation:** WS1's Rust workspace setup should be the absolute first task. Nothing else in the Rust domain can start without it.

**Bottleneck 2: Phase 2 gate depends on GPU availability.** The Phase 2 gate requires `gpu_speedup >= 15.0` (benchmarks.toml:124) and `node_throughput >= 200` (benchmarks.toml:126). These cannot be measured without A100/H100 GPU access. Consumer GPUs (RTX series) have 1/32 float64 throughput, making it impossible to hit these targets on developer hardware.

**Recommendation:** GPU compute access should be secured before Phase 2 begins (month 14 in the original timeline, month 12 in the revised timeline). This is an operational dependency, not a code dependency, but it can block the gate.

**Bottleneck 3: The `.nl` file parser is on the testing critical path.** All 24 `KNOWN_OPTIMA` instances in `test_correctness.py` are MINLPLib problems in `.nl` format. Without the `.nl` parser (WS1), the correctness test suite cannot run. The modeling API can construct some test problems programmatically, but the MINLPLib validation requires `.nl` parsing.

**Recommendation:** Prioritize the `.nl` parser within WS1. It can be implemented as a standalone Rust module before the full expression IR is complete. Alternatively, a Python-side `.nl` parser (using the `amplpy` or `pyomo` `.nl` reader) could serve as a temporary bridge.

---

## 3. Task List Dependency Mapping

The development plan describes 21 implied tasks across Week 1-12 and Phases 2-4 (inferred from the Phase Gate Checklist at lines 388-427 and the Team Allocation table at lines 357-365). The current task tracking system has 3 review tasks (#1-#3), not the 21 implementation tasks.

### 3.1 Mapping Work Streams to Implementation Tasks

The Phase Gate Checklist (lines 388-427) maps tasks to work streams correctly:

| Phase 1 Gate Criterion | Work Stream | Dependency Chain |
|----------------------|-------------|------------------|
| `minlplib_solved_count >= 25` | WS5 | WS1 + WS2 + WS3 -> WS5 |
| `nlp_convergence_rate >= 0.80` | WS4 | WS3 -> WS4 |
| `lp_netlib_pass_rate >= 0.95` | WS3 | Independent |
| `lp_vs_highs_geomean <= 3.0` | WS3 | Independent |
| `sparse_accuracy <= 1e-12` | WS3 | Independent |
| `relaxation_valid = 1.0` | WS2 + WS5 | WS1 + WS2 -> WS5 |
| `interop_overhead <= 0.05` | WS5 | WS1 -> WS5 |
| `zero_incorrect = 0` | WS5 | All Phase 1 WS |

This mapping is internally consistent.

### 3.2 Issue: Phase 1 Gate Criteria Conflict With Revised Plan

**ISSUE 5 (HIGH severity):** The Phase 1 gate includes criteria that the feasibility assessment recommends deferring:

- `lp_netlib_pass_rate >= 0.95` (benchmarks.toml:110) -- This evaluates the *custom* Rust LP solver (WS3) against Netlib. If using HiGHS for Phase 1, this criterion measures HiGHS performance, which is guaranteed to pass but does not validate any code the team wrote.
- `lp_vs_highs_geomean <= 3.0` (benchmarks.toml:111) -- This compares the custom LP solver to HiGHS. If the team *uses* HiGHS as the LP solver, this comparison is meaningless (ratio = 1.0 trivially).
- `sparse_accuracy <= 1e-12` (benchmarks.toml:112) -- This evaluates the custom sparse factorization (WS3) against SuiteSparse. If using Ipopt's MUMPS for sparse LA, this criterion tests MUMPS, not custom code.
- `nlp_convergence_rate >= 0.80` (benchmarks.toml:109) -- This evaluates the custom hybrid IPM (WS4). If using Ipopt for Phase 1 NLP, this criterion tests Ipopt with JAX callbacks. It validates the JAX evaluation layer but not a custom solver.

**Finding:** The Phase 1 gate criteria in `benchmarks.toml` were designed for the "build everything from scratch" plan (original WS3 scope). Under the revised plan (use HiGHS + Ipopt), four of eight Phase 1 criteria either become trivially satisfied or measure external library performance rather than custom code.

**Recommendation:** If the HiGHS+Ipopt strategy is adopted, the Phase 1 gate criteria should be revised to:
1. Keep: `minlplib_solved_count >= 25`, `relaxation_valid = 1.0`, `interop_overhead <= 0.05`, `zero_incorrect = 0` (these validate the team's code)
2. Revise: `lp_netlib_pass_rate` and `lp_vs_highs_geomean` should be deferred to a "custom LP" milestone in Phase 2-3
3. Revise: `sparse_accuracy` should be deferred to Phase 2-3 when custom sparse LA is built
4. Keep with caveat: `nlp_convergence_rate >= 0.80` validates the JAX evaluator integration even with Ipopt, so it has value, but the gate description should note that this tests JAX+Ipopt, not a custom solver

---

## 4. Architecture Consistency: HiGHS+Ipopt Alignment

### 4.1 Does the Feasibility Assessment's Recommendation Align With the Task List?

The feasibility assessment recommends a revised Phase 1 approach (Section 4.5, lines 265-276):

```
Rust workspace + PyO3 (2-3mo) ------+
DAG-to-JAX compiler (2-4mo) --------+--> B&B engine + HiGHS/Ipopt integration (3-4mo)
McCormick relaxations (2-4mo) ------+
```

This maps to work streams: WS1 (subset) + WS2 + WS5 (with HiGHS/Ipopt). WS3 and WS4 are deferred.

**Alignment assessment:**

| Plan Element | Development Plan | Feasibility Assessment | Aligned? |
|-------------|-----------------|----------------------|----------|
| Phase 1 LP solver | Custom Rust simplex (WS3) | HiGHS (highspy) | **NO** -- contradictory |
| Phase 1 NLP solver | Custom hybrid IPM (WS4) | Ipopt (cyipopt) | **NO** -- contradictory |
| Phase 1 sparse LA | Custom Rust LU/Cholesky (WS3) | Ipopt's MUMPS | **NO** -- contradictory |
| Phase 1 B&B engine | Custom Rust (WS5) | Custom Rust (WS5) | YES |
| Phase 1 DAG compiler | JAX (WS2) | JAX (WS2) | YES |
| Phase 1 McCormick | JAX (WS2) | JAX (WS2) | YES |
| Phase 1 Rust IR | Rust (WS1) | Rust (WS1) | YES |
| Phase 1 gate criteria | Custom LP/NLP metrics | Unchanged | **NO** -- criteria don't match revised approach |

**Finding:** The development plan and feasibility assessment are fundamentally misaligned on Phase 1 strategy. The development plan assumes building LP, NLP, and sparse LA from scratch. The feasibility assessment recommends using external libraries for Phase 1 and building custom implementations later. These are two different plans with different timelines, staffing needs, and gate criteria.

### 4.2 Impact on Work Stream Definitions

If the HiGHS+Ipopt strategy is adopted:

- **WS3 (LP + sparse LA):** Scope shrinks drastically in Phase 1. Becomes "integrate HiGHS via highspy" (2-4 weeks, not 9 months). Custom LP development moves to Phase 2-3.
- **WS4 (Hybrid IPM):** Phase 1 scope becomes "build JAX NLPEvaluator + integrate with Ipopt callbacks" (2-3 months). Custom IPM moves to Phase 2.
- **WS5 (B&B):** Unchanged in scope. Uses HiGHS for node LP relaxations and Ipopt for NLP heuristics instead of custom implementations. The `Model.solve()` integration point is the same.
- **WS1, WS2, WS9:** Unchanged.
- **WS6, WS7:** Delayed impact. OBBT warm-starting (WS7) must use HiGHS's API instead of custom dual simplex. GPU batching (WS6) still requires a custom JAX IPM.
- **WS8:** Feasibility assessment recommends cutting entirely for Phase 1-2.

**Recommendation:** The two documents need reconciliation. Either:
1. Adopt the HiGHS+Ipopt strategy formally, update WS3/WS4 scope and Phase 1 gate criteria accordingly, OR
2. Retain the build-from-scratch plan but accept the longer Phase 1 timeline (14-18 months)

The feasibility assessment's approach is clearly more pragmatic. If adopted, the development plan's WS3 and WS4 descriptions should be rewritten to reflect the phased approach (external libraries first, custom implementations second).

---

## 5. Phase Gate Alignment

### 5.1 Phase 1 Gate (benchmarks.toml:104-115)

| Criterion | benchmarks.toml Key | Value | Plan WS | Validates |
|-----------|-------------------|-------|---------|-----------|
| Solved count | `minlplib_solved_count` | >= 25 | WS5 | End-to-end solver works |
| NLP convergence | `nlp_convergence_rate` | >= 0.80 | WS4 | NLP subsolver works |
| LP Netlib pass | `lp_netlib_pass_rate` | >= 0.95 | WS3 | LP solver works |
| LP vs HiGHS | `lp_vs_highs_geomean` | <= 3.0 | WS3 | LP performance |
| Sparse accuracy | `sparse_accuracy` | <= 1e-12 | WS3 | Sparse factorization |
| Relaxation valid | `relaxation_valid` | = 1.0 | WS2+WS5 | McCormick soundness |
| Interop overhead | `interop_overhead` | <= 0.05 | WS5 | Rust/JAX boundary |
| Zero incorrect | `zero_incorrect` | = 0 | WS5 | Correctness invariant |

**Assessment:** Under the original plan, all 8 criteria correctly map to their work streams and test real deliverables. Under the revised plan, 4 criteria need revision (see Issue 5 above).

### 5.2 Phase 2 Gate (benchmarks.toml:117-128)

| Criterion | Value | Plan WS | Issue |
|-----------|-------|---------|-------|
| 30-var solved >= 55 | WS5+WS7 | OK |
| 50-var solved >= 25 | WS5+WS7 | OK |
| geomean vs Couenne <= 3.0 | WS5+WS7 | OK |
| GPU speedup >= 15.0 | WS6 | OK -- but requires A100/H100 hardware |
| Root gap vs BARON <= 1.3 | WS7 | **AGGRESSIVE** -- feasibility assessment flags this as very aggressive for 26 months |
| Node throughput >= 200 | WS6 | OK |
| Rust overhead <= 0.05 | WS5 | OK |
| Zero incorrect = 0 | All | OK |

**ISSUE 6 (MEDIUM severity):** The Phase 2 criterion `root_gap_vs_baron <= 1.3` is flagged by the feasibility assessment (Section 2.2, line 60) as "very aggressive for 26 months of development." BARON has 25+ years of specialized relaxation rules, adaptive partitioning, and convex envelopes. The plan's WS7 (bound tightening) delivers FBBT + OBBT, which is necessary but likely insufficient to close the gap to 1.3x BARON. This criterion may block the Phase 2 gate even if all other criteria pass.

**Recommendation:** Consider softening to `root_gap_vs_baron <= 2.0` for Phase 2 and tightening to 1.3 for Phase 3 when WS10 (advanced relaxations, piecewise McCormick, alphaBB) contributes.

### 5.3 Phase 3 Gate (benchmarks.toml:130-140)

| Criterion | Value | Plan WS | Issue |
|-----------|-------|---------|-------|
| 30-var solved >= 75 | WS10 | OK |
| 50-var solved >= 45 | WS10 | OK |
| 100-var solved >= 20 | WS10 | OK -- but Tier 3 sparse LA may be needed |
| geomean vs BARON <= 2.5 | WS10 | Ambitious but plausible with advanced relaxations |
| GPU class vs BARON <= 1.0 on pooling | WS10 | OK -- pooling is GPU-amenable |
| Learned branching >= 20% | WS10 | OK -- 20% node reduction is validated by literature |
| Zero incorrect = 0 on full MINLPLib | All | OK |

**Assessment:** Phase 3 criteria are appropriately ambitious. The `gpu_class_vs_baron <= 1.0` on pooling is the most achievable "beat BARON" target because pooling problems are bilinear-dominated and GPU-amenable. This is well-chosen.

### 5.4 Phase 4 Gate (benchmarks.toml:142-152)

| Criterion | Value | Issue |
|-----------|-------|-------|
| 30-var solved >= 85 | OK |
| 100-var solved >= 30 | OK |
| geomean vs BARON <= 1.5 | **UNREALISTIC** per feasibility assessment |
| Classes faster than BARON >= 2 | Plausible for GPU-amenable classes |
| Beats Couenne >= 1.0 | OK |
| Beats Bonmin >= 1.0 | OK |
| Zero incorrect = 0 on full MINLPLib | OK |

**ISSUE 7 (HIGH severity):** The feasibility assessment (Section 5.4, lines 354-361) rates the likelihood of matching BARON within 1.5x on general MINLP at 15-25%, estimating 60-84+ months instead of 48. The Phase 4 criterion `geomean_vs_baron <= 1.5` may be unachievable on the stated timeline.

**Recommendation:** Revise to `geomean_vs_baron <= 3.0` for general MINLP plus `gpu_class_vs_baron <= 1.0` for 3+ specific problem classes. This targets the areas where JaxMINLP has structural advantages (batch-amenable, dense nonlinear) rather than all of BARON's territory.

---

## 6. Cross-Document Consistency Issues

### 6.1 Ecosystem Vision vs Development Plan

The ecosystem vision document describes a layered package structure (`jax-optcore`, `jax-lp`, `jax-milp`, `jax-miqp`, `jax-minlp`) with Phase 2 extraction of `jax-optcore`. The development plan does not mention this extraction anywhere. WS10 mentions "Release" but not ecosystem decomposition.

**Finding:** The ecosystem expansion is not captured in any work stream. If `jax-optcore` extraction is a Phase 2 deliverable, it needs to be added to WS9 or a new work stream. If deferred (as the feasibility assessment recommends), the ecosystem vision's Phase 2 scope is overcommitted.

**Recommendation:** Add an explicit "ecosystem extraction" task to Phase 3 or later. Phase 2 should focus exclusively on GPU batching and bound tightening, not package restructuring.

### 6.2 Team Allocation vs Feasibility Assessment

The development plan allocates 5 roles across Phase 1 (lines 357-365):
- Rust Developer: WS1 -> WS3 -> WS5
- Numerical Specialist: WS3 -> WS4
- JAX/GPU Engineer: WS2 -> WS4
- LLM Engineer: WS8
- DevOps: WS9

The feasibility assessment recommends 2-3 core developers for Phase 1 and cutting WS8 (LLM) entirely. This means:
- The Rust Developer and JAX/GPU Engineer are essential
- The Numerical Specialist role is absorbed (HiGHS/Ipopt replace custom LP/NLP)
- The LLM Engineer is not needed
- DevOps can be part-time

**Finding:** The team allocation table in the development plan does not reflect the feasibility assessment's staffing recommendation. If 2-3 developers are realistic, the work stream parallelism in Phase 1 (5 streams active) is overstated.

---

## 7. Summary of Findings

### Confirmed Correct

1. The WS5 -> WS1 + WS2 + WS3 dependency chain is correct and necessary.
2. No circular dependencies exist in the work stream graph.
3. Phase gate criteria correctly map to work streams (under the original plan).
4. The Phase 2 and Phase 3 gate criteria are appropriately structured with increasing difficulty.
5. The critical path identification (WS1+WS2 -> WS5 -> WS6 -> WS10) is correct for the revised plan.
6. The `zero_incorrect = 0` invariant at every phase gate is correctly enforced and non-negotiable.
7. `benchmarks.toml` is self-consistent -- suite references, metric names, and threshold values all form a coherent evaluation framework.

### Issues Requiring Resolution

| # | Severity | Finding | Section |
|---|----------|---------|---------|
| 1 | MEDIUM | WS5's dependency on WS4 is not strictly required for Phase 1 gate | 1.3 |
| 2 | LOW | WS2 has soft dependency on WS1's `.nl` parser for testing | 1.3 |
| 3 | MEDIUM | WS7 OBBT warm-start integration differs between custom LP and HiGHS | 1.3 |
| 4 | -- | No circular dependencies (confirmed negative) | 1.3 |
| 5 | HIGH | Phase 1 gate criteria conflict with HiGHS+Ipopt revised plan | 3.2 |
| 6 | MEDIUM | Phase 2 `root_gap_vs_baron <= 1.3` is very aggressive | 5.2 |
| 7 | HIGH | Phase 4 `geomean_vs_baron <= 1.5` rated 15-25% likely by feasibility assessment | 5.4 |

### Critical Recommendations

1. **Reconcile the two plans.** The development plan and feasibility assessment are contradictory on Phase 1 strategy (build vs buy for LP/NLP). A single authoritative plan must be established before implementation begins.

2. **Revise Phase 1 gate criteria if HiGHS+Ipopt strategy is adopted.** Four of eight criteria need updating to reflect that external libraries handle LP/NLP/sparse LA in Phase 1.

3. **Soften BARON comparison targets.** Replace `root_gap_vs_baron <= 1.3` (Phase 2) with `<= 2.0`, and replace `geomean_vs_baron <= 1.5` (Phase 4) with `<= 3.0` general + `<= 1.0` on 3+ GPU-amenable problem classes. The feasibility assessment's own analysis supports this.

4. **Make WS4 non-blocking for Phase 1.** The B&B engine with McCormick LP relaxations can pass the Phase 1 gate without a custom NLP subsolver. WS4 should be reclassified as a Phase 1-2 deliverable that is not on the Phase 1 critical path.

5. **Establish implementation task list with explicit dependency edges.** The current 21 implied tasks (from the phase gate checklist and team allocation) should be converted into tracked tasks with `blockedBy` relationships that encode the work stream dependencies analyzed here.
