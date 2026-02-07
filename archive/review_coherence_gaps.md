# Coherence, Gaps, and Risk Alignment Review

**Reviewer:** coherence-reviewer
**Date:** 2026-02-07
**Documents Analyzed:**
1. `reports/jaxminlp_development_plan.md` (Development Plan)
2. `reports/feasibility_assessment.md` (Feasibility Assessment)
3. `JAX_OPTIMIZATION_ECOSYSTEM_VISION.md` (Vision Document)
4. `README.md`
5. `CLAUDE.md`
6. `jaxminlp_benchmarks/config/benchmarks.toml` (Phase Gate Config)

---

## 1. Cross-Document Consistency

### 1.1 Timeline Agreement

| Dimension | Development Plan | Feasibility Assessment | Vision Document |
|-----------|-----------------|----------------------|-----------------|
| Overall duration | 48 months | 48 months (but recommends 8-12 months to Phase 1) | 48 months (same phase boundaries) |
| Phase 1 completion | Month 14 | Month 8-12 (with external libraries) | Month 14 |
| Phase 2 completion | Month 26 | Month 18-24 (revised) | Month 26 |
| Phase 3 completion | Month 38 | Month 28-36 (revised) | Month 38 |
| Phase 4 / v1.0 | Month 48 | Month 42-48 | Month 48 |

**Finding: INCONSISTENCY.** The Development Plan and Vision Document agree on a Month 14 Phase 1 deadline, but the Feasibility Assessment argues this is unnecessarily slow and proposes reaching Phase 1 in 8-12 months by using HiGHS/Ipopt as scaffolding. The Development Plan's WS3 (custom Rust LP solver) and WS4 (custom NLP subsolver) are on the Phase 1 critical path, but the Feasibility Assessment explicitly recommends deferring both to Phase 2-3. **There is no resolution document indicating which timeline is authoritative.**

### 1.2 Build-vs-Buy Agreement

| Component | Development Plan | Feasibility Assessment | Conflict? |
|-----------|-----------------|----------------------|-----------|
| LP solver | Build custom Rust simplex (WS3, Phase 1) | Use HiGHS for Phase 1, build custom later | **YES** |
| NLP solver | Build custom hybrid IPM (WS4, Phase 1-2) | Use Ipopt for Phase 1, replace Phase 2 | **YES** |
| Sparse linear algebra | Build from scratch (WS3, Phase 1) | Use SuiteSparse/MUMPS/lineax | **YES** |
| McCormick relaxations | Build (WS2, Phase 1) | Build (same) | No |
| B&B engine | Build in Rust (WS5, Phase 1-2) | Build in Rust (same) | No |
| ML stack | Equinox/Optax/jraph (WS10, Phase 3) | Same | No |

**Finding: MAJOR INCONSISTENCY.** The Development Plan commits to building LP, NLP, and sparse LA from scratch as Phase 1 deliverables (WS3, WS4). The Feasibility Assessment explicitly argues against this, calling WS3 "a multi-year effort by itself" and recommending HiGHS and Ipopt as Phase 1 scaffolding. This is the single largest strategic disagreement between the documents. The Feasibility Assessment estimates this decision saves 3-6 months on the critical path.

### 1.3 Scope Agreement (WS8 / LLM)

| Feature | Development Plan | Feasibility Assessment | Conflict? |
|---------|-----------------|----------------------|-----------|
| `from_description()` | WS8, Phase 2 | **Cut entirely** | **YES** |
| `explain()` enhancement | WS8, Phase 2 | **Cut entirely** | **YES** |
| Configuration advisor | WS8, Phase 2 | **Cut entirely** | **YES** |
| Reformulation advisor | WS8, Phase 3 | **Cut entirely** | **YES** |
| `jaxminlp.chat()` REPL | WS8, Phase 4 | **Cut entirely** | **YES** |
| RAG reformulation KB | WS8, Phase 4 | **Cut entirely** | **YES** |
| LLM evaluation benchmark | WS8, Phase 4 | **Cut entirely** | **YES** |
| Safety infrastructure | WS8, Phase 1-2 | Not mentioned (implied cut) | **YES** |

**Finding: MAJOR CONTRADICTION.** The Development Plan includes WS8 (LLM Advisory Layer) as a full work stream spanning Phase 2-4 with a dedicated LLM engineer. The Feasibility Assessment recommends cutting WS8 entirely from Phase 1-2, calling it "a research project within a research project" and stating "LLM features can be reconsidered after the solver is functional and has users." The Vision Document does not mention LLM features at all. **These documents fundamentally disagree on whether LLM features belong in the project's first 26 months.**

### 1.4 Phase Gate Criteria Agreement

Phase gate criteria are consistent across the Development Plan (Section "Phase Gate Checklist"), Feasibility Assessment (references same gates), and `benchmarks.toml` (source of truth). All three agree on the specific numeric thresholds.

**Finding: CONSISTENT.** The `benchmarks.toml` file is the authoritative source, and both documents correctly reference its values.

### 1.5 Team Size Agreement

| Document | Team Size |
|----------|-----------|
| Development Plan | 3-5 core (Rust dev, numerical specialist, JAX/GPU engineer, LLM engineer, part-time DevOps) |
| Feasibility Assessment | 2-3 core (recommends starting with 2, deferring LLM engineer) |
| Vision Document | Not specified |

**Finding: INCONSISTENCY.** The team size differs because the Feasibility Assessment cuts the LLM engineer role entirely from early phases. If WS8 is cut, the 5-person plan has a role with no work stream.

---

## 2. Gap Analysis: Phase Gate Requirements vs. Task Coverage

### 2.1 Phase 1 Gate Criteria Coverage

| Criterion | `benchmarks.toml` Reference | Covered By Work Stream | Gap? |
|-----------|---------------------------|----------------------|------|
| `minlplib_solved_count >= 25` | `gates.phase1.criteria` line 108 | WS5 (B&B Engine) | No |
| `nlp_convergence_rate >= 0.80` | line 109 | WS4 (Hybrid IPM) | **Conditional** - covered by WS4 if built from scratch, but Feasibility says use Ipopt. If using Ipopt, no task explicitly validates CUTEst convergence rate with Ipopt. |
| `lp_netlib_pass_rate >= 0.95` | line 110 | WS3 (LP Solver) | **Conditional** - WS3 builds Rust simplex, but if HiGHS is used instead, need task to validate HiGHS against Netlib in the benchmark framework. |
| `lp_vs_highs_geomean <= 3.0` | line 111 | WS3 (LP Solver) | **Gap if HiGHS is the LP solver.** This criterion compares "our LP" to HiGHS. If HiGHS IS our LP solver, this metric is always 1.0 and the gate is trivially passed but meaningless. The gate needs revision if HiGHS is adopted. |
| `sparse_accuracy <= 1e-12` | line 112 | WS3 (Sparse LA) | **Gap if external LA is used.** If SuiteSparse/MUMPS replaces custom sparse LA, this criterion tests SuiteSparse against itself. Need to decide: is this testing our code or validating external library compatibility? |
| `relaxation_valid = 1.0` | line 113 | WS2 (McCormick) + WS5 | No |
| `interop_overhead <= 0.05` | line 114 | WS5 (B&B batch dispatch) | No |
| `zero_incorrect = 0` | line 115 | WS5 (all) | No |

**Finding: 3 conditional gaps in Phase 1.** If the Feasibility Assessment's build-vs-buy recommendations are adopted, 3 Phase 1 gate criteria become either trivially satisfied or meaningless. The gate criteria were designed for a build-from-scratch plan and need revision for a HiGHS/Ipopt-scaffolded approach.

### 2.2 Phase 2 Gate Criteria Coverage

| Criterion | Covered By | Gap? |
|-----------|-----------|------|
| `minlplib_30var_solved >= 55` | WS5 + WS7 | No |
| `minlplib_50var_solved >= 25` | WS5 + WS7 | No |
| `geomean_vs_couenne <= 3.0` | WS5 + WS7 | No |
| `gpu_speedup >= 15.0` | WS6 (GPU Batching) | No |
| `root_gap_vs_baron <= 1.3` | WS7 (Bound Tightening) | **Feasibility flags as very aggressive.** No mitigation task if target is not met. |
| `node_throughput >= 200` | WS6 | No |
| `rust_overhead <= 0.05` | WS5 | No |
| `zero_incorrect = 0` | All | No |

**Finding: 1 risk with no contingency.** The Feasibility Assessment calls `root_gap_vs_baron <= 1.3` "very aggressive for 26 months of development." There is no task or fallback plan for what happens if this gate fails. All other Phase 2 criteria have clear task ownership.

### 2.3 Phase 3 and 4 Gate Criteria

Phase 3 and 4 criteria are covered by WS10 (Advanced Algorithms) in the Development Plan. The Feasibility Assessment rates `geomean_vs_baron <= 1.5` (Phase 4) as "not realistic on the stated timeline" and suggests `3-5x BARON` as a more achievable 48-month target. **No revision to the Phase 4 gate criteria reflects this assessment.**

### 2.4 Missing Tasks for Feasibility Assessment Recommendations

The Feasibility Assessment makes several recommendations that have no corresponding task in any work stream:

| Recommendation | Status |
|---------------|--------|
| 2-4 week architectural spike (Rust-JAX latency validation) | **No task exists.** The Feasibility Assessment calls this "the critical first step." |
| Evaluate MPAX as LP/QP backend | **No task exists.** Mentioned as "deserving evaluation" in Feasibility Section 4.1. |
| Define minimum viable product scope | **Defined in Feasibility Section 7.6 but not formalized as a milestone in the Development Plan.** |
| Mixed-precision strategy for consumer GPUs | **No task exists.** Mentioned as a concern in Feasibility Section 2.4 but not in any work stream. |
| Funding/grant strategy | **No task exists.** Feasibility Section 6.2 identifies this as HIGH risk. |

---

## 3. Risk-to-Mitigation Mapping

The Feasibility Assessment identifies 6 risks (Section 6). Below is whether each has corresponding mitigation tasks in the Development Plan or task list.

### Risk 1: Team Recruitment (CRITICAL)

| Mitigation Strategy (from Feasibility 6.1) | Has Task? |
|---------------------------------------------|-----------|
| Start with 2 core developers | **No task.** Organizational, not a code task. |
| Use external libraries to reduce specialist need | **Conflicts with Development Plan** which plans to build LP/NLP from scratch. |
| Recruit from Julia/Rust/JAX communities | **No task.** |
| Academic collaborations for algorithmic work | **No task.** |

**Assessment: UNMITIGATED in the task list.** The mitigation strategies are organizational, not engineering tasks, and none are captured in the plan's work streams.

### Risk 2: Funding (HIGH)

| Mitigation Strategy (from Feasibility 6.2) | Has Task? |
|---------------------------------------------|-----------|
| Define MVP achievable with smaller team | **Defined in Feasibility Section 7.6 but not in Development Plan.** |
| Use MVP to seek larger funding | **No task.** |
| Pursue NSF CSSI / DOE ASCR grants | **No task.** |
| Explore industry partnerships | **No task.** |

**Assessment: UNMITIGATED.** No funding-related tasks exist.

### Risk 3: Gurobi 13.0 MINLP Competition (MEDIUM-HIGH)

| Mitigation Strategy (from Feasibility 6.3) | Has Task? |
|---------------------------------------------|-----------|
| Focus on vmap/grad/jit differentiators, not single-instance speed | **Partially covered.** WS6 (GPU batching) addresses vmap. But no explicit task to implement differentiable solving (`custom_jvp`/`custom_vjp`), which the Feasibility Assessment identifies as a key differentiator. The Vision Document describes 4 levels of differentiable B&B but the Development Plan has no corresponding work stream for Levels 1-3. |

**Assessment: PARTIALLY MITIGATED.** GPU batching is covered, but differentiable solving -- arguably the strongest differentiator -- has no dedicated work stream or tasks in the Development Plan.

### Risk 4: Scope Creep (HIGH)

| Mitigation Strategy (from Feasibility 6.4) | Has Task? |
|---------------------------------------------|-----------|
| Cut WS8 (LLM) from Phase 1-2 | **Development Plan does NOT cut WS8.** Direct contradiction. |
| Cut `from_description()`, chat REPL, RAG, etc. | **Development Plan includes all of these.** Direct contradiction. |

**Assessment: UNMITIGATED.** The Development Plan includes exactly the scope the Feasibility Assessment says to cut. There is no reconciliation.

### Risk 5: Infrastructure Without a Solver (MEDIUM)

| Mitigation Strategy (from Feasibility 6.5) | Has Task? |
|---------------------------------------------|-----------|
| Prioritize solving `ex1221` by month 5-6 | **No explicit task.** The Development Plan's earliest solvable milestone is Phase 1 gate at Month 14. |
| Use HiGHS + Ipopt for fast first solve | **Conflicts with Development Plan** which builds from scratch. |

**Assessment: UNMITIGATED in Development Plan.** The Feasibility Assessment's key recommendation -- get to a solvable problem fast -- is not reflected in the work stream timeline.

### Risk 6: GPU-CPU Transfer Overhead (MEDIUM)

| Mitigation Strategy (from Feasibility 6.6) | Has Task? |
|---------------------------------------------|-----------|
| 2-4 week architectural spike | **No task exists.** |
| Measure Rust-to-JAX roundtrip latency with realistic array sizes | **No task exists.** WS5 includes `interop_overhead <= 0.05` as a Phase 1 gate criterion but no early validation task. |

**Assessment: UNMITIGATED.** The spike that the Feasibility Assessment calls "the critical first step" has no corresponding task.

### Summary: Risk-to-Mitigation Coverage

| Risk | Severity | Mitigated in Task List? |
|------|----------|------------------------|
| Team Recruitment | CRITICAL | No |
| Funding | HIGH | No |
| Gurobi 13.0 Competition | MEDIUM-HIGH | Partially (GPU yes, differentiability no) |
| Scope Creep | HIGH | No (Development Plan contradicts Feasibility) |
| Infrastructure Without Solver | MEDIUM | No |
| GPU-CPU Transfer Overhead | MEDIUM | No |

**Finding: 0 of 6 risks are fully mitigated by the current task list. 1 is partially mitigated.**

---

## 4. MVP Alignment

The Feasibility Assessment defines an MVP in Section 7.6 (achievable in 10-14 months with 2-3 developers):

| MVP Requirement | Covered by Development Plan Tasks Through Week 12? |
|----------------|-----------------------------------------------------|
| Solves all 24 `KNOWN_OPTIMA` correctly | WS5 targets this, but at Month 14 (not Month 12). With HiGHS/Ipopt scaffolding, Feasibility says Month 8-12. |
| Spatial B&B in Rust with McCormick relaxations | WS1 + WS2 + WS5 cover this. |
| HiGHS for LP relaxations | **Not in Development Plan.** WS3 builds custom LP. |
| Ipopt for NLP subproblems | **Not in Development Plan.** WS4 builds custom IPM. |
| GPU batch relaxation evaluation via `jax.vmap` | WS6, but this is Phase 2 (Month 14-26) in the Development Plan, not part of MVP-timeline scope. |
| Level 1 differentiable solving | **Not in any work stream.** The Vision Document describes it but the Development Plan has no task for `custom_jvp` implementation. |
| `pip install jaxminlp` with Rust extension | WS9 (release automation), but at Month 26+, not MVP timeline. |
| 3-5 example notebooks | **No task exists.** WS10 mentions "documentation" but at Phase 3-4. |

**Finding: The Development Plan does NOT produce the Feasibility Assessment's MVP by Week 12 (Month 3).** Even interpreting "through Week 12" generously as "through Month 12," the Development Plan's Phase 1 deadline is Month 14, and it builds custom LP/NLP instead of using HiGHS/Ipopt. Critical MVP components (GPU batch evaluation, differentiable solving, installable package) are Phase 2-4 scope in the Development Plan.

The documents describe two fundamentally different Phase 1 strategies:
- **Development Plan:** Build everything from scratch, reach Phase 1 gate at Month 14.
- **Feasibility Assessment:** Scaffold with HiGHS/Ipopt, reach MVP at Month 10-14, reach unique value props (GPU, differentiability) by Month 14-18.

---

## 5. Naming Consistency

The CLAUDE.md file references `discopt` in the Implementation Status section:

> "The development plan for `discopt` has been organized into tasks..."

All other documents use `JaxMINLP` or `jaxminlp`:
- Development Plan: "JaxMINLP" throughout
- Feasibility Assessment: "JaxMINLP" throughout
- Vision Document: "JaxMINLP" throughout, with ecosystem packages named `jax-lp`, `jax-qp`, etc.
- `benchmarks.toml`: `[solvers.jaxminlp]`
- All directory names: `jaxminlp_benchmarks/`, `jaxminlp_api/`

The Vision Document's ecosystem envisions the package as `jax-minlp` (with hyphen) within a broader `jax-opt` meta-package. The Vision Document also mentions `import jaxopt as jo` in the API example, which conflicts with the deprecated JAXopt library.

**Findings:**
1. **`discopt` appears only in CLAUDE.md** and nowhere else. This rename is not reflected in any other document, directory name, import path, or configuration. It is unclear whether `discopt` is the intended new name or an artifact.
2. **`jaxopt` import name conflicts** with the deprecated Google JAXopt library. The Vision Document uses `import jaxopt as jo` which would create a namespace collision.
3. **Hyphenation is inconsistent**: `jax-minlp` (repo name, Vision ecosystem), `jaxminlp` (code directories, config), `JaxMINLP` (prose). This is cosmetic but should be resolved before v1.0.

---

## 6. Specific Contradictions Between Documents

### Contradiction 1: Custom vs. External LP/NLP Solvers (Critical)

- **Development Plan WS3:** Build custom Rust revised simplex, dual simplex, sparse LU with Markowitz pivoting, sparse Cholesky, LDL^T -- all from scratch in Phase 1.
- **Feasibility Assessment Section 4.1:** "Building a competitive LP solver (WS3 in the plan) is a multi-year effort by itself. HiGHS is MIT-licensed, state-of-the-art...Using it for CPU-path LP relaxations saves 3-6 months."
- **Feasibility Assessment Section 7.3:** Does not list WS3 LP as "cut" but Section 7.2 defers it to Phase 2-3.

**Impact:** This is the highest-impact contradiction. It determines whether Phase 1 takes 14 months or 8-12 months, and whether the team needs a sparse LA specialist from day one.

### Contradiction 2: LLM Scope (Major)

- **Development Plan WS8:** Full LLM advisory layer as a dedicated work stream with a dedicated engineer, starting Phase 2.
- **Feasibility Assessment Section 7.3:** "Cut entirely" -- `from_description()`, chat REPL, RAG, LLM benchmark, configuration advisor, reformulation advisor.
- **Vision Document:** No mention of LLM features at all.

**Impact:** One full-time engineer role and 12+ features over 38 months of effort depend on this decision.

### Contradiction 3: Phase 4 Performance Target (Significant)

- **Development Plan:** Phase 4 gate `geomean_vs_baron <= 1.5` (at Month 48).
- **Feasibility Assessment Section 5.4:** "The plan's Phase 4 target of 'geomean <= 1.5x BARON' is not realistic on the stated timeline." Recommends 3-5x BARON as achievable in 48 months. Estimates 60-84+ months for 1.5x.
- **`benchmarks.toml` line 148:** `geomean_vs_baron = { max = 1.5 }` (matches Development Plan).

**Impact:** If the Feasibility Assessment is correct, the Phase 4 gate as currently defined will fail. The `benchmarks.toml` needs revision, or the Phase 4 timeline needs extension.

### Contradiction 4: Differentiable Solving Ownership (Significant)

- **Vision Document:** Describes 4 levels of differentiable B&B in detail (LP relaxation sensitivity, soft B&B, implicit differentiation, neural B&B). Level 1 is tagged "Build now."
- **Development Plan:** No work stream covers differentiable solving. `custom_jvp`/`custom_vjp` implementation is not mentioned in any WS.
- **Feasibility Assessment Section 5.2:** Rates Level 1 differentiable solving at 80% likelihood, 12-18 months -- calls it a key differentiator.

**Impact:** One of the project's three core value propositions (differentiable optimization) has detailed design in the Vision Document but zero implementation tasks in the Development Plan.

### Contradiction 5: Ecosystem Expansion Timing

- **Vision Document Phase 2:** Extract `jax-optcore`, build GPU IPM that serves entire ecosystem.
- **Development Plan:** No work stream for `jax-optcore` extraction. No mention of ecosystem packages before Phase 3-4.
- **Feasibility Assessment Section 7.2:** Defers ecosystem expansion to "Phase 4+ or cut."

**Impact:** The Vision Document's Phase 2 ecosystem work is not in the Development Plan and is explicitly deferred by the Feasibility Assessment.

### Contradiction 6: Architectural Spike

- **Feasibility Assessment:** "The critical first step: Before committing to the full plan, execute a 2-4 week architectural spike." Repeated in Section 7.1, 6.6, and 8.
- **Development Plan:** No architectural spike. WS1 starts immediately with full Cargo workspace setup.
- **Vision Document:** No spike mentioned.

**Impact:** The Feasibility Assessment conditions the entire project on a validation that the Development Plan skips.

---

## 7. Summary of Key Findings

### Critical Issues (must resolve before proceeding)

1. **Build-vs-buy for LP/NLP is unresolved.** The Development Plan and Feasibility Assessment take opposite positions on the most resource-intensive Phase 1 work. This must be decided before assigning tasks.

2. **LLM scope (WS8) is unresolved.** The Development Plan includes it; the Feasibility Assessment cuts it entirely. A full-time engineer role depends on this decision.

3. **Differentiable solving has no tasks.** One of three core value propositions is designed in the Vision Document but absent from the Development Plan's work streams.

4. **No risks are fully mitigated in the task list.** The Feasibility Assessment identifies 6 risks; the Development Plan addresses none of them with explicit tasks or contingencies.

### Significant Issues (should resolve soon)

5. **Phase gate criteria need revision if HiGHS/Ipopt are adopted.** Three Phase 1 criteria become meaningless if external solvers replace custom implementations.

6. **Phase 4 BARON performance target is likely unrealistic.** The Feasibility Assessment estimates 60-84+ months for `geomean_vs_baron <= 1.5`. The `benchmarks.toml` should be revised to a more achievable target (e.g., 3.0x) with the 1.5x target moved to a stretch milestone.

7. **Naming (`discopt` vs `JaxMINLP` vs `jaxminlp` vs `jax-minlp`) is inconsistent.** CLAUDE.md references `discopt`; all other documents use `JaxMINLP`.

8. **The architectural spike recommended by the Feasibility Assessment is not in any plan.** This 2-4 week validation is called "the critical first step" but has no task.

### Observations (informational)

9. **The Vision Document is aspirational, not operational.** It describes a full ecosystem (jax-lp, jax-qp, jax-milp, jax-miqp, jax-minlp) that the other documents do not plan to build in the 48-month window. It should be treated as long-term direction, not near-term scope.

10. **The documents were clearly written in sequence:** Development Plan first (build-from-scratch approach), then Feasibility Assessment (challenging and revising the plan), then Vision Document (expanding scope beyond MINLP). They have not been reconciled into a single authoritative plan.

---

## 8. Recommendations

1. **Produce a single authoritative "Project Plan v2"** that resolves the build-vs-buy decision, LLM scope, and timeline conflicts between the Development Plan and Feasibility Assessment.

2. **Add an "Architectural Spike" task** as the first task in any work plan, before committing to full development.

3. **Add a "Differentiable Solving" work stream** (or add Level 1 differentiability tasks to WS5/WS6) to ensure this core value proposition has implementation tasks.

4. **Revise Phase 1 gate criteria** to be meaningful under a HiGHS/Ipopt scaffolding strategy (if adopted).

5. **Revise Phase 4 `geomean_vs_baron`** from 1.5 to a more achievable target based on the Feasibility Assessment's analysis.

6. **Resolve the `discopt` naming** -- either rename consistently across all documents and code, or remove the reference from CLAUDE.md.

7. **Add non-engineering tasks** for risk mitigations: funding strategy, recruitment plan, and industry partnership exploration.
