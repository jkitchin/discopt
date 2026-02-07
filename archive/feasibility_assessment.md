# JaxMINLP Feasibility and Competitiveness Assessment

**Date:** February 7, 2026
**Classification:** Internal Decision Document
**Status:** Initial Assessment (Pre-Development)

---

## 1. Executive Summary

JaxMINLP is an ambitious 48-month project to build a JAX-native, GPU-accelerated MINLP solver combining a Rust backend for tree search and LP solving, JAX for continuous relaxations and autodiff, and an LLM advisory layer. The repository currently contains approximately 5,100 lines of Python infrastructure code -- a modeling API, benchmark framework, test harness, and literature review agent -- but **zero solver code**. No LP solver, no NLP solver, no branch-and-bound engine, no McCormick relaxations, no Rust code, and no JAX computation code exists. The project cannot solve even the simplest optimization problem today.

**Can this project succeed?** Conditionally yes, but the definition of "success" matters enormously.

- **As a research prototype demonstrating vmap batch solving and differentiable MINLP:** Feasible within 18-24 months with a focused team of 2-3. This is the strongest value proposition because no one else is doing it.
- **As a competitive open-source MINLP solver matching Couenne/Bonmin:** Plausible within 30-42 months, but requires sustained funding and successful recruitment of rare talent (Rust + numerical optimization).
- **As a solver competitive with BARON on general MINLP within 48 months:** Unlikely. BARON represents 25+ years of accumulated relaxation rules, presolve techniques, and algorithmic engineering. Matching its relaxation quality alone is a multi-year research problem. The plan's Phase 4 target of "geomean <= 1.5x BARON" is not realistic on the stated timeline.

**The honest assessment:** The project has identified three genuine gaps in the ecosystem (batch integer program solving, differentiable MINLP, modern open-source MINLP). The architecture decisions are sound and validated by analogous projects (MPAX, cuOpt, CuClarabel). The infrastructure code is well-structured. But the project is at Step 0 of actual solver development. The plan requires assembling a rare team, sustaining multi-year funding, and executing on 10 parallel work streams simultaneously. The most probable path to impact is to narrow scope aggressively, leverage existing solvers (HiGHS, Ipopt) for immediate capability, and demonstrate the unique value propositions (vmap, grad) on small-to-medium problems before attempting to compete on raw performance.

**The critical first step:** Before committing to the full plan, execute a 2-4 week architectural spike to validate the Rust-JAX batch evaluation loop with real GPU measurements. This validates the core thesis at minimal cost.

---

## 2. Technical Feasibility

### 2.1 Is the Rust+JAX+Python Architecture Sound?

**Assessment: Yes, with caveats.**

The architecture cleanly separates discrete logic (branch-and-bound tree, presolve, branching decisions) into Rust and continuous computation (relaxation evaluation, gradient computation, LP/NLP subproblems) into JAX on GPU. This split is well-motivated:

- **Rust for tree search:** B&B tree management is pointer-heavy, sequential, and variable-length -- everything JAX's functional model handles poorly. Rust's ownership model prevents the memory safety bugs that plague C++ solver implementations. This is the right choice.
- **JAX for GPU compute:** Relaxation evaluation across hundreds of B&B nodes is embarrassingly parallel and a natural fit for vmap + JIT. Autodiff through relaxations is free with JAX. This is the core architectural advantage.
- **Precedent:** NVIDIA's cuOpt uses a similar hybrid CPU/GPU architecture for vehicle routing. MPAX demonstrates JAX-native LP/QP solving works. CuClarabel (Yale, Boyd group) shows GPU interior point methods are viable. linrax implements JAX-native simplex for LP. These validate the general approach from multiple angles.

**Caveats that must be validated early:**

1. **Rust-to-JAX boundary latency.** The plan claims ~50-100 microseconds per round trip, <1% overhead for batches of 100+ nodes. This is plausible but unverified. If batch sizes are small (as they are early in B&B when the tree is shallow), overhead could dominate. An architectural spike should measure this before committing to the full plan.

2. **Two GPU-CPU transfers per IPM iteration.** The hybrid IPM design has JAX evaluate f/gradient/Hessian on GPU, transfer the KKT matrix to CPU for Rust factorization, then transfer the search direction back to GPU for line search. Two transfers per iteration, 15-50 iterations per subproblem. At scale this could be a bottleneck.

3. **JAX JIT recompilation.** If node relaxations have varying structures (different active constraints, different variable counts after presolve), JAX will recompile for each new shape. A fixed-shape padding strategy is needed to ensure JIT compilation happens once and only once.

4. **Build complexity.** The three-language architecture (Rust + Python + JAX/XLA) means three debugging environments, three performance profiling tools, cross-platform builds with Rust extensions (maturin + PyO3), and a much larger surface area for integration bugs. This is manageable but not trivial.

### 2.2 Can McCormick Relaxations Be Made Competitive?

**Assessment: Adequate for many problems, but will not match BARON's relaxation quality for years.**

McCormick relaxations for factorable programs are the foundation of spatial branch-and-bound for MINLP. The approach is well-understood theoretically: decompose an expression into elementary operations, apply convex/concave envelopes to each operation. JAX's expression tracing makes this natural to implement -- the existing Expression DAG in `core.py` (lines 60-400) already captures the right structure.

However, BARON's competitive advantage comes not from the basic McCormick framework but from 25+ years of accumulated refinements:
- Hundreds of specialized relaxation rules for specific subexpression patterns (trilinear, fractional, signomial, pooling-specific)
- Adaptive partitioning schemes that tighten relaxations where they matter most
- Range reduction techniques (FBBT, OBBT) deeply integrated with relaxation tightening
- Proprietary convex envelopes that are provably tighter than standard McCormick on many function classes
- Cutting planes (RLT cuts, gradient-based outer approximation) that supplement relaxation quality

A fresh implementation will start with basic McCormick (bilinear products, univariate convex/concave envelopes for exp, log, sin, etc.) and will produce relaxations that are significantly weaker than BARON's on most problem classes. The plan's Phase 2 target of "root gap <= 1.3x BARON" is very aggressive for 26 months of development.

**Where JaxMINLP can compensate:** Weaker relaxations mean more B&B nodes. GPU batching can process nodes 10-100x faster than sequential CPU solvers. This is the fundamental bet: trade relaxation quality for node throughput. This strategy works well for:
- Problems with moderate-depth B&B trees (hundreds to low thousands of nodes)
- Problems where relaxation evaluation is the bottleneck (dense constraints, many nonlinear terms)
- Batch scenarios where thousands of related instances are solved simultaneously

It breaks down for:
- Problems requiring millions of nodes (where tight relaxations are essential)
- Highly sparse problems (where GPU dense operations have high overhead)
- Single-instance scenarios (where BARON's tighter relaxations win outright)

### 2.3 Can a GPU IPM Be Built That Is Fast Enough?

**Assessment: Yes, for dense-to-moderate problems. This is well-validated by recent projects.**

Multiple recent projects validate GPU interior point methods:
- **cuPDLP** (Google/NVIDIA): First-order LP solver on GPU, competitive with commercial solvers on large sparse LPs
- **CuClarabel** (Yale, Boyd group): GPU conic solver achieving 10-50x speedups on SOCP/SDP
- **MPAX**: JAX-native LP/QP solver (pure JAX, vmappable)

The three-tier approach in the plan is sensible:

- **Tier 1 (dense GPU Cholesky)** covers problems up to ~5,000 variables. This is straightforward to implement in pure JAX using `jax.scipy.linalg.cholesky`. Most MINLPLib benchmarks and many industrial problems fall in this range. Fully vmap-compatible. This should be the first GPU IPM implementation.
- **Tier 2 (PCG iterative solver)** is pure JAX and extends to ~50,000 variables. Moderate accuracy (1e-6 to 1e-8), which is sufficient for B&B node solves where exact optimality is not required. lineax already provides the building blocks (CG, BiCGSTAB, GMRES).
- **Tier 3 (sparse GPU Cholesky via jax.extend.ffi)** is needed for large-scale problems (100K+ variables) but requires significant engineering (custom CUDA kernels, symbolic factorization in Rust, cuSOLVER integration). This is Phase 3-4 work and should not block initial development.

**IPM warm-starting risk:** IPM lacks warm-starting, which is a significant disadvantage in B&B where sequential LP resolves benefit enormously from simplex warm-starting (a single dual pivot vs. a full IPM solve). The plan acknowledges this and proposes compensating with GPU parallelism (solve many nodes simultaneously rather than quickly resolving one at a time). This is a reasonable strategy validated by the back-of-envelope calculation in the vision document:
```
CPU Simplex:  1000 nodes x 5 pivots x 0.1ms/pivot  = 500ms (sequential)
GPU IPM:      1000 nodes x 40 iters batched on GPU  = 50ms  (parallel)
```
However, this assumes 1000 nodes are available for batching simultaneously, which is only true for deep trees with many open nodes. Early in the search when the tree is small, this advantage disappears.

### 2.4 Float64 on Consumer GPUs

**Assessment: A real but manageable concern.**

NVIDIA consumer GPUs (RTX series) have severely throttled float64 throughput: 1/32 of float32 on RTX 4090 versus 1/2 on A100. Optimization requires float64 for numerical stability, especially in IPM where the KKT system becomes increasingly ill-conditioned near optimality (the plan's factorization tolerance is 1e-12, which demands float64).

Practical implications:
- Development and testing on consumer hardware will be 16-32x slower for float64 operations than on data center GPUs (A100, H100)
- Mixed precision strategies (float32 for initial IPM iterations, float64 for refinement) can mitigate this but add complexity and are not mentioned in the plan
- The target deployment environment (cloud data center GPUs) does not have this limitation
- JAX transparently handles the float64/float32 distinction, so code does not need to change

This is not a blocker but affects the development experience and cost calculations. Budget should assume A100/H100 access for serious benchmarking ($60-120K/year compute is reasonable).

### 2.5 JAX Sparse Maturity

**Assessment: Improving but still a constraint for Tier 3.**

JAX's `jax.experimental.sparse` module has matured since 2024 but still has limitations:
- BCOO/BCSR formats are supported with growing operation coverage
- Sparse-dense matrix multiplication is well-supported
- Sparse direct solvers (Cholesky, LU) are not natively available in JAX -- must use `jax.extend.ffi` to call cuSOLVER or SuiteSparse
- vmap over sparse operations has limited but growing support

For Tier 1 (dense) and Tier 2 (iterative/PCG), JAX's existing capabilities are sufficient. Tier 3 (sparse direct) will require custom FFI work regardless. This is not a showstopper for the initial development phases.

---

## 3. Competitive Positioning Assessment

### 3.1 Where JaxMINLP Can Realistically Win

**1. Batch solving of integer programs (vmap) -- STRONGEST DIFFERENTIATOR**

This is the most defensible value proposition. No existing solver -- commercial or open-source -- can solve thousands of related optimization problems in parallel on GPU. Gurobi, BARON, SCIP, and HiGHS all process problems sequentially. For applications like:
- Portfolio optimization across thousands of scenarios
- Supply chain planning with stochastic demand realizations
- Parametric sensitivity analysis (sweeping over parameter values)
- Hyperparameter search over optimization-embedded ML pipelines
- Monte Carlo simulation with optimization at each sample

...a vmappable solver provides a genuine capability that does not exist anywhere else. Even a solver that is 5x slower per instance but can batch 512 instances simultaneously wins by 100x on total throughput.

**2. Differentiable MINLP -- UNIQUE IN JAX ECOSYSTEM**

No existing tool provides differentiable optimization through integer programs in JAX. cvxpylayers handles convex-only and is not JAX-native. PyEPO and the predict-then-optimize literature use perturbation-based approaches in PyTorch. A JAX solver with `custom_jvp`/`custom_vjp` for LP relaxation sensitivity (Level 1 differentiability) enables:
- Decision-focused learning (predict-then-optimize)
- Bilevel optimization with integer lower level
- End-to-end differentiable discrete optimization in neural networks
- Sensitivity analysis of optimal decisions to problem parameters

This is an active and growing research area (NeurIPS, ICML, ICLR) with no production-quality tools.

**3. Open-source MINLP gap -- REAL BUT COMPETITIVE**

Couenne and Bonmin are in maintenance mode (last meaningful updates 2019-2020). SCIP is the only actively developed open-source solver handling MINLP, and its MINLP capabilities are limited compared to its MIP strengths. There is genuine demand for a modern, well-maintained open-source MINLP solver, especially one with a Pythonic API. Being the "scikit-learn of MINLP" -- not the fastest, but the most accessible, well-documented, and composable -- is a viable positioning.

**4. ML integration / Learn-to-optimize -- BEST RESEARCH PLATFORM**

The JAX ecosystem (Equinox, Flax, Optax, jraph) makes it natural to train ML models (GNN branching policies, learned warm-starting, neural cut selection) that are called inside the solver loop. This is extremely difficult with C/C++ solvers, requiring awkward Python callbacks and data serialization. JaxMINLP could become the reference platform for ML-for-optimization research, a rapidly growing field.

### 3.2 Where JaxMINLP Will Lose

**1. Single-instance speed vs BARON/Gurobi**

On a single MINLP instance, BARON will be faster for the foreseeable future. Its relaxations are tighter (fewer nodes), its presolve is more sophisticated, and it has 25+ years of hand-tuned heuristics. Gurobi 13.0 adding MINLP support means another well-funded competitor with decades of MIP and NLP engineering. JaxMINLP should not compete on this axis.

**2. Enterprise adoption and support**

Commercial solvers come with support contracts, SLAs, documented APIs, and decades of production deployments. JaxMINLP will be an open-source project with no commercial entity behind it. Enterprise customers in energy, logistics, and manufacturing -- the primary MINLP markets -- require reliability guarantees that an early-stage open-source project cannot provide.

**3. Large-scale sparse problems**

Problems with 100,000+ variables and sparse constraint matrices are where commercial solvers excel due to decades of sparse linear algebra optimization. JaxMINLP's dense-first approach (Tier 1) will not be competitive here. Tier 3 (sparse GPU Cholesky) is Phase 3-4 work, and even then, catching up to CHOLMOD/MUMPS quality sparse factorization is a massive undertaking.

**4. Breadth of problem types**

BARON handles MINLP, NLP, MILP, QP, and LP. Gurobi handles MILP, MIQP, MIQCQP, LP, QP, and (soon) MINLP. These solvers auto-detect structure and route to specialized algorithms. JaxMINLP initially handles only MINLP. The ecosystem vision (jax-lp, jax-milp, jax-miqp, etc.) addresses this but is Phase 3-4 scope.

### 3.3 The Right Competitive Framing

JaxMINLP should **not** be positioned as "an alternative to BARON" or "a faster MINLP solver." It should be positioned as:

> "The first JAX-native optimization solver with integer variables: batch-solve thousands of problems in parallel, differentiate through optimization, and train ML models that improve solving."

The competitive frame is not "we solve MINLPs faster" but "we enable workflows that are impossible with existing solvers." This avoids a head-to-head comparison that JaxMINLP will lose for years and focuses on the genuine technical gaps that no competitor addresses.

### 3.4 Competitive Threat Matrix

| Competitor | Threat Level | Why | JaxMINLP's Counter |
|-----------|-------------|-----|-------------------|
| **Gurobi 13.0** (adding MINLP) | HIGH | Massive team, 2x faster MINLPs in v13, OBBT added | They cannot offer vmap/grad/jit -- fundamental architecture difference |
| **BARON** | MEDIUM | Gold standard but expensive, no API modernization | Target different users (ML researchers vs enterprise) |
| **SCIP 10.0** | MEDIUM | Best open-source, active development, no GPU | No JAX integration, no batch solving, no differentiability |
| **MPAX** | LOW-MEDIUM | JAX-native LP/QP but no integers | Could be a collaborator, not a competitor; potential Phase 1 LP backend |
| **cuOpt** (NVIDIA) | LOW | GPU-optimized but focused on VRP/routing | Not general MINLP |
| **Couenne/Bonmin** | NONE | Maintenance mode | JaxMINLP should surpass these |

---

## 4. External Library Strategy

This section directly addresses the user's question about what external libraries will be needed.

### 4.1 LP: HiGHS Fallback + Custom JAX GPU IPM

**Recommendation: Use HiGHS (via highspy) as the initial LP solver. Build custom JAX GPU IPM in Phase 2.**

| Path | Library | Use Case | Phase | Timeline Impact |
|------|---------|----------|-------|-----------------|
| CPU LP (B&B relaxations, OBBT) | **HiGHS** (highspy, MIT license) | Single-instance LP relaxations, OBBT subproblems | Phase 1-2 | Saves 3-6 months vs building Rust simplex |
| GPU batch LP | **Build JAX IPM** (dense Cholesky, Tier 1) | Batch evaluation of 128-1024 node relaxations on GPU | Phase 2+ | Core value proposition, must be custom |
| Validation / comparison | **HiGHS** | Correctness validation baseline | All phases | Free |
| Potential bridge | **MPAX** | JAX-native LP/QP, vmappable, could serve batch LP needs | Phase 1-2 evaluation | May eliminate need for custom LP |

**Rationale:** Building a competitive LP solver (WS3 in the plan) is a multi-year effort by itself. HiGHS is MIT-licensed, state-of-the-art among open-source LP solvers, actively maintained, and has excellent Python bindings. Using it for CPU-path LP relaxations saves 3-6 months and lets the team focus on the unique GPU/JAX value.

The Rust simplex (from the original plan) could still be built later for cases where zero-copy warm-start integration with the Rust B&B tree matters -- but only after the solver works end-to-end with HiGHS.

MPAX deserves evaluation as a middle-ground: it is JAX-native and vmappable, which means it could serve as both the CPU and GPU LP backend. If MPAX is fast enough for node relaxations, it could replace both HiGHS and a custom IPM for LP subproblems.

### 4.2 NLP: Ipopt for Validation + Custom JAX IPM

**Recommendation: Use Ipopt (via cyipopt) for Phase 1 NLP subproblems. Build custom JAX evaluator that feeds Ipopt, then replace Ipopt with custom IPM in Phase 2.**

| Path | Library | Use Case | Phase | Timeline Impact |
|------|---------|----------|-------|-----------------|
| Phase 1 NLP | **Ipopt** (cyipopt, EPL-2.0 license) | Node NLP subproblems, correctness validation | Phase 1 | Saves 4-8 months on Phase 1 |
| Phase 2+ NLP | **Build custom JAX IPM** | GPU-batched NLP, differentiable, vmappable | Phase 2+ | Core value proposition |
| Linear algebra for IPM | **Ipopt's MUMPS** (Phase 1) -> custom (Phase 2+) | KKT factorization | Phase 1 free | Phase 2+ significant effort |

**Rationale:** Ipopt is battle-tested, handles large-scale NLP, and has Python bindings via cyipopt. Its callback interface accepts user-provided gradient and Hessian functions -- which is exactly what JAX autodiff produces. This means:

1. Build the JAX-side `NLPEvaluator` in Phase 1 (compile Expression DAG to JAX, use `jax.grad` for gradients, `jax.hessian` for Hessians)
2. Feed these callbacks to Ipopt for Phase 1 NLP solving
3. Replace Ipopt with a custom JAX IPM in Phase 2 when GPU batching and differentiability matter

The JAX evaluation layer is built once and reused. Ipopt is a temporary scaffold that de-risks Phase 1.

**License note:** EPL-2.0 is permissive for calling Ipopt as a dependency. MUMPS (Ipopt's default linear solver) is public domain. HSL routines (MA27/MA57/MA86) are free for academic use but commercial-restricted -- the plan should default to MUMPS.

### 4.3 Sparse Linear Algebra: Tiered External Libraries

**Recommendation: Use SuiteSparse for CPU sparse factorization, lineax for JAX iterative solving, cuSOLVER FFI only when needed.**

| Tier | Library | Scope | Phase |
|------|---------|-------|-------|
| Tier 0 | **Ipopt's MUMPS** | NLP factorization (free with Ipopt) | Phase 1 |
| Tier 1 | **JAX dense Cholesky** (`jax.scipy.linalg`) | GPU IPM, problems up to ~5K variables | Phase 2 |
| Tier 2 | **lineax** (JAX-native) | Iterative solver (CG, BiCGSTAB), up to ~50K variables | Phase 2-3 |
| Tier 2.5 | **SuiteSparse** (CHOLMOD/UMFPACK via scikit-sparse) | CPU sparse direct, for validation and fallback | Phase 2-3 |
| Tier 3 | **cuSOLVER** via `jax.extend.ffi` | GPU sparse Cholesky, 100K+ variables | Phase 3-4 |

Building custom sparse factorization from scratch (as WS3 in the plan suggests) is not justified until the solver handles problems large enough to need it. The existing ecosystem of sparse LA libraries covers Phase 1-3 needs.

### 4.4 ML Stack

**Recommendation: Use the standard JAX ML ecosystem. No custom ML infrastructure needed.**

| Library | Purpose | Phase |
|---------|---------|-------|
| **Equinox** | Neural network framework (JAX-native, pytree-based) | Phase 3 |
| **Optax** | Training optimizer (Adam, schedules) for GNN policies | Phase 3 |
| **jraph** | GNN message passing for bipartite variable-constraint graphs | Phase 3 |

These are mature, well-maintained JAX libraries used broadly in the research community. The GNN branching policy (WS10 in the plan) can be built entirely with these libraries. The bipartite graph structure (variable nodes + constraint nodes) maps directly to jraph's APIs.

### 4.5 What This Means for the Build Timeline

Using external libraries for LP (HiGHS), NLP (Ipopt), and sparse LA (SuiteSparse/MUMPS) fundamentally changes the critical path:

**Original plan (build everything from scratch):**
```
Sparse LA (3-6mo) -> LP solver (3-6mo) -> NLP solver (4-8mo) -> B&B (3-6mo) = 14-26 months to Phase 1
```

**With external libraries:**
```
Rust workspace + PyO3 (2-3mo) ---+
DAG-to-JAX compiler (2-4mo) ----+---> B&B engine + HiGHS/Ipopt integration (3-4mo) = 8-12 months
McCormick relaxations (2-4mo) --+
```

**Phase 1 could be reached in 8-12 months instead of 14-26 months.** This means the unique value propositions (GPU batching, differentiability) arrive 6-12 months earlier, when they matter for attracting users and funding.

### 4.6 Complete Dependency Map

| Library | Role | Phase | License | Build-vs-Buy |
|---------|------|-------|---------|---------------|
| **JAX + jaxlib** | GPU compute, autodiff, JIT, vmap | All | Apache 2.0 | Buy (foundation) |
| **maturin + PyO3** | Rust-Python build/bindings | All | MIT/Apache | Buy (tooling) |
| **numpy** | Array interchange | All | BSD | Buy (already used) |
| **scipy** | Statistics, LA fallback | All | BSD | Buy (already used) |
| **HiGHS** (highspy) | LP relaxations in B&B, OBBT | Phase 1-2 | MIT | Buy (accelerates Phase 1) |
| **Ipopt** (cyipopt) | NLP subproblems | Phase 1 (replaced Phase 2) | EPL-2.0 | Buy (accelerates Phase 1) |
| **lineax** | JAX linear solvers (CG, BiCGSTAB) | Phase 2-3 | Apache 2.0 | Buy (Tier 2 LA) |
| **Equinox** | Neural networks for learned heuristics | Phase 3 | Apache 2.0 | Buy (ML framework) |
| **Optax** | Training optimizer | Phase 3 | Apache 2.0 | Buy (ML training) |
| **jraph** | GNN message passing | Phase 3 | Apache 2.0 | Buy (GNN) |
| **Rust B&B engine** | Tree search, branching, pruning | Phase 1+ | -- | Build (core) |
| **JAX DAG compiler** | Expression -> jax.numpy callable | Phase 1+ | -- | Build (core) |
| **McCormick relaxations** | Convex relaxation primitives | Phase 1+ | -- | Build (core) |
| **JAX GPU IPM** | Batched interior point method | Phase 2+ | -- | Build (core) |

---

## 5. Likelihood of Success Assessment

### 5.1 vmap Batch Solving of Integer Programs

| Dimension | Assessment |
|-----------|-----------|
| **Likelihood** | **High (75-85%)** |
| **Timeline** | 12-18 months for LP/QP batching, 18-24 months for MINLP |
| **Impact** | **Very High** -- genuinely unique capability, no competition |
| **Key dependency** | JAX GPU IPM (Tier 1 dense) must be vmappable |
| **Key risk** | GPU batch overhead may not deliver 15x speedup; tree starvation early in search |

This is the most feasible and most valuable proposition. Even a solver that is 3x slower than BARON per instance but can solve 512 instances in parallel would be transformative for parametric optimization, scenario analysis, and decision-focused learning.

The core requirement is a vmappable LP/NLP relaxation solver. MPAX already demonstrates JAX-native LP/QP is viable. Dense GPU Cholesky in JAX is straightforward. For MINLP, the Rust B&B tree is inherently sequential (each instance has its own tree), so true vmap means running multiple independent solvers in parallel, not vmapping a single solver. This still provides massive throughput gains via `pmap` or JAX parallelism but is architecturally different from vmapping a single forward pass.

For simpler problem classes (LP, QP, MILP with LP relaxation), true vmap batch solving of the LP relaxation across B&B nodes within a single solve is highly feasible and could be demonstrated within 12 months.

### 5.2 Differentiable MINLP

| Dimension | Assessment |
|-----------|-----------|
| **Likelihood** | **High for Level 1** (80%), **Medium for Level 3** (50-60%) |
| **Timeline** | Level 1: 12-18 months. Level 3: 24-36 months. |
| **Impact** | **High** -- enables decision-focused learning, bilevel optimization |
| **Key dependency** | Working solver + `jax.custom_jvp/vjp` implementation |
| **Key risk** | Differentiating through B&B is fundamentally hard (combinatorial discontinuities) |

**Level 1 (LP relaxation sensitivity):** Well-understood mathematically and mechanically simple. Solve the LP relaxation, obtain dual variables, wrap in `custom_jvp`. This requires a working LP solver and JAX integration but no novel research. Immediately useful for predict-then-optimize applications.

**Level 3 (implicit differentiation at optimal active set):** Requires solving the MINLP to identify the active set, then applying the implicit function theorem. Established theory with precedent (cvxpylayers for convex programs, Amos & Kolter for QP layers). Implementation is delicate (degenerate cases, active set identification, perturbation smoothing at non-smooth points). Feasible but requires care and 6-12 months of focused effort.

**Levels 2 and 4** (soft B&B, fully differentiable neural B&B) are research contributions, not engineering deliverables. They should not be planned as milestones.

### 5.3 Learn-to-Optimize Integration

| Dimension | Assessment |
|-----------|-----------|
| **Likelihood** | **Medium-High (60-70%)** for GNN branching. **Medium (40-50%)** for meaningful wall-clock speedup |
| **Timeline** | 24-36 months for trained GNN branching policy |
| **Impact** | **Medium** -- 20% node reduction is meaningful but not transformative |
| **Key dependency** | Working solver generating training data, jraph integration |
| **Key risk** | GNN inference overhead may negate node reduction gains |

GNN branching policies (Gasse et al. 2019) are well-established in the MILP literature. Training one for MILP is a solved problem; extending to MINLP is straightforward in principle. The main challenge is generating sufficient training data (strong branching labels), which requires thousands of solver runs on representative instances.

The claimed 20% node reduction target is realistic for MILP but less certain for MINLP, where branching interacts with nonlinear relaxations in complex ways. More importantly, even achieving 20% node reduction, the wall-clock speedup may be modest if GNN inference (even at sub-millisecond latency) adds up over thousands of branching decisions.

The real value of ML integration may be as a research platform rather than a production speedup. JaxMINLP could become the standard tool for testing learned solver heuristics, attracting ML-for-optimization researchers as users and contributors.

### 5.4 Competitive Open-Source MINLP Solver

| Dimension | Assessment |
|-----------|-----------|
| **Likelihood** | **Medium (50-60%)** for matching Couenne/Bonmin. **Low (15-25%)** for matching BARON |
| **Timeline** | 30-42 months for Couenne/Bonmin parity. 5-7+ years for BARON competitiveness |
| **Impact** | **Very High** if achieved -- fills major ecosystem gap |
| **Key dependency** | Relaxation quality, presolve sophistication, sustained multi-year development |

Matching Couenne/Bonmin is achievable because these solvers are no longer actively developed and use older algorithmic approaches. A modern implementation with GPU batching, better presolve, and learned heuristics should eventually match or exceed them.

Matching BARON is a much harder problem. BARON's relaxation quality is the result of decades of specialized research. The plan's Phase 4 target of "geomean <= 1.5x BARON" would be a remarkable achievement requiring significant algorithmic innovation beyond what is described in the current plan. Being within 3-5x of BARON on general MINLP benchmarks is a more realistic 48-month target. Being within 1.0x of BARON on GPU-amenable problem classes (pooling, portfolio, batch scenarios) is achievable and should be the benchmark focus.

---

## 6. Critical Risk Analysis

### 6.1 Team Recruitment

**Risk Level: CRITICAL (highest risk)**

The plan requires 3-5 core developers with an unusual combination of skills:
- **Rust developer with optimization expertise:** Rust is growing rapidly, but Rust + numerical optimization + LP/MIP algorithm knowledge is an extremely rare combination. There are perhaps a few dozen people globally with this exact profile. Most work at well-funded companies (Gurobi, Google OR, Meta, Databricks).
- **JAX/GPU engineer with optimization background:** More available than the Rust role (ML researchers increasingly use JAX), but the intersection of JAX expertise and mathematical optimization knowledge is still uncommon. Most JAX experts focus on ML training, not mathematical programming.
- **Numerical computing specialist:** Sparse linear algebra, interior point methods, KKT systems. This expertise exists in applied math departments but competes with quantitative finance for talent.
- **LLM integration engineer:** More available, but the LLM work stream should be deferred (see Recommendations).

**Mitigation strategies:**
- Start with 2 core developers (one strong in Rust systems, one strong in JAX/numerical). Defer hiring until the architecture is validated.
- Use external libraries (HiGHS, Ipopt) to reduce the need for deep LP/NLP implementation expertise in Phase 1.
- Recruit from adjacent communities: Julia optimization community (familiar with solver architecture), Rust scientific computing community (familiar with Rust + numerics), JAX research community (familiar with JAX + custom differentiation rules).
- Consider academic collaborations for algorithmic work (McCormick envelope research, learned branching) that does not require full-time hires.

### 6.2 Funding

**Risk Level: HIGH**

The plan implicitly assumes:
- 3-5 full-time engineers for 4 years at competitive salaries: $1.2M-$3.5M total compensation
- GPU compute: $60K-$120K/year for cloud GPU access: $240K-$480K over 4 years
- **Total cost estimate: $1.5M-$4M over 4 years**

No funding plan, grant strategy, or commercial model is described in any project document. For an academic project, this would require multiple NSF/DOE grants (each typically $500K-$1.5M over 3 years). For a startup, this requires VC funding with an unclear path to revenue (open-source solvers are notoriously difficult to monetize). For an internal tool at a company, this requires a strong business case.

**Mitigation:**
- Define a minimum viable product (see Section 7.6) achievable with a smaller team (2 people, 12 months, $300-$400K).
- Use the MVP to demonstrate value and seek larger funding.
- Consider NSF CSSI (Cyberinfrastructure for Sustained Scientific Innovation) grants, DOE ASCR applied math programs, or NumFOCUS/CZI open-source science grants.
- Explore industry partnerships with companies that would benefit from batch optimization or differentiable optimization (energy, logistics, ML platforms).

### 6.3 Gurobi 13.0 Adding MINLP

**Risk Level: MEDIUM-HIGH**

Gurobi announced MINLP support in version 13.0 (November 2025), with 2x speed improvements on MINLP benchmarks and OBBT integration. Gurobi has:
- 200+ engineers (vs. 0-5 for JaxMINLP)
- Decades of MIP infrastructure to build on
- An installed base of 50,000+ commercial customers
- Resources to iterate rapidly on MINLP quality

If Gurobi ships a high-quality MINLP solver by 2028, the competitive landscape shifts dramatically: commercial users who already have Gurobi licenses will use Gurobi for MINLP, and the "open-source MINLP gap" becomes less compelling.

**Mitigation:** This risk reinforces the importance of not competing on single-instance speed. JaxMINLP's differentiability, batch solving, and JAX ecosystem integration are capabilities that Gurobi will not provide -- they require a fundamentally different architecture. Gurobi entering MINLP is actually an argument for narrowing JaxMINLP's competitive scope to the JAX/ML angle, not broadening it.

### 6.4 Scope Creep

**Risk Level: HIGH**

The current plan includes:
- 10 parallel work streams
- An LLM advisory layer with 12+ features spanning 38 months
- A full ecosystem of solvers (jax-lp, jax-qp, jax-milp, jax-miqp, jax-minlp)
- Literature review agents
- Natural-language formulation interface
- Conversational REPL
- RAG-based reformulation knowledge base

This is far too much for a 3-5 person team that has not yet written a single line of solver code. The LLM work stream alone (WS8) could consume the equivalent of a full-time engineer for 3+ years.

**Mitigation:** Ruthlessly cut scope. See Section 7.3 for specific recommendations on what to cut.

### 6.5 The "Infrastructure Without a Solver" Problem

**Risk Level: MEDIUM**

The project has invested significant effort in infrastructure (5,100 lines of benchmarks, tests, metrics, modeling API, examples, literature review) before writing any solver code. This creates several risks:

1. **API-driven design without implementation feedback:** The modeling API and benchmark framework were designed without the experience of actually solving problems. When solver development begins, the API may need changes that cascade through the test and benchmark infrastructure.

2. **Untested infrastructure:** All 112 tests skip with "JaxMINLP not yet available." The benchmark runner returns placeholder results. The metrics code computes results on synthetic data but has never processed real solver output. This infrastructure is plausible but unvalidated.

3. **False sense of progress:** 5,100 lines of code and a comprehensive plan can create the impression of significant progress. In reality, the project is at Step 0 of solver development. The ratio of infrastructure code to solver code is currently infinite.

4. **Morale risk:** A team that spends months building infrastructure without solving a single problem may lose motivation and credibility with stakeholders.

**Mitigation:** Prioritize getting to a solvable problem as fast as possible. Even solving `ex1221` (a tiny MINLP from MINLPLib with 5 variables, 3 constraints, known optimum 7.6672) correctly would be a major milestone that validates the architecture end-to-end. Using HiGHS + Ipopt as external subsolvers can make this happen in 4-6 months.

### 6.6 GPU-CPU Transfer Overhead

**Risk Level: MEDIUM**

The entire batch solving value proposition rests on GPU batch evaluation being faster than serial CPU evaluation. If:
- Transfer overhead is too high (slow PyO3 roundtrips), or
- Batch sizes are too small (tree starvation early in search), or
- GPU utilization is low (too much time spent in Python orchestration)

...then the GPU provides no benefit and the fundamental thesis fails.

**Mitigation:** The recommended 2-4 week architectural spike validates this empirically before committing to the full plan. Build a minimal Rust-to-JAX roundtrip with realistic array sizes (e.g., 512 nodes x 50 variables = 25,600 floats), measure latency, and compare to the plan's <100 microsecond budget.

---

## 7. Recommendations

### 7.1 What to Focus on First (Months 1-8)

Priority order for reaching the first solvable MINLP instance:

1. **Architectural spike (Month 1):** Build a minimal Rust-to-JAX roundtrip and measure GPU batch evaluation latency. Validate the core thesis before committing resources.

2. **Rust workspace + PyO3 bindings (Months 1-3):** Cargo workspace (`jaxminlp-core` + `jaxminlp-python`), maturin build, expression graph IR in Rust mirroring the Python DAG types from `core.py`, `.pyi` type stubs for mypy.

3. **JAX DAG compiler (Months 2-5):** Compile the existing `Expression` tree (core.py lines 60-400) to a pure `jax.numpy` callable. This enables `jax.jit(f)`, `jax.grad(f)`, `jax.hessian(f)` on any model expression. Test on all 7 examples from `examples.py`.

4. **McCormick relaxation primitives (Months 3-6):** Implement basic McCormick for bilinear products and univariate functions (exp, log, sin, cos, sqrt, abs). Validate soundness invariant: `relaxation_lower <= true_value` at 10,000 random points.

5. **B&B engine with external subsolvers (Months 4-8):** Implement spatial B&B in Rust with best-first node selection, most-fractional branching, and bound-based pruning. Use HiGHS for LP relaxation at each node, Ipopt for NLP subproblems. Wire into `Model.solve()` at `core.py:814-820`.

6. **CI/CD (Months 1-3):** GitHub Actions with cargo test, ruff check, mypy, pytest -m smoke. Activate tests as solver capabilities come online.

**Goal:** Correctly solve `ex1221` (optimum 7.6672) by month 5-6. Pass Phase 1 gate (25 MINLPLib instances) by month 8-12.

### 7.2 What to Defer

| Component | Original Phase | Defer To | Rationale |
|-----------|---------------|----------|-----------|
| Custom LP solver in Rust (WS3 LP) | Phase 1 | Phase 2-3 | Use HiGHS for Phase 1 |
| Custom sparse LA (WS3 sparse) | Phase 1 | Phase 3 | Use SuiteSparse/MUMPS via Ipopt |
| LLM advisory layer (WS8) | Phase 2 | Phase 3+ | Solver does not exist yet |
| Multi-GPU support (WS6) | Phase 2 | Phase 3-4 | Single GPU first |
| Advanced relaxations (WS10) | Phase 3 | Phase 3 (unchanged) | Depends on basic relaxations |
| Ecosystem expansion (jax-lp, etc.) | Phase 2 extract | Phase 4+ | MINLP solver first |
| Literature review automation | Phase 1 | Indefinite | Already works, no further investment needed |
| `.gms` file parser | Phase 1 | Phase 4+ or cut | Low priority format |
| `from_pyomo()` bridge | Phase 4 | Phase 4 or community contribution | Users can export to `.nl` |

### 7.3 What to Cut Entirely

These items should be removed from the plan to reduce scope:

1. **`from_description()` (LLM formulation from natural language):** JaxMINLP's initial users will be optimization researchers who know how to formulate models. They do not need LLM hand-holding. This feature targets a user base (non-expert modelers) that the project cannot serve until the solver itself is mature. Cut.

2. **Conversational REPL (`jaxminlp.chat()`):** Irrelevant to solver quality or the core value propositions. Cut.

3. **RAG-based reformulation knowledge base:** Research project within a research project. Cut.

4. **LLM evaluation benchmark (100 formulation tasks):** Evaluating LLM formulation accuracy is a separate research agenda. Cut.

5. **LLM configuration advisor and reformulation advisor:** Premature optimization of the user experience when no solver exists. Cut.

6. **`from_gams()` parser:** GAMS has its own solver ecosystem. Extremely low priority. Cut.

7. **Teaching mode, privacy mode, multi-model LLM pipeline:** These provide no solver capability. Cut.

Cutting these items eliminates WS8 (LLM Advisory Layer) as a dedicated work stream for Phase 1-2, freeing one full-time engineer equivalent for core solver work. LLM features can be reconsidered after the solver is functional and has users.

### 7.4 External Library Strategy (Summary Table)

| Component | Phase 1 (Months 1-12) | Phase 2 (Months 12-24) | Phase 3-4 (Months 24-48) |
|-----------|----------------------|----------------------|--------------------------|
| **LP solver** | HiGHS (highspy) | HiGHS + JAX GPU IPM (dense Tier 1) | Custom GPU IPM (iterative Tier 2) |
| **NLP solver** | Ipopt (cyipopt) + JAX grad/hess | Custom JAX IPM + Ipopt fallback | Custom JAX IPM only |
| **Sparse LA** | Ipopt's MUMPS | lineax (iterative), JAX dense Cholesky | cuSOLVER FFI (if needed) |
| **ML stack** | -- | -- | Equinox + Optax + jraph |
| **File formats** | `.nl` via Rust parser | MINLPLib loader | `.mps` (if needed) |

### 7.5 Revised Realistic Timeline

| Milestone | Original Plan | Revised Estimate | Key Enabler |
|-----------|--------------|-----------------|-------------|
| Architectural spike validated | -- | Month 1 | New: validate GPU thesis first |
| Rust workspace + PyO3 working | Month 3 | Month 2-3 | Same |
| DAG compiler + McCormick basics | Month 6-10 | Month 4-6 | Narrower scope (basics only) |
| First MINLP solved correctly | ~Month 10-12 | Month 5-6 | HiGHS + Ipopt as subsolvers |
| 25 MINLPLib instances solved | Month 14 | Month 8-12 | External subsolvers save 3-6 months |
| GPU batch evaluator working | Month 18 | Month 12-16 | Dense IPM + vmap on small problems |
| vmap batch solving demo | Not explicit | Month 14-18 | MPAX or custom dense IPM |
| Level 1 differentiable solving | ~Month 26 | Month 14-18 | custom_jvp on LP relaxation |
| Phase 2 gate (GPU speedup) | Month 26 | Month 18-24 | Custom JAX IPM replaces Ipopt |
| Competitive with Couenne | Month 38 | Month 28-36 | Custom IPM + presolve + OBBT |
| Competitive with Bonmin | Month 48 | Month 36-42 | Sustained effort |
| Learned branching prototype | Month 33 | Month 24-30 | After sufficient training data |
| Within 3x of BARON (GPU classes) | Month 48 | Month 36-48 | Achievable on favorable problems |
| Within 1.5x of BARON (general) | Month 48 | 60-84+ months | Requires research breakthroughs |
| v1.0 release | Month 48 | Month 42-48 | With narrower scope |

### 7.6 Minimum Viable Product Definition

The MVP is the smallest deliverable that demonstrates JaxMINLP's unique value and attracts early adopters:

**MVP = Spatial B&B solver that correctly solves small MINLPs (up to ~50 variables) with GPU-batched McCormick relaxations, demonstrating vmap batch relaxation evaluation and Level 1 differentiable solving.**

Concretely, the MVP includes:
- Solves all 24 `KNOWN_OPTIMA` instances from `test_correctness.py` correctly (zero incorrect)
- Spatial B&B in Rust with McCormick relaxations compiled to JAX
- HiGHS for LP relaxations at B&B nodes, Ipopt for NLP subproblems
- GPU batch relaxation evaluation via `jax.vmap` (even if full solver loop is not yet vmappable)
- Level 1 differentiable solving: `result.gradient(param)` returns d(obj*)/d(param) via LP relaxation sensitivity
- `pip install jaxminlp` installs working package (Rust extension via maturin)
- Modeling API works end-to-end: `m = Model() -> ... -> result = m.solve() -> result.objective`
- Documentation with 3-5 example notebooks showing modeling, batch solving, and sensitivity analysis

The MVP does **not** include:
- Custom LP or NLP solver (uses external libraries)
- LLM features of any kind
- Learned branching or ML integration
- Multi-GPU support
- Sparse Tier 3 support
- Ecosystem packages (jax-lp, etc.)

**This MVP is achievable in 10-14 months with 2-3 focused developers** and provides:
- Empirical validation that the architecture works
- A functioning solver for early adopters
- Demonstration of unique capabilities (batch, differentiable) for funding proposals
- A foundation to iterate on for subsequent phases

---

## 8. Bottom Line

JaxMINLP is a well-conceived project targeting genuine gaps in the optimization landscape. The three unique value propositions -- vmap batch solving, differentiable MINLP, and learned solver heuristics -- are real differentiators that no existing tool provides. The architecture is sound and validated by analogous projects. The infrastructure code is well-structured and defines the right contracts.

**The project can succeed if it:**
1. Uses HiGHS and Ipopt to reach a working solver fast (8-12 months, not 14-26)
2. Validates the GPU thesis with an architectural spike before committing full resources
3. Focuses on the unique value propositions (batch, differentiable, learnable) rather than competing on single-instance speed against BARON
4. Secures sustained funding and recruits 2-3 specialized developers
5. Ships useful increments at each phase rather than waiting for the complete vision
6. Cuts LLM scope aggressively and defers ecosystem expansion

**The project will fail if it:**
1. Tries to build every component from scratch (LP, NLP, sparse LA) before having a working solver
2. Spreads resources across 10 work streams + LLM features simultaneously
3. Cannot recruit the specialized developers needed (Rust + optimization is exceptionally rare)
4. Runs out of funding before Phase 2 delivers the GPU value proposition
5. Chases BARON performance benchmarks instead of doubling down on unique capabilities
6. Treats the 48-month plan as immutable rather than adapting based on early empirical results

**The recommended path:** Architectural spike (month 1) -> HiGHS/Ipopt-scaffolded MVP (months 8-12) -> custom JAX GPU engine + differentiability (months 12-24) -> learned heuristics + competitive performance (months 24-36) -> polish and release (months 36-48).

---

## Appendix A: Current Codebase Inventory

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| Modeling API + Expression DAG | `jaxminlp_api/core.py` | 1,022 | Complete (model building works, solve raises NotImplementedError) |
| Example models | `jaxminlp_api/examples.py` | 602 | Complete (7 runnable model-building examples) |
| Benchmark metrics + phase gates | `benchmarks/metrics.py` | 664 | Complete (12+ metrics implemented, untested against real data) |
| Benchmark runner | `benchmarks/runner.py` | 314 | Stub (_run_jaxminlp returns placeholder results) |
| Dolan-More profiles + statistics | `utils/statistics.py` | 212 | Complete |
| Report generation | `utils/reporting.py` | 302 | Complete |
| Pytest fixtures + markers | `tests/conftest.py` | 216 | Complete (9 markers, 7 fixtures) |
| Correctness tests | `tests/test_correctness.py` | 276 | All skip (24 known optima defined, no solver to test) |
| Interop tests | `tests/test_interop.py` | 240 | All skip (14 test stubs for PyO3 API surface) |
| Literature review agent | `agents/lit_review.py` | 1,028 | Production-ready (only fully functional component) |
| Config | `config/benchmarks.toml` | ~190 | Complete (phase gate criteria, suite definitions) |
| CLI entry point | `run_benchmarks.py` | 198 | Complete (wire to solver pending) |
| **Total infrastructure** | | **~5,145** | **0 lines of solver code** |

## Appendix B: Key External Projects Referenced

| Project | What It Is | Relevance | Status (Feb 2026) |
|---------|-----------|-----------|-------------------|
| **MPAX** | JAX-native LP/QP solver | Validates JAX optimization feasibility; potential Phase 1 LP backend | Active development |
| **linrax** | JAX-native simplex for LP | Demonstrates JAX LP is feasible | Research prototype |
| **cuOpt** (NVIDIA) | GPU-accelerated vehicle routing | Validates hybrid GPU/CPU optimization architecture | Commercial product |
| **cuPDLP** (Google/NVIDIA) | GPU first-order LP solver | Shows GPU LP methods are competitive | Active research |
| **CuClarabel** (Yale) | GPU conic solver (SOCP/SDP) | Demonstrates GPU IPM achieves 10-50x speedups | Active development |
| **HiGHS** | Open-source LP/MIP solver | Recommended Phase 1 LP backend | Mature, MIT license |
| **Ipopt** | Open-source NLP solver | Recommended Phase 1 NLP backend | Mature, EPL-2.0 license |
| **lineax** | JAX-native linear system solvers | Tier 2 iterative LA for GPU | Active, Apache 2.0 |
| **BARON** | Commercial global MINLP solver | Primary performance benchmark (25+ years of development) | Industry standard |
| **Gurobi 13.0** | Commercial MIP/QP/MINLP solver | Competitive threat adding MINLP | Major new competitor |
| **SCIP** | Open-source MIP/MINLP solver | Only actively developed open-source MINLP | Active development |
| **Couenne** | Open-source MINLP (COIN-OR) | Target to surpass (maintenance mode) | Stagnant |
| **Bonmin** | Open-source MINLP (COIN-OR) | Target to surpass (maintenance mode) | Stagnant |
| **cvxpylayers** | Differentiable convex optimization | Closest existing tool for differentiable optimization; convex-only | Mature |
| **JAXopt** | JAX optimization library | Was the JAX optimization entry point | **Deprecated** |
| **Equinox** | JAX neural network framework | ML stack for learned heuristics | Mature, Apache 2.0 |
| **jraph** | JAX graph neural network library | GNN branching policy | Mature, Apache 2.0 |

---

*This assessment reflects the state of the project as of February 7, 2026. It should be revisited and updated at two key milestones: (1) after the architectural spike validates or invalidates the GPU thesis, and (2) after the first MINLP instance is solved correctly, which will provide critical empirical data on architecture viability, subsolver integration complexity, and performance characteristics.*
