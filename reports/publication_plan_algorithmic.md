# discopt: Algorithmic & Methods Publication Plan

**Reviewer**: methods-reviewer
**Date**: 2026-02-07
**Based on**: reconciled_development_plan_v2.md, discopt_positioning.md

---

## Overview

This report identifies six natural publication milestones for algorithmic and methods papers emerging from the discopt development plan. The papers are ordered to create a coherent narrative arc: early papers establish the framework and its unique architectural choices, middle papers demonstrate capabilities no existing solver provides, and late papers show competitive or superior results on recognized benchmarks.

The pacing targets roughly one paper every 6-8 months, with the first submittable around Month 8-10 and the last around Month 34-36.

---

## Paper 1: Compiling Expression DAGs to JAX-Traceable McCormick Relaxations

### Working Title
"Automatic McCormick Relaxation via Expression DAG Compilation to JAX: Enabling Differentiable and Batchable Convex Relaxations for Nonlinear Programs"

### Timing
**Month 8-10** (end of Phase 1). Submittable once WS2 (DAG compiler + McCormick) is complete and validated on the 24 MINLPLib instances. Does not require the full B&B solver to be competitive -- the contribution is the compilation approach itself.

### Core Contribution
A compiler that walks an expression DAG and emits pure `jax.numpy` code implementing McCormick convex/concave relaxation envelopes for arbitrary compositions of elementary operations. The compiled relaxation functions are compatible with `jax.jit` (zero overhead), `jax.vmap` (batch evaluation over bound vectors), and `jax.grad` (differentiable relaxation bounds). This is the first system that makes McCormick relaxations automatically composable with JAX's transformation system.

Prior work on automatic McCormick relaxations exists (McCormick.jl in Julia, mc++ in C++) but none produces functions that are simultaneously JIT-compilable, batchable via vmap, and differentiable -- the three properties needed for GPU-accelerated spatial branch-and-bound with learned heuristics.

### Minimum Viable Results
1. **Soundness validation**: All 14 operation types pass the 10,000-point soundness test (relaxation_lower <= true_value <= relaxation_upper) with tolerance 1e-10.
2. **Coverage**: All 24 MINLPLib known-optimum instances from the test suite produce valid relaxations.
3. **JIT traceability**: `jax.make_jaxpr` succeeds on all 7 example model objectives.
4. **Gradient correctness**: `jax.grad` of compiled relaxations matches finite differences (atol=1e-6) at 100 random interior points per operation.
5. **Batch throughput**: `jax.vmap` over 128-1024 bound vectors shows near-linear scaling on GPU, with wall-clock comparison to serial McCormick.jl evaluation.
6. **Gap convergence**: Relaxation gap decreases monotonically as bounds tighten (5 progressive steps, 10 expressions) -- demonstrating the compiled relaxations preserve the theoretical convergence properties.

### Target Venue
**INFORMS Journal on Computing (IJOC)** or **Mathematical Programming Computation (MPC)**

**Rationale**: This is fundamentally an optimization computation paper -- it presents a software system for computing mathematical objects (McCormick relaxations) in a new computational paradigm (JAX transformations). IJOC and MPC are the premier venues for optimization software contributions with rigorous computational experiments. IJOC has published McCormick relaxation papers before (Tsoukalas & Mitsos, 2014 on multivariate McCormick; Mitsos et al. on MC++). MPC requires open-source software availability, which aligns with discopt's MIT/Apache license.

Alternative: Optimization Methods and Software (OMS) if the paper is more implementation-focused than algorithmic.

### Dependencies (from development plan)
- T4: JAX DAG compiler complete
- T5: McCormick relaxation primitives complete (all 14 operation types)
- T6: Relaxation compiler (compile_relaxation) complete and vmap-validated
- T15: MINLPLib loading (for the 24-instance validation)
- T10: CI/CD (for reproducibility claims)

### Builds On
This is the first discopt paper -- no prior discopt work to cite. Cites McCormick (1976), Tsoukalas & Mitsos (2014), Scott et al. (2011 on McCormick.jl), the JAX paper (Bradbury et al., 2018), and the broader automatic relaxation literature.

---

## Paper 2: Batch Spatial Branch-and-Bound on GPU via Hybrid Rust-JAX Architecture

### Working Title
"GPU-Accelerated Spatial Branch-and-Bound for Mixed-Integer Nonlinear Programming via Batch Relaxation Evaluation"

### Timing
**Month 14-16** (early Phase 2). Submittable once WS6 (GPU batching) demonstrates the 15x GPU speedup target on the comparison benchmark suite, and the custom IPM (WS4 Tier 1) is working. This is the flagship architecture paper.

### Core Contribution
A new computational architecture for spatial branch-and-bound (sBB) that separates tree management (Rust, CPU) from relaxation evaluation (JAX, GPU). The CPU manages node selection, branching, pruning, and incumbent tracking in a Rust B&B engine. At each iteration, it exports a batch of open nodes (bound vectors) to the GPU, where a single fused XLA kernel evaluates convex relaxations for all nodes simultaneously via `jax.vmap`. Results are imported back to update the tree.

This is the first MINLP solver to evaluate hundreds of branch-and-bound nodes in parallel on GPU. The architectural contribution is showing that the inherently sequential B&B algorithm can be restructured into a "batch-sequential" pattern that exposes massive data parallelism without sacrificing correctness or global optimality guarantees.

### Minimum Viable Results
1. **Correctness**: All 24 MINLPLib known-optimum instances solved correctly (zero incorrect) with the batch architecture.
2. **GPU speedup**: Batch of 512 nodes >= 15x faster than serial CPU evaluation (the Phase 2 gate target).
3. **Node throughput**: >= 200 nodes/sec on 50-variable problems.
4. **Scaling curve**: GPU speedup vs batch size (1, 32, 64, 128, 256, 512, 1024) showing the parallelism threshold and saturation point.
5. **Component profiling**: Rust tree management, JAX relaxation evaluation, and Python orchestration fractions sum to ~1.0, with interop overhead <= 5%.
6. **Comparison to Couenne**: Shifted geometric mean time ratio on the phase2 comparison suite (target: <= 3.0x Couenne). Note: the paper does NOT claim to beat BARON -- it demonstrates a new computational paradigm with significant speedups over open-source baselines.
7. **Batch size ablation**: How does optimal batch size vary with problem structure (number of variables, constraint density, nonlinearity)?

### Target Venue
**NeurIPS** (Neural Information Processing Systems) or **ICML** (International Conference on Machine Learning)

**Rationale**: The GPU batching architecture is directly relevant to ML workflows that embed optimization (decision-focused learning, structured prediction). NeurIPS and ICML have published GPU optimization papers (MPAX at NeurIPS, CuClarabel at ICML workshop, differentiable optimization papers). The "batch solving" contribution is a systems-level innovation that enables ML applications, making it appropriate for a top ML venue. The paper should frame the contribution around enabling "optimization-in-the-loop ML" rather than purely as an optimization advance.

Alternative: If the paper leans more heavily on the optimization theory (convergence guarantees for batch sBB), **Mathematical Programming** Series A would be the premium optimization venue, but the turnaround time (12-18 months) is much slower than NeurIPS/ICML (6 months).

### Dependencies
- T14: End-to-end solver working
- T15: Full MINLPLib correctness validation
- T17: Dense GPU IPM (Tier 1) -- needed for vmappable relaxation solves
- T19: Batch relaxation evaluator (the core of this paper)
- T24: Custom IPM replaces Ipopt in solver loop (enables true GPU batching)
- T25: Performance measurement infrastructure

### Builds On
- **Paper 1** (McCormick compilation): The batch evaluator uses Paper 1's compiled relaxations. This paper extends the contribution from "compile relaxations to JAX" to "use those relaxations inside a full GPU-batched B&B solver."

---

## Paper 3: Differentiable Mixed-Integer Nonlinear Optimization via JAX Transformations

### Working Title
"Differentiating Through Discrete Optimization: Multi-Level Gradient Computation for Mixed-Integer Nonlinear Programs in JAX"

### Timing
**Month 16-20** (mid-to-late Phase 2). Submittable once WS-D delivers Level 1 (LP relaxation sensitivity via custom_jvp) and Level 3 (implicit differentiation at optimal active set) with perturbation smoothing as fallback. Can be submitted shortly after Paper 2.

### Core Contribution
A multi-level differentiability framework for MINLP solvers, implemented via JAX's `custom_jvp` and composable with `jax.grad`, `jax.vmap`, and `jax.jit`. The three levels provide a graceful degradation from exact to approximate gradients:

- **Level 1** (LP relaxation sensitivity): Exact dual-variable-based gradients of the relaxed LP objective w.r.t. problem parameters. Mechanically simple, theoretically sound, immediately useful for decision-focused learning.
- **Level 3** (implicit differentiation at active set): At the MINLP solution, active constraints define an implicit system. The implicit function theorem gives exact gradients of continuous variables w.r.t. parameters. Extends cvxpylayers to the mixed-integer nonlinear case.
- **Perturbation smoothing** (fallback): `jax.vmap` over perturbed instances estimates gradients via Stein's lemma. Expensive in serial, cheap in batch.

The novelty relative to PyEPO (integer linear only) and cvxpylayers (continuous convex only) is handling *both* integer variables *and* nonlinear constraints simultaneously.

### Minimum Viable Results
1. **Level 1 correctness**: `jax.grad` through LP relaxation matches known LP sensitivity theory on 5 parametric LP instances.
2. **Level 3 correctness**: Implicit differentiation gradients match finite-difference approximation (rel_tol=1e-3) on 5 parametric MINLP problems.
3. **Perturbation smoothing convergence**: Gradient estimates converge as number of perturbations increases (4, 8, 16, 32, 64).
4. **Decision-focused learning demo**: End-to-end training of a simple neural network predictor where gradients flow through discopt.solve, showing improved decision quality vs. two-stage (predict-then-optimize) baseline on at least 2 benchmark problems from the DFL literature.
5. **Composability**: `jax.grad(lambda p: jax.vmap(lambda s: solve(model(p, s)))(scenarios).objective.mean())(p0)` -- gradient of expected objective over scenarios w.r.t. parameters -- works and returns correct values.
6. **Comparison to finite differences**: Wall-clock time and gradient accuracy vs. finite-difference approximation at varying perturbation sizes.

### Target Venue
**NeurIPS** or **ICML**

**Rationale**: Differentiable optimization is a core ML topic. NeurIPS/ICML have published all the key differentiable optimization papers: cvxpylayers (NeurIPS 2019), OptNet (ICML 2017), PyEPO (CPAIOR 2022 but heavily cited by ML community), Blondel et al. implicit differentiation (ICML 2022). Extending differentiability to MINLP is a clear contribution to this literature. The decision-focused learning experiments directly address current ML research interests.

Alternative: **AAAI** or **IJCAI** if the paper emphasizes the AI planning/combinatorial optimization angle over the pure ML angle.

### Dependencies
- T14: End-to-end solver working (needed as the forward pass)
- T22: Level 1 differentiable solving (custom_jvp on LP relaxation)
- T23: Level 3 differentiable solving (implicit differentiation)
- T19: Batch evaluator (needed for perturbation smoothing to be computationally tractable)

### Builds On
- **Paper 1** (McCormick compilation): The relaxation gradients from Paper 1 are prerequisites for Level 1 sensitivity.
- **Paper 2** (batch B&B): The batch solving from Paper 2 makes perturbation smoothing practical (32 perturbed solves cost the same as one).

---

## Paper 4: A vmappable Interior Point Method for Batch Nonlinear Subproblem Solving

### Working Title
"Dense GPU Interior Point Methods for Batch NLP Subproblems in Branch-and-Bound: JAX-Native Implementation via vmap"

### Timing
**Month 18-22** (late Phase 2 / early Phase 3). Submittable once WS4 (Tier 1 dense IPM + Tier 2 PCG iterative) has replaced Ipopt in the solver loop and performance measurements are available.

### Core Contribution
A JAX-native interior point method (IPM) for LP and NLP subproblems that is fully compatible with `jax.vmap` and `jax.grad`. Unlike traditional IPM implementations (Ipopt, OOQP, PDLP) that rely on sparse factorizations, this IPM uses dense Cholesky factorization via `jax.scipy.linalg.cholesky` on GPU, exploiting the fact that MINLP relaxation subproblems are typically small-to-medium (up to ~5,000 variables) but solved hundreds of times per B&B iteration. The key insight is that for batch MINLP solving, the efficiency bottleneck is not single-instance solve time but throughput across instances -- and dense GPU linear algebra is far more amenable to batching than sparse CPU factorization.

The Tier 2 PCG extension uses preconditioned conjugate gradients (pure JAX, also vmappable) to scale to ~50,000 variables with moderate accuracy.

### Minimum Viable Results
1. **Correctness**: IPM matches Ipopt solution on all 7 example NLP problems (rel_tol=1e-4) and all 7 Netlib LP instances (abs_tol=1e-6).
2. **KKT residual**: < 1e-10 at solution for LP instances, < 1e-8 for NLP instances.
3. **vmap correctness**: `jax.vmap(ipm_solve)(batch_of_64_problems)` produces correct results.
4. **Batch speedup**: Batch of 64-512 IPM solves on GPU vs. 64-512 sequential Ipopt calls on CPU. Target: >= 10x at batch 64, >= 50x at batch 512.
5. **Single-instance comparison**: IPM vs. Ipopt wall-clock on individual problems of varying size (10, 50, 100, 500, 1000, 5000 variables). The IPM will be slower for single instances -- the paper honestly reports this and frames the contribution as batch throughput.
6. **Differentiability**: `jax.grad(lambda p: ipm_solve(p).objective)(params)` returns finite, correct values.
7. **End-to-end solver impact**: Comparison of full discopt solver with Ipopt backend vs. custom IPM backend on the Phase 2 benchmark suite (time, correctness, GPU utilization).
8. **PCG scaling**: Tier 2 iterative solver convergence on problems up to 50,000 variables, with preconditioning ablation.

### Target Venue
**Mathematical Programming Computation (MPC)** or **Optimization Methods and Software (OMS)**

**Rationale**: This is fundamentally an optimization computation paper about implementing a well-known algorithm (IPM) in a novel computational framework (JAX/GPU batching). MPC and OMS are the right venues for rigorous computational studies of optimization algorithms. MPC requires open-source code, which aligns with discopt. The paper is less suitable for NeurIPS/ICML because the IPM itself is not novel -- what's novel is the implementation strategy for batch GPU solving, which is a computational contribution.

Alternative: **SIAM Journal on Optimization (SIOPT)** if the paper includes convergence analysis of the dense-Cholesky IPM under batching (e.g., showing that numerical properties are preserved when solving 512 instances simultaneously with shared precision).

### Dependencies
- T8: JAX NLP evaluator (gradient/Hessian callbacks)
- T17: Dense GPU IPM (Tier 1)
- T18: PCG iterative solver (Tier 2)
- T24: Custom IPM replaces Ipopt in solver loop
- T25: Performance measurement

### Builds On
- **Paper 1** (McCormick compilation): The NLP subproblems being solved are McCormick relaxations compiled by Paper 1's system.
- **Paper 2** (batch B&B): The IPM is the component that makes Paper 2's batch architecture fully GPU-native (replacing the CPU Ipopt bottleneck).

---

## Paper 5: GNN Branching Policies for Mixed-Integer Nonlinear Programs

### Working Title
"Learning to Branch in Spatial Branch-and-Bound: Graph Neural Network Policies for MINLP"

### Timing
**Month 28-32** (Phase 3). Submittable once WS10-c delivers the GNN branching policy with >= 20% node reduction and < 0.1ms inference latency. This paper requires significant training data from Phase 2 solver runs.

### Core Contribution
Extension of GNN branching policies from MILP (Gasse et al., NeurIPS 2019; Nair et al., ICML 2020) to the spatial branch-and-bound algorithm for MINLP. Key differences from the MILP case:

1. **Bipartite graph structure includes nonlinear relaxation information**: In MILP, the bipartite graph encodes LP constraint coefficients. In MINLP, the graph must also encode McCormick relaxation quality (gap between relaxation and true function) and bound tightening history.
2. **Branching on continuous variables**: MILP only branches on integer variables. Spatial B&B also branches on continuous variables (spatial branching), meaning the action space is fundamentally different.
3. **Training loop is end-to-end differentiable in JAX**: Because discopt's B&B tree, relaxation evaluator, and GNN are all in JAX, the entire training pipeline -- from strong branching data collection through imitation learning to RL fine-tuning -- executes in a single JIT-compiled loop. This is infeasible with C/C++ MILP solvers that require Python callbacks.

The paper uses Equinox for the GNN architecture, jraph for graph operations, and Optax for training -- all JAX-native.

### Minimum Viable Results
1. **Node reduction**: >= 20% fewer nodes than most-fractional branching on the Phase 3 benchmark suite (the gate target).
2. **Inference latency**: < 0.1ms per branching decision (the gate target), so the GNN does not bottleneck the solver.
3. **Comparison to classical heuristics**: GNN vs. reliability branching, strong branching, most-fractional on >= 30 MINLPLib instances.
4. **Generalization**: Train on 50-variable problems, test on 100-variable problems. Report both in-distribution and out-of-distribution performance.
5. **Training efficiency**: Imitation learning (IL) vs. IL + RL fine-tuning ablation showing that RL improves node counts beyond pure imitation.
6. **MINLP vs. MILP comparison**: Same GNN architecture applied to MILP instances (via SCIP or HiGHS) and MINLP instances (via discopt), showing that MINLP-specific features (relaxation gap, bound history) matter.

### Target Venue
**NeurIPS** or **ICML**

**Rationale**: GNN branching for combinatorial optimization is a major ML-for-optimization thread at NeurIPS/ICML. Gasse et al. (NeurIPS 2019) and follow-ups have established this as an active research area. Extending from MILP to MINLP is a clear and well-motivated contribution. The end-to-end JAX training loop is a secondary contribution that makes the training methodology significantly cleaner than prior work (which used SCIP callbacks + PyTorch with data serialization).

Alternative: **CPAIOR** (Integration of Constraint Programming, AI, and Operations Research) for a more optimization-focused audience. CPAIOR has published several ML-for-optimization papers and would appreciate the B&B-specific contributions.

### Dependencies
- T14: End-to-end solver (generates training data)
- T19: Batch evaluator (fast enough to generate training data at scale)
- T29: GNN branching policy (the core implementation)
- T27: Piecewise McCormick (provides tighter relaxations that affect branching quality)

### Builds On
- **Paper 1** (McCormick compilation): The relaxation gap features in the GNN input come from Paper 1's compiled relaxations.
- **Paper 2** (batch B&B): The B&B framework that the GNN plugs into.
- **Paper 3** (differentiable optimization): The end-to-end differentiable training pipeline uses Paper 3's custom_jvp machinery.
- **Paper 4** (GPU IPM): The batch NLP solves during training data collection use Paper 4's IPM.

---

## Paper 6: Advanced Convex Relaxations for Global Optimization in JAX: Piecewise McCormick and alphaBB

### Working Title
"Tight Convex Relaxations for Global Optimization via Adaptive Piecewise McCormick and alphaBB in JAX"

### Timing
**Month 30-36** (late Phase 3 / early Phase 4). Submittable once WS10-a delivers piecewise McCormick with adaptive partitioning and alphaBB relaxations, and the Phase 3 gate (root gap <= 1.3x BARON) is approached or met.

### Core Contribution
Two advanced relaxation techniques implemented in JAX for the first time:

1. **Adaptive piecewise McCormick**: Partition the variable domain into k=4-16 pieces and compute tighter McCormick relaxations on each sub-domain. The partition is adapted during B&B based on violation patterns. The JAX implementation is vmappable: all k partitions are evaluated simultaneously via vmap, and the tightest relaxation is selected. This is substantially cheaper on GPU than the serial approach used in BARON's piecewise relaxations.

2. **alphaBB relaxations in JAX**: The alphaBB method (Androulakis et al., 1995) adds a convex underestimator `alpha * sum((x_i - lb_i)(ub_i - x_i))` where alpha is determined by the Hessian eigenvalue. In JAX, `jax.hessian` provides exact Hessian computation, and eigenvalue estimation can be done via `jax.numpy.linalg.eigvalsh`. The entire alphaBB construction is differentiable and JIT-compilable.

The novelty is not the relaxation techniques themselves (both are well-known) but their JAX implementation: batchable, differentiable, and composable with the GPU solver. This enables, for the first time, running tight relaxation techniques at GPU speed.

### Minimum Viable Results
1. **Soundness**: Both relaxation types pass the 10,000-point soundness test at tolerance 1e-10.
2. **Gap reduction**: >= 60% root gap reduction vs. standard McCormick on a suite of at least 20 MINLPLib instances.
3. **Root gap comparison**: Root gap ratio vs. BARON <= 1.3 on the comparison suite (the Phase 3 gate target).
4. **GPU throughput**: Piecewise McCormick evaluation (k=8) on GPU is no more than 2x slower than standard McCormick for the same batch size (the extra tightness is worth the cost).
5. **Adaptive partitioning**: Show that adapting partition points during B&B reduces total nodes compared to uniform partitioning.
6. **AlphaBB vs. McCormick**: Head-to-head comparison on problems where McCormick relaxations are known to be weak (multilinear, signomial, trigonometric).
7. **End-to-end solver improvement**: Full discopt solver with advanced relaxations vs. standard McCormick on Phase 3 benchmark suite.

### Target Venue
**Journal of Global Optimization (JOGO)** or **Mathematical Programming** Series A

**Rationale**: This is a pure global optimization paper. JOGO is the home venue for McCormick relaxation and alphaBB work -- the original alphaBB paper and many piecewise relaxation papers appeared there. Mathematical Programming is appropriate if the paper includes theoretical tightness results (e.g., proving convergence rate improvements for the adaptive partitioning scheme). Both venues have long review cycles (12-18 months), so an early submission at Month 30 targets acceptance by Month 42-48.

Alternative: **Computers & Chemical Engineering** if the paper emphasizes chemical engineering benchmark problems (pooling, blending, reactor networks) where these relaxations have the most impact.

### Dependencies
- T6: Relaxation compiler (base McCormick)
- T19: Batch evaluator (GPU evaluation)
- T27: Piecewise McCormick + alphaBB implementation
- T21: OBBT (needed for competitive root gap)

### Builds On
- **Paper 1** (McCormick compilation): This paper directly extends Paper 1's compiler to support piecewise and alphaBB relaxations.
- **Paper 2** (batch B&B): The GPU batching framework evaluates piecewise relaxations in batch.
- **Paper 4** (GPU IPM): The NLP subproblems generated by tighter relaxations are solved by Paper 4's IPM.

---

## Publication Timeline Summary

| Paper | Month Written | Month Submitted | Phase | Target Venue | Topic |
|-------|:----------:|:-----------:|:-----:|-------------|-------|
| 1 | 8-10 | 10 | Phase 1 end | IJOC / MPC | McCormick-to-JAX compiler |
| 2 | 14-16 | 16 | Phase 2 mid | NeurIPS / ICML | Batch spatial B&B on GPU |
| 3 | 16-20 | 20 | Phase 2 end | NeurIPS / ICML | Differentiable MINLP |
| 4 | 18-22 | 22 | Phase 2-3 | MPC / OMS | Batch GPU IPM |
| 5 | 28-32 | 32 | Phase 3 end | NeurIPS / ICML | GNN branching for MINLP |
| 6 | 30-36 | 36 | Phase 3-4 | JOGO / Math Prog | Advanced relaxations in JAX |

### Cadence
- Paper 1 (Month 10) -> Paper 2 (Month 16): 6 months apart
- Paper 2 (Month 16) -> Paper 3 (Month 20): 4 months apart (these can be written in parallel, different lead authors)
- Paper 3 (Month 20) -> Paper 4 (Month 22): 2 months apart (parallel writing; Paper 4 is more technical/computational)
- Paper 4 (Month 22) -> Paper 5 (Month 32): 10 months apart (Phase 3 development time for GNN training)
- Paper 5 (Month 32) -> Paper 6 (Month 36): 4 months apart

The larger gap between Papers 4 and 5 reflects the Phase 3 development cycle for GNN branching. Consider submitting a workshop paper at NeurIPS/ICML around Month 26-28 with preliminary GNN results to maintain visibility during this gap.

### Narrative Arc

1. **Foundation** (Paper 1, Month 10): "We can compile McCormick relaxations to JAX, making them batchable and differentiable for the first time."
2. **Architecture** (Paper 2, Month 16): "Using those relaxations, we built the first GPU-batched spatial B&B solver, achieving 15x+ speedups."
3. **Differentiability** (Paper 3, Month 20): "The solver is differentiable end-to-end, enabling decision-focused learning with discrete optimization for the first time."
4. **Subsolver** (Paper 4, Month 22): "We replaced Ipopt with a vmappable JAX IPM, making the entire solver GPU-native."
5. **Learning** (Paper 5, Month 32): "With everything in JAX, we train GNN branching policies end-to-end, extending ML-for-MILP to MINLP."
6. **Tightness** (Paper 6, Month 36): "Advanced relaxations in JAX close the gap to BARON while maintaining GPU throughput."

Each paper cites and builds on the previous ones, creating a coherent body of work that establishes discopt as both a practical tool and a research contribution.

---

## Risk Assessment for Each Paper

| Paper | Primary Risk | Mitigation |
|-------|-------------|------------|
| 1 (McCormick compiler) | Low novelty -- McCormick is well-known | Emphasize the JAX compilation and vmap/grad properties as genuinely new; extensive experimental comparison vs. McCormick.jl and mc++ |
| 2 (Batch B&B) | GPU speedup may be lower than 15x on some problems | Report honestly, identify problem classes where batch helps most (dense nonlinear) vs. least (sparse linear); the architecture contribution stands even at 5-10x |
| 3 (Differentiable MINLP) | Level 3 implicit diff may fail on degenerate problems | Perturbation smoothing is the fallback; report Level 1 results as the primary and Level 3 as stretch contribution |
| 4 (GPU IPM) | Dense IPM is slower than Ipopt for single instances | Frame explicitly as batch throughput contribution, not single-instance; honest head-to-head comparison |
| 5 (GNN branching) | Node reduction may be below 20% on hard MINLP | Include MILP comparison showing the technique works; investigate MINLP-specific features; 10-15% reduction is still publishable with good analysis |
| 6 (Advanced relaxations) | Root gap target 1.3x BARON may not be met | Shift framing from "matching BARON" to "closing the gap by X% using JAX-native relaxations at GPU speed" |

---

## Additional Publication Opportunities (Not Full Papers)

### Workshop Papers
- **NeurIPS OPT Workshop** (Month 12-14): Preliminary batch B&B results, before the full Paper 2
- **ICML Differentiable Everything Workshop** (Month 14-16): Preview of differentiable MINLP, before Paper 3
- **NeurIPS ML4CO Workshop** (Month 26-28): Preliminary GNN branching, before Paper 5

### System Demonstration / Short Papers
- **CPAIOR System Demo** (Month 16-18): discopt as a working system with Python API, batch solving, and differentiability
- **SciPy Conference Talk** (Month 20-24): JAX ecosystem integration and scientific application demos

### Benchmark Papers
- **MINLPLib Results Paper** (Month 32-36): Comprehensive benchmarking of discopt against BARON, Couenne, SCIP, Bonmin on the full MINLPLib library. Target: Optimization Methods and Software or INFORMS JoC.

---

## Recommendation on Paper Ordering and Priority

**Highest priority**: Paper 2 (Batch B&B) and Paper 3 (Differentiable MINLP). These are the flagship contributions that establish discopt's unique value proposition. They target top ML venues (NeurIPS/ICML) with the highest visibility and impact.

**Second priority**: Paper 1 (McCormick compiler). Although it is chronologically first, it is the most "incremental" contribution (new implementation of known technique). It establishes the foundation but is less exciting on its own. Writing it first serves the narrative arc and provides a citable reference for Papers 2-6.

**Third priority**: Paper 5 (GNN branching). This is timely given the ML-for-combinatorial-optimization trend and directly leverages discopt's unique all-in-JAX advantage.

**Lower priority**: Papers 4 and 6 are important but more niche (optimization computation audience rather than broad ML audience). They can be written by different team members in parallel with Papers 3 and 5 respectively.
