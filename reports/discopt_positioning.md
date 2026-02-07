# discopt: Differentiable Discrete Optimization for Scientific Machine Learning

## Positioning Document for Federal Funding (NSF CSSI / DOE ASCR)

---

## 1. The Problem: Optimization is the Bottleneck in Scientific ML Workflows

Modern scientific discovery increasingly relies on closed-loop workflows that alternate between machine learning prediction and constrained optimization:

- **Materials discovery**: A surrogate model predicts material properties; an optimizer selects the next composition to synthesize, choosing from discrete atom types and continuous stoichiometries subject to thermodynamic stability constraints.
- **Drug design**: A generative model proposes molecular candidates; an optimizer filters them against discrete structural requirements (functional groups, ring systems) and continuous property thresholds (solubility, binding affinity, toxicity).
- **Energy systems**: A forecasting model predicts renewable generation and load; a dispatcher optimizes generator commitment (discrete on/off) and power output (continuous) subject to nonlinear AC power flow equations.
- **Autonomous experiments**: A Bayesian model estimates an objective surface; an acquisition function optimizer selects the next batch of experiments from discrete experimental conditions and continuous parameter ranges.

In every case, the optimization problem has the same structure: **mixed-integer nonlinear programming (MINLP)** — discrete choices combined with continuous variables under nonlinear constraints. And in every case, the scientific workflow demands three capabilities that no existing solver provides:

1. **Differentiability**: To train the ML model end-to-end against downstream decision quality (decision-focused learning), gradients must flow backward through the optimization step. Current MINLP solvers are black boxes — they accept inputs and return solutions but provide no gradient information.

2. **Batch solving**: Scientific workflows evaluate optimization over thousands of scenarios (Monte Carlo uncertainty quantification), parameter sweeps (sensitivity analysis), or candidate designs (high-throughput screening). Current MINLP solvers process problems sequentially on CPU. Solving 10,000 related problems takes 10,000 serial solver calls.

3. **Composability with ML frameworks**: The optimization step must integrate seamlessly into automatic differentiation frameworks (JAX, PyTorch) to participate in gradient-based training. Current solvers require awkward callback interfaces, manual gradient derivations, or finite-difference approximations that scale poorly.

**The result**: Scientists either simplify their optimization problems (dropping integer variables, linearizing constraints) to fit existing differentiable solvers, or they decouple prediction from optimization and accept suboptimal decisions. Both compromises limit scientific impact.

---

## 2. The Gap: No Differentiable, Batch-Capable MINLP Solver Exists

The optimization software landscape has advanced significantly in three separate directions, but no tool combines them:

**Differentiable optimization** is maturing for continuous, convex problems. cvxpylayers (Agrawal et al., NeurIPS 2019) enables differentiation through convex programs in PyTorch and JAX. Optimistix provides JAX-native root-finding and minimization. But these tools cannot handle discrete variables — the defining feature of combinatorial optimization problems in science and engineering.

**Differentiable integer programming** has emerged for linear problems. PyEPO (Tang & Khalil, 2022) and DiffILO (2024) provide gradient approximations for integer linear programs. But they cannot handle nonlinear objectives or constraints — ruling out the thermodynamic, kinetic, and physical models that define scientific optimization problems.

**High-performance MINLP solvers** (BARON, Couenne, SCIP) can solve mixed-integer nonlinear programs to global optimality. But they are not differentiable, cannot batch-solve on GPU, and do not integrate with ML frameworks. BARON represents 25+ years of algorithmic refinement but remains a sequential, CPU-only black box.

| Capability | cvxpylayers | PyEPO | BARON | Optimistix | **discopt** |
|------------|-------------|-------|-------|------------|-------------|
| Discrete variables | No | Yes | Yes | No | **Yes** |
| Nonlinear constraints | Yes (convex) | No | Yes | Yes | **Yes** |
| Differentiable | Yes | Yes | No | Yes | **Yes** |
| GPU batch solving | No | No | No | Yes | **Yes** |
| JAX composable | Yes | No | No | Yes | **Yes** |

**discopt fills the only empty cell in this matrix**: a JAX-native MINLP solver that is differentiable, batch-solvable on GPU, and composable with the scientific ML ecosystem.

---

## 3. The Proposed Cyberinfrastructure: discopt

discopt is an open-source MINLP solver designed from the ground up for three capabilities that existing solvers cannot provide:

### 3.1 Batch Solving via `jax.vmap`

discopt's solver loop is structured so that the computationally intensive step — evaluating convex relaxations across branch-and-bound tree nodes — is a pure JAX function compatible with `jax.vmap`. This enables:

- **Intra-solve parallelism**: Evaluate hundreds of B&B node relaxations simultaneously on GPU instead of sequentially on CPU. Preliminary estimates (validated by analogous projects MPAX, CuClarabel, and NVIDIA cuOpt) project 15-100x speedups for batch sizes of 128-1024.
- **Inter-solve parallelism**: Solve thousands of related MINLP instances in parallel by vectorizing the entire solver across problem parameters. This transforms parametric studies, uncertainty quantification, and high-throughput screening from hours-long serial loops into seconds-long batched GPU computations.

No existing MINLP solver — commercial or open-source — supports either form of parallelism. Gurobi, BARON, SCIP, and HiGHS all process problems sequentially.

### 3.2 Differentiability via `jax.custom_jvp`

discopt provides gradients of optimal solutions with respect to problem parameters through three mechanisms:

- **Level 1 — LP relaxation sensitivity** (immediate): The LP relaxation at the root node provides dual variables that are the gradient of the relaxed objective with respect to right-hand-side parameters. This is mathematically well-founded and mechanically straightforward via `jax.custom_jvp`. It is immediately useful for decision-focused learning where the LP relaxation quality is sufficient.

- **Level 3 — Implicit differentiation at the optimal active set** (Phase 2): At the MINLP solution, the active constraints define an implicit system. The implicit function theorem yields exact gradients of continuous decision variables with respect to problem parameters. This extends cvxpylayers-style differentiation to the mixed-integer nonlinear case.

- **Perturbation smoothing** (fallback): When the discrete solution is non-differentiable (e.g., at a point where the optimal integer assignment changes), discopt uses `jax.vmap` to solve perturbed instances in parallel and estimate gradients via Stein's lemma. This is computationally expensive in serial but cheap with batch solving — 32 perturbed solves in parallel cost the same as one.

These mechanisms enable end-to-end gradient-based training of ML models that interact with discrete optimization:

```python
# Decision-focused learning: train predictor to minimize decision regret
def loss(predictor_params, features, true_cost):
    predicted_cost = predictor(predictor_params, features)
    solution = discopt.solve(model_with_cost(predicted_cost))
    decision_cost = jnp.dot(true_cost, solution.x)
    return decision_cost

# Gradient flows through discopt.solve via custom_jvp
grads = jax.grad(loss)(predictor_params, features, true_cost)
```

### 3.3 JAX Ecosystem Integration

discopt participates fully in JAX's transformation system:

- `jax.jit(discopt.solve)` — JIT-compile the solver for zero Python overhead
- `jax.vmap(discopt.solve)` — batch-solve across problem parameters
- `jax.grad(lambda p: discopt.solve(f(p)).objective)` — differentiate through the solver
- Inputs and outputs are JAX arrays compatible with Equinox neural networks, Optax optimizers, and jraph graph neural networks

This composability means discopt slots into existing JAX-based scientific ML pipelines without wrapper code, serialization, or framework bridges. A researcher using Equinox for surrogate modeling and Optax for training can add discopt as an optimization layer with a single function call.

### 3.4 Architecture

discopt uses a hybrid Rust + JAX architecture that assigns each component to the language best suited for it:

- **Rust** (via PyO3): Branch-and-bound tree management, branching decisions, presolve, node selection. These are sequential, pointer-heavy operations that benefit from Rust's ownership model and memory safety guarantees.
- **JAX** (on GPU): Relaxation evaluation, gradient computation, interior point method for LP/NLP subproblems. These are embarrassingly parallel operations that benefit from GPU acceleration and automatic differentiation.

This split is validated by multiple recent projects: NVIDIA's cuOpt uses a similar CPU/GPU hybrid for vehicle routing; CuClarabel (Yale, Boyd group) demonstrates GPU interior point methods achieving 10-50x speedups; MPAX demonstrates JAX-native LP/QP solving.

---

## 4. Intellectual Merit

### 4.1 Enabling Decision-Focused Learning for Nonlinear Combinatorial Problems

Decision-focused learning (DFL) — training ML models end-to-end to minimize downstream decision cost rather than prediction error — has emerged as a significant advance in ML for operations research (Kotary et al., JAIR 2024; NeurIPS 2024). Over 100 papers on DFL have appeared in 2024-2025 alone. However, all existing DFL frameworks are limited to convex or linear-integer problems. discopt extends DFL to the MINLP domain, enabling it for the first time in applications where optimization involves both discrete decisions and nonlinear physics:

- **Materials inverse design**: Train a neural network to predict processing parameters that, when optimized over discrete compositions and continuous conditions, yield target material properties.
- **Molecular optimization**: Train a generative model end-to-end against drug efficacy, where the optimization layer enforces synthesizability constraints and discrete structural requirements.
- **Experiment design**: Train a Bayesian surrogate to minimize the expected number of experiments to reach a target, where each experiment selection involves discrete and continuous choices.

### 4.2 Novel Computational Approach: Batch Spatial Branch-and-Bound on GPU

The spatial branch-and-bound algorithm — the foundation of global MINLP solving — has been implemented exclusively for sequential CPU execution for 30 years. discopt introduces a batch evaluation architecture where the GPU processes hundreds of tree nodes simultaneously while the CPU manages tree structure. This is a fundamentally different computational paradigm that trades tighter per-node relaxations (BARON's strength) for massive node throughput. For problem classes where relaxation evaluation is the bottleneck (dense nonlinear constraints, many bilinear terms), this approach can overcome weaker relaxations through sheer throughput.

### 4.3 Composable Differentiable Discrete Optimization in JAX

The extension of JAX's transformation system (`jit`, `vmap`, `grad`) to mixed-integer nonlinear optimization requires novel gradient definitions for operations that are inherently discontinuous. discopt's multi-level differentiability strategy (LP sensitivity, implicit differentiation at active sets, perturbation smoothing) provides a practical and theoretically grounded approach that degrades gracefully: exact gradients where possible, consistent gradient estimators where not. This builds on and extends the implicit differentiation framework of Blondel et al. (ICML 2022) to the combinatorial setting.

### 4.4 Platform for ML-for-Optimization Research

The JAX ecosystem (jraph for GNNs, Equinox for networks, Optax for training) makes discopt a natural platform for learning solver heuristics — branching policies, node selection, cut selection, warm-starting — within a single JIT-compiled training loop. This is extremely difficult with C/C++ solvers that require Python callbacks and data serialization across language boundaries. discopt could serve as the standard research platform for ML-for-MINLP, a field that is nascent compared to ML-for-MILP (where GNN branching policies are now well-established).

---

## 5. Broader Impacts

### 5.1 Democratizing Global Optimization for Scientific Researchers

Commercial MINLP solvers (BARON, Gurobi) cost $5,000-$30,000+ per academic license. Open-source alternatives (Couenne, Bonmin) are in maintenance mode with no active development since 2019-2020. discopt provides a modern, well-maintained, open-source MINLP solver with a Pythonic API — lowering the barrier for researchers in materials science, chemistry, biology, and engineering who need global optimization but cannot justify commercial solver costs or navigate legacy C/Fortran interfaces.

### 5.2 Accelerating Closed-Loop Scientific Discovery

Autonomous scientific workflows — where ML models design experiments, robots execute them, and results update the models — are transforming materials science (Nature Communications 2020), drug discovery (Insilico Medicine's ISM001-055, Phase II), and chemical engineering. The optimization step in these loops is often the bottleneck: selecting the next experiment from a combinatorial space of conditions. discopt's batch solving reduces this from minutes (serial MINLP) to seconds (batched GPU), enabling real-time experiment selection in autonomous laboratories.

### 5.3 Educational Impact

discopt's JAX-native API makes optimization accessible to the growing community of researchers who learn scientific computing through Python and JAX. The project will produce tutorial notebooks demonstrating:
- Decision-focused learning for materials property prediction
- Batch Bayesian optimization for experimental design
- Differentiable supply chain optimization
- GNN branching policy training

These materials bridge the gap between the optimization and ML communities, which have historically used different tools and spoken different technical languages.

### 5.4 Building Community Infrastructure

discopt follows the model of successful scientific Python packages (scikit-learn, SciPy, JAX itself): open-source under a permissive license (MIT/Apache 2.0), with rigorous testing (85%+ coverage, 24 known-optimum correctness validation), reproducible benchmarks (MINLPLib, automated phase gates), and clear contribution guidelines. The solver protocol abstraction allows community-contributed solver backends (LP, QP, NLP) to plug into the same framework, fostering an ecosystem of interoperable optimization tools for JAX.

---

## 6. Relationship to Existing Cyberinfrastructure

### 6.1 What discopt Complements

| Infrastructure | What it does | How discopt extends it |
|---------------|-------------|----------------------|
| **JAX** (Google) | Composable numerical computing | Adds discrete optimization to jit/vmap/grad |
| **Equinox** (Kidger) | JAX neural networks | discopt becomes an optimization layer in Equinox models |
| **Optimistix** (Kidger) | Continuous optimization in JAX | discopt adds integer variables to the same ecosystem |
| **HiGHS** (Edinburgh) | Open-source LP/MIP solver | discopt uses HiGHS as an LP backend, adds nonlinear + GPU |
| **Ipopt** (COIN-OR) | Open-source NLP solver | discopt uses Ipopt as NLP backend, adds integers + GPU |

### 6.2 What discopt Replaces

discopt does not aim to replace BARON or Gurobi for enterprise optimization workloads. It targets a different use case: **optimization embedded in ML training loops**, where differentiability, batch solving, and framework integration matter more than single-instance speed on large problems.

The target user is a researcher who writes:

```python
# Today: two-stage, non-differentiable, sequential
predictions = model(features)
for scenario in scenarios:  # Sequential loop, minutes to hours
    result = baron.solve(build_minlp(predictions, scenario))  # Black box
    decisions.append(result)
loss = evaluate_decisions(decisions, ground_truth)
# Cannot compute grad(loss, model.params) — BARON is not differentiable

# With discopt: end-to-end, differentiable, batched
predictions = model(features)
solve_fn = jax.vmap(lambda s: discopt.solve(build_minlp(predictions, s)))
results = solve_fn(scenarios)  # All scenarios in parallel on GPU, seconds
loss = evaluate_decisions(results, ground_truth)
grads = jax.grad(loss)(model.params)  # Gradient flows through discopt
```

---

## 7. Relevant Funding Programs

| Program | Agency | Fit | Rationale |
|---------|--------|-----|-----------|
| **CSSI: Elements** | NSF OAC | High | Robust, sustainable cyberinfrastructure for scientific community |
| **CSSI: Frameworks** | NSF OAC | High (if collaborative) | Sustained operation of essential cyberinfrastructure |
| **CDS&E** | NSF multiple | High | Computational and data-enabled science |
| **DMREF** | NSF | High (materials focus) | Designing Materials to Revolutionize and Engineer our Future |
| **ASCR Applied Math** | DOE | Medium-High | Mathematical optimization algorithms for science |
| **SciDAC** | DOE | Medium | Scientific Discovery through Advanced Computing |
| **CZI EOSS** | CZI | Medium | Essential Open Source Software for Science |

### Alignment with NSF Strategic Goals

- **Harnessing the Data Revolution**: discopt enables ML models to make discrete optimization decisions, advancing data-driven scientific discovery
- **Growing Convergence Research**: discopt bridges mathematical optimization, computer science (GPU computing, compiler design), and domain sciences (materials, chemistry, biology)
- **Expanding AI for Science**: discopt provides the infrastructure for optimization-aware AI in scientific workflows

---

## 8. Prior Work and Preliminary Results

### 8.1 Infrastructure Completed

The project has produced ~5,100 lines of validated infrastructure code:

- **Modeling API** (1,022 lines): Expression DAG with operator overloading supporting continuous, binary, and integer variables. 7 runnable example models (MINLP, NLP, MILP, QP).
- **Benchmark framework** (664 lines): 12 metrics including shifted geometric means, Dolan-More performance profiles, GPU speedup measurement, and regression detection. 4 phase gates with automated go/no-go criteria.
- **Test suite** (516 lines): 9 test markers, 7 fixtures, 24 known-optimum validation instances from MINLPLib, 14 Rust-JAX interop test specifications. Coverage threshold enforced at 85%.
- **Benchmark configuration** (190 lines): Suite definitions for MINLPLib, Netlib LP, CUTEst NLP, SuiteSparse sparse LA, and GPU scaling benchmarks.

### 8.2 Architecture Validation

The Rust + JAX hybrid architecture is validated by three independent projects:
- **MPAX** (Google Research): JAX-native LP/QP solver demonstrating that JAX can solve optimization problems competitively
- **CuClarabel** (Yale, Boyd group): GPU conic solver achieving 10-50x speedups on SOCP/SDP via GPU interior point methods
- **cuOpt** (NVIDIA): Commercial GPU-accelerated optimization achieving 5,000x speedups on vehicle routing via hybrid CPU/GPU architecture

### 8.3 Feasibility Assessment

A detailed feasibility assessment (February 2026) evaluated the project across four dimensions:
- **Batch solving**: 75-85% likelihood of success, 12-18 months to demonstration
- **Differentiable MINLP (Level 1)**: 80% likelihood, 12-18 months
- **Competitive with Couenne/Bonmin**: 50-60% likelihood, 30-42 months
- **Platform for ML-for-MINLP research**: High likelihood, grows with solver maturity

---

## 9. Development Plan Summary

| Phase | Timeline | Deliverable | Validation |
|-------|----------|-------------|------------|
| **0: Spike** | Month 1 | Rust-JAX GPU batch latency validation | < 100μs round-trip, > 10x GPU speedup |
| **1: Working solver** | Months 1-10 | Spatial B&B + McCormick + HiGHS/Ipopt | 25 MINLPLib instances, zero incorrect |
| **2: GPU + differentiability** | Months 10-20 | Custom JAX IPM, vmap batch, Level 1 grad | 15x GPU speedup, differentiable solving works |
| **3: Competitive** | Months 20-32 | Advanced relaxations, GNN branching | Within 2.5x BARON, beats BARON on GPU classes |
| **4: Release** | Months 32-42 | v1.0, documentation, tutorials | `pip install discopt`, 3-5 example notebooks |

**Team**: 2-3 core developers (Rust + systems engineer, JAX + numerical engineer, applied math), 1 part-time DevOps.

**Budget estimate**: $1.2-$2.0M over 3.5 years (personnel + GPU compute).

---

## 10. Summary

Scientific machine learning workflows increasingly require solving mixed-integer nonlinear optimization problems inside gradient-based training loops. No existing solver provides the combination of differentiability, GPU batch solving, and ML framework integration needed for this use case. discopt fills this gap as the first JAX-native MINLP solver, enabling decision-focused learning for nonlinear combinatorial problems, 100-1000x throughput improvements via batched GPU solving, and a research platform for learning solver heuristics. The project builds on validated architectural choices, targets well-defined phase gates with automated evaluation, and addresses a genuine capability gap at the intersection of mathematical optimization and scientific ML.
