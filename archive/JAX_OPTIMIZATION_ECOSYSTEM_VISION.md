# JAX Optimization Ecosystem Vision

## Context

JaxMINLP is currently a MINLP solver with a Rust backend (LP solving, B&B tree management), JAX (autodiff, NLP relaxations, GPU acceleration), and Python orchestration. This document captures the vision for expanding it into a **comprehensive JAX-native optimization ecosystem** covering LP, MILP, QP, MIQP, NLP, and MINLP — with differentiable solving, batch solving via `vmap`, and learned solver heuristics.

### Why This Ecosystem

No JAX-native solver ecosystem exists that handles integer variables with both differentiability and competitive performance. The closest alternatives (JAXopt, CVXPY+diffcp) lack integer support, GPU-native solving, or JAX composability. This is the gap.

### Three Value Propositions

1. **`vmap` batch solving**: Solve thousands of related optimization problems in parallel on GPU. No existing solver can do this. Gurobi/HiGHS process problems sequentially.
2. **Differentiable optimization layers**: Optimization as a differentiable layer in neural networks for predict-then-optimize, decision-focused learning, and learning constraints.
3. **Learn-to-optimize**: Train ML models (branching policies, cut selection, IPM warm-starting) using JAX-traceable solvers with `jit`, `vmap`, and `grad`.

---

## Architecture

### Layered Package Structure

```
jax-optcore    (expression DAG, variable types, solver protocol, JAX utils)
  ├── jax-lp      (simplex + GPU interior point)
  ├── jax-qp      (GPU interior point, active set)
  ├── jax-milp    (branch-and-cut, uses jax-lp for relaxations)
  ├── jax-miqp    (branch-and-cut, uses jax-qp for relaxations)
  └── jax-minlp   (spatial B&B, McCormick, uses jax-lp) ← THIS REPO
```

Optional meta-package `jax-opt` provides unified import with structure detection.

### Rust + JAX Hybrid Backend

**Decision**: Rust for CPU-bound sequential logic, JAX for GPU-parallel computation, targeted CUDA kernels via `jax.extend.ffi` for specialized operations (sparse Cholesky).

The boundary aligns with the **discrete/continuous split**:

| Component | Best In | Why |
|---|---|---|
| B&B tree traversal | **Rust** | Sequential, pointer-heavy, ownership model prevents bugs |
| Branch/cut decisions | **Rust** (action) + **JAX** (policy) | Discrete decisions, learnable via policy gradient |
| Presolve, bound tightening | **Rust** | Sequential reductions |
| LP/QP subproblem solving | **JAX + GPU** | Dense linear algebra, embarrassingly parallel across nodes |
| NLP relaxation evaluation | **JAX + GPU** | Autodiff, GPU-parallel McCormick |
| GNN policy evaluation | **JAX + GPU** | Matrix multiply, message passing |
| Sparse symbolic factorization | **Rust** | Graph algorithm, one-time cost |
| Sparse numeric factorization | **CUDA via jax.extend.ffi** | GPU-accelerated, reuses symbolic pattern |

**Why not C++/CUDA monolith**: Loses `vmap`, `grad`, and Flax/Equinox composability — kills all three value propositions.

**Why not pure JAX**: B&B tree management fights JAX's functional model (variable-length, pointer-heavy, sequential). Rust excels here.

**Transfer overhead**: Rust↔Python↔JAX round-trip is ~50-100μs. For batches of 100+ nodes with GPU compute of 1-100ms, overhead is <1%. Phase 2 gate validates: `rust_overhead ≤ 0.05`, `interop_overhead ≤ 0.05`.

---

## API Design

### Philosophy: Hybrid Familiar + JAX-Native

Familiar `Model()` API with auto-detection (Philosophy A) plus `.to_jax_function()` escape hatch for power users (Philosophy C). Structure detection is automatic with user override.

### Unified Interface Across Problem Types

```python
import jaxopt as jo

m = jo.Model("my_problem")
x = m.continuous("x", shape=(n,), lb=0, ub=10)
y = m.binary("y", shape=(k,))
price = m.parameter("price", value=50.0)

m.minimize(objective_expression)
m.subject_to(constraints)

# Standard solve — auto-detects problem type (LP/MILP/QP/MIQP/NLP/MINLP)
result = m.solve()
result.problem_type  # "MIQP" — what was detected

# Override detection
result = m.solve(solver="minlp")

# Sensitivity (already in current API)
grad = result.gradient(price)

# Batch solve via vmap
batch_results = m.batch_solve(parameter_values)

# Export for advanced JAX use
solve_fn = m.to_jax_function()  # vmappable, differentiable
jax.vmap(solve_fn)(batch_params)
jax.grad(lambda p: solve_fn(p).objective)(params)
```

### Structure Detection

The expression DAG already contains enough information to classify problems. Detection walks the DAG:
- Any `FunctionCall` nodes (exp, log, sin)? → nonlinear
- Any `BinaryOp("mul", Variable, Variable)` (bilinear)? → nonlinear
- Quadratic terms only? → QP/MIQP
- All linear? → LP/MILP
- Integer/binary variables present? → mixed-integer variant

---

## Solver Algorithms

### GPU Interior Point Method — The Universal Engine

GPU IPM is the foundation for the entire ecosystem:

| Problem | IPM Configuration |
|---|---|
| **LP** | Standard IPM (Q=0) |
| **QP** | IPM with Hessian in Newton system |
| **MILP** | Batched LP relaxation IPM across B&B nodes |
| **MIQP** | Batched QP relaxation IPM across B&B nodes |
| **NLP** | IPM for NLP subproblems (SQP outer loop) |
| **MINLP** | IPM for relaxed NLP subproblems |

Each IPM iteration solves: `(A D² Aᵀ) Δy = rhs` (normal equations, symmetric positive definite).

### B&B Warm-Starting Strategy

IPM lacks warm-starting (unlike simplex). GPU parallelism compensates:

```
CPU Simplex:  1000 nodes × 5 pivots × 0.1ms/pivot  = 500ms (sequential)
GPU IPM:      1000 nodes × 40 iters batched on GPU  = 50ms  (parallel)
```

Additional strategies:
- IPM warm-start from parent iterate (Gondzio): reduces iterations from 40 → 10-15
- Neural warm-starting: predict IPM starting point from problem features (learn-to-optimize)
- Crossover at root: IPM → basic solution → simplex for children (hybrid)

### Rust Simplex Fallback

Keep existing Rust simplex for small/sparse single-instance problems where GPU overhead isn't worth it. Auto-select based on problem size.

### Cutting Planes (MILP)

Generated in Rust (Gomory, MIR, cover cuts), applied in batched GPU LP solves. Cut selection is a learning opportunity (GNN policy).

---

## Sparse Linear Algebra — Tiered Approach

### Tier 1: Dense GPU Cholesky (Ship First)

```python
@jax.jit
def ipm_iteration(A, Q, D, rhs):
    M = A @ jnp.diag(1.0 / (jnp.diag(Q) + 1.0/D**2)) @ A.T
    L = jax.scipy.linalg.cholesky(M, lower=True)
    return jax.scipy.linalg.cho_solve((L, True), rhs)

batch_solve = jax.vmap(ipm_iteration, in_axes=(None, None, 0, 0))
```

- Covers problems up to ~5,000 variables
- Pure JAX — fully `jit`, `vmap`, `grad` compatible
- Covers: portfolio, scheduling, facility location, ML problems

### Tier 2: Iterative Solver (PCG) — Pure JAX

- Preconditioned Conjugate Gradient with cross-iteration warm-starting
- Scales to ~50K variables without forming dense matrices
- Pure JAX — `vmap` compatible
- Moderate accuracy (1e-6 to 1e-8), sufficient for B&B node solves

### Tier 3: Sparse GPU Cholesky — Custom CUDA

- cuSOLVER sparse Cholesky via `jax.extend.ffi`
- Symbolic factorization in Rust (one-time), numeric on GPU
- Scales to 1M+ variables
- For: power systems, transportation networks, large MINLPLib instances

### Tier 4: Learned Preconditioners (Research)

- GNN predicts preconditioner for a given sparsity pattern
- Fully JAX-native, `vmap`-compatible

---

## Learn-to-Optimize

### Learnable Decisions in B&B

| Decision | Impact | Training Method |
|---|---|---|
| Variable branching | Very high | Imitation learning (strong branching expert) → RL fine-tuning |
| Node selection | High | Learned priority function |
| Cut selection | High | Policy network |
| IPM warm-starting | High | Supervised (predict IPM solution from problem features) |
| Primal heuristics | Medium | Learned search strategies |

### Architecture: Bipartite GNN

State-of-the-art for MILP (Gasse et al. 2019). Bipartite graph with variable nodes and constraint nodes connected by coefficient edges. Works for MILP, MIQP, and MINLP — same structure, same training pipeline.

Variable features: type, objective coeff, LP value, fractionality, reduced cost, bound tightness.
Constraint features: slack, dual value, RHS, type, is-cut flag.
Edge features: coefficient A[i,j].

### Training Pipeline

1. **Imitation learning**: Collect expert decisions from strong branching. Supervised cross-entropy loss.
2. **RL fine-tuning**: REINFORCE with reward = -nodes_explored. `vmap` over 1000 instances for parallel training.
3. **Problem-specific fine-tuning**: Adapt pretrained policy to user's problem distribution.

### JAX Advantage

- `vmap`: Train on 1000 instances simultaneously (vs. sequential in PyTorch)
- `jit`: Policy evaluation in microseconds (called thousands of times per solve)
- `grad`: End-to-end differentiation through IPM subproblems for warm-start learning

### Pluggable Interface

```python
result = m.solve(branching=jo.load_policy("branching_gnn.pkl"))
```

---

## Differentiable B&B

### Four Levels (Build Incrementally)

**Level 1 — LP relaxation sensitivity (Build now)**:
Forward: exact MILP solve (Rust B&B). Backward: differentiate LP relaxation at root via KKT duality. Implemented via `jax.custom_jvp`. Immediately useful for predict-then-optimize.

**Level 2 — Truncated soft B&B (Research prototype)**:
Replace hard argmax branching with softmax (temperature-annealed). Explore top-k branches to depth 3-5. Fully differentiable but approximate. Use for policy training only, not deployment.

**Level 3 — Implicit differentiation at optimal active set (Build with MILP/MIQP)**:
At the MILP solution, active constraints define an implicit system. Apply implicit function theorem for exact gradients of continuous variables w.r.t. parameters. Falls back to perturbation smoothing at degenerate points.

**Level 4 — Fully differentiable neural B&B (Open research)**:
Replace entire B&B with a fixed-depth differentiable architecture. Aspirational — nobody has made this work at scale.

### Perturbation Smoothing (Practical Fallback)

Zeroth-order gradient estimation: solve N perturbed instances via `vmap`, estimate gradient via Stein's lemma. `vmap` makes this practical — 32 perturbed solves in parallel.

### Predict-Then-Optimize Application

Train ML models to predict optimization parameters that lead to good decisions (decision-focused learning), not just accurate predictions. Requires differentiating through the solver — Level 1 or Level 3.

---

## Phasing Strategy

### Phase 1 (Now → Month 14): MINLP Foundation

**Focus**: Ship core MINLP solver. No ecosystem expansion yet.

Infrastructure built that feeds ecosystem later:
- Rust B&B tree → reusable for MILP/MIQP
- Expression DAG → shared across all types
- Python API → unified interface
- Test/benchmark framework → shared

### Phase 2 (Month 14 → 26): GPU Engine + Extract Core

**MINLP**: GPU-batched McCormick, 15x GPU speedup.

**Ecosystem**:
- Extract `jax-optcore` (shared DAG, types, solver protocol, JAX utils)
- Build JAX GPU IPM (dense, Tier 1) — serves MINLP AND becomes LP/QP engine
- Level 1 differentiable solving (`custom_jvp` for LP relaxation sensitivity)

### Phase 3 (Month 26 → 38): MILP + MIQP + Learned Branching

**MINLP**: Learned branching (20% node reduction), ≤2.5x vs BARON.

**Ecosystem**:
- Ship `jax-milp`: Rust B&B + GPU LP relaxations + Gomory/MIR cuts
- Ship `jax-miqp`: Rust B&B + GPU QP relaxations
- GNN branching policy (shared across MILP/MIQP/MINLP)
- Iterative solver Tier 2 (PCG)
- Level 3 differentiable solving (implicit differentiation)

### Phase 4 (Month 38 → 48): Polish + Differentiable + Scale

**MINLP**: Release quality, beat Couenne/Bonmin.

**Ecosystem**:
- `custom_jvp`/`custom_vjp` for all solvers
- `to_jax_function()` interface for `vmap` batch solving
- Sparse Tier 3 (cuSOLVER via `jax.extend.ffi`)
- Meta-package `jax-opt` with structure detection + auto-routing
- Predict-then-optimize toolkit

### Suggested MILP/MIQP Phase Gates

```toml
[gates.milp_alpha]
description = "MILP Alpha: basic correctness"
  miplib_easy_solved = { min = 20 }
  geomean_vs_highs = { max = 10.0 }
  zero_incorrect = { max = 0 }

[gates.milp_beta]
description = "MILP Beta: competitive performance"
  miplib_solved = { min = 50 }
  geomean_vs_highs = { max = 3.0 }
  gpu_batch_speedup = { min = 10.0 }
  zero_incorrect = { max = 0 }
```

---

## Shared Infrastructure Multiplier

| Component | Built In | Used By |
|---|---|---|
| Rust B&B engine | Phase 1 | MINLP, MILP, MIQP |
| Expression DAG | Phase 1 | All |
| Python API | Phase 1 | All |
| GPU IPM (dense) | Phase 2 | LP, QP, MILP, MIQP, NLP, MINLP |
| Benchmark framework | Phase 1 | All |
| GNN branching policy | Phase 3 | MILP, MIQP, MINLP |
| Differentiable interface | Phase 4 | All |

By Phase 3, adding MILP is "plug LP relaxation into existing B&B" — marginal effort is small relative to infrastructure already built.

---

## Implementation: What to Do Now

This vision document captures design decisions for the full ecosystem. **No code changes are needed now.** The current MINLP Phase 1 work proceeds as planned. The ecosystem expands starting mid-Phase 2 when GPU IPM and core extraction become relevant.

**Immediate action**: Save this document as a reference alongside the project. Revisit at the Phase 1→2 boundary to plan the `jax-optcore` extraction and GPU IPM implementation.
