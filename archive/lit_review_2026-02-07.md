# JaxMINLP Literature Review

**Period:** 2026-01-24 to 2026-02-07
**Generated:** 2026-02-07T01:25:41.077227
**Papers scanned:** 5
**Relevant papers:** 4

## Executive Summary

- **1 critical** papers requiring immediate team review
- **3 high-relevance** papers with potential implementation value
- **0 medium-relevance** papers for background awareness

## Action Items

- [CRITICAL] Review: "Tighter McCormick Relaxations via Affine Arithmetic for Spatial Branch-and-Bound" (components: bound_tightening, branching, global_optimization, relaxation, software)
- [HIGH] Review: "GPU-Accelerated Branch-and-Bound for Mixed-Integer Nonlinear Programs" (components: branching, global_optimization, software)
- [HIGH] Review: "Improved Cutting Planes for Quadratically Constrained Programs via Perspective Reformulation" (components: cutting_planes, global_optimization, relaxation, software)
- [HIGH] Review: "Learning Branching Heuristics for MINLP via Graph Neural Networks" (components: branching, global_optimization, software)

## Papers by JaxMINLP Component

### Global Optimization (4 papers)

- Tighter McCormick Relaxations via Affine Arithmetic for Spatial Branch-and-Bound
- GPU-Accelerated Branch-and-Bound for Mixed-Integer Nonlinear Programs
- Improved Cutting Planes for Quadratically Constrained Programs via Perspective Reformulation
- Learning Branching Heuristics for MINLP via Graph Neural Networks

### Software (4 papers)

- Tighter McCormick Relaxations via Affine Arithmetic for Spatial Branch-and-Bound
- GPU-Accelerated Branch-and-Bound for Mixed-Integer Nonlinear Programs
- Improved Cutting Planes for Quadratically Constrained Programs via Perspective Reformulation
- Learning Branching Heuristics for MINLP via Graph Neural Networks

### Branching (3 papers)

- Tighter McCormick Relaxations via Affine Arithmetic for Spatial Branch-and-Bound
- GPU-Accelerated Branch-and-Bound for Mixed-Integer Nonlinear Programs
- Learning Branching Heuristics for MINLP via Graph Neural Networks

### Relaxation (2 papers)

- Tighter McCormick Relaxations via Affine Arithmetic for Spatial Branch-and-Bound
- Improved Cutting Planes for Quadratically Constrained Programs via Perspective Reformulation

### Bound Tightening (1 papers)

- Tighter McCormick Relaxations via Affine Arithmetic for Spatial Branch-and-Bound

### Cutting Planes (1 papers)

- Improved Cutting Planes for Quadratically Constrained Programs via Perspective Reformulation

## Critical — Immediate Review Required

### Tighter McCormick Relaxations via Affine Arithmetic for Spatial Branch-and-Bound

- **Authors:** Smith, A., Jones, B.
- **Published:** 2026-01-30
- **Source:** [arxiv](https://arxiv.org/abs/2601.99999)
- **Relevance score:** 0.85
- **Components:** bound_tightening, branching, global_optimization, relaxation, software
- **Keywords matched:** affine arithmetic, baron, branch-and-bound, convex envelope, interval arithmetic, mccormick, minlp, minlplib

**Abstract (excerpt):** We present a novel approach to construct McCormick relaxations using affine arithmetic instead of standard interval arithmetic. Our method tracks linear correlations between variables, producing convex envelopes that are provably tighter than classical McCormick on problems with repeated variable oc...

---

## High Relevance

### GPU-Accelerated Branch-and-Bound for Mixed-Integer Nonlinear Programs

- **Authors:** Chen, X., Wang, Y.
- **Published:** 2026-02-01
- **Source:** [arxiv](https://arxiv.org/abs/2601.88888)
- **Relevance score:** 0.60
- **Components:** branching, global_optimization, software
- **Keywords matched:** baron, benchmark, branch-and-bound, minlp, mixed-integer nonlinear, pooling problem, scip, spatial branch

**Abstract (excerpt):** We demonstrate that spatial branch-and-bound for MINLP can be effectively parallelized on GPUs by batching node relaxation evaluations. Our CUDA implementation evaluates up to 512 node relaxations simultaneously, achieving 20x speedup over serial evaluation on pooling problems. We compare against BA...

---

### Improved Cutting Planes for Quadratically Constrained Programs via Perspective Reformulation

- **Authors:** Garcia, M., Brown, T.
- **Published:** 2026-02-04
- **Source:** [arxiv](https://arxiv.org/abs/2602.33333)
- **Relevance score:** 0.55
- **Components:** cutting_planes, global_optimization, relaxation, software
- **Keywords matched:** baron, benchmark, cutting plane, miqcqp, qcqp, quadratically constrained, rlt

**Abstract (excerpt):** We derive new families of valid inequalities for MIQCQP based on the perspective reformulation. Our cuts significantly tighten the LP relaxation for problems with on/off constraints (indicator variables multiplied by quadratic terms). Experiments on process synthesis benchmarks show 40% root gap clo...

---

### Learning Branching Heuristics for MINLP via Graph Neural Networks

- **Authors:** Kim, D., Patel, R.
- **Published:** 2026-02-03
- **Source:** [arxiv](https://arxiv.org/abs/2602.11111)
- **Relevance score:** 0.50
- **Components:** branching, global_optimization, software
- **Keywords matched:** branch-and-bound, minlp, minlplib, nonconvex, reliability branching, scip, spatial branch, strong branching

**Abstract (excerpt):** We train a graph neural network to predict strong branching decisions in spatial branch-and-bound for nonconvex MINLP. The GNN operates on a bipartite graph representation of the LP relaxation at each node. Experiments on MINLPLib show 30% node count reduction versus reliability branching with negli...

---

---
*Generated by JaxMINLP Literature Review Agent. Keyword-based scoring with optional LLM analysis. Review critical papers within 1 week; high-relevance within 2 weeks.*