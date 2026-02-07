# discopt: Publication Milestones and Strategy

**Date:** 2026-02-07
**Author:** Software Reviewer (Publication Review Team)

---

## Executive Summary

The discopt development plan (42-month timeline, 4 phases) creates at least **8 natural publication opportunities** spanning workshop papers, software papers, benchmark studies, systems papers, and position/survey papers. The strategy sequences publications to build credibility incrementally: early workshop papers establish the ideas, a JOSS paper legitimizes the software, benchmark and systems papers demonstrate value, and position/survey papers claim the intellectual territory.

Key finding: the **earliest publishable units emerge around Month 6-8** (workshop papers), the **JOSS software paper becomes viable at Month 12-14**, and the **highest-impact publications** (benchmark comparison, differentiable optimization) require Phase 2 completion (~Month 20).

---

## Publication Timeline Overview

| # | Paper (working title) | Type | Timing | Venue | Phase Dependency |
|---|----------------------|------|--------|-------|------------------|
| P1 | Workshop: Batch Spatial B&B on GPU | Workshop | Month 6-8 | NeurIPS/ICML OPT workshop | Phase 0-1 partial |
| P2 | Position: Differentiable Discrete Optimization for Scientific ML | Position/Survey | Month 8-12 | arXiv preprint / AAAI workshop | Conceptual (minimal code) |
| P3 | JOSS: discopt — A JAX-native MINLP Solver | Software | Month 12-14 | JOSS | Phase 1 complete |
| P4 | Systems: Rust+JAX Hybrid Architecture for GPU-Accelerated Optimization | Systems | Month 16-20 | SoftwareX or SC/PPoPP workshop | Phase 2 partial |
| P5 | Benchmark: discopt vs BARON/Couenne/SCIP on MINLPLib | Benchmark | Month 20-24 | Mathematical Programming Computation | Phase 2 complete |
| P6 | Methods: Differentiable MINLP via Multi-Level Gradient Strategies | Methods | Month 20-24 | ICML / NeurIPS main | Phase 2 complete |
| P7 | Tutorial: Decision-Focused Learning with discopt | Tutorial/Educational | Month 22-28 | Computing in Science & Engineering or LJCMS | Phase 2 complete |
| P8 | Methods: Learned Branching Policies for Spatial B&B | Methods | Month 28-34 | AAAI / CPAIOR | Phase 3 partial |

---

## Detailed Paper Analyses

### P1: Workshop Paper — Batch Spatial Branch-and-Bound on GPU via JAX

**Working title:** "Batch Spatial Branch-and-Bound: GPU-Accelerated Node Evaluation for Mixed-Integer Nonlinear Programming"

**Timing:** Month 6-8 (writable once first MINLP instances are being solved, even if Phase 1 gate is not yet passed)

**Core contribution:**
- Introduce the architectural idea of batching B&B node relaxation evaluations via `jax.vmap` on GPU
- Present the Rust (tree management) + JAX (node evaluation) split as a design pattern
- Show preliminary speedup numbers on a handful of solved instances (even 5-10 instances suffice for a workshop paper)
- Position this as a fundamentally different computational paradigm from BARON/Couenne/SCIP's sequential approach

**Minimum viable results:**
- Architectural spike data (T0): round-trip latency < 100us, GPU vs CPU comparison
- At least 5 MINLPLib instances solved correctly (from T14/T15 in progress)
- Batch evaluation speedup measurements at batch sizes 32-512 (even on synthetic or partially-solved trees)
- Does NOT require competitive performance with BARON -- only needs to show the mechanism works

**Target venue:**
- **Primary:** NeurIPS Optimization for Machine Learning (OPT) Workshop (December deadline ~September)
- **Alternative:** ICML Workshop on Computational Optimization (deadline ~May)
- **Alternative:** INFORMS Computing Society Conference

**Dependencies:** T0 (spike), T1-T2 (Rust IR), T4-T6 (relaxation compiler), T11-T12 (B&B + batch dispatch), T14 (first solve). Not all of T15 (full 24 instances) required.

**Builds on:** None (first discopt publication)

**Strategic value:** Establishes priority on the batch B&B idea. Workshop papers have low review bars (4-6 page extended abstracts) and fast turnaround (~2 months). Gets discopt's name into the NeurIPS/ICML community early. Can later be expanded into P4 (systems paper) or P5 (benchmark paper).

---

### P2: Position/Survey Paper — Differentiable Discrete Optimization for Scientific ML

**Working title:** "The Missing Layer: Differentiable Discrete Optimization for Scientific Machine Learning Workflows"

**Timing:** Month 8-12 (can be written with minimal code results; primarily a conceptual contribution)

**Core contribution:**
- Survey the gap between differentiable optimization (cvxpylayers, Optimistix) and integer programming (BARON, SCIP)
- Catalog scientific ML workflows that require differentiable MINLP: materials design, drug discovery, autonomous experiments, energy systems
- Present the multi-level differentiability strategy (LP sensitivity, implicit differentiation, perturbation smoothing) as a framework
- Position discopt as the first tool addressing this gap
- This is essentially a refined, peer-reviewed version of the positioning document with a thorough literature survey

**Minimum viable results:**
- Literature survey of decision-focused learning (100+ papers in 2024-2025)
- The capability gap table from the positioning document, expanded with detailed comparisons
- Conceptual architecture diagram of discopt
- Does NOT require discopt to fully work -- this is a "vision + gap analysis" paper
- Ideally includes a small proof-of-concept demo (e.g., Level 1 differentiability on a toy problem)

**Target venue:**
- **Primary:** arXiv preprint first, then submit to a workshop (AAAI Workshop on AI for Scientific Discovery, or NeurIPS AI4Science)
- **Alternative:** ACM Computing Surveys (if expanded significantly, 30+ pages)
- **Alternative:** WIREs Computational Molecular Science (if scoped to chemistry/materials)
- **Journal version:** Annual Review of Control, Robotics, and Autonomous Systems (high visibility, invited review style)

**Dependencies:** Primarily literature review effort. Benefits from T0 (spike data) and T14 (first solve) for a proof-of-concept, but these are not strictly required.

**Builds on:** None (first conceptual paper)

**Strategic value:** Claims the intellectual territory for "differentiable discrete optimization for scientific ML." Even without a finished solver, this paper frames the problem space and positions discopt as the natural solution. If published before competitors, it becomes the cited reference when anyone works in this space. Critically, this paper supports grant applications (NSF CSSI, DOE ASCR) by demonstrating the team understands the landscape.

---

### P3: JOSS Software Paper — discopt

**Working title:** "discopt: A JAX-Native Mixed-Integer Nonlinear Programming Solver for Differentiable and Batch Optimization"

**Timing:** Month 12-14 (after Phase 1 gate is passed, with early Phase 2 work demonstrating differentiability)

**Core contribution:**
- Describe the software: Python API, Rust B&B engine, JAX relaxation compiler, McCormick relaxations
- Demonstrate correctness on 25+ MINLPLib instances
- Show the three unique capabilities: `jax.vmap` batch solving (even partial), `jax.grad` through LP relaxation (Level 1), and JAX ecosystem composability
- Open-source availability, testing infrastructure, CI/CD

**Minimum viable results (JOSS requirements):**
- Installable software (`pip install discopt` or `maturin develop`)
- Documented API (at least docstrings + README with examples)
- Automated test suite with >85% coverage
- At least one domain example (e.g., materials optimization or experiment design)
- Statement of need distinguishing from BARON, Couenne, SCIP, cvxpylayers
- Community guidelines (CONTRIBUTING.md)
- JOSS requires: "substantial scholarly effort" and "not a minor utility"

**JOSS-specific considerations:**
- JOSS papers are short (typically 1-2 pages of text + references) — the software IS the contribution
- JOSS review is primarily about software quality (tests, docs, installation), not algorithmic novelty
- JOSS does NOT require the software to be "finished" — it requires it to be useful and well-engineered
- Phase 1 completion (25 MINLPLib instances, correctness validated, CI green) meets the JOSS bar
- The Rust+JAX hybrid and McCormick relaxations represent substantial scholarly effort
- Turnaround: ~2-4 months from submission to acceptance (fast by journal standards)

**Target venue:** Journal of Open Source Software (JOSS)

**Dependencies:** T15 (24 MINLPLib solved), T16 (Phase 1 gate passed), T10 (CI/CD), documentation (partial T31). Level 1 differentiability (T22, even partial) strongly recommended to demonstrate the key value proposition.

**Builds on:** P1 (workshop paper, if published, provides precedent); P2 (position paper, if posted, provides context)

**Strategic value:** The JOSS paper is the **citable reference** for the software. Every subsequent paper that uses discopt cites this. It also legitimizes the software in the academic community — reviewers and grant panels treat JOSS-published software as vetted. JOSS papers are indexed in Crossref and have DOIs, enabling citation tracking. This is the single most important publication for establishing discopt as a real project.

---

### P4: Systems Paper — Rust+JAX Hybrid Architecture

**Working title:** "Zero-Copy GPU Dispatch: A Rust+JAX Hybrid Architecture for GPU-Accelerated Mathematical Optimization"

**Timing:** Month 16-20 (requires Phase 2 GPU results, specifically custom IPM and batch evaluator)

**Core contribution:**
- Present the Rust+JAX hybrid architecture as a reusable design pattern for GPU-accelerated scientific computing
- Detailed analysis of zero-copy array transfer (PyO3 + numpy crate), JIT compilation overhead, batch dispatch protocol
- Performance breakdown: Rust tree management overhead (<5%), JAX GPU evaluation throughput, Python orchestration fraction
- Comparison with alternative architectures: pure Python (slow), pure Rust (no GPU), C++/CUDA (no composability), pure JAX (no tree management)
- The argument: "Rust for control flow + JAX for data parallelism" is a general pattern applicable beyond optimization

**Minimum viable results:**
- Layer profiling data from T14/T19: `rust_time_fraction`, `jax_time_fraction`, `python_time_fraction`
- GPU speedup measurements (T19): 15x+ at batch size 512
- Zero-copy verification data (T12)
- Comparison: discopt architecture vs naive Python orchestration, showing overhead breakdown
- At least 3 problem sizes showing scaling behavior (10, 50, 100 variables)

**Target venue:**
- **Primary:** SoftwareX (Elsevier, open access, focus on software impact)
- **Alternative:** SC (Supercomputing) workshop paper or poster
- **Alternative:** PPoPP (Principles and Practice of Parallel Programming) — if the parallelism angle is strong enough
- **Alternative:** Computing in Science & Engineering (IEEE, practical focus)

**Dependencies:** T0 (spike data), T12 (batch dispatch), T14 (solver orchestrator), T17 (custom IPM), T19 (batch evaluator), T25 (performance measurement). Essentially requires Phase 2 to be well underway.

**Builds on:** P1 (workshop paper introduced the idea), P3 (JOSS paper described the software)

**Strategic value:** This paper targets the systems/HPC community rather than the optimization community. It demonstrates that Rust+JAX is a viable architecture for performance-critical scientific computing, potentially attracting contributors from outside the optimization world. The "design pattern" framing makes it citeable by anyone building Rust+JAX scientific software.

---

### P5: Benchmark Paper — discopt vs BARON/Couenne/SCIP

**Working title:** "Benchmarking discopt: GPU-Accelerated MINLP Solving on MINLPLib Instances"

**Timing:** Month 20-24 (requires Phase 2 gate passed for credible GPU speedup numbers)

**Core contribution:**
- Rigorous, reproducible benchmark comparison of discopt against BARON, Couenne, SCIP, and Bonmin on MINLPLib instances
- Performance profiles (Dolan-More), shifted geometric means, root gap analysis
- Separate analysis for problem classes: convex MINLP, nonconvex, pooling, bilinear, portfolio
- The key narrative: discopt is NOT faster than BARON on general MINLP (within 2-3x), BUT is dramatically faster for batch solving and provides capabilities (differentiability, vmap) that no competitor offers
- Transparent about limitations: which problem classes discopt struggles on, where BARON dominance is clear

**Minimum viable results:**
- Phase 2 gate data: 55+ solved (30-var), 25+ solved (50-var), 15x GPU speedup
- Root gap within 2.0x BARON
- Performance profiles across at least 50 MINLPLib instances
- At least one problem class (pooling, portfolio) where discopt matches or beats BARON
- Batch solving comparison: 100-1000 related instances, discopt vmap vs BARON serial
- All benchmark infrastructure automated and reproducible (`run_benchmarks.py`)

**Target venue:**
- **Primary:** Mathematical Programming Computation (MPC) — the gold standard for solver benchmarking
- **Alternative:** INFORMS Journal on Computing (shorter turnaround)
- **Alternative:** Optimization Methods and Software

**Dependencies:** T26 (Phase 2 gate passed), T25 (benchmark runner fully functional), T19 (GPU batch evaluator), T21 (OBBT for competitive root gaps). Also requires access to BARON license for comparison runs.

**Builds on:** P3 (JOSS paper as the software reference), P4 (systems paper for architecture details)

**Strategic value:** The benchmark paper is how the optimization community evaluates new solvers. Without it, discopt lacks credibility in the OR/optimization world. The key is honest reporting: BARON is better on general MINLP, but discopt enables things BARON cannot (batch, grad, vmap). MPC is very selective but highly respected; INFORMS JoC is a strong fallback. This paper would be cited by anyone comparing MINLP solvers for the next decade.

---

### P6: Methods Paper — Differentiable MINLP

**Working title:** "Differentiable Mixed-Integer Nonlinear Programming via Multi-Level Gradient Strategies"

**Timing:** Month 20-24 (requires Level 1 + Level 3 differentiability working, Phase 2 complete)

**Core contribution:**
- Present the multi-level differentiability framework for MINLP: LP relaxation sensitivity (Level 1), implicit differentiation at active sets (Level 3), perturbation smoothing (fallback)
- Theoretical analysis: when each level provides exact vs approximate gradients, consistency guarantees
- Empirical demonstration: decision-focused learning on 3-5 scientific problems (materials, drug design, experiment design)
- Comparison with alternatives: finite differences (O(n) cost, noisy), STE/straight-through estimators (biased), relaxation-then-round (loses integer structure)
- Show that perturbation smoothing + vmap is uniquely enabled by discopt's batch solving

**Minimum viable results:**
- Level 1 differentiability working end-to-end (T22): `jax.grad` through solve on 5+ problems
- Level 3 on at least 3 problems (T23)
- Decision-focused learning demo: train a predictor end-to-end where the inner optimization is MINLP
- Show gradient quality: compare discopt gradients to finite-difference baseline on parametric problems
- At least one domain application (materials property optimization or Bayesian experiment design)

**Target venue:**
- **Primary:** ICML or NeurIPS (main conference, not workshop) — high visibility, establishes methodological contribution
- **Alternative:** ICLR (if framing emphasizes representation learning + optimization)
- **Alternative:** Mathematical Programming (if framing emphasizes optimization theory)
- **Fallback:** AAAI (lower acceptance rate but strong optimization track)

**Dependencies:** T22 (Level 1), T23 (Level 3), T14 (solver works), T19 (batch evaluator for perturbation smoothing). Also needs domain-specific problem formulations (materials, drug design).

**Builds on:** P2 (position paper framed the problem), P3 (JOSS paper for the software), P5 (benchmark paper for solver credibility)

**Strategic value:** This is potentially the **highest-impact publication** in the entire pipeline. Differentiable MINLP for scientific ML is a genuinely novel contribution that no competitor offers. If accepted at ICML/NeurIPS, it would generate significant attention and position discopt as the go-to tool for decision-focused learning with discrete variables. The position paper (P2) primes the community; this paper delivers the method.

---

### P7: Tutorial Paper — Decision-Focused Learning with discopt

**Working title:** "A Practitioner's Guide to Decision-Focused Learning with Mixed-Integer Optimization in JAX"

**Timing:** Month 22-28 (requires stable API, working differentiability, and polished examples)

**Core contribution:**
- Step-by-step tutorial showing how to embed MINLP optimization inside a JAX training loop
- Three worked examples: (1) Bayesian optimization with discrete experimental conditions, (2) decision-focused materials design, (3) portfolio optimization with cardinality constraints
- Best practices: when to use Level 1 vs Level 3, batch size selection, convergence monitoring
- Common pitfalls: gradient quality, branching sensitivity to parameters, scaling issues
- Reproducible Jupyter notebooks + Google Colab compatibility

**Minimum viable results:**
- Stable `discopt` API (Phase 2 complete)
- Working `jax.grad` and `jax.vmap` through solver
- 3 self-contained example notebooks that run end-to-end
- Performance sufficient that tutorials complete in reasonable time (< 5 minutes each)

**Target venue:**
- **Primary:** Computing in Science & Engineering (IEEE CiSE) — tutorial-focused, computational science audience
- **Alternative:** Living Journal of Computational Molecular Science (LJCMS) — if scoped to molecular/materials science
- **Alternative:** Journal of Chemical Education (if scoped to chemistry education)
- **Alternative:** Patterns (Cell Press) — data science tutorials

**Dependencies:** T22 (differentiability), T19 (batch solving), Phase 2 gate, partial T31 (documentation/notebooks). Requires stable, user-friendly API.

**Builds on:** P3 (JOSS paper for software citation), P6 (methods paper for theoretical foundation)

**Strategic value:** Tutorial papers have outsized impact per citation because they are the entry point for new users. A practitioner who reads P7 and successfully runs the notebooks becomes a discopt user and potential contributor. Tutorials also serve as evidence of broader impacts in grant reports and renewals. This paper targets the ML-adjacent scientific community (chemists, materials scientists, engineers) who want to use discopt but need guidance, not the optimization theory community.

---

### P8: Methods Paper — Learned Branching for Spatial B&B

**Working title:** "Graph Neural Network Branching Policies for GPU-Accelerated Spatial Branch-and-Bound"

**Timing:** Month 28-34 (requires GNN branching prototype from Phase 3)

**Core contribution:**
- GNN branching policy for spatial B&B (extending the well-established MILP GNN branching literature to MINLP)
- Key novelty: the bipartite variable-constraint graph representation adapted for nonlinear constraints and McCormick relaxations
- Training within JAX ecosystem: jraph (GNN), Equinox (network), Optax (training), discopt (solver) in one JIT-compiled loop
- Comparison with classical branching (most-fractional, reliability branching) and strong branching
- Show 20%+ node reduction while maintaining correct solutions

**Minimum viable results:**
- GNN branching policy trained on at least 100 MINLPLib instances (T29)
- Node reduction >= 20% vs classical branching on held-out test set
- Inference latency < 0.1ms (must not bottleneck the solver)
- Zero incorrect solutions (the correctness invariant must hold)
- Comparison on problem classes: convex vs nonconvex, different constraint types

**Target venue:**
- **Primary:** AAAI (strong optimization + ML track)
- **Alternative:** CPAIOR (Integration of Constraint Programming, AI, and Operations Research) — niche but perfectly targeted
- **Alternative:** NeurIPS (if results are strong enough to compete at the top venue)

**Dependencies:** T29 (GNN branching), T14 (solver working), T19 (GPU batch evaluation for training data generation). Also needs Phase 2 solver to generate training data via strong branching.

**Builds on:** P3 (software), P5 (benchmark baseline), P6 (differentiability enables training)

**Strategic value:** ML-for-MILP branching is a well-established literature (Gasse et al., NeurIPS 2019; Khalil et al., AAAI 2016). Extending this to MINLP/spatial B&B is natural but under-explored because no open-source spatial B&B solver exists with ML-friendly interfaces. discopt's JAX-native architecture makes it the ideal platform for this research. This paper demonstrates that discopt is not just a solver but a research platform.

---

## Publication Sequencing Strategy

### The Credibility Cascade

Publications are sequenced to build credibility incrementally:

```
Month 6-8:   P1 (Workshop) -- establishes the batch B&B idea, gets discopt name out
                |
Month 8-12:  P2 (Position/Survey) -- claims intellectual territory, supports grants
                |
Month 12-14: P3 (JOSS) -- legitimizes the software, becomes the canonical citation
                |         \
Month 16-20: P4 (Systems) -- Rust+JAX architecture for HPC/systems community
                |
Month 20-24: P5 (Benchmark) + P6 (Methods) -- the twin "proof" papers
                |              |
Month 22-28: P7 (Tutorial) -- onboards practitioners
                |
Month 28-34: P8 (Learned Branching) -- demonstrates research platform capability
```

### Key Principle: Software Paper Before Methods Papers

The JOSS paper (P3) should be published **before** the benchmark paper (P5) and the methods paper (P6). This is critical because:

1. P5 and P6 need to cite a published, peer-reviewed software artifact
2. Grant reviewers check whether the software has been independently reviewed (JOSS provides this)
3. JOSS is fast (~2-4 months), so submitting at Month 12 yields acceptance by Month 14-16
4. Without P3, P5 and P6 are "results from our unpublished software" — much weaker

### Workshop Papers as Draft Vehicles

P1 (workshop paper at Month 6-8) serves multiple purposes:
- Forces the team to articulate the key ideas early
- Gets feedback from the NeurIPS/ICML community before investing in full papers
- Establishes a publication date for priority claims
- Can be expanded into P4 (systems paper) or P5 (benchmark paper) with additional results
- Workshop papers are NOT considered prior publication at most venues (NeurIPS, ICML explicitly allow this)

### Grant Alignment

The publication timeline aligns with the grant strategy from the positioning document:

| Grant program | Deadline (typical) | Supporting publications at submission |
|--------------|-------------------|--------------------------------------|
| NSF CSSI Elements | October annually | P1 (workshop), P2 (position), P3 (JOSS if timing works) |
| DOE ASCR Applied Math | February annually | P1, P2, possibly P3 |
| CZI EOSS Cycle 7+ | June annually | P3 (JOSS), P1 |

Having P2 (position paper) and P3 (JOSS paper) ready by Month 14 means a grant submission at Month 14-18 can cite two peer-reviewed publications demonstrating both the vision and the working software.

---

## Risk Assessment for Publications

| Risk | Mitigation |
|------|-----------|
| Phase 1 delayed, P3 (JOSS) not ready by Month 14 | P2 (position paper) and P1 (workshop) can proceed independently of solver completion |
| BARON comparison unfavorable for P5 | Frame P5 around batch solving and capability comparison, not single-instance speed. "discopt is 3x slower per instance but 50x faster for 512 instances via vmap" is a compelling narrative |
| P6 rejected at ICML/NeurIPS | Resubmit at AAAI or ICLR. Workshop version (P1 expanded) serves as fallback venue |
| Phase 2 differentiability (T22/T23) proves harder than expected | P6 can be scoped to Level 1 only (LP relaxation sensitivity). Level 3 becomes future work |
| JOSS reviewers request extensive documentation | Budget 2-4 weeks of documentation effort before JOSS submission. This effort also supports P7 (tutorial) |
| Competing work appears (e.g., PyTorch MINLP solver) | Accelerate P2 (position paper) to establish priority on the conceptual framework. P1 (workshop) provides timestamp |

---

## Summary Table

| Paper | Phase | Month | Venue | Dependencies | Status After |
|-------|-------|-------|-------|-------------|-------------|
| P1: Workshop (Batch B&B) | 1 (partial) | 6-8 | NeurIPS OPT | T0, T4-T6, T11-T12, T14 | Idea established |
| P2: Position (Diff. Discrete Opt.) | 1 | 8-12 | arXiv + workshop | Literature review + T0 | Territory claimed |
| P3: JOSS (discopt software) | 1 (complete) | 12-14 | JOSS | T15, T16, T10 | Software legitimized |
| P4: Systems (Rust+JAX) | 2 (partial) | 16-20 | SoftwareX | T12, T17, T19, T25 | Architecture published |
| P5: Benchmark (vs BARON) | 2 (complete) | 20-24 | Math. Prog. Comp. | T26, T25, T21 | Solver credentialed |
| P6: Methods (Diff. MINLP) | 2 (complete) | 20-24 | ICML/NeurIPS | T22, T23, T19 | Methods contribution |
| P7: Tutorial (DFL guide) | 2+ | 22-28 | IEEE CiSE / LJCMS | T22, T19, T31 | Practitioners onboarded |
| P8: Learned Branching | 3 (partial) | 28-34 | AAAI / CPAIOR | T29, T19 | Platform demonstrated |

**Total by end of project (Month 42):** 8 publications (1 workshop, 1 position/survey, 1 software, 1 systems, 1 benchmark, 2 methods, 1 tutorial). This is a strong publication record for a 3.5-year software project and supports both tenure cases and grant renewals.
