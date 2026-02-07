# discopt: Unified Publication Roadmap

**Date:** 2026-02-07
**Synthesized from:** Three independent reviewer reports (methods, applications, software/infrastructure)

---

## Executive Summary

This roadmap consolidates findings from three scientific reviewers into a unified publication strategy spanning the 42-month discopt development plan. The strategy identifies **22 distinct publication opportunities** (plus 1 open-source release milestone) organized into **5 tiers** by priority and timing:

1. **Tier 1 — Foundation** (Months 6-14): Establish priority, claim intellectual territory, publish first algorithmic results
2. **Tier 2 — Flagship Methods** (Months 14-22): The core algorithmic contributions at top venues
3. **Tier 3 — Application Demonstrations** (Months 14-30): Domain papers proving the value propositions
4. **Tier 4 — Software & Advanced Methods** (Months 20-36): Software formalization, deeper algorithmic contributions
5. **Tier 5 — Capstone** (Months 28-42): Platform maturity, research platform demonstration

**Realistic target**: 12-16 published papers over 42 months (not all 22 — some are alternatives for the same slot). A strong outcome is **4 top-venue papers** (NeurIPS/ICML/Nature-family), **4-5 domain journal papers**, **2-3 optimization computation papers**, and **1-2 software/tutorial papers**.

---

## The Three Value Propositions and Their Paper Families

Every discopt paper demonstrates one or more of three unique capabilities:

| Value Proposition | Key JAX Feature | Becomes Available | Paper Family |
|------------------|----------------|------------------|-------------|
| **Batch solving** | `jax.vmap` | Phase 2 (~Month 14) | Batch B&B, portfolio, batch BO, autonomous experiments, molecular screening |
| **Differentiability** | `jax.grad` / `custom_jvp` | Phase 2 (~Month 18) | Differentiable MINLP, DFL/ChemE, inverse design, energy dispatch |
| **JAX composability** | `jit`, ecosystem | Phase 1 (~Month 10) | McCormick compiler, systems architecture, JOSS, tutorial |

---

## Tier 1 — Foundation (Months 6-14)

These papers establish discopt's existence and claim intellectual territory before the solver is competitive.

### 1. Workshop Paper: Batch Spatial B&B on GPU via JAX
- **Month**: 6-8 (submittable with partial Phase 1 results)
- **Venue**: NeurIPS OPT Workshop or ICML Workshop on Computational Optimization
- **Contribution**: Introduce the batch B&B architecture — Rust tree management + JAX vmap node evaluation
- **Minimum results**: 5+ MINLPLib instances solved, batch speedup at sizes 32-512, architectural spike data
- **Strategic value**: Establishes priority, 4-6 page extended abstract, fast turnaround (~2 months). NOT prior publication at NeurIPS/ICML for later full papers
- **Dependencies**: T0, T4-T6, T11-T12, T14 (partial)

### 2. Position Paper: Differentiable Discrete Optimization for Scientific ML
- **Month**: 8-12 (primarily conceptual, minimal code required)
- **Venue**: arXiv preprint → AAAI AI4Science Workshop or NeurIPS AI4Science
- **Contribution**: Survey the gap between differentiable optimization and integer programming; catalog scientific ML workflows needing differentiable MINLP; present discopt's multi-level framework
- **Minimum results**: Literature survey (100+ papers), capability gap matrix, conceptual architecture, optional small demo
- **Strategic value**: Claims intellectual territory for "differentiable discrete optimization for scientific ML." Becomes the cited reference. Directly supports grant applications (NSF CSSI, DOE ASCR)
- **Dependencies**: Literature review effort; benefits from T0 and T14 but not strictly required

### 3. McCormick-to-JAX Compiler Paper
- **Month**: 8-10 (end of Phase 1)
- **Venue**: INFORMS Journal on Computing (IJOC) or Mathematical Programming Computation (MPC)
- **Contribution**: Compiler that walks expression DAGs and emits pure `jax.numpy` McCormick relaxations — first system making McCormick automatically JIT-compilable, vmappable, and differentiable
- **Minimum results**: 14 operation types pass 10K-point soundness test; 24 MINLPLib instances produce valid relaxations; `jax.grad` matches finite differences; vmap scaling on GPU
- **Strategic value**: First peer-reviewed algorithmic contribution. Establishes the compilation approach that all later papers build on. MPC requires open-source, aligning with discopt's license
- **Dependencies**: T4, T5, T6, T15, T10

### 4. Open-Source Release (not a paper — enables JOSS clock)
- **Month**: ~10 (at or shortly after Phase 1 gate)
- **Action**: Make the repository public on GitHub. This starts the 6-month maintenance clock required by JOSS
- **Strategic value**: Enables community feedback, signals commitment, and begins the JOSS eligibility period. Methods papers [5, 6] can reference the open-source repo with a GitHub URL or Zenodo DOI before JOSS is published
- **Note**: JOSS submission moves to Month 20-24 (see Paper 17 in Tier 4)

---

## Tier 2 — Flagship Methods (Months 14-22)

These are the highest-impact contributions targeting top ML and optimization venues.

### 5. Batch Spatial B&B Architecture (Full Paper)
- **Month**: 14-16 (early Phase 2)
- **Venue**: NeurIPS or ICML (main conference)
- **Contribution**: First MINLP solver to evaluate hundreds of B&B nodes in parallel on GPU. Restructures the inherently sequential B&B into "batch-sequential" pattern. Demonstrates Rust (CPU tree) + JAX (GPU relaxation) split
- **Minimum results**: 24 MINLPLib correct, >=15x GPU speedup at batch 512, >=200 nodes/sec, interop overhead <=5%, comparison to Couenne (<=3x), batch size ablation
- **Strategic value**: Flagship architecture paper. Frames as enabling "optimization-in-the-loop ML," not just optimization speed
- **Dependencies**: T14, T15, T17, T19, T24, T25
- **Builds on**: Papers 1, 3

### 6. Differentiable MINLP via Multi-Level Gradients
- **Month**: 16-20 (mid-late Phase 2)
- **Venue**: NeurIPS or ICML (main conference)
- **Contribution**: Multi-level differentiability framework — Level 1 (LP relaxation sensitivity via `custom_jvp`), Level 3 (implicit differentiation at active set), perturbation smoothing fallback. First system providing gradients through MINLP solving
- **Minimum results**: Level 1 on 5+ parametric LP instances, Level 3 on 5 MINLP problems, perturbation smoothing convergence, decision-focused learning demo, composability with vmap
- **Strategic value**: **Potentially the highest-impact publication.** Extends cvxpylayers/PyEPO to MINLP. NeurIPS/ICML have published all key differentiable optimization papers
- **Dependencies**: T14, T22, T23, T19
- **Builds on**: Papers 3, 5

### 7. Ipopt-to-Rust Translation: A Memory-Safe Interior Point Method
- **Month**: 12-20 (Phase 1-2, can begin early as infrastructure work)
- **Venue**: Mathematical Programming Computation (MPC), Optimization Methods and Software (OMS), or SoftwareX
- **Contribution**: A faithful translation of Ipopt's core interior point algorithm from C++ to Rust, producing a standalone, memory-safe NLP solver. Rather than designing a novel IPM from scratch — with all the risk of subtle numerical bugs and years of missed edge cases — this approach preserves 20+ years of hardened algorithmic engineering (filter line search, inertia correction, warm-starting, restoration phase) while gaining Rust's safety guarantees and modern tooling
- **Why translation over from-scratch**:
  1. **De-risking**: Ipopt handles hundreds of numerical edge cases accumulated over two decades (degenerate Jacobians, rank-deficient KKT systems, cycling detection, tiny step recovery). A from-scratch IPM would rediscover these failures one benchmark at a time. Translation preserves this institutional knowledge
  2. **Correctness by construction**: Each translated function can be validated against Ipopt's own output on the same problem instance, providing a natural regression test suite. Bit-for-bit matching on well-conditioned problems, tolerance matching on ill-conditioned ones
  3. **Path to vmappability**: Once the core algorithm is in Rust with explicit state, refactoring for batch execution (struct-of-arrays layout, removing global state, parameterizing over problem data) is substantially easier than wrapping C++ Ipopt through FFI. The Rust version becomes the foundation for a vmappable JAX IPM — the translation is Phase 1 scaffolding that evolves into Phase 2 infrastructure rather than being discarded
  4. **Community value**: There is no production-quality Rust NLP solver. A Rust Ipopt would be immediately useful to the growing Rust scientific computing ecosystem (Argmin, Russell, faer), independent of discopt
  5. **Maintenance**: Rust's ownership model eliminates use-after-free and data races that have caused real Ipopt bugs. `cargo clippy` and `miri` catch entire categories of issues at compile time
  6. **LLM-assisted translation**: Modern LLMs can accelerate C++-to-Rust translation significantly, making this more feasible now than even two years ago. The structured, algorithmic nature of Ipopt's code (numerical linear algebra, iterative loops, well-defined interfaces) is particularly amenable to automated translation
- **EPL-2.0 licensing issue**: Ipopt is licensed under the Eclipse Public License 2.0. A line-by-line translation to Rust is almost certainly a derivative work under copyright law, meaning the Rust translation must also be distributed under EPL-2.0 (or a compatible secondary license if the original contributors designate one). **This creates a license boundary within discopt**: the Rust Ipopt module would be EPL-2.0, while the rest of discopt could remain MIT/Apache-2.0. Options:
  - **(a) Dual-module licensing** (recommended): Keep the Rust Ipopt as a separate crate (`ipopt-rs`) under EPL-2.0. discopt depends on it but is not itself EPL-covered. EPL-2.0 explicitly permits this — it is not a "viral" license like GPL. The Ipopt crate is a separately distributable component with its own license
  - **(b) Clean-room reimplementation**: Study Ipopt's algorithms from published papers (Wächter & Biegler 2006, SIAM J. Optimization), then reimplement from the mathematical description without reference to source code. Produces a fully MIT-licensable result but loses the edge-case hardening and is higher risk
  - **(c) Upstream relicensing**: Contact COIN-OR / Ipopt maintainers about adding MIT/Apache as a secondary license under EPL-2.0 Section 3. Worth attempting but unlikely to succeed quickly for a codebase with many contributors
  - **(d) Functional API boundary**: If the translation produces a solver with a sufficiently different internal architecture (e.g., different data structures, different memory layout, different linear algebra backend), it may qualify as an independent implementation inspired by the same algorithms. Legal gray area — requires counsel
- **Minimum results**: Matches Ipopt's C++ output on all 7 NLP benchmark problems (rel_tol=1e-6); matches on all 7 Netlib LP instances; passes Ipopt's own test suite (translated); handles at least 3 failure-mode tests (degenerate, unbounded, infeasible) correctly; wall-clock within 2x of C++ Ipopt on single instances (Rust overhead acceptable given safety benefits); clean `cargo clippy`, zero `unsafe` in core algorithm
- **Stretch results**: vmappable Rust IPM that can batch-solve via JAX integration (>=10x at batch 64), differentiable via `custom_jvp`
- **Strategic value**: **This is the lowest-risk path to a GPU-native subsolver.** Translation gives a correct baseline immediately; optimization for batching/GPU comes afterward from a known-good starting point. The alternative — building a custom dense GPU IPM from scratch — is higher novelty but also higher risk of subtle numerical failures that only manifest on hard MINLPLib instances at Phase 2/3 gates
- **Dependencies**: Ipopt source code (publicly available), Rust toolchain, faer or nalgebra for linear algebra
- **Builds on**: Papers 3, 5

---

## Tier 3 — Application Demonstrations (Months 14-30)

Domain papers proving discopt's value propositions in scientific contexts. Each needs a domain expert co-author.

### 8. Solver Announcement Paper (with domain demos)
- **Month**: 10-14 (immediately after Phase 1 gate)
- **Venue**: INFORMS Journal on Computing or arXiv preprint
- **Contribution**: Introduce discopt's architecture, validate correctness on 25+ MINLPLib instances, show small-scale application examples (process synthesis, pooling, portfolio from existing codebase). Establishes that the solver exists and works
- **Note**: This is a fuller paper than the workshop [1]. Can serve as the citable software reference until JOSS is published at Month 20-24
- **Dependencies**: Phase 1 gate (T16)

### 9. Cardinality-Constrained Portfolio at Scale
- **Month**: 14-18 (early Phase 2) — **quickest application win**
- **Venue**: Operations Research (INFORMS) or Quantitative Finance
- **Contribution**: Solve portfolio MIQCQP across 10,000 market scenarios via `jax.vmap`. Monte Carlo risk assessment infeasible with serial solvers. Level 1 sensitivity for model risk
- **Minimum results**: N=20-50 assets, K=5-15 cardinality, 10K scenarios batched, 20-100x vs Gurobi serial, sensitivity analysis, S&P 500 backtest
- **Strategic value**: Problem already exists in codebase (`example_portfolio()`). MIQCQP well within capabilities. Large finance audience. Fastest path to a published application
- **Dependencies**: WS6 (T19), WS5 (T14-T15), WS-D Level 1 (T22)
- **Builds on**: Paper 4 (JOSS)

### 10. Batch BO for Materials Discovery
- **Month**: 18-22 (late Phase 2)
- **Venue**: npj Computational Materials or Digital Discovery (RSC)
- **Contribution**: `jax.vmap` over discopt for batch acquisition optimization in mixed-integer BO. Discrete atom types + continuous stoichiometries. 10-50x speedup at batch 128+
- **Minimum results**: Synthetic benchmark (Thompson sampling, 5-10 discrete classes), surrogate case study on alloy dataset, batch BO vs serial comparison
- **Strategic value**: Flagship demonstration of vmap value proposition. Materials science community investing heavily in ML
- **Dependencies**: WS6 (T19), WS5 (T14-T15); differentiability enhances but not required
- **Co-author needed**: Materials scientist with alloy datasets + DFT access

### 11. Autonomous Experiment Design
- **Month**: 16-22 (late Phase 2)
- **Venue**: Nature Machine Intelligence or Chemical Science (RSC)
- **Contribution**: Real-time batch optimization for autonomous experiment loops. Batch q-EI acquisition over mixed-integer experimental conditions in <10 seconds
- **Minimum results**: Simulated loop on published reaction dataset (Shields 2021), batch sizes 4-32, wall-clock < 10s, fewer total experiments than serial BO
- **Strategic value**: Hot topic (self-driving labs). "Real-time" angle is distinctive
- **Dependencies**: WS6 (T19), WS5 (T14-T15)
- **Co-author needed**: Experimental chemist with autonomous lab access

### 12. Decision-Focused Learning for Chemical Process Optimization
- **Month**: 20-26 (post Phase 2 gate)
- **Venue**: Computers & Chemical Engineering
- **Contribution**: End-to-end training of process surrogates by backpropagating through discopt. Decision-focused learning reduces "decision regret" 15-40% vs two-stage
- **Minimum results**: 3-5 unit process network, Equinox surrogate, three training paradigms compared (MSE vs Level 1 vs Level 3), sensitivity analysis, 2-3 benchmark problems from literature
- **Strategic value**: Flagship demonstration of differentiability value proposition. ChemE community understands MINLP deeply; DFL angle is novel for this audience
- **Dependencies**: WS-D Level 1 (T22, required), Level 3 (T23, preferred), WS5, WS4
- **Co-author needed**: Chemical engineer with process simulation expertise

### 13. High-Throughput Molecular Screening
- **Month**: 22-28 (Phase 2/3)
- **Venue**: Journal of Chemical Information and Modeling (ACS JCIM)
- **Contribution**: Screen 1K-10K molecular candidates simultaneously via `jax.vmap(discopt.solve)` with QSPR constraints. 50-200x speedup for 1000+ candidates
- **Dependencies**: WS6, WS5, Phase 2 gate
- **Co-author needed**: Computational chemist with QSPR expertise

### 14. Stochastic Energy System Dispatch
- **Month**: 24-30 (Phase 3)
- **Venue**: Applied Energy or IEEE Transactions on Power Systems
- **Contribution**: Decision-focused unit commitment — train load forecaster to minimize dispatch cost, not RMSE. Batch stochastic programming (100-500 scenarios via vmap)
- **Dependencies**: WS-D (T22/T23), WS6 (T19), WS10-a (T27)
- **Co-author needed**: Power systems engineer

### 15. Differentiable Inverse Materials Design
- **Month**: 24-32 (Phase 3) — **highest-risk, highest-reward application**
- **Venue**: Nature Communications or ACS Central Science
- **Contribution**: discopt as differentiable optimization layer inside a neural network for materials inverse design. All three value propositions simultaneously
- **Minimum results**: Materials Project dataset, MINLP layer enforces physical constraints, 20-50% higher designability vs post-hoc enforcement, Level 1 vs Level 3 ablation, 3-5 designs validated against DFT
- **Dependencies**: WS-D Level 3 (T23), WS4 (T17), WS6 (T19), WS10-a (T27)
- **Co-author needed**: Materials scientist with DFT + ML expertise

---

## Tier 4 — Software Formalization & Advanced Methods (Months 16-36)

### 16. Systems Paper: Rust+JAX Hybrid Architecture
- **Month**: 16-20 (Phase 2 partial)
- **Venue**: SoftwareX or SC/PPoPP workshop
- **Contribution**: Rust+JAX as a reusable design pattern for GPU-accelerated scientific computing. Zero-copy transfer, JIT overhead, batch dispatch protocol
- **Dependencies**: T0, T12, T14, T17, T19, T25
- **Builds on**: Papers 1, 3

### 17. JOSS Software Paper: discopt
- **Month**: 20-24 (requires 6 months of open-source maintenance history; methods papers [5, 6] published or submitted first)
- **Venue**: Journal of Open Source Software (JOSS)
- **Contribution**: Formalize discopt as peer-reviewed open-source software. By this point, methods papers have already demonstrated the algorithmic contributions — JOSS certifies the software artifact itself
- **Minimum results**: Installable software, documented API, >85% test coverage, 25+ MINLPLib correct, community guidelines, CONTRIBUTING.md, 6+ months of open-source commit history with issue tracking
- **Strategic value**: Becomes the canonical software citation going forward. JOSS review validates software quality. Grant panels treat JOSS-published software as vetted. By publishing after methods papers, JOSS can reference published results rather than making unsupported claims
- **JOSS eligibility**: Repository must be public for 6 months. If open-sourced at Month 10, eligible from Month 16. Targeting Month 20-24 allows methods papers to establish the contribution first
- **Dependencies**: T15, T16 (Phase 1 gate), T10, 6 months open-source history, at least one methods paper [5 or 6] submitted

### 18. Benchmark Paper: discopt vs BARON/Couenne/SCIP
- **Month**: 20-24 (Phase 2 complete)
- **Venue**: Mathematical Programming Computation (MPC)
- **Contribution**: Rigorous Dolan-More profiles, shifted geometric means, root gap analysis on 50+ MINLPLib instances. Honest framing: "3x slower per instance, 50x faster for 512 instances via vmap"
- **Strategic value**: How the optimization community evaluates new solvers. Would be cited by anyone comparing MINLP solvers for a decade
- **Dependencies**: T26 (Phase 2 gate), T25, T19, T21. BARON license needed
- **Builds on**: Papers 5, 6; cites JOSS [17] if published, otherwise GitHub/Zenodo

### 19. Tutorial: Decision-Focused Learning with discopt
- **Month**: 22-28 (stable API required)
- **Venue**: IEEE Computing in Science & Engineering or Living Journal of Computational Molecular Science
- **Contribution**: Step-by-step tutorial with 3 worked examples (BO, materials design, portfolio). Reproducible Jupyter notebooks + Colab
- **Strategic value**: Outsized impact per citation — entry point for new users. Evidence of broader impacts for grant reports
- **Dependencies**: T22, T19, Phase 2 gate, T31 (documentation)
- **Builds on**: Papers 6, 17 (JOSS)

### 20. Advanced Relaxations: Piecewise McCormick and alphaBB in JAX
- **Month**: 30-36 (late Phase 3 / early Phase 4)
- **Venue**: Journal of Global Optimization (JOGO) or Mathematical Programming Series A
- **Contribution**: Adaptive piecewise McCormick (k=4-16 partitions, vmappable) and alphaBB relaxations in JAX (exact Hessian via `jax.hessian`). Not novel techniques — novel implementation enabling batch GPU evaluation
- **Minimum results**: Soundness validation, >=60% root gap reduction vs standard McCormick, root gap <=1.3x BARON, GPU throughput within 2x of standard McCormick
- **Dependencies**: T6, T19, T27, T21
- **Builds on**: Papers 3, 5, 7

---

## Tier 5 — Capstone (Months 28-42)

### 21. GNN Branching Policies for Spatial B&B
- **Month**: 28-34 (Phase 3)
- **Venue**: NeurIPS or ICML (main) / AAAI / CPAIOR
- **Contribution**: Extend GNN branching from MILP to MINLP. Novel: bipartite graph includes relaxation quality features, branching on continuous variables, end-to-end differentiable training in JAX
- **Minimum results**: >=20% node reduction, <0.1ms inference, comparison to classical heuristics, generalization to larger problems, IL vs IL+RL ablation
- **Strategic value**: Demonstrates discopt as a research platform, not just a solver
- **Dependencies**: T14, T19, T29, T27
- **Builds on**: Papers 3, 5, 6, 7

### 22. Comprehensive MINLPLib Benchmark
- **Month**: 32-36 (Phase 3/4)
- **Venue**: Optimization Methods and Software or INFORMS JoC
- **Contribution**: Full MINLPLib benchmarking with all advanced features enabled. Definitive performance characterization
- **Dependencies**: Phase 3 gate, all advanced relaxations, GNN branching

### 23. Survey/Review: The State of Differentiable Discrete Optimization
- **Month**: 36-42 (Phase 4)
- **Venue**: Annual Review of Control, Robotics, and Autonomous Systems or ACM Computing Surveys
- **Contribution**: Expanded, retrospective version of Paper 2 with 3+ years of results. Positions discopt within the broader landscape
- **Strategic value**: Capstone intellectual contribution. Invited review style

---

## Unified Timeline

```
Phase 1 (Months 1-10)
  Month 6-8:   [1] Workshop paper (batch B&B idea)
  Month 8-10:  [3] McCormick compiler paper (submit to IJOC/MPC)
  Month 8-12:  [2] Position paper (arXiv + workshop)
  Month ~10:   [4] Open-source release (starts JOSS 6-month clock)

Phase 2 (Months 10-20)
  Month 10-14: [8] Solver announcement (IJOC or arXiv)
  Month 12-20: [7] Ipopt-to-Rust translation (MPC/OMS/SoftwareX)
  Month 14-16: [5] Batch B&B full paper (NeurIPS/ICML) ← HIGHEST PRIORITY
  Month 14-18: [9] Portfolio application (Operations Research)
  Month 16-20: [6] Differentiable MINLP (NeurIPS/ICML) ← HIGHEST IMPACT
  Month 16-20: [16] Systems paper (SoftwareX)
  Month 16-22: [11] Autonomous experiments (Nature MI)
  Month 18-22: [10] Batch BO materials (npj Comp. Mat.)

Phase 3 (Months 20-32)
  Month 20-24: [17] JOSS software paper ← after methods, after 6mo maintenance
  Month 20-24: [18] Benchmark paper (MPC) ← SOLVER CREDIBILITY
  Month 20-26: [12] DFL for ChemE (Comp. & Chem. Eng.)
  Month 22-28: [13] Molecular screening (JCIM)
  Month 22-28: [19] Tutorial paper (IEEE CiSE)
  Month 24-30: [14] Energy dispatch (Applied Energy)
  Month 24-32: [15] Inverse materials design (Nature Comms) ← MARQUEE APPLICATION
  Month 28-34: [21] GNN branching (NeurIPS/ICML/AAAI)

Phase 4 (Months 32-42)
  Month 30-36: [20] Advanced relaxations (JOGO)
  Month 32-36: [22] Full MINLPLib benchmark (OMS)
  Month 36-42: [23] Survey/review (ACM Computing Surveys)
```

---

## Credibility Cascade

Publications are sequenced so that methods papers establish the intellectual contribution first, and the JOSS software paper formalizes the artifact afterward:

```
[1] Workshop ─── establishes the idea, gets name out
 │
[2] Position ─── claims intellectual territory, supports grants
 │
[3] McCormick ── first algorithmic contribution (optimization journal)
 │
[4] Open-source release ── starts JOSS 6-month maintenance clock
 │  \
 │   [8] Solver announcement ── citable reference until JOSS
 │
[5] Batch B&B ── flagship architecture (top ML venue)
 │  \
 │   [9] Portfolio ── quick application win (vmap demo)
 │
[6] Diff MINLP ─ flagship methods (top ML venue)
 │  \
 │   [12] DFL ChemE ── differentiability in domain context
 │
[17] JOSS ─────── formalizes software after methods prove the contribution
 │
[18] Benchmark ── solver credentialed by optimization community
 │
[19] Tutorial ── onboards practitioners
 │  \
 │   [15] Inverse design ── all value propositions combined
 │
[21] GNN branch ─ demonstrates research platform capability
```

**Sequencing rationale**: Methods papers [5, 6] come before JOSS because (a) JOSS now requires 6 months of open-source maintenance history, and (b) the algorithmic contributions are the real intellectual advances — JOSS formalizes the software artifact after those contributions are established. Methods papers can reference the open-source repository via GitHub URL or Zenodo DOI. The solver announcement paper [8] or arXiv preprint serves as a citable reference in the interim.

---

## Recommended Priority Ranking

### Must-publish (core credibility — methods first)
1. **Differentiable MINLP** [6] — highest novelty, top venue potential
2. **Batch B&B architecture** [5] — flagship systems contribution
3. **Benchmark paper** [18] — solver credibility in optimization community
4. **McCormick compiler** [3] — first algorithmic contribution, foundation for all later work

### High priority (applications + software formalization)
5. **Portfolio application** [9] — quickest application win
6. **Batch BO for materials** [10] — flagship vmap application
7. **DFL for ChemE** [12] — flagship differentiability application
8. **JOSS paper** [17] — formalizes software after methods establish the contribution

### Medium priority (territory + breadth)
9. **Position paper** [2] — claims territory, supports grants
10. **Workshop paper** [1] — establishes priority early
11. **Tutorial paper** [19] — user onboarding, broader impacts
12. **GNN branching** [21] — research platform demonstration
13. **Autonomous experiments** [11] — hot topic application

### Lower priority (niche or stretch)
14. **Ipopt-to-Rust** [7] — de-risks subsolver, community value, narrow audience but strategic foundation
15. **Systems paper** [16] — HPC community outreach
16. **Advanced relaxations** [20] — global optimization niche
17. **Energy dispatch** [14] — requires domain collaborator
18. **Molecular screening** [13] — requires domain collaborator
19. **Inverse materials design** [15] — highest risk, highest reward
20. **Full MINLPLib benchmark** [22] — comprehensive but duplicates [18]
21. **Survey/review** [23] — capstone if invited
22. **Solver announcement** [8] — interim reference until JOSS

---

## Collaboration Requirements

Each application paper benefits from domain expert co-authors. Recruiting should begin in Months 10-14 (while Phase 2 work proceeds):

| Paper | Domain Expert Needed | When to Recruit |
|-------|---------------------|----------------|
| [9] Portfolio | Quantitative finance researcher | Month 10 |
| [10] Batch BO | Materials scientist (alloy/composition) | Month 10 |
| [11] Autonomous experiments | Experimental chemist (autonomous lab) | Month 12 |
| [12] DFL ChemE | Chemical engineer (process synthesis) | Month 12 |
| [13] Molecular screening | Computational chemist (QSPR) | Month 16 |
| [14] Energy dispatch | Power systems engineer | Month 18 |
| [15] Inverse design | Materials scientist (DFT + ML) | Month 18 |

---

## Grant Alignment

| Grant Program | Typical Deadline | Supporting Publications at Submission |
|--------------|-----------------|--------------------------------------|
| NSF CSSI Elements | October annually | [1] workshop, [2] position, [3] McCormick, [8] solver announcement |
| DOE ASCR Applied Math | February annually | [1], [2], [3], open-source repo |
| CZI EOSS Cycle 7+ | June annually | [1], [2], [3], open-source repo with 6+ months history |
| NSF CSSI Framework (renewal) | October (Year 3+) | [5]-[6], [9]-[12], [17] JOSS, [18] benchmark — full portfolio |

For early grant submissions (Month 14-18), the position paper [2], McCormick compiler [3], and open-source repo with active development history provide strong evidence of vision and capability. Methods papers [5, 6] may be submitted/under review by then, demonstrable as preprints.

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Phase 1 delayed → methods papers delayed | HIGH — pushes entire cascade | [1] workshop and [2] position proceed independently; [3] McCormick compiler has fewer dependencies |
| BARON comparison unfavorable | MEDIUM — weakens [18] | Frame around batch/diff capabilities, not single-instance speed |
| Differentiability (Level 3) proves harder than expected | MEDIUM — weakens [6], [12], [15] | Scope [6] to Level 1 only; [15] becomes future work |
| Competing work (PyTorch MINLP solver) appears | MEDIUM — reduces novelty | Accelerate [2] position paper for priority; [1] provides timestamp |
| Cannot recruit domain collaborators | MEDIUM — blocks application papers | Start with [9] portfolio (no collaborator strictly needed) and [12] ChemE (closest to existing codebase examples) |
| GPU speedup <15x | LOW — weakens [5] | Architecture contribution stands even at 5-10x; honest reporting |
| JOSS maintenance requirement not met | LOW — delays [17] | Open-source early (Month 10); active issue tracking and community engagement during Phases 1-2 builds the required history naturally |

---

## Summary: Realistic 42-Month Publication Output

**Conservative estimate** (12 papers):
- 2 workshop papers [1, +1 NeurIPS ML4CO at Month 26]
- 1 position paper [2]
- 2 top-venue methods papers [5, 6]
- 2 optimization computation papers [3, 18]
- 3 domain application papers [9, 10, 12]
- 1 JOSS paper [17]
- 1 tutorial [19]

**Optimistic estimate** (16 papers):
- Add [7] GPU IPM, [11] autonomous experiments, [15] inverse design, [21] GNN branching

This represents a strong publication record for a 3.5-year computational infrastructure project and supports both tenure cases and grant renewals.

---

## Source Reports

The following detailed reports were produced by independent reviewers and contain expanded analysis for each paper:

- `reports/publication_plan_algorithmic.md` — 6 methods/algorithmic papers with detailed MVRs and venue analysis
- `reports/application_publication_milestones.md` — 8 application papers with domain-specific requirements and collaboration needs
- `reports/publication_milestones.md` — 8 software/infrastructure papers with credibility cascade analysis and grant alignment
