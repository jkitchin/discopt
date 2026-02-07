# discopt: Application and Domain-Science Publication Milestones

**Reviewer**: applications-reviewer
**Date**: 2026-02-07
**Based on**: `reconciled_development_plan_v2.md`, `discopt_positioning.md`

---

## Executive Summary

This report identifies **8 application papers** spanning materials science, chemical engineering, autonomous experimentation, energy systems, drug design, and finance. Papers are sequenced to match solver maturity: Phase 1 enables proof-of-concept demonstrations on small academic problems; Phase 2 unlocks the unique value propositions (batch vmap, differentiability) that justify compelling domain publications; Phase 3+ enables competitive-scale results suitable for high-impact domain journals.

A critical strategic observation: **the most impactful application papers require Phase 2 capabilities** (vmap batch solving and differentiability). Phase 1 produces a working solver but without the differentiators that make discopt novel compared to existing tools. Application papers from Phase 1 must therefore be framed carefully -- as methodology demonstrations on canonical problems, not as claims of superiority.

---

## Paper A1: Batch Bayesian Optimization with Discrete Choices for Materials Discovery

- **Title (working)**: "Batch Bayesian Optimization over Mixed-Integer Spaces via GPU-Vectorized MINLP Solving: Application to Alloy Composition Design"
- **Timing**: Month 18-20 (late Phase 2), publishable ~Month 22
- **Core contribution**: Demonstrates that `jax.vmap` over discopt's solver enables batch acquisition function optimization where each candidate evaluation involves solving an MINLP (discrete atom types + continuous stoichiometries). Compares wall-clock time and sample efficiency against sequential BO with Gurobi/BARON for the acquisition solve.
- **Domain**: Materials science (alloy design / composition optimization)
- **Minimum viable results**:
  1. Synthetic benchmark: Thompson sampling acquisition over 5-10 discrete material classes x 3-5 continuous composition variables. Batch sizes 32, 64, 128, 256. Show wall-clock speedup vs serial BARON calls.
  2. Surrogate-based case study: Train a Gaussian process (or small neural network) surrogate on a published alloy dataset (e.g., Citrine Informatics open datasets or AFLOW). Use discopt to optimize batch acquisition with discrete element choices (e.g., select 3 elements from a set of 10) and continuous composition fractions subject to thermodynamic stability constraints.
  3. Compare total optimization-loop time: discopt (batched GPU) vs. BARON (serial CPU) vs. Gurobi (serial CPU, MIQP approximation). Target: 10-50x speedup at batch size 128+.
  4. Demonstrate that batch BO with discopt achieves the same or better sample efficiency as sequential BO (same number of rounds to reach target property), but each round completes in seconds instead of minutes.
- **Target venue**: **npj Computational Materials** or **Digital Discovery** (RSC)
  - Rationale: npj Comp. Mat. publishes computational methods for materials discovery and has high impact (IF ~12). Digital Discovery is newer, explicitly targets ML-for-materials methods, and has a faster review cycle. Both value methodological novelty in how optimization is done, not just the materials result.
  - Alternative: **JMLR** (ML track) if framed as a general batch BO methodology paper with materials as the application.
- **Dependencies**: WS6 (GPU batch evaluator with vmap, T19), WS5 (working solver, T14-T15), WS-D (differentiability not strictly needed but enhances the story if Level 1 is available, T22)
- **Builds on**: Cites the core discopt solver paper (if published) and the positioning as "the missing infrastructure for optimization-in-the-loop scientific ML." This paper is the flagship demonstration of the vmap value proposition.

---

## Paper A2: Decision-Focused Learning for Chemical Process Optimization

- **Title (working)**: "End-to-End Learning of Process Surrogates via Differentiable Mixed-Integer Optimization"
- **Timing**: Month 20-24 (Phase 2 gate passed), publishable ~Month 26
- **Core contribution**: Trains a neural network surrogate for chemical process behavior end-to-end by backpropagating through discopt's solver. The surrogate predicts yield/cost as a function of operating conditions; the optimizer selects discrete unit configurations (reactor types, catalyst choices) and continuous operating parameters (temperature, pressure, flow rate). Decision-focused learning produces a surrogate that is optimized for decision quality, not just prediction accuracy.
- **Domain**: Chemical engineering (process synthesis and design)
- **Minimum viable results**:
  1. Small process network: 3-5 processing units with binary build/no-build decisions and continuous flows (similar to `example_process_synthesis()` in the existing codebase). Train an Equinox neural network surrogate on simulated process data.
  2. Compare three training paradigms: (a) two-stage (train surrogate on MSE, then optimize), (b) decision-focused with discopt Level 1 gradients (`custom_jvp`), (c) decision-focused with Level 3 implicit differentiation.
  3. Show that decision-focused training reduces "decision regret" (gap between actual cost and optimal cost under true model) by 15-40% compared to two-stage, even when prediction accuracy (MSE) is worse.
  4. Sensitivity analysis: use `result.gradient(param)` to compute how optimal process design changes with feedstock prices, demonstrating parametric sensitivity as a practical engineering tool.
  5. Reproduce on 2-3 process synthesis benchmark problems from the literature (e.g., Duran & Grossmann 1986 problems, or problems from Grossmann & Trespalacios 2013 review).
- **Target venue**: **Computers & Chemical Engineering** (Elsevier)
  - Rationale: This is the premier journal for computational methods in chemical engineering. It has published extensively on MINLP for process synthesis (Grossmann's group) and is now publishing ML-for-process-engineering work. A paper showing differentiable MINLP for process design would be highly novel here.
  - Alternative: **AIChE Journal** (broader chemical engineering audience, higher impact) if the process case study is industrially compelling.
- **Dependencies**: WS-D Level 1 (T22, required), WS-D Level 3 (T23, strongly preferred), WS5 (working solver, T14-T15), WS4 (custom IPM for vmappable perturbation smoothing, T17)
- **Builds on**: Cites discopt solver paper, Paper A1 (batch solving capability). This paper is the flagship demonstration of the differentiability value proposition.

---

## Paper A3: High-Throughput Screening via Batch MINLP: Molecular Property Optimization

- **Title (working)**: "GPU-Accelerated High-Throughput Molecular Design with Mixed-Integer Nonlinear Constraints"
- **Timing**: Month 22-26 (Phase 2/early Phase 3), publishable ~Month 28
- **Core contribution**: Uses discopt's batch solving to screen thousands of molecular candidates simultaneously, where each candidate is defined by discrete structural choices (functional group selection, ring system choices) and continuous property variables, subject to nonlinear constraints (QSPR models for solubility, toxicity, synthesizability). This transforms molecular optimization from a sequential enumeration problem into a batched GPU computation.
- **Domain**: Computational chemistry / drug design
- **Minimum viable results**:
  1. Define a molecular design space with 10-20 discrete building blocks and 5-10 continuous parameters (substituent positions, chain lengths treated continuously for optimization purposes). Formulate as MINLP with QSPR constraints.
  2. Screen 1,000-10,000 candidate formulations in parallel using `jax.vmap(discopt.solve)` over parameterized molecular templates.
  3. Compare wall-clock time: discopt (batched GPU) vs. BARON (serial) vs. enumeration-then-filter. Target: 50-200x speedup for 1000+ candidates.
  4. Validate on a published molecular property dataset (e.g., ESOL solubility, Tox21, or FreeSolv). Show that MINLP-based screening identifies the same top-K candidates as exhaustive evaluation but in a fraction of the time.
  5. Demonstrate integration with a JAX-based molecular featurization pipeline (e.g., using jraph for molecular graphs) -- the screening MINLP solver and the property predictor share the same JAX computation graph.
- **Target venue**: **Journal of Chemical Information and Modeling** (ACS JCIM)
  - Rationale: JCIM is the standard venue for computational methods in molecular design. It has published extensively on optimization-based molecular design and is open to GPU-acceleration methodology papers.
  - Alternative: **Journal of Cheminformatics** (open access, BMC) for faster publication, or **Nature Computational Science** if results are dramatic enough.
- **Dependencies**: WS6 (batch evaluator, T19), WS5 (solver correctness, T14-T15), Phase 2 gate for GPU speedup validation (T26)
- **Builds on**: Cites discopt solver paper, Paper A1 (batch solving methodology). Demonstrates inter-solve parallelism as opposed to A1's intra-solve parallelism.

---

## Paper A4: Differentiable Inverse Design for Materials with Discrete-Continuous Optimization

- **Title (working)**: "Differentiable Inverse Materials Design: Learning Processing-Structure Maps through Mixed-Integer Optimization Layers"
- **Timing**: Month 24-30 (Phase 3), publishable ~Month 32
- **Core contribution**: Uses discopt as a differentiable optimization layer inside a neural network for materials inverse design. The network takes a target material property (e.g., bandgap, tensile strength) and outputs processing parameters by solving an embedded MINLP that enforces physical constraints. The entire pipeline -- property target to processing recipe -- is trained end-to-end. Discrete variables represent crystal structure choices, dopant selections, or heat treatment protocols. Continuous variables represent composition fractions, temperatures, and durations.
- **Domain**: Materials science (inverse design)
- **Minimum viable results**:
  1. Demonstrate on a materials property prediction dataset (e.g., Materials Project bandgap data, or Matbench suite). Train a model that maps target bandgap to optimal composition + structure.
  2. The MINLP layer enforces: charge neutrality, Pauling electronegativity rules, discrete crystal structure selection (from a finite set), continuous composition fractions summing to 1.
  3. Show that end-to-end training produces designs with 20-50% higher "designability" (fraction of predicted designs that are thermodynamically stable per convex hull analysis) compared to post-hoc constraint enforcement.
  4. Ablation: compare Level 1 vs Level 3 differentiability. Show that Level 3 (implicit differentiation at the active set) provides more useful gradients for training.
  5. Validate 3-5 predicted designs against DFT calculations or literature data.
- **Target venue**: **Nature Communications** or **ACS Central Science**
  - Rationale: The combination of differentiable optimization + discrete materials design + end-to-end learning is novel enough for a high-impact venue. Nature Communications publishes computational materials methods with demonstrated novelty. ACS Central Science reaches the broader chemistry audience.
  - Alternative: **npj Computational Materials** if Nature Communications is too ambitious, or **ICLR** if framed as a machine learning methods contribution with materials as the application domain.
- **Dependencies**: WS-D Level 3 (T23, strongly preferred), WS4 (custom IPM for vmappable solving, T17), WS6 (GPU batch, T19), WS10-a (advanced relaxations improve solution quality for real problems, T27)
- **Builds on**: Cites Paper A2 (decision-focused learning methodology), discopt solver paper. This is the "marquee" application paper that demonstrates all three value propositions simultaneously.

---

## Paper A5: Autonomous Experiment Design with Real-Time Discrete Optimization

- **Title (working)**: "Real-Time Autonomous Experimental Design with Batch Mixed-Integer Optimization on GPU"
- **Timing**: Month 16-20 (late Phase 2), publishable ~Month 22
- **Core contribution**: Integrates discopt into an autonomous experiment design loop where, at each iteration, a Bayesian model is updated with new data and an acquisition function is optimized over a mixed-integer space of experimental conditions (discrete: catalyst type, solvent choice, equipment configuration; continuous: temperature, concentration, time). The key insight: discopt's batch solving allows evaluating many candidate experiments simultaneously, enabling look-ahead strategies that are computationally infeasible with serial solvers.
- **Domain**: Autonomous experimentation / laboratory automation
- **Minimum viable results**:
  1. Simulated autonomous loop on a published reaction optimization dataset (e.g., Shields et al. 2021 Buchwald-Hartwig dataset, or a dataset from the Aspuru-Guzik group).
  2. At each iteration, optimize a batch expected improvement (q-EI) acquisition function over 5-8 discrete choices and 3-5 continuous parameters. Solve for batch sizes of 4, 8, 16, 32 experiments per round.
  3. Compare: (a) random selection, (b) serial BO with continuous relaxation, (c) serial BO with BARON for exact MINLP, (d) batch BO with discopt. Show that (d) achieves the target objective in fewer total experiments and less wall-clock time.
  4. Demonstrate real-time capability: acquisition optimization completes in < 10 seconds for batch size 32, enabling integration with robotic platforms (where experiments take minutes to hours).
  5. Optional stretch: Connect to a simulated robotic experiment scheduler that dispatches batches to parallel reactors.
- **Target venue**: **Nature Machine Intelligence** or **Chemical Science** (RSC)
  - Rationale: Nature Machine Intelligence publishes autonomous science workflows and values computational infrastructure contributions. Chemical Science has a strong track record of publishing autonomous chemistry optimization papers (Shields, Aspuru-Guzik group).
  - Alternative: **JACS Au** (ACS, open access) if the chemistry application is compelling, or **Patterns** (Cell Press) if framed as a data science methodology.
- **Dependencies**: WS6 (batch evaluator, T19), WS5 (working solver, T14-T15), basic Phase 2 capabilities. Does not strictly require differentiability but benefits from it.
- **Builds on**: Cites discopt solver paper, Paper A1 (batch BO methodology). Focuses on the "real-time" and "autonomous loop" angles rather than the pure batch speedup.

---

## Paper A6: Stochastic Energy System Dispatch with Differentiable MINLP

- **Title (working)**: "Differentiable Unit Commitment for Learning-Based Energy System Operation under Uncertainty"
- **Timing**: Month 24-28 (Phase 3), publishable ~Month 30
- **Core contribution**: Applies decision-focused learning to the unit commitment problem in power systems: train a neural network load/generation forecaster end-to-end to minimize dispatch cost, where the optimization layer is a MINLP with binary generator on/off decisions and continuous power output subject to nonlinear AC power flow constraints. Uses discopt's batch solving for stochastic programming (solving across many scenarios in parallel) and differentiability for end-to-end training.
- **Domain**: Energy systems / power systems
- **Minimum viable results**:
  1. Small power system (IEEE 14-bus or 30-bus) with 5-10 generators, each with binary commitment and continuous output variables. Nonlinear constraints from simplified AC power flow (or DC power flow + nonlinear cost curves).
  2. Stochastic dispatch: generate 100-500 renewable generation/load scenarios from a probabilistic forecaster. Solve all scenarios in parallel with `jax.vmap(discopt.solve)`. Compare vs. sequential BARON or Bonmin.
  3. Decision-focused training: train an Equinox-based load forecaster that minimizes expected dispatch cost (not forecast RMSE). Show 10-25% cost reduction vs. two-stage predict-then-optimize.
  4. Sensitivity analysis: compute `d(total_cost)/d(renewable_capacity)` to quantify the marginal value of additional solar/wind capacity.
  5. Scale analysis: show how batch solving enables consideration of 100+ scenarios where serial solving is limited to 5-10 scenarios in practice.
- **Target venue**: **Applied Energy** (Elsevier) or **IEEE Transactions on Power Systems**
  - Rationale: Applied Energy (IF ~11) publishes ML-for-energy-systems work and values methodological novelty. IEEE T-PWRS is the premier power systems venue and has published extensively on unit commitment optimization. Both audiences would find differentiable MINLP dispatch novel.
  - Alternative: **NeurIPS** (ML conference, applications track) if framed as a decision-focused learning methodology paper.
- **Dependencies**: WS-D Level 1 or Level 3 (T22/T23), WS6 (batch solving, T19), WS10-a (advanced relaxations for nonlinear power flow, T27)
- **Builds on**: Cites Paper A2 (decision-focused learning for process engineering, same methodology different domain), discopt solver paper. Demonstrates scalability of the approach to a different engineering domain.

---

## Paper A7: Cardinality-Constrained Portfolio Optimization at Scale

- **Title (working)**: "GPU-Batched Cardinality-Constrained Portfolio Optimization: Sensitivity Analysis over 10,000 Market Scenarios"
- **Timing**: Month 14-18 (early-mid Phase 2), publishable ~Month 20
- **Core contribution**: Uses discopt to solve the cardinality-constrained portfolio optimization problem (MIQCQP) -- select at most K assets from N candidates to minimize portfolio variance subject to return targets. The key demonstration: solve the same portfolio problem across 10,000 market scenarios (different return vectors, covariance matrices) in parallel using `jax.vmap`, enabling Monte Carlo risk assessment that is infeasible with serial solvers. Additionally, uses Level 1 differentiability to compute portfolio sensitivity to return estimates.
- **Domain**: Quantitative finance / operations research
- **Minimum viable results**:
  1. Portfolio problem with N=20-50 assets, K=5-15 cardinality constraint (directly extends `example_portfolio()` in the codebase). This is MIQCQP, well within Phase 2 capabilities.
  2. Monte Carlo stress testing: sample 10,000 market scenarios from a multivariate distribution. Solve portfolio optimization under each scenario with `jax.vmap(discopt.solve)`. Report full distribution of optimal portfolio compositions and risk metrics.
  3. Wall-clock comparison: discopt (batched GPU, 10,000 scenarios in one call) vs. Gurobi (serial, 10,000 sequential solves) vs. CPLEX. Target: 20-100x speedup.
  4. Sensitivity: `d(optimal_variance)/d(expected_return_i)` via Level 1 differentiability. Show this identifies assets whose return estimates most impact the optimal portfolio, useful for model risk assessment.
  5. Backtest on historical S&P 500 data: show that batch scenario analysis enables robust portfolio construction that outperforms single-scenario optimization in out-of-sample returns.
- **Target venue**: **Operations Research** (INFORMS) or **Quantitative Finance**
  - Rationale: Operations Research is the top venue for optimization methodology in practice. Quantitative Finance targets the finance practitioner audience. Both would value the batch-solving angle for scenario analysis.
  - Alternative: **Management Science** (broader business audience) or **Computational Management Science** (focused on computational methods).
- **Dependencies**: WS6 (batch evaluator, T19), WS5 (working solver, T14-T15), WS-D Level 1 (T22, for sensitivity demonstration)
- **Builds on**: Cites discopt solver paper. This is a "quick win" application paper because the problem formulation already exists in the codebase (`example_portfolio()`), MIQCQP is well within solver capabilities by Phase 2, and the finance audience is large and engaged.

---

## Paper A8: Proof-of-Concept: Small-Scale MINLP Benchmarks Solved in JAX

- **Title (working)**: "discopt: A JAX-Native Mixed-Integer Nonlinear Programming Solver"
- **Timing**: Month 10-12 (immediately after Phase 1 gate), publishable ~Month 14
- **Core contribution**: This is the **solver paper**, not an application paper per se, but it includes domain-motivated demonstrations. It introduces discopt's architecture, validates correctness on 25+ MINLPLib instances, and demonstrates small-scale application examples (process synthesis, pooling, reactor design from the existing examples). The application demonstrations are modest -- the point is that a JAX-native MINLP solver exists and works.
- **Domain**: Optimization methodology (with chemical engineering + operations research examples)
- **Minimum viable results**:
  1. Correctness: 25 MINLPLib instances solved with zero incorrect results.
  2. Architecture: Rust B&B + JAX relaxation + HiGHS LP + Ipopt NLP demonstrated end-to-end.
  3. Small application demos: solve all 7 example models from the codebase (simple MINLP, pooling, process synthesis, portfolio, reactor design, facility location, parametric).
  4. Preliminary performance: comparison vs. Couenne and Bonmin on the 25 solved instances (not expected to win, but establishes baseline).
  5. Preview of Phase 2 capabilities: mention ongoing work on vmap batch solving, custom GPU IPM, differentiability.
- **Target venue**: **JOSS** (Journal of Open Source Software) or **INFORMS Journal on Computing**
  - Rationale: JOSS is ideal for a software paper announcing discopt's existence (peer-reviewed, indexed, low overhead). INFORMS JoC is the top venue for optimization software and would give more academic credibility but requires stronger computational results.
  - Alternative: **Mathematical Programming Computation** (MPC) once Phase 2 results are available -- this may be better as a follow-up once GPU results exist.
- **Dependencies**: Phase 1 gate passed (T16). All Phase 1 work streams complete.
- **Builds on**: Nothing prior -- this is the foundational paper that all application papers cite.

---

## Sequencing and Strategic Considerations

### Publication Timeline

| Month | Paper | Phase | Key Capability Demonstrated |
|-------|-------|-------|---------------------------|
| ~14 | **A8** (solver paper / JOSS) | Post-Phase 1 | Correctness, JAX-native MINLP exists |
| ~20 | **A7** (portfolio / finance) | Phase 2 | vmap batch over 10K scenarios |
| ~22 | **A1** (batch BO / materials) | Phase 2 | vmap for batch acquisition optimization |
| ~22 | **A5** (autonomous experiments) | Phase 2 | Real-time batch optimization for experiment loops |
| ~26 | **A2** (decision-focused learning / ChemE) | Post-Phase 2 | Differentiability + process engineering |
| ~28 | **A3** (molecular screening) | Phase 2/3 | Inter-solve parallelism at scale |
| ~30 | **A6** (energy dispatch) | Phase 3 | Differentiable + batch + stochastic |
| ~32 | **A4** (inverse materials design) | Phase 3 | All three value propositions combined |

### Strategic Observations

**1. Phase 1 is necessary but not sufficient for compelling application papers.**
Phase 1 produces a correct solver, but without batch solving (vmap) or differentiability, it has no advantage over existing tools for domain scientists. The solver paper (A8) can be published from Phase 1, but all real application papers require Phase 2 capabilities.

**2. The "quick wins" after Phase 2 are portfolio (A7) and batch BO (A1, A5).**
These papers rely primarily on vmap batch solving, which is the most mechanically straightforward capability. They do not require Level 3 differentiability or advanced relaxations. They should be targeted first.

**3. Differentiability papers (A2, A4, A6) require Phase 2 completion and are higher risk.**
Decision-focused learning through a MINLP solver is genuinely novel -- no one has demonstrated this. But it also means there is no established benchmark or comparison methodology. These papers need careful experimental design.

**4. Domain choice affects accessibility and impact differently.**
- **Materials science** (A1, A4): Large, active community investing heavily in ML-for-materials. Competition is high but the discrete optimization angle is underexplored.
- **Chemical engineering** (A2): Long history of MINLP use; audience understands the problem deeply. The decision-focused learning angle is novel for this community.
- **Drug design** (A3): Enormous audience and funding interest, but the MINLP formulation of molecular design is less standard than for process engineering. Requires more setup/justification.
- **Energy** (A6): Very active ML-for-energy community. Unit commitment is a well-understood MINLP. Stochastic scenario analysis is a pressing need.
- **Finance** (A7): Cardinality-constrained portfolio is a well-studied MIQCQP. The scenario analysis angle is practical and immediately compelling. Fastest path to a published application.
- **Autonomous experiments** (A5): Hot topic area (self-driving labs). The "real-time" angle is distinctive but requires demonstrating that optimization is actually the bottleneck in a real workflow.

**5. Citation cascading: the solver paper is critical infrastructure.**
Paper A8 (solver/JOSS paper) must be published first so that all application papers can cite it. If A8 is delayed, all application papers are delayed or must include excessive solver description.

**6. Collaboration strategy.**
Each application paper benefits from a domain expert co-author:
- A1, A4: Materials scientist with access to validation datasets and DFT
- A2: Chemical engineer with process simulation expertise
- A3: Computational chemist with QSPR modeling experience
- A5: Experimental chemist with autonomous lab access or simulation
- A6: Power systems engineer with unit commitment modeling expertise
- A7: Quantitative finance researcher with portfolio modeling experience

Recruiting these collaborators early (Months 10-14, while Phase 2 work proceeds) is essential.

### Risk Assessment for Application Papers

| Paper | Risk | Mitigation |
|-------|------|------------|
| A8 (solver) | LOW: just needs Phase 1 completion | Submit to JOSS immediately after Phase 1 gate |
| A7 (portfolio) | LOW: MIQCQP is well within capabilities, problem already coded | Start experiments as soon as vmap batch works |
| A1 (batch BO) | MEDIUM: need GP/NN surrogate + BO loop integration | Use existing BO libraries (e.g., BoTorch concepts adapted to JAX) |
| A5 (autonomous) | MEDIUM: real-time claims need wall-clock benchmarks | Use published reaction datasets for simulated autonomous loop |
| A2 (decision-focused) | MEDIUM-HIGH: novel methodology, no established baselines | Start with simple process networks, validate gradient quality carefully |
| A3 (molecular) | MEDIUM-HIGH: MINLP formulation of molecular design is non-standard | Collaborate with computational chemist; use established QSPR models |
| A6 (energy) | MEDIUM: standard problem formulation but nonlinear AC power flow is hard | Use simplified (DC) power flow + nonlinear costs as fallback |
| A4 (inverse design) | HIGH: requires Level 3 differentiability + meaningful materials validation | This is the "home run" paper; accept that it may take longer |

---

## Appendix: Capability-to-Paper Dependency Matrix

| Capability | A8 | A7 | A1 | A5 | A2 | A3 | A6 | A4 |
|-----------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Working solver (Phase 1) | REQ | REQ | REQ | REQ | REQ | REQ | REQ | REQ |
| vmap batch solving (WS6) | -- | REQ | REQ | REQ | OPT | REQ | REQ | REQ |
| Level 1 differentiability (WS-D) | -- | OPT | OPT | OPT | REQ | -- | REQ | OPT |
| Level 3 differentiability (WS-D) | -- | -- | -- | -- | PREF | -- | OPT | REQ |
| Custom JAX IPM (WS4) | -- | PREF | PREF | PREF | PREF | PREF | PREF | REQ |
| Advanced relaxations (WS10-a) | -- | -- | -- | -- | -- | OPT | OPT | PREF |
| GPU speedup >= 15x (Phase 2 gate) | -- | REQ | REQ | REQ | -- | REQ | REQ | REQ |

REQ = required, PREF = strongly preferred, OPT = optional enhancement, -- = not needed

---

*This report should be revisited after Phase 1 gate (Month 10) to reassess timing and adjust based on actual solver capabilities and any new collaboration opportunities.*
