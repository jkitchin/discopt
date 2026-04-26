# Presolve / Preprocessing Roadmap

A roadmap for evolving discopt's presolve from the current set of independent
passes (FBBT, OBBT, probing, simplify) into a structural NLP/MINLP
preprocessing layer comparable to what modern MIP solvers ship.

This document is a planning artifact. Items are grouped by theme, with a
suggested phasing at the end. Each item lists what it does, where in the
codebase it lands, references, dependencies, and a rough effort estimate.

## Motivation

Branch-and-bound for MINLP pays exponential cost per node in the worst case.
Every preprocessing dollar amortizes over the entire tree, so the leverage on
preprocessing is structurally larger than on per-node techniques. Three
specific reasons preprocessing matters more for MINLP than even for MIP:

1. **Relaxation tightness is bounds-driven.** Every relaxation arithmetic
   discopt ships — McCormick, αBB, future Taylor/Chebyshev models — gets
   monotonically tighter as variable bounds tighten. Root-node OBBT is often
   worth more than an extra cutting-plane pass at every node.
2. **Convexity unlocks downstream specialization.** Convex constraints can
   skip relaxation entirely. Recognizing convex substructure at presolve time
   is a transformation, not just a label.
3. **Structural detection is preprocessing.** Polynomial-to-quadratic
   reformulation, reduction-constraint detection, symmetry, separability —
   none of these are recoverable by branching once you've started the tree.

The high-level pitch: build a presolve pipeline that treats the MINLP
*structure* as a first-class output, not just tightened bounds. Downstream
components (relaxation compiler, branching, primal heuristics) consume that
structure.

## Current state

### Rust side (`crates/discopt-core/src/presolve/`)

| Pass         | File          | Capability                                    |
|--------------|---------------|-----------------------------------------------|
| FBBT         | `fbbt.rs`     | Forward/backward bound tightening on the IR   |
| OBBT         | `obbt.rs`     | LP-based bound tightening                     |
| Probing      | `probing.rs`  | Binary fix-and-propagate                      |
| Simplify     | `simplify.rs` | Algebraic simplification, big-M cleanup       |

These run as independent functions invoked by the solver orchestrator.
They do not share a fixed-point loop, do not report change deltas to one
another, and do not have a unified budget.

### Python side (relevant preprocessing-adjacent code)

| Module                                | Capability                                        |
|---------------------------------------|---------------------------------------------------|
| `_jax/convexity/`                     | Convexity certificates (interval AD, Gershgorin)  |
| `_jax/term_classifier.py`             | Term-level structure detection                    |
| `_jax/problem_classifier.py`          | Top-level problem classification                  |
| `_jax/sparsity.py`, `sparse_*`        | Sparsity exploitation                             |
| `_jax/scaling.py`                     | Numerical scaling                                 |
| `_jax/gdp_reformulate.py`             | GDP big-M / hull reformulation                    |
| `_jax/obbt.py`                        | Python-side OBBT (LP-driven via JAX relaxations)  |

Python preprocessing logic exists but is dispatched ad-hoc from the solver,
not orchestrated as a pipeline.

### What's missing relative to a serious presolve layer

- No orchestrator: passes don't iterate to a fixed point.
- No change-tracking: a pass that tightens bounds doesn't trigger re-running
  passes that consume bounds.
- No structural-output contract: the relaxation compiler can't ask "is this
  block convex?" or "is this monomial reformulable as quadratic?" because
  presolve doesn't produce that as output, only bound deltas.
- No global budget or determinism controls: presolve can't be capped or
  reproduced exactly across runs.
- No Rust↔Python presolve handshake: convexity detection (Python) and bound
  tightening (Rust) live in separate worlds.

## Design principles

The roadmap is driven by five principles, in priority order:

1. **Correctness is non-negotiable.** Every pass must be sound: bounds may
   only tighten, never widen; reformulations must preserve the feasible set
   and optimum to within documented tolerances.
2. **Passes are pluggable and report deltas.** A pass returns "what changed"
   (variables tightened, constraints removed, structure detected). The
   orchestrator uses this to decide what to re-run.
3. **Fixed-point with budget.** The orchestrator iterates passes until no
   pass reports progress or until a global budget (time, iterations, work
   units) is exhausted.
4. **Determinism by default.** Same input + same options produce the same
   presolved model. Tie-breaks, hash orders, and parallelism are seeded.
5. **Structure is a first-class output.** Presolve emits a `PresolveResult`
   that carries not just the tightened model but a structural manifest:
   detected convex blocks, polynomial degrees, symmetry orbits, aggregated
   variable substitutions, etc.

## Roadmap items

Items are independent unless an explicit dependency is called out. Effort
labels: **S** = days, **M** = 1–3 weeks, **L** = > 1 month.

### Track A — Orchestrator and infrastructure

#### A1. Presolve orchestrator (S–M)

A central driver that runs registered passes to a fixed point under a budget,
collects deltas, and reports a structured `PresolveResult`. Passes register
with metadata: what they read, what they write, expected cost.

**Where.** `crates/discopt-core/src/presolve/orchestrator.rs` (new), called
from the solver entry point in place of the current ad-hoc invocations.

**Dependencies.** None — but everything else in the roadmap depends on this.

**Acceptance.** Existing passes (`fbbt`, `obbt`, `probing`, `simplify`) run
through the orchestrator with no behavioral change. Presolve produces a
deterministic delta report. Budget cap honored within ±10% wall.

#### A2. Pass-delta protocol (S)

Define the `PresolveDelta` type: bounds tightened (per-variable), constraints
removed, variables fixed/aggregated, structure flags raised. Each existing
pass refactored to return one.

**Where.** `crates/discopt-core/src/presolve/delta.rs` (new); refactors to
`fbbt.rs`, `obbt.rs`, `probing.rs`, `simplify.rs`.

**Dependencies.** A1.

#### A3. Rust↔Python presolve handshake (M)

Allow Python-side passes (convexity detection, polynomial structure) to run
inside the orchestrator's fixed-point loop. Python passes call back into the
Rust IR through PyO3 with the same `PresolveDelta` contract.

**Where.** `crates/discopt-python/` PyO3 bindings; Python entry points in
`python/discopt/_jax/presolve/` (new package).

**Dependencies.** A1, A2.

#### A4. Determinism + reproducibility harness (S)

Seeded RNG, stable iteration order, golden-file tests for presolve output on
a fixed instance set.

**Where.** `crates/discopt-core/tests/presolve_determinism.rs`.

**Dependencies.** A1.

### Track B — Bound and structural propagation

#### B1. Implied-bound propagation across nonlinear constraints (M)

Existing FBBT propagates through the expression DAG of each constraint
independently. Implied-bound propagation uses bounds learned on one
constraint to tighten another that shares variables. Standard in MIP,
underused in MINLP.

**Where.** `crates/discopt-core/src/presolve/implied_bounds.rs` (new).

**References.**
- Achterberg & Wunderling (2013), *Mixed Integer Programming: Analyzing 12 Years of Progress*.
- Belotti et al. (2009), *Branching and bounds tightening techniques for non-convex MINLP*, Optimization Methods and Software.

**Dependencies.** A1, A2.

#### B2. Reverse-mode interval AD on JAX DAG (M)

Mirror of `presolve/fbbt.rs` on the JAX relaxation DAG. Lets Python-side
relaxations participate in bound tightening directly, without bouncing
through Rust.

**Where.** Extend `python/discopt/_jax/convexity/interval_ad.py`.

**References.**
- Schichl & Neumaier (2005), *Interval analysis on directed acyclic graphs for global optimization*, J. Global Optim. 33.

**Dependencies.** A3.

#### B3. Persistent bound tightening across the tree (M)

Run lightweight FBBT/probing at strategically chosen tree nodes (not just
root). Re-use root presolve work via a delta encoding so the in-tree pass
only does incremental work.

**Where.** `crates/discopt-core/src/bnb/` integration with `presolve/`.

**References.**
- Belotti, Lee, Liberti, Margot, Wächter (2009).
- Khajavirad & Sahinidis (2018), *A hybrid LP/NLP paradigm for global optimization relaxations*, Math. Prog. Comp. 10.

**Dependencies.** A1.

### Track C — Aggregation, substitution, redundancy

#### C1. Variable aggregation and substitution (S–M)

Detect equalities of the form `x = a*y + b` (or more general affine
combinations) and globally substitute. Standard MIP presolve, currently
absent from discopt.

**Where.** `crates/discopt-core/src/presolve/aggregate.rs` (new).

**References.**
- Achterberg, Bixby, Gu, Rothberg, Weninger (2020), *Presolve reductions in mixed integer programming*, INFORMS J. Computing 32.

**Dependencies.** A1, A2.

#### C2. Variable elimination in factorable expressions (M)

Distinct from C1: eliminates intermediate variables in factorable
expressions where one factor is uniquely determined by the others. MC++
implements this as a presolve.

**Where.** `crates/discopt-core/src/presolve/factorable_elim.rs` (new).

**References.** MC++ source (`omega-icl/mcpp`); related ideas in Smith &
Pantelides (1999).

**Dependencies.** A1, A2.

#### C3. Redundancy detection (S)

Dominated constraints, parallel constraints with looser RHS, redundant
integrality. Cheap and high-yield on real-world MINLPs.

**Where.** `crates/discopt-core/src/presolve/redundancy.rs` (new).

**Dependencies.** A1, A2.

#### C4. Coefficient strengthening for mixed-integer rows (M)

Well-understood for MIP (Savelsbergh, 1994). Extension to MINLP rows that
mix integer and continuous variables is partially open territory; the
linear part of any constraint can still be strengthened using the standard
techniques.

**Where.** `crates/discopt-core/src/presolve/coefficient_strengthening.rs`.

**References.**
- Savelsbergh (1994), *Preprocessing and probing techniques for mixed integer programming problems*, ORSA J. Computing 6.

**Dependencies.** A1, A2.

### Track D — Structural detection (the structure manifest)

The output of these passes feeds the relaxation compiler, branching, and
primal heuristics. They don't necessarily change the model; they annotate
it.

#### D1. Convexity-aware reformulation (M)

Promote `convexity/` from a certificate to a presolve pass that *rewrites*
constraints into forms whose convex hull is tractable (e.g., epigraph
substitution, exp-cone form for log-sum-exp, second-order-cone reformulation
of quadratics with PSD structure).

**Where.** New `python/discopt/_jax/presolve/convex_reform.py`, consuming
`convexity/`.

**References.**
- Lubin, Yamangil, Bent, Vielma (2018), *Polyhedral approximation in mixed-integer convex optimization*, Math. Prog. 172.
- Boyd & Vandenberghe, *Convex Optimization*, Ch. 4.

**Dependencies.** A3.

#### D2. Polynomial-to-quadratic reformulation (M–L)

Detect sparse polynomial subexpressions and reformulate to quadratic form
before relaxing. Two-step pipeline gives substantially tighter bounds than
direct McCormick on high-degree monomials.

**Where.** `crates/discopt-core/src/presolve/polynomial_quadratic.rs` (new).

**References.**
- Karia, Adjiman, Chachuat (2022), *Assessment of a two-step approach for global optimization of mixed-integer polynomial programs using quadratic reformulation*, Comput. Chem. Eng. 165, 107909.

**Dependencies.** A1, A2.

#### D3. Reduction-constraint detection in sparse polynomials (S–M)

Derive implicit equalities/inequalities from polynomial structure (e.g.,
implied SOS1 from a sum-of-squares pattern, implied complementarity).
Closely related to D2 and shares structure-detection machinery.

**Where.** Same module as D2 ideally.

**References.** Smith & Pantelides (1999); MC++ documentation.

**Dependencies.** D2 recommended.

#### D4. Symmetry detection and breaking (M)

Detect permutation symmetries in the model graph (e.g., interchangeable
units in design problems, identical reactor stages). Add symmetry-breaking
constraints or use orbital branching.

**Where.** `crates/discopt-core/src/presolve/symmetry.rs` (new); integration
with branching in `bnb/`.

**References.**
- Margot (2010), *Symmetry in integer linear programming*, in 50 Years of Integer Programming.
- Liberti, Ostrowski (2014), *Stabilizer-based symmetry breaking constraints for mathematical programs*, J. Global Optim.

**Dependencies.** A1, A2.

#### D5. Separability detection (S)

Detect block-separable structure in the objective/constraints. Enables
parallel relaxation evaluation and decomposition methods.

**Where.** `python/discopt/_jax/presolve/separability.py` (new).

**Dependencies.** A3.

### Track E — Numerics and scaling

#### E1. Presolve-time row/column equilibration (S)

Lift `_jax/scaling.py` into a presolve pass that produces a single scaling
applied consistently to LP, NLP, and IPM relaxations. Currently each solver
does its own scaling; that's wasted work and a source of inconsistency.

**Where.** `crates/discopt-core/src/presolve/scaling.rs` (new), with Python
mirror in `_jax/presolve/`.

**References.**
- Curtis & Reid (1972), *On the automatic scaling of matrices for Gaussian elimination*.

**Dependencies.** A1, A2.

#### E2. Domain reduction via duality (S–M)

Use LP-relaxation duals at the root to fix variables (analog of reduced-cost
fixing). Cheap once an OBBT pass has run.

**Where.** Extension to `crates/discopt-core/src/presolve/obbt.rs`.

**References.**
- Khajavirad & Sahinidis (2018).

**Dependencies.** A1, A2.

### Track F — GDP and disjunctive structure

#### F1. Big-M vs. hull reformulation as a presolve choice (M)

`_jax/gdp_reformulate.py` exists but the choice between big-M and hull (and
when to disaggregate) is currently fixed by the user. Make it a presolve
decision driven by problem structure: tightness of big-M, cost of
hull-reformulation auxiliaries, density of the disjunction.

**Where.** Lift `gdp_reformulate.py` into the presolve pipeline.

**References.**
- Grossmann & Trespalacios (2013), *Systematic modeling of discrete-continuous optimization models through generalized disjunctive programming*, AIChE J. 59.

**Dependencies.** A3.

#### F2. Implied-clique extraction across binaries (S)

Detect cliques among binary variables from constraint structure; add as
explicit clique constraints to tighten the LP relaxation.

**Where.** `crates/discopt-core/src/presolve/cliques.rs` (new).

**References.** Achterberg et al. (2020).

**Dependencies.** A1, A2.

## Phasing

A suggested three-phase rollout. Each phase is independently shippable.

### Phase P1 — Foundations (orchestrator + delta protocol)

- A1, A2, A4. Refactor existing passes onto the new protocol with no behavior change.
- Deliverable: presolve runs to a fixed point, reports a structured delta,
  is deterministic, has golden-file tests.
- Risk: low. Pure refactor.

### Phase P2 — Quick wins (aggregation, redundancy, implied bounds)

- C1, C3, B1, E1, E2.
- Deliverable: measurable reduction in variables/constraints on MINLPLib;
  measurable reduction in root LP gap from tighter bounds.
- Risk: low–medium. Standard MIP-style techniques, NLP extensions are
  incremental.

### Phase P3 — Structural detection layer

- D1, D2, D3, A3, B2, F1, F2.
- Deliverable: presolve emits a structural manifest. Relaxation compiler
  consumes it to produce tighter relaxations on convex blocks and high-degree
  polynomials.
- Risk: medium. D2 is architecturally invasive. D1 requires non-trivial
  integration between Python-side convexity detection and the Rust IR.

### Phase P4 — Advanced (symmetry, in-tree presolve, factorable elimination)

- B3, C2, C4, D4, D5.
- Deliverable: persistent presolve in the tree; symmetry-breaking on
  combinatorial MINLPs.
- Risk: medium–high. Symmetry detection has known scaling cliffs; in-tree
  presolve interacts with branching heuristics in subtle ways.

## Success metrics

Track on the existing benchmark suite:

- **Geomean root-node LP gap reduction** vs. current presolve.
- **Geomean tree size reduction** on MINLPLib instances that solve.
- **Number of MINLPLib instances solved within the time limit.**
- **Presolve wall time** as a fraction of total solve time (cap target: 10%).
- **Variable/constraint reduction ratios.**
- **Determinism**: bit-identical presolve output on repeat runs of the same
  input.

Phase gate: every phase must satisfy `incorrect_count == 0` on the
correctness suite (per the project-wide invariant).

## Open questions

1. **Where to draw the Rust/Python line for structural detection.** D2 in
   particular could live in either; Rust gives faster iteration on large
   models, Python is closer to the JAX relaxation compiler that consumes
   the output. Probably Rust for detection, Python for consumption, with a
   well-defined manifest contract crossing the boundary.
2. **In-tree presolve cost model.** Naive re-running of FBBT at every node
   is too expensive. We need a cost model that decides when an incremental
   presolve pays for itself — likely tied to the size of the current bound
   change since the last pass.
3. **Symmetry detection scalability.** Off-the-shelf graph isomorphism
   (nauty, bliss) is fast on most instances but has pathological cases.
   Decide upfront whether to ship a time-budgeted attempt or to gate
   symmetry detection on a structural pre-filter (regular reactor stages,
   identical units in a flowsheet).
4. **Determinism vs. parallelism.** Some passes (OBBT especially) are
   embarrassingly parallel. Pick a parallel-but-deterministic strategy
   (fixed work-stealing seed, deterministic reduction order) early — fixing
   this retroactively is painful.
5. **Interaction with the LLM layer.** `python/discopt/llm/reformulation.py`
   already detects big-M, weak bounds, symmetry, bilinear structure for
   advisory output. Decide whether the LLM advisor consumes the structural
   manifest produced by presolve, or whether they remain independent. The
   former is more coherent but creates a new coupling.

## Related

- Top-level `ROADMAP.md` — project-wide phase plan.
- `python/discopt/ro/ROADMAP.md` — robust-optimization module roadmap.
- GitHub issue #51 — bounding-arithmetic enhancements (Taylor/Chebyshev
  models, multivariate McCormick). Items D2 and D3 here cross-reference
  items 4 and 5 in that issue and should be treated as the presolve
  formulation of the same work.
