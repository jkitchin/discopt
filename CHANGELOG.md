# Changelog

All notable changes to discopt are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The release procedure that produces these entries is documented in
[`RELEASE.md`](RELEASE.md).

## [Unreleased]

### Added

- **Continuous stratified multistart at the root** (`feat`, #188). Pure-continuous
  nonconvex models had zero basin diversification on the spatial McCormick-LP
  path: the integer-centric primal heuristics (pump/ILS/diving/RINS/RENS) all
  no-op without integers, the root multistart NLP is skipped there, and the
  strided node NLP warm-starts from the parent point — so the incumbent stayed
  locked in the first LP-vertex basin (kall_congruentcircles_c51: parked at the
  1.5371 two-row packing vs the 1.0730 global). `solve_model` now runs a
  budgeted, deadline-gated stratified multistart
  (`primal_heuristics.continuous_multistart`) once at the root for nonconvex
  models with no integer variables; the c51-class reconstruction reaches the
  1.07301 global on the default path (siblings and the C-38 `kall_circles_c8a`
  soundness lock unregressed). Primal-only (heuristic-policy regime): every
  point is constraint-re-verified and `inject_incumbent` enforces strict
  improvement, so dual bounds and certificates are untouched.
  `DISCOPT_CONTINUOUS_MULTISTART=0` / `SolverTuning.continuous_multistart=False`
  restores the prior behavior.

### Changed

- **OBBT-on-auxiliaries reverse-FBBT cascade graduated default-ON** (`perf`, #208).
  The root branch-and-reduce fixpoint now propagates OBBT-tightened auxiliary
  (product/ratio) column bounds back onto the original variables through the
  nonlinear term definitions — the hyperbolic/root bounds the linear McCormick rows
  cannot express (`w=a·b ⟹ a∈[w]/[b]`, `w=aᵖ ⟹` p-th-root box, plus the
  trilinear/multilinear and ratio-of-products generalizations). The extra aux
  min/max LPs are budgeted to the reverse-FBBT-*reachable* columns
  (`obbt.cascade_reachable_aux`), which is bound-neutral vs a blanket cascade but
  drops ~87% of the aux probes, and the cascade runs root-only (no per-node cost).
  Graduation gate (`design/ab_cascade_aux.py`, 65-instance corpus, fair 30 s
  budget): cert-clean — 0 differential soundness violations, 0 optimum mismatches,
  0 cert regressions, +1 cert gain (`tls2` F→T) — and net-positive with 0
  regression: node-neutral on the convergent integer-heavy majority (converged-only
  2228 vs 2228) and helpful on the continuous spatial-branch class (`tspn08/10/12`
  prune to 1 node, `heatexch_gen3` 208→31 s wall; all-instances node_count −2.5%).
  An earlier too-tight 8 s A/B had read as net-negative; that was time-limited noise
  (`fac2`/`cvxnonsep_nsig30` converge bit-identically at a fair budget).
  `DISCOPT_OBBT_CASCADE_AUX=0` restores the prior default-OFF behavior.

## [0.6.0] - 2026-07-12

### Removed

- **AC-OPF and pooling application builders extracted to the `discopt-apps`
  plugin** (`refactor`, #431). `python/discopt/opf.py` (rectangular AC-OPF:
  `build_ac_opf_rectangular`, `ACOPF`, `Bus`, `Line`, `admittance_matrix`,
  `two_bus_example`) and `python/discopt/pooling.py` (pq-formulation:
  `build_pq_formulation`, `PoolingProblem`, `Input`, `Pool`, `Output`,
  `haverly_hpp1`), their tests, and the `ac_opf`/`pooling_pq` doc notebooks now
  live in the standalone
  [discopt-apps](https://github.com/jkitchin/discopt-apps) package, mirroring
  the course extraction (#430). Because discopt is a namespace package,
  `pip install discopt-apps` restores `discopt.opf` and `discopt.pooling`
  imports unchanged. These are pure builders over `discopt.modeling.core`; no
  core solver behavior changes. (The in-core
  `discopt.modeling.examples.example_pooling_haverly` example stays.)
- **DOE module extracted to the `discopt-doe` plugin** (`refactor!`, #389).
  `python/discopt/doe/` (design, screening, FIM, identifiability,
  discrimination, workbook CLI, Streamlit GUI), its 19 test files, 11 doc
  notebooks, notebook build scripts, and the doe skill/agents now live in the
  standalone [discopt-doe](https://github.com/jkitchin/discopt-doe) package.
  Because discopt is a namespace package, `pip install discopt-doe` restores
  `discopt.doe` imports and the `discopt doe` subcommand unchanged. The
  `doe`/`doe-gui` extras are gone — use `discopt-doe[gui,ml]`. discopt ≤0.5.x
  is the last line with DOE built in.

### Added

- **Public parametric-compilation API** (`feat(api)`, #389). New
  `discopt.parametric` module — the stable contract for external plugins
  (discopt-doe, discopt-mkm, ...) that compile model expressions into
  JAX-differentiable functions of `(x_flat, p_flat)`: `compile_expression`,
  `compile_response_function`, `extract_x_flat`, `flatten_params`,
  `param_total_size`, `variable_total_size`, `variable_slices`. The in-tree
  DOE module and `discopt.estimate` now consume this API instead of reaching
  into `discopt._jax` internals.
- **Generic CLI plugin subcommands** (`feat(cli)`, #389). External packages
  can register subcommands via the `"discopt.cli"` entry-point group (name =
  subcommand; value = module exposing `add_subparser(subparsers)` and
  `run(args)`). Plugin modules load lazily — only for their own subcommand or
  full help — and built-in names cannot be shadowed. The in-tree `discopt doe`
  subcommand now registers through this mechanism, ahead of its extraction to
  the standalone `discopt-doe` package (#389).
- **Reduced-space McCormick relaxation (MAiNGO-parity, opt-in)** (`feat(mcbox)`,
  #574, #575, #576, #577, #579, #580, #583). A propagating McCormick type with
  rule-based subgradients that relaxes in the original variable space (no
  auxiliary lifting), plus a jitted Kelley-cut LP on the in-house simplex,
  S-shaped intrinsics with per-regime subgradients, and sign-definite reciprocal
  division. Selected with `DISCOPT_RELAX_SPACE=reduced` (**default-OFF**); sound
  where it engages, pursued for generality/robustness parity with MAiNGO rather
  than as a speed lever. See `docs/dev/maingo-parity-plan.md`.
- **Hybrid physics + ML: trainable surrogates and simultaneous neural-DAE
  training** (`feat(nn)`, #595). Surrogate weights become decision variables so a
  neural rate law can be trained jointly with a physics model (e.g. inside a
  collocation DAE); see `discopt.nn.trainable` / `discopt.nn.surrogate` and the
  multi-experiment fitting glue in `discopt.dae.fit`.
- **Entropy-family relaxation: centropy tangent-plane underestimator**
  (`feat(relax)`, #597). Linearizes the `x·log(x)` / entropy atom so entropy-style
  objectives become certifiable rather than local-NLP-only.
- **Exact 1-D univariate-composite and positive-product envelopes (opt-in)**
  (`cert:LR-2`, #627). `DISCOPT_UNIVARIATE_ENVELOPE` (H-UNI) and
  `DISCOPT_LOG_MONOMIAL` (H-LOG) add exact convex/concave hulls for
  single-variable composites and log-space positive products (root-certifies
  nvs09). Both **default-OFF**; graduation to default-ON is tracked in #632
  (canonical factorable normal form). OFF is byte-identical to prior main.
- **Binary (`b`) `.nl` parsing** (`cert:TAIL-1b`, #593). Binary-format AMPL `.nl`
  files are transcoded to text and read (e.g. `st_miqp5`).

### Changed

- **`pounce-solver` minimum raised to `>=0.8`** (`chore(deps)`, #633). pounce 0.8
  makes `Problem` a compiled object; discopt's usage is unaffected (method
  dispatch behind `hasattr`).
- **`feral` LP engine bumped to crates.io `0.14.0`** (`chore(deps)`, #628),
  retiring the git-rev pin from #112.
- **Default-ON graduation of three per-node levers** (`graduate`, #616): the
  density-aware LU route (`lu_density_route`), objective-branch-priority
  (`obj_branch_priority`), and loose-product lifting (`lift_loose_products`) are
  now on by default, validated soundness-neutral (`incorrect_count = 0`).
- **Density-aware dense/sparse LU route for wide-McCormick node bases**
  (`perf(lp)`, #573, #591) with a failure-triggered dense retry that cures the
  nvs21 certificate loss; plus zero-pivot / refactorization-breakdown fixes in
  the LP engine (#570, #571).
- **Documentation build is now zero-warning** (`docs`, #605; 132 → 0).

### Fixed

- **AMP false-feasible from per-iteration MILP-budget starvation** (`fix(amp)`,
  #621). The per-iteration MILP time budget was sized off the `max_iter` cap, so
  a large `max_iter` starved each MILP on cold runs; it is now sized by an
  expected iteration horizon.
- **Gap-closed solves are now certified soundly** (`fix(cert)`, #604, #603,
  #613). A node-LP failure no longer poisons certification when the bound
  accounting stays rigorous (per-node sound accounting, #604); tainted trees now
  report the strongest rigorous dual bound instead of the taint-dropped one
  (#603); spatial gap-closed solves certify over soundly-floored sentinel
  removals (#613).
- **`nvs22` false-optimal on the reduced-space and cut-inheritance paths**
  (`fix`, #582, #568). The reduced-space evaluator now refuses non-finite boxes;
  column-identity-safe cut inheritance fixes the cut-inherit false-optimal.
- **H-UNI no longer builds a hull over effectively-unbounded boxes** (`cert:LR-3`,
  #631) — was a false-infeasible on the opt-in envelope path; guarded by a
  solver-sense finiteness check.
- **AMP integration suite de-flaked** (`ci(amp)`, #607, #608) via pytest-xdist
  process distribution and isolation of the slow trig/abs certification tests.

## [0.5.0] - 2026-07-01

### Added

- **Implicit-function expression node** (`feat(modeling)`, #379). `m.implicit(
  residual, u_inputs, n_unknowns, x0=)` defines a vector `v` by a square system
  `g(u, v) = 0`, compiled to a differentiable JAX inner solve (Newton forward;
  implicit-function-theorem derivatives via `jax.lax.custom_root`, which supports
  the higher-order AD the NLP Hessian needs). Rides on `CustomCall`, so it is
  **local-NLP-only** (no global certificate) and rejects integers. The core-side
  primitive for implicit variable-aggregation of irreducible cyclic blocks;
  documented in `docs/notebooks/implicit_function_node.ipynb`.
- **Hardened pure-Rust LP engine** (`feat(lp)`, #368). Numeric-focus LU with
  in-engine iterative refinement and condition/growth signals (via `feral`
  0.12.0), primal + dual refined recovery on a drifted-Optimal, dual-simplex
  anti-cycling (Bland + stall counter), and **EXPAND anti-degeneracy** (Gill et
  al.) in the Harris ratio test — ~15× fewer degenerate pivots on the
  lifted-relaxation corpus, validated soundness-neutral against the gauntlet and
  a BARON head-to-head.
- **Namespace-package support** (`feat(packaging)`). `discopt` now extends its
  `__path__` via `pkgutil.extend_path`, so external distributions (e.g. a
  `discopt-aggregation` plugin) can contribute submodules under the `discopt.*`
  namespace from a separate location on `sys.path` without modifying the core.
- **Set & index abstractions** (`feat(modeling)`). A Pyomo/JuMP-style named-set
  layer for sparse models, implemented as a pure-Python desugaring over the
  existing flat model (no solver/backend changes). Completes the Phase 7
  roadmap item. New public API:
  - `discopt.Set` / `discopt.RangeSet` (+ `ProductSet`) and `Model.set(...)`:
    arbitrary hashable members with inferred/declared `dimen`, set algebra
    (`|`, `&`, `-`, `*`), and filtering (`Set.where`, `with_first`,
    `with_last`). Product sets are lazy and accept flat or nested keys.
  - `Model.continuous/binary/integer/parameter(..., over=SET)` returning
    `IndexedVar` / `IndexedParam` backed by a single flat variable/parameter;
    per-key bounds/values via scalar, `dict`, or callable.
  - `Model.constraint(SET, rule, name=)` generating one constraint per member
    (named `name[key]`), with a `Skip` sentinel; `subject_to` now accepts
    generators of constraints. `dm.sum`/`dm.prod` aggregate over sets.
  - A transparent **linear fast path**: single-variable-affine, uniform-sense
    families are emitted as one sparse-matrix builder call (`fast=True`
    default) with identical results, falling back automatically otherwise.
  Documented in `docs/notebooks/sets_and_indexing.ipynb`; design in
  `docs/design/sets-and-indexing.md`; examples
  `example_transportation` / `example_assignment` /
  `example_multicommodity_flow`.
- **Decomposition benchmark instances** (`test(decomposition)`). Block-structured
  / two-stage MILP instances with known optima (`decomposition_problems.py`,
  registered in the `milp` suite) plus a consolidated correctness gate
  (`test_decomposition_benchmarks.py`) that checks Benders and Lagrangian
  against the known optima and the monolithic solver.
- **Generalized Benders Decomposition** (`feat(decomposition)`, Geoffrion 1972).
  `solve_benders` now handles a **convex nonlinear recourse** subproblem, not
  just a linear LP: when the model has a nonlinear objective or constraints it
  dispatches to `solve_gbd` (`discopt.solve_gbd`). Each optimality cut is the
  **Lagrangian dual value** as an affine function of the first-stage variables —
  `eta >= [L(x̂,ŷ) + m_y] + ∇_x L^T (x − x̂)` with sign-projected (dual-feasible)
  multipliers and the closed-form recourse-box correction `m_y` — which is a
  valid lower bound for *any* recourse point by the joint-subgradient inequality,
  so it stays sound even if the recourse NLP returns an inexact primal (the
  analogue of classical Benders' complete-dual cut). Recourse infeasibility at a
  0/1 first-stage point is excluded with a no-good cut. The reported lower bound
  is rigorous when the model is convex (gated on `classify_oa_cut_convexity`); on
  a nonconvex model GBD runs heuristically and reports `bound=None` so the
  `incorrect_count <= 0` gate is never threatened. POUNCE-only (no HiGHS).
- **Lagrangian B&B node-bound hook** (`feat(decomposition)`). `model.solve(
  lagrangian_bound=True)` fixes Lagrangian multipliers at the root and, at each
  MILP branch-and-bound node, combines a valid Lagrangian dual lower bound with
  the node's LP relaxation bound (`max()`), tightening pruning when the block
  subproblems lack the integrality property. Opt-in (default off), applies to
  linear minimization models with coupling structure, and no-ops cleanly
  otherwise; bounds are verified sound against brute-force enumeration.
- **Lagrangian relaxation solver** (`feat(decomposition)`). Dualizes coupling
  constraints (annotate with `model.mark_coupling(...)`) to produce a rigorous
  dual lower bound via `model.solve(decomposition="lagrangian")` /
  `discopt.solve_lagrangian`. The dual is maximized by a subgradient method
  (Polyak step) or a bundle / cutting-plane method, and a Lagrangian heuristic
  recovers a feasible primal incumbent. Documented in
  `docs/notebooks/tutorial_lagrangian.ipynb`.
- **Benders decomposition solver** (`feat(decomposition)`). Classical Benders
  for two-stage / block-angular (mixed-integer) linear programs via
  `model.solve(decomposition="benders")` / `discopt.solve_benders`. The master
  holds the complicating variables; the recourse-LP duals generate optimality
  cuts and a slack-penalized feasibility LP generates feasibility cuts. Cuts are
  **anchored at the primal recourse value** with a row-dual slope, so they stay
  sound even when the recourse optimum is set by variable bounds and with
  POUNCE's interior-point duals — **no HiGHS dependency** (runs on the POUNCE
  LP/MILP stack). Every cut is a global under-estimator, so the master objective
  is a rigorous lower bound (gap certified on convergence). Documented in
  `docs/notebooks/tutorial_benders.ipynb`.
- **Decomposition structure layer** (`feat(decomposition)`). Foundation for the
  upcoming Benders / Lagrangian solvers: a `Model` annotation API
  (`first_stage`/`second_stage`/`set_stage`/`set_block`/`mark_coupling`) and
  `discopt.detect_decomposition(model)`, which resolves annotations and
  auto-detects block structure (complicating variables default to integers;
  coupling constraints via a bridge heuristic reusing the separability scan).
  Exposed as `discopt.detect_decomposition` / `DecompositionStructure`.
- **Irreducible Infeasible Subsystem (IIS)** (`feat(infeasibility)`, #227). New
  `compute_iis(model)` returns a minimal infeasible subset of constraints/bounds
  via deletion filtering — exact for LP/MILP/convex, best-effort for nonconvex.
  Exposed as `discopt.compute_iis` / `IISResult`; documented in
  `docs/notebooks/infeasibility_iis.ipynb`.
- **Complementarity constraints via GDP disjunction** (`feat(modeling)`, #231).
  `Model.complementarity(x, y)` now reformulates through a GDP disjunction by
  default (`method="gdp"`), alongside the existing Scholtes regularization and
  SOS1/disjunctive paths, all unified behind one front-end and the
  `discopt.mpec` reformulation module. Documented in
  `docs/notebooks/complementarity_mpec.ipynb`.
- **RLT as a first-class solve option** (`feat(rlt)`, #223, #212). Level-1
  Reformulation-Linearization-Technique cuts are now a first-class solver choice
  (`rlt_cuts=True`) with per-node targeted RLT cut separation.
- **PSD / SOC cuts for QCQP + AC optimal power flow** (`feat(cuts)`, #203, #209).
  Dense-moment (PSD) eigenvalue cut separator and second-order-cone cuts for
  QCQP structure, plus a rectangular AC-OPF builder (`discopt.opf`:
  `build_ac_opf_rectangular`, `Bus`, `Line`, `ACOPF`) as the capstone target.
- **Differentiable MILP / MIQP** (`feat(diff)`, #221). Fix-and-differentiate
  framework propagates parameter sensitivities through integer programs via
  implicit (KKT) differentiation of the fixed continuous subproblem.
- **Conflict analysis / no-good cuts** (`feat(conflict)`). FBBT-driven conflict
  analysis derives no-good cuts from infeasible nodes.
- **Standard pooling problem + pq-formulation** (`feat(pooling)`). Bilinear
  quality-blending model builder with pq-cuts that tighten the McCormick
  relaxation; documented in `docs/notebooks/pooling_pq.ipynb`.
- **Geometric programming detection + log-space reformulation** (`feat(gp)`).
  `discopt.gp` recognizes posynomial structure (`classify_gp`) and solves via the
  convex log-space transformation (`as_geometric_program`, `solve_gp`).
- **Primal improvement heuristics** (`feat(heuristics)`). Diving, RINS, and
  local-branching heuristics added to the B&B primal-side search.
- **Public FBBT bound-tightening API** (`feat(tightening)`, #198). `discopt.tightening`
  exposes feasibility-based bound tightening for manual use; documented in
  `docs/notebooks/bound_tightening.ipynb`.
- **Integrality-aware FBBT bound snapping** (`feat(fbbt)`). Binary-indicator
  propagation snaps tightened bounds to integer values for sharper inference.
- **Periodic-variable bound reduction** (`feat(presolve)`, #215). Presolve pass
  reduces bounds of variables that only enter through periodic functions
  (`sin`/`cos`), unblocking otherwise-free angular variables.
- **`cuts='auto'` is the solver default** (`feat(cuts)`, #217). Auto cut
  selection balances bound tightening against node-count reduction.
- **Best-estimate node selection** (`feat(bnb)`) and **objective-gating priority
  branching** (`feat(bnb)`, #184) B&B search strategies.
- **Transcendental relaxation coverage** (`feat(relax)`, #216, #218). LP relaxer
  engaged for general transcendental nonlinearity; `asin`/`acos`/`acosh` gaps
  closed; non-smooth `abs`/`min`/`max` fixed; relaxation coverage audit added.
- **Run discopt as a GAMS solver** (`feat(gams)`, #119). GMO/GEV control-file
  link lets GAMS call discopt as an external solver; see `docs/gams_solver_link.md`.
- **Batched / multiple-RHS LP solving** (`feat(simplex)`). Shared-matrix batched
  ftran/btran for solving many RHS over one factorization, used by node-relaxation
  and DoE batches.
- **Per-node lifted-LP FBBT** (`feat(relaxation)`, #184). Opt-in
  (`DISCOPT_LIFTED_FBBT=1`) feasibility-based bound tightening that propagates
  the McCormick relaxation's *own* rows (`A_ub·z ≤ b_ub`, spanning the lifted
  product/monomial columns), recovering the bilinear-implied factor bounds that
  purely linear FBBT misses, then rebuilds the relaxation on the tightened box.
  This lifts `ex1252`'s structurally-zero node bound off 0 (a branched node goes
  `bound 0 → ~18987`, sound) so the B&B can certify optimality. Implemented in
  `discopt._jax.mccormick_lp` (vectorised over the sparse matrix); pinned
  multilinear factors are un-pinned by a hair so the build keeps the term at
  full arity and never drops `objective_bound_valid`. Sound by construction —
  only valid rows tighten, and the un-pin only enlarges the box. Regression
  locks in `test_bucket2_sound_bounds.py`.

- **Lifted-relaxation LP equilibration** (`feat(relaxation)`, #184). The lifted
  McCormick rows of a product over a wide variable box mix tiny constants (~1e-9)
  with large bound-derived coefficients (~1e7), giving a >1e15 coefficient spread
  on ex1252's boundary sub-boxes. HiGHS stalls on it (a 452×96 LP hits its time
  limit; the per-node soundness re-verifications then dominate the solve) while
  the pure-Rust simplex, which equilibrates internally, solves it in ~0.03s.
  `equilibrate_relaxation_lp` (`discopt._jax.milp_relaxation`) applies
  geometric-mean (Ruiz) row/column scaling, snapped to powers of two, before the
  external (HiGHS/POUNCE) backend solve when the spread exceeds 1e6 — turning
  previously timing-out boundary boxes into ~3-4s converged solves (e.g. the
  `x36=1,x37=1` box: time-out → 4.5s). The rescaling is exact (bound/feasibility
  unchanged, integer columns never scaled, solution mapped back through the
  column scale), so it only ever conditions — never alters — the result.
  Regression-locked in `test_bucket2_sound_bounds.py`.

- **Objective-gating priority branching** (`feat(bnb)`, #184). Opt-in
  (`DISCOPT_OBJ_BRANCH_PRIORITY=1`) branching-order heuristic that branches the
  integer variables gating the objective's nonlinear terms (those appearing in,
  or equality-linked to, a lifted product/monomial — e.g. ex1252's line-selection
  binaries `x36/x37/x38`) before other integers. The global dual bound is the
  minimum over the open frontier, so on problems whose bound is structurally 0
  until a *set* of binaries is jointly fixed it stays pinned at 0 under
  most-fractional branching (no single-variable score sees the joint jump);
  branching the gating binaries first reaches the depth where the per-node
  relaxation lifts each leaf off 0. Implemented via the existing `set_branch_hints`
  path in `solve_model` (`discopt.solver`) — pure search reordering over already
  fractional integer candidates, so it can never affect a bound or feasibility
  verdict. Detector locked by `test_bucket2_sound_bounds.py`.

- **POUNCE NLP backend declared as a dependency** (`feat(solvers)`). New
  `pounce` optional extra (`pip install discopt[pounce]`, dist `pounce-solver`)
  and a `requires_pounce` test marker. POUNCE is a standalone pure-Rust port of
  Ipopt (https://github.com/jkitchin/pounce). Added `solve_nlp_from_model` to
  `discopt.solvers.nlp_pounce` for parity with the cyipopt wrapper, plus
  `python/tests/test_nlp_pounce.py`.

### Changed

- **`feral` pinned to the crates.io 0.12.0 release** (`chore(deps)`, #375),
  carrying the LU-hardening APIs (element-growth getters, unsymmetric-LU
  condition estimate, richer `update()` instability signal) the numeric-focus
  simplex consumes; replaces the temporary git-rev pin.
- **Minimum `pounce-solver` bumped to 0.7** (`chore(deps)`). The interior-point
  KKT solve (`solve_lp_kkt`) the differentiable LP/QP layers and crossover use
  after the JAX LP-IPM retirement requires POUNCE ≥ 0.7.
- **POUNCE is now the default single-solve NLP backend** (`feat(solvers)`).
  For single continuous solves the `ipm` default is promoted to a KKT-valid
  backend via `_default_nlp_solver()`, resolving to POUNCE when installed and
  falling back to cyipopt. B&B convex-polish / dual-recovery passes likewise
  prefer POUNCE through the new `_solve_node_nlp_kkt` wrapper.
- **LP / MILP / QP / MIQP solves stay JAX-free** (`perf(solver)`, #224, #225).
  Linear and quadratic solve paths no longer import JAX, removing the cold-start
  compile tax on fresh solves; node QP relaxations now route through POUNCE.
- **Faster simplex** (`perf(simplex)`, #178, #180). Dual Devex pricing and a
  bound-flipping ratio test; one LU factorization is reused across a node's
  strong-branch probes, and one equilibration is shared across each batch of
  node solves.

### Removed

- **BREAKING: the JAX LP interior-point method is retired** (`refactor(lp)!`,
  #368, #371, #373). The MILP/MINLP node LP relaxations and the standalone LP
  path now use the pure-Rust simplex (degrading to POUNCE); `nlp_solver` governs
  only the NLP subproblem solver. `discopt._jax.lp_ipm` was deleted. This
  completes retirement of the LP fallback chain (`Rust simplex → HiGHS → POUNCE
  → JAX-IPM`).
- **HiGHS removed from the LP/MILP path** (`feat(solvers)`, #356) and the QP path
  (`qp_highs`, #359) — the pure-Rust core is the sole LP/MILP engine.
- **BREAKING: removed the deprecated `ripopt` aliases** (`feat(solvers)!`). The
  old in-repo Rust IPM crate `ripopt` was already superseded by POUNCE; the
  remaining compatibility shims are gone: the `discopt.solvers.nlp_ripopt`
  module, `nlp_solver="ripopt"` (now raises `ValueError`),
  `DISCOPT_MCCORMICK_BACKEND=ripopt`, `sipopt.ripopt_sensitivity`, and the
  `discopt_ripopt` benchmark key. Use `pounce` / `pounce_sensitivity` instead.

### Fixed

- **Sound lower bounds for `log²`/`exp²` objectives** (`fix(relax)`, #372, closes
  #369). The objective linearizer now registers squares of *any* lifted
  univariate call (not just trig), so a mixed objective like nvs09's
  `Σ log(·)² − (∏x)^0.2` produces a sound lower bound instead of falling back to
  a feasibility objective with no bound.
- **Pytest virtual-address cap raised 16 → 32 GB** (`ci`, #360) so JAX/XLA
  compilation no longer aborts with `std::bad_alloc` / exit 134 in CI.
- **Decomposition stage annotation on indexed variables** (`fix(decomposition)`).
  `model.first_stage(y[i])` / `set_stage` / `set_block` on an indexed element
  (`y[i]`) stringified to a stray key (`"y[3][0]"`) that never matched the
  variable name, so the annotated variable silently fell into the recourse
  subproblem and tripped the "integer in recourse" guard. The annotation now
  resolves an indexed reference (or single-variable expression) to its base
  variable name. Surfaced by the new curated adversarial example suite
  (`test_decomposition_adversarial.py`), which carries hand-crafted Benders/GBD
  instances with analytically known optima for each correctness hazard.
- **`.nl` export of builder constraints and objectives** (`fix(export)`). Linear
  constraints built directly into the Rust builder — via the fast-construction
  `add_linear_constraints` API and the indexed-constraint fast path — were
  silently omitted from `to_nl`, which reads `model._constraints`; likewise an
  objective set via `add_linear_objective` / `add_quadratic_objective` was
  exported as a zero placeholder. The model now records each emitted block
  (constraints and the `0.5 x'Qx + c'x + constant` objective) and the `.nl`
  writer reconstructs them — the quadratic part as an n-ary `SUMLIST` nonlinear
  objective term — so a fast-construction model round-trips through `.nl` with
  all constraints and the correct linear/quadratic objective (including a
  constant offset and `maximize` sense) intact.
- **MILP B&B node bound soundness** (`fix(solver)`). The per-node LP soundness
  gate now also rejects a relaxation point that violates the node's variable
  bounds, not only its constraint rows. The pure-Rust simplex adapter (and the
  POUNCE IPM) could return a basic point that violated the variable box on mixed
  equality/inequality nodes; such a point can be integral but off-bound (e.g. a
  binary at -1), pass the row check, and be accepted by the tree as a spurious
  integer incumbent — returning a wrong (too-low) optimum on some
  generalized-assignment-style MILPs. Regression covered in
  `python/tests/test_milp_node_bound_soundness.py`.
- **Clean errors for unsupported decomposition models** (`fix(decomposition)`).
  `solve_benders` / `solve_lagrangian` (and the B&B hook) now raise a clear
  `NotImplementedError` on models the linear extractor cannot handle — e.g.
  multi-dimensional indexed variables — instead of a stray internal `TypeError`.
- **Simplex equilibration over-scaling on noise entries** (`fix(simplex)`). The
  root cause of the MILP wrong-optimum bug below: the geometric-mean
  equilibration treated a numerically-negligible matrix entry (e.g. a ~1e-16 cut
  coefficient that is float noise, not structure) as a genuine nonzero, so a
  column's scale factor blew up to ~1e8. That pinned the variable's *scaled*
  bounds to ~0; the scaled simplex returned a within-tolerance value that
  unscaled into a gross original-space bound violation (a `[0,1]` variable at
  -1), accepted as a spurious integer incumbent. The equilibration now ignores
  entries more than ten orders of magnitude below a line's maximum when forming
  the factor, bounding the per-line dynamic range. The simplex now returns the
  correct vertex (verified against brute-force enumeration and HiGHS). Rust
  regression in `scaling::tests::noise_entry_does_not_overscale_column`.
- **MILP B&B node bound soundness** (`fix(solver)`). Defense in depth for the
  above: the per-node LP soundness gate now also rejects a relaxation point that
  violates the node's variable bounds, not only its constraint rows, so a
  bound-violating point can never seed a spurious integer incumbent. Regression
  covered in `python/tests/test_milp_node_bound_soundness.py`.
- **Relaxation soundness hardening** across the global-opt loop: reject a
  fabricated finite bound on an unbounded McCormick relaxation (`himmel16`,
  `fix(soundness)`); never trust an unconverged simplex objective as an LP lower
  bound (`gear4`, `fix(soundness)`); tangent-separate lifted univariate squares
  (#199); pre-reform interval bound + even-power FBBT (`rbrock` 43s→1.3s, #204);
  fold variable-free product factors and emit sound feasible-exit certificates
  (#179); reject denominator clearing that fabricates a false infeasibility;
  certify `du-opt` globally via epigraph relaxation + rank-1 Hessian (#182).
- **Bound convexity classification can no longer blow the time limit**
  (`fix(solver)`, #228).
- **Preserve integrality for discrete vars in nonlinear `.nl` export**
  (`fix(export)`, #214).
- **`from_gams` correctness on real GAMS files** (`fix(gams)`, #176): 1-D
  parameters and embedded objective variables now translate correctly.
- **Keep wrongly-omitted constraints in the AMP MILP relaxation** (`fix(amp)`, #200).
- **Corrected 9 wrong known-optima** in the benchmark set against MINLPLib
  `minlplib.solu` (`fix(benchmarks)`).

## [0.4.0] - 2026-05-17

### Added

- **AMP global MINLP solver, hardened end-to-end** (`feat(amp)`, #86, #15, #71). Adaptive Multivariate Partitioning gets the contributor build from #44 promoted to first-class status: lifted fractional powers to MILP aux variables (`d8ebffa`); piecewise secants + cover for every nonlinear term (`cc8f741`); piecewise secants for concave fractional powers (`9248fa1`); β-driven piecewise McCormick on bilinear-with-fp (`6cd81e3`); opt-in OBBT-on-relaxation (`e595a11`); cutoff-OBBT now honors `obbt_with_cutoff` and uses live `disc_state` (#71). New README section + worked tutorial at `docs/notebooks/amp_global_minlp.ipynb`.
- **Structural presolve pipeline (#53)** (`feat(presolve)`, #77). Orchestrator wiring 22 structural passes; M4+M5, M9, M10 wired into the root presolve pipeline; presolve roadmap grounded in the literature with B4/D6 prioritization (`fc268a1`, `cfe5b4f`, `22c6298`, `b23c0e7`).
- **Convexification roadmap M1–M11** (`feat(relaxation)`, #51, #75, #79). Permutation-symmetric trilinear McCormick (`70008ef`); M2/M3 relaxation arithmetics + M6 eigenvalue bound (#79); rank-1 certificate path for `x^2/y` on wide boxes (#74).
- **Examiner / KKT validator + solver-dual plumbing** (`feat(examiner)`, `feat(validation)`, #55, #65, #83). New `Model.solve(validate=True)`; `SolveResult` now carries solver duals; Examiner-style KKT validator with independent dual recovery; `minlptests` validator re-validates the primal at the returned `x`.
- **30-lesson optimization course + `discopt tutor` CLI** (`feat(course)`, #85). Full tutorial curriculum and interactive tutor CLI.
- **Deadline-aware JAX IPM** (`feat(deadline)`, #80). Wall-clock `time_limit` honored inside JAX-compiled IPM `while_loop`s.
- **Slice indexing on `IndexExpression`** (`feat(modeling)`, #61). `IndexExpression` now supports Python slice syntax.
- **Tiered Python test suite + ripopt 0.8** (`test`, #69). Fast PR-tier markers separated from full and integration tiers; ripopt bumped to 0.8.
- `discopt-dev` script splits developer commands out of the main `discopt` CLI (`a003ac3`).

### Changed

- **CI Python tests parallelized; coverage moved off PR path** (`ci`, #68, #72). `pytest-xdist` parallel execution by default; coverage job runs nightly + on push-to-main + on `coverage`-labeled PRs to keep PR turnaround fast.
- **Coverage floor temporarily lowered 85% → 70%** (`ci`, #88, tracking #87). AMP merge added ~7k statements without proportional smoke-test coverage; 85% target restored once the AMP test surface is expanded.
- `make test` now matches CI's parallel xdist invocation (`chore(test)`, #68, #84).

### Fixed

- **LOA/OA gap computation near-zero objective** (`fix(loa,oa)`, `9838fdb`). Relative gap was undefined when the objective was near zero; now uses a safe denominator.
- **Serial Ipopt B&B incumbent injection + NaN guards** (`fix(solver)`, #34, #73). `inject_incumbent` now wired into the serial Ipopt B&B path; starting points are clipped before evaluation to suppress NaNs.
- **Convexity certificate for `x^2/y` on wide boxes** (`fix(convexity)`, #42, #74). Rank-1 certificate path correctly identifies convexity over wide variable boxes.
- **LP-data extraction for vector-valued constraint bodies** (`fix(classifier)`, #67). `extract_lp_data` no longer drops vector-valued constraint bodies.
- **Latent mypy + clippy after #53** (`fix(ci)`, `32eb334`). Cleared lint failures introduced by the presolve merge.
- **Large-bound conservatism** (carried forward from `[Unreleased]`). Large-bound warnings remain conservative when nonlinear tightening can infer a smaller box but that tightened box is not applied to every solve path.

## [0.3.0] - 2026-04-24

This release skips the never-tagged `0.2.6` and folds its draft entries into `0.3.0` along with the post-`0.2.6` feature and infrastructure work.

### Added

- **`discopt.mo` -- multi-objective optimization** (`feat(mo): multi-objective optimization via scalarization`). Weighted-sum, AUGMECON2 ε-constraint, weighted-Tchebycheff, NBI, and NNC scalarizations; ideal/nadir payoff-table utilities; `ParetoFront` container; hypervolume / IGD / spread / ε-indicator quality metrics under `discopt.mo.indicators`.
- **`discopt.doe` -- model-based design of experiments**. Identifiability + estimability + profile-likelihood analysis (`feat(doe): identifiability + estimability + profile likelihood`, #48); model discrimination criteria + selection + sequential-design loop (`feat(doe): model discrimination`, #49, #50); batch / parallel experimental design (`feat(doe): batch / parallel experimental design`).
- **AMP -- Adaptive Multivariate Partitioning global MINLP solver** (`feat(amp)`, #44). Iterates MILP relaxation -> NLP subproblem -> partition refinement with the soundness guarantee `LB_k <= global_opt <= UB_k` at every iteration.
- **SUSPECT-style convexity detector** with sound certificates (#46). Structural convexity / concavity / monotonicity proofs for use by the convex NLP fast path and `discopt.mo` reformulations.
- **Claude Code skills + CLI installer** (`feat(cli): ship Claude Code skills in package + discopt install-skills`, `feat(skills): 20 discopt feature / algorithm expert agents`). 20 expert agents shipped in the package and installable into a user's `~/.claude/skills/` via `discopt install-skills`.
- **Crucible knowledge base** tracked in git (`feat(crucible): track wiki, bib, and 3 new articles in git`).
- **Zenodo metadata** and refined manuscript sections (#47).
- `RELEASE.md` -- authoritative release checklist documenting the procedure for cutting a discopt release.
- `CHANGELOG.md` -- this file, in Keep a Changelog format.
- Local `cargo-fmt` pre-commit hook so Rust formatting is enforced alongside `ruff` and `mypy`.

### Changed

- **`ripopt` workspace dependency `0.6.1` -> `0.7.0`** (via `0.6.2`; `Cargo.toml`, `Cargo.lock`). The `0.6.2` step transitively updated `rmumps` `0.1.0` -> `0.1.1`; the `0.7.0` step adapted `crates/discopt-python/src/ripopt_bindings.rs` to the new `NlpProblem` trait signatures: evaluation methods (`objective`, `gradient`, `constraints`, `jacobian_values`, `hessian_values`) now take an explicit `new_x: bool` flag and return `bool` (success / evaluation-failure), matching Ipopt's TNLP contract. Added match arms for the new `SolveStatus::Acceptable`, `SolveStatus::EvaluationError`, and `SolveStatus::UserRequestedStop` variants, surfaced as `"acceptable"` / `"evaluation_error"` / `"user_requested_stop"` on the Python side; `acceptable` maps to `SolveStatus.OPTIMAL` (KKT residuals within Ipopt's relaxed-acceptable-level tolerances).
- `_solve_continuous` (pure-continuous NLP fast path) now promotes the default `nlp_solver="ipm"` to `"ipopt"` for single-problem solves. The pure-JAX IPM's acceptable-tolerance check only covers variable-bound complementarity, so on problems with unbounded variables plus inequality constraints it could terminate at a non-KKT point and report OPTIMAL. Ipopt is more reliable for single solves; the JAX IPM remains the default for B&B subproblems.
- `differentiable_solve` and `differentiable_solve_l3` default backend changed from `"ipm"` to `"ipopt"` for the same reason.
- `solver` now routes pure-MILP problems through HiGHS MIP with a B&B fallback (`fix(solver): route MILP through HiGHS MIP with B&B fallback`).
- **DAE collocation perf** (`perf(dae): vectorize collocation and fix sparse Jacobian for NMPC warm solves`). Vectorized collocation residuals; sparse Jacobian assembly fixed so NMPC warm-start solves don't densify.
- `manuscript/discopt.tex` is no longer tracked -- it is generated from `manuscript/discopt.org`.

### Fixed

- **Jupyter Book docs build with zero warnings**. Cleaned up RST-formatting issues in module docstrings for `benchmarks/problems/gas_network_minlp.py`, `modeling/core.py`, `ro/formulations/box.py`, `solvers/qp_highs.py`, `solvers/sipopt.py`, `doe/discrimination.py`, `doe/discrimination_sequential.py`, `doe/selection.py`, `mo/indicators.py`, `mo/scalarization.py`, `mo/utils.py`, and `solvers/amp.py`; suppressed autoapi import-resolution warnings for the compiled `discopt._rust` extension; escaped `**kwargs` parameter entries to keep Sphinx from parsing the leading `**` as inline strong.
- **HiGHS LP/QP false optimality on wide bounds**: `solvers/qp_highs.py` and `solvers/lp_highs.py` now clip any bound with magnitude `>= 1e15` to `highspy.kHighsInf` before passing to HiGHS. Bounds like discopt's default `+/-9.999e19` fall just below HiGHS's internal infinity threshold (`1e20`) and caused HiGHS to return false-optimal solutions on convex QPs with unbounded variables.
- **Single-solve starting point**: `_solve_continuous` now clips the default starting point to `+/-10` (respecting actual bounds) instead of the previous `+/-100`, preventing ipopt from exploding on exp/log NLPs with one-sided large bounds.
- **Stationary-point starting point**: Fully unbounded variables (`|lb| > 1e15` and `|ub| > 1e15`) now start at `0.5` instead of the midpoint of `0`. Zero is a stationary point of periodic functions (sin, cos) and even functions generally; starting at `0.5` lets first-order NLP methods pick a descent direction and escape local maxima of the objective. Same fix applied in `_jax/differentiable.py::_safe_x0`.
- `_solve_qp_highs` and `_solve_qp_jax` now set `SolveResult.convex_fast_path = True` when solving a detected convex QP directly, matching the semantics of the convex NLP fast path.
- **Cutting planes with bilinear terms (#35)**: `_jax/cutting_planes.py::generate_rlt_cuts` no longer emits unsound inequalities when a bilinear term has no auxiliary `w_index`. The old no-auxiliary branch produced cuts purely in the original variable space that were not valid relaxations of the product (e.g. with `x, y in [0.1, 5]` it emitted `0.1*x + 0.1*y <= 0.01`, excluding every feasible point). Since `detect_bilinear_terms` always returns `w_index=None`, every RLT cut fed into `_AugmentedEvaluator` made the NLP infeasible at every B&B node, so no incumbent could be accepted on mixed convex/nonconvex MINLPs when `cutting_planes=True`. The function now returns `[]` in that case. Fixed in `383985e`.
- `fix(estimate)`: `discopt.estimate` now uses all array observations in residuals and Fisher information, instead of dropping all but the first row.
- `fix(ci)`: cleared a clippy `collapsible_match` and repaired the T24 `vmap` path after the MILP rerouting change.

## [0.2.5] and earlier

Historical releases (`v0.2.0` through `v0.2.5`) are not backfilled in this file.
For commit-level history of those releases, see:

```bash
git log v0.2.4..v0.2.5
git log v0.2.3..v0.2.4
git log v0.2.2..v0.2.3
git log v0.2.1..v0.2.2
git log v0.2.0..v0.2.1
```

Going forward, every release will have a section above with curated entries.

[Unreleased]: https://github.com/jkitchin/discopt/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/jkitchin/discopt/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/jkitchin/discopt/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/jkitchin/discopt/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/jkitchin/discopt/compare/v0.2.5...v0.3.0
