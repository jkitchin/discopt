# Changelog

All notable changes to discopt are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The release procedure that produces these entries is documented in
[`RELEASE.md`](RELEASE.md).

## [Unreleased]

### Added

### Changed

### Fixed

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

[Unreleased]: https://github.com/jkitchin/discopt/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/jkitchin/discopt/compare/v0.2.5...v0.3.0
