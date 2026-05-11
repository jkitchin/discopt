//! Adapter shims wrapping each existing presolve kernel as a
//! [`PresolvePass`].
//!
//! The kernels live in `fbbt.rs`, `simplify.rs`, `probing.rs`,
//! `eliminate.rs`, and `polynomial.rs`. They retain their original
//! function signatures and tests; this module adds thin adapter structs
//! that conform to the orchestrator's [`PresolvePass`] trait.
//!
//! # OBBT
//!
//! `obbt.rs` is intentionally not wrapped here. OBBT requires an LP
//! solver, which lives on the Python side; integration via the A3
//! Rust↔Python handshake is deferred to phase P3 of the roadmap.
//!
//! # Determinism
//!
//! Every wrapper is a pure function of `(model, bounds)`; no RNG,
//! no unsorted-iteration sources. See `tests/presolve_determinism.rs`.

use super::aggregate::aggregate_variables;
use super::cliques::extract_cliques;
use super::coefficient_strengthening::coefficient_strengthening;
use super::delta::{count_tightened, Implication as DeltaImpl, PresolveDelta, VarAggregation};
use super::duality::{reduced_cost_fixing, ReducedCostInfo};
use super::eliminate::eliminate_variables;
use super::factorable_elim::factorable_eliminate;
use super::fbbt::{fbbt_with_cutoff, Interval};
use super::fbbt_fp::{fbbt_fixed_point, FbbtFpOptions};
use super::implied_bounds::propagate_implied_bounds;
use super::pass::{PassCategory, PresolveContext, PresolvePass};
use super::polynomial::reformulate_polynomial;
use super::probing::probe_binary_vars;
use super::reduction_constraints::detect_reduction_constraints;
use super::redundancy::detect_row_redundancy;
use super::scaling::compute_equilibration;
use super::simplify::simplify;

// ─────────────────────────────────────────────────────────────────
// FBBT
// ─────────────────────────────────────────────────────────────────

/// Adapter for `fbbt::fbbt_with_cutoff`.
#[derive(Debug, Clone)]
pub struct FbbtPass {
    /// Maximum forward/backward sweeps inside the FBBT kernel.
    pub max_iter: usize,
    /// Convergence tolerance for the FBBT inner loop.
    pub tol: f64,
    /// Optional incumbent objective bound for cutoff propagation.
    pub incumbent_bound: Option<f64>,
}

impl Default for FbbtPass {
    fn default() -> Self {
        Self {
            max_iter: 20,
            tol: 1e-8,
            incumbent_bound: None,
        }
    }
}

impl PresolvePass for FbbtPass {
    fn name(&self) -> &'static str {
        "fbbt"
    }
    fn category(&self) -> PassCategory {
        PassCategory::BoundsOnly
    }
    fn run(&mut self, ctx: &mut PresolveContext) -> PresolveDelta {
        let before = ctx.bounds.clone();
        let new_bounds =
            fbbt_with_cutoff(&ctx.model, self.max_iter, self.tol, self.incumbent_bound);
        // Intersect with the orchestrator's running bounds — the kernel
        // ignores the caller's current state and re-derives from the
        // model's declared bounds, so we have to fold in any prior
        // tightening.
        let mut after = ctx.bounds.clone();
        let n = after.len().min(new_bounds.len());
        for i in 0..n {
            after[i] = after[i].intersect(&new_bounds[i]);
        }
        ctx.bounds = after.clone();

        let mut delta = PresolveDelta::empty("fbbt", ctx.iter);
        delta.bounds_tightened = count_tightened(&before, &ctx.bounds);
        delta.var_bounds_after = Some(ctx.bounds.clone());
        delta.work_units = self.max_iter as u64;
        delta
    }
}

// ─────────────────────────────────────────────────────────────────
// Simplify
// ─────────────────────────────────────────────────────────────────

/// Adapter for `simplify::simplify`.
#[derive(Debug, Default, Clone)]
pub struct SimplifyPass;

impl PresolvePass for SimplifyPass {
    fn name(&self) -> &'static str {
        "simplify"
    }
    fn category(&self) -> PassCategory {
        PassCategory::BoundsOnly
    }
    fn run(&mut self, ctx: &mut PresolveContext) -> PresolveDelta {
        let before = ctx.bounds.clone();
        let result = simplify(&ctx.model, &mut ctx.bounds);
        let mut delta = PresolveDelta::empty("simplify", ctx.iter);
        delta.bounds_tightened = count_tightened(&before, &ctx.bounds);
        delta.var_bounds_after = Some(ctx.bounds.clone());
        delta.constraints_removed = result.redundant_constraints.clone();
        delta.work_units = (result.bigm_tightened
            + result.integer_bounds_tightened
            + result.constraints_removed) as u64;
        delta
    }
}

// ─────────────────────────────────────────────────────────────────
// Probing
// ─────────────────────────────────────────────────────────────────

/// Adapter for `probing::probe_binary_vars`.
#[derive(Debug, Default, Clone)]
pub struct ProbingPass;

impl PresolvePass for ProbingPass {
    fn name(&self) -> &'static str {
        "probing"
    }
    fn category(&self) -> PassCategory {
        PassCategory::BoundsOnly
    }
    fn run(&mut self, ctx: &mut PresolveContext) -> PresolveDelta {
        let before = ctx.bounds.clone();
        let result = probe_binary_vars(&ctx.model, &ctx.bounds);
        // probing returns full tightened-bounds vector; intersect with
        // existing.
        let n = ctx.bounds.len().min(result.tightened_bounds.len());
        for i in 0..n {
            ctx.bounds[i] = ctx.bounds[i].intersect(&result.tightened_bounds[i]);
        }
        // Also pin any fixed vars.
        for &(idx, val) in &result.fixed_vars {
            if idx < ctx.bounds.len() {
                ctx.bounds[idx] = Interval::point(val);
            }
        }

        let mut delta = PresolveDelta::empty("probing", ctx.iter);
        delta.bounds_tightened = count_tightened(&before, &ctx.bounds);
        delta.var_bounds_after = Some(ctx.bounds.clone());
        delta.vars_fixed = result.fixed_vars.clone();
        delta.structure.implications = result
            .implications
            .iter()
            .map(|imp| DeltaImpl {
                binary_var: imp.binary_var,
                binary_val: imp.binary_val,
                implied_var: imp.implied_var,
                implied_lo: imp.implied_bound.lo,
                implied_hi: imp.implied_bound.hi,
            })
            .collect();
        delta.work_units = result.implications.len() as u64;
        delta
    }
}

// ─────────────────────────────────────────────────────────────────
// Eliminate (M10 — variable elimination via singleton equality)
// ─────────────────────────────────────────────────────────────────

/// Adapter for `eliminate::eliminate_variables`.
#[derive(Debug, Default, Clone)]
pub struct EliminatePass;

impl PresolvePass for EliminatePass {
    fn name(&self) -> &'static str {
        "eliminate"
    }
    fn category(&self) -> PassCategory {
        PassCategory::RewritesModel
    }
    fn run(&mut self, ctx: &mut PresolveContext) -> PresolveDelta {
        let n_constr_before = ctx.model.constraints.len();
        let (new_model, stats) = eliminate_variables(&ctx.model);
        let n_constr_after = new_model.constraints.len();
        ctx.model = new_model;
        // resync_bounds_after_rewrite is called by the orchestrator.

        let mut delta = PresolveDelta::empty("eliminate", ctx.iter);
        // We don't have per-index removal lists from the kernel; record
        // the count using a synthetic "removed" range. For diagnostics
        // only — downstream code uses counts, not specific indices.
        if stats.constraints_removed > 0 {
            delta.constraints_removed = (n_constr_after..n_constr_before).collect::<Vec<usize>>();
        }
        // vars_fixed: the kernel pins lb=ub but doesn't list which.
        // We don't reconstruct it here; the count is implied by
        // stats.variables_fixed. Leave list empty to avoid lying.
        delta.work_units = stats.candidates_examined as u64;
        // Use the count directly — the empty `vars_fixed` would not
        // signal progress otherwise.
        if stats.variables_fixed > 0 {
            delta.bounds_tightened = (stats.variables_fixed * 2) as u32;
        }
        delta
    }
}

// ─────────────────────────────────────────────────────────────────
// Aggregate (C1 — variable aggregation / affine substitution)
// ─────────────────────────────────────────────────────────────────

/// Adapter for `aggregate::aggregate_variables` (C1 of the roadmap).
///
/// Removes a continuous scalar variable when it appears in exactly one
/// constraint — an equality of the form `c_x · x + c_y · y + c0 == rhs` —
/// and nowhere else. The variable's block is dropped, the equality is
/// dropped, and the surviving variable's bounds are tightened from the
/// implied interval. Each successful aggregation is recorded in
/// `delta.vars_aggregated` for downstream post-solve recovery.
#[derive(Debug, Default, Clone)]
pub struct AggregatePass;

impl PresolvePass for AggregatePass {
    fn name(&self) -> &'static str {
        "aggregate"
    }
    fn category(&self) -> PassCategory {
        PassCategory::RewritesModel
    }
    fn run(&mut self, ctx: &mut PresolveContext) -> PresolveDelta {
        let n_constr_before = ctx.model.constraints.len();
        let n_vars_before = ctx.model.variables.len();
        let (new_model, stats) = aggregate_variables(&ctx.model);
        let n_constr_after = new_model.constraints.len();
        let n_vars_after = new_model.variables.len();
        ctx.model = new_model;

        let mut delta = PresolveDelta::empty("aggregate", ctx.iter);
        delta.work_units = stats.candidates_examined as u64;
        if stats.variables_aggregated > 0 {
            delta.constraints_removed = (n_constr_after..n_constr_before).collect::<Vec<usize>>();
            // Use bounds_tightened to expose progress to the
            // orchestrator's `made_progress()` check, since dropping a
            // variable block also tightens (in fact, replaces) bounds
            // for the surviving variable.
            delta.bounds_tightened = (n_vars_before.saturating_sub(n_vars_after)) as u32;
            for r in &stats.aggregations {
                delta.vars_aggregated.push(VarAggregation {
                    target: r.eliminated_block,
                    sources: vec![r.source_block],
                    coeffs: vec![r.coeff],
                    constant: r.offset,
                });
            }
        }
        delta
    }
}

// ─────────────────────────────────────────────────────────────────
// Redundancy (C3 — parallel-row / duplicate constraint detection)
// ─────────────────────────────────────────────────────────────────

/// Adapter for `redundancy::detect_row_redundancy` (C3 of the roadmap).
///
/// Drops constraints whose linear part is a positive scalar multiple
/// of another constraint with a looser RHS, or duplicates of an
/// equality. Pure structural redundancy: independent of variable
/// bounds (the bound-driven case is `simplify`'s job).
#[derive(Debug, Default, Clone)]
pub struct RedundancyPass;

impl PresolvePass for RedundancyPass {
    fn name(&self) -> &'static str {
        "redundancy"
    }
    fn category(&self) -> PassCategory {
        PassCategory::RewritesModel
    }
    fn run(&mut self, ctx: &mut PresolveContext) -> PresolveDelta {
        let (new_model, stats) = detect_row_redundancy(&ctx.model);
        ctx.model = new_model;

        let mut delta = PresolveDelta::empty("redundancy", ctx.iter);
        delta.constraints_removed = stats.removed_indices.clone();
        delta.work_units = stats.pairs_examined as u64;
        delta
    }
}

// ─────────────────────────────────────────────────────────────────
// Equilibration scaling (E1 — Curtis–Reid row/col scale factors)
// ─────────────────────────────────────────────────────────────────

/// Adapter for `scaling::compute_equilibration` (E1 of the roadmap).
///
/// Computes Curtis–Reid geometric-mean scale factors for the linear
/// part of the model and exposes them on the delta (`row_scales`,
/// `col_scales`). The pass does not modify the model or bounds — it
/// only emits the numbers, so its delta does NOT count as progress
/// for the orchestrator's fixed-point check. Downstream LP/NLP
/// solvers consume the factors from the delta log.
#[derive(Debug, Default, Clone)]
pub struct ScalingPass;

impl PresolvePass for ScalingPass {
    fn name(&self) -> &'static str {
        "scaling"
    }
    fn category(&self) -> PassCategory {
        PassCategory::BoundsOnly
    }
    fn run(&mut self, ctx: &mut PresolveContext) -> PresolveDelta {
        let (factors, stats) = compute_equilibration(&ctx.model);
        let mut delta = PresolveDelta::empty("scaling", ctx.iter);
        delta.row_scales = Some(factors.row_scales);
        delta.col_scales = Some(factors.col_scales);
        delta.work_units = stats.linear_rows_sampled as u64;
        delta
    }
}

// ─────────────────────────────────────────────────────────────────
// Watch-list FBBT (B4 — fixed-point bound propagation)
// ─────────────────────────────────────────────────────────────────

/// Adapter for `fbbt_fp::fbbt_fixed_point` (B4 of the roadmap).
///
/// Runs forward/backward bound propagation as a constraint work-queue,
/// re-queuing only constraints that share a freshly-tightened
/// variable. Terminates at the true fixed point on monotone-DAG
/// instances and on the linear subset; the outer cap covers
/// pathological non-monotone cycles.
#[derive(Debug, Clone)]
pub struct FbbtFixedPointPass {
    /// Convergence tolerance — bound changes below this don't requeue.
    pub tol: f64,
    /// Hard cap on constraint visits (0 ⇒ derive from model size).
    pub max_visits: usize,
}

impl Default for FbbtFixedPointPass {
    fn default() -> Self {
        Self {
            tol: 1e-9,
            max_visits: 0,
        }
    }
}

impl PresolvePass for FbbtFixedPointPass {
    fn name(&self) -> &'static str {
        "fbbt_fixed_point"
    }
    fn category(&self) -> PassCategory {
        PassCategory::BoundsOnly
    }
    fn run(&mut self, ctx: &mut PresolveContext) -> PresolveDelta {
        let before = ctx.bounds.clone();
        let stats = fbbt_fixed_point(
            &ctx.model,
            &mut ctx.bounds,
            FbbtFpOptions {
                tol: self.tol,
                max_visits: self.max_visits,
            },
        );

        let mut delta = PresolveDelta::empty("fbbt_fixed_point", ctx.iter);
        delta.bounds_tightened = count_tightened(&before, &ctx.bounds);
        delta.var_bounds_after = Some(ctx.bounds.clone());
        delta.work_units = stats.constraint_visits as u64;
        delta
    }
}

// ─────────────────────────────────────────────────────────────────
// Implied-bound propagation (B1 — linear-row activity tightening)
// ─────────────────────────────────────────────────────────────────

/// Adapter for `implied_bounds::propagate_implied_bounds` (B1 of the
/// roadmap).
///
/// Sweeps every linear constraint once and tightens `ctx.bounds` from
/// the row activity. Complements `FbbtPass` by being a pure linear
/// pass with no DAG traversal — typically faster on the linear
/// subset of a model.
#[derive(Debug, Default, Clone)]
pub struct ImpliedBoundsPass;

impl PresolvePass for ImpliedBoundsPass {
    fn name(&self) -> &'static str {
        "implied_bounds"
    }
    fn category(&self) -> PassCategory {
        PassCategory::BoundsOnly
    }
    fn run(&mut self, ctx: &mut PresolveContext) -> PresolveDelta {
        let before = ctx.bounds.clone();
        let stats = propagate_implied_bounds(&ctx.model, &mut ctx.bounds);

        let mut delta = PresolveDelta::empty("implied_bounds", ctx.iter);
        delta.bounds_tightened = count_tightened(&before, &ctx.bounds);
        delta.var_bounds_after = Some(ctx.bounds.clone());
        delta.work_units = stats.linear_rows_examined as u64;
        delta
    }
}

// ─────────────────────────────────────────────────────────────────
// Implied-clique extraction (F2 — pairwise binary conflicts)
// ─────────────────────────────────────────────────────────────────

/// Adapter for `cliques::extract_cliques` (F2 of the roadmap).
///
/// Detects pairs of binary variables that cannot simultaneously be 1
/// under some linear constraint. Edges land in
/// `delta.structure.cliques`. Diagnostic only — never modifies the
/// model or bounds, so the delta does NOT count as progress for the
/// orchestrator's fixed-point check (otherwise the orchestrator would
/// loop forever as long as conflict edges exist).
#[derive(Debug, Default, Clone)]
pub struct CliquePass;

impl PresolvePass for CliquePass {
    fn name(&self) -> &'static str {
        "cliques"
    }
    fn category(&self) -> PassCategory {
        PassCategory::BoundsOnly
    }
    fn run(&mut self, ctx: &mut PresolveContext) -> PresolveDelta {
        let (set, stats) = extract_cliques(&ctx.model);
        let mut delta = PresolveDelta::empty("cliques", ctx.iter);
        delta.structure.cliques = set.edges;
        delta.work_units = stats.linear_rows_scanned as u64;
        delta
    }
}

// ─────────────────────────────────────────────────────────────────
// Reduced-cost fixing (E2 — domain reduction via LP duality)
// ─────────────────────────────────────────────────────────────────

/// Adapter for `duality::reduced_cost_fixing` (E2 of the roadmap).
///
/// Holds the LP-duality information (`lp_value`, `cutoff`,
/// `reduced_costs`) needed to apply reduced-cost fixing. The orchestrator
/// runs this pass after the relaxation has been solved upstream; the
/// pass adapter is constructed with the LP info already populated.
///
/// When `info` is `None` the pass is a no-op — the LP info has not been
/// supplied. This lets the pass be present in a default pipeline before
/// the A3 Rust↔Python handshake (P3) is wired through.
#[derive(Debug, Clone, Default)]
pub struct ReducedCostFixingPass {
    /// LP / NLP relaxation info. `None` ⇒ no-op.
    pub info: Option<ReducedCostInfo>,
}

impl PresolvePass for ReducedCostFixingPass {
    fn name(&self) -> &'static str {
        "reduced_cost_fixing"
    }
    fn category(&self) -> PassCategory {
        PassCategory::BoundsOnly
    }
    fn run(&mut self, ctx: &mut PresolveContext) -> PresolveDelta {
        let mut delta = PresolveDelta::empty("reduced_cost_fixing", ctx.iter);
        let info = match &self.info {
            Some(i) => i,
            None => return delta,
        };
        let before = ctx.bounds.clone();
        let stats = reduced_cost_fixing(&ctx.model, &mut ctx.bounds, info);
        delta.bounds_tightened = count_tightened(&before, &ctx.bounds);
        delta.var_bounds_after = Some(ctx.bounds.clone());
        delta.vars_fixed = stats.vars_fixed;
        delta.work_units = stats.blocks_examined as u64;
        delta
    }
}

// ─────────────────────────────────────────────────────────────────
// Polynomial reformulation (M4 + M5)
// ─────────────────────────────────────────────────────────────────

/// Adapter for `polynomial::reformulate_polynomial`.
#[derive(Debug, Default, Clone)]
pub struct PolynomialReformPass;

impl PresolvePass for PolynomialReformPass {
    fn name(&self) -> &'static str {
        "polynomial_reform"
    }
    fn category(&self) -> PassCategory {
        PassCategory::RewritesModel
    }
    fn run(&mut self, ctx: &mut PresolveContext) -> PresolveDelta {
        let n_constr_before = ctx.model.constraints.len();
        let (new_model, stats) = reformulate_polynomial(&ctx.model);
        let n_constr_after = new_model.constraints.len();
        ctx.model = new_model;

        let mut delta = PresolveDelta::empty("polynomial_reform", ctx.iter);
        delta.aux_vars_introduced = stats.aux_variables_introduced as u32;
        delta.aux_constraints_introduced = (n_constr_after.saturating_sub(n_constr_before)) as u32;
        // Constraints "rewritten" — we know the count but not specific
        // indices. Leaving the index list empty; the count is implied
        // by stats.constraints_rewritten and made_progress() picks up
        // on aux_vars_introduced or aux_constraints_introduced.
        if stats.constraints_rewritten > 0 && delta.aux_constraints_introduced == 0 {
            // Edge case: rewritten in place without adding aux constraints.
            // Mark progress via a single-entry vec to satisfy made_progress().
            delta.constraints_rewritten.push(0);
        }
        delta.work_units = (stats.constraints_rewritten + stats.constraints_skipped) as u64;
        delta.structure.reformulated_polynomial_constraints =
            (0..stats.constraints_rewritten).collect();
        delta
    }
}

// ─────────────────────────────────────────────────────────────────
// Reduction-constraint detection (D3 of #53)
// ─────────────────────────────────────────────────────────────────

/// Adapter for `reduction_constraints::detect_reduction_constraints`.
///
/// Detects sum-of-squares (or sum-of-even-powers) inequalities of the
/// form `Σ cᵢ · ∏ xⱼ^(2 kⱼ) ≤ r` with `r ≤ 0`, and tightens every
/// participating variable's bounds to `[0, 0]`. Marks the constraint
/// as redundant and stamps the indices into
/// `delta.constraints_removed` so the redundancy pass can drop them on
/// the next sweep. Surfaces structural infeasibility immediately
/// (e.g. `x² ≤ -1`).
#[derive(Debug, Default, Clone)]
pub struct ReductionConstraintsPass;

impl PresolvePass for ReductionConstraintsPass {
    fn name(&self) -> &'static str {
        "reduction_constraints"
    }
    fn category(&self) -> PassCategory {
        PassCategory::BoundsOnly
    }
    fn run(&mut self, ctx: &mut PresolveContext) -> PresolveDelta {
        let before = ctx.bounds.clone();
        let stats = detect_reduction_constraints(&ctx.model, &mut ctx.bounds);
        let mut delta = PresolveDelta::empty("reduction_constraints", ctx.iter);
        delta.bounds_tightened = count_tightened(&before, &ctx.bounds);
        delta.var_bounds_after = Some(ctx.bounds.clone());
        delta.vars_fixed = stats
            .vars_fixed_to_zero
            .into_iter()
            .map(|b| (b, 0.0))
            .collect();
        delta.constraints_removed = stats.constraints_made_redundant;
        delta.work_units = stats.constraints_examined as u64;
        // Infeasibility surfaces through the bounds array (the kernel
        // writes an empty Interval); the orchestrator's standard
        // is_empty() check terminates the loop with TerminationReason::
        // Infeasible.
        delta
    }
}

// ─────────────────────────────────────────────────────────────────
// Coefficient strengthening (C4 — Savelsbergh row strengthening)
// ─────────────────────────────────────────────────────────────────

/// Adapter for `coefficient_strengthening::coefficient_strengthening`
/// (C4 of the roadmap).
///
/// Tightens the coefficient of every binary variable whose
/// LP-slack-when-fixed-to-1 is smaller than its current coefficient.
/// Bound-preserving on integer corners; tightens the LP relaxation.
#[derive(Debug, Default, Clone)]
pub struct CoefficientStrengtheningPass;

impl PresolvePass for CoefficientStrengtheningPass {
    fn name(&self) -> &'static str {
        "coefficient_strengthening"
    }
    fn category(&self) -> PassCategory {
        PassCategory::RewritesModel
    }
    fn run(&mut self, ctx: &mut PresolveContext) -> PresolveDelta {
        let (new_model, stats) = coefficient_strengthening(&ctx.model);
        ctx.model = new_model;

        let mut delta = PresolveDelta::empty("coefficient_strengthening", ctx.iter);
        delta.constraints_rewritten = stats.rewritten_indices.clone();
        delta.work_units = stats.rows_examined as u64;
        delta
    }
}

// ─────────────────────────────────────────────────────────────────
// Factorable-expression variable elimination (C2 of issue #51)
// ─────────────────────────────────────────────────────────────────

/// Adapter for `factorable_elim::factorable_eliminate` (C2 of the
/// roadmap).
///
/// Drops equality constraints that uniquely determine a continuous
/// scalar variable that appears nowhere else in the model. Generalises
/// M10 (`EliminatePass`) to factorable right-hand sides — the
/// determining equation may contain arbitrary nonlinear terms in
/// *other* variables, so long as the candidate variable itself enters
/// only as a single linear monomial.
#[derive(Debug, Default, Clone)]
pub struct FactorableElimPass;

impl PresolvePass for FactorableElimPass {
    fn name(&self) -> &'static str {
        "factorable_elim"
    }
    fn category(&self) -> PassCategory {
        PassCategory::RewritesModel
    }
    fn run(&mut self, ctx: &mut PresolveContext) -> PresolveDelta {
        let (new_model, stats) = factorable_eliminate(&ctx.model);
        ctx.model = new_model;

        let mut delta = PresolveDelta::empty("factorable_elim", ctx.iter);
        delta.constraints_removed = stats.removed_constraint_indices.clone();
        delta.work_units = stats.candidates_examined as u64;
        delta
    }
}

// ─────────────────────────────────────────────────────────────────
// Test-only helper: a pass that always reports progress.
// Used by the orchestrator unit tests to drive iteration-cap behaviour.
// ─────────────────────────────────────────────────────────────────

/// Test-only no-op pass that always reports progress, used to drive
/// orchestrator iteration-cap tests without depending on a real kernel.
#[cfg(test)]
#[derive(Debug, Default, Clone)]
pub struct AlwaysProgressPass;

#[cfg(test)]
impl PresolvePass for AlwaysProgressPass {
    fn name(&self) -> &'static str {
        "always_progress"
    }
    fn category(&self) -> PassCategory {
        PassCategory::BoundsOnly
    }
    fn run(&mut self, ctx: &mut PresolveContext) -> PresolveDelta {
        let mut delta = PresolveDelta::empty("always_progress", ctx.iter);
        // Synthesise progress without actually changing anything that
        // would affect downstream passes.
        delta.bounds_tightened = 1;
        delta
    }
}
