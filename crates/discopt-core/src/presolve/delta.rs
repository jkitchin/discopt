//! Pass-delta protocol for the presolve orchestrator (item A2 of the
//! roadmap in `crates/discopt-core/src/presolve/ROADMAP.md`).
//!
//! Each presolve pass returns a [`PresolveDelta`] summarising what it
//! did. The orchestrator (item A1) iterates passes to a fixed point,
//! using `PresolveDelta::made_progress` to decide when to stop.
//!
//! Fields are deliberately broad so that future passes (B-track bound
//! tightening, C-track aggregation, D-track structural detection) can
//! emit their results through the same contract without reshuffling the
//! API. Fields a given pass does not touch stay at their default
//! (zero / empty / `None`).
//!
//! The protocol is intentionally minimal in P1 — `StructureManifest` and
//! `VarAggregation` exist as empty placeholders so that downstream
//! consumers can already pattern-match on them, but no current pass
//! populates them.
//!
//! # Determinism
//!
//! `PresolveDelta` and its sub-types contain only ordered collections
//! (`Vec`) and POD scalars. There are no `HashMap`/`HashSet` fields, so
//! a sequence of deltas is byte-deterministic given byte-deterministic
//! pass kernels. See `tests/presolve_determinism.rs`.

use super::fbbt::Interval;

/// Manifest of structural facts a pass detected about the model.
///
/// Empty in P1. D-track passes (D1 convex reformulation, D2/D3
/// polynomial-quadratic / reduction-constraint detection, D4 symmetry,
/// D5 separability, D6 NN presolve) populate the relevant fields.
#[derive(Debug, Clone, Default)]
pub struct StructureManifest {
    /// Constraint indices detected as convex blocks (D1).
    pub convex_constraints: Vec<usize>,
    /// Constraint indices detected as polynomial of degree > 2 that
    /// have already been reformulated to bilinear (D2).
    pub reformulated_polynomial_constraints: Vec<usize>,
    /// (binary_var, value, implied_var, implied_bound) — implications
    /// discovered during probing or by structural inspection.
    pub implications: Vec<Implication>,
    /// Pairwise binary conflict edges (item F2 of the roadmap). Each
    /// edge `(i, j)` with `i < j` records that binary variable blocks
    /// `i` and `j` cannot simultaneously equal 1 under some constraint.
    /// Sorted lexicographically.
    pub cliques: Vec<(usize, usize)>,
}

/// One implication tuple. Mirrors `presolve::probing::Implication` but
/// keeps the orchestrator independent of the probing module's types.
#[derive(Debug, Clone)]
pub struct Implication {
    /// Index of the binary variable being conditioned on.
    pub binary_var: usize,
    /// The value (false=0, true=1) that triggers the implication.
    pub binary_val: bool,
    /// Index of the variable whose bounds are tightened.
    pub implied_var: usize,
    /// Implied lower bound on `implied_var`.
    pub implied_lo: f64,
    /// Implied upper bound on `implied_var`.
    pub implied_hi: f64,
}

/// One variable aggregation: `target = sum_i (coeffs[i] * sources[i]) + constant`.
///
/// Empty / unused in P1 because the C1 (variable aggregation) pass is
/// not yet implemented. Reserved so that the type does not need to
/// change once C1 lands.
#[derive(Debug, Clone)]
pub struct VarAggregation {
    /// Variable that gets eliminated in favor of the linear combination.
    pub target: usize,
    /// Source variable indices in the linear combination.
    pub sources: Vec<usize>,
    /// Coefficients aligned with `sources`.
    pub coeffs: Vec<f64>,
    /// Additive constant in the aggregation.
    pub constant: f64,
}

/// Per-pass delta. Returned by every [`PresolvePass::run`] invocation.
///
/// All counts default to zero; all collections default to empty. The
/// orchestrator treats a delta as "made progress" iff at least one
/// counter is positive *or* at least one collection is non-empty.
#[derive(Debug, Clone)]
pub struct PresolveDelta {
    /// Stable identifier for the pass that produced this delta.
    pub pass_name: &'static str,
    /// Sweep iteration (0-based) in which this pass ran. Useful for
    /// detecting per-pass convergence vs. global convergence.
    pub pass_iter: u32,

    // ─── Bound changes ────────────────────────────────────────────
    /// Number of *(lb, ub)* pairs that strictly tightened. Counts
    /// each side independently, so a pass that tightens both the
    /// lower and upper bound on one variable contributes 2.
    pub bounds_tightened: u32,
    /// Snapshot of variable bounds *after* the pass ran. `None` for
    /// passes that don't touch bounds. Used by the orchestrator to
    /// detect convergence and by tests for golden-file comparisons.
    pub var_bounds_after: Option<Vec<Interval>>,

    // ─── Variable changes ─────────────────────────────────────────
    /// `(var_index, value)` pairs the pass fixed.
    pub vars_fixed: Vec<(usize, f64)>,
    /// Aggregations introduced (empty in P1; reserved for C1).
    pub vars_aggregated: Vec<VarAggregation>,
    /// Number of auxiliary variables introduced (e.g. by polynomial
    /// reformulation). Counted, not enumerated, because the indices
    /// are simply `n_vars_old .. n_vars_new`.
    pub aux_vars_introduced: u32,

    // ─── Constraint changes ───────────────────────────────────────
    /// Indices of constraints removed (relative to the input model;
    /// after rewrite passes these indices no longer apply to the
    /// output, but they are useful for logging/diagnostics).
    pub constraints_removed: Vec<usize>,
    /// Indices of constraints rewritten (e.g. polynomial reformulation
    /// replaced the body but kept the slot).
    pub constraints_rewritten: Vec<usize>,
    /// Number of auxiliary constraints introduced.
    pub aux_constraints_introduced: u32,

    // ─── Structural manifest (empty in P1) ────────────────────────
    /// Structural facts detected by this pass.
    pub structure: StructureManifest,

    // ─── Diagnostics that DO NOT count as progress ───────────────
    /// Curtis–Reid row scale factors, one per constraint. `None` for
    /// passes that do not compute scaling. Recorded for downstream
    /// LP/NLP solvers to consume; populating this field does not
    /// trigger orchestrator iteration on its own.
    pub row_scales: Option<Vec<f64>>,
    /// Curtis–Reid column scale factors, one per variable block.
    pub col_scales: Option<Vec<f64>>,

    // ─── Accounting ───────────────────────────────────────────────
    /// Wall-clock time spent in this pass invocation (milliseconds).
    pub wall_time_ms: f64,
    /// Pass-defined work units (e.g. LP solves for OBBT, propagator
    /// invocations for FBBT). Used by the orchestrator's global budget.
    pub work_units: u64,
}

impl PresolveDelta {
    /// Construct an empty delta tagged with the given pass name and
    /// iteration index. Useful for early-exit / no-op cases.
    pub fn empty(pass_name: &'static str, iter: u32) -> Self {
        Self {
            pass_name,
            pass_iter: iter,
            bounds_tightened: 0,
            var_bounds_after: None,
            vars_fixed: Vec::new(),
            vars_aggregated: Vec::new(),
            aux_vars_introduced: 0,
            constraints_removed: Vec::new(),
            constraints_rewritten: Vec::new(),
            aux_constraints_introduced: 0,
            structure: StructureManifest::default(),
            row_scales: None,
            col_scales: None,
            wall_time_ms: 0.0,
            work_units: 0,
        }
    }

    /// Whether this delta represents observable progress. Used by the
    /// orchestrator to detect a fixed point (a full sweep with every
    /// pass returning `made_progress() == false`).
    pub fn made_progress(&self) -> bool {
        self.bounds_tightened > 0
            || !self.vars_fixed.is_empty()
            || !self.vars_aggregated.is_empty()
            || self.aux_vars_introduced > 0
            || !self.constraints_removed.is_empty()
            || !self.constraints_rewritten.is_empty()
            || self.aux_constraints_introduced > 0
            || !self
                .structure
                .reformulated_polynomial_constraints
                .is_empty()
            || !self.structure.implications.is_empty()
        // NOTE: `structure.cliques` and `structure.convex_constraints`
        // are intentionally NOT checked here. Both are diagnostic —
        // they describe the model's shape without modifying it — so
        // they must not trigger orchestrator iteration on their own
        // (otherwise the orchestrator would loop forever as long as
        // such structural findings exist).
    }
}

/// Why the orchestrator stopped iterating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminationReason {
    /// One full sweep over every pass produced no progress.
    NoProgress,
    /// `OrchestratorOptions::max_iterations` was reached.
    IterationCap,
    /// `OrchestratorOptions::time_limit_ms` was reached.
    TimeBudget,
    /// `OrchestratorOptions::work_unit_budget` was reached.
    WorkBudget,
    /// A pass detected infeasibility (e.g. FBBT empty interval).
    Infeasible,
}

/// Count strictly tightened bound endpoints between two snapshots.
///
/// Counts each endpoint (lo, hi) independently. Handy helper for pass
/// adapters that take a bounds snapshot before invoking the kernel and
/// compute the delta after.
pub fn count_tightened(before: &[Interval], after: &[Interval]) -> u32 {
    let n = before.len().min(after.len());
    let mut tight: u32 = 0;
    for i in 0..n {
        if after[i].lo > before[i].lo + 0.0 {
            tight += 1;
        }
        if after[i].hi < before[i].hi - 0.0 {
            tight += 1;
        }
    }
    tight
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_delta_does_not_report_progress() {
        let d = PresolveDelta::empty("test", 0);
        assert!(!d.made_progress());
    }

    #[test]
    fn bounds_tightened_signals_progress() {
        let mut d = PresolveDelta::empty("test", 0);
        d.bounds_tightened = 1;
        assert!(d.made_progress());
    }

    #[test]
    fn fixed_var_signals_progress() {
        let mut d = PresolveDelta::empty("test", 0);
        d.vars_fixed.push((0, 1.0));
        assert!(d.made_progress());
    }

    #[test]
    fn count_tightened_counts_each_endpoint() {
        let before = vec![Interval::new(0.0, 10.0), Interval::new(-5.0, 5.0)];
        let after = vec![Interval::new(1.0, 9.0), Interval::new(-5.0, 5.0)];
        assert_eq!(count_tightened(&before, &after), 2);
    }
}
