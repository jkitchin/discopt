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
    /// Constraint indices a pass proved redundant *without removing
    /// them* — the row is implied by the current variable bounds, but
    /// the pass that noticed is `PassCategory::BoundsOnly` and holds
    /// `&ctx.model` immutably, so it cannot drop anything.
    ///
    /// This is emphatically NOT `PresolveDelta::constraints_removed`,
    /// which means the row is gone from the model. Conflating the two
    /// is #1053: `simplify` stamped its bound-redundancy findings into
    /// `constraints_removed`, `made_progress()` read that as a model
    /// change, the model was in fact unchanged, so the next sweep
    /// re-derived the identical list and the fixed point became
    /// unreachable. Every presolve on a model with one bound-redundant
    /// row ran to the iteration cap.
    pub redundant_constraints: Vec<usize>,
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
        // NOTE: every `structure` field except
        // `reformulated_polynomial_constraints` is intentionally absent
        // from this list. They are *diagnostic*: they describe the
        // model's shape without changing it, so a pass that emits one
        // has not moved the fixed point. `reformulated_polynomial_
        // constraints` is the exception because that pass really did
        // rewrite the rows it names.
        //
        // The rule exists because breaking it is unrecoverable rather
        // than merely wasteful: a detection derived from unchanged
        // inputs is re-derived identically on the next sweep, so the
        // signal never clears, `made_progress()` is true forever, and
        // the `NoProgress` break can never fire. Presolve then always
        // runs to `max_iterations` regardless of whether anything is
        // left to do. That is #1053, and it was reached two ways at
        // once — `structure.implications` was in this list, and
        // `simplify` was stamping bound-redundancy *detections* into
        // `constraints_removed` (see `redundant_constraints` above,
        // which is where they go now).
        //
        // Measured on `hda` (722 vars, MINLPLib) at a 30 s presolve
        // budget: the reduction is complete by sweep 4 (1 var fixed,
        // 1 constraint removed, 6 bounds tightened) and unchanged
        // through sweep 15, yet the loop ran the full 16 sweeps at
        // ~1.4 s each. Removing `implications` alone did not fix it —
        // `simplify` re-reported the same 80 rows on every sweep from
        // 1 to 15 — which is why both are addressed here.
        //
        // No actionable output is lost. Probing's tightened bounds and
        // fixed variables are counted above; the detections themselves
        // still reach consumers on the delta. They just no longer buy
        // another sweep. Only a pass that changed something earns one.
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

/// Relative floor on what counts as a *tightening*, below which a moved
/// endpoint is numerical noise rather than progress (#1053).
///
/// A zero threshold makes the presolve fixed point unreachable on any model
/// whose bounds converge asymptotically: every sweep nudges some endpoint in
/// the last bits, `count_tightened` reports it, `made_progress()` is true, and
/// the loop runs until the iteration or time cap. Measured on MINLPLib `hda`
/// (782 vars) with `max_iterations` pinned: sweeps 5..14 move the returned
/// bound vector by at most **4.0e-14** in total, while each of those sweeps
/// reports 3-9 "tightenings" and buys the next one. Presolve filled its entire
/// 15 s root budget producing that 4e-14.
///
/// 1e-9 sits three orders below `fbbt::FEAS_TOL` (1e-6) — so nothing that
/// could change a feasibility decision is ever dismissed as noise — and five
/// orders above the movement actually observed.
///
/// Suppressing the *signal* does not discard the *tightening*: the bound is
/// already in `ctx.bounds` and is still returned. Only the claim "this sweep
/// made progress" is withheld, so presolve stops one sweep earlier with a
/// bound vector looser by at most this tolerance. Sound in the only direction
/// that matters — a looser presolve box never cuts off a feasible point.
pub const TIGHTEN_PROGRESS_TOL: f64 = 1e-9;

/// Did an endpoint move by more than numerical noise?
///
/// Scaled by the *smaller* of the two magnitudes, which is what makes an
/// unbounded endpoint becoming finite always count: the threshold is taken
/// from the finite side, so `-inf -> -1000` is measured against 1000 and not
/// against the infinity. Scaling by the larger magnitude would swallow the
/// single most valuable tightening presolve can make.
///
/// `Interval` here holds true `f64::INFINITY` rather than the LP layer's 1e20
/// `INF` sentinel, but declared bounds arriving from a model may carry the
/// sentinel. The `min` handles both: 1e20 -> 1e15 is a real tightening
/// (scale 1e15, movement ~1e20), while 1e20 -> 1e20 - 1e9 is not.
fn moved_meaningfully(before: f64, after: f64) -> bool {
    let scale = before.abs().min(after.abs()).max(1.0);
    (after - before).abs() > TIGHTEN_PROGRESS_TOL * scale
}

/// Count strictly tightened bound endpoints between two snapshots.
///
/// Counts each endpoint (lo, hi) independently, ignoring movement below
/// [`TIGHTEN_PROGRESS_TOL`]. Handy helper for pass adapters that take a
/// bounds snapshot before invoking the kernel and compute the delta after.
/// Every bound-tightening pass routes its `bounds_tightened` through here, so
/// this is the single place the presolve fixed point is decided.
pub fn count_tightened(before: &[Interval], after: &[Interval]) -> u32 {
    let n = before.len().min(after.len());
    let mut tight: u32 = 0;
    for i in 0..n {
        if after[i].lo > before[i].lo && moved_meaningfully(before[i].lo, after[i].lo) {
            tight += 1;
        }
        if after[i].hi < before[i].hi && moved_meaningfully(before[i].hi, after[i].hi) {
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

    /// #1053: probing re-derives an identical implication set from
    /// unchanged bounds on every sweep. Counting that as progress makes
    /// `made_progress()` permanently true and the fixed point
    /// unreachable. The actionable half of probing's output —
    /// tightened bounds, fixed variables — is counted above.
    #[test]
    fn diagnostic_structure_alone_is_not_progress() {
        let mut d = PresolveDelta::empty("test", 0);
        d.structure.implications.push(Implication {
            binary_var: 0,
            binary_val: true,
            implied_var: 1,
            implied_lo: 0.0,
            implied_hi: 1.0,
        });
        d.structure.cliques.push((0, 1));
        d.structure.convex_constraints.push(0);
        assert!(!d.made_progress());

        // ANTI-VACUITY CONTROL: the same delta *with* an actionable
        // change must still report progress, so the assertion above is
        // not passing because the predicate is stuck at false.
        d.bounds_tightened = 1;
        assert!(d.made_progress());
    }

    #[test]
    fn count_tightened_counts_each_endpoint() {
        let before = vec![Interval::new(0.0, 10.0), Interval::new(-5.0, 5.0)];
        let after = vec![Interval::new(1.0, 9.0), Interval::new(-5.0, 5.0)];
        assert_eq!(count_tightened(&before, &after), 2);
    }

    /// #1053: last-bit movement is not progress.
    ///
    /// On `hda` the bound vector moved by 4.0e-14 across ten sweeps while
    /// each of those sweeps reported 3-9 tightenings, so presolve never
    /// reached its fixed point and burned its whole time budget.
    #[test]
    fn count_tightened_ignores_numerical_noise() {
        let before = vec![Interval::new(0.0, 10.0), Interval::new(-500.0, 500.0)];
        let after = vec![
            // Absolute noise on a small endpoint.
            Interval::new(1.0e-14, 10.0 - 1.0e-14),
            // Relative noise on a large one: 500 * 1e-15 in ulps.
            Interval::new(-500.0 + 5.0e-13, 500.0 - 5.0e-13),
        ];
        assert_eq!(count_tightened(&before, &after), 0);

        // ANTI-VACUITY CONTROL: a real tightening on the same endpoints
        // must still count, so the zero above is the tolerance talking and
        // not a helper that always returns false.
        let real = vec![Interval::new(1.0, 9.0), Interval::new(-499.0, 499.0)];
        assert_eq!(count_tightened(&before, &real), 4);
    }

    /// An unbounded endpoint becoming finite is always progress, however the
    /// tolerance is scaled. A threshold scaled by the *larger* magnitude would
    /// be infinite here (or 1e11 for a 1e20 sentinel bound) and would discard
    /// the single most valuable tightening presolve can make.
    #[test]
    fn count_tightened_counts_an_infinite_bound_becoming_finite() {
        let before = vec![Interval::new(f64::NEG_INFINITY, f64::INFINITY)];
        let after = vec![Interval::new(-1000.0, 1000.0)];
        assert_eq!(count_tightened(&before, &after), 2);

        // An unchanged infinite endpoint is not progress. `inf - inf` is NaN
        // and every NaN comparison is false, so this also pins that the NaN
        // does not leak out as a phantom tightening.
        assert_eq!(count_tightened(&before, &before), 0);

        // Same two properties for a model that arrived carrying the 1e20
        // sentinel instead: a real narrowing counts, a 1e9 nudge does not.
        const SENTINEL: f64 = 1e20;
        let sent = vec![Interval::new(-SENTINEL, SENTINEL)];
        assert_eq!(count_tightened(&sent, &after), 2);
        let nudged = vec![Interval::new(-SENTINEL + 1.0e9, SENTINEL - 1.0e9)];
        assert_eq!(count_tightened(&sent, &nudged), 0);
    }

    /// The tolerance must not let a *loosening* count, and must not let a
    /// pass claim progress for a bound that did not move at all.
    #[test]
    fn count_tightened_ignores_loosening_and_no_ops() {
        let before = vec![Interval::new(0.0, 10.0)];
        let looser = vec![Interval::new(-1.0, 11.0)];
        assert_eq!(count_tightened(&before, &looser), 0);
        assert_eq!(count_tightened(&before, &before), 0);
    }
}
