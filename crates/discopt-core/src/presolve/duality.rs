//! Domain reduction via LP duality (item E2 of the presolve roadmap).
//!
//! ## What this pass does
//!
//! Implements the classical *reduced-cost fixing* (Land & Powell 1979,
//! Khajavirad & Sahinidis 2018) rule. Given:
//!
//! - `lp_value`   — the optimum objective value of the continuous LP /
//!   convex-NLP relaxation at the current node (a valid lower bound for
//!   minimization);
//! - `cutoff`     — an upper bound on the optimal objective coming from
//!   any feasible primal point (the incumbent);
//! - `reduced_costs[j]` — the LP reduced cost (a.k.a. dual price for
//!   simple bounds) of variable block `j` at the LP optimum.
//!
//! the pass tightens variable bounds by the inequality
//!
//! ```text
//!     lp_value + c̄_j · (x_j − x*_j) ≤ cutoff
//! ```
//!
//! where `x*_j` is implicitly the bound at which `c̄_j` was sampled. In
//! the standard interpretation:
//!
//! - If `c̄_j > 0` then any feasible solution improving the cutoff must
//!   satisfy `x_j ≤ lb_j + (cutoff − lp_value) / c̄_j`.
//! - If `c̄_j < 0` then any feasible solution improving the cutoff must
//!   satisfy `x_j ≥ ub_j + (cutoff − lp_value) / c̄_j`.
//!
//! Integer variables additionally floor / ceil the new endpoint.
//!
//! ## Scope (v0)
//!
//! - Only scalar variable blocks (`size == 1`) are tightened. Tensor
//!   blocks are skipped because the LP duals exposed at the Python
//!   boundary are per-block scalars; per-element duals are an A3
//!   handshake feature.
//! - The model is never rewritten: the pass produces a `ReducedCostStats`
//!   with the bound deltas; the pass adapter applies them via the
//!   shared `bounds: &mut [Interval]` slice.
//!
//! ## Determinism
//!
//! Iteration is over `model.variables` in declaration order. No
//! `HashMap`/`HashSet` reads on the hot path.
//!
//! ## Floating-point discipline (#1409 Finding 2)
//!
//! `c̄_j` is a cancelling difference `c_j − A_jᵀ y`, so `|c̄_j|` says nothing about
//! its own accuracy: the round-off is governed by `S_j = |c_j| + Σ_i |a_ij y_i|`.
//! Dividing the gap by an **over**-stated `|c̄_j|` shrinks the endpoint this pass
//! writes and can exclude an improving point. Two things follow, and both are now
//! enforced above:
//!
//! 1. The gap is widened outward by `1e-6 · (1 + |cutoff|)`, matching
//!    [`crate::bnb::milp_driver::reduced_cost_fix`]. This pass used the bare
//!    difference, which is what made it strictly weaker than the live path. Note the
//!    consequence recorded in `zero_gap_no_longer_pinches_an_ordinary_column`: a zero
//!    raw gap no longer collapses a variable to a point, which is the correct answer.
//! 2. The divisor is deflated toward zero by
//!    [`ReducedCostInfo::reduced_cost_errors`] before the division, and a block whose
//!    sign does not survive that deflation is skipped.
//!
//! **The issue's own recommendation was falsified and is not what was implemented.**
//! #1409 proposed that this pass and `bnb::milp_driver`'s "share one helper" that
//! returns the dot product alongside its magnitude sum. They cannot: this pass
//! receives `reduced_costs` as bare per-block scalars with no column, no constraint
//! matrix and no dual vector, so it has no way to compute `S_j` at all — only the
//! *producer* of `c̄_j` does. What the two paths share is therefore the bound
//! *formula* in [`crate::numeric`], not a dot-product helper; here the bound is a
//! required input, and its absence is a loud refusal
//! ([`ReducedCostStats::refused_unbounded_error`]) rather than a constant stand-in,
//! because any constant would reintroduce exactly the scale-blind absolute tolerance
//! #1397 set out to remove.

use crate::expr::{ModelRepr, VarType};

use super::fbbt::Interval;

/// LP-duality information needed to apply reduced-cost fixing.
///
/// `reduced_costs` is indexed by variable *block*, not by flat scalar
/// index, and must be the same length as `model.variables`.
#[derive(Debug, Clone)]
pub struct ReducedCostInfo {
    /// LP / NLP relaxation optimum at the root.
    pub lp_value: f64,
    /// Cutoff (incumbent objective bound). For minimization, the best
    /// known feasible objective; for maximization, take the negative.
    pub cutoff: f64,
    /// Reduced cost per variable block. Length must equal
    /// `model.variables.len()` or the pass is a no-op.
    pub reduced_costs: Vec<f64>,
    /// An **absolute upper bound on the error** of each entry of `reduced_costs`,
    /// i.e. `|reduced_costs[j] − c̄_j_true| ≤ reduced_cost_errors[j]`.
    ///
    /// This is required, not optional: `c̄_j` is a cancelling difference
    /// `c_j − A_jᵀy`, so its round-off is governed by
    /// `S_j = |c_j| + Σ_i |a_ij y_i|` and **not** by `|c̄_j|`. Dividing the gap by an
    /// over-stated `|c̄_j|` shrinks the bound this pass writes and can exclude an
    /// improving point (#1409). This pass is handed `c̄_j` as a bare scalar with no
    /// column, no matrix and no duals, so — unlike
    /// [`crate::bnb::milp_driver`]'s reduced-cost fixing, which computes the dot
    /// itself and can bound it on the spot — it **cannot derive this number**. Only
    /// the producer of `c̄_j` can.
    ///
    /// The producer computes it with [`crate::numeric::gamma`]:
    /// `reduced_cost_errors[j] = gamma(nnz_j + 2) * S_j`.
    ///
    /// When this is empty or the wrong length the pass tightens **nothing** and sets
    /// [`ReducedCostStats::refused_unbounded_error`]. That is a deliberate loud
    /// refusal rather than a cheap default: any constant stand-in here would be
    /// exactly the scale-blind absolute tolerance #1397 set out to remove
    /// (CLAUDE.md §3).
    pub reduced_cost_errors: Vec<f64>,
}

/// Per-pass diagnostics for reduced-cost fixing.
#[derive(Debug, Clone, Default)]
pub struct ReducedCostStats {
    /// Number of bound endpoints strictly tightened.
    pub bounds_tightened: u32,
    /// `(block_index, value)` pairs the pass collapsed to a point.
    pub vars_fixed: Vec<(usize, f64)>,
    /// Number of variable blocks examined (= number of scalar blocks).
    pub blocks_examined: usize,
    /// `true` iff the gap `cutoff − lp_value` is negative — the LP
    /// already proves the cutoff infeasible, so the search node can be
    /// pruned. The pass does not modify bounds in that case.
    pub infeasible: bool,
    /// `true` iff the pass refused to tighten anything because
    /// [`ReducedCostInfo::reduced_cost_errors`] was missing or the wrong length, so
    /// no reduced cost could be bounded (#1409). Distinct from "nothing to tighten":
    /// a caller seeing this has supplied an incomplete certificate, not a model
    /// without reductions.
    pub refused_unbounded_error: bool,
}

/// Apply reduced-cost fixing to `bounds` in place.
///
/// Pure function. Returns the diagnostic struct; never panics.
pub fn reduced_cost_fixing(
    model: &ModelRepr,
    bounds: &mut [Interval],
    info: &ReducedCostInfo,
) -> ReducedCostStats {
    let mut stats = ReducedCostStats::default();
    if info.reduced_costs.len() != model.variables.len() {
        return stats;
    }
    // #1409: without a per-block error bound no reduced cost here can be bounded, so
    // no tightening can be justified. Refuse loudly rather than substitute a constant.
    if info.reduced_cost_errors.len() != info.reduced_costs.len() {
        stats.refused_unbounded_error = true;
        return stats;
    }
    let gap = info.cutoff - info.lp_value;
    if gap < -1e-9 {
        stats.infeasible = true;
        return stats;
    }
    // Widen the gap outward by the same relative slack the live path in
    // `bnb::milp_driver::reduced_cost_fix` carries, so round-off in `cutoff` and
    // `lp_value` can only *loosen* the fixing. This pass previously used the bare
    // difference, which made it strictly weaker than the live path (#1409 Finding 2).
    // Note this slack covers the GAP only — the divisor's own error is a different
    // currency and is handled per block below.
    let gap = gap.max(0.0) + 1e-6 * (1.0 + info.cutoff.abs());
    if !gap.is_finite() {
        return stats;
    }

    for (block_idx, var) in model.variables.iter().enumerate() {
        if var.size != 1 {
            continue;
        }
        if block_idx >= bounds.len() {
            continue;
        }
        stats.blocks_examined += 1;
        let cbar = info.reduced_costs[block_idx];
        if !cbar.is_finite() {
            continue;
        }
        // Deflate the divisor toward zero by its own error bound before dividing:
        // over-stating `|c̄_j|` shrinks the endpoint and can exclude an improving
        // point. `None` means `c̄_j`'s sign is not resolved at its own scale, so no
        // tightening from this block is justified (#1409).
        let err = info.reduced_cost_errors[block_idx];
        if !(err >= 0.0) {
            // NaN or negative: not a bound. Refuse this block rather than trust it.
            continue;
        }
        let cbar_mag = cbar.abs() - err;
        if cbar_mag <= 0.0 {
            continue;
        }
        let cur = bounds[block_idx];
        let lb = cur.lo;
        let ub = cur.hi;
        if lb > ub {
            continue;
        }
        let is_integer = matches!(var.var_type, VarType::Binary | VarType::Integer);

        let mut new_lb = lb;
        let mut new_ub = ub;

        // The deadband is now on the *error-resolved* magnitude, not on the raw
        // `c̄_j`: `cbar_mag > 0` already means the sign is certain, which is exactly
        // what the old absolute `1e-12` was standing in for and failing to establish
        // at scale. Per #1397's non-goal, the yardstick changed, not a tolerance.
        if cbar > 0.0 && lb.is_finite() {
            // ub_new = lb + gap / |c̄_j|, with the divisor deflated toward zero
            let candidate = lb + gap / cbar_mag;
            let candidate = if is_integer {
                candidate.floor()
            } else {
                candidate
            };
            if candidate < new_ub {
                new_ub = candidate;
            }
        }
        if cbar < 0.0 && ub.is_finite() {
            // lb_new = ub − gap / |c̄_j|
            let candidate = ub - gap / cbar_mag;
            let candidate = if is_integer {
                candidate.ceil()
            } else {
                candidate
            };
            if candidate > new_lb {
                new_lb = candidate;
            }
        }

        if new_lb > new_ub {
            // Reduced-cost fixing proved this variable empty under the
            // cutoff: equivalent to infeasibility for the relaxation.
            stats.infeasible = true;
            return stats;
        }
        if new_lb > lb {
            stats.bounds_tightened += 1;
            bounds[block_idx].lo = new_lb;
        }
        if new_ub < ub {
            stats.bounds_tightened += 1;
            bounds[block_idx].hi = new_ub;
        }
        if (bounds[block_idx].hi - bounds[block_idx].lo).abs() < 1e-12
            && (new_lb > lb || new_ub < ub)
        {
            stats.vars_fixed.push((
                block_idx,
                0.5 * (bounds[block_idx].lo + bounds[block_idx].hi),
            ));
        }
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{
        ConstraintRepr, ConstraintSense, ExprArena, ExprNode, ModelRepr, ObjectiveSense, VarInfo,
        VarType,
    };

    fn cont(name: &str, lo: f64, hi: f64) -> VarInfo {
        VarInfo {
            name: name.into(),
            var_type: VarType::Continuous,
            offset: 0,
            size: 1,
            shape: vec![],
            lb: vec![lo],
            ub: vec![hi],
        }
    }

    fn int_(name: &str, lo: f64, hi: f64) -> VarInfo {
        VarInfo {
            name: name.into(),
            var_type: VarType::Integer,
            offset: 0,
            size: 1,
            shape: vec![],
            lb: vec![lo],
            ub: vec![hi],
        }
    }

    fn trivial_model(vars: Vec<VarInfo>) -> ModelRepr {
        let mut arena = ExprArena::new();
        let zero = arena.add(ExprNode::Constant(0.0));
        let n = vars.len();
        ModelRepr {
            arena,
            objective: zero,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: zero,
                sense: ConstraintSense::Le,
                rhs: 0.0,
                name: None,
            }],
            variables: vars,
            n_vars: n,
        }
    }

    /// The gap slack this pass now carries (#1409), for tests that pin an endpoint.
    /// Mirrors `bnb::milp_driver::reduced_cost_fix`: relative to the cutoff, so the
    /// endpoint lands *outside* the exact-arithmetic one by `slack / |c̄_j|`.
    fn gap_slack(cutoff: f64) -> f64 {
        1e-6 * (1.0 + cutoff.abs())
    }

    #[test]
    fn positive_reduced_cost_tightens_upper_bound() {
        // gap = 10, cbar = 2, lb = 0 ⇒ new_ub = 0 + 10/2 = 5 in exact arithmetic.
        // The pass deliberately writes the endpoint *outward* of that by
        // `gap_slack(cutoff) / cbar` (#1409), so 5 is a floor, not an equality.
        let model = trivial_model(vec![cont("x", 0.0, 100.0)]);
        let mut bounds = vec![Interval::new(0.0, 100.0)];
        let info = ReducedCostInfo {
            lp_value: 0.0,
            cutoff: 10.0,
            // A literal reduced cost, so its error really is zero. A producer
            // computing `c_j − A_jᵀy` must supply `gamma(nnz+2) * S_j` here.
            reduced_costs: vec![2.0],
            reduced_cost_errors: vec![0.0],
        };
        let s = reduced_cost_fixing(&model, &mut bounds, &info);
        assert_eq!(s.bounds_tightened, 1);
        assert!(
            bounds[0].hi >= 5.0,
            "the endpoint must never fall inside the exact-arithmetic 5.0: {}",
            bounds[0].hi
        );
        assert!(bounds[0].hi <= 5.0 + gap_slack(10.0) / 2.0 + 1e-12);
        assert!((bounds[0].lo - 0.0).abs() < 1e-9);
        assert!(!s.infeasible);
    }

    #[test]
    fn negative_reduced_cost_tightens_lower_bound() {
        // gap = 6, cbar = -3, ub = 10 ⇒ new_lb = 10 + 6 / (-3) = 8 exactly; the slack
        // moves the endpoint *down* (outward) from 8 (#1409).
        let model = trivial_model(vec![cont("x", 0.0, 10.0)]);
        let mut bounds = vec![Interval::new(0.0, 10.0)];
        let info = ReducedCostInfo {
            lp_value: 0.0,
            cutoff: 6.0,
            reduced_costs: vec![-3.0],
            reduced_cost_errors: vec![0.0],
        };
        let s = reduced_cost_fixing(&model, &mut bounds, &info);
        assert_eq!(s.bounds_tightened, 1);
        assert!(
            bounds[0].lo <= 8.0,
            "the endpoint must never rise above the exact-arithmetic 8.0: {}",
            bounds[0].lo
        );
        assert!(bounds[0].lo >= 8.0 - gap_slack(6.0) / 3.0 - 1e-12);
        assert!((bounds[0].hi - 10.0).abs() < 1e-9);
    }

    #[test]
    fn zero_reduced_cost_no_change() {
        let model = trivial_model(vec![cont("x", 0.0, 10.0)]);
        let mut bounds = vec![Interval::new(0.0, 10.0)];
        let info = ReducedCostInfo {
            lp_value: 0.0,
            cutoff: 5.0,
            reduced_costs: vec![0.0],
            reduced_cost_errors: vec![0.0],
        };
        let s = reduced_cost_fixing(&model, &mut bounds, &info);
        assert_eq!(s.bounds_tightened, 0);
    }

    #[test]
    fn integer_floors_upper_bound() {
        // gap = 10, cbar = 3, lb = 0 ⇒ raw = 3.333; integer ⇒ 3.
        let model = trivial_model(vec![int_("z", 0.0, 100.0)]);
        let mut bounds = vec![Interval::new(0.0, 100.0)];
        let info = ReducedCostInfo {
            lp_value: 0.0,
            cutoff: 10.0,
            reduced_costs: vec![3.0],
            reduced_cost_errors: vec![0.0],
        };
        let s = reduced_cost_fixing(&model, &mut bounds, &info);
        assert_eq!(s.bounds_tightened, 1);
        assert!((bounds[0].hi - 3.0).abs() < 1e-12);
    }

    #[test]
    fn negative_gap_flags_infeasible() {
        let model = trivial_model(vec![cont("x", 0.0, 10.0)]);
        let mut bounds = vec![Interval::new(0.0, 10.0)];
        let info = ReducedCostInfo {
            lp_value: 5.0,
            cutoff: 3.0,
            reduced_costs: vec![1.0],
            reduced_cost_errors: vec![0.0],
        };
        let s = reduced_cost_fixing(&model, &mut bounds, &info);
        assert!(s.infeasible);
        assert_eq!(s.bounds_tightened, 0);
    }

    #[test]
    fn mismatched_lengths_no_op() {
        let model = trivial_model(vec![cont("x", 0.0, 10.0), cont("y", 0.0, 10.0)]);
        let mut bounds = vec![Interval::new(0.0, 10.0), Interval::new(0.0, 10.0)];
        let info = ReducedCostInfo {
            lp_value: 0.0,
            cutoff: 1.0,
            reduced_costs: vec![1.0], // wrong length
            reduced_cost_errors: vec![0.0],
        };
        let s = reduced_cost_fixing(&model, &mut bounds, &info);
        assert_eq!(s.bounds_tightened, 0);
        assert_eq!(s.blocks_examined, 0);
        // A reduced-cost length mismatch is the pre-existing no-op, not the #1409
        // missing-error-bound refusal; keep the two diagnoses distinguishable.
        assert!(!s.refused_unbounded_error);
    }

    #[test]
    fn vars_fixed_when_pinch_to_point() {
        // gap = 0, cbar > 0 ⇒ ub collapses to lb.
        //
        // #1409 changed what "collapses" requires. The pass now widens the gap by
        // `gap_slack(cutoff)`, so a zero raw gap still leaves a window of
        // `gap_slack / |c̄_j|` open; the collapse detector's `1e-12` therefore needs
        // `|c̄_j| > gap_slack / 1e-12`. `c̄_j = 1e7` against `gap_slack(5) = 6e-6`
        // gives a window of 6e-13, which still pinches. See
        // `zero_gap_no_longer_pinches_an_ordinary_column` for the complementary
        // half: at an ordinary `|c̄_j|` the pass deliberately declines to collapse.
        let model = trivial_model(vec![cont("x", 0.0, 10.0)]);
        let mut bounds = vec![Interval::new(0.0, 10.0)];
        let info = ReducedCostInfo {
            lp_value: 5.0,
            cutoff: 5.0,
            reduced_costs: vec![1e7],
            reduced_cost_errors: vec![0.0],
        };
        let s = reduced_cost_fixing(&model, &mut bounds, &info);
        assert!(s.bounds_tightened >= 1);
        assert_eq!(s.vars_fixed.len(), 1);
        assert_eq!(s.vars_fixed[0].0, 0);
        assert!(s.vars_fixed[0].1.abs() < 1e-9);
    }

    #[test]
    fn zero_gap_no_longer_pinches_an_ordinary_column() {
        // The behaviour change from adding the live path's gap slack (#1409), recorded
        // rather than hidden: with `cutoff == lp_value` and an ordinary reduced cost
        // the pass leaves a `gap_slack / |c̄_j|` window instead of fixing the variable
        // to its lower bound. That is the correct conservative answer — at a zero raw
        // gap, round-off in `cutoff` and `lp_value` is larger than the gap itself, so
        // "no improving point has x_j > lb" is not certifiable.
        let model = trivial_model(vec![cont("x", 0.0, 10.0)]);
        let mut bounds = vec![Interval::new(0.0, 10.0)];
        let info = ReducedCostInfo {
            lp_value: 5.0,
            cutoff: 5.0,
            reduced_costs: vec![1.0],
            reduced_cost_errors: vec![0.0],
        };
        let s = reduced_cost_fixing(&model, &mut bounds, &info);
        assert!(s.bounds_tightened >= 1, "it must still tighten 10 -> ~6e-6");
        assert!(
            s.vars_fixed.is_empty(),
            "a zero raw gap must not fix the var"
        );
        assert!(bounds[0].hi > 0.0, "the slack window must stay open");
        assert!(bounds[0].hi <= gap_slack(5.0) + 1e-18);
    }

    #[test]
    fn missing_error_bounds_refuse_loudly_instead_of_tightening() {
        // #1409 Finding 2: this pass cannot derive `c̄_j`'s error itself (it never sees
        // the column, the matrix or the duals), so an absent bound is a refusal, not a
        // licence to assume exactness. Counted assertions per CLAUDE.md §6.
        let mut asserts = 0usize;
        for errors in [vec![], vec![0.0, 0.0]] {
            let model = trivial_model(vec![cont("x", 0.0, 100.0)]);
            let mut bounds = vec![Interval::new(0.0, 100.0)];
            let info = ReducedCostInfo {
                lp_value: 0.0,
                cutoff: 10.0,
                reduced_costs: vec![2.0],
                reduced_cost_errors: errors,
            };
            let s = reduced_cost_fixing(&model, &mut bounds, &info);
            assert!(s.refused_unbounded_error, "the refusal must be reported");
            asserts += 1;
            assert_eq!(s.bounds_tightened, 0, "a refusal must not tighten");
            asserts += 1;
            assert!((bounds[0].hi - 100.0).abs() < 1e-12, "bounds untouched");
            asserts += 1;
        }
        assert_eq!(asserts, 6, "probe must have executed every assertion");
    }

    #[test]
    fn an_error_bound_swamping_the_reduced_cost_blocks_the_tightening() {
        // The divisor-deflation half of Finding 1, at this pass's interface: when the
        // supplied error bound exceeds `|c̄_j|` the sign of `c̄_j` is not resolved, so
        // no endpoint follows from it. The *same* fixture tightens when the bound is
        // small, which is what makes this a discriminating test and not a no-op.
        let mut asserts = 0usize;
        let fixture = |err: f64| {
            let model = trivial_model(vec![cont("x", 0.0, 100.0)]);
            let mut bounds = vec![Interval::new(0.0, 100.0)];
            let info = ReducedCostInfo {
                lp_value: 0.0,
                cutoff: 10.0,
                reduced_costs: vec![2.0],
                reduced_cost_errors: vec![err],
            };
            let s = reduced_cost_fixing(&model, &mut bounds, &info);
            (s, bounds[0].hi)
        };

        // `c̄_j = 2` known only to ±3: could be negative, so no upper bound follows.
        let (s_blind, hi_blind) = fixture(3.0);
        assert_eq!(
            s_blind.bounds_tightened, 0,
            "unresolved sign must not tighten"
        );
        asserts += 1;
        assert!((hi_blind - 100.0).abs() < 1e-12);
        asserts += 1;
        assert!(
            !s_blind.refused_unbounded_error,
            "a supplied bound is not a refusal"
        );
        asserts += 1;

        // Same `c̄_j`, a bound a real producer would report: the pass tightens, and
        // deflating the divisor pushes the endpoint outward of the undeflated 5.0.
        let (s_ok, hi_ok) = fixture(1e-9);
        assert_eq!(
            s_ok.bounds_tightened, 1,
            "a resolved sign must still tighten"
        );
        asserts += 1;
        assert!(
            hi_ok > 5.0,
            "the deflated divisor must widen the endpoint: {hi_ok}"
        );
        asserts += 1;
        assert!(hi_ok < 100.0, "it must remain a real tightening");
        asserts += 1;

        // Monotonicity in the safe direction: a larger error bound never yields a
        // tighter endpoint.
        let (_, hi_looser) = fixture(1e-3);
        assert!(
            hi_looser >= hi_ok,
            "more uncertainty must not tighten further"
        );
        asserts += 1;

        // A NaN or negative "bound" is not a bound; refuse the block.
        for bad in [f64::NAN, -1.0] {
            let (s_bad, hi_bad) = fixture(bad);
            assert_eq!(s_bad.bounds_tightened, 0, "a non-bound must not tighten");
            asserts += 1;
            assert!((hi_bad - 100.0).abs() < 1e-12);
            asserts += 1;
        }

        assert_eq!(asserts, 11, "probe must have executed every assertion");
    }
}
