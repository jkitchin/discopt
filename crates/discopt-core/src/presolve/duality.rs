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
    let gap = info.cutoff - info.lp_value;
    if gap < -1e-9 {
        stats.infeasible = true;
        return stats;
    }
    let gap = gap.max(0.0);
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
        let cur = bounds[block_idx];
        let lb = cur.lo;
        let ub = cur.hi;
        if lb > ub {
            continue;
        }
        let is_integer = matches!(var.var_type, VarType::Binary | VarType::Integer);

        let mut new_lb = lb;
        let mut new_ub = ub;

        if cbar > 1e-12 && lb.is_finite() {
            // ub_new = lb + gap / cbar
            let candidate = lb + gap / cbar;
            let candidate = if is_integer {
                candidate.floor()
            } else {
                candidate
            };
            if candidate < new_ub {
                new_ub = candidate;
            }
        }
        if cbar < -1e-12 && ub.is_finite() {
            // lb_new = ub + gap / cbar (cbar < 0 ⇒ adds a negative number)
            let candidate = ub + gap / cbar;
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

    #[test]
    fn positive_reduced_cost_tightens_upper_bound() {
        // gap = 10, cbar = 2, lb = 0 ⇒ new_ub = 0 + 10/2 = 5.
        let model = trivial_model(vec![cont("x", 0.0, 100.0)]);
        let mut bounds = vec![Interval::new(0.0, 100.0)];
        let info = ReducedCostInfo {
            lp_value: 0.0,
            cutoff: 10.0,
            reduced_costs: vec![2.0],
        };
        let s = reduced_cost_fixing(&model, &mut bounds, &info);
        assert_eq!(s.bounds_tightened, 1);
        assert!((bounds[0].hi - 5.0).abs() < 1e-9);
        assert!((bounds[0].lo - 0.0).abs() < 1e-9);
        assert!(!s.infeasible);
    }

    #[test]
    fn negative_reduced_cost_tightens_lower_bound() {
        // gap = 6, cbar = -3, ub = 10 ⇒ new_lb = 10 + 6 / (-3) = 8.
        let model = trivial_model(vec![cont("x", 0.0, 10.0)]);
        let mut bounds = vec![Interval::new(0.0, 10.0)];
        let info = ReducedCostInfo {
            lp_value: 0.0,
            cutoff: 6.0,
            reduced_costs: vec![-3.0],
        };
        let s = reduced_cost_fixing(&model, &mut bounds, &info);
        assert_eq!(s.bounds_tightened, 1);
        assert!((bounds[0].lo - 8.0).abs() < 1e-9);
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
        };
        let s = reduced_cost_fixing(&model, &mut bounds, &info);
        assert_eq!(s.bounds_tightened, 0);
        assert_eq!(s.blocks_examined, 0);
    }

    #[test]
    fn vars_fixed_when_pinch_to_point() {
        // gap = 0, cbar > 0 ⇒ ub collapses to lb.
        let model = trivial_model(vec![cont("x", 0.0, 10.0)]);
        let mut bounds = vec![Interval::new(0.0, 10.0)];
        let info = ReducedCostInfo {
            lp_value: 5.0,
            cutoff: 5.0,
            reduced_costs: vec![1.0],
        };
        let s = reduced_cost_fixing(&model, &mut bounds, &info);
        assert!(s.bounds_tightened >= 1);
        assert_eq!(s.vars_fixed.len(), 1);
        assert_eq!(s.vars_fixed[0].0, 0);
        assert!(s.vars_fixed[0].1.abs() < 1e-9);
    }
}
