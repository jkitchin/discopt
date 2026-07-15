//! Binary variable probing.
//!
//! For each binary variable, temporarily fix it to 0 and 1, run FBBT,
//! and use the results to detect fixings, tightened bounds, and implications.

use std::time::Instant;

use super::fbbt::{fbbt, fbbt_with_cutoff, Interval, FEAS_TOL};
use crate::expr::{ModelRepr, VarType};

/// Result of probing all binary variables.
#[derive(Debug, Clone)]
pub struct ProbingResult {
    /// Variables fixed during probing: (var_index, fixed_value).
    pub fixed_vars: Vec<(usize, f64)>,
    /// Tightened variable bounds (indexed by variable index).
    pub tightened_bounds: Vec<Interval>,
    /// Discovered implications.
    pub implications: Vec<Implication>,
}

/// An implication discovered during probing.
#[derive(Debug, Clone)]
pub struct Implication {
    /// Index of the binary variable being probed.
    pub binary_var: usize,
    /// The value (false=0, true=1) that triggers the implication.
    pub binary_val: bool,
    /// The variable whose bounds are tightened.
    pub implied_var: usize,
    /// The tightened bounds for the implied variable.
    pub implied_bound: Interval,
}

/// Probe all binary variables in the model.
///
/// Uses FBBT with each binary variable fixed to 0 and 1 to discover:
/// - Variables that must be fixed (one fixing causes infeasibility).
/// - Tighter bounds from intersecting the two cases.
/// - Implications for bound tightening.
pub fn probe_binary_vars(model: &ModelRepr, var_bounds: &[Interval]) -> ProbingResult {
    probe_binary_vars_until(model, var_bounds, None)
}

/// Like [`probe_binary_vars`] but stops once `deadline` passes, returning the
/// implications gathered so far. Probing clones the model and runs FBBT per
/// binary variable, so a full pass on a large model can take minutes — far
/// past the orchestrator's *between-passes* time check. Bailing mid-loop only
/// performs *fewer* tightenings; every implication already found stays valid,
/// so the early exit is sound.
pub fn probe_binary_vars_until(
    model: &ModelRepr,
    var_bounds: &[Interval],
    deadline: Option<Instant>,
) -> ProbingResult {
    let n = var_bounds.len();
    let mut fixed_vars: Vec<(usize, f64)> = Vec::new();
    let mut best_bounds = var_bounds.to_vec();
    let mut implications: Vec<Implication> = Vec::new();

    let max_fbbt_iter = 5;
    let fbbt_tol = 1e-8;

    for (i, vinfo) in model.variables.iter().enumerate() {
        if let Some(dl) = deadline {
            if Instant::now() >= dl {
                break;
            }
        }
        if vinfo.var_type != VarType::Binary {
            continue;
        }

        // Probe y=0.
        let mut bounds_zero = var_bounds.to_vec();
        bounds_zero[i] = Interval::new(0.0, 0.0);
        let model_zero = make_probing_model(model, &bounds_zero);
        let result_zero = fbbt(&model_zero, max_fbbt_iter, fbbt_tol);
        // Declare a fixing infeasible only when a bound is empty beyond the
        // feasibility tolerance. An eps-scale inverted interval (e.g. from a
        // GDP hull perspective residual) is numerical noise, not proof that
        // the selector cannot take this value — treating it as such would fix
        // the disjunction wrongly and yield an unsound bound.
        let infeasible_zero = result_zero.iter().any(|b| b.is_empty_beyond(FEAS_TOL));

        // Probe y=1.
        let mut bounds_one = var_bounds.to_vec();
        bounds_one[i] = Interval::new(1.0, 1.0);
        let model_one = make_probing_model(model, &bounds_one);
        let result_one = fbbt(&model_one, max_fbbt_iter, fbbt_tol);
        let infeasible_one = result_one.iter().any(|b| b.is_empty_beyond(FEAS_TOL));

        if infeasible_zero && infeasible_one {
            // Both infeasible — model is infeasible. Mark bounds as empty.
            for b in &mut best_bounds {
                *b = Interval::empty();
            }
            return ProbingResult {
                fixed_vars,
                tightened_bounds: best_bounds,
                implications,
            };
        } else if infeasible_zero {
            // y=0 infeasible => y must be 1.
            fixed_vars.push((i, 1.0));
            best_bounds[i] = Interval::new(1.0, 1.0);
            // Apply bounds from the y=1 case.
            for j in 0..n {
                best_bounds[j] = best_bounds[j].intersect(&result_one[j]);
            }
        } else if infeasible_one {
            // y=1 infeasible => y must be 0.
            fixed_vars.push((i, 0.0));
            best_bounds[i] = Interval::new(0.0, 0.0);
            // Apply bounds from the y=0 case.
            for j in 0..n {
                best_bounds[j] = best_bounds[j].intersect(&result_zero[j]);
            }
        } else {
            // Both feasible: tighten bounds by intersection.
            for j in 0..n {
                if j == i {
                    continue;
                }
                let combined = Interval::new(
                    result_zero[j].lo.min(result_one[j].lo),
                    result_zero[j].hi.max(result_one[j].hi),
                );
                let tightened = best_bounds[j].intersect(&combined);
                best_bounds[j] = tightened;

                // Record implications if one case produces tighter bounds.
                if result_zero[j].lo > var_bounds[j].lo + fbbt_tol
                    || result_zero[j].hi < var_bounds[j].hi - fbbt_tol
                {
                    implications.push(Implication {
                        binary_var: i,
                        binary_val: false,
                        implied_var: j,
                        implied_bound: result_zero[j],
                    });
                }
                if result_one[j].lo > var_bounds[j].lo + fbbt_tol
                    || result_one[j].hi < var_bounds[j].hi - fbbt_tol
                {
                    implications.push(Implication {
                        binary_var: i,
                        binary_val: true,
                        implied_var: j,
                        implied_bound: result_one[j],
                    });
                }
            }
        }
    }

    ProbingResult {
        fixed_vars,
        tightened_bounds: best_bounds,
        implications,
    }
}

/// Create a model copy with modified variable bounds for probing.
fn make_probing_model(model: &ModelRepr, var_bounds: &[Interval]) -> ModelRepr {
    let mut probe_model = model.clone();
    for (i, vinfo) in probe_model.variables.iter_mut().enumerate() {
        if i < var_bounds.len() {
            vinfo.lb = vec![var_bounds[i].lo];
            vinfo.ub = vec![var_bounds[i].hi];
        }
    }
    probe_model
}

// ─────────────────────────────────────────────────────────────
// Per-node probing (P3 branch-and-reduce, issue #632)
// ─────────────────────────────────────────────────────────────

/// Result of per-node probing over discrete variables.
#[derive(Debug, Clone)]
pub struct NodeProbeResult {
    /// Tightened variable bounds (one interval per variable). Always a subset
    /// of the input box — probing never loosens a bound.
    pub tightened_bounds: Vec<Interval>,
    /// Variables permanently fixed by probing: `(var_index, fixed_value)`.
    pub fixed_vars: Vec<(usize, f64)>,
    /// `true` iff probing proved the node infeasible (an empty box).
    pub infeasible: bool,
    /// Number of half-bounds tightened (either endpoint contracted).
    pub n_tightened: usize,
    /// Number of discrete variables actually probed (budget accounting).
    pub n_probed: usize,
}

/// Cap on integer-endpoint peels per variable per call. A wide integer domain
/// whose whole range is infeasible would otherwise spin the inner FBBT solve
/// linearly in the domain width at one node; this bounds the per-node cost
/// (prototype budget — the surviving box is always valid regardless of the cap).
const MAX_INT_PEEL: usize = 8;

/// Per-node probing over binary **and** general-integer variables (P3).
///
/// For each discrete variable (up to `max_vars`, in index order), tentatively
/// fix it at a bound, run cutoff-aware FBBT on the resulting box, and CONTRACT
/// the domain on any *proven-infeasible* fixing:
///
///   * **binary `y`** — fix to 0 and to 1; if exactly one branch is infeasible
///     the other value is forced (permanent fix + intersect the surviving
///     branch's implied bounds into the box); if both are infeasible the node
///     is infeasible; if both are feasible, tighten every *other* variable to
///     the interval hull of the two branches (a sound implication cut);
///   * **integer `x ∈ [l, u]`** — probe the two endpoints; a proven-infeasible
///     `x = l` justifies `x ≥ l+1`, a proven-infeasible `x = u` justifies
///     `x ≤ u-1` (repeated up to [`MAX_INT_PEEL`] times per endpoint).
///
/// # Soundness
/// A bound is contracted ONLY when a fixing is *proven* infeasible via
/// [`Interval::is_empty_beyond`] at `FEAS_TOL` — an eps-scale inverted interval
/// is treated as numerical noise, never as proof (mirrors [`probe_binary_vars`]).
/// FBBT itself only ever derives valid bounds, so intersecting a surviving
/// branch's result can only tighten. When `incumbent` is `Some`, the cutoff
/// makes probing optimality-aware (`objective ⋈ incumbent`): a fixing whose
/// sub-box cannot reach the incumbent is discarded — sound inside branch &
/// bound because such points cannot improve the incumbent. Every returned bound
/// is a subset of the corresponding input bound.
///
/// `var_bounds` is the node box (one interval per variable). `deadline` bounds
/// wall time; on expiry the pass returns whatever it has contracted so far
/// (each contraction is independently valid, so an early exit is sound).
pub fn probe_node_bounds(
    model: &ModelRepr,
    var_bounds: &[Interval],
    max_vars: usize,
    max_fbbt_iter: usize,
    tol: f64,
    incumbent: Option<f64>,
    deadline: Option<Instant>,
) -> NodeProbeResult {
    let n = var_bounds.len();
    let mut bounds = var_bounds.to_vec();
    let mut fixed_vars: Vec<(usize, f64)> = Vec::new();
    let mut n_tightened = 0usize;
    let mut n_probed = 0usize;

    // Run cutoff-aware FBBT on `bounds` with variable `idx` pinned to
    // `[val, val]`. Returns the per-variable intervals; `is_empty_beyond`
    // on any of them proves the fixing infeasible.
    let probe = |bnds: &[Interval], idx: usize, val: f64| -> Vec<Interval> {
        let mut b = bnds.to_vec();
        b[idx] = Interval::new(val, val);
        let m = make_probing_model(model, &b);
        fbbt_with_cutoff(&m, max_fbbt_iter, tol, incumbent)
    };
    let branch_infeasible =
        |res: &[Interval]| -> bool { res.iter().any(|b| b.is_empty_beyond(FEAS_TOL)) };

    for (i, vinfo) in model.variables.iter().enumerate() {
        if n_probed >= max_vars {
            break;
        }
        if let Some(dl) = deadline {
            if Instant::now() >= dl {
                break;
            }
        }
        if i >= n {
            break;
        }

        match vinfo.var_type {
            VarType::Binary => {
                // Skip binaries already pinned by branching / earlier probing.
                if bounds[i].width() < 0.5 {
                    continue;
                }
                let res0 = probe(&bounds, i, 0.0);
                let res1 = probe(&bounds, i, 1.0);
                let infeas0 = branch_infeasible(&res0);
                let infeas1 = branch_infeasible(&res1);
                n_probed += 1;

                if infeas0 && infeas1 {
                    return NodeProbeResult {
                        tightened_bounds: vec![Interval::empty(); n],
                        fixed_vars,
                        infeasible: true,
                        n_tightened,
                        n_probed,
                    };
                } else if infeas0 {
                    // y = 0 infeasible ⇒ y must be 1.
                    fixed_vars.push((i, 1.0));
                    bounds[i] = Interval::new(1.0, 1.0);
                    n_tightened += 1;
                    for j in 0..n {
                        bounds[j] = bounds[j].intersect(&res1[j]);
                    }
                } else if infeas1 {
                    // y = 1 infeasible ⇒ y must be 0.
                    fixed_vars.push((i, 0.0));
                    bounds[i] = Interval::new(0.0, 0.0);
                    n_tightened += 1;
                    for j in 0..n {
                        bounds[j] = bounds[j].intersect(&res0[j]);
                    }
                } else {
                    // Both branches feasible: any feasible point lies in
                    // res0 ∪ res1, hence in their interval hull. Intersecting
                    // that hull into every OTHER variable is a sound cut.
                    for j in 0..n {
                        if j == i {
                            continue;
                        }
                        let hull =
                            Interval::new(res0[j].lo.min(res1[j].lo), res0[j].hi.max(res1[j].hi));
                        let before = bounds[j];
                        bounds[j] = bounds[j].intersect(&hull);
                        if bounds[j].lo > before.lo + tol || bounds[j].hi < before.hi - tol {
                            n_tightened += 1;
                        }
                    }
                }
                if bounds[i].is_empty_beyond(FEAS_TOL) {
                    return NodeProbeResult {
                        tightened_bounds: vec![Interval::empty(); n],
                        fixed_vars,
                        infeasible: true,
                        n_tightened,
                        n_probed,
                    };
                }
            }
            VarType::Integer => {
                // Only finite, non-degenerate integer domains have endpoints
                // to peel.
                if !bounds[i].lo.is_finite() || !bounds[i].hi.is_finite() {
                    continue;
                }
                if bounds[i].width() < 0.5 {
                    continue;
                }
                n_probed += 1;

                // Peel infeasible lower endpoints: x = l infeasible ⇒ x ≥ l+1.
                for _ in 0..MAX_INT_PEEL {
                    if bounds[i].width() < 0.5 {
                        break;
                    }
                    let lo = bounds[i].lo;
                    if branch_infeasible(&probe(&bounds, i, lo)) {
                        bounds[i] = Interval::new(lo + 1.0, bounds[i].hi);
                        n_tightened += 1;
                    } else {
                        break;
                    }
                }
                // Peel infeasible upper endpoints: x = u infeasible ⇒ x ≤ u-1.
                for _ in 0..MAX_INT_PEEL {
                    if bounds[i].width() < 0.5 {
                        break;
                    }
                    let hi = bounds[i].hi;
                    if branch_infeasible(&probe(&bounds, i, hi)) {
                        bounds[i] = Interval::new(bounds[i].lo, hi - 1.0);
                        n_tightened += 1;
                    } else {
                        break;
                    }
                }
                if bounds[i].is_empty_beyond(FEAS_TOL) {
                    return NodeProbeResult {
                        tightened_bounds: vec![Interval::empty(); n],
                        fixed_vars,
                        infeasible: true,
                        n_tightened,
                        n_probed,
                    };
                }
                // A fully-pinned integer is a fixing worth recording.
                if bounds[i].width() < 0.5 {
                    fixed_vars.push((i, bounds[i].lo.round()));
                }
            }
            VarType::Continuous => {}
        }
    }

    NodeProbeResult {
        tightened_bounds: bounds,
        fixed_vars,
        infeasible: false,
        n_tightened,
        n_probed,
    }
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::fbbt::Interval;
    use super::*;
    use crate::expr::*;

    #[test]
    fn test_probing_fixes_binary_var() {
        // x + 10*y <= 5, x in [0, 10], y binary
        // If y=1: x + 10 <= 5 => x <= -5. But x >= 0 => infeasible.
        // Therefore y must be 0.
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let y = arena.add(ExprNode::Variable {
            name: "y".into(),
            index: 1,
            size: 1,
            shape: vec![],
        });
        let c10 = arena.add(ExprNode::Constant(10.0));
        let prod = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: c10,
            right: y,
        });
        let sum = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x,
            right: prod,
        });
        let model = ModelRepr {
            arena,
            objective: ExprId(0),
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: sum,
                sense: ConstraintSense::Le,
                rhs: 5.0,
                name: Some("c1".into()),
            }],
            variables: vec![
                VarInfo {
                    name: "x".into(),
                    var_type: VarType::Continuous,
                    offset: 0,
                    size: 1,
                    shape: vec![],
                    lb: vec![0.0],
                    ub: vec![10.0],
                },
                VarInfo {
                    name: "y".into(),
                    var_type: VarType::Binary,
                    offset: 1,
                    size: 1,
                    shape: vec![],
                    lb: vec![0.0],
                    ub: vec![1.0],
                },
            ],
            n_vars: 2,
        };

        let var_bounds = vec![Interval::new(0.0, 10.0), Interval::new(0.0, 1.0)];
        let result = probe_binary_vars(&model, &var_bounds);

        // y should be fixed to 0.
        assert_eq!(result.fixed_vars.len(), 1);
        assert_eq!(result.fixed_vars[0].0, 1);
        assert!((result.fixed_vars[0].1 - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_probing_honors_past_deadline() {
        // Same model as `test_probing_fixes_binary_var` (probing would fix y=0),
        // but with a deadline already in the past. The orchestrator only checks
        // the time budget *between* passes, so probing — which clones the model
        // and runs FBBT per binary — must poll the deadline itself and bail
        // before doing work. Returning fewer tightenings is always sound.
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let y = arena.add(ExprNode::Variable {
            name: "y".into(),
            index: 1,
            size: 1,
            shape: vec![],
        });
        let c10 = arena.add(ExprNode::Constant(10.0));
        let prod = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: c10,
            right: y,
        });
        let sum = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x,
            right: prod,
        });
        let model = ModelRepr {
            arena,
            objective: ExprId(0),
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: sum,
                sense: ConstraintSense::Le,
                rhs: 5.0,
                name: Some("c1".into()),
            }],
            variables: vec![
                VarInfo {
                    name: "x".into(),
                    var_type: VarType::Continuous,
                    offset: 0,
                    size: 1,
                    shape: vec![],
                    lb: vec![0.0],
                    ub: vec![10.0],
                },
                VarInfo {
                    name: "y".into(),
                    var_type: VarType::Binary,
                    offset: 1,
                    size: 1,
                    shape: vec![],
                    lb: vec![0.0],
                    ub: vec![1.0],
                },
            ],
            n_vars: 2,
        };

        let var_bounds = vec![Interval::new(0.0, 10.0), Interval::new(0.0, 1.0)];

        // No deadline: probing does its work and fixes y=0 (control).
        let unbounded = probe_binary_vars_until(&model, &var_bounds, None);
        assert_eq!(unbounded.fixed_vars.len(), 1);

        // Deadline already in the past: probing bails at the first loop check,
        // before probing any binary, so no fixings are reported.
        let past = std::time::Instant::now();
        let bailed = probe_binary_vars_until(&model, &var_bounds, Some(past));
        assert!(
            bailed.fixed_vars.is_empty(),
            "probing must bail before any work once the deadline has passed"
        );
        // Bounds pass through unchanged on the early exit.
        assert_eq!(bailed.tightened_bounds.len(), var_bounds.len());
        for (got, want) in bailed.tightened_bounds.iter().zip(var_bounds.iter()) {
            assert_eq!(got.lo, want.lo);
            assert_eq!(got.hi, want.hi);
        }
    }

    #[test]
    fn test_probing_no_fixing_needed() {
        // x + y <= 10, x in [0, 5], y binary
        // Both y=0 and y=1 are feasible.
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let y = arena.add(ExprNode::Variable {
            name: "y".into(),
            index: 1,
            size: 1,
            shape: vec![],
        });
        let sum = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x,
            right: y,
        });
        let model = ModelRepr {
            arena,
            objective: ExprId(0),
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: sum,
                sense: ConstraintSense::Le,
                rhs: 10.0,
                name: Some("c1".into()),
            }],
            variables: vec![
                VarInfo {
                    name: "x".into(),
                    var_type: VarType::Continuous,
                    offset: 0,
                    size: 1,
                    shape: vec![],
                    lb: vec![0.0],
                    ub: vec![5.0],
                },
                VarInfo {
                    name: "y".into(),
                    var_type: VarType::Binary,
                    offset: 1,
                    size: 1,
                    shape: vec![],
                    lb: vec![0.0],
                    ub: vec![1.0],
                },
            ],
            n_vars: 2,
        };

        let var_bounds = vec![Interval::new(0.0, 5.0), Interval::new(0.0, 1.0)];
        let result = probe_binary_vars(&model, &var_bounds);

        // No variables should be fixed.
        assert!(result.fixed_vars.is_empty());
    }

    #[test]
    fn test_probing_discovers_implications() {
        // x + 5*y <= 8, x in [0, 10], y binary
        // y=0: x <= 8
        // y=1: x <= 3
        // Both feasible, but y=1 tightens x.
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let y = arena.add(ExprNode::Variable {
            name: "y".into(),
            index: 1,
            size: 1,
            shape: vec![],
        });
        let c5 = arena.add(ExprNode::Constant(5.0));
        let prod = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: c5,
            right: y,
        });
        let sum = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x,
            right: prod,
        });
        let model = ModelRepr {
            arena,
            objective: ExprId(0),
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: sum,
                sense: ConstraintSense::Le,
                rhs: 8.0,
                name: Some("c1".into()),
            }],
            variables: vec![
                VarInfo {
                    name: "x".into(),
                    var_type: VarType::Continuous,
                    offset: 0,
                    size: 1,
                    shape: vec![],
                    lb: vec![0.0],
                    ub: vec![10.0],
                },
                VarInfo {
                    name: "y".into(),
                    var_type: VarType::Binary,
                    offset: 1,
                    size: 1,
                    shape: vec![],
                    lb: vec![0.0],
                    ub: vec![1.0],
                },
            ],
            n_vars: 2,
        };

        let var_bounds = vec![Interval::new(0.0, 10.0), Interval::new(0.0, 1.0)];
        let result = probe_binary_vars(&model, &var_bounds);

        // Should have implications for x from both y=0 and y=1 fixings.
        assert!(!result.implications.is_empty());

        // Find the y=1 implication on x.
        let y1_impl = result
            .implications
            .iter()
            .find(|imp| imp.binary_var == 1 && imp.binary_val && imp.implied_var == 0);
        assert!(y1_impl.is_some());
        let imp = y1_impl.unwrap();
        // x should be tightened to at most 3.
        assert!((imp.implied_bound.hi - 3.0).abs() < 1e-8);
    }

    #[test]
    fn test_probing_tightens_bounds() {
        // x + 5*y <= 8, x in [0, 10], y binary
        // Union of y=0 case (x<=8) and y=1 case (x<=3) is x<=8.
        // So best_bounds for x should be [0, 8] (intersected with original [0,10]).
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let y = arena.add(ExprNode::Variable {
            name: "y".into(),
            index: 1,
            size: 1,
            shape: vec![],
        });
        let c5 = arena.add(ExprNode::Constant(5.0));
        let prod = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: c5,
            right: y,
        });
        let sum = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x,
            right: prod,
        });
        let model = ModelRepr {
            arena,
            objective: ExprId(0),
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: sum,
                sense: ConstraintSense::Le,
                rhs: 8.0,
                name: Some("c1".into()),
            }],
            variables: vec![
                VarInfo {
                    name: "x".into(),
                    var_type: VarType::Continuous,
                    offset: 0,
                    size: 1,
                    shape: vec![],
                    lb: vec![0.0],
                    ub: vec![10.0],
                },
                VarInfo {
                    name: "y".into(),
                    var_type: VarType::Binary,
                    offset: 1,
                    size: 1,
                    shape: vec![],
                    lb: vec![0.0],
                    ub: vec![1.0],
                },
            ],
            n_vars: 2,
        };

        let var_bounds = vec![Interval::new(0.0, 10.0), Interval::new(0.0, 1.0)];
        let result = probe_binary_vars(&model, &var_bounds);

        // x should be tightened.
        assert!(result.tightened_bounds[0].hi <= 8.0 + 1e-8);
    }

    #[test]
    fn test_probing_ignores_eps_residual() {
        // x + 1e-9*y <= 0, with x fixed to [0, 0], y binary.
        // Fixing y=1 makes the body 1e-9 — an eps-scale violation of the bound,
        // exactly the shape of a GDP hull perspective residual at an integer
        // face. This is feasible within tolerance and MUST NOT fix y=0; doing so
        // would wrongly eliminate a disjunct and yield an unsound bound (#27a).
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let y = arena.add(ExprNode::Variable {
            name: "y".into(),
            index: 1,
            size: 1,
            shape: vec![],
        });
        let eps = arena.add(ExprNode::Constant(1e-9));
        let prod = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: eps,
            right: y,
        });
        let sum = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x,
            right: prod,
        });
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: sum,
                sense: ConstraintSense::Le,
                rhs: 0.0,
                name: Some("c1".into()),
            }],
            variables: vec![
                VarInfo {
                    name: "x".into(),
                    var_type: VarType::Continuous,
                    offset: 0,
                    size: 1,
                    shape: vec![],
                    lb: vec![0.0],
                    ub: vec![0.0],
                },
                VarInfo {
                    name: "y".into(),
                    var_type: VarType::Binary,
                    offset: 1,
                    size: 1,
                    shape: vec![],
                    lb: vec![0.0],
                    ub: vec![1.0],
                },
            ],
            n_vars: 2,
        };

        let var_bounds = vec![Interval::new(0.0, 0.0), Interval::new(0.0, 1.0)];
        let result = probe_binary_vars(&model, &var_bounds);

        // y must remain free: the eps-scale residual is not a real infeasibility.
        assert!(
            result.fixed_vars.is_empty(),
            "eps-scale residual must not fix the binary selector"
        );
    }

    #[test]
    fn test_probing_fixes_on_real_violation() {
        // x + 1.0*y <= 0, with x fixed to [0, 0], y binary.
        // Fixing y=1 makes the body 1.0 — a violation well beyond the
        // feasibility tolerance — so y must genuinely be fixed to 0.
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let y = arena.add(ExprNode::Variable {
            name: "y".into(),
            index: 1,
            size: 1,
            shape: vec![],
        });
        let one = arena.add(ExprNode::Constant(1.0));
        let prod = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: one,
            right: y,
        });
        let sum = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x,
            right: prod,
        });
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: sum,
                sense: ConstraintSense::Le,
                rhs: 0.0,
                name: Some("c1".into()),
            }],
            variables: vec![
                VarInfo {
                    name: "x".into(),
                    var_type: VarType::Continuous,
                    offset: 0,
                    size: 1,
                    shape: vec![],
                    lb: vec![0.0],
                    ub: vec![0.0],
                },
                VarInfo {
                    name: "y".into(),
                    var_type: VarType::Binary,
                    offset: 1,
                    size: 1,
                    shape: vec![],
                    lb: vec![0.0],
                    ub: vec![1.0],
                },
            ],
            n_vars: 2,
        };

        let var_bounds = vec![Interval::new(0.0, 0.0), Interval::new(0.0, 1.0)];
        let result = probe_binary_vars(&model, &var_bounds);

        assert_eq!(result.fixed_vars.len(), 1);
        assert_eq!(result.fixed_vars[0].0, 1);
        assert!((result.fixed_vars[0].1 - 0.0).abs() < 1e-15);
    }

    // ── Per-node probing (P3 branch-and-reduce) ───────────────────

    fn var(name: &str, ty: VarType, lb: f64, ub: f64, idx: usize) -> VarInfo {
        VarInfo {
            name: name.into(),
            var_type: ty,
            offset: idx,
            size: 1,
            shape: vec![],
            lb: vec![lb],
            ub: vec![ub],
        }
    }

    /// x + 10·b ≤ 5, x ∈ [0,10], b binary. At a node where branching pinned
    /// x ≥ 3, b = 1 forces x ≤ -5 (infeasible), so per-node probing must fix
    /// b = 0 — a fixing keyed to the *node* box, not the root box.
    #[test]
    fn probe_node_fixes_binary_at_node_box() {
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let b = arena.add(ExprNode::Variable {
            name: "b".into(),
            index: 1,
            size: 1,
            shape: vec![],
        });
        let c10 = arena.add(ExprNode::Constant(10.0));
        let prod = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: c10,
            right: b,
        });
        let sum = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x,
            right: prod,
        });
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: sum,
                sense: ConstraintSense::Le,
                rhs: 5.0,
                name: None,
            }],
            variables: vec![
                var("x", VarType::Continuous, 0.0, 10.0, 0),
                var("b", VarType::Binary, 0.0, 1.0, 1),
            ],
            n_vars: 2,
        };
        // Node box: branching pinned x ∈ [3, 10].
        let box_ = vec![Interval::new(3.0, 10.0), Interval::new(0.0, 1.0)];
        let res = probe_node_bounds(&model, &box_, 32, 8, 1e-9, None, None);
        assert!(!res.infeasible);
        assert_eq!(res.fixed_vars.len(), 1);
        assert_eq!(res.fixed_vars[0].0, 1);
        assert!((res.fixed_vars[0].1 - 0.0).abs() < 1e-9);
        assert!(res.tightened_bounds[1].hi <= 1e-9, "b must be fixed to 0");
    }

    /// x·x ≤ 5, x integer ∈ [1, 4]. Endpoint probing peels x = 4 (16 > 5) and
    /// x = 3 (9 > 5) as infeasible, contracting to x ≤ 2 — soundly (every
    /// removed value violates the constraint) and never below the true feasible
    /// max x = 2.
    #[test]
    fn probe_node_peels_integer_endpoint() {
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let sq = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: x,
            right: x,
        });
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: sq,
                sense: ConstraintSense::Le,
                rhs: 5.0,
                name: None,
            }],
            variables: vec![var("x", VarType::Integer, 1.0, 4.0, 0)],
            n_vars: 1,
        };
        let box_ = vec![Interval::new(1.0, 4.0)];
        let res = probe_node_bounds(&model, &box_, 32, 8, 1e-9, None, None);
        assert!(!res.infeasible);
        assert!(
            res.tightened_bounds[0].hi <= 2.0 + 1e-9,
            "x must be peeled to ≤ 2, got {}",
            res.tightened_bounds[0].hi
        );
        // Soundness: the feasible optimum x = 2 is retained.
        assert!(res.tightened_bounds[0].contains(2.0));
        assert!(
            res.tightened_bounds[0].lo <= 1.0 + 1e-9,
            "lb must not over-tighten"
        );
    }

    /// Both binary branches infeasible ⇒ node infeasible. x = 8 (node-pinned),
    /// x + 10·b ≤ 5 fails for b = 0 (8 > 5) and b = 1 (18 > 5).
    #[test]
    fn probe_node_detects_infeasibility() {
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let b = arena.add(ExprNode::Variable {
            name: "b".into(),
            index: 1,
            size: 1,
            shape: vec![],
        });
        let c10 = arena.add(ExprNode::Constant(10.0));
        let prod = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: c10,
            right: b,
        });
        let sum = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x,
            right: prod,
        });
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: sum,
                sense: ConstraintSense::Le,
                rhs: 5.0,
                name: None,
            }],
            variables: vec![
                var("x", VarType::Continuous, 0.0, 10.0, 0),
                var("b", VarType::Binary, 0.0, 1.0, 1),
            ],
            n_vars: 2,
        };
        let box_ = vec![Interval::new(8.0, 8.0), Interval::new(0.0, 1.0)];
        let res = probe_node_bounds(&model, &box_, 32, 8, 1e-9, None, None);
        assert!(res.infeasible);
    }

    /// A slack node with no active implications: probing must return the box
    /// unchanged (never loosens, records no spurious fixings).
    #[test]
    fn probe_node_is_noop_when_slack() {
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let b = arena.add(ExprNode::Variable {
            name: "b".into(),
            index: 1,
            size: 1,
            shape: vec![],
        });
        let sum = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x,
            right: b,
        });
        // x + b ≤ 100 — never binding on the node box.
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: sum,
                sense: ConstraintSense::Le,
                rhs: 100.0,
                name: None,
            }],
            variables: vec![
                var("x", VarType::Continuous, 0.0, 10.0, 0),
                var("b", VarType::Binary, 0.0, 1.0, 1),
            ],
            n_vars: 2,
        };
        let box_ = vec![Interval::new(0.0, 10.0), Interval::new(0.0, 1.0)];
        let res = probe_node_bounds(&model, &box_, 32, 8, 1e-9, None, None);
        assert!(!res.infeasible);
        assert_eq!(res.n_tightened, 0);
        assert!(res.fixed_vars.is_empty());
        assert!((res.tightened_bounds[0].lo - 0.0).abs() < 1e-12);
        assert!((res.tightened_bounds[0].hi - 10.0).abs() < 1e-12);
        assert!((res.tightened_bounds[1].hi - 1.0).abs() < 1e-12);
    }
}
