//! Binary variable probing.
//!
//! For each binary variable, temporarily fix it to 0 and 1, run FBBT,
//! and use the results to detect fixings, tightened bounds, and implications.

use super::fbbt::{fbbt, Interval};
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
    let n = var_bounds.len();
    let mut fixed_vars: Vec<(usize, f64)> = Vec::new();
    let mut best_bounds = var_bounds.to_vec();
    let mut implications: Vec<Implication> = Vec::new();

    let max_fbbt_iter = 5;
    let fbbt_tol = 1e-8;

    for (i, vinfo) in model.variables.iter().enumerate() {
        if vinfo.var_type != VarType::Binary {
            continue;
        }

        // Probe y=0.
        let mut bounds_zero = var_bounds.to_vec();
        bounds_zero[i] = Interval::new(0.0, 0.0);
        let model_zero = make_probing_model(model, &bounds_zero);
        let result_zero = fbbt(&model_zero, max_fbbt_iter, fbbt_tol);
        let infeasible_zero = result_zero.iter().any(|b| b.is_empty());

        // Probe y=1.
        let mut bounds_one = var_bounds.to_vec();
        bounds_one[i] = Interval::new(1.0, 1.0);
        let model_one = make_probing_model(model, &bounds_one);
        let result_one = fbbt(&model_one, max_fbbt_iter, fbbt_tol);
        let infeasible_one = result_one.iter().any(|b| b.is_empty());

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
}
