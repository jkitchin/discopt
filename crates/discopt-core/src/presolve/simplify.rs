//! Model simplification: Big-M strengthening, integer bound tightening,
//! and redundant constraint removal.

use crate::expr::{
    BinOp, ConstraintSense, ExprArena, ExprId, ExprNode, ModelRepr, UnOp, VarType,
};
use super::fbbt::{forward_propagate, Interval};

/// Result of simplification.
#[derive(Debug, Clone)]
pub struct SimplifyResult {
    /// Number of Big-M coefficients tightened.
    pub bigm_tightened: usize,
    /// Number of integer bounds rounded.
    pub integer_bounds_tightened: usize,
    /// Number of redundant constraints removed.
    pub constraints_removed: usize,
    /// Updated variable bounds.
    pub var_bounds: Vec<Interval>,
    /// Indices of constraints detected as redundant.
    pub redundant_constraints: Vec<usize>,
    /// Big-M detections: (constraint_index, original_M, tightened_M).
    pub bigm_detections: Vec<(usize, f64, f64)>,
}

/// Run simplification passes on the model.
///
/// 1. Integer bound tightening (ceil lb, floor ub for integer vars).
/// 2. Big-M strengthening (detect M*y patterns, tighten M).
/// 3. Redundant constraint removal (always-satisfied constraints).
pub fn simplify(model: &ModelRepr, var_bounds: &mut [Interval]) -> SimplifyResult {
    let mut result = SimplifyResult {
        bigm_tightened: 0,
        integer_bounds_tightened: 0,
        constraints_removed: 0,
        var_bounds: var_bounds.to_vec(),
        redundant_constraints: Vec::new(),
        bigm_detections: Vec::new(),
    };

    // 1. Integer bound tightening.
    for (i, vinfo) in model.variables.iter().enumerate() {
        if i >= var_bounds.len() {
            continue;
        }
        match vinfo.var_type {
            VarType::Integer => {
                let old_lo = var_bounds[i].lo;
                let old_hi = var_bounds[i].hi;
                let new_lo = old_lo.ceil();
                let new_hi = old_hi.floor();
                if new_lo > old_lo || new_hi < old_hi {
                    var_bounds[i] = Interval::new(new_lo, new_hi);
                    result.integer_bounds_tightened += 1;
                }
            }
            VarType::Binary => {
                // Binary variables are already 0/1; clamp to [0, 1].
                let new_lo = var_bounds[i].lo.max(0.0);
                let new_hi = var_bounds[i].hi.min(1.0);
                if new_lo > var_bounds[i].lo || new_hi < var_bounds[i].hi {
                    var_bounds[i] = Interval::new(new_lo, new_hi);
                }
            }
            VarType::Continuous => {}
        }
    }

    // 2. Big-M strengthening.
    for (ci, constr) in model.constraints.iter().enumerate() {
        if constr.sense != ConstraintSense::Le {
            continue;
        }
        if let Some((big_m, _binary_var_idx, other_expr_id)) =
            detect_bigm_pattern(&model.arena, constr.body, &model.variables)
        {
            let node_bounds = forward_propagate(&model.arena, other_expr_id, var_bounds);
            let other_ub = node_bounds[other_expr_id.0].hi;
            if other_ub.is_finite() && other_ub < big_m {
                let tightened_m = other_ub;
                result.bigm_detections.push((ci, big_m, tightened_m));
                result.bigm_tightened += 1;
            }
        }
    }

    // 3. Redundant constraint removal.
    for (ci, constr) in model.constraints.iter().enumerate() {
        let node_bounds = forward_propagate(&model.arena, constr.body, var_bounds);
        let body_interval = node_bounds[constr.body.0];

        let redundant = match constr.sense {
            ConstraintSense::Le => body_interval.hi <= constr.rhs,
            ConstraintSense::Ge => body_interval.lo >= constr.rhs,
            ConstraintSense::Eq => {
                body_interval.lo >= constr.rhs && body_interval.hi <= constr.rhs
            }
        };

        if redundant {
            result.redundant_constraints.push(ci);
            result.constraints_removed += 1;
        }
    }

    result.var_bounds = var_bounds.to_vec();
    result
}

/// Detect the Big-M pattern: constraint body is of the form
/// `expr - M * y` (Le constraint), where y is binary and M is a constant.
///
/// Returns (M_value, binary_var_index, other_expr_id) if detected.
fn detect_bigm_pattern(
    arena: &ExprArena,
    body: ExprId,
    variables: &[crate::expr::VarInfo],
) -> Option<(f64, usize, ExprId)> {
    match arena.get(body) {
        ExprNode::BinaryOp {
            op: BinOp::Sub,
            left,
            right,
        } => {
            // right should be M*y.
            if let Some((m, var_idx)) = detect_const_times_binary(arena, *right, variables) {
                return Some((m, var_idx, *left));
            }
        }
        ExprNode::BinaryOp {
            op: BinOp::Add,
            left,
            right,
        } => {
            // Check if right is Neg(M*y).
            if let ExprNode::UnaryOp {
                op: UnOp::Neg,
                operand,
            } = arena.get(*right)
            {
                if let Some((m, var_idx)) = detect_const_times_binary(arena, *operand, variables) {
                    return Some((m, var_idx, *left));
                }
            }
            // Check for Mul(negative_const, y).
            if let Some((m, var_idx)) = detect_neg_const_times_binary(arena, *right, variables) {
                return Some((m, var_idx, *left));
            }
            // Symmetric: left could be the -M*y part.
            if let ExprNode::UnaryOp {
                op: UnOp::Neg,
                operand,
            } = arena.get(*left)
            {
                if let Some((m, var_idx)) = detect_const_times_binary(arena, *operand, variables) {
                    return Some((m, var_idx, *right));
                }
            }
            if let Some((m, var_idx)) = detect_neg_const_times_binary(arena, *left, variables) {
                return Some((m, var_idx, *right));
            }
        }
        _ => {}
    }
    None
}

/// Detect `M * y` where M is a positive constant and y is binary.
fn detect_const_times_binary(
    arena: &ExprArena,
    id: ExprId,
    variables: &[crate::expr::VarInfo],
) -> Option<(f64, usize)> {
    if let ExprNode::BinaryOp {
        op: BinOp::Mul,
        left,
        right,
    } = arena.get(id)
    {
        if let (Some(m), Some(var_idx)) = (
            arena.try_constant_value_pub(*left),
            get_binary_var_index(arena, *right, variables),
        ) {
            if m > 0.0 {
                return Some((m, var_idx));
            }
        }
        if let (Some(m), Some(var_idx)) = (
            arena.try_constant_value_pub(*right),
            get_binary_var_index(arena, *left, variables),
        ) {
            if m > 0.0 {
                return Some((m, var_idx));
            }
        }
    }
    None
}

/// Detect `(-M) * y` where M is positive and y is binary.
fn detect_neg_const_times_binary(
    arena: &ExprArena,
    id: ExprId,
    variables: &[crate::expr::VarInfo],
) -> Option<(f64, usize)> {
    if let ExprNode::BinaryOp {
        op: BinOp::Mul,
        left,
        right,
    } = arena.get(id)
    {
        if let (Some(m), Some(var_idx)) = (
            arena.try_constant_value_pub(*left),
            get_binary_var_index(arena, *right, variables),
        ) {
            if m < 0.0 {
                return Some((-m, var_idx));
            }
        }
        if let (Some(m), Some(var_idx)) = (
            arena.try_constant_value_pub(*right),
            get_binary_var_index(arena, *left, variables),
        ) {
            if m < 0.0 {
                return Some((-m, var_idx));
            }
        }
    }
    None
}

/// If the expression is a binary variable, return its variable index.
fn get_binary_var_index(
    arena: &ExprArena,
    id: ExprId,
    variables: &[crate::expr::VarInfo],
) -> Option<usize> {
    if let ExprNode::Variable { index, .. } = arena.get(id) {
        if *index < variables.len() && variables[*index].var_type == VarType::Binary {
            return Some(*index);
        }
    }
    None
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::*;
    use super::super::fbbt::Interval;

    #[test]
    fn test_integer_bound_tightening() {
        // x integer, lb=1.3, ub=4.7 => lb=2, ub=4
        let mut arena = ExprArena::new();
        let _x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let model = ModelRepr {
            arena,
            objective: ExprId(0),
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![],
            variables: vec![VarInfo {
                name: "x".into(),
                var_type: VarType::Integer,
                offset: 0,
                size: 1,
                shape: vec![],
                lb: vec![1.3],
                ub: vec![4.7],
            }],
            n_vars: 1,
        };
        let mut var_bounds = vec![Interval::new(1.3, 4.7)];
        let result = simplify(&model, &mut var_bounds);
        assert_eq!(result.integer_bounds_tightened, 1);
        assert!((var_bounds[0].lo - 2.0).abs() < 1e-15);
        assert!((var_bounds[0].hi - 4.0).abs() < 1e-15);
    }

    #[test]
    fn test_integer_bound_already_tight() {
        let mut arena = ExprArena::new();
        let _x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let model = ModelRepr {
            arena,
            objective: ExprId(0),
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![],
            variables: vec![VarInfo {
                name: "x".into(),
                var_type: VarType::Integer,
                offset: 0,
                size: 1,
                shape: vec![],
                lb: vec![2.0],
                ub: vec![4.0],
            }],
            n_vars: 1,
        };
        let mut var_bounds = vec![Interval::new(2.0, 4.0)];
        let result = simplify(&model, &mut var_bounds);
        assert_eq!(result.integer_bounds_tightened, 0);
    }

    #[test]
    fn test_bigm_detection_and_tightening() {
        // x - 100*y <= 0 with x_ub=50, y binary
        // Big-M is 100, could be tightened to 50.
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
        let c100 = arena.add(ExprNode::Constant(100.0));
        let my = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: c100,
            right: y,
        });
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Sub,
            left: x,
            right: my,
        });
        let model = ModelRepr {
            arena,
            objective: ExprId(0),
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 0.0,
                name: Some("bigm".into()),
            }],
            variables: vec![
                VarInfo {
                    name: "x".into(),
                    var_type: VarType::Continuous,
                    offset: 0,
                    size: 1,
                    shape: vec![],
                    lb: vec![0.0],
                    ub: vec![50.0],
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
        let mut var_bounds = vec![Interval::new(0.0, 50.0), Interval::new(0.0, 1.0)];
        let result = simplify(&model, &mut var_bounds);

        assert_eq!(result.bigm_tightened, 1);
        assert_eq!(result.bigm_detections.len(), 1);
        let (ci, orig_m, tight_m) = result.bigm_detections[0];
        assert_eq!(ci, 0);
        assert!((orig_m - 100.0).abs() < 1e-15);
        assert!((tight_m - 50.0).abs() < 1e-15);
    }

    #[test]
    fn test_redundant_constraint_removal() {
        // x <= 10 with x in [0, 5] => always satisfied => redundant.
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let model = ModelRepr {
            arena,
            objective: ExprId(0),
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: x,
                sense: ConstraintSense::Le,
                rhs: 10.0,
                name: Some("c1".into()),
            }],
            variables: vec![VarInfo {
                name: "x".into(),
                var_type: VarType::Continuous,
                offset: 0,
                size: 1,
                shape: vec![],
                lb: vec![0.0],
                ub: vec![5.0],
            }],
            n_vars: 1,
        };
        let mut var_bounds = vec![Interval::new(0.0, 5.0)];
        let result = simplify(&model, &mut var_bounds);

        assert_eq!(result.constraints_removed, 1);
        assert_eq!(result.redundant_constraints, vec![0]);
    }

    #[test]
    fn test_non_redundant_constraint() {
        // x <= 3 with x in [0, 5] => not always satisfied.
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let model = ModelRepr {
            arena,
            objective: ExprId(0),
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: x,
                sense: ConstraintSense::Le,
                rhs: 3.0,
                name: Some("c1".into()),
            }],
            variables: vec![VarInfo {
                name: "x".into(),
                var_type: VarType::Continuous,
                offset: 0,
                size: 1,
                shape: vec![],
                lb: vec![0.0],
                ub: vec![5.0],
            }],
            n_vars: 1,
        };
        let mut var_bounds = vec![Interval::new(0.0, 5.0)];
        let result = simplify(&model, &mut var_bounds);

        assert_eq!(result.constraints_removed, 0);
    }

    #[test]
    fn test_ge_redundant_constraint() {
        // x >= 0 with x in [5, 10] => always satisfied.
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let model = ModelRepr {
            arena,
            objective: ExprId(0),
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: x,
                sense: ConstraintSense::Ge,
                rhs: 0.0,
                name: Some("c1".into()),
            }],
            variables: vec![VarInfo {
                name: "x".into(),
                var_type: VarType::Continuous,
                offset: 0,
                size: 1,
                shape: vec![],
                lb: vec![5.0],
                ub: vec![10.0],
            }],
            n_vars: 1,
        };
        let mut var_bounds = vec![Interval::new(5.0, 10.0)];
        let result = simplify(&model, &mut var_bounds);

        assert_eq!(result.constraints_removed, 1);
    }

    #[test]
    fn test_multiple_integer_vars() {
        let mut arena = ExprArena::new();
        let _x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let _y = arena.add(ExprNode::Variable {
            name: "y".into(),
            index: 1,
            size: 1,
            shape: vec![],
        });
        let model = ModelRepr {
            arena,
            objective: ExprId(0),
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![],
            variables: vec![
                VarInfo {
                    name: "x".into(),
                    var_type: VarType::Integer,
                    offset: 0,
                    size: 1,
                    shape: vec![],
                    lb: vec![0.5],
                    ub: vec![3.8],
                },
                VarInfo {
                    name: "y".into(),
                    var_type: VarType::Integer,
                    offset: 1,
                    size: 1,
                    shape: vec![],
                    lb: vec![2.0],
                    ub: vec![5.0],
                },
            ],
            n_vars: 2,
        };
        let mut var_bounds = vec![Interval::new(0.5, 3.8), Interval::new(2.0, 5.0)];
        let result = simplify(&model, &mut var_bounds);

        assert_eq!(result.integer_bounds_tightened, 1);
        assert!((var_bounds[0].lo - 1.0).abs() < 1e-15);
        assert!((var_bounds[0].hi - 3.0).abs() < 1e-15);
        assert!((var_bounds[1].lo - 2.0).abs() < 1e-15);
        assert!((var_bounds[1].hi - 5.0).abs() < 1e-15);
    }

    #[test]
    fn test_combined_simplifications() {
        // x integer lb=1.3 ub=4.7, y binary
        // constraint: x <= 10 (redundant after tightening since x_ub=4)
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let _y = arena.add(ExprNode::Variable {
            name: "y".into(),
            index: 1,
            size: 1,
            shape: vec![],
        });
        let model = ModelRepr {
            arena,
            objective: ExprId(0),
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: x,
                sense: ConstraintSense::Le,
                rhs: 10.0,
                name: Some("c1".into()),
            }],
            variables: vec![
                VarInfo {
                    name: "x".into(),
                    var_type: VarType::Integer,
                    offset: 0,
                    size: 1,
                    shape: vec![],
                    lb: vec![1.3],
                    ub: vec![4.7],
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
        let mut var_bounds = vec![Interval::new(1.3, 4.7), Interval::new(0.0, 1.0)];
        let result = simplify(&model, &mut var_bounds);

        assert_eq!(result.integer_bounds_tightened, 1);
        assert!((var_bounds[0].lo - 2.0).abs() < 1e-15);
        assert!((var_bounds[0].hi - 4.0).abs() < 1e-15);

        assert_eq!(result.constraints_removed, 1);
    }
}
