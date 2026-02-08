//! Optimality-Based Bound Tightening (OBBT).
//!
//! For each variable x_i, solves two LPs:
//!   min x_i  subject to LP relaxation  -> tightened lower bound
//!   max x_i  subject to LP relaxation  -> tightened upper bound
//!
//! This module provides:
//! - Linear coefficient extraction from the expression DAG
//! - OBBT result types
//! - Filtering logic to skip variables unlikely to benefit

use crate::expr::{BinOp, ConstraintSense, ExprArena, ExprId, ExprNode, ModelRepr, UnOp};
use super::fbbt::Interval;

// ─────────────────────────────────────────────────────────────
// Linear row extraction
// ─────────────────────────────────────────────────────────────

/// A linear constraint row: sum(coeffs[i] * x[var_indices[i]]) sense rhs.
#[derive(Debug, Clone)]
pub struct LinearRow {
    /// Variable indices (into the flat variable vector).
    pub var_indices: Vec<usize>,
    /// Corresponding coefficients.
    pub coeffs: Vec<f64>,
    /// Constant offset (moved to rhs).
    pub offset: f64,
    /// Constraint sense.
    pub sense: ConstraintSense,
    /// Right-hand side (adjusted for offset).
    pub rhs: f64,
}

/// Extract linear coefficients from an expression.
///
/// Returns `Some((coeffs_map, offset))` where `coeffs_map` maps
/// variable index -> coefficient. Returns `None` if the expression
/// is not linear.
fn extract_linear_coeffs(
    arena: &ExprArena,
    id: ExprId,
) -> Option<(Vec<(usize, f64)>, f64)> {
    match arena.get(id) {
        ExprNode::Constant(v) => Some((vec![], *v)),
        ExprNode::ConstantArray(data, _) => {
            if data.len() == 1 {
                Some((vec![], data[0]))
            } else {
                None
            }
        }
        ExprNode::Variable { index, size, .. } => {
            if *size == 1 {
                Some((vec![(*index, 1.0)], 0.0))
            } else {
                None // Array variables need indexing first
            }
        }
        ExprNode::Parameter { value, .. } => {
            if value.len() == 1 {
                Some((vec![], value[0]))
            } else {
                None
            }
        }
        ExprNode::BinaryOp { op, left, right } => {
            match op {
                BinOp::Add => {
                    let (mut lc, lo) = extract_linear_coeffs(arena, *left)?;
                    let (rc, ro) = extract_linear_coeffs(arena, *right)?;
                    for (idx, coeff) in rc {
                        if let Some(entry) = lc.iter_mut().find(|(i, _)| *i == idx) {
                            entry.1 += coeff;
                        } else {
                            lc.push((idx, coeff));
                        }
                    }
                    Some((lc, lo + ro))
                }
                BinOp::Sub => {
                    let (mut lc, lo) = extract_linear_coeffs(arena, *left)?;
                    let (rc, ro) = extract_linear_coeffs(arena, *right)?;
                    for (idx, coeff) in rc {
                        if let Some(entry) = lc.iter_mut().find(|(i, _)| *i == idx) {
                            entry.1 -= coeff;
                        } else {
                            lc.push((idx, -coeff));
                        }
                    }
                    Some((lc, lo - ro))
                }
                BinOp::Mul => {
                    let l = extract_linear_coeffs(arena, *left);
                    let r = extract_linear_coeffs(arena, *right);
                    match (l, r) {
                        (Some((lc, lo)), Some((rc, ro))) => {
                            // c * expr or expr * c
                            if lc.is_empty() {
                                // Left is constant
                                let scale = lo;
                                let coeffs: Vec<(usize, f64)> =
                                    rc.iter().map(|(i, c)| (*i, c * scale)).collect();
                                Some((coeffs, scale * ro))
                            } else if rc.is_empty() {
                                // Right is constant
                                let scale = ro;
                                let coeffs: Vec<(usize, f64)> =
                                    lc.iter().map(|(i, c)| (*i, c * scale)).collect();
                                Some((coeffs, lo * scale))
                            } else {
                                // Both sides have variables — not linear
                                None
                            }
                        }
                        _ => None,
                    }
                }
                BinOp::Div => {
                    let l = extract_linear_coeffs(arena, *left)?;
                    let r = extract_linear_coeffs(arena, *right)?;
                    if r.0.is_empty() && r.1.abs() > 1e-30 {
                        // Division by constant
                        let scale = 1.0 / r.1;
                        let coeffs: Vec<(usize, f64)> =
                            l.0.iter().map(|(i, c)| (*i, c * scale)).collect();
                        Some((coeffs, l.1 * scale))
                    } else {
                        None
                    }
                }
                BinOp::Pow => None, // Not linear
            }
        }
        ExprNode::UnaryOp { op, operand } => {
            match op {
                UnOp::Neg => {
                    let (coeffs, offset) = extract_linear_coeffs(arena, *operand)?;
                    let neg_coeffs: Vec<(usize, f64)> =
                        coeffs.iter().map(|(i, c)| (*i, -c)).collect();
                    Some((neg_coeffs, -offset))
                }
                UnOp::Abs => None, // Not linear
            }
        }
        ExprNode::Index { base, index } => {
            // For indexed variables, extract the specific variable element
            match arena.get(*base) {
                ExprNode::Variable { index: var_idx, .. } => {
                    let offset = match index {
                        crate::expr::IndexSpec::Scalar(i) => *i,
                        crate::expr::IndexSpec::Tuple(indices) => {
                            // For now, handle 1D indexing only
                            if indices.len() == 1 {
                                indices[0]
                            } else {
                                return None;
                            }
                        }
                    };
                    Some((vec![(*var_idx + offset, 1.0)], 0.0))
                }
                _ => None,
            }
        }
        ExprNode::Sum { operand, .. } => {
            // Sum of a linear expression is linear
            extract_linear_coeffs(arena, *operand)
        }
        ExprNode::SumOver { terms } => {
            let mut all_coeffs: Vec<(usize, f64)> = vec![];
            let mut total_offset = 0.0;
            for t in terms {
                let (coeffs, offset) = extract_linear_coeffs(arena, *t)?;
                total_offset += offset;
                for (idx, coeff) in coeffs {
                    if let Some(entry) = all_coeffs.iter_mut().find(|(i, _)| *i == idx) {
                        entry.1 += coeff;
                    } else {
                        all_coeffs.push((idx, coeff));
                    }
                }
            }
            Some((all_coeffs, total_offset))
        }
        ExprNode::FunctionCall { .. } | ExprNode::MatMul { .. } => None,
    }
}

/// Extract linear constraints from a model.
///
/// Returns a vector of `LinearRow` for each constraint that is linear.
/// Non-linear constraints are skipped.
pub fn extract_linear_rows(model: &ModelRepr) -> Vec<LinearRow> {
    let mut rows = Vec::new();
    for constr in &model.constraints {
        if !model.arena.is_linear(constr.body) {
            continue;
        }
        if let Some((coeffs_map, offset)) = extract_linear_coeffs(&model.arena, constr.body) {
            let var_indices: Vec<usize> = coeffs_map.iter().map(|(i, _)| *i).collect();
            let coeffs: Vec<f64> = coeffs_map.iter().map(|(_, c)| *c).collect();
            rows.push(LinearRow {
                var_indices,
                coeffs,
                offset,
                sense: constr.sense,
                rhs: constr.rhs - offset, // Move offset to rhs
            });
        }
    }
    rows
}

// ─────────────────────────────────────────────────────────────
// OBBT result
// ─────────────────────────────────────────────────────────────

/// Result of running OBBT on a model.
#[derive(Debug, Clone)]
pub struct ObbtResult {
    /// Tightened variable bounds (indexed by variable block index).
    pub tightened_bounds: Vec<Interval>,
    /// Number of LP solves performed.
    pub n_lp_solves: usize,
    /// Number of bounds tightened.
    pub n_tightened: usize,
    /// Total LP solve time in seconds.
    pub total_lp_time: f64,
}

/// Determine which variables are candidates for OBBT.
///
/// Filters out variables with already-tight bounds (width < tol)
/// and fixed variables.
pub fn obbt_candidates(
    var_bounds: &[Interval],
    min_width: f64,
) -> Vec<usize> {
    var_bounds
        .iter()
        .enumerate()
        .filter(|(_, b)| {
            !b.is_empty()
                && b.width() > min_width
                && b.lo.is_finite()
                && b.hi.is_finite()
        })
        .map(|(i, _)| i)
        .collect()
}

/// Apply OBBT results to tighten variable bounds.
///
/// For each candidate variable, updates the bound if the LP-based bound
/// is tighter than the current bound.
pub fn apply_obbt_bounds(
    var_bounds: &mut [Interval],
    candidates: &[usize],
    lb_results: &[Option<f64>],
    ub_results: &[Option<f64>],
) -> usize {
    let mut n_tightened = 0;
    for (k, &var_idx) in candidates.iter().enumerate() {
        if var_idx >= var_bounds.len() {
            continue;
        }
        if let Some(new_lb) = lb_results[k] {
            if new_lb > var_bounds[var_idx].lo + 1e-8 {
                var_bounds[var_idx].lo = new_lb;
                n_tightened += 1;
            }
        }
        if let Some(new_ub) = ub_results[k] {
            if new_ub < var_bounds[var_idx].hi - 1e-8 {
                var_bounds[var_idx].hi = new_ub;
                n_tightened += 1;
            }
        }
    }
    n_tightened
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::*;

    fn make_linear_model_2var() -> ModelRepr {
        // x + 2y <= 10, 3x + y <= 12, x,y >= 0
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

        // x + 2*y
        let c2 = arena.add(ExprNode::Constant(2.0));
        let two_y = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: c2,
            right: y,
        });
        let sum1 = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x,
            right: two_y,
        });

        // 3*x + y
        let c3 = arena.add(ExprNode::Constant(3.0));
        let three_x = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: c3,
            right: x,
        });
        let sum2 = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: three_x,
            right: y,
        });

        ModelRepr {
            arena,
            objective: ExprId(0),
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![
                ConstraintRepr {
                    body: sum1,
                    sense: ConstraintSense::Le,
                    rhs: 10.0,
                    name: Some("c1".into()),
                },
                ConstraintRepr {
                    body: sum2,
                    sense: ConstraintSense::Le,
                    rhs: 12.0,
                    name: Some("c2".into()),
                },
            ],
            variables: vec![
                VarInfo {
                    name: "x".into(),
                    var_type: VarType::Continuous,
                    offset: 0,
                    size: 1,
                    shape: vec![],
                    lb: vec![0.0],
                    ub: vec![100.0],
                },
                VarInfo {
                    name: "y".into(),
                    var_type: VarType::Continuous,
                    offset: 1,
                    size: 1,
                    shape: vec![],
                    lb: vec![0.0],
                    ub: vec![100.0],
                },
            ],
            n_vars: 2,
        }
    }

    #[test]
    fn test_extract_linear_rows() {
        let model = make_linear_model_2var();
        let rows = extract_linear_rows(&model);

        assert_eq!(rows.len(), 2);

        // First row: x + 2*y <= 10
        assert_eq!(rows[0].var_indices.len(), 2);
        assert!((rows[0].rhs - 10.0).abs() < 1e-10);

        // Second row: 3*x + y <= 12
        assert_eq!(rows[1].var_indices.len(), 2);
        assert!((rows[1].rhs - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_extract_linear_coeffs_constant() {
        let mut arena = ExprArena::new();
        let c = arena.add(ExprNode::Constant(5.0));
        let result = extract_linear_coeffs(&arena, c).unwrap();
        assert!(result.0.is_empty());
        assert!((result.1 - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_extract_linear_coeffs_variable() {
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let result = extract_linear_coeffs(&arena, x).unwrap();
        assert_eq!(result.0.len(), 1);
        assert_eq!(result.0[0].0, 0);
        assert!((result.0[0].1 - 1.0).abs() < 1e-10);
        assert!((result.1 - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_extract_linear_coeffs_scaled_var() {
        let mut arena = ExprArena::new();
        let c = arena.add(ExprNode::Constant(3.0));
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let prod = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: c,
            right: x,
        });
        let result = extract_linear_coeffs(&arena, prod).unwrap();
        assert_eq!(result.0.len(), 1);
        assert_eq!(result.0[0].0, 0);
        assert!((result.0[0].1 - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_extract_linear_coeffs_sum() {
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
        let result = extract_linear_coeffs(&arena, sum).unwrap();
        assert_eq!(result.0.len(), 2);
    }

    #[test]
    fn test_extract_linear_nonlinear_returns_none() {
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        // x * x (quadratic, not linear)
        let prod = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: x,
            right: x,
        });
        let result = extract_linear_coeffs(&arena, prod);
        assert!(result.is_none());
    }

    #[test]
    fn test_obbt_candidates() {
        let bounds = vec![
            Interval::new(0.0, 100.0),
            Interval::new(5.0, 5.0),       // Fixed
            Interval::new(0.0, 50.0),
            Interval::new(f64::NEG_INFINITY, f64::INFINITY), // Infinite
        ];
        let candidates = obbt_candidates(&bounds, 1e-6);
        // Should include index 0 and 2 (finite, non-fixed)
        assert_eq!(candidates, vec![0, 2]);
    }

    #[test]
    fn test_apply_obbt_bounds_tightens() {
        let mut bounds = vec![
            Interval::new(0.0, 100.0),
            Interval::new(0.0, 100.0),
        ];
        let candidates = vec![0, 1];
        let lb_results = vec![Some(5.0), Some(2.0)];
        let ub_results = vec![Some(80.0), Some(50.0)];

        let n = apply_obbt_bounds(&mut bounds, &candidates, &lb_results, &ub_results);
        assert_eq!(n, 4); // All four bounds tightened
        assert!((bounds[0].lo - 5.0).abs() < 1e-10);
        assert!((bounds[0].hi - 80.0).abs() < 1e-10);
        assert!((bounds[1].lo - 2.0).abs() < 1e-10);
        assert!((bounds[1].hi - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_obbt_bounds_no_tightening() {
        let mut bounds = vec![
            Interval::new(5.0, 80.0),
        ];
        let candidates = vec![0];
        // LP results are looser than current bounds
        let lb_results = vec![Some(3.0)];
        let ub_results = vec![Some(90.0)];

        let n = apply_obbt_bounds(&mut bounds, &candidates, &lb_results, &ub_results);
        assert_eq!(n, 0);
        // Bounds unchanged
        assert!((bounds[0].lo - 5.0).abs() < 1e-10);
        assert!((bounds[0].hi - 80.0).abs() < 1e-10);
    }

    #[test]
    fn test_extract_linear_with_neg() {
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let neg_x = arena.add(ExprNode::UnaryOp {
            op: UnOp::Neg,
            operand: x,
        });
        let result = extract_linear_coeffs(&arena, neg_x).unwrap();
        assert_eq!(result.0.len(), 1);
        assert!((result.0[0].1 - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_extract_linear_with_offset() {
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let c5 = arena.add(ExprNode::Constant(5.0));
        let sum = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x,
            right: c5,
        });
        let result = extract_linear_coeffs(&arena, sum).unwrap();
        assert_eq!(result.0.len(), 1);
        assert!((result.0[0].1 - 1.0).abs() < 1e-10);
        assert!((result.1 - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_row_rhs_adjusted_for_offset() {
        // Constraint: x + 5 <= 10 should give rhs = 5 (10 - 5)
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let c5 = arena.add(ExprNode::Constant(5.0));
        let sum = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x,
            right: c5,
        });
        let model = ModelRepr {
            arena,
            objective: ExprId(0),
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: sum,
                sense: ConstraintSense::Le,
                rhs: 10.0,
                name: None,
            }],
            variables: vec![VarInfo {
                name: "x".into(),
                var_type: VarType::Continuous,
                offset: 0,
                size: 1,
                shape: vec![],
                lb: vec![0.0],
                ub: vec![100.0],
            }],
            n_vars: 1,
        };
        let rows = extract_linear_rows(&model);
        assert_eq!(rows.len(), 1);
        assert!((rows[0].rhs - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_extract_sum_over() {
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
        let c3 = arena.add(ExprNode::Constant(3.0));
        let sum_over = arena.add(ExprNode::SumOver {
            terms: vec![x, y, c3],
        });
        let result = extract_linear_coeffs(&arena, sum_over).unwrap();
        assert_eq!(result.0.len(), 2);
        assert!((result.1 - 3.0).abs() < 1e-10);
    }
}
