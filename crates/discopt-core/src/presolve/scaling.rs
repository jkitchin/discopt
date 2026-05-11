//! Row/column equilibration scaling (E1 of the presolve roadmap).
//!
//! ## What this pass does
//!
//! Computes Curtis–Reid-style geometric-mean scale factors for the
//! linear part of the model and stores them on the pass delta so
//! downstream solvers can apply them consistently. The pass itself
//! does not rewrite the model — it only emits the numbers.
//!
//! For each linear constraint `Σ a_ij x_j  ⊙  b_i`, define
//!
//! ```text
//!     row_scale[i]  = 1 / sqrt(max_j |a_ij| · min_{j: a_ij ≠ 0} |a_ij|)
//!     col_scale[j]  = 1 / sqrt(max_i |a_ij| · min_{i: a_ij ≠ 0} |a_ij|)
//! ```
//!
//! Constraints with no linear part contribute `1.0` (identity scale).
//! Variables that appear in no linear constraint contribute `1.0`.
//!
//! ## Why this is a presolve pass
//!
//! Today, every solver (LP, NLP, IPM) computes its own scaling and
//! they sometimes disagree, which costs cache and cross-checks. By
//! pinning the scaling decision in presolve and surfacing it in the
//! delta, every downstream solver can apply the same factors. The
//! actual application is deferred to the solver — the pass is purely
//! diagnostic.
//!
//! ## Reference
//!
//! Curtis & Reid (1972), *On the automatic scaling of matrices for
//! Gaussian elimination*. The geometric-mean balance is the simplest
//! choice that handles wide dynamic ranges; richer iterative schemes
//! (Sinkhorn, Knight–Ruiz) are a future replacement.
//!
//! ## Determinism
//!
//! Constraints and variables are scanned in their natural order; min
//! / max are taken over `f64` with explicit handling of zeros. No
//! `HashMap` iteration on the hot path.

use super::polynomial::try_polynomial;
use crate::expr::ModelRepr;

/// Per-pass scaling diagnostics.
#[derive(Debug, Clone, Default)]
pub struct ScalingStats {
    /// Number of linear constraints whose coefficients were sampled.
    pub linear_rows_sampled: usize,
    /// Largest ratio (max / min) observed in any single row before
    /// scaling. A useful single-number diagnostic for badly scaled
    /// inputs.
    pub worst_row_dynamic_range: f64,
    /// Largest ratio (max / min) observed in any single column before
    /// scaling.
    pub worst_col_dynamic_range: f64,
}

/// Scale-factor result of running [`compute_equilibration`].
#[derive(Debug, Clone, Default)]
pub struct ScalingFactors {
    /// One scale per constraint. Identity (`1.0`) for non-linear or
    /// empty rows.
    pub row_scales: Vec<f64>,
    /// One scale per variable block. Identity (`1.0`) for variables
    /// that appear in no linear constraint.
    pub col_scales: Vec<f64>,
}

/// Compute equilibration factors. Pure function; does not mutate the
/// model. Returns the factors and a `ScalingStats` summary.
pub fn compute_equilibration(model: &ModelRepr) -> (ScalingFactors, ScalingStats) {
    let n_rows = model.constraints.len();
    let n_cols = model.variables.len();
    let mut row_max = vec![0.0_f64; n_rows];
    let mut row_min = vec![f64::INFINITY; n_rows];
    let mut col_max = vec![0.0_f64; n_cols];
    let mut col_min = vec![f64::INFINITY; n_cols];
    let mut row_has_entry = vec![false; n_rows];
    let mut col_has_entry = vec![false; n_cols];
    let mut stats = ScalingStats::default();

    for (i, c) in model.constraints.iter().enumerate() {
        let poly = match try_polynomial(&model.arena, c.body) {
            Some(p) => p,
            None => continue,
        };
        if poly.max_total_degree() > 1 {
            continue;
        }
        let mut any = false;
        for m in &poly.monomials {
            if m.factors.len() != 1 || m.factors[0].1 != 1 {
                continue;
            }
            let leaf = m.factors[0].0;
            // Resolve leaf to a column index via Variable.index.
            let col = match model.arena.get(leaf) {
                crate::expr::ExprNode::Variable { index, .. } => *index,
                crate::expr::ExprNode::Index { base, .. } => match model.arena.get(*base) {
                    crate::expr::ExprNode::Variable { index, .. } => *index,
                    _ => continue,
                },
                _ => continue,
            };
            let a = m.coeff.abs();
            if a <= 1e-15 {
                continue;
            }
            any = true;
            row_has_entry[i] = true;
            row_max[i] = row_max[i].max(a);
            row_min[i] = row_min[i].min(a);
            if col < n_cols {
                col_has_entry[col] = true;
                col_max[col] = col_max[col].max(a);
                col_min[col] = col_min[col].min(a);
            }
        }
        if any {
            stats.linear_rows_sampled += 1;
        }
    }

    let mut factors = ScalingFactors {
        row_scales: vec![1.0; n_rows],
        col_scales: vec![1.0; n_cols],
    };

    for i in 0..n_rows {
        if !row_has_entry[i] {
            continue;
        }
        let lo = row_min[i].max(1e-300);
        let hi = row_max[i];
        let dyn_range = hi / lo;
        if dyn_range.is_finite() && dyn_range > stats.worst_row_dynamic_range {
            stats.worst_row_dynamic_range = dyn_range;
        }
        let g = (lo * hi).sqrt();
        if g > 0.0 && g.is_finite() {
            factors.row_scales[i] = 1.0 / g;
        }
    }
    for j in 0..n_cols {
        if !col_has_entry[j] {
            continue;
        }
        let lo = col_min[j].max(1e-300);
        let hi = col_max[j];
        let dyn_range = hi / lo;
        if dyn_range.is_finite() && dyn_range > stats.worst_col_dynamic_range {
            stats.worst_col_dynamic_range = dyn_range;
        }
        let g = (lo * hi).sqrt();
        if g > 0.0 && g.is_finite() {
            factors.col_scales[j] = 1.0 / g;
        }
    }

    (factors, stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{
        BinOp, ConstraintRepr, ConstraintSense, ExprArena, ExprNode, ModelRepr, ObjectiveSense,
        VarInfo, VarType,
    };

    fn scalar_var(arena: &mut ExprArena, name: &str, idx: usize) -> ExprId {
        arena.add(ExprNode::Variable {
            name: name.into(),
            index: idx,
            size: 1,
            shape: vec![],
        })
    }

    use crate::expr::ExprId;

    fn vinfo(name: &str, offset: usize) -> VarInfo {
        VarInfo {
            name: name.into(),
            var_type: VarType::Continuous,
            offset,
            size: 1,
            shape: vec![],
            lb: vec![0.0],
            ub: vec![1.0],
        }
    }

    fn lin(arena: &mut ExprArena, c: f64, var: ExprId) -> ExprId {
        let cn = arena.add(ExprNode::Constant(c));
        arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: cn,
            right: var,
        })
    }

    fn add(arena: &mut ExprArena, a: ExprId, b: ExprId) -> ExprId {
        arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: a,
            right: b,
        })
    }

    /// `100 x + 0.01 y ≤ 1`: row dynamic range is 100 / 0.01 = 1e4.
    /// row_scale = 1 / sqrt(100 * 0.01) = 1 / sqrt(1) = 1.
    #[test]
    fn balanced_row_unit_scale() {
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let body = {
            let a = lin(&mut arena, 100.0, x);
            let b = lin(&mut arena, 0.01, y);
            add(&mut arena, a, b)
        };
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 1.0,
                name: None,
            }],
            variables: vec![vinfo("x", 0), vinfo("y", 1)],
            n_vars: 2,
        };
        let (f, s) = compute_equilibration(&model);
        assert_eq!(f.row_scales.len(), 1);
        assert!(
            (f.row_scales[0] - 1.0).abs() < 1e-9,
            "row = {}",
            f.row_scales[0]
        );
        assert!((s.worst_row_dynamic_range - 1e4).abs() < 1e-3);
        assert_eq!(s.linear_rows_sampled, 1);
    }

    /// One row, one variable: `4 x ≤ 1`. Column scale should be
    /// `1 / sqrt(4 * 4) = 0.25`.
    #[test]
    fn single_term_col_scale() {
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let body = lin(&mut arena, 4.0, x);
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 1.0,
                name: None,
            }],
            variables: vec![vinfo("x", 0)],
            n_vars: 1,
        };
        let (f, _) = compute_equilibration(&model);
        assert!((f.col_scales[0] - 0.25).abs() < 1e-9);
        assert!((f.row_scales[0] - 0.25).abs() < 1e-9);
    }

    /// Variable not used in any linear row gets identity scale.
    #[test]
    fn unused_var_identity() {
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let _y = scalar_var(&mut arena, "y", 1);
        let body = lin(&mut arena, 2.0, x);
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 1.0,
                name: None,
            }],
            variables: vec![vinfo("x", 0), vinfo("y", 1)],
            n_vars: 2,
        };
        let (f, _) = compute_equilibration(&model);
        assert_eq!(f.col_scales[1], 1.0);
    }

    /// Empty / nonlinear model: every row scale stays identity.
    #[test]
    fn nonlinear_row_skipped() {
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let two = arena.add(ExprNode::Constant(2.0));
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Pow,
            left: x,
            right: two,
        });
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 5.0,
                name: None,
            }],
            variables: vec![vinfo("x", 0)],
            n_vars: 1,
        };
        let (f, s) = compute_equilibration(&model);
        assert_eq!(f.row_scales[0], 1.0);
        assert_eq!(s.linear_rows_sampled, 0);
    }
}
