//! Row/column equilibration scaling (E1 of the presolve roadmap).
//!
//! ## What this pass does
//!
//! Computes Curtis–Reid-style geometric-mean scale factors for the
//! linear part of the model and stores them on the pass delta. The pass
//! itself does not rewrite the model — it only emits the numbers.
//!
//! **What actually consumes this, as of the diagnostics change.** The two
//! *dynamic ranges* have a consumer: they ride the delta out to Python and
//! are read directly by `PyModelRepr::scaling_diagnostics`, which backs the
//! once-per-solve "badly scaled model" warning in `solver.py`. The *scale
//! factors* still have none — no LP, NLP or IPM path reads `row_scales` /
//! `col_scales` off the delta, and the `scaling` pass is off by default in
//! `_relax/presolve_pipeline.py`. This paragraph used to assert the
//! opposite ("downstream solvers can apply them consistently", below, as if
//! it were shipped behaviour); it was aspirational when written and was
//! never true. Applying the factors changes the LP the solver sees, so it
//! is a bound-changing change under CLAUDE.md §5 and needs a differential
//! panel before any path may consume them. Do not read the "Why this is a
//! presolve pass" section below as a description of what happens today.
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
//! ## Work budget (#1686)
//!
//! Each row is expanded with [`try_polynomial_budgeted`], not the unbounded
//! [`super::polynomial::try_polynomial`]. This runs before the solve's time
//! limit (from `solver.py`'s `_check_model_scaling`), and a row that is a
//! product of `k` width-`w` sums expands to `w^k` monomials only to be
//! discarded for having degree > 1: measured, a 9 x 7 product held a
//! `time_limit=5` solve for ~115 s before branch and bound started. The
//! budget is an operation count (never a clock, #912), in two currencies
//! (see [`PolyBudget`]):
//!
//! - *expansion* (cross-products of non-constant polynomials):
//!   [`ROW_EXPANSION_BUDGET`] per row and [`MODEL_EXPANSION_BUDGET`] for the
//!   whole pass, so many individually-affordable rows cannot add up to an
//!   unbounded total;
//! - *linear* (node visits, leaf monomials, constant-factor copies):
//!   proportional to the arena size, per row and per pass. An unshared
//!   linear row spends roughly its own node count, so this only bites on
//!   shared subtrees re-walked as a tree.
//!
//! A row that exhausts either is **abstained on** exactly like a
//! non-polynomial row: it contributes nothing and keeps identity scale. It
//! is counted in [`ScalingStats::rows_over_budget`] so the diagnostic says
//! it did not look, rather than reading as "this row is fine". Within budget
//! the result is bit-identical to the unbudgeted expansion.
//!
//! ## Determinism
//!
//! Constraints and variables are scanned in their natural order; min
//! / max are taken over `f64` with explicit handling of zeros. No
//! `HashMap` iteration on the hot path.

use super::polynomial::{try_polynomial_budgeted, PolyBudget, PolyBudgetedError};
use crate::expr::ModelRepr;

/// Expansion units (monomials created by multiplying two non-constant
/// polynomials) one row may spend before it is abstained on. A row that can
/// contribute to equilibration has degree ≤ 1, which needs no genuine
/// expansion unless products cancel; this leaves ample room for that.
pub const ROW_EXPANSION_BUDGET: u64 = 200_000;

/// Expansion units the whole pass may spend across all rows (#1456's lesson:
/// a per-row limit alone bounds nothing over many rows).
pub const MODEL_EXPANSION_BUDGET: u64 = 2_000_000;

/// Linear allowance per row is `LINEAR_ARENA_FACTOR * arena.len() +
/// LINEAR_BASE`; for the pass it is four times that.
pub const LINEAR_ARENA_FACTOR: u64 = 4;
/// See [`LINEAR_ARENA_FACTOR`].
pub const LINEAR_BASE: u64 = 100_000;

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
    /// Index of the constraint attaining `worst_row_dynamic_range`,
    /// and of the variable attaining `worst_col_dynamic_range`.
    /// `None` when no linear row (resp. column) was sampled.
    ///
    /// The ranges alone say "something in this model is badly scaled"
    /// without saying *what*, which a user cannot act on. The maxima
    /// are taken in a loop that already knows the index, so recording
    /// it is free and turns the number into a pointer at the offending
    /// row or column.
    pub worst_row_index: Option<usize>,
    /// See [`Self::worst_row_index`].
    pub worst_col_index: Option<usize>,
    /// Rows whose polynomial expansion exceeded the work budget and were
    /// therefore not examined (#1686). Nonzero means the reported ranges
    /// cover only the rows that were.
    pub rows_over_budget: usize,
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

    let arena_len = model.arena.len() as u64;
    let row_linear = LINEAR_ARENA_FACTOR
        .saturating_mul(arena_len)
        .saturating_add(LINEAR_BASE);
    let mut pass_linear = row_linear.saturating_mul(4);
    let mut pass_expansion = MODEL_EXPANSION_BUDGET;

    for (i, c) in model.constraints.iter().enumerate() {
        let budget = PolyBudget {
            linear: row_linear.min(pass_linear),
            expansion: ROW_EXPANSION_BUDGET.min(pass_expansion),
        };
        let (res, spent) = try_polynomial_budgeted(&model.arena, c.body, budget);
        pass_linear -= spent.linear;
        pass_expansion -= spent.expansion;
        let poly = match res {
            Ok(p) => p,
            Err(PolyBudgetedError::NotPolynomial) => continue,
            Err(PolyBudgetedError::BudgetExhausted) => {
                stats.rows_over_budget += 1;
                continue;
            }
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
            stats.worst_row_index = Some(i);
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
            stats.worst_col_index = Some(j);
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

    /// The worst row/column must be *identifiable*, not just measurable.
    ///
    /// Two rows, the second far worse than the first: the stats must report
    /// the second one's range AND point at the second one. A diagnostic that
    /// says "some row spans 1e12" without saying which is not actionable, and
    /// before these indices existed that is all a caller could have been told.
    #[test]
    fn worst_row_and_column_are_identified() {
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let y = scalar_var(&mut arena, "y", 1);
        // row 0: 1 x + 1 y  -> range 1
        let row0 = {
            let a = lin(&mut arena, 1.0, x);
            let b = lin(&mut arena, 1.0, y);
            add(&mut arena, a, b)
        };
        // row 1: 1e6 x + 1e-6 y -> range 1e12, and column x now spans
        // 1e6 / 1 = 1e6 across the two rows while y spans 1 / 1e-6 = 1e6.
        let row1 = {
            let a = lin(&mut arena, 1e6, x);
            let b = lin(&mut arena, 1e-6, y);
            add(&mut arena, a, b)
        };
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![
                ConstraintRepr {
                    body: row0,
                    sense: ConstraintSense::Le,
                    rhs: 1.0,
                    name: Some("fine".into()),
                },
                ConstraintRepr {
                    body: row1,
                    sense: ConstraintSense::Le,
                    rhs: 1.0,
                    name: Some("badly_scaled".into()),
                },
            ],
            variables: vec![vinfo("x", 0), vinfo("y", 1)],
            n_vars: 2,
        };
        let (_f, s) = compute_equilibration(&model);
        assert_eq!(s.linear_rows_sampled, 2);
        assert!(
            (s.worst_row_dynamic_range - 1e12).abs() / 1e12 < 1e-9,
            "row range = {}",
            s.worst_row_dynamic_range
        );
        assert_eq!(s.worst_row_index, Some(1), "must point at the bad row");
        assert!(
            (s.worst_col_dynamic_range - 1e6).abs() / 1e6 < 1e-9,
            "col range = {}",
            s.worst_col_dynamic_range
        );
        // Both columns span 1e6, so the first to attain it wins -- the point
        // is that *a* column is named, deterministically.
        assert_eq!(s.worst_col_index, Some(0));
    }

    /// A model with no linear rows leaves both indices `None`, so a caller
    /// cannot mistake "nothing measured" for "row 0 is the problem".
    #[test]
    fn no_linear_rows_leaves_indices_unset() {
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![],
            variables: vec![vinfo("x", 0)],
            n_vars: 1,
        };
        let (_f, s) = compute_equilibration(&model);
        assert_eq!(s.linear_rows_sampled, 0);
        assert_eq!(s.worst_row_index, None);
        assert_eq!(s.worst_col_index, None);
    }

    /// Build `prod_{k<n_factors} (x_{k,0} + 1*x_{k,1} + ... + (w-1)*x_{k,w-1})`:
    /// a product of sums whose full expansion has `w^n_factors` monomials.
    /// Returns (arena, body, variables).
    fn product_of_sums(
        arena: &mut ExprArena,
        n_factors: usize,
        width: usize,
        first_var: usize,
    ) -> (ExprId, Vec<VarInfo>) {
        let mut vars = Vec::new();
        let mut body: Option<ExprId> = None;
        for k in 0..n_factors {
            let mut s: Option<ExprId> = None;
            for j in 0..width {
                let idx = first_var + k * width + j;
                let name = format!("x{idx}");
                let v = scalar_var(arena, &name, idx);
                vars.push(vinfo(&name, idx));
                let t = lin(arena, (j + 1) as f64, v);
                s = Some(match s {
                    None => t,
                    Some(acc) => add(arena, acc, t),
                });
            }
            let s = s.unwrap();
            body = Some(match body {
                None => s,
                Some(acc) => arena.add(ExprNode::BinaryOp {
                    op: BinOp::Mul,
                    left: acc,
                    right: s,
                }),
            });
        }
        (body.unwrap(), vars)
    }

    /// #1686 THE REGRESSION. A 12-factor product of 7-term sums expands to
    /// 7^12 ≈ 1.4e10 monomials; unbudgeted this does not return in any useful
    /// time (the 9 x 7 case took ~115 s). Budgeted, the row is abstained on
    /// and counted, and a linear row in the same model is still sampled
    /// exactly as before.
    #[test]
    fn product_of_sums_row_is_abstained_on_not_expanded() {
        let mut arena = ExprArena::new();
        let (blowup, mut vars) = product_of_sums(&mut arena, 12, 7, 0);
        let n = vars.len();
        let a = scalar_var(&mut arena, "a", n);
        let b = scalar_var(&mut arena, "b", n + 1);
        vars.push(vinfo("a", n));
        vars.push(vinfo("b", n + 1));
        let linear = {
            let l = lin(&mut arena, 1e3, a);
            let r = lin(&mut arena, 1e-3, b);
            add(&mut arena, l, r)
        };
        let n_vars = vars.len();
        let model = ModelRepr {
            arena,
            objective: a,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![
                ConstraintRepr {
                    body: blowup,
                    sense: ConstraintSense::Le,
                    rhs: 1.0,
                    name: None,
                },
                ConstraintRepr {
                    body: linear,
                    sense: ConstraintSense::Le,
                    rhs: 1.0,
                    name: None,
                },
            ],
            variables: vars,
            n_vars,
        };
        let (f, s) = compute_equilibration(&model);
        assert_eq!(
            s.rows_over_budget, 1,
            "the product row must be abstained on"
        );
        assert_eq!(
            s.linear_rows_sampled, 1,
            "the linear row must still be sampled"
        );
        assert_eq!(f.row_scales[0], 1.0, "abstained row keeps identity scale");
        assert!((s.worst_row_dynamic_range - 1e6).abs() / 1e6 < 1e-9);
        assert_eq!(s.worst_row_index, Some(1));
    }

    /// The per-PASS cap: many rows, each individually inside the per-row
    /// budget, must not add up to more than `MODEL_EXPANSION_BUDGET` of
    /// expansion. 7^6 = 117,649 monomials per row (< ROW_EXPANSION_BUDGET);
    /// forty of them would be ~4.7e6 (> MODEL_EXPANSION_BUDGET).
    #[test]
    fn expansion_budget_is_a_running_total() {
        let mut arena = ExprArena::new();
        let mut vars = Vec::new();
        let mut cons = Vec::new();
        for r in 0..40 {
            let (body, v) = product_of_sums(&mut arena, 6, 7, r * 42);
            vars.extend(v);
            cons.push(ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 1.0,
                name: None,
            });
        }
        let n_vars = vars.len();
        let model = ModelRepr {
            arena,
            objective: cons[0].body,
            objective_sense: ObjectiveSense::Minimize,
            constraints: cons,
            variables: vars,
            n_vars,
        };
        // Each row alone fits the per-row budget...
        let (one, spent) = try_polynomial_budgeted(
            &model.arena,
            model.constraints[0].body,
            PolyBudget {
                linear: u64::MAX,
                expansion: ROW_EXPANSION_BUDGET,
            },
        );
        assert!(
            one.is_ok(),
            "fixture: a single row must fit the per-row budget"
        );
        assert!(
            spent.expansion * 40 > MODEL_EXPANSION_BUDGET,
            "fixture too small"
        );
        // ...but the pass stops paying once the running total is spent.
        let (_f, s) = compute_equilibration(&model);
        let paid = 40 - s.rows_over_budget;
        assert!(s.rows_over_budget > 0, "running total was not enforced");
        assert!(
            (paid as u64) * spent.expansion <= MODEL_EXPANSION_BUDGET,
            "{paid} rows x {} units exceeds the pass budget",
            spent.expansion
        );
    }

    /// Within budget the budgeted expansion is identical to the unbudgeted
    /// one (bound-neutral by construction): a row that cancels to linear,
    /// `(x + y) * (x - y) - x*x + y*y + 3 z`, is sampled exactly as before.
    #[test]
    fn budgeted_matches_unbudgeted_on_cancelling_row() {
        use super::super::polynomial::try_polynomial;
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let z = scalar_var(&mut arena, "z", 2);
        let s = add(&mut arena, x, y);
        let d = arena.add(ExprNode::BinaryOp {
            op: BinOp::Sub,
            left: x,
            right: y,
        });
        let p = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: s,
            right: d,
        });
        let xx = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: x,
            right: x,
        });
        let yy = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: y,
            right: y,
        });
        let t = arena.add(ExprNode::BinaryOp {
            op: BinOp::Sub,
            left: p,
            right: xx,
        });
        let t = add(&mut arena, t, yy);
        let z3 = lin(&mut arena, 3.0, z);
        let body = add(&mut arena, t, z3);
        let full = try_polynomial(&arena, body).unwrap();
        let (b, _) = try_polynomial_budgeted(
            &arena,
            body,
            PolyBudget {
                linear: 1_000,
                expansion: 1_000,
            },
        );
        let b = b.unwrap();
        assert_eq!(full.max_total_degree(), 1);
        assert_eq!(b.monomials.len(), full.monomials.len());
        for (m1, m2) in b.monomials.iter().zip(full.monomials.iter()) {
            assert_eq!(m1.coeff.to_bits(), m2.coeff.to_bits());
            assert_eq!(m1.factors, m2.factors);
        }
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
            variables: vec![vinfo("x", 0), vinfo("y", 1), vinfo("z", 2)],
            n_vars: 3,
        };
        let (f, st) = compute_equilibration(&model);
        assert_eq!(st.rows_over_budget, 0);
        assert_eq!(st.linear_rows_sampled, 1);
        assert!((f.row_scales[0] - 1.0 / 3.0).abs() < 1e-12);
    }
}
