//! Implied-bound propagation across linear constraints (B1 of the
//! presolve roadmap).
//!
//! ## What this pass does
//!
//! For every constraint whose body is a degree-≤1 polynomial, derive an
//! implied bound on each of its variables from the row activity under the
//! current scalar variable bounds, and tighten the running bound vector.
//!
//! For a row `Σ c_i x_i + c0  ⊙  rhs` with `⊙ ∈ {≤, =, ≥}` and current
//! per-variable intervals `x_i ∈ [lb_i, ub_i]`, define the activity
//! contribution interval of variable `i`
//!
//! ```text
//!     a_i = c_i · [lb_i, ub_i]
//! ```
//!
//! and the activity interval of the rest of the row
//!
//! ```text
//!     R_i = Σ_{k≠i} a_k + c0.
//! ```
//!
//! For a `≤` constraint, isolating `c_i x_i ≤ rhs − R_i` yields an
//! implied bound on `x_i`. For `≥` the bound is dual; for `=` both
//! bounds apply. The pass intersects every such implied bound with
//! the running vector.
//!
//! ## Distinction from `fbbt`
//!
//! [`super::fbbt::fbbt_with_cutoff`] is a general DAG-walking
//! propagator: it is correct for arbitrary expressions but pays
//! proportional cost in tree-traversal overhead. This pass is the
//! linear specialisation: one O(nnz) sweep per constraint, no DAG
//! descent. It often establishes the same fixed point on linear
//! subproblems orders of magnitude faster, and provides cross-row
//! propagation as a single deterministic sweep — the same constraint
//! tightens every variable in its row in one pass.
//!
//! Nonlinear and partially-nonlinear rows are simply skipped; FBBT
//! handles those.
//!
//! ## Scope (intentional, conservative)
//!
//! - Only rows whose body is a degree-≤1 polynomial in scalar
//!   variable leaves.
//! - Each leaf must resolve to a scalar `Variable` block (size == 1).
//!   Indexed accesses into vector variables are skipped to keep the
//!   pass simple; a future revision can add the bound-vector
//!   bookkeeping to handle them.
//! - One sweep over constraints per call. Iterating to a fixed point
//!   is the orchestrator's responsibility.
//!
//! ## Determinism
//!
//! Constraints are scanned in `model.constraints` order; within each
//! constraint, leaves are visited in arena-id order. No `HashMap`
//! iteration on the hot path.

use super::fbbt::Interval;
use super::polynomial::try_polynomial;
use crate::expr::{ConstraintSense, ExprArena, ExprId, ExprNode, ModelRepr};

/// Per-pass statistics from implied-bound propagation.
#[derive(Debug, Clone, Default)]
pub struct ImpliedBoundsStats {
    /// Number of constraints whose body was a usable linear row.
    pub linear_rows_examined: usize,
    /// Number of variables whose `lb` or `ub` was strictly tightened.
    pub bounds_tightened: usize,
    /// Number of constraints flagged infeasible (empty intersection
    /// against the current box). Detection only — the pass does not
    /// raise.
    pub rows_flagged_infeasible: usize,
}

/// Run implied-bound propagation over `model`'s linear rows, tightening
/// `bounds` in place. Pure function on the inputs apart from the
/// `bounds` argument.
pub fn propagate_implied_bounds(model: &ModelRepr, bounds: &mut [Interval]) -> ImpliedBoundsStats {
    let mut stats = ImpliedBoundsStats::default();

    for c in &model.constraints {
        let row = match extract_linear_row(model, c.body) {
            Some(r) => r,
            None => continue,
        };
        stats.linear_rows_examined += 1;
        propagate_row(&row, c.sense, c.rhs, bounds, &mut stats);
    }
    stats
}

// ─────────────────────────────────────────────────────────────
// Row extraction
// ─────────────────────────────────────────────────────────────

/// A linear row, normalised to scalar bound indices.
#[derive(Debug, Clone)]
struct LinearRow {
    /// `(bound_index, coeff)` pairs in arena-id order. Coefficients
    /// with absolute value below 1e-15 are dropped.
    terms: Vec<(usize, f64)>,
    /// Constant offset of the row (`c0` in `Σ c_i x_i + c0`).
    constant: f64,
}

/// Extract a linear-row representation if `body` is a degree-≤1
/// polynomial in scalar variables. Returns `None` for nonlinear bodies,
/// indexed variable accesses, or rows whose all coefficients vanish.
fn extract_linear_row(model: &ModelRepr, body: ExprId) -> Option<LinearRow> {
    let poly = try_polynomial(&model.arena, body)?;
    if poly.max_total_degree() > 1 {
        return None;
    }
    let mut terms: Vec<(usize, f64)> = Vec::with_capacity(poly.monomials.len());
    for m in &poly.monomials {
        if m.factors.len() != 1 || m.factors[0].1 != 1 {
            return None;
        }
        let leaf = m.factors[0].0;
        let bidx = leaf_scalar_bound_index(&model.arena, model, leaf)?;
        if m.coeff.abs() <= 1e-15 {
            continue;
        }
        terms.push((bidx, m.coeff));
    }
    if terms.is_empty() {
        return None;
    }
    // Sort by bound index for stable iteration; arena ordering would
    // also work but bound-index sort makes the activity update loop
    // robust to leaf interning order.
    terms.sort_by_key(|(b, _)| *b);
    Some(LinearRow {
        terms,
        constant: poly.constant,
    })
}

/// Resolve a polynomial leaf id to a scalar bound index. Returns
/// `None` if the leaf is not a scalar `Variable` or if the index is
/// out of range.
fn leaf_scalar_bound_index(arena: &ExprArena, model: &ModelRepr, leaf: ExprId) -> Option<usize> {
    match arena.get(leaf) {
        ExprNode::Variable { index, size, .. } if *size == 1 => {
            let v = model.variables.get(*index)?;
            Some(v.offset)
        }
        _ => None,
    }
}

// ─────────────────────────────────────────────────────────────
// Per-row propagation
// ─────────────────────────────────────────────────────────────

/// Tighten variable bounds from a single linear row.
///
/// The activity interval `R = Σ c_i [lb_i, ub_i] + c0` is computed
/// once. For each term, the contribution is subtracted, the sense
/// applied, and the result intersected back. The contribution itself
/// uses interval arithmetic so vanishing or sign-changing coefficients
/// drop out naturally.
fn propagate_row(
    row: &LinearRow,
    sense: ConstraintSense,
    rhs: f64,
    bounds: &mut [Interval],
    stats: &mut ImpliedBoundsStats,
) {
    // Bail if any term references an out-of-range bound index — happens
    // briefly during fixture construction in tests.
    for (idx, _) in &row.terms {
        if *idx >= bounds.len() {
            return;
        }
    }
    // Compute total activity Σ c_i [lb_i, ub_i] + constant.
    let mut act_lo = row.constant;
    let mut act_hi = row.constant;
    for (idx, c) in &row.terms {
        let b = bounds[*idx];
        let (lo, hi) = scaled_interval(*c, b.lo, b.hi);
        act_lo += lo;
        act_hi += hi;
    }
    // Quick infeasibility check — only flag, don't act on it.
    match sense {
        ConstraintSense::Le if act_lo > rhs + 1e-9 => {
            stats.rows_flagged_infeasible += 1;
        }
        ConstraintSense::Ge if act_hi < rhs - 1e-9 => {
            stats.rows_flagged_infeasible += 1;
        }
        ConstraintSense::Eq if act_lo > rhs + 1e-9 || act_hi < rhs - 1e-9 => {
            stats.rows_flagged_infeasible += 1;
        }
        _ => {}
    }

    for &(idx, c) in &row.terms {
        let b = bounds[idx];
        let (own_lo, own_hi) = scaled_interval(c, b.lo, b.hi);
        // Activity of the rest of the row.
        let rest_lo = act_lo - own_lo;
        let rest_hi = act_hi - own_hi;
        // Implied bound on c_i * x_i:
        //   Le:  c_i x_i ≤ rhs − rest_lo
        //   Ge:  c_i x_i ≥ rhs − rest_hi
        //   Eq:  rhs − rest_hi ≤ c_i x_i ≤ rhs − rest_lo
        let (cx_lo, cx_hi) = match sense {
            ConstraintSense::Le => (f64::NEG_INFINITY, rhs - rest_lo),
            ConstraintSense::Ge => (rhs - rest_hi, f64::INFINITY),
            ConstraintSense::Eq => (rhs - rest_hi, rhs - rest_lo),
        };
        // Divide by c to get bound on x_i. interval_div would handle
        // sign-flipping; do it explicitly here for clarity.
        let (x_lo, x_hi) = if c > 0.0 {
            (cx_lo / c, cx_hi / c)
        } else {
            // Negative coeff flips the inequality direction.
            (cx_hi / c, cx_lo / c)
        };
        let implied = Interval::new(x_lo, x_hi);
        let new_b = bounds[idx].intersect(&implied);
        // Floating-point only — only count strict tightenings.
        let tol = 1e-12;
        if new_b.lo > bounds[idx].lo + tol || new_b.hi < bounds[idx].hi - tol {
            bounds[idx] = new_b;
            stats.bounds_tightened += 1;
        }
    }
}

/// `c · [lo, hi]` with the sign of `c` taken into account. Returns
/// `(lo, hi)` of the resulting interval.
fn scaled_interval(c: f64, lo: f64, hi: f64) -> (f64, f64) {
    if c >= 0.0 {
        (c * lo, c * hi)
    } else {
        (c * hi, c * lo)
    }
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

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

    fn vinfo(name: &str, offset: usize, lb: f64, ub: f64) -> VarInfo {
        VarInfo {
            name: name.into(),
            var_type: VarType::Continuous,
            offset,
            size: 1,
            shape: vec![],
            lb: vec![lb],
            ub: vec![ub],
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

    /// `2 x + y ≤ 10` with `x ∈ [0, 10], y ∈ [0, 10]` should not
    /// tighten anything: the row is satisfied at every corner.
    #[test]
    fn loose_row_does_not_tighten() {
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let body = {
            let lhs = lin(&mut arena, 2.0, x);
            let rhs = lin(&mut arena, 1.0, y);
            add(&mut arena, lhs, rhs)
        };
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 100.0,
                name: None,
            }],
            variables: vec![vinfo("x", 0, 0.0, 10.0), vinfo("y", 1, 0.0, 10.0)],
            n_vars: 2,
        };
        let mut bounds = vec![Interval::new(0.0, 10.0); 2];
        let stats = propagate_implied_bounds(&model, &mut bounds);
        assert_eq!(stats.bounds_tightened, 0);
        assert_eq!(bounds[0].lo, 0.0);
        assert_eq!(bounds[0].hi, 10.0);
        assert_eq!(bounds[1].lo, 0.0);
        assert_eq!(bounds[1].hi, 10.0);
    }

    /// `2 x + y ≤ 5` with `x ∈ [0, 10], y ∈ [0, 10]`: the row
    /// implies `x ≤ 2.5` (when `y = 0`) and `y ≤ 5` (when `x = 0`).
    #[test]
    fn tight_le_row_pushes_down_uppers() {
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let body = {
            let lhs = lin(&mut arena, 2.0, x);
            let rhs = lin(&mut arena, 1.0, y);
            add(&mut arena, lhs, rhs)
        };
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
            variables: vec![vinfo("x", 0, 0.0, 10.0), vinfo("y", 1, 0.0, 10.0)],
            n_vars: 2,
        };
        let mut bounds = vec![Interval::new(0.0, 10.0); 2];
        let stats = propagate_implied_bounds(&model, &mut bounds);
        assert_eq!(stats.bounds_tightened, 2);
        assert!((bounds[0].hi - 2.5).abs() < 1e-9);
        assert!((bounds[1].hi - 5.0).abs() < 1e-9);
        assert_eq!(bounds[0].lo, 0.0);
        assert_eq!(bounds[1].lo, 0.0);
    }

    /// `x + y ≥ 8` with both in `[0, 5]`: implies `x ≥ 3, y ≥ 3`.
    #[test]
    fn ge_row_pushes_up_lowers() {
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let body = {
            let lhs = lin(&mut arena, 1.0, x);
            let rhs = lin(&mut arena, 1.0, y);
            add(&mut arena, lhs, rhs)
        };
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Ge,
                rhs: 8.0,
                name: None,
            }],
            variables: vec![vinfo("x", 0, 0.0, 5.0), vinfo("y", 1, 0.0, 5.0)],
            n_vars: 2,
        };
        let mut bounds = vec![Interval::new(0.0, 5.0); 2];
        let stats = propagate_implied_bounds(&model, &mut bounds);
        assert_eq!(stats.bounds_tightened, 2);
        assert!((bounds[0].lo - 3.0).abs() < 1e-9);
        assert!((bounds[1].lo - 3.0).abs() < 1e-9);
    }

    /// `2 x − y = 4` with `x ∈ [1, 5], y ∈ [0, 10]`:
    ///   `y = 2 x − 4`   → `y ∈ [−2, 6] ∩ [0, 10] = [0, 6]`
    ///   `x = (y + 4) / 2` → `x ∈ [2, 7] ∩ [1, 5] = [2, 5]`
    /// After one sweep, both endpoints tighten.
    #[test]
    fn eq_row_tightens_both_sides() {
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let body = {
            let lhs = lin(&mut arena, 2.0, x);
            let rhs = lin(&mut arena, -1.0, y);
            add(&mut arena, lhs, rhs)
        };
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Eq,
                rhs: 4.0,
                name: None,
            }],
            variables: vec![vinfo("x", 0, 1.0, 5.0), vinfo("y", 1, 0.0, 10.0)],
            n_vars: 2,
        };
        let mut bounds = vec![Interval::new(1.0, 5.0), Interval::new(0.0, 10.0)];
        let _ = propagate_implied_bounds(&model, &mut bounds);
        // After the sweep, x ≥ 2 (forced by y ≤ 10 ⇒ 2x = y + 4 ≤ 14
        // is loose; tighter is y ≥ 0 ⇒ 2x ≥ 4 ⇒ x ≥ 2), and y ≤ 6
        // (forced by x ≤ 5).
        assert!((bounds[0].lo - 2.0).abs() < 1e-9, "x.lo = {}", bounds[0].lo);
        assert_eq!(bounds[0].hi, 5.0);
        assert_eq!(bounds[1].lo, 0.0);
        assert!((bounds[1].hi - 6.0).abs() < 1e-9, "y.hi = {}", bounds[1].hi);
    }

    /// Negative coefficient on the variable being bounded:
    /// `−x + y ≥ −3` with `x ∈ [0, 10], y ∈ [0, 1]` rearranges to
    /// `x ≤ y + 3`, hence `x ≤ 4`. The negative-coefficient code
    /// path must flip the inequality direction when dividing by `c`.
    #[test]
    fn negative_coeff_correct_sign() {
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let body = {
            let lhs = lin(&mut arena, -1.0, x);
            let rhs = lin(&mut arena, 1.0, y);
            add(&mut arena, lhs, rhs)
        };
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Ge,
                rhs: -3.0,
                name: None,
            }],
            variables: vec![vinfo("x", 0, 0.0, 10.0), vinfo("y", 1, 0.0, 1.0)],
            n_vars: 2,
        };
        let mut bounds = vec![Interval::new(0.0, 10.0), Interval::new(0.0, 1.0)];
        let stats = propagate_implied_bounds(&model, &mut bounds);
        assert!((bounds[0].hi - 4.0).abs() < 1e-9, "x.hi = {}", bounds[0].hi);
        assert_eq!(bounds[0].lo, 0.0);
        assert_eq!(bounds[1].lo, 0.0);
        assert_eq!(bounds[1].hi, 1.0);
        assert!(stats.bounds_tightened >= 1);
    }

    /// Nonlinear rows are skipped silently.
    #[test]
    fn nonlinear_row_skipped() {
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let two = arena.add(ExprNode::Constant(2.0));
        // x^2 + 1 ≤ 5  → max_total_degree == 2, skipped.
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
            variables: vec![vinfo("x", 0, -5.0, 5.0)],
            n_vars: 1,
        };
        let mut bounds = vec![Interval::new(-5.0, 5.0)];
        let stats = propagate_implied_bounds(&model, &mut bounds);
        assert_eq!(stats.linear_rows_examined, 0);
        assert_eq!(stats.bounds_tightened, 0);
    }

    /// Two rows feeding each other should each contribute. One sweep
    /// is enough on this fixture; the orchestrator iterates if needed.
    #[test]
    fn cross_row_propagation_in_one_sweep() {
        // x + y ≤ 5;  x − y ≤ 1;  both vars in [0, 10].
        // Activity-only implications:
        //   row 1:  x ≤ 5,  y ≤ 5
        //   row 2:  x ≤ 11 (loose),  y ≥ −1 (loose)
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let r1 = {
            let a = lin(&mut arena, 1.0, x);
            let b = lin(&mut arena, 1.0, y);
            add(&mut arena, a, b)
        };
        let r2 = {
            let a = lin(&mut arena, 1.0, x);
            let b = lin(&mut arena, -1.0, y);
            add(&mut arena, a, b)
        };
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![
                ConstraintRepr {
                    body: r1,
                    sense: ConstraintSense::Le,
                    rhs: 5.0,
                    name: None,
                },
                ConstraintRepr {
                    body: r2,
                    sense: ConstraintSense::Le,
                    rhs: 1.0,
                    name: None,
                },
            ],
            variables: vec![vinfo("x", 0, 0.0, 10.0), vinfo("y", 1, 0.0, 10.0)],
            n_vars: 2,
        };
        let mut bounds = vec![Interval::new(0.0, 10.0); 2];
        let stats = propagate_implied_bounds(&model, &mut bounds);
        assert_eq!(stats.linear_rows_examined, 2);
        assert!(stats.bounds_tightened >= 2);
        assert!((bounds[0].hi - 5.0).abs() < 1e-9);
        assert!((bounds[1].hi - 5.0).abs() < 1e-9);
    }

    /// Idempotence: a second call on already-tightened bounds reports
    /// zero new tightenings and produces identical bounds.
    #[test]
    fn idempotent_on_second_call() {
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let body = {
            let lhs = lin(&mut arena, 2.0, x);
            let rhs = lin(&mut arena, 1.0, y);
            add(&mut arena, lhs, rhs)
        };
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
            variables: vec![vinfo("x", 0, 0.0, 10.0), vinfo("y", 1, 0.0, 10.0)],
            n_vars: 2,
        };
        let mut bounds = vec![Interval::new(0.0, 10.0); 2];
        let _ = propagate_implied_bounds(&model, &mut bounds);
        let snapshot: Vec<(f64, f64)> = bounds.iter().map(|b| (b.lo, b.hi)).collect();
        let stats2 = propagate_implied_bounds(&model, &mut bounds);
        assert_eq!(stats2.bounds_tightened, 0);
        let after: Vec<(f64, f64)> = bounds.iter().map(|b| (b.lo, b.hi)).collect();
        assert_eq!(after, snapshot);
    }
}
