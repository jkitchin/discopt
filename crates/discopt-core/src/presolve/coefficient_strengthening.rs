//! Savelsbergh-style coefficient strengthening for mixed-integer rows
//! (C4 of issue #53).
//!
//! Strengthens the coefficient of a binary variable in a linear `≤`
//! (or `≥` after reflection) constraint when the LP slack on the
//! "binary fixed at 1" branch exceeds the binary's coefficient. The
//! transformation is bound-preserving: any feasible point of the
//! original constraint is feasible in the strengthened one and
//! vice versa, but the LP relaxation strictly tightens.
//!
//! ## The rule
//!
//! For `Σ aⱼ xⱼ ≤ b` with binary `x_k ∈ {0, 1}` and `a_k > 0`, let
//!
//! ```text
//!   U_minus_k = Σ_{j ≠ k} max(aⱼ · lbⱼ, aⱼ · ubⱼ)
//!   slack     = b - U_minus_k
//! ```
//!
//! `U_minus_k` is the worst-case (largest) value of the rest of the
//! LHS. If `0 < slack < a_k` (i.e. setting `x_k = 1` with the rest
//! at its max would violate the row, but the rest alone fits inside
//! the RHS), simultaneously shrink the binary's coefficient and the
//! RHS by the slack:
//!
//! ```text
//!   a_k' = a_k - slack
//!   b'   = b   - slack
//! ```
//!
//! At `x_k = 0` both `a_k x_k = 0` and the constraint reduces to
//! `Σ_{j≠k} aⱼ xⱼ ≤ b'`, which is implied by `Σ ≤ U_minus_k = b'`,
//! the worst case the rest can reach — so the row is trivially
//! satisfied as it was before. At `x_k = 1` both rows reduce to
//! `Σ_{j≠k} aⱼ xⱼ ≤ b - a_k`, which is exactly the original
//! restriction. The transformation is therefore equivalent on
//! integer corners but cuts off LP-feasible points with fractional
//! `x_k`. Savelsbergh (1994) formalises the rule for pure MIP; the
//! same algebra applies row-wise to MINLP rows whose linear part
//! satisfies the assumptions.
//!
//! ## Scope (intentional, conservative)
//!
//! - Only **linear** constraint bodies — those whose total degree
//!   is exactly 1 in [`super::polynomial::try_polynomial`].
//! - Only **binary** coefficient targets. Integer-but-not-binary
//!   strengthening needs an extra rounding step which v0 skips.
//! - `≤` and `≥` constraints (`≥` is reflected internally to `≤` and
//!   reflected back). Equalities are left to other passes.
//! - Coefficients are tightened down toward the binary's slack; we do
//!   not strengthen bounds on continuous variables here (FBBT's job).
//! - Bound source: `model.variables[*].lb / .ub`, not the orchestrator's
//!   running interval array — staying consistent with `aggregate.rs`
//!   and `redundancy.rs` which also work off declared bounds.
//!
//! ## Determinism
//!
//! Constraints scanned in `model.constraints` order. Within a row,
//! coefficients are visited in the canonical (leaf-id-sorted) order
//! produced by [`super::polynomial::try_polynomial`]. No `HashMap`
//! iteration on the hot path.

use super::polynomial::try_polynomial;
use crate::expr::{
    BinOp, ConstraintRepr, ConstraintSense, ExprArena, ExprId, ExprNode, ModelRepr, VarType,
};

/// Per-pass statistics from coefficient strengthening.
#[derive(Debug, Clone, Default)]
pub struct CoefficientStrengtheningStats {
    /// Number of constraint rows that had at least one coefficient strengthened.
    pub constraints_strengthened: usize,
    /// Total number of binary coefficients tightened (one row may strengthen several).
    pub coefficients_strengthened: usize,
    /// Number of linear rows examined.
    pub rows_examined: usize,
    /// Indices (in input model order) of constraints that were rewritten.
    pub rewritten_indices: Vec<usize>,
}

/// Run coefficient-strengthening on `model`. Returns a new model with
/// rewritten constraint bodies, plus per-pass statistics.
///
/// The model is cloned in full; any constraint that *cannot* be
/// strengthened is preserved verbatim. This keeps the pass safely
/// composable inside the orchestrator's fixed-point loop: idempotent
/// after one fully-converged run, and safe to interleave with other
/// rewriters.
pub fn coefficient_strengthening(model: &ModelRepr) -> (ModelRepr, CoefficientStrengtheningStats) {
    let mut stats = CoefficientStrengtheningStats::default();
    let mut out = model.clone();

    for ci in 0..out.constraints.len() {
        let row = match extract_linear_row(&out, ci) {
            Some(r) => r,
            None => continue,
        };
        stats.rows_examined += 1;

        let new_row = match strengthen_row(&out, &row) {
            Some(nr) => nr,
            None => continue,
        };
        if new_row.tightened == 0 {
            continue;
        }

        let new_body = build_linear_body(&mut out.arena, &new_row);
        out.constraints[ci] = ConstraintRepr {
            body: new_body,
            sense: new_row.sense,
            rhs: new_row.rhs,
            name: out.constraints[ci].name.clone(),
        };
        stats.constraints_strengthened += 1;
        stats.coefficients_strengthened += new_row.tightened;
        stats.rewritten_indices.push(ci);
    }

    (out, stats)
}

// ─────────────────────────────────────────────────────────────────────
// Internal: linear-row representation + strengthening kernel
// ─────────────────────────────────────────────────────────────────────

/// A fully-resolved linear constraint body: `Σ (coeff_j · var_id_j) + const ⟂ rhs`.
#[derive(Debug, Clone)]
struct LinearRow {
    /// `(arena ExprId of variable leaf, variable block index, coefficient)`.
    terms: Vec<(ExprId, usize, f64)>,
    /// Polynomial constant offset (free term).
    constant: f64,
    sense: ConstraintSense,
    rhs: f64,
    /// Number of coefficients tightened by `strengthen_row`. Defaults to 0.
    tightened: usize,
}

/// Try to view `model.constraints[ci]` as a linear row over variable
/// leaves. Returns `None` if the body is non-linear, has no variable
/// terms, or contains a factor whose underlying ExprId is not a
/// scalar variable (e.g. an indexed-into vector — strengthening such
/// terms requires per-element bounds and is out of scope for v0).
fn extract_linear_row(model: &ModelRepr, ci: usize) -> Option<LinearRow> {
    let c = &model.constraints[ci];
    if c.sense == ConstraintSense::Eq {
        // Skip equalities — the strengthening rule is asymmetric and
        // equality rows don't have the slack semantics we depend on.
        return None;
    }

    let poly = try_polynomial(&model.arena, c.body)?;
    if poly.max_total_degree() > 1 {
        return None;
    }
    if poly.monomials.is_empty() {
        return None;
    }

    let mut terms: Vec<(ExprId, usize, f64)> = Vec::with_capacity(poly.monomials.len());
    for m in &poly.monomials {
        if m.factors.len() != 1 || m.factors[0].1 != 1 {
            return None;
        }
        let leaf_id = m.factors[0].0;
        let block = match model.arena.get(leaf_id) {
            ExprNode::Variable {
                index, size, shape, ..
            } => {
                // v0: scalar variables only. Indexed access into an
                // array variable shows up here too but we can't read
                // per-element bounds without an explicit element
                // offset, so skip such rows.
                if *size != 1 || !shape.is_empty() {
                    return None;
                }
                *index
            }
            _ => return None,
        };
        terms.push((leaf_id, block, m.coeff));
    }
    Some(LinearRow {
        terms,
        constant: poly.constant,
        sense: c.sense,
        rhs: c.rhs,
        tightened: 0,
    })
}

/// Apply Savelsbergh strengthening to `row` and return the modified
/// row, or `None` if no coefficient could be strengthened. Reflects
/// `≥` rows internally so the kernel only deals with `≤`.
fn strengthen_row(model: &ModelRepr, row: &LinearRow) -> Option<LinearRow> {
    // Reflect ≥ to ≤ by negating LHS + RHS.
    let (mut terms, mut rhs, _sense) = match row.sense {
        ConstraintSense::Le => (
            row.terms.clone(),
            row.rhs - row.constant,
            ConstraintSense::Le,
        ),
        ConstraintSense::Ge => (
            row.terms.iter().map(|(id, b, c)| (*id, *b, -*c)).collect(),
            -(row.rhs - row.constant),
            ConstraintSense::Le,
        ),
        ConstraintSense::Eq => return None,
    };

    // Worst-case rest activity per coefficient. We compute the global
    // worst-case once and subtract each candidate's own contribution.
    let mut total_worst = 0.0;
    let mut worst_contributions: Vec<f64> = Vec::with_capacity(terms.len());
    for &(_, block, coeff) in &terms {
        let v = &model.variables[block];
        let lo_term = coeff * v.lb[0];
        let hi_term = coeff * v.ub[0];
        let w = lo_term.max(hi_term);
        if !w.is_finite() {
            return None;
        }
        total_worst += w;
        worst_contributions.push(w);
    }

    let mut tightened = 0;
    for ti in 0..terms.len() {
        let (_, block, coeff) = terms[ti];
        if coeff <= 0.0 {
            continue; // v0: positive-coefficient binary case only
        }
        if model.variables[block].var_type != VarType::Binary {
            continue;
        }
        let u_minus_k = total_worst - worst_contributions[ti];
        if !u_minus_k.is_finite() {
            continue;
        }
        // slack = rhs - U_minus_k; valid strengthening fires when
        // 0 < slack < coeff (else either redundant at x_k=1 or
        // infeasible at x_k=0).
        let slack = rhs - u_minus_k;
        if slack <= 1e-9 || slack + 1e-9 >= coeff {
            continue;
        }
        let new_coeff = coeff - slack;
        let new_rhs = rhs - slack;
        terms[ti].2 = new_coeff;
        rhs = new_rhs;
        // Update worst-contribution bookkeeping so a subsequent
        // coefficient in the same row sees the post-strengthening
        // activity.
        let v = &model.variables[block];
        let new_w = (new_coeff * v.lb[0]).max(new_coeff * v.ub[0]);
        total_worst += new_w - worst_contributions[ti];
        worst_contributions[ti] = new_w;
        tightened += 1;
    }

    if tightened == 0 {
        return None;
    }

    // Reflect back to original sense if we flipped at the start.
    let (final_terms, final_rhs) = match row.sense {
        ConstraintSense::Le => (terms, rhs + row.constant),
        ConstraintSense::Ge => (
            terms.into_iter().map(|(id, b, c)| (id, b, -c)).collect(),
            -rhs + row.constant,
        ),
        ConstraintSense::Eq => unreachable!("equalities filtered earlier"),
    };

    Some(LinearRow {
        terms: final_terms,
        constant: row.constant,
        sense: row.sense,
        rhs: final_rhs,
        tightened,
    })
}

/// Build a fresh expression body: `Σ (c_j · x_j) + constant`.
fn build_linear_body(arena: &mut ExprArena, row: &LinearRow) -> ExprId {
    let mut term_ids: Vec<ExprId> = Vec::with_capacity(row.terms.len());
    for &(leaf_id, _, coeff) in &row.terms {
        if coeff.abs() < 1e-15 {
            continue;
        }
        let term = if (coeff - 1.0).abs() < 1e-15 {
            leaf_id
        } else {
            let c = arena.add(ExprNode::Constant(coeff));
            arena.add(ExprNode::BinaryOp {
                op: BinOp::Mul,
                left: c,
                right: leaf_id,
            })
        };
        term_ids.push(term);
    }
    if row.constant.abs() > 1e-15 {
        term_ids.push(arena.add(ExprNode::Constant(row.constant)));
    }
    if term_ids.is_empty() {
        return arena.add(ExprNode::Constant(0.0));
    }
    let mut acc = term_ids[0];
    for &nxt in &term_ids[1..] {
        acc = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: acc,
            right: nxt,
        });
    }
    acc
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{
        BinOp, ConstraintRepr, ConstraintSense, ExprArena, ExprNode, ModelRepr, ObjectiveSense,
        VarInfo, VarType,
    };

    fn scalar_var(arena: &mut ExprArena, name: &str, index: usize) -> ExprId {
        arena.add(ExprNode::Variable {
            name: name.to_string(),
            index,
            size: 1,
            shape: vec![],
        })
    }

    fn vinfo(name: &str, vt: VarType, lb: f64, ub: f64) -> VarInfo {
        VarInfo {
            name: name.to_string(),
            var_type: vt,
            offset: 0,
            size: 1,
            shape: vec![],
            lb: vec![lb],
            ub: vec![ub],
        }
    }

    fn term(arena: &mut ExprArena, c: f64, x: ExprId) -> ExprId {
        let cn = arena.add(ExprNode::Constant(c));
        arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: cn,
            right: x,
        })
    }

    fn add_terms(arena: &mut ExprArena, a: ExprId, b: ExprId) -> ExprId {
        arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: a,
            right: b,
        })
    }

    /// Compute the row activity `Σ aⱼ xⱼ` from a constraint body. Used
    /// to verify the strengthened constraint at integer corners.
    fn eval_at(arena: &ExprArena, body: ExprId, vals: &[f64]) -> f64 {
        match arena.get(body) {
            ExprNode::Constant(v) => *v,
            ExprNode::Variable { index, .. } => vals[*index],
            ExprNode::BinaryOp { op, left, right } => {
                let l = eval_at(arena, *left, vals);
                let r = eval_at(arena, *right, vals);
                match op {
                    BinOp::Add => l + r,
                    BinOp::Sub => l - r,
                    BinOp::Mul => l * r,
                    BinOp::Div => l / r,
                    _ => f64::NAN,
                }
            }
            _ => f64::NAN,
        }
    }

    #[test]
    fn strengthens_binary_when_slack_smaller_than_coeff() {
        // 5·b + 1·y ≤ 5 with b ∈ {0, 1}, y ∈ [0, 2].
        // U_minus_b = 1 · 2 = 2; slack = 5 - 2 = 3; 0 < 3 < 5 ⇒
        // a_b' = 5 - 3 = 2; b' = 5 - 3 = 2.
        let mut arena = ExprArena::new();
        let b = scalar_var(&mut arena, "b", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let t1 = term(&mut arena, 5.0, b);
        let t2 = term(&mut arena, 1.0, y);
        let body = add_terms(&mut arena, t1, t2);
        let model = ModelRepr {
            arena,
            objective: b,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 5.0,
                name: None,
            }],
            variables: vec![
                vinfo("b", VarType::Binary, 0.0, 1.0),
                vinfo("y", VarType::Continuous, 0.0, 2.0),
            ],
            n_vars: 2,
        };
        let (out, stats) = coefficient_strengthening(&model);
        assert_eq!(stats.constraints_strengthened, 1);
        assert_eq!(stats.coefficients_strengthened, 1);
        // At b=1, y=2: original LHS = 5 + 2 = 7 > 5 ⇒ infeasible.
        // Strengthened: a_b' = 2, b' = 2, so LHS = 2 + 2 = 4 > 2 ⇒
        // also infeasible (consistent).
        let lhs = eval_at(&out.arena, out.constraints[0].body, &[1.0, 2.0]);
        assert!((lhs - 4.0).abs() < 1e-9, "expected 4, got {lhs}");
        assert!((out.constraints[0].rhs - 2.0).abs() < 1e-9);
        // At b=0, y=2: rest = 2 ≤ 2 ⇒ feasible (consistent with original).
        let lhs0 = eval_at(&out.arena, out.constraints[0].body, &[0.0, 2.0]);
        assert!(lhs0 - 2.0 <= 1e-9);
    }

    #[test]
    fn skips_when_slack_already_meets_coeff() {
        // 2·b + 1·y ≤ 10, y ∈ [0, 2]. U_minus_b = 2; slack = 8 ≥ 2 ⇒
        // nothing to strengthen.
        let mut arena = ExprArena::new();
        let b = scalar_var(&mut arena, "b", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let t1 = term(&mut arena, 2.0, b);
        let t2 = term(&mut arena, 1.0, y);
        let body = add_terms(&mut arena, t1, t2);
        let model = ModelRepr {
            arena,
            objective: b,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 10.0,
                name: None,
            }],
            variables: vec![
                vinfo("b", VarType::Binary, 0.0, 1.0),
                vinfo("y", VarType::Continuous, 0.0, 2.0),
            ],
            n_vars: 2,
        };
        let (_, stats) = coefficient_strengthening(&model);
        assert_eq!(stats.constraints_strengthened, 0);
        assert_eq!(stats.coefficients_strengthened, 0);
    }

    #[test]
    fn skips_nonbinary_integer() {
        // 5·z + 1·y ≤ 5 with z ∈ {0, 3} integer (not binary). v0 skips.
        let mut arena = ExprArena::new();
        let z = scalar_var(&mut arena, "z", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let t1 = term(&mut arena, 5.0, z);
        let t2 = term(&mut arena, 1.0, y);
        let body = add_terms(&mut arena, t1, t2);
        let model = ModelRepr {
            arena,
            objective: z,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 5.0,
                name: None,
            }],
            variables: vec![
                vinfo("z", VarType::Integer, 0.0, 3.0),
                vinfo("y", VarType::Continuous, 0.0, 2.0),
            ],
            n_vars: 2,
        };
        let (_, stats) = coefficient_strengthening(&model);
        assert_eq!(stats.constraints_strengthened, 0);
    }

    #[test]
    fn handles_ge_via_reflection() {
        // -5·b + 1·y ≥ -3 with y ∈ [-2, 0] reflects to
        // 5·b - y ≤ 3. In the reflected ≤ row:
        //   coeff(b)=5 (positive binary), coeff(y)=-1, y ∈ [-2, 0].
        //   U_minus_b = max(-1·(-2), -1·0) = 2.
        //   slack = 3 - 2 = 1; 0 < 1 < 5 ⇒ a_b' = 4, b' = 2.
        // Reflect back: -4·b + 1·y ≥ -2.
        //   At (b=1, y=0): old LHS = -5 ≥ -3? FALSE. New LHS = -4 ≥ -2? FALSE. ✓
        //   At (b=1, y=-2): old LHS = -7 ≥ -3? FALSE. New LHS = -6 ≥ -2? FALSE. ✓
        //   At (b=0, y=-2): old LHS = -2 ≥ -3? TRUE. New LHS = -2 ≥ -2? TRUE. ✓
        //   At (b=0, y=0): both LHS = 0. Both feasible. ✓
        let mut arena = ExprArena::new();
        let b = scalar_var(&mut arena, "b", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let t1 = term(&mut arena, -5.0, b);
        let t2 = term(&mut arena, 1.0, y);
        let body = add_terms(&mut arena, t1, t2);
        let model = ModelRepr {
            arena,
            objective: b,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Ge,
                rhs: -3.0,
                name: None,
            }],
            variables: vec![
                vinfo("b", VarType::Binary, 0.0, 1.0),
                vinfo("y", VarType::Continuous, -2.0, 0.0),
            ],
            n_vars: 2,
        };
        let (out, stats) = coefficient_strengthening(&model);
        assert_eq!(stats.constraints_strengthened, 1);
        assert!((out.constraints[0].rhs - (-2.0)).abs() < 1e-9);
        // Verify all four binary corners.
        for b_v in [0.0, 1.0] {
            for y_v in [-2.0, 0.0] {
                let orig = -5.0 * b_v + y_v >= -3.0 - 1e-9;
                let new_lhs = eval_at(&out.arena, out.constraints[0].body, &[b_v, y_v]);
                let new = new_lhs >= out.constraints[0].rhs - 1e-9;
                assert_eq!(orig, new, "ge corner ({b_v}, {y_v}) differs");
            }
        }
    }

    #[test]
    fn skips_nonlinear_constraint() {
        // x² + b ≤ 5 — body is non-linear, skip.
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let b = scalar_var(&mut arena, "b", 1);
        let xx = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: x,
            right: x,
        });
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: xx,
            right: b,
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
            variables: vec![
                vinfo("x", VarType::Continuous, 0.0, 5.0),
                vinfo("b", VarType::Binary, 0.0, 1.0),
            ],
            n_vars: 2,
        };
        let (_, stats) = coefficient_strengthening(&model);
        assert_eq!(stats.rows_examined, 0);
    }

    #[test]
    fn strengthening_preserves_binary_corners() {
        // Direct correctness check: at every binary corner the
        // strengthened row is satisfied iff the original was.
        // 5·b1 + 4·b2 + 1·y ≤ 6 with b1, b2 ∈ {0,1}, y ∈ [0, 2].
        // U_minus_b1 = 4 + 2 = 6; slack = 0 ⇒ no positive a' (skip).
        // U_minus_b2 = 5 + 2 = 7; slack = -1 < 0 ⇒ skip.
        // Tighter rhs = 7: U_minus_b1 = 6 ⇒ slack 1 < 5 ⇒ a_b1 → 1.
        let mut arena = ExprArena::new();
        let b1 = scalar_var(&mut arena, "b1", 0);
        let b2 = scalar_var(&mut arena, "b2", 1);
        let y = scalar_var(&mut arena, "y", 2);
        let t1 = term(&mut arena, 5.0, b1);
        let t2 = term(&mut arena, 4.0, b2);
        let t3 = term(&mut arena, 1.0, y);
        let mid = add_terms(&mut arena, t1, t2);
        let body = add_terms(&mut arena, mid, t3);
        let model = ModelRepr {
            arena,
            objective: b1,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 7.0,
                name: None,
            }],
            variables: vec![
                vinfo("b1", VarType::Binary, 0.0, 1.0),
                vinfo("b2", VarType::Binary, 0.0, 1.0),
                vinfo("y", VarType::Continuous, 0.0, 2.0),
            ],
            n_vars: 3,
        };
        let (out, stats) = coefficient_strengthening(&model);
        assert!(stats.coefficients_strengthened >= 1);
        // Binary-corner sweep: (b1, b2) ∈ {0,1}², y ∈ {0, 2}.
        for b1_v in [0.0, 1.0] {
            for b2_v in [0.0, 1.0] {
                for y_v in [0.0, 1.0, 2.0] {
                    let orig = 5.0 * b1_v + 4.0 * b2_v + y_v <= 7.0 + 1e-9;
                    let new = eval_at(&out.arena, out.constraints[0].body, &[b1_v, b2_v, y_v])
                        <= 7.0 + 1e-9;
                    assert_eq!(orig, new, "corner ({b1_v}, {b2_v}, {y_v}) differs");
                }
            }
        }
    }
}
