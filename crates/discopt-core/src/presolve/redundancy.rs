//! Parallel-row and duplicate-constraint redundancy detection
//! (C3 of the presolve roadmap).
//!
//! ## What this pass does
//!
//! Drops constraints whose linear part duplicates another constraint
//! up to a positive scalar factor and a possibly looser RHS. Three
//! patterns are handled:
//!
//! 1. **Identical row, identical sense**: the second constraint is a
//!    duplicate. Drop it.
//! 2. **Parallel row, same sense, looser RHS**: the looser one is
//!    dominated by the tighter. Drop the looser.
//! 3. **Parallel row, opposite senses, RHS encloses an equality**:
//!    e.g. `2x + y ≤ 5` and `2x + y ≥ 5` together imply equality;
//!    promote the inequality to equality and drop the duplicate. v0
//!    skips this case to keep the rewrite local; equality detection
//!    is handled by FBBT and `eliminate`.
//!
//! ## Distinction from `simplify`
//!
//! [`super::simplify::simplify`] flags a constraint redundant when
//! its **interval evaluation** under the current bounds already
//! satisfies the RHS. That's interval-redundancy: the constraint
//! itself is locally trivial. This pass is row-redundancy: the
//! constraint is dominated by **another** constraint in the model,
//! independent of variable bounds.
//!
//! ## Scope (intentional, conservative)
//!
//! - Linear constraints only — those whose body interpreted by
//!   [`super::polynomial::try_polynomial`] has total degree exactly 1.
//! - Constraints with a constant body or no recognised polynomial form
//!   are left for `simplify` to handle.
//! - Pairwise comparison is `O(n²)` in the number of linear
//!   constraints. v0 ships this; a hash-keyed signature pass is
//!   planned once profiling shows it is needed.
//!
//! ## Determinism
//!
//! Constraints are scanned in `model.constraints` order. When a pair
//! is dominated, the one with the **larger index** is removed so the
//! relative order of surviving constraints is stable. No `HashMap`
//! iteration on the hot path.

use std::collections::HashSet;

use super::polynomial::try_polynomial;
use crate::expr::{ConstraintSense, ExprId, ModelRepr};

/// Per-pass statistics from row-redundancy detection.
#[derive(Debug, Clone, Default)]
pub struct RedundancyStats {
    /// Number of constraints dropped as redundant duplicates.
    pub constraints_removed: usize,
    /// Indices of the removed constraints in the *input* model order.
    pub removed_indices: Vec<usize>,
    /// Number of pairs examined.
    pub pairs_examined: usize,
}

/// Run row-redundancy detection on `model`. Pure function.
pub fn detect_row_redundancy(model: &ModelRepr) -> (ModelRepr, RedundancyStats) {
    let mut stats = RedundancyStats::default();

    // Extract canonical signatures for each constraint that is a
    // pure linear polynomial.
    let n = model.constraints.len();
    let mut sigs: Vec<Option<Signature>> = Vec::with_capacity(n);
    for c in &model.constraints {
        sigs.push(canonical_signature(model, c.body, c.sense, c.rhs));
    }

    // Pairwise scan. The constraint with the LARGER index is
    // removed when dominated, so we mark a `dropped` set and rebuild.
    let mut dropped: HashSet<usize> = HashSet::new();
    for i in 0..n {
        if dropped.contains(&i) {
            continue;
        }
        let si = match &sigs[i] {
            Some(s) => s,
            None => continue,
        };
        for (j, sj_opt) in sigs.iter().enumerate().skip(i + 1) {
            if dropped.contains(&j) {
                continue;
            }
            let sj = match sj_opt {
                Some(s) => s,
                None => continue,
            };
            stats.pairs_examined += 1;
            if !same_normalised_lhs(si, sj) {
                continue;
            }
            // Same canonical LHS direction. Compare senses & RHS.
            // Note `rhs_after_norm` already accounts for sign-flip
            // applied during normalisation: the effective sense is
            // `si.sense_after_norm`.
            let dominate = dominates(si, sj);
            match dominate {
                Dom::IDominatesJ => {
                    dropped.insert(j);
                }
                Dom::JDominatesI => {
                    dropped.insert(i);
                    break;
                }
                Dom::Equal => {
                    // Identical → drop the larger index for stable
                    // surviving order.
                    dropped.insert(j);
                }
                Dom::Independent => {}
            }
        }
    }

    if dropped.is_empty() {
        return (model.clone(), stats);
    }

    let mut removed_sorted: Vec<usize> = dropped.iter().copied().collect();
    removed_sorted.sort_unstable();
    stats.constraints_removed = removed_sorted.len();
    stats.removed_indices = removed_sorted.clone();

    let mut out = model.clone();
    // Remove from the back so earlier indices stay valid.
    for &idx in removed_sorted.iter().rev() {
        out.constraints.remove(idx);
    }
    (out, stats)
}

// ─────────────────────────────────────────────────────────────
// Canonical signature
// ─────────────────────────────────────────────────────────────

/// Canonical form of a linear constraint after sign normalisation.
///
/// Coefficients are stored as `(leaf_id, coeff)` pairs sorted by
/// `leaf_id`. Sign is normalised so the first coefficient is
/// strictly positive; if a flip happened, both `sense_after_norm`
/// and `rhs_after_norm` reflect that.
#[derive(Debug, Clone)]
struct Signature {
    /// Canonical coefficient pairs, divided by the absolute value of
    /// the first non-zero coefficient and sign-flipped so leading is
    /// positive.
    coeffs_norm: Vec<(ExprId, f64)>,
    /// `Le` / `Eq` / `Ge` after normalisation.
    sense_after_norm: ConstraintSense,
    /// RHS after subtracting the polynomial constant offset, dividing
    /// by the same scaling factor, and possibly flipping sign.
    rhs_after_norm: f64,
}

/// Build a [`Signature`] for the constraint body. Returns `None` if
/// the body is not a degree-≤1 polynomial or has no variable terms.
fn canonical_signature(
    model: &ModelRepr,
    body: ExprId,
    sense: ConstraintSense,
    rhs: f64,
) -> Option<Signature> {
    let poly = try_polynomial(&model.arena, body)?;
    if poly.max_total_degree() > 1 {
        return None;
    }
    if poly.monomials.is_empty() {
        // Constant body: leave for simplify to detect.
        return None;
    }
    // Each monomial has factors == [(leaf, 1)] (degree 1).
    let mut coeffs: Vec<(ExprId, f64)> = poly
        .monomials
        .iter()
        .filter_map(|m| {
            if m.factors.len() != 1 || m.factors[0].1 != 1 {
                None
            } else {
                Some((m.factors[0].0, m.coeff))
            }
        })
        .collect();
    if coeffs.len() != poly.monomials.len() {
        return None;
    }
    coeffs.sort_by_key(|(id, _)| id.0);
    // try_polynomial canonicalises but in the `is_some` arm there
    // may still be near-zero coefficients; drop them so different
    // arena spellings of the same row collapse.
    coeffs.retain(|(_, c)| c.abs() > 1e-15);
    if coeffs.is_empty() {
        return None;
    }
    let lead = coeffs[0].1;
    let scale = lead.abs();
    let flip = lead < 0.0;
    for (_, c) in coeffs.iter_mut() {
        *c = if flip { -*c / scale } else { *c / scale };
    }
    let rhs_shifted = rhs - poly.constant;
    let rhs_norm = if flip {
        -rhs_shifted / scale
    } else {
        rhs_shifted / scale
    };
    let sense_norm = if flip { flip_sense(sense) } else { sense };
    Some(Signature {
        coeffs_norm: coeffs,
        sense_after_norm: sense_norm,
        rhs_after_norm: rhs_norm,
    })
}

fn flip_sense(s: ConstraintSense) -> ConstraintSense {
    match s {
        ConstraintSense::Le => ConstraintSense::Ge,
        ConstraintSense::Ge => ConstraintSense::Le,
        ConstraintSense::Eq => ConstraintSense::Eq,
    }
}

/// True if both signatures have identical normalised LHS coefficient
/// vectors.
fn same_normalised_lhs(a: &Signature, b: &Signature) -> bool {
    if a.coeffs_norm.len() != b.coeffs_norm.len() {
        return false;
    }
    for (pa, pb) in a.coeffs_norm.iter().zip(b.coeffs_norm.iter()) {
        if pa.0 != pb.0 {
            return false;
        }
        if (pa.1 - pb.1).abs() > 1e-9 {
            return false;
        }
    }
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dom {
    IDominatesJ,
    JDominatesI,
    Equal,
    Independent,
}

/// With identical normalised LHS, decide which of the two row
/// constraints is dominant.
fn dominates(i: &Signature, j: &Signature) -> Dom {
    use ConstraintSense::*;
    match (i.sense_after_norm, j.sense_after_norm) {
        (Le, Le) => {
            let d = i.rhs_after_norm - j.rhs_after_norm;
            if d.abs() < 1e-9 {
                Dom::Equal
            } else if d < 0.0 {
                // i.rhs < j.rhs: i ≤ smaller is tighter than j ≤ bigger
                Dom::IDominatesJ
            } else {
                Dom::JDominatesI
            }
        }
        (Ge, Ge) => {
            let d = i.rhs_after_norm - j.rhs_after_norm;
            if d.abs() < 1e-9 {
                Dom::Equal
            } else if d > 0.0 {
                // i.rhs > j.rhs: i ≥ bigger is tighter than j ≥ smaller
                Dom::IDominatesJ
            } else {
                Dom::JDominatesI
            }
        }
        (Eq, Eq) => {
            if (i.rhs_after_norm - j.rhs_after_norm).abs() < 1e-9 {
                Dom::Equal
            } else {
                // Inconsistent equalities → infeasible. Leave for FBBT.
                Dom::Independent
            }
        }
        (Eq, _) => Dom::IDominatesJ,
        (_, Eq) => Dom::JDominatesI,
        // Mixed inequality senses: skip in v0.
        _ => Dom::Independent,
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
            name: name.to_string(),
            index: idx,
            size: 1,
            shape: vec![],
        })
    }

    fn vinfo(name: &str, lb: f64, ub: f64) -> VarInfo {
        VarInfo {
            name: name.to_string(),
            var_type: VarType::Continuous,
            offset: 0,
            size: 1,
            shape: vec![],
            lb: vec![lb],
            ub: vec![ub],
        }
    }

    fn linear_2var(arena: &mut ExprArena, cx: f64, x: ExprId, cy: f64, y: ExprId) -> ExprId {
        let cx_n = arena.add(ExprNode::Constant(cx));
        let cy_n = arena.add(ExprNode::Constant(cy));
        let cxx = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: cx_n,
            right: x,
        });
        let cyy = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: cy_n,
            right: y,
        });
        arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: cxx,
            right: cyy,
        })
    }

    #[test]
    fn drops_duplicate_constraint() {
        // Two identical constraints: 2x + y <= 5.
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let b1 = linear_2var(&mut arena, 2.0, x, 1.0, y);
        let b2 = linear_2var(&mut arena, 2.0, x, 1.0, y);
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![
                ConstraintRepr {
                    body: b1,
                    sense: ConstraintSense::Le,
                    rhs: 5.0,
                    name: None,
                },
                ConstraintRepr {
                    body: b2,
                    sense: ConstraintSense::Le,
                    rhs: 5.0,
                    name: None,
                },
            ],
            variables: vec![vinfo("x", 0.0, 10.0), vinfo("y", 0.0, 10.0)],
            n_vars: 2,
        };
        let (out, stats) = detect_row_redundancy(&model);
        assert_eq!(stats.constraints_removed, 1);
        assert_eq!(out.constraints.len(), 1);
    }

    #[test]
    fn drops_dominated_le_constraint() {
        // 2x + y <= 5 dominates 2x + y <= 10.
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let b1 = linear_2var(&mut arena, 2.0, x, 1.0, y);
        let b2 = linear_2var(&mut arena, 2.0, x, 1.0, y);
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![
                ConstraintRepr {
                    body: b1,
                    sense: ConstraintSense::Le,
                    rhs: 5.0,
                    name: None,
                },
                ConstraintRepr {
                    body: b2,
                    sense: ConstraintSense::Le,
                    rhs: 10.0,
                    name: None,
                },
            ],
            variables: vec![vinfo("x", 0.0, 10.0), vinfo("y", 0.0, 10.0)],
            n_vars: 2,
        };
        let (out, stats) = detect_row_redundancy(&model);
        assert_eq!(stats.constraints_removed, 1);
        // Index 1 (the looser one) is removed.
        assert_eq!(stats.removed_indices, vec![1]);
        assert_eq!(out.constraints.len(), 1);
        assert_eq!(out.constraints[0].rhs, 5.0);
    }

    #[test]
    fn drops_parallel_scaled_row() {
        // 2x + y <= 5  vs  4x + 2y <= 12 (scaled by 2 → 2x+y <= 6).
        // First dominates second.
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let b1 = linear_2var(&mut arena, 2.0, x, 1.0, y);
        let b2 = linear_2var(&mut arena, 4.0, x, 2.0, y);
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![
                ConstraintRepr {
                    body: b1,
                    sense: ConstraintSense::Le,
                    rhs: 5.0,
                    name: None,
                },
                ConstraintRepr {
                    body: b2,
                    sense: ConstraintSense::Le,
                    rhs: 12.0,
                    name: None,
                },
            ],
            variables: vec![vinfo("x", 0.0, 10.0), vinfo("y", 0.0, 10.0)],
            n_vars: 2,
        };
        let (out, stats) = detect_row_redundancy(&model);
        assert_eq!(stats.constraints_removed, 1);
        assert_eq!(stats.removed_indices, vec![1]);
        assert_eq!(out.constraints.len(), 1);
    }

    #[test]
    fn drops_negated_parallel_row() {
        // 2x + y <= 5  vs  -4x - 2y >= -12 (the same as 2x+y <= 6).
        // First dominates.
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let b1 = linear_2var(&mut arena, 2.0, x, 1.0, y);
        let b2 = linear_2var(&mut arena, -4.0, x, -2.0, y);
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![
                ConstraintRepr {
                    body: b1,
                    sense: ConstraintSense::Le,
                    rhs: 5.0,
                    name: None,
                },
                ConstraintRepr {
                    body: b2,
                    sense: ConstraintSense::Ge,
                    rhs: -12.0,
                    name: None,
                },
            ],
            variables: vec![vinfo("x", 0.0, 10.0), vinfo("y", 0.0, 10.0)],
            n_vars: 2,
        };
        let (out, stats) = detect_row_redundancy(&model);
        assert_eq!(stats.constraints_removed, 1);
        assert_eq!(out.constraints.len(), 1);
    }

    #[test]
    fn keeps_independent_constraints() {
        // Different LHS shapes — neither dominates.
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let b1 = linear_2var(&mut arena, 2.0, x, 1.0, y);
        let b2 = linear_2var(&mut arena, 3.0, x, -1.0, y);
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![
                ConstraintRepr {
                    body: b1,
                    sense: ConstraintSense::Le,
                    rhs: 5.0,
                    name: None,
                },
                ConstraintRepr {
                    body: b2,
                    sense: ConstraintSense::Le,
                    rhs: 10.0,
                    name: None,
                },
            ],
            variables: vec![vinfo("x", 0.0, 10.0), vinfo("y", 0.0, 10.0)],
            n_vars: 2,
        };
        let (out, stats) = detect_row_redundancy(&model);
        assert_eq!(stats.constraints_removed, 0);
        assert_eq!(out.constraints.len(), 2);
    }

    #[test]
    fn equality_dominates_inequality() {
        // 2x + y == 5 dominates 2x + y <= 7 (== implies <=).
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let b1 = linear_2var(&mut arena, 2.0, x, 1.0, y);
        let b2 = linear_2var(&mut arena, 2.0, x, 1.0, y);
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![
                ConstraintRepr {
                    body: b1,
                    sense: ConstraintSense::Eq,
                    rhs: 5.0,
                    name: None,
                },
                ConstraintRepr {
                    body: b2,
                    sense: ConstraintSense::Le,
                    rhs: 7.0,
                    name: None,
                },
            ],
            variables: vec![vinfo("x", 0.0, 10.0), vinfo("y", 0.0, 10.0)],
            n_vars: 2,
        };
        let (out, stats) = detect_row_redundancy(&model);
        assert_eq!(stats.constraints_removed, 1);
        assert_eq!(stats.removed_indices, vec![1]);
        assert_eq!(out.constraints.len(), 1);
        assert_eq!(out.constraints[0].sense, ConstraintSense::Eq);
    }

    #[test]
    fn skips_nonlinear_constraints() {
        // x^2 + y <= 5: not linear, signature returns None.
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let xx = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: x,
            right: x,
        });
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: xx,
            right: y,
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
            variables: vec![vinfo("x", 0.0, 10.0), vinfo("y", 0.0, 10.0)],
            n_vars: 2,
        };
        let (_, stats) = detect_row_redundancy(&model);
        assert_eq!(stats.constraints_removed, 0);
    }
}
