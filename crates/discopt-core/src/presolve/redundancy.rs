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
    /// Whether the values above were divided by the lead magnitude (exactly).
    /// An unscaled row is compared only with another unscaled row (#1817).
    scaled: bool,
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
    // #1817: only an exact zero is not a term. Dropping |c| < 1e-15 so that
    // "different spellings of the same row collapse" made two rows that differ
    // in a real term compare equal, and then deleted one of them.
    coeffs.retain(|(_, c)| *c != 0.0);
    if coeffs.is_empty() {
        return None;
    }
    let lead = coeffs[0].1;
    let scale = lead.abs();
    let flip = lead < 0.0;
    // #1817: every value the comparison reads must be the row's own value
    // exactly, or two rows that are not parallel (or a looser row and a tighter
    // one) can compare as the same. Divide by the lead magnitude only when every
    // quotient is exact; otherwise keep the row unscaled (sign-normalised only),
    // which still matches an identical row and never a merely nearby one.
    let rhs_shifted = exact_difference(rhs, poly.constant)?;
    let divides_exactly = coeffs
        .iter()
        .map(|&(_, c)| c)
        .chain(std::iter::once(rhs_shifted))
        .all(|v| (v / scale).mul_add(scale, -v) == 0.0);
    let (unit, scaled) = if divides_exactly {
        (scale, true)
    } else {
        (1.0, false)
    };
    for (_, c) in coeffs.iter_mut() {
        let q = *c / unit;
        *c = if flip { -q } else { q };
    }
    let q = rhs_shifted / unit;
    let rhs_norm = if flip { -q } else { q };
    let sense_norm = if flip { flip_sense(sense) } else { sense };
    Some(Signature {
        coeffs_norm: coeffs,
        scaled,
        sense_after_norm: sense_norm,
        rhs_after_norm: rhs_norm,
    })
}

/// `a - b` when the floating-point subtraction is exact, else `None`.
///
/// TwoSum (Knuth): `s + err == a - b` exactly, so `err == 0` means `s` is the
/// exact difference.
fn exact_difference(a: f64, b: f64) -> Option<f64> {
    let nb = -b;
    let s = a + nb;
    let bb = s - a;
    let err = (a - (s - bb)) + (nb - bb);
    if s.is_finite() && err == 0.0 {
        Some(s)
    } else {
        None
    }
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
    if a.scaled != b.scaled || a.coeffs_norm.len() != b.coeffs_norm.len() {
        return false;
    }
    for (pa, pb) in a.coeffs_norm.iter().zip(b.coeffs_norm.iter()) {
        if pa.0 != pb.0 {
            return false;
        }
        // #1817: exact. A 1e-9 tolerance called rows parallel that are not, and
        // the pass then deleted one of them.
        if pa.1 != pb.1 {
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
    // #1817: exact comparisons (the signature values are exact, see
    // `canonical_signature`). A row is dropped only when the other implies it:
    // the old 1e-9 "equal" band dropped a row up to 1e-9 tighter, and an
    // equality dropped every parallel inequality whatever its RHS -- with
    // `x + y == 5` and `x + y <= 3` the pass deleted the `<=` row and the
    // infeasible pair became feasible.
    let (ri, rj) = (i.rhs_after_norm, j.rhs_after_norm);
    match (i.sense_after_norm, j.sense_after_norm) {
        (Le, Le) => {
            if ri == rj {
                Dom::Equal
            } else if ri < rj {
                // i.rhs < j.rhs: i ≤ smaller is tighter than j ≤ bigger
                Dom::IDominatesJ
            } else {
                Dom::JDominatesI
            }
        }
        (Ge, Ge) => {
            if ri == rj {
                Dom::Equal
            } else if ri > rj {
                // i.rhs > j.rhs: i ≥ bigger is tighter than j ≥ smaller
                Dom::IDominatesJ
            } else {
                Dom::JDominatesI
            }
        }
        (Eq, Eq) => {
            if ri == rj {
                Dom::Equal
            } else {
                // Inconsistent equalities → infeasible. Leave for FBBT.
                Dom::Independent
            }
        }
        // An equality implies a parallel inequality only when it satisfies it.
        // Otherwise the pair is infeasible: keep both and leave it for FBBT.
        (Eq, Le) if ri <= rj => Dom::IDominatesJ,
        (Eq, Ge) if ri >= rj => Dom::IDominatesJ,
        (Le, Eq) if rj <= ri => Dom::JDominatesI,
        (Ge, Eq) if rj >= ri => Dom::JDominatesI,
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

    /// Two-variable model with rows `(cx, cy, sense, rhs)`, all over `x, y`.
    fn two_var_model(rows: &[(f64, f64, ConstraintSense, f64)]) -> ModelRepr {
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let constraints = rows
            .iter()
            .map(|&(cx, cy, sense, rhs)| ConstraintRepr {
                body: linear_2var(&mut arena, cx, x, cy, y),
                sense,
                rhs,
                name: None,
            })
            .collect();
        ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints,
            variables: vec![vinfo("x", 0.0, 10.0), vinfo("y", 0.0, 10.0)],
            n_vars: 2,
        }
    }

    // #1817: an equality dominated every parallel inequality whatever its RHS,
    // so `x + y == 5` with `2x + 2y <= 6` (infeasible) lost the `<=` row.
    #[test]
    fn equality_does_not_drop_an_inequality_it_violates() {
        use ConstraintSense::*;
        for rows in [
            [(1.0, 1.0, Eq, 5.0), (2.0, 2.0, Le, 6.0)],
            [(2.0, 2.0, Le, 6.0), (1.0, 1.0, Eq, 5.0)],
            [(1.0, 1.0, Eq, 5.0), (2.0, 2.0, Ge, 12.0)],
            [(2.0, 2.0, Ge, 12.0), (1.0, 1.0, Eq, 5.0)],
        ] {
            let (out, stats) = detect_row_redundancy(&two_var_model(&rows));
            assert_eq!(stats.constraints_removed, 0, "{rows:?}");
            assert_eq!(out.constraints.len(), 2);
        }
        // An equality that satisfies the inequality still drops it.
        let (out, stats) =
            detect_row_redundancy(&two_var_model(&[(2.0, 2.0, Le, 12.0), (1.0, 1.0, Eq, 5.0)]));
        assert_eq!(stats.removed_indices, vec![0]);
        assert_eq!(out.constraints[0].sense, Eq);
    }

    // #1817: the 1e-9 band called rows parallel (LHS) or equal (RHS) that are not.
    #[test]
    fn nearly_parallel_or_nearly_equal_rows_are_kept() {
        use ConstraintSense::*;
        let eps = 2.0_f64.powi(-40);
        for rows in [
            [(1.0, 1.0, Le, 5.0), (1.0, 1.0 + eps, Le, 5.0)],
            [(1.0, 1.0, Le, 5.0), (1.0, 1.0, Le, 5.0 - eps)],
        ] {
            let (_, stats) = detect_row_redundancy(&two_var_model(&rows));
            if rows[1].1 == 1.0 {
                // Same LHS, the second RHS is tighter: the FIRST row goes.
                assert_eq!(stats.removed_indices, vec![0], "{rows:?}");
            } else {
                assert_eq!(stats.constraints_removed, 0, "{rows:?}");
            }
        }
    }

    // #1817: a tiny coefficient is a term, not noise to drop before comparing.
    #[test]
    fn tiny_coefficient_is_not_dropped_before_comparing() {
        use ConstraintSense::*;
        let (_, stats) = detect_row_redundancy(&two_var_model(&[
            (1.0, 1e-17, Le, 5.0),
            (1.0, 0.0, Le, 5.0),
        ]));
        assert_eq!(stats.constraints_removed, 0);
    }

    // #1817: a normalisation that rounds is not used; the row is compared
    // unscaled, so an identical row still matches and a scaled copy does not.
    #[test]
    fn inexact_normalisation_compares_unscaled() {
        use ConstraintSense::*;
        // 0.1 / 3.0 rounds. The identical pair still collapses (looser goes)...
        let (_, stats) =
            detect_row_redundancy(&two_var_model(&[(3.0, 0.1, Le, 5.0), (3.0, 0.1, Le, 4.0)]));
        assert_eq!(stats.removed_indices, vec![0]);
        // ...but a copy scaled by 2 cannot be shown parallel with exact values
        // through a rounding division, so both are kept.
        let (_, stats) =
            detect_row_redundancy(&two_var_model(&[(3.0, 0.1, Le, 5.0), (6.0, 0.2, Le, 4.0)]));
        assert_eq!(stats.constraints_removed, 0);
        // Exactly representable multiples are still found.
        let (_, stats) =
            detect_row_redundancy(&two_var_model(&[(1.0, 1.0, Le, 5.0), (2.0, 2.0, Le, 6.0)]));
        assert_eq!(stats.removed_indices, vec![0]);
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
