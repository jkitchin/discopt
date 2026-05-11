//! Reduction-constraint detection in sparse polynomials (D3 of #53).
//!
//! Detects polynomial constraints whose structure forces variables to
//! specific values. The canonical pattern is a sum-of-squares (or
//! sum-of-even-powers) bounded above by a non-positive value:
//!
//! ```text
//! Σᵢ cᵢ · ∏ⱼ xⱼ^(2 kⱼ)   ≤   r    with cᵢ > 0 and r ≤ 0
//! ```
//!
//! Each monomial in the sum is non-negative on the entire real line
//! (every factor has an even exponent and the coefficient is positive),
//! so the only way the inequality holds is if every monomial equals
//! zero. That in turn forces every variable that appears in any
//! monomial to equal zero. The constraint then becomes redundant.
//!
//! The same logic handles `body = r` with `r ≤ 0` (forces zero), and
//! flags infeasibility when `body ≤ r` with `r < 0` strictly (a sum of
//! non-negatives cannot be strictly negative).
//!
//! ## References
//!
//! Smith & Pantelides (1999), *A symbolic reformulation/spatial branch-
//! and-bound algorithm for the global optimisation of nonconvex MINLPs*,
//! Comput. Chem. Eng. 23(4-5).
//! Chachuat, MC++ documentation — implicit equality detection.
//!
//! ## Dependencies
//!
//! Shares the polynomial-detection machinery with D2
//! (`reformulate_polynomial`); operates on the same ``try_polynomial``
//! decomposition, so the cost is a single arena walk per constraint
//! plus a constant amount of arithmetic.

use crate::expr::{ConstraintSense, ExprId, ExprNode, ModelRepr, VarType};

use super::fbbt::Interval;
use super::polynomial::try_polynomial;

/// Per-pass accounting + side-effects of a reduction-constraint sweep.
#[derive(Debug, Clone, Default)]
pub struct ReductionStats {
    /// Number of constraints scanned.
    pub constraints_examined: usize,
    /// Endpoint updates that strictly tightened a variable bound.
    pub bounds_tightened: usize,
    /// Variable block indices fixed to {0}.
    pub vars_fixed_to_zero: Vec<usize>,
    /// Constraint indices that became redundant after fixing.
    pub constraints_made_redundant: Vec<usize>,
    /// True iff the structural analysis proved infeasibility.
    pub infeasible: bool,
}

/// Detect reduction constraints and tighten ``bounds`` accordingly.
///
/// Mutates ``bounds`` in place (intersecting with `[0, 0]` for any
/// variable forced to zero). Does NOT remove the redundant constraints
/// from the model — that is the orchestrator/redundancy pass's job;
/// here we only record their indices so the redundancy pass can drop
/// them on the next sweep.
pub fn detect_reduction_constraints(model: &ModelRepr, bounds: &mut [Interval]) -> ReductionStats {
    let leaves = build_leaf_to_block_map(model);
    let mut stats = ReductionStats::default();

    for (ci, c) in model.constraints.iter().enumerate() {
        stats.constraints_examined += 1;
        let poly = match try_polynomial(&model.arena, c.body) {
            Some(p) => p,
            None => continue,
        };
        if poly.monomials.is_empty() {
            continue;
        }
        // Sum-of-non-negatives: every monomial coefficient strictly
        // positive AND every factor exponent even. (Even exponents make
        // each factor non-negative; positive coefficients keep the sum
        // non-negative.)
        let nonneg = poly
            .monomials
            .iter()
            .all(|m| m.coeff > 0.0 && m.factors.iter().all(|(_, e)| *e % 2 == 0));
        if !nonneg {
            continue;
        }

        // Need every leaf to resolve to a scalar variable block — Index
        // nodes pointing into multi-element blocks aren't supported in
        // v0 because tightening one element doesn't have a clean
        // representation in the per-block ``Interval`` array.
        let mut vars_in_poly: Vec<usize> = Vec::new();
        let mut bail = false;
        for m in &poly.monomials {
            for (leaf, _) in &m.factors {
                match leaves.get(leaf) {
                    Some(&block) => {
                        if !vars_in_poly.contains(&block) {
                            vars_in_poly.push(block);
                        }
                    }
                    None => {
                        bail = true;
                        break;
                    }
                }
            }
            if bail {
                break;
            }
        }
        if bail {
            continue;
        }

        // body = sum_of_nonneg_monomials + constant
        // Constraint: body OP rhs ⇔ sum OP (rhs - constant)
        let target = c.rhs - poly.constant;

        // Infeasibility checks (sum ≥ 0 always for valid bound regions).
        // Mark the first variable's bounds empty as a structural witness
        // so the orchestrator detects infeasibility via its standard
        // empty-interval path.
        let infeas = matches!(
            (c.sense, target < -1e-12),
            (ConstraintSense::Le, true) | (ConstraintSense::Eq, true)
        );
        if infeas {
            if let Some(&first_block) = vars_in_poly.first() {
                bounds[first_block] = Interval::empty();
            }
            stats.infeasible = true;
            return stats;
        }

        // Force-zero condition.
        let force_zero = match c.sense {
            // sum ≤ target with target ≤ 0 ⇒ sum = 0 (since sum ≥ 0)
            ConstraintSense::Le => target <= 1e-12,
            // sum = target with target ≤ 0 ⇒ sum = 0 + infeasibility caught
            // above for target < 0.
            ConstraintSense::Eq => target.abs() <= 1e-12,
            // sum ≥ target gives no information when target ≤ 0 (always true).
            ConstraintSense::Ge => false,
        };

        if !force_zero {
            continue;
        }

        // Each variable that appears in any monomial must be exactly 0.
        for block in &vars_in_poly {
            let cur = bounds[*block];
            let new_lo = cur.lo.max(0.0);
            let new_hi = cur.hi.min(0.0);
            if new_lo > new_hi + 1e-12 {
                // Empty interval — write an explicitly-empty interval so
                // downstream orchestrator infeasibility detection picks
                // this up via its standard ``Interval::is_empty`` path.
                bounds[*block] = Interval::empty();
                stats.infeasible = true;
                return stats;
            }
            let mut changed = false;
            if new_lo > cur.lo + 1e-12 {
                changed = true;
                stats.bounds_tightened += 1;
            }
            if new_hi < cur.hi - 1e-12 {
                changed = true;
                stats.bounds_tightened += 1;
            }
            bounds[*block] = Interval::new(new_lo, new_hi);
            if changed && !stats.vars_fixed_to_zero.contains(block) {
                stats.vars_fixed_to_zero.push(*block);
            }
        }
        stats.constraints_made_redundant.push(ci);
    }

    // Sort outputs for determinism (preserves the contract that no
    // delta is order-dependent on HashMap iteration).
    stats.vars_fixed_to_zero.sort_unstable();
    stats.constraints_made_redundant.sort_unstable();
    stats
}

/// Build a map from polynomial leaf `ExprId` to its scalar variable
/// block index. Restricted to size-1 `Variable` nodes; aliased blocks
/// (multiple arena leaves for the same block) are excluded so that the
/// pass never tightens a bound it cannot back out cleanly.
fn build_leaf_to_block_map(model: &ModelRepr) -> std::collections::HashMap<ExprId, usize> {
    let mut by_index: std::collections::HashMap<usize, ExprId> = std::collections::HashMap::new();
    let mut multi: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for nid in 0..model.arena.len() {
        if let ExprNode::Variable { index, size, .. } = model.arena.get(ExprId(nid)) {
            if *size != 1 {
                continue;
            }
            if by_index.insert(*index, ExprId(nid)).is_some() {
                multi.insert(*index);
            }
        }
    }
    let mut out: std::collections::HashMap<ExprId, usize> = std::collections::HashMap::new();
    for (idx, leaf) in by_index {
        if multi.contains(&idx) {
            continue;
        }
        if idx >= model.variables.len() {
            continue;
        }
        let v = &model.variables[idx];
        if v.size != 1 {
            continue;
        }
        // Continuous and integer variables: zero is always representable.
        // Binary variables: 0 is in domain too.
        match v.var_type {
            VarType::Continuous | VarType::Integer | VarType::Binary => {
                out.insert(leaf, idx);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{
        BinOp, ConstraintRepr, ConstraintSense, ExprArena, ExprNode, ModelRepr, ObjectiveSense,
        VarInfo, VarType,
    };

    fn make_var(arena: &mut ExprArena, name: &str, idx: usize) -> ExprId {
        arena.add(ExprNode::Variable {
            name: name.into(),
            index: idx,
            size: 1,
            shape: vec![],
        })
    }

    fn varinfo(name: &str, lo: f64, hi: f64) -> VarInfo {
        VarInfo {
            name: name.into(),
            var_type: VarType::Continuous,
            offset: 0,
            size: 1,
            shape: vec![],
            lb: vec![lo],
            ub: vec![hi],
        }
    }

    fn x_squared(arena: &mut ExprArena, x: ExprId) -> ExprId {
        let two = arena.add(ExprNode::Constant(2.0));
        arena.add(ExprNode::BinaryOp {
            op: BinOp::Pow,
            left: x,
            right: two,
        })
    }

    #[test]
    fn detects_sum_of_squares_le_zero() {
        // x^2 + y^2 ≤ 0 ⇒ x = y = 0.
        let mut arena = ExprArena::new();
        let x = make_var(&mut arena, "x", 0);
        let y = make_var(&mut arena, "y", 1);
        let xsq = x_squared(&mut arena, x);
        let ysq = x_squared(&mut arena, y);
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: xsq,
            right: ysq,
        });
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 0.0,
                name: None,
            }],
            variables: vec![varinfo("x", -3.0, 3.0), varinfo("y", -3.0, 3.0)],
            n_vars: 2,
        };
        let mut bounds = vec![Interval::new(-3.0, 3.0), Interval::new(-3.0, 3.0)];
        let stats = detect_reduction_constraints(&model, &mut bounds);
        assert!(!stats.infeasible);
        assert_eq!(stats.constraints_made_redundant, vec![0]);
        assert_eq!(stats.vars_fixed_to_zero, vec![0, 1]);
        assert_eq!(stats.bounds_tightened, 4); // 2 lo + 2 hi
        assert_eq!(bounds[0].lo, 0.0);
        assert_eq!(bounds[0].hi, 0.0);
        assert_eq!(bounds[1].lo, 0.0);
        assert_eq!(bounds[1].hi, 0.0);
    }

    #[test]
    fn detects_sum_of_squares_eq_zero() {
        let mut arena = ExprArena::new();
        let x = make_var(&mut arena, "x", 0);
        let body = x_squared(&mut arena, x);
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Eq,
                rhs: 0.0,
                name: None,
            }],
            variables: vec![varinfo("x", -2.0, 2.0)],
            n_vars: 1,
        };
        let mut bounds = vec![Interval::new(-2.0, 2.0)];
        let stats = detect_reduction_constraints(&model, &mut bounds);
        assert!(!stats.infeasible);
        assert_eq!(stats.vars_fixed_to_zero, vec![0]);
        assert_eq!(bounds[0].lo, 0.0);
        assert_eq!(bounds[0].hi, 0.0);
    }

    #[test]
    fn flags_infeasible_for_negative_target() {
        // x^2 ≤ -1 is infeasible.
        let mut arena = ExprArena::new();
        let x = make_var(&mut arena, "x", 0);
        let body = x_squared(&mut arena, x);
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: -1.0,
                name: None,
            }],
            variables: vec![varinfo("x", -2.0, 2.0)],
            n_vars: 1,
        };
        let mut bounds = vec![Interval::new(-2.0, 2.0)];
        let stats = detect_reduction_constraints(&model, &mut bounds);
        assert!(stats.infeasible);
    }

    #[test]
    fn skips_non_sos_polynomials() {
        // x^2 + y ≤ 0 — y has odd exponent, not a sum of non-negatives.
        let mut arena = ExprArena::new();
        let x = make_var(&mut arena, "x", 0);
        let y = make_var(&mut arena, "y", 1);
        let xsq = x_squared(&mut arena, x);
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: xsq,
            right: y,
        });
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 0.0,
                name: None,
            }],
            variables: vec![varinfo("x", -2.0, 2.0), varinfo("y", -2.0, 2.0)],
            n_vars: 2,
        };
        let mut bounds = vec![Interval::new(-2.0, 2.0), Interval::new(-2.0, 2.0)];
        let stats = detect_reduction_constraints(&model, &mut bounds);
        assert!(!stats.infeasible);
        assert!(stats.vars_fixed_to_zero.is_empty());
        assert!(stats.constraints_made_redundant.is_empty());
    }

    #[test]
    fn skips_when_target_is_strictly_positive() {
        // x^2 ≤ 4 — feasible, no forcing.
        let mut arena = ExprArena::new();
        let x = make_var(&mut arena, "x", 0);
        let body = x_squared(&mut arena, x);
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 4.0,
                name: None,
            }],
            variables: vec![varinfo("x", -3.0, 3.0)],
            n_vars: 1,
        };
        let mut bounds = vec![Interval::new(-3.0, 3.0)];
        let stats = detect_reduction_constraints(&model, &mut bounds);
        assert!(stats.vars_fixed_to_zero.is_empty());
        assert!(stats.constraints_made_redundant.is_empty());
    }

    #[test]
    fn handles_constant_offset() {
        // x^2 + 1 ≤ 1 ⇒ x = 0.
        let mut arena = ExprArena::new();
        let x = make_var(&mut arena, "x", 0);
        let xsq = x_squared(&mut arena, x);
        let one = arena.add(ExprNode::Constant(1.0));
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: xsq,
            right: one,
        });
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
            variables: vec![varinfo("x", -2.0, 2.0)],
            n_vars: 1,
        };
        let mut bounds = vec![Interval::new(-2.0, 2.0)];
        let stats = detect_reduction_constraints(&model, &mut bounds);
        assert_eq!(stats.vars_fixed_to_zero, vec![0]);
        assert_eq!(bounds[0].lo, 0.0);
        assert_eq!(bounds[0].hi, 0.0);
    }

    #[test]
    fn detects_higher_even_powers() {
        // x^4 + y^6 ≤ 0 ⇒ x = y = 0.
        let mut arena = ExprArena::new();
        let x = make_var(&mut arena, "x", 0);
        let y = make_var(&mut arena, "y", 1);
        let four = arena.add(ExprNode::Constant(4.0));
        let six = arena.add(ExprNode::Constant(6.0));
        let xq = arena.add(ExprNode::BinaryOp {
            op: BinOp::Pow,
            left: x,
            right: four,
        });
        let ys = arena.add(ExprNode::BinaryOp {
            op: BinOp::Pow,
            left: y,
            right: six,
        });
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: xq,
            right: ys,
        });
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 0.0,
                name: None,
            }],
            variables: vec![varinfo("x", -2.0, 2.0), varinfo("y", -2.0, 2.0)],
            n_vars: 2,
        };
        let mut bounds = vec![Interval::new(-2.0, 2.0), Interval::new(-2.0, 2.0)];
        let stats = detect_reduction_constraints(&model, &mut bounds);
        assert_eq!(stats.vars_fixed_to_zero, vec![0, 1]);
    }
}
