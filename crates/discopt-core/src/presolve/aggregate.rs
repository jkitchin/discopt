//! Variable aggregation / affine substitution (C1 of the presolve
//! roadmap).
//!
//! ## What this pass does
//!
//! Detects equalities of the form
//!
//! ```text
//!     c_x · x + c_y · y + c0 == rhs        (c_x ≠ 0, c_y ≠ 0)
//! ```
//!
//! where the eliminable target `x` is a continuous, scalar variable
//! that appears **only in this equality** — neither in the objective
//! nor in any other constraint. When such a pair is found:
//!
//! 1. The equality is dropped from the model.
//! 2. The variable block for `x` is removed, and every `Variable`
//!    node whose `index` exceeds `x`'s block is renumbered.
//! 3. The surviving variable `y`'s bounds are tightened to the
//!    intersection of its current bounds and the bounds implied by
//!    `x`'s original interval under the relation
//!    `y = (rhs − c0 − c_x · x) / c_y`.
//!
//! The aggregation is recorded in [`AggregationStats::aggregations`]
//! so a post-solve recovery step can recompute `x` from `y`.
//!
//! ## Distinction from `eliminate`
//!
//! [`super::eliminate::eliminate_variables`] handles **singleton**
//! equalities: one variable, one equality, fix to a constant. This
//! pass handles the next case up: **two** variables in an affine
//! equality where one is otherwise unused. Together they cover the
//! "variable is fully determined by a linear equation" branch of
//! standard MIP presolve, modulo the "otherwise unused" condition
//! that keeps v0 contained.
//!
//! ## Scope (intentional, conservative)
//!
//! - Continuous, scalar variables only on both sides.
//! - Body must be polynomial of total degree 1, with exactly 2 distinct
//!   variable leaves.
//! - The eliminable target must appear in exactly **one** expression
//!   (the determining equality itself). Any other appearance — in
//!   the objective or another constraint — disqualifies the
//!   candidate.
//! - The implied bound on the surviving variable must intersect its
//!   current `[lb, ub]` non-trivially. Empty intersection signals
//!   infeasibility, which is left for FBBT to detect.
//!
//! ## Determinism
//!
//! Equalities are scanned in `model.constraints` order; the eliminable
//! target is chosen by the lower variable-block index when both leaves
//! are eligible. No `HashMap` iteration on the hot path.

use std::collections::HashSet;

use super::polynomial::try_polynomial;
use crate::expr::{ConstraintSense, ExprArena, ExprId, ExprNode, ModelRepr, VarInfo, VarType};

/// Per-pass statistics from variable aggregation.
#[derive(Debug, Clone, Default)]
pub struct AggregationStats {
    /// Number of variable blocks removed.
    pub variables_aggregated: usize,
    /// Number of equalities dropped (one per aggregation).
    pub equalities_dropped: usize,
    /// Number of equality candidates examined.
    pub candidates_examined: usize,
    /// Recorded substitutions, in the order applied.
    pub aggregations: Vec<AggregationRecord>,
}

/// Substitution record: `eliminated = coeff * source + offset`.
///
/// Block indices are reported as they were **before** any renumbering
/// caused by this pass — i.e., they refer to the input model. The
/// orchestrator does not currently chain post-solve recovery, but the
/// record is shaped so a future post-solve pass can pop them in
/// reverse order and reconstruct the eliminated values.
#[derive(Debug, Clone)]
pub struct AggregationRecord {
    /// Block index of the eliminated variable, in the *input* model.
    pub eliminated_block: usize,
    /// Block index of the surviving variable, in the *input* model.
    pub source_block: usize,
    /// Linear coefficient: `eliminated = coeff * source + offset`.
    pub coeff: f64,
    /// Affine offset.
    pub offset: f64,
}

/// Run variable aggregation on `model`. Pure function.
pub fn aggregate_variables(model: &ModelRepr) -> (ModelRepr, AggregationStats) {
    let mut out = model.clone();
    let mut stats = AggregationStats::default();

    // Iterate to a fixed point. Each successful aggregation may
    // expose another (e.g. when removing x lets y become a singleton
    // equality target on a different constraint), but we only handle
    // one per outer loop and let the orchestrator interleave with the
    // other passes.
    loop {
        let n_before = stats.variables_aggregated;
        if let Some((new_model, record)) = try_one_aggregation(&out, &mut stats.candidates_examined)
        {
            out = new_model;
            stats.aggregations.push(record);
            stats.variables_aggregated += 1;
            stats.equalities_dropped += 1;
        }
        if stats.variables_aggregated == n_before {
            break;
        }
    }

    (out, stats)
}

/// Try to find and apply exactly one aggregation. Returns the new
/// model and the record on success, or `None` if no candidate
/// applied.
fn try_one_aggregation(
    model: &ModelRepr,
    candidates_examined: &mut usize,
) -> Option<(ModelRepr, AggregationRecord)> {
    // Build leaf → block index map for scalar continuous variables.
    let leaves = scalar_continuous_var_leaves(model);

    for (ci, c) in model.constraints.iter().enumerate() {
        if c.sense != ConstraintSense::Eq {
            continue;
        }
        let poly = match try_polynomial(&model.arena, c.body) {
            Some(p) if p.max_total_degree() == 1 => p,
            _ => continue,
        };

        // Collect (leaf_id, coeff) for monomials of degree exactly 1.
        // try_polynomial may emit duplicate-leaf monomials before
        // canonicalisation; canonicalise() inside try_polynomial dedupes.
        let mut linear: Vec<(ExprId, f64)> = Vec::with_capacity(2);
        let mut bail = false;
        for m in &poly.monomials {
            if m.factors.len() != 1 || m.factors[0].1 != 1 {
                bail = true;
                break;
            }
            linear.push((m.factors[0].0, m.coeff));
        }
        if bail || linear.len() != 2 {
            continue;
        }
        *candidates_examined += 1;

        // Both leaves must be scalar continuous variables we know about.
        let (leaf_a, ca) = linear[0];
        let (leaf_b, cb) = linear[1];
        let block_a = leaves.iter().find(|(_, l)| *l == leaf_a).map(|(b, _)| *b);
        let block_b = leaves.iter().find(|(_, l)| *l == leaf_b).map(|(b, _)| *b);
        let (block_a, block_b) = match (block_a, block_b) {
            (Some(a), Some(b)) if a != b => (a, b),
            _ => continue,
        };
        if ca == 0.0 || cb == 0.0 {
            continue;
        }

        // For each leaf, count its appearances in expressions OTHER
        // than this equality. Both must already be skipped — i.e.
        // appear nowhere else — for that leaf to be eliminable.
        let appears_a_elsewhere =
            leaf_appears_outside(model, ci, leaf_a) || leaf_block_aliased(model, block_a, leaf_a);
        let appears_b_elsewhere =
            leaf_appears_outside(model, ci, leaf_b) || leaf_block_aliased(model, block_b, leaf_b);

        // Pick the eliminable target: prefer the one with no other
        // appearances. If both qualify, choose the lower block index
        // for determinism.
        let pick_a = match (appears_a_elsewhere, appears_b_elsewhere) {
            (false, false) => block_a < block_b,
            (false, true) => true,
            (true, false) => false,
            (true, true) => continue,
        };
        let (elim_block, _elim_leaf, elim_coeff, src_block, _src_leaf, src_coeff) = if pick_a {
            (block_a, leaf_a, ca, block_b, leaf_b, cb)
        } else {
            (block_b, leaf_b, cb, block_a, leaf_a, ca)
        };

        // Substitution: eliminated = coeff * source + offset
        // From `c_e * e + c_s * s + p.constant == rhs`:
        //   e = (rhs - p.constant - c_s * s) / c_e
        let offset = (c.rhs - poly.constant) / elim_coeff;
        let coeff = -src_coeff / elim_coeff;
        if !offset.is_finite() || !coeff.is_finite() {
            continue;
        }

        // Tighten the surviving variable's bounds via the relation
        //   s = (rhs - p.constant - c_e * e) / c_s
        // applied over e's interval.
        let e_lb = model.variables[elim_block].lb[0];
        let e_ub = model.variables[elim_block].ub[0];
        let s_implied_lo;
        let s_implied_hi;
        {
            let (lo1, hi1) = scale_interval(elim_coeff, e_lb, e_ub);
            // Numerator interval = (rhs - p.constant) - [lo1, hi1]
            let num_lo = (c.rhs - poly.constant) - hi1;
            let num_hi = (c.rhs - poly.constant) - lo1;
            // Divide by src_coeff (constant; can be negative).
            let (a, b) = scale_interval(1.0 / src_coeff, num_lo, num_hi);
            s_implied_lo = a;
            s_implied_hi = b;
        }
        let cur_s_lb = model.variables[src_block].lb[0];
        let cur_s_ub = model.variables[src_block].ub[0];
        let new_s_lb = cur_s_lb.max(s_implied_lo);
        let new_s_ub = cur_s_ub.min(s_implied_hi);
        if new_s_lb > new_s_ub + 1e-12 {
            // Empty intersection — let FBBT report infeasibility on a
            // later pass instead of mutating here.
            continue;
        }

        // Apply the rewrite.
        let mut new_model = model.clone();
        new_model.constraints.remove(ci);
        new_model.variables.remove(elim_block);
        new_model.arena = renumber_variable_indices(&model.arena, elim_block);
        // Recompute n_vars + offsets.
        new_model.n_vars = 0;
        let mut running = 0usize;
        for v in new_model.variables.iter_mut() {
            v.offset = running;
            running += v.size;
        }
        new_model.n_vars = running;
        // Tighten s bounds.
        // After renumbering, src_block may have shifted by 1 if it
        // came after elim_block.
        let new_src_block = if src_block > elim_block {
            src_block - 1
        } else {
            src_block
        };
        new_model.variables[new_src_block].lb[0] = new_s_lb.max(cur_s_lb);
        new_model.variables[new_src_block].ub[0] = new_s_ub.min(cur_s_ub);

        let record = AggregationRecord {
            eliminated_block: elim_block,
            source_block: src_block,
            coeff,
            offset,
        };
        return Some((new_model, record));
    }

    None
}

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

/// Scale interval `[lo, hi]` by a scalar; preserves min/max ordering.
fn scale_interval(c: f64, lo: f64, hi: f64) -> (f64, f64) {
    let a = c * lo;
    let b = c * hi;
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Returns the unique `(block_index, expr_id)` pairs for scalar
/// continuous variables that have a `Variable` leaf in the arena.
/// Variables with multiple `Variable` nodes (alias situations) are
/// excluded — see [`leaf_block_aliased`].
fn scalar_continuous_var_leaves(model: &ModelRepr) -> Vec<(usize, ExprId)> {
    let mut by_index: std::collections::HashMap<usize, ExprId> = std::collections::HashMap::new();
    let mut multi: HashSet<usize> = HashSet::new();
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
    let mut out: Vec<(usize, ExprId)> = by_index
        .into_iter()
        .filter(|(idx, _)| !multi.contains(idx))
        .filter(|(idx, _)| {
            *idx < model.variables.len()
                && model.variables[*idx].var_type == VarType::Continuous
                && model.variables[*idx].size == 1
        })
        .collect();
    out.sort_by_key(|(idx, _)| *idx);
    out
}

/// True if the variable block has more than one `Variable` arena leaf.
/// Alias situations are out of scope for v0.
fn leaf_block_aliased(model: &ModelRepr, block: usize, _leaf: ExprId) -> bool {
    let mut count = 0;
    for nid in 0..model.arena.len() {
        if let ExprNode::Variable { index, size, .. } = model.arena.get(ExprId(nid)) {
            if *index == block && *size == 1 {
                count += 1;
                if count > 1 {
                    return true;
                }
            }
        }
    }
    false
}

/// True if `leaf` appears in any constraint body except `skip_ci`,
/// or in the objective.
fn leaf_appears_outside(model: &ModelRepr, skip_ci: usize, leaf: ExprId) -> bool {
    if expr_contains_leaf(&model.arena, model.objective, leaf) {
        return true;
    }
    for (ci, c) in model.constraints.iter().enumerate() {
        if ci == skip_ci {
            continue;
        }
        if expr_contains_leaf(&model.arena, c.body, leaf) {
            return true;
        }
    }
    false
}

fn expr_contains_leaf(arena: &ExprArena, root: ExprId, target: ExprId) -> bool {
    let mut visited: HashSet<usize> = HashSet::new();
    let mut stack: Vec<ExprId> = vec![root];
    while let Some(id) = stack.pop() {
        if id == target {
            return true;
        }
        if !visited.insert(id.0) {
            continue;
        }
        match arena.get(id) {
            ExprNode::Constant(_)
            | ExprNode::ConstantArray(_, _)
            | ExprNode::Parameter { .. }
            | ExprNode::Variable { .. } => {}
            ExprNode::BinaryOp { left, right, .. } | ExprNode::MatMul { left, right } => {
                stack.push(*left);
                stack.push(*right);
            }
            ExprNode::UnaryOp { operand, .. } | ExprNode::Sum { operand, .. } => {
                stack.push(*operand);
            }
            ExprNode::FunctionCall { args, .. } => {
                stack.extend(args.iter().copied());
            }
            ExprNode::Index { base, .. } => stack.push(*base),
            ExprNode::SumOver { terms } => stack.extend(terms.iter().copied()),
        }
    }
    false
}

/// Build a new arena identical to `arena` except every `Variable`
/// node with `index > drop_block` has its index decremented. Node
/// ids are preserved by emitting nodes in the same order.
fn renumber_variable_indices(arena: &ExprArena, drop_block: usize) -> ExprArena {
    let mut new = ExprArena::with_capacity(arena.len());
    for nid in 0..arena.len() {
        let node = match arena.get(ExprId(nid)) {
            ExprNode::Variable {
                name,
                index,
                size,
                shape,
            } => ExprNode::Variable {
                name: name.clone(),
                index: if *index > drop_block {
                    index - 1
                } else {
                    *index
                },
                size: *size,
                shape: shape.clone(),
            },
            other => other.clone(),
        };
        new.add(node);
    }
    new
}

#[allow(dead_code)]
fn vinfo_dummy() -> VarInfo {
    // Keep the import alive for tests that build small VarInfos.
    VarInfo {
        name: String::new(),
        var_type: VarType::Continuous,
        offset: 0,
        size: 1,
        shape: vec![],
        lb: vec![0.0],
        ub: vec![0.0],
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

    /// Build the equality `c_x · x + c_y · y == rhs` and return its
    /// ConstraintRepr. The arena must already contain the leaves `x`
    /// and `y`.
    fn affine_eq(
        arena: &mut ExprArena,
        cx: f64,
        x: ExprId,
        cy: f64,
        y: ExprId,
        rhs: f64,
    ) -> ConstraintRepr {
        let cx_node = arena.add(ExprNode::Constant(cx));
        let cy_node = arena.add(ExprNode::Constant(cy));
        let cxx = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: cx_node,
            right: x,
        });
        let cyy = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: cy_node,
            right: y,
        });
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: cxx,
            right: cyy,
        });
        ConstraintRepr {
            body,
            sense: ConstraintSense::Eq,
            rhs,
            name: None,
        }
    }

    #[test]
    fn aggregates_unused_variable() {
        // x = 2*y + 1, with x appearing only in this equality.
        // Equation: x - 2*y == 1, written as 1*x + (-2)*y == 1.
        // Objective uses only y. After aggregation x's block is gone.
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let eq = affine_eq(&mut arena, 1.0, x, -2.0, y, 1.0);

        let model = ModelRepr {
            arena,
            objective: y, // x not in objective
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![eq],
            variables: vec![vinfo("x", -10.0, 10.0), vinfo("y", 0.0, 5.0)],
            n_vars: 2,
        };

        let (out, stats) = aggregate_variables(&model);
        assert_eq!(stats.variables_aggregated, 1);
        assert_eq!(stats.equalities_dropped, 1);
        assert_eq!(stats.aggregations.len(), 1);
        let r = &stats.aggregations[0];
        assert_eq!(r.eliminated_block, 0);
        assert_eq!(r.source_block, 1);
        assert!((r.coeff - 2.0).abs() < 1e-12, "coeff was {}", r.coeff);
        assert!((r.offset - 1.0).abs() < 1e-12, "offset was {}", r.offset);

        // Output model has 1 variable (y) and 0 constraints.
        assert_eq!(out.variables.len(), 1);
        assert_eq!(out.constraints.len(), 0);
        assert_eq!(out.n_vars, 1);
        // y is now at block index 0.
        assert_eq!(out.variables[0].name, "y");
    }

    #[test]
    fn skips_when_target_appears_in_other_constraint() {
        // x - 2y = 1 AND x <= 7. x is referenced twice → not eliminable.
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let eq = affine_eq(&mut arena, 1.0, x, -2.0, y, 1.0);
        let extra = ConstraintRepr {
            body: x,
            sense: ConstraintSense::Le,
            rhs: 7.0,
            name: None,
        };
        let model = ModelRepr {
            arena,
            objective: y,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![eq, extra],
            variables: vec![vinfo("x", -10.0, 10.0), vinfo("y", 0.0, 5.0)],
            n_vars: 2,
        };
        let (out, stats) = aggregate_variables(&model);
        // y appears in objective and in eq → not unique-appearance either.
        // Neither side qualifies → no aggregation.
        assert_eq!(stats.variables_aggregated, 0);
        assert_eq!(out.constraints.len(), 2);
        assert_eq!(out.variables.len(), 2);
    }

    #[test]
    fn renumbers_variable_indices_after_drop() {
        // Three variables: x(0), y(1), z(2). Only y appears in the
        // objective and an extra constraint; x is in the equality
        // only; z is referenced in an inequality. After aggregation
        // x is dropped; y's block index becomes 0, z's becomes 1.
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let z = scalar_var(&mut arena, "z", 2);
        let eq = affine_eq(&mut arena, 1.0, x, -1.0, y, 0.0);
        let z_le = ConstraintRepr {
            body: z,
            sense: ConstraintSense::Le,
            rhs: 5.0,
            name: None,
        };
        let model = ModelRepr {
            arena,
            objective: y,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![eq, z_le],
            variables: vec![
                vinfo("x", -10.0, 10.0),
                vinfo("y", 0.0, 5.0),
                vinfo("z", 0.0, 100.0),
            ],
            n_vars: 3,
        };
        let (out, stats) = aggregate_variables(&model);
        assert_eq!(stats.variables_aggregated, 1);
        assert_eq!(out.variables.len(), 2);
        assert_eq!(out.variables[0].name, "y");
        assert_eq!(out.variables[1].name, "z");
        // The remaining z constraint should still mention z (block 1
        // after renumbering, was block 2 before).
        let c = &out.constraints[0];
        if let ExprNode::Variable { index, .. } = out.arena.get(c.body) {
            assert_eq!(*index, 1);
        } else {
            panic!("expected Variable leaf in remaining constraint");
        }
    }

    #[test]
    fn tightens_surviving_var_bounds() {
        // x = 2y + 1, x ∈ [3, 7], y ∈ [-10, 10]. After substitution:
        //   y = (x − 1) / 2 → for x ∈ [3, 7], y ∈ [1, 3].
        // y's new bounds should be [1, 3] (intersection with [-10,10]).
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let eq = affine_eq(&mut arena, 1.0, x, -2.0, y, 1.0);
        let model = ModelRepr {
            arena,
            objective: y,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![eq],
            variables: vec![vinfo("x", 3.0, 7.0), vinfo("y", -10.0, 10.0)],
            n_vars: 2,
        };
        let (out, stats) = aggregate_variables(&model);
        assert_eq!(stats.variables_aggregated, 1);
        assert_eq!(out.variables.len(), 1);
        let yv = &out.variables[0];
        assert!((yv.lb[0] - 1.0).abs() < 1e-9, "y.lb={}", yv.lb[0]);
        assert!((yv.ub[0] - 3.0).abs() < 1e-9, "y.ub={}", yv.ub[0]);
    }

    #[test]
    fn skips_inequality_constraint() {
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let mut le = affine_eq(&mut arena, 1.0, x, -2.0, y, 1.0);
        le.sense = ConstraintSense::Le;
        let model = ModelRepr {
            arena,
            objective: y,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![le],
            variables: vec![vinfo("x", -10.0, 10.0), vinfo("y", 0.0, 5.0)],
            n_vars: 2,
        };
        let (_out, stats) = aggregate_variables(&model);
        assert_eq!(stats.variables_aggregated, 0);
    }

    #[test]
    fn idempotent_on_second_call() {
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let eq = affine_eq(&mut arena, 1.0, x, -2.0, y, 1.0);
        let model = ModelRepr {
            arena,
            objective: y,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![eq],
            variables: vec![vinfo("x", -10.0, 10.0), vinfo("y", 0.0, 5.0)],
            n_vars: 2,
        };
        let (m1, s1) = aggregate_variables(&model);
        assert_eq!(s1.variables_aggregated, 1);
        let (_m2, s2) = aggregate_variables(&m1);
        assert_eq!(s2.variables_aggregated, 0);
    }
}
