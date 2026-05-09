//! Variable elimination via singleton equality detection (M10 of issue #51).
//!
//! ## What this pass does
//!
//! Detects continuous scalar variables that are *uniquely determined*
//! by exactly one equality constraint of the form
//!
//! ```text
//!     coeff · v + const  ==  rhs        (coeff ≠ 0)
//! ```
//!
//! where `v` is the only variable appearing in the constraint body.
//! When such a `v` is found, its declared bounds are pinned to the
//! derived value `(rhs − const) / coeff` and the determining equality
//! is dropped.
//!
//! Distinct from `simplify::simplify`, which strengthens bounds and
//! drops *redundant* constraints but leaves variables in the active
//! set. This pass takes one further step: when a variable is fixed by
//! its determining equation alone, both the variable's degree of
//! freedom and the equation are eliminated from the active model.
//!
//! ## Why the variable may still appear in the objective
//!
//! Once `v.lb == v.ub == value`, every feasible point of the new
//! model has `v = value` exactly, so any subsequent objective or
//! constraint reference to `v` evaluates to the same number it would
//! have under the original model. We do not need to rewrite those
//! references — pinning the bounds suffices. This is the same
//! soundness argument used by LP-presolve free-column elimination.
//!
//! ## Scope (intentional, conservative)
//!
//! - Continuous, scalar (`size == 1`) variables only.
//! - The candidate variable must appear (as an arena leaf) in the
//!   *body* of exactly one constraint, and that constraint must be
//!   an equality.
//! - The body must be polynomial (interpretable by `try_polynomial`)
//!   and reduce to `coeff · v + const` after canonicalisation. Other
//!   monomials referring to *other* variables are not allowed in
//!   the determining equality (that case requires arena-level
//!   substitution, which is a follow-up).
//! - The derived value must lie inside `v`'s current `[lb, ub]`. If
//!   not, the pass abstains and leaves infeasibility detection to
//!   FBBT.
//!
//! Anything outside this scope is left untouched.
//!
//! ## Acceptance criteria from issue #51
//!
//! - **Feasibility/optimality preservation:** post-pass model has the
//!   same feasible set as the original model (verified by sampling).
//! - **No feasible point lost:** `v` was uniquely determined by the
//!   dropped constraint, so pinning its bound to the derived value
//!   does not exclude any feasible solution.
//! - **Idempotent:** a second application makes no further changes
//!   (every fixed variable is already pinned, every singleton
//!   equation already removed).

use std::collections::HashSet;

use super::polynomial::try_polynomial;
use crate::expr::{ConstraintSense, ExprArena, ExprId, ExprNode, ModelRepr, VarType};

/// Per-pass statistics from variable elimination.
#[derive(Debug, Clone, Default)]
pub struct EliminationStats {
    /// Number of variables fixed by elimination (lb := ub := value).
    pub variables_fixed: usize,
    /// Number of equality constraints dropped because they uniquely
    /// determined the eliminated variable.
    pub constraints_removed: usize,
    /// Number of variables that were inspected as candidates.
    pub candidates_examined: usize,
}

/// Run M10 variable elimination on `model`.
///
/// Pure function: input is not modified. The caller decides whether
/// to swap in the result.
pub fn eliminate_variables(model: &ModelRepr) -> (ModelRepr, EliminationStats) {
    let mut out = model.clone();
    let mut stats = EliminationStats::default();

    // Iterate to a fixed point. Each pass may expose new fixings (e.g.
    // when a previous fix made another constraint singleton).
    loop {
        let n_before = stats.variables_fixed;

        // For each candidate variable, find its unique scalar leaf id
        // and run the elimination check. We rebuild the leaf map each
        // outer iteration because constraint indices may have shifted.
        let leaves = scalar_continuous_var_leaves(&out);
        stats.candidates_examined += leaves.len();

        let mut victim_ci: Option<usize> = None;
        let mut victim_vblock: Option<usize> = None;
        let mut victim_value: f64 = 0.0;

        'cands: for (vblock, leaf) in &leaves {
            // Skip variables already pinned (lb == ub).
            let lb = out.variables[*vblock].lb[0];
            let ub = out.variables[*vblock].ub[0];
            if lb == ub {
                continue;
            }

            // Count constraint appearances of `leaf`.
            let mut containing: Vec<usize> = Vec::new();
            for (ci, c) in out.constraints.iter().enumerate() {
                if expr_contains_leaf(&out.arena, c.body, *leaf) {
                    containing.push(ci);
                    if containing.len() > 1 {
                        continue 'cands;
                    }
                }
            }
            if containing.len() != 1 {
                continue;
            }
            let ci = containing[0];
            if out.constraints[ci].sense != ConstraintSense::Eq {
                continue;
            }

            // The body must be polynomial, with a single linear
            // monomial in `leaf` and only constant terms otherwise.
            let body = out.constraints[ci].body;
            let poly = match try_polynomial(&out.arena, body) {
                Some(p) => p,
                None => continue,
            };

            let mut coeff = 0.0;
            let mut leaf_count = 0usize;
            let mut other_var_terms = false;
            for m in &poly.monomials {
                let touches_leaf = m.factors.iter().any(|(fid, _)| *fid == *leaf);
                if touches_leaf {
                    if m.factors.len() == 1 && m.factors[0].1 == 1 {
                        coeff += m.coeff;
                        leaf_count += 1;
                    } else {
                        // Non-linear in v (e.g. v^2 or v*other). Out of scope.
                        other_var_terms = true;
                        break;
                    }
                } else {
                    // Term in other variables — out of scope for v0.
                    other_var_terms = true;
                    break;
                }
            }
            if other_var_terms || leaf_count == 0 || coeff == 0.0 {
                continue;
            }

            let value = (out.constraints[ci].rhs - poly.constant) / coeff;
            if !value.is_finite() {
                continue;
            }
            // Conservative bound check — leave infeasibility for FBBT.
            if value < lb - 1e-12 || value > ub + 1e-12 {
                continue;
            }

            victim_ci = Some(ci);
            victim_vblock = Some(*vblock);
            victim_value = value.clamp(lb, ub);
            break;
        }

        match (victim_ci, victim_vblock) {
            (Some(ci), Some(vb)) => {
                out.variables[vb].lb[0] = victim_value;
                out.variables[vb].ub[0] = victim_value;
                out.constraints.remove(ci);
                stats.variables_fixed += 1;
                stats.constraints_removed += 1;
            }
            _ => break,
        }

        // Defensive guard against accidental non-termination.
        if stats.variables_fixed == n_before {
            break;
        }
    }

    (out, stats)
}

// ─────────────────────────────────────────────────────────────
// Internals
// ─────────────────────────────────────────────────────────────

/// Return `(var_block_index, leaf_expr_id)` for every continuous,
/// scalar variable in the model whose `Variable` node is present in
/// the arena.
fn scalar_continuous_var_leaves(model: &ModelRepr) -> Vec<(usize, ExprId)> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for nid in 0..model.arena.len() {
        if let ExprNode::Variable { index, size, .. } = model.arena.get(ExprId(nid)) {
            if *size != 1 {
                continue;
            }
            if !seen.insert(*index) {
                // Multiple Variable nodes for the same block — skip.
                continue;
            }
            if *index < model.variables.len()
                && model.variables[*index].var_type == VarType::Continuous
                && model.variables[*index].size == 1
            {
                out.push((*index, ExprId(nid)));
            }
        }
    }
    out
}

/// True if `target` appears in the arena DAG rooted at `root`.
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

    #[test]
    fn fixes_singleton_equality() {
        // Model: minimize x ; subject to 2*x == 6 ; x ∈ [0, 10]
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let two = arena.add(ExprNode::Constant(2.0));
        let two_x = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: two,
            right: x,
        });
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: two_x,
                sense: ConstraintSense::Eq,
                rhs: 6.0,
                name: None,
            }],
            variables: vec![vinfo("x", 0.0, 10.0)],
            n_vars: 1,
        };
        let (out, stats) = eliminate_variables(&model);
        assert_eq!(stats.variables_fixed, 1);
        assert_eq!(stats.constraints_removed, 1);
        assert_eq!(out.constraints.len(), 0);
        assert_eq!(out.variables[0].lb[0], 3.0);
        assert_eq!(out.variables[0].ub[0], 3.0);
    }

    #[test]
    fn skips_inequality_constraint() {
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: x,
                sense: ConstraintSense::Le,
                rhs: 5.0,
                name: None,
            }],
            variables: vec![vinfo("x", 0.0, 10.0)],
            n_vars: 1,
        };
        let (out, stats) = eliminate_variables(&model);
        assert_eq!(stats.variables_fixed, 0);
        assert_eq!(out.constraints.len(), 1);
    }

    #[test]
    fn skips_when_var_appears_in_two_constraints() {
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![
                ConstraintRepr {
                    body: x,
                    sense: ConstraintSense::Eq,
                    rhs: 3.0,
                    name: None,
                },
                ConstraintRepr {
                    body: x,
                    sense: ConstraintSense::Le,
                    rhs: 5.0,
                    name: None,
                },
            ],
            variables: vec![vinfo("x", 0.0, 10.0)],
            n_vars: 1,
        };
        let (out, stats) = eliminate_variables(&model);
        assert_eq!(stats.variables_fixed, 0);
        assert_eq!(out.constraints.len(), 2);
    }

    #[test]
    fn skips_when_other_var_in_equality() {
        // 2*x + y == 5, both vars present — out of scope for v0.
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let two = arena.add(ExprNode::Constant(2.0));
        let two_x = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: two,
            right: x,
        });
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: two_x,
            right: y,
        });
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Eq,
                rhs: 5.0,
                name: None,
            }],
            variables: vec![vinfo("x", 0.0, 10.0), vinfo("y", 0.0, 10.0)],
            n_vars: 2,
        };
        let (out, stats) = eliminate_variables(&model);
        assert_eq!(stats.variables_fixed, 0);
        assert_eq!(out.constraints.len(), 1);
    }

    #[test]
    fn skips_when_value_outside_bounds() {
        // 2*x == 6 but x ∈ [0, 2]. Derived value 3 outside bounds:
        // leave for FBBT to flag.
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let two = arena.add(ExprNode::Constant(2.0));
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: two,
            right: x,
        });
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Eq,
                rhs: 6.0,
                name: None,
            }],
            variables: vec![vinfo("x", 0.0, 2.0)],
            n_vars: 1,
        };
        let (out, stats) = eliminate_variables(&model);
        assert_eq!(stats.variables_fixed, 0);
        assert_eq!(out.constraints.len(), 1);
    }

    #[test]
    fn idempotent_on_second_pass() {
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let two = arena.add(ExprNode::Constant(2.0));
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: two,
            right: x,
        });
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Eq,
                rhs: 6.0,
                name: None,
            }],
            variables: vec![vinfo("x", 0.0, 10.0)],
            n_vars: 1,
        };
        let (m1, s1) = eliminate_variables(&model);
        assert_eq!(s1.variables_fixed, 1);
        let (m2, s2) = eliminate_variables(&m1);
        assert_eq!(s2.variables_fixed, 0);
        assert_eq!(m2.constraints.len(), m1.constraints.len());
        assert_eq!(m2.variables[0].lb[0], m1.variables[0].lb[0]);
    }

    #[test]
    fn cascades_two_eliminations() {
        // Two independent singleton equalities.
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![
                ConstraintRepr {
                    body: x,
                    sense: ConstraintSense::Eq,
                    rhs: 1.5,
                    name: None,
                },
                ConstraintRepr {
                    body: y,
                    sense: ConstraintSense::Eq,
                    rhs: 2.5,
                    name: None,
                },
            ],
            variables: vec![vinfo("x", 0.0, 10.0), vinfo("y", 0.0, 10.0)],
            n_vars: 2,
        };
        let (out, stats) = eliminate_variables(&model);
        assert_eq!(stats.variables_fixed, 2);
        assert_eq!(stats.constraints_removed, 2);
        assert_eq!(out.constraints.len(), 0);
        assert_eq!(out.variables[0].lb[0], 1.5);
        assert_eq!(out.variables[1].ub[0], 2.5);
    }
}
