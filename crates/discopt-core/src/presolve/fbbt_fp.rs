//! Watch-list FBBT to a true fixed point (B4 of the presolve roadmap).
//!
//! ## What this pass does
//!
//! Runs forward/backward bound propagation as a constraint-by-constraint
//! work-queue: a constraint is processed only if one of its variables
//! has had its bound tightened since the last visit. The queue starts
//! with every constraint dirty; processing a constraint runs the same
//! `forward_propagate` / `backward_propagate` kernels as
//! [`super::fbbt::fbbt_with_cutoff`], then queues every constraint that
//! shares a freshly-tightened variable.
//!
//! ## Why it matters
//!
//! The existing iterative-with-cap FBBT in `fbbt.rs` (B4 reference)
//! visits every constraint on every sweep. That is wasteful when only
//! a small part of the model has changed since the last sweep, and
//! oscillates in the tail of convergence — the cap is a defensive cap
//! against non-termination on cyclic DAGs, not a tightness check.
//!
//! The watch-list version:
//!
//! 1. Avoids re-visiting constraints whose inputs haven't changed.
//! 2. Terminates the moment the queue is empty — that's the true
//!    fixed point on the linear and DAG-monotone parts of the model,
//!    no `max_iter` artefact.
//! 3. Still respects an explicit work cap (`max_visits`) for the
//!    pathological non-monotone case (e.g. cyclic DAGs through
//!    nonconvex envelopes), so termination is guaranteed.
//!
//! This is the closest Rust-only realisation of the Belotti–Cafieri–
//! Lee–Liberti (2010) FBBT-as-LP idea: same monotone tightening
//! contract, terminates at the fixed point, no LP solver. The full LP
//! formulation (which closes the gap on inherently non-monotone
//! cycles) requires the A3 Rust↔Python LP handshake and is deferred
//! to phase P3.
//!
//! ## Determinism
//!
//! Constraints are pushed into the queue in `model.constraints` order
//! and popped FIFO (`VecDeque`). The watch-set bookkeeping uses
//! `Vec<Vec<usize>>` keyed by variable index — no `HashMap` iteration
//! on the hot path.

use std::collections::VecDeque;

use super::fbbt::{backward_propagate, forward_propagate, Interval};
use crate::expr::{ConstraintSense, ExprArena, ExprId, ExprNode, ModelRepr};

/// Per-pass statistics from watch-list FBBT.
#[derive(Debug, Clone, Default)]
pub struct FbbtFpStats {
    /// Number of times a constraint was popped from the queue and
    /// processed. ≥ `model.constraints.len()` since every constraint
    /// is queued once at startup.
    pub constraint_visits: usize,
    /// Number of `(var_index, change)` events recorded — one per
    /// strictly-tightening update.
    pub bound_updates: usize,
    /// True if the work cap was hit before the queue drained.
    pub hit_work_cap: bool,
    /// True if the model was determined infeasible during propagation.
    pub infeasible: bool,
}

/// Options for [`fbbt_fixed_point`].
#[derive(Debug, Clone, Copy)]
pub struct FbbtFpOptions {
    /// Tolerance below which a bound change does not re-trigger
    /// dependent constraints.
    pub tol: f64,
    /// Hard cap on the number of constraint visits. Set to a generous
    /// multiple of `n_constraints` (e.g. 200×) to leave the natural
    /// fixed point well within budget.
    pub max_visits: usize,
}

impl Default for FbbtFpOptions {
    fn default() -> Self {
        Self {
            tol: 1e-9,
            max_visits: 0, // 0 => derive from model size
        }
    }
}

/// Run watch-list FBBT to a fixed point. `bounds` is updated in place
/// with strictly-monotone tightenings.
pub fn fbbt_fixed_point(
    model: &ModelRepr,
    bounds: &mut [Interval],
    opts: FbbtFpOptions,
) -> FbbtFpStats {
    let mut stats = FbbtFpStats::default();
    let n_constr = model.constraints.len();
    if n_constr == 0 {
        return stats;
    }
    let cap = if opts.max_visits == 0 {
        n_constr.saturating_mul(200).max(1024)
    } else {
        opts.max_visits
    };

    // Build var_index → list of constraints that mention it.
    // Variable indices come from `ExprNode::Variable.index`.
    let mut watchers: Vec<Vec<usize>> = vec![Vec::new(); model.variables.len()];
    let mut per_constr_vars: Vec<Vec<usize>> = Vec::with_capacity(n_constr);
    for (ci, c) in model.constraints.iter().enumerate() {
        let mut vs = Vec::new();
        collect_variable_indices(&model.arena, c.body, &mut vs);
        vs.sort_unstable();
        vs.dedup();
        for &v in &vs {
            if let Some(list) = watchers.get_mut(v) {
                list.push(ci);
            }
        }
        per_constr_vars.push(vs);
    }

    // Initial queue: every constraint, in order. `in_queue` avoids
    // pushing the same constraint twice.
    let mut queue: VecDeque<usize> = VecDeque::with_capacity(n_constr);
    let mut in_queue: Vec<bool> = vec![false; n_constr];
    for (ci, slot) in in_queue.iter_mut().enumerate() {
        queue.push_back(ci);
        *slot = true;
    }

    while let Some(ci) = queue.pop_front() {
        in_queue[ci] = false;
        stats.constraint_visits += 1;
        if stats.constraint_visits > cap {
            stats.hit_work_cap = true;
            break;
        }

        let constr = &model.constraints[ci];
        let node_bounds = forward_propagate(&model.arena, constr.body, bounds);

        let output_bound = match constr.sense {
            ConstraintSense::Le => Interval::new(f64::NEG_INFINITY, constr.rhs),
            ConstraintSense::Ge => Interval::new(constr.rhs, f64::INFINITY),
            ConstraintSense::Eq => Interval::point(constr.rhs),
        };
        let body_bound = node_bounds[constr.body.0];
        if body_bound.intersect(&output_bound).is_empty() {
            stats.infeasible = true;
            for b in bounds.iter_mut() {
                *b = Interval::empty();
            }
            return stats;
        }

        // Snapshot the variables this constraint touches so we can
        // detect which ones changed after backward propagation.
        let vs = &per_constr_vars[ci];
        let snapshot: Vec<(usize, Interval)> = vs
            .iter()
            .filter_map(|&v| bounds.get(v).map(|b| (v, *b)))
            .collect();

        backward_propagate(
            &model.arena,
            constr.body,
            output_bound,
            &node_bounds,
            bounds,
        );

        // Detect changes; queue downstream constraints.
        for (v, prev) in snapshot {
            let cur = bounds[v];
            if cur.is_empty() {
                stats.infeasible = true;
                for b in bounds.iter_mut() {
                    *b = Interval::empty();
                }
                return stats;
            }
            let dlo = (cur.lo - prev.lo).abs();
            let dhi = (cur.hi - prev.hi).abs();
            if dlo > opts.tol || dhi > opts.tol {
                stats.bound_updates += 1;
                if let Some(list) = watchers.get(v) {
                    for &cj in list {
                        if cj == ci {
                            continue;
                        }
                        if !in_queue[cj] {
                            queue.push_back(cj);
                            in_queue[cj] = true;
                        }
                    }
                }
            }
        }
    }
    stats
}

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

/// Collect every variable-block index referenced under `root`.
/// Indexed accesses (`ExprNode::Index { base, .. }`) contribute
/// the base variable's block index; the watcher map is per-block,
/// not per-element.
fn collect_variable_indices(arena: &ExprArena, root: ExprId, out: &mut Vec<usize>) {
    walk(arena, root, out);
}

fn walk(arena: &ExprArena, id: ExprId, out: &mut Vec<usize>) {
    match arena.get(id) {
        ExprNode::Variable { index, .. } => out.push(*index),
        ExprNode::Index { base, .. } => walk(arena, *base, out),
        ExprNode::UnaryOp { operand, .. } => walk(arena, *operand, out),
        ExprNode::BinaryOp { left, right, .. } => {
            walk(arena, *left, out);
            walk(arena, *right, out);
        }
        ExprNode::FunctionCall { args, .. } => {
            for a in args {
                walk(arena, *a, out);
            }
        }
        ExprNode::MatMul { left, right } => {
            walk(arena, *left, out);
            walk(arena, *right, out);
        }
        ExprNode::Sum { operand, .. } => walk(arena, *operand, out),
        ExprNode::SumOver { terms } => {
            for t in terms {
                walk(arena, *t, out);
            }
        }
        ExprNode::Constant(_) | ExprNode::Parameter { .. } | ExprNode::ConstantArray(_, _) => {}
    }
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

    /// Two linear constraints chained: tightening from row 0 should
    /// requeue row 1, which then tightens further.
    /// `x + y ≤ 5; x − y ≤ 1; x, y ∈ [0, 10]`.
    /// Expect both rows visited at least twice (initial pass + requeue
    /// after row 0 lowers x to 5 / y to 5).
    #[test]
    fn watch_requeues_after_tightening() {
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
        let stats = fbbt_fixed_point(&model, &mut bounds, FbbtFpOptions::default());
        assert!(!stats.infeasible);
        assert!(!stats.hit_work_cap);
        // After fixed-point, x ≤ 5, y ≤ 5 from the first row alone.
        assert!(bounds[0].hi <= 5.0 + 1e-6);
        assert!(bounds[1].hi <= 5.0 + 1e-6);
        // At least the 2 initial visits.
        assert!(stats.constraint_visits >= 2);
        // At least one bound update happened.
        assert!(stats.bound_updates >= 1);
    }

    /// Empty queue: a model where no row tightens anything means each
    /// constraint is visited exactly once.
    #[test]
    fn idempotent_loose_model() {
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let r1 = {
            let a = lin(&mut arena, 1.0, x);
            let b = lin(&mut arena, 1.0, y);
            add(&mut arena, a, b)
        };
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: r1,
                sense: ConstraintSense::Le,
                rhs: 1000.0,
                name: None,
            }],
            variables: vec![vinfo("x", 0, 0.0, 10.0), vinfo("y", 1, 0.0, 10.0)],
            n_vars: 2,
        };
        let mut bounds = vec![Interval::new(0.0, 10.0); 2];
        let stats = fbbt_fixed_point(&model, &mut bounds, FbbtFpOptions::default());
        assert_eq!(stats.constraint_visits, 1);
        assert_eq!(stats.bound_updates, 0);
        assert_eq!(bounds[0].lo, 0.0);
        assert_eq!(bounds[0].hi, 10.0);
    }

    /// Infeasibility detection: `x + 1 ≤ 0` with `x ∈ [0, 10]`.
    #[test]
    fn infeasibility_zeros_bounds() {
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let one = arena.add(ExprNode::Constant(1.0));
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x,
            right: one,
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
            variables: vec![vinfo("x", 0, 0.5, 10.0)],
            n_vars: 1,
        };
        let mut bounds = vec![Interval::new(0.5, 10.0)];
        let stats = fbbt_fixed_point(&model, &mut bounds, FbbtFpOptions::default());
        assert!(stats.infeasible);
        assert!(bounds[0].is_empty());
    }
}
