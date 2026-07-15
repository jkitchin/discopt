//! Persistent in-tree bound tightening (B3 of issue #51).
//!
//! Runs lightweight FBBT on a B&B node's local bounds, returning the
//! tightened intervals. Tightenings persist by virtue of the B&B
//! contract: any child node inherits its parent's bounds, so a
//! tightening applied at a node automatically propagates to its
//! subtree.
//!
//! ## Why this is cheap
//!
//! The expression DAG and constraint structure are identical at every
//! node — only the variable bounds change. So FBBT at a child node
//! re-uses all of the topology and only re-evaluates intervals on
//! shifted leaves. The marginal work per node is proportional to the
//! number of variables that *changed* relative to the parent (in
//! principle); this kernel runs the full pass for now and leaves the
//! incremental optimisation to a follow-up.
//!
//! ## Scheduling
//!
//! In-tree FBBT is gated by [`InTreePresolveOptions::depth_stride`] —
//! the pass runs only when `node_depth % depth_stride == 0`, so the
//! caller can amortise the cost over the tree without paying it at
//! every node. `depth_stride = 1` runs at every node;
//! `depth_stride = 0` disables the pass.

use crate::expr::ModelRepr;
use crate::presolve::fbbt::{fbbt_with_cutoff, Interval};
use crate::presolve::probing::probe_node_bounds;

/// Options controlling persistent in-tree bound tightening.
#[derive(Debug, Clone)]
pub struct InTreePresolveOptions {
    /// Run the pass at every `depth_stride`-th tree depth. `0` disables
    /// the pass entirely; `1` runs at every node.
    pub depth_stride: u32,
    /// FBBT inner-loop iteration cap.
    pub max_iter: usize,
    /// FBBT inner-loop convergence tolerance.
    pub tol: f64,
    /// Run per-node probing (P3 branch-and-reduce) after FBBT. Probing
    /// tentatively fixes each discrete variable at a bound and re-runs FBBT,
    /// contracting the domain on any proven-infeasible fixing. Off by default
    /// (it costs O(discrete) extra FBBT solves per node); sound when on.
    pub probing: bool,
    /// Cap on the number of discrete variables probed per node (budget).
    pub probe_max_vars: usize,
}

impl Default for InTreePresolveOptions {
    fn default() -> Self {
        Self {
            depth_stride: 4,
            max_iter: 8,
            tol: 1e-6,
            probing: false,
            probe_max_vars: 32,
        }
    }
}

/// Per-node tightening result.
#[derive(Debug, Clone, Default)]
pub struct InTreeDelta {
    /// Tightened lower bounds (one per variable).
    pub lb: Vec<f64>,
    /// Tightened upper bounds (one per variable).
    pub ub: Vec<f64>,
    /// Number of variables whose bounds tightened (either side).
    pub bounds_tightened: u32,
    /// True if the kernel detected infeasibility (empty interval).
    pub infeasible: bool,
    /// True iff the schedule actually ran the pass at this node.
    pub ran: bool,
}

/// Run in-tree FBBT at a node with the given local bounds.
///
/// `model` is the **root** model (variable bounds inside it are
/// ignored — `node_lb`/`node_ub` override them). Returns an
/// [`InTreeDelta`] containing the post-tightening bounds.
///
/// The pass is a no-op (returns `ran = false`, copies `node_lb` /
/// `node_ub` unchanged) when the schedule says to skip this depth.
pub fn run_in_tree_presolve(
    model: &ModelRepr,
    node_lb: &[f64],
    node_ub: &[f64],
    node_depth: usize,
    incumbent: Option<f64>,
    opts: &InTreePresolveOptions,
) -> InTreeDelta {
    assert_eq!(node_lb.len(), model.variables.len());
    assert_eq!(node_ub.len(), model.variables.len());

    if opts.depth_stride == 0 || (node_depth as u32) % opts.depth_stride != 0 {
        return InTreeDelta {
            lb: node_lb.to_vec(),
            ub: node_ub.to_vec(),
            bounds_tightened: 0,
            infeasible: false,
            ran: false,
        };
    }

    // Patch the model's variable bounds with the node-local bounds.
    // We clone only the lightweight `variables` Vec, not the arena.
    let mut patched = model.clone();
    for (i, vinfo) in patched.variables.iter_mut().enumerate() {
        if !vinfo.lb.is_empty() {
            vinfo.lb[0] = node_lb[i];
        }
        if !vinfo.ub.is_empty() {
            vinfo.ub[0] = node_ub[i];
        }
    }

    let bounds: Vec<Interval> = fbbt_with_cutoff(&patched, opts.max_iter, opts.tol, incumbent);
    let mut infeasible = bounds.iter().any(|iv| iv.is_empty());

    let mut new_lb = node_lb.to_vec();
    let mut new_ub = node_ub.to_vec();
    let mut tightened = 0u32;
    if !infeasible {
        for i in 0..bounds.len() {
            let iv = bounds[i];
            // Floor with the node's bounds — never relax.
            if iv.lo > new_lb[i] + opts.tol {
                new_lb[i] = iv.lo;
                tightened += 1;
            }
            if iv.hi < new_ub[i] - opts.tol {
                new_ub[i] = iv.hi;
                tightened += 1;
            }
        }
    }

    // P3 probing pass: contract discrete-variable domains by tentatively fixing
    // each at a bound and re-running FBBT (proven-infeasible fixings only).
    // Runs on the FBBT-tightened box; folds its (subset) result back, never
    // loosening. `patched` carries the node bounds; probing re-seeds fully from
    // the explicit interval box, so the two boxes agree.
    if opts.probing && !infeasible {
        let node_box: Vec<Interval> = (0..new_lb.len())
            .map(|i| Interval::new(new_lb[i], new_ub[i]))
            .collect();
        let pr = probe_node_bounds(
            &patched,
            &node_box,
            opts.probe_max_vars,
            opts.max_iter,
            opts.tol,
            incumbent,
            None,
        );
        if pr.infeasible {
            infeasible = true;
        } else {
            for i in 0..pr.tightened_bounds.len().min(new_lb.len()) {
                let iv = pr.tightened_bounds[i];
                if iv.lo > new_lb[i] + opts.tol {
                    new_lb[i] = iv.lo;
                    tightened += 1;
                }
                if iv.hi < new_ub[i] - opts.tol {
                    new_ub[i] = iv.hi;
                    tightened += 1;
                }
                if new_lb[i] > new_ub[i] + opts.tol {
                    infeasible = true;
                }
            }
        }
    }

    InTreeDelta {
        lb: new_lb,
        ub: new_ub,
        bounds_tightened: tightened,
        infeasible,
        ran: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{
        BinOp, ConstraintRepr, ConstraintSense, ExprArena, ExprId, ExprNode, ModelRepr,
        ObjectiveSense, VarInfo, VarType,
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

    fn x_plus_y_le_5() -> ModelRepr {
        // x + y <= 5, x ∈ [0, 10], y ∈ [0, 10], min x+y
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x,
            right: y,
        });
        ModelRepr {
            arena,
            objective: body,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 5.0,
                name: None,
            }],
            variables: vec![vinfo("x", 0.0, 10.0), vinfo("y", 0.0, 10.0)],
            n_vars: 2,
        }
    }

    #[test]
    fn tightens_at_node_with_branching_bound() {
        // Branch: x ∈ [3, 10] in the node. The constraint x+y≤5 then
        // forces y ≤ 2.
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            max_iter: 16,
            tol: 1e-9,
            ..Default::default()
        };
        let delta = run_in_tree_presolve(&model, &[3.0, 0.0], &[10.0, 10.0], 1, None, &opts);
        assert!(delta.ran);
        assert!(!delta.infeasible);
        assert!(delta.bounds_tightened >= 1);
        assert!((delta.ub[1] - 2.0).abs() <= 1e-6);
        // Lower bounds are not relaxed.
        assert_eq!(delta.lb[0], 3.0);
    }

    #[test]
    fn infers_indicator_binary_at_node() {
        // Guard x ≤ 10·b, x ∈ [0, 10], b binary. At a node where branching has
        // tightened x to [3, 10], FBBT infers b ≥ 0.3 and snaps it to b = 1 —
        // per-node indicator propagation (issue #230). This is the integration
        // the root-only probing pass cannot deliver inside the tree.
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let b = scalar_var(&mut arena, "b", 1);
        let m = arena.add(ExprNode::Constant(10.0));
        let mb = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: m,
            right: b,
        });
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Sub,
            left: x,
            right: mb,
        });
        let mut bvar = vinfo("b", 0.0, 1.0);
        bvar.var_type = VarType::Binary;
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
            variables: vec![vinfo("x", 0.0, 10.0), bvar],
            n_vars: 2,
        };
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            max_iter: 16,
            tol: 1e-9,
            ..Default::default()
        };
        let delta = run_in_tree_presolve(&model, &[3.0, 0.0], &[10.0, 1.0], 1, None, &opts);
        assert!(delta.ran);
        assert!(!delta.infeasible);
        assert!(
            (delta.lb[1] - 1.0).abs() <= 1e-6,
            "binary should be fixed to 1 at the node, got [{}, {}]",
            delta.lb[1],
            delta.ub[1]
        );
    }

    #[test]
    fn skips_when_depth_stride_zero() {
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 0,
            ..Default::default()
        };
        let delta = run_in_tree_presolve(&model, &[3.0, 0.0], &[10.0, 10.0], 1, None, &opts);
        assert!(!delta.ran);
        assert_eq!(delta.bounds_tightened, 0);
        assert_eq!(delta.lb, vec![3.0, 0.0]);
        assert_eq!(delta.ub, vec![10.0, 10.0]);
    }

    #[test]
    fn skips_off_schedule_depths() {
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 4,
            ..Default::default()
        };
        // depth=1 is not a multiple of 4 ⇒ skipped.
        let d = run_in_tree_presolve(&model, &[3.0, 0.0], &[10.0, 10.0], 1, None, &opts);
        assert!(!d.ran);
        // depth=4 ⇒ runs.
        let d4 = run_in_tree_presolve(&model, &[3.0, 0.0], &[10.0, 10.0], 4, None, &opts);
        assert!(d4.ran);
        assert!(d4.bounds_tightened >= 1);
    }

    #[test]
    fn detects_infeasibility() {
        // Branch: x ∈ [10, 10] AND y ∈ [10, 10]. x+y=20 > 5 — infeasible.
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            ..Default::default()
        };
        let delta = run_in_tree_presolve(&model, &[10.0, 10.0], &[10.0, 10.0], 1, None, &opts);
        assert!(delta.ran);
        assert!(delta.infeasible);
    }

    #[test]
    fn never_relaxes_input_bounds() {
        // Bounds tighter than what FBBT alone would derive must be kept.
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            ..Default::default()
        };
        // Caller-supplied tighter ub on x.
        let delta = run_in_tree_presolve(&model, &[0.0, 0.0], &[1.0, 10.0], 0, None, &opts);
        assert!(delta.ran);
        // The ub on x must remain at 1.0 (or tighter), never relax to 5.
        assert!(delta.ub[0] <= 1.0 + 1e-9);
    }

    #[test]
    fn probing_fixes_binary_at_node() {
        // x ≤ 10·b, x ∈ [0,10], b binary; node branch x ∈ [3,10] ⇒ b = 1.
        // The probing pass (opts.probing = true) must fix b to 1 at the node.
        let model = {
            let mut arena = ExprArena::new();
            let x = scalar_var(&mut arena, "x", 0);
            let b = scalar_var(&mut arena, "b", 1);
            let m = arena.add(ExprNode::Constant(10.0));
            let mb = arena.add(ExprNode::BinaryOp {
                op: BinOp::Mul,
                left: m,
                right: b,
            });
            let body = arena.add(ExprNode::BinaryOp {
                op: BinOp::Sub,
                left: x,
                right: mb,
            });
            let mut bvar = vinfo("b", 0.0, 1.0);
            bvar.var_type = VarType::Binary;
            ModelRepr {
                arena,
                objective: x,
                objective_sense: ObjectiveSense::Minimize,
                constraints: vec![ConstraintRepr {
                    body,
                    sense: ConstraintSense::Le,
                    rhs: 0.0,
                    name: None,
                }],
                variables: vec![vinfo("x", 0.0, 10.0), bvar],
                n_vars: 2,
            }
        };
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            max_iter: 16,
            tol: 1e-9,
            probing: true,
            probe_max_vars: 32,
        };
        let delta = run_in_tree_presolve(&model, &[3.0, 0.0], &[10.0, 1.0], 1, None, &opts);
        assert!(delta.ran);
        assert!(!delta.infeasible);
        assert!(
            (delta.lb[1] - 1.0).abs() <= 1e-6,
            "b should be fixed to 1 at the node, got [{}, {}]",
            delta.lb[1],
            delta.ub[1]
        );
    }

    #[test]
    fn probing_off_by_default_is_byte_neutral() {
        // With probing disabled (default), the delta matches the FBBT-only path.
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            max_iter: 16,
            tol: 1e-9,
            ..Default::default()
        };
        assert!(!opts.probing);
        let delta = run_in_tree_presolve(&model, &[3.0, 0.0], &[10.0, 10.0], 1, None, &opts);
        assert!(delta.ran);
        assert!(!delta.infeasible);
        assert!((delta.ub[1] - 2.0).abs() <= 1e-6);
    }
}
