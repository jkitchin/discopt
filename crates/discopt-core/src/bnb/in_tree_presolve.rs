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
use crate::presolve::fbbt::{
    any_empty_beyond, fbbt_with_cutoff, repair_subtol_crossings, Interval, FEAS_TOL,
};
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
    /// How many sub-`FEAS_TOL` bound crossings were repaired (#907).
    ///
    /// Surfaced rather than absorbed: a rising count is a numerical smell, and
    /// before #907 each of these events could have fathomed a live node.
    pub subtol_repaired: usize,
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
            subtol_repaired: 0,
            ran: false,
        };
    }

    // #907. Sanitize the INCOMING node box before anything reads it. A caller
    // upstream (or an earlier `in_tree_presolve` on a parent node) may hand us a
    // box already inverted by rounding noise; patching that straight onto
    // `VarInfo` would seed FBBT from an inverted domain and manufacture the very
    // emptiness we are trying not to over-read.
    let mut node_box: Vec<Interval> = (0..node_lb.len())
        .map(|i| Interval::new(node_lb[i], node_ub[i]))
        .collect();
    let mut subtol_repaired = repair_subtol_crossings(&mut node_box, FEAS_TOL);

    // Patch the model's variable bounds with the node-local bounds.
    // We clone only the lightweight `variables` Vec, not the arena.
    let mut patched = model.clone();
    for (i, vinfo) in patched.variables.iter_mut().enumerate() {
        if !vinfo.lb.is_empty() {
            vinfo.lb[0] = node_box[i].lo;
        }
        if !vinfo.ub.is_empty() {
            vinfo.ub[0] = node_box[i].hi;
        }
    }

    // #907. `infeasible` is consumed by the B&B loop as a RIGOROUS FATHOM — the
    // subtree is pruned outright — so it must never be set by floating-point
    // noise. Repair sub-`FEAS_TOL` crossings, then conclude infeasibility only
    // beyond the tolerance, exactly as `fbbt`, `fbbt_fp` and `probing` do.
    //
    // Loosening a fathom is the SOUND direction: the node is explored rather
    // than discarded. Genuine detections are unaffected — corpus instrumentation
    // found every real fathom carried either the `[inf, -inf]` empty sentinel or
    // a crossing of exactly 1.0 (binary domain wipeout), 6+ orders above
    // `FEAS_TOL`.
    let mut bounds: Vec<Interval> = fbbt_with_cutoff(&patched, opts.max_iter, opts.tol, incumbent);
    subtol_repaired += repair_subtol_crossings(&mut bounds, FEAS_TOL);
    let mut infeasible = any_empty_beyond(&bounds, FEAS_TOL);

    let mut new_lb: Vec<f64> = node_box.iter().map(|b| b.lo).collect();
    let mut new_ub: Vec<f64> = node_box.iter().map(|b| b.hi).collect();
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
                // #907. NO infeasibility verdict here. This loop used to test
                // `new_lb[i] > new_ub[i] + opts.tol`, but `opts.tol` is the FBBT
                // *convergence* tolerance — independently settable and smaller
                // than `FEAS_TOL` in practice — so a crossing in
                // `(opts.tol, FEAS_TOL]` set the rigorous-fathom flag on exactly
                // the noise this fix exists to tolerate. The single exit below
                // repairs and then decides, at `FEAS_TOL`, for every path.
            }
        }
    }

    // #907. Final sanitation at the single exit. The probing branch above gates
    // its own emptiness test on `opts.tol` (a different, smaller tolerance), so it
    // can fold back a box that is inverted by a sub-`FEAS_TOL` amount WITHOUT
    // setting `infeasible`. Returning that inverted box would push `lo > hi` onto
    // an LP column bound downstream, reproducing the false infeasibility one layer
    // down — declining to *declare* emptiness is not enough on its own.
    let mut out: Vec<Interval> = (0..new_lb.len())
        .map(|i| Interval::new(new_lb[i], new_ub[i]))
        .collect();
    subtol_repaired += repair_subtol_crossings(&mut out, FEAS_TOL);
    if !infeasible && any_empty_beyond(&out, FEAS_TOL) {
        infeasible = true;
    }
    for (i, b) in out.iter().enumerate() {
        new_lb[i] = b.lo;
        new_ub[i] = b.hi;
    }
    debug_assert!(
        infeasible || new_lb.iter().zip(&new_ub).all(|(l, u)| l <= u),
        "#907: in_tree_presolve returned an inverted box without declaring infeasible"
    );

    InTreeDelta {
        lb: new_lb,
        ub: new_ub,
        bounds_tightened: tightened,
        infeasible,
        subtol_repaired,
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

    // ── #907: a sub-FEAS_TOL crossing must not FATHOM a node ─────────────
    //
    // `InTreeDelta::infeasible` is consumed by the B&B loop as a *rigorous
    // fathom*: the subtree is pruned outright. Setting it from a crossing of
    // 8.5e-14 discards a region that may contain feasible points, violating the
    // zero-slack `incorrect_count <= 0` gate with no flag set.

    /// A node box inverted by rounding noise must be explored, not fathomed,
    /// and must not be returned inverted.
    #[test]
    fn subtol_inverted_node_box_is_not_fathomed() {
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            max_iter: 16,
            tol: 1e-9,
            probing: false,
            probe_max_vars: 0,
        };
        // x fixed at 2.5 by two derivations disagreeing in the last ulps.
        let lo = [2.5, 0.0];
        let hi = [2.5 - 1e-14, 10.0];
        let d = run_in_tree_presolve(&model, &lo, &hi, 0, None, &opts);

        assert!(d.ran);
        assert!(
            !d.infeasible,
            "a 1e-14 crossing fathomed a live node — #907 regressed"
        );
        assert_eq!(d.subtol_repaired, 1);
        // The returned box must be well-formed: an inverted interval reaching an
        // LP column bound reproduces the false infeasibility one layer down.
        for i in 0..d.lb.len() {
            assert!(
                d.lb[i] <= d.ub[i],
                "returned an inverted box at var{i}: [{}, {}]",
                d.lb[i],
                d.ub[i]
            );
        }
        // Repair widens to contain both endpoints, so the feasible point x=2.5
        // survives.
        assert!(d.lb[0] <= 2.5 && 2.5 <= d.ub[0]);
    }

    /// ANTI-PERMISSIVENESS CONTROL: a genuinely empty node box must STILL
    /// fathom. Without this the change is a tolerance-tweak, not a fix.
    #[test]
    fn genuine_empty_node_box_still_fathoms() {
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            max_iter: 16,
            tol: 1e-9,
            probing: false,
            probe_max_vars: 0,
        };
        // x >= 10 AND y >= 10 with x + y <= 5 — infeasible by 15, not by noise.
        let d = run_in_tree_presolve(&model, &[10.0, 10.0], &[10.0, 10.0], 0, None, &opts);
        assert!(d.ran);
        assert!(d.infeasible, "a genuine infeasibility stopped fathoming");
        assert_eq!(d.subtol_repaired, 0);
    }

    /// The repair must never cut a point the caller's box contained: sweep a
    /// feasible point through many noise-inverted boxes and assert containment
    /// survives. Prints nothing, but the assertion count is the point (§6).
    #[test]
    fn repair_never_cuts_a_contained_feasible_point() {
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            max_iter: 16,
            tol: 1e-9,
            probing: false,
            probe_max_vars: 0,
        };
        let mut checked = 0usize;
        for k in 0..40 {
            let v = 0.1 * k as f64; // feasible x value in [0, 3.9]
            for eps in [1e-16, 1e-14, 1e-12, 1e-9, 1e-7] {
                let d = run_in_tree_presolve(&model, &[v, 0.0], &[v - eps, 10.0], 0, None, &opts);
                assert!(!d.infeasible, "fathomed a live node at eps={eps}");
                assert!(
                    d.lb[0] <= v && v <= d.ub[0],
                    "repair cut x={v} at eps={eps}: [{}, {}]",
                    d.lb[0],
                    d.ub[0]
                );
                checked += 1;
            }
        }
        assert_eq!(
            checked, 200,
            "probe did not execute the comparisons it claims"
        );
    }

    /// #907, probing path enabled: a node box inverted by sub-`FEAS_TOL` noise
    /// must not be fathomed, and must not come back inverted or with the point
    /// cut. Runs with `opts.tol` two orders BELOW `FEAS_TOL` so the two
    /// tolerances are distinguishable.
    ///
    /// SCOPE, stated honestly: this covers the incoming-box sanitation on the
    /// probing path (it fails on pre-#907 `main`). It does NOT isolate the
    /// fold-back verdict that used to read `new_lb[i] > new_ub[i] + opts.tol` —
    /// reverting that one line alone leaves this test green, because reaching it
    /// requires `probe_node_bounds` to itself return an interval inverted by
    /// `(opts.tol, FEAS_TOL]`, which no toy model here produces. That line is
    /// changed on consistency grounds — `opts.tol` is a convergence tolerance and
    /// must not gate an infeasibility verdict — and is UNCOVERED by a
    /// fail-before test.
    #[test]
    fn probing_path_does_not_fathom_subtol_inverted_box() {
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            max_iter: 16,
            tol: 1e-8, // << FEAS_TOL (1e-6): the window the old test fathomed in
            probing: true,
            probe_max_vars: 8,
        };
        let mut checked = 0usize;
        for eps in [1e-7, 5e-7, 9e-7] {
            // Crossing strictly inside (opts.tol, FEAS_TOL] — noise, not a proof.
            let d = run_in_tree_presolve(&model, &[2.5, 0.0], &[2.5 - eps, 10.0], 0, None, &opts);
            assert!(d.ran);
            assert!(
                !d.infeasible,
                "probing path fathomed a live node at a {eps:e} crossing (< FEAS_TOL)"
            );
            assert!(d.lb[0] <= d.ub[0], "probing path returned an inverted box");
            assert!(d.lb[0] <= 2.5 && 2.5 <= d.ub[0], "probing path cut x=2.5");
            checked += 1;
        }
        assert_eq!(
            checked, 3,
            "probe did not execute the comparisons it claims"
        );
    }

    /// ANTI-PERMISSIVENESS CONTROL for the probing path: a genuine infeasibility
    /// must still fathom with probing on and a small `opts.tol`.
    #[test]
    fn probing_path_still_fathoms_genuine_infeasibility() {
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            max_iter: 16,
            tol: 1e-8,
            probing: true,
            probe_max_vars: 8,
        };
        let d = run_in_tree_presolve(&model, &[10.0, 10.0], &[10.0, 10.0], 0, None, &opts);
        assert!(d.ran);
        assert!(
            d.infeasible,
            "probing path stopped fathoming a real infeasibility"
        );
    }
}
