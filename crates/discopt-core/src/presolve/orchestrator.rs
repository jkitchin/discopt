//! Presolve orchestrator (item A1 of the roadmap).
//!
//! Drives a list of [`PresolvePass`] objects to a fixed point under a
//! global budget. Replaces the previous ad-hoc sequence of bespoke
//! presolve calls.
//!
//! Termination conditions, in order of priority:
//!
//! 1. A pass detected infeasibility (any bound's `is_empty()`).
//! 2. The configured time budget was exhausted.
//! 3. The configured work-unit budget was exhausted.
//! 4. `max_iterations` sweeps completed.
//! 5. A full sweep produced no progress on any pass — the fixed point.
//!
//! The orchestrator itself is deterministic: passes run in the order
//! supplied via `OrchestratorOptions::pass_order`; no parallelism, no
//! RNG, no `HashMap` iteration in the hot path. Together with the
//! determinism of every individual pass kernel (verified in
//! `tests/presolve_determinism.rs`), this makes the entire run
//! byte-reproducible.

use std::time::Instant;

use super::delta::{PresolveDelta, TerminationReason};
use super::pass::{PassCategory, PresolveContext, PresolvePass};

/// Tunables for one orchestrator run.
pub struct OrchestratorOptions {
    /// Maximum number of sweeps over the registered passes.
    pub max_iterations: u32,
    /// Wall-clock cap (milliseconds). 0 disables the time budget.
    pub time_limit_ms: u64,
    /// Aggregate work-unit cap. 0 disables the work budget.
    pub work_unit_budget: u64,
    /// The passes to run, in order. Each sweep walks the list once.
    pub pass_order: Vec<Box<dyn PresolvePass>>,
}

impl OrchestratorOptions {
    /// Default budgets: 16 sweeps, no time / work caps. Caller must
    /// supply the pass list — there is no implicit default pass set,
    /// because the orchestrator deliberately knows nothing about the
    /// concrete pass implementations.
    pub fn with_passes(passes: Vec<Box<dyn PresolvePass>>) -> Self {
        Self {
            max_iterations: 16,
            time_limit_ms: 0,
            work_unit_budget: 0,
            pass_order: passes,
        }
    }
}

/// Outcome of an orchestrator run.
pub struct PresolveResult {
    /// Final (possibly rewritten) model.
    pub model: crate::expr::ModelRepr,
    /// Final tightened variable bounds (one per variable block).
    pub bounds: Vec<super::fbbt::Interval>,
    /// Chronological log of every pass invocation. Used for
    /// determinism tests and by Python-side stats reporting.
    pub deltas: Vec<PresolveDelta>,
    /// Number of full sweeps actually run.
    pub iterations: u32,
    /// Why the loop stopped.
    pub terminated_by: TerminationReason,
}

/// Run the fixed-point loop on `model` with the given options.
pub fn run(model: crate::expr::ModelRepr, mut opts: OrchestratorOptions) -> PresolveResult {
    let started = Instant::now();
    let mut ctx = PresolveContext::from_model(model);
    // Expose the run deadline so long passes (probing) can bail mid-loop
    // instead of only being stopped by the between-passes check below.
    ctx.deadline = if opts.time_limit_ms > 0 {
        Some(started + std::time::Duration::from_millis(opts.time_limit_ms))
    } else {
        None
    };
    let mut deltas: Vec<PresolveDelta> = Vec::new();
    let mut terminated_by = TerminationReason::IterationCap;
    let mut last_iter: u32 = 0;

    'outer: for sweep in 0..opts.max_iterations {
        ctx.iter = sweep;
        last_iter = sweep + 1;
        let mut sweep_progress = false;

        for pass in opts.pass_order.iter_mut() {
            // Snapshot category up front: invoking `run` may mutate the
            // pass's own state but the category is stable per impl.
            let category = pass.category();

            let pass_started = Instant::now();
            let mut delta = pass.run(&mut ctx);
            let elapsed_ms = pass_started.elapsed().as_secs_f64() * 1000.0;

            // Carry-through accounting in case the pass didn't fill it.
            if delta.wall_time_ms == 0.0 {
                delta.wall_time_ms = elapsed_ms;
            }
            ctx.time_used_ms += delta.wall_time_ms;
            ctx.work_units_used += delta.work_units;

            if matches!(category, PassCategory::RewritesModel) {
                ctx.resync_bounds_after_rewrite();
            }

            if any_empty(&ctx.bounds) {
                deltas.push(delta);
                terminated_by = TerminationReason::Infeasible;
                break 'outer;
            }

            if delta.made_progress() {
                sweep_progress = true;
            }
            deltas.push(delta);

            if opts.time_limit_ms > 0
                && (started.elapsed().as_secs_f64() * 1000.0) as u64 >= opts.time_limit_ms
            {
                terminated_by = TerminationReason::TimeBudget;
                break 'outer;
            }
            if opts.work_unit_budget > 0 && ctx.work_units_used >= opts.work_unit_budget {
                terminated_by = TerminationReason::WorkBudget;
                break 'outer;
            }
        }

        if !sweep_progress {
            terminated_by = TerminationReason::NoProgress;
            break;
        }
    }

    // NOTE: the tightened bounds are returned in `PresolveResult::bounds`
    // but deliberately NOT written back into `ctx.model`'s `VarInfo`. The
    // returned model is consumed downstream for solving, dual recovery,
    // and feasibility checks; mutating its declared variable bounds there
    // — even with values that are valid for *optimization* — can flip an
    // inactive bound to active (changing LP duals) or, if a tightening was
    // ever cutoff/incumbent-derived, manufacture a false infeasibility on
    // re-solve. Callers that want the tightened bounds read them from
    // `bounds`; fixed-point detection within a single `run` already works
    // because `ctx.bounds` carries across sweeps.

    PresolveResult {
        model: ctx.model,
        bounds: ctx.bounds,
        deltas,
        iterations: last_iter,
        terminated_by,
    }
}

fn any_empty(bounds: &[super::fbbt::Interval]) -> bool {
    bounds.iter().any(|b| b.is_empty())
}

#[cfg(test)]
mod tests {
    use super::super::passes;
    use super::*;
    use crate::expr::*;

    fn trivial_model() -> ModelRepr {
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![],
            variables: vec![VarInfo {
                name: "x".into(),
                var_type: VarType::Continuous,
                offset: 0,
                size: 1,
                shape: vec![],
                lb: vec![-1.0],
                ub: vec![1.0],
            }],
            n_vars: 1,
        }
    }

    #[test]
    fn orchestrator_terminates_on_no_progress() {
        let model = trivial_model();
        let opts = OrchestratorOptions::with_passes(vec![Box::new(passes::FbbtPass::default())]);
        let result = run(model, opts);
        assert_eq!(result.terminated_by, TerminationReason::NoProgress);
        assert_eq!(result.bounds.len(), 1);
        // Empty pass set runs zero iterations? No, at least one sweep
        // ran: a sweep with one no-op pass returns NoProgress.
        assert!(result.iterations >= 1);
    }

    #[test]
    fn orchestrator_honors_iteration_cap() {
        let model = trivial_model();
        let mut opts = OrchestratorOptions::with_passes(vec![Box::new(passes::AlwaysProgressPass)]);
        opts.max_iterations = 3;
        let result = run(model, opts);
        assert_eq!(result.terminated_by, TerminationReason::IterationCap);
        assert_eq!(result.iterations, 3);
        assert_eq!(result.deltas.len(), 3);
    }

    // ── cert:C-16 regression tests ────────────────────────────────────
    //
    // A variable-removing (shrinking) pass renumbers later variables
    // down. Before the fix, `resync_bounds_after_rewrite` intersected
    // `ctx.bounds[i]` (OLD variable i) with NEW variable i's declared
    // bounds *positionally* — fusing two unrelated variables' intervals.
    // An empty intersection surfaced as a false `Infeasible`; a tighter
    // non-empty one silently cut an unrelated survivor. Both models below
    // aggregate x1 (index 1, source = x0) while an *unrelated* x2 sits at
    // index 2 and slides into new index 1, colliding with old x1's bounds.

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

    /// Build `cx·x + cy·y == rhs`; leaves must already be in `arena`.
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
    fn c16_shrink_does_not_report_feasible_model_infeasible() {
        // x0 ∈ [0, 300] (objective + eq partner), x1 ∈ [100, 200]
        // (eq-only, eliminable), x2 ∈ [0, 10] (objective, unrelated).
        // Equality x1 − x0 == 0 ⇒ x0 = x1 ∈ [100, 200] — feasible.
        // Aggregation removes x1; x2 slides to new index 1. The buggy
        // resync fuses old x1's [100, 200] with x2's [0, 10] → empty →
        // false Infeasible. The fix rebuilds bounds from the new model.
        let mut arena = ExprArena::new();
        let x0 = scalar_var(&mut arena, "x0", 0);
        let x1 = scalar_var(&mut arena, "x1", 1);
        let x2 = scalar_var(&mut arena, "x2", 2);
        let eq = affine_eq(&mut arena, 1.0, x1, -1.0, x0, 0.0);
        let obj = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x0,
            right: x2,
        });
        let model = ModelRepr {
            arena,
            objective: obj,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![eq],
            variables: vec![
                vinfo("x0", 0.0, 300.0),
                vinfo("x1", 100.0, 200.0),
                vinfo("x2", 0.0, 10.0),
            ],
            n_vars: 3,
        };

        let opts = OrchestratorOptions::with_passes(vec![Box::new(passes::AggregatePass)]);
        let result = run(model, opts);

        assert_ne!(
            result.terminated_by,
            TerminationReason::Infeasible,
            "C-16: feasible model reported infeasible after aggregation shrink"
        );
        // x1 removed → 2 survivors; x2 keeps its true [0, 10] bounds.
        assert_eq!(result.bounds.len(), 2);
        let x2_bounds = result.bounds[1];
        assert!(
            (x2_bounds.lo - 0.0).abs() < 1e-9 && (x2_bounds.hi - 10.0).abs() < 1e-9,
            "C-16: unrelated survivor x2 bounds fused, got [{}, {}]",
            x2_bounds.lo,
            x2_bounds.hi
        );
    }

    #[test]
    fn c16_shrink_does_not_silently_tighten_unrelated_survivor() {
        // Same shape but x1 ∈ [3, 7] overlaps x2 ∈ [0, 10]. The buggy
        // resync fuses old x1's [3, 7] into x2 (non-empty but WRONG),
        // silently cutting x2's feasible region [0, 3) ∪ (7, 10].
        let mut arena = ExprArena::new();
        let x0 = scalar_var(&mut arena, "x0", 0);
        let x1 = scalar_var(&mut arena, "x1", 1);
        let x2 = scalar_var(&mut arena, "x2", 2);
        let eq = affine_eq(&mut arena, 1.0, x1, -1.0, x0, 0.0);
        let obj = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x0,
            right: x2,
        });
        let model = ModelRepr {
            arena,
            objective: obj,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![eq],
            variables: vec![
                vinfo("x0", 0.0, 100.0),
                vinfo("x1", 3.0, 7.0),
                vinfo("x2", 0.0, 10.0),
            ],
            n_vars: 3,
        };

        let opts = OrchestratorOptions::with_passes(vec![Box::new(passes::AggregatePass)]);
        let result = run(model, opts);

        assert_ne!(result.terminated_by, TerminationReason::Infeasible);
        assert_eq!(result.bounds.len(), 2);
        let x2_bounds = result.bounds[1];
        assert!(
            (x2_bounds.lo - 0.0).abs() < 1e-9 && (x2_bounds.hi - 10.0).abs() < 1e-9,
            "C-16: unrelated survivor x2 silently tightened to [{}, {}] (true [0, 10])",
            x2_bounds.lo,
            x2_bounds.hi
        );
    }

    #[test]
    fn c16_property_survivor_bounds_contain_feasible_point() {
        // Randomized oracle: build aggregation-eliminable models around a
        // *known feasible point*, run presolve, and assert every survivor's
        // post-presolve bounds still contain its feasible coordinate. A
        // sound presolve may only tighten toward feasibility, so excluding
        // a known-feasible point is a bug — this catches the C-16 fusion
        // class over many random shapes without needing a dense solver.
        //
        // Layout per model: index 0 = `s` (eq partner, in objective),
        // index 1 = `x_e` (eq-only, eliminable), indices 2.. = unrelated
        // survivors (in objective). Removing x_e slides the unrelated
        // survivors down into x_e's old slot — the collision the bug hits.
        let mut seed: u64 = 0x2545_F491_4F6C_DD1D;
        let mut unit = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / ((1u64 << 31) as f64) // [0, 1)
        };
        // Nonzero coefficient in [-3, 3] \ {~0}.
        let nonzero_coeff = |u: &mut dyn FnMut() -> f64| {
            let mut c = (u() * 6.0) - 3.0;
            if c.abs() < 0.5 {
                c += if c >= 0.0 { 0.5 } else { -0.5 };
            }
            c
        };

        for trial in 0..300 {
            let n_surv = 2 + (unit() * 4.0) as usize; // survivors: 2..=5
            let n_others = n_surv - 1; // ≥ 1 unrelated survivor at index ≥ 2

            // Feasible coordinates for s and the unrelated survivors.
            let s_val = (unit() * 20.0) - 10.0;
            let other_vals: Vec<f64> = (0..n_others).map(|_| (unit() * 20.0) - 10.0).collect();
            let x_e_val = (unit() * 20.0) - 10.0;
            let c = nonzero_coeff(&mut unit);
            // Equality: 1·x_e + c·s == rhs, satisfied at the feasible point.
            let rhs = x_e_val + c * s_val;

            // Bounds are random intervals that *contain* each coordinate.
            let mut vinfo_at = |name: &str, v: f64| {
                let lo = v - (unit() * 5.0 + 0.1);
                let hi = v + (unit() * 5.0 + 0.1);
                vinfo(name, lo, hi)
            };

            let mut arena = ExprArena::new();
            let s = scalar_var(&mut arena, "s", 0);
            let x_e = scalar_var(&mut arena, "x_e", 1);
            let others: Vec<ExprId> = (0..n_others)
                .map(|k| scalar_var(&mut arena, &format!("o{k}"), 2 + k))
                .collect();
            let eq = affine_eq(&mut arena, 1.0, x_e, c, s, rhs);
            // Objective = s + Σ others (x_e absent → x_e is the target).
            let mut obj = s;
            for &o in &others {
                obj = arena.add(ExprNode::BinaryOp {
                    op: BinOp::Add,
                    left: obj,
                    right: o,
                });
            }

            let mut variables = vec![vinfo_at("s", s_val), vinfo_at("x_e", x_e_val)];
            for (k, ov) in other_vals.iter().enumerate() {
                variables.push(vinfo_at(&format!("o{k}"), *ov));
            }
            let model = ModelRepr {
                arena,
                objective: obj,
                objective_sense: ObjectiveSense::Minimize,
                constraints: vec![eq],
                variables,
                n_vars: n_surv + 1,
            };

            let opts = OrchestratorOptions::with_passes(vec![Box::new(passes::AggregatePass)]);
            let result = run(model, opts);

            assert_ne!(
                result.terminated_by,
                TerminationReason::Infeasible,
                "C-16 trial {trial}: feasible model reported infeasible"
            );
            assert_eq!(
                result.bounds.len(),
                n_surv,
                "C-16 trial {trial}: expected {n_surv} survivors"
            );
            // Survivor order after removing x_e: [s, o0, o1, ...].
            let feasible: Vec<f64> = std::iter::once(s_val).chain(other_vals).collect();
            for (k, &fv) in feasible.iter().enumerate() {
                let b = result.bounds[k];
                assert!(
                    b.lo <= fv + 1e-9 && fv - 1e-9 <= b.hi,
                    "C-16 trial {trial}: survivor {k} bounds [{}, {}] exclude feasible {fv}",
                    b.lo,
                    b.hi
                );
            }
        }
    }
}
