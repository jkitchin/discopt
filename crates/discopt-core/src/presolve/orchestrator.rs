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
}
