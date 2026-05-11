//! `PresolvePass` trait — uniform contract for orchestrator-driven
//! presolve passes (item A1 of the roadmap).
//!
//! A pass is a small object that reads a [`PresolveContext`] (model +
//! current variable bounds + accounting), mutates it in-place, and
//! returns a [`PresolveDelta`] describing what it did.
//!
//! # Two pass categories
//!
//! Existing presolve kernels split cleanly into two shapes:
//!
//! - [`PassCategory::BoundsOnly`] — operates on `&ModelRepr` plus a
//!   `&mut Vec<Interval>` of bounds. The model itself is unchanged.
//!   Examples: `fbbt`, `simplify`, `probing`.
//! - [`PassCategory::RewritesModel`] — returns a *new* `ModelRepr`
//!   whose variable / constraint indexing may differ from the input.
//!   Examples: `eliminate_variables`, `reformulate_polynomial`.
//!
//! The orchestrator uses this category tag to decide whether to
//! invalidate cached state (notably the bounds vector) when a pass
//! mutates the model topology.
//!
//! # Determinism
//!
//! All P1 passes are deterministic by construction (no `rand`, no
//! unsorted `HashMap` iteration in the result path). New passes added
//! later must preserve this.

use super::delta::PresolveDelta;
use super::fbbt::Interval;
use crate::expr::ModelRepr;

/// Whether a pass mutates only bounds, or also rewrites the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassCategory {
    /// Pass mutates `PresolveContext::bounds` only. The orchestrator
    /// keeps the model and bounds vector aligned across calls.
    BoundsOnly,
    /// Pass replaces `PresolveContext::model`. The orchestrator
    /// recomputes the bounds vector from the new model after the
    /// call (or trusts the pass to also write it).
    RewritesModel,
}

/// Mutable state passed through the fixed-point loop.
///
/// All pass kernels operate against this struct. A pass mutates
/// `model` and/or `bounds` in place, then returns a [`PresolveDelta`]
/// describing what it did. The orchestrator updates `iter`,
/// `work_units_used`, and `time_used_ms` between calls.
pub struct PresolveContext {
    /// Current model. May be replaced by `RewritesModel` passes.
    pub model: ModelRepr,
    /// Variable bounds, one entry per variable *block* (matching
    /// `model.variables.len()`). Indexed identically to
    /// `fbbt::fbbt_with_cutoff`'s return value.
    pub bounds: Vec<Interval>,
    /// Sweep index (0-based). Incremented by the orchestrator before
    /// each full sweep of the registered passes.
    pub iter: u32,
    /// Cumulative work-units consumed by passes in this run.
    pub work_units_used: u64,
    /// Cumulative wall time consumed by passes in this run.
    pub time_used_ms: f64,
}

impl PresolveContext {
    /// Initialise a context from a model. The bounds vector is filled
    /// from each variable block's first scalar `lb`/`ub`, matching
    /// `fbbt_with_cutoff`'s convention.
    pub fn from_model(model: ModelRepr) -> Self {
        let bounds: Vec<Interval> = model
            .variables
            .iter()
            .map(|v| {
                Interval::new(
                    v.lb.first().copied().unwrap_or(f64::NEG_INFINITY),
                    v.ub.first().copied().unwrap_or(f64::INFINITY),
                )
            })
            .collect();
        Self {
            model,
            bounds,
            iter: 0,
            work_units_used: 0,
            time_used_ms: 0.0,
        }
    }

    /// Resize the bounds vector to match the (possibly larger) model
    /// after a `RewritesModel` pass added auxiliary variables. New
    /// entries inherit the new variable's declared bounds.
    pub fn resync_bounds_after_rewrite(&mut self) {
        let n = self.model.variables.len();
        if self.bounds.len() == n {
            return;
        }
        let mut new_bounds = Vec::with_capacity(n);
        for (i, v) in self.model.variables.iter().enumerate() {
            let lb = v.lb.first().copied().unwrap_or(f64::NEG_INFINITY);
            let ub = v.ub.first().copied().unwrap_or(f64::INFINITY);
            // Preserve any tightening already in `bounds` for
            // pre-existing variables; new aux vars take the model's
            // declared bounds.
            if i < self.bounds.len() {
                let prior = self.bounds[i];
                new_bounds.push(Interval::new(prior.lo.max(lb), prior.hi.min(ub)));
            } else {
                new_bounds.push(Interval::new(lb, ub));
            }
        }
        self.bounds = new_bounds;
    }
}

/// Uniform contract for any presolve pass.
pub trait PresolvePass {
    /// Stable name for logging and the `PresolveDelta::pass_name` field.
    fn name(&self) -> &'static str;

    /// Whether this pass rewrites the model topology.
    fn category(&self) -> PassCategory;

    /// Invoke the pass against `ctx`, mutating it in place. Returns a
    /// delta describing what changed.
    fn run(&mut self, ctx: &mut PresolveContext) -> PresolveDelta;
}
