//! Provenance: the main loop of `HighsMipSolver::run` (install, process, return open
//! nodes, update the lower bound, check limits) from HiGHS
//! (https://github.com/ERGO-Code/HiGHS, Copyright (c) HiGHS, MIT License), translated
//! into Rust and wired to discopt's components, for deep integration with discopt's LP,
//! safety layer and certificate machinery and a pure-Rust build with no C/C++ compiler
//! to ship. See `THIRD_PARTY_NOTICES.md` at the repository root.
//!
//! Stages 1d-4/1d-5 of `docs/dev/highs-mip-replacement-plan.md`: the production
//! [`SearchEnv`] -- the node queue, limits, the local domain, the evaluation
//! (`highs_eval`), branching (`highs_branch`) with the `HighsPseudocost` translation,
//! and the rounding heuristic at nodes -- and [`run`], which searches the whole tree
//! from the working LP the driver has at its root.
//!
//! Limits and certificates are discopt's:
//! * the cutoff is the incumbent value itself (nodes are pruned only when their safe
//!   bound exceeds it), the optimality limit the incumbent less the caller's gap
//!   allowance; nodes between the two stay in the queue's suboptimal set and keep
//!   counting for the dual bound, as in HiGHS;
//! * the dual bound is `min(incumbent, best queued bound, unresolved floor)` -- the
//!   floor holds the bounds of nodes no LP could resolve (HiGHS declares those
//!   infeasible);
//! * time, node and open-node limits stop the search with that valid bound,
//!   uncertified.
//!
//! Not yet re-hosted (the driver keeps the batched search for these): lazy
//! constraints, node and bound hooks, the debugger, node cuts (the working LP is
//! fixed during this search), and an unbounded LP relaxation.

use super::highs_branch::{BranchCtx, BranchPseudocost, ChildSelectionRule};
use super::highs_eval::HighsEval;
use super::*;
use crate::bnb::highs_search::node_queue::{DomainChange as QDomainChange, NodeQueue, OpenNode};
use crate::bnb::highs_search::pseudocost::{Pseudocost, DEFAULT_MIN_RELIABLE};
use crate::bnb::highs_search::search::{
    BranchOutcome, BranchStep, MainLoop, NodeData, NodeResult as SearchResult, Search, SearchEnv,
    NO_NODE_LIMIT,
};
use crate::lp::highs_cutgen::domain::{BoundType, DomainChange, Reason};

#[cfg(test)]
thread_local! {
    /// Searches run on this thread (tests prove the route fired).
    pub(super) static RUNS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Limits and tolerances for [`run`].
pub(super) struct Limits {
    /// Nodes this search may evaluate.
    pub max_nodes: usize,
    /// Open-node cap (`MilpOptions::max_open_nodes`).
    pub max_open_nodes: Option<usize>,
    /// Wall-clock deadline.
    pub deadline: Option<std::time::Instant>,
    /// Relative gap tolerance and its denominator rule, absolute gap tolerance.
    pub gap_tol: f64,
    pub rel_gap_max_abs: bool,
    pub abs_gap_tol: Option<f64>,
    /// Run the rounding heuristic at nodes.
    pub heuristics: bool,
}

/// Why the search stopped early.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Stop {
    Nodes,
    OpenNodes,
    Time,
}

/// The outcome of [`run`].
#[derive(Debug, Clone)]
pub(super) struct Outcome {
    /// Best verified point (structurals) and its value (with `obj_const`).
    pub best: Option<(Vec<f64>, f64)>,
    /// Valid dual bound (with `obj_const`).
    pub bound: f64,
    /// The search finished (queue empty or the gap closed) without a limit.
    pub complete: bool,
    /// The dual bound rests on nodes no LP resolved.
    pub unresolved: bool,
    pub stop: Option<Stop>,
    pub nodes: i64,
    pub lp_iters: i64,
    pub sb_iters: i64,
    pub commits: usize,
    pub branched: usize,
    pub heuristic_hits: usize,
}

/// The relative and absolute gap test the driver uses (`gap_closed`).
fn gap_closed(lim: &Limits, inc: f64, lb: f64) -> bool {
    if !inc.is_finite() || lb == f64::NEG_INFINITY {
        return false;
    }
    let raw = inc - lb;
    let denom = if lim.rel_gap_max_abs {
        inc.abs().max(lb.abs()).max(1e-10)
    } else {
        inc.abs().max(1.0)
    };
    (raw / denom).max(0.0) <= lim.gap_tol || lim.abs_gap_tol.is_some_and(|t| raw.max(0.0) <= t)
}

/// The largest bound at which the gap counts as closed against `inc` (HiGHS's
/// `optimality_limit`): nodes above it are suboptimal.
fn optimality_limit(lim: &Limits, inc: f64) -> f64 {
    if !inc.is_finite() {
        return f64::INFINITY;
    }
    let rel = if lim.rel_gap_max_abs {
        lim.gap_tol * inc.abs()
    } else {
        lim.gap_tol * inc.abs().max(1.0)
    };
    let allow = rel.max(lim.abs_gap_tol.unwrap_or(0.0));
    inc - allow
}

fn to_q(c: &DomainChange) -> QDomainChange {
    QDomainChange {
        col: c.column as i32,
        upper: c.boundtype == BoundType::Upper,
        val: c.boundval,
    }
}

fn from_q(c: &QDomainChange) -> DomainChange {
    DomainChange {
        boundval: c.val,
        column: c.col as usize,
        boundtype: if c.upper {
            BoundType::Upper
        } else {
            BoundType::Lower
        },
    }
}

/// The production search environment.
pub(super) struct HighsEnv<'a, 'b> {
    ctx: &'a NodeCtx<'b>,
    lim: &'a Limits,
    pub eval: HighsEval,
    pub pc: Pseudocost,
    pub q: NodeQueue,
    pub best: Option<(Vec<f64>, f64)>,
    nseen: usize,
    pub nodes: i64,
    pub commits: usize,
    pub branched: usize,
    pub heuristic_hits: usize,
    pub stop: Option<Stop>,
    gap_stop: bool,
}

impl<'a, 'b> HighsEnv<'a, 'b> {
    /// Take up incumbents the components found, and set the cutoff to the best value.
    fn sync(&mut self) {
        while self.nseen < self.eval.found.len() {
            let (x, v) = self.eval.found[self.nseen].clone();
            self.nseen += 1;
            self.accept(x, v);
        }
    }

    fn accept(&mut self, x: Vec<f64>, v: f64) {
        if self.best.as_ref().is_none_or(|b| v < b.1) {
            self.best = Some((x, v));
            self.eval.cutoff = v;
        }
    }

    fn optlim(&self) -> f64 {
        optimality_limit(self.lim, self.best.as_ref().map_or(f64::INFINITY, |b| b.1))
    }

    fn check_limits(&mut self) -> bool {
        if self.stop.is_some() {
            return true;
        }
        if self.nodes as usize >= self.lim.max_nodes {
            self.stop = Some(Stop::Nodes);
        } else if self
            .lim
            .max_open_nodes
            .is_some_and(|k| self.q.num_nodes() as usize > k)
        {
            self.stop = Some(Stop::OpenNodes);
        } else if self
            .lim
            .deadline
            .is_some_and(|d| std::time::Instant::now() >= d)
        {
            self.stop = Some(Stop::Time);
        }
        self.stop.is_some()
    }
}

impl SearchEnv for HighsEnv<'_, '_> {
    fn queue_emplace(
        &mut self,
        d: Vec<QDomainChange>,
        b: Vec<i32>,
        lb: f64,
        est: f64,
        depth: i32,
    ) -> f64 {
        self.q.emplace_node(d, b, lb, est, depth).1
    }
    fn queue_pop_best(&mut self) -> OpenNode {
        self.q.pop_best_node().1
    }
    fn queue_pop_best_bound(&mut self) -> OpenNode {
        self.q.pop_best_bound_node().1
    }
    fn queue_is_empty(&self) -> bool {
        self.q.is_empty()
    }
    fn queue_best_lower_bound(&self) -> f64 {
        self.q.best_lower_bound()
    }
    fn queue_best_bound_stack_size(&self) -> i32 {
        self.q.best_bound_domchg_stack_size()
    }
    fn cutoff_bound(&self) -> f64 {
        self.eval.cutoff
    }
    fn optimality_limit(&self) -> f64 {
        self.optlim()
    }
    fn feastol(&self) -> f64 {
        1e-6
    }
    fn search_check_limits(&mut self, _nnodes: i64) -> bool {
        self.check_limits()
    }
    fn mip_check_limits(&mut self) -> bool {
        self.check_limits()
    }
    fn domain_install(&mut self, node: &OpenNode) {
        self.eval.dom.backtrack_to_global();
        let stack: Vec<DomainChange> = node.domchgstack.iter().map(from_q).collect();
        let br: Vec<usize> = node.branchings.iter().map(|&p| p as usize).collect();
        self.eval.dom.set_domain_change_stack(&stack, &br);
        self.eval.subrootsol.clear();
    }
    fn domain_stack_len(&self) -> i32 {
        self.eval.dom.domain_change_stack().len() as i32
    }
    fn domain_backtrack(&mut self) {
        self.eval.dom.backtrack();
    }
    fn domain_backtrack_to_global(&mut self) {
        self.eval.dom.backtrack_to_global();
    }
    fn domain_propagate_after_backtrack(&mut self, _depth: i32) -> bool {
        self.eval.dom.propagate();
        self.eval.dom.infeasible()
    }
    fn domain_try_sibling(&mut self, d: QDomainChange, skip_propagate: bool) -> bool {
        self.eval.dom.change_bound(from_q(&d), Reason::Branching);
        if !skip_propagate && !self.eval.dom.infeasible() {
            self.eval.dom.propagate();
        }
        self.eval.dom.infeasible()
    }
    fn domain_undo_sibling(&mut self) {
        self.eval.dom.backtrack();
        self.eval.dom.clear_changed_cols();
    }
    fn domain_propagate_for_queue(&mut self) -> bool {
        let n = self.eval.dom.changed_cols().len();
        self.eval.dom.propagate();
        self.eval.dom.clear_changed_cols_from(n);
        self.eval.dom.infeasible()
    }
    fn domain_reduced_stack(&mut self) -> (Vec<QDomainChange>, Vec<i32>) {
        let (s, p) = self.eval.dom.reduced_domain_change_stack();
        (
            s.iter().map(to_q).collect(),
            p.iter().map(|&x| x as i32).collect(),
        )
    }
    fn domain_objective_lower_bound(&mut self) -> f64 {
        match self.eval.dom.objective_lower() {
            Some((v, 0)) => v + self.ctx.obj_const,
            _ => f64::NEG_INFINITY,
        }
    }
    fn global_propagate(&mut self) {}
    fn global_infeasible(&self) -> bool {
        false
    }
    fn evaluate(
        &mut self,
        node: &mut NodeData,
        parent: Option<&NodeData>,
        _depth: i32,
    ) -> SearchResult {
        self.nodes += 1;
        let limit = self.optlim();
        let r = self
            .eval
            .evaluate(self.ctx, node, parent, &mut self.pc, false, limit);
        self.sync();
        r
    }
    fn branch(&mut self, top: &NodeData, parent: Option<&NodeData>, _depth: i32) -> BranchOutcome {
        let q = &self.q;
        let nodes = |c: usize| (q.num_nodes_up(c), q.num_nodes_down(c));
        let deadline = self.lim.deadline;
        let limits = || deadline.is_some_and(|d| std::time::Instant::now() >= d);
        let bc = BranchCtx {
            optimality_limit: optimality_limit(
                self.lim,
                self.best.as_ref().map_or(f64::INFINITY, |b| b.1),
            ),
            upper_limit: self.eval.cutoff,
            feastol: 1e-6,
            epsilon: 1e-9,
            nodes_up_down: &nodes,
            check_limits: &limits,
            child_rule: ChildSelectionRule::RootSol,
            inheuristic: false,
        };
        let out = self.eval.branch(self.ctx, &mut self.pc, top, parent, &bc);
        self.commits += out
            .steps
            .iter()
            .filter(|s| matches!(s, BranchStep::Commit { .. }))
            .count();
        if out.branched.is_some() {
            self.branched += 1;
        }
        self.sync();
        out
    }
    fn plunge_scores(
        &mut self,
        node: (i32, f64, bool),
        ancestor: Option<(i32, f64, bool)>,
    ) -> (f64, Option<(f64, f64)>) {
        let (c, p, up) = node;
        let s = if up {
            self.pc.score_up(c as usize, p)
        } else {
            self.pc.score_down(c as usize, p)
        };
        let a = ancestor.map(|(ac, ap, lower)| {
            let (d, u) = (
                self.pc.score_down(ac as usize, ap),
                self.pc.score_up(ac as usize, ap),
            );
            if lower {
                (d, u)
            } else {
                (u, d)
            }
        });
        (s, a)
    }
    fn separate(&mut self) -> bool {
        false
    }
    fn heuristics_allowed(&mut self) -> bool {
        self.lim.heuristics
    }
    fn run_heuristics(&mut self) {
        // discopt's rounding of the node LP point (HiGHS: randomized rounding, RENS,
        // RINS -- stage 6).
        let Some(x) = self.eval.point_full.clone() else {
            return;
        };
        let ctx = self.ctx;
        if let Some((cand, v)) = try_rounding_csc(
            &x,
            ctx.ns,
            ctx.is_int,
            ctx.csc_rc,
            ctx.b_w,
            ctx.c_w,
            ctx.l_w,
            ctx.u_w,
            ctx.n_orig_rows,
            ctx.n_w,
            ctx.obj_const,
        ) {
            self.heuristic_hits += 1;
            self.accept(cand, v);
        }
    }
    fn after_round(&mut self) -> bool {
        self.q.perform_bounding(self.eval.cutoff);
        let optlim = self.optlim();
        self.q.set_optimality_limit(optlim);
        let inc = self.best.as_ref().map_or(f64::INFINITY, |b| b.1);
        let lb = if self.q.is_empty() {
            inc
        } else {
            inc.min(self.q.best_lower_bound())
        }
        .min(self.eval.unresolved_floor);
        if gap_closed(self.lim, inc, lb) {
            self.gap_stop = true;
            return true;
        }
        self.check_limits()
    }
}

/// Search the whole tree from the driver's root working LP. `initial` is the best
/// verified point the driver already holds. `None` when the root LP is unbounded or
/// the rows are infeasible over the global box before any search (the driver's own
/// handling then decides).
pub(super) fn run(
    ctx: &NodeCtx<'_>,
    lim: &Limits,
    initial: Option<(Vec<f64>, f64)>,
) -> Option<Outcome> {
    #[cfg(test)]
    RUNS.with(|r| r.set(r.get() + 1));
    let mut eval = HighsEval::new(ctx)?;
    eval.tighten_cutoff = false;
    let mut env = HighsEnv {
        ctx,
        lim,
        eval,
        pc: Pseudocost::new(ctx.ns, DEFAULT_MIN_RELIABLE),
        q: NodeQueue::new(ctx.ns),
        best: None,
        nseen: 0,
        nodes: 0,
        commits: 0,
        branched: 0,
        heuristic_hits: 0,
        stop: None,
        gap_stop: false,
    };
    if let Some((x, v)) = initial {
        env.accept(x, v);
    }
    env.q.emplace_node(
        Vec::new(),
        Vec::new(),
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
        1,
    );
    let mut search = Search {
        count_tree_weight: true,
        ..Default::default()
    };
    let mut ml = MainLoop::default();
    let mut root = true;
    while !env.queue_is_empty() {
        let stop = ml.round(&mut env, &mut search, root, NO_NODE_LIMIT);
        if root && env.eval.unbounded {
            return None;
        }
        root = false;
        if stop {
            break;
        }
    }
    let inc = env.best.as_ref().map_or(f64::INFINITY, |b| b.1);
    let open = if env.q.is_empty() {
        f64::INFINITY
    } else {
        env.q.best_lower_bound()
    };
    let bound = inc.min(open).min(env.eval.unresolved_floor);
    Some(Outcome {
        best: env.best.clone(),
        bound,
        complete: env.stop.is_none() && (env.q.is_empty() || env.gap_stop),
        unresolved: env.eval.unresolved_floor < inc,
        stop: env.stop,
        nodes: env.nodes,
        lp_iters: env.eval.lp_iters,
        sb_iters: env.eval.sb_iters,
        commits: env.commits,
        branched: env.branched,
        heuristic_hits: env.heuristic_hits,
    })
}
