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
    /// Cuts folded into the search's LP on this thread.
    pub(super) static FOLDED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
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
    pub sep_calls: usize,
    pub sep_lifts: usize,
    /// Cuts folded into the search's LP.
    pub folded: usize,
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

/// The search's state that persists across main-loop rounds (and across folds of
/// cuts into its LP between rounds).
pub(super) struct State {
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
    /// Node separation's cuts (globally valid Gomory cuts found at nodes), offered to
    /// later nodes' separation and folded into the LP between rounds; with
    /// signatures for deduplication, the best efficacy each reached at a node this
    /// round, and whether it is in the LP.
    pool: Vec<GomoryCut>,
    pool_sigs: std::collections::HashSet<Vec<(u32, i64)>>,
    pool_score: std::collections::HashMap<usize, f64>,
    pool_in_lp: Vec<bool>,
    pub sep_calls: usize,
    pub sep_lifts: usize,
    pub folded: usize,
}

/// The production search environment for one round: the round's LP context, the
/// limits, and the persistent state.
pub(super) struct HighsEnv<'a, 'b> {
    ctx: &'a NodeCtx<'b>,
    lim: &'a Limits,
    pub st: State,
}

impl<'a, 'b> HighsEnv<'a, 'b> {
    /// Take up incumbents the components found, and set the cutoff to the best value.
    fn sync(&mut self) {
        while self.st.nseen < self.st.eval.found.len() {
            let (x, v) = self.st.eval.found[self.st.nseen].clone();
            self.st.nseen += 1;
            self.accept(x, v);
        }
    }

    fn accept(&mut self, x: Vec<f64>, v: f64) {
        if self.st.best.as_ref().is_none_or(|b| v < b.1) {
            self.st.best = Some((x, v));
            self.st.eval.cutoff = v;
        }
    }

    fn optlim(&self) -> f64 {
        optimality_limit(
            self.lim,
            self.st.best.as_ref().map_or(f64::INFINITY, |b| b.1),
        )
    }

    fn check_limits(&mut self) -> bool {
        if self.st.stop.is_some() {
            return true;
        }
        if self.st.nodes as usize >= self.lim.max_nodes {
            self.st.stop = Some(Stop::Nodes);
        } else if self
            .lim
            .max_open_nodes
            .is_some_and(|k| self.st.q.num_nodes() as usize > k)
        {
            self.st.stop = Some(Stop::OpenNodes);
        } else if self
            .lim
            .deadline
            .is_some_and(|d| std::time::Instant::now() >= d)
        {
            self.st.stop = Some(Stop::Time);
        }
        self.st.stop.is_some()
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
        self.st.q.emplace_node(d, b, lb, est, depth).1
    }
    fn queue_pop_best(&mut self) -> OpenNode {
        self.st.q.pop_best_node().1
    }
    fn queue_pop_best_bound(&mut self) -> OpenNode {
        self.st.q.pop_best_bound_node().1
    }
    fn queue_is_empty(&self) -> bool {
        self.st.q.is_empty()
    }
    fn queue_best_lower_bound(&self) -> f64 {
        self.st.q.best_lower_bound()
    }
    fn queue_best_bound_stack_size(&self) -> i32 {
        self.st.q.best_bound_domchg_stack_size()
    }
    fn cutoff_bound(&self) -> f64 {
        self.st.eval.cutoff
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
        self.st.eval.dom.backtrack_to_global();
        let stack: Vec<DomainChange> = node.domchgstack.iter().map(from_q).collect();
        let br: Vec<usize> = node.branchings.iter().map(|&p| p as usize).collect();
        self.st.eval.dom.set_domain_change_stack(&stack, &br);
        self.st.eval.subrootsol.clear();
    }
    fn domain_stack_len(&self) -> i32 {
        self.st.eval.dom.domain_change_stack().len() as i32
    }
    fn domain_backtrack(&mut self) {
        self.st.eval.dom.backtrack();
    }
    fn domain_backtrack_to_global(&mut self) {
        self.st.eval.dom.backtrack_to_global();
    }
    fn domain_propagate_after_backtrack(&mut self, _depth: i32) -> bool {
        self.st.eval.dom.propagate();
        self.st.eval.dom.infeasible()
    }
    fn domain_try_sibling(&mut self, d: QDomainChange, skip_propagate: bool) -> bool {
        self.st.eval.dom.change_bound(from_q(&d), Reason::Branching);
        if !skip_propagate && !self.st.eval.dom.infeasible() {
            self.st.eval.dom.propagate();
        }
        self.st.eval.dom.infeasible()
    }
    fn domain_undo_sibling(&mut self) {
        self.st.eval.dom.backtrack();
        self.st.eval.dom.clear_changed_cols();
    }
    fn domain_propagate_for_queue(&mut self) -> bool {
        let n = self.st.eval.dom.changed_cols().len();
        self.st.eval.dom.propagate();
        self.st.eval.dom.clear_changed_cols_from(n);
        self.st.eval.dom.infeasible()
    }
    fn domain_reduced_stack(&mut self) -> (Vec<QDomainChange>, Vec<i32>) {
        let (s, p) = self.st.eval.dom.reduced_domain_change_stack();
        (
            s.iter().map(to_q).collect(),
            p.iter().map(|&x| x as i32).collect(),
        )
    }
    fn domain_objective_lower_bound(&mut self) -> f64 {
        match self.st.eval.dom.objective_lower() {
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
        self.st.nodes += 1;
        let limit = self.optlim();
        let r = self
            .st
            .eval
            .evaluate(self.ctx, node, parent, &mut self.st.pc, false, limit);
        self.sync();
        r
    }
    fn branch(&mut self, top: &NodeData, parent: Option<&NodeData>, _depth: i32) -> BranchOutcome {
        let q = &self.st.q;
        let nodes = |c: usize| (q.num_nodes_up(c), q.num_nodes_down(c));
        let deadline = self.lim.deadline;
        let limits = || deadline.is_some_and(|d| std::time::Instant::now() >= d);
        let bc = BranchCtx {
            optimality_limit: optimality_limit(
                self.lim,
                self.st.best.as_ref().map_or(f64::INFINITY, |b| b.1),
            ),
            upper_limit: self.st.eval.cutoff,
            feastol: 1e-6,
            epsilon: 1e-9,
            nodes_up_down: &nodes,
            check_limits: &limits,
            child_rule: ChildSelectionRule::RootSol,
            inheuristic: false,
        };
        let out = self
            .st
            .eval
            .branch(self.ctx, &mut self.st.pc, top, parent, &bc);
        self.st.commits += out
            .steps
            .iter()
            .filter(|s| matches!(s, BranchStep::Commit { .. }))
            .count();
        if out.branched.is_some() {
            self.st.branched += 1;
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
            self.st.pc.score_up(c as usize, p)
        } else {
            self.st.pc.score_down(c as usize, p)
        };
        let a = ancestor.map(|(ac, ap, lower)| {
            let (d, u) = (
                self.st.pc.score_down(ac as usize, ap),
                self.st.pc.score_up(ac as usize, ap),
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
        // `separateAndStoreBasis` with discopt's node separation: node-local Gomory
        // rounds plus the pool cuts this node violates, on a copy of the node LP; the
        // bound they prove holds on the node's box and lifts the node when the dive
        // re-evaluates it. (HiGHS adds its cuts to the shared LP; the working LP is
        // fixed during this search, so the cuts stay node-local and in the pool.)
        let ctx = self.ctx;
        if !ctx.node_gmi {
            return false;
        }
        let (Some(x), Some(y), Some(basis), Some((fl, fu))) = (
            self.st.eval.point_full.clone(),
            self.st.eval.last_dual_raw.clone(),
            self.st.eval.basis.clone(),
            self.st.eval.last_box.clone(),
        ) else {
            return false;
        };
        let integral = (0..ctx.ns).all(|j| !ctx.is_int[j] || frac(x[j]) <= INT_TOL);
        if integral
            || self
                .lim
                .deadline
                .is_some_and(|d| std::time::Instant::now() >= d)
        {
            return false;
        }
        let mut pick: Vec<(f64, usize)> = Vec::new();
        for (k, cut) in self.st.pool.iter().enumerate() {
            if self.st.pool_in_lp[k] {
                continue;
            }
            let (mut act, mut nrm) = (0.0, 0.0);
            for (j, &a) in cut.coeffs.iter().enumerate() {
                if a != 0.0 {
                    act += a * x[j];
                    nrm += a * a;
                }
            }
            let viol = cut.rhs - act;
            if nrm > 0.0 && viol > 0.0 && viol / f64::sqrt(nrm) >= CUT_MIN_EFFICACY {
                pick.push((viol / f64::sqrt(nrm), k));
            }
        }
        pick.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        pick.truncate(NODE_GMI_POOL_PER_NODE);
        for &(eff, k) in &pick {
            let e = self.st.pool_score.entry(k).or_insert(eff);
            *e = e.max(eff);
        }
        let rows: Vec<GomoryCut> = pick
            .iter()
            .map(|&(_, k)| {
                let mut coeffs = self.st.pool[k].coeffs.clone();
                coeffs.resize(ctx.n_w, 0.0);
                GomoryCut {
                    coeffs,
                    rhs: self.st.pool[k].rhs,
                }
            })
            .collect();
        self.st.sep_calls += 1;
        let lg = node_local_gmi_bound(
            ctx,
            &fl,
            &fu,
            &x,
            &y,
            &basis,
            &rows,
            NODE_GMI_ROUNDS,
            self.st.nodes,
        );
        self.st.eval.lp_iters += lg.pivots as i64;
        if let Some(g) = lg.bound {
            self.st.sep_lifts += 1;
            self.st.eval.pending_lift = Some(g + ctx.obj_const);
        }
        for cut in lg.new_cuts {
            if self.st.pool_sigs.insert(cut_signature(&cut)) {
                // Its efficacy at the node point that produced it.
                let (mut act, mut nrm) = (0.0, 0.0);
                for (j, &a) in cut.coeffs.iter().enumerate() {
                    if a != 0.0 && j < x.len() {
                        act += a * x[j];
                        nrm += a * a;
                    }
                }
                let k = self.st.pool.len();
                if nrm > 0.0 && cut.rhs - act > 0.0 {
                    self.st
                        .pool_score
                        .insert(k, (cut.rhs - act) / f64::sqrt(nrm));
                }
                self.st.pool.push(cut);
                self.st.pool_in_lp.push(false);
            }
        }
        false
    }
    fn heuristics_allowed(&mut self) -> bool {
        self.lim.heuristics
    }
    fn run_heuristics(&mut self) {
        // discopt's rounding of the node LP point (HiGHS: randomized rounding, RENS,
        // RINS -- stage 6).
        let Some(x) = self.st.eval.point_full.clone() else {
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
            self.st.heuristic_hits += 1;
            self.accept(cand, v);
        }
    }
    fn after_round(&mut self) -> bool {
        self.st.q.perform_bounding(self.st.eval.cutoff);
        let optlim = self.optlim();
        self.st.q.set_optimality_limit(optlim);
        let inc = self.st.best.as_ref().map_or(f64::INFINITY, |b| b.1);
        let lb = if self.st.q.is_empty() {
            inc
        } else {
            inc.min(self.st.q.best_lower_bound())
        }
        .min(self.st.eval.unresolved_floor);
        if gap_closed(self.lim, inc, lb) {
            self.st.gap_stop = true;
            return true;
        }
        self.check_limits()
    }
}

/// Search the whole tree from the driver's root working LP. `initial` is the best
/// verified point the driver already holds; `root_bound` a valid bound on the root box
/// the driver already proved (its root evaluation, node-local cuts included), with
/// `obj_const` (`-inf` for none). `None` when the root LP is unbounded or
/// the rows are infeasible over the global box before any search (the driver's own
/// handling then decides).
pub(super) fn run(
    ctx: &NodeCtx<'_>,
    lim: &Limits,
    initial: Option<(Vec<f64>, f64)>,
    root_bound: f64,
) -> Option<Outcome> {
    #[cfg(test)]
    RUNS.with(|r| r.set(r.get() + 1));
    let mut eval = HighsEval::new(ctx)?;
    eval.tighten_cutoff = false;
    let mut st = State {
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
        pool: Vec::new(),
        pool_sigs: std::collections::HashSet::new(),
        pool_score: std::collections::HashMap::new(),
        pool_in_lp: Vec::new(),
        sep_calls: 0,
        sep_lifts: 0,
        folded: 0,
    };
    st.q.emplace_node(Vec::new(), Vec::new(), root_bound, root_bound, 1);
    // The search's own copy of the working LP, grown by folding cuts between rounds
    // (HiGHS adds separated cuts to its LP; the driver's LP stays as it was).
    let mut own = OwnedLp::from_ctx(ctx);
    let fold_cap = (ctx.n_orig_rows / 2).max(FOLD_CAP_MIN);
    let mut search = Search {
        count_tree_weight: true,
        ..Default::default()
    };
    let mut ml = MainLoop::default();
    let mut root = true;
    let mut accepted_initial = initial;
    loop {
        let rctx = own.ctx(ctx);
        let mut env = HighsEnv {
            ctx: &rctx,
            lim,
            st,
        };
        if let Some((x, v)) = accepted_initial.take() {
            env.accept(x, v);
        }
        let mut stop = false;
        let mut rounds = 0usize;
        // Rounds on this LP until there are cuts worth folding.
        while !env.queue_is_empty() {
            stop = ml.round(&mut env, &mut search, root, NO_NODE_LIMIT);
            if root && env.st.eval.unbounded {
                return None;
            }
            root = false;
            rounds += 1;
            if stop || (env.st.folded < fold_cap && fold_candidates(&env.st) >= FOLD_MIN) {
                break;
            }
        }
        let _ = rounds;
        st = env.st;
        if stop || st.q.is_empty() {
            break;
        }
        // Fold the best-scoring pool cuts into the LP and rebuild what depends on it.
        let mut cands: Vec<(f64, usize)> = st
            .pool_score
            .iter()
            .filter(|(k, _)| !st.pool_in_lp[**k])
            .map(|(&k, &sc)| (sc, k))
            .collect();
        cands.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        cands.truncate(FOLD_PER_ROUND.min(fold_cap - st.folded));
        let cuts: Vec<GomoryCut> = cands.iter().map(|&(_, k)| st.pool[k].clone()).collect();
        for &(_, k) in &cands {
            st.pool_in_lp[k] = true;
        }
        st.pool_score.clear();
        st.folded += cuts.len();
        #[cfg(test)]
        FOLDED.with(|f| f.set(f.get() + cuts.len()));
        own.add_cuts(&cuts);
        let rctx = own.ctx(ctx);
        if !st.eval.relp(&rctx) {
            // The rows (cuts included, all valid for every integer point of the
            // box) are infeasible over the global box: no integer point is left.
            st.q.clear();
            break;
        }
    }
    let inc = st.best.as_ref().map_or(f64::INFINITY, |b| b.1);
    let open = if st.q.is_empty() {
        f64::INFINITY
    } else {
        st.q.best_lower_bound()
    };
    let bound = inc.min(open).min(st.eval.unresolved_floor);
    Some(Outcome {
        best: st.best.clone(),
        bound,
        complete: st.stop.is_none() && (st.q.is_empty() || st.gap_stop),
        unresolved: st.eval.unresolved_floor < inc,
        stop: st.stop,
        nodes: st.nodes,
        lp_iters: st.eval.lp_iters,
        sb_iters: st.eval.sb_iters,
        commits: st.commits,
        branched: st.branched,
        heuristic_hits: st.heuristic_hits,
        sep_calls: st.sep_calls,
        sep_lifts: st.sep_lifts,
        folded: st.folded,
    })
}

/// Fold once any scored cut is waiting (HiGHS adds separated cuts to its LP at once), at
/// most this many per fold, and
/// at most `max(FOLD_CAP_MIN, original rows / 2)` in all.
const FOLD_MIN: usize = 1;
const FOLD_PER_ROUND: usize = 20;
const FOLD_CAP_MIN: usize = 50;

/// Pool cuts scored since the last fold that are not in the LP yet.
fn fold_candidates(st: &State) -> usize {
    st.pool_score.keys().filter(|&&k| !st.pool_in_lp[k]).count()
}

/// The search's own working LP `A x = b` (the driver's at the root, then grown by
/// folded cuts `a.x - s = rhs`, `s >= 0`) with the solve structures over it.
pub(super) struct OwnedLp {
    csc: SparseCols,
    b: Vec<f64>,
    c: Vec<f64>,
    l: Vec<f64>,
    u: Vec<f64>,
    is_int_full: Vec<bool>,
    slack_l: Vec<f64>,
    slack_u: Vec<f64>,
    problem: ScaledProblem,
    m: usize,
    n: usize,
}

impl OwnedLp {
    fn from_ctx(ctx: &NodeCtx<'_>) -> Self {
        let csc = ctx.csc_rc.clone();
        let problem = ScaledProblem::new(csc.clone(), ctx.m_w, ctx.n_w, ctx.c_w, ctx.b_w);
        OwnedLp {
            csc,
            b: ctx.b_w[..ctx.m_w].to_vec(),
            c: ctx.c_w[..ctx.n_w].to_vec(),
            l: ctx.l_w[..ctx.n_w].to_vec(),
            u: ctx.u_w[..ctx.n_w].to_vec(),
            is_int_full: ctx.is_int_full[..ctx.n_w].to_vec(),
            slack_l: ctx.slack_l.to_vec(),
            slack_u: ctx.slack_u.to_vec(),
            problem,
            m: ctx.m_w,
            n: ctx.n_w,
        }
    }

    /// The node context over this LP (everything else from the driver's).
    fn ctx<'s>(&'s self, base: &NodeCtx<'s>) -> NodeCtx<'s> {
        let p = &self.problem;
        NodeCtx {
            b_w: &self.b,
            c_w: &self.c,
            l_w: &self.l,
            u_w: &self.u,
            scaling: p.scaling(),
            sc: p.solve_c(),
            sb: p.solve_b(),
            csc: p.solve_matrix(),
            csc_rc: p.original_matrix(),
            problem: p,
            slack_l: &self.slack_l,
            slack_u: &self.slack_u,
            is_int_full: &self.is_int_full,
            m_w: self.m,
            n_w: self.n,
            // The driver's root basis is over its LP's columns: valid until the
            // first fold (the search keeps its own basis from the first solve on).
            root_warm_basis: if self.n == base.n_w {
                base.root_warm_basis
            } else {
                None
            },
            cut_lp_rows: &[],
            gmi_pool: &[],
            ..*base
        }
    }

    /// Append each cut `a.x >= rhs` as the row `a.x - s = rhs` with a new surplus
    /// slack `s` in `[0, inf)`.
    fn add_cuts(&mut self, cuts: &[GomoryCut]) {
        if cuts.is_empty() {
            return;
        }
        let (cp, ri, rv) = self.csc.raw();
        let mut cols: Vec<Vec<(usize, f64)>> = (0..self.n)
            .map(|j| (cp[j]..cp[j + 1]).map(|k| (ri[k], rv[k])).collect())
            .collect();
        for cut in cuts {
            let row = self.m;
            for (j, &a) in cut.coeffs.iter().enumerate() {
                if a != 0.0 && j < self.n {
                    cols[j].push((row, a));
                }
            }
            cols.push(vec![(row, -1.0)]);
            self.b.push(cut.rhs);
            self.c.push(0.0);
            self.l.push(0.0);
            self.u.push(INF);
            self.is_int_full.push(false);
            self.slack_l.push(0.0);
            self.slack_u.push(INF);
            self.m += 1;
            self.n += 1;
        }
        let mut cpn = vec![0usize];
        let (mut rin, mut rvn) = (Vec::new(), Vec::new());
        for col in &cols {
            for &(i, v) in col {
                rin.push(i);
                rvn.push(v);
            }
            cpn.push(rin.len());
        }
        self.csc = SparseCols::from_csc(cpn, rin, rvn);
        self.problem = ScaledProblem::new(self.csc.clone(), self.m, self.n, &self.c, &self.b);
    }
}
