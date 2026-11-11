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
//! The debugger (`MilpDebugHook`, which also serves `node_callback`) runs on this
//! search ([`Debugger`]): every evaluated node is a batch of one, with discopt's node
//! identity (`NodeIdent`) carried through the stack and the queue.
//!
//! Not yet re-hosted (the driver keeps the batched search for these): lazy
//! constraints, node and bound hooks, node cuts (the working LP is fixed during this
//! search), and an unbounded LP relaxation. Also routed there, as
//! options this search does not implement: a branching rule other than reliability,
//! a selection other than best-first, `node_propagation = false`, and the off-root
//! repair dive (`DISCOPT_MILP_DIVE_STRIDE`).

use super::highs_branch::{BranchCtx, BranchPseudocost, ChildSelectionRule};
use super::highs_eval::HighsEval;
use super::*;
use crate::bnb::highs_search::node_queue::NodeIdent;
use crate::bnb::highs_search::node_queue::{DomainChange as QDomainChange, NodeQueue, OpenNode};
use crate::bnb::highs_search::pseudocost::{Pseudocost, DEFAULT_MIN_RELIABLE};
use crate::bnb::highs_search::search::{
    BranchOutcome, BranchStep, MainLoop, NodeData, NodeFate, NodeResult as SearchResult, Search,
    SearchEnv, NO_NODE_LIMIT,
};
use crate::bnb::node::{BranchRecord, NodeStatus};
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
    /// Strong branching allowed (`MilpOptions::strong_branch`; off: `minreliable` 0).
    pub strong_branch: bool,
    /// `MilpOptions::stall_time_s` as an instant: there, a search with no incumbent
    /// stops (uncertified), and one with an incumbent runs the stall re-solve once.
    pub stall_at: Option<std::time::Instant>,
}

/// Why the search stopped early.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Stop {
    Nodes,
    OpenNodes,
    Time,
    /// The stall guard: no incumbent at `stall_at`.
    Stall,
    /// The attached debugger asked the search to stop.
    Debugger,
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
    /// Better points the sub-MIP heuristics found.
    pub sub_mip_hits: usize,
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

/// A call into the driver's heuristics.
pub(super) enum HeurCall<'x> {
    /// RENS/RINS (the driver's schedule) from an LP point (structurals), with the
    /// incumbent, at a node count.
    SubMip {
        x: &'x [f64],
        inc: Option<(&'x [f64], f64)>,
        nodes: usize,
    },
    /// The once-per-solve stall re-solve (`try_stall_redive`) around the incumbent.
    StallRedive { inc: (&'x [f64], f64) },
}

/// The driver's heuristics: a verified better point or `None`.
pub(super) type SubMip<'a> = &'a mut dyn FnMut(HeurCall<'_>) -> Option<(Vec<f64>, f64)>;

/// The debugger attached to the search (`MilpDebugHook`), and what it needs to report
/// the tree the way the batched search does: every evaluated node is a batch of one
/// (`AfterSelect` before its first evaluation, `AfterProcess` when it is branched or
/// closed), with its lineage and the nodes created since the last report.
pub(super) struct Debugger<'a> {
    pub hook: &'a dyn MilpDebugHook,
    pub t_start: std::time::Instant,
    /// The driver's checkpoint counter, continued.
    pub iter: usize,
    /// The root's id; the driver already fired `IterStart`/`AfterSelect` for it.
    pub root_id: u64,
    lineage: HashMap<u64, NodeLineage>,
    /// Created since the last `AfterProcess`.
    created: Vec<u64>,
    /// Ever selected; selected and not yet processed.
    selected: HashSet<u64>,
    open: HashSet<u64>,
    /// Nodes whose evaluation produced a verified integral point.
    fathomed: HashSet<u64>,
    /// The box reported for a selected node, with how many of its children have been
    /// settled (reported, or pruned at creation); dropped once two have (a branched
    /// node has two; a node committed to one child by strong branching keeps its box
    /// until the search ends, which is bounded by the number of commits). A node reinstalled from the queue carries only
    /// HiGHS's reduced domain-change stack, so before its own propagation its domain
    /// can be wider than its parent's was; its region is inside the parent's box, so
    /// the reported box is the domain intersected with the parent's.
    boxes: HashMap<u64, (Vec<f64>, Vec<f64>, usize)>,
    /// The LP bound and point of the node evaluated last.
    best: Option<(u64, f64, Vec<f64>)>,
    /// A valid dual bound: the root's, then each round's (the stack's nodes are not in
    /// the queue mid-round, so the queue alone is not a bound there).
    bound: f64,
    last_inc: f64,
}

impl<'a> Debugger<'a> {
    pub fn new(
        hook: &'a dyn MilpDebugHook,
        t_start: std::time::Instant,
        iter: usize,
        root_id: u64,
        root_box: (Vec<f64>, Vec<f64>),
        root_relaxation: Option<(f64, Vec<f64>)>,
        last_inc: f64,
    ) -> Self {
        let mut lineage = HashMap::new();
        lineage.insert(
            root_id,
            NodeLineage {
                id: NodeId(root_id as usize),
                parent: None,
                depth: 0,
                branch: None,
                status: NodeStatus::Pending,
            },
        );
        // The driver selected the root (its root batch's `AfterSelect`).
        let selected: HashSet<u64> = [root_id].into_iter().collect();
        let open = selected.clone();
        Debugger {
            hook,
            t_start,
            iter,
            root_id,
            lineage,
            created: Vec::new(),
            selected,
            open,
            fathomed: HashSet::new(),
            // The root's box exactly as the driver's `AfterSelect` reported it.
            boxes: [(root_id, (root_box.0, root_box.1, 0))]
                .into_iter()
                .collect(),
            // The driver's root evaluation, for the root's `AfterProcess` when the
            // search closes the root without evaluating it again.
            best: root_relaxation.map(|(b, x)| (root_id, b, x)),
            bound: f64::NEG_INFINITY,
            last_inc,
        }
    }
}

/// One node's box and id, as a debugger batch: `(lower, upper, ids)`.
type DbgBatch<'x> = (&'x [Vec<f64>], &'x [Vec<f64>], &'x [NodeId]);

/// The production search environment.
pub(super) struct HighsEnv<'a, 'b> {
    ctx: &'a NodeCtx<'b>,
    lim: &'a Limits,
    sub_mip: Option<SubMip<'a>>,
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
    stall_redive_done: bool,
    /// Open subtrees on the search's node stack at the last limit check.
    open_on_stack: usize,
    /// Node separation's cuts (globally valid Gomory cuts found at nodes), offered to
    /// later nodes' separation; with signatures for deduplication.
    pool: Vec<GomoryCut>,
    pool_sigs: std::collections::HashSet<Vec<(u32, i64)>>,
    pub sep_calls: usize,
    pub sep_lifts: usize,
    pub sub_mip_hits: usize,
    /// The queue slot of the node emplaced last (for [`SearchEnv::queue_tag`]).
    last_slot: i64,
    pub dbg: Option<Debugger<'a>>,
}

impl<'a, 'b> HighsEnv<'a, 'b> {
    /// Fire a debugger checkpoint; a `Stop` ends the search (uncertified).
    #[allow(clippy::too_many_arguments)]
    fn fire(
        &mut self,
        cp: MilpCheckpoint,
        batch: Option<DbgBatch<'_>>,
        lineage: Option<&[NodeLineage]>,
        created: Option<&[NodeLineage]>,
        best: Option<(f64, &[f64])>,
    ) {
        let Some(d) = self.dbg.as_ref() else { return };
        let inc = self.best.as_ref().map(|b| b.1);
        let gap = match inc {
            Some(v) if d.bound.is_finite() => ((v - d.bound) / v.abs().max(1.0)).max(0.0),
            _ => f64::INFINITY,
        };
        let state = MilpDebugState {
            checkpoint: cp,
            iteration: d.iter,
            // The count the result reports (`Outcome::nodes`).
            total_nodes: self.nodes.max(0) as usize,
            open_nodes: self.q.num_nodes() as usize + self.open_on_stack,
            incumbent: inc,
            bound: d.bound,
            gap,
            elapsed: d.t_start.elapsed().as_secs_f64(),
            n_vars: self.ctx.ns,
            batch_lb: batch.map(|b| b.0),
            batch_ub: batch.map(|b| b.1),
            batch_ids: batch.map(|b| b.2),
            batch_lineage: lineage,
            created,
            batch_best_bound: best.map(|b| b.0),
            batch_best_x: best.map(|b| b.1),
        };
        if matches!(d.hook.checkpoint(&state), MilpDebugControl::Stop) && self.stop.is_none() {
            self.stop = Some(Stop::Debugger);
        }
    }

    /// `AfterProcess` for a selected node: its lineage (with how it closed), the nodes
    /// created since the last report, and its LP bound and point.
    fn dbg_processed(&mut self, id: u64) {
        let Some(d) = self.dbg.as_mut() else { return };
        if !d.open.remove(&id) {
            return;
        }
        let lin = [d.lineage[&id]];
        let created: Vec<NodeLineage> = std::mem::take(&mut d.created)
            .iter()
            .map(|c| d.lineage[c])
            .collect();
        let best = d.best.take().filter(|b| b.0 == id);
        let best_ref = best.as_ref().map(|b| (b.1, b.2.as_slice()));
        self.fire(
            MilpCheckpoint::AfterProcess,
            None,
            Some(&lin),
            Some(&created),
            best_ref,
        );
        if let Some(d) = self.dbg.as_mut() {
            d.iter += 1;
        }
    }

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
            // The driver's objective lattice (HiGHS: `isIntegral` and the floored
            // upper limit): with lattice spacing g a better point is at least g
            // below the incumbent.
            self.eval.cutoff = self.ctx.tm.cutoff_for(v);
            if self.dbg.as_ref().is_some_and(|d| v < d.last_inc - 1e-9) {
                self.dbg.as_mut().unwrap().last_inc = v;
                self.fire(MilpCheckpoint::IncumbentFound, None, None, None, None);
            }
        }
    }

    /// A valid bound lifted to the objective lattice (`round_bound_up`).
    fn lift(&self, v: f64) -> f64 {
        self.ctx.tm.round_bound_up(v)
    }

    fn optlim(&self) -> f64 {
        optimality_limit(self.lim, self.best.as_ref().map_or(f64::INFINITY, |b| b.1))
            .min(self.eval.cutoff)
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
            .is_some_and(|k| self.q.num_nodes() as usize + self.open_on_stack > k)
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
        let (slot, w) = self.q.emplace_node(d, b, lb, est, depth);
        self.last_slot = slot;
        w
    }
    fn queue_tag(&mut self, ident: NodeIdent) {
        self.q.set_ident(self.last_slot, ident);
    }
    fn node_created(&mut self, ident: NodeIdent) {
        let Some(d) = self.dbg.as_mut() else { return };
        let depth = ident
            .parent
            .and_then(|p| d.lineage.get(&p))
            .map_or(0, |p| p.depth + 1);
        let branch = ident.branch.map(|c| BranchRecord {
            var: c.col as usize,
            // An upper-bound change (`x <= v`) makes the down child.
            up: !c.upper,
            value: c.val,
        });
        d.lineage.insert(
            ident.id,
            NodeLineage {
                id: NodeId(ident.id as usize),
                parent: ident.parent.map(|p| NodeId(p as usize)),
                depth,
                branch,
                status: NodeStatus::Pending,
            },
        );
        d.created.push(ident.id);
    }
    fn node_selected(&mut self, node: &NodeData) {
        let id = node.ident.id;
        let Some(d) = self.dbg.as_mut() else { return };
        // The node's relaxation as known now: a child committed by strong branching
        // was evaluated inside `branch` (its bound and the LP point are current); a
        // node about to be evaluated gets the fresh one from `evaluate`.
        if let Some(x) = self.eval.point.as_ref() {
            if node.lower_bound.is_finite() && x.len() >= self.ctx.ns {
                d.best = Some((id, node.lower_bound, x[..self.ctx.ns].to_vec()));
            }
        }
        if !d.selected.insert(id) {
            return;
        }
        d.open.insert(id);
        let ns = self.ctx.ns;
        let mut lo = self.eval.dom.col_lower[..ns].to_vec();
        let mut hi = self.eval.dom.col_upper[..ns].to_vec();
        if let Some(p) = node.ident.parent {
            if let Some((plo, phi, settled)) = d.boxes.get_mut(&p) {
                for j in 0..ns {
                    lo[j] = lo[j].max(plo[j]);
                    hi[j] = hi[j].min(phi[j]);
                }
                *settled += 1;
                if *settled >= 2 {
                    d.boxes.remove(&p);
                }
            }
        }
        d.boxes.insert(id, (lo.clone(), hi.clone(), 0));
        let lin = [d.lineage[&id]];
        self.fire(MilpCheckpoint::IterStart, None, None, None, None);
        let lb = [lo];
        let ub = [hi];
        let ids = [NodeId(id as usize)];
        self.fire(
            MilpCheckpoint::AfterSelect,
            Some((&lb, &ub, &ids)),
            Some(&lin),
            None,
            None,
        );
    }
    fn node_closed(&mut self, id: u64, fate: NodeFate) {
        let Some(d) = self.dbg.as_mut() else { return };
        let status = match fate {
            NodeFate::Branched => NodeStatus::Branched,
            NodeFate::Pruned if d.fathomed.contains(&id) => NodeStatus::Fathomed,
            NodeFate::Pruned => NodeStatus::Pruned,
        };
        if let Some(l) = d.lineage.get_mut(&id) {
            l.status = status;
        }
        if status != NodeStatus::Branched {
            d.boxes.remove(&id);
        }
        // A child pruned at creation (never selected) settles too.
        if !d.selected.contains(&id) {
            if let Some(p) = d.lineage.get(&id).and_then(|l| l.parent) {
                let p = p.0 as u64;
                if let Some(b) = d.boxes.get_mut(&p) {
                    b.2 += 1;
                    if b.2 >= 2 {
                        d.boxes.remove(&p);
                    }
                }
            }
        }
        self.dbg_processed(id);
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
    fn search_check_limits(&mut self, _nnodes: i64, open_on_stack: usize) -> bool {
        self.open_on_stack = open_on_stack;
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
        let found0 = self.eval.found.len();
        // Only this evaluation's own LP point describes this node (none when it is
        // infeasible).
        if let Some(d) = self.dbg.as_mut() {
            d.best = None;
        }
        let r = self
            .eval
            .evaluate(self.ctx, node, parent, &mut self.pc, false, limit);
        if let Some(d) = self.dbg.as_mut() {
            let id = node.ident.id;
            if self.eval.found.len() > found0 {
                d.fathomed.insert(id);
            }
            if let Some(x) = self.eval.point.as_ref() {
                if node.lower_bound.is_finite() && x.len() >= self.ctx.ns {
                    d.best = Some((id, node.lower_bound, x[..self.ctx.ns].to_vec()));
                }
            }
        }
        self.sync();
        r
    }
    fn branch(&mut self, top: &NodeData, parent: Option<&NodeData>, _depth: i32) -> BranchOutcome {
        let optlim = self.optlim();
        let q = &self.q;
        let nodes = |c: usize| (q.num_nodes_up(c), q.num_nodes_down(c));
        let deadline = self.lim.deadline;
        let limits = || deadline.is_some_and(|d| std::time::Instant::now() >= d);
        let bc = BranchCtx {
            optimality_limit: optlim,
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
            self.eval.point_full.clone(),
            self.eval.last_dual_raw.clone(),
            self.eval.basis.clone(),
            self.eval.last_box.clone(),
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
        for (k, cut) in self.pool.iter().enumerate() {
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
        let rows: Vec<GomoryCut> = pick
            .iter()
            .map(|&(_, k)| {
                let mut coeffs = self.pool[k].coeffs.clone();
                coeffs.resize(ctx.n_w, 0.0);
                GomoryCut {
                    coeffs,
                    rhs: self.pool[k].rhs,
                }
            })
            .collect();
        self.sep_calls += 1;
        let lg = node_local_gmi_bound(
            ctx,
            &fl,
            &fu,
            &x,
            &y,
            &basis,
            &rows,
            NODE_GMI_ROUNDS,
            self.nodes,
        );
        self.eval.lp_iters += lg.pivots as i64;
        if let Some(g) = lg.bound {
            self.sep_lifts += 1;
            self.eval.pending_lift = Some(g + ctx.obj_const);
        }
        for cut in lg.new_cuts {
            if self.pool_sigs.insert(cut_signature(&cut)) {
                self.pool.push(cut);
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
        // RENS / RINS (HiGHS's `runHeuristics`: RENS without an incumbent, RINS
        // with one), on the driver's schedule.
        if let (Some(sm), Some(xs)) = (self.sub_mip.as_mut(), self.eval.point.clone()) {
            let inc = self.best.as_ref().map(|(bx, bv)| (bx.as_slice(), *bv));
            let call = HeurCall::SubMip {
                x: &xs,
                inc,
                nodes: self.nodes.max(0) as usize,
            };
            if let Some((cand, v)) = sm(call) {
                self.sub_mip_hits += 1;
                self.accept(cand, v);
            }
        }
    }
    fn after_round(&mut self) -> bool {
        // The stall guard (the batched loop's `stall_time_s`, gap plan P1): past the
        // slice, stop without an incumbent; with one, run the stall re-solve once.
        if self
            .lim
            .stall_at
            .is_some_and(|t| std::time::Instant::now() >= t)
        {
            match self.best.clone() {
                None => {
                    crate::profile::incr(crate::profile::Ctr::MilpStallGuardStop);
                    self.stop = Some(Stop::Stall);
                    return true;
                }
                Some((bx, bv)) if !self.stall_redive_done => {
                    self.stall_redive_done = true;
                    if let Some(sm) = self.sub_mip.as_mut() {
                        if let Some((cand, v)) = sm(HeurCall::StallRedive { inc: (&bx, bv) }) {
                            self.accept(cand, v);
                        }
                    }
                }
                _ => {}
            }
        }
        self.q.perform_bounding(self.eval.cutoff);
        let optlim = self.optlim();
        self.q.set_optimality_limit(optlim);
        let inc = self.best.as_ref().map_or(f64::INFINITY, |b| b.1);
        let lb = if self.q.is_empty() {
            inc
        } else {
            inc.min(self.lift(self.q.best_lower_bound()))
        }
        .min(self.lift(self.eval.unresolved_floor));
        if let Some(d) = self.dbg.as_mut() {
            d.bound = d.bound.max(lb);
        }
        if gap_closed(self.lim, inc, lb) {
            self.gap_stop = true;
            return true;
        }
        self.check_limits()
    }
}

/// Search the whole tree from the driver's root working LP. `initial` is the best
/// verified point the driver already holds; `root_bound` a valid bound on the root box
/// the driver already proved (its root evaluation, node-local cuts included), with
/// `obj_const` (`-inf` for none). `None` when any LP of the search is unbounded or
/// the rows are infeasible over the global box before any search (the driver's own
/// handling then decides).
pub(super) fn run<'a>(
    ctx: &'a NodeCtx<'_>,
    lim: &'a Limits,
    initial: Option<(Vec<f64>, f64)>,
    root_bound: f64,
    sub_mip: Option<SubMip<'a>>,
    dbg: Option<Debugger<'a>>,
) -> Option<Outcome> {
    #[cfg(test)]
    RUNS.with(|r| r.set(r.get() + 1));
    let mut eval = HighsEval::new(ctx)?;
    eval.tighten_cutoff = false;
    let mut env = HighsEnv {
        ctx,
        lim,
        sub_mip,
        eval,
        pc: Pseudocost::new(
            ctx.ns,
            if lim.strong_branch {
                DEFAULT_MIN_RELIABLE
            } else {
                0
            },
        ),
        q: NodeQueue::new(ctx.ns),
        best: None,
        nseen: 0,
        nodes: 0,
        commits: 0,
        branched: 0,
        heuristic_hits: 0,
        stop: None,
        gap_stop: false,
        stall_redive_done: false,
        open_on_stack: 0,
        pool: Vec::new(),
        pool_sigs: std::collections::HashSet::new(),
        sep_calls: 0,
        sep_lifts: 0,
        sub_mip_hits: 0,
        last_slot: -1,
        dbg,
    };
    let root_id = env.dbg.as_ref().map_or(0, |d| d.root_id);
    if let Some(d) = env.dbg.as_mut() {
        d.bound = root_bound;
    }
    if let Some((x, v)) = initial {
        // `IncumbentFound` when it improves on the last one the driver reported.
        env.accept(x, v);
    }
    let (slot, _) = env
        .q
        .emplace_node(Vec::new(), Vec::new(), root_bound, root_bound, 1);
    env.q.set_ident(
        slot,
        NodeIdent {
            id: root_id,
            parent: None,
            branch: None,
        },
    );
    let mut search = Search {
        count_tree_weight: true,
        next_id: root_id + 1,
        ..Default::default()
    };
    let mut ml = MainLoop::default();
    let mut root = true;
    while !env.queue_is_empty() {
        let stop = ml.round(&mut env, &mut search, root, NO_NODE_LIMIT);
        // An unbounded LP anywhere: no bound from this search is a certificate.
        // Hand the problem back to the driver (its batched search decides
        // Unbounded), as for an unbounded root.
        if env.eval.unbounded_seen {
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
        env.lift(env.q.best_lower_bound())
    };
    let bound = inc.min(open).min(env.lift(env.eval.unresolved_floor));
    if let Some(d) = env.dbg.as_mut() {
        d.bound = d.bound.max(bound);
    }
    // The root the driver selected and the search closed without evaluating (already
    // within the gap): its `AfterProcess` is still owed.
    if env.dbg.as_ref().is_some_and(|d| d.open.contains(&root_id)) {
        env.node_closed(root_id, NodeFate::Pruned);
    }
    env.fire(MilpCheckpoint::Terminated, None, None, None, None);
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
        sep_calls: env.sep_calls,
        sep_lifts: env.sep_lifts,
        sub_mip_hits: env.sub_mip_hits,
    })
}
