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
//! So does the external node bound (`MilpBoundHook`, [`HighsEnv::apply_bound_hook`]).
//!
//! Not yet re-hosted (the driver keeps the batched search for these): lazy
//! constraints, node hooks, node cuts (the working LP is fixed during this
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

mod nodebox;
mod refcheck;

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
    /// HiGHS's heuristic effort budget (`moreHeuristicsAllowed`,
    /// [`heur_effort_enabled`]) gates the node heuristics.
    pub heur_effort: bool,
    /// This solve is a heuristic sub-MIP (`mipsolver.submip`).
    pub submip: bool,
    /// LP iterations the driver spent before the search (root LP and cut loop).
    pub iters_before: i64,
    /// `MilpOptions::max_stall_nodes` (0 = none).
    pub max_stall_nodes: usize,
    /// `MilpOptions::max_leaves` (0 = none).
    pub max_leaves: usize,
}

/// `DISCOPT_HIGHS_HEUR_EFFORT`: gate the search's node heuristics (rounding, RENS,
/// RINS) by HiGHS's `moreHeuristicsAllowed` -- heuristic LP iterations held to
/// `mip_heuristic_effort` (5 %) of the search's, a sub-MIP charged by the share of
/// columns its fixings left free (`highs_search::heur_effort`). Member of
/// `DISCOPT_HIGHS_CONFIG`; garbage refused. Entry evidence: on 23588 under the
/// configuration six sub-MIPs took ~5 of ~17 s (each runs its own root cut loop),
/// and with sub-MIPs off the configuration certifies in 12.6 s (deterministic, 1,258
/// nodes twice) against 17-22 s.
pub(super) fn heur_effort_enabled() -> bool {
    match std::env::var("DISCOPT_HIGHS_HEUR_EFFORT") {
        Err(_) => crate::highs_config::enabled(),
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" | "" => false,
            other => {
                panic!("DISCOPT_HIGHS_HEUR_EFFORT={other:?} is not recognized (expected 1 or 0)")
            }
        },
    }
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
    /// `mip_max_stall_nodes` consecutive installs at or above the cutoff.
    StallNodes,
    /// `mip_max_leaves` leaves.
    Leaves,
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
    /// `MilpBoundHook` calls, and the node bounds they raised.
    pub bound_hook_calls: usize,
    pub bound_hook_raised: usize,
    /// Step 2.1: cut rows added to and deleted from the search's LP, and the root
    /// cuts its pool started with.
    pub cut_rows_added: usize,
    pub cut_rows_deleted: usize,
    pub seeded_pool: usize,
    /// `performAging(true)` epochs on the search's LP.
    pub cut_epochs: u32,
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

/// `DISCOPT_HIGHS_SEARCH_CUT_LP` (step 2.1 of `docs/dev/highs-mip-replacement-plan.md`):
/// cuts separated at nodes join the search's own LP as rows that persist for every
/// later node, aged out when non-binding (HiGHS `mip_lp_age_limit`) and re-added from
/// the pool when violated again. Default OFF until its §5 panel; read per solve,
/// garbage refused. Entry evidence: HiGHS with cuts aged out at once goes from 102
/// nodes to 92k on `b-ball` and from a 4.9 to a 23.0 % gap on `gsvm2rl3`, where the
/// search with a fixed LP sits (`data/highs-parity/search/step2_entry.txt`).
pub(super) fn search_cut_lp_enabled() -> bool {
    match std::env::var("DISCOPT_HIGHS_SEARCH_CUT_LP") {
        Err(_) => crate::highs_config::enabled(),
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" | "" => false,
            other => {
                panic!("DISCOPT_HIGHS_SEARCH_CUT_LP={other:?} is not recognized (expected 1 or 0)")
            }
        },
    }
}

/// Cut rows the search's LP may carry at once, beyond the root's: bounds the per-LP
/// cost (the falsified 1d-5 fold doubled it with unmanaged rows).
fn search_cut_cap(base_rows: usize) -> usize {
    base_rows.max(100)
}

/// The row update of HiGHS `performAging(deleteRows)` (`HighsLpRelaxation.cpp:
/// 629-677`): a row whose slack is basic ages (without `delete_rows` only once it
/// has started to), a nonbasic row with a nonzero dual is reset. With `delete_rows`
/// an epoch passes; every `max(agelimit/2, 2)`-th epoch the limit `min(epochs,
/// agelimit)` is returned, and the basic rows older than it are due for deletion.
fn highs_age_rows(
    rows: &mut [CutLpRow],
    basic: impl Fn(usize) -> bool,
    y: &[f64],
    delete_rows: bool,
    epochs: &mut u32,
) -> Option<u32> {
    let mut due = None;
    if delete_rows {
        *epochs += 1;
        if *epochs % (CUT_LP_AGE_LIMIT >> 1).max(2) == 0 {
            due = Some(CUT_LP_AGE_LIMIT.min(*epochs));
        }
    }
    for r in rows.iter_mut() {
        if basic(r.col) {
            if delete_rows || r.age != 0 {
                r.age = r.age.saturating_add(1);
            }
        } else if y[r.row].abs() > CUT_LP_DUAL_TOL {
            r.age = 0;
        }
    }
    due
}

/// New cut rows one node's separation may add.
const SEARCH_CUTS_PER_NODE: usize = 20;

/// The search's own copy of the working LP once it carries cut rows (step 2.1): the
/// driver's root working LP plus managed cut rows `a·x - s = rhs`, `s >= 0`. Every
/// cut row's coefficients are over the base columns (the driver's working columns):
/// a cut found at a node, which may use cut slacks, is rewritten by substituting
/// `s_r = a_r·x - rhs_r` (an identity on the LP) before it is stored, so deleting a
/// row never touches another.
pub(super) struct SearchLp {
    csc: SparseCols,
    b: Vec<f64>,
    c: Vec<f64>,
    l: Vec<f64>,
    u: Vec<f64>,
    is_int_full: Vec<bool>,
    m: usize,
    n: usize,
    problem: ScaledProblem,
    slack_l: Vec<f64>,
    slack_u: Vec<f64>,
    rows: Vec<CutLpRow>,
    /// HiGHS `HighsLpRelaxation::epochs`: `performAging(true)` calls so far.
    epochs: u32,
    /// HiGHS `lastAgeCall`: the LP iteration count at the last ageing.
    last_age_iters: i64,
    /// Separation appended rows; the next dual-feasible solve is the resolve after
    /// which HiGHS's `separationRound` calls `performAging(true)`.
    epoch_due: bool,
    /// An epoch fell due for deletion (`min(epochs, agelimit)`); `after_round`
    /// deletes the rows above it whose slack is still basic.
    delete_limit: Option<u32>,
}

impl SearchLp {
    fn from_base(base: &NodeCtx<'_>) -> Self {
        let csc = base.csc_rc.clone();
        let problem = ScaledProblem::new(csc.clone(), base.m_w, base.n_w, base.c_w, base.b_w);
        SearchLp {
            csc,
            b: base.b_w.to_vec(),
            c: base.c_w.to_vec(),
            l: base.l_w.to_vec(),
            u: base.u_w.to_vec(),
            is_int_full: base.is_int_full.to_vec(),
            m: base.m_w,
            n: base.n_w,
            problem,
            slack_l: base.slack_l.to_vec(),
            slack_u: base.slack_u.to_vec(),
            rows: Vec::new(),
            epochs: 0,
            last_age_iters: -1,
            epoch_due: false,
            delete_limit: None,
        }
    }

    fn rebuild(&mut self, ns: usize) {
        self.problem = ScaledProblem::new(self.csc.clone(), self.m, self.n, &self.c, &self.b);
        self.slack_l = self.l[ns..].to_vec();
        self.slack_u = self.u[ns..].to_vec();
    }

    /// `cut` (over the current columns) over the base columns `0..n0`.
    fn to_base(&self, cut: &GomoryCut, n0: usize) -> GomoryCut {
        let mut out = GomoryCut {
            coeffs: cut.coeffs[..cut.coeffs.len().min(n0)].to_vec(),
            rhs: cut.rhs,
        };
        out.coeffs.resize(n0, 0.0);
        for r in &self.rows {
            let g = cut.coeffs.get(r.col).copied().unwrap_or(0.0);
            if g == 0.0 {
                continue;
            }
            for (j, &v) in r.cut.coeffs.iter().enumerate() {
                if v != 0.0 {
                    out.coeffs[j] += g * v;
                }
            }
            out.rhs += g * r.cut.rhs;
        }
        out
    }
}

/// The working LP the search's components see: the driver's, or the search's own once
/// it carries cut rows.
fn make_ctx<'x>(base: &NodeCtx<'x>, lp: &'x Option<SearchLp>) -> NodeCtx<'x> {
    let Some(lp) = lp else { return *base };
    NodeCtx {
        b_w: &lp.b,
        c_w: &lp.c,
        l_w: &lp.l,
        u_w: &lp.u,
        scaling: lp.problem.scaling(),
        sc: lp.problem.solve_c(),
        sb: lp.problem.solve_b(),
        csc: lp.problem.solve_matrix(),
        csc_rc: lp.problem.original_matrix(),
        problem: &lp.problem,
        slack_l: &lp.slack_l,
        slack_u: &lp.slack_u,
        is_int_full: &lp.is_int_full,
        m_w: lp.m,
        n_w: lp.n,
        ..*base
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
    /// The external node bound (`MilpBoundHook`): distinct nodes offered so far, the
    /// last node's value (a node is re-evaluated within a round; the hook is asked
    /// once), and the call/raise counts.
    hook_nodes: usize,
    hook_last: Option<(u64, Option<f64>)>,
    pub bound_hook_calls: usize,
    pub bound_hook_raised: usize,
    /// Step 2.1 (`DISCOPT_HIGHS_SEARCH_CUT_LP`): the search's own LP once it carries
    /// cut rows, and the signatures of the cuts currently in it.
    cut_lp: bool,
    lp: Option<SearchLp>,
    /// The last evaluation stored duals of the current LP (`last_dual`).
    dual_fresh: bool,
    in_lp: std::collections::HashSet<Vec<(u32, i64)>>,
    pub cut_rows_added: usize,
    pub cut_rows_deleted: usize,
    /// Root cuts the pool started with (step 2.1).
    pub seeded_pool: usize,
    /// `DISCOPT_HIGHS_SEARCH_CHECK_POINT` (debug instrument, [`refcheck`]).
    refp: Option<refcheck::RefCheck>,
    /// Tree weight the queue pruned (`emplaceNode`, `performBounding`): with the
    /// search's own `treeweight`, HiGHS's `pruned_treeweight`.
    queue_pruned: f64,
    /// The search's `(treeweight, nleaves, nodes)` at the last node processed.
    progress: (f64, i64, i64),
    /// Consecutive stall installs (`numStallNodes`).
    stall_nodes: usize,
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

    /// `DISCOPT_HIGHS_SEARCH_CHECK_POINT`: whether the local domain holds the point.
    fn ref_in_domain(&self) -> Option<bool> {
        let rc = self.refp.as_ref()?;
        let ns = self.ctx.ns;
        Some(
            !self.eval.dom.infeasible()
                && rc.contains(
                    &self.eval.dom.col_lower[..ns],
                    &self.eval.dom.col_upper[..ns],
                ),
        )
    }

    /// A popped queue slot: forget it if it held the point.
    fn ref_popped(&mut self, how: &str, slot: i64) {
        if let Some(rc) = self.refp.as_mut() {
            if rc.slots.remove(&slot) {
                eprintln!(
                    "HIGHS_SEARCH_TRACE popped {how} slot={slot} n={}",
                    self.nodes
                );
            }
        }
    }

    /// Every tracked queue slot holding the point must still be queued.
    fn ref_slots_kept(&mut self, kind: &str) {
        let Some(rc) = self.refp.as_mut() else { return };
        let gone: Vec<i64> = rc
            .slots
            .iter()
            .copied()
            .filter(|&s| !self.q.is_linked(s))
            .collect();
        for s in gone {
            rc.slots.remove(&s);
            let lb = self.q.node(s).lower_bound;
            rc.bound(
                kind,
                self.nodes,
                f64::INFINITY,
                self.eval.cutoff,
                &format!(
                    "slot={s} lb={lb:.12} optlim={:.12}",
                    self.q.optimality_limit()
                ),
            );
        }
    }

    /// The domain held the point before a reduction (`was_in`): it must still.
    fn ref_box_kept(&mut self, kind: &str, was_in: Option<bool>) {
        if was_in != Some(true) {
            return;
        }
        let still = self.ref_in_domain() == Some(true);
        let (nodes, cutoff) = (self.nodes, self.eval.cutoff);
        let infeasible = self.eval.dom.infeasible();
        let Some(rc) = self.refp.as_mut() else { return };
        if !still && rc.active(cutoff) {
            rc.bound(
                kind,
                nodes,
                f64::INFINITY,
                cutoff,
                &format!("infeasible={infeasible}"),
            );
        }
    }

    /// Append `cuts` (over the current columns) to the search's LP as persistent rows.
    fn append_cut_rows(&mut self, cuts: &[GomoryCut]) {
        if cuts.is_empty() {
            return;
        }
        let base = self.ctx;
        let n0 = base.n_w;
        let lp = self.lp.get_or_insert_with(|| SearchLp::from_base(base));
        let room = search_cut_cap(base.m_w).saturating_sub(lp.rows.len());
        let mut add: Vec<GomoryCut> = Vec::new();
        for cut in cuts {
            if add.len() >= room {
                break;
            }
            let c = lp.to_base(cut, n0);
            if let Some(rc) = self.refp.as_mut() {
                rc.cut("lp-row", self.nodes, &c, &rc.x.clone());
            }
            if self.in_lp.insert(cut_signature(&c)) {
                add.push(c);
            }
        }
        if add.is_empty() {
            return;
        }
        let (m0, n0w) = (lp.m, lp.n);
        let (csc, m, n) = augment_csc_with_cuts(
            &lp.csc,
            &mut lp.b,
            &mut lp.c,
            &mut lp.l,
            &mut lp.u,
            &mut lp.is_int_full,
            lp.m,
            lp.n,
            &add,
        );
        lp.csc = csc;
        lp.m = m;
        lp.n = n;
        for (i, cut) in add.into_iter().enumerate() {
            lp.rows.push(CutLpRow {
                row: m0 + i,
                col: n0w + i,
                age: 0,
                cut,
                relaxed: false,
            });
        }
        lp.rebuild(base.ns);
        if let Some(rc) = self.refp.as_mut() {
            rc.lp("append", self.nodes, lp);
        }
        self.cut_rows_added += m - m0;
        // The basis takes the new slacks as basic (block-triangular, nonsingular);
        // reduced costs of basic columns are zero. The other caches describe the
        // old LP: the next evaluation (the dive's, right after separation) rebuilds
        // them.
        self.eval.basis = self.eval.basis.take().map(|b| extend_basis(b, n));
        if let Some(d) = self.eval.basis_d.as_mut() {
            d.resize(n, 0.0);
        }
        self.clear_lp_caches();
    }

    fn clear_lp_caches(&mut self) {
        self.eval.point_full = None;
        self.eval.last_dual = None;
        self.dual_fresh = false;
        self.eval.last_dual_raw = None;
        self.eval.last_box = None;
    }

    /// HiGHS `HighsLpRelaxation::performAging(deleteRows)` (`HighsLpRelaxation.cpp:
    /// 629-677`) on the last solve: only with valid duals (`dual_fresh`); a row whose
    /// slack is basic ages (without `delete_rows`, only once it has started to), a
    /// nonbasic row with a nonzero dual is reset. With `delete_rows` an epoch passes,
    /// and every `max(agelimit/2, 2)`-th epoch the rows older than
    /// `min(epochs, agelimit)` are due for deletion. Without `delete_rows` the call is
    /// skipped when no LP iteration happened since the last one (`lastAgeCall`).
    fn age_cut_rows(&mut self, delete_rows: bool) {
        if !self.dual_fresh {
            return;
        }
        let iters = self.eval.lp_iters;
        let (Some(lp), Some(y), Some(basis)) = (
            self.lp.as_mut(),
            self.eval.last_dual.as_ref(),
            self.eval.basis.as_ref(),
        ) else {
            return;
        };
        if y.len() != lp.m {
            return;
        }
        if !delete_rows && lp.last_age_iters == iters {
            return;
        }
        lp.last_age_iters = iters;
        let due = highs_age_rows(
            &mut lp.rows,
            |col| basis.col_status.get(col).is_some_and(|&st| st == BASIC),
            y,
            delete_rows,
            &mut lp.epochs,
        );
        if let Some(limit) = due {
            lp.delete_limit = Some(lp.delete_limit.map_or(limit, |d| d.min(limit)));
        }
    }

    /// Delete the cut rows an ageing epoch found due (`delete_limit`): older than the
    /// limit and with the slack basic in the LP's basis (dropping a basic slack with
    /// its row keeps the basis a basis). HiGHS deletes inside `performAging`; here it
    /// waits for the round's end, because the node's point and duals are still in use
    /// after its evaluation, and the conditions are re-checked on the current basis.
    /// The cuts stay in the pool and are re-added when violated again.
    fn delete_aged_cut_rows(&mut self) {
        let Some(lp) = self.lp.as_mut() else { return };
        let Some(limit) = lp.delete_limit.take() else {
            return;
        };
        let Some(basis) = self.eval.basis.as_ref() else {
            return;
        };
        let (mut gone, mut kept): (Vec<CutLpRow>, Vec<CutLpRow>) =
            lp.rows.drain(..).partition(|r| {
                r.age > limit && basis.col_status.get(r.col).is_some_and(|&s| s == BASIC)
            });
        if gone.is_empty() {
            lp.rows = kept;
            return;
        }
        for r in &gone {
            self.in_lp.remove(&cut_signature(&r.cut));
        }
        let d = delete_cut_rows(&lp.csc, lp.m, lp.n, &lp.b, &mut gone, &mut kept);
        let col_map = d.col_map.clone();
        let keep = |j: usize| col_map[j] != usize::MAX;
        lp.c =
            lp.c.iter()
                .enumerate()
                .filter(|(j, _)| keep(*j))
                .map(|(_, &x)| x)
                .collect();
        lp.l =
            lp.l.iter()
                .enumerate()
                .filter(|(j, _)| keep(*j))
                .map(|(_, &x)| x)
                .collect();
        lp.u =
            lp.u.iter()
                .enumerate()
                .filter(|(j, _)| keep(*j))
                .map(|(_, &x)| x)
                .collect();
        lp.is_int_full = lp
            .is_int_full
            .iter()
            .enumerate()
            .filter(|(j, _)| keep(*j))
            .map(|(_, &x)| x)
            .collect();
        lp.csc = d.csc;
        lp.b = d.b;
        lp.m -= gone.len();
        lp.n -= gone.len();
        lp.rows = kept;
        lp.rebuild(self.ctx.ns);
        if let Some(rc) = self.refp.as_mut() {
            rc.lp("delete", self.nodes, lp);
        }
        self.cut_rows_deleted += gone.len();
        self.eval.basis = self
            .eval
            .basis
            .take()
            .and_then(|b| remap_basis_after_delete(&b, &col_map));
        self.eval.basis_d = self.eval.basis_d.take().map(|v| {
            v.iter()
                .enumerate()
                .filter(|(j, _)| keep(*j))
                .map(|(_, &x)| x)
                .collect()
        });
        self.clear_lp_caches();
    }

    /// The external node bound (`MilpBoundHook`, `Model.solve(lagrangian_bound=True)`),
    /// as the batched search applies it: on a node whose LP bound is real and whose
    /// point is fractional (`Open`), every `bound_hook_every`-th node, over the node's
    /// structural box (its local domain); the node keeps `max(LP bound, hook bound)`.
    /// A raised bound past the cutoff prunes the node (`BoundExceeding`); one past the
    /// optimality limit makes the control queue it as suboptimal. The hook must return
    /// a valid bound over the box, so a valid hook can only tighten.
    fn apply_bound_hook(&mut self, node: &mut NodeData, r: SearchResult) -> SearchResult {
        let Some(h) = self.ctx.opts.bound_hook.as_ref() else {
            return r;
        };
        let lb0 = node.lower_bound;
        if r != SearchResult::Open || !lb0.is_finite() || lb0 >= INFEAS_SENTINEL {
            return r;
        }
        let id = node.ident.id;
        let v = match self.hook_last {
            Some((k, v)) if k == id => v,
            _ => {
                let due = self.hook_nodes % self.ctx.opts.bound_hook_every.max(1) == 0;
                self.hook_nodes += 1;
                let v = if due {
                    self.bound_hook_calls += 1;
                    let ns = self.ctx.ns;
                    h.node_bound(
                        &self.eval.dom.col_lower[..ns],
                        &self.eval.dom.col_upper[..ns],
                    )
                    .filter(|v| v.is_finite() && *v < INFEAS_SENTINEL)
                } else {
                    None
                };
                self.hook_last = Some((id, v));
                if v.is_some_and(|v| v > lb0) {
                    self.bound_hook_raised += 1;
                }
                v
            }
        };
        match v {
            Some(v) if v > lb0 => {
                node.lower_bound = v;
                if v > self.eval.cutoff {
                    SearchResult::BoundExceeding
                } else {
                    r
                }
            }
            _ => r,
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

    /// The search's dual bound: `min(incumbent, best queued bound, unresolved floor)`,
    /// HiGHS's `min(upper_bound, nodequeue.getBestLowerBound())`. The queued bound
    /// covers the suboptimal nodes too (bound above the optimality limit, below the
    /// cutoff): they are not searched, but the optimum may lie in one of them, so the
    /// bound must not exceed theirs. `NodeQueue::is_empty` counts active nodes only and
    /// is no test for this: with only suboptimal nodes left it is true, and gating the
    /// queued bound on it reported the incumbent itself as the bound (neos17 under
    /// `DISCOPT_HIGHS_SEARCH_CUT_LP=1 DISCOPT_HIGHS_SEPA_NODE=1`: bound 0.150069 above
    /// the optimum 0.1500026). `best_lower_bound` is `+inf` on an empty queue.
    fn dual_bound(&self, inc: f64) -> f64 {
        inc.min(self.lift(self.q.best_lower_bound()))
            .min(self.lift(self.eval.unresolved_floor))
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
        let ns = self.ctx.ns;
        let mut ref_q = false;
        if let Some(rc) = self.refp.as_mut() {
            let (lo, hi) = refcheck::stack_box(&self.ctx.l_w[..ns], &self.ctx.u_w[..ns], &d);
            rc.queue_checked += 1;
            ref_q = rc.contains(&lo, &hi);
            if ref_q {
                rc.queue_containing += 1;
                eprintln!(
                    "HIGHS_SEARCH_TRACE queue n={} lb={lb:.12} est={est:.12} depth={depth} \
                     domchg={}",
                    self.nodes,
                    d.len()
                );
                let cutoff = self.eval.cutoff;
                if rc.active(cutoff) {
                    rc.bound("queue", self.nodes, lb, cutoff, &format!("depth={depth}"));
                }
            }
        }
        let (slot, w) = self.q.emplace_node(d, b, lb, est, depth);
        self.queue_pruned += w;
        if ref_q {
            eprintln!(
                "HIGHS_SEARCH_TRACE queued slot={slot} linked={} optlim={:.12}",
                self.q.is_linked(slot),
                self.q.optimality_limit()
            );
            if let Some(rc) = self.refp.as_mut() {
                rc.slots.insert(slot);
            }
        }
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
        let (slot, n) = self.q.pop_best_node();
        self.ref_popped("best", slot);
        n
    }
    fn queue_pop_best_bound(&mut self) -> OpenNode {
        let (slot, n) = self.q.pop_best_bound_node();
        self.ref_popped("best-bound", slot);
        n
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
        if let Some(rc) = self.refp.as_ref() {
            let ns = self.ctx.ns;
            let (lo, hi) =
                refcheck::stack_box(&self.ctx.l_w[..ns], &self.ctx.u_w[..ns], &node.domchgstack);
            let queued_in = rc.contains(&lo, &hi);
            let dom_in = self.ref_in_domain() == Some(true);
            if queued_in || dom_in {
                eprintln!(
                    "HIGHS_SEARCH_TRACE install n={} id={} lb={:.12} depth={} queued_in={queued_in} \
                     dom_in={dom_in} infeasible={}",
                    self.nodes,
                    node.ident.id,
                    node.lower_bound,
                    node.depth,
                    self.eval.dom.infeasible()
                );
                if queued_in && !dom_in {
                    let bad: Vec<String> = (0..ns)
                        .filter(|&j| {
                            let v = rc.x[j];
                            v < self.eval.dom.col_lower[j] - 1e-9
                                || v > self.eval.dom.col_upper[j] + 1e-9
                        })
                        .take(8)
                        .map(|j| {
                            format!(
                                "col {j} x={} dom=[{}, {}] queued=[{}, {}] global=[{}, {}]",
                                rc.x[j],
                                self.eval.dom.col_lower[j],
                                self.eval.dom.col_upper[j],
                                lo[j],
                                hi[j],
                                self.ctx.l_w[j],
                                self.ctx.u_w[j]
                            )
                        })
                        .collect();
                    eprintln!("HIGHS_SEARCH_INVALID kind=install-box {bad:?}");
                }
            }
        }
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
        let was_in = self.ref_in_domain();
        self.eval.dom.propagate();
        self.ref_box_kept("prop-backtrack", was_in);
        self.eval.dom.infeasible()
    }
    fn domain_try_sibling(&mut self, d: QDomainChange, skip_propagate: bool) -> bool {
        self.eval.dom.change_bound(from_q(&d), Reason::Branching);
        let was_in = self.ref_in_domain();
        if was_in == Some(true) {
            eprintln!(
                "HIGHS_SEARCH_TRACE sibling n={} {d:?} skip_propagate={skip_propagate}",
                self.nodes
            );
        }
        if !skip_propagate && !self.eval.dom.infeasible() {
            self.eval.dom.propagate();
        }
        self.ref_box_kept("prop-sibling", was_in);
        self.eval.dom.infeasible()
    }
    fn domain_undo_sibling(&mut self) {
        self.eval.dom.backtrack();
        self.eval.dom.clear_changed_cols();
    }
    fn domain_propagate_for_queue(&mut self) -> bool {
        let n = self.eval.dom.changed_cols().len();
        let was_in = self.ref_in_domain();
        self.eval.dom.propagate();
        self.ref_box_kept("prop-queue", was_in);
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
        let ctx = make_ctx(self.ctx, &self.lp);
        let ns = self.ctx.ns;
        let ref_in = self.refp.as_ref().map(|rc| {
            rc.contains(
                &self.eval.dom.col_lower[..ns],
                &self.eval.dom.col_upper[..ns],
            )
        });
        let lb_in = node.lower_bound;
        let dual_epoch0 = self.eval.dual_epoch;
        let r = self
            .eval
            .evaluate(&ctx, node, parent, &mut self.pc, false, limit);
        self.dual_fresh = self.eval.dual_epoch != dual_epoch0;
        if !matches!(
            r,
            SearchResult::DomainInfeasible | SearchResult::LpInfeasible
        ) {
            self.ref_box_kept("eval-box", ref_in);
        }
        if let (Some(rc), Some(inside)) = (self.refp.as_mut(), ref_in) {
            rc.nodes_checked += 1;
            if inside {
                rc.nodes_containing += 1;
                let cutoff = self.eval.cutoff;
                eprintln!(
                    "HIGHS_SEARCH_TRACE eval n={} id={} lb_in={lb_in:.12} r={r:?} lb={:.12} \
                     cutoff={cutoff:.12} rows={}",
                    self.nodes,
                    node.ident.id,
                    node.lower_bound,
                    self.lp.as_ref().map_or(0, |l| l.rows.len())
                );
                if rc.active(cutoff) {
                    let what = format!(
                        "result={r:?} lb_in={lb_in:.12} rows={}",
                        self.lp.as_ref().map_or(0, |l| l.rows.len())
                    );
                    let b = match r {
                        SearchResult::DomainInfeasible | SearchResult::LpInfeasible => {
                            f64::INFINITY
                        }
                        _ => node.lower_bound,
                    };
                    rc.bound("node", self.nodes, b, cutoff, &what);
                }
            }
        }
        // HiGHS ages after every node LP (`HighsSearch.cpp:955-958`); the first solve
        // after a separation round is that round's resolve, which ages with deletion
        // (`HighsSeparation.cpp:122-124`). A solve without valid duals ends the round
        // without an epoch, as `performAging` returns early there.
        let epoch = self
            .lp
            .as_mut()
            .is_some_and(|lp| std::mem::take(&mut lp.epoch_due));
        self.age_cut_rows(epoch);
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
        let r = self.apply_bound_hook(node, r);
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
        let ctx = make_ctx(self.ctx, &self.lp);
        let ns = self.ctx.ns;
        let ref_box = self.refp.as_ref().map(|_| {
            (
                self.eval.dom.col_lower[..ns].to_vec(),
                self.eval.dom.col_upper[..ns].to_vec(),
            )
        });
        let out = self.eval.branch(&ctx, &mut self.pc, top, parent, &bc);
        if let (Some(rc), Some((mut lo, mut hi))) = (self.refp.as_mut(), ref_box) {
            if rc.contains(&lo, &hi) && rc.active(self.eval.cutoff) {
                let cutoff = self.eval.cutoff;
                let side = |d: &QDomainChange, x: &[f64]| {
                    let v = x[d.col as usize];
                    if d.upper {
                        v <= d.val + 1e-9
                    } else {
                        v >= d.val - 1e-9
                    }
                };
                eprintln!(
                    "HIGHS_SEARCH_TRACE branch n={} top={} r={:?} steps={:?} branched={:?} \
                     closed={:?}",
                    self.nodes, top.ident.id, out.result, out.steps, out.branched, out.closed
                );
                let mut inside = true;
                for s in &out.steps {
                    match s {
                        BranchStep::Commit { decision, .. } => {
                            if !side(decision, &rc.x) {
                                rc.bound(
                                    "sb-commit",
                                    self.nodes,
                                    f64::INFINITY,
                                    cutoff,
                                    &format!("{decision:?}"),
                                );
                                inside = false;
                                break;
                            }
                            let j = decision.col as usize;
                            if decision.upper {
                                hi[j] = hi[j].min(decision.val);
                            } else {
                                lo[j] = lo[j].max(decision.val);
                            }
                        }
                        BranchStep::Update { lower_bound, .. } => {
                            rc.bound("sb-update", self.nodes, *lower_bound, cutoff, "");
                        }
                    }
                }
                if inside {
                    if let Some((d, _, other_lb, lb, _, child_lb, _)) = out.branched.as_ref() {
                        rc.bound("branch-node", self.nodes, *lb, cutoff, "");
                        let (b, which) = if side(d, &rc.x) {
                            (*child_lb, "branch-child")
                        } else {
                            (*other_lb, "branch-other")
                        };
                        rc.bound(which, self.nodes, b, cutoff, &format!("{d:?}"));
                    } else if let Some((lb, ..)) = out.closed.as_ref() {
                        let b = match out.result {
                            SearchResult::DomainInfeasible | SearchResult::LpInfeasible => {
                                f64::INFINITY
                            }
                            _ => *lb,
                        };
                        rc.bound(
                            "branch-closed",
                            self.nodes,
                            b,
                            cutoff,
                            &format!("{:?}", out.result),
                        );
                    }
                }
            }
        }
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
        // re-evaluates it. Under `DISCOPT_HIGHS_SEARCH_CUT_LP` (step 2.1) the cuts also
        // join the search's LP as persistent rows, as HiGHS adds its cuts to the shared
        // LP; otherwise they stay node-local and in the pool.
        let ctx = make_ctx(self.ctx, &self.lp);
        let ctx = &ctx;
        if !ctx.node_gmi {
            return false;
        }
        let (Some(x), Some(y), Some(basis), Some((fl, fu))) = (
            self.eval.point_full.clone(),
            self.eval.last_dual_raw.clone(),
            self.eval.basis.clone(),
            self.eval.last_box.clone(),
        ) else {
            // HiGHS `separate`'s else branch: no optimal LP, just ageing (a no-op
            // without valid duals).
            self.age_cut_rows(true);
            return false;
        };
        let integral = (0..ctx.ns).all(|j| !ctx.is_int[j] || frac(x[j]) <= INT_TOL);
        if integral {
            // The else branch again: no fractional integers, just ageing.
            self.age_cut_rows(true);
            return false;
        }
        if self
            .lim
            .deadline
            .is_some_and(|d| std::time::Instant::now() >= d)
        {
            return false;
        }
        let mut pick: Vec<(f64, usize)> = Vec::new();
        for (k, cut) in self.pool.iter().enumerate() {
            if self.cut_lp && self.in_lp.contains(&cut_signature(cut)) {
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
        let ref_cur = self.refp.as_ref().map(|rc| rc.current(self.lp.as_ref()));
        if let (Some(rc), Some(xc)) = (self.refp.as_mut(), ref_cur.as_ref()) {
            for r in &rows {
                rc.cut("pool-pick", self.nodes, r, xc);
            }
        }
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
            self.best.as_ref().map(|b| b.1),
        );
        self.eval.lp_iters += lg.pivots as i64;
        if let (Some(rc), Some(xc)) = (self.refp.as_mut(), ref_cur.as_ref()) {
            for c in &lg.new_cuts {
                rc.cut("node-new", self.nodes, c, xc);
            }
            if let Some(g) = lg.bound {
                if rc.contains(&fl[..ctx.ns], &fu[..ctx.ns]) {
                    rc.bound(
                        "sep-lift",
                        self.nodes,
                        g + ctx.obj_const,
                        self.eval.cutoff,
                        &format!("pool_rows={} new_cuts={}", rows.len(), lg.new_cuts.len()),
                    );
                }
            }
        }
        if let Some(g) = lg.bound {
            self.sep_lifts += 1;
            self.eval.pending_lift = Some(g + ctx.obj_const);
        }
        if !self.cut_lp {
            for cut in lg.new_cuts {
                if self.pool_sigs.insert(cut_signature(&cut)) {
                    self.pool.push(cut);
                }
            }
            return false;
        }
        // Step 2.1: the pool holds cuts over the base columns; the LP gets the pool
        // cuts this node violates and its new cuts, as persistent rows.
        let n0 = self.ctx.n_w;
        let new_base: Vec<GomoryCut> = match self.lp.as_ref() {
            Some(lp) => lg.new_cuts.iter().map(|c| lp.to_base(c, n0)).collect(),
            None => lg.new_cuts,
        };
        if let Some(rc) = self.refp.as_mut() {
            let xb = rc.x.clone();
            for c in &new_base {
                rc.cut("node-new-base", self.nodes, c, &xb);
            }
        }
        let mut add: Vec<GomoryCut> = pick.iter().map(|&(_, k)| self.pool[k].clone()).collect();
        for cut in new_base {
            if self.pool_sigs.insert(cut_signature(&cut)) {
                self.pool.push(cut.clone());
            }
            add.push(cut);
        }
        add.truncate(SEARCH_CUTS_PER_NODE);
        let rows0 = self.cut_rows_added;
        self.append_cut_rows(&add);
        if self.cut_rows_added > rows0 {
            // `separationRound`: cuts went into the LP, so its resolve ages with
            // deletion (`HighsSeparation.cpp:122-124`).
            if let Some(lp) = self.lp.as_mut() {
                lp.epoch_due = true;
            }
        }
        false
    }
    fn heuristics_allowed(&mut self) -> bool {
        if !self.lim.heuristics {
            return false;
        }
        if !self.lim.heur_effort {
            return true;
        }
        let heur = super::heur_lp_iters_here();
        let st = crate::bnb::highs_search::heur_effort::EffortState {
            submip: self.lim.submip,
            heuristic_lp_iterations: heur,
            total_lp_iterations: self.lim.iters_before
                + self.eval.lp_iters
                + self.eval.sb_iters
                + heur,
            sb_lp_iterations: self.eval.sb_iters,
            pruned_treeweight: self.progress.0 + self.queue_pruned,
            num_leaves: self.progress.1,
            num_nodes: self.progress.2,
        };
        let ok = crate::bnb::highs_search::heur_effort::more_heuristics_allowed(
            &st,
            crate::bnb::highs_search::heur_effort::HEURISTIC_EFFORT,
        );
        crate::profile::incr(if ok {
            crate::profile::Ctr::HeurEffortAllowed
        } else {
            crate::profile::Ctr::HeurEffortRefused
        });
        ok
    }
    fn note_search_progress(&mut self, treeweight: f64, nleaves: i64, nnodes: i64) {
        self.progress = (treeweight, nleaves, nnodes);
        if self.lim.max_leaves > 0 && nleaves >= self.lim.max_leaves as i64 && self.stop.is_none() {
            self.stop = Some(Stop::Leaves);
        }
    }
    fn note_installed_estimate(&mut self, estimate: f64) -> bool {
        if self.lim.max_stall_nodes == 0 {
            return false;
        }
        // `HighsMipSolver.cpp:641-650`: an installed node whose estimate cannot beat the
        // incumbent is a stall node; the counter resets on any other.
        if estimate >= self.eval.cutoff {
            self.stall_nodes += 1;
            if self.stall_nodes >= self.lim.max_stall_nodes && self.stop.is_none() {
                crate::profile::incr(crate::profile::Ctr::SearchStallNodeStops);
                self.stop = Some(Stop::StallNodes);
                return true;
            }
        } else {
            self.stall_nodes = 0;
        }
        false
    }
    fn run_heuristics(&mut self) {
        // discopt's rounding of the node LP point (HiGHS: randomized rounding, RENS,
        // RINS -- stage 6).
        let Some(x) = self.eval.point_full.clone() else {
            return;
        };
        // The driver's working LP (cut rows are implied by it); the point's prefix over
        // its columns (cut slacks come after them).
        let ctx = self.ctx;
        let x = &x[..ctx.n_w.min(x.len())];
        if let Some((cand, v)) = try_rounding_csc(
            x,
            ctx.ns,
            ctx.heur_is_int,
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
        self.delete_aged_cut_rows();
        self.ref_slots_kept("queue-lost-before-bounding");
        self.queue_pruned += self.q.perform_bounding(self.eval.cutoff);
        self.ref_slots_kept("queue-bounding");
        let optlim = self.optlim();
        self.q.set_optimality_limit(optlim);
        let inc = self.best.as_ref().map_or(f64::INFINITY, |b| b.1);
        let lb = self.dual_bound(inc);
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

impl<'a, 'b> HighsEnv<'a, 'b> {
    /// A search environment over the driver's root working LP, before any node.
    fn new(
        ctx: &'a NodeCtx<'b>,
        lim: &'a Limits,
        sub_mip: Option<SubMip<'a>>,
        dbg: Option<Debugger<'a>>,
        refp: Option<refcheck::RefCheck>,
    ) -> Option<Self> {
        let mut eval = HighsEval::new(ctx)?;
        eval.tighten_cutoff = false;
        Some(HighsEnv {
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
            hook_nodes: 0,
            hook_last: None,
            bound_hook_calls: 0,
            bound_hook_raised: 0,
            cut_lp: search_cut_lp_enabled(),
            dual_fresh: false,
            lp: None,
            in_lp: std::collections::HashSet::new(),
            cut_rows_added: 0,
            cut_rows_deleted: 0,
            seeded_pool: 0,
            refp,
            queue_pruned: 0.0,
            progress: (0.0, 0, 0),
            stall_nodes: 0,
        })
    }

    /// Step 2.1: HiGHS's search starts with the root's whole cut pool (every separated
    /// cut, and the ones root cleanup dropped from the LP); with persistent cuts on,
    /// so does this one (the pool feeds the LP when a node violates a cut).
    fn seed_pool(&mut self, seed_pool: Vec<GomoryCut>) {
        if self.cut_lp {
            for cut in self.ctx.gmi_pool.iter().cloned().chain(seed_pool) {
                if self.pool_sigs.insert(cut_signature(&cut)) {
                    self.pool.push(cut);
                }
            }
            self.seeded_pool = self.pool.len();
        }
    }

    /// Rows of the LP the search's components see (model rows, root cuts, and the
    /// search's own cut rows).
    fn lp_rows(&self) -> usize {
        self.lp.as_ref().map_or(self.ctx.m_w, |l| l.m)
    }
}

/// Search the whole tree from the driver's root working LP. `initial` is the best
/// verified point the driver already holds; `root_bound` a valid bound on the root box
/// the driver already proved (its root evaluation, node-local cuts included), with
/// `obj_const` (`-inf` for none). `None` when any LP of the search is unbounded or
/// the rows are infeasible over the global box before any search (the driver's own
/// handling then decides).
#[allow(clippy::too_many_arguments)]
pub(super) fn run<'a>(
    ctx: &'a NodeCtx<'_>,
    lim: &'a Limits,
    initial: Option<(Vec<f64>, f64)>,
    root_bound: f64,
    sub_mip: Option<SubMip<'a>>,
    dbg: Option<Debugger<'a>>,
    seed_pool: Vec<GomoryCut>,
    top_level: bool,
) -> Option<Outcome> {
    #[cfg(test)]
    RUNS.with(|r| r.set(r.get() + 1));
    // `DISCOPT_NODEBOX_FILE` (diagnostic, top-level solve only): evaluate external node
    // boxes on environments of their own, built exactly as this search's, before the
    // search starts; the search below is untouched by it.
    if top_level {
        nodebox::harness(ctx, lim, initial.as_ref(), &seed_pool);
    }
    let mut env = HighsEnv::new(ctx, lim, sub_mip, dbg, refcheck::RefCheck::load(ctx))?;
    if let Some(rc) = env.refp.as_mut() {
        let xb = rc.x.clone();
        for cut in ctx.gmi_pool.iter() {
            rc.cut("seed-gmi", 0, cut, &xb);
        }
        for cut in seed_pool.iter() {
            rc.cut("seed-highs", 0, cut, &xb);
        }
        rc.bound("root", 0, root_bound, f64::INFINITY, "");
    }
    env.seed_pool(seed_pool);
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
    let bound = env.dual_bound(inc);
    if let Some(rc) = env.refp.as_ref() {
        rc.summary(bound);
    }
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
        bound_hook_calls: env.bound_hook_calls,
        bound_hook_raised: env.bound_hook_raised,
        cut_rows_added: env.cut_rows_added,
        cut_rows_deleted: env.cut_rows_deleted,
        cut_epochs: env.lp.as_ref().map_or(0, |l| l.epochs),
        seeded_pool: env.seeded_pool,
    })
}

#[cfg(test)]
mod ageing_tests {
    use super::*;

    fn row(i: usize, age: u32) -> CutLpRow {
        CutLpRow {
            row: i,
            col: 10 + i,
            age,
            cut: GomoryCut {
                coeffs: vec![1.0],
                rhs: 0.0,
            },
            relaxed: false,
        }
    }

    /// HiGHS `performAging`: per-node ageing advances only rows already ageing whose
    /// slack is basic; a nonbasic row with a nonzero dual is reset, one with a zero
    /// dual keeps its age; separation epochs age every basic row and make deletion
    /// due every 5th epoch at `min(epochs, 10)`.
    #[test]
    fn search_lp_rows_age_as_highs_performaging() {
        // rows: 0 basic fresh, 1 basic ageing, 2 nonbasic binding, 3 nonbasic dual 0.
        let mut rows = vec![row(0, 0), row(1, 3), row(2, 7), row(3, 4)];
        let basic = |c: usize| c == 10 || c == 11;
        let y = [0.0, 0.0, -2.0, 0.0];
        let mut epochs = 0u32;

        assert_eq!(
            highs_age_rows(&mut rows, basic, &y, false, &mut epochs),
            None
        );
        let ages: Vec<u32> = rows.iter().map(|r| r.age).collect();
        assert_eq!(ages, vec![0, 4, 0, 4]);
        assert_eq!(epochs, 0);

        let mut due = Vec::new();
        for _ in 0..10 {
            due.push(highs_age_rows(&mut rows, basic, &y, true, &mut epochs));
        }
        assert_eq!(epochs, 10);
        let expect: Vec<Option<u32>> = (1..=10u32)
            .map(|e| (e % 5 == 0).then_some(e.min(CUT_LP_AGE_LIMIT)))
            .collect();
        assert_eq!(due, expect);
        let ages: Vec<u32> = rows.iter().map(|r| r.age).collect();
        assert_eq!(ages, vec![10, 14, 0, 4]);
    }
}
