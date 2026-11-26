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
use crate::bnb::highs_search::pseudocost::{
    Pseudocost, DEFAULT_MIN_RELIABLE, HIGHS_FEASIBILITY_WEIGHT,
};
use crate::bnb::highs_search::search::{
    BranchOutcome, BranchStep, MainLoop, NodeData, NodeFate, NodeResult as SearchResult, Search,
    SearchEnv, NO_NODE_LIMIT,
};
use crate::bnb::node::{BranchRecord, NodeStatus};
use crate::lp::highs_cutgen::conflict_pool::{ConflictDomainObservers, ConflictPool};
use crate::lp::highs_cutgen::core::{CutGenCore, CutGenParams};
use crate::lp::highs_cutgen::domain::{BoundType, ConflictLinks, Domain, DomainChange, Reason};
use crate::lp::highs_cutgen::pool::{CutPool, DomainObservers};
use crate::lp::highs_cutgen::separators::tighten_coefficients_with;

mod nodebox;
mod refcheck;

#[cfg(test)]
thread_local! {
    /// Searches run on this thread (tests prove the route fired).
    pub(super) static RUNS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// The simplex `max_iter` each node evaluation of the search solved under
    /// ([`node_iter_limit_enabled`]'s regression test).
    pub(super) static NODE_MAX_ITER: std::cell::RefCell<Vec<usize>> =
        const { std::cell::RefCell::new(Vec::new()) };
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
    /// stops (uncertified) unless `stall_keeps_searching`, and one with an incumbent
    /// runs the stall re-solve once.
    pub stall_at: Option<std::time::Instant>,
    /// The caller offered `MilpOptions::extend_no_incumbent_s`: a search with no
    /// incumbent at `stall_at` keeps searching to its deadline (`stall_stops` in `milp_driver.rs`).
    pub stall_keeps_searching: bool,
    /// HiGHS's heuristic effort budget (`moreHeuristicsAllowed`,
    /// [`heur_effort_enabled`]) gates the node heuristics.
    pub heur_effort: bool,
    /// This solve is a heuristic sub-MIP (`mipsolver.submip`).
    pub submip: bool,
    /// LP iterations the driver spent before the search (root LP and cut loop).
    pub iters_before: i64,
    /// Root LP solves behind `iters_before` (their mean is `avgrootlpiters`).
    pub root_solves: usize,
    /// [`node_iter_limit_enabled`]: HiGHS's node iteration limit on the search's LPs.
    pub node_iter_limit: bool,
    /// `MilpOptions::max_stall_nodes` (0 = none).
    pub max_stall_nodes: usize,
    /// `MilpOptions::max_leaves` (0 = none).
    pub max_leaves: usize,
    /// `MilpOptions::max_work` (`super::sub_mip_work`), counted with
    /// `iters_before` and `nodes_before`.
    pub max_work: Option<u64>,
    /// `m + n` of the solve, for `super::sub_mip_work`.
    pub work_size: usize,
    /// Nodes the driver created before the search.
    pub nodes_before: usize,
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

#[cfg(test)]
thread_local! {
    /// Branchings taken with the raised pre-incumbent weight on this thread.
    pub(super) static PREINC_BRANCHINGS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    /// Tests force `DISCOPT_MILP_PREINC_SCORE` without touching the process
    /// environment (tests run in parallel).
    pub(super) static FORCE_PREINC_SCORE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// The weight [`preinc_score_enabled`] gives the cutoff and inference terms of the
/// branching scores while the search has no incumbent.
pub(super) const PREINC_FEASIBILITY_WEIGHT: f64 = 1.0;

/// `DISCOPT_MILP_PREINC_SCORE` (#2224; not a HiGHS component): while the top-level
/// search holds no incumbent (`self.best` is `None`, so the cutoff is `+inf` and the
/// cost term of the branching score cannot prune anything), weight the cutoff-rate and
/// inference terms of the pseudocost scores (`get_score*`) at
/// [`PREINC_FEASIBILITY_WEIGHT`] instead of HiGHS's `1e-4`; from the first incumbent on
/// (the driver's initial point included, which is accepted before the root) the scores
/// are HiGHS's exactly. It changes which column is branched on and which child is
/// plunged first, never a bound, a pruning test or an incumbent check.
///
/// Top-level search only: a sub-MIP (RENS/RINS, `milp_depth > 0`) keeps HiGHS's weights
/// even before its own first point, because the hypothesis is about a solve that has no
/// primal bound at all, and a sub-MIP runs under a parent that may already hold one.
///
/// Default ON (graduated, #2224 / PR #2229 review round). Not a HiGHS component, so the
/// bar was "measurably helpful": on 47 MIPLIB 2017 collection instances selected by the
/// property "the ON search branched before its first incumbent" (none from the
/// enlight*/amur families the first entry rested on), 7 gains, 0 losses, 0 violations;
/// with the first entry population and the MIPLIB-38+M panel in §8.16. `0` (or empty)
/// restores HiGHS's `1e-4` throughout.
///
/// `1`/`0` (also `true`/`yes`/`on`, `false`/`no`/`off`/empty, as
/// [`heur_effort_enabled`]); anything else is refused. Profile counter
/// `PreIncScoreBranchings`: branchings taken with the raised weight. Measurement:
/// `docs/dev/flag-retirement-audit.md` and `docs/dev/highs-mip-replacement-plan.md` §8.16.
pub(super) fn preinc_score_enabled() -> bool {
    #[cfg(test)]
    if let Some(v) = FORCE_PREINC_SCORE.with(|f| f.get()) {
        return v;
    }
    match std::env::var("DISCOPT_MILP_PREINC_SCORE") {
        Err(_) => true,
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" | "" => false,
            other => {
                panic!("DISCOPT_MILP_PREINC_SCORE={other:?} is not recognized (expected 1 or 0)")
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
    /// `MilpOptions::max_work` work units.
    Work,
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
    /// Branchings taken with the raised pre-incumbent score weight
    /// (`DISCOPT_MILP_PREINC_SCORE`).
    pub preinc_branchings: usize,
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
    /// The evaluation's verdict counters.
    pub eval: super::highs_eval::EvalStats,
    /// `DISCOPT_MILP_CONFLICT`: (conflicts from infeasible domains, from LP dual
    /// proofs, global bound changes carried into the local baseline).
    pub conflicts: (u64, u64, u64),
    /// `DISCOPT_MILP_CONFLICT`: (conflict cuts `generateConflict` added to the cut
    /// pool, pool cuts the search's domains propagate).
    pub conflict_cuts: (u64, u64),
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

/// `mipdata_->feastol` in `HighsSeparation::separate`'s progress test.
const FEASTOL_SEPA: f64 = 1e-6;

/// `DISCOPT_HIGHS_NODE_ITER_LIMIT`: the search's LP solves run under HiGHS's node
/// iteration limit. `processNodes` sets `iterlimit = max(10000, 10 * max(avgiter,
/// avgrootlpiters), 3 * firstrootlpiters / 2)` on the worker's LP before each dive
/// (`HighsMipSolver.cpp`, HiGHS 1.16 `f3f113d`), and it holds until set again, so
/// it also bounds the next node's evaluation, strong-branching probes and separation
/// re-solves; before the first node the root's `max(10000, 10 * avgrootlpiters)`
/// holds. Here it is the simplex's `max_iter` (capped by the driver's own), a per
/// simplex-run cap where HiGHS's counts the whole solve. A solve that reaches it ends
/// `IterLimit`, which a node takes as untrusted: it stays open on the bound it
/// inherited and is never fathomed (HiGHS's `kError`; its 200-iteration IPM rescue
/// is not ported). Read once per search, garbage refused. Entry evidence (neos17, 2,000
/// nodes, deterministic, with [`search_sepa_enabled`]): 1,244,493 primal phase-1
/// pivots against 35,192 without it, from warm starts the dual refuses as dual
/// infeasible (`DualPrepRejectDualInf` 104 against 74) that fall to a warm primal
/// solve grinding to the 100,000-pivot default; `NodeLpSolve` took 209 of 213 s.
/// With the limit (same build, 3 interleaved reps, load ~11): phase-1 pivots
/// 1,244,493 -> 171,123, LP iterations 223,557 -> 133,877, wall 115.2 +/- 10.4 s ->
/// 24.5 +/- 2.7 s (separation off: 11.3 +/- 1.2 s), same incumbent and bound at the
/// cap; neos-911970 unchanged (713,488 -> 715,507 iterations, 41.6 -> 40.5 s).
///
/// Default OFF until its §5 panel; the row is in `docs/dev/flag-retirement-audit.md`.
pub(super) fn node_iter_limit_enabled() -> bool {
    match std::env::var("DISCOPT_HIGHS_NODE_ITER_LIMIT") {
        Err(_) => false,
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" | "" => false,
            other => panic!(
                "DISCOPT_HIGHS_NODE_ITER_LIMIT={other:?} is not recognized (expected 1 or 0)"
            ),
        },
    }
}

/// HiGHS's node iteration limit (see [`node_iter_limit_enabled`]).
pub(super) fn node_iteration_limit(
    avgiter: f64,
    avg_root_iters: f64,
    first_root_iters: i64,
) -> usize {
    let iterlimit = (10.0 * avgiter.max(avg_root_iters)) as i64;
    10_000i64.max(iterlimit).max((3 * first_root_iters) / 2) as usize
}

/// `DISCOPT_HIGHS_SEARCH_SEPA`: node separation in the search is HiGHS's
/// `HighsSeparation::separationRound` on the root's persistent separation subsystem
/// (clique, tableau, path and mod-k separators into the global cut pool, the pool's
/// selection over the node's box, the cuts entering the search LP as rows), instead
/// of discopt's per-node GMI over a fresh pool. Needs the search cut LP
/// ([`search_cut_lp_enabled`]). As in HiGHS, the search's separators start fresh
/// ([`RootSeparation::begin_search`]: `HighsMipWorker` owns its own
/// `HighsSeparation`), and the transformed LP's variable-bound choice reads the node
/// queue's open-node counts and the node LP's reduced costs (`getBestVub/Vlb`'s
/// `numNodesDown/Up` rank and `kUseDualsForBreakingTies`). Read per solve, garbage
/// refused. Entry evidence (bell5, presolve off, matched node boxes): at the first
/// diverging node HiGHS's round adds 3 rows and raises the bound by 3,762 while
/// discopt's adds 1 row and 164; downstream discopt is weaker on 1,349 of 1,727 boxes.
///
/// Default OFF: its 2026-11-26 §5 panel (MIPLIB-38 + M, 2,000 nodes) was bar-1 clean
/// with -35 % nodes over the both-certified rows, but lost bell5's certificate at the
/// node cap (one draw; four permutations all favour the flag) and failed bars 2a/2e.
/// It graduates as a member of `DISCOPT_HIGHS_CONFIG` on a passing re-panel; the row
/// is in `docs/dev/flag-retirement-audit.md`.
///
/// [`RootSeparation::begin_search`]: crate::lp::highs_cutgen::bridge::RootSeparation::begin_search
pub(super) fn search_sepa_enabled() -> bool {
    match std::env::var("DISCOPT_HIGHS_SEARCH_SEPA") {
        Err(_) => false,
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" | "" => false,
            other => {
                panic!("DISCOPT_HIGHS_SEARCH_SEPA={other:?} is not recognized (expected 1 or 0)")
            }
        },
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

/// `DISCOPT_MILP_CONFLICT`: HiGHS conflict analysis in the search (`HighsSearch`'s
/// `conflictAnalysis` at every infeasible local domain, `addInfeasibleConflict` from
/// the verified Farkas ray of an infeasible node LP) into a conflict pool that both
/// the local domain and a global domain propagate (`HighsMipSolverData::
/// conflictPool`), aged before each round. A conflict propagated to an empty global
/// domain proves the search complete. Default ON since 2026-11-26 (#2350): graduated
/// jointly with `DISCOPT_MILP_CLIQUE_DOMAIN` (the pair is the HiGHS default) on a
/// CLAUDE.md §5 panel scored with permutation reps (row in
/// `docs/dev/flag-retirement-audit.md`); `=0` restores the search without conflict
/// analysis or the cut pool. Read per search, garbage refused.
pub(super) fn conflict_enabled() -> bool {
    match std::env::var("DISCOPT_MILP_CONFLICT") {
        Err(_) => true,
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" | "" => false,
            other => panic!("DISCOPT_MILP_CONFLICT={other:?} is not recognized (expected 1 or 0)"),
        },
    }
}

/// `DISCOPT_MILP_CLIQUE_DOMAIN`: the root's clique table (`mipdata_->cliquetable`,
/// the ported `extractCliques` + `runCliqueMerging` of the root separation), filled by
/// HiGHS's presolve probing schedule (`HPresolve::runProbing`: every binary, scored and
/// budgeted as in HiGHS) and by probing the root LP's fractional binaries
/// (`separateImpliedBounds`' probing loop), both in `lp/highs_cutgen/probing.rs`, then
/// attached with the probing implications to the search's local and global domains.
/// Fixing a binary then fixes the other literals of its cliques (`HighsDomain::
/// changeBound`'s `cliquetable.addImplications`) and re-applies what its probe implied
/// (`implications.applyImplications`), with `Reason::cliqueTable` reasons conflict
/// analysis explains. With `DISCOPT_MILP_CONFLICT` (which supplies the global domain)
/// node separation also runs `separateImpliedBounds`: probe the node LP's unprobed
/// fractional binaries and add the violated implied-bound cuts. Bound-changing.
/// Default ON since 2026-11-26 (#2350), graduated jointly with `DISCOPT_MILP_CONFLICT`
/// on a §5 panel scored with permutation reps (row in
/// `docs/dev/flag-retirement-audit.md`: cert-clean, nodes -20 % to -22 %, no dual loss,
/// median primal gap lower at both caps); `=0` restores the search without the table
/// hook or probing. The per-round root probing is still approximated by one pass at
/// search start (the root separation has no `Domain`). Read per search, garbage refused.
pub(super) fn clique_domain_enabled() -> bool {
    match std::env::var("DISCOPT_MILP_CLIQUE_DOMAIN") {
        Err(_) => true,
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" | "" => false,
            other => {
                panic!("DISCOPT_MILP_CLIQUE_DOMAIN={other:?} is not recognized (expected 1 or 0)")
            }
        },
    }
}

/// The search's conflict state (`DISCOPT_MILP_CONFLICT`).
///
/// Scope: one [`run`] on one model. A conflict or a global fixing is derived from that
/// model's rows, which under `DISCOPT_MILP_SYM_LEX` include the lex-leader rows: it
/// is valid only for the symmetry-reduced problem. Nothing here leaves the run. The
/// pool is created in `enable_conflicts` and dropped with the environment. The
/// driver's result carries only `status`, `x`, `obj` and `bound`. Each of those is
/// valid for the unreduced model: the rows keep a lex-max point of every orbit, so
/// they keep the optimal value and the feasibility verdict. No thread-local or static
/// state holds a conflict. A re-entry or re-solve builds a fresh pool on its own model.
pub(super) struct Conflicts {
    /// HiGHS `conflictPool(5 * mip_pool_age_limit, mip_pool_soft_limit)`.
    pool: ConflictPool,
    /// The global domain (HiGHS `mipdata_->domain`): the local domain's baseline
    /// plus the pool's globally valid consequences.
    gdom: Domain,
    /// The pool's index in both domains.
    cp: usize,
    /// `gdom`'s change-stack length at the last sync into the local baseline.
    synced: usize,
    /// Conflicts from infeasible domains, from LP dual proofs, and global bound
    /// changes carried into the local baseline.
    pub added: u64,
    pub proof_added: u64,
    pub global_fixings: u64,
    /// HiGHS `mipdata_->cutpool`: the conflict cuts `generateConflict` derives from
    /// LP dual proofs and the search's pool cuts, propagated by both domains
    /// (`addCutpool`, HiGHS's `CutpoolPropagation`). Same validity scope as the
    /// conflicts: one run, this model.
    cutpool: CutPool,
    /// The cut pool's index among both domains' cut pools.
    cpi: usize,
    /// The working rows' structural parts, for rewriting a working-LP cut over the
    /// structurals ([`StructRows::to_struct`]).
    srows: StructRows,
    pub conflict_cuts: u64,
    pub pool_cuts: u64,
}

/// The working rows `A [x; s] = b` split into their structural part and their
/// singleton slack, to rewrite a cut over working columns (slacks included) as a row
/// over the structurals the domains propagate.
struct StructRows {
    ns: usize,
    rows: Vec<(Vec<usize>, Vec<f64>)>,
    /// The row holds a non-structural column other than its singleton slack.
    other: Vec<bool>,
    /// Slack column -> (its row, its coefficient).
    slack_row: std::collections::HashMap<usize, (usize, f64)>,
    b: Vec<f64>,
    /// Each row's acceptance tolerance ([`row_tolerances`] over the global box).
    eps: Vec<f64>,
    l: Vec<f64>,
    u: Vec<f64>,
}

impl StructRows {
    fn new(ctx: &NodeCtx<'_>, slacks: &[Option<(usize, f64)>]) -> Self {
        let (m, n, ns) = (ctx.m_w, ctx.n_w, ctx.ns);
        let mut rows: Vec<(Vec<usize>, Vec<f64>)> = vec![(Vec::new(), Vec::new()); m];
        let mut nonstruct: Vec<Vec<usize>> = vec![Vec::new(); m];
        for j in 0..n {
            let (ri, rv) = ctx.csc_rc.col(j);
            for (&i, &a) in ri.iter().zip(rv) {
                if i >= m || a == 0.0 {
                    continue;
                }
                if j < ns {
                    rows[i].0.push(j);
                    rows[i].1.push(a);
                } else {
                    nonstruct[i].push(j);
                }
            }
        }
        let mut slack_row = std::collections::HashMap::new();
        let mut other = vec![false; m];
        for i in 0..m {
            let sl = slacks.get(i).copied().flatten();
            if let Some((c, coef)) = sl {
                slack_row.insert(c, (i, coef));
            }
            other[i] = nonstruct[i].iter().any(|&j| sl.is_none_or(|(c, _)| c != j));
        }
        StructRows {
            ns,
            rows,
            other,
            slack_row,
            b: ctx.b_w[..m].to_vec(),
            eps: row_tolerances(ctx.csc_rc, m, n, ctx.b_w, ctx.l_w, ctx.u_w),
            l: ctx.l_w[..ns].to_vec(),
            u: ctx.u_w[..ns].to_vec(),
        }
    }

    /// The working-LP cut `cut.coeffs . [x; s] >= cut.rhs` as `vals . x[inds] <= rhs`
    /// over the structurals, valid for every point the incumbent check accepts: each
    /// slack is replaced through its row (`s = (b_r - A_r x) / c`), the rhs widened by
    /// the row's acceptance tolerance times the multiplier, plus a relative margin as
    /// in [`farkas_proof_row`]. `None` when a slack has no singleton row, a used row
    /// holds another non-structural column, or nothing structural remains.
    fn to_struct(&self, cut: &GomoryCut) -> Option<(Vec<i32>, Vec<f64>, f64)> {
        let ns = self.ns;
        let mut w = vec![0.0f64; ns];
        let mut r = cut.rhs;
        let mut widen = 0.0f64;
        let mut scale = 1.0 + cut.rhs.abs();
        for (j, &a) in cut.coeffs.iter().enumerate() {
            if a == 0.0 {
                continue;
            }
            if j < ns {
                w[j] += a;
                continue;
            }
            let &(i, c) = self.slack_row.get(&j)?;
            if self.other[i] || c == 0.0 {
                return None;
            }
            let f = a / c;
            r -= f * self.b[i];
            scale += (f * self.b[i]).abs();
            widen += f.abs() * self.eps[i];
            let (ri, rv) = &self.rows[i];
            for (&k, &v) in ri.iter().zip(rv) {
                w[k] -= f * v;
            }
        }
        // `w . x >= r - widen`, i.e. `-w . x <= widen - r`.
        let wmax = w.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
        if wmax == 0.0 || !r.is_finite() {
            return None;
        }
        let mut rhs = widen - r;
        let (mut inds, mut vals) = (Vec::new(), Vec::new());
        for (j, &wj) in w.iter().enumerate() {
            let a = -wj;
            if a == 0.0 {
                continue;
            }
            let (lj, uj) = (self.l[j], self.u[j]);
            let mag = [lj, uj]
                .iter()
                .filter(|v| v.abs() < INF)
                .fold(0.0f64, |acc, v| acc.max(v.abs()));
            scale += a.abs() * mag;
            if a.abs() <= 1e-12 * wmax {
                let lo = if a > 0.0 { lj } else { uj };
                if lo.abs() < INF {
                    rhs -= a * lo;
                    continue;
                }
            }
            inds.push(j as i32);
            vals.push(a);
        }
        if inds.is_empty() || !rhs.is_finite() {
            return None;
        }
        rhs += 1e-9 * scale;
        Some((inds, vals, rhs))
    }
}

/// `HighsLpRelaxation::computeDualInfProof` in discopt's working form: the verified
/// Farkas ray `ray` (solve coordinates) of an infeasible node LP, times `sgn`, as a
/// proof row `vals . x[inds] <= rhs` over the structurals that every point the
/// incumbent check accepts satisfies (so it is valid globally, not just at the node).
///
/// With `y` the ray in original units and `w = A^T y` over all `n_w` columns, every
/// `x` with `A [x; s] = b` has `sum_{j<ns} w_j x_j = y.b - sum_{j>=ns} w_j s_j`, so over
/// the GLOBAL slack box `-sum_{j<ns} w_j x_j <= -(y.b - sum_{j>=ns} max(w_j l_j, w_j
/// u_j))`. The rhs is widened by `sum_i |y_i| eps_i` (each row's acceptance tolerance,
/// [`row_tolerances`] over the global box) and a relative margin, and a structural with
/// a negligible coefficient is moved to the rhs at its worst global bound. `None` when
/// a slack (or such a structural) the proof needs is unbounded.
fn farkas_proof_row(
    ctx: &NodeCtx<'_>,
    ray: &[f64],
    sgn: f64,
) -> Option<(Vec<usize>, Vec<f64>, f64)> {
    let (m, n, ns) = (ctx.m_w, ctx.n_w, ctx.ns);
    let mut y: Vec<f64> = ray.iter().take(m).map(|v| sgn * v).collect();
    if y.len() < m {
        return None;
    }
    if let Some(sc) = ctx.scaling {
        sc.unscale_dual(&mut y);
    }
    proof_row_from_dual(ctx.csc_rc, m, n, ns, ctx.b_w, ctx.l_w, ctx.u_w, y)
}

/// The body of [`farkas_proof_row`] once the ray `y` is in original units, over the
/// unscaled working matrix `csc` (`m` rows, `n` columns, the first `ns` structural)
/// with rhs `b` and the global box `l`/`u`.
#[allow(clippy::too_many_arguments)]
fn proof_row_from_dual(
    csc: &SparseCols,
    m: usize,
    n: usize,
    ns: usize,
    b: &[f64],
    l: &[f64],
    u: &[f64],
    mut y: Vec<f64>,
) -> Option<(Vec<usize>, Vec<f64>, f64)> {
    let eps = row_tolerances(csc, m, n, b, l, u);
    let col_w = |j: usize, y: &[f64]| -> f64 {
        let (rows, vals) = csc.col(j);
        rows.iter()
            .zip(vals)
            .filter(|(i, _)| **i < m)
            .map(|(&i, &a)| a * y[i])
            .sum()
    };
    // HiGHS `storeDualInfProof` aggregates only the rows whose ray weight points at a
    // finite side (`if (lp.row_upper_[iRow] == kHighsInf) continue;` and the lower
    // mirror): a row whose weight would need its infinite side is left out of the
    // aggregation instead of failing the proof. Any subset of the rows, each with a
    // weight of the sign its finite side allows, aggregates to a valid inequality, so
    // dropping a row is sound (the proof can only get weaker, and conflict analysis
    // then simply finds nothing). In this working form a row's sides are its singleton
    // slack's bounds: the weight `w_s = sigma * y_i` needs `u_s` when positive and `l_s`
    // when negative. A slack shared by several rows is not dropped (`None` below).
    for j in ns..n {
        let (rows, vals) = csc.col(j);
        let mut it = rows.iter().zip(vals).filter(|(i, a)| **i < m && **a != 0.0);
        let (Some((&i, &sigma)), None) = (it.next(), it.next()) else {
            continue;
        };
        let wj = sigma * y[i];
        if wj == 0.0 {
            continue;
        }
        let side = if wj > 0.0 { u[j] } else { l[j] };
        if side.abs() >= INF {
            y[i] = 0.0;
        }
    }
    let w: Vec<f64> = (0..n).map(|j| col_w(j, &y)).collect();
    let yb: f64 = (0..m).map(|i| y[i] * b[i]).sum();
    let mut scale = 1.0 + (0..m).map(|i| (y[i] * b[i]).abs()).sum::<f64>();
    let mut beta = yb;
    for j in ns..n {
        let wj = w[j];
        if wj == 0.0 {
            continue;
        }
        let hi = if wj > 0.0 { u[j] } else { l[j] };
        if hi.abs() >= INF {
            return None;
        }
        beta -= wj * hi;
        scale += (wj * hi).abs();
    }
    let wmax = w[..ns].iter().fold(0.0f64, |a, v| a.max(v.abs()));
    if wmax == 0.0 {
        return None;
    }
    let mut rhs = -beta + (0..m).map(|i| y[i].abs() * eps[i]).sum::<f64>();
    let (mut inds, mut vals) = (Vec::new(), Vec::new());
    for j in 0..ns {
        let a = -w[j];
        if a == 0.0 {
            continue;
        }
        let mag = [l[j], u[j]]
            .iter()
            .filter(|v| v.abs() < INF)
            .fold(0.0f64, |acc, v| acc.max(v.abs()));
        scale += a.abs() * mag;
        if a.abs() <= 1e-12 * wmax {
            // Drop `a x_j`: the rest is `<= rhs - a x_j <= rhs - min(a l_j, a u_j)`
            // over the global box.
            let lo = if a > 0.0 { l[j] } else { u[j] };
            if lo.abs() >= INF {
                inds.push(j);
                vals.push(a);
                continue;
            }
            rhs -= a * lo;
            continue;
        }
        inds.push(j);
        vals.push(a);
    }
    if inds.is_empty() || !rhs.is_finite() {
        return None;
    }
    rhs += 1e-9 * scale;
    Some((inds, vals, rhs))
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
fn make_ctx<'x>(
    base: &NodeCtx<'x>,
    lp: &'x Option<SearchLp>,
    simplex: &'x Option<SimplexOptions>,
) -> NodeCtx<'x> {
    let base = &NodeCtx {
        simplex: simplex.as_ref().unwrap_or(base.simplex),
        ..*base
    };
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
    /// `DISCOPT_MILP_PREINC_SCORE` is on for this search, and the branchings taken
    /// with its raised weight (no incumbent yet).
    preinc_score: bool,
    pub preinc_branchings: usize,
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
    /// LP cut rows selected from the conflict `cutpool` (signature -> pool index):
    /// HiGHS `lpCutRemoved` when the row leaves the LP, so the cut ages again.
    pool_lp: std::collections::HashMap<Vec<(u32, i64)>, i32>,
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
    /// `DISCOPT_MILP_CONFLICT`.
    conf: Option<Conflicts>,
    /// `DISCOPT_HIGHS_SEARCH_SEPA`: the root's separation subsystem (separators,
    /// clique table, variable bounds and cut pool), carried into the search as HiGHS
    /// carries `mipdata.cutpool` and the `HighsSeparation` separators; node separation
    /// then runs [`RootSeparation::round_node`] on the search's LP. `None` runs
    /// discopt's node GMI.
    hsep: Option<&'a mut crate::lp::highs_cutgen::bridge::RootSeparation>,
    /// [`node_iter_limit_enabled`]: the simplex options of the search's LP solves,
    /// `max_iter` the node iteration limit; `None` uses the driver's.
    node_simplex: Option<SimplexOptions>,
    /// `mipdata_->avgrootlpiters` and `firstrootlpiters`.
    avg_root_iters: f64,
    first_root_iters: i64,
    /// Pool index of each search-LP cut row that came from [`Self::hsep`] (by the
    /// base-form cut's signature), reported back when the row leaves the LP.
    hsep_ids: std::collections::HashMap<Vec<(u32, i64)>, i32>,
    /// LP iterations already reported to [`Self::hsep`] (its separators' seed).
    hsep_iters: i64,
    /// The root LP's objective (`rootlpsolobj`), the reference of
    /// [`Self::separate_highs`]'s progress test.
    root_obj: f64,
    /// The node LP's objective at its last evaluation or separation resolve.
    last_lp_obj: f64,
    /// The last separation resolve was not trusted (`!scaledOptimal`).
    resolved_untrusted: bool,
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
    ///
    /// `idx[k]` is `cuts[k]`'s conflict-`cutpool` index when it was selected from it;
    /// a selected cut that does not enter (no room, already a row) is released at once.
    fn append_cut_rows(&mut self, cuts: &[GomoryCut], idx: &[Option<i32>]) {
        self.append_cut_rows_tagged(cuts, idx, &[]);
    }

    /// [`Self::append_cut_rows`], with also the [`Self::hsep`] pool index of each cut
    /// (`ids[k]` for `cuts[k]`; empty when the cuts are not that pool's). A cut from
    /// either pool that does not enter the LP (no room, already in it) is reported
    /// back to its pool at once, so the pool offers it again.
    fn append_cut_rows_tagged(&mut self, cuts: &[GomoryCut], idx: &[Option<i32>], ids: &[i32]) {
        debug_assert_eq!(cuts.len(), idx.len());
        if cuts.is_empty() {
            return;
        }
        let base = self.ctx;
        let n0 = base.n_w;
        let lp = self.lp.get_or_insert_with(|| SearchLp::from_base(base));
        let room = search_cut_cap(base.m_w).saturating_sub(lp.rows.len());
        let mut add: Vec<GomoryCut> = Vec::new();
        let mut released: Vec<i32> = Vec::new();
        let mut refused: Vec<i32> = Vec::new();
        for (j, (cut, &k)) in cuts.iter().zip(idx).enumerate() {
            let id = ids.get(j).copied();
            if add.len() >= room {
                released.extend(k);
                refused.extend(id);
                continue;
            }
            let c = lp.to_base(cut, n0);
            if let Some(rc) = self.refp.as_mut() {
                rc.cut("lp-row", self.nodes, &c, &rc.x.clone());
            }
            let sig = cut_signature(&c);
            if self.in_lp.insert(sig.clone()) {
                if let Some(k) = k {
                    self.pool_lp.insert(sig.clone(), k);
                }
                if let Some(id) = id {
                    self.hsep_ids.insert(sig, id);
                }
                add.push(c);
            } else {
                released.extend(k);
                refused.extend(id);
            }
        }
        if let Some(cf) = self.conf.as_mut() {
            for k in released {
                cf.cutpool.lp_cut_removed(k);
            }
        }
        if let Some(hs) = self.hsep.as_deref_mut() {
            for id in refused {
                hs.lp_cut_removed(id);
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

    /// `HighsSeparation::separate` at a node (`DISCOPT_HIGHS_SEARCH_SEPA`): rounds of
    /// `separationRound` -- the clique, tableau, path and mod-k separators of the
    /// root's subsystem on the search LP's point, the persistent pool's selection
    /// over the node's box, the selected cuts appended to the search LP as rows
    /// (`lp->addCuts`), the LP resolved over the node's box and aged with deletion --
    /// while a round adds cuts, the LP stays optimal and fractional, and the objective
    /// keeps improving by more than 1 % of the last round's gain over the root's.
    /// Every cut is a globally valid inequality of the pool, so the LP bound it gives
    /// holds on the node's box; the dive's next evaluation of the node takes it.
    /// Returns whether the global domain became infeasible (never here).
    fn separate_highs(&mut self) -> bool {
        let ns = self.ctx.ns;
        let fractional = |x: &[f64]| (0..ns).any(|j| self.ctx.is_int[j] && frac(x[j]) > INT_TOL);
        let ok = self.eval.last_dual_raw.is_some()
            && self.eval.basis.is_some()
            && self.eval.last_box.is_some()
            && self.eval.point_full.as_deref().is_some_and(fractional);
        if !ok {
            // `separate`'s else branch: no optimal LP or no fractional integer, just
            // ageing.
            self.age_cut_rows(true);
            return false;
        }
        let first = self.root_obj;
        loop {
            if self.last_lp_obj >= self.optlim()
                || self
                    .lim
                    .deadline
                    .is_some_and(|d| std::time::Instant::now() >= d)
            {
                break;
            }
            let last = self.last_lp_obj;
            if !self.separation_round() {
                break;
            }
            let still = self.eval.point_full.as_deref().is_some_and(fractional);
            if !still || self.resolved_untrusted {
                break;
            }
            if self.last_lp_obj - first <= f64::max(last - first, FEASTOL_SEPA) * 1.01 {
                break;
            }
        }
        false
    }

    /// One `separationRound` on the node's current LP point; whether it added cuts
    /// and the resolve over the node's box was trusted (`scaledOptimal`).
    fn separation_round(&mut self) -> bool {
        let ns = self.ctx.ns;
        let (Some(x), Some(y), Some(basis), Some((fl, fu))) = (
            self.eval.point_full.clone(),
            self.eval.last_dual_raw.clone(),
            self.eval.basis.clone(),
            self.eval.last_box.clone(),
        ) else {
            return false;
        };
        let iters = self.eval.lp_iters + self.eval.sb_iters;
        let rc = {
            let ctx = make_ctx(self.ctx, &self.lp, &self.node_simplex);
            if x.len() != ctx.n_w || y.len() != ctx.m_w {
                return false;
            }
            let Some(hs) = self.hsep.as_deref_mut() else {
                return false;
            };
            hs.add_lp_iterations(iters - self.hsep_iters);
            hs.round_node(
                &crate::lp::highs_cutgen::bridge::StdFormRound {
                    csc: ctx.csc_rc,
                    m: ctx.m_w,
                    n: ctx.n_w,
                    ns,
                    b: ctx.b_w,
                    l: ctx.l_w,
                    u: ctx.u_w,
                    is_int: ctx.is_int_full,
                    x: &x,
                    y: &y,
                    basic_vars: &basis.basic_vars,
                    deadline: self.lim.deadline,
                    open_nodes: Some(&self.q),
                    col_dual: self.eval.redcost.as_deref(),
                },
                &fl[..ns],
                &fu[..ns],
            )
        };
        self.hsep_iters = iters;
        self.sep_calls += 1;
        crate::profile::incr_by(crate::profile::Ctr::SepCmirCuts, rc.generated as u64);
        let (ids, mut cuts): (Vec<i32>, Vec<GomoryCut>) = rc.cuts.into_iter().unzip();
        let mut idx: Vec<Option<i32>> = vec![None; cuts.len()];
        // `separationRound`'s `mipdata.cutpool.separate`: discopt keeps the search's
        // conflict / dual-proof cuts in their own cutpool (#2350), so it is separated
        // here too, as `separate` does on the default path.
        if let Some(cf) = self.conf.as_mut() {
            let n_w = make_ctx(self.ctx, &self.lp, &self.node_simplex).n_w;
            let mut cutset = Vec::new();
            let mut obs =
                DomainObservers(vec![(&mut cf.gdom, cf.cpi), (&mut self.eval.dom, cf.cpi)]);
            let sel =
                cf.cutpool
                    .separate(&mut obs, &x[..ns], &fl[..ns], &fu[..ns], 1e-6, &mut cutset);
            for c in sel {
                // `vals . x <= rhs` as `coeffs . x >= rhs` over the working columns.
                let mut coeffs = vec![0.0; n_w];
                for (&j, &v) in c.inds.iter().zip(&c.vals) {
                    coeffs[j as usize] = -v;
                }
                cuts.push(GomoryCut {
                    coeffs,
                    rhs: -c.rhs,
                });
                idx.push(Some(c.index));
            }
        }
        let rows0 = self.cut_rows_added;
        self.append_cut_rows_tagged(&cuts, &idx, &ids);
        crate::profile::incr(crate::profile::Ctr::HighsMipSearchSepaRounds);
        crate::profile::incr_by(
            crate::profile::Ctr::HighsMipSearchSepaRows,
            (self.cut_rows_added - rows0) as u64,
        );
        if self.cut_rows_added == rows0 {
            return false;
        }
        // `lp->resolveLp(&propdomain)` over the node's box, then `performAging(true)`.
        let ctx = make_ctx(self.ctx, &self.lp, &self.node_simplex);
        let (fl, fu) = self.eval.lp_box(&ctx);
        let start = super::highs_eval::fit_basis_to_box(
            self.eval.basis.clone(),
            self.eval.basis_d.as_deref(),
            &fl,
            &fu,
        );
        let nl = node_lp(&ctx, start, &fl, &fu);
        self.eval.lp_iters += nl.sol.iters as i64;
        self.eval.note_solve(&nl.sol);
        self.resolved_untrusted = true;
        if let NodeLpVerdict::Bound { .. } = nl.verdict {
            self.resolved_untrusted = false;
            self.eval.basis = Some(nl.sol.basis.clone());
            self.eval.redcost = Some(super::highs_eval::structural_redcost(&ctx, &nl));
            self.eval.last_dual = Some(nl.rc_dual().to_vec());
            self.eval.dual_epoch += 1;
            self.eval.basis_d = Some(super::highs_eval::full_redcost(&ctx, &nl));
            self.eval.last_dual_raw = Some(nl.sol.dual.clone());
            self.eval.last_box = Some((fl, fu));
            self.eval.point = Some(nl.sol.x[..ns].to_vec());
            self.eval.point_full = Some(nl.sol.x.clone());
            self.last_lp_obj = nl.sol.obj + ctx.obj_const;
        }
        self.age_cut_rows(true);
        !self.resolved_untrusted
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
            let sig = cut_signature(&r.cut);
            if let (Some(k), Some(cf)) = (self.pool_lp.remove(&sig), self.conf.as_mut()) {
                cf.cutpool.lp_cut_removed(k);
            }
            if let Some(id) = self.hsep_ids.remove(&sig) {
                if let Some(hs) = self.hsep.as_deref_mut() {
                    // `HighsLpRelaxation::removeCuts` -> `cutpool.lpCutRemoved`.
                    hs.lp_cut_removed(id);
                }
            }
            self.in_lp.remove(&sig);
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
            // From the first incumbent on the scores are HiGHS's.
            self.pc.set_feasibility_weight(HIGHS_FEASIBILITY_WEIGHT);
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

    /// `setIterationLimit(limit)` on the search's LP (see
    /// [`node_iter_limit_enabled`]); a no-op when the flag is off.
    fn set_iteration_limit(&mut self, limit: usize) {
        if !self.lim.node_iter_limit {
            self.node_simplex = None;
            return;
        }
        let mut o = self.ctx.simplex.clone();
        o.max_iter = o.max_iter.min(limit);
        crate::profile::incr(crate::profile::Ctr::HighsMipSearchIterLimitSet);
        self.node_simplex = Some(o);
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
        } else if self.lim.max_work.is_some_and(|k| {
            let it = self.lim.iters_before + self.eval.lp_iters + self.eval.sb_iters;
            let nodes = self.lim.nodes_before + self.nodes.max(0) as usize;
            super::sub_mip_work(it.max(0) as usize, nodes, self.lim.work_size) >= k
        }) {
            self.stop = Some(Stop::Work);
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
        self.sync_global();
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
        self.sync_global();
    }
    fn domain_propagate_after_backtrack(&mut self, _depth: i32) -> bool {
        let was_in = self.ref_in_domain();
        self.eval.dom.propagate();
        self.ref_box_kept("prop-backtrack", was_in);
        self.analyse_conflict();
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
        self.analyse_conflict();
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
        self.analyse_conflict();
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
    fn global_propagate(&mut self) {
        let lim = self.eval.cutoff - self.ctx.obj_const;
        if let Some(cf) = self.conf.as_mut() {
            cf.gdom.set_upper_limit(lim);
            cf.gdom.propagate();
        }
    }
    fn global_infeasible(&self) -> bool {
        self.conf.as_ref().is_some_and(|cf| cf.gdom.infeasible())
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
        let ctx = make_ctx(self.ctx, &self.lp, &self.node_simplex);
        #[cfg(test)]
        NODE_MAX_ITER.with(|v| v.borrow_mut().push(ctx.simplex.max_iter));
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
        self.last_lp_obj = node.lp_objective;
        match r {
            SearchResult::DomainInfeasible => self.analyse_conflict(),
            SearchResult::LpInfeasible => {
                use crate::profile::{incr, Ctr};
                let ray = self.eval.ray.take();
                if self.conf.is_some() {
                    incr(Ctr::HighsMipSearchProofCalls);
                    if ray.is_none() {
                        incr(Ctr::HighsMipSearchProofNoRay);
                    }
                }
                let proofs: Vec<_> = match (self.conf.is_some(), ray) {
                    (true, Some(ray)) => [1.0, -1.0]
                        .iter()
                        .filter_map(|&sgn| farkas_proof_row(&ctx, &ray, sgn))
                        .collect(),
                    _ => Vec::new(),
                };
                if self.conf.is_some() && proofs.is_empty() {
                    incr(Ctr::HighsMipSearchProofNoRow);
                }
                self.analyse_infeasible_lp(proofs)
            }
            _ => {}
        }
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
        let ctx = make_ctx(self.ctx, &self.lp, &self.node_simplex);
        let ns = self.ctx.ns;
        let ref_box = self.refp.as_ref().map(|_| {
            (
                self.eval.dom.col_lower[..ns].to_vec(),
                self.eval.dom.col_upper[..ns].to_vec(),
            )
        });
        if self.preinc_score && self.best.is_none() {
            debug_assert_eq!(self.pc.feasibility_weight(), PREINC_FEASIBILITY_WEIGHT);
            self.preinc_branchings += 1;
            #[cfg(test)]
            PREINC_BRANCHINGS.with(|c| c.set(c.get() + 1));
        }
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
    fn avg_solve_iters(&self) -> f64 {
        self.eval.avg_solve_iters
    }
    fn set_node_iteration_limit(&mut self, avgiter: f64) {
        let limit = node_iteration_limit(avgiter, self.avg_root_iters, self.first_root_iters);
        self.set_iteration_limit(limit);
    }
    fn separate(&mut self) -> bool {
        // `separateAndStoreBasis` with discopt's node separation: node-local Gomory
        // rounds plus the pool cuts this node violates, on a copy of the node LP; the
        // bound they prove holds on the node's box and lifts the node when the dive
        // re-evaluates it. Under `DISCOPT_HIGHS_SEARCH_CUT_LP` (step 2.1) the cuts also
        // join the search's LP as persistent rows, as HiGHS adds its cuts to the shared
        // LP; otherwise they stay node-local and in the pool.
        if self.hsep.is_some() && self.cut_lp {
            return self.separate_highs();
        }
        let ctx = make_ctx(self.ctx, &self.lp, &self.node_simplex);
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
        // `separationRound` starts with `implications.separateImpliedBounds` on the
        // global domain: probe the LP's unprobed fractional binaries, then the
        // implied-bound cuts the LP point violates, into the cut pool.
        let mut implied: Vec<GomoryCut> = Vec::new();
        if clique_domain_enabled() {
            if let Some(cf) = self.conf.as_mut() {
                let feastol = self.eval.dom.feastol();
                let fr: Vec<usize> = (0..ctx.ns)
                    .filter(|&j| ctx.is_int[j] && frac(x[j]) > feastol)
                    .collect();
                let (st, cuts) = crate::lp::highs_cutgen::probing::separate_implied_bounds(
                    &mut cf.gdom,
                    &mut self.eval.dom,
                    &x,
                    &fr,
                );
                // Probing committed its reductions to the global baseline: the next
                // `sync_global` copies the global bounds into the local baseline.
                cf.synced = usize::MAX;
                crate::profile::incr_by(crate::profile::Ctr::MilpProbeNode, st.probed as u64);
                crate::profile::incr_by(
                    crate::profile::Ctr::MilpImpliedBoundCuts,
                    cuts.len() as u64,
                );
                for (inds, vals, rhs, _) in cuts {
                    // `vals . x <= rhs` as `coeffs . x >= rhs`.
                    let mut coeffs = vec![0.0; self.ctx.n_w];
                    coeffs[inds[0]] -= vals[0];
                    coeffs[inds[1]] -= vals[1];
                    implied.push(GomoryCut { coeffs, rhs: -rhs });
                }
            }
            if self.conf.as_ref().is_some_and(|cf| cf.gdom.infeasible()) {
                return false;
            }
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
        let mut rows: Vec<GomoryCut> = pick
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
        for cut in &implied {
            let mut coeffs = cut.coeffs.clone();
            coeffs.resize(ctx.n_w, 0.0);
            rows.push(GomoryCut {
                coeffs,
                rhs: cut.rhs,
            });
        }
        // `separationRound` ends with `cutpool.separate`: the conflict cuts this LP
        // point violates, scored and filtered for parallelism; a cut it passes over
        // ages, and is deleted past the age limit.
        let mut from_pool: Vec<(i32, GomoryCut)> = Vec::new();
        if let Some(cf) = self.conf.as_mut() {
            let ns = ctx.ns;
            let mut cutset = Vec::new();
            let mut obs =
                DomainObservers(vec![(&mut cf.gdom, cf.cpi), (&mut self.eval.dom, cf.cpi)]);
            let sel =
                cf.cutpool
                    .separate(&mut obs, &x[..ns], &fl[..ns], &fu[..ns], 1e-6, &mut cutset);
            for c in sel {
                // `vals . x <= rhs` as `coeffs . x >= rhs` over the working columns.
                let mut coeffs = vec![0.0; ctx.n_w];
                for (&j, &v) in c.inds.iter().zip(&c.vals) {
                    coeffs[j as usize] = -v;
                }
                from_pool.push((
                    c.index,
                    GomoryCut {
                        coeffs,
                        rhs: -c.rhs,
                    },
                ));
            }
        }
        rows.extend(from_pool.iter().map(|(_, c)| c.clone()));
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
            for cut in implied.into_iter().chain(lg.new_cuts) {
                self.pool_add(cut);
            }
            // The selected pool cuts served this node's lift only: out of the LP.
            if let Some(cf) = self.conf.as_mut() {
                for (k, _) in &from_pool {
                    cf.cutpool.lp_cut_removed(*k);
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
        let mut add: Vec<(GomoryCut, Option<i32>)> = pick
            .iter()
            .map(|&(_, k)| (self.pool[k].clone(), None))
            .collect();
        add.extend(from_pool.into_iter().map(|(k, c)| (c, Some(k))));
        for cut in implied.into_iter().chain(new_base) {
            self.pool_add(cut.clone());
            add.push((cut, None));
        }
        if add.len() > SEARCH_CUTS_PER_NODE {
            let over: Vec<i32> = add
                .drain(SEARCH_CUTS_PER_NODE..)
                .filter_map(|(_, k)| k)
                .collect();
            if let Some(cf) = self.conf.as_mut() {
                for k in over {
                    cf.cutpool.lp_cut_removed(k);
                }
            }
        }
        let (add, idx): (Vec<GomoryCut>, Vec<Option<i32>>) = add.into_iter().unzip();
        let rows0 = self.cut_rows_added;
        self.append_cut_rows(&add, &idx);
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
                None if self.lim.stall_keeps_searching => {}
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
        let preinc_score = preinc_score_enabled();
        let mut pc = Pseudocost::new(
            ctx.ns,
            if lim.strong_branch {
                DEFAULT_MIN_RELIABLE
            } else {
                0
            },
        );
        if preinc_score {
            pc.set_feasibility_weight(PREINC_FEASIBILITY_WEIGHT);
        }
        Some(HighsEnv {
            ctx,
            lim,
            sub_mip,
            eval,
            pc,
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
            preinc_score,
            preinc_branchings: 0,
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
            pool_lp: std::collections::HashMap::new(),
            cut_rows_added: 0,
            cut_rows_deleted: 0,
            seeded_pool: 0,
            refp,
            queue_pruned: 0.0,
            progress: (0.0, 0, 0),
            stall_nodes: 0,
            conf: None,
            hsep: None,
            node_simplex: None,
            avg_root_iters: if lim.root_solves > 0 {
                lim.iters_before as f64 / lim.root_solves as f64
            } else {
                0.0
            },
            first_root_iters: lim.iters_before,
            hsep_ids: std::collections::HashMap::new(),
            hsep_iters: 0,
            root_obj: f64::NEG_INFINITY,
            last_lp_obj: f64::NEG_INFINITY,
            resolved_untrusted: false,
        })
    }

    /// Turn on conflict analysis: register the pool on the local domain and take the
    /// global domain as a copy of its committed baseline.
    fn enable_conflicts(&mut self) {
        let cp = self.eval.dom.add_conflict_pool();
        let cpi = self.eval.dom.add_cutpool();
        let (model_nnz, model_rows) = {
            let m0 = self.ctx.n_orig_rows.min(self.ctx.m_w);
            let nnz: usize = (0..self.ctx.ns)
                .map(|j| self.ctx.csc_rc.col(j).0.iter().filter(|&&i| i < m0).count())
                .sum();
            (nnz.max(1) as i64, m0.max(1) as i64)
        };
        // HiGHS `cutpool(numCol, mip_pool_age_limit = 30, mip_pool_soft_limit = 10000)`.
        let cutpool = CutPool::new(self.ctx.ns, 30, 10000, model_nnz, model_rows);
        let srows = StructRows::new(self.ctx, &self.eval.slacks);
        let mut gdom = self.eval.dom.clone();
        gdom.backtrack_to_global();
        gdom.set_global(true);
        self.eval.keep_ray = true;
        let synced = gdom.domain_change_stack().len();
        self.conf = Some(Conflicts {
            pool: ConflictPool::new(150, 10000),
            gdom,
            cp,
            synced,
            added: 0,
            proof_added: 0,
            global_fixings: 0,
            cutpool,
            cpi,
            srows,
            conflict_cuts: 0,
            pool_cuts: 0,
        });
    }

    /// A pool cut over the working columns into the propagated cut pool (HiGHS adds
    /// every separated cut to `mipdata_->cutpool`, which both domains propagate).
    fn propagate_pool_cut(&mut self, cut: &GomoryCut) {
        let Some(cf) = self.conf.as_mut() else {
            return;
        };
        if cf.gdom.infeasible() || self.eval.dom.infeasible() {
            return;
        }
        let Some((inds, vals, rhs)) = cf.srows.to_struct(cut) else {
            return;
        };
        let is_int = self.ctx.is_int_full;
        let integral = inds.iter().all(|&c| is_int[c as usize]);
        let mut obs = DomainObservers(vec![(&mut cf.gdom, cf.cpi), (&mut self.eval.dom, cf.cpi)]);
        if cf
            .cutpool
            .add_cut(&mut obs, &inds, &vals, rhs, integral, true, false)
            != -1
        {
            cf.pool_cuts += 1;
        }
    }

    /// A cut for the node-separation pool, deduplicated, and into the propagated
    /// cut pool under conflict analysis.
    fn pool_add(&mut self, cut: GomoryCut) {
        if self.pool_sigs.insert(cut_signature(&cut)) {
            self.propagate_pool_cut(&cut);
            self.pool.push(cut);
        }
    }

    /// HiGHS `HighsCutGeneration::generateConflict` on a dual proof `vals . x[inds]
    /// <= rhs` of the current (infeasible) node: a lifted cover or c-MIR cut from the
    /// proof complemented at the global bounds, post-processed, its coefficients
    /// tightened on the global domain (`tightenCoefficients`) and added to the cut
    /// pool, which both domains propagate (`HighsCutGeneration.cpp:1208-1299`). The
    /// cut is valid wherever the proof is: it is a valid inequality of the proof row
    /// over the global box.
    fn generate_conflict(&mut self, inds: &[usize], vals: &[f64], rhs: f64) {
        let ns = self.ctx.ns;
        let is_int = self.ctx.is_int_full;
        let Some(cf) = self.conf.as_mut() else {
            return;
        };
        if cf.gdom.infeasible() || inds.iter().any(|&j| j >= ns) {
            return;
        }
        let pinds: Vec<i32> = inds.iter().map(|&j| j as i32).collect();
        let glb: Vec<f64> = inds.iter().map(|&j| cf.gdom.col_lower[j]).collect();
        let gub: Vec<f64> = inds.iter().map(|&j| cf.gdom.col_upper[j]).collect();
        let llb: Vec<f64> = inds.iter().map(|&j| self.eval.dom.col_lower[j]).collect();
        let lub: Vec<f64> = inds.iter().map(|&j| self.eval.dom.col_upper[j]).collect();
        let col_integral: Vec<bool> = inds.iter().map(|&j| is_int[j]).collect();
        let p = CutGenParams {
            feastol: 1e-6,
            epsilon: 1e-9,
            num_cols: ns,
        };
        let mut gen = CutGenCore::seeded(p, 0, self.eval.lp_iters, cf.cutpool.num_cuts() as i32);
        let Some(core) = gen.generate_conflict_core(
            &pinds,
            vals,
            rhs,
            &glb,
            &gub,
            &llb,
            &lub,
            &col_integral,
            &self.q,
        ) else {
            return;
        };
        let gd = &cf.gdom;
        let lb = |c: i32| gd.col_lower[c as usize];
        let ub = |c: i32| gd.col_upper[c as usize];
        let isi = |c: i32| is_int[c as usize];
        let Some(cut) = gen.postprocess_cut(core, &lb, &ub, &isi) else {
            return;
        };
        let integral = cut.integral_support && cut.integral_coefficients;
        let (cinds, mut cvals, mut crhs) = (cut.inds, cut.vals, cut.rhs.to_f64());
        tighten_coefficients_with(&lb, &ub, &isi, 1e-6, &cinds, &mut cvals, &mut crhs);
        if cinds.is_empty() || !crhs.is_finite() {
            return;
        }
        let mut obs = DomainObservers(vec![(&mut cf.gdom, cf.cpi), (&mut self.eval.dom, cf.cpi)]);
        if cf
            .cutpool
            .add_cut(&mut obs, &cinds, &cvals, crhs, integral, true, true)
            == -1
        {
            return;
        }
        cf.conflict_cuts += 1;
        // Node separation reaches the cut through `cutpool.separate` (HiGHS
        // `separationRound`), which ages it out once it stops being violated. It used
        // to be copied into the never-pruned `pool` as well, whose per-node scan grew
        // with the run (market split, 80k nodes: 5.2x the wall without conflicts).
    }

    /// HiGHS `conflictPool.performAging()` (before each dive).
    fn age_conflicts(&mut self) {
        let Some(cf) = self.conf.as_mut() else {
            return;
        };
        let mut obs =
            ConflictDomainObservers(vec![(&mut self.eval.dom, cf.cp), (&mut cf.gdom, cf.cp)]);
        cf.pool.perform_aging(false, &mut obs);
        let mut obs = DomainObservers(vec![(&mut cf.gdom, cf.cpi), (&mut self.eval.dom, cf.cpi)]);
        cf.cutpool.perform_aging(&mut obs);
    }

    /// The conflicts the domains used while propagating (`resetAge`).
    fn reset_conflict_ages(&mut self) {
        let Some(cf) = self.conf.as_mut() else {
            return;
        };
        let resets: Vec<(usize, usize)> = self
            .eval
            .dom
            .take_conflict_age_resets()
            .into_iter()
            .chain(cf.gdom.take_conflict_age_resets())
            .collect();
        for (pool, c) in resets {
            if pool == cf.cp {
                cf.pool.reset_age(c);
            }
        }
        let cut_resets: Vec<(usize, usize)> = self
            .eval
            .dom
            .take_cut_age_resets()
            .into_iter()
            .chain(cf.gdom.take_cut_age_resets())
            .collect();
        for (pool, c) in cut_resets {
            if pool == cf.cpi && cf.cutpool.is_live(c as i32) {
                cf.cutpool.reset_age(c as i32);
            }
        }
    }

    /// `localdom.conflictAnalysis(conflictPool)` on an infeasible local domain.
    fn analyse_conflict(&mut self) {
        let lim = self.eval.cutoff - self.ctx.obj_const;
        let Some(cf) = self.conf.as_mut() else {
            return;
        };
        if !self.eval.dom.infeasible() {
            return;
        }
        cf.gdom.set_upper_limit(lim);
        let links = ConflictLinks {
            local: Some(cf.cp),
            global: Some(cf.cp),
        };
        let added = self.eval.dom.conflict_analysis(
            &mut cf.pool,
            links,
            &mut cf.gdom,
            &mut self.pc,
            &self.q,
        );
        cf.added += added.len() as u64;
        crate::profile::incr(crate::profile::Ctr::HighsMipSearchDomConflictCalls);
        if !added.is_empty() {
            crate::profile::incr(crate::profile::Ctr::HighsMipSearchDomConflictHits);
        }
        self.reset_conflict_ages();
    }

    /// `addInfeasibleConflict`: the verified Farkas ray of the infeasible node LP as
    /// a globally valid proof row over the structurals ([`farkas_proof_row`]),
    /// explained by conflict analysis.
    fn analyse_infeasible_lp(&mut self, proofs: Vec<(Vec<usize>, Vec<f64>, f64)>) {
        if self.conf.is_none() {
            return;
        }
        let lim = self.eval.cutoff - self.ctx.obj_const;
        for (inds, vals, rhs) in proofs {
            let cf = self.conf.as_mut().expect("checked above");
            cf.gdom.set_upper_limit(lim);
            let links = ConflictLinks {
                local: Some(cf.cp),
                global: Some(cf.cp),
            };
            let added = self.eval.dom.conflict_analysis_proof(
                &inds,
                &vals,
                rhs,
                &mut cf.pool,
                links,
                &mut cf.gdom,
                &mut self.pc,
                &self.q,
            );
            cf.proof_added += added.len() as u64;
            self.reset_conflict_ages();
            self.generate_conflict(&inds, &vals, rhs);
            if !added.is_empty() {
                crate::profile::incr(crate::profile::Ctr::HighsMipSearchProofHits);
                break;
            }
        }
    }

    /// Carry the global domain's bounds into the local domain's baseline (just after
    /// `backtrack_to_global`), as HiGHS's local domain starts from the global one.
    /// The baseline then equals the global domain's bounds, the condition conflict
    /// analysis rests on: a baseline bound has no stack position, so explanations
    /// treat it as globally valid.
    fn sync_global(&mut self) {
        let lim = self.eval.cutoff - self.ctx.obj_const;
        let Some(cf) = self.conf.as_mut() else {
            return;
        };
        cf.gdom.set_upper_limit(lim);
        cf.gdom.propagate();
        let len = cf.gdom.domain_change_stack().len();
        if cf.gdom.infeasible() || len == cf.synced {
            return;
        }
        let dom = &mut self.eval.dom;
        if dom.infeasible() {
            return;
        }
        assert!(
            dom.domain_change_stack().is_empty(),
            "sync_global needs the local domain at its global baseline"
        );
        cf.synced = len;
        let mut k = 0u64;
        for j in 0..dom.col_lower.len().min(cf.gdom.col_lower.len()) {
            let (gl, gu) = (cf.gdom.col_lower[j], cf.gdom.col_upper[j]);
            if gl > dom.col_lower[j] {
                dom.change_bound(
                    DomainChange {
                        boundval: gl,
                        column: j,
                        boundtype: BoundType::Lower,
                    },
                    Reason::Unspecified,
                );
                k += 1;
            }
            if gu < dom.col_upper[j] {
                dom.change_bound(
                    DomainChange {
                        boundval: gu,
                        column: j,
                        boundtype: BoundType::Upper,
                    },
                    Reason::Unspecified,
                );
                k += 1;
            }
        }
        if k > 0 {
            assert!(
                !dom.infeasible(),
                "the global domain's bounds are consistent"
            );
            dom.commit_as_global();
            cf.global_fixings += k;
        }
    }

    /// Step 2.1: HiGHS's search starts with the root's whole cut pool (every separated
    /// cut, and the ones root cleanup dropped from the LP); with persistent cuts on,
    /// so does this one (the pool feeds the LP when a node violates a cut).
    fn seed_pool(&mut self, seed_pool: Vec<GomoryCut>) {
        if self.conf.is_some() {
            // HiGHS's search domains propagate the root's whole cut pool.
            let root: Vec<GomoryCut> = self
                .ctx
                .gmi_pool
                .iter()
                .chain(seed_pool.iter())
                .cloned()
                .collect();
            for cut in &root {
                self.propagate_pool_cut(cut);
            }
        }
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
    cliques: Option<(crate::lp::highs_cutgen::clique::CliqueTable, Vec<f64>)>,
    mut hsep: Option<&'a mut crate::lp::highs_cutgen::bridge::RootSeparation>,
) -> Option<Outcome> {
    #[cfg(test)]
    RUNS.with(|r| r.set(r.get() + 1));
    // `DISCOPT_NODEBOX_FILE` (diagnostic, top-level solve only): evaluate external node
    // boxes on environments of their own, built exactly as this search's, before the
    // search starts; the search below is untouched by it.
    if top_level {
        nodebox::harness(
            ctx,
            lim,
            initial.as_ref(),
            &seed_pool,
            hsep.as_deref_mut(),
            root_bound,
        );
    }
    let mut env = HighsEnv::new(ctx, lim, sub_mip, dbg, refcheck::RefCheck::load(ctx))?;
    if let Some(hs) = hsep.as_deref_mut() {
        hs.begin_search();
    }
    env.hsep = hsep;
    env.root_obj = root_bound;
    // The root's `setIterationLimit(max(10000, 10 * avgrootlpiters))`, in force until
    // the first `processNodes` sets the node limit.
    let root_limit = 10_000usize.max((10.0 * env.avg_root_iters) as usize);
    env.set_iteration_limit(root_limit);
    if !top_level {
        // `DISCOPT_MILP_PREINC_SCORE` acts on the top-level search only.
        env.preinc_score = false;
        env.pc.set_feasibility_weight(HIGHS_FEASIBILITY_WEIGHT);
    }
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
    // `DISCOPT_MILP_CLIQUE_DOMAIN`: `mipdata_->cliquetable` and `mipdata_->implications`
    // on the local domain before the global one is cloned from it
    // (`enable_conflicts`), so both domains run `addImplications` and
    // `applyImplications`. The table is the root separation's; probing extends it
    // first (`separateImpliedBounds`' probing loop over the root LP's fractional
    // binaries), on the domain at its global baseline.
    let mut probe_infeasible = false;
    if let Some((t, x)) = cliques {
        assert_eq!(
            t.num_cols(),
            ctx.ns,
            "the clique table must index the search's structural columns"
        );
        assert!(
            env.eval.dom.domain_change_stack().is_empty(),
            "probing needs the search's domain at its global baseline"
        );
        let feastol = env.eval.dom.feastol();
        // `separateImpliedBounds` probes the binaries the LP solution leaves
        // fractional; here, those of the final root LP.
        let cands: Vec<usize> = (0..ctx.ns.min(x.len()))
            .filter(|&j| ctx.is_int[j] && (x[j] - x[j].round()).abs() > feastol)
            .collect();
        // HiGHS probes every binary in presolve first (`HPresolve::runProbing`), at
        // its default `random_seed = 0`.
        let pp = crate::lp::highs_cutgen::probing::PresolveProbing {
            submip: lim.submip,
            random_seed: 0,
            nrows: ctx.n_orig_rows.min(ctx.m_w),
        };
        let st = crate::lp::highs_cutgen::probing::probe_and_attach(
            &mut env.eval.dom,
            t,
            &cands,
            Some(pp),
        );
        {
            use crate::profile::{incr_by, Ctr};
            incr_by(Ctr::MilpProbeProbed, st.probed as u64);
            incr_by(Ctr::MilpProbeReductions, st.reductions as u64);
            incr_by(Ctr::MilpProbeEntries, st.new_entries.max(0) as u64);
        }
        probe_infeasible = env.eval.dom.infeasible();
    }
    if conflict_enabled() {
        env.enable_conflicts();
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
    // Probing proved the global box holds no (improving) point: the queue starts
    // empty, as when conflict propagation empties the global domain.
    if !probe_infeasible {
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
    }
    let mut search = Search {
        count_tree_weight: true,
        next_id: root_id + 1,
        ..Default::default()
    };
    let mut ml = MainLoop::default();
    let mut root = true;
    while !env.queue_is_empty() {
        env.age_conflicts();
        let stop = ml.round(&mut env, &mut search, root, NO_NODE_LIMIT);
        if env.global_infeasible() {
            // Conflict propagation emptied the global domain: every conflict is
            // valid for each improving point, so no open node holds one.
            env.q.clear();
            break;
        }
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
        preinc_branchings: env.preinc_branchings,
        bound_hook_calls: env.bound_hook_calls,
        bound_hook_raised: env.bound_hook_raised,
        cut_rows_added: env.cut_rows_added,
        cut_rows_deleted: env.cut_rows_deleted,
        cut_epochs: env.lp.as_ref().map_or(0, |l| l.epochs),
        seeded_pool: env.seeded_pool,
        eval: env.eval.stats.clone(),
        conflicts: env
            .conf
            .as_ref()
            .map_or((0, 0, 0), |c| (c.added, c.proof_added, c.global_fixings)),
        conflict_cuts: env
            .conf
            .as_ref()
            .map_or((0, 0), |c| (c.conflict_cuts, c.pool_cuts)),
    })
}

#[cfg(test)]
mod proof_tests {
    use super::*;

    /// HiGHS `storeDualInfProof` skips a row whose ray weight needs its infinite side
    /// instead of giving up on the proof. Working form `A [x; s] = b`, `x in [0,1]^2`:
    /// row 0 `x0 + x1 - s0 = 0`, `s0 >= 3` (the infeasible row); row 1 `x0 - s1 = 0`,
    /// `s1 <= 5` (a row with only an upper side). The ray `y = (1, 0.5)` gives row 1's
    /// slack the weight `-0.5`, which needs `s1`'s missing lower side. Before the fix
    /// the whole proof was `None`; now row 1 is left out and the proof is row 0,
    /// `-(x0 + x1) <= -3`, which separates the box. Soundness: every point satisfying
    /// the rows (structurals sampled in `[-6, 6]^2`, far past their box, since the
    /// proof must not rely on them) satisfies the proof.
    #[test]
    fn proof_row_drops_a_row_whose_needed_side_is_infinite() {
        let (m, n, ns) = (2, 4, 2);
        // columns x0 x1 s0 s1
        let a = [1.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0, -1.0];
        let csc = SparseCols::from_dense(&a, m, n);
        let b = [0.0, 0.0];
        let l = [0.0, 0.0, 3.0, -INF];
        let u = [1.0, 1.0, INF, 5.0];
        let (inds, vals, rhs) =
            proof_row_from_dual(&csc, m, n, ns, &b, &l, &u, vec![1.0, 0.5]).expect("a proof");
        // The proof row is violated by every point of the structural box.
        let max_act = inds
            .iter()
            .zip(&vals)
            .map(|(&j, &v)| if v > 0.0 { v * u[j] } else { v * l[j] })
            .sum::<f64>();
        let min_act = inds
            .iter()
            .zip(&vals)
            .map(|(&j, &v)| if v > 0.0 { v * l[j] } else { v * u[j] })
            .sum::<f64>();
        assert!(max_act >= min_act);
        assert!(
            min_act > rhs,
            "proof {vals:?} <= {rhs} does not separate the box"
        );
        let mut checked = 0;
        for i0 in -24..=24 {
            for i1 in -24..=24 {
                let x = [i0 as f64 * 0.25, i1 as f64 * 0.25];
                let s0 = x[0] + x[1];
                let s1 = x[0];
                if s0 < l[2] || s1 > u[3] {
                    continue;
                }
                let act: f64 = inds.iter().zip(&vals).map(|(&j, &v)| v * x[j]).sum();
                assert!(act <= rhs + 1e-12, "row-feasible {x:?} violates the proof");
                checked += 1;
            }
        }
        assert!(checked > 100, "only {checked} row-feasible points checked");
    }
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
