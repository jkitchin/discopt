//! Rust-internal MILP branch-and-bound driven by the warm-started simplex.
//!
//! This runs the *entire* pure-MILP solve in Rust (roadmap P4): it reuses the
//! existing [`TreeManager`] (selection, pruning, branching, pseudocosts,
//! incumbent, gap) but solves each node's LP relaxation in-process with the
//! simplex — the root cold ([`solve_lp`]) and every child warm
//! ([`solve_lp_warm`]) from the basis it inherited from its parent. The optimal
//! basis of each node is stored back on the node so its children inherit it.
//!
//! No per-node Python round-trip: the whole search is one call, exposed to
//! Python by a single PyO3 entry. MINLP/MIQP/NLP are untouched (they keep the
//! POUNCE/JAX path); only linear MILP reaches here.

use std::collections::{HashMap, HashSet};

use crate::bnb::branching::VarBranchInfo;
use crate::bnb::node::NodeId;
use crate::bnb::pool::SelectionStrategy;
use crate::bnb::tree_manager::{NodeResult, TreeManager};
use crate::lp::basis::{Basis, AT_LOWER, AT_UPPER, BASIC};
use crate::lp::cover::separate_cover_csc;
use crate::lp::crossover::LpView;
use crate::lp::cut_select::select_cuts;
use crate::lp::gomory::{separate_gomory_cols, GomoryCut};
use crate::lp::simplex::linsolve::{FeralLU, LinearSolver};
use crate::lp::simplex::sparse::SparseCols;
use crate::lp::simplex::{
    solve_lp, solve_lp_cols, solve_lp_cols_scaled, solve_lp_warm, solve_lp_warm_scaled_csc,
    tighten_bounds_csc, LpStatus, PreparedDual, Scaling, SimplexOptions,
};

const INF: f64 = 1e20;
const INFEAS_SENTINEL: f64 = 1e30;
const INT_TOL: f64 = 1e-5;

/// Minimum efficacy (normalized violation) for a cut to be worth adding under
/// cut selection — below this it barely separates the point and only bloats the LP.
const CUT_MIN_EFFICACY: f64 = 1e-4;
/// Drop a candidate cut whose direction is more than this parallel (|cos|) to an
/// already-selected cut — keeps the kept set spanning diverse faces.
const CUT_MAX_PARALLEL: f64 = 0.99;

/// Terminal status of a MILP solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MilpStatus {
    /// Proven optimal: the final frontier gap (which folds in every valid
    /// inherited bound, including the unresolved-fathom floor; #598) closed
    /// within tolerance, with no search truncation.
    Optimal,
    /// A feasible incumbent found but optimality not proven (limit / numerical).
    Feasible,
    /// No feasible integer solution exists.
    Infeasible,
    /// The relaxation is unbounded.
    Unbounded,
    /// Node limit reached without proving optimality.
    NodeLimit,
}

/// Result of a Rust-internal MILP solve.
#[derive(Debug, Clone)]
pub struct MilpResult {
    /// Terminal status.
    pub status: MilpStatus,
    /// Best incumbent over the structural variables (length `n_struct`).
    pub x: Vec<f64>,
    /// Incumbent objective `cᵀx + obj_const` (when an incumbent exists).
    pub obj: f64,
    /// Global lower bound at termination.
    pub bound: f64,
    /// Total B&B nodes created.
    pub nodes: usize,
    /// Total simplex pivots across all node solves.
    pub lp_iters: usize,
    /// Number of times a [`MilpLazyHook`] separator was actually invoked.
    ///
    /// Zero whenever no hook was attached. With a hook attached it is the
    /// anti-vacuity counter (CLAUDE.md §6): a caller that wires a separator in
    /// and gets `lazy_calls == 0` back learns that the separator never saw a
    /// point, which is a different failure from "the separator accepted
    /// everything" and must not be reported as one.
    pub lazy_calls: usize,
    /// Number of times a node was re-queued because the separator vetoed its
    /// integral relaxation solution. Zero without a hook. Together with
    /// `lazy_calls` this distinguishes "the separator accepted everything" from
    /// "the separator re-searched boxes", which is the signal the OA caller
    /// needs to tell a converged single-tree run from a vacuous one.
    pub lazy_requeues: usize,
    /// Number of times a [`MilpNodeHook`] separator was actually invoked at a
    /// **fractional** node relaxation. Zero whenever no node hook was attached.
    ///
    /// The anti-vacuity counter for fractional separation (CLAUDE.md §6): a
    /// caller that wires a node separator in and gets `node_calls == 0` back
    /// learns the separator never saw a fractional point — a different failure
    /// from "the separator found nothing to cut", and one that must never be
    /// reported as convergence.
    pub node_calls: usize,
    /// Rows a [`MilpNodeHook`] separator returned that were actually folded into
    /// the shared relaxation (post-dedup, post-cap). `node_calls > 0` with
    /// `node_cuts_added == 0` means the separator ran and cut nothing.
    pub node_cuts_added: usize,
}

/// A node-lifecycle checkpoint fired to an attached [`MilpDebugHook`].
///
/// Mirrors the Python-side `discopt.debug.Checkpoint` so the pure-Rust MILP
/// fast-path is inspectable by the same debugger that drives the spatial /
/// MIQP / NLP-BB loops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MilpCheckpoint {
    /// Top of a batch iteration.
    IterStart,
    /// The batch of open nodes was just exported — their boxes are available.
    AfterSelect,
    /// After the batch's results were imported and prune/branch/fathom ran.
    AfterProcess,
    /// A strictly-better incumbent was just adopted.
    IncumbentFound,
    /// Once, after the search loop exits (final / limit / infeasible).
    Terminated,
}

/// Aggregate solver state passed to a debug hook at a checkpoint. Read-only.
///
/// The `batch_*` fields are populated only at [`MilpCheckpoint::AfterSelect`],
/// where the current batch of open-node boxes is in scope; they are `None`
/// elsewhere. The lifetime `'a` borrows those boxes from the export batch.
#[derive(Debug, Clone, Copy)]
pub struct MilpDebugState<'a> {
    /// Which checkpoint fired.
    pub checkpoint: MilpCheckpoint,
    /// Batch-iteration counter (0-based), mirroring the Python loops.
    pub iteration: usize,
    /// Total B&B nodes created so far.
    pub total_nodes: usize,
    /// Open nodes remaining in the frontier.
    pub open_nodes: usize,
    /// Incumbent objective (internal min sense), or `None` if none yet.
    pub incumbent: Option<f64>,
    /// Global lower (dual) bound.
    pub bound: f64,
    /// Current relative optimality gap.
    pub gap: f64,
    /// Wall-clock seconds since the solve started.
    pub elapsed: f64,
    /// Number of structural variables (box length reference).
    pub n_vars: usize,
    /// Per-node lower-bound boxes of the exported batch (AfterSelect only).
    pub batch_lb: Option<&'a [Vec<f64>]>,
    /// Per-node upper-bound boxes of the exported batch (AfterSelect only).
    pub batch_ub: Option<&'a [Vec<f64>]>,
    /// Node ids of the exported batch (AfterSelect only).
    pub batch_ids: Option<&'a [NodeId]>,
}

/// What a debug hook tells the search to do after a checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MilpDebugControl {
    /// Keep searching.
    Continue,
    /// Stop the search now (graceful — a valid uncertified result is built).
    Stop,
}

/// A debugger attached to the Rust MILP search. Implemented on the Python side
/// by a GIL-reacquiring adapter over a Python callable. Must be `Sync` because
/// the solve runs under `Python::allow_threads`.
///
/// **Zero effect when absent:** every fire-site is gated on `Option::is_some`,
/// so a `None` hook leaves the search bit-for-bit identical (bound-neutral).
pub trait MilpDebugHook: Sync {
    /// Called at each fired checkpoint; return [`MilpDebugControl::Stop`] to
    /// end the search gracefully.
    fn checkpoint(&self, state: &MilpDebugState<'_>) -> MilpDebugControl;
}

/// What a [`MilpLazyHook`] says about an integer-feasible point.
#[derive(Debug, Clone)]
pub enum MilpLazyVerdict {
    /// No lazy constraint is violated: adopt the point exactly as an unhooked
    /// search would.
    Accept,
    /// The point violates a lazy constraint. Each returned row is a
    /// **globally valid** `coeffs · x ≥ rhs` (coefficients indexed by
    /// structural column; trailing columns are implicitly zero) that cuts this
    /// point off. Never empty — an empty separation is [`Self::Accept`].
    Reject(Vec<GomoryCut>),
    /// The separator itself failed (e.g. the Python callable raised). The
    /// search stops and the result is reported uncertified. A failed separator
    /// must never be treated as "no cut" — that silently drops a constraint
    /// the caller believes is being enforced (CLAUDE.md §7).
    Failed,
}

/// A lazy-constraint separator attached to the Rust MILP search: the native
/// equivalent of a Gurobi `MIPSOL` callback, and the mechanism single-tree
/// LP/NLP branch-and-bound (`mip_nlp_method="lp_nlp_bb"`) runs on (#1060).
///
/// `separate` is called with each integer-feasible **structural** point the
/// search finds — a node relaxation that came out integral, a primal
/// heuristic's candidate, or a caller-supplied seed — *before* that point can
/// become the incumbent. Must be `Sync`: the solve runs under
/// `Python::allow_threads`, and the Python adapter re-acquires the GIL.
///
/// **Zero effect when absent:** every fire-site is gated on `Option::is_some`,
/// so a `None` hook leaves the search bit-for-bit identical (bound-neutral).
pub trait MilpLazyHook: Sync {
    /// Separate `x` (length `n_struct`) against the caller's lazy constraints.
    fn separate(&self, x: &[f64]) -> MilpLazyVerdict;
}

/// What a [`MilpNodeHook`] says about a **fractional** node relaxation solution.
#[derive(Debug, Clone)]
pub enum MilpNodeVerdict {
    /// Nothing to separate at this point: the node's LP result is imported
    /// exactly as an unhooked search would import it.
    None,
    /// Globally valid `coeffs · x ≥ rhs` rows (coefficients indexed by
    /// structural column; trailing columns implicitly zero) that cut this
    /// fractional point off. Never empty — an empty separation is [`Self::None`].
    Cuts(Vec<GomoryCut>),
    /// The separator itself failed (e.g. the Python callable raised). The search
    /// stops and the result is reported uncertified — a failed separator must
    /// never be silently read as "found nothing" (CLAUDE.md §7).
    Failed,
}

/// A **fractional**-node cut separator attached to the Rust MILP search: the
/// native equivalent of a Gurobi `MIPNODE` user-cut callback, and the missing
/// half of single-tree LP/NLP branch-and-cut (#1141).
///
/// [`MilpLazyHook`] fires only at *integer-feasible* points, so a caller whose
/// relaxation ignores a nonlinear constraint pays a full NLP at every integer
/// proposal and gets no help at the fractional nodes in between. This hook fires
/// at the fractional ones: for a convex MINLP the caller returns the
/// first-order linearization `g(x̄) + ∇g(x̄)·(x − x̄) ≤ 0` of each violated
/// constraint, which is a supporting hyperplane and therefore globally valid.
/// That is exactly what SCIP does, and what makes an LP node cheap *and* tight
/// instead of cheap *or* tight.
///
/// **Soundness contract.** Every returned row must be valid for the WHOLE
/// problem, not just this node's box: the driver folds it into the shared
/// matrix. A row valid only under the node's local bounds would cut feasible
/// points out of sibling subtrees and produce a false certificate.
///
/// **Not a veto.** Unlike a lazy veto, a node separation never rejects a point
/// that could become the incumbent — a fractional LP solution is not a
/// candidate solution in the first place. So exhausting the per-node round
/// budget is benign: the node's own (valid) LP bound is imported and the search
/// continues, with certification untouched.
///
/// Must be `Sync`: the solve runs under `Python::allow_threads` and the Python
/// adapter re-acquires the GIL.
///
/// **Zero effect when absent:** every fire-site is gated on `Option::is_some`
/// AND on `MilpOptions::node_hook_rounds > 0`, so a `None` hook leaves the
/// search bit-for-bit identical (bound-neutral).
pub trait MilpNodeHook: Sync {
    /// Separate the fractional point `x` (length `n_struct`).
    fn separate_node(&self, x: &[f64]) -> MilpNodeVerdict;
}

/// How many times one node may be re-solved after a lazy veto before the search
/// gives up on refining it.
///
/// Each veto folds in a cut that provably separates the vetoed point, so a
/// node's re-solve cannot return the same point and the loop terminates on any
/// well-behaved separator (finite outer approximation). The cap only bounds a
/// pathological one — a separator that vetoes without separating. Exhausting it
/// sentinels the node, which is a *non-rigorous* fathom, so the driver also
/// drops certification (see the call site): the result may then be `Feasible`,
/// never a false `Optimal`.
const LAZY_REQUEUE_CAP: u32 = 1000;

/// Hard cap on how many *non-root* continuous-repair dives one MILP solve may
/// run (#1060). The dive only fires while the search has no incumbent at all, so
/// on a model where it succeeds it stops by itself after one or two firings; the
/// cap bounds the other case — a model where the dive can never repair anything —
/// so a hopeless dive cannot become a per-node tax on a million-node search.
const DIVE_NO_INCUMBENT_CAP: usize = 64;

/// Batch stride for the no-incumbent dive schedule (#1060), read once from
/// `DISCOPT_MILP_DIVE_STRIDE`.
///
/// `0` restores the legacy root-only dive exactly. Any `n > 0` lets the dive
/// re-fire on the first node of every `n`-th batch **while the tree holds no
/// incumbent** — the regime in which every node is doing bound-free work anyway,
/// because there is nothing to prune against.
///
/// Why this exists: `try_dive_repair`'s own comment predicts that on a
/// weak-relaxation (big-M) master "rounding ... finds no incumbent at all and the
/// search runs with no bound-based pruning (tree explosion)" — and then runs the
/// dive only at the root. Measured on the Quesada-Grossmann single-tree master
/// for `rsyn0840m` (issue #1060): 150,193 nodes, exactly **one** integer-feasible
/// candidate reaching the lazy callback in 60 s, final incumbent 103.5% short of
/// the reference optimum. `rsyn0805m`, which does solve, surfaces 24.
///
/// Read once per process: the schedule changes which nodes are explored, so
/// flipping it mid-solve would make an A/B measurement incomparable with itself.
fn dive_stride() -> usize {
    static STRIDE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *STRIDE.get_or_init(|| {
        let raw = std::env::var("DISCOPT_MILP_DIVE_STRIDE").unwrap_or_default();
        parse_dive_stride(&raw).unwrap_or_else(|e| panic!("{e}"))
    })
}

/// The parse table behind [`dive_stride`]. An unrecognized value is **refused**,
/// not defaulted: an A/B arm whose value silently reads as the default makes the
/// harness measure one arm twice (the `DISCOPT_LP_REFAC_INTERVAL` precedent).
fn parse_dive_stride(raw: &str) -> Result<usize, String> {
    let t = raw.trim();
    if t.is_empty() {
        return Ok(DIVE_STRIDE_DEFAULT);
    }
    t.parse::<usize>().map_err(|_| {
        format!("DISCOPT_MILP_DIVE_STRIDE: expected a non-negative integer, got {t:?}")
    })
}

/// Default batch stride. Default-off (`0` = legacy root-only) until the CLAUDE.md
/// §5 differential panel graduates it.
const DIVE_STRIDE_DEFAULT: usize = 0;

/// Does this batch get a continuous-repair dive away from the root?
///
/// Pure so the schedule can be asserted directly rather than inferred from a
/// solve's node count (CLAUDE.md §6). Four independent gates, all of which must
/// hold:
///
/// * `stride > 0` — the feature is on at all. `0` is the legacy root-only dive,
///   under which this function is `false` for every batch and the driver is
///   byte-identical to the version before the schedule existed.
/// * `!has_incumbent` — the tree holds nothing to prune against, so every node in
///   the batch is doing bound-free work. Once any incumbent lands, the ordinary
///   search (warm starts, cuts, reduced-cost fixing) is strictly the better use of
///   the budget and the schedule switches itself off.
/// * `nonroot_dives < DIVE_NO_INCUMBENT_CAP` — a model the dive can never repair
///   cannot turn it into a per-node tax.
/// * `batch_index % stride == 0` — spread the budget over batches instead of
///   spending it all in the first few.
fn dive_batch_eligible(
    stride: usize,
    batch_index: usize,
    has_incumbent: bool,
    nonroot_dives: usize,
) -> bool {
    stride > 0
        && !has_incumbent
        && nonroot_dives < DIVE_NO_INCUMBENT_CAP
        && batch_index % stride == 0
}

/// Options for the MILP driver.
pub struct MilpOptions {
    /// Number of structural (model) variables; columns `[n_struct, n)` are slacks.
    pub n_struct: usize,
    /// Structural column indices that are integer-constrained.
    pub integer_cols: Vec<usize>,
    /// Node-creation cap.
    pub max_nodes: usize,
    /// Optional wall-time cap in seconds. `None` = unlimited. Checked at each
    /// batch boundary of the B&B search; on expiry the search stops and returns
    /// the incumbent with a valid (uncertified) dual bound — never a false
    /// "optimal". Keeps a single atomic MILP solve from overrunning a caller's
    /// `time_limit` (the McCormick LP relaxer node solve was the worst offender).
    pub time_limit_s: Option<f64>,
    /// Relative gap tolerance for proving optimality.
    pub gap_tol: f64,
    /// Max Gomory mixed-integer cuts to add at the root (0 disables), summed
    /// over rounds. Derived from the root's *native* simplex basis — no
    /// crossover needed.
    pub root_cuts: usize,
    /// Max root cut rounds (separate → re-solve → separate). 1 = single pass.
    pub cut_rounds: usize,
    /// Separate Gomory mixed-integer cuts (off the tableau) in the root loop.
    /// GMI cuts are typically **dense** (a tableau row `B⁻¹A` mixes all columns),
    /// so on a sparse-row model they densify the cut-augmented matrix and make
    /// every node's LP re-solve expensive — erasing the cut benefit in wall time.
    /// Disable to keep cuts sparse (cover cuts are row-local), preserving the
    /// sparse-LP fast path. `true` keeps GMI (good on dense models where tableau
    /// cuts add bound the sparse cover cuts miss).
    pub gmi_cuts: bool,
    /// Apply efficacy + orthogonality cut selection ([`crate::lp::cut_select`])
    /// to each round's candidate cuts: keep only the strongest, most diverse few
    /// (up to the remaining `root_cuts` budget) instead of every cut found. With
    /// a small `root_cuts` cap and many `cut_rounds`, this keeps the active cut
    /// set small while still iterating — the win on sparse-row MILPs, where cuts
    /// close the node gap but carrying all of them is too expensive per node.
    pub cut_select: bool,
    /// Separate globally-valid cover cuts at fractional nodes into a shared pool.
    pub node_cuts: bool,
    /// Cap on the total number of pooled cuts (root + node).
    pub max_pool_cuts: usize,
    /// Rounding primal heuristic at fractional nodes (early incumbents).
    pub heuristics: bool,
    /// Root feasibility-based bound tightening (sound, dimension-preserving).
    pub presolve: bool,
    /// Limited strong branching on unreliable candidates (reliability branching).
    /// Probe objective degradations are fed back into the shared pseudocosts (see
    /// [`TreeManager::record_sb_observations`]) so probed variables graduate and
    /// stop being re-probed — the reliability-branching feedback loop.
    pub strong_branch: bool,
    /// Run feasibility-based bound tightening (FBBT) at every node, not just the
    /// root. Deep in the tree, branching tightens constraint slacks, so FBBT
    /// fixes/contracts further integer variables that the LP relaxation leaves
    /// fractional — pruning infeasible subtrees and shrinking children (which
    /// inherit the tightened bounds). Sound: a pure contraction, no postsolve.
    pub node_propagation: bool,
    /// Reduced-cost (objective) fixing at every node: using the node's LP dual
    /// bound `z`, the incumbent `U`, and each nonbasic integer variable's reduced
    /// cost `d`, fix the variable's bound when moving it off its current bound
    /// would push the objective to/past `U` (no improving solution can). This is
    /// the dominant node lever in the proving phase (where `U − z` is small) and
    /// is what the LP relaxation alone leaves on the table. Sound: only removes
    /// solutions no better than the incumbent. Children inherit the fixings.
    pub reduced_cost_fixing: bool,
    /// Max candidates probed per node when strong branching.
    pub sb_max_cands: usize,
    /// Only strong-branch while the tree is smaller than this many nodes — the
    /// early region where branching choices shape the whole search. Beyond it,
    /// matured pseudocosts decide (avoids probing overhead deep in large trees).
    pub sb_node_budget: usize,
    /// Optional caller-supplied point over the structural columns (length
    /// `n_struct`) to seed the incumbent before the search starts — e.g. a
    /// warm start mapped through an exact reformulation, or a primal
    /// heuristic's solution. The driver VALIDATES it (integrality, bounds,
    /// original-row feasibility) and recomputes the objective from `c`; an
    /// invalid seed is silently ignored. The caller's point is never trusted:
    /// an infeasible incumbent would prune the true optimum — a false
    /// certificate.
    pub initial_incumbent: Option<Vec<f64>>,
    /// Max separation rounds one node may run against an attached
    /// [`MilpNodeHook`] before its LP result is imported as-is. `0` disables
    /// fractional separation entirely (the default, and what makes an attached
    /// hook a no-op unless the caller asks for it).
    ///
    /// Each round costs one hook call plus one warm re-solve of the node LP, and
    /// buys a tighter bound at that node (and, since the rows are global, at
    /// every later node). A small budget is the point: the cuts are global, so
    /// what one node does not separate the next one will.
    pub node_hook_rounds: usize,
    /// Cap on the total number of rows an attached [`MilpNodeHook`] may fold
    /// into the shared relaxation over the whole solve. Unlike a lazy cut — which
    /// is mandatory, because it is the only thing keeping a vetoed point out — a
    /// fractional cut is optional: it only tightens. So it is budgeted, and once
    /// the budget is spent the hook stops being called and the search runs on the
    /// relaxation it has.
    pub node_hook_cut_cap: usize,
    /// Wall-clock ceiling for the ROOT CUT LOOP alone, in seconds.
    ///
    /// `None` (the value at every existing call site) leaves the loop bounded
    /// only by `root_cuts` / `cut_rounds` / tailing-off, exactly as before.
    ///
    /// A value makes a many-round configuration safe to ask for. Raising
    /// `cut_rounds` is how a weak root relaxation gets closed — on a lot-sizing
    /// model the default single pass leaves the root bound at a fraction of the
    /// optimum — but the separation phase is worth several seconds only if it
    /// still leaves time to branch. This is the knob that says how much of the
    /// budget the root may spend, so the round cap can be set by what the model
    /// needs rather than by what a worst case can afford.
    ///
    /// Checked at the top of each round, so a round already in flight always
    /// completes and the loop never abandons a partially augmented matrix. It
    /// therefore bounds *entry* into rounds, not the duration of one round; a
    /// single very slow round can still overrun it, bounded in turn by
    /// `time_limit_s` through the LP's own deadline.
    ///
    /// This is not the ceiling on the whole solve -- that stays `time_limit_s`,
    /// which the loop also honors (see the guard at the top of the round).
    pub root_cut_time_s: Option<f64>,

    /// Drop root cuts that are slack at the final root LP before the tree starts.
    ///
    /// Every root cut rides in every node LP for the rest of the solve. A cut that
    /// is not tight at the root optimum has a zero dual multiplier there, so
    /// removing it leaves the root bound and its certificate untouched while
    /// taking a row out of every node — the measured cost of NOT doing this is
    /// severe (30-instance MILP panel, `root_cuts=500, cut_rounds=50`: node counts
    /// fell 2.4x but total wall rose from 103 s to 145 s, and `cflp_s101` went
    /// from 0.04 s to 5.85 s at an unchanged node count).
    ///
    /// Deeper nodes may get a weaker bound than they would with the full set —
    /// this is bound-CHANGING under CLAUDE.md §5, never bound-INVALIDATING, since
    /// what remains is a subset of globally valid cuts. Kept as a switch so the
    /// two can be measured against each other in one interleaved process.
    ///
    /// Inert unless `cut_rounds > 1`: with a single round every cut was separated
    /// off the one root solve and is violated there, so nothing is ever slack.
    pub root_cut_prune: bool,
    /// LP solver options.
    pub simplex: SimplexOptions,
}

/// Map the search's terminal state to a [`MilpStatus`]. Pure so it can be
/// unit-tested against the exact orphaned-node scenario (C-2) without driving a
/// full solve.
///
/// The C-2 invariant lives here: `Infeasible` is returned **only** on a rigorous
/// empty-tree proof — `tree_finished && !search_incomplete && !tree_unresolved`.
/// `search_incomplete` is true whenever a node was deferred un-solved (deadline),
/// which leaves it popped off the heap and `Evaluated`, invisible to the tree's
/// `open_count()`. `tree_unresolved` is true whenever the tree fathomed a node
/// without a proof (a failed relaxation with no branch direction left — the
/// `bound_unresolved` pin of #467 or the `unresolved_floor` of #598): its
/// subtree was never searched, so an empty tree is not an emptiness proof. In
/// either case the honest terminus is a limit status, never a false
/// "infeasible".
///
/// `Optimal` requires the final gap to actually be closed (`gap_closed` is
/// computed from the tree's frontier bound, which already folds in the #598
/// unresolved floor) with no search truncation (`gap_certified`). A finished
/// tree does NOT imply a closed gap: an unresolved-floor fathom can drain the
/// open set while the bound honestly stays below the incumbent, and that must
/// exit `Feasible`, not `Optimal`. (Conversely, a rigorously drained tree with
/// an incumbent always collapses the bound to the incumbent — gap 0 — so
/// dropping the old `tree_finished ||` disjunct loses no true certificate.)
// Eight orthogonal terminal-state bits; a parameter struct would only obscure
// the truth table this function IS.
#[allow(clippy::too_many_arguments)]
fn decide_status(
    unbounded: bool,
    has_inc: bool,
    tree_finished: bool,
    search_incomplete: bool,
    tree_unresolved: bool,
    gap_closed: bool,
    gap_certified: bool,
    node_limit_hit: bool,
) -> MilpStatus {
    if unbounded {
        MilpStatus::Unbounded
    } else if !has_inc {
        if tree_finished && !search_incomplete && !tree_unresolved {
            MilpStatus::Infeasible
        } else {
            MilpStatus::NodeLimit
        }
    } else if gap_closed && gap_certified {
        MilpStatus::Optimal
    } else if node_limit_hit {
        MilpStatus::NodeLimit
    } else {
        MilpStatus::Feasible
    }
}

/// Solve `min cᵀx + obj_const s.t. A x = b, l ≤ x ≤ u` with `integer_cols`
/// integer-constrained, by Rust-internal warm-started-simplex branch and bound.
pub fn solve_milp(lp: &LpView<'_>, b: &[f64], obj_const: f64, opts: &MilpOptions) -> MilpResult {
    // Dense entry: build the working CSC from `lp.a` once and hand it to the fully
    // sparse driver core. The CSC entry (`solve_milp_csc`) skips this densify.
    solve_milp_hooked(
        SparseCols::from_dense(lp.a, lp.m, lp.n),
        lp.m,
        lp.n,
        lp.c,
        lp.l,
        lp.u,
        b,
        obj_const,
        opts,
        None,
    )
}

/// The MILP branch-and-bound driver core, taking the constraint matrix as a
/// column-major [`SparseCols`] (`m` rows, `n` cols) — **no dense `m×n` matrix is ever
/// materialized** (docs/dev/sparse-milp-plan.md T3b5/T4). `hook` is an optional
/// interactive-debugger hook; when `None` this is bit-for-bit identical to a plain
/// solve (all fire-sites short-circuit).
#[allow(clippy::too_many_arguments)]
pub fn solve_milp_hooked(
    csc_w: SparseCols,
    m: usize,
    n: usize,
    c: &[f64],
    l: &[f64],
    u: &[f64],
    b: &[f64],
    obj_const: f64,
    opts: &MilpOptions,
    hook: Option<&dyn MilpDebugHook>,
) -> MilpResult {
    solve_milp_lazy_hooked(csc_w, m, n, c, l, u, b, obj_const, opts, hook, None)
}

/// [`solve_milp_hooked`] plus an optional lazy-constraint separator (#1060).
///
/// `lazy` turns the driver into a single-tree lazy-cut engine: every
/// integer-feasible point is offered to the separator before it can become the
/// incumbent, and the globally-valid rows a veto returns are folded into the
/// shared matrix at the batch boundary — the same mechanism the node cut pool
/// already uses. A vetoed node is **re-queued**, not fathomed. When `lazy` is
/// `None` this is exactly [`solve_milp_hooked`].
#[allow(clippy::too_many_arguments)]
pub fn solve_milp_lazy_hooked(
    csc_w: SparseCols,
    m: usize,
    n: usize,
    c: &[f64],
    l: &[f64],
    u: &[f64],
    b: &[f64],
    obj_const: f64,
    opts: &MilpOptions,
    hook: Option<&dyn MilpDebugHook>,
    lazy: Option<&dyn MilpLazyHook>,
) -> MilpResult {
    solve_milp_node_hooked(csc_w, m, n, c, l, u, b, obj_const, opts, hook, lazy, None)
}

/// [`solve_milp_lazy_hooked`] plus an optional **fractional**-node cut separator
/// (#1141).
///
/// `node` completes the single-tree branch-and-cut picture: `lazy` fires at
/// integer-feasible points (the caller's constraints must hold there), `node`
/// fires at the fractional node relaxations in between (the caller's convex
/// constraints can be *linearized* there). Rows from either are globally valid
/// and fold into the shared matrix at the batch boundary.
///
/// The two hooks differ in what a separation means. A lazy veto is mandatory —
/// it is the only thing keeping a point out of the incumbent — so a vetoed node
/// is re-queued and exhausting the re-queue cap costs certification. A node
/// separation is optional — it only tightens a relaxation — so exhausting
/// `MilpOptions::node_hook_rounds` simply imports the node's own valid LP bound
/// and certification is untouched.
///
/// When `node` is `None` (or `opts.node_hook_rounds == 0`) this is exactly
/// [`solve_milp_lazy_hooked`].
#[allow(clippy::too_many_arguments)]
pub fn solve_milp_node_hooked(
    mut csc_w: SparseCols,
    m: usize,
    n: usize,
    c: &[f64],
    l: &[f64],
    u: &[f64],
    b: &[f64],
    obj_const: f64,
    opts: &MilpOptions,
    hook: Option<&dyn MilpDebugHook>,
    lazy: Option<&dyn MilpLazyHook>,
    node: Option<&dyn MilpNodeHook>,
) -> MilpResult {
    // A hook with a zero round budget can never fire; drop it here so every
    // fire-site downstream is a single `Option` test and the "zero effect when
    // absent" guarantee covers "present but disabled" too.
    let node = if opts.node_hook_rounds == 0 || opts.node_hook_cut_cap == 0 {
        None
    } else {
        node
    };
    crate::profile::begin_solve();
    let ns = opts.n_struct;
    let is_int = {
        let mut v = vec![false; ns];
        for &j in &opts.integer_cols {
            if j < ns {
                v[j] = true;
            }
        }
        v
    };
    let int_info: Vec<VarBranchInfo> = (0..ns)
        .filter(|&j| is_int[j])
        .map(|j| VarBranchInfo {
            offset: j,
            size: 1,
            is_integer: true,
        })
        .collect();

    let mut is_int_full = vec![false; n];
    is_int_full[..ns].copy_from_slice(&is_int);

    // --- presolve: sound, dimension-preserving root bound tightening ---
    // Only narrows bounds (interval/FBBT contraction), so it never cuts a
    // feasible solution and needs no postsolve; the tightened bounds seed both
    // the tree's global bounds and the node LPs. A proven-empty box ⇒ infeasible.
    let (base_l, base_u) = if opts.presolve {
        // cert:T0.3 — time root presolve bound reduction.
        let pr = {
            let _t = crate::profile::Timer::new(crate::profile::Phase::NodeReduce);
            tighten_bounds_csc(&csc_w, m, n, l, u, b, &is_int_full, opts.simplex.tol)
        };
        if pr.infeasible {
            return MilpResult {
                status: MilpStatus::Infeasible,
                x: vec![0.0; ns],
                obj: f64::INFINITY,
                bound: f64::INFINITY,
                nodes: 0,
                lp_iters: 0,
                lazy_calls: 0,
                lazy_requeues: 0,
                node_calls: 0,
                node_cuts_added: 0,
            };
        }
        (pr.l, pr.u)
    } else {
        (l.to_vec(), u.to_vec())
    };

    let glb = base_l[..ns].to_vec();
    let gub = base_u[..ns].to_vec();
    let mut tm = TreeManager::new(ns, glb, gub, int_info, SelectionStrategy::BestFirst);
    // Objective-lattice fathoming. Derived from the ORIGINAL cost vector and
    // integrality pattern, before any cut adds a slack column to `c_w`: the
    // lattice is a property of the model, and appended slacks carry zero cost so
    // they would not change it, but reading `c` here keeps that independent of
    // what the cut loop does later.
    tm.set_objective_lattice(
        if obj_integrality_enabled() {
            crate::bnb::obj_integral::objective_granularity(&c[..ns], &is_int)
        } else {
            None
        },
        obj_const,
    );
    tm.initialize();

    // Working LP, possibly augmented with root cuts. Cuts add rows + slack
    // columns; structural columns [0, ns) are untouched, so the tree's
    // structural bounds still apply unchanged.
    let mut b_w = b.to_vec();
    let mut c_w = c.to_vec();
    let mut l_w = base_l;
    let mut u_w = base_u;
    let mut m_w = m;
    let mut n_w = n;

    // --- Lazy separation state (only live when `lazy` is Some) ---
    // Cuts a lazy separator produced, awaiting the next batch-boundary fold.
    // Kept apart from the node-cut pool because a lazy cut is not optional: it
    // is the only thing that stops the vetoed point from coming back, so it is
    // exempt from the pool's `node_cut_cap` budget.
    let mut lazy_pending: Vec<GomoryCut> = Vec::new();
    let mut lazy_sigs: HashSet<Vec<(u32, i64)>> = HashSet::new();
    // Per-node lazy re-solve counts, against `LAZY_REQUEUE_CAP`.
    let mut lazy_requeues: HashMap<NodeId, u32> = HashMap::new();
    // Set when the separator itself failed; the search stops uncertified.
    let mut lazy_failed = false;
    // Set when the FRACTIONAL separator itself failed; the search stops
    // uncertified for the same reason (CLAUDE.md §7): a caller that asked for
    // cuts and got an exception must not be handed a result that reads as if the
    // separator simply found nothing.
    let mut node_failed = false;
    // Executed separator calls, for the caller's anti-vacuity check (§6).
    let mut lazy_calls: usize = 0;
    // Total re-queue events across all nodes (the per-node counts above are
    // consumed by the cap; this one is reported).
    let mut lazy_requeue_events: usize = 0;

    // --- Fractional-node separation state (only live when `node` is Some) ---
    // Rows the node separator produced, awaiting the next batch-boundary fold.
    // Kept in the same vector family as the lazy ones but budgeted separately:
    // a fractional cut only tightens, so it is optional and capped, where a lazy
    // cut is mandatory and exempt.
    let mut node_pending: Vec<GomoryCut> = Vec::new();
    // Executed node-separator calls, for the caller's anti-vacuity check (§6).
    let mut node_calls: usize = 0;
    // Rows the node separator contributed that survived dedup and the cap.
    let mut node_cuts_added: usize = 0;
    // Per-node separation-round counts, against `opts.node_hook_rounds`.
    let mut node_rounds: HashMap<NodeId, u32> = HashMap::new();
    // Dedup + record a batch of separator rows into `lazy_pending`. Returns how
    // many were new (a fully-duplicate veto cannot make progress, so the caller
    // counts on the re-queue cap to break the loop).
    macro_rules! stage_lazy_cuts {
        ($cuts:expr) => {{
            let mut fresh = 0usize;
            for cut in $cuts {
                if lazy_sigs.insert(cut_signature(&cut)) {
                    lazy_pending.push(cut);
                    fresh += 1;
                }
            }
            fresh
        }};
    }

    // Dedup + record a batch of fractional-separator rows into `node_pending`,
    // honouring `node_hook_cut_cap`. Shares `lazy_sigs` with the lazy separator
    // so a row already in the relaxation is never added twice, whichever hook
    // produced it. Returns how many were new (zero means the round separated
    // nothing the matrix does not already carry, so re-solving the node cannot
    // move its bound and the caller must not re-queue it).
    macro_rules! stage_node_cuts {
        ($cuts:expr) => {{
            let mut fresh = 0usize;
            for cut in $cuts {
                if node_cuts_added + fresh >= opts.node_hook_cut_cap {
                    break;
                }
                if lazy_sigs.insert(cut_signature(&cut)) {
                    node_pending.push(cut);
                    fresh += 1;
                }
            }
            node_cuts_added += fresh;
            fresh
        }};
    }

    // --- Caller-seeded incumbent: validate, then inject before the search ---
    // Runs on the pre-cut matrix (all rows are original rows here) with the
    // presolve-tightened bounds (FBBT never cuts a feasible point, so a
    // genuinely feasible seed survives the tightening). A seed that cannot be
    // proven feasible is dropped silently — seeding is an optimization and
    // must never be able to produce a false certificate.
    if let Some(seed) = opts.initial_incumbent.as_ref() {
        if let Some((sx, sobj)) =
            validate_seed_incumbent(seed, ns, &is_int, &csc_w, &b_w, &c_w, &l_w, &u_w, m_w, n_w)
        {
            // A seed is an integer-feasible point like any other, so it goes to
            // the separator first: seeding an incumbent the caller's lazy
            // constraints exclude would prune the tree against a point that is
            // not actually feasible. A vetoed seed is simply not seeded (seeding
            // is an optimization), but its cuts are kept — they are globally
            // valid and the first batch boundary folds them in.
            let accept = match lazy {
                None => true,
                Some(h) => {
                    lazy_calls += 1;
                    match h.separate(&sx) {
                        MilpLazyVerdict::Accept => true,
                        MilpLazyVerdict::Reject(cuts) => {
                            let _ = stage_lazy_cuts!(cuts);
                            false
                        }
                        MilpLazyVerdict::Failed => {
                            lazy_failed = true;
                            false
                        }
                    }
                }
            };
            if accept {
                tm.inject_incumbent(sx, sobj + obj_const);
            }
        }
    }

    let mut lp_iters = 0usize;
    let mut unbounded = false;
    // False once the SEARCH was truncated (deadline deferral, node/time limit,
    // debugger stop): the driver then reports a limit/feasible status, never
    // `Optimal`. A node LP *failure* does NOT clear this (#598): the tree keeps
    // failed nodes soundly accounted (parent-bound floor + branch, or the
    // permanent `unresolved_floor`), so a closed gap stays a rigorous
    // certificate — see the IterLimit/Numerical arm of `solve_node`.
    let mut gap_certified = true;
    // C-2: set whenever a node is dropped un-solved (deadline deferral). A
    // deferred node was already popped off the heap and left `Evaluated`, so it
    // is invisible to `open_count()` — `is_finished()` can then read `true` even
    // though that node's subtree was never searched. Without this flag, an empty
    // tree with no incumbent is mislabeled `Infeasible` (a false certificate)
    // when the real terminus is a time-limit cut-off. The no-incumbent status
    // branch gates on this so a deadline yields a limit status, not `Infeasible`.
    let mut search_incomplete = false;

    // Original constraint rows (before any cuts) are the knapsack candidates for
    // cover separation; later rows are themselves cuts.
    let n_orig_rows = m_w;
    // Global cut pool signatures — globally-valid cover cuts found anywhere in
    // the tree are added once and shared by all nodes.
    let mut pool_sigs: HashSet<Vec<(u32, i64)>> = HashSet::new();

    // Node-cut policy (issue #331 node-count sweep). Separating cover cuts at
    // fractional nodes closes the integrality gap deep in the tree and cuts the
    // node count hard (−70–80% on sparse knapsacks, toward SCIP). Two grounded
    // guardrails make it a *wall* win rather than a 2× regression:
    //   * Density gate. A cover cut spans the support of its source row, so on
    //     dense-row models it is itself dense and bloats every node's LP for no
    //     node benefit (measured: dense knapsacks +2× wall, ~0 node change). Only
    //     separate when the structural rows are sparse, where cover cuts are
    //     row-local and cheap. (Set-covering ≥-rows yield no knapsack covers, so
    //     this is a no-op there regardless.)
    //   * Tight pool cap ≈ 2× the original row count. The win is at a small active
    //     set; loose caps (≈8×) drive the per-node LP cost up and erase it. Cuts
    //     are kept globally valid and never removed, so a tight cap is the cheap
    //     stand-in for SCIP-style aging.
    // Nonzeros in the structural columns [0, ns): the CSC's column pointer at `ns`
    // (columns are stored contiguously) — identical to the dense per-row count.
    let struct_nnz: usize = csc_w.raw().0[ns];
    let row_density = struct_nnz as f64 / (m_w.max(1) * ns.max(1)) as f64;
    const NODE_CUT_MAX_DENSITY: f64 = 0.5;
    let node_cuts_on = opts.node_cuts && row_density < NODE_CUT_MAX_DENSITY;
    let node_cut_cap = (2 * n_orig_rows).min(opts.max_pool_cuts);

    // Absolute wall-clock deadline for the whole solve (root cuts + B&B),
    // computed up front so even the root-cut LP solves below honour it. The
    // simplex options carry it into every primal/dual loop, so a single
    // pathological dense LP cannot run past the budget; `node_simplex` is the
    // deadline-aware variant used for all LP solves here. See
    // `SimplexOptions::deadline`.
    let t_start = std::time::Instant::now();
    let deadline = opts
        .time_limit_s
        .map(|tl| t_start + std::time::Duration::from_secs_f64(tl));
    let node_simplex = {
        let mut sx = opts.simplex.clone();
        sx.deadline = deadline;
        sx
    };

    // --- P5/P8: multi-round root GMI cuts from the native simplex basis ---
    // Each round re-solves the (growing) root LP and separates GMI cuts off its
    // native basis, adding them as `coeffs·x − s = rhs` surplus rows. Iterating
    // rounds (Gomory's classic approach) tightens the relaxation far more than a
    // single pass; we stop on the cut cap, when no violated cut is found, or when
    // the bound stops improving (tailing off).
    // Optimal basis from the last successful root-cut solve, with the matrix size
    // (`n_w`) it was computed at. Reused to warm-start the root B&B node so it does
    // not re-derive the augmented LP from a cold slack basis. See `root_warm_basis`.
    let mut root_basis: Option<(Basis, usize)> = None;
    // Snapshot of the matrix as it stood before any root cut, kept so the cut
    // cleanup after the loop can rebuild from it with a subset of the cuts. Only
    // taken when more than one round can run: with `cut_rounds == 1` every cut is
    // separated off the single root solve and is violated there by construction,
    // so the cleanup below provably keeps all of them and the clone would be pure
    // cost. (`cut_rounds == 1` is the shipped default.)
    let prune_base: Option<(SparseCols, usize, usize)> =
        if opts.root_cut_prune && opts.root_cuts > 0 && opts.cut_rounds > 1 {
            Some((csc_w.clone(), m_w, n_w))
        } else {
            None
        };
    let mut all_root_cuts: Vec<GomoryCut> = Vec::new();
    let mut last_root_x: Option<Vec<f64>> = None;
    if opts.root_cuts > 0 {
        let _t = crate::profile::Timer::new(crate::profile::Phase::RootCutLoop);
        let mut total_cuts = 0usize;
        let mut prev_obj = f64::NEG_INFINITY;
        // Wall bound for the separation phase: the earlier of the caller's own
        // deadline and the (optional) root-cut sub-budget. Checked at the top of
        // each round -- a round in flight always finishes, so the loop never
        // leaves a half-augmented matrix behind.
        //
        // `root_cut_time_s` is the substantive half: it is the only thing that
        // stops a high `cut_rounds` from spending the whole budget separating.
        //
        // The `deadline` half is a cheap early exit, NOT a fix for an overrun.
        // The loop already stopped at the caller's deadline indirectly: each
        // round's root LP carries `node_simplex.deadline`, and a deadline-aborted
        // LP comes back non-`Optimal`, which breaks the loop below. Testing the
        // clock here just skips mounting a round whose LP would abort anyway,
        // along with the separation work that would follow it.
        let root_cut_deadline: Option<std::time::Instant> = match (deadline, opts.root_cut_time_s) {
            (Some(dl), Some(rc)) => {
                Some(dl.min(t_start + std::time::Duration::from_secs_f64(rc.max(0.0))))
            }
            (Some(dl), None) => Some(dl),
            (None, Some(rc)) => Some(t_start + std::time::Duration::from_secs_f64(rc.max(0.0))),
            (None, None) => None,
        };
        for _round in 0..opts.cut_rounds {
            if total_cuts >= opts.root_cuts {
                break;
            }
            if let Some(dl) = root_cut_deadline {
                if std::time::Instant::now() >= dl {
                    break;
                }
            }
            crate::profile::incr(crate::profile::Ctr::RootCutRounds);
            let root = {
                let _t = crate::profile::Timer::new(crate::profile::Phase::RootSolve);
                // Re-optimize from the PREVIOUS round's optimal basis instead of
                // re-deriving the augmented LP from scratch. Appending a cut adds a
                // row and its surplus slack; `extend_basis` makes that slack basic,
                // which leaves the basis nonsingular and still dual-feasible (the
                // slack's cost is 0), so a warm DUAL re-solve is the textbook
                // re-optimizer — it pivots in the few violated cut rows rather than
                // running a fresh primal phase-1 over the whole matrix.
                //
                // The loop already kept `root_basis` for the post-loop root node; it
                // simply never fed it back in, so every round paid a cold solve.
                // Measured on the rsyn0840m OA master at `root_cuts=500,
                // cut_rounds=15`: 14 cold root solves = 23.1 s of the 24.2 s cut
                // loop, 16127 phase-1 pivots, `EntryColsWarm = 0`.
                //
                // Bound-safe: the LP optimum does not depend on the starting basis,
                // and the warm solve falls back to the cold `solve_lp_root_csc` on
                // IterLimit/Numerical exactly as the node path does. Only which
                // optimal vertex is reached — hence which Gomory cuts are read off
                // the basis — can differ, the same latitude the post-loop
                // `root_warm_basis` path already has.
                // Borrowed, not taken: a round whose LP comes back non-Optimal
                // breaks out of the loop *without* refreshing `root_basis`, and the
                // post-loop root node still wants the last good basis to warm-start
                // from. Taking it here would silently cold-solve that root.
                match root_basis.as_ref() {
                    Some(&(ref b, basis_n)) => {
                        crate::profile::incr(crate::profile::Ctr::RootCutWarmReopt);
                        let b = if basis_n < n_w {
                            extend_basis(b.clone(), n_w)
                        } else {
                            b.clone()
                        };
                        let lp = LpView {
                            a: &[],
                            m: m_w,
                            n: n_w,
                            c: &c_w,
                            l: &l_w,
                            u: &u_w,
                        };
                        let warm = solve_lp_warm_scaled_csc(
                            &lp,
                            &b_w,
                            &b,
                            &warm_root_opts(&node_simplex, m_w, n_w),
                            &csc_w,
                        );
                        match warm.status {
                            LpStatus::IterLimit | LpStatus::Numerical => solve_lp_root_csc(
                                &csc_w,
                                m_w,
                                n_w,
                                &c_w,
                                &l_w,
                                &u_w,
                                &b_w,
                                &node_simplex,
                            ),
                            _ => warm,
                        }
                    }
                    None => {
                        solve_lp_root_csc(&csc_w, m_w, n_w, &c_w, &l_w, &u_w, &b_w, &node_simplex)
                    }
                }
            };
            lp_iters += root.iters;
            if root.status != LpStatus::Optimal {
                break;
            }
            // Keep this round's optimal basis for the root B&B node. If the loop
            // exits now (tailing off / no cuts) it already spans the final matrix;
            // if more cuts are appended below it is extended to the new size after
            // the loop. The clone is cheap next to the solve it came from.
            root_basis = Some((root.basis.clone(), n_w));
            if prune_base.is_some() {
                last_root_x = Some(root.x.clone());
            }
            // Tailing off: stop once added cuts barely move the bound.
            if root.obj <= prev_obj + 1e-7 * (1.0 + prev_obj.abs()) && prev_obj > f64::NEG_INFINITY
            {
                break;
            }
            prev_obj = root.obj;

            // Knapsack cover cuts (sparse, strong on knapsack structure) plus
            // Gomory mixed-integer cuts off the native basis.
            let mut cuts = {
                let _t = crate::profile::Timer::new(crate::profile::Phase::SepCover);
                separate_cover_csc(
                    &csc_w,
                    n_w,
                    m_w,
                    &l_w,
                    &u_w,
                    &b_w,
                    &root.x,
                    ns,
                    &is_int_full,
                    n_orig_rows,
                    opts.simplex.tol,
                )
            };
            if opts.gmi_cuts {
                let _t = crate::profile::Timer::new(crate::profile::Phase::SepGomory);
                cuts.extend(separate_gomory_cols(
                    &csc_w,
                    m_w,
                    n_w,
                    &l_w,
                    &u_w,
                    &b_w,
                    &root.basis,
                    &is_int_full,
                    opts.simplex.tol,
                    1e7,
                ));
            }
            // Keep the strongest, most diverse few (efficacy + orthogonality)
            // up to the remaining root-cut budget; otherwise add first-come.
            let remaining = opts.root_cuts - total_cuts;
            let selected = if opts.cut_select {
                select_cuts(cuts, &root.x, remaining, CUT_MIN_EFFICACY, CUT_MAX_PARALLEL)
            } else {
                cuts.truncate(remaining);
                cuts
            };
            let new_cuts = dedup_new_cuts(selected, &mut pool_sigs, usize::MAX);
            if new_cuts.is_empty() {
                break;
            }
            total_cuts += new_cuts.len();
            let (nw_csc, nm, nn) = augment_csc_with_cuts(
                &csc_w,
                &mut b_w,
                &mut c_w,
                &mut l_w,
                &mut u_w,
                &mut is_int_full,
                m_w,
                n_w,
                &new_cuts,
            );
            csc_w = nw_csc;
            m_w = nm;
            n_w = nn;
            if prune_base.is_some() {
                all_root_cuts.extend(new_cuts);
            }
        }
    }

    // --- Root cut cleanup: keep only the cuts the root LP actually leans on ---
    //
    // Every cut generated above rides in *every* node LP for the rest of the
    // solve. Across 50 rounds that is hundreds of extra rows, and the measured
    // cost is severe: on the cflp family the 500-cut budget turned a 0.04 s solve
    // into 31 s while the bound was no better. But a cut that is *slack* at the
    // final root solution contributed nothing to the root bound, and dropping it
    // is provably bound-neutral there: its dual multiplier is zero, so the root
    // optimum and its KKT certificate survive the row's removal unchanged. Deeper
    // in the tree a dropped cut could have become tight, so the node bounds may
    // differ — weaker, never invalid, since the remaining rows are still globally
    // valid cuts. This is a bound-CHANGING change under CLAUDE.md #5.
    //
    // A GMI cut's coefficients span the slack columns of the cuts before it, so a
    // cut can only be dropped if nothing kept depends on it; the reverse sweep
    // closes that dependency set before anything is removed.
    if let (Some((csc_base, m_base, n_base)), Some(x)) = (prune_base.as_ref(), last_root_x.as_ref())
    {
        let (m_base, n_base) = (*m_base, *n_base);
        let mut keep = vec![false; all_root_cuts.len()];
        for ci in (0..all_root_cuts.len()).rev() {
            let cut = &all_root_cuts[ci];
            // `coeffs · x >= rhs`; the surplus is the amount by which the final
            // root solution over-satisfies the cut. Cuts separated after the last
            // solve have a negative surplus there (they were violated, which is
            // why they were separated) and are always kept.
            let act: f64 = cut
                .coeffs
                .iter()
                .enumerate()
                .take(x.len())
                .map(|(j, &a)| a * x[j])
                .sum();
            let tight = act - cut.rhs <= 1e-6 * (1.0 + cut.rhs.abs());
            if tight || keep[ci] {
                keep[ci] = true;
                for (j, &a) in cut.coeffs.iter().enumerate() {
                    if a != 0.0 && j >= n_base {
                        let dep = j - n_base;
                        // A cut only ever sees the slacks of cuts before it, so
                        // `dep < ci` and the reverse sweep still has it ahead.
                        if dep < ci {
                            keep[dep] = true;
                        }
                    }
                }
            }
        }
        let n_keep = keep.iter().filter(|&&k| k).count();
        crate::profile::incr_by(
            crate::profile::Ctr::RootCutsGenerated,
            all_root_cuts.len() as u64,
        );
        crate::profile::incr_by(crate::profile::Ctr::RootCutsKept, n_keep as u64);
        if n_keep < all_root_cuts.len() {
            let mut newidx = vec![usize::MAX; all_root_cuts.len()];
            let mut next = 0usize;
            for (c, &k) in keep.iter().enumerate() {
                if k {
                    newidx[c] = next;
                    next += 1;
                }
            }
            let kept: Vec<GomoryCut> = all_root_cuts
                .iter()
                .enumerate()
                .filter(|(c, _)| keep[*c])
                .map(|(_, cut)| {
                    let mut coeffs = vec![0.0; n_base + n_keep];
                    for (j, &a) in cut.coeffs.iter().enumerate() {
                        if a == 0.0 {
                            continue;
                        }
                        if j < n_base {
                            coeffs[j] = a;
                        } else {
                            // Kept by the dependency closure above; indexing a
                            // `usize::MAX` here would panic rather than corrupt.
                            coeffs[n_base + newidx[j - n_base]] = a;
                        }
                    }
                    GomoryCut {
                        coeffs,
                        rhs: cut.rhs,
                    }
                })
                .collect();
            b_w.truncate(m_base);
            c_w.truncate(n_base);
            l_w.truncate(n_base);
            u_w.truncate(n_base);
            is_int_full.truncate(n_base);
            for cut in &kept {
                b_w.push(cut.rhs);
                c_w.push(0.0);
                l_w.push(0.0);
                u_w.push(INF);
                is_int_full.push(false);
            }
            csc_w = rebuild_csc_with_cuts(csc_base, m_base, n_base, &kept);
            m_w = m_base + kept.len();
            n_w = n_base + kept.len();
            // The stored basis indexes the pre-cleanup matrix (dropped rows'
            // slacks are basic in it, by definition of being slack), so it cannot
            // be extended onto the smaller one. The root node cold-solves the
            // reduced LP instead -- cheaper than the solve that produced the
            // basis, and far cheaper than carrying the dropped rows all tree.
            root_basis = None;
        }
    }

    let mut slack_l = l_w[ns..].to_vec();
    let mut slack_u = u_w[ns..].to_vec();

    // Extend the kept root basis to the final matrix size: rounds after the last
    // solve appended cut rows/slacks, and `extend_basis` makes those new slacks
    // basic (a valid, dual-repairable starting basis). When the last solve already
    // spanned the final matrix this is a no-op. The root node warm-starts from it.
    let root_warm_basis: Option<Basis> = root_basis.map(|(b, basis_n)| {
        if basis_n < n_w {
            extend_basis(b, n_w)
        } else {
            b
        }
    });

    // `deadline` (computed above, before the root cuts) drives two layers of
    // budget enforcement: the loop-top check below stops *dispatching* new
    // batches once it passes; `solve_node` reads it to drop each node's
    // *optional* effort (strong branching, cover separation, rounding) once it is
    // reached — leaving only the cheap LP solve that yields the node's valid
    // bound — so the in-flight batch drains quickly instead of running to
    // completion; and `node_simplex.deadline` bounds each individual LP solve
    // from inside the simplex loop. Soundness is untouched: those steps only
    // change branching choice / cut tightness / early incumbents, never a
    // bound's validity.
    let _t_search = crate::profile::Timer::new(crate::profile::Phase::SearchLoop);

    // Interactive debugger bookkeeping. `dbg_iter` mirrors the Python loops'
    // iteration counter; `dbg_last_inc` tracks the incumbent so the
    // `IncumbentFound` event fires exactly on a strict improvement. When `hook`
    // is `None` the macro below expands to `false` and nothing is read.
    let mut dbg_iter: usize = 0;
    let mut dbg_last_inc: f64 = f64::INFINITY;
    macro_rules! fire_dbg {
        ($cp:expr) => {{
            if let Some(h) = hook {
                let s = tm.stats();
                let inc = tm.incumbent().map(|(_, v)| v);
                let state = MilpDebugState {
                    checkpoint: $cp,
                    iteration: dbg_iter,
                    total_nodes: s.total_nodes,
                    open_nodes: s.open_nodes,
                    incumbent: inc,
                    bound: s.global_lower_bound,
                    gap: s.gap,
                    elapsed: t_start.elapsed().as_secs_f64(),
                    n_vars: ns,
                    batch_lb: None,
                    batch_ub: None,
                    batch_ids: None,
                };
                matches!(h.checkpoint(&state), MilpDebugControl::Stop)
            } else {
                false
            }
        }};
    }

    // #1060 no-incumbent dive schedule. Both counters live in the sequential
    // driver loop (never inside the parallel node map), so the schedule is a pure
    // function of batch order and the search stays deterministic.
    let mut batch_index: usize = 0;
    let mut nonroot_dives: usize = 0;

    'search: loop {
        if fire_dbg!(MilpCheckpoint::IterStart) {
            gap_certified = false;
            break;
        }
        if tm.is_finished() || tm.gap() <= opts.gap_tol {
            break;
        }
        if tm.stats().total_nodes >= opts.max_nodes {
            gap_certified = false;
            break;
        }
        if let Some(tl) = opts.time_limit_s {
            if t_start.elapsed().as_secs_f64() >= tl {
                gap_certified = false;
                break;
            }
        }
        let batch = tm.export_batch(64);
        if batch.node_ids.is_empty() {
            break;
        }

        // Interactive debugger: nodes selected — expose the batch's boxes/ids so
        // `print node <i>` works on this pure-Rust path too. Gated on the hook.
        if let Some(h) = hook {
            let s = tm.stats();
            let inc = tm.incumbent().map(|(_, v)| v);
            let state = MilpDebugState {
                checkpoint: MilpCheckpoint::AfterSelect,
                iteration: dbg_iter,
                total_nodes: s.total_nodes,
                open_nodes: s.open_nodes,
                incumbent: inc,
                bound: s.global_lower_bound,
                gap: s.gap,
                elapsed: t_start.elapsed().as_secs_f64(),
                n_vars: ns,
                batch_lb: Some(&batch.lb),
                batch_ub: Some(&batch.ub),
                batch_ids: Some(&batch.node_ids),
            };
            if matches!(h.checkpoint(&state), MilpDebugControl::Stop) {
                gap_certified = false;
                break 'search;
            }
        }

        // Equilibration scaling for the working matrix, computed once per batch
        // and shared by every node solve below. The matrix is constant within a
        // batch (cuts are folded in only between batches), so re-equilibrating it
        // per node — as the auto-scaling entry points would — is pure waste. On an
        // ill-scaled lifted LP this is the dominant per-node cost; sharing it lets
        // the 64 nodes pay one equilibration. When the matrix is well-conditioned
        // (`None`) the nodes solve the original LP unchanged.
        let scaling = Scaling::from_sparse(&csc_w, m_w, n_w);
        let (c_s, b_s) = match &scaling {
            Some(s) => (s.scale_c(&c_w), s.scale_b(&b_w)),
            None => (Vec::new(), Vec::new()),
        };
        // Solve-space objective/rhs: the scaled copies when scaling, else the
        // originals (borrowed). Node bounds are scaled per node (cheap).
        let (sc, sb): (&[f64], &[f64]) = match &scaling {
            Some(_) => (&c_s, &b_s),
            None => (&c_w, &b_w),
        };
        // Solve-space CSC of the working matrix, built once and shared by every node
        // solve in this batch (the matrix is constant within a batch): scale the
        // working CSC by the batch factors, or reuse it unscaled. Never densified.
        let csc_batch = match &scaling {
            Some(s) => {
                let mut c = csc_w.clone();
                s.scale_cols(&mut c);
                c
            }
            None => csc_w.clone(),
        };
        // Reduced-cost fixing (and the unscaled per-node consumers) reason in true
        // objective units, so they use the CSC of the *unscaled* working matrix. When
        // the matrix isn't scaled this is exactly `csc_batch` (no second build).
        let csc_unscaled_owned = scaling.as_ref().map(|_| csc_w.clone());
        let csc_rc: &SparseCols = csc_unscaled_owned.as_ref().unwrap_or(&csc_batch);
        let sb_active = opts.strong_branch && tm.stats().total_nodes < opts.sb_node_budget;
        let mut results = Vec::with_capacity(batch.node_ids.len());
        let mut pending_cuts: Vec<GomoryCut> = Vec::new();

        // --- per-node evaluation (parallelizable) ---
        // Each node's relaxation solve is independent and reads only the
        // immutable working LP plus a snapshot of the tree's read-only state, so
        // the bodies run concurrently and the resulting `NodeOutput`s are folded
        // back into the tree sequentially, in batch order, below. The snapshot
        // (incumbent/reliability/pool-room) is taken once per batch; pseudocosts
        // are likewise constant within a batch (updated in `process_evaluated`),
        // so each node's computation is independent of thread scheduling and the
        // search stays deterministic.
        // `ctx` is scoped to the map so its immutable borrow of `tm` ends before
        // the mutable reduce below.
        // #1060: does this batch get a continuous-repair dive away from the root?
        // Only while the tree holds NO incumbent — with nothing to prune against,
        // every node in the batch is doing bound-free work, so spending one node's
        // budget on the one heuristic that can produce a first feasible point is
        // the cheapest thing in the batch. The moment an incumbent exists the
        // schedule switches off on its own.
        let this_batch = batch_index;
        batch_index += 1;
        let dive_batch = dive_batch_eligible(
            dive_stride(),
            this_batch,
            tm.incumbent().is_some(),
            nonroot_dives,
        );
        if dive_batch {
            nonroot_dives += 1;
        }

        let outputs: Vec<NodeOutput> = {
            let ctx = NodeCtx {
                b_w: &b_w,
                c_w: &c_w,
                l_w: &l_w,
                u_w: &u_w,
                scaling: scaling.as_ref(),
                sc,
                sb,
                csc: &csc_batch,
                csc_rc,
                slack_l: &slack_l,
                slack_u: &slack_u,
                is_int: &is_int,
                is_int_full: &is_int_full,
                ns,
                m_w,
                n_w,
                n_orig_rows,
                obj_const,
                opts,
                simplex: &node_simplex,
                sb_active,
                inc_snapshot: tm.incumbent().map(|(_, inc)| inc),
                reliability: tm.get_reliability_threshold(),
                pool_room: node_cuts_on && pool_sigs.len() < node_cut_cap,
                root_warm_basis: root_warm_basis.as_ref(),
                deadline,
                tm: &tm,
            };
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                // Small batches don't amortize task-spawn overhead; solve those
                // serially. PAR_MIN_BATCH is conservative — the bench tunes it.
                const PAR_MIN_BATCH: usize = 4;
                if batch.node_ids.len() >= PAR_MIN_BATCH {
                    (0..batch.node_ids.len())
                        .into_par_iter()
                        .map(|k| {
                            solve_node(
                                batch.node_ids[k],
                                &batch.lb[k],
                                &batch.ub[k],
                                &ctx,
                                dive_batch && k == 0,
                            )
                        })
                        .collect()
                } else {
                    (0..batch.node_ids.len())
                        .map(|k| {
                            solve_node(
                                batch.node_ids[k],
                                &batch.lb[k],
                                &batch.ub[k],
                                &ctx,
                                dive_batch && k == 0,
                            )
                        })
                        .collect()
                }
            }
            #[cfg(not(feature = "parallel"))]
            {
                (0..batch.node_ids.len())
                    .map(|k| {
                        solve_node(
                            batch.node_ids[k],
                            &batch.lb[k],
                            &batch.ub[k],
                            &ctx,
                            dive_batch && k == 0,
                        )
                    })
                    .collect()
            }
        };

        // --- sequential reduce: apply tree mutations in batch order ---
        let mut hit_unbounded = false;
        for (k, mut out) in outputs.into_iter().enumerate() {
            let id = batch.node_ids[k];
            lp_iters += out.iters;
            if out.deferred {
                // Deadline hit before this node's LP solve. Skip it entirely: do
                // not import a result, so the node keeps its parent-inherited
                // bound. This is a search TRUNCATION (unlike a node-LP failure,
                // which stays certifiable — see below), so the gap is not
                // certified; the loop-top deadline check breaks the search on
                // the next iteration.
                //
                // C-2: the node was popped off the heap by `export_batch` and is
                // now stuck `Evaluated`, so `open_count()` no longer sees it. If
                // the rest of this final batch fathoms rigorously and no incumbent
                // exists, `is_finished()` would read `true` and the driver would
                // return `Infeasible` — a false certificate, since this node's
                // subtree was never searched. Record that the search was cut short
                // so the no-incumbent status resolves to a limit status instead.
                gap_certified = false;
                search_incomplete = true;
                continue;
            }
            if out.unbounded {
                hit_unbounded = true;
                break;
            }
            // An iter-limit/numerical node LP exit does NOT decertify the gap
            // (#598). Such a node is handed back with a raw -inf bound and an
            // untrusted midpoint solution; the tree keeps it SOUND end to end:
            // `import_results` floors the -inf at the node's parent-inherited
            // bound (valid over the child box), and `process_evaluated` only
            // ever (a) prunes it against the incumbent using that valid bound,
            // (b) branches it — its children re-solve fresh LPs, so the subtree
            // stays searched and its bounds stay in the frontier minimum — or
            // (c) when no branch direction remains, fathoms it into the
            // tree's permanent `unresolved_floor`, which participates in
            // `tm.gap()`. Every removal is therefore proof-backed or floored,
            // so a closed gap is a rigorous certificate even after node-LP
            // failures; withholding `Optimal` here was pure over-conservatism.
            if let Some(basis) = out.basis {
                tm.set_node_basis(id, Some(basis));
            }
            if let Some((cand, cobj)) = out.incumbent {
                // A heuristic candidate is an integer-feasible point, so it goes
                // through the separator on the same terms as a node's own
                // integral relaxation. Nothing to re-queue here: the candidate is
                // not this node's LP solution, so vetoing it leaves the node's
                // own result (fractional, or handled below) untouched.
                let accept = match lazy {
                    None => true,
                    Some(h) => {
                        lazy_calls += 1;
                        match h.separate(&cand) {
                            MilpLazyVerdict::Accept => true,
                            MilpLazyVerdict::Reject(cuts) => {
                                let _ = stage_lazy_cuts!(cuts);
                                false
                            }
                            MilpLazyVerdict::Failed => {
                                lazy_failed = true;
                                false
                            }
                        }
                    }
                };
                if accept {
                    tm.inject_incumbent(cand, cobj);
                }
            }
            if let Some(v) = out.branch_hint {
                tm.set_branch_hint(id, v);
            }
            // Feed this node's strong-branch probes into the shared pseudocosts,
            // in batch order, so the reliability mechanism graduates those
            // variables and stops re-probing them at later nodes. Selection-only:
            // never affects a bound, so determinism and soundness hold.
            if !out.sb_observations.is_empty() {
                tm.record_sb_observations(&out.sb_observations);
            }
            // Store node-propagation tightened bounds so children inherit them.
            if let Some((tl, tu)) = out.tightened {
                tm.set_node_bounds(id, tl, tu);
            }
            // Dedup this node's cuts against the shared pool *in order*, with the
            // same room check the serial path applied, so the pool is identical.
            if !out.found_cuts.is_empty() && pool_sigs.len() < node_cut_cap {
                pending_cuts.extend(dedup_new_cuts(out.found_cuts, &mut pool_sigs, node_cut_cap));
            }
            // Lazy separation of this node's OWN relaxation solution. The point
            // has to be screened here, before `import_results`, because
            // `process_evaluated` fathoms an integer-feasible node and promotes
            // its solution to the incumbent in one step — after that the point
            // is already the answer.
            //
            // The predicate must match the one `process_evaluated` uses
            // (`result.is_feasible || is_integer_feasible(solution)`), or a point
            // it treats as integral would slip past unseparated.
            // UNCHANGED from the pre-#1141 lazy predicate — it must keep matching
            // the one `process_evaluated` uses (`result.is_feasible ||
            // is_integer_feasible(solution)`), or a point that path treats as
            // integral would slip past unseparated.
            // `INFEAS_SENTINEL` is a FINITE 1e30, so `is_finite()` alone admits an
            // INFEASIBLE node -- and such a node carries a placeholder solution
            // vector of zeros, which `solution_is_integral` then happily accepts.
            // The lazy separator was therefore called on infeasible nodes with a
            // meaningless point, returned a (valid but useless) cut for it, and the
            // node was re-queued... against a matrix the point still violates,
            // because the point was never a solution of it. Measured on MINLPLib
            // `tls2` (#1141 item 4): ONE distinct assignment re-proposed 1386 times,
            // the returned point violating 31 of the 35 cut rows already in its own
            // matrix, until `LAZY_REQUEUE_CAP` exhausted -- which drops
            // `gap_certified` and marks the search incomplete. That is why the
            // in-house master could not certify instances the HiGHS master closes in
            // 13 subproblems, and it cost `tls2` its incumbent entirely.
            //
            // Same test the fractional hook already applies (`node_result_usable`);
            // it was added there in this issue and the lazy path was left alone,
            // which is the bug.
            let node_usable = out.result.lower_bound.is_finite()
                && out.result.lower_bound < INFEAS_SENTINEL - 1.0;
            let integral = node_usable
                && (out.result.is_feasible || solution_is_integral(&out.result.solution, &is_int));
            // The fractional separator's own admission test is stricter, and only
            // it: `INFEAS_SENTINEL` (1e30) is a FINITE number meaning "this node's
            // LP was infeasible or excluded", carried with a placeholder solution.
            // Firing on that placeholder is sound (a convex tangent is valid wherever
            // it is taken) but pure waste, and it would re-queue a node the tree is
            // about to fathom — invisible until an incumbent exists to make the prune
            // test below catch it.
            let node_result_usable = out.result.lower_bound.is_finite()
                && out.result.lower_bound < INFEAS_SENTINEL - 1.0;
            // Fractional separation (#1141). Only at a point the integral branch
            // below will not handle, and only while this node has separation
            // rounds left. A node whose own bound already meets the incumbent is
            // about to be pruned, so separating there buys nothing.
            if let Some(h) = node {
                let prunable = tm
                    .incumbent()
                    .map(|(_, v)| out.result.lower_bound >= v)
                    .unwrap_or(false);
                let rounds = node_rounds.get(&id).copied().unwrap_or(0);
                if node_result_usable
                    && !integral
                    && !prunable
                    && (rounds as usize) < opts.node_hook_rounds
                    && node_cuts_added < opts.node_hook_cut_cap
                {
                    node_calls += 1;
                    match h.separate_node(&out.result.solution) {
                        MilpNodeVerdict::None => {}
                        MilpNodeVerdict::Cuts(cuts) => {
                            let fresh = stage_node_cuts!(cuts);
                            if fresh > 0 {
                                // Re-open the node so it re-solves against the
                                // tightened matrix (the rows fold in at this
                                // batch's boundary). Its stored basis was saved
                                // above, so the re-solve warm-starts through the
                                // dual simplex once the cut rows are appended.
                                //
                                // Discarding this evaluation costs nothing in
                                // soundness: the node keeps its parent-inherited
                                // bound, which is valid over its box, and the
                                // re-solve can only sharpen it.
                                node_rounds.insert(id, rounds + 1);
                                // Keep the bound this evaluation proved before
                                // re-opening. `requeue_node` alone would leave
                                // the node on its parent-inherited bound, and the
                                // frontier minimum is what the driver reports as
                                // the dual bound — so discarding a valid, tighter
                                // node bound shows up directly as a weaker
                                // certificate at the time limit (`clay0303hfsg`:
                                // 1700.0 -> 3.98e-12). Sound: the node's LP
                                // optimum over its own box is a valid lower bound
                                // for that box, and the cuts just staged can only
                                // tighten it further.
                                tm.raise_node_bound(id, out.result.lower_bound);
                                tm.requeue_node(id);
                                continue;
                            }
                            // Every row was already in the relaxation, so a
                            // re-solve would return this same point. Import the
                            // result and let the node branch.
                        }
                        MilpNodeVerdict::Failed => {
                            node_failed = true;
                        }
                    }
                }
            }
            if let Some(h) = lazy {
                if integral {
                    lazy_calls += 1;
                    match h.separate(&out.result.solution) {
                        MilpLazyVerdict::Accept => {}
                        MilpLazyVerdict::Reject(cuts) => {
                            let _ = stage_lazy_cuts!(cuts);
                            let seen = lazy_requeues.entry(id).or_insert(0);
                            *seen += 1;
                            if *seen <= LAZY_REQUEUE_CAP {
                                lazy_requeue_events += 1;
                                // Re-open the node instead of importing a result.
                                // The cuts fold in at this batch's boundary, so
                                // the re-solve runs against a matrix that
                                // excludes the vetoed point. The node keeps its
                                // parent-inherited bound meanwhile — valid, just
                                // not sharpened by this evaluation.
                                // The node's optimal basis was already
                                // stored above, so the re-solve warm-starts from
                                // it through the dual simplex once the new cut
                                // rows are appended (`extend_basis`).
                                tm.requeue_node(id);
                                continue;
                            }
                            // Cap exhausted: a separator that keeps vetoing
                            // without making progress. Fall back to the sentinel
                            // (the node is excluded) — which is a NON-RIGOROUS
                            // fathom, so certification is dropped: this run may
                            // report `Feasible`, never a false `Optimal`.
                            out.result.lower_bound = INFEAS_SENTINEL;
                            out.result.is_feasible = false;
                            gap_certified = false;
                            search_incomplete = true;
                        }
                        MilpLazyVerdict::Failed => {
                            lazy_failed = true;
                        }
                    }
                }
            }
            results.push(out.result);
        }
        if lazy_failed || node_failed {
            // A separator that raised leaves constraints unenforced; continuing
            // would report a result built against a model the caller did not
            // ask for. Stop, uncertified (CLAUDE.md §7).
            gap_certified = false;
            search_incomplete = true;
            tm.import_results(&results);
            tm.process_evaluated();
            break 'search;
        }
        if hit_unbounded {
            unbounded = true;
            break 'search;
        }
        tm.import_results(&results);
        tm.process_evaluated();

        // Interactive debugger: post prune/branch/fathom, plus the new-incumbent
        // event on a strict improvement (from any source this batch).
        if let Some(v) = tm.incumbent().map(|(_, v)| v) {
            if v < dbg_last_inc - 1e-9 {
                dbg_last_inc = v;
                if fire_dbg!(MilpCheckpoint::IncumbentFound) {
                    gap_certified = false;
                    break 'search;
                }
            }
        }
        if fire_dbg!(MilpCheckpoint::AfterProcess) {
            gap_certified = false;
            break 'search;
        }
        dbg_iter += 1;

        // Lazy cuts join the fold unconditionally — after the node-cut dedup
        // above and outside the `node_cut_cap` budget, because dropping one
        // would leave the vetoed point in the relaxation and the node would be
        // re-queued forever against a matrix that never changed.
        if !lazy_pending.is_empty() {
            pending_cuts.append(&mut lazy_pending);
        }
        // Fractional-separator rows join the same fold. They are already capped
        // by `node_hook_cut_cap` at staging time, so they do not go through the
        // node-cut pool budget a second time.
        if !node_pending.is_empty() {
            pending_cuts.append(&mut node_pending);
        }

        // Fold this batch's newly-found global cuts into the shared matrix.
        // Stored node bases are extended lazily on their next solve, so children
        // warm-start through the dual simplex from the cut-augmented basis.
        if !pending_cuts.is_empty() {
            let (nw_csc, nm, nn) = augment_csc_with_cuts(
                &csc_w,
                &mut b_w,
                &mut c_w,
                &mut l_w,
                &mut u_w,
                &mut is_int_full,
                m_w,
                n_w,
                &pending_cuts,
            );
            csc_w = nw_csc;
            m_w = nm;
            n_w = nn;
            slack_l = l_w[ns..].to_vec();
            slack_u = u_w[ns..].to_vec();
        }
    }

    // Interactive debugger: terminal checkpoint (return value is advisory only —
    // the solve is already over, so its control is ignored).
    let _ = fire_dbg!(MilpCheckpoint::Terminated);

    let stats = tm.stats();
    let bound = stats.global_lower_bound;
    // An all-integer placeholder at an infeasible node can be fathomed by the
    // tree as a sentinel-valued "incumbent"; it never blocks a real (finite)
    // incumbent, so treat obj ≥ the sentinel threshold as "no real solution".
    let (x, obj, has_inc) = match tm.incumbent() {
        Some((xi, oi)) if oi < INFEAS_SENTINEL - 1.0 => (xi.to_vec(), oi, true),
        _ => (vec![0.0; ns], f64::INFINITY, false),
    };
    let status = decide_status(
        unbounded,
        has_inc,
        tm.is_finished(),
        search_incomplete,
        // A subtree was removed without proof (#467 -inf pin or #598 floor):
        // an empty tree is then not an emptiness proof. The Optimal arm needs
        // no such flag — the floor already participates in `tm.gap()`.
        stats.bound_unresolved || stats.unresolved_floor.is_finite(),
        tm.gap() <= opts.gap_tol,
        gap_certified,
        stats.total_nodes >= opts.max_nodes,
    );

    MilpResult {
        status,
        x,
        obj,
        bound,
        nodes: stats.total_nodes,
        lp_iters,
        lazy_calls,
        lazy_requeues: lazy_requeue_events,
        node_calls,
        node_cuts_added,
    }
}

/// Immutable per-batch context shared by every node evaluation. Holds the
/// working LP (constant within a batch), the options, and a snapshot of the
/// tree's read-only state. All fields are `Sync`, so `solve_node` runs under
/// `rayon`'s `into_par_iter` over the batch.
struct NodeCtx<'a> {
    b_w: &'a [f64],
    c_w: &'a [f64],
    l_w: &'a [f64],
    u_w: &'a [f64],
    /// Equilibration for the working matrix (shared across the batch), or `None`
    /// when it is well-conditioned. When `Some`, the node LP is solved on the
    /// pre-scaled `csc`/`sc`/`sb` and the solution is unscaled before use.
    scaling: Option<&'a Scaling>,
    /// Solve-space objective / rhs: scaled copies when `scaling` is `Some`, else the
    /// originals (`c_w`/`b_w`).
    sc: &'a [f64],
    sb: &'a [f64],
    /// CSC view of the solve-space (scaled when `scaling` is `Some`) working matrix,
    /// built **once per batch** and shared by every node/strong-branch/dive LP solve
    /// in the batch. The working matrix is constant within a batch (cuts fold in only
    /// between batches). The MILP driver is fully sparse: no dense `m×n` working
    /// matrix is ever materialized, so a large sparse relaxation never blows up.
    csc: &'a SparseCols,
    /// CSC view of the **unscaled** working matrix `a_w`, for reduced-cost fixing
    /// (which reasons in true objective units, so it needs unscaled duals/reduced
    /// costs). Equals `csc` when the matrix isn't scaled. Lets `reduced_cost_fix`
    /// compute the node duals via the sparse LU (btran, O(nnz)) instead of a dense
    /// O(m³) refactor — the asymmetry that made it cheap on knapsack but ruinous
    /// on many-row covering LPs.
    csc_rc: &'a SparseCols,
    slack_l: &'a [f64],
    slack_u: &'a [f64],
    is_int: &'a [bool],
    is_int_full: &'a [bool],
    ns: usize,
    m_w: usize,
    n_w: usize,
    n_orig_rows: usize,
    obj_const: f64,
    opts: &'a MilpOptions,
    /// Deadline-aware simplex options (a clone of `opts.simplex` carrying the
    /// solve's wall-clock `deadline`). Used for every node/strong-branch LP solve
    /// so a single dense, degenerate relaxation cannot overrun the time budget.
    simplex: &'a SimplexOptions,
    sb_active: bool,
    /// Incumbent value at batch start (for the strong-branch prunable check).
    inc_snapshot: Option<f64>,
    reliability: u32,
    /// Whether the cut pool had room at batch start (gates separation work).
    pool_room: bool,
    /// Optimal basis of the (final) root-cut LP, extended to the current matrix
    /// size. The root B&B node — the only node solved cold — warm-starts from it
    /// instead of the slack basis, so it pivots in just the few cut rows added
    /// after the last root solve rather than re-deriving the whole augmented LP.
    /// `None` when no root cuts ran (then the root falls back to the slack basis).
    root_warm_basis: Option<&'a Basis>,
    /// Absolute wall-clock deadline. Once passed, each node still computes its
    /// (valid) LP bound but skips the optional rounding heuristic, cover
    /// separation, and strong branching — none of which affect bound validity
    /// or feasibility — so the in-flight batch drains quickly. `None` = no limit.
    deadline: Option<std::time::Instant>,
    tm: &'a TreeManager,
}

/// One strong-branch pseudocost sample harvested from a probe:
/// `(var_index, frac, Δobj, is_down)`. Fed into the shared pseudocost tracker so
/// the reliability mechanism can graduate the variable. See [`strong_branch`].
type SbObservation = (usize, f64, f64, bool);

/// The product of one node's evaluation, applied to the tree later in the
/// sequential reduce (in batch order). Keeping every tree mutation out of the
/// parallel region is what preserves determinism: parallelism changes only
/// *when* a node's LP is solved, never the order results fold into the tree.
struct NodeOutput {
    result: NodeResult,
    /// Optimal basis to store on the node (for children to warm-start from).
    basis: Option<Basis>,
    /// Incumbent candidate from the rounding heuristic.
    incumbent: Option<(Vec<f64>, f64)>,
    /// Cover cuts found at this node (raw; deduped against the pool in reduce).
    found_cuts: Vec<GomoryCut>,
    /// Strong-branching variable hint.
    branch_hint: Option<usize>,
    /// Pseudocost samples harvested from the strong-branch probes at this node,
    /// each `(var, frac, Δobj, is_down)`. Applied to the shared pseudocost tracker
    /// in the sequential reduce (batch order) so the reliability mechanism can
    /// graduate these variables and stop re-probing them. Empty when the node did
    /// no strong branching.
    sb_observations: Vec<SbObservation>,
    /// Node-propagation result: tightened structural bounds `(lb, ub)` to store
    /// on the node so its children inherit them. `None` when propagation is off
    /// or changed nothing.
    tightened: Option<(Vec<f64>, Vec<f64>)>,
    /// Simplex pivots spent on this node (LP solve + strong-branch probes).
    iters: usize,
    /// Relaxation was unbounded — the whole search terminates.
    unbounded: bool,
    /// The wall-clock deadline had already passed when this node was dequeued, so
    /// its (expensive) LP solve was skipped entirely. The reduce drops it without
    /// importing a result, leaving the node Evaluated with its parent-inherited
    /// bound — so the returned dual bound stays valid (just not sharpened by this
    /// node) and the in-flight batch drains in O(0) instead of running every
    /// remaining node's relaxation past the deadline.
    deferred: bool,
}

/// Evaluate a single B&B node: solve its LP relaxation (cold, or warm from the
/// basis it inherited), then run the optional rounding heuristic, cover
/// separation, and strong branching. Pure given `ctx` (the immutable working LP
/// plus a read-only tree snapshot), so it is safe to call concurrently across a
/// batch. Returns a [`NodeOutput`] the caller folds into the tree sequentially.
///
/// `dive_slot` is the caller's per-node permission to run the continuous-repair
/// dive away from the root (#1060). The caller grants it to exactly one node per
/// eligible batch, chosen by position and not by thread order, so the schedule is
/// the same on every run — the determinism the batch map relies on.
fn solve_node(
    id: NodeId,
    lb_k: &[f64],
    ub_k: &[f64],
    ctx: &NodeCtx<'_>,
    dive_slot: bool,
) -> NodeOutput {
    // Deadline guard BEFORE the expensive LP solve. The loop-top check only stops
    // *dispatching* new batches; a single batch of N nodes whose per-node LP costs
    // ~seconds (e.g. a dense lifted McCormick relaxation) would otherwise run the
    // whole batch past the deadline — N x per-node-LP overshoot. Checking here lets
    // every node dequeued after the deadline return immediately, so the in-flight
    // batch drains and the loop-top check fires, bounding overshoot to the handful
    // of nodes already mid-solve. Sound: the node is left Evaluated with its
    // parent-inherited bound (the reduce skips importing a deferred result), so the
    // returned dual lower bound stays valid — only sharpening is skipped, never a
    // bound's validity. gap is decertified on the deadline path regardless.
    if ctx.deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return NodeOutput {
            result: NodeResult {
                node_id: id,
                lower_bound: f64::NEG_INFINITY,
                solution: Vec::new(),
                is_feasible: false,
                certified_infeasible: false,
            },
            basis: None,
            incumbent: None,
            found_cuts: Vec::new(),
            branch_hint: None,
            sb_observations: Vec::new(),
            tightened: None,
            iters: 0,
            unbounded: false,
            deferred: true,
        };
    }
    let mut full_l = vec![0.0; ctx.n_w];
    let mut full_u = vec![0.0; ctx.n_w];
    full_l[..ctx.ns].copy_from_slice(lb_k);
    full_u[..ctx.ns].copy_from_slice(ub_k);
    full_l[ctx.ns..].copy_from_slice(ctx.slack_l);
    full_u[ctx.ns..].copy_from_slice(ctx.slack_u);

    // Node-level propagation (FBBT): a sound contraction of this node's bounds.
    // Deep in the tree branching has tightened the slacks, so FBBT fixes/contracts
    // further integer variables the LP would leave fractional — proving some
    // subtrees infeasible outright and shrinking the rest (children inherit the
    // tightened structural bounds via the reduce). Runs on the working matrix
    // (including cut rows) with the node's local bounds.
    let mut tightened: Option<(Vec<f64>, Vec<f64>)> = None;
    if ctx.opts.node_propagation {
        let pr = {
            // cert:T0.3 — time per-node FBBT/constraint propagation. T3b5: FBBT on
            // the UNSCALED working CSC (`ctx.csc_rc`; the dense `prop_lp.a` was
            // `ctx.a_w`, unscaled), bit-identical to the dense `tighten_bounds`.
            let _t = crate::profile::Timer::new(crate::profile::Phase::Fbbt);
            tighten_bounds_csc(
                ctx.csc_rc,
                ctx.m_w,
                ctx.n_w,
                &full_l,
                &full_u,
                ctx.b_w,
                ctx.is_int_full,
                ctx.opts.simplex.tol,
            )
        };
        if pr.infeasible {
            // Proven-empty box ⇒ prune this node (a valid fathom, like an
            // infeasible LP). No incumbent, no basis, nothing to branch.
            return NodeOutput {
                result: NodeResult {
                    node_id: id,
                    lower_bound: INFEAS_SENTINEL,
                    solution: vec![0.0; ctx.ns],
                    is_feasible: false,
                    certified_infeasible: false,
                },
                basis: None,
                incumbent: None,
                found_cuts: Vec::new(),
                branch_hint: None,
                sb_observations: Vec::new(),
                tightened: None,
                iters: 0,
                unbounded: false,
                deferred: false,
            };
        }
        // Record tightened structural bounds for the children to inherit, only
        // if propagation actually changed something.
        if pr.l[..ctx.ns] != full_l[..ctx.ns] || pr.u[..ctx.ns] != full_u[..ctx.ns] {
            tightened = Some((pr.l[..ctx.ns].to_vec(), pr.u[..ctx.ns].to_vec()));
        }
        full_l = pr.l;
        full_u = pr.u;
    }
    // Solve on the batch's shared (pre-scaled, when ill-conditioned) matrix. Only
    // the per-node bounds are scaled here; the matrix/objective/rhs were scaled
    // once for the whole batch. The basis is scaling-invariant, so a warm start
    // and the returned basis transfer across the original/scaled spaces; the
    // objective is invariant too. We unscale the primal `x` back to the original
    // space so everything downstream (separation, rounding, branching) is unchanged.
    // Lazily extend a basis stored before later cuts grew the matrix: the
    // appended cut slacks become basic (a valid, dual-repairable starting basis).
    let (sl, su) = match ctx.scaling {
        Some(s) => (s.scale_lower(&full_l), s.scale_upper(&full_u)),
        None => (full_l.clone(), full_u.clone()),
    };
    let solve_lp_view = LpView {
        a: &[], // T3b5: matrix comes from `ctx.csc`; `.a` unused by the CSC solvers.
        m: ctx.m_w,
        n: ctx.n_w,
        c: ctx.sc,
        l: &sl,
        u: &su,
    };
    // The root is the only node solved cold (no inherited basis); the diving
    // heuristic runs there once so its cost (up to n_int warm re-solves) is paid
    // a single time for the whole search.
    let is_root = ctx.tm.node_basis(id).is_none();
    let _t_node = crate::profile::Timer::new(crate::profile::Phase::NodeLpSolve);
    let mut sol = match ctx.tm.node_basis(id) {
        Some(basis) => {
            let basis = extend_basis(basis, ctx.n_w);
            solve_lp_warm_scaled_csc(&solve_lp_view, ctx.sb, &basis, ctx.simplex, ctx.csc)
        }
        // The only node solved cold is the root. Prefer the root-cut loop's own
        // optimal basis (extended over any cuts added after its last solve): the
        // augmented LP is already solved there, so the root node just pivots in the
        // few trailing cut rows instead of re-deriving it. Failing that, try the
        // dual simplex from the slack basis (built from the unscaled working matrix
        // — scaling-invariant, dual feasibility preserved) before the cold primal,
        // the same covering-LP speedup the root-cut loop gets. The warm solve
        // re-verifies the start basis and cold-solves on any difficulty, so the
        // optimal objective — hence the node's bound — is identical either way; only
        // which optimal vertex is reached (and thus branching) can differ, exactly
        // as the existing slack-basis path already does. The batch CSC (`ctx.csc`)
        // is reused so this cold-root warm solve also skips the per-solve rebuild.
        None => {
            if let Some(rb) = ctx.root_warm_basis {
                let basis = extend_basis(rb.clone(), ctx.n_w);
                solve_lp_warm_scaled_csc(&solve_lp_view, ctx.sb, &basis, ctx.simplex, ctx.csc)
            } else {
                match dual_slack_basis(
                    // The unscaled CSC of the working matrix (== `from_dense(a_w)`);
                    // dual_slack_basis reads only singleton structure + `c`, both
                    // scale-invariant, so this is bit-identical to the old dense arg.
                    ctx.csc_rc,
                    ctx.m_w,
                    ctx.n_w,
                    ctx.c_w,
                    &full_l,
                    &full_u,
                    ctx.simplex.tol,
                ) {
                    // Same pivot-bounded guard as `solve_lp_root` (#350): a qualifying
                    // slack basis whose dual solve then stalls on an ill-conditioned
                    // relaxation must fall back to the cold primal, not grind to the
                    // deadline.
                    Some(basis) => {
                        let warm = solve_lp_warm_scaled_csc(
                            &solve_lp_view,
                            ctx.sb,
                            &basis,
                            &warm_root_opts(ctx.simplex, ctx.m_w, ctx.n_w),
                            ctx.csc,
                        );
                        match warm.status {
                            LpStatus::IterLimit | LpStatus::Numerical => solve_lp_cols(
                                ctx.csc.clone(),
                                ctx.m_w,
                                ctx.n_w,
                                ctx.sc,
                                &sl,
                                &su,
                                ctx.sb,
                                ctx.simplex,
                            ),
                            _ => warm,
                        }
                    }
                    None => solve_lp_cols(
                        ctx.csc.clone(),
                        ctx.m_w,
                        ctx.n_w,
                        ctx.sc,
                        &sl,
                        &su,
                        ctx.sb,
                        ctx.simplex,
                    ),
                }
            }
        }
    };
    if let Some(s) = ctx.scaling {
        s.unscale_x(&mut sol.x);
        // NOTE (#1066): `sol.dual` deliberately stays in the SCALED space here. The
        // `LpStatus::Infeasible` arm below verifies the Farkas ray against the scaled
        // working CSC (`ctx.csc`, `ctx.sb`, `sl`/`su`) and unscaling it there would
        // mix coordinate systems, fail every verification, and stop the driver from
        // fathoming provably empty nodes. Reduced-cost fixing wants the unscaled
        // duals instead, so it unscales its own copy at the call site.
    }
    drop(_t_node);
    let sol = sol;

    let mut out = NodeOutput {
        result: NodeResult {
            node_id: id,
            lower_bound: 0.0,
            solution: Vec::new(),
            is_feasible: false,
            certified_infeasible: false,
        },
        basis: None,
        incumbent: None,
        found_cuts: Vec::new(),
        branch_hint: None,
        sb_observations: Vec::new(),
        tightened,
        iters: sol.iters,
        unbounded: false,
        deferred: false,
    };

    match sol.status {
        LpStatus::Optimal => {
            out.basis = Some(sol.basis.clone());
            let xs = &sol.x[..ctx.ns];
            let feasible = ctx
                .is_int
                .iter()
                .enumerate()
                .all(|(j, &it)| !it || frac(xs[j]) <= INT_TOL);
            // Past the deadline, skip every optional per-node effort below. The
            // LP bound (above) is already computed and valid; the heuristic,
            // cover separation, and strong branching only sharpen branching /
            // cuts / early incumbents, never the bound or feasibility. Dropping
            // them lets the in-flight batch drain in cheap-LP time so the
            // loop-top deadline check can actually fire instead of being
            // overshot by a batch of expensive strong-branch probes.
            let time_up = ctx.deadline.is_some_and(|d| std::time::Instant::now() >= d);
            // Reduced-cost fixing: contract this node's integer bounds against the
            // incumbent using the LP duals. Children inherit the fixings (merged
            // with any FBBT tightening already in `out.tightened`). Pure bound
            // contraction — never touches the node's own valid LP bound.
            if ctx.opts.reduced_cost_fixing && !feasible {
                let _t_rcf = crate::profile::Timer::new(crate::profile::Phase::RedCostFix);
                // #1066: the solve exports the row duals in the scaled space (`ŷ`);
                // `reduced_cost_fix` reads them against the UNSCALED working matrix
                // `ctx.csc_rc` / cost `ctx.c_w`, so map this copy back with the same
                // exact power-of-two factors `unscale_x` uses. `sol.dual` itself is
                // left scaled for the Farkas arm (see the note after the node solve).
                let rc_dual: std::borrow::Cow<'_, [f64]> = match ctx.scaling {
                    Some(s) => {
                        let mut d = sol.dual.clone();
                        s.unscale_dual(&mut d);
                        std::borrow::Cow::Owned(d)
                    }
                    None => std::borrow::Cow::Borrowed(&sol.dual),
                };
                if let Some((rl, ru)) = reduced_cost_fix(
                    ctx.csc_rc,
                    ctx.m_w,
                    ctx.c_w,
                    &sol.basis,
                    &rc_dual,
                    sol.obj + ctx.obj_const,
                    ctx.inc_snapshot.unwrap_or(f64::INFINITY),
                    &full_l,
                    &full_u,
                    ctx.ns,
                    ctx.is_int_full,
                    ctx.opts.simplex.tol,
                ) {
                    out.tightened = Some(match out.tightened.take() {
                        Some((mut tl, mut tu)) => {
                            for j in 0..ctx.ns {
                                tl[j] = tl[j].max(rl[j]);
                                tu[j] = tu[j].min(ru[j]);
                            }
                            (tl, tu)
                        }
                        None => (rl, ru),
                    });
                }
            }
            // Primal heuristic: round this fractional point so the reduce can
            // inject a feasible incumbent early and prune more of the tree.
            if ctx.opts.heuristics && !feasible && !time_up {
                out.incumbent = try_rounding_csc(
                    &sol.x,
                    ctx.ns,
                    ctx.is_int,
                    ctx.csc_rc, // T3b5: unscaled working CSC (was ctx.a_w)
                    ctx.b_w,
                    ctx.c_w,
                    ctx.l_w,
                    ctx.u_w,
                    ctx.n_orig_rows,
                    ctx.n_w,
                    ctx.obj_const,
                );
                // Continuous-repair fractional dive: plain rounding never
                // re-solves the continuous variables for the rounded integer
                // assignment, so on weak-relaxation (big-M) models it finds no
                // incumbent at all and the search runs with no bound-based
                // pruning (tree explosion). The dive fixes integers one at a time
                // and re-solves between fixes, repairing the continuous variables
                // and avoiding infeasible combinations.
                //
                // It runs at the root, and — when `DISCOPT_MILP_DIVE_STRIDE` is
                // on — again on scheduled batches for as long as the tree holds
                // NO incumbent (#1060). Root-only was the whole story before, and
                // it leaves exactly one chance to find a first feasible point: if
                // the root dive misses, or (on the single-tree LP/NLP-BB master)
                // its candidate is cut off by the lazy separator, the search then
                // runs to its node limit with nothing to prune against. The
                // caller grants the slot to one node per eligible batch and stops
                // granting it the moment an incumbent exists, so the cost is
                // bounded and the warm-started search + cuts still take over.
                //
                // Sound at any node: the dive fixes within `[lb_k, ub_k]`, a
                // restriction of the root box, so any point it returns is feasible
                // for the model — and `inject_incumbent` re-validates it besides.
                if out.incumbent.is_none() && (is_root || dive_slot) {
                    let _t = crate::profile::Timer::new(crate::profile::Phase::DiveRepair);
                    out.incumbent = try_dive_repair(ctx, lb_k, ub_k, &sol.x, &sol.basis);
                    if !is_root {
                        crate::profile::incr(crate::profile::Ctr::DiveOffRoot);
                        if out.incumbent.is_some() {
                            crate::profile::incr(crate::profile::Ctr::DiveOffRootHits);
                        }
                    }
                }
            }
            // Node-level cover separation: a fractional node exposes violated
            // covers the root never sees. These are globally valid; the reduce
            // dedups them into the shared pool to tighten the whole tree.
            if !feasible && ctx.pool_room && !time_up {
                let _t = crate::profile::Timer::new(crate::profile::Phase::SepCover);
                out.found_cuts = separate_cover_csc(
                    ctx.csc_rc, // T3b5: unscaled working CSC (was node_lp.a = ctx.a_w)
                    ctx.n_w,
                    ctx.m_w,
                    ctx.l_w,
                    ctx.u_w,
                    ctx.b_w,
                    &sol.x,
                    ctx.ns,
                    ctx.is_int_full,
                    ctx.n_orig_rows,
                    ctx.opts.simplex.tol,
                );
            }
            // Strong branching: for a fractional node that won't be pruned, probe
            // the unreliable candidates and hint the best branching variable.
            // Only the *choice* of variable changes, so this never affects
            // correctness — only the node count. The prunable check uses the
            // batch-start incumbent snapshot (an effort decision, not a bound).
            if ctx.sb_active && !feasible && !time_up {
                let node_bound = sol.obj + ctx.obj_const;
                let prunable = ctx
                    .inc_snapshot
                    .map(|inc| node_bound >= inc - 1e-9)
                    .unwrap_or(false);
                if !prunable {
                    let _t = crate::profile::Timer::new(crate::profile::Phase::StrongBranch);
                    let cands = ctx.tm.score_candidates(xs);
                    let (best, piv, sb_obs) =
                        strong_branch(ctx, &full_l, &full_u, &sol.basis, &sol.x, sol.obj, &cands);
                    out.iters += piv;
                    out.branch_hint = best;
                    out.sb_observations = sb_obs;
                }
            }
            out.result = NodeResult {
                node_id: id,
                lower_bound: sol.obj + ctx.obj_const,
                solution: xs.to_vec(),
                is_feasible: feasible,
                certified_infeasible: false,
            };
        }
        LpStatus::Infeasible => {
            // C-14: the simplex reports `Infeasible` with a Farkas *dual ray
            // candidate* in `sol.dual` (contract: `lp/simplex/mod.rs`), sound only
            // once the caller verifies it. Fathoming on the status alone can drop a
            // node — possibly containing the optimum — when a numerically tight
            // phase-1 artificial sum trips the absolute threshold on a feasible box.
            // Verify the ray (`g0(±y) > margin`, a weak-duality certificate of
            // emptiness) on the *scaled solve-space* data — where the ray lives —
            // before pruning. The safe-bound identity is invariant under
            // equilibration (`scaling.rs`), so the scaled-space verdict equals the
            // original-space one. On verification failure the box is NOT provably
            // empty: never fathom — hand the node back with a raw -inf (non-pruning)
            // bound and midpoint so it is branched/re-solved and the optimum can
            // never be silently cut (same sound handling as the IterLimit /
            // Numerical arm below). A sound infeasible LP always exports a
            // verifiable ray, so this costs one mat-vec and never changes a
            // correct fathom.
            // T3b5: farkas check on the SCALED working CSC (`ctx.csc`; the dual/ray
            // and `sl`/`su`/`ctx.sb` are all in scaled space, as `ctx.sa` was).
            if verify_farkas_infeasible_csc(&sol.dual, ctx.csc, ctx.sb, &sl, &su, ctx.m_w, ctx.n_w)
            {
                out.result = NodeResult {
                    node_id: id,
                    lower_bound: INFEAS_SENTINEL, // pruned
                    solution: vec![0.0; ctx.ns],
                    is_feasible: false,
                    certified_infeasible: false,
                };
            } else {
                out.result = NodeResult {
                    node_id: id,
                    lower_bound: f64::NEG_INFINITY,
                    solution: midpoint(lb_k, ub_k),
                    is_feasible: false,
                    certified_infeasible: false,
                };
            }
        }
        LpStatus::Unbounded => {
            out.unbounded = true;
        }
        LpStatus::IterLimit | LpStatus::Numerical => {
            // Cannot trust this LP's bound: never prune off it (could drop the
            // optimum). Hand the node back with a raw -inf (non-pruning) bound
            // and the box midpoint. This stays fully SOUND without decertifying
            // the whole search (#598): `import_results` floors the -inf at the
            // node's parent-inherited bound — a valid bound over this child box,
            // proved at the ancestor's LP solve — and `process_evaluated` then
            // only prunes it against the incumbent on that valid bound, branches
            // it (children re-solve fresh LPs, so the subtree stays searched), or
            // — when nothing is left to branch — folds the valid bound into the
            // tree's permanent `unresolved_floor`. The untrusted midpoint can
            // never fathom the node as integer-feasible or become the incumbent
            // (`PendingResult::bound_trusted` gates both), so a closed gap
            // remains a rigorous optimality certificate.
            out.result = NodeResult {
                node_id: id,
                lower_bound: f64::NEG_INFINITY,
                solution: midpoint(lb_k, ub_k),
                is_feasible: false,
                certified_infeasible: false,
            };
        }
    }
    out
}

/// Verify a Farkas infeasibility certificate for the LP `{ A x = b, l ≤ x ≤ u }`.
///
/// The simplex exports `y` (length `m`) on `LpStatus::Infeasible` as a *candidate*
/// dual ray (contract in `lp/simplex/mod.rs`); this is the caller-side check the
/// contract requires before an infeasible fathom is trusted. By Farkas' lemma /
/// weak duality, the box is provably empty iff the objective-free safe bound
///
/// ```text
///     g0(y) = bᵀy + Σⱼ min_{zⱼ∈[lⱼ,uⱼ]} (−Aᵀy)ⱼ zⱼ
/// ```
///
/// is strictly positive for `y` or `−y` (the ray sign the simplex returns is not
/// fixed): `g0 > 0` means every point of the box violates `Σ yᵢ(Aᵢx − bᵢ) = 0`, so
/// no `x` in the box satisfies `Ax = b`. `g0(y) ≤ 0 ≤ g0` by weak duality for any
/// feasible LP, so a positive value can only arise when the feasible set is truly
/// empty — the check never false-certifies emptiness.
///
/// The margin is scaled by the magnitudes entering `g0` (‖b‖∞ and the max
/// per-column box contribution) so a genuinely-empty box clears it while a ray that
/// only grazes zero — the numerically-tight case C-14 is about — does not, forcing
/// the caller to keep (branch) the node instead of fathoming it.
///
/// Columns with an infinite bound on the contributing side yield `−∞` (that column
/// cannot help certify emptiness); such a `g0` is `≤ 0` and correctly fails to
/// certify. Runs on the scaled solve-space data (`sa`, `sb`, scaled `l`/`u`), where
/// the returned ray lives; the safe-bound identity is invariant under
/// equilibration, so the verdict matches the original space.
#[allow(dead_code)] // retained as the differential oracle for the CSC port (tests)
fn verify_farkas_infeasible(
    y: &[f64],
    a: &[f64],
    b: &[f64],
    l: &[f64],
    u: &[f64],
    m: usize,
    n: usize,
) -> bool {
    // No certificate exported ⇒ cannot verify ⇒ do not trust the fathom.
    if y.len() != m || m == 0 {
        return false;
    }
    farkas_safe_bound(y, a, b, l, u, m, n) || {
        let neg: Vec<f64> = y.iter().map(|v| -v).collect();
        farkas_safe_bound(&neg, a, b, l, u, m, n)
    }
}

/// Objective-free safe bound `g0(y) = bᵀy + Σⱼ min_box((−Aᵀy)ⱼ zⱼ)`, returning
/// `true` iff it clears a magnitude-scaled positive margin — a rigorous certificate
/// that `{Ax=b, l≤x≤u}` is empty for this `y`.
///
/// A column open to ±∞ can only contribute a finite term when its reduced cost
/// `(Aᵀy)ⱼ` is zero; the warm-simplex ray carries rounding noise there, so a reduced
/// cost within a ray-scaled tolerance of zero is treated as exactly zero (otherwise a
/// `1e-18` dribble would send `g0` to `−∞` and reject a valid certificate). A reduced
/// cost genuinely past that tolerance toward an infinite bound does push `g0` to
/// `−∞`: this ray cannot certify emptiness and the caller keeps the node.
#[allow(dead_code)] // retained as the differential oracle for the CSC port (tests)
fn farkas_safe_bound(
    y: &[f64],
    a: &[f64],
    b: &[f64],
    l: &[f64],
    u: &[f64],
    m: usize,
    n: usize,
) -> bool {
    // Reduced-cost zero-tolerance, scaled by the ray magnitude so it tracks the
    // noise floor of `Aᵀy` rather than being an absolute constant.
    let ynorm = y.iter().fold(0.0f64, |acc, &v| acc.max(v.abs()));
    let rc_tol = 1e-7 * ynorm.max(1.0);
    let mut g = 0.0f64;
    let mut scale = 0.0f64; // running magnitude of the terms, for the margin
    for i in 0..m {
        g += b[i] * y[i];
        scale = scale.max((b[i] * y[i]).abs());
    }
    for j in 0..n {
        let mut aty = 0.0f64;
        for i in 0..m {
            aty += a[i * n + j] * y[i];
        }
        let mut rc = -aty; // objective is zero: reduced cost is −(Aᵀy)ⱼ
        if rc.abs() <= rc_tol {
            rc = 0.0;
        }
        let term = if rc > 0.0 {
            if l[j] <= -INF {
                return false; // genuine −∞ contribution ⇒ this y can't certify
            }
            rc * l[j]
        } else if rc < 0.0 {
            if u[j] >= INF {
                return false;
            }
            rc * u[j]
        } else {
            0.0
        };
        g += term;
        scale = scale.max(term.abs());
    }
    // Magnitude-scaled margin: a genuinely-empty box clears `g0 > 0` with room to
    // spare, while a ray grazing zero on a numerically-tight feasible box does not.
    let margin = 1e-9 * scale.max(1.0);
    g > margin
}

/// CSC port of [`farkas_safe_bound`] (docs/dev/sparse-milp-plan.md T3b1). Bit-identical:
/// the only matrix use is `(Aᵀy)ⱼ`, which `csc.dot(j, y)` computes as the same sum of
/// the same nonzero products (multiplication commutes; structural zeros add `0.0`
/// exactly; CSC preserves ascending row order). Never materializes the dense matrix.
#[allow(dead_code)] // wired into the driver at T3b5
fn farkas_safe_bound_csc(
    y: &[f64],
    csc: &SparseCols,
    b: &[f64],
    l: &[f64],
    u: &[f64],
    m: usize,
    n: usize,
) -> bool {
    let ynorm = y.iter().fold(0.0f64, |acc, &v| acc.max(v.abs()));
    let rc_tol = 1e-7 * ynorm.max(1.0);
    let mut g = 0.0f64;
    let mut scale = 0.0f64;
    for i in 0..m {
        g += b[i] * y[i];
        scale = scale.max((b[i] * y[i]).abs());
    }
    for j in 0..n {
        let aty = csc.dot(j, y);
        let mut rc = -aty;
        if rc.abs() <= rc_tol {
            rc = 0.0;
        }
        let term = if rc > 0.0 {
            if l[j] <= -INF {
                return false;
            }
            rc * l[j]
        } else if rc < 0.0 {
            if u[j] >= INF {
                return false;
            }
            rc * u[j]
        } else {
            0.0
        };
        g += term;
        scale = scale.max(term.abs());
    }
    let margin = 1e-9 * scale.max(1.0);
    g > margin
}

/// CSC port of [`verify_farkas_infeasible`].
#[allow(dead_code)] // wired into the driver at T3b5
fn verify_farkas_infeasible_csc(
    y: &[f64],
    csc: &SparseCols,
    b: &[f64],
    l: &[f64],
    u: &[f64],
    m: usize,
    n: usize,
) -> bool {
    if y.len() != m || m == 0 {
        return false;
    }
    farkas_safe_bound_csc(y, csc, b, l, u, m, n) || {
        let neg: Vec<f64> = y.iter().map(|v| -v).collect();
        farkas_safe_bound_csc(&neg, csc, b, l, u, m, n)
    }
}

/// Limited strong branching. For the *unreliable* fractional candidates (those
/// whose pseudocosts aren't trusted yet), probe both child bounds with a warm
/// dual re-solve from the node's basis and pick the variable with the best
/// product score `max(Δ↓,ε)·max(Δ↑,ε)` (an infeasible child scores high — it
/// prunes immediately). Returns the chosen structural variable, if any, and the
/// simplex pivots spent. Cheap because each probe is a few warm pivots, and it
/// tapers automatically as pseudocosts mature past the reliability threshold.
fn strong_branch(
    ctx: &NodeCtx<'_>,
    orig_l: &[f64],
    orig_u: &[f64],
    basis: &Basis,
    x: &[f64],
    node_obj: f64,
    cands: &[(usize, f64, u32, f64)],
) -> (Option<usize>, usize, Vec<SbObservation>) {
    let simplex = ctx.simplex;
    // Unreliable candidates, most-fractional (nearest 0.5) first.
    let mut cand: Vec<(usize, f64)> = cands
        .iter()
        .filter(|c| c.2 < ctx.reliability)
        .map(|c| (c.0, c.1))
        .collect();
    if cand.is_empty() {
        return (None, 0, Vec::new());
    }
    cand.sort_by(|a, c| {
        (c.1 - 0.5)
            .abs()
            .partial_cmp(&(a.1 - 0.5).abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    cand.truncate(ctx.opts.sb_max_cands.max(1));

    const INFEAS_DELTA: f64 = 1e7; // a pruned child is a strong branching signal
    let eps = 1e-6;
    // Every probe re-optimizes from the *same* node basis on the *same* matrix,
    // differing only in one bound. Prepare (factorize + verify dual feasibility)
    // that basis once on the batch's pre-scaled matrix (`ctx.sa`/`sc`/`sb`); each
    // probe then clones the pristine factorization instead of refactorizing the
    // identical basis ~2·max_cands times. Branching bounds (floor/ceil of the
    // fractional value) are set in the original space, then scaled to match. The
    // basis is scaling-invariant and the objective gap `obj − node_obj` is
    // invariant, so scores match an unscaled probe — and a wrong score could only
    // pick a worse branching variable, never an unsound bound. If the basis is not
    // warm-startable, fall back to a per-probe scaled warm solve.
    let mut l = orig_l.to_vec();
    let mut u = orig_u.to_vec();
    let scale_bounds = |l: &[f64], u: &[f64]| -> (Vec<f64>, Vec<f64>) {
        match ctx.scaling {
            Some(s) => (s.scale_lower(l), s.scale_upper(u)),
            None => (l.to_vec(), u.to_vec()),
        }
    };
    // Reference scaled bounds (the node's own) at which the basis is dual-feasible.
    let (ref_l, ref_u) = scale_bounds(orig_l, orig_u);
    let prep_view = LpView {
        // T3b3: strong branching solves only through `PreparedDual`/`solve_lp_warm_scaled_csc`,
        // which read the matrix from `ctx.csc`, never `LpView.a`. The `.a` is vestigial —
        // pass an empty slice so this path carries no dense-matrix dependency.
        a: &[],
        m: ctx.m_w,
        n: ctx.n_w,
        c: ctx.sc,
        l: &ref_l,
        u: &ref_u,
    };
    let prepared = PreparedDual::prepare(&prep_view, basis, simplex, ctx.csc);
    let probe = |l: &[f64], u: &[f64]| -> crate::lp::simplex::LpSolve {
        let (sl, su) = scale_bounds(l, u);
        match &prepared {
            Some(p) => p.reoptimize(&sl, &su, ctx.sb, simplex),
            None => {
                let view = LpView {
                    a: &[], // T3b3: matrix comes from `ctx.csc`; `.a` unused here.
                    m: ctx.m_w,
                    n: ctx.n_w,
                    c: ctx.sc,
                    l: &sl,
                    u: &su,
                };
                solve_lp_warm_scaled_csc(&view, ctx.sb, basis, simplex, ctx.csc)
            }
        }
    };
    let mut best: Option<usize> = None;
    let mut best_score = f64::NEG_INFINITY;
    let mut pivots = 0usize;
    // Exact pseudocost samples harvested from the probes: `(var, frac, Δobj,
    // is_down)`. Each optimal probe *is* a pseudocost observation (the canonical
    // reliability-branching feedback); recording them lets a variable reach the
    // reliability threshold and drop out of strong branching at later nodes,
    // instead of being re-probed every time it turns up fractional. Infeasible /
    // non-optimal probes are excluded — a pruned child is a branching signal, not
    // a finite degradation sample, and feeding it would corrupt the average.
    let mut obs: Vec<SbObservation> = Vec::new();
    for (idx, _f) in cand {
        let xi = x[idx];
        let (lo0, hi0) = (orig_l[idx], orig_u[idx]);
        let frac = xi - xi.floor();

        // Down branch: x_idx ≤ floor(x_idx).
        u[idx] = xi.floor();
        let dn = probe(&l, &u);
        u[idx] = hi0;
        pivots += dn.iters;
        let d_dn = match dn.status {
            LpStatus::Optimal => {
                let d = (dn.obj - node_obj).max(0.0);
                obs.push((idx, frac, d, true));
                d
            }
            LpStatus::Infeasible => INFEAS_DELTA,
            _ => 0.0,
        };

        // Up branch: x_idx ≥ ceil(x_idx).
        l[idx] = xi.ceil();
        let up = probe(&l, &u);
        l[idx] = lo0;
        pivots += up.iters;
        let d_up = match up.status {
            LpStatus::Optimal => {
                let d = (up.obj - node_obj).max(0.0);
                obs.push((idx, frac, d, false));
                d
            }
            LpStatus::Infeasible => INFEAS_DELTA,
            _ => 0.0,
        };

        let score = d_dn.max(eps) * d_up.max(eps);
        if score > best_score {
            best_score = score;
            best = Some(idx);
        }
    }
    (best, pivots, obs)
}

/// Opt-out (`DISCOPT_MILP_RC_FIX_REFACTOR=1`) that forces reduced-cost fixing to
/// re-derive the node duals from a fresh sparse LU instead of reusing the ones the
/// LP solve already exported. Off by default; it exists so the reuse path and the
/// legacy refactor path can be run against each other on the same binary (the
/// differential test in this module, and any interleaved A/B timing per CLAUDE.md §9).
/// Whether objective-lattice fathoming is active (`DISCOPT_OBJ_INTEGRALITY`,
/// default ON; set to `0` for the legacy incumbent-only cutoff).
///
/// A bound-changing lever, so it ships with the CLAUDE.md §5 opt-out intact: the
/// OFF arm is the exact pre-change search, which is what the differential panel
/// A/Bs against.
fn obj_integrality_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !std::env::var("DISCOPT_OBJ_INTEGRALITY")
            .is_ok_and(|v| matches!(v.trim(), "0" | "false" | "no"))
    })
}

fn rc_fix_force_refactor() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DISCOPT_MILP_RC_FIX_REFACTOR").is_ok_and(|v| v.trim() == "1"))
}

/// Reduced-cost (objective) fixing at a node. Given the node's optimal basis,
/// its LP dual bound `node_obj` (= `z`), and the incumbent `incumbent` (= `U`,
/// both in the engine's minimize sense), tighten each nonbasic integer
/// variable's bound: a variable at its lower bound with reduced cost `d > 0` can
/// rise at most `⌊(U − z)/d⌋` units before the objective reaches `U`, so any
/// improving solution keeps it within that — symmetrically at the upper bound.
///
/// Duals are recovered from the (scaling-invariant) basis on the **unscaled**
/// working matrix: solve `Bᵀ y = c_B`, then `d_j = c_j − A_jᵀ y`. Returns the
/// tightened structural bounds `(l, u)` when anything changed, else `None`.
///
/// Sound: it only ever removes solutions whose objective is `≥ U` (no better than
/// the incumbent). A small positive slack on the gap and an inward integer floor
/// keep numerical error on the *safe* side (never fixing out an improving point);
/// a singular/ill-conditioned basis solve returns `None` (no fixing).
#[allow(clippy::too_many_arguments)]
fn reduced_cost_fix(
    sp: &SparseCols,
    m: usize,
    c: &[f64],
    basis: &Basis,
    dual: &[f64],
    node_obj: f64,
    incumbent: f64,
    l: &[f64],
    u: &[f64],
    ns: usize,
    is_int: &[bool],
    tol: f64,
) -> Option<(Vec<f64>, Vec<f64>)> {
    if !incumbent.is_finite() || m == 0 {
        return None;
    }
    // Gap U − z, with a small positive slack so floating-point noise can only
    // *loosen* the fixing (never cut an improving solution).
    let gap = (incumbent - node_obj) + 1e-6 * (1.0 + incumbent.abs());
    if gap <= 0.0 {
        return None; // node should be pruned anyway; nothing improving here
    }

    // Node duals `y = B⁻ᵀ c_B`. The LP solve already computed exactly this vector
    // from its own final factorization and exported it as `LpSolve::dual` (both the
    // warm dual and the cold primal fill it on `Optimal`), so #1066 takes it as-is
    // instead of rebuilding the basis' sparse LU from scratch. That rebuild was a
    // second full factorization *per node* — on rsyn0820m02m it was 47% of every LU
    // factorization in the solve — spent recomputing a value already in hand.
    //
    // The fallback below is not dead: `dual` is empty when the solve's own btran
    // failed, and `DISCOPT_MILP_RC_FIX_REFACTOR=1` forces it so the two paths can be
    // differentially tested against each other. It uses the *sparse* LU (factorize
    // O(nnz) + btran), not a dense m×m refactor: the earlier dense `solve_dense` was
    // O(m³) per node, trivial on few-row knapsacks but ruinous on 800–1500-row
    // covering LPs. Either way the basis is scaling-invariant and `sp`/`c` are
    // unscaled, so `y` is the unscaled dual the integer bound fixing needs, and a
    // singular basis ⇒ `None` (sound: just no fixing).
    let y: Vec<f64> =
        if !rc_fix_force_refactor() && dual.len() == m && dual.iter().all(|v| v.is_finite()) {
            crate::profile::incr(crate::profile::Ctr::RcFixDualReuse);
            dual.to_vec()
        } else {
            crate::profile::incr(crate::profile::Ctr::RcFixRefactor);
            let mut lu = FeralLU::new();
            let cols: Vec<Vec<(usize, f64)>> = basis
                .basic_vars
                .iter()
                .map(|&bv| {
                    let (rows, vals) = sp.col(bv);
                    rows.iter().zip(vals).map(|(&r, &v)| (r, v)).collect()
                })
                .collect();
            if lu.factorize_sparse(m, &cols).is_err() {
                return None;
            }
            let mut y: Vec<f64> = basis.basic_vars.iter().map(|&bv| c[bv]).collect();
            if lu.btran(&mut y).is_err() || !y.iter().all(|v| v.is_finite()) {
                return None;
            }
            y
        };

    let mut new_l = l[..ns].to_vec();
    let mut new_u = u[..ns].to_vec();
    let mut changed = false;
    for j in 0..ns {
        if !is_int[j] || basis.col_status[j] == BASIC {
            continue;
        }
        // Reduced cost d_j = c_j − A_jᵀ y (sparse dot over column j's nonzeros).
        let dj = c[j] - sp.dot(j, &y);
        match basis.col_status[j] {
            x if x == AT_LOWER && dj > tol => {
                let maxk = (gap / dj).floor();
                let nu = new_l[j] + maxk;
                if nu < new_u[j] - 0.5 {
                    new_u[j] = nu;
                    changed = true;
                }
            }
            x if x == AT_UPPER && dj < -tol => {
                let maxk = (gap / -dj).floor();
                let nl = new_u[j] - maxk;
                if nl > new_l[j] + 0.5 {
                    new_l[j] = nl;
                    changed = true;
                }
            }
            _ => {}
        }
    }
    if changed {
        Some((new_l, new_u))
    } else {
        None
    }
}

/// Append `cuts` (each `coeffs·x ≥ rhs`, coeffs length `n_w`) to the working LP
/// as `coeffs·x − s = rhs` surplus rows (`s ≥ 0`), growing the dense matrix and
/// the bound/cost/integrality vectors. Returns the new `(m, n)`.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // retained as the differential oracle for the CSC port (tests)
fn augment_with_cuts(
    a_w: &mut Vec<f64>,
    b_w: &mut Vec<f64>,
    c_w: &mut Vec<f64>,
    l_w: &mut Vec<f64>,
    u_w: &mut Vec<f64>,
    is_int_full: &mut Vec<bool>,
    m_w: usize,
    n_w: usize,
    cuts: &[GomoryCut],
) -> (usize, usize) {
    let k = cuts.len();
    if k == 0 {
        return (m_w, n_w);
    }
    let _t = crate::profile::Timer::new(crate::profile::Phase::Augment);
    let (m_old, n_old) = (m_w, n_w);
    let (m_new, n_new) = (m_old + k, n_old + k);
    let mut a_new = vec![0.0; m_new * n_new];
    for i in 0..m_old {
        a_new[i * n_new..i * n_new + n_old].copy_from_slice(&a_w[i * n_old..(i + 1) * n_old]);
    }
    for (ci, cut) in cuts.iter().enumerate() {
        let row = m_old + ci;
        let w = cut.coeffs.len().min(n_old);
        a_new[row * n_new..row * n_new + w].copy_from_slice(&cut.coeffs[..w]);
        a_new[row * n_new + n_old + ci] = -1.0; // surplus: coeffs·x − s = rhs
        b_w.push(cut.rhs);
        c_w.push(0.0);
        l_w.push(0.0);
        u_w.push(INF);
        is_int_full.push(false);
    }
    *a_w = a_new;
    (m_new, n_new)
}

/// CSC analogue of [`augment_with_cuts`]'s MATRIX augmentation (docs/dev/sparse-milp-plan.md,
/// T3): append `k` cut rows and `k` surplus-slack columns to a column-major matrix,
/// producing EXACTLY the nonzeros `from_dense(augment_with_cuts(dense,..))` would.
/// Bit-identical by construction and `O(nnz + cut_nnz)` — never materializes the dense
/// `m×n` matrix, which is what lets the driver drop the dense `a_w` at T3b. The caller
/// appends to `b/c/l/u/is_int` itself (those are independent of the matrix layout).
#[allow(dead_code)] // wired into the driver at T3b (replaces the dense a_w path)
fn augment_cols_with_cuts(sp: &SparseCols, m: usize, n: usize, cuts: &[GomoryCut]) -> SparseCols {
    let k = cuts.len();
    if k == 0 {
        return sp.clone();
    }
    let (col_ptr, row_idx, vals) = sp.raw();
    let mut new_col_ptr: Vec<usize> = Vec::with_capacity(n + k + 1);
    let mut new_row_idx: Vec<usize> = Vec::with_capacity(row_idx.len() + n * k + k);
    let mut new_vals: Vec<f64> = Vec::with_capacity(vals.len() + n * k + k);
    new_col_ptr.push(0);
    for j in 0..n {
        // Existing entries (rows 0..m, already sorted ascending) …
        for idx in col_ptr[j]..col_ptr[j + 1] {
            new_row_idx.push(row_idx[idx]);
            new_vals.push(vals[idx]);
        }
        // … then this column's coefficient in each cut row (rows m..m+k, strictly
        // greater, so the column stays row-sorted). Matches the dense path's
        // `coeffs[..min(len, n)]`: columns j >= cut.coeffs.len() contribute nothing.
        for (ci, cut) in cuts.iter().enumerate() {
            if let Some(&v) = cut.coeffs.get(j) {
                if v != 0.0 {
                    new_row_idx.push(m + ci);
                    new_vals.push(v);
                }
            }
        }
        new_col_ptr.push(new_row_idx.len());
    }
    // Surplus slack columns n..n+k: a singleton `-1.0` at the cut's row.
    for ci in 0..k {
        new_row_idx.push(m + ci);
        new_vals.push(-1.0);
        new_col_ptr.push(new_row_idx.len());
    }
    SparseCols::from_csc(new_col_ptr, new_row_idx, new_vals)
}

/// Rebuild the pre-cut matrix with an explicit, ordered subset of root cuts.
///
/// [`augment_cols_with_cuts`] appends a *batch* of cuts to a matrix whose column
/// count already covers every column those cuts reference; it scans `j in 0..n`
/// and therefore cannot place a coefficient that lands on a slack column the same
/// call is creating. That is sound for the root loop, where a round's cuts are
/// separated off the matrix as it stood *before* the round. It is not sound for a
/// rebuild: a GMI cut's coefficient vector spans all columns of the LP it was read
/// off, cut slacks included, so replaying round 7's cuts onto the original matrix
/// would silently drop their coefficients on rounds 1-6's slack columns and leave
/// a *different, unjustified* row in the LP.
///
/// This builds the whole augmented matrix in one pass instead. `cuts[i].coeffs`
/// may reference any column `< n_base + i` — the original columns plus the slacks
/// of the cuts before it — and every such coefficient is placed. Cuts referencing
/// a *dropped* cut's slack are the caller's problem: it must have closed the
/// dependency set first (see the cleanup in `solve_milp`).
fn rebuild_csc_with_cuts(
    base: &SparseCols,
    m_base: usize,
    n_base: usize,
    cuts: &[GomoryCut],
) -> SparseCols {
    let k = cuts.len();
    if k == 0 {
        return base.clone();
    }
    let (col_ptr, row_idx, vals) = base.raw();
    let mut new_col_ptr: Vec<usize> = Vec::with_capacity(n_base + k + 1);
    let mut new_row_idx: Vec<usize> = Vec::with_capacity(row_idx.len() + n_base * k + k);
    let mut new_vals: Vec<f64> = Vec::with_capacity(vals.len() + n_base * k + k);
    new_col_ptr.push(0);
    // Original columns: their base entries (rows 0..m_base) then their coefficient
    // in each cut row (rows m_base.., strictly greater, so rows stay sorted).
    for j in 0..n_base {
        for idx in col_ptr[j]..col_ptr[j + 1] {
            new_row_idx.push(row_idx[idx]);
            new_vals.push(vals[idx]);
        }
        for (ci, cut) in cuts.iter().enumerate() {
            if let Some(&v) = cut.coeffs.get(j) {
                if v != 0.0 {
                    new_row_idx.push(m_base + ci);
                    new_vals.push(v);
                }
            }
        }
        new_col_ptr.push(new_row_idx.len());
    }
    // Cut slack columns: the surplus `-1` on the cut's own row, then any later
    // cut's coefficient on this slack (again a strictly greater row index).
    for c in 0..k {
        new_row_idx.push(m_base + c);
        new_vals.push(-1.0);
        for (ci, cut) in cuts.iter().enumerate().skip(c + 1) {
            if let Some(&v) = cut.coeffs.get(n_base + c) {
                if v != 0.0 {
                    new_row_idx.push(m_base + ci);
                    new_vals.push(v);
                }
            }
        }
        new_col_ptr.push(new_row_idx.len());
    }
    SparseCols::from_csc(new_col_ptr, new_row_idx, new_vals)
}

/// Full CSC analogue of [`augment_with_cuts`] (T3b5): augment the matrix via
/// [`augment_cols_with_cuts`] AND append the `k` surplus rows/cols to
/// `b/c/l/u/is_int` exactly as the dense version does. Returns the grown CSC and new
/// `(m, n)`.
#[allow(clippy::too_many_arguments)]
fn augment_csc_with_cuts(
    csc: &SparseCols,
    b_w: &mut Vec<f64>,
    c_w: &mut Vec<f64>,
    l_w: &mut Vec<f64>,
    u_w: &mut Vec<f64>,
    is_int_full: &mut Vec<bool>,
    m_w: usize,
    n_w: usize,
    cuts: &[GomoryCut],
) -> (SparseCols, usize, usize) {
    let k = cuts.len();
    if k == 0 {
        return (csc.clone(), m_w, n_w);
    }
    let new_csc = augment_cols_with_cuts(csc, m_w, n_w, cuts);
    for cut in cuts {
        b_w.push(cut.rhs);
        c_w.push(0.0);
        l_w.push(0.0);
        u_w.push(INF);
        is_int_full.push(false);
    }
    (new_csc, m_w + k, n_w + k)
}

/// Sparse signature of a cut for pool deduplication: its nonzero `(col, coeff)`
/// pairs (quantized) plus the rhs, so an identical cut found at many nodes is
/// added to the pool only once.
fn cut_signature(cut: &GomoryCut) -> Vec<(u32, i64)> {
    let mut s: Vec<(u32, i64)> = cut
        .coeffs
        .iter()
        .enumerate()
        .filter(|(_, &v)| v != 0.0)
        .map(|(j, &v)| (j as u32, (v * 1e6).round() as i64))
        .collect();
    s.push((u32::MAX, (cut.rhs * 1e6).round() as i64));
    s
}

/// Keep only cuts whose signature is new (recording it), up to a pool budget.
fn dedup_new_cuts(
    cuts: Vec<GomoryCut>,
    sigs: &mut HashSet<Vec<(u32, i64)>>,
    max_pool: usize,
) -> Vec<GomoryCut> {
    let mut out = Vec::new();
    for cut in cuts {
        if sigs.len() >= max_pool {
            break;
        }
        if sigs.insert(cut_signature(&cut)) {
            out.push(cut);
        }
    }
    out
}

/// Extend a stored basis to the current matrix size after cuts were appended:
/// each new column is a cut's surplus slack (a −e on its new row), so making the
/// new slacks basic gives a valid (block-triangular, nonsingular) basis the dual
/// simplex can repair from. No-op when the basis already spans the matrix.
fn extend_basis(mut basis: Basis, n_w: usize) -> Basis {
    let n0 = basis.col_status.len();
    for j in n0..n_w {
        basis.col_status.push(BASIC);
        basis.basic_vars.push(j);
    }
    basis
}

/// Rounding primal heuristic: round the integer variables of an LP point and,
/// if the rounded point satisfies the original `≤` rows and the (global)
/// variable bounds, return it with its objective. Tries nearest-rounding, then
/// floor (which can only lower the activity of a nonnegative-weight `≤` row, so
/// it is feasible for knapsack-like rows). Returns the better feasible candidate.
/// Cheap (`O(ns · n_orig_rows)`); an early incumbent prunes the whole tree.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // retained as the differential oracle for the CSC port (tests)
fn try_rounding(
    x: &[f64],
    ns: usize,
    is_int: &[bool],
    a_w: &[f64],
    b_w: &[f64],
    c_w: &[f64],
    l_w: &[f64],
    u_w: &[f64],
    n_orig_rows: usize,
    n_w: usize,
    obj_const: f64,
) -> Option<(Vec<f64>, f64)> {
    let feasible = |xc: &[f64]| -> bool {
        for i in 0..n_orig_rows {
            let mut act = 0.0;
            for j in 0..ns {
                act += a_w[i * n_w + j] * xc[j];
            }
            // The row's slack columns must cover the residual `b - act`. Sum the
            // achievable range of the slack contributions over this row: an
            // equality row has no slack (range [0, 0], so `act` must equal `b`);
            // a `<=` row a non-negative slack (range [0, +∞), so `act <= b`); a
            // `>=` row a non-positive one. A plain `act <= b` test is unsound for
            // equality rows — it wrongly accepts e.g. all-zeros for `Σx == k`,
            // injecting an infeasible incumbent (the zero-objective feasibility
            // MILP failure). Using the slack bounds makes the check correct for
            // every row sense.
            let resid = b_w[i] - act;
            let mut lo = 0.0;
            let mut hi = 0.0;
            for k in ns..n_w {
                let aik = a_w[i * n_w + k];
                if aik == 0.0 {
                    continue;
                }
                let (c1, c2) = (aik * l_w[k], aik * u_w[k]);
                lo += c1.min(c2);
                hi += c1.max(c2);
            }
            if resid < lo - 1e-6 || resid > hi + 1e-6 {
                return false;
            }
        }
        true
    };
    let obj = |xc: &[f64]| -> f64 { (0..ns).map(|j| c_w[j] * xc[j]).sum::<f64>() + obj_const };

    let make = |round: &dyn Fn(f64) -> f64| -> Vec<f64> {
        (0..ns)
            .map(|j| {
                let v = if is_int[j] { round(x[j]) } else { x[j] };
                // Guard against rounding-induced bound inversion (l_w[j] a few ULP
                // above u_w[j] on a near-fixed variable): f64::clamp panics when
                // min > max. Clamp into the well-ordered interval — identical to
                // the direct clamp when bounds are ordered, and collapses to the
                // degenerate (ULP-wide) box when they cross.
                let (lo, hi) = if l_w[j] <= u_w[j] {
                    (l_w[j], u_w[j])
                } else {
                    (u_w[j], l_w[j])
                };
                v.clamp(lo, hi)
            })
            .collect()
    };

    let mut best: Option<(Vec<f64>, f64)> = None;
    let mut consider = |xc: Vec<f64>| {
        if feasible(&xc) {
            let o = obj(&xc);
            if best.as_ref().map(|(_, bo)| o < *bo).unwrap_or(true) {
                best = Some((xc, o));
            }
        }
    };
    consider(make(&|v: f64| v.round()));
    consider(make(&|v: f64| v.floor()));
    best
}

/// CSC port of [`try_rounding`] (docs/dev/sparse-milp-plan.md T3b1). Bit-identical:
/// the slack range (`slack_lo/hi`, xc-independent) and the per-row activity `act` are
/// accumulated by CSC column iteration in the SAME ascending order as the dense
/// per-row loops, and a structural `0.0` adds exactly, so every row sum matches
/// term-for-term. Never materializes the dense matrix.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // wired into the driver at T3b5
fn try_rounding_csc(
    x: &[f64],
    ns: usize,
    is_int: &[bool],
    csc: &SparseCols,
    b_w: &[f64],
    c_w: &[f64],
    l_w: &[f64],
    u_w: &[f64],
    n_orig_rows: usize,
    n_w: usize,
    obj_const: f64,
) -> Option<(Vec<f64>, f64)> {
    let (col_ptr, row_idx, vals) = csc.raw();
    let mut slack_lo = vec![0.0f64; n_orig_rows];
    let mut slack_hi = vec![0.0f64; n_orig_rows];
    for k in ns..n_w {
        for idx in col_ptr[k]..col_ptr[k + 1] {
            let i = row_idx[idx];
            if i >= n_orig_rows {
                continue;
            }
            let aik = vals[idx];
            let (c1, c2) = (aik * l_w[k], aik * u_w[k]);
            slack_lo[i] += c1.min(c2);
            slack_hi[i] += c1.max(c2);
        }
    }
    let feasible = |xc: &[f64]| -> bool {
        let mut act = vec![0.0f64; n_orig_rows];
        for j in 0..ns {
            for idx in col_ptr[j]..col_ptr[j + 1] {
                let i = row_idx[idx];
                if i >= n_orig_rows {
                    continue;
                }
                act[i] += vals[idx] * xc[j];
            }
        }
        for i in 0..n_orig_rows {
            let resid = b_w[i] - act[i];
            if resid < slack_lo[i] - 1e-6 || resid > slack_hi[i] + 1e-6 {
                return false;
            }
        }
        true
    };
    let obj = |xc: &[f64]| -> f64 { (0..ns).map(|j| c_w[j] * xc[j]).sum::<f64>() + obj_const };
    let make = |round: &dyn Fn(f64) -> f64| -> Vec<f64> {
        (0..ns)
            .map(|j| {
                let v = if is_int[j] { round(x[j]) } else { x[j] };
                let (lo, hi) = if l_w[j] <= u_w[j] {
                    (l_w[j], u_w[j])
                } else {
                    (u_w[j], l_w[j])
                };
                v.clamp(lo, hi)
            })
            .collect()
    };
    let mut best: Option<(Vec<f64>, f64)> = None;
    let mut consider = |xc: Vec<f64>| {
        if feasible(&xc) {
            let o = obj(&xc);
            if best.as_ref().map(|(_, bo)| o < *bo).unwrap_or(true) {
                best = Some((xc, o));
            }
        }
    };
    consider(make(&|v: f64| v.round()));
    consider(make(&|v: f64| v.floor()));
    best
}

/// Validate a caller-supplied structural seed point (see
/// [`MilpOptions::initial_incumbent`]) and return `(x, cᵀx)` ready for
/// `inject_incumbent`, or `None` when it cannot be *proven* feasible here.
/// Mirrors [`try_rounding_csc`]'s feasibility test: integer columns must sit
/// on integer values (within `1e-6`, then snapped exactly), every column must
/// lie within its (presolve-tightened) bounds (within `1e-6`, then clamped),
/// and each original row's residual must be coverable by its slack columns'
/// bound range. The objective is recomputed from `c_w` — a caller-claimed
/// value is never trusted. Rejection is silent: an unverifiable seed simply
/// does not seed, so a bad caller point can never prune the true optimum.
// The arguments ARE the standard-form LP slices, same shape as the sibling
// heuristics (`try_rounding_csc`); a parameter struct would only obscure that.
#[allow(clippy::too_many_arguments)]
fn validate_seed_incumbent(
    seed: &[f64],
    ns: usize,
    is_int: &[bool],
    csc: &SparseCols,
    b_w: &[f64],
    c_w: &[f64],
    l_w: &[f64],
    u_w: &[f64],
    n_orig_rows: usize,
    n_w: usize,
) -> Option<(Vec<f64>, f64)> {
    const TOL: f64 = 1e-6;
    if seed.len() != ns {
        return None;
    }
    let mut x = Vec::with_capacity(ns);
    for j in 0..ns {
        let mut v = seed[j];
        if !v.is_finite() {
            return None;
        }
        if is_int[j] {
            let r = v.round();
            if (v - r).abs() > TOL {
                return None;
            }
            v = r;
        }
        let (lo, hi) = if l_w[j] <= u_w[j] {
            (l_w[j], u_w[j])
        } else {
            (u_w[j], l_w[j])
        };
        if v < lo - TOL || v > hi + TOL {
            return None;
        }
        x.push(v.clamp(lo, hi));
    }
    let (col_ptr, row_idx, vals) = csc.raw();
    let mut slack_lo = vec![0.0f64; n_orig_rows];
    let mut slack_hi = vec![0.0f64; n_orig_rows];
    for k in ns..n_w {
        for idx in col_ptr[k]..col_ptr[k + 1] {
            let i = row_idx[idx];
            if i >= n_orig_rows {
                continue;
            }
            let aik = vals[idx];
            let (c1, c2) = (aik * l_w[k], aik * u_w[k]);
            slack_lo[i] += c1.min(c2);
            slack_hi[i] += c1.max(c2);
        }
    }
    let mut act = vec![0.0f64; n_orig_rows];
    for (j, &xj) in x.iter().enumerate() {
        if xj == 0.0 {
            continue;
        }
        for idx in col_ptr[j]..col_ptr[j + 1] {
            let i = row_idx[idx];
            if i >= n_orig_rows {
                continue;
            }
            act[i] += vals[idx] * xj;
        }
    }
    for i in 0..n_orig_rows {
        let resid = b_w[i] - act[i];
        if resid < slack_lo[i] - TOL || resid > slack_hi[i] + TOL {
            return None;
        }
    }
    let obj = x
        .iter()
        .zip(c_w.iter())
        .map(|(xj, cj)| cj * xj)
        .sum::<f64>();
    Some((x, obj))
}

/// Fractional-diving primal heuristic with continuous repair: repeatedly fix the
/// most-fractional unfixed integer to its nearest integer and **re-solve the LP**
/// (warm-started — the bound-change case the dual simplex re-optimizes cheaply),
/// until every integer is integral (a feasible incumbent) or a fix makes the LP
/// infeasible (dive abandoned). Returns the incumbent (structural `x`, true
/// objective) or `None`.
///
/// Re-solving *between* fixes is the whole point: it repairs the continuous
/// variables to each partial integer assignment and keeps the remaining
/// relaxed integers feasible-fractional, so the dive avoids the infeasible
/// combinations (e.g. cyclic big-M precedences) that defeat one-shot rounding.
/// Plain [`try_rounding`] fixes nothing and never re-solves, so on weak-
/// relaxation (big-M) models it finds no incumbent at all — leaving the search
/// with no bound-based pruning. Up to `n_int` warm solves; run once at the root
/// so the cost is bounded. Sound: integers land on integer values and the final
/// LP optimum satisfies every row, so the point is feasible; the caller's
/// `inject_incumbent` enforces strict improvement.
fn try_dive_repair(
    ctx: &NodeCtx<'_>,
    lb_k: &[f64],
    ub_k: &[f64],
    x_start: &[f64],
    start_basis: &Basis,
) -> Option<(Vec<f64>, f64)> {
    let ns = ctx.ns;
    // Structural bounds, progressively fixed; slacks keep their bounds.
    let mut l = lb_k.to_vec();
    let mut u = ub_k.to_vec();
    let mut x = x_start.to_vec();
    // Warm-start each fix from the previous step's optimal basis: fixing one
    // integer is a single bound change, exactly the dual-simplex re-optimization
    // case (a few pivots), versus a cold phase-1/phase-2 solve from scratch per
    // step. `solve_lp_warm_scaled` falls back to a cold solve on *any* difficulty
    // (a dual-infeasible/over-updated basis, the iteration cap, the wall-clock
    // deadline), so the big-M robustness the cold-per-step recipe gave is retained
    // — the warm path only ever saves time, never changes a result. The dive is a
    // heuristic, so a (possibly different) warm optimum is just as valid an
    // incumbent; `inject_incumbent` still enforces strict improvement and the
    // Python feasibility gate re-checks the point.
    let mut cur_basis = start_basis.clone();
    let max_steps = ctx.is_int.iter().filter(|&&it| it).count() + 1;
    for _ in 0..max_steps {
        // Most-fractional unfixed integer column.
        let mut pick: Option<usize> = None;
        let mut best = INT_TOL;
        for j in 0..ns {
            if ctx.is_int[j] && l[j] != u[j] {
                let f = frac(x[j]);
                if f > best {
                    best = f;
                    pick = Some(j);
                }
            }
        }
        let j = match pick {
            // No fractional integer remains: round the (near-integral) integers
            // exactly and return the repaired point.
            None => {
                let mut xc = x[..ns].to_vec();
                for (k, xk) in xc.iter_mut().enumerate() {
                    if ctx.is_int[k] {
                        *xk = xk.round();
                    }
                }
                let obj = (0..ns).map(|k| ctx.c_w[k] * xc[k]).sum::<f64>() + ctx.obj_const;
                return Some((xc, obj));
            }
            Some(j) => j,
        };
        // Fix the picked integer and re-solve. Try the nearest integer first; if
        // that makes the LP infeasible (the rounded assignment is part of an
        // infeasible combination, e.g. a cyclic big-M precedence), try the other
        // rounding direction before abandoning the dive — a cheap single-variable
        // backtrack that rescues many big-M dives.
        let (lo, hi) = if l[j] <= u[j] {
            (l[j], u[j])
        } else {
            (u[j], l[j])
        };
        let nearest = x[j].round().clamp(lo, hi);
        let other = if x[j] >= nearest {
            (nearest + 1.0).min(hi)
        } else {
            (nearest - 1.0).max(lo)
        };
        let mut next_x: Option<Vec<f64>> = None;
        let mut tried = Vec::with_capacity(2);
        for &v in &[nearest, other] {
            if tried.contains(&v.to_bits()) {
                continue;
            }
            tried.push(v.to_bits());
            // Re-solve on the batch's shared (pre-scaled, when ill-conditioned)
            // matrix — the same robust recipe as the node LP solve. Warm-start
            // from the running basis (one fixed bound → a few dual pivots); the
            // warm path cold-solves on any difficulty, so this is never less
            // robust than the old cold-per-step solve, only faster on the common
            // well-conditioned case.
            let mut full_l = vec![0.0; ctx.n_w];
            let mut full_u = vec![0.0; ctx.n_w];
            full_l[..ns].copy_from_slice(&l);
            full_u[..ns].copy_from_slice(&u);
            full_l[j] = v;
            full_u[j] = v;
            full_l[ns..].copy_from_slice(ctx.slack_l);
            full_u[ns..].copy_from_slice(ctx.slack_u);
            let (sl, su) = match ctx.scaling {
                Some(s) => (s.scale_lower(&full_l), s.scale_upper(&full_u)),
                None => (full_l.clone(), full_u.clone()),
            };
            let view = LpView {
                a: &[], // T3b5: matrix comes from `ctx.csc`; `.a` unused by CSC solve.
                m: ctx.m_w,
                n: ctx.n_w,
                c: ctx.sc,
                l: &sl,
                u: &su,
            };
            let mut sol = solve_lp_warm_scaled_csc(&view, ctx.sb, &cur_basis, ctx.simplex, ctx.csc);
            if sol.status == LpStatus::Optimal {
                if let Some(s) = ctx.scaling {
                    s.unscale_x(&mut sol.x);
                }
                l[j] = v;
                u[j] = v;
                // Carry this step's optimal basis into the next fix's warm start.
                cur_basis = sol.basis;
                next_x = Some(sol.x);
                break;
            }
        }
        // `None` means both roundings were infeasible -> abandon the dive.
        x = next_x?;
    }
    None
}

fn frac(v: f64) -> f64 {
    let f = v - v.floor();
    f.min(1.0 - f)
}

/// Integrality screen for the lazy separator, deliberately identical to the one
/// `TreeManager::process_evaluated` applies (`branching::is_integer_feasible`
/// over the tree's `integer_vars`, tolerance `INTEGRALITY_TOL`).
///
/// It is re-derived here over the structural mask rather than borrowed from the
/// tree because `int_info` is moved into `TreeManager::new`. The two must agree:
/// a point the tree treats as integer-feasible but this screen skips would be
/// promoted to the incumbent without ever reaching the separator. The driver's
/// `is_int` is built from the same `opts.integer_cols`, and every `VarBranchInfo`
/// the driver constructs has `size == 1`, so the predicates coincide.
///
/// `INT_TOL` here and `branching::INTEGRALITY_TOL` there are both 1e-5; the
/// latter is private to that module, so the agreement is pinned by
/// `lazy_screen_matches_tree_integrality_predicate` in the tests below rather
/// than by a shared constant.
fn solution_is_integral(solution: &[f64], is_int: &[bool]) -> bool {
    for (j, &it) in is_int.iter().enumerate() {
        if !it {
            continue;
        }
        // Matches `is_integer_feasible`'s short-circuit on a truncated solution.
        if j >= solution.len() {
            return false;
        }
        if frac(solution[j]) > INT_TOL {
            return false;
        }
    }
    true
}

fn midpoint(lb: &[f64], ub: &[f64]) -> Vec<f64> {
    lb.iter()
        .zip(ub)
        .map(|(&l, &u)| 0.5 * (l.clamp(-INF, INF) + u.clamp(-INF, INF)))
        .collect()
}

/// Build the **slack starting basis** for a standard-form LP: one zero-cost
/// singleton (slack) column per row, basic; every other column nonbasic at the
/// bound that makes its `y = 0` reduced cost `d_j = c_j` dual-feasible (lower if
/// `c_j ≥ 0`, upper if `c_j < 0`). On a covering/packing-style LP this basis is
/// dual-feasible and primal-infeasible — the dual simplex's home turf, solving it
/// in a fraction of the cold primal's phase-1+phase-2 pivots.
///
/// Returns `None` when a row has no available zero-cost singleton, or a nonbasic
/// variable can't be made dual-feasible at a *finite* bound (a free variable with
/// nonzero cost) — the caller then cold-solves. The returned basis is only ever a
/// *hint*: [`solve_lp_warm`] re-verifies dual feasibility and falls back to the
/// cold primal if it does not actually hold, so a wrong guess costs one
/// factorization, never correctness.
// The nonbasic sweep indexes `c`/`l`/`u`/`col_status` by the same `j`, so a range
// loop reads clearer than zipping four slices (matches the simplex modules).
#[allow(clippy::needless_range_loop)]
fn dual_slack_basis(
    sp: &SparseCols,
    m: usize,
    n: usize,
    c: &[f64],
    l: &[f64],
    u: &[f64],
    tol: f64,
) -> Option<Basis> {
    // Assign each row a distinct zero-cost singleton column (a slack).
    let mut row_basic: Vec<i64> = vec![-1; m];
    for j in 0..n {
        if c[j] != 0.0 {
            continue; // slacks carry no objective cost (keeps y = 0)
        }
        let (rows, _vals) = sp.col(j);
        if rows.len() == 1 {
            let i = rows[0];
            if row_basic[i] < 0 {
                row_basic[i] = j as i64;
            }
        }
    }
    if row_basic.iter().any(|&x| x < 0) {
        return None; // some row has no slack (e.g. a pure equality) → cold solve
    }
    let mut is_basic = vec![false; n];
    for &j in &row_basic {
        is_basic[j as usize] = true;
    }
    // Nonbasic columns sit at the dual-feasible bound for d_j = c_j.
    let mut col_status = vec![AT_LOWER; n];
    for j in 0..n {
        if is_basic[j] {
            col_status[j] = BASIC;
        } else if c[j] > tol {
            if l[j] <= -INF {
                return None; // free var, c_j > 0 → not dual-feasible at a bound
            }
            col_status[j] = AT_LOWER;
        } else if c[j] < -tol {
            if u[j] >= INF {
                return None;
            }
            col_status[j] = AT_UPPER;
        } else {
            // |c_j| ≈ 0: dual-feasible at either bound; prefer a finite one.
            col_status[j] = if l[j] > -INF { AT_LOWER } else { AT_UPPER };
        }
    }
    let basic_vars: Vec<usize> = row_basic.iter().map(|&j| j as usize).collect();
    Some(Basis {
        col_status,
        basic_vars,
    })
}

/// Cold-solve an LP, but try the dual simplex from the [`dual_slack_basis`] first
/// (a large win on covering/packing relaxations where the cold primal stalls in
/// degenerate phase-2 pivots). `solve_lp_warm` falls back to the cold primal when
/// the slack basis is unavailable or not actually dual-feasible, so the result is
/// always the same optimum — only the path differs.
///
/// Superseded in the driver by [`solve_lp_root_csc`] (T2); retained as the
/// differential ORACLE the `sparse_milp_diff` tests check the CSC root solve against
/// bit-for-bit. Removed with the dense driver at T5.
#[allow(dead_code)]
fn solve_lp_root(lp: &LpView<'_>, b: &[f64], opts: &SimplexOptions) -> crate::lp::simplex::LpSolve {
    match dual_slack_basis(
        &SparseCols::from_dense(lp.a, lp.m, lp.n),
        lp.m,
        lp.n,
        lp.c,
        lp.l,
        lp.u,
        opts.tol,
    ) {
        Some(basis) => {
            // The dual-slack warm start is an *optimization*, not a requirement: on
            // covering/packing relaxations the dual simplex reaches the optimum in a
            // handful of pivots (the #334 win). But on an ill-conditioned relaxation
            // — e.g. nvs06's geometric-mean-equilibrated McCormick LP (#350) — the
            // dual simplex degenerate-cycles to `max_iter` and burns the whole
            // enclosing MILP budget, while the cold primal solves it instantly. The
            // existing fallback only fires when the slack basis does not *qualify*
            // (`dual_slack_basis` -> None); it does NOT catch a qualifying basis whose
            // dual solve then stalls. Cap the warm attempt to a small pivot budget and
            // fall back to the cold primal when it stalls (IterLimit / Numerical).
            // Optimal/Infeasible/Unbounded warm results are exact and kept.
            let warm = solve_lp_warm(lp, b, &basis, &warm_root_opts(opts, lp.m, lp.n));
            match warm.status {
                LpStatus::IterLimit | LpStatus::Numerical => solve_lp(lp, b, opts),
                _ => warm,
            }
        }
        None => solve_lp(lp, b, opts),
    }
}

/// Sparse-native equivalent of [`solve_lp_root`] (docs/dev/sparse-milp-plan.md, T2).
/// Bit-identical to `solve_lp_root` on the same LP — the dual-slack warm start reads
/// only singleton structure + `c` (scale-invariant), the warm re-solve uses the same
/// `solve_lp_warm_scaled_csc` the node solves already trust with the same pivot cap,
/// and the cold fallback uses [`solve_lp_cols_scaled`] which reproduces `solve_lp`'s
/// `ScaledLp` equilibration exactly. It NEVER materializes the dense `m×n` matrix, so
/// the root relaxation of a large sparse binary QP solves from CSC without the dense
/// blow-up. `cols` is the (unscaled) CSC of the working matrix; `c/l/u/b` and the
/// slack layout match the dense root LP.
#[allow(clippy::too_many_arguments)] // inherent LP signature: cols + m,n,c,l,u,b,opts
fn solve_lp_root_csc(
    cols: &SparseCols,
    m: usize,
    n: usize,
    c: &[f64],
    l: &[f64],
    u: &[f64],
    b: &[f64],
    opts: &SimplexOptions,
) -> crate::lp::simplex::LpSolve {
    match dual_slack_basis(cols, m, n, c, l, u, opts.tol) {
        Some(basis) => {
            let lp = LpView {
                a: &[],
                m,
                n,
                c,
                l,
                u,
            };
            let warm = solve_lp_warm_scaled_csc(&lp, b, &basis, &warm_root_opts(opts, m, n), cols);
            match warm.status {
                LpStatus::IterLimit | LpStatus::Numerical => {
                    solve_lp_cols_scaled(cols.clone(), m, n, c, l, u, b, opts)
                }
                _ => warm,
            }
        }
        None => solve_lp_cols_scaled(cols.clone(), m, n, c, l, u, b, opts),
    }
}

/// Pivot-bounded options for a dual-slack *warm* root attempt. The dual-slack
/// start only ever pays off when it converges quickly (the covering-LP win is a
/// few hundred to a low-thousands pivots); past a generous multiple of the problem
/// size it is stalling, so cap it and let the caller cold-solve. The
/// size-proportional `8·(m+n)` term sits far above the largest validated
/// covering-LP win (sc2000 root ≈ 1384 pivots at m+n ≈ 2800 ⇒ cap ≈ 22400), so
/// that win is untouched; the small absolute floor only affects genuinely tiny LPs.
fn warm_root_opts(opts: &SimplexOptions, m: usize, n: usize) -> SimplexOptions {
    // Covering/packing relaxations converge in O(m+n) dual pivots (sc2000 root:
    // ≈1384 for m+n≈2800), so a generous size-proportional cap preserves that win
    // by a wide margin while a degenerate stall (nvs06: ~max_iter pivots on a tiny
    // ill-conditioned LP, #350) trips it almost immediately. The small floor keeps
    // genuinely tiny LPs from a too-eager bail (their cold solve is cheap anyway).
    let cap = (8 * (m + n)).max(512);
    let mut o = opts.clone();
    o.max_iter = o.max_iter.min(cap);
    o
}

/// Reconstruct the dense row-major `m×n` matrix from a column-major
/// [`SparseCols`]. **Temporary T1 bridge** (docs/dev/sparse-milp-plan.md): the CSC
/// entry densifies here and calls the reference dense driver so the CSC path is
/// provably bit-identical to the dense path, while T2/T3 sparsify the driver
/// internals (root solve, scaling, cut appends) and remove this densification. It
/// therefore does NOT yet fix the memory blow-up on a large sparse relaxation —
/// that is T3's job — it only establishes the entry point and its differential gate.
fn csc_to_dense(csc: &SparseCols, m: usize, n: usize) -> Vec<f64> {
    let mut a = vec![0.0; m * n];
    for j in 0..n {
        let (rows, vals) = csc.col(j);
        for (&i, &v) in rows.iter().zip(vals) {
            a[i * n + j] = v;
        }
    }
    a
}

/// CSC-input entry to the MILP branch-and-bound driver
/// (docs/dev/sparse-milp-plan.md, T1). Identical contract and result to
/// [`solve_milp`], but the equality-constraint matrix arrives column-major as a
/// [`SparseCols`] (`m` rows, `n` columns) so a large *sparse* relaxation need not be
/// densified by the caller / Python boundary. T1 bridges to the dense driver via
/// [`csc_to_dense`]; the differential harness gates it bit-identical to
/// [`solve_milp`] on the panel, and T2/T3 remove the internal densification so the
/// sparse matrix flows through untouched.
#[allow(clippy::too_many_arguments)] // inherent LP signature: csc + m,n,c,l,u,b,obj_const,opts
pub fn solve_milp_csc(
    csc: &SparseCols,
    m: usize,
    n: usize,
    c: &[f64],
    l: &[f64],
    u: &[f64],
    b: &[f64],
    obj_const: f64,
    opts: &MilpOptions,
) -> MilpResult {
    let a = csc_to_dense(csc, m, n);
    let lp = LpView {
        a: &a,
        m,
        n,
        c,
        l,
        u,
    };
    solve_milp(&lp, b, obj_const, opts)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- objective-lattice fathoming (DISCOPT_OBJ_INTEGRALITY) ----

    /// The odd-hole vertex cover on `k` vertices (`k` odd): `min Σ x_i` subject to
    /// `x_i + x_{i+1} >= 1` around the cycle, `x` binary. Its LP relaxation sits at
    /// exactly `k/2` (all halves) while the integer optimum is `(k+1)/2` — a gap of
    /// half a unit that **no** cut in the engine closes and that branching can only
    /// close by enumerating the cycle. The half unit is precisely what objective
    /// integrality is for, so this fixture isolates the rule.
    ///
    /// Working form: `A x = b` with a surplus column per row.
    fn odd_cycle_cover(
        k: usize,
    ) -> (
        SparseCols,
        usize,
        usize,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
    ) {
        assert!(k >= 3 && k % 2 == 1, "odd hole needs an odd k >= 3");
        let (ns, m) = (k, k);
        let n = ns + m;
        let mut dense = vec![0.0f64; m * n];
        for i in 0..m {
            dense[i * n + i] = 1.0;
            dense[i * n + (i + 1) % k] = 1.0;
            dense[i * n + ns + i] = -1.0;
        }
        let sp = SparseCols::from_dense(&dense, m, n);
        let mut c = vec![0.0f64; n];
        for cj in c.iter_mut().take(ns) {
            *cj = 1.0;
        }
        let b = vec![1.0; m];
        let l = vec![0.0; n];
        let mut u = vec![INF; n];
        for uj in u.iter_mut().take(ns) {
            *uj = 1.0;
        }
        (sp, m, ns, c, b, l, u)
    }

    /// Cycle length for the fixture. Odd, and large enough that the OFF arm has to
    /// enumerate rather than stumble into the answer at the root.
    const K: usize = 15;

    /// Driver options for the cover fixture: the engine's own defaults for every
    /// lever, so the test measures the lattice rule and not a bespoke configuration.
    fn cover_opts(ns: usize) -> MilpOptions {
        MilpOptions {
            n_struct: ns,
            integer_cols: (0..ns).collect(),
            max_nodes: 100_000,
            time_limit_s: None,
            gap_tol: 1e-9,
            root_cuts: 0,
            cut_rounds: 1,
            gmi_cuts: false,
            cut_select: false,
            node_cuts: false,
            max_pool_cuts: 0,
            heuristics: true,
            presolve: true,
            strong_branch: true,
            node_propagation: false,
            reduced_cost_fixing: true,
            sb_max_cands: 8,
            sb_node_budget: 1000,
            initial_incumbent: None,
            node_hook_rounds: 0,
            node_hook_cut_cap: 0,
            root_cut_time_s: None,
            root_cut_prune: true,
            simplex: SimplexOptions::default(),
        }
    }

    #[test]
    fn odd_cycle_cover_lp_bound_really_is_fractional() {
        // Anti-vacuity (CLAUDE.md §6): if the relaxation were already integral the
        // fathoming test below would prove nothing about the lattice rule.
        let (sp, m, ns, c, b, l, u) = odd_cycle_cover(K);
        let n = l.len();
        let sol =
            crate::lp::simplex::solve_lp_cols(sp, m, n, &c, &l, &u, &b, &SimplexOptions::default());
        assert_eq!(sol.status, LpStatus::Optimal);
        assert!(
            (sol.obj - K as f64 / 2.0).abs() < 1e-6,
            "fixture must relax to k/2, got {} (ns={ns})",
            sol.obj
        );
    }

    /// End-to-end soundness: the driver wires the lattice in, and the answer is
    /// still right. The node-level behaviour of the rule is pinned in
    /// `tree_manager`'s tests; what this covers is the wiring — a granularity read
    /// off the wrong vector, or applied in the wrong objective sense, shows up here
    /// as a wrong optimum or a bound above it.
    #[test]
    fn odd_cycle_cover_solves_correctly_with_the_lattice_wired_in() {
        let (sp, m, ns, c, b, l, u) = odd_cycle_cover(K);
        let n = l.len();
        let is_int: Vec<bool> = (0..n).map(|j| j < ns).collect();
        // Anti-vacuity (CLAUDE.md §6): the driver's detector must actually fire on
        // this fixture, or the run below exercises nothing.
        assert_eq!(
            crate::bnb::obj_integral::objective_granularity(&c[..ns], &is_int[..ns]),
            Some(1.0),
            "fixture must present a unit objective lattice"
        );
        let opts = cover_opts(ns);
        let res = solve_milp_csc(&sp, m, n, &c, &l, &u, &b, 0.0, &opts);
        let optimum = ((K + 1) / 2) as f64;
        assert_eq!(
            res.status,
            MilpStatus::Optimal,
            "must certify, not merely find"
        );
        assert!(
            (res.obj - optimum).abs() < 1e-6,
            "optimum is {optimum}, got {} -- a wrong objective here means the cutoff \
             fathomed the optimum (CLAUDE.md §1)",
            res.obj
        );
        assert!(
            res.bound <= optimum + 1e-6,
            "dual bound {} exceeds the true optimum -- false certificate",
            res.bound
        );
    }

    /// Same cover, but one unit of continuous cost. The objective can now take any
    /// value, so the detector must refuse — pruning on `U - 1` here could fathom a
    /// solution worth `U - 0.3`.
    #[test]
    fn a_continuous_cost_column_disables_the_lattice_rule() {
        let (sp, m, ns, mut c, b, l, mut u) = odd_cycle_cover(K);
        let n = l.len();
        // Give the first surplus column a real cost and a finite bound.
        c[ns] = 0.3;
        u[ns] = 1.0;
        let is_int: Vec<bool> = (0..n).map(|j| j < ns).collect();
        assert_eq!(
            crate::bnb::obj_integral::objective_granularity(&c, &is_int),
            None,
            "a costed continuous column must disable the lattice"
        );
        let opts = cover_opts(ns);
        let res = solve_milp_csc(&sp, m, n, &c, &l, &u, &b, 0.0, &opts);
        assert_eq!(res.status, MilpStatus::Optimal);
        assert!(
            res.bound <= res.obj + 1e-6,
            "bound {} above incumbent {}",
            res.bound,
            res.obj
        );
    }

    // ---- #1066: reduced-cost fixing reuses the LP's own duals ----

    /// A small covering-style LP with integer structurals, in the driver's working
    /// form `A x = b, l <= x <= u` (explicit slacks), plus an incumbent loose enough
    /// that the gap actually admits fixings.
    fn rc_fix_fixture() -> (
        SparseCols,
        usize,
        usize,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
    ) {
        // min  4a + 3b + 5c   s.t.  a + b + c >= 1,  a + 2b + 4c >= 1,  0 <= . <= 3
        // Rows carry surplus columns (coefficient -1) to reach equality form.
        //
        // The upper bound is 3, not 1, on purpose: with a unit box the optimum puts
        // `b` ON its bound, so every integer column comes back nonbasic-at-bound or
        // basic-and-skipped and reduced-cost fixing has nothing to fix. Here the
        // optimum is `b = 1` (basic, interior), `a = c = 0` at their lower bounds
        // with strictly positive reduced costs 1 and 2 -- so the fixing actually
        // bites and a test over this fixture is not vacuous (CLAUDE.md §6).
        let (ns, m) = (3usize, 2usize);
        let n = ns + m;
        let mut dense = vec![0.0f64; m * n];
        let rows = [[1.0, 1.0, 1.0], [1.0, 2.0, 4.0]];
        for (i, r) in rows.iter().enumerate() {
            for (j, v) in r.iter().enumerate() {
                dense[i * n + j] = *v;
            }
            dense[i * n + ns + i] = -1.0;
        }
        let sp = SparseCols::from_dense(&dense, m, n);
        let c = vec![4.0, 3.0, 5.0, 0.0, 0.0];
        let b = vec![1.0, 1.0];
        let l = vec![0.0; n];
        let u = vec![3.0, 3.0, 3.0, INF, INF];
        (sp, m, ns, c, b, l, u)
    }

    #[test]
    fn rc_fix_reuses_the_solve_duals_and_matches_the_refactor_path() {
        // The #1066 claim: the duals the LP solve exports ARE `y = B^-T c_B`, so
        // reusing them must produce exactly the fixings the from-scratch sparse
        // refactorization produced. If the two disagree the reuse is not a
        // bound-neutral refactor and must not ship.
        let (sp, m, ns, c, b, l, u) = rc_fix_fixture();
        let n = l.len();
        let opts = SimplexOptions::default();
        let sol = crate::lp::simplex::solve_lp_cols(sp.clone(), m, n, &c, &l, &u, &b, &opts);
        assert_eq!(sol.status, LpStatus::Optimal, "fixture LP must solve");
        assert_eq!(
            sol.dual.len(),
            m,
            "Optimal must export row duals (#1066 rests on it)"
        );
        let is_int = vec![true, true, true, false, false];

        let mut checks = 0;
        let mut fixed = 0;
        // Sweep incumbents so the gap ranges from tight (fixings bite) to loose
        // (none do): both regimes must agree, not just the one that happens to fix.
        for k in 0..12 {
            let incumbent = sol.obj + 0.25 * k as f64;
            let reuse = reduced_cost_fix(
                &sp, m, &c, &sol.basis, &sol.dual, sol.obj, incumbent, &l, &u, ns, &is_int,
                opts.tol,
            );
            // An empty `dual` is exactly what a failed btran leaves behind, and it
            // is what drives the legacy from-scratch factorization branch.
            let refac = reduced_cost_fix(
                &sp,
                m,
                &c,
                &sol.basis,
                &[],
                sol.obj,
                incumbent,
                &l,
                &u,
                ns,
                &is_int,
                opts.tol,
            );
            assert_eq!(
                reuse.is_some(),
                refac.is_some(),
                "k={k}: the two dual sources disagree on whether anything was fixed"
            );
            if let (Some((rl, ru)), Some((fl, fu))) = (reuse, refac) {
                assert_eq!(rl, fl, "k={k}: lower bounds differ between dual sources");
                assert_eq!(ru, fu, "k={k}: upper bounds differ between dual sources");
                fixed += 1;
            }
            checks += 1;
        }
        assert_eq!(checks, 12, "probe must have compared every incumbent");
        // Agreement on twelve `None`s would be agreement about nothing: at least one
        // rung must actually have produced a fixing for the comparison to mean
        // anything (CLAUDE.md §6).
        assert!(
            fixed >= 1,
            "no incumbent on the ladder produced a fixing at all"
        );
    }

    #[test]
    fn rc_fix_with_solve_duals_does_not_factorize() {
        // The point of the change is the factorization it removes; without a probe
        // for it the reuse could silently fall through to the legacy branch on every
        // node and read exactly like a working optimization (CLAUDE.md §6).
        //
        // The discriminator is a deliberately SINGULAR basis: `basic_vars` is every
        // column set to the same index, so `factorize_sparse` cannot succeed. Only
        // the legacy branch touches `basic_vars` (the reuse branch reads `dual` and
        // `col_status` alone), so "still returns a fixing" is proof that no
        // factorization was attempted. This is checked on the return value rather
        // than a `profile` counter on purpose: `ENABLED` and the counter arrays are
        // process-global and `solve_milp` calls `init_from_env()` on entry, so any
        // driver test on a sibling thread disarms a counter-based probe mid-flight
        // and it reads 0 -- the exact §6 failure the counters were added to catch.
        let (sp, m, ns, c, b, l, u) = rc_fix_fixture();
        let n = l.len();
        let opts = SimplexOptions::default();
        let sol = crate::lp::simplex::solve_lp_cols(sp.clone(), m, n, &c, &l, &u, &b, &opts);
        assert_eq!(sol.status, LpStatus::Optimal);
        assert_eq!(
            sol.dual.len(),
            m,
            "Optimal must export row duals (#1066 rests on it)"
        );
        let is_int = vec![true, true, true, false, false];

        let mut singular = sol.basis.clone();
        for bv in singular.basic_vars.iter_mut() {
            *bv = 0;
        }
        // Sweep the same incumbent ladder as the sibling test: with the basis
        // sabotaged, the legacy path can never return a fixing, while the reuse path
        // must still return the ones the intact basis produced -- at least one of
        // them, or the probe would pass vacuously on a ladder where nothing fixes.
        let mut reuse_fixed = 0;
        let mut refac_attempts = 0;
        for k in 0..12 {
            let incumbent = sol.obj + 0.25 * k as f64;
            assert!(
                reduced_cost_fix(
                    &sp,
                    m,
                    &c,
                    &singular,
                    &[],
                    sol.obj,
                    incumbent,
                    &l,
                    &u,
                    ns,
                    &is_int,
                    opts.tol,
                )
                .is_none(),
                "k={k}: a singular basis must defeat the refactor path -- otherwise \
                 this test cannot tell the two paths apart"
            );
            refac_attempts += 1;
            if reduced_cost_fix(
                &sp, m, &c, &singular, &sol.dual, sol.obj, incumbent, &l, &u, ns, &is_int, opts.tol,
            )
            .is_some()
            {
                reuse_fixed += 1;
            }
        }
        assert_eq!(
            refac_attempts, 12,
            "probe must have exercised every incumbent"
        );
        assert!(
            reuse_fixed >= 1,
            "with the solve's own duals in hand, reduced-cost fixing must produce a \
             fixing without ever factorizing the basis (0/12 did)"
        );
    }

    // ---- #1060: the no-incumbent continuous-repair dive schedule ----

    #[test]
    fn dive_stride_zero_is_the_legacy_root_only_dive() {
        // The shipped default. Every batch must be ineligible, which is what makes
        // an unset `DISCOPT_MILP_DIVE_STRIDE` bit-identical to the driver before
        // the schedule existed -- the claim the golden/determinism panels below
        // silently rest on.
        let mut checks = 0;
        for b in 0..64 {
            assert!(
                !dive_batch_eligible(0, b, false, 0),
                "batch {b} eligible at stride 0"
            );
            checks += 1;
        }
        assert_eq!(DIVE_STRIDE_DEFAULT, 0);
        assert_eq!(checks, 64);
    }

    #[test]
    fn dive_schedule_stops_the_moment_an_incumbent_exists() {
        // The whole justification for diving off-root is that there is nothing to
        // prune against. With an incumbent in hand the ordinary search is the
        // better use of the node budget, so the schedule must switch itself off.
        assert!(dive_batch_eligible(8, 0, false, 0));
        assert!(!dive_batch_eligible(8, 0, true, 0));
        assert!(dive_batch_eligible(8, 16, false, 3));
        assert!(!dive_batch_eligible(8, 16, true, 3));
    }

    #[test]
    fn dive_schedule_honors_stride_and_cap() {
        let stride = 8;
        let eligible: Vec<usize> = (0..40)
            .filter(|&b| dive_batch_eligible(stride, b, false, 0))
            .collect();
        assert_eq!(eligible, vec![0, 8, 16, 24, 32]);
        // A model the dive can never repair must not pay for it forever.
        assert!(!dive_batch_eligible(
            stride,
            0,
            false,
            DIVE_NO_INCUMBENT_CAP
        ));
        assert!(!dive_batch_eligible(
            stride,
            800,
            false,
            DIVE_NO_INCUMBENT_CAP + 5
        ));
    }

    #[test]
    fn dive_stride_parse_refuses_garbage_instead_of_defaulting() {
        // An A/B arm whose value silently reads as the default makes the harness
        // measure one arm twice (the DISCOPT_LP_REFAC_INTERVAL precedent).
        assert_eq!(parse_dive_stride("").unwrap(), DIVE_STRIDE_DEFAULT);
        assert_eq!(parse_dive_stride("  ").unwrap(), DIVE_STRIDE_DEFAULT);
        assert_eq!(parse_dive_stride("0").unwrap(), 0);
        assert_eq!(parse_dive_stride("8").unwrap(), 8);
        assert_eq!(parse_dive_stride(" 12 ").unwrap(), 12);
        for bad in ["-1", "eight", "1.5", "8x"] {
            assert!(parse_dive_stride(bad).is_err(), "{bad:?} must be refused");
        }
    }

    fn opts(ns: usize, int_cols: Vec<usize>) -> MilpOptions {
        MilpOptions {
            n_struct: ns,
            integer_cols: int_cols,
            max_nodes: 100_000,
            time_limit_s: None,
            gap_tol: 1e-9,
            root_cuts: 16,
            cut_rounds: 3,
            gmi_cuts: true,
            cut_select: true,
            node_cuts: true,
            max_pool_cuts: 500,
            heuristics: true,
            presolve: true,
            strong_branch: true,
            node_propagation: true,
            reduced_cost_fixing: true,
            sb_max_cands: 8,
            sb_node_budget: 1024,
            initial_incumbent: None,
            node_hook_rounds: 0,
            node_hook_cut_cap: 0,
            root_cut_time_s: None,
            root_cut_prune: true,
            simplex: SimplexOptions::default(),
        }
    }

    #[test]
    fn binary_knapsack_optimum() {
        // max 10x0+9x1+8x2+x3 s.t. 5Σx ≤ 9, x binary. Slack s. min -obj.
        // Optimum: a single item fits (5+5=10>9), best is x0 → -10.
        let a = [5.0, 5.0, 5.0, 5.0, 1.0];
        let c = [-10.0, -9.0, -8.0, -1.0, 0.0];
        let l = [0.0; 5];
        let u = [1.0, 1.0, 1.0, 1.0, INF];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 5,
            c: &c,
            l: &l,
            u: &u,
        };
        let r = solve_milp(&lp, &[9.0], 0.0, &opts(4, vec![0, 1, 2, 3]));
        assert_eq!(r.status, MilpStatus::Optimal);
        assert!((r.obj - (-10.0)).abs() < 1e-6, "obj {}", r.obj);
    }

    /// Same knapsack, driven to a hard truncation (`max_nodes = 0`) so the only
    /// possible incumbent is the seeded one: a valid seed must be adopted (and
    /// reported), an infeasible / fractional / wrong-length one silently
    /// rejected (never trusted — an infeasible incumbent would be a false
    /// certificate). Root machinery (cuts/heuristics/presolve) is disabled so
    /// no other incumbent source can mask the seeding behavior.
    #[test]
    fn seeded_incumbent_validated_then_adopted() {
        let a = [5.0, 5.0, 5.0, 5.0, 1.0];
        let c = [-10.0, -9.0, -8.0, -1.0, 0.0];
        let l = [0.0; 5];
        let u = [1.0, 1.0, 1.0, 1.0, INF];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 5,
            c: &c,
            l: &l,
            u: &u,
        };
        let bare = |seed: Option<Vec<f64>>| {
            let mut o = opts(4, vec![0, 1, 2, 3]);
            o.max_nodes = 0;
            o.root_cuts = 0;
            o.cut_rounds = 0;
            o.gmi_cuts = false;
            o.node_cuts = false;
            o.heuristics = false;
            o.presolve = false;
            o.strong_branch = false;
            o.initial_incumbent = seed;
            solve_milp(&lp, &[9.0], 0.0, &o)
        };
        // Feasible seed (x1 = 1, others 0, obj -9): adopted and reported.
        let r = bare(Some(vec![0.0, 1.0, 0.0, 0.0]));
        assert_ne!(r.status, MilpStatus::Optimal, "truncated search");
        assert!((r.obj - (-9.0)).abs() < 1e-9, "seed not adopted: {}", r.obj);
        assert!((r.x[1] - 1.0).abs() < 1e-9);
        // Row-infeasible seed (two items, 10 > 9): rejected, no incumbent.
        let r = bare(Some(vec![1.0, 1.0, 0.0, 0.0]));
        assert!(!r.obj.is_finite(), "infeasible seed adopted: {}", r.obj);
        // Fractional integer column: rejected.
        let r = bare(Some(vec![0.5, 0.0, 0.0, 0.0]));
        assert!(!r.obj.is_finite(), "fractional seed adopted: {}", r.obj);
        // Wrong length: rejected.
        let r = bare(Some(vec![1.0, 0.0]));
        assert!(!r.obj.is_finite(), "wrong-length seed adopted: {}", r.obj);
        // Out-of-bounds value: rejected.
        let r = bare(Some(vec![2.0, 0.0, 0.0, 0.0]));
        assert!(!r.obj.is_finite(), "out-of-bounds seed adopted: {}", r.obj);
        // A seeded search left to run must still certify the true optimum
        // (seeding is monotone: it can only help pruning, never change math).
        let mut o = opts(4, vec![0, 1, 2, 3]);
        o.initial_incumbent = Some(vec![0.0, 1.0, 0.0, 0.0]);
        let r = solve_milp(&lp, &[9.0], 0.0, &o);
        assert_eq!(r.status, MilpStatus::Optimal);
        assert!((r.obj - (-10.0)).abs() < 1e-6, "obj {}", r.obj);
    }

    /// A no-op debug hook must be bound-neutral: identical status / obj / nodes
    /// vs. no hook (CLAUDE.md §5), while still firing at least one checkpoint.
    #[test]
    fn debug_hook_is_bound_neutral() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct Counter(AtomicUsize);
        impl MilpDebugHook for Counter {
            fn checkpoint(&self, _s: &MilpDebugState<'_>) -> MilpDebugControl {
                self.0.fetch_add(1, Ordering::Relaxed);
                MilpDebugControl::Continue
            }
        }

        // A knapsack big enough to branch (multiple nodes, an incumbent event).
        let a = [5.0, 3.0, 2.0, 4.0, 3.0, 5.0, 1.0];
        let c = [-8.0, -5.0, -3.0, -6.0, -4.0, -7.0, 0.0];
        let l = [0.0; 7];
        let u = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, INF];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 7,
            c: &c,
            l: &l,
            u: &u,
        };
        let base = solve_milp(&lp, &[10.0], 0.0, &opts(6, vec![0, 1, 2, 3, 4, 5]));

        let hook = Counter(AtomicUsize::new(0));
        let hooked = solve_milp_hooked(
            SparseCols::from_dense(&a, 1, 7),
            1,
            7,
            &c,
            &l,
            &u,
            &[10.0],
            0.0,
            &opts(6, vec![0, 1, 2, 3, 4, 5]),
            Some(&hook),
        );

        assert_eq!(base.status, hooked.status);
        assert_eq!(base.nodes, hooked.nodes, "node count drifted with hook");
        assert!(
            (base.obj - hooked.obj).abs() < 1e-12,
            "obj drifted with hook"
        );
        assert!(
            hook.0.load(Ordering::Relaxed) > 0,
            "hook never fired — checkpoints not wired"
        );
    }

    #[test]
    fn general_integer_optimum() {
        // min -x0 - x1 s.t. x0 + x1 + s = 3 (s≥0), 0≤x≤2 integer.
        // Optimum x0=2,x1=1 (or 1,2) → -3.
        let a = [1.0, 1.0, 1.0];
        let c = [-1.0, -1.0, 0.0];
        let l = [0.0, 0.0, 0.0];
        let u = [2.0, 2.0, INF];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 3,
            c: &c,
            l: &l,
            u: &u,
        };
        let r = solve_milp(&lp, &[3.0], 0.0, &opts(2, vec![0, 1]));
        assert_eq!(r.status, MilpStatus::Optimal);
        assert!((r.obj - (-3.0)).abs() < 1e-6, "obj {}", r.obj);
    }

    #[test]
    fn infeasible_milp() {
        // x0 + s = 1 (s≥0), x0 ∈ [2,5] integer → x0≥2 but ≤1 → infeasible.
        let a = [1.0, 1.0];
        let c = [1.0, 0.0];
        let l = [2.0, 0.0];
        let u = [5.0, INF];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 2,
            c: &c,
            l: &l,
            u: &u,
        };
        let r = solve_milp(&lp, &[1.0], 0.0, &opts(1, vec![0]));
        assert_eq!(r.status, MilpStatus::Infeasible);
    }

    // ---- C-2: deadline that orphans a deferred node must NOT report Infeasible ----
    //
    // These drive the terminal-status logic (`decide_status`) directly, which is
    // where the false certificate lived: when the last batch fathoms rigorously
    // and no incumbent exists, a deferred (un-solved) node has been popped off the
    // heap and left `Evaluated`, so the tree reads `is_finished() == true`. The
    // pre-fix code returned `Infeasible` unconditionally on that branch — a false
    // "infeasible" on a time-limit termination whose orphaned subtree may contain
    // the optimum. The fix gates `Infeasible` on `!search_incomplete`.

    #[test]
    fn c2_deferred_node_orphaned_by_deadline_is_not_infeasible() {
        // Empty tree, no incumbent, but a node was deferred un-solved: the search
        // was cut short by the deadline, so the honest status is a limit status,
        // NEVER Infeasible. This is the exact scenario the C-2 card describes.
        let status = decide_status(
            /*unbounded=*/ false, /*has_inc=*/ false,
            /*tree_finished=*/ true, // open_count()==0 because the orphan is Evaluated
            /*search_incomplete=*/ true, // a node was deferred un-solved
            /*tree_unresolved=*/ false, /*gap_closed=*/ false,
            /*gap_certified=*/ false, // gap is correctly decertified on defer
            /*node_limit_hit=*/ false,
        );
        assert_ne!(
            status,
            MilpStatus::Infeasible,
            "deferred-node orphaning must not yield a false Infeasible certificate"
        );
        assert_eq!(status, MilpStatus::NodeLimit);
    }

    #[test]
    fn c2_genuine_infeasible_still_reported_when_search_complete() {
        // Rigorous empty-tree proof: every node fathomed, nothing deferred. The
        // Infeasible certificate must survive the fix — do not weaken it.
        let status = decide_status(
            /*unbounded=*/ false, /*has_inc=*/ false, /*tree_finished=*/ true,
            /*search_incomplete=*/ false, // no node was ever dropped un-solved
            /*tree_unresolved=*/ false, // every removal was proof-backed
            /*gap_closed=*/ false, /*gap_certified=*/ true,
            /*node_limit_hit=*/ false,
        );
        assert_eq!(
            status,
            MilpStatus::Infeasible,
            "a rigorously drained empty tree must still certify Infeasible"
        );
    }

    #[test]
    fn c2_end_to_end_genuine_infeasible_unaffected() {
        // The full-driver infeasible path (no deferral) is unchanged: x0 ∈ [2,5]
        // integer with x0 ≤ 1 is genuinely infeasible and no deadline is set, so
        // `search_incomplete` stays false and the status is Infeasible.
        let a = [1.0, 1.0];
        let c = [1.0, 0.0];
        let l = [2.0, 0.0];
        let u = [5.0, INF];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 2,
            c: &c,
            l: &l,
            u: &u,
        };
        let r = solve_milp(&lp, &[1.0], 0.0, &opts(1, vec![0]));
        assert_eq!(r.status, MilpStatus::Infeasible);
    }

    #[test]
    fn c2_deferred_with_incumbent_reports_feasible_not_infeasible() {
        // Orthogonal guard: a deferred node with an incumbent present is a
        // time-limited feasible solve, not Infeasible (that branch never touched
        // `has_inc==true`, but pin it so a future refactor can't regress it).
        let status = decide_status(
            /*unbounded=*/ false, /*has_inc=*/ true, /*tree_finished=*/ false,
            /*search_incomplete=*/ true, /*tree_unresolved=*/ false,
            /*gap_closed=*/ false, /*gap_certified=*/ false,
            /*node_limit_hit=*/ false,
        );
        assert_eq!(status, MilpStatus::Feasible);
    }

    // ---- #598 (B1-FIX): certification semantics of decide_status ----

    #[test]
    fn b1_gap_closed_certifies_optimal_without_tree_finished() {
        // The certification criterion is the CLOSED FRONTIER GAP (which folds in
        // every valid inherited bound and the unresolved floor), not tree
        // exhaustion: a search that closed the gap mid-tree — even one whose
        // node LPs failed along the way, since those nodes stay soundly
        // accounted (parent-bound floor + branch / unresolved_floor) — is a
        // rigorous optimum.
        let status = decide_status(
            /*unbounded=*/ false, /*has_inc=*/ true, /*tree_finished=*/ false,
            /*search_incomplete=*/ false, /*tree_unresolved=*/ false,
            /*gap_closed=*/ true, /*gap_certified=*/ true,
            /*node_limit_hit=*/ false,
        );
        assert_eq!(status, MilpStatus::Optimal);
    }

    #[test]
    fn b1_finished_tree_with_open_gap_is_feasible_not_optimal() {
        // A drained tree whose gap did NOT close (an unresolved-floor fathom
        // kept the honest bound below the incumbent) must exit Feasible. The
        // pre-fix `(tree_finished || gap_closed)` disjunct would have stamped
        // this Optimal — a false certificate.
        let status = decide_status(
            /*unbounded=*/ false, /*has_inc=*/ true, /*tree_finished=*/ true,
            /*search_incomplete=*/ false, /*tree_unresolved=*/ true,
            /*gap_closed=*/ false, /*gap_certified=*/ true,
            /*node_limit_hit=*/ false,
        );
        assert_eq!(status, MilpStatus::Feasible);
    }

    #[test]
    fn b1_unresolved_fathom_blocks_false_infeasible() {
        // No incumbent and the tree drained, but a subtree was removed without
        // proof (failed relaxation, nothing left to branch): the empty tree is
        // NOT an emptiness proof — a limit status, never Infeasible.
        let status = decide_status(
            /*unbounded=*/ false, /*has_inc=*/ false, /*tree_finished=*/ true,
            /*search_incomplete=*/ false, /*tree_unresolved=*/ true,
            /*gap_closed=*/ false, /*gap_certified=*/ true,
            /*node_limit_hit=*/ false,
        );
        assert_ne!(status, MilpStatus::Infeasible);
        assert_eq!(status, MilpStatus::NodeLimit);
    }

    #[test]
    fn root_cuts_reduce_nodes() {
        // Symmetric knapsack 5Σx + s = 9, x binary, min -16Σx. Optimum -16
        // (one item). The fractional root [0.45]^4 yields a GMI cut Σx ≤ 1.
        let a = [5.0, 5.0, 5.0, 5.0, 1.0];
        let c = [-16.0, -16.0, -16.0, -16.0, 0.0];
        let l = [0.0; 5];
        let u = [1.0, 1.0, 1.0, 1.0, INF];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 5,
            c: &c,
            l: &l,
            u: &u,
        };
        let mut o_cut = opts(4, vec![0, 1, 2, 3]);
        o_cut.root_cuts = 16;
        let mut o_no = opts(4, vec![0, 1, 2, 3]);
        o_no.root_cuts = 0;
        let r_cut = solve_milp(&lp, &[9.0], 0.0, &o_cut);
        let r_no = solve_milp(&lp, &[9.0], 0.0, &o_no);
        assert_eq!(r_cut.status, MilpStatus::Optimal);
        assert_eq!(r_no.status, MilpStatus::Optimal);
        assert!((r_cut.obj - (-16.0)).abs() < 1e-6 && (r_no.obj - (-16.0)).abs() < 1e-6);
        assert!(
            r_cut.nodes <= r_no.nodes,
            "cuts {} vs no-cuts {}",
            r_cut.nodes,
            r_no.nodes
        );
    }

    #[test]
    fn root_cut_rounds_reoptimize_warm_instead_of_cold_solving() {
        // The root cut loop kept each round's optimal basis for the post-loop root
        // node but never fed it back into the NEXT round, so every round re-derived
        // the augmented LP with a cold primal phase-1. Measured on the rsyn0840m OA
        // master at `root_cuts=500, cut_rounds=15`: 14 cold root solves = 23.1 s of
        // the 24.2 s cut loop, 16127 phase-1 pivots, and the enclosing solve could
        // never close the gap. Warm-starting the rounds makes real cut budgets
        // affordable (same instance: 22559 nodes to a certified optimum).
        //
        // The invariant: however many rounds run, only the FIRST derives its basis
        // from scratch; every later round re-optimizes from the previous round's
        // optimum, which `RootCutWarmReopt` records.
        //
        // `RootCutRounds` is asserted `>= 3` so the check cannot pass vacuously on a
        // fixture that happens to finish in one round (CLAUDE.md §6) -- with a single
        // round `EntryCols == 1` holds with or without the fix.
        let _g = crate::profile::test_guard();
        // 3 knapsack rows over 8 binaries with coefficients chosen so the LP
        // relaxation stays fractional across several GMI rounds.
        // 8 binaries + one explicit surplus slack per row (`Σ a·x + s = b`, the
        // standard form this driver's `LpView` takes).
        let n = 11;
        let a = vec![
            7.0, 5.0, 4.0, 3.0, 9.0, 6.0, 5.0, 4.0, 1.0, 0.0, 0.0, //
            3.0, 8.0, 6.0, 7.0, 4.0, 5.0, 9.0, 3.0, 0.0, 1.0, 0.0, //
            6.0, 4.0, 9.0, 5.0, 3.0, 8.0, 4.0, 7.0, 0.0, 0.0, 1.0,
        ];
        let c = vec![
            -9.0, -7.0, -8.0, -6.0, -10.0, -8.0, -7.0, -6.0, 0.0, 0.0, 0.0,
        ];
        let l = vec![0.0; n];
        let mut u = vec![1.0; n];
        u[8] = INF;
        u[9] = INF;
        u[10] = INF;
        let lp = LpView {
            a: &a,
            m: 3,
            n,
            c: &c,
            l: &l,
            u: &u,
        };
        let b = [15.0, 17.0, 16.0];
        let mut o = opts(8, (0..8).collect());
        o.root_cuts = 200;
        o.cut_rounds = 12;
        o.cut_select = true;
        o.gmi_cuts = true;

        // Reference optimum from the same driver with cuts switched off entirely:
        // the cut loop may never move the certified objective.
        let mut o_ref = opts(8, (0..8).collect());
        o_ref.root_cuts = 0;
        let r_ref = solve_milp(&lp, &b, 0.0, &o_ref);
        assert_eq!(r_ref.status, MilpStatus::Optimal);

        // `set_enabled(true)` raises the test override, which keeps both halves of
        // the driver's `profile::begin_solve()` -- the env re-arm and the per-solve
        // counter reset -- from touching this measurement, on this thread or a
        // sibling's. Setting `DISCOPT_PROFILE` process-wide from a threaded test
        // (the old way here) armed every concurrent solve as well.
        crate::profile::set_enabled(true);
        crate::profile::reset();
        let r = solve_milp(&lp, &b, 0.0, &o);
        let rounds = crate::profile::counter(crate::profile::Ctr::RootCutRounds);
        let warm_reopt = crate::profile::counter(crate::profile::Ctr::RootCutWarmReopt);
        crate::profile::set_enabled(false);
        assert_eq!(r.status, MilpStatus::Optimal);
        assert!(
            (r.obj - r_ref.obj).abs() < 1e-6,
            "cut loop moved the optimum: {} vs {}",
            r.obj,
            r_ref.obj
        );
        assert!(
            rounds >= 3,
            "fixture ran only {rounds} cut round(s); the cold-solve check would be vacuous"
        );
        assert_eq!(
            warm_reopt,
            rounds - 1,
            "{rounds} cut rounds but only {warm_reopt} warm re-optimizes; every round \
             after the first must start from the previous round's optimal basis"
        );
    }

    #[test]
    fn presolve_matches_no_presolve() {
        // Equality-constrained MILP where FBBT actually fires:
        //   min -x0 - 2x1 - x2  s.t.  x0 + x1 + x2 = 3,  2x1 + x2 + s = 4,
        //   x∈[0,3] integer, s≥0. Presolve must not change the optimum.
        let a = [1.0, 1.0, 1.0, 0.0, 0.0, 2.0, 1.0, 1.0];
        let c = [-1.0, -2.0, -1.0, 0.0];
        let l = [0.0, 0.0, 0.0, 0.0];
        let u = [3.0, 3.0, 3.0, INF];
        let lp = LpView {
            a: &a,
            m: 2,
            n: 4,
            c: &c,
            l: &l,
            u: &u,
        };
        let mut on = opts(3, vec![0, 1, 2]);
        on.presolve = true;
        let mut off = opts(3, vec![0, 1, 2]);
        off.presolve = false;
        let r_on = solve_milp(&lp, &[3.0, 4.0], 0.0, &on);
        let r_off = solve_milp(&lp, &[3.0, 4.0], 0.0, &off);
        assert_eq!(r_on.status, MilpStatus::Optimal);
        assert_eq!(r_off.status, MilpStatus::Optimal);
        assert!(
            (r_on.obj - r_off.obj).abs() < 1e-6,
            "presolve {} vs no-presolve {}",
            r_on.obj,
            r_off.obj
        );
        // Tightening should not increase node count.
        assert!(
            r_on.nodes <= r_off.nodes,
            "{} vs {}",
            r_on.nodes,
            r_off.nodes
        );
    }

    #[test]
    fn obj_const_applied() {
        // Same knapsack but with obj_const 100 → optimum -10 + 100 = 90.
        let a = [5.0, 5.0, 5.0, 5.0, 1.0];
        let c = [-10.0, -9.0, -8.0, -1.0, 0.0];
        let l = [0.0; 5];
        let u = [1.0, 1.0, 1.0, 1.0, INF];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 5,
            c: &c,
            l: &l,
            u: &u,
        };
        let r = solve_milp(&lp, &[9.0], 100.0, &opts(4, vec![0, 1, 2, 3]));
        assert_eq!(r.status, MilpStatus::Optimal);
        assert!((r.obj - 90.0).abs() < 1e-6, "obj {}", r.obj);
    }

    // --- C-14: Farkas verification before an infeasible fathom ---------------

    /// A genuinely infeasible LP's exported dual ray verifies, so the node is
    /// (soundly) fathomable. `x0 + s = 1`, `s∈[0,∞)`, `x0∈[2,∞)` ⇒ `x0≥2` yet
    /// `x0≤1` — empty.
    #[test]
    fn c14_valid_farkas_ray_certifies_emptiness() {
        let a = [1.0, 1.0];
        let b = [1.0];
        let c = [0.0, 0.0];
        let l = [2.0, 0.0];
        let u = [INF, INF];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 2,
            c: &c,
            l: &l,
            u: &u,
        };
        let r = solve_lp(&lp, &b, &SimplexOptions::default());
        assert_eq!(r.status, LpStatus::Infeasible);
        assert!(
            verify_farkas_infeasible(&r.dual, &a, &b, &l, &u, 1, 2),
            "a real infeasible LP's ray must verify (else we would refuse a valid fathom)"
        );
    }

    /// The C-14 defect class: a node the simplex *labels* Infeasible but whose
    /// exported ray does NOT certify emptiness (here a corrupted/zeroed ray) must
    /// be refused — `verify_farkas_infeasible` returns false so the caller keeps
    /// the node instead of fathoming a region that may hold the optimum.
    #[test]
    fn c14_non_certifying_ray_is_refused() {
        let a = [1.0, 1.0];
        let b = [1.0];
        let l = [2.0, 0.0];
        let u = [INF, INF];
        // Zero ray: g0 ≡ 0, no certificate ⇒ must not fathom.
        assert!(
            !verify_farkas_infeasible(&[0.0], &a, &b, &l, &u, 1, 2),
            "a zero ray certifies nothing; the node must not be fathomed"
        );
        // Empty certificate (nothing exported) ⇒ must not fathom.
        assert!(
            !verify_farkas_infeasible(&[], &a, &b, &l, &u, 1, 2),
            "an absent ray must never license a fathom"
        );
        // A ray that only *grazes* zero (g0 = 0 exactly) must not clear the
        // magnitude-scaled margin: box {x0∈[0,1], s=... } here is actually
        // feasible, and no free-sign y makes g0 strictly positive.
        let a2 = [1.0, 1.0];
        let b2 = [1.0];
        let l2 = [0.0, 0.0];
        let u2 = [1.0, INF];
        assert!(
            !verify_farkas_infeasible(&[1.0], &a2, &b2, &l2, &u2, 1, 2),
            "a feasible box must not be certified empty by any ray"
        );
    }

    /// #1066 regression. The driver verifies the node's Farkas ray against the
    /// *scaled* batch CSC (`ctx.csc`, `ctx.sb`, scaled node bounds), so `sol.dual`
    /// must stay in solve-space coordinates all the way to that arm. An earlier cut
    /// of the reduced-cost-fixing change unscaled `sol.dual` in place right after
    /// the node solve; the ray then no longer matched the matrix it was checked
    /// against, every verification failed, and provably empty nodes stopped being
    /// fathomed — sound (the bound only ever got weaker) but ruinous, turning a
    /// one-node refutation into a full branch-out. Pinned here on node count.
    ///
    /// The model is LP-infeasible but not row-wise infeasible (`x0 + x1 >= 1.5` and
    /// `x0 + x1 <= 0.5` with `x0, x1` binary), with presolve and node propagation
    /// off so nothing but the LP's own certificate can refute it, and a 1e8 value
    /// range so `Scaling::from_sparse` actually returns `Some` — asserted in the
    /// fixture, because without it the scaled path is never entered and the test
    /// proves nothing.
    #[test]
    fn rc_fix_dual_unscale_does_not_break_the_node_farkas_fathom() {
        // The two refuting rows carry deliberately non-unit magnitudes (2^-10 and
        // 2^10) so equilibration gives them non-unit ROW factors — without that the
        // scaled and unscaled rays coincide on exactly the rows the certificate
        // uses and the coupling under test is invisible. A third, harmless row adds
        // the 1e8 entry that pushes the dynamic range past `SCALE_TRIGGER` (1e6).
        let k0 = 2f64.powi(-10);
        let k1 = 2f64.powi(10);
        let a = [
            k0, k0, 0.0, -k0, 0.0, 0.0, // x0 + x1 - s0 = 1.5  (x0 + x1 >= 1.5)
            k1, k1, 0.0, 0.0, k1, 0.0, // x0 + x1 + s1 = 0.5  (x0 + x1 <= 0.5)
            0.0, 0.0, 1e8, 0.0, 0.0, 1.0, // 1e8·x2 + s2 = 1e8    (feasible; sets the range)
        ];
        let b = [1.5 * k0, 0.5 * k1, 1e8];
        let c = [0.0; 6];
        let l = [0.0; 6];
        let u = [1.0, 1.0, 1.0, INF, INF, INF];
        assert!(
            Scaling::from_sparse(&SparseCols::from_dense(&a, 3, 6), 3, 6).is_some(),
            "fixture must actually trigger equilibration or the scaled path is untested"
        );
        let lp = LpView {
            a: &a,
            m: 3,
            n: 6,
            c: &c,
            l: &l,
            u: &u,
        };
        let o = MilpOptions {
            n_struct: 3,
            integer_cols: vec![0, 1, 2],
            max_nodes: 5_000,
            time_limit_s: None,
            gap_tol: 1e-9,
            root_cuts: 0,
            cut_rounds: 1,
            gmi_cuts: false,
            cut_select: false,
            node_cuts: false,
            max_pool_cuts: 0,
            heuristics: false,
            presolve: false,
            node_propagation: false,
            strong_branch: false,
            reduced_cost_fixing: true,
            sb_max_cands: 0,
            sb_node_budget: 0,
            initial_incumbent: None,
            node_hook_rounds: 0,
            node_hook_cut_cap: 0,
            root_cut_time_s: None,
            root_cut_prune: true,
            simplex: SimplexOptions::default(),
        };
        let r = solve_milp(&lp, &b, 0.0, &o);
        assert_eq!(
            r.status,
            MilpStatus::Infeasible,
            "status (nodes {})",
            r.nodes
        );
        assert_eq!(
            r.nodes, 1,
            "the ROOT ray must refute the model outright; branching at all means the \
             scaled certificate was checked in the wrong coordinates"
        );
    }

    /// Scale-invariance of the certificate: an infeasible LP whose exported ray
    /// verifies still verifies after both A/b and the ray are equilibrated by an
    /// arbitrary positive row factor — the property the fathom relies on to check
    /// in scaled solve-space. (`g0(R·ŷ)` over `Â=RA`, `b̂=Rb` equals `g0(y)`.)
    #[test]
    fn c14_certificate_is_scale_invariant() {
        let a = [1.0, 1.0];
        let b = [1.0];
        let c = [0.0, 0.0];
        let l = [2.0, 0.0];
        let u = [INF, INF];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 2,
            c: &c,
            l: &l,
            u: &u,
        };
        let r = solve_lp(&lp, &b, &SimplexOptions::default());
        assert_eq!(r.status, LpStatus::Infeasible);
        // Equilibrate row 0 by 8 (a power of two, as real equilibration snaps to);
        // the scaled ray is ŷ = y / 8, and g0 is invariant, so it still verifies.
        let s = 8.0;
        let a_s = [a[0] * s, a[1] * s];
        let b_s = [b[0] * s];
        let y_s: Vec<f64> = r.dual.iter().map(|v| v / s).collect();
        assert!(
            verify_farkas_infeasible(&y_s, &a_s, &b_s, &l, &u, 1, 2),
            "the safe-bound certificate must be invariant under row equilibration"
        );
    }

    /// Noise robustness (the AMP-relaxation regression that surfaced during the
    /// fix): a warm-simplex ray carries rounding noise, so an infinite-bounded
    /// column with a *noise-level* reduced cost must NOT push `g0` to `−∞` and
    /// reject an otherwise-valid certificate. Here `x0 + s = 1`, `x0∈[2,∞)`,
    /// `s∈[0,∞)` is infeasible; the clean ray is `y=[-1]` (g0 = -1·1 + min over
    /// x0≥2 of (1)·x0 = -1 + 2 = 1 > 0). Perturb `A` on the *infinite* slack column
    /// so `(Aᵀy)` there is a tiny nonzero — the tolerance must absorb it.
    #[test]
    fn c14_infinite_column_noise_does_not_reject_valid_ray() {
        // Ray y=[-1]; column 1 (slack, u=∞) has a[1]=1 → (Aᵀy)_1 = -1, rc=+1 with
        // l=0 contributes 0; column 0 (x0, l=2) rc from a[0]: use the real solve.
        let a = [1.0, 1.0];
        let b = [1.0];
        let l = [2.0, 0.0];
        let u = [INF, INF];
        // A hand-built clean ray for this system: y = [-1].
        let y = [-1.0];
        assert!(
            verify_farkas_infeasible(&y, &a, &b, &l, &u, 1, 2),
            "clean ray must certify the infeasible box"
        );
        // Now perturb the *infinite-bounded* slack column by noise 1e-10: with the
        // ray-scaled rc tolerance, g0 is unchanged and the ray still certifies.
        let a_noisy = [1.0, 1.0 + 1e-10];
        assert!(
            verify_farkas_infeasible(&y, &a_noisy, &b, &l, &u, 1, 2),
            "noise-level reduced cost on an ∞-bounded column must not reject the ray"
        );
    }
}

/// T0 (docs/dev/sparse-milp-plan.md): differential harness for the sparse-MILP
/// conversion. A fixed panel of small MILPs solved through the reference dense
/// [`solve_milp`]. Today it pins the dense results and their determinism; at T1 the
/// CSC entry point plugs into [`Case::solve_csc`] and [`assert_same`] gates it
/// bit-identical to the dense path (same status / obj / bound / node count) — the
/// invariant that keeps the representation change from perturbing any dual bound.
#[cfg(test)]
mod sparse_milp_diff {
    use super::*;
    use crate::lp::simplex::sparse::SparseCols;

    /// One MILP in the panel. Owns its data so the borrowing [`LpView`] can be
    /// rebuilt per solve.
    struct Case {
        name: &'static str,
        a: Vec<f64>, // row-major, m*n
        m: usize,
        n: usize,
        c: Vec<f64>,
        l: Vec<f64>,
        u: Vec<f64>,
        b: Vec<f64>,
        ns: usize,
        int_cols: Vec<usize>,
    }

    impl Case {
        fn opts(&self) -> MilpOptions {
            MilpOptions {
                n_struct: self.ns,
                integer_cols: self.int_cols.clone(),
                max_nodes: 100_000,
                time_limit_s: None,
                gap_tol: 1e-9,
                root_cuts: 16,
                cut_rounds: 3,
                gmi_cuts: true,
                cut_select: true,
                node_cuts: true,
                max_pool_cuts: 500,
                heuristics: true,
                presolve: true,
                strong_branch: true,
                node_propagation: true,
                reduced_cost_fixing: true,
                sb_max_cands: 8,
                sb_node_budget: 1024,
                initial_incumbent: None,
                node_hook_rounds: 0,
                node_hook_cut_cap: 0,
                root_cut_time_s: None,
                root_cut_prune: true,
                simplex: SimplexOptions::default(),
            }
        }

        fn dense_view(&self) -> LpView<'_> {
            LpView {
                a: &self.a,
                m: self.m,
                n: self.n,
                c: &self.c,
                l: &self.l,
                u: &self.u,
            }
        }

        /// Reference solve through the dense driver.
        fn solve_dense(&self) -> MilpResult {
            solve_milp(&self.dense_view(), &self.b, 0.0, &self.opts())
        }

        /// CSC view of this instance's matrix, fed to the CSC driver entry.
        fn csc(&self) -> SparseCols {
            SparseCols::from_dense(&self.a, self.m, self.n)
        }

        /// Solve through the T1 CSC entry point [`solve_milp_csc`].
        fn solve_csc(&self) -> MilpResult {
            let sp = self.csc();
            solve_milp_csc(
                &sp,
                self.m,
                self.n,
                &self.c,
                &self.l,
                &self.u,
                &self.b,
                0.0,
                &self.opts(),
            )
        }
    }

    /// Panel: pure-LP, small binary knapsack, general integer, infeasible,
    /// unbounded, and a cuts-firing knapsack (branches + fires GMI cuts).
    fn panel() -> Vec<Case> {
        vec![
            // pure LP: min -x0 s.t. x0 + s = 1, x0 in [0,1] continuous -> -1.
            Case {
                name: "pure_lp",
                a: vec![1.0, 1.0],
                m: 1,
                n: 2,
                c: vec![-1.0, 0.0],
                l: vec![0.0, 0.0],
                u: vec![1.0, INF],
                b: vec![1.0],
                ns: 1,
                int_cols: vec![],
            },
            // binary knapsack: max 10x0+9x1+8x2+x3 s.t. 5*sum <= 9, binary -> -10.
            Case {
                name: "binary_knapsack",
                a: vec![5.0, 5.0, 5.0, 5.0, 1.0],
                m: 1,
                n: 5,
                c: vec![-10.0, -9.0, -8.0, -1.0, 0.0],
                l: vec![0.0; 5],
                u: vec![1.0, 1.0, 1.0, 1.0, INF],
                b: vec![9.0],
                ns: 4,
                int_cols: vec![0, 1, 2, 3],
            },
            // general integer: min -x0-x1 s.t. x0+x1+s=3, x in [0,2] int -> -3.
            Case {
                name: "general_integer",
                a: vec![1.0, 1.0, 1.0],
                m: 1,
                n: 3,
                c: vec![-1.0, -1.0, 0.0],
                l: vec![0.0, 0.0, 0.0],
                u: vec![2.0, 2.0, INF],
                b: vec![3.0],
                ns: 2,
                int_cols: vec![0, 1],
            },
            // infeasible: x0 + s = 1, x0 in [2,5] int -> infeasible.
            Case {
                name: "infeasible",
                a: vec![1.0, 1.0],
                m: 1,
                n: 2,
                c: vec![1.0, 0.0],
                l: vec![2.0, 0.0],
                u: vec![5.0, INF],
                b: vec![1.0],
                ns: 1,
                int_cols: vec![0],
            },
            // unbounded: min -x0 s.t. 0*x0 + s = 1, x0 in [0,INF) -> unbounded.
            Case {
                name: "unbounded",
                a: vec![0.0, 1.0],
                m: 1,
                n: 2,
                c: vec![-1.0, 0.0],
                l: vec![0.0, 0.0],
                u: vec![INF, INF],
                b: vec![1.0],
                ns: 1,
                int_cols: vec![0],
            },
            // cuts-firing knapsack: 6 binaries, branches and fires GMI cuts.
            Case {
                name: "cuts_firing_knapsack",
                a: vec![5.0, 3.0, 2.0, 4.0, 3.0, 5.0, 1.0],
                m: 1,
                n: 7,
                c: vec![-8.0, -5.0, -3.0, -6.0, -4.0, -7.0, 0.0],
                l: vec![0.0; 7],
                u: vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, INF],
                b: vec![10.0],
                ns: 6,
                int_cols: vec![0, 1, 2, 3, 4, 5],
            },
        ]
    }

    /// Bit-identical gate used by the dense-vs-dense determinism check now and by
    /// the dense-vs-CSC check at T1. Status and node count must match exactly;
    /// obj/bound match to a tight tolerance (finite cases only).
    fn assert_same(name: &str, a: &MilpResult, b: &MilpResult) {
        assert_eq!(a.status, b.status, "{name}: status drift");
        assert_eq!(a.nodes, b.nodes, "{name}: node-count drift");
        if a.obj.is_finite() && b.obj.is_finite() {
            assert!(
                (a.obj - b.obj).abs() < 1e-9,
                "{name}: obj drift {} {}",
                a.obj,
                b.obj
            );
        }
        if a.bound.is_finite() && b.bound.is_finite() {
            assert!(
                (a.bound - b.bound).abs() < 1e-9,
                "{name}: bound drift {} {}",
                a.bound,
                b.bound
            );
        }
    }

    #[test]
    fn dense_panel_reference_values() {
        for case in panel() {
            let r = case.solve_dense();
            match case.name {
                "pure_lp" => {
                    assert_eq!(r.status, MilpStatus::Optimal, "pure_lp");
                    assert!((r.obj - (-1.0)).abs() < 1e-6, "pure_lp obj {}", r.obj);
                }
                "binary_knapsack" => {
                    assert_eq!(r.status, MilpStatus::Optimal, "binary_knapsack");
                    assert!((r.obj - (-10.0)).abs() < 1e-6, "knapsack obj {}", r.obj);
                }
                "general_integer" => {
                    assert_eq!(r.status, MilpStatus::Optimal, "general_integer");
                    assert!((r.obj - (-3.0)).abs() < 1e-6, "genint obj {}", r.obj);
                }
                "infeasible" => assert_eq!(r.status, MilpStatus::Infeasible, "infeasible"),
                "unbounded" => assert_eq!(r.status, MilpStatus::Unbounded, "unbounded"),
                "cuts_firing_knapsack" => {
                    assert_eq!(r.status, MilpStatus::Optimal, "cuts_firing");
                    assert!(
                        r.obj.is_finite() && r.obj < 0.0,
                        "cuts_firing obj {}",
                        r.obj
                    );
                }
                other => panic!("unhandled case {other}"),
            }
        }
    }

    /// Golden lock on the CURRENT driver's per-instance solve — status, objective,
    /// bound, node count, and simplex pivot count. This is the **driver-wide**
    /// bit-identity gate for the sparse conversion: T2/T3 change the driver internals
    /// for BOTH the dense and CSC entry points, so `csc_entry_matches_dense_on_panel`
    /// (dense-vs-CSC) alone can no longer catch a regression against the *original*
    /// behavior — after conversion both sides move together. `lp_iters` is the
    /// sensitive discriminator: a different root-solve pivot path drifts it even when
    /// the B&B tree is a single node. A change to any value here is a red flag — the
    /// sparse path is a pure representation change and must reproduce these exactly.
    ///
    /// `lp_iters` was re-baselined once, for two compounding reasons, both measured
    /// on this panel with status/nodes/obj/bound bit-identical throughout:
    ///   1. `primal.rs` used to `assemble` every cold solve with a hardcoded
    ///      `iters: 0`, so the primal contributed nothing to this sum (CLAUDE.md §6).
    ///      With the honest pivot count: binary_knapsack 1 → 5, cuts_firing 1 → 8.
    ///   2. The root cut loop now warm dual re-optimizes each round from the previous
    ///      round's optimal basis instead of cold-solving the augmented LP, which is
    ///      the whole point of the change: binary_knapsack 5 → 2, cuts_firing 8 → 2.
    /// The recorded values are therefore the *real* pivot counts of the *current*
    /// engine; the old ones were an under-report of a slower path.
    #[test]
    fn driver_matches_golden() {
        for case in panel() {
            let r = case.solve_dense();
            let (status, obj, bound, nodes, iters): (MilpStatus, f64, f64, usize, usize) =
                match case.name {
                    "pure_lp" => (MilpStatus::Optimal, -1.0, -1.0, 1, 0),
                    "binary_knapsack" => (MilpStatus::Optimal, -10.0, -10.0, 1, 2),
                    "general_integer" => (MilpStatus::Optimal, -3.0, -3.0, 1, 1),
                    "infeasible" => (MilpStatus::Infeasible, f64::INFINITY, f64::INFINITY, 0, 0),
                    "unbounded" => (
                        MilpStatus::Unbounded,
                        f64::INFINITY,
                        f64::NEG_INFINITY,
                        1,
                        0,
                    ),
                    "cuts_firing_knapsack" => (MilpStatus::Optimal, -16.0, -16.0, 1, 2),
                    other => panic!("unhandled case {other}"),
                };
            assert_eq!(r.status, status, "{}: status", case.name);
            assert_eq!(r.nodes, nodes, "{}: nodes", case.name);
            assert_eq!(r.lp_iters, iters, "{}: lp_iters", case.name);
            if obj.is_finite() {
                assert!(
                    (r.obj - obj).abs() < 1e-9,
                    "{}: obj {} != {obj}",
                    case.name,
                    r.obj
                );
            } else {
                assert_eq!(r.obj, obj, "{}: obj", case.name);
            }
            if bound.is_finite() {
                assert!(
                    (r.bound - bound).abs() < 1e-9,
                    "{}: bound {} != {bound}",
                    case.name,
                    r.bound
                );
            } else {
                assert_eq!(r.bound, bound, "{}: bound", case.name);
            }
        }
    }

    /// T3b6 branching integration golden. The panel solves at the root (nodes=1) so
    /// it never runs the per-node engine. This instance FORCES a 13-node tree
    /// (heuristics off, no root GMI) while keeping node cover separation, strong
    /// branching, and FBBT propagation ON — so `separate_cover`, `strong_branch`,
    /// and `tighten_bounds` (and their CSC ports wired in at T3b5) are exercised
    /// end-to-end. The rewire must leave status/obj/node-count/`lp_iters` EXACTLY
    /// unchanged; any drift is a per-node-engine regression the panel would miss.
    #[test]
    fn branching_golden() {
        let a = [
            3.0, 4.0, 5.0, 2.0, 6.0, 1.0, 1.0, 0.0, //
            2.0, 3.0, 1.0, 5.0, 2.0, 4.0, 0.0, 1.0,
        ];
        let c = [-5.0, -6.0, -7.0, -4.0, -8.0, -3.0, 0.0, 0.0];
        let l = [0.0; 8];
        let u = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, INF, INF];
        let lp = LpView {
            a: &a,
            m: 2,
            n: 8,
            c: &c,
            l: &l,
            u: &u,
        };
        let o = MilpOptions {
            n_struct: 6,
            integer_cols: vec![0, 1, 2, 3, 4, 5],
            max_nodes: 100_000,
            time_limit_s: None,
            gap_tol: 1e-9,
            root_cuts: 0,
            cut_rounds: 3,
            gmi_cuts: false,
            cut_select: true,
            node_cuts: true,
            max_pool_cuts: 500,
            heuristics: false,
            presolve: true,
            strong_branch: true,
            node_propagation: true,
            reduced_cost_fixing: true,
            sb_max_cands: 8,
            sb_node_budget: 1024,
            initial_incumbent: None,
            node_hook_rounds: 0,
            node_hook_cut_cap: 0,
            root_cut_time_s: None,
            root_cut_prune: true,
            simplex: SimplexOptions::default(),
        };
        let r = solve_milp(&lp, &[10.0, 9.0], 0.0, &o);
        assert_eq!(r.status, MilpStatus::Optimal, "status");
        assert!((r.obj - (-16.0)).abs() < 1e-9, "obj {}", r.obj);
        assert_eq!(r.nodes, 13, "node-count drift (per-node engine)");
        assert_eq!(r.lp_iters, 39, "lp_iters drift (per-node engine)");
    }

    /// Determinism: re-solving is bit-identical. This is exactly the property the
    /// CSC path must satisfy against the dense path at T1, so the harness proves the
    /// gate is meaningful (the dense driver itself is reproducible).
    #[test]
    fn dense_panel_is_deterministic() {
        for case in panel() {
            let r1 = case.solve_dense();
            let r2 = case.solve_dense();
            assert_same(case.name, &r1, &r2);
            assert_eq!(r1.lp_iters, r2.lp_iters, "{}: lp_iters drift", case.name);
        }
    }

    /// T1 gate: the CSC entry point [`solve_milp_csc`] is bit-identical to the
    /// dense [`solve_milp`] on every panel case — same status, node count, objective,
    /// bound, and incumbent length. Any drift means the CSC path perturbed the
    /// solve, which would corrupt a dual bound (the whole point of the gate).
    #[test]
    fn csc_entry_matches_dense_on_panel() {
        for case in panel() {
            let dense = case.solve_dense();
            let csc = case.solve_csc();
            assert_same(case.name, &dense, &csc);
            assert_eq!(
                dense.x.len(),
                csc.x.len(),
                "{}: incumbent length",
                case.name
            );
            if dense.status == MilpStatus::Optimal {
                for (k, (xd, xc)) in dense.x.iter().zip(csc.x.iter()).enumerate() {
                    assert!(
                        (xd - xc).abs() < 1e-9,
                        "{}: incumbent[{k}] drift {xd} {xc}",
                        case.name
                    );
                }
            }
        }
    }

    /// T2 direct gate: [`solve_lp_root_csc`] reproduces the dense [`solve_lp_root`]
    /// pivot-for-pivot (status, objective, and **iteration count**) on both a
    /// well-conditioned LP and an ILL-conditioned one whose 1e8 dynamic range trips
    /// the `ScaledLp`/`Scaling::from_sparse` equilibration — the exact path whose
    /// dense-vs-CSC equivalence option A rests on. `iters` drift here would mean the
    /// CSC root solve takes a different pivot path and is NOT bit-identical.
    #[test]
    fn solve_lp_root_csc_matches_dense() {
        // (name, a row-major m*n, m, n, c, l, u, b)
        let cases: Vec<(
            &str,
            Vec<f64>,
            usize,
            usize,
            Vec<f64>,
            Vec<f64>,
            Vec<f64>,
            Vec<f64>,
        )> = vec![
            (
                "well_conditioned",
                vec![1.0, 1.0],
                1,
                2,
                vec![-1.0, 0.0],
                vec![0.0, 0.0],
                vec![1.0, INF],
                vec![1.0],
            ),
            (
                // 1e8*x0 + 1.0*s = 1e8, x0 in [0,1] -> x0=1, obj -1. Range 1e8 > 1e6
                // SCALE_TRIGGER, so both paths equilibrate.
                "ill_conditioned",
                vec![1e8, 1.0],
                1,
                2,
                vec![-1.0, 0.0],
                vec![0.0, 0.0],
                vec![1.0, INF],
                vec![1e8],
            ),
        ];
        let opts = SimplexOptions::default();
        for (name, a, m, n, c, l, u, b) in cases {
            let lp = LpView {
                a: &a,
                m,
                n,
                c: &c,
                l: &l,
                u: &u,
            };
            let dense = solve_lp_root(&lp, &b, &opts);
            let sp = SparseCols::from_dense(&a, m, n);
            let csc = solve_lp_root_csc(&sp, m, n, &c, &l, &u, &b, &opts);
            assert_eq!(dense.status, csc.status, "{name}: status");
            assert_eq!(dense.iters, csc.iters, "{name}: iters (pivot-path) drift");
            assert!(
                (dense.obj - csc.obj).abs() < 1e-6 * (1.0 + dense.obj.abs()),
                "{name}: obj {} vs {}",
                dense.obj,
                csc.obj
            );
        }
    }

    /// T3 gate: the CSC cut augmentation reproduces the dense `augment_with_cuts`
    /// matrix exactly — `augment_cols_with_cuts(from_dense(A))` equals `from_dense`
    /// of the dense-augmented A, nonzero-for-nonzero (same col_ptr/row_idx/vals).
    /// This is what lets T3b append cuts to the CSC and drop `a_w` without perturbing
    /// a single coefficient.
    #[test]
    fn csc_augment_matches_dense_augment() {
        use crate::lp::gomory::GomoryCut;
        // 2×3 base with a structural zero; two cuts, one carrying a zero coeff.
        let a = vec![1.0, 0.0, 2.0, 0.0, 3.0, 4.0];
        let (m, n) = (2usize, 3usize);
        let cuts = vec![
            GomoryCut {
                coeffs: vec![1.0, 0.0, -2.0],
                rhs: 1.0,
            },
            GomoryCut {
                coeffs: vec![0.0, 5.0, 0.0],
                rhs: 2.0,
            },
        ];
        // Reference: dense augment, then to CSC.
        let mut a_w = a.clone();
        let mut b = vec![0.0; m];
        let mut c = vec![0.0; n];
        let mut l = vec![0.0; n];
        let mut u = vec![INF; n];
        let mut ii = vec![false; n];
        let (mn, nn) = augment_with_cuts(
            &mut a_w, &mut b, &mut c, &mut l, &mut u, &mut ii, m, n, &cuts,
        );
        let csc_ref = SparseCols::from_dense(&a_w, mn, nn);
        // CSC augment of the CSC of A.
        let csc_test = augment_cols_with_cuts(&SparseCols::from_dense(&a, m, n), m, n, &cuts);
        assert_eq!(
            csc_ref.raw(),
            csc_test.raw(),
            "csc augment != from_dense(dense augment)"
        );
        // b/c/l/u/is_int side-effects (independent of the matrix layout) match the k
        // appended surplus rows/cols.
        assert_eq!((mn, nn), (m + cuts.len(), n + cuts.len()));
    }

    /// T3b1 gate: `farkas_safe_bound_csc`/`verify_farkas_infeasible_csc` give the
    /// identical verdict to their dense oracles across several rays.
    #[test]
    fn farkas_csc_matches_dense() {
        let a = vec![2.0, 0.0, -1.0, 0.0, 3.0, 1.0, 1.0, -2.0, 0.0]; // 3×3
        let (m, n) = (3usize, 3usize);
        let b = vec![1.0, -2.0, 0.5];
        let l = vec![0.0, 0.0, 0.0];
        let u = vec![INF, INF, INF];
        let csc = SparseCols::from_dense(&a, m, n);
        for y in [
            vec![1.0, -1.0, 2.0],
            vec![0.5, 0.5, 0.5],
            vec![-3.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ] {
            assert_eq!(
                farkas_safe_bound(&y, &a, &b, &l, &u, m, n),
                farkas_safe_bound_csc(&y, &csc, &b, &l, &u, m, n),
                "farkas verdict drift for y={y:?}"
            );
            assert_eq!(
                verify_farkas_infeasible(&y, &a, &b, &l, &u, m, n),
                verify_farkas_infeasible_csc(&y, &csc, &b, &l, &u, m, n),
            );
        }
    }

    /// T3b1 gate: `try_rounding_csc` returns the identical incumbent (or `None`) as
    /// the dense oracle across fractional/integral start points.
    #[test]
    fn try_rounding_csc_matches_dense() {
        let a = vec![1.0, 1.0, 1.0]; // 1×3: x0 + x1 + s = b
        let (ns, n_orig_rows, n_w) = (2usize, 1usize, 3usize);
        let b = vec![1.0];
        let c = vec![-1.0, -1.0, 0.0];
        let l = vec![0.0, 0.0, 0.0];
        let u = vec![1.0, 1.0, INF];
        let is_int = vec![true, true, false];
        let csc = SparseCols::from_dense(&a, n_orig_rows, n_w);
        for x in [vec![0.6, 0.3], vec![1.0, 1.0], vec![0.0, 0.0]] {
            let dense = try_rounding(&x, ns, &is_int, &a, &b, &c, &l, &u, n_orig_rows, n_w, 0.0);
            let cscr =
                try_rounding_csc(&x, ns, &is_int, &csc, &b, &c, &l, &u, n_orig_rows, n_w, 0.0);
            assert_eq!(dense, cscr, "try_rounding drift for x={x:?}");
        }
    }

    /// The CSC of each instance round-trips the dense matrix's exact nonzeros
    /// (T1 relies on this equivalence). Sanity-checks `from_dense` on the panel.
    #[test]
    fn csc_roundtrips_dense_nonzeros() {
        for case in panel() {
            let sp = case.csc();
            let mut dense_nnz = 0usize;
            for &v in &case.a {
                if v != 0.0 {
                    dense_nnz += 1;
                }
            }
            let (_col_ptr, _row_idx, vals) = sp.raw();
            assert_eq!(vals.len(), dense_nnz, "{}: csc nnz != dense nnz", case.name);
        }
    }
}

/// #1060: the lazy-constraint separator hook that makes single-tree LP/NLP-BB
/// possible without a commercial MILP backend.
///
/// The properties pinned here are the ones the OA caller depends on: a vetoed
/// point never becomes the answer, a vetoed node stays in the search instead of
/// being fathomed, a separator that cannot make progress costs the certificate
/// rather than producing a false one, and `lazy: None` is inert.
#[cfg(test)]
mod lazy_separation {
    use super::*;
    use crate::bnb::branching::is_integer_feasible;
    use crate::lp::simplex::sparse::SparseCols;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Binary knapsack: `min -10x0 -9x1 -8x2 -x3` s.t. `5(x0+x1+x2+x3) + s = 9`,
    /// all four structural columns binary. The budget admits exactly one item,
    /// so the unconstrained optimum is `x0 = 1`, obj `-10`; with `x0` excluded
    /// it is `x1 = 1`, obj `-9`.
    struct Knapsack {
        a: Vec<f64>,
        m: usize,
        n: usize,
        c: Vec<f64>,
        l: Vec<f64>,
        u: Vec<f64>,
        b: Vec<f64>,
    }

    impl Knapsack {
        fn new() -> Self {
            Knapsack {
                a: vec![5.0, 5.0, 5.0, 5.0, 1.0],
                m: 1,
                n: 5,
                c: vec![-10.0, -9.0, -8.0, -1.0, 0.0],
                l: vec![0.0; 5],
                u: vec![1.0, 1.0, 1.0, 1.0, INF],
                b: vec![9.0],
            }
        }

        fn opts(&self) -> MilpOptions {
            MilpOptions {
                n_struct: 4,
                integer_cols: vec![0, 1, 2, 3],
                max_nodes: 100_000,
                time_limit_s: None,
                gap_tol: 1e-9,
                root_cuts: 16,
                cut_rounds: 3,
                gmi_cuts: true,
                cut_select: true,
                node_cuts: true,
                max_pool_cuts: 500,
                heuristics: true,
                presolve: true,
                strong_branch: true,
                node_propagation: true,
                reduced_cost_fixing: true,
                sb_max_cands: 8,
                sb_node_budget: 1024,
                initial_incumbent: None,
                node_hook_rounds: 0,
                node_hook_cut_cap: 0,
                root_cut_time_s: None,
                root_cut_prune: true,
                simplex: SimplexOptions::default(),
            }
        }

        fn solve_with(&self, lazy: Option<&dyn MilpLazyHook>) -> MilpResult {
            solve_milp_lazy_hooked(
                SparseCols::from_dense(&self.a, self.m, self.n),
                self.m,
                self.n,
                &self.c,
                &self.l,
                &self.u,
                &self.b,
                0.0,
                &self.opts(),
                None,
                lazy,
            )
        }
    }

    /// Rejects any point that uses item 0, returning the globally valid row
    /// `x0 ≤ 0` in the driver's `coeffs · x ≥ rhs` form (`-x0 ≥ 0`).
    struct VetoItemZero {
        calls: AtomicUsize,
    }

    impl MilpLazyHook for VetoItemZero {
        fn separate(&self, x: &[f64]) -> MilpLazyVerdict {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if x.first().copied().unwrap_or(0.0) > 0.5 {
                let mut coeffs = vec![0.0; 4];
                coeffs[0] = -1.0;
                MilpLazyVerdict::Reject(vec![GomoryCut { coeffs, rhs: 0.0 }])
            } else {
                MilpLazyVerdict::Accept
            }
        }
    }

    /// Rejects every point and returns nothing that separates it — the
    /// pathological separator the re-queue cap exists for.
    struct VetoEverythingUselessly {
        calls: AtomicUsize,
    }

    impl MilpLazyHook for VetoEverythingUselessly {
        fn separate(&self, _x: &[f64]) -> MilpLazyVerdict {
            self.calls.fetch_add(1, Ordering::SeqCst);
            MilpLazyVerdict::Reject(Vec::new())
        }
    }

    /// A separator that raised on the Python side.
    struct SeparatorFails {
        calls: AtomicUsize,
    }

    impl MilpLazyHook for SeparatorFails {
        fn separate(&self, _x: &[f64]) -> MilpLazyVerdict {
            self.calls.fetch_add(1, Ordering::SeqCst);
            MilpLazyVerdict::Failed
        }
    }

    /// The headline behaviour: the separator sees the MILP optimum, vetoes it,
    /// and the driver returns the best point that survives the cut — still with
    /// a certificate, because the veto was resolved by a genuine cut and not by
    /// removing a box.
    #[test]
    fn veto_of_the_optimum_returns_the_next_best_certified_point() {
        let k = Knapsack::new();
        let baseline = k.solve_with(None);
        assert_eq!(baseline.status, MilpStatus::Optimal, "baseline status");
        assert!(
            (baseline.obj - (-10.0)).abs() < 1e-6,
            "baseline obj {}",
            baseline.obj
        );
        assert_eq!(
            baseline.lazy_calls, 0,
            "no hook must mean no separator call"
        );

        let hook = VetoItemZero {
            calls: AtomicUsize::new(0),
        };
        let r = k.solve_with(Some(&hook));
        assert_eq!(r.status, MilpStatus::Optimal, "vetoed solve status");
        assert!((r.obj - (-9.0)).abs() < 1e-6, "vetoed solve obj {}", r.obj);
        assert!(
            r.x[0] < 0.5,
            "the vetoed item is still in the answer: {:?}",
            r.x
        );
        // Anti-vacuity (CLAUDE.md §6): a pass here must mean the separator ran.
        let fired = hook.calls.load(Ordering::SeqCst);
        assert!(fired > 0, "separator never fired");
        assert_eq!(r.lazy_calls, fired, "reported lazy_calls != actual calls");
        assert!(
            r.lazy_requeues > 0,
            "the node fire-site never re-queued: the veto was resolved somewhere \
             else, so this test does not exercise the mechanism it claims to"
        );
        assert_eq!(baseline.lazy_requeues, 0, "no hook must mean no re-queue");
    }

    /// A separator that keeps vetoing without ever separating must cost the
    /// certificate, never produce a false `Optimal`. This is the property that
    /// makes the re-queue cap safe (CLAUDE.md §1).
    #[test]
    fn a_separator_that_never_makes_progress_loses_the_certificate() {
        let k = Knapsack::new();
        let hook = VetoEverythingUselessly {
            calls: AtomicUsize::new(0),
        };
        let r = k.solve_with(Some(&hook));
        assert_ne!(
            r.status,
            MilpStatus::Optimal,
            "a search that excluded a box it never proved empty must not certify"
        );
        assert!(
            hook.calls.load(Ordering::SeqCst) > 0,
            "separator never fired"
        );
        assert_eq!(r.lazy_calls, hook.calls.load(Ordering::SeqCst));
        assert!(
            r.lazy_requeues >= LAZY_REQUEUE_CAP as usize,
            "the cap was never reached ({} re-queues), so this test did not \
             exercise the exhaustion path",
            r.lazy_requeues
        );
    }

    /// A separator failure is surfaced, not swallowed (CLAUDE.md §7): the search
    /// stops and returns uncertified rather than reporting a result computed
    /// against constraints that were never enforced.
    #[test]
    fn separator_failure_stops_the_search_uncertified() {
        let k = Knapsack::new();
        let hook = SeparatorFails {
            calls: AtomicUsize::new(0),
        };
        let r = k.solve_with(Some(&hook));
        assert_ne!(
            r.status,
            MilpStatus::Optimal,
            "failed separator must not certify"
        );
        assert!(
            hook.calls.load(Ordering::SeqCst) > 0,
            "separator never fired"
        );
    }

    /// `lazy: None` must be inert — same status, objective, bound and node count
    /// as the plain CSC entry point (CLAUDE.md §5, bound-neutral regime).
    #[test]
    fn lazy_none_is_bound_neutral() {
        let k = Knapsack::new();
        let with_none = k.solve_with(None);
        let plain = solve_milp_csc(
            &SparseCols::from_dense(&k.a, k.m, k.n),
            k.m,
            k.n,
            &k.c,
            &k.l,
            &k.u,
            &k.b,
            0.0,
            &k.opts(),
        );
        assert_eq!(with_none.status, plain.status, "status drift");
        assert_eq!(with_none.nodes, plain.nodes, "node-count drift");
        assert!((with_none.obj - plain.obj).abs() < 1e-12, "obj drift");
        assert!((with_none.bound - plain.bound).abs() < 1e-12, "bound drift");
    }

    /// The separator's integrality screen must agree with the one
    /// `TreeManager::process_evaluated` uses, or a point the tree promotes to
    /// the incumbent could bypass separation entirely. Compared directly against
    /// `branching::is_integer_feasible` on the same mask.
    #[test]
    fn lazy_screen_matches_tree_integrality_predicate() {
        let is_int = vec![true, false, true, true];
        let vars: Vec<VarBranchInfo> = (0..is_int.len())
            .filter(|&j| is_int[j])
            .map(|j| VarBranchInfo {
                offset: j,
                size: 1,
                is_integer: true,
            })
            .collect();
        // Integral, fractional on an integer column, fractional only on the
        // continuous column, at-tolerance either side, and negative values.
        let points: Vec<Vec<f64>> = vec![
            vec![0.0, 0.5, 1.0, 3.0],
            vec![0.5, 0.5, 1.0, 3.0],
            vec![0.0, 0.5, 1.4, 3.0],
            vec![1.0 - 1e-6, 0.5, 1.0, 3.0],
            vec![1.0 + 1e-6, 0.5, 1.0, 3.0],
            vec![1.0 - 1e-3, 0.5, 1.0, 3.0],
            vec![-2.0, -0.25, -3.0, 0.0],
            vec![-2.25, 0.0, 0.0, 0.0],
        ];
        let mut compared = 0usize;
        for p in &points {
            assert_eq!(
                solution_is_integral(p, &is_int),
                is_integer_feasible(p, &vars),
                "screen disagrees with the tree on {p:?}"
            );
            compared += 1;
        }
        // A truncated solution is rejected by both.
        let short = vec![0.0, 0.5];
        assert_eq!(
            solution_is_integral(&short, &is_int),
            is_integer_feasible(&short, &vars),
            "screen disagrees with the tree on a truncated solution"
        );
        compared += 1;
        assert_eq!(compared, 9, "integrality-screen comparison count");
    }
}

/// #1141 — fractional-node cut separation ([`MilpNodeHook`]).
///
/// The hook fires where [`MilpLazyHook`] cannot: at node relaxation solutions
/// that are still fractional. Every test here asserts the separator actually
/// fired (`node_calls`) before asserting anything about the answer — a driver
/// that silently ignored the hook would still answer these models correctly and
/// would read as a pass while testing nothing (CLAUDE.md §6).
#[cfg(test)]
mod node_separation {
    use super::*;
    use crate::lp::simplex::sparse::SparseCols;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Binary knapsack `min −5x0 − 4x1 − 3x2` s.t. `2x0 + 3x1 + x2 + s = 4`.
    ///
    /// Its LP relaxation is fractional (`x1 = 1/3` at the root), which is what
    /// gives the fractional separator anything to see; the plain integer optimum
    /// is `(1, 0, 1)`, objective `−8`. [`CapPair`] holds one more constraint,
    /// `x0 + x2 ≤ 1`, that exists ONLY in the hooks — with it the answer is
    /// `(0, 1, 1)`, objective `−7`.
    struct FracKnapsack {
        a: Vec<f64>,
        m: usize,
        n: usize,
        c: Vec<f64>,
        l: Vec<f64>,
        u: Vec<f64>,
        b: Vec<f64>,
    }

    impl FracKnapsack {
        fn new() -> Self {
            FracKnapsack {
                a: vec![2.0, 3.0, 1.0, 1.0],
                m: 1,
                n: 4,
                c: vec![-5.0, -4.0, -3.0, 0.0],
                l: vec![0.0; 4],
                u: vec![1.0, 1.0, 1.0, INF],
                b: vec![4.0],
            }
        }

        fn opts(&self, rounds: usize, cap: usize) -> MilpOptions {
            MilpOptions {
                n_struct: 3,
                integer_cols: vec![0, 1, 2],
                max_nodes: 100_000,
                time_limit_s: None,
                gap_tol: 1e-9,
                root_cuts: 0,
                cut_rounds: 0,
                gmi_cuts: false,
                cut_select: false,
                node_cuts: false,
                max_pool_cuts: 128,
                heuristics: false,
                presolve: true,
                strong_branch: true,
                node_propagation: false,
                reduced_cost_fixing: true,
                sb_max_cands: 6,
                sb_node_budget: 48,
                initial_incumbent: None,
                node_hook_rounds: rounds,
                node_hook_cut_cap: cap,
                root_cut_time_s: None,
                root_cut_prune: true,
                simplex: SimplexOptions::default(),
            }
        }

        fn solve(
            &self,
            rounds: usize,
            cap: usize,
            lazy: Option<&dyn MilpLazyHook>,
            node: Option<&dyn MilpNodeHook>,
        ) -> MilpResult {
            solve_milp_node_hooked(
                SparseCols::from_dense(&self.a, self.m, self.n),
                self.m,
                self.n,
                &self.c,
                &self.l,
                &self.u,
                &self.b,
                0.0,
                &self.opts(rounds, cap),
                None,
                lazy,
                node,
            )
        }
    }

    /// Holds `x0 + x2 ≤ 1` (driver form `−x0 − x2 ≥ −1`) and separates it at any
    /// violating point, fractional or integral. Serves as both hooks so the two
    /// arms of the agreement test enforce exactly the same constraint.
    struct CapPair {
        node_calls: AtomicUsize,
        fractional_points: AtomicUsize,
    }

    impl CapPair {
        fn new() -> Self {
            CapPair {
                node_calls: AtomicUsize::new(0),
                fractional_points: AtomicUsize::new(0),
            }
        }

        fn row(x: &[f64]) -> Option<GomoryCut> {
            if x[0] + x[2] > 1.0 + 1e-9 {
                Some(GomoryCut {
                    coeffs: vec![-1.0, 0.0, -1.0],
                    rhs: -1.0,
                })
            } else {
                None
            }
        }
    }

    impl MilpNodeHook for CapPair {
        fn separate_node(&self, x: &[f64]) -> MilpNodeVerdict {
            self.node_calls.fetch_add(1, Ordering::SeqCst);
            if x[..3].iter().any(|v| (v - v.round()).abs() > 1e-7) {
                self.fractional_points.fetch_add(1, Ordering::SeqCst);
            }
            match Self::row(x) {
                Some(cut) => MilpNodeVerdict::Cuts(vec![cut]),
                None => MilpNodeVerdict::None,
            }
        }
    }

    impl MilpLazyHook for CapPair {
        fn separate(&self, x: &[f64]) -> MilpLazyVerdict {
            match Self::row(x) {
                Some(cut) => MilpLazyVerdict::Reject(vec![cut]),
                None => MilpLazyVerdict::Accept,
            }
        }
    }

    struct NodeSeparatorFails;

    impl MilpNodeHook for NodeSeparatorFails {
        fn separate_node(&self, _x: &[f64]) -> MilpNodeVerdict {
            MilpNodeVerdict::Failed
        }
    }

    #[test]
    fn absent_hook_leaves_the_search_untouched() {
        let k = FracKnapsack::new();
        let r = k.solve(4, 100, None, None);
        assert_eq!(r.status, MilpStatus::Optimal);
        assert_eq!(r.node_calls, 0, "no hook must mean no calls");
        assert_eq!(r.node_cuts_added, 0, "no hook must mean no rows");
        assert!((r.obj + 8.0).abs() < 1e-9, "unhooked optimum: {}", r.obj);
    }

    #[test]
    fn a_zero_budget_hook_is_treated_as_absent() {
        let k = FracKnapsack::new();
        for (rounds, cap) in [(0usize, 100usize), (4, 0)] {
            let hook = CapPair::new();
            let r = k.solve(rounds, cap, None, Some(&hook));
            assert_eq!(r.node_calls, 0, "rounds={rounds} cap={cap}");
            assert_eq!(hook.node_calls.load(Ordering::SeqCst), 0);
            assert!((r.obj + 8.0).abs() < 1e-9);
        }
    }

    #[test]
    fn the_separator_sees_fractional_points_and_its_calls_are_counted() {
        let k = FracKnapsack::new();
        let hook = CapPair::new();
        let r = k.solve(4, 100, Some(&hook), Some(&hook));
        assert!(r.node_calls > 0, "the fractional separator never fired");
        assert_eq!(
            r.node_calls,
            hook.node_calls.load(Ordering::SeqCst),
            "the driver's call count disagrees with the hook's own"
        );
        assert!(
            hook.fractional_points.load(Ordering::SeqCst) > 0,
            "the separator only ever saw integral points; this is the LAZY hook's \
             job and the fractional path was never exercised"
        );
        assert!(
            r.node_cuts_added > 0,
            "the fractional separator added no rows"
        );
    }

    #[test]
    fn both_hooks_together_agree_with_the_lazy_only_answer() {
        let k = FracKnapsack::new();
        let lazy_only = CapPair::new();
        let a = k.solve(4, 100, Some(&lazy_only), None);
        let both = CapPair::new();
        let b = k.solve(4, 100, Some(&both), Some(&both));
        assert_eq!(a.status, MilpStatus::Optimal, "lazy-only status");
        assert_eq!(b.status, MilpStatus::Optimal, "both-hooks status");
        assert!(b.node_calls > 0, "the fractional separator never fired");
        assert!(
            (a.obj - b.obj).abs() < 1e-9,
            "adding fractional separation moved the certified optimum: {} vs {}",
            a.obj,
            b.obj
        );
        // The constraint the hooks hold is what makes -7 the answer; -8 would mean
        // it was dropped on both arms and the agreement is vacuous.
        assert!((a.obj + 7.0).abs() < 1e-9, "lazy-only optimum: {}", a.obj);
        assert!(b.bound <= b.obj + 1e-9, "dual bound above the incumbent");
    }

    #[test]
    fn the_cut_cap_bounds_the_rows() {
        let k = FracKnapsack::new();
        let hook = CapPair::new();
        let r = k.solve(4, 1, Some(&hook), Some(&hook));
        assert!(r.node_calls > 0, "the fractional separator never fired");
        assert!(
            r.node_cuts_added <= 1,
            "cap ignored: {} rows",
            r.node_cuts_added
        );
    }

    #[test]
    fn exhausting_the_round_budget_keeps_the_certificate() {
        // A node separation is never a veto: when the per-node round budget runs
        // out the node's own valid LP bound is imported and the search continues,
        // certified. (The lazy hook still enforces the constraint, so the answer
        // is the constrained optimum either way.)
        let k = FracKnapsack::new();
        let hook = CapPair::new();
        let r = k.solve(1, 100, Some(&hook), Some(&hook));
        assert!(r.node_calls > 0, "the fractional separator never fired");
        assert_eq!(
            r.status,
            MilpStatus::Optimal,
            "a spent round budget must not cost certification"
        );
        assert!((r.obj + 7.0).abs() < 1e-9, "optimum: {}", r.obj);
        assert!(r.bound <= r.obj + 1e-9);
    }

    #[test]
    fn a_failed_node_separator_stops_the_search_uncertified() {
        let k = FracKnapsack::new();
        let r = k.solve(4, 100, None, Some(&NodeSeparatorFails));
        assert!(r.node_calls > 0, "the fractional separator never fired");
        assert_ne!(
            r.status,
            MilpStatus::Optimal,
            "a separator that failed left rows unenforced; the run must not certify"
        );
    }
}

#[cfg(test)]
mod lazy_infeasible_node {
    use super::*;
    use crate::lp::simplex::sparse::SparseCols;
    use std::sync::Mutex;

    /// Records every point the lazy separator is shown, and never accepts.
    ///
    /// Rejecting unconditionally is the point: a separator that always vetoes
    /// makes any node it is called on get re-queued, so if the driver calls it on
    /// an INFEASIBLE node the search cannot terminate until `LAZY_REQUEUE_CAP`
    /// exhausts -- which drops `gap_certified`.
    struct RecordAll {
        seen: Mutex<Vec<Vec<f64>>>,
    }

    impl MilpLazyHook for RecordAll {
        fn separate(&self, x: &[f64]) -> MilpLazyVerdict {
            self.seen.lock().unwrap().push(x.to_vec());
            // `-x0 >= 1` i.e. `x0 <= -1`: unsatisfiable inside `x >= 0`, so this
            // rejects whatever it is shown.
            MilpLazyVerdict::Reject(vec![GomoryCut {
                coeffs: vec![-1.0, 0.0],
                rhs: 1.0,
            }])
        }
    }

    /// `x0 + x1 = 1` with `x0, x1` binary and an extra row forcing `x0 >= 2`:
    /// every node is infeasible, so the driver must never reach the separator.
    fn infeasible_binary() -> (
        SparseCols,
        usize,
        usize,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
    ) {
        // rows: x0 + x1 + s = 1 ; x0 - t = 2   (s, t >= 0)
        let a = vec![
            1.0, 1.0, 1.0, 0.0, //
            1.0, 0.0, 0.0, -1.0,
        ];
        let (m, n) = (2usize, 4usize);
        (
            SparseCols::from_dense(&a, m, n),
            m,
            n,
            vec![1.0, 1.0, 0.0, 0.0], // c
            vec![0.0, 0.0, 0.0, 0.0], // l
            vec![1.0, 1.0, INF, INF], // u
            vec![1.0, 2.0],           // b
        )
    }

    fn opts() -> MilpOptions {
        MilpOptions {
            n_struct: 2,
            integer_cols: vec![0, 1],
            max_nodes: 100_000,
            time_limit_s: None,
            gap_tol: 1e-9,
            root_cuts: 0,
            cut_rounds: 0,
            gmi_cuts: false,
            cut_select: false,
            node_cuts: false,
            max_pool_cuts: 128,
            heuristics: false,
            presolve: false,
            strong_branch: false,
            node_propagation: false,
            reduced_cost_fixing: false,
            sb_max_cands: 6,
            sb_node_budget: 48,
            initial_incumbent: None,
            node_hook_rounds: 0,
            node_hook_cut_cap: 0,
            root_cut_time_s: None,
            root_cut_prune: true,
            simplex: SimplexOptions::default(),
        }
    }

    /// #1141 item 4: the lazy separator must not be called on an INFEASIBLE node.
    ///
    /// `INFEAS_SENTINEL` is a finite `1e30`, so the old admission test
    /// (`lower_bound.is_finite()`) let an infeasible node through -- and such a
    /// node carries a placeholder solution of zeros, which `solution_is_integral`
    /// accepts. The separator was then handed a point that is not a solution of
    /// anything, returned a cut for it, and the node was re-queued against a
    /// matrix the point still violated. Measured on MINLPLib `tls2`: ONE distinct
    /// assignment re-proposed 1386 times, the point violating 31 of the 35 cut
    /// rows in its own matrix, until the re-queue cap dropped `gap_certified`.
    ///
    /// Fails before the fix: the separator is called and the model is reported
    /// with certification dropped rather than as a clean `Infeasible`.
    #[test]
    fn the_lazy_hook_is_never_called_on_an_infeasible_node() {
        let (csc, m, n, c, l, u, b) = infeasible_binary();
        let hook = RecordAll {
            seen: Mutex::new(Vec::new()),
        };
        let r = solve_milp_lazy_hooked(csc, m, n, &c, &l, &u, &b, 0.0, &opts(), None, Some(&hook));
        let seen = hook.seen.lock().unwrap();
        assert!(
            seen.is_empty(),
            "separator was called {} time(s) on an infeasible model; first point {:?}",
            seen.len(),
            seen.first()
        );
        assert_eq!(r.status, MilpStatus::Infeasible, "bound={}", r.bound);
        assert_eq!(r.lazy_calls, 0);
        assert_eq!(
            r.lazy_requeues, 0,
            "an infeasible node must not be re-queued"
        );
    }

    /// The guard must not cost the separator a node it SHOULD see: the same hook
    /// shape on a feasible model still gets called. Without this the test above
    /// would pass on a driver that never calls the separator at all (§6).
    #[test]
    fn a_feasible_node_still_reaches_the_lazy_hook() {
        // rows: x0 + x1 + s = 1 (drop the infeasible row)
        let a = vec![1.0, 1.0, 1.0];
        let csc = SparseCols::from_dense(&a, 1, 3);
        let hook = RecordAll {
            seen: Mutex::new(Vec::new()),
        };
        let r = solve_milp_lazy_hooked(
            csc,
            1,
            3,
            &[1.0, 1.0, 0.0],
            &[0.0, 0.0, 0.0],
            &[1.0, 1.0, INF],
            &[1.0],
            0.0,
            &opts(),
            None,
            Some(&hook),
        );
        assert!(
            !hook.seen.lock().unwrap().is_empty(),
            "separator never ran on a FEASIBLE model; the guard above proves nothing"
        );
        let _ = r;
    }
}

/// Tests for the root-cut wall bound (`MilpOptions::root_cut_time_s`) and for the
/// root cut loop's newly-honored `time_limit_s` deadline.
#[cfg(test)]
mod root_cut_budget_tests {
    use super::*;

    /// A 0/1 knapsack big enough that cover/GMI cuts at the root measurably
    /// shrink the tree — the differential the tests below read.
    ///
    /// `max Σ p_j x_j  s.t.  Σ w_j x_j ≤ cap`, written in the driver's minimize
    /// form with one surplus column.
    fn knapsack(
        n: usize,
    ) -> (
        SparseCols,
        usize,
        usize,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
    ) {
        // Deterministic pseudo-random weights/profits (no rng dependency): the
        // pattern is arbitrary but fixed, so node counts are reproducible.
        let w: Vec<f64> = (0..n).map(|j| (7 * j % 23 + 11) as f64).collect();
        let p: Vec<f64> = (0..n).map(|j| (13 * j % 29 + 17) as f64).collect();
        let cap: f64 = w.iter().sum::<f64>() / 2.0;
        let mut dense = vec![0.0; n + 1];
        dense[..n].copy_from_slice(&w);
        dense[n] = 1.0; // slack:  Σ w x + s = cap,  s ≥ 0
        let sp = SparseCols::from_dense(&dense, 1, n + 1);
        let c: Vec<f64> = p.iter().map(|v| -v).chain(std::iter::once(0.0)).collect();
        let l = vec![0.0; n + 1];
        let mut u = vec![1.0; n + 1];
        u[n] = INF;
        (sp, 1, n, c, vec![cap], l, u)
    }

    fn budget_opts(ns: usize, root_cut_time_s: Option<f64>) -> MilpOptions {
        MilpOptions {
            n_struct: ns,
            integer_cols: (0..ns).collect(),
            max_nodes: 200_000,
            time_limit_s: None,
            gap_tol: 1e-9,
            root_cuts: 200,
            cut_rounds: 30,
            gmi_cuts: true,
            cut_select: true,
            node_cuts: false,
            max_pool_cuts: 500,
            heuristics: true,
            presolve: true,
            strong_branch: true,
            node_propagation: false,
            reduced_cost_fixing: true,
            sb_max_cands: 8,
            sb_node_budget: 1024,
            initial_incumbent: None,
            node_hook_rounds: 0,
            node_hook_cut_cap: 0,
            root_cut_time_s,
            root_cut_prune: true,
            simplex: SimplexOptions::default(),
        }
    }

    /// The budget really gates the separation phase, and gating it is visible.
    ///
    /// Same instance, same everything else: `root_cut_time_s: Some(0.0)` expires
    /// before the first round is entered, so no root cuts are separated. The
    /// cut-strengthened arm closes the 40-item knapsack at the root (1 node); the
    /// budgeted-out arm has to branch (9 nodes). That strict inequality is the
    /// anti-vacuity check (CLAUDE.md §6) -- if the budget were ignored both arms
    /// would separate cuts and the node counts would be equal.
    #[test]
    fn a_zero_root_cut_budget_skips_the_separation_phase() {
        let (sp, m, ns, c, b, l, u) = knapsack(40);
        let n = l.len();
        let full = solve_milp_csc(&sp, m, n, &c, &l, &u, &b, 0.0, &budget_opts(ns, None));
        let zero = solve_milp_csc(&sp, m, n, &c, &l, &u, &b, 0.0, &budget_opts(ns, Some(0.0)));

        assert_eq!(
            full.status,
            MilpStatus::Optimal,
            "unbudgeted arm must certify"
        );
        assert_eq!(
            zero.status,
            MilpStatus::Optimal,
            "budgeted arm must certify"
        );
        assert!(
            zero.nodes > full.nodes,
            "the root-cut budget did not gate the loop: {} nodes with cuts, {} without \
             -- an equal count means both arms separated and the test proves nothing",
            full.nodes,
            zero.nodes
        );
    }

    /// Gating the cuts must not change the ANSWER. The budget trades search
    /// effort for wall time; a different optimum, or a dual bound above it, would
    /// mean the loop is load-bearing for correctness rather than for speed
    /// (CLAUDE.md §1).
    #[test]
    fn the_root_cut_budget_never_changes_the_certified_optimum() {
        for n_items in [40usize, 300] {
            let (sp, m, ns, c, b, l, u) = knapsack(n_items);
            let n = l.len();
            let full = solve_milp_csc(&sp, m, n, &c, &l, &u, &b, 0.0, &budget_opts(ns, None));
            let zero = solve_milp_csc(&sp, m, n, &c, &l, &u, &b, 0.0, &budget_opts(ns, Some(0.0)));
            assert_eq!(full.status, MilpStatus::Optimal, "n={n_items}");
            assert_eq!(zero.status, MilpStatus::Optimal, "n={n_items}");
            assert!(
                (full.obj - zero.obj).abs() < 1e-6,
                "n={n_items}: budgeted arm found {} but unbudgeted found {}",
                zero.obj,
                full.obj
            );
            for (name, r) in [("unbudgeted", &full), ("budgeted", &zero)] {
                assert!(
                    r.bound <= r.obj + 1e-6,
                    "n={n_items} {name}: dual bound {} above incumbent {} -- false \
                     certificate",
                    r.bound,
                    r.obj
                );
            }
        }
    }

    /// A budget larger than the separation phase needs is a no-op: the loop runs
    /// to its natural stop (cut cap / tailing off) and the tree is the same size
    /// as with no budget at all. Without this the test above could pass with a
    /// budget that always expires, which would make the option useless rather
    /// than working.
    #[test]
    fn a_generous_root_cut_budget_is_indistinguishable_from_none() {
        let (sp, m, ns, c, b, l, u) = knapsack(300);
        let n = l.len();
        let none = solve_milp_csc(&sp, m, n, &c, &l, &u, &b, 0.0, &budget_opts(ns, None));
        let generous = solve_milp_csc(
            &sp,
            m,
            n,
            &c,
            &l,
            &u,
            &b,
            0.0,
            &budget_opts(ns, Some(600.0)),
        );
        assert_eq!(none.status, generous.status);
        assert_eq!(
            none.nodes, generous.nodes,
            "a 600 s budget must not perturb a separation phase that takes \
             milliseconds"
        );
        assert!((none.obj - generous.obj).abs() < 1e-12);
    }

    /// Densify a CSC for element-wise comparison in the tests below.
    fn dense_of(sp: &SparseCols, m: usize, n: usize) -> Vec<f64> {
        let mut d = vec![0.0; m * n];
        for j in 0..n {
            let (rows, vals) = sp.col(j);
            for (&r, &v) in rows.iter().zip(vals) {
                d[r * n + j] = v;
            }
        }
        d
    }

    /// The whole reason `rebuild_csc_with_cuts` exists: a GMI cut's coefficient
    /// vector spans the slack columns of the cuts before it, and the batch
    /// appender cannot place those.
    ///
    /// Two cuts on a 1x2 base; the second has a coefficient on the FIRST cut's
    /// surplus column (`n_base + 0`). The rebuild must carry it. The assertion
    /// against `augment_cols_with_cuts` on the same input is the anti-vacuity
    /// half (CLAUDE.md §6): it pins that the two functions genuinely differ here,
    /// so a future "simplification" back to the batch appender fails loudly
    /// instead of silently installing a different row.
    #[test]
    fn the_rebuild_places_a_cut_coefficient_on_an_earlier_cuts_slack() {
        let base = SparseCols::from_dense(&[2.0, 3.0], 1, 2);
        let (m_base, n_base) = (1usize, 2usize);
        let cuts = vec![
            GomoryCut {
                coeffs: vec![1.0, 1.0],
                rhs: 1.0,
            },
            GomoryCut {
                // ... plus 0.5 * (cut 0's surplus), column n_base + 0 == 2.
                coeffs: vec![1.0, 0.0, 0.5],
                rhs: 0.25,
            },
        ];
        let m_new = m_base + 2;
        let n_new = n_base + 2;

        let rebuilt = dense_of(
            &rebuild_csc_with_cuts(&base, m_base, n_base, &cuts),
            m_new,
            n_new,
        );
        // row 0: the base row, no surplus columns
        assert_eq!(&rebuilt[0..4], &[2.0, 3.0, 0.0, 0.0]);
        // row 1: cut 0 with its own -1 surplus
        assert_eq!(&rebuilt[4..8], &[1.0, 1.0, -1.0, 0.0]);
        // row 2: cut 1, including the 0.5 on cut 0's surplus column
        assert_eq!(&rebuilt[8..12], &[1.0, 0.0, 0.5, -1.0]);

        let batched = dense_of(
            &augment_cols_with_cuts(&base, m_base, n_base, &cuts),
            m_new,
            n_new,
        );
        assert_eq!(
            batched[8 + 2],
            0.0,
            "augment_cols_with_cuts is expected to DROP the cross-slack coefficient \
             (it only scans j < n_base); if it now keeps it, rebuild_csc_with_cuts \
             is redundant and this test is no longer proving anything"
        );
    }

    /// The cleanup must remove cuts, and removing them must not move the answer.
    ///
    /// A 300-item knapsack under a 200-cut / 30-round budget separates far more
    /// cuts than the final root LP leans on. The counters make the drop visible
    /// (CLAUDE.md §6 -- without the strict `kept < generated` this test would pass
    /// on a cleanup that never fired), and the certified optimum is compared
    /// against the same solve with the separation phase budgeted out entirely.
    #[test]
    fn the_root_cut_cleanup_drops_slack_cuts_without_moving_the_optimum() {
        let _g = crate::profile::test_guard();
        let (sp, m, ns, c, b, l, u) = knapsack(300);
        let n = l.len();

        let reference = solve_milp_csc(&sp, m, n, &c, &l, &u, &b, 0.0, &budget_opts(ns, Some(0.0)));

        crate::profile::set_enabled(true);
        crate::profile::reset();
        let cut = solve_milp_csc(&sp, m, n, &c, &l, &u, &b, 0.0, &budget_opts(ns, None));
        let generated = crate::profile::counter(crate::profile::Ctr::RootCutsGenerated);
        let kept = crate::profile::counter(crate::profile::Ctr::RootCutsKept);
        crate::profile::set_enabled(false);

        assert!(
            generated > 0,
            "no root cuts were separated -- the cleanup had nothing to act on and \
             this test is vacuous"
        );
        assert!(
            kept < generated,
            "cleanup kept all {generated} root cuts; it never removed a row, so the \
             per-node cost it exists to avoid is still being paid"
        );
        assert_eq!(cut.status, MilpStatus::Optimal);
        assert_eq!(reference.status, MilpStatus::Optimal);
        assert!(
            (cut.obj - reference.obj).abs() < 1e-6,
            "cleanup changed the optimum: {} vs {}",
            cut.obj,
            reference.obj
        );
        assert!(
            cut.bound <= cut.obj + 1e-6,
            "dual bound {} above incumbent {} after cleanup -- false certificate",
            cut.bound,
            cut.obj
        );
    }
}
