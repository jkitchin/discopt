//! Native spatial branch-and-bound tree loop (issue #764, C1 build-order item 4).
//!
//! Composes [`solve_spatial_node`] into a full spatial B&B: pop a box, solve the
//! node relaxation (rigorous safe bound), OBBT-tighten the box, and either accept a
//! feasible point as the incumbent or branch. Runs entirely in `discopt-core`.
//!
//! Soundness (the non-negotiable, correctness-first contract):
//! * **Pruning** uses only the [`ns_safe_bound_csc`](crate::lp::simplex::refine)
//!   safe lower bound (`<=` the true node optimum), so a node is fathomed only when
//!   its relaxation *provably* cannot beat the incumbent.
//! * **Incumbent acceptance** requires a *sufficient* feasibility condition — every
//!   integer candidate integral AND every lifted term tight (`|x_aux − f(operands)|
//!   <= mccormick_tol`, i.e. the McCormick relaxation is exact at the point) — so an
//!   accepted point is genuinely feasible for the original nonconvex problem and its
//!   linear objective value is valid. A looser check could bless an infeasible point
//!   (`incorrect_count > 0`); this never does.
//! * **Branching** partitions the box into two covering children (`ub = p` and
//!   `lo = p` for a split point `p in [lo,hi]`), whose union is the parent — so no
//!   feasible point is ever lost.

use crate::bnb::convex_quad_cuts::{detect_convex_quad_forms, separate_round, ConvexQuadForm};
use crate::bnb::spatial_kernel::{
    solve_spatial_node_with_cuts, EnvTerm, FixedRow, SpatialKernelSpec, SpatialNodeResult,
};
use crate::bnb::spatial_propagate::propagate_spec_fixpoint;
use crate::lp::basis::Basis;
use crate::lp::simplex::{LpStatus, SimplexOptions};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::rc::Rc;
use std::time::{Duration, Instant};

/// An open B&B node in the best-bound frontier: its box and the inherited lower
/// bound `pb` (the parent's rigorous bound, a valid lower bound for this region).
/// Ordered so a max-heap yields the SMALLEST `pb` first (best-bound search): the
/// lowest-bound region is explored + tightened first, which is what lifts the global
/// frontier minimum — pure DFS leaves low-bound siblings unexplored and the reported
/// bound stuck at the root value.
struct QNode {
    pb: f64,
    lo: Vec<f64>,
    hi: Vec<f64>,
    /// #1913: the integer branch that created this node -- `(column, up, distance
    /// the LP value moved)` -- so its LP bound can update that column's pseudocost.
    /// `None` at the root and for spatial children.
    br: Option<(usize, bool, f64)>,
    /// #1970: the same for a spatial branch -- `(column, up, width cut off)` --
    /// recorded only under `SpatialTreeConfig::spatial_score`.
    sbr: Option<(usize, bool, f64)>,
    /// The node-LP basis this node starts from, shared by both children: the
    /// parent's optimal basis, or, when the parent's LP decided nothing, the one
    /// the parent itself inherited. `None` unless
    /// [`SpatialTreeConfig::warm_start_children`] is on (and at the root).
    basis: Option<Rc<Basis>>,
    /// The `node_count` id of the node that pushed this one (`None` at the root);
    /// read only by a [`NodeObserver`].
    parent: Option<usize>,
    /// #1962: the globally valid cuts this node's LP starts with -- the ones that
    /// were binding at its parent's final LP. They are the last rows of the node
    /// LP, in this order, so `basis` (which was taken over the same rows) fits it.
    cuts: Rc<Vec<FixedRow>>,
}

impl PartialEq for QNode {
    fn eq(&self, o: &Self) -> bool {
        self.pb == o.pb
    }
}
impl Eq for QNode {}
impl PartialOrd for QNode {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for QNode {
    fn cmp(&self, o: &Self) -> Ordering {
        // Reverse: BinaryHeap is a max-heap, we want the min `pb` on top.
        o.pb.partial_cmp(&self.pb).unwrap_or(Ordering::Equal)
    }
}

/// What a node LP's terminal status licenses the tree to conclude about the region
/// the node covers.
///
/// #927: this mapping is the soundness hinge of the whole search. Only a
/// Farkas-certified [`LpStatus::Infeasible`] proves a region is empty; every other
/// non-optimal status is the LP declining to decide, and pruning on it fabricates
/// an emptiness proof the solver never had. The `match` is exhaustive on purpose —
/// a new `LpStatus` variant must make an explicit choice here rather than silently
/// inheriting "prune it".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NodeVerdict {
    /// The LP solved: its safe bound is a valid lower bound for the region.
    Bound,
    /// The region is PROVEN empty (certified infeasible) — it contributes nothing.
    EmptyRegion,
    /// The LP decided nothing. The region may well contain the optimum, so it must
    /// be branched (or closed with its inherited bound), never pruned.
    Undecided,
}

fn verdict_for(status: LpStatus) -> NodeVerdict {
    match status {
        LpStatus::Optimal => NodeVerdict::Bound,
        // `Infeasible` is returned only after the simplex verifies a Farkas dual
        // ray (see `lp::simplex`), so it is a proof.
        LpStatus::Infeasible => NodeVerdict::EmptyRegion,
        LpStatus::Unbounded | LpStatus::IterLimit | LpStatus::Numerical => NodeVerdict::Undecided,
    }
}

/// Termination status of the tree solve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TreeStatus {
    /// The gap closed to `gap_tol` (global bound met the incumbent) — a genuine
    /// certificate: every region was either explored or fathomed with a rigorous
    /// bound `>= incumbent - gap_tol`.
    Optimal,
    /// The node budget was exhausted with the gap still open.
    NodeLimit,
    /// The wall-clock budget was exhausted with the gap still open.
    TimeLimit,
    /// The worklist emptied but the certified global bound did NOT reach
    /// `incumbent - gap_tol` — some regions could only be closed with weaker
    /// rigorous bounds (width-exhausted boxes, uncertifiable node duals). The
    /// incumbent and `bound` are both valid; the gap between them is honest
    /// residual uncertainty. NEVER reported as `Optimal`.
    Exhausted,
    /// No feasible point exists (the root relaxation was infeasible).
    Infeasible,
    /// The caller's [`IncumbentValueFn`] answered [`IncumbentPrice::Decline`]: at a
    /// McCormick-tight point the lifted objective and the caller's objective
    /// disagree beyond what slack explains, so this kernel's model is not the
    /// caller's problem (#1656). Neither the incumbent nor the bound is a claim
    /// about the caller's problem; the caller solves it another way.
    Declined,
}

/// Whether `bound` closes the region against `inc` under the configured gap.
///
/// The single place the gap criterion is spelled out, so the fathoming tests,
/// the node-limit exit and the terminal `Optimal` verdict can never drift apart
/// — which is exactly how a fathom looser than the certificate becomes a false
/// `Optimal`.
///
/// The test is three clauses, all of which must hold:
/// `inc - bound <= gap_tol` AND (`inc - bound <= abs_gap_tol` OR
/// `inc - bound <= rel_gap_tol * max(|inc_pub|, |bound_pub|)`). What a caller's
/// `gap_tolerance` means therefore depends on how `solver.py` fills the fields
/// (#1615 D-12, #1678):
///
/// - `gap_tolerance <= 1e-4` (the default or tighter): `gap_tol = gap_tolerance`
///   (or `min` with `abs_gap_tolerance`), so the stop is ABSOLUTE at
///   `gap_tolerance`, further conjoined with the relative clause below unit
///   objective magnitude (#1263).
/// - `gap_tolerance > 1e-4` (an explicit loosening): `gap_tol = 1e300`, so the
///   first clause is vacuous for any finite bound and the stop is the Python
///   tree's disjunction — absolute `abs_gap_tol` OR RELATIVE `gap_tolerance`.
///
/// History of the absolute clause below. The kernel was purely absolute
/// before #1263/#1615; #1243 briefly gave it a
/// relative second arm so it would match the Python tree's disjunction; that was
/// reverted, because the relative arm never existed here and adding one can only
/// LOOSEN the fathom — a caller tightening `abs_gap_tolerance` would have
/// widened the effective tolerance by orders of magnitude on a large objective.
/// `solver.py` now maps a caller's absolute tolerance through `min`, so this
/// route honours a tightening and declines a loosening.
///
/// #1263: the absolute `gap_tol` is additionally CONJOINED with the documented
/// `solver.py` criterion (absolute `abs_gap_tol` OR relative `rel_gap_tol` against
/// `max(|inc|, |bound|)`). `solver.py` passes `gap_tol = gap_tolerance`, so below
/// unit objective magnitude the absolute arm alone was exactly the 1.0-floored
/// relative test `_DEFAULT_ABS_GAP_TOL` exists to correct (`st_z`: `Optimal` at
/// incumbent 2.7e-5 over a true optimum of 0). A conjunction can only TIGHTEN, and
/// at `|inc| >= 1` with `rel_gap_tol >= gap_tol` the second clause is implied by
/// the first, so those solves are unchanged. The defaults (`rel_gap_tol = inf`)
/// reproduce the purely absolute test.
///
/// #1656: the relative arm is measured on the objective the caller is handed,
/// `inc + obj_offset`, not on the kernel's constant-free internal value. The two
/// differ whenever the objective carries a constant: `(xy-1)^2 + ...` expands to
/// an internal optimum of -1.1026 with offset +1.13, published as 0.02735. The
/// internal test closed a 9.7e-5 gap as 8.8e-5 relative; against the published
/// 0.02735 it is 3.6e-3, and `_refuse_unclosed_published_pair` (#1536) withdrew
/// the certificate the kernel had stopped on. The gap itself is offset-invariant.
#[inline]
fn gap_closed(bound: f64, inc: f64, config: &SpatialTreeConfig) -> bool {
    let absolute_closed = bound >= inc - config.gap_tol; // false on NaN
    if !absolute_closed {
        return false;
    }
    let gap = inc - bound;
    if gap <= config.abs_gap_tol {
        return true;
    }
    let (inc_pub, bound_pub) = (inc + config.obj_offset, bound + config.obj_offset);
    gap <= config.rel_gap_tol * inc_pub.abs().max(bound_pub.abs()).max(1e-10)
}

/// Tunables for [`solve_spatial_tree`].
#[derive(Clone, Copy, Debug)]
pub struct SpatialTreeConfig {
    /// Maximum nodes to process before returning [`TreeStatus::NodeLimit`].
    pub max_nodes: usize,
    /// Absolute monotonic-clock deadline. Checked before every node so an expired
    /// solve returns its best incumbent and rigorous frontier bound without ever
    /// claiming [`TreeStatus::Optimal`]. `None` means no wall-clock limit.
    pub deadline: Option<Instant>,
    /// Absolute gap `incumbent − global_bound` at/below which the solve stops.
    pub gap_tol: f64,
    /// Relative gap (against `max(|incumbent|, |bound|)`) that must ALSO hold
    /// unless the absolute gap is within [`Self::abs_gap_tol`] (#1263). `inf`
    /// (the default) disables the extra clause.
    pub rel_gap_tol: f64,
    /// Absolute gap that satisfies the #1263 clause on its own.
    pub abs_gap_tol: f64,
    /// Constant added to the internal objective to give the published one
    /// (`meta_obj_offset`). Only the relative arm of [`gap_closed`] reads it, so the
    /// relative gap is judged on the scale the caller sees (#1656). `0.0` (the
    /// default) reproduces the pre-#1656 test.
    pub obj_offset: f64,
    /// Integrality tolerance for incumbent acceptance / integer branching.
    pub int_tol: f64,
    /// McCormick-exactness tolerance for incumbent acceptance (`|x_aux − f|`).
    pub mccormick_tol: f64,
    /// Minimum box width worth spatial-branching (avoids infinite splitting).
    pub min_box_width: f64,
    /// Whether to run the in-kernel OBBT sweep at each node. Default OFF since the
    /// C2 entry experiment (2026-07-19): cheap FBBT propagation is the validated
    /// default tightening; OBBT's ~2·n LP probes/node are the expensive substitute
    /// it replaces.
    pub run_obbt: bool,
    /// Whether to run the FBBT fixpoint propagation at each node BEFORE the LP:
    /// linear rows, products with extended division, sqrt/monomial/affine-square,
    /// integer rounding, and the objective cutoff. Default ON — the C2-validated
    /// mechanism that climbs the dual bound (zero LP solves).
    pub run_propagation: bool,
    /// Fixpoint round cap for the per-node propagation.
    pub propagation_rounds: usize,
    /// Externally-supplied valid upper bound — the objective value (internal
    /// minimize units) of a KNOWN feasible point (e.g. from an NLP heuristic).
    /// Seeded as the initial incumbent (with an empty `incumbent_x`, since the
    /// point lives with the caller): it prunes and cutoff-propagates exactly like
    /// an internally-found incumbent, and is reported back unchanged if never
    /// improved. Soundness requires the value to genuinely be attained by a
    /// feasible point.
    pub initial_incumbent: Option<f64>,
    /// Extra wall-clock time this search may take when [`Self::deadline`] expires —
    /// but ONLY if it already holds an incumbent (#917).
    ///
    /// `Model.solve` withholds 35% of the caller's `time_limit` as a reserve for the
    /// #844 no-incumbent fallback and spends it only when the primary returns
    /// nothing, so a primary that finds an incumbent and then hits its reduced
    /// deadline forfeits that slice outright — nobody spends it. Reclaiming it here is
    /// safe by construction: the fallback exists exclusively for the no-incumbent
    /// case, so the state that unlocks this extension is precisely the state in which
    /// the fallback provably has nothing to contribute.
    ///
    /// Taken at most once. `None` (the default) reproduces the pre-#917 deadline
    /// exactly. Measured on the in-repo corpus at a 60 s budget, giving the reserve
    /// back is worth a lot on this path: nvs17's bound goes -1149.20 -> -1100.40
    /// (closing onto its own incumbent), nvs19 -4017.37 -> -2303.40, nvs23
    /// -23735.23 -> -18951.65.
    pub incumbent_time_extension: Option<Duration>,
    /// Extra wall-clock time this search may take when [`Self::deadline`] expires —
    /// but ONLY if its global dual bound is already finite (#933).
    ///
    /// This is the kernel-side mirror of `Model.solve`'s #844 policy ("withhold the
    /// root-fallback reserve precisely while the search is bound-less"): the caller
    /// shortens the kernel's deadline by the reserve it wants to keep for the
    /// root-relaxation fallback and passes that reserve here. A kernel whose
    /// frontier is finite at the shortened deadline reclaims the slice (the
    /// fallback has nothing to contribute — the kernel's own bound is at least as
    /// tight as any root bound); a kernel still bound-less exits immediately,
    /// leaving the reserve unspent so the caller can compute a root-relaxation
    /// bound INSIDE the original budget instead of reporting no bound at all.
    ///
    /// Taken at most once. `None` (the default) reproduces the prior deadline
    /// exactly.
    pub bound_time_extension: Option<Duration>,
    /// #1619 D-11: choose the spatial branch scale-free. Off (the default), the
    /// term to branch is the one with the largest absolute McCormick gap
    /// `|x_aux - f(x)|` and its operand the one with the widest absolute box, so
    /// the tree depends on the units the model is written in: the issue's pooling
    /// plant took 15,745 nodes with flows in barrels and 593 in hundreds of
    /// barrels. On, each term's gap is divided by its aux column's root width and
    /// each operand's width by its own root width, which a change of units (a
    /// positive rescaling of a column) leaves unchanged. Branching order only:
    /// every child pair still covers its parent box, so no bound moves. `false`
    /// here (the struct default) is the legacy rule, and so is the Python
    /// orchestrator's default since 2026-10-22 (`DISCOPT_SCALE_FREE_BRANCHING=1`
    /// opts in): scaling by possibly-loose ROOT widths lost tanksize's certificate.
    pub scale_free_branching: bool,
    /// #1913: choose the fractional integer column to branch on by pseudocost
    /// (product rule over the per-unit bound gains observed on each side, unknown
    /// sides taking the average of the known ones). Off (the struct default) is the
    /// legacy rule: the lowest-index fractional column; the Python orchestrator turns
    /// it on by default since #1941 (`DISCOPT_KERNEL_PSEUDOCOST=0` opts out).
    /// Branching order only: every child pair still covers its parent box, so no
    /// bound moves.
    pub pseudocost_branching: bool,
    /// #1913: reliability initialisation for `pseudocost_branching`. A fractional
    /// candidate with fewer than this many observations on either side has both
    /// child LPs solved (after propagation) before the pick, and their gains are
    /// recorded as observations (at most `STRONG_BRANCH_CANDIDATES` per node, and
    /// skipped unless ~512 more nodes fit in the time left). `0` is plain pseudocost branching.
    pub pseudocost_reliability: usize,
    /// #1935 warm-start experiment: start each child's node LP from its parent's
    /// optimal basis (dual simplex) instead of solving it from scratch. Off (the
    /// default) is the cold solve every node has always had.
    ///
    /// Why it is worth measuring: #1935 replaced the node-LP engine with HiGHS
    /// warm-started from the parent and measured the median per-node LP time
    /// falling 6.27 -> 0.81 ms, but its baseline was this kernel's COLD solve, so
    /// the number does not separate the engine from the warm start. This flag is
    /// the other half of that comparison, on the in-house simplex.
    ///
    /// Bound-changing, not bound-neutral: a different starting basis can end on a
    /// different optimal vertex of a degenerate node LP, with different duals and
    /// so a different safe bound, LP point and branching decision. Soundness does
    /// not depend on the start (see `solve_spatial_node_from`). Cost: each queued
    /// node holds one shared parent basis (`n_total` status bytes plus `m` indices).
    pub warm_start_children: bool,
    /// #1962: separation rounds at the root node. Each round adds the violated
    /// outer-approximation cut of every certified convex quadratic form
    /// ([`crate::bnb::convex_quad_cuts`]) at the node LP's point and re-solves the
    /// LP warm. `0` (the default), together with `cut_rounds_node == 0`, is the
    /// single-solve node every release before it has had.
    pub cut_rounds_root: usize,
    /// #1962: separation rounds at every non-root node (see `cut_rounds_root`).
    pub cut_rounds_node: usize,
    /// #1962: most cuts one round adds (the most violated first).
    pub max_cuts_per_round: usize,
    /// #1962: the integral-point probe. When a node's LP point is integral on every
    /// integer column but some lifted term is not tight, re-solve the node LP with
    /// the integer original columns fixed at their LP values (once per assignment)
    /// and judge that LP's point for acceptance like any other. Primal only. Off
    /// (the default) is the tree before it.
    pub int_fix_probe: bool,
    /// #1970: choose the spatial branching column by a score over candidates
    /// (SCIP's nonlinear-handler rule, Bestuzheva et al. 2023) instead of the
    /// widest operand of the single largest-gap term. Every operand of every
    /// violated term is a candidate; its violation score is the sum, over the
    /// terms it appears in, of that term's gap split evenly across the term's
    /// operands. With [`Self::spatial_pscost_weight`] `> 0` a spatial pseudocost
    /// (per-side bound gain per unit of width cut off, recorded as children are
    /// solved) is added, each component normalised by its maximum over the
    /// candidates. Off (the default) is the legacy rule. Branching order only:
    /// every child pair still covers its parent box, so no bound moves.
    pub spatial_score: bool,
    /// #1970: weight of the pseudocost component under [`Self::spatial_score`]
    /// (the violation component has weight 1). `0` scores by violation alone.
    pub spatial_pscost_weight: f64,
    /// #1970: spatial split point. `0` (the default) splits at the LP value
    /// (pulled to the midpoint only within ~1e-9 of a bound); `m in (0, 1]` takes
    /// `m * mid + (1 - m) * lp` (SCIP's `branching/midpull`), which keeps the
    /// point at least `m / 2` of the width from either bound.
    pub spatial_midpull: f64,
    /// #1995: when a node LP decides nothing (`Numerical` / `IterLimit` /
    /// `Unbounded`) over a box that propagation tightened, solve the node again
    /// over the box it had BEFORE propagation and use that result. The objective
    /// cutoff can collapse a box against an incumbent at the LP optimum into a
    /// sliver of width ~1e-8, where the McCormick LP is numerically degenerate;
    /// every descendant inherits the sliver, so the undecided arm branches forever.
    /// Sound: the pre-propagation box contains every point of the node's region
    /// (propagation removes only points that are infeasible or no better than the
    /// incumbent), so its LP bound, Farkas emptiness proof and McCormick-tight
    /// points are valid for the region -- that node is solved exactly as
    /// `run_propagation = false` would solve it. Changes nothing at a node whose
    /// first LP decides. Off (the default) is the tree before it.
    pub undecided_unpropagated_retry: bool,
}

impl Default for SpatialTreeConfig {
    fn default() -> Self {
        Self {
            max_nodes: 100_000,
            deadline: None,
            gap_tol: 1e-6,
            rel_gap_tol: f64::INFINITY,
            abs_gap_tol: 0.0,
            obj_offset: 0.0,
            int_tol: 1e-5,
            mccormick_tol: 1e-6,
            min_box_width: 1e-9,
            run_obbt: false,
            run_propagation: true,
            propagation_rounds: 15,
            initial_incumbent: None,
            incumbent_time_extension: None,
            bound_time_extension: None,
            scale_free_branching: false,
            pseudocost_branching: false,
            pseudocost_reliability: 0,
            warm_start_children: false,
            cut_rounds_root: 0,
            cut_rounds_node: 0,
            max_cuts_per_round: 50,
            int_fix_probe: false,
            spatial_score: false,
            spatial_pscost_weight: 0.0,
            spatial_midpull: 0.0,
            undecided_unpropagated_retry: false,
        }
    }
}

/// Result of the tree solve.
#[derive(Clone, Debug)]
pub struct SpatialTreeResult {
    /// Termination status.
    pub status: TreeStatus,
    /// Best feasible objective found, or `None` if no incumbent was accepted.
    pub incumbent: Option<f64>,
    /// The incumbent point (structural columns), empty if none.
    pub incumbent_x: Vec<f64>,
    /// Global lower bound (min sense): `<=` the true optimum.
    pub bound: f64,
    /// Nodes processed.
    pub node_count: usize,
    /// Total LP solves across all nodes (relaxation + OBBT probes).
    pub n_lp_solves: usize,
    /// Nodes whose LP solved to optimality but whose Neumaier–Shcherbina safe
    /// bound could NOT be certified (`-inf` — non-finite duals / infinite-bound
    /// reduced costs). These nodes carry only their inherited parent bound, so a
    /// subtree of them freezes the frontier — the diagnostic for a bound plateau
    /// caused by certification failure rather than relaxation looseness.
    pub n_uncertified: usize,
    /// #927: nodes whose LP came back with a status that DECIDES NOTHING
    /// (`Numerical` / `IterLimit` / `Unbounded`). These are branched rather than
    /// fathomed — a nonzero count means the search hit ill-conditioned node LPs and
    /// paid for them in extra nodes, which is exactly the trade that keeps the
    /// certificate honest. Reported so "the fallback fired" is observable.
    pub n_undecided: usize,
    /// #917: seconds of the caller's withheld reserve this search actually reclaimed
    /// (0.0 when it never did). Reported so "the extension fired" is directly
    /// observable instead of inferred from a wall-clock reading — a panel that can
    /// only guess whether the mechanism ran cannot score it (CLAUDE.md §6).
    pub incumbent_extension_s: f64,
    /// #933: seconds of the caller's withheld bound-fallback reserve this search
    /// reclaimed because its dual bound was already finite at the shortened
    /// deadline (0.0 when it never did — including the bound-less exit that
    /// leaves the reserve to the caller's root-relaxation fallback). Reported
    /// for the same §6 observability reason as `incumbent_extension_s`.
    pub bound_extension_s: f64,
    /// #1236: the valid lower bound this search established for the ROOT region —
    /// the contribution node 1 makes to `bound` before any branching. `-inf` when
    /// the root never produced one (the search exited on its deadline before the
    /// root node finished, or the root LP could not be certified).
    ///
    /// This is what makes the kernel's dual side diagnosable at all. Everything
    /// the caller could previously see was the FINAL `bound`, so "the relaxation
    /// is loose at the root and the tree closes it" and "the root is already tight
    /// and the tree is spending its nodes on the primal side" were the same
    /// observation — and the Python `SolveResult` reported `root_bound=None` on
    /// every kernel-routed solve (CLAUDE.md §6).
    pub root_bound: f64,
    /// #1236: seconds elapsed when `root_bound` was established (0.0 when it never
    /// was). The kernel's analogue of the `root_time` every Python driver reports.
    pub root_time_s: f64,
    /// #1236 (review finding 8): WHICH arm node 1 left by, because `root_bound`
    /// alone cannot say. Three distinct outcomes carry a non-finite bound —
    /// `"infeasible"` (the root region was certified empty, `+inf`), `"undecided"`
    /// (the root LP decided nothing, so the region closes with the parent's `-inf`)
    /// and `"not_reached"` (the search exited before node 1 finished) — and the
    /// Python mapping turns every non-finite value into `root_bound=None`. Without
    /// this they are the same observation from Python, which is the §6 conflation
    /// this instrumentation exists to remove, reproduced inside it.
    pub root_status: &'static str,
    /// #1522: how many times the caller's primal hook was consulted, and how many
    /// of those calls replaced the incumbent. Reported so a panel can tell "the
    /// hook never fired" from "it fired and found nothing" (CLAUDE.md §6). Both
    /// are 0 when no hook was supplied.
    pub n_primal_hook_calls: usize,
    /// #1522: primal-hook calls that replaced the incumbent (see above).
    pub n_primal_hook_improvements: usize,
    /// #1656: how many McCormick-tight LP points the caller's
    /// [`IncumbentValueFn`] priced, and how many of those it priced ABOVE `cᵀx`
    /// (the point's lifted objective was optimistic). Both 0 with no function.
    pub n_incumbent_value_calls: usize,
    /// #1656: [`IncumbentValueFn`] calls whose value exceeded `cᵀx` (see above).
    pub n_incumbent_value_raised: usize,
    /// #1913: integer branches chosen by pseudocost (`config.pseudocost_branching`);
    /// 0 when the flag is off -- the probe that the rule fired.
    pub n_pseudocost_picks: usize,
    /// #1970: spatial branches chosen by the candidate score
    /// (`config.spatial_score`); 0 when the flag is off.
    pub n_spatial_score_picks: usize,
    /// #1935: node LPs started from the parent's basis
    /// ([`SpatialTreeConfig::warm_start_children`]). 0 with the flag off, which is
    /// how a panel tells "the warm start never engaged" from "it engaged and did
    /// not help" (CLAUDE.md §6). Whether a warm start then stayed warm or fell back
    /// to a cold solve inside the simplex is `DualWarmSolves`/`DualColdFallbacks`
    /// under `DISCOPT_PROFILE`.
    pub n_warm_starts: usize,
    /// #1935: nodes that had a parent basis but were solved without it because
    /// the node LP's dimensions differ from the parent's.
    pub n_warm_shape_mismatch: usize,
    /// #1935: simplex pivots summed over node relaxation solves (OBBT probes
    /// excluded). Deterministic, unlike `node_lp_s`.
    pub n_node_pivots: usize,
    /// #1935: wall seconds spent inside the node solve (LP assembly, the LP, the
    /// safe bound and any OBBT sweep).
    pub node_lp_s: f64,
    /// #1962: certified convex quadratic forms the separator found in the spec
    /// (0 with separation off, where detection is skipped).
    pub n_cut_forms: usize,
    /// #1962: separation rounds that added at least one cut and re-solved the LP.
    pub n_cut_rounds: usize,
    /// #1962: cuts added across all rounds (inherited cuts are not recounted).
    pub n_cuts_added: usize,
    /// #1962: integral-point probes run ([`SpatialTreeConfig::int_fix_probe`]) and
    /// how many of them replaced the incumbent. Both 0 with the probe off.
    pub n_int_probes: usize,
    /// #1962: integral-point probes that replaced the incumbent (see above).
    pub n_int_probe_improvements: usize,
    /// #1995: undecided node LPs re-solved over their pre-propagation box
    /// ([`SpatialTreeConfig::undecided_unpropagated_retry`]), and how many of those
    /// re-solves decided (bounded or certified empty). Both 0 with the flag off.
    pub n_prop_retries: usize,
    /// #1995: pre-propagation re-solves that decided (see above).
    pub n_prop_retry_decided: usize,
}

/// One solved node, as a [`NodeObserver`] sees it (#1935 engine comparison).
pub struct NodeObservation<'a> {
    /// This node's id (the tree's `node_count` when it was solved; the root is 1).
    pub id: usize,
    /// The id of the node that pushed it; `None` at the root.
    pub parent: Option<usize>,
    /// The box the node LP was assembled over (after propagation).
    pub lo: &'a [f64],
    /// Upper side of that box.
    pub hi: &'a [f64],
    /// The basis the LP was started from, if any.
    pub start: Option<&'a Basis>,
    /// The node solve's result.
    pub result: &'a crate::bnb::spatial_kernel::SpatialNodeResult,
}

/// Called after every node LP with that node's [`NodeObservation`]. It observes
/// only: nothing it does reaches the search.
pub type NodeObserver<'a> = dyn FnMut(&NodeObservation<'_>) + 'a;

/// Where the primal hook runs on a model with integer columns (#1990,
/// `DISCOPT_HOOK_INTEGRAL_POINTS`, default ON; `=0` restores the #1522 schedule).
///
/// #1522 called the hook at nodes 1, 2, 4, 8, ... from whatever LP point the node
/// had, and the caller rounds the integer columns and fixes them. At a fractional
/// point that rounding is a guess, and a guess the local NLP usually cannot make
/// feasible: on `genpooling_lee1` all 10 calls (about 1 s each, a third of a 30 s
/// solve) found nothing, and the slowdown cost its certificate; on ex1252a the 8
/// calls at fractional points took 1-3 s each and failed, and the one that found
/// the optimum (0.06 s) was at a point integral on every integer column. ON, each
/// scheduled call is held until the next node whose LP point is integral on every
/// integer column with an integer assignment not yet tried, which is SCIP's
/// sub-NLP rule. The number of scheduled calls stays at most one per power of two;
/// a continuous model keeps the #1522 schedule, and a newly accepted incumbent is
/// still handed to the hook. Primal only.
fn hook_integral_points_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("DISCOPT_HOOK_INTEGRAL_POINTS")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .map(|v| !matches!(v.trim(), "0" | "false" | "False"))
            .unwrap_or(true)
    })
}

/// A caller-supplied primal heuristic for [`solve_spatial_tree_with_hook`] (#1522).
///
/// Called with a node's LP point (all `n_cols` columns) and the current incumbent
/// value; returns `Some((value, point))` ONLY for a point the caller has
/// independently VERIFIED feasible for the problem being solved, `value` being that
/// point's objective in the kernel's internal minimize units. The kernel adopts it
/// as the incumbent when it strictly improves, and records `point` as
/// `incumbent_x` (so `incumbent_x` may then be shorter than `n_cols`: it holds
/// whatever columns the caller's point carries).
///
/// Why the kernel needs one. Its only internal source of incumbents is an LP vertex
/// at which every lifted term is McCormick-tight to `mccormick_tol`. When the
/// optimum lies on a curved manifold (an equality whose lifted terms are tight only
/// on tiny boxes) that condition is met rarely and far from the optimum: on MINLPLib
/// `prob09` with no external seed, 100k nodes produced incumbents 2e-3 above an
/// optimum the root bound had already proven, so the gap could never close. One
/// local NLP from the first such vertex lands on the optimum.
///
/// Soundness. The value only prunes: the reported bound is still the min of the
/// closed regions' rigorous bounds, so a hook can never make the bound invalid —
/// but a value that no feasible point attains would prune regions that hold the
/// optimum, which is why the contract demands a verified point.
pub type PrimalHook<'a> = dyn FnMut(&[f64], Option<f64>) -> Option<(f64, Vec<f64>)> + 'a;

/// The caller's objective at a McCormick-tight LP point, for
/// [`solve_spatial_tree_with_hooks`] (#1656).
///
/// Called with the point (all `n_cols` columns) before the tree accepts it as the
/// incumbent; returns the objective, in the kernel's internal minimize units, of
/// the point the caller will PUBLISH from it, or `None` when that objective cannot
/// be evaluated (the point is then not accepted).
///
/// Why the kernel needs one. A point is accepted when every lifted term is tight
/// to `mccormick_tol`, but `cᵀx` is the true objective only when the terms are
/// exact: each term's slack composes through the lift. On `(xy-1)^2 + ...` over
/// `[-100, 100]^2` the lift `w = x^2 * y^2` reads `x^2` with slack 8.1e-7 and
/// multiplies it by `y^2 ~ 7.8`, so `cᵀx` sat 7.5e-6 BELOW the objective at the
/// point -- 2.7x the relative gap the caller asked for on an optimum of 0.0274. The
/// tree pruned and stopped against a value no point attains, and the published
/// pair (the objective AT the point) could not close.
///
/// The tree accepts at `max(cᵀx, value)`: never more optimistic than before, and
/// never more optimistic than the value the caller publishes.
///
/// Called with the point and its `cᵀx`; see [`IncumbentPrice`] for the answers.
pub type IncumbentValueFn<'a> = dyn FnMut(&[f64], f64) -> IncumbentPrice + 'a;

/// An [`IncumbentValueFn`]'s answer for one McCormick-tight point (#1656).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IncumbentPrice {
    /// The caller's objective at the point, in internal minimize units.
    Value(f64),
    /// The objective cannot be evaluated there: the point is not accepted.
    Reject,
    /// The point's `cᵀx` and the caller's objective disagree by more than lifted
    /// slack can explain, so the lift is not the caller's problem (MINLPLib
    /// `prob10`: `cᵀx = 2.345` at a point whose objective is 3.446, a lift
    /// strictly looser than the model). The search stops with
    /// [`TreeStatus::Declined`]: pricing alone would keep it honest but not
    /// finished, since its bound can never pass the lift's own optimum and it
    /// would run to its node or time limit.
    Decline,
}

/// True value of a lifted term at the point `x` (structural columns), for the
/// McCormick-exactness feasibility test. `None` for a sqrt of a negative argument
/// (infeasible point — never accepted).
fn term_true_value(t: &EnvTerm, x: &[f64]) -> Option<f64> {
    Some(match *t {
        EnvTerm::Bilinear { i, j, .. } => x[i] * x[j],
        EnvTerm::Monomial { i, p, .. } => x[i].powi(p),
        EnvTerm::AffineSquare { j, coeff, cst, .. } => {
            let t = coeff * x[j] + cst;
            t * t
        }
        EnvTerm::Sqrt {
            x: xc, coeff, cst, ..
        } => {
            let arg = coeff * x[xc] + cst;
            if arg < 0.0 {
                return None;
            }
            arg.sqrt()
        }
    })
}

/// The operand column a spatial branch on this term should split, and the term's
/// current McCormick gap `|x_aux − f(operands)|`. For a bilinear term we split the
/// operand with the wider box (passed in via `width`); for the 1-D terms the sole
/// operand.
fn term_gap_and_branch_col(t: &EnvTerm, x: &[f64], width: &dyn Fn(usize) -> f64) -> (f64, usize) {
    let aux = t.aux_col();
    let true_v = term_true_value(t, x);
    let gap = match true_v {
        Some(v) => (x[aux] - v).abs(),
        None => f64::INFINITY, // infeasible operand: force a branch here
    };
    let col = match *t {
        EnvTerm::Bilinear { i, j, .. } => {
            if width(i) >= width(j) {
                i
            } else {
                j
            }
        }
        EnvTerm::Monomial { i, .. } => i,
        EnvTerm::AffineSquare { j, .. } => j,
        EnvTerm::Sqrt { x: xc, .. } => xc,
    };
    (gap, col)
}

/// The [`SimplexOptions`] a node LP runs under, given the caller's base options
/// and the tree's *live* deadline (#1009).
///
/// The tree checks its clock only BETWEEN nodes, so without this the wall-clock
/// budget bounds nothing: one pathological node LP runs uninterruptibly to
/// `max_iter` and the whole search overruns by an unbounded margin. Measured on
/// QPLIB_1157 with `DISCOPT_RLT_LINEQ=1` and a 20 s `time_limit`: **240.83 s on
/// node 1**, 11.9x over budget, because the root LP alone could not be stopped.
///
/// Takes the EARLIER of the two when both are set: the caller's own per-LP cap
/// (e.g. the `DISCOPT_LP_WARM_DEADLINE` path) is a cap on a single solve and the
/// tree's is a cap on the whole search — honoring the later of them would let one
/// of the two budgets be silently ignored, which is the bug being fixed.
///
/// Sound by construction: a deadline bail is reported as [`LpStatus::IterLimit`],
/// which the loop below routes to `NodeVerdict::Undecided` — the region is
/// BRANCHED with the parent's (valid) bound inherited, never fathomed, and
/// `n_uncertified`/`n_undecided` record it. Cutting an LP short can therefore
/// only make the reported bound looser, never wrong. When `config.deadline` is
/// `None` and the caller set none either, this is `base` unchanged and the solve
/// is bit-identical to before.
fn node_lp_opts(base: &SimplexOptions, live_deadline: Option<Instant>) -> SimplexOptions {
    let deadline = match (base.deadline, live_deadline) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    };
    SimplexOptions {
        deadline,
        ..base.clone()
    }
}

/// Solve `spec` by native spatial branch-and-bound. `spec.global_lo/global_hi` is
/// the root box; `spec.integrality` marks integer columns; `spec.obbt_candidates`
/// (if `config.run_obbt`) are probed per node.
pub fn solve_spatial_tree(
    spec: &SpatialKernelSpec,
    config: &SpatialTreeConfig,
    opts: &SimplexOptions,
) -> SpatialTreeResult {
    solve_spatial_tree_with_hook(spec, config, opts, None)
}

/// [`solve_spatial_tree`] with an optional caller primal heuristic (#1522); see
/// [`PrimalHook`]. It is consulted when the tree accepts a new McCormick-tight
/// incumbent and at nodes 1, 2, 4, 8, ... (a geometric schedule, so the calls grow
/// with the logarithm of the node count). `None` is exactly [`solve_spatial_tree`].
pub fn solve_spatial_tree_with_hook(
    spec: &SpatialKernelSpec,
    config: &SpatialTreeConfig,
    opts: &SimplexOptions,
    hook: Option<&mut PrimalHook<'_>>,
) -> SpatialTreeResult {
    solve_spatial_tree_with_hooks(spec, config, opts, hook, None)
}

/// [`solve_spatial_tree_with_hook`] with an optional [`IncumbentValueFn`] (#1656)
/// pricing every McCormick-tight LP point before it is accepted. `None` is exactly
/// [`solve_spatial_tree_with_hook`].
pub fn solve_spatial_tree_with_hooks(
    spec: &SpatialKernelSpec,
    config: &SpatialTreeConfig,
    opts: &SimplexOptions,
    hook: Option<&mut PrimalHook<'_>>,
    value_fn: Option<&mut IncumbentValueFn<'_>>,
) -> SpatialTreeResult {
    solve_spatial_tree_observed(spec, config, opts, hook, value_fn, None)
}

/// [`solve_spatial_tree_with_hooks`] that also reports every solved node to
/// `observer` (#1935: capturing the node-LP sequence to replay it through another
/// LP engine). `None` is exactly [`solve_spatial_tree_with_hooks`].
pub fn solve_spatial_tree_observed(
    spec: &SpatialKernelSpec,
    config: &SpatialTreeConfig,
    opts: &SimplexOptions,
    mut hook: Option<&mut PrimalHook<'_>>,
    mut value_fn: Option<&mut IncumbentValueFn<'_>>,
    mut observer: Option<&mut NodeObserver<'_>>,
) -> SpatialTreeResult {
    let mut n_primal_hook_calls = 0usize;
    let mut n_primal_hook_improvements = 0usize;
    // #1990: the hook's integral-point schedule (`hook_integral_points_enabled`).
    let hook_has_int = spec.integrality[..spec.n_orig].iter().any(|&b| b);
    let mut hook_armed = false;
    let mut hook_tried: std::collections::HashSet<Vec<i64>> = std::collections::HashSet::new();
    let mut n_incumbent_value_calls = 0usize;
    let mut n_incumbent_value_raised = 0usize;
    let mut n_pseudocost_picks = 0usize;
    let mut n_spatial_score_picks = 0usize;
    let mut n_warm_starts = 0usize;
    let mut n_warm_shape_mismatch = 0usize;
    let mut n_node_pivots = 0usize;
    let mut node_lp_s = 0.0f64;
    // #1962: the separator's certified convex quadratic forms (none unless
    // separation is configured, so the default path never runs detection).
    let cut_forms: Vec<ConvexQuadForm> = if config.cut_rounds_root > 0 || config.cut_rounds_node > 0
    {
        detect_convex_quad_forms(spec)
    } else {
        Vec::new()
    };
    let mut n_cut_rounds = 0usize;
    let mut n_cuts_added = 0usize;
    // #1962: the integer assignments the integral-point probe has tried.
    let mut int_probed: std::collections::HashSet<Vec<i64>> = std::collections::HashSet::new();
    let mut n_int_probes = 0usize;
    let mut n_int_probe_improvements = 0usize;
    // #1995: pre-propagation re-solves of undecided node LPs (probe counters).
    let mut n_prop_retries = 0usize;
    let mut n_prop_retry_decided = 0usize;
    // Best-bound frontier: a min-heap on the inherited lower bound. Exploring the
    // lowest-bound region first lifts the global frontier minimum (the reported dual
    // bound) as fast as possible — the key to certifying instances like tanksize
    // whose bound climbs only slowly.
    let mut heap: BinaryHeap<QNode> = BinaryHeap::new();
    heap.push(QNode {
        pb: f64::NEG_INFINITY,
        lo: spec.global_lo.clone(),
        hi: spec.global_hi.clone(),
        br: None,
        sbr: None,
        basis: None,
        parent: None,
        cuts: Rc::new(Vec::new()),
    });
    // #1913: per-column pseudocosts, `[down, up]` sums of per-unit bound gains and
    // their observation counts. Read only under `config.pseudocost_branching`.
    let mut pc_sum = [vec![0.0f64; spec.n_cols], vec![0.0f64; spec.n_cols]];
    let mut pc_cnt = [vec![0usize; spec.n_cols], vec![0usize; spec.n_cols]];
    // #1970: spatial pseudocosts, the same layout, per unit of width cut off.
    let mut spc_sum = [vec![0.0f64; spec.n_cols], vec![0.0f64; spec.n_cols]];
    let mut spc_cnt = [vec![0usize; spec.n_cols], vec![0usize; spec.n_cols]];
    // Root box widths (for root-relative branch scoring in the fallback rule).
    let root_w: Vec<f64> = spec
        .global_lo
        .iter()
        .zip(spec.global_hi.iter())
        .map(|(l, h)| (h - l).max(1e-12))
        .collect();
    // An externally-supplied feasible value seeds the incumbent (empty point —
    // the caller holds it); it prunes and cutoff-propagates like any incumbent.
    let mut incumbent: Option<f64> = config.initial_incumbent;
    let mut incumbent_x: Vec<f64> = Vec::new();
    // #917: the live deadline, which the incumbent-conditional extension below may
    // push out ONCE. Kept local because `config` is `Copy` and shared.
    let mut deadline = config.deadline;
    let mut extension_taken = false;
    let mut extension_s = 0.0f64;
    let mut bound_extension_taken = false;
    let mut bound_extension_s = 0.0f64;
    let mut node_count = 0usize;
    let mut n_lp_solves = 0usize;
    let mut n_uncertified = 0usize;
    let mut n_undecided = 0usize;
    // #1236: the root region's proven lower bound, and when it was proven. Set at
    // whichever of the four arms node 1 leaves by (propagation-fathom, certified-
    // empty, undecided-LP, bounded), and read only when the result is built -- the
    // search never branches on either, which is what keeps this bound-neutral.
    let mut root_bound = f64::NEG_INFINITY;
    let mut root_time_s = 0.0f64;
    // Stays "not_reached" unless node 1 actually finishes; each arm below names
    // itself, so a non-finite `root_bound` is never ambiguous downstream.
    let mut root_status: &'static str = "not_reached";
    let t_tree_start = Instant::now();

    // Global lower bound = min, over every region that leaves the tree WITHOUT being
    // subdivided (pruned / infeasible / feasible-leaf / width-exhausted), of a valid
    // lower bound for that region. Each contribution is a safe bound (`<=` the
    // region's true optimum), so the accumulated min is `<=` the true global optimum
    // — a rigorous lower bound. Branched regions contribute nothing (their children
    // do). Open frontier nodes carry their `pb` as a valid region lower bound; under
    // best-bound the heap top is exactly that frontier minimum.
    let mut global_lb_closed = f64::INFINITY;

    while let Some(QNode {
        pb: parent_bound,
        lo,
        hi,
        br,
        sbr,
        basis: parent_basis,
        parent: parent_id,
        cuts: inherited_cuts,
    }) = heap.pop()
    {
        if deadline.is_some_and(|d| Instant::now() >= d) && !extension_taken && incumbent.is_some()
        {
            // #917: reclaim the caller's withheld #844 reserve, once, and only while
            // this search already holds an incumbent — the one state in which that
            // fallback provably has nothing to contribute. Purely additive: with no
            // extension configured, or with no incumbent, the exit below is the
            // pre-#917 one.
            //
            // This is the kernel's own reclaim point. The Python node loops have one
            // (`_extend_budget_for_incumbent`) but this kernel is a single
            // uninterruptible call that never enters them, so before this a
            // kernel-routed solve forfeited the reserve outright: nvs17 at a 60 s
            // budget stopped at 39.4 s with its bound 4.4% short of its own incumbent.
            if let (Some(d), Some(ext)) = (deadline, config.incumbent_time_extension) {
                deadline = Some(d + ext);
                extension_taken = true;
                extension_s = ext.as_secs_f64();
            }
        }
        if deadline.is_some_and(|d| Instant::now() >= d)
            && !bound_extension_taken
            && config.bound_time_extension.is_some()
        {
            // #933: the caller shortened this kernel's deadline by the reserve it
            // withholds for its root-relaxation fallback. If the global bound is
            // already finite, that fallback has nothing to contribute (the
            // kernel's frontier bound is at least as tight as any root bound), so
            // reclaim the slice — once — and keep searching. If the bound is
            // still -inf, fall through to the exit below with the reserve
            // unspent, so the caller can still prove a root bound inside the
            // original budget instead of reporting none at all.
            let frontier = heap.iter().map(|n| n.pb).fold(f64::INFINITY, f64::min);
            let gb = global_lb_closed.min(frontier).min(parent_bound);
            if gb.is_finite() {
                if let (Some(d), Some(ext)) = (deadline, config.bound_time_extension) {
                    deadline = Some(d + ext);
                    bound_extension_taken = true;
                    bound_extension_s = ext.as_secs_f64();
                }
            }
        }
        if deadline.is_some_and(|d| Instant::now() >= d) {
            // Global bound = min(closed regions, open frontier, this unprocessed
            // node). Every term is rigorous for its region, so the partial result
            // remains an honest certificate even though the gap is still open.
            let frontier = heap.iter().map(|n| n.pb).fold(f64::INFINITY, f64::min);
            let gb = global_lb_closed.min(frontier).min(parent_bound);
            return SpatialTreeResult {
                status: TreeStatus::TimeLimit,
                incumbent,
                incumbent_x,
                bound: gb,
                node_count,
                n_lp_solves,
                n_uncertified,
                n_undecided,
                incumbent_extension_s: extension_s,
                bound_extension_s,
                root_bound,
                root_time_s,
                root_status,
                n_primal_hook_calls,
                n_primal_hook_improvements,
                n_incumbent_value_calls,
                n_incumbent_value_raised,
                n_pseudocost_picks,
                n_spatial_score_picks,
                n_warm_starts,
                n_warm_shape_mismatch,
                n_node_pivots,
                node_lp_s,
                n_cut_forms: cut_forms.len(),
                n_cut_rounds,
                n_cuts_added,
                n_int_probes,
                n_int_probe_improvements,
                n_prop_retries,
                n_prop_retry_decided,
            };
        }
        // Fathom by the parent bound if the incumbent already dominates it. The
        // region's valid lower bound is `parent_bound`.
        if let Some(inc) = incumbent {
            if gap_closed(parent_bound, inc, config) {
                global_lb_closed = global_lb_closed.min(parent_bound);
                continue;
            }
        }
        if node_count >= config.max_nodes {
            // Global bound = min(closed regions, open frontier, this unpopped node).
            // Under best-bound the heap top is the frontier minimum, but take the full
            // min defensively.
            let frontier = heap.iter().map(|n| n.pb).fold(f64::INFINITY, f64::min);
            let gb = global_lb_closed.min(frontier).min(parent_bound);
            return SpatialTreeResult {
                status: TreeStatus::NodeLimit,
                incumbent,
                incumbent_x,
                bound: gb,
                node_count,
                n_lp_solves,
                n_uncertified,
                n_undecided,
                incumbent_extension_s: extension_s,
                bound_extension_s,
                root_bound,
                root_time_s,
                root_status,
                n_primal_hook_calls,
                n_primal_hook_improvements,
                n_incumbent_value_calls,
                n_incumbent_value_raised,
                n_pseudocost_picks,
                n_spatial_score_picks,
                n_warm_starts,
                n_warm_shape_mismatch,
                n_node_pivots,
                node_lp_s,
                n_cut_forms: cut_forms.len(),
                n_cut_rounds,
                n_cuts_added,
                n_int_probes,
                n_int_probe_improvements,
                n_prop_retries,
                n_prop_retry_decided,
            };
        }
        node_count += 1;

        // FBBT fixpoint propagation BEFORE the LP (C2, entry experiment GO
        // 2026-07-19): linear rows + products (extended division) + 1-D terms +
        // integer rounding + objective cutoff, zero LP solves. Tightens the box the
        // LP and the children see; a box proven empty under the cutoff is fathomed
        // with region lower bound `incumbent` (no feasible point beats it there),
        // or `+inf` when no cutoff was active (genuinely empty region).
        let mut lo = lo;
        let mut hi = hi;
        // #1995: the box as the node was pushed, kept only under the retry flag.
        let mut pre_propagation = (config.undecided_unpropagated_retry && config.run_propagation)
            .then(|| (lo.clone(), hi.clone()));
        if config.run_propagation
            && !propagate_spec_fixpoint(
                spec,
                &mut lo,
                &mut hi,
                incumbent,
                config.propagation_rounds,
            )
        {
            let contrib = incumbent.unwrap_or(f64::INFINITY).max(parent_bound);
            global_lb_closed = global_lb_closed.min(contrib);
            if node_count == 1 {
                root_bound = contrib;
                root_time_s = t_tree_start.elapsed().as_secs_f64();
                root_status = "propagation_fathom";
            }
            continue;
        }

        // #1009: hand the node LP the tree's LIVE deadline (post-extension), not
        // `config.deadline` — an extension taken above must reach the LP too, or
        // the extra slice the tree just granted itself is spent on an LP that
        // still cannot be interrupted. See `node_lp_opts`.
        let node_opts = node_lp_opts(opts, deadline);
        let t_node = Instant::now();
        let mut node = solve_spatial_node_with_cuts(
            spec,
            &lo,
            &hi,
            config.run_obbt,
            &node_opts,
            parent_basis.as_deref(),
            &inherited_cuts,
        );
        // #1962: the separation loop. Every LP solved here is a relaxation of the
        // node region (each cut holds at every feasible point), so each one's safe
        // bound is valid and the node keeps the best of them; a round whose LP
        // decides nothing is dropped and the node keeps the last decided result.
        let rounds = if node_count == 1 {
            config.cut_rounds_root
        } else {
            config.cut_rounds_node
        };
        let mut node_cuts: Rc<Vec<FixedRow>> = inherited_cuts.clone();
        if rounds > 0 && !cut_forms.is_empty() {
            let (cuts, rounds_done, added) = separation_loop(
                spec,
                (&lo, &hi),
                config,
                &node_opts,
                &cut_forms,
                rounds,
                &mut node,
                &inherited_cuts,
                &mut n_lp_solves,
                &mut n_node_pivots,
            );
            node_cuts = cuts;
            n_cut_rounds += rounds_done;
            n_cuts_added += added;
        }
        // #1995: an undecided LP over a box propagation tightened is re-solved over
        // the pre-propagation box (see `SpatialTreeConfig::undecided_unpropagated_retry`
        // for why that box's verdict holds for this region). A decided re-solve
        // replaces the node -- box included, so the children split the box the LP
        // was solved over -- and the first attempt's work is still counted.
        if verdict_for(node.status) == NodeVerdict::Undecided {
            if let Some((plo, phi)) = pre_propagation.take().filter(|(l, h)| *l != lo || *h != hi) {
                n_prop_retries += 1;
                n_lp_solves += node.n_lp_solves;
                n_node_pivots += node.n_pivots;
                let retry = solve_spatial_node_with_cuts(
                    spec,
                    &plo,
                    &phi,
                    config.run_obbt,
                    &node_opts,
                    parent_basis.as_deref(),
                    &inherited_cuts,
                );
                if verdict_for(retry.status) != NodeVerdict::Undecided {
                    n_prop_retry_decided += 1;
                    node = retry;
                    lo = plo;
                    hi = phi;
                    node_cuts = inherited_cuts.clone();
                } else {
                    n_lp_solves += retry.n_lp_solves;
                    n_node_pivots += retry.n_pivots;
                }
            }
        }
        let node_time = t_node.elapsed();
        node_lp_s += node_time.as_secs_f64();
        n_lp_solves += node.n_lp_solves;
        n_node_pivots += node.n_pivots;
        n_warm_starts += usize::from(node.warm_started);
        n_warm_shape_mismatch += usize::from(node.warm_shape_mismatch);
        if let Some(obs) = observer.as_mut() {
            obs(&NodeObservation {
                id: node_count,
                parent: parent_id,
                lo: &lo,
                hi: &hi,
                start: parent_basis.as_deref(),
                result: &node,
            });
        }
        if node.status == LpStatus::Optimal && node.bound == f64::NEG_INFINITY {
            n_uncertified += 1;
        }

        let verdict = verdict_for(node.status);
        if verdict != NodeVerdict::Bound {
            // Certified-infeasible node: empty region, contributes +inf (nothing).
            if verdict == NodeVerdict::EmptyRegion {
                if node_count == 1 {
                    root_bound = f64::INFINITY;
                    root_time_s = t_tree_start.elapsed().as_secs_f64();
                    root_status = "infeasible";
                }
                continue;
            }
            // #927: every OTHER non-Optimal status (`Numerical`, `IterLimit`,
            // `Unbounded`) is the LP *failing to decide*, NOT a proof that the
            // region is empty — and the simplex contract is explicit that an
            // unproven verdict may "force a fallback, never an unsound fathom".
            // Treating them as empty is a false emptiness proof: on ex1252 under
            // `DISCOPT_INTEGER_MULTILINEAR_REFORM` + `DISCOPT_MULTILINEAR_COUPLING_RLT`
            // the reformulation lifts x^3 of a variable near 2950, so the envelope
            // rows carry magnitudes near 1e11 and the node LP on a tight box came
            // back `Numerical`. The region it covered held the true optimum
            // (128893.74); fathoming it produced a certified `optimal` at
            // 216826.52 — a false certificate.
            //
            // An undecided LP is a reason to BRANCH, not to prune: the children
            // are better conditioned (narrower boxes → smaller envelope
            // coefficients) and each carries `parent_bound`, which is a valid lower
            // bound for this region because the parent's region contains it. When
            // no branchable column is left, the region closes with that same honest
            // bound rather than with `+inf`.
            n_undecided += 1;
            if node_count == 1 {
                root_bound = parent_bound;
                root_time_s = t_tree_start.elapsed().as_secs_f64();
                root_status = "undecided";
            }
            let split = widest_original_col(spec, &lo, &hi, &root_w, config.min_box_width);
            match split {
                Some(j) => {
                    let at = clamp_interior(0.5 * (lo[j] + hi[j]), lo[j], hi[j]);
                    // An undecided LP has no optimal basis of its own, so the
                    // children inherit the one THIS node was started from (its
                    // nearest solved ancestor's). Handing down nothing made the
                    // whole subtree cold, and cold solves are what fail: on
                    // MINLPLib `hybriddynamic_var` 85% of cold node LPs end
                    // `Numerical` against 1% of warm ones, so one undecided node
                    // turned its subtree undecided (#1935). The children's boxes
                    // lie inside that ancestor's, so the basis fits their LP; it
                    // is a start only, never a verdict. `None` with the flag off.
                    push_children(
                        &mut heap,
                        &lo,
                        &hi,
                        j,
                        at,
                        parent_bound,
                        spec.integrality[j],
                        None,
                        parent_basis.clone(),
                        Some(node_count),
                        inherited_cuts.clone(),
                        false,
                    );
                }
                None => global_lb_closed = global_lb_closed.min(parent_bound),
            }
            continue;
        }
        // Rigorous safe lower bound for this region, inheriting the parent's bound as
        // a floor: `parent_bound` is a valid lower bound for the parent region, which
        // CONTAINS this child, so the child's region optimum is `>= parent_bound`.
        // Taking the max keeps the bound finite and monotone when the node's own safe
        // bound is looser or uncertifiable (`-inf` — e.g. non-finite duals from an
        // ill-conditioned McCormick LP on tanksize), which would otherwise poison the
        // global lower bound. Sound: `max(safe, parent)` is still `<=` the true region
        // optimum since both terms are.
        let bound = node.bound.max(parent_bound);
        if let Some((col, up, dist)) = br {
            if node.bound.is_finite() && parent_bound.is_finite() && dist > 0.0 {
                let d = usize::from(up);
                pc_sum[d][col] += (node.bound - parent_bound).max(0.0) / dist;
                pc_cnt[d][col] += 1;
            }
        }
        if let Some((col, up, dist)) = sbr {
            if node.bound.is_finite() && parent_bound.is_finite() && dist > 0.0 {
                let d = usize::from(up);
                spc_sum[d][col] += (node.bound - parent_bound).max(0.0) / dist;
                spc_cnt[d][col] += 1;
            }
        }
        // #1236: node 1 IS the root region, and `bound` is the rigorous lower bound
        // just proven for it — the same value that reaches `global_lb_closed` or the
        // children's `pb`. Recorded here (and at the three other arms node 1 can
        // leave by) so the caller can tell a loose root relaxation from a tight one,
        // which the final `bound` alone cannot say.
        if node_count == 1 {
            root_bound = bound;
            root_time_s = t_tree_start.elapsed().as_secs_f64();
            root_status = "bounded";
        }
        // Fathom by bound vs incumbent. The region's valid lower bound is `bound`.
        if let Some(inc) = incumbent {
            if gap_closed(bound, inc, config) {
                global_lb_closed = global_lb_closed.min(bound);
                continue;
            }
        }

        // Apply OBBT-tightened bounds to the box (tighten-only, sound).
        if config.run_obbt {
            for (k, &cand) in spec.obbt_candidates.iter().enumerate() {
                if k < node.tightened.len() {
                    let (glo, ghi) = node.tightened[k];
                    lo[cand] = lo[cand].max(glo);
                    hi[cand] = hi[cand].min(ghi);
                }
            }
        }

        let x = &node.x;

        // --- Feasibility test (sufficient condition for a valid incumbent) --- //
        // (a) integer candidates integral.
        let mut int_ok = true;
        let mut frac_int: Option<usize> = None;
        for (j, &is_int) in spec.integrality.iter().enumerate().take(spec.n_cols) {
            if is_int && (x[j] - x[j].round()).abs() > config.int_tol {
                int_ok = false;
                frac_int = Some(j);
                break;
            }
        }
        if config.pseudocost_branching && frac_int.is_some() {
            // Strong branching amortises only over a long tree, so skip the pass
            // unless ~512 more nodes fit in the time left -- judged both at the
            // tree's measured wall time per node (which includes the propagation and
            // primal-hook work around the LP, and earlier passes) and at the pass's
            // own worst case, two child LPs per candidate at this node's solve time.
            // graphpart_clique-40 at 120 s spends ~0.7 s per node mostly outside the
            // LP: an LP-time-only test let the pass run and halved its nodes (166 ->
            // 98) for a looser bound. A global LP-count or wall-time quota instead
            // throttled the cheap-node instances that gain from it (st_e31,
            // genpooling_meyer04).
            let affordable = deadline.is_none_or(|d| {
                let left = d.saturating_duration_since(Instant::now());
                let per_node = t_tree_start.elapsed() / node_count.max(1) as u32;
                per_node * 512 <= left
                    && node_time * (2 * STRONG_BRANCH_CANDIDATES as u32) <= left / 32
            });
            if config.pseudocost_reliability > 0 && bound.is_finite() && affordable {
                n_lp_solves += strong_branch_init(
                    spec,
                    x,
                    (&lo, &hi),
                    bound,
                    incumbent,
                    config,
                    &node_opts,
                    (&mut pc_sum, &mut pc_cnt),
                    &node_cuts,
                );
            }
            frac_int = pseudocost_pick(spec, x, config.int_tol, &pc_sum, &pc_cnt);
            n_pseudocost_picks += 1;
        }
        // (b) every lifted term McCormick-tight — fixed-width EnvTerms and the
        //     affine-form product (BlfTerm) terms alike.
        let width = |c: usize| hi[c] - lo[c];
        // #1619 D-11: the widths and gaps the branch CHOICE compares. Scale-free,
        // both are relative to the column's root width; otherwise absolute.
        let choice_width = |c: usize| {
            if config.scale_free_branching {
                width(c) / root_w[c]
            } else {
                width(c)
            }
        };
        let choice_gap = |gap: f64, aux: usize| {
            if config.scale_free_branching {
                gap / root_w[aux]
            } else {
                gap
            }
        };
        // `worst_gap` (absolute) decides McCormick-tightness; `worst_score` picks
        // the term to branch. They coincide unless `scale_free_branching`.
        let mut worst_gap = 0.0f64;
        let mut worst_score = 0.0f64;
        let mut branch_col: Option<usize> = None;
        for t in &spec.terms {
            let (gap, col) = term_gap_and_branch_col(t, x, &choice_width);
            worst_gap = worst_gap.max(gap);
            let score = choice_gap(gap, t.aux_col());
            if score > worst_score {
                worst_score = score;
                branch_col = Some(col);
            }
        }
        for t in &spec.blf_terms {
            let a_val = t.a_const + dot_form(&t.a_cols, &t.a_coeffs, x);
            let b_val = t.b_const + dot_form(&t.b_cols, &t.b_coeffs, x);
            let gap = (x[t.w] - a_val * b_val).abs();
            worst_gap = worst_gap.max(gap);
            let score = choice_gap(gap, t.w);
            if score > worst_score {
                worst_score = score;
                // Spatial-branch the widest operand column across A ∪ B.
                let mut best = None;
                let mut best_w = -1.0f64;
                for &c in t.a_cols.iter().chain(t.b_cols.iter()) {
                    let cw = choice_width(c);
                    if cw > best_w {
                        best_w = cw;
                        best = Some(c);
                    }
                }
                branch_col = best;
            }
        }
        let terms_tight = worst_gap <= config.mccormick_tol;

        let mut accepted_lp_point = false;
        // #1938: the caller's pricer rejected this point (see below).
        let mut rejected_lp_point = false;
        if int_ok && terms_tight {
            // Feasible point: accept if it improves the incumbent. The objective is
            // linear over the (now-tight) lifted columns, so `cᵀx` is the objective
            // at this point up to the terms' composed slack; the caller's
            // `value_fn` prices that slack in (#1656, see [`IncumbentValueFn`]).
            let lp_obj = dot(&spec.c, x);
            if incumbent.map(|inc| lp_obj < inc - 1e-12).unwrap_or(true) {
                let priced = match value_fn.as_mut() {
                    None => Some(lp_obj),
                    Some(f) => {
                        n_incumbent_value_calls += 1;
                        match f(x, lp_obj) {
                            // NaN fails `is_finite` and rejects the point.
                            IncumbentPrice::Value(v) if v.is_finite() => {
                                if v > lp_obj {
                                    n_incumbent_value_raised += 1;
                                }
                                Some(v.max(lp_obj))
                            }
                            IncumbentPrice::Value(_) | IncumbentPrice::Reject => None,
                            IncumbentPrice::Decline => {
                                let frontier =
                                    heap.iter().map(|n| n.pb).fold(f64::INFINITY, f64::min);
                                return SpatialTreeResult {
                                    status: TreeStatus::Declined,
                                    incumbent,
                                    incumbent_x,
                                    bound: global_lb_closed.min(frontier).min(bound),
                                    node_count,
                                    n_lp_solves,
                                    n_uncertified,
                                    n_undecided,
                                    incumbent_extension_s: extension_s,
                                    bound_extension_s,
                                    root_bound,
                                    root_time_s,
                                    root_status,
                                    n_primal_hook_calls,
                                    n_primal_hook_improvements,
                                    n_incumbent_value_calls,
                                    n_incumbent_value_raised,
                                    n_pseudocost_picks,
                                    n_spatial_score_picks,
                                    n_warm_starts,
                                    n_warm_shape_mismatch,
                                    n_node_pivots,
                                    node_lp_s,
                                    n_cut_forms: cut_forms.len(),
                                    n_cut_rounds,
                                    n_cuts_added,
                                    n_int_probes,
                                    n_int_probe_improvements,
                                    n_prop_retries,
                                    n_prop_retry_decided,
                                };
                            }
                        }
                    }
                };
                rejected_lp_point = priced.is_none();
                if let Some(obj) = priced {
                    if incumbent.map(|inc| obj < inc - 1e-12).unwrap_or(true) {
                        incumbent = Some(obj);
                        incumbent_x = x[..spec.n_cols].to_vec();
                        accepted_lp_point = true;
                    }
                }
            }
        }
        // #1962: the integral-point probe. The LP point is integral but some lifted
        // term is not tight, so it cannot be accepted; fix every integer original
        // column at its LP value and re-solve over that sub-box (once per integer
        // assignment). When every term reads only integer columns the fixed box pins
        // them all, so the probe's LP point is McCormick-tight and is judged by the
        // SAME acceptance test and pricing as any LP point. Primal only: nothing it
        // does reaches a bound, and a probe that does not decide is ignored.
        if config.int_fix_probe
            && int_ok
            && !terms_tight
            && !accepted_lp_point
            && !rejected_lp_point
            && spec.integrality[..spec.n_orig].iter().any(|&b| b)
        {
            let key: Vec<i64> = (0..spec.n_orig)
                .filter(|&j| spec.integrality[j])
                .map(|j| x[j].round() as i64)
                .collect();
            if int_probed.insert(key) {
                n_int_probes += 1;
                let mut l2 = lo.clone();
                let mut h2 = hi.clone();
                for j in 0..spec.n_orig {
                    if spec.integrality[j] {
                        let v = x[j].round().clamp(lo[j], hi[j]);
                        l2[j] = v;
                        h2[j] = v;
                    }
                }
                let live = !config.run_propagation
                    || propagate_spec_fixpoint(
                        spec,
                        &mut l2,
                        &mut h2,
                        incumbent,
                        config.propagation_rounds,
                    );
                if live {
                    let probe = solve_spatial_node_with_cuts(
                        spec, &l2, &h2, false, &node_opts, None, &node_cuts,
                    );
                    n_lp_solves += probe.n_lp_solves;
                    let px = &probe.x;
                    let tight = probe.status == LpStatus::Optimal
                        && (0..spec.n_cols).all(|j| {
                            !spec.integrality[j] || (px[j] - px[j].round()).abs() <= config.int_tol
                        })
                        && max_term_gap(spec, px) <= config.mccormick_tol;
                    if tight {
                        let lp_obj = dot(&spec.c, px);
                        if incumbent.map(|inc| lp_obj < inc - 1e-12).unwrap_or(true) {
                            // A `Decline` is left to the main path to act on: the
                            // probe only ever declines to accept.
                            let priced = match value_fn.as_mut() {
                                None => Some(lp_obj),
                                Some(f) => {
                                    n_incumbent_value_calls += 1;
                                    match f(px, lp_obj) {
                                        IncumbentPrice::Value(v) if v.is_finite() => {
                                            if v > lp_obj {
                                                n_incumbent_value_raised += 1;
                                            }
                                            Some(v.max(lp_obj))
                                        }
                                        _ => None,
                                    }
                                }
                            };
                            if let Some(obj) = priced {
                                if incumbent.map(|inc| obj < inc - 1e-12).unwrap_or(true) {
                                    incumbent = Some(obj);
                                    incumbent_x = px[..spec.n_cols].to_vec();
                                    accepted_lp_point = true;
                                    n_int_probe_improvements += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        // #1522: the caller's local primal step, from this node's LP point. Adopted
        // only on strict improvement; the caller has verified the point.
        if let Some(h) = hook.as_mut() {
            // #1990: on a model with integer columns, a scheduled call waits for an
            // LP point integral on every integer column whose assignment has not
            // been tried (see `hook_integral_points_enabled`).
            if node_count.is_power_of_two() {
                hook_armed = true;
            }
            let scheduled = if hook_has_int && hook_integral_points_enabled() {
                hook_armed
                    && (0..spec.n_orig).all(|j| {
                        !spec.integrality[j] || (x[j] - x[j].round()).abs() <= config.int_tol
                    })
                    && hook_tried.insert(
                        (0..spec.n_orig)
                            .filter(|&j| spec.integrality[j])
                            .map(|j| x[j].round() as i64)
                            .collect::<Vec<i64>>(),
                    )
            } else {
                node_count.is_power_of_two()
            };
            if scheduled {
                hook_armed = false;
            }
            if accepted_lp_point || scheduled {
                n_primal_hook_calls += 1;
                if let Some((v, xp)) = h(x, incumbent) {
                    if v.is_finite() && incumbent.map(|inc| v < inc - 1e-12).unwrap_or(true) {
                        incumbent = Some(v);
                        incumbent_x = xp;
                        n_primal_hook_improvements += 1;
                    }
                }
            }
        }
        // CLOSE the region ONLY when its rigorous bound certifies that no point in
        // it beats the incumbent by more than the gap. A feasible point does NOT
        // prove the region optimal — with a loose bound the region may hold BETTER
        // points, so it must be branched further (closing here would be a premature
        // fathom → a false certificate). The incumbent can only have changed since
        // the bound-vs-incumbent fathom above through the two updates just made, so
        // with no hook this fires exactly where it did when it sat inside the
        // feasible branch.
        if let Some(inc_now) = incumbent {
            if gap_closed(bound, inc_now, config) {
                global_lb_closed = global_lb_closed.min(bound);
                continue;
            }
        }
        // fall through to branching (branch_col may be None: all terms tight — the
        // widest-column fallback below picks the split).

        // --- Branch --- //
        // Prefer closing an integer infeasibility; else spatial-branch the worst
        // McCormick gap; else (all terms tight but the bound uncertified) the widest
        // root-relative original column. A region with no branchable column left is
        // closed with its honest rigorous `bound` (surfaced as `Exhausted` if that
        // leaves the gap open — never silently upgraded to `Optimal`).
        let fallback = || -> Option<(usize, f64)> {
            widest_original_col(spec, &lo, &hi, &root_w, config.min_box_width)
                .map(|j| (j, clamp_interior(x[j], lo[j], hi[j])))
        };
        // #1938: a rejected point passed `int_tol` but not the caller's check on
        // its integral realisation (a binary at 1e-6 on a big-M row). Spatial
        // branching leaves that value inside both children and the LP returns it
        // again, so branch the unfixed integer column farthest from an integer at
        // its floor/ceil: both children exclude the value.
        let near_int = if rejected_lp_point && frac_int.is_none() {
            near_integral_col(spec, x, &lo, &hi)
        } else {
            None
        };
        // #1913: only a genuinely fractional pick feeds the pseudocosts; a #1938
        // near-integral column moves ~1e-6, which would inflate its per-unit gain.
        let int_frac = frac_int.map(|j| x[j] - x[j].floor());
        let pick = if let Some(j) = frac_int.or(near_int) {
            Some((j, x[j].floor() + 0.5)) // integer branch: <= floor, >= ceil
        } else if let Some(col) = branch_col.filter(|&c| width(c) > config.min_box_width) {
            // #1970: score the candidates instead of taking the worst term's
            // widest operand. `None` only if every candidate is too narrow to
            // split, which the legacy column above is not.
            let col = if config.spatial_score {
                n_spatial_score_picks += 1;
                spatial_score_pick(spec, x, (&lo, &hi), &root_w, config, (&spc_sum, &spc_cnt))
                    .unwrap_or(col)
            } else {
                col
            };
            // Spatial branch at the LP value, pulled to the interior.
            Some((col, spatial_split_point(x[col], lo[col], hi[col], config)))
        } else {
            fallback()
        };
        let Some((split_col, split_at)) = pick else {
            // No branchable column: close with the honest rigorous bound.
            global_lb_closed = global_lb_closed.min(bound);
            continue;
        };

        // Two covering children. #1962: they inherit the cuts binding at this
        // node's final LP (a cut whose slack is basic is dropped, and with it that
        // slack from the basis, which leaves the basis exact for the kept rows).
        let (child_cuts, child_basis) = binding_cuts(
            spec.n_cols,
            &node_cuts,
            node.basis,
            config.warm_start_children,
        );
        push_children(
            &mut heap,
            &lo,
            &hi,
            split_col,
            split_at,
            bound,
            spec.integrality[split_col],
            int_frac,
            child_basis.map(Rc::new),
            Some(node_count),
            child_cuts,
            config.spatial_score,
        );
    }

    // Worklist empty: every region was explored or fathomed. The reported bound is
    // the min rigorous bound over all closed regions — a valid global lower bound
    // (`<=` the true optimum), never the incumbent (an upper bound). `Optimal` is
    // claimed ONLY when that bound actually closes the gap; a residual gap (from
    // width-exhausted boxes or uncertifiable node duals) is surfaced honestly as
    // `Exhausted` — both the incumbent and the bound remain valid, but the tree
    // does NOT certify optimality.
    match incumbent {
        Some(inc) => {
            let bound = global_lb_closed.min(inc);
            let status = if gap_closed(bound, inc, config) {
                TreeStatus::Optimal
            } else {
                TreeStatus::Exhausted
            };
            SpatialTreeResult {
                status,
                incumbent,
                incumbent_x,
                bound,
                node_count,
                n_lp_solves,
                n_uncertified,
                n_undecided,
                incumbent_extension_s: extension_s,
                bound_extension_s,
                root_bound,
                root_time_s,
                root_status,
                n_primal_hook_calls,
                n_primal_hook_improvements,
                n_incumbent_value_calls,
                n_incumbent_value_raised,
                n_pseudocost_picks,
                n_spatial_score_picks,
                n_warm_starts,
                n_warm_shape_mismatch,
                n_node_pivots,
                node_lp_s,
                n_cut_forms: cut_forms.len(),
                n_cut_rounds,
                n_cuts_added,
                n_int_probes,
                n_int_probe_improvements,
                n_prop_retries,
                n_prop_retry_decided,
            }
        }
        None => SpatialTreeResult {
            status: TreeStatus::Infeasible,
            incumbent: None,
            incumbent_x: Vec::new(),
            bound: f64::INFINITY,
            node_count,
            n_lp_solves,
            n_uncertified,
            n_undecided,
            incumbent_extension_s: extension_s,
            bound_extension_s,
            root_bound,
            root_time_s,
            root_status,
            n_primal_hook_calls,
            n_primal_hook_improvements,
            n_incumbent_value_calls,
            n_incumbent_value_raised,
            n_pseudocost_picks,
            n_spatial_score_picks,
            n_warm_starts,
            n_warm_shape_mismatch,
            n_node_pivots,
            node_lp_s,
            n_cut_forms: cut_forms.len(),
            n_cut_rounds,
            n_cuts_added,
            n_int_probes,
            n_int_probe_improvements,
            n_prop_retries,
            n_prop_retry_decided,
        },
    }
}

/// Widest ORIGINAL column relative to its root width, or `None` when every one is
/// already at or below `min_box_width` (nothing left to branch on).
fn widest_original_col(
    spec: &SpatialKernelSpec,
    lo: &[f64],
    hi: &[f64],
    root_w: &[f64],
    min_box_width: f64,
) -> Option<usize> {
    let mut best: Option<(f64, usize)> = None;
    for j in 0..spec.n_orig {
        let wj = hi[j] - lo[j];
        if wj > min_box_width.max(1e-9) {
            let rw = wj / root_w[j];
            if best.map(|(bw, _)| rw > bw).unwrap_or(true) {
                best = Some((rw, j));
            }
        }
    }
    best.map(|(_, j)| j)
}

/// The unfixed integer column whose value is farthest from an integer, among those
/// with a NONZERO distance, or `None` (#1938). Used only after the caller rejected a
/// point every integer column of which is within `int_tol`.
fn near_integral_col(spec: &SpatialKernelSpec, x: &[f64], lo: &[f64], hi: &[f64]) -> Option<usize> {
    let mut best: Option<usize> = None;
    let mut best_d = 0.0f64;
    for (j, &is_int) in spec.integrality.iter().enumerate().take(spec.n_cols) {
        if !is_int || hi[j] - lo[j] < 0.5 {
            continue;
        }
        let d = (x[j] - x[j].round()).abs();
        if d > best_d {
            best_d = d;
            best = Some(j);
        }
    }
    best
}

/// Push the two covering children of `[lo, hi]` split on `split_col` at `split_at`,
/// each inheriting `pb` as its (valid) region lower bound. The two children cover
/// the parent box exactly, so no feasible point is lost.
#[allow(clippy::too_many_arguments)]
fn push_children(
    heap: &mut BinaryHeap<QNode>,
    lo: &[f64],
    hi: &[f64],
    split_col: usize,
    split_at: f64,
    pb: f64,
    is_int: bool,
    frac: Option<f64>,
    basis: Option<Rc<Basis>>,
    parent: Option<usize>,
    cuts: Rc<Vec<FixedRow>>,
    record_spatial: bool,
) {
    // #1913: a fractional-integer branch records its column, side and the distance
    // the LP value moves, for the pseudocost update when each child is solved.
    let br = |up: bool| frac.map(|f| (split_col, up, if up { 1.0 - f } else { f }));
    if is_int {
        // integer: child1 x<=floor, child2 x>=ceil
        let f = (split_at - 0.5).floor();
        let lo1 = lo.to_vec();
        let mut hi1 = hi.to_vec();
        hi1[split_col] = f;
        let mut lo2 = lo.to_vec();
        let hi2 = hi.to_vec();
        lo2[split_col] = f + 1.0;
        if hi1[split_col] >= lo1[split_col] - 1e-12 {
            heap.push(QNode {
                pb,
                lo: lo1,
                hi: hi1,
                br: br(false),
                sbr: None,
                basis: basis.clone(),
                parent,
                cuts: cuts.clone(),
            });
        }
        if hi2[split_col] >= lo2[split_col] - 1e-12 {
            heap.push(QNode {
                pb,
                lo: lo2,
                hi: hi2,
                br: br(true),
                sbr: None,
                basis: basis.clone(),
                parent,
                cuts: cuts.clone(),
            });
        }
    } else {
        // #1970: a spatial child records the width it cut off its parent's box.
        let sbr = |up: bool| {
            let cut = if up {
                split_at - lo[split_col]
            } else {
                hi[split_col] - split_at
            };
            record_spatial.then_some((split_col, up, cut))
        };
        let mut hi1 = hi.to_vec();
        hi1[split_col] = split_at;
        let mut lo2 = lo.to_vec();
        lo2[split_col] = split_at;
        heap.push(QNode {
            pb,
            lo: lo.to_vec(),
            hi: hi1,
            br: None,
            sbr: sbr(false),
            basis: basis.clone(),
            parent,
            cuts: cuts.clone(),
        });
        heap.push(QNode {
            pb,
            lo: lo2,
            hi: hi.to_vec(),
            br: None,
            sbr: sbr(true),
            basis: basis.clone(),
            parent,
            cuts: cuts.clone(),
        });
    }
}

/// #1970: the spatial split point for a column with LP value `xv` on `[lo, hi]`:
/// `config.spatial_midpull` of the way from `xv` to the midpoint (`0` is the
/// legacy LP-value split). Any point strictly inside the box gives two covering
/// children, so the choice moves no bound.
fn spatial_split_point(xv: f64, lo: f64, hi: f64, config: &SpatialTreeConfig) -> f64 {
    let m = config.spatial_midpull.clamp(0.0, 1.0);
    let p = if m > 0.0 {
        m * 0.5 * (lo + hi) + (1.0 - m) * xv
    } else {
        xv
    };
    clamp_interior(p, lo, hi)
}

/// The operand columns of a lifted term (a bilinear term's two, a 1-D term's one).
fn term_operands(t: &EnvTerm) -> [Option<usize>; 2] {
    match *t {
        EnvTerm::Bilinear { i, j, .. } => [Some(i), (j != i).then_some(j)],
        EnvTerm::Monomial { i, .. } => [Some(i), None],
        EnvTerm::AffineSquare { j, .. } => [Some(j), None],
        EnvTerm::Sqrt { x: xc, .. } => [Some(xc), None],
    }
}

/// #1970: the spatial branching column under `config.spatial_score`, after SCIP's
/// nonlinear-handler rule (Bestuzheva et al. 2023, `constraints/nonlinear/branching`):
/// every operand of a term with a nonzero McCormick gap is a candidate, its
/// violation score the sum of those terms' gaps split evenly over their operands.
/// With `spatial_pscost_weight > 0`, the product of the two children's estimated
/// gains (spatial pseudocost per unit of width cut off, times the width each side
/// cuts; a column not yet observed on a side takes the mean over observed columns)
/// is added, each component divided by its maximum over the candidates. Columns
/// no wider than `min_box_width` are not candidates. `None` when none remains.
/// Under `config.scale_free_branching` each gap is divided by its aux column's
/// root width, as the legacy rule does, so the pick does not depend on units (the
/// pseudocost estimate, a gain, already does not).
fn spatial_score_pick(
    spec: &SpatialKernelSpec,
    x: &[f64],
    (lo, hi): (&[f64], &[f64]),
    root_w: &[f64],
    config: &SpatialTreeConfig,
    (spc_sum, spc_cnt): (&[Vec<f64>; 2], &[Vec<usize>; 2]),
) -> Option<usize> {
    let mut viol = vec![0.0f64; spec.n_cols];
    let mut add = |cols: &mut dyn Iterator<Item = usize>, gap: f64, aux: usize| {
        let gap = if config.scale_free_branching {
            gap / root_w[aux]
        } else {
            gap
        };
        let cols: Vec<usize> = cols
            .filter(|&c| hi[c] - lo[c] > config.min_box_width)
            .collect();
        let share = gap / cols.len().max(1) as f64;
        for c in cols {
            viol[c] += share;
        }
    };
    for t in &spec.terms {
        let gap = match term_true_value(t, x) {
            Some(v) => (x[t.aux_col()] - v).abs(),
            None => f64::INFINITY,
        };
        if gap > 0.0 {
            add(
                &mut term_operands(t).into_iter().flatten(),
                gap,
                t.aux_col(),
            );
        }
    }
    for t in &spec.blf_terms {
        let a_val = t.a_const + dot_form(&t.a_cols, &t.a_coeffs, x);
        let b_val = t.b_const + dot_form(&t.b_cols, &t.b_coeffs, x);
        let gap = (x[t.w] - a_val * b_val).abs();
        if gap > 0.0 {
            let mut cols: Vec<usize> = t.a_cols.iter().chain(t.b_cols.iter()).copied().collect();
            cols.sort_unstable();
            cols.dedup();
            add(&mut cols.into_iter(), gap, t.w);
        }
    }
    let cands: Vec<usize> = (0..spec.n_cols).filter(|&c| viol[c] > 0.0).collect();
    if cands.is_empty() {
        return None;
    }
    // An infinite gap (an operand outside a term's domain) outranks everything.
    if let Some(&c) = cands.iter().find(|&&c| viol[c].is_infinite()) {
        return Some(c);
    }
    let vmax = cands.iter().map(|&c| viol[c]).fold(0.0f64, f64::max);
    let w = config.spatial_pscost_weight;
    let mut ps = vec![0.0f64; cands.len()];
    let mut pmax = 0.0f64;
    if w > 0.0 {
        let mean = |d: usize| {
            let (s, n) = spc_sum[d]
                .iter()
                .zip(spc_cnt[d].iter())
                .filter(|(_, &n)| n > 0)
                .fold((0.0, 0usize), |(s, n), (&ps, &pn)| {
                    (s + ps / pn as f64, n + 1)
                });
            (n > 0).then(|| s / n as f64)
        };
        // Before any observation the component is uniform, so it changes nothing.
        if let (Some(m0), Some(m1)) = (mean(0), mean(1)) {
            let avg = [m0, m1];
            let pc = |d: usize, c: usize| {
                if spc_cnt[d][c] > 0 {
                    spc_sum[d][c] / spc_cnt[d][c] as f64
                } else {
                    avg[d]
                }
            };
            const EPS: f64 = 1e-6;
            for (k, &c) in cands.iter().enumerate() {
                let p = spatial_split_point(x[c], lo[c], hi[c], config);
                let down = (pc(0, c) * (hi[c] - p)).max(EPS);
                let up = (pc(1, c) * (p - lo[c])).max(EPS);
                ps[k] = down * up;
                pmax = pmax.max(ps[k]);
            }
        }
    }
    let mut best: Option<(f64, usize)> = None;
    for (k, &c) in cands.iter().enumerate() {
        let mut score = viol[c] / vmax;
        if pmax > 0.0 {
            score += w * ps[k] / pmax;
        }
        if best.map(|(b, _)| score > b).unwrap_or(true) {
            best = Some((score, c));
        }
    }
    best.map(|(_, c)| c)
}

/// #1913: the fractional integer column with the largest pseudocost product score
/// `max(f_down * pc_down, eps) * max(f_up * pc_up, eps)` (Achterberg, Koch & Martin
/// 2005), ties to the lowest index. A side never observed takes the mean of the
/// observed pseudocosts on that side, or 1 before any observation, so the first
/// choices fall back to the most balanced fractionality. Not `branching::Pseudocosts`:
/// its unobserved default is a fixed 1.0, which on an objective scaled like nvs17's
/// (gains ~1e3 per unit) ranks every unobserved column last, so it is never tried.
fn pseudocost_pick(
    spec: &SpatialKernelSpec,
    x: &[f64],
    int_tol: f64,
    pc_sum: &[Vec<f64>; 2],
    pc_cnt: &[Vec<usize>; 2],
) -> Option<usize> {
    let mean = |d: usize| {
        let (s, n) = pc_sum[d]
            .iter()
            .zip(pc_cnt[d].iter())
            .filter(|(_, &n)| n > 0)
            .fold((0.0, 0usize), |(s, n), (&ps, &pn)| {
                (s + ps / pn as f64, n + 1)
            });
        if n > 0 {
            s / n as f64
        } else {
            1.0
        }
    };
    let avg = [mean(0), mean(1)];
    let pc = |d: usize, j: usize| {
        if pc_cnt[d][j] > 0 {
            pc_sum[d][j] / pc_cnt[d][j] as f64
        } else {
            avg[d]
        }
    };
    const EPS: f64 = 1e-6;
    let mut best: Option<(f64, usize)> = None;
    for (j, &is_int) in spec.integrality.iter().enumerate().take(spec.n_cols) {
        if !is_int || (x[j] - x[j].round()).abs() <= int_tol {
            continue;
        }
        let f = x[j] - x[j].floor();
        let score = (f * pc(0, j)).max(EPS) * ((1.0 - f) * pc(1, j)).max(EPS);
        if best.map(|(b, _)| score > b).unwrap_or(true) {
            best = Some((score, j));
        }
    }
    best.map(|(_, j)| j)
}

/// #1913: most candidates strong-branched at one node by [`strong_branch_init`].
const STRONG_BRANCH_CANDIDATES: usize = 8;

/// #1913 reliability initialisation: for the (at most [`STRONG_BRANCH_CANDIDATES`])
/// best-scoring fractional integer columns observed fewer than
/// `config.pseudocost_reliability` times on either side, solve both child LPs after
/// propagation and record each child's per-unit bound gain as a pseudocost
/// observation. A child proven empty records the gap to the incumbent (or one unit
/// of the bound's magnitude without one), so a column that prunes a side ranks
/// high. Only the branching choice reads these numbers; no bound or box of the tree is
/// changed. Returns the LP solves spent.
#[allow(clippy::too_many_arguments)]
fn strong_branch_init(
    spec: &SpatialKernelSpec,
    x: &[f64],
    (lo, hi): (&[f64], &[f64]),
    bound: f64,
    incumbent: Option<f64>,
    config: &SpatialTreeConfig,
    opts: &SimplexOptions,
    (pc_sum, pc_cnt): (&mut [Vec<f64>; 2], &mut [Vec<usize>; 2]),
    cuts: &[FixedRow],
) -> usize {
    let rel = config.pseudocost_reliability;
    let mut cands: Vec<(f64, usize)> = Vec::new();
    for (j, &is_int) in spec.integrality.iter().enumerate().take(spec.n_cols) {
        if !is_int || (x[j] - x[j].round()).abs() <= config.int_tol {
            continue;
        }
        if pc_cnt[0][j].min(pc_cnt[1][j]) >= rel {
            continue;
        }
        let f = x[j] - x[j].floor();
        cands.push(((f * (1.0 - f)), j));
    }
    // Most balanced fractionality first; ties to the lowest index.
    cands.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    let pruned_gain = incumbent.map_or(bound.abs().max(1.0), |inc| (inc - bound).max(0.0));
    let mut n_lp = 0usize;
    for &(_, j) in cands.iter().take(STRONG_BRANCH_CANDIDATES) {
        let f = x[j] - x[j].floor();
        for up in [false, true] {
            if opts.deadline.is_some_and(|d| Instant::now() >= d) {
                return n_lp;
            }
            let mut l2 = lo.to_vec();
            let mut h2 = hi.to_vec();
            if up {
                l2[j] = x[j].ceil();
            } else {
                h2[j] = x[j].floor();
            }
            let dist = if up { 1.0 - f } else { f };
            let empty = l2[j] > h2[j]
                || (config.run_propagation
                    && !propagate_spec_fixpoint(
                        spec,
                        &mut l2,
                        &mut h2,
                        incumbent,
                        config.propagation_rounds,
                    ));
            let gain = if empty {
                Some(pruned_gain)
            } else {
                let child = solve_spatial_node_with_cuts(spec, &l2, &h2, false, opts, None, cuts);
                n_lp += child.n_lp_solves;
                match verdict_for(child.status) {
                    NodeVerdict::EmptyRegion => Some(pruned_gain),
                    NodeVerdict::Bound if child.bound.is_finite() => {
                        Some((child.bound - bound).max(0.0))
                    }
                    _ => None,
                }
            };
            if let Some(g) = gain {
                let d = usize::from(up);
                pc_sum[d][j] += g / dist.max(1e-9);
                pc_cnt[d][j] += 1;
            }
        }
    }
    n_lp
}

/// #1962: run up to `rounds` separation rounds on a solved node. Each round adds
/// the violated cuts of `forms` at `node`'s LP point, extends its basis by the new
/// rows (each new slack basic, so the extended basis is primal-feasible except on
/// the violated rows, the dual simplex's textbook restart), and re-solves.
///
/// On return `node` is the last LP that decided (`Optimal`, or a Farkas-certified
/// `Infeasible`) with its `bound` raised to the best safe bound any round proved,
/// and the returned cut list is exactly the cut block of that LP. Stops early when a
/// round finds nothing violated, when the LP objective stalls, or when an LP decides
/// nothing (that round's cuts are dropped). Returns `(cuts, rounds, cuts added)`.
#[allow(clippy::too_many_arguments)]
fn separation_loop(
    spec: &SpatialKernelSpec,
    (lo, hi): (&[f64], &[f64]),
    config: &SpatialTreeConfig,
    opts: &SimplexOptions,
    forms: &[ConvexQuadForm],
    rounds: usize,
    node: &mut SpatialNodeResult,
    inherited: &Rc<Vec<FixedRow>>,
    n_lp_solves: &mut usize,
    n_node_pivots: &mut usize,
) -> (Rc<Vec<FixedRow>>, usize, usize) {
    use crate::lp::basis::BASIC;
    let mut cuts: Vec<FixedRow> = inherited.as_ref().clone();
    let mut best_bound = node.bound;
    let mut done = 0usize;
    let mut added = 0usize;
    for _ in 0..rounds {
        if node.status != LpStatus::Optimal {
            break;
        }
        if opts.deadline.is_some_and(|d| Instant::now() >= d) {
            break;
        }
        let new = separate_round(forms, &node.x, 1e-6, config.max_cuts_per_round);
        if new.is_empty() {
            break;
        }
        let k = new.len();
        let mut warm = node.basis.clone();
        let n_total = warm.col_status.len();
        warm.col_status.extend(std::iter::repeat_n(BASIC, k));
        warm.basic_vars.extend(n_total..n_total + k);
        let base_len = cuts.len();
        cuts.extend(new);
        let next = solve_spatial_node_with_cuts(spec, lo, hi, false, opts, Some(&warm), &cuts);
        *n_lp_solves += next.n_lp_solves;
        *n_node_pivots += next.n_pivots;
        match next.status {
            LpStatus::Optimal => {
                done += 1;
                added += k;
                let stalled = next.obj - node.obj <= 1e-6 * node.obj.abs().max(1.0);
                best_bound = best_bound.max(next.bound);
                // The re-solve skips OBBT; keep the first solve's tightenings. It
                // is also always warm, so keep the first solve's warm-start record:
                // that is what `n_warm_starts` reports (the inherited basis).
                let tightened = std::mem::take(&mut node.tightened);
                let (ws, wm) = (node.warm_started, node.warm_shape_mismatch);
                *node = next;
                node.tightened = tightened;
                node.warm_started = ws;
                node.warm_shape_mismatch = wm;
                if stalled {
                    break;
                }
            }
            LpStatus::Infeasible => {
                // A verified Farkas ray over valid rows: the region holds no
                // feasible point.
                done += 1;
                added += k;
                let (ws, wm) = (node.warm_started, node.warm_shape_mismatch);
                *node = next;
                node.warm_started = ws;
                node.warm_shape_mismatch = wm;
                return (Rc::new(cuts), done, added);
            }
            LpStatus::Unbounded | LpStatus::IterLimit | LpStatus::Numerical => {
                cuts.truncate(base_len);
                break;
            }
        }
    }
    node.bound = best_bound;
    (Rc::new(cuts), done, added)
}

/// #1962: the cuts a node hands its children -- those whose slack is NONBASIC
/// (binding) in its final basis -- and, when `keep_basis`, that basis restricted to
/// the kept rows. A dropped cut's slack is basic, so removing the row and its slack
/// together leaves a square basis over the remaining rows with the same basic
/// structurals: still the node's optimal vertex, now of the smaller LP.
fn binding_cuts(
    n_cols: usize,
    cuts: &Rc<Vec<FixedRow>>,
    basis: Basis,
    keep_basis: bool,
) -> (Rc<Vec<FixedRow>>, Option<Basis>) {
    use crate::lp::basis::BASIC;
    if cuts.is_empty() {
        return (cuts.clone(), keep_basis.then_some(basis));
    }
    let m = basis.basic_vars.len();
    let n_total = basis.col_status.len();
    // A basis that does not span this node's rows (an undecided LP has none):
    // hand down the cuts unchanged and no basis.
    if n_total != n_cols + m || m < cuts.len() {
        return (cuts.clone(), None);
    }
    let first = n_cols + m - cuts.len(); // slack column of the first cut row
    let keep: Vec<bool> = (0..cuts.len())
        .map(|t| basis.col_status[first + t] != BASIC)
        .collect();
    if keep.iter().all(|&k| k) {
        return (cuts.clone(), keep_basis.then_some(basis));
    }
    let kept: Vec<FixedRow> = cuts
        .iter()
        .zip(keep.iter())
        .filter(|(_, &k)| k)
        .map(|(c, _)| c.clone())
        .collect();
    let basis = keep_basis.then(|| {
        let mut col_status = basis.col_status[..first].to_vec();
        col_status.extend(
            (0..cuts.len())
                .filter(|&t| keep[t])
                .map(|t| basis.col_status[first + t]),
        );
        // Every cut slack still basic is a dropped one; nothing else moves.
        let basic_vars = basis
            .basic_vars
            .into_iter()
            .filter(|&c| c < first)
            .collect();
        Basis {
            col_status,
            basic_vars,
        }
    });
    (Rc::new(kept), basis)
}

/// The largest McCormick gap `|x_aux − f(operands)|` over every lifted term at `x`
/// (`inf` when a term cannot be evaluated there) -- the acceptance test's measure.
fn max_term_gap(spec: &SpatialKernelSpec, x: &[f64]) -> f64 {
    let mut worst = 0.0f64;
    for t in &spec.terms {
        let gap = match term_true_value(t, x) {
            Some(v) => (x[t.aux_col()] - v).abs(),
            None => f64::INFINITY,
        };
        worst = worst.max(gap);
    }
    for t in &spec.blf_terms {
        let a_val = t.a_const + dot_form(&t.a_cols, &t.a_coeffs, x);
        let b_val = t.b_const + dot_form(&t.b_cols, &t.b_coeffs, x);
        worst = worst.max((x[t.w] - a_val * b_val).abs());
    }
    if worst.is_nan() {
        f64::INFINITY
    } else {
        worst
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// `Σ coeffs[k] * x[cols[k]]` — the value of a sparse affine form's linear part.
fn dot_form(cols: &[usize], coeffs: &[f64], x: &[f64]) -> f64 {
    cols.iter()
        .zip(coeffs.iter())
        .map(|(&c, &a)| a * x[c])
        .sum()
}

/// Pull a split point strictly inside `(lo, hi)` so both children are nonempty; if
/// `p` sits at a bound, use the midpoint.
fn clamp_interior(p: f64, lo: f64, hi: f64) -> f64 {
    let eps = 1e-9 * (1.0 + (hi - lo).abs());
    if p <= lo + eps || p >= hi - eps {
        0.5 * (lo + hi)
    } else {
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bnb::spatial_kernel::FixedRow;

    // minimize w = x*y  s.t.  x + y >= 3,  x,y in [0,2].
    // True feasible region on [0,2]^2 with x+y>=3: the min of x*y is 2 (corners
    // (2,1),(1,2)); the interior x=y=1.5 gives 2.25 > 2. McCormick underestimates
    // at the root, so B&B must branch to certify 2.0.
    pub(super) fn xy_min_spec() -> SpatialKernelSpec {
        SpatialKernelSpec {
            n_cols: 3,
            n_orig: 2,
            c: vec![0.0, 0.0, 1.0], // minimize w
            integrality: vec![false, false, false],
            global_lo: vec![0.0, 0.0, -1e20],
            global_hi: vec![2.0, 2.0, 1e20],
            // x + y >= 3  ==>  -x - y <= -3
            fixed_rows: vec![FixedRow {
                cols: vec![0, 1],
                coeffs: vec![-1.0, -1.0],
                rhs: -3.0,
            }],
            terms: vec![EnvTerm::Bilinear { i: 0, j: 1, w: 2 }],
            blf_terms: vec![],
            obbt_candidates: vec![0, 1],
        }
    }

    /// #1619 D-11: the operand a bilinear branch splits must not depend on the
    /// units a column is written in. Node widths `(0.75, 2.5)` inside root widths
    /// `(1.5, 2.5)`, then the same node with `y` rescaled by 0.01: the legacy
    /// (absolute) rule flips from `y` to `x`; the root-relative rule keeps `y`.
    #[test]
    fn scale_free_operand_choice_is_unit_invariant() {
        let t = EnvTerm::Bilinear { i: 0, j: 1, w: 2 };
        let x = [1.0, 1.0, 1.0];
        let pick = |node: [f64; 2], root: [f64; 2], relative: bool| {
            let width = move |c: usize| {
                if relative {
                    node[c] / root[c]
                } else {
                    node[c]
                }
            };
            term_gap_and_branch_col(&t, &x, &width).1
        };
        for relative in [false, true] {
            let a = pick([0.75, 2.5], [1.5, 2.5], relative);
            let b = pick([0.75, 0.025], [1.5, 0.025], relative);
            if relative {
                assert_eq!((a, b), (1, 1), "relative rule changed with the units");
            } else {
                assert_eq!(
                    (a, b),
                    (1, 0),
                    "the probe must show the legacy unit dependence"
                );
            }
        }
    }

    #[test]
    fn branches_to_certify_bilinear_min() {
        let spec = xy_min_spec();
        let opts = SimplexOptions::default();
        let cfg = SpatialTreeConfig {
            max_nodes: 5000,
            gap_tol: 1e-5,
            ..SpatialTreeConfig::default()
        };
        let res = solve_spatial_tree(&spec, &cfg, &opts);
        assert_eq!(res.status, TreeStatus::Optimal, "did not converge: {res:?}");
        let inc = res.incumbent.expect("incumbent found");
        // Global optimum is 2.0.
        assert!((inc - 2.0).abs() < 1e-3, "incumbent {inc} != 2.0");
        // Soundness: global bound never above the true optimum.
        assert!(
            res.bound <= 2.0 + 1e-6,
            "bound {} above optimum 2.0",
            res.bound
        );
        // The incumbent point is feasible: x*y == w and x+y>=3.
        let x = &res.incumbent_x;
        assert!((x[0] * x[1] - x[2]).abs() < 1e-4, "w != x*y at incumbent");
        assert!(x[0] + x[1] >= 3.0 - 1e-4, "x+y>=3 violated");
    }

    /// #1236 review finding 8: a non-finite `root_bound` must still say WHICH
    /// outcome it was.
    ///
    /// `root_bound` is `+inf` when the root region was certified empty, the
    /// parent's `-inf` when the root LP decided nothing, and the initial `-inf`
    /// when the search never finished node 1. The Python mapping turns every
    /// non-finite value into `root_bound=None`, so all three read as "the root
    /// never produced a bound" — the §6 conflation this instrumentation exists to
    /// remove, reproduced inside it. `root_status` names the arm.
    #[test]
    fn a_non_finite_root_bound_still_names_its_outcome() {
        let spec = xy_min_spec();

        // Arm 1: a normal solve leaves by the bounded arm and says so.
        let cfg = SpatialTreeConfig {
            max_nodes: 5000,
            gap_tol: 1e-5,
            ..SpatialTreeConfig::default()
        };
        let res = solve_spatial_tree(&spec, &cfg, &SimplexOptions::default());
        assert_eq!(
            res.root_status, "bounded",
            "a solve whose root produced a finite bound must say `bounded`"
        );
        assert!(res.root_bound.is_finite());

        // Arm 2: stopped before node 1 can finish. `root_bound` is non-finite and
        // MUST be distinguishable from the two arms that also leave it non-finite.
        let cfg0 = SpatialTreeConfig {
            max_nodes: 0,
            gap_tol: 1e-5,
            ..SpatialTreeConfig::default()
        };
        let res0 = solve_spatial_tree(&spec, &cfg0, &SimplexOptions::default());
        assert_eq!(res0.node_count, 0, "node 1 must not have run");
        assert!(
            !res0.root_bound.is_finite(),
            "expected a non-finite root bound, got {}",
            res0.root_bound
        );
        assert_eq!(
            res0.root_status, "not_reached",
            "a search that never reached node 1 must say so, not share \
             `root_bound = None` with a root that was certified empty"
        );

        // The point of the whole thing: the two runs are distinguishable even
        // though a Python caller sees `root_bound=None` for one of them.
        assert_ne!(res.root_status, res0.root_status);
    }

    /// #1236: the root region's bound is recorded, is a VALID lower bound on the
    /// true optimum, and is no tighter than the final tree bound.
    ///
    /// The kernel used to report only its final `bound`, which left every
    /// kernel-routed Python `SolveResult` with `root_bound=None` — so "the
    /// relaxation is loose at the root and the tree closed it" was
    /// indistinguishable from "the root was already tight". On this spec the
    /// McCormick root strictly underestimates (that is what
    /// `branches_to_certify_bilinear_min` pins), so the root bound must be BELOW
    /// the optimum 2.0 while the final bound reaches it — a single assertion that
    /// the recorded value is the root's and not a copy of the final bound.
    #[test]
    fn records_the_root_region_bound() {
        let spec = xy_min_spec();
        let cfg = SpatialTreeConfig {
            max_nodes: 5000,
            gap_tol: 1e-5,
            ..SpatialTreeConfig::default()
        };
        let res = solve_spatial_tree(&spec, &cfg, &SimplexOptions::default());
        assert_eq!(res.status, TreeStatus::Optimal, "did not converge: {res:?}");
        assert!(
            res.root_bound.is_finite(),
            "root bound was never recorded: {}",
            res.root_bound
        );
        // Soundness: a root bound is a lower bound on the true optimum (2.0).
        assert!(
            res.root_bound <= 2.0 + 1e-6,
            "root bound {} above the optimum 2.0",
            res.root_bound
        );
        // Monotonicity: branching only ever tightens the global bound.
        assert!(
            res.root_bound <= res.bound + 1e-9,
            "root bound {} tighter than the final bound {}",
            res.root_bound,
            res.bound
        );
        assert!(res.root_time_s >= 0.0);

        // It is the ROOT's value, not a copy of the final bound. Spec-independent
        // identity: a search stopped after node 1 reports exactly the bound the
        // root region proved, so the full search's `root_bound` must equal the
        // one-node search's `bound`. (On THIS spec McCormick happens to be tight
        // at the root, so a "strictly looser than the final bound" assertion would
        // pass only by accident of the instance -- the identity holds either way.)
        let cfg1 = SpatialTreeConfig {
            max_nodes: 1,
            ..cfg
        };
        let res1 = solve_spatial_tree(&spec, &cfg1, &SimplexOptions::default());
        assert_eq!(
            res1.node_count, 1,
            "one-node run processed {}",
            res1.node_count
        );
        assert!(
            (res1.bound - res.root_bound).abs() <= 1e-9,
            "root bound {} != the one-node search's bound {}",
            res.root_bound,
            res1.bound
        );
        assert!(
            (res1.root_bound - res.root_bound).abs() <= 1e-9,
            "the two runs disagree on the root bound: {} vs {}",
            res1.root_bound,
            res.root_bound
        );
    }

    /// #1236 companion: a spec whose McCormick root is genuinely LOOSE, so the
    /// recorded root bound is visibly weaker than the bound the tree ends with.
    /// Minimize `-x*y` on `x + y = 2`, `x,y in [0,2]`: the optimum is -1 at
    /// (1,1), while the root McCormick overestimator allows `w <= 2` and so bounds
    /// the objective only by -2. Without a root bound on the result there is no
    /// way to see that from Python -- which is the whole point of the field.
    #[test]
    fn root_bound_shows_a_loose_root_relaxation() {
        let spec = SpatialKernelSpec {
            n_cols: 3,
            n_orig: 2,
            c: vec![0.0, 0.0, -1.0], // minimize -w
            integrality: vec![false, false, false],
            global_lo: vec![0.0, 0.0, -1e20],
            global_hi: vec![2.0, 2.0, 1e20],
            // x + y == 2, as the two inequalities the fixed-row form takes.
            fixed_rows: vec![
                FixedRow {
                    cols: vec![0, 1],
                    coeffs: vec![1.0, 1.0],
                    rhs: 2.0,
                },
                FixedRow {
                    cols: vec![0, 1],
                    coeffs: vec![-1.0, -1.0],
                    rhs: -2.0,
                },
            ],
            terms: vec![EnvTerm::Bilinear { i: 0, j: 1, w: 2 }],
            blf_terms: vec![],
            obbt_candidates: vec![0, 1],
        };
        let cfg = SpatialTreeConfig {
            max_nodes: 5000,
            gap_tol: 1e-5,
            ..SpatialTreeConfig::default()
        };
        let res = solve_spatial_tree(&spec, &cfg, &SimplexOptions::default());
        assert!(res.root_bound.is_finite(), "root bound never recorded");
        // Sound: below the true optimum -1.
        assert!(
            res.root_bound <= -1.0 + 1e-6,
            "root bound {} above the optimum -1.0",
            res.root_bound
        );
        // Loose: the root envelope gives about -2, and the tree tightens it.
        assert!(
            res.root_bound < res.bound - 1e-3,
            "root bound {} not looser than the final bound {}",
            res.root_bound,
            res.bound
        );
    }

    /// Regression (premature-fathom fix): a feasible leaf whose region bound does
    /// NOT certify must keep branching until the gap genuinely closes — `Optimal`
    /// is only ever reported with `bound >= incumbent - gap_tol`.
    #[test]
    fn optimal_status_implies_certified_gap() {
        let spec = xy_min_spec();
        let cfg = SpatialTreeConfig {
            max_nodes: 5000,
            gap_tol: 1e-5,
            ..SpatialTreeConfig::default()
        };
        let res = solve_spatial_tree(&spec, &cfg, &SimplexOptions::default());
        if res.status == TreeStatus::Optimal {
            let inc = res.incumbent.expect("Optimal implies incumbent");
            assert!(
                res.bound >= inc - cfg.gap_tol - 1e-12,
                "Optimal with open gap: bound {} vs incumbent {}",
                res.bound,
                inc
            );
        }
        // And an externally seeded incumbent is honored + reported back.
        let cfg2 = SpatialTreeConfig {
            initial_incumbent: Some(2.0), // the true optimum, externally known
            ..cfg
        };
        let res2 = solve_spatial_tree(&spec, &cfg2, &SimplexOptions::default());
        assert!(res2.incumbent.is_some());
        assert!(res2.incumbent.unwrap() <= 2.0 + 1e-9);
        if res2.status == TreeStatus::Optimal {
            assert!(res2.bound >= res2.incumbent.unwrap() - cfg2.gap_tol - 1e-12);
        }
    }

    // A pure integer-branch case: minimize -x s.t. x in [0,2] integer, no terms.
    // Optimum x=2, obj -2. Exercises integer branching + acceptance.
    #[test]
    fn integer_branch_finds_optimum() {
        let spec = SpatialKernelSpec {
            n_cols: 1,
            n_orig: 1,
            c: vec![-1.0],
            integrality: vec![true],
            global_lo: vec![0.0],
            global_hi: vec![2.0],
            fixed_rows: vec![],
            terms: vec![],
            blf_terms: vec![],
            obbt_candidates: vec![],
        };
        let cfg = SpatialTreeConfig {
            run_obbt: false,
            ..SpatialTreeConfig::default()
        };
        let res = solve_spatial_tree(&spec, &cfg, &SimplexOptions::default());
        assert_eq!(res.status, TreeStatus::Optimal);
        assert!((res.incumbent.unwrap() - (-2.0)).abs() < 1e-9);
        assert!((res.incumbent_x[0] - 2.0).abs() < 1e-9);
    }

    /// #1913: with no observations every side takes pseudocost 1, so the product
    /// rule picks the most balanced fractionality; observed gains then decide, and
    /// integral or continuous columns are never picked.
    #[test]
    fn pseudocost_pick_prefers_balanced_then_observed_gains() {
        let spec = SpatialKernelSpec {
            n_cols: 4,
            n_orig: 4,
            c: vec![0.0; 4],
            integrality: vec![true, true, true, false],
            global_lo: vec![0.0; 4],
            global_hi: vec![5.0; 4],
            fixed_rows: vec![],
            terms: vec![],
            blf_terms: vec![],
            obbt_candidates: vec![],
        };
        let x = [1.1, 2.5, 3.0, 0.5];
        let zero_s = [vec![0.0; 4], vec![0.0; 4]];
        let zero_n = [vec![0usize; 4], vec![0usize; 4]];
        assert_eq!(pseudocost_pick(&spec, &x, 1e-5, &zero_s, &zero_n), Some(1));
        // Column 0 has gained 10 per unit on both sides, column 1 only 0.1.
        let s = [vec![10.0, 0.1, 0.0, 0.0], vec![10.0, 0.1, 0.0, 0.0]];
        let n = [vec![1, 1, 0, 0], vec![1, 1, 0, 0]];
        assert_eq!(pseudocost_pick(&spec, &x, 1e-5, &s, &n), Some(0));
        // No fractional integer column: nothing to pick.
        let xi = [1.0, 2.0, 3.0, 0.5];
        assert_eq!(pseudocost_pick(&spec, &xi, 1e-5, &s, &n), None);
    }

    /// #1913: the rule changes only which column is branched, so a pure integer
    /// program certifies the same optimum either way; the probe counter is 0 with
    /// the rule off and positive with it on.
    #[test]
    fn pseudocost_branching_certifies_the_same_optimum() {
        // max 5a + 4b + 3c  s.t.  2a + 3b + c <= 5.5,  4a + b + 2c <= 10.5,
        // 3a + 4b + 2c <= 7.5, a,b,c in [0, 5] integer; the optimum is found by
        // enumeration below.
        let rows = vec![
            (vec![2.0, 3.0, 1.0], 5.5),
            (vec![4.0, 1.0, 2.0], 10.5),
            (vec![3.0, 4.0, 2.0], 7.5),
        ];
        let c = vec![-5.0, -4.0, -3.0];
        let mut best = f64::INFINITY;
        for a in 0..=5 {
            for b in 0..=5 {
                for cc in 0..=5 {
                    let v = [a as f64, b as f64, cc as f64];
                    if rows.iter().all(|(r, rhs)| dot(r, &v) <= *rhs) {
                        best = best.min(dot(&c, &v));
                    }
                }
            }
        }
        let spec = SpatialKernelSpec {
            n_cols: 3,
            n_orig: 3,
            c,
            integrality: vec![true, true, true],
            global_lo: vec![0.0; 3],
            global_hi: vec![5.0; 3],
            fixed_rows: rows
                .iter()
                .map(|(r, rhs)| FixedRow {
                    cols: vec![0, 1, 2],
                    coeffs: r.clone(),
                    rhs: *rhs,
                })
                .collect(),
            terms: vec![],
            blf_terms: vec![],
            obbt_candidates: vec![],
        };
        for (on, rel) in [(false, 0), (true, 0), (true, 2)] {
            let cfg = SpatialTreeConfig {
                pseudocost_branching: on,
                pseudocost_reliability: rel,
                ..SpatialTreeConfig::default()
            };
            let res = solve_spatial_tree(&spec, &cfg, &SimplexOptions::default());
            assert_eq!(res.status, TreeStatus::Optimal);
            assert!((res.incumbent.unwrap() - best).abs() < 1e-9);
            assert!(res.bound <= best + 1e-9);
            assert_eq!(res.n_pseudocost_picks > 0, on);
        }
    }

    /// #1970: the score sums each column's share of every violated term's gap, so
    /// a column in two moderately violated terms outranks the operands of the one
    /// term with the largest gap (which the legacy rule splits).
    #[test]
    fn spatial_score_prefers_the_column_shared_by_violated_terms() {
        let spec = SpatialKernelSpec {
            n_cols: 8,
            n_orig: 5,
            c: vec![0.0; 8],
            integrality: vec![false; 8],
            global_lo: vec![0.0; 8],
            global_hi: vec![2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0],
            fixed_rows: vec![],
            terms: vec![
                EnvTerm::Bilinear { i: 0, j: 1, w: 5 },
                EnvTerm::Bilinear { i: 0, j: 2, w: 6 },
                EnvTerm::Bilinear { i: 3, j: 4, w: 7 },
            ],
            blf_terms: vec![],
            obbt_candidates: vec![],
        };
        // Every product is 1; the gaps are 1, 1 and 1.5.
        let x = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.5];
        let (lo, hi) = (spec.global_lo.clone(), spec.global_hi.clone());
        let none = [vec![0.0; 8], vec![0.0; 8]];
        let zero = [vec![0usize; 8], vec![0usize; 8]];
        for w in [0.0, 1.0] {
            let cfg = SpatialTreeConfig {
                spatial_score: true,
                spatial_pscost_weight: w,
                ..SpatialTreeConfig::default()
            };
            // Column 0 scores 0.5 + 0.5; columns 3 and 4 score 0.75 each. With
            // no observation the pseudocost component is uniform and moves nothing.
            assert_eq!(
                spatial_score_pick(&spec, &x, (&lo, &hi), &[1.0; 8], &cfg, (&none, &zero)),
                Some(0)
            );
        }
        // An observed pseudocost can outweigh the violation: column 3 gains a lot
        // per unit cut on both sides, column 0 nothing.
        let mut s = [vec![0.0; 8], vec![0.0; 8]];
        let mut n = [vec![1usize; 8], vec![1usize; 8]];
        s[0][3] = 10.0;
        s[1][3] = 10.0;
        n[0][0] = 1;
        let cfg = SpatialTreeConfig {
            spatial_score: true,
            spatial_pscost_weight: 1.0,
            ..SpatialTreeConfig::default()
        };
        assert_eq!(
            spatial_score_pick(&spec, &x, (&lo, &hi), &[1.0; 8], &cfg, (&s, &n)),
            Some(3)
        );
        // A column too narrow to split is never a candidate.
        let mut hi2 = hi.clone();
        hi2[0] = lo[0] + 1e-12;
        let cfg0 = SpatialTreeConfig {
            spatial_score: true,
            ..SpatialTreeConfig::default()
        };
        let pick = spatial_score_pick(&spec, &x, (&lo, &hi2), &[1.0; 8], &cfg0, (&none, &zero));
        // Column 0's shares go to its partners: columns 1 and 2 score 1 each.
        assert_eq!(pick, Some(1));
    }

    #[test]
    fn spatial_midpull_moves_the_split_toward_the_midpoint() {
        let cfg = |m: f64| SpatialTreeConfig {
            spatial_midpull: m,
            ..SpatialTreeConfig::default()
        };
        assert_eq!(spatial_split_point(0.1, 0.0, 4.0, &cfg(0.0)), 0.1);
        assert!((spatial_split_point(0.1, 0.0, 4.0, &cfg(0.75)) - 1.525).abs() < 1e-12);
        assert_eq!(spatial_split_point(0.1, 0.0, 4.0, &cfg(1.0)), 2.0);
        // At a bound the legacy rule already takes the midpoint.
        assert_eq!(spatial_split_point(0.0, 0.0, 4.0, &cfg(0.0)), 2.0);
    }

    /// #1970: the scored rule certifies the same optima as the legacy one, and
    /// reports that it ran. `min -x*y - y*z  s.t.  x + y + z <= 3` on `[0, 2]^3`:
    /// `-y * (x + z) >= -y * (3 - y) >= -2.25`, attained at `y = 1.5`.
    #[test]
    fn spatial_score_certifies_the_same_optimum() {
        let spec = SpatialKernelSpec {
            n_cols: 5,
            n_orig: 3,
            c: vec![0.0, 0.0, 0.0, -1.0, -1.0],
            integrality: vec![false; 5],
            global_lo: vec![0.0, 0.0, 0.0, -1e20, -1e20],
            global_hi: vec![2.0, 2.0, 2.0, 1e20, 1e20],
            fixed_rows: vec![FixedRow {
                cols: vec![0, 1, 2],
                coeffs: vec![1.0, 1.0, 1.0],
                rhs: 3.0,
            }],
            terms: vec![
                EnvTerm::Bilinear { i: 0, j: 1, w: 3 },
                EnvTerm::Bilinear { i: 1, j: 2, w: 4 },
            ],
            blf_terms: vec![],
            obbt_candidates: vec![],
        };
        for (score, w, m) in [(false, 0.0, 0.0), (true, 0.0, 0.0), (true, 1.0, 0.75)] {
            for s in [&spec, &xy_min_spec()] {
                let cfg = SpatialTreeConfig {
                    spatial_score: score,
                    spatial_pscost_weight: w,
                    spatial_midpull: m,
                    gap_tol: 1e-5,
                    ..SpatialTreeConfig::default()
                };
                let res = solve_spatial_tree(s, &cfg, &SimplexOptions::default());
                let opt = if s.n_cols == 5 { -2.25 } else { 2.0 };
                assert_eq!(res.status, TreeStatus::Optimal, "{score} {w} {m}");
                assert!((res.incumbent.unwrap() - opt).abs() < 1e-4, "{res:?}");
                assert!(res.bound <= opt + 1e-9);
                // `xy_min_spec` certifies at its root; the other has to branch.
                if s.n_cols == 5 {
                    assert_eq!(res.n_spatial_score_picks > 0, score, "{res:?}");
                    assert!(res.node_count > 1);
                }
            }
        }
    }

    /// Issue #788: an expired wall-clock budget must retain the best known
    /// incumbent/bound and, above all, must never be upgraded to `Optimal`.
    #[test]
    fn expired_deadline_returns_honest_time_limit() {
        let spec = xy_min_spec();
        let cfg = SpatialTreeConfig {
            deadline: Some(Instant::now()),
            initial_incumbent: Some(2.0),
            ..SpatialTreeConfig::default()
        };
        let res = solve_spatial_tree(&spec, &cfg, &SimplexOptions::default());
        assert_eq!(res.status, TreeStatus::TimeLimit);
        assert_eq!(res.node_count, 0);
        assert_eq!(res.incumbent, Some(2.0));
        assert_eq!(res.bound, f64::NEG_INFINITY);
    }

    // ---- #917: incumbent-conditional budget extension ----------------------

    #[test]
    fn extension_is_taken_only_with_an_incumbent_in_hand() {
        // Same already-expired deadline as above, but the caller withheld a reserve
        // it is willing to hand back. An incumbent IS held, so the search reclaims it
        // and actually runs instead of exiting at node 0.
        let spec = xy_min_spec();
        let cfg = SpatialTreeConfig {
            deadline: Some(Instant::now()),
            initial_incumbent: Some(2.0),
            incumbent_time_extension: Some(Duration::from_secs(30)),
            ..SpatialTreeConfig::default()
        };
        let res = solve_spatial_tree(&spec, &cfg, &SimplexOptions::default());
        assert!(
            res.node_count > 0,
            "the extension must let the search proceed, got node_count=0"
        );
        assert!(
            res.bound > f64::NEG_INFINITY,
            "a running search must produce a bound"
        );
    }

    #[test]
    fn extension_is_declined_without_an_incumbent() {
        // The whole safety argument: with no incumbent the search must stop at its
        // reduced deadline, because that is exactly when the caller's #844
        // no-incumbent fallback needs the reserve it withheld.
        let spec = xy_min_spec();
        let cfg = SpatialTreeConfig {
            deadline: Some(Instant::now()),
            initial_incumbent: None,
            incumbent_time_extension: Some(Duration::from_secs(30)),
            ..SpatialTreeConfig::default()
        };
        let res = solve_spatial_tree(&spec, &cfg, &SimplexOptions::default());
        assert_eq!(res.status, TreeStatus::TimeLimit);
        assert_eq!(res.node_count, 0, "no incumbent means no extension");
    }

    #[test]
    fn extension_is_taken_at_most_once() {
        // A zero-length extension with an incumbent: the guard fires once, the
        // deadline does not move, and the very next check exits. If the "once" latch
        // were missing this would spin forever rather than return.
        let spec = xy_min_spec();
        let cfg = SpatialTreeConfig {
            deadline: Some(Instant::now()),
            initial_incumbent: Some(2.0),
            incumbent_time_extension: Some(Duration::from_secs(0)),
            ..SpatialTreeConfig::default()
        };
        let res = solve_spatial_tree(&spec, &cfg, &SimplexOptions::default());
        assert_eq!(res.status, TreeStatus::TimeLimit);
    }

    #[test]
    fn no_extension_configured_is_the_pre_917_deadline() {
        // The default must be bit-identical to the old behaviour.
        let spec = xy_min_spec();
        let cfg = SpatialTreeConfig {
            deadline: Some(Instant::now()),
            initial_incumbent: Some(2.0),
            incumbent_time_extension: None,
            ..SpatialTreeConfig::default()
        };
        let res = solve_spatial_tree(&spec, &cfg, &SimplexOptions::default());
        assert_eq!(res.status, TreeStatus::TimeLimit);
        assert_eq!(res.node_count, 0);
        assert_eq!(res.bound, f64::NEG_INFINITY);
    }

    /// #1009: the node LP must run under the tree's LIVE deadline.
    ///
    /// Before this, `SimplexOptions::deadline` was left `None` on the kernel path
    /// and the tree could only check its clock BETWEEN nodes, so a single slow
    /// node LP overran the budget without limit — measured 240.83 s against a
    /// 20 s `time_limit` on QPLIB_1157 (`DISCOPT_RLT_LINEQ=1`), 11.9x over, on
    /// node 1.
    #[test]
    fn node_lp_opts_takes_the_earlier_deadline_and_preserves_everything_else() {
        let now = Instant::now();
        let early = now + Duration::from_secs(1);
        let late = now + Duration::from_secs(100);
        let base = SimplexOptions::default();
        let mut checked = 0usize;

        // No deadline anywhere: unchanged, so a solve with no time limit stays
        // bit-identical to the pre-#1009 path.
        assert_eq!(node_lp_opts(&base, None).deadline, None);
        checked += 1;
        // Tree deadline only — the case the bug dropped on the floor.
        assert_eq!(node_lp_opts(&base, Some(late)).deadline, Some(late));
        checked += 1;
        // Caller's per-LP cap only: still honored when the tree has no budget.
        let capped = SimplexOptions {
            deadline: Some(early),
            ..base.clone()
        };
        assert_eq!(node_lp_opts(&capped, None).deadline, Some(early));
        checked += 1;
        // Both set: the EARLIER wins, in either order. Taking the later would let
        // one of the two budgets be silently ignored — the bug being fixed.
        assert_eq!(node_lp_opts(&capped, Some(late)).deadline, Some(early));
        checked += 1;
        let capped_late = SimplexOptions {
            deadline: Some(late),
            ..base.clone()
        };
        assert_eq!(
            node_lp_opts(&capped_late, Some(early)).deadline,
            Some(early)
        );
        checked += 1;

        // Every other field is carried through untouched: this composes a
        // deadline, it does not reset the caller's tuning (a silent revert of
        // `cold_dual_start` or `max_iter` here would be invisible in a bound).
        let tuned = SimplexOptions {
            max_iter: 12345,
            tol: 1e-9,
            cold_dual_start: true,
            warm_stall_guard: false,
            ..base.clone()
        };
        let out = node_lp_opts(&tuned, Some(late));
        assert_eq!(out.max_iter, 12345);
        assert_eq!(out.tol, 1e-9);
        assert!(out.cold_dual_start);
        assert!(!out.warm_stall_guard);
        checked += 4;

        assert_eq!(checked, 9, "probe fired on every arm");
    }

    /// The other half of the chain: an expired deadline reaching the LP must come
    /// back as `IterLimit`, and the tree must treat that as UNDECIDED — branch the
    /// region with the parent's valid bound inherited, never fathom it.
    ///
    /// This is why threading the deadline in is sound rather than merely
    /// convenient: cutting an LP short can only make the reported bound looser.
    /// If a future change ever routed `IterLimit` to a fathom, this fails.
    #[test]
    fn an_interrupted_node_lp_is_undecided_never_fathomed() {
        let spec = xy_min_spec();
        // Expired per-LP deadline, no tree deadline: the tree's between-node check
        // passes, so nodes really are processed and every node LP is cut at entry
        // (the iteration loop polls at `_iter == 0`).
        let opts = SimplexOptions {
            deadline: Some(Instant::now() - Duration::from_secs(1)),
            ..SimplexOptions::default()
        };
        let cfg = SpatialTreeConfig {
            max_nodes: 8,
            gap_tol: 1e-5,
            ..SpatialTreeConfig::default()
        };
        let res = solve_spatial_tree(&spec, &cfg, &opts);

        assert!(
            res.node_count > 0,
            "no node was processed, probe never fired"
        );
        assert!(
            res.n_undecided >= 1,
            "an LP cut by its deadline was not recorded undecided: {res:?}"
        );
        // Never a false certificate off an interrupted LP.
        assert_ne!(
            res.status,
            TreeStatus::Optimal,
            "claimed Optimal with every node LP interrupted: {res:?}"
        );
        // The bound stays a valid lower bound for the true optimum (2.0).
        assert!(
            res.bound <= 2.0 + 1e-6,
            "interrupted search reported bound {} above the true optimum 2.0",
            res.bound
        );

        // Control: the identical search with no deadline decides its nodes, so the
        // assertions above are about the deadline and not about this spec.
        let clean = solve_spatial_tree(&spec, &cfg, &SimplexOptions::default());
        assert_eq!(
            clean.n_undecided, 0,
            "control run should decide every node LP: {clean:?}"
        );
    }

    /// #927 regression: ONLY a certified `Infeasible` licenses the tree to treat a
    /// region as empty.
    ///
    /// The false certificate on ex1252 (a certified `optimal` 216826.52 against a
    /// true optimum of 128893.74, under `DISCOPT_INTEGER_MULTILINEAR_REFORM` +
    /// `DISCOPT_MULTILINEAR_COUPLING_RLT`) came from exactly one line: the tree
    /// fathomed on `status != Optimal`, and the node LP had returned `Numerical`.
    /// That reformulation lifts `x^3` for an `x` near 2950, so the cubic's tangent
    /// envelope rows carry coefficients near `3*2950^2 = 2.6e7` and right-hand sides
    /// near `7.7e10` formed by a cancelling subtraction; at the box corner `x = ui`
    /// the row and the auxiliary column's own upper bound `ui^3` are two different
    /// roundings of the same number, leaving an absolute residual near 1e-5 that no
    /// LP feasibility tolerance can absorb. The simplex said so honestly
    /// (`Numerical`, not `Infeasible`) and was overruled.
    ///
    /// Pinning the mapping is what keeps that from coming back: a region is empty
    /// only when something PROVED it empty.
    #[test]
    fn only_certified_infeasible_proves_a_region_empty() {
        assert_eq!(verdict_for(LpStatus::Optimal), NodeVerdict::Bound);
        assert_eq!(verdict_for(LpStatus::Infeasible), NodeVerdict::EmptyRegion);
        for undecided in [
            LpStatus::Numerical,
            LpStatus::IterLimit,
            LpStatus::Unbounded,
        ] {
            assert_eq!(
                verdict_for(undecided),
                NodeVerdict::Undecided,
                "{undecided:?} is not a proof of emptiness and must not fathom"
            );
        }
    }
}

#[cfg(test)]
mod gap_criterion_tests {
    use super::*;

    #[test]
    fn gap_closed_is_the_absolute_test_it_consolidates() {
        // `gap_closed` exists so the fathoming tests, the node-limit exit and
        // the terminal `Optimal` verdict cannot drift apart. It must therefore
        // stay EXACTLY the `bound >= inc - gap_tol` each of them spelled out
        // inline before, across the whole range of magnitudes -- a helper that
        // quietly differs from the four sites it replaced is worse than none.
        // (#1263's extra clause is disabled by the defaults used here.)
        let config = SpatialTreeConfig {
            gap_tol: 1e-4,
            ..Default::default()
        };
        let mut checked = 0usize;
        for &inc in &[-1e6, -1.0, -1e-8, 0.0, 1e-8, 1.0, 1e6] {
            for &gap in &[0.0, 1e-9, 1e-5, 1e-4, 1.1e-4, 1.0] {
                let bound = inc - gap;
                assert_eq!(
                    gap_closed(bound, inc, &config),
                    bound >= inc - config.gap_tol,
                    "inc={inc} gap={gap}"
                );
                checked += 1;
            }
        }
        assert_eq!(checked, 42, "probe ran {checked} comparisons");
    }

    #[test]
    fn relative_clause_rejects_small_magnitude_open_gaps() {
        // #1263: with `gap_tol = 1e-4` alone, st_z closed at incumbent 2.727e-5
        // over a bound of -5.6e-10 (true optimum 0). The conjoined clause is the
        // Python `_gap_values_converged` test: absolute 1e-6 OR relative 1e-4.
        let config = SpatialTreeConfig {
            gap_tol: 1e-4,
            rel_gap_tol: 1e-4,
            abs_gap_tol: 1e-6,
            ..Default::default()
        };
        let cases: [(f64, f64, bool); 8] = [
            (2.727e-5, -5.6e-10, false),       // st_z
            (0.1, 0.09995, false),             // relative 5e-4
            (-0.68607228, -0.68616931, false), // mathopt5_8 shape
            (4e-12, 0.0, true),                // absolute arm
            (0.1, 0.1 - 5e-7, true),           // absolute arm
            (100.0, 100.0 - 5e-5, true),       // |inc| > 1: unchanged
            (-1e6, -1e6 - 50.0, false),        // old absolute test already open
            (5.0, 5.0 - 5e-5, true),           // |inc| > 1: unchanged
        ];
        let mut checked = 0usize;
        for (inc, bound, want) in cases {
            assert_eq!(
                gap_closed(bound, inc, &config),
                want,
                "inc={inc} bound={bound}"
            );
            // The clause only ever tightens the absolute test.
            if want {
                assert!(bound >= inc - config.gap_tol);
            }
            checked += 1;
        }
        // At |inc| >= 1 with rel_gap_tol >= gap_tol the result is unchanged.
        for &inc in &[-1e6, -1.0, 1.0, 1e6] {
            for &gap in &[0.0, 1e-9, 1e-5, 1e-4, 1.1e-4, 1.0] {
                let bound = inc - gap;
                assert_eq!(
                    gap_closed(bound, inc, &config),
                    bound >= inc - config.gap_tol,
                    "inc={inc} gap={gap}"
                );
                checked += 1;
            }
        }
        assert_eq!(checked, 32, "probe ran {checked} comparisons");
    }

    #[test]
    fn conjoined_clause_never_closes_what_the_absolute_test_leaves_open() {
        // The #1260 hazard was a relative DISJUNCT loosening the fathom. #1263's
        // clause is a conjunct: for any `rel_gap_tol` / `abs_gap_tol` it may only
        // close a subset of what the absolute test alone closes.
        let mut checked = 0usize;
        for &inc in &[-1e5, -1.0, -1e-3, 0.0, 1e-3, 1.0, 1e5] {
            for &gap in &[0.0, 1e-9, 1e-7, 1e-5, 1e-4, 1e-2, 1.0, 11.0] {
                for &rel in &[0.0, 1e-9, 1e-4, 1.0, f64::INFINITY] {
                    for &abs in &[0.0, 1e-6, 1e-4, 10.0] {
                        let bound = inc - gap;
                        let base = SpatialTreeConfig {
                            gap_tol: 1e-4,
                            ..Default::default()
                        };
                        let conj = SpatialTreeConfig {
                            gap_tol: 1e-4,
                            rel_gap_tol: rel,
                            abs_gap_tol: abs,
                            ..Default::default()
                        };
                        if gap_closed(bound, inc, &conj) {
                            assert!(
                                gap_closed(bound, inc, &base),
                                "inc={inc} gap={gap} rel={rel} abs={abs}"
                            );
                        }
                        checked += 1;
                    }
                }
            }
        }
        assert_eq!(checked, 7 * 8 * 5 * 4, "probe ran {checked} comparisons");
    }

    #[test]
    fn a_tighter_gap_tol_never_widens_the_fathom() {
        // The #1260 review finding, pinned at the level it happened. A relative
        // second arm was briefly added here so this route would match the Python
        // tree's disjunction; on a large objective it turned a caller's TIGHTER
        // absolute tolerance into a fathom orders of magnitude LOOSER
        // (1e-9 requested, 10.0 effective at |inc| ~ 1e5). The criterion must be
        // monotone in `gap_tol`: shrinking it can only ever close fewer regions.
        let mut checked = 0usize;
        for &inc in &[-1e5, -1.0, 0.0, 1.0, 1e5] {
            for &gap in &[0.0, 1e-9, 1e-7, 1e-4, 1e-2, 1.0, 11.0] {
                let bound = inc - gap;
                let loose = SpatialTreeConfig {
                    gap_tol: 1e-4,
                    ..Default::default()
                };
                let tight = SpatialTreeConfig {
                    gap_tol: 1e-9,
                    ..Default::default()
                };
                if gap_closed(bound, inc, &tight) {
                    assert!(
                        gap_closed(bound, inc, &loose),
                        "tightening closed a region the looser tolerance does not: \
                         inc={inc} gap={gap}"
                    );
                }
                checked += 1;
            }
        }
        assert_eq!(checked, 35, "probe ran {checked} comparisons");
    }

    /// #1656: the relative arm is judged on `inc + obj_offset`, the objective the
    /// caller is handed. The internal pair of the `(xy-1)^2` least squares
    /// (-1.102645878, -1.102743252; offset 1.13) is 8.8e-5 relative internally but
    /// 3.6e-3 relative once published, so it must stay open.
    #[test]
    fn relative_clause_uses_published_objective() {
        let base = SpatialTreeConfig {
            gap_tol: 1e-4,
            rel_gap_tol: 1e-4,
            abs_gap_tol: 1e-6,
            ..Default::default()
        };
        let with_off = SpatialTreeConfig {
            obj_offset: 1.13,
            ..base
        };
        let (inc, bound) = (-1.102645878110912, -1.1027432524576144);
        assert!(gap_closed(bound, inc, &base), "internal scale closes it");
        assert!(
            !gap_closed(bound, inc, &with_off),
            "published scale keeps it open"
        );
        // A gap within 1e-4 of the published 0.0273541 still closes.
        assert!(gap_closed(inc - 2.0e-6, inc, &with_off));
        // The offset only rescales the relative arm: the absolute arms ignore it.
        assert!(gap_closed(inc - 5e-7, inc, &with_off));
        assert!(!gap_closed(inc - 2e-4, inc, &with_off));
    }

    /// #1522: with no hook, `solve_spatial_tree_with_hook` is `solve_spatial_tree`
    /// node-for-node — the incumbent re-check was moved out of the feasible branch,
    /// and this pins that the move changed nothing.
    #[test]
    fn no_hook_is_node_for_node_identical() {
        let spec = super::tests::xy_min_spec();
        let cfg = SpatialTreeConfig {
            max_nodes: 5000,
            gap_tol: 1e-5,
            ..SpatialTreeConfig::default()
        };
        let a = solve_spatial_tree(&spec, &cfg, &SimplexOptions::default());
        let b = solve_spatial_tree_with_hook(&spec, &cfg, &SimplexOptions::default(), None);
        assert_eq!(a.status, b.status);
        assert_eq!(a.node_count, b.node_count);
        assert_eq!(a.incumbent, b.incumbent);
        assert_eq!(a.bound, b.bound);
        assert_eq!(
            (b.n_primal_hook_calls, b.n_primal_hook_improvements),
            (0, 0)
        );
    }

    /// `max x*y s.t. x + y <= 3` over `[0,2]^2` as `min -w`, optimum -2.25 at
    /// (1.5, 1.5). The root McCormick LP sits at `w = 3`, not tight, so the kernel
    /// has to branch before it can accept a point of its own.
    fn xy_max_spec() -> SpatialKernelSpec {
        SpatialKernelSpec {
            n_cols: 3,
            n_orig: 2,
            c: vec![0.0, 0.0, -1.0],
            integrality: vec![false, false, false],
            global_lo: vec![0.0, 0.0, -1e20],
            global_hi: vec![2.0, 2.0, 1e20],
            fixed_rows: vec![crate::bnb::spatial_kernel::FixedRow {
                cols: vec![0, 1],
                coeffs: vec![1.0, 1.0],
                rhs: 3.0,
            }],
            terms: vec![EnvTerm::Bilinear { i: 0, j: 1, w: 2 }],
            blf_terms: vec![],
            obbt_candidates: vec![0, 1],
        }
    }

    /// #1656: an [`IncumbentValueFn`] that echoes `cᵀx` is node-for-node the
    /// unpriced search; one that raises the value keeps the incumbent honest;
    /// `Reject` keeps every LP point out; `Decline` stops at the first tight point.
    #[test]
    fn incumbent_value_fn_answers() {
        let spec = super::tests::xy_min_spec();
        let cfg = SpatialTreeConfig {
            max_nodes: 5000,
            gap_tol: 1e-5,
            ..SpatialTreeConfig::default()
        };
        let opts = SimplexOptions::default();
        let base = solve_spatial_tree(&spec, &cfg, &opts);

        let mut echo = |_x: &[f64], lp: f64| IncumbentPrice::Value(lp);
        let e = solve_spatial_tree_with_hooks(&spec, &cfg, &opts, None, Some(&mut echo));
        assert_eq!((e.status, e.node_count), (base.status, base.node_count));
        assert_eq!((e.incumbent, e.bound), (base.incumbent, base.bound));
        assert!(
            e.n_incumbent_value_calls >= 1,
            "the function was never consulted"
        );
        assert_eq!(e.n_incumbent_value_raised, 0);

        let mut raise = |_x: &[f64], lp: f64| IncumbentPrice::Value(lp + 1.0);
        let r = solve_spatial_tree_with_hooks(&spec, &cfg, &opts, None, Some(&mut raise));
        assert!(r.n_incumbent_value_raised >= 1);
        // Every accepted value is some tight point's `cᵀx + 1`, and a tight point's
        // `cᵀx` is the optimum up to lifted slack.
        if let (Some(ri), Some(bi)) = (r.incumbent, base.incumbent) {
            assert!(
                ri >= bi + 1.0 - 1e-4,
                "a raised price was not kept: {ri} vs {bi}"
            );
        }

        let mut reject = |_x: &[f64], _lp: f64| IncumbentPrice::Reject;
        let j = solve_spatial_tree_with_hooks(&spec, &cfg, &opts, None, Some(&mut reject));
        assert!(
            j.incumbent.is_none(),
            "a rejected point became the incumbent"
        );

        let mut decline = |_x: &[f64], _lp: f64| IncumbentPrice::Decline;
        let d = solve_spatial_tree_with_hooks(&spec, &cfg, &opts, None, Some(&mut decline));
        assert_eq!(d.status, TreeStatus::Declined);
        assert_eq!(d.n_incumbent_value_calls, 1);
        assert!(d.node_count <= base.node_count);
    }

    /// #1938: a pricer that rejects a point whose binary sits within `int_tol` of
    /// an integer but whose integral realisation violates a big-M row must not
    /// stall the tree. `min -y*v - x + c*z + w` s.t. `x - M*z <= 0`,
    /// `x - w <= eps`, `y + v = 2`: the node LPs return `z = eps/M = 1e-6`, and
    /// spatial branching keeps that value in both children. Branching the
    /// near-integral `z` at its floor/ceil excludes it; the tree then certifies
    /// the true optimum `-1 - eps + c = -1.005` at `z = 1`.
    #[test]
    fn rejected_near_integral_point_branches_the_integer() {
        use crate::bnb::spatial_kernel::FixedRow;
        let (eps, big_m, c) = (1e-2, 1e4, 5e-3);
        // cols: x, w, z (binary), y, v, t = y*v
        let spec = SpatialKernelSpec {
            n_cols: 6,
            n_orig: 5,
            c: vec![-1.0, 1.0, c, 0.0, 0.0, -1.0],
            integrality: vec![false, false, true, false, false, false],
            global_lo: vec![0.0, 0.0, 0.0, 0.0, 0.0, -1e20],
            global_hi: vec![big_m, big_m, 1.0, 2.0, 2.0, 1e20],
            fixed_rows: vec![
                FixedRow {
                    cols: vec![0, 2],
                    coeffs: vec![1.0, -big_m],
                    rhs: 0.0,
                },
                FixedRow {
                    cols: vec![0, 1],
                    coeffs: vec![1.0, -1.0],
                    rhs: eps,
                },
                FixedRow {
                    cols: vec![3, 4],
                    coeffs: vec![1.0, 1.0],
                    rhs: 2.0,
                },
                FixedRow {
                    cols: vec![3, 4],
                    coeffs: vec![-1.0, -1.0],
                    rhs: -2.0,
                },
            ],
            terms: vec![EnvTerm::Bilinear { i: 3, j: 4, w: 5 }],
            blf_terms: vec![],
            obbt_candidates: vec![],
        };
        let cfg = SpatialTreeConfig {
            max_nodes: 20_000,
            gap_tol: 1e-6,
            ..SpatialTreeConfig::default()
        };
        // The integral realisation the caller verifies: `z` snapped.
        let mut rejected = 0usize;
        let mut snap_check = |x: &[f64], lp: f64| {
            if x[0] - big_m * x[2].round() > 1e-6 {
                rejected += 1;
                IncumbentPrice::Reject
            } else {
                IncumbentPrice::Value(lp)
            }
        };
        let r = solve_spatial_tree_with_hooks(
            &spec,
            &cfg,
            &SimplexOptions::default(),
            None,
            Some(&mut snap_check),
        );
        assert!(
            rejected >= 1,
            "the probe never produced a near-integral point"
        );
        assert_eq!(r.status, TreeStatus::Optimal, "{r:?}");
        let inc = r.incumbent.expect("no incumbent");
        let opt = -1.0 - eps + c;
        assert!((inc - opt).abs() < 1e-5, "incumbent {inc} vs optimum {opt}");
        assert!(
            r.bound <= opt + 1e-6,
            "bound {} above optimum {opt}",
            r.bound
        );
        assert!(
            (r.incumbent_x[2] - 1.0).abs() < 1e-9,
            "z = {}",
            r.incumbent_x[2]
        );
    }

    /// #1522: a hook handing back the optimum at the root is adopted, recorded as
    /// the incumbent point, never moves the bound above the optimum, and costs no
    /// nodes; a hook that finds nothing is consulted but leaves the search
    /// node-for-node unchanged.
    #[test]
    fn primal_hook_is_adopted_only_on_improvement() {
        let spec = xy_max_spec();
        let cfg = SpatialTreeConfig {
            max_nodes: 5000,
            gap_tol: 1e-5,
            ..SpatialTreeConfig::default()
        };
        let base = solve_spatial_tree(&spec, &cfg, &SimplexOptions::default());
        assert!(
            base.node_count > 1,
            "the control must need to branch: {base:?}"
        );

        let mut calls = 0usize;
        let mut good = |_x: &[f64], _inc: Option<f64>| -> Option<(f64, Vec<f64>)> {
            calls += 1;
            Some((-2.25, vec![1.5, 1.5, 2.25])) // feasible and optimal
        };
        let res =
            solve_spatial_tree_with_hook(&spec, &cfg, &SimplexOptions::default(), Some(&mut good));
        assert_eq!(res.status, TreeStatus::Optimal);
        let inc = res.incumbent.expect("incumbent");
        assert!((inc + 2.25).abs() < 1e-9, "incumbent {inc} != -2.25");
        assert!(
            res.bound <= -2.25 + 1e-6,
            "bound {} above the optimum",
            res.bound
        );
        assert!(res.n_primal_hook_improvements >= 1);
        assert_eq!(res.n_primal_hook_calls, calls);
        assert!(
            res.node_count <= base.node_count,
            "an optimal cutoff from node 1 cost nodes: {} > {}",
            res.node_count,
            base.node_count
        );

        let mut nothing = |_x: &[f64], _inc: Option<f64>| -> Option<(f64, Vec<f64>)> { None };
        let res_w = solve_spatial_tree_with_hook(
            &spec,
            &cfg,
            &SimplexOptions::default(),
            Some(&mut nothing),
        );
        assert!(
            res_w.n_primal_hook_calls >= 1,
            "the hook was never consulted"
        );
        assert_eq!(res_w.n_primal_hook_improvements, 0);
        assert_eq!(res_w.node_count, base.node_count);
        assert_eq!(res_w.incumbent, base.incumbent);
        assert_eq!(res_w.bound, base.bound);
    }

    /// #1935: `warm_start_children` changes how node LPs START, never what the
    /// search may conclude. On a nonconvex box-constrained problem with several
    /// lifted terms (two bilinear products, a square, a linear coupling row), the
    /// warm and cold searches must both certify, agree on the optimum to the gap,
    /// keep each bound below the other arm's incumbent, and the warm arm must
    /// actually have warm-started (the probe-fired check, CLAUDE.md §6). Off, no
    /// node may warm-start.
    #[test]
    fn warm_start_children_certifies_the_cold_optimum() {
        use crate::bnb::spatial_kernel::FixedRow;
        // min  w01 - 2 w12 + s0 - x2   with w01 = x0*x1, w12 = x1*x2, s0 = x0^2
        // s.t. x0 + x1 + x2 <= 2.5,   x in [-1, 2]^3.
        let spec = SpatialKernelSpec {
            n_cols: 6,
            n_orig: 3,
            c: vec![0.0, 0.0, -1.0, 1.0, -2.0, 1.0],
            integrality: vec![false; 6],
            global_lo: vec![-1.0, -1.0, -1.0, -1e20, -1e20, -1e20],
            global_hi: vec![2.0, 2.0, 2.0, 1e20, 1e20, 1e20],
            fixed_rows: vec![FixedRow {
                cols: vec![0, 1, 2],
                coeffs: vec![1.0, 1.0, 1.0],
                rhs: 2.5,
            }],
            terms: vec![
                EnvTerm::Bilinear { i: 0, j: 1, w: 3 },
                EnvTerm::Bilinear { i: 1, j: 2, w: 4 },
                EnvTerm::Monomial { i: 0, s: 5, p: 2 },
            ],
            blf_terms: vec![],
            obbt_candidates: vec![],
        };
        let base = SpatialTreeConfig {
            max_nodes: 20_000,
            gap_tol: 1e-6,
            ..SpatialTreeConfig::default()
        };
        let warm_cfg = SpatialTreeConfig {
            warm_start_children: true,
            ..base
        };
        let opts = SimplexOptions::default();
        let cold = solve_spatial_tree(&spec, &base, &opts);
        let warm = solve_spatial_tree(&spec, &warm_cfg, &opts);
        let mut checked = 0usize;

        assert_eq!(cold.status, TreeStatus::Optimal, "{cold:?}");
        assert_eq!(warm.status, TreeStatus::Optimal, "{warm:?}");
        checked += 2;
        assert!(cold.node_count > 3, "too few nodes to exercise inheritance");
        assert_eq!(cold.n_warm_starts, 0, "flag off must never warm-start");
        assert_eq!(cold.n_warm_shape_mismatch, 0);
        checked += 3;
        assert!(warm.n_warm_starts > 0, "warm arm never warm-started");
        // Every node but the root has an inherited basis.
        assert!(warm.n_warm_starts + warm.n_warm_shape_mismatch < warm.node_count);
        checked += 2;
        let (ic, iw) = (cold.incumbent.unwrap(), warm.incumbent.unwrap());
        assert!((ic - iw).abs() <= 2e-6, "incumbents {ic} vs {iw}");
        // Each arm's certified bound is a valid lower bound: never above an
        // incumbent either arm found.
        assert!(warm.bound <= ic + 1e-9 && warm.bound <= iw + 1e-9);
        assert!(cold.bound <= ic + 1e-9 && cold.bound <= iw + 1e-9);
        checked += 3;
        // The incumbent is feasible for the original problem.
        let x = &warm.incumbent_x;
        assert!(x[0] + x[1] + x[2] <= 2.5 + 1e-6);
        let obj = x[0] * x[1] - 2.0 * x[1] * x[2] + x[0] * x[0] - x[2];
        assert!(
            (obj - iw).abs() <= 1e-4,
            "objective at point {obj} vs incumbent {iw}"
        );
        checked += 2;
        assert_eq!(
            checked, 12,
            "warm-start tree probe did not run every assertion"
        );
    }

    /// #1935: a node whose LP decides nothing hands its children the basis it was
    /// itself started from. Forcing undecided LPs with a tiny pivot cap, every
    /// LP-solved node except the root must then be warm-started: the root solved,
    /// and each later node inherits from its nearest solved ancestor. Before this,
    /// an undecided node handed down nothing and its whole subtree solved cold.
    #[test]
    fn undecided_nodes_pass_their_inherited_basis_down() {
        let spec = probe_spec_1935();
        let cfg = SpatialTreeConfig {
            max_nodes: 4_000,
            gap_tol: 1e-6,
            warm_start_children: true,
            ..SpatialTreeConfig::default()
        };
        let mut checked = 0usize;
        let mut exercised = 0usize;
        for max_iter in [3usize, 4, 5, 6, 8] {
            let opts = SimplexOptions {
                max_iter,
                ..SimplexOptions::default()
            };
            let res = solve_spatial_tree(&spec, &cfg, &opts);
            if res.root_status != "bounded" {
                continue;
            }
            assert_eq!(
                res.n_warm_starts + res.n_warm_shape_mismatch,
                res.n_lp_solves - 1,
                "max_iter={max_iter}: a solved node started cold: {res:?}"
            );
            checked += 1;
            if res.n_undecided > 0 {
                exercised += 1;
            }
        }
        assert!(checked > 0, "no run solved its root");
        assert!(exercised > 0, "no run produced an undecided node LP");
    }

    /// #1995: the pre-propagation re-solve keeps the certificate honest where node
    /// LPs fail. A small `max_iter` makes node LPs end `IterLimit` (undecided); with
    /// the retry ON every run must still report a bound `<=` the true optimum and
    /// never certify a wrong value, and the retry must actually run in some of them.
    #[test]
    fn undecided_retry_never_certifies_a_wrong_value() {
        let spec = probe_spec_1935();
        let base = SpatialTreeConfig {
            max_nodes: 4_000,
            gap_tol: 1e-6,
            ..SpatialTreeConfig::default()
        };
        let reference = solve_spatial_tree(&spec, &base, &SimplexOptions::default());
        assert_eq!(reference.status, TreeStatus::Optimal, "{reference:?}");
        let opt = reference.incumbent.expect("the control run certifies");
        let cfg = SpatialTreeConfig {
            undecided_unpropagated_retry: true,
            ..base
        };
        let mut checked = 0usize;
        let mut retried = 0usize;
        for max_iter in [2usize, 3, 4, 5, 6, 8] {
            let opts = SimplexOptions {
                max_iter,
                ..SimplexOptions::default()
            };
            let res = solve_spatial_tree(&spec, &cfg, &opts);
            assert!(
                res.bound <= opt + 1e-6,
                "max_iter={max_iter}: bound {} above the optimum {opt}",
                res.bound
            );
            if res.status == TreeStatus::Optimal {
                let inc = res.incumbent.expect("optimal carries an incumbent");
                assert!(
                    (inc - opt).abs() <= 1e-5,
                    "max_iter={max_iter}: {inc} vs {opt}"
                );
            }
            assert!(res.n_prop_retry_decided <= res.n_prop_retries);
            checked += 1;
            retried += res.n_prop_retries;
        }
        assert!(checked > 0);
        assert!(retried > 0, "no run re-solved an undecided node");
    }

    /// #1995: where every node LP decides, the retry flag changes nothing.
    #[test]
    fn undecided_retry_is_neutral_when_every_node_decides() {
        for spec in [
            super::tests::xy_min_spec(),
            probe_spec_1935(),
            convex_iqp_spec(),
        ] {
            let off = SpatialTreeConfig {
                max_nodes: 4_000,
                gap_tol: 1e-6,
                ..SpatialTreeConfig::default()
            };
            let on = SpatialTreeConfig {
                undecided_unpropagated_retry: true,
                ..off
            };
            let a = solve_spatial_tree(&spec, &off, &SimplexOptions::default());
            let b = solve_spatial_tree(&spec, &on, &SimplexOptions::default());
            assert_eq!(a.n_undecided, 0, "control must decide every node: {a:?}");
            assert_eq!(b.n_prop_retries, 0);
            assert_eq!(a.status, b.status);
            assert_eq!(a.node_count, b.node_count);
            assert_eq!(a.n_lp_solves, b.n_lp_solves);
            assert_eq!(a.bound.to_bits(), b.bound.to_bits());
            assert_eq!(a.incumbent.map(f64::to_bits), b.incumbent.map(f64::to_bits));
        }
    }

    fn probe_spec_1935() -> SpatialKernelSpec {
        use crate::bnb::spatial_kernel::FixedRow;
        SpatialKernelSpec {
            n_cols: 6,
            n_orig: 3,
            c: vec![0.0, 0.0, -1.0, 1.0, -2.0, 1.0],
            integrality: vec![false; 6],
            global_lo: vec![-1.0, -1.0, -1.0, -1e20, -1e20, -1e20],
            global_hi: vec![2.0, 2.0, 2.0, 1e20, 1e20, 1e20],
            fixed_rows: vec![FixedRow {
                cols: vec![0, 1, 2],
                coeffs: vec![1.0, 1.0, 1.0],
                rhs: 2.5,
            }],
            terms: vec![
                EnvTerm::Bilinear { i: 0, j: 1, w: 3 },
                EnvTerm::Bilinear { i: 1, j: 2, w: 4 },
                EnvTerm::Monomial { i: 0, s: 5, p: 2 },
            ],
            blf_terms: vec![],
            obbt_candidates: vec![],
        }
    }

    /// #1962: `min x0² + x1² + x0·x1 − 7x0 − 8x1` over integers in `[0, 10]²` with
    /// `x0 + x1 <= 6`. Columns `x0, x1, s0 = x0², s1 = x1², w = x0·x1`; the
    /// objective's quadratic part is positive definite, which the per-term
    /// McCormick relaxation cannot see.
    fn convex_iqp_spec() -> SpatialKernelSpec {
        use crate::bnb::spatial_kernel::BlfTerm;
        SpatialKernelSpec {
            n_cols: 5,
            n_orig: 2,
            c: vec![-7.0, -8.0, 1.0, 1.0, 1.0],
            integrality: vec![true, true, false, false, false],
            global_lo: vec![0.0, 0.0, 0.0, 0.0, 0.0],
            global_hi: vec![10.0, 10.0, 100.0, 100.0, 100.0],
            fixed_rows: vec![FixedRow {
                cols: vec![0, 1],
                coeffs: vec![1.0, 1.0],
                rhs: 6.0,
            }],
            terms: vec![
                EnvTerm::Monomial { i: 0, s: 2, p: 2 },
                EnvTerm::Monomial { i: 1, s: 3, p: 2 },
            ],
            blf_terms: vec![BlfTerm {
                a_cols: vec![0],
                a_coeffs: vec![1.0],
                a_const: 0.0,
                b_cols: vec![1],
                b_coeffs: vec![1.0],
                b_const: 0.0,
                w: 4,
            }],
            obbt_candidates: vec![],
        }
    }

    /// #1990: on a model with integer columns the hook runs only at LP points
    /// integral on every integer column, never twice on one integer assignment
    /// (a kernel-accepted incumbent aside), and a hook that finds nothing leaves
    /// the search node-for-node unchanged.
    #[test]
    fn primal_hook_waits_for_integral_unseen_points() {
        let spec = convex_iqp_spec();
        let cfg = SpatialTreeConfig::default();
        let opts = SimplexOptions::default();
        let base = solve_spatial_tree(&spec, &cfg, &opts);
        assert_eq!(base.status, TreeStatus::Optimal);
        let mut seen: Vec<(Vec<i64>, bool)> = Vec::new();
        let mut rec = |x: &[f64], _inc: Option<f64>| -> Option<(f64, Vec<f64>)> {
            let integral = (0..2).all(|j| (x[j] - x[j].round()).abs() <= 1e-5);
            seen.push(((0..2).map(|j| x[j].round() as i64).collect(), integral));
            None
        };
        let res = solve_spatial_tree_with_hook(&spec, &cfg, &opts, Some(&mut rec));
        assert_eq!(res.node_count, base.node_count);
        assert_eq!(res.incumbent, base.incumbent);
        assert_eq!(res.bound, base.bound);
        assert_eq!(res.n_primal_hook_calls, seen.len());
        assert!(!seen.is_empty(), "the hook was never consulted");
        // Every call is at an integral point (a kernel incumbent is one too).
        let mut checks = 0usize;
        for (_, integral) in &seen {
            assert!(*integral, "hook called at a fractional point: {seen:?}");
            checks += 1;
        }
        // At most one repeat per accepted incumbent (those bypass the dedup).
        let mut keys: Vec<&Vec<i64>> = seen.iter().map(|(k, _)| k).collect();
        keys.sort();
        keys.dedup();
        assert!(seen.len() - keys.len() <= base.node_count, "{seen:?}");
        assert!(checks >= 1);
    }

    /// #1962: the separation loop tightens the root, keeps every bound valid, and
    /// certifies the same optimum as the single-solve tree, cold and warm.
    #[test]
    fn convex_cut_loop_tightens_root_and_certifies_the_same_optimum() {
        let spec = convex_iqp_spec();
        // Brute-force oracle over the integer grid.
        let mut opt = f64::INFINITY;
        for a in 0..=10 {
            for b in 0..=10 {
                let (x, y) = (a as f64, b as f64);
                if x + y <= 6.0 {
                    opt = opt.min(x * x + y * y + x * y - 7.0 * x - 8.0 * y);
                }
            }
        }
        let opts = SimplexOptions::default();
        let mut checked = 0usize;
        for warm in [false, true] {
            let off = SpatialTreeConfig {
                warm_start_children: warm,
                ..SpatialTreeConfig::default()
            };
            let on = SpatialTreeConfig {
                cut_rounds_root: 20,
                cut_rounds_node: 3,
                ..off
            };
            let r_off = solve_spatial_tree(&spec, &off, &opts);
            let r_on = solve_spatial_tree(&spec, &on, &opts);
            assert_eq!(r_off.status, TreeStatus::Optimal, "{r_off:?}");
            assert_eq!(r_on.status, TreeStatus::Optimal, "{r_on:?}");
            assert_eq!(
                r_off.n_cut_forms + r_off.n_cut_rounds + r_off.n_cuts_added,
                0
            );
            assert_eq!(r_on.n_cut_forms, 1, "the objective form was not certified");
            assert!(r_on.n_cut_rounds > 0 && r_on.n_cuts_added > 0, "{r_on:?}");
            checked += 5;
            for r in [&r_off, &r_on] {
                assert!((r.incumbent.unwrap() - opt).abs() <= 1e-6, "{r:?} vs {opt}");
                assert!(r.bound <= opt + 1e-9 && r.root_bound <= opt + 1e-9, "{r:?}");
                checked += 2;
            }
            // The cuts close most of the root gap the envelopes leave.
            assert!(
                r_on.root_bound > r_off.root_bound + 0.5 * (opt - r_off.root_bound),
                "root {} (off {}) vs optimum {opt}",
                r_on.root_bound,
                r_off.root_bound
            );
            assert!(r_on.node_count <= r_off.node_count, "{r_on:?} vs {r_off:?}");
            checked += 2;
            if warm {
                // The extended / restricted bases always fit their node LPs.
                assert!(r_on.n_warm_starts > 0);
                assert_eq!(r_on.n_warm_shape_mismatch, 0);
                checked += 2;
            }
        }
        assert_eq!(
            checked,
            2 * 11 + 2,
            "cut-loop probe did not run every assertion"
        );
    }

    /// #1962: a child inherits exactly the cuts binding at its parent, and the
    /// parent's basis restricted to them stays square and consistent.
    #[test]
    fn binding_cuts_drops_basic_cut_rows_and_their_slacks() {
        use crate::lp::basis::{AT_LOWER, AT_UPPER, BASIC};
        let cut = |r: f64| FixedRow {
            cols: vec![0],
            coeffs: vec![1.0],
            rhs: r,
        };
        let cuts = Rc::new(vec![cut(1.0), cut(2.0), cut(3.0)]);
        // n_cols = 2, two base rows (slacks 2, 3) and three cut rows (slacks 4..7).
        let basis = Basis {
            col_status: vec![BASIC, AT_LOWER, AT_LOWER, BASIC, BASIC, AT_UPPER, BASIC],
            basic_vars: vec![0, 3, 4, 6, 2],
        };
        // Row 2 (slack 4) and row 4 (slack 6) are basic, so only the middle cut is binding.
        let (kept, b) = binding_cuts(2, &cuts, basis.clone(), true);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].rhs, 2.0);
        let b = b.unwrap();
        assert_eq!(
            b.col_status,
            vec![BASIC, AT_LOWER, AT_LOWER, BASIC, AT_UPPER]
        );
        assert_eq!(b.basic_vars, vec![0, 3, 2]);
        assert_eq!(b.col_status.len(), 2 + b.basic_vars.len());
        // Without a warm start the cuts are filtered the same way.
        let (kept2, b2) = binding_cuts(2, &cuts, basis, false);
        assert_eq!(kept2.len(), 1);
        assert!(b2.is_none());
        // A basis that does not span the rows hands the cuts down unfiltered.
        let short = Basis {
            col_status: vec![BASIC; 3],
            basic_vars: vec![0],
        };
        let (kept3, b3) = binding_cuts(2, &cuts, short, true);
        assert_eq!(kept3.len(), 3);
        assert!(b3.is_none());
        // Every cut binding (all slacks nonbasic): the basis passes through
        // unchanged. n_cols = 5, two base rows and the three cuts.
        let all = Basis {
            col_status: [vec![BASIC; 5], vec![AT_LOWER; 5]].concat(),
            basic_vars: vec![0, 1, 2, 3, 4],
        };
        let (kept4, b4) = binding_cuts(5, &cuts, all.clone(), true);
        assert_eq!(kept4.len(), 3);
        assert_eq!(b4.unwrap().col_status, all.col_status);
    }

    /// #1962: the integral-point probe finds a tight point from an integral but
    /// untight LP point and changes no bound's validity.
    #[test]
    fn int_fix_probe_accepts_only_tight_points() {
        let spec = convex_iqp_spec();
        let opts = SimplexOptions::default();
        let base = SpatialTreeConfig::default();
        let probe = SpatialTreeConfig {
            int_fix_probe: true,
            ..base
        };
        let r0 = solve_spatial_tree(&spec, &base, &opts);
        let r1 = solve_spatial_tree(&spec, &probe, &opts);
        assert_eq!(r0.n_int_probes, 0);
        assert_eq!(r1.status, TreeStatus::Optimal, "{r1:?}");
        assert!(r1.n_int_probes > 0, "probe never fired: {r1:?}");
        assert!((r1.incumbent.unwrap() - r0.incumbent.unwrap()).abs() <= 1e-6);
        // The accepted point is genuinely feasible: integral, and its objective
        // recomputed from the operands matches the incumbent.
        let x = &r1.incumbent_x;
        assert!((x[0] - x[0].round()).abs() <= 1e-6 && (x[1] - x[1].round()).abs() <= 1e-6);
        let obj = x[0] * x[0] + x[1] * x[1] + x[0] * x[1] - 7.0 * x[0] - 8.0 * x[1];
        assert!(
            (obj - r1.incumbent.unwrap()).abs() <= 1e-6,
            "{obj} vs {r1:?}"
        );
        assert!(x[0] + x[1] <= 6.0 + 1e-9);
    }
}
