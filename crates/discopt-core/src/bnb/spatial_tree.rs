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

use crate::bnb::spatial_kernel::{solve_spatial_node, EnvTerm, SpatialKernelSpec};
use crate::bnb::spatial_propagate::propagate_spec_fixpoint;
use crate::lp::simplex::{LpStatus, SimplexOptions};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
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
}

/// Whether `bound` closes the region against `inc` under the configured gap.
///
/// The single place the gap criterion is spelled out, so the fathoming tests,
/// the node-limit exit and the terminal `Optimal` verdict can never drift apart
/// — which is exactly how a fathom looser than the certificate becomes a false
/// `Optimal`.
///
/// Purely ABSOLUTE, as this kernel has always been. #1243 briefly gave it a
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
    gap <= config.rel_gap_tol * inc.abs().max(bound.abs()).max(1e-10)
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
}

impl Default for SpatialTreeConfig {
    fn default() -> Self {
        Self {
            max_nodes: 100_000,
            deadline: None,
            gap_tol: 1e-6,
            rel_gap_tol: f64::INFINITY,
            abs_gap_tol: 0.0,
            int_tol: 1e-5,
            mccormick_tol: 1e-6,
            min_box_width: 1e-9,
            run_obbt: false,
            run_propagation: true,
            propagation_rounds: 15,
            initial_incumbent: None,
            incumbent_time_extension: None,
            bound_time_extension: None,
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
    // Best-bound frontier: a min-heap on the inherited lower bound. Exploring the
    // lowest-bound region first lifts the global frontier minimum (the reported dual
    // bound) as fast as possible — the key to certifying instances like tanksize
    // whose bound climbs only slowly.
    let mut heap: BinaryHeap<QNode> = BinaryHeap::new();
    heap.push(QNode {
        pb: f64::NEG_INFINITY,
        lo: spec.global_lo.clone(),
        hi: spec.global_hi.clone(),
    });
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
            }
            continue;
        }

        // #1009: hand the node LP the tree's LIVE deadline (post-extension), not
        // `config.deadline` — an extension taken above must reach the LP too, or
        // the extra slice the tree just granted itself is spent on an LP that
        // still cannot be interrupted. See `node_lp_opts`.
        let node_opts = node_lp_opts(opts, deadline);
        let node = solve_spatial_node(spec, &lo, &hi, config.run_obbt, &node_opts);
        n_lp_solves += node.n_lp_solves;
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
            }
            let split = widest_original_col(spec, &lo, &hi, &root_w, config.min_box_width);
            match split {
                Some(j) => {
                    let at = clamp_interior(0.5 * (lo[j] + hi[j]), lo[j], hi[j]);
                    push_children(
                        &mut heap,
                        &lo,
                        &hi,
                        j,
                        at,
                        parent_bound,
                        spec.integrality[j],
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
        // #1236: node 1 IS the root region, and `bound` is the rigorous lower bound
        // just proven for it — the same value that reaches `global_lb_closed` or the
        // children's `pb`. Recorded here (and at the three other arms node 1 can
        // leave by) so the caller can tell a loose root relaxation from a tight one,
        // which the final `bound` alone cannot say.
        if node_count == 1 {
            root_bound = bound;
            root_time_s = t_tree_start.elapsed().as_secs_f64();
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
        // (b) every lifted term McCormick-tight — fixed-width EnvTerms and the
        //     affine-form product (BlfTerm) terms alike.
        let width = |c: usize| hi[c] - lo[c];
        let mut worst_gap = 0.0f64;
        let mut branch_col: Option<usize> = None;
        for t in &spec.terms {
            let (gap, col) = term_gap_and_branch_col(t, x, &width);
            if gap > worst_gap {
                worst_gap = gap;
                branch_col = Some(col);
            }
        }
        for t in &spec.blf_terms {
            let a_val = t.a_const + dot_form(&t.a_cols, &t.a_coeffs, x);
            let b_val = t.b_const + dot_form(&t.b_cols, &t.b_coeffs, x);
            let gap = (x[t.w] - a_val * b_val).abs();
            if gap > worst_gap {
                worst_gap = gap;
                // Spatial-branch the widest operand column across A ∪ B.
                let mut best = None;
                let mut best_w = -1.0f64;
                for &c in t.a_cols.iter().chain(t.b_cols.iter()) {
                    let cw = width(c);
                    if cw > best_w {
                        best_w = cw;
                        best = Some(c);
                    }
                }
                branch_col = best;
            }
        }
        let terms_tight = worst_gap <= config.mccormick_tol;

        if int_ok && terms_tight {
            // Feasible point: accept if it improves the incumbent. The objective is
            // linear over the (now-tight) lifted columns, so `cᵀx` is the true
            // objective at this feasible point.
            let obj = dot(&spec.c, x);
            if incumbent.map(|inc| obj < inc - 1e-12).unwrap_or(true) {
                incumbent = Some(obj);
                incumbent_x = x[..spec.n_cols].to_vec();
            }
            // CLOSE the region ONLY when its rigorous bound certifies that no
            // point in it beats the incumbent by more than the gap. A feasible
            // point does NOT prove the region optimal — with a loose bound the
            // region may hold BETTER points, so it must be branched further
            // (closing here would be a premature fathom → a false certificate).
            let inc_now = incumbent.unwrap();
            if gap_closed(bound, inc_now, config) {
                global_lb_closed = global_lb_closed.min(bound);
                continue;
            }
            // fall through to branching (branch_col may be None: all terms tight —
            // the widest-column fallback below picks the split).
        }

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
        let pick = if let Some(j) = frac_int {
            Some((j, x[j].floor() + 0.5)) // integer branch: <= floor, >= ceil
        } else if let Some(col) = branch_col.filter(|&c| width(c) > config.min_box_width) {
            // Spatial branch at the LP value, pulled to the interior.
            Some((col, clamp_interior(x[col], lo[col], hi[col])))
        } else {
            fallback()
        };
        let Some((split_col, split_at)) = pick else {
            // No branchable column: close with the honest rigorous bound.
            global_lb_closed = global_lb_closed.min(bound);
            continue;
        };

        // Two covering children.
        push_children(
            &mut heap,
            &lo,
            &hi,
            split_col,
            split_at,
            bound,
            spec.integrality[split_col],
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
) {
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
            });
        }
        if hi2[split_col] >= lo2[split_col] - 1e-12 {
            heap.push(QNode {
                pb,
                lo: lo2,
                hi: hi2,
            });
        }
    } else {
        let mut hi1 = hi.to_vec();
        hi1[split_col] = split_at;
        let mut lo2 = lo.to_vec();
        lo2[split_col] = split_at;
        heap.push(QNode {
            pb,
            lo: lo.to_vec(),
            hi: hi1,
        });
        heap.push(QNode {
            pb,
            lo: lo2,
            hi: hi.to_vec(),
        });
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
    fn xy_min_spec() -> SpatialKernelSpec {
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
}
