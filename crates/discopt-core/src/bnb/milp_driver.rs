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

use std::collections::HashSet;

use crate::bnb::branching::VarBranchInfo;
use crate::bnb::node::NodeId;
use crate::bnb::pool::SelectionStrategy;
use crate::bnb::tree_manager::{NodeResult, TreeManager};
use crate::lp::basis::{Basis, AT_LOWER, AT_UPPER, BASIC};
use crate::lp::cover::separate_cover;
use crate::lp::crossover::LpView;
use crate::lp::cut_select::select_cuts;
use crate::lp::gomory::{separate_gomory, GomoryCut};
use crate::lp::simplex::linsolve::{FeralLU, LinearSolver};
use crate::lp::simplex::sparse::SparseCols;
use crate::lp::simplex::{
    solve_lp, solve_lp_cols_scaled, solve_lp_scaled, solve_lp_warm, solve_lp_warm_scaled_csc,
    tighten_bounds, LpStatus, PreparedDual, Scaling, SimplexOptions,
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
    solve_milp_hooked(lp, b, obj_const, opts, None)
}

/// [`solve_milp`] with an optional interactive-debugger hook. When `hook` is
/// `None` this is bit-for-bit identical to a plain solve (all fire-sites
/// short-circuit); the `solve_milp` wrapper above passes `None`.
pub fn solve_milp_hooked(
    lp: &LpView<'_>,
    b: &[f64],
    obj_const: f64,
    opts: &MilpOptions,
    hook: Option<&dyn MilpDebugHook>,
) -> MilpResult {
    crate::profile::init_from_env();
    crate::profile::reset();
    let n = lp.n;
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
            tighten_bounds(lp, b, &is_int_full, opts.simplex.tol)
        };
        if pr.infeasible {
            return MilpResult {
                status: MilpStatus::Infeasible,
                x: vec![0.0; ns],
                obj: f64::INFINITY,
                bound: f64::INFINITY,
                nodes: 0,
                lp_iters: 0,
            };
        }
        (pr.l, pr.u)
    } else {
        (lp.l.to_vec(), lp.u.to_vec())
    };

    let glb = base_l[..ns].to_vec();
    let gub = base_u[..ns].to_vec();
    let mut tm = TreeManager::new(ns, glb, gub, int_info, SelectionStrategy::BestFirst);
    tm.initialize();

    // Working LP, possibly augmented with root cuts. Cuts add rows + slack
    // columns; structural columns [0, ns) are untouched, so the tree's
    // structural bounds still apply unchanged.
    let mut a_w = lp.a.to_vec();
    let mut b_w = b.to_vec();
    let mut c_w = lp.c.to_vec();
    let mut l_w = base_l;
    let mut u_w = base_u;
    let mut m_w = lp.m;
    let mut n_w = n;

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
    let struct_nnz: usize = (0..m_w)
        .map(|i| (0..ns).filter(|&j| a_w[i * n_w + j] != 0.0).count())
        .sum();
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
    if opts.root_cuts > 0 {
        let _t = crate::profile::Timer::new(crate::profile::Phase::RootCutLoop);
        let mut total_cuts = 0usize;
        let mut prev_obj = f64::NEG_INFINITY;
        for _round in 0..opts.cut_rounds {
            if total_cuts >= opts.root_cuts {
                break;
            }
            let root_lp = LpView {
                a: &a_w,
                m: m_w,
                n: n_w,
                c: &c_w,
                l: &l_w,
                u: &u_w,
            };
            let root = {
                let _t = crate::profile::Timer::new(crate::profile::Phase::RootSolve);
                // T2 (docs/dev/sparse-milp-plan.md): solve the root relaxation from
                // CSC via the bit-identical `solve_lp_root_csc`. The CSC is derived
                // from `a_w` here only until T3 removes the dense working matrix; the
                // node solves already run on CSC, so this makes the WHOLE per-round
                // solve path sparse, gated bit-identical by `driver_matches_golden`.
                let root_csc = SparseCols::from_dense(&a_w, m_w, n_w);
                solve_lp_root_csc(&root_csc, m_w, n_w, &c_w, &l_w, &u_w, &b_w, &node_simplex)
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
                separate_cover(
                    &root_lp,
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
                cuts.extend(separate_gomory(
                    &root_lp,
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
            let (nm, nn) = augment_with_cuts(
                &mut a_w,
                &mut b_w,
                &mut c_w,
                &mut l_w,
                &mut u_w,
                &mut is_int_full,
                m_w,
                n_w,
                &new_cuts,
            );
            m_w = nm;
            n_w = nn;
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
        let scaling = Scaling::from_matrix(&a_w, m_w, n_w);
        let (a_s, c_s, b_s) = match &scaling {
            Some(s) => (s.scale_matrix(&a_w), s.scale_c(&c_w), s.scale_b(&b_w)),
            None => (Vec::new(), Vec::new(), Vec::new()),
        };
        // Solve-space matrix/objective/rhs: the scaled copies when scaling, else
        // the originals (borrowed). Node bounds are scaled per node (cheap).
        let (sa, sc, sb): (&[f64], &[f64], &[f64]) = match &scaling {
            Some(_) => (&a_s, &c_s, &b_s),
            None => (&a_w, &c_w, &b_w),
        };
        // CSC of the solve-space matrix, built once and shared by every node solve
        // in this batch (the matrix is constant within a batch). This lifts the
        // per-node `SparseCols::from_dense` O(m·n) rebuild out of the warm solve.
        let csc_batch = SparseCols::from_dense(sa, m_w, n_w);
        // Reduced-cost fixing needs the *unscaled* duals/reduced costs (the integer
        // bound fixing reasons in true objective units), so it gets the CSC of the
        // unscaled working matrix. When the matrix isn't scaled this is exactly
        // `csc_batch` (no second build); otherwise build it once for the batch.
        let csc_unscaled_owned = scaling
            .as_ref()
            .map(|_| SparseCols::from_dense(&a_w, m_w, n_w));
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
        let outputs: Vec<NodeOutput> = {
            let ctx = NodeCtx {
                a_w: &a_w,
                b_w: &b_w,
                c_w: &c_w,
                l_w: &l_w,
                u_w: &u_w,
                scaling: scaling.as_ref(),
                sa,
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
                        .map(|k| solve_node(batch.node_ids[k], &batch.lb[k], &batch.ub[k], &ctx))
                        .collect()
                } else {
                    (0..batch.node_ids.len())
                        .map(|k| solve_node(batch.node_ids[k], &batch.lb[k], &batch.ub[k], &ctx))
                        .collect()
                }
            }
            #[cfg(not(feature = "parallel"))]
            {
                (0..batch.node_ids.len())
                    .map(|k| solve_node(batch.node_ids[k], &batch.lb[k], &batch.ub[k], &ctx))
                    .collect()
            }
        };

        // --- sequential reduce: apply tree mutations in batch order ---
        let mut hit_unbounded = false;
        for (k, out) in outputs.into_iter().enumerate() {
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
                tm.inject_incumbent(cand, cobj);
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
            results.push(out.result);
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

        // Fold this batch's newly-found global cuts into the shared matrix.
        // Stored node bases are extended lazily on their next solve, so children
        // warm-start through the dual simplex from the cut-augmented basis.
        if !pending_cuts.is_empty() {
            let (nm, nn) = augment_with_cuts(
                &mut a_w,
                &mut b_w,
                &mut c_w,
                &mut l_w,
                &mut u_w,
                &mut is_int_full,
                m_w,
                n_w,
                &pending_cuts,
            );
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
    }
}

/// Immutable per-batch context shared by every node evaluation. Holds the
/// working LP (constant within a batch), the options, and a snapshot of the
/// tree's read-only state. All fields are `Sync`, so `solve_node` runs under
/// `rayon`'s `into_par_iter` over the batch.
struct NodeCtx<'a> {
    a_w: &'a [f64],
    b_w: &'a [f64],
    c_w: &'a [f64],
    l_w: &'a [f64],
    u_w: &'a [f64],
    /// Equilibration for the working matrix (shared across the batch), or `None`
    /// when it is well-conditioned. When `Some`, the node LP is solved on the
    /// pre-scaled `sa`/`sc`/`sb` and the solution is unscaled before use.
    scaling: Option<&'a Scaling>,
    /// Solve-space matrix / objective / rhs: scaled copies when `scaling` is
    /// `Some`, else the originals (`a_w`/`c_w`/`b_w`).
    sa: &'a [f64],
    sc: &'a [f64],
    sb: &'a [f64],
    /// CSC view of the solve-space matrix `sa`, built **once per batch** and
    /// shared by every node/strong-branch/dive LP solve in the batch. The working
    /// matrix is constant within a batch (cuts fold in only between batches), so
    /// this removes the per-node `SparseCols::from_dense` rebuild from the warm
    /// solve. Built from `sa`, so it matches whichever (scaled or raw) matrix the
    /// node solves see.
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
fn solve_node(id: NodeId, lb_k: &[f64], ub_k: &[f64], ctx: &NodeCtx<'_>) -> NodeOutput {
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
        let prop_lp = LpView {
            a: ctx.a_w,
            m: ctx.m_w,
            n: ctx.n_w,
            c: ctx.c_w,
            l: &full_l,
            u: &full_u,
        };
        let pr = {
            // cert:T0.3 — time per-node FBBT/constraint propagation.
            let _t = crate::profile::Timer::new(crate::profile::Phase::Fbbt);
            tighten_bounds(&prop_lp, ctx.b_w, ctx.is_int_full, ctx.opts.simplex.tol)
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
    // Original-space LP, used by the cut separators, rounding, and strong
    // branching (which all reason about the model's true coefficients/values).
    let node_lp = LpView {
        a: ctx.a_w,
        m: ctx.m_w,
        n: ctx.n_w,
        c: ctx.c_w,
        l: &full_l,
        u: &full_u,
    };

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
        a: ctx.sa,
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
                            LpStatus::IterLimit | LpStatus::Numerical => {
                                solve_lp_scaled(&solve_lp_view, ctx.sb, ctx.simplex)
                            }
                            _ => warm,
                        }
                    }
                    None => solve_lp_scaled(&solve_lp_view, ctx.sb, ctx.simplex),
                }
            }
        }
    };
    if let Some(s) = ctx.scaling {
        s.unscale_x(&mut sol.x);
    }
    drop(_t_node);
    let sol = sol;

    let mut out = NodeOutput {
        result: NodeResult {
            node_id: id,
            lower_bound: 0.0,
            solution: Vec::new(),
            is_feasible: false,
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
                if let Some((rl, ru)) = reduced_cost_fix(
                    ctx.csc_rc,
                    ctx.m_w,
                    ctx.c_w,
                    &sol.basis,
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
                out.incumbent = try_rounding(
                    &sol.x,
                    ctx.ns,
                    ctx.is_int,
                    ctx.a_w,
                    ctx.b_w,
                    ctx.c_w,
                    ctx.l_w,
                    ctx.u_w,
                    ctx.n_orig_rows,
                    ctx.n_w,
                    ctx.obj_const,
                );
                // Continuous-repair fractional dive at the root: plain rounding
                // never re-solves the continuous variables for the rounded
                // integer assignment, so on weak-relaxation (big-M) models it
                // finds no incumbent at all and the search runs with no
                // bound-based pruning (tree explosion). The dive fixes integers
                // one at a time and re-solves between fixes, repairing the
                // continuous variables and avoiding infeasible combinations. Run
                // only at the root (when rounding found nothing) so its cost is
                // bounded; the warm-started search + cuts take over thereafter.
                if out.incumbent.is_none() && is_root {
                    let _t = crate::profile::Timer::new(crate::profile::Phase::DiveRepair);
                    out.incumbent = try_dive_repair(ctx, lb_k, ub_k, &sol.x, &sol.basis);
                }
            }
            // Node-level cover separation: a fractional node exposes violated
            // covers the root never sees. These are globally valid; the reduce
            // dedups them into the shared pool to tighten the whole tree.
            if !feasible && ctx.pool_room && !time_up {
                let _t = crate::profile::Timer::new(crate::profile::Phase::SepCover);
                out.found_cuts = separate_cover(
                    &node_lp,
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
            if verify_farkas_infeasible(&sol.dual, ctx.sa, ctx.sb, &sl, &su, ctx.m_w, ctx.n_w) {
                out.result = NodeResult {
                    node_id: id,
                    lower_bound: INFEAS_SENTINEL, // pruned
                    solution: vec![0.0; ctx.ns],
                    is_feasible: false,
                };
            } else {
                out.result = NodeResult {
                    node_id: id,
                    lower_bound: f64::NEG_INFINITY,
                    solution: midpoint(lb_k, ub_k),
                    is_feasible: false,
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
        a: ctx.sa,
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
                    a: ctx.sa,
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

    // Node duals y = B⁻ᵀ c_B via the *sparse* LU (factorize O(nnz) + btran), the
    // same path the dual simplex uses — not a dense m×m refactor. The earlier dense
    // `solve_dense` was O(m³) per node: trivial on few-row knapsacks but ruinous on
    // 800–1500-row covering LPs, where it dominated the whole solve. The basis is
    // scaling-invariant and `sp`/`c` are unscaled, so `y` is the unscaled dual the
    // integer bound fixing needs. A singular basis ⇒ `None` (sound: just no fixing).
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
                a: ctx.sa,
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
            &lp,
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
    #[test]
    fn driver_matches_golden() {
        for case in panel() {
            let r = case.solve_dense();
            let (status, obj, bound, nodes, iters): (MilpStatus, f64, f64, usize, usize) =
                match case.name {
                    "pure_lp" => (MilpStatus::Optimal, -1.0, -1.0, 1, 0),
                    "binary_knapsack" => (MilpStatus::Optimal, -10.0, -10.0, 1, 1),
                    "general_integer" => (MilpStatus::Optimal, -3.0, -3.0, 1, 1),
                    "infeasible" => (MilpStatus::Infeasible, f64::INFINITY, f64::INFINITY, 0, 0),
                    "unbounded" => (
                        MilpStatus::Unbounded,
                        f64::INFINITY,
                        f64::NEG_INFINITY,
                        1,
                        0,
                    ),
                    "cuts_firing_knapsack" => (MilpStatus::Optimal, -16.0, -16.0, 1, 1),
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
