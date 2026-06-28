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
use crate::lp::gomory::{separate_gomory, GomoryCut};
use crate::lp::simplex::sparse::SparseCols;
use crate::lp::simplex::{
    solve_lp, solve_lp_scaled, solve_lp_warm, solve_lp_warm_scaled, tighten_bounds, LpStatus,
    PreparedDual, Scaling, SimplexOptions,
};

const INF: f64 = 1e20;
const INFEAS_SENTINEL: f64 = 1e30;
const INT_TOL: f64 = 1e-5;

/// Terminal status of a MILP solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MilpStatus {
    /// Proven optimal (gap closed, no uncertified node bounds).
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
    /// Separate globally-valid cover cuts at fractional nodes into a shared pool.
    pub node_cuts: bool,
    /// Cap on the total number of pooled cuts (root + node).
    pub max_pool_cuts: usize,
    /// Rounding primal heuristic at fractional nodes (early incumbents).
    pub heuristics: bool,
    /// Root feasibility-based bound tightening (sound, dimension-preserving).
    pub presolve: bool,
    /// Limited strong branching on unreliable candidates (reliability branching).
    pub strong_branch: bool,
    /// Max candidates probed per node when strong branching.
    pub sb_max_cands: usize,
    /// Only strong-branch while the tree is smaller than this many nodes — the
    /// early region where branching choices shape the whole search. Beyond it,
    /// matured pseudocosts decide (avoids probing overhead deep in large trees).
    pub sb_node_budget: usize,
    /// LP solver options.
    pub simplex: SimplexOptions,
}

/// Solve `min cᵀx + obj_const s.t. A x = b, l ≤ x ≤ u` with `integer_cols`
/// integer-constrained, by Rust-internal warm-started-simplex branch and bound.
pub fn solve_milp(lp: &LpView<'_>, b: &[f64], obj_const: f64, opts: &MilpOptions) -> MilpResult {
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
        let pr = tighten_bounds(lp, b, &is_int_full, opts.simplex.tol);
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
    let mut gap_certified = true;

    // Original constraint rows (before any cuts) are the knapsack candidates for
    // cover separation; later rows are themselves cuts.
    let n_orig_rows = m_w;
    // Global cut pool signatures — globally-valid cover cuts found anywhere in
    // the tree are added once and shared by all nodes.
    let mut pool_sigs: HashSet<Vec<(u32, i64)>> = HashSet::new();

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
    if opts.root_cuts > 0 {
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
            let root = solve_lp_root(&root_lp, &b_w, &node_simplex);
            lp_iters += root.iters;
            if root.status != LpStatus::Optimal {
                break;
            }
            // Tailing off: stop once added cuts barely move the bound.
            if root.obj <= prev_obj + 1e-7 * (1.0 + prev_obj.abs()) && prev_obj > f64::NEG_INFINITY
            {
                break;
            }
            prev_obj = root.obj;

            // Knapsack cover cuts (sparse, strong on knapsack structure) plus
            // Gomory mixed-integer cuts off the native basis.
            let mut cuts = separate_cover(
                &root_lp,
                &b_w,
                &root.x,
                ns,
                &is_int_full,
                n_orig_rows,
                opts.simplex.tol,
            );
            cuts.extend(separate_gomory(
                &root_lp,
                &b_w,
                &root.basis,
                &is_int_full,
                opts.simplex.tol,
                1e7,
            ));
            cuts.truncate(opts.root_cuts - total_cuts);
            let new_cuts = dedup_new_cuts(cuts, &mut pool_sigs, usize::MAX);
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
    'search: loop {
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
                pool_room: pool_sigs.len() < opts.max_pool_cuts,
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
                // bound and stays a valid open node. The gap cannot be certified
                // once any node went unsolved, and the loop-top deadline check
                // breaks the search on the next iteration.
                gap_certified = false;
                continue;
            }
            if out.unbounded {
                hit_unbounded = true;
                break;
            }
            if out.uncertified {
                // An untrusted (iter-limit/numerical) node bound: never claim
                // optimality from this search.
                gap_certified = false;
            }
            if let Some(basis) = out.basis {
                tm.set_node_basis(id, Some(basis));
            }
            if let Some((cand, cobj)) = out.incumbent {
                tm.inject_incumbent(cand, cobj);
            }
            if let Some(v) = out.branch_hint {
                tm.set_branch_hint(id, v);
            }
            // Dedup this node's cuts against the shared pool *in order*, with the
            // same room check the serial path applied, so the pool is identical.
            if !out.found_cuts.is_empty() && pool_sigs.len() < opts.max_pool_cuts {
                pending_cuts.extend(dedup_new_cuts(
                    out.found_cuts,
                    &mut pool_sigs,
                    opts.max_pool_cuts,
                ));
            }
            results.push(out.result);
        }
        if hit_unbounded {
            unbounded = true;
            break 'search;
        }
        tm.import_results(&results);
        tm.process_evaluated();

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

    let stats = tm.stats();
    let bound = stats.global_lower_bound;
    // An all-integer placeholder at an infeasible node can be fathomed by the
    // tree as a sentinel-valued "incumbent"; it never blocks a real (finite)
    // incumbent, so treat obj ≥ the sentinel threshold as "no real solution".
    let (x, obj, has_inc) = match tm.incumbent() {
        Some((xi, oi)) if oi < INFEAS_SENTINEL - 1.0 => (xi.to_vec(), oi, true),
        _ => (vec![0.0; ns], f64::INFINITY, false),
    };
    let status = if unbounded {
        MilpStatus::Unbounded
    } else if !has_inc {
        if tm.is_finished() {
            MilpStatus::Infeasible
        } else {
            MilpStatus::NodeLimit
        }
    } else if (tm.is_finished() || tm.gap() <= opts.gap_tol) && gap_certified {
        MilpStatus::Optimal
    } else if stats.total_nodes >= opts.max_nodes {
        MilpStatus::NodeLimit
    } else {
        MilpStatus::Feasible
    };

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
    /// Absolute wall-clock deadline. Once passed, each node still computes its
    /// (valid) LP bound but skips the optional rounding heuristic, cover
    /// separation, and strong branching — none of which affect bound validity
    /// or feasibility — so the in-flight batch drains quickly. `None` = no limit.
    deadline: Option<std::time::Instant>,
    tm: &'a TreeManager,
}

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
    /// Simplex pivots spent on this node (LP solve + strong-branch probes).
    iters: usize,
    /// Relaxation was unbounded — the whole search terminates.
    unbounded: bool,
    /// LP hit iter-limit / numerical breakdown — gap can't be certified.
    uncertified: bool,
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
            iters: 0,
            unbounded: false,
            uncertified: true,
            deferred: true,
        };
    }
    let mut full_l = vec![0.0; ctx.n_w];
    let mut full_u = vec![0.0; ctx.n_w];
    full_l[..ctx.ns].copy_from_slice(lb_k);
    full_u[..ctx.ns].copy_from_slice(ub_k);
    full_l[ctx.ns..].copy_from_slice(ctx.slack_l);
    full_u[ctx.ns..].copy_from_slice(ctx.slack_u);
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
    let mut sol = match ctx.tm.node_basis(id) {
        Some(basis) => {
            let basis = extend_basis(basis, ctx.n_w);
            solve_lp_warm_scaled(&solve_lp_view, ctx.sb, &basis, ctx.simplex)
        }
        // The only node solved cold is the root. Try the dual simplex from the
        // slack basis (built from the unscaled working matrix — scaling-invariant,
        // dual feasibility preserved) before the cold primal, the same covering-LP
        // speedup the root-cut loop gets. `solve_lp_warm_scaled` cold-solves if the
        // slack basis is unavailable or not dual-feasible, so the result is unchanged.
        None => match dual_slack_basis(
            ctx.a_w,
            ctx.m_w,
            ctx.n_w,
            ctx.c_w,
            &full_l,
            &full_u,
            ctx.simplex.tol,
        ) {
            Some(basis) => solve_lp_warm_scaled(&solve_lp_view, ctx.sb, &basis, ctx.simplex),
            None => solve_lp_scaled(&solve_lp_view, ctx.sb, ctx.simplex),
        },
    };
    if let Some(s) = ctx.scaling {
        s.unscale_x(&mut sol.x);
    }
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
        iters: sol.iters,
        unbounded: false,
        uncertified: false,
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
                    out.incumbent = try_dive_repair(ctx, lb_k, ub_k, &sol.x, &sol.basis);
                }
            }
            // Node-level cover separation: a fractional node exposes violated
            // covers the root never sees. These are globally valid; the reduce
            // dedups them into the shared pool to tighten the whole tree.
            if ctx.opts.node_cuts && !feasible && ctx.pool_room && !time_up {
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
                    let cands = ctx.tm.score_candidates(xs);
                    let (best, piv) =
                        strong_branch(ctx, &full_l, &full_u, &sol.basis, &sol.x, sol.obj, &cands);
                    out.iters += piv;
                    out.branch_hint = best;
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
            out.result = NodeResult {
                node_id: id,
                lower_bound: INFEAS_SENTINEL, // pruned
                solution: vec![0.0; ctx.ns],
                is_feasible: false,
            };
        }
        LpStatus::Unbounded => {
            out.unbounded = true;
        }
        LpStatus::IterLimit | LpStatus::Numerical => {
            // Cannot trust a bound: never prune (could drop the optimum). Give a
            // non-pruning bound and leave the node to be branched; the reduce
            // marks the gap uncertified so we never claim optimality.
            out.uncertified = true;
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
) -> (Option<usize>, usize) {
    let simplex = ctx.simplex;
    // Unreliable candidates, most-fractional (nearest 0.5) first.
    let mut cand: Vec<(usize, f64)> = cands
        .iter()
        .filter(|c| c.2 < ctx.reliability)
        .map(|c| (c.0, c.1))
        .collect();
    if cand.is_empty() {
        return (None, 0);
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
    let prepared = PreparedDual::prepare(&prep_view, basis, simplex);
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
                solve_lp_warm_scaled(&view, ctx.sb, basis, simplex)
            }
        }
    };
    let mut best: Option<usize> = None;
    let mut best_score = f64::NEG_INFINITY;
    let mut pivots = 0usize;
    for (idx, _f) in cand {
        let xi = x[idx];
        let (lo0, hi0) = (orig_l[idx], orig_u[idx]);

        // Down branch: x_idx ≤ floor(x_idx).
        u[idx] = xi.floor();
        let dn = probe(&l, &u);
        u[idx] = hi0;
        pivots += dn.iters;
        let d_dn = match dn.status {
            LpStatus::Optimal => (dn.obj - node_obj).max(0.0),
            LpStatus::Infeasible => INFEAS_DELTA,
            _ => 0.0,
        };

        // Up branch: x_idx ≥ ceil(x_idx).
        l[idx] = xi.ceil();
        let up = probe(&l, &u);
        l[idx] = lo0;
        pivots += up.iters;
        let d_up = match up.status {
            LpStatus::Optimal => (up.obj - node_obj).max(0.0),
            LpStatus::Infeasible => INFEAS_DELTA,
            _ => 0.0,
        };

        let score = d_dn.max(eps) * d_up.max(eps);
        if score > best_score {
            best_score = score;
            best = Some(idx);
        }
    }
    (best, pivots)
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
            let mut sol = solve_lp_warm_scaled(&view, ctx.sb, &cur_basis, ctx.simplex);
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
        match next_x {
            Some(xx) => x = xx,
            None => return None, // both roundings infeasible -> abandon the dive
        }
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
    a: &[f64],
    m: usize,
    n: usize,
    c: &[f64],
    l: &[f64],
    u: &[f64],
    tol: f64,
) -> Option<Basis> {
    let sp = SparseCols::from_dense(a, m, n);
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
fn solve_lp_root(lp: &LpView<'_>, b: &[f64], opts: &SimplexOptions) -> crate::lp::simplex::LpSolve {
    match dual_slack_basis(lp.a, lp.m, lp.n, lp.c, lp.l, lp.u, opts.tol) {
        Some(basis) => solve_lp_warm(lp, b, &basis, opts),
        None => solve_lp(lp, b, opts),
    }
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
            node_cuts: true,
            max_pool_cuts: 500,
            heuristics: true,
            presolve: true,
            strong_branch: true,
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
}
