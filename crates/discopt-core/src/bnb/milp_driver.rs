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

use crate::bnb::branching::VarBranchInfo;
use crate::bnb::pool::SelectionStrategy;
use crate::bnb::tree_manager::{NodeResult, TreeManager};
use crate::lp::basis::Basis;
use crate::lp::crossover::LpView;
use crate::lp::gomory::separate_gomory;
use crate::lp::simplex::{solve_lp, solve_lp_warm, tighten_bounds, LpStatus, SimplexOptions};

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
    /// Relative gap tolerance for proving optimality.
    pub gap_tol: f64,
    /// Max Gomory mixed-integer cuts to add at the root (0 disables). Derived
    /// from the root's *native* simplex basis — no crossover needed.
    pub root_cuts: usize,
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

    // --- P5: root GMI cuts from the native simplex basis ---
    if opts.root_cuts > 0 {
        let root_lp = LpView {
            a: &a_w,
            m: m_w,
            n: n_w,
            c: &c_w,
            l: &l_w,
            u: &u_w,
        };
        let root = solve_lp(&root_lp, &b_w, &opts.simplex);
        lp_iters += root.iters;
        if root.status == LpStatus::Optimal {
            let cuts = separate_gomory(
                &root_lp,
                &b_w,
                &root.basis,
                &is_int_full,
                opts.simplex.tol,
                1e7,
            );
            let k = cuts.len().min(opts.root_cuts);
            if k > 0 {
                let (m_old, n_old) = (m_w, n_w);
                let (m_new, n_new) = (m_old + k, n_old + k);
                let mut a_new = vec![0.0; m_new * n_new];
                for i in 0..m_old {
                    a_new[i * n_new..i * n_new + n_old]
                        .copy_from_slice(&a_w[i * n_old..(i + 1) * n_old]);
                }
                for (ci, cut) in cuts.iter().take(k).enumerate() {
                    let row = m_old + ci;
                    a_new[row * n_new..row * n_new + n_old].copy_from_slice(&cut.coeffs);
                    a_new[row * n_new + n_old + ci] = -1.0; // surplus: coeffs·x − s = rhs
                    b_w.push(cut.rhs);
                    c_w.push(0.0);
                    l_w.push(0.0);
                    u_w.push(INF);
                    is_int_full.push(false);
                }
                a_w = a_new;
                m_w = m_new;
                n_w = n_new;
            }
        }
    }
    let slack_l = l_w[ns..].to_vec();
    let slack_u = u_w[ns..].to_vec();

    'search: loop {
        if tm.is_finished() || tm.gap() <= opts.gap_tol {
            break;
        }
        if tm.stats().total_nodes >= opts.max_nodes {
            gap_certified = false;
            break;
        }
        let batch = tm.export_batch(64);
        if batch.node_ids.is_empty() {
            break;
        }
        let sb_active = opts.strong_branch && tm.stats().total_nodes < opts.sb_node_budget;
        let mut results = Vec::with_capacity(batch.node_ids.len());
        for k in 0..batch.node_ids.len() {
            let id = batch.node_ids[k];
            let mut full_l = vec![0.0; n_w];
            let mut full_u = vec![0.0; n_w];
            full_l[..ns].copy_from_slice(&batch.lb[k]);
            full_u[..ns].copy_from_slice(&batch.ub[k]);
            full_l[ns..].copy_from_slice(&slack_l);
            full_u[ns..].copy_from_slice(&slack_u);
            let node_lp = LpView {
                a: &a_w,
                m: m_w,
                n: n_w,
                c: &c_w,
                l: &full_l,
                u: &full_u,
            };

            let sol = match tm.node_basis(id) {
                Some(basis) => solve_lp_warm(&node_lp, &b_w, &basis, &opts.simplex),
                None => solve_lp(&node_lp, &b_w, &opts.simplex),
            };
            lp_iters += sol.iters;

            let result = match sol.status {
                LpStatus::Optimal => {
                    tm.set_node_basis(id, Some(sol.basis.clone()));
                    let xs = &sol.x[..ns];
                    let feasible = is_int
                        .iter()
                        .enumerate()
                        .all(|(j, &it)| !it || frac(xs[j]) <= INT_TOL);
                    // Strong branching: for a fractional node that won't be
                    // pruned, probe the unreliable candidates and hint the best
                    // branching variable. Only the *choice* of variable changes,
                    // so this never affects correctness — only the node count.
                    if sb_active && !feasible {
                        let node_bound = sol.obj + obj_const;
                        let prunable = tm
                            .incumbent()
                            .map(|(_, inc)| node_bound >= inc - 1e-9)
                            .unwrap_or(false);
                        if !prunable {
                            let reliability = tm.get_reliability_threshold();
                            let cands = tm.score_candidates(xs);
                            let (best, piv) = strong_branch(
                                &node_lp,
                                &b_w,
                                &sol.basis,
                                &sol.x,
                                sol.obj,
                                &cands,
                                reliability,
                                opts.sb_max_cands,
                                &opts.simplex,
                            );
                            lp_iters += piv;
                            if let Some(v) = best {
                                tm.set_branch_hint(id, v);
                            }
                        }
                    }
                    NodeResult {
                        node_id: id,
                        lower_bound: sol.obj + obj_const,
                        solution: xs.to_vec(),
                        is_feasible: feasible,
                    }
                }
                LpStatus::Infeasible => NodeResult {
                    node_id: id,
                    lower_bound: INFEAS_SENTINEL, // pruned
                    solution: vec![0.0; ns],
                    is_feasible: false,
                },
                LpStatus::Unbounded => {
                    unbounded = true;
                    break 'search;
                }
                LpStatus::IterLimit | LpStatus::Numerical => {
                    // Cannot trust a bound: never prune (could drop the optimum).
                    // Give a non-pruning bound and leave the node to be branched;
                    // mark the gap uncertified so we never claim optimality.
                    gap_certified = false;
                    NodeResult {
                        node_id: id,
                        lower_bound: f64::NEG_INFINITY,
                        solution: midpoint(&batch.lb[k], &batch.ub[k]),
                        is_feasible: false,
                    }
                }
            };
            results.push(result);
        }
        tm.import_results(&results);
        tm.process_evaluated();
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

/// Limited strong branching. For the *unreliable* fractional candidates (those
/// whose pseudocosts aren't trusted yet), probe both child bounds with a warm
/// dual re-solve from the node's basis and pick the variable with the best
/// product score `max(Δ↓,ε)·max(Δ↑,ε)` (an infeasible child scores high — it
/// prunes immediately). Returns the chosen structural variable, if any, and the
/// simplex pivots spent. Cheap because each probe is a few warm pivots, and it
/// tapers automatically as pseudocosts mature past the reliability threshold.
#[allow(clippy::too_many_arguments)]
fn strong_branch(
    lp: &LpView<'_>,
    b: &[f64],
    basis: &Basis,
    x: &[f64],
    node_obj: f64,
    cands: &[(usize, f64, u32, f64)],
    reliability: u32,
    max_cands: usize,
    simplex: &SimplexOptions,
) -> (Option<usize>, usize) {
    // Unreliable candidates, most-fractional (nearest 0.5) first.
    let mut cand: Vec<(usize, f64)> = cands
        .iter()
        .filter(|c| c.2 < reliability)
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
    cand.truncate(max_cands.max(1));

    const INFEAS_DELTA: f64 = 1e7; // a pruned child is a strong branching signal
    let eps = 1e-6;
    let mut l = lp.l.to_vec();
    let mut u = lp.u.to_vec();
    let mut best: Option<usize> = None;
    let mut best_score = f64::NEG_INFINITY;
    let mut pivots = 0usize;
    for (idx, _f) in cand {
        let xi = x[idx];
        let (lo0, hi0) = (lp.l[idx], lp.u[idx]);

        // Down branch: x_idx ≤ floor(x_idx).
        u[idx] = xi.floor();
        let dn = solve_lp_warm(
            &LpView {
                a: lp.a,
                m: lp.m,
                n: lp.n,
                c: lp.c,
                l: &l,
                u: &u,
            },
            b,
            basis,
            simplex,
        );
        u[idx] = hi0;
        pivots += dn.iters;
        let d_dn = match dn.status {
            LpStatus::Optimal => (dn.obj - node_obj).max(0.0),
            LpStatus::Infeasible => INFEAS_DELTA,
            _ => 0.0,
        };

        // Up branch: x_idx ≥ ceil(x_idx).
        l[idx] = xi.ceil();
        let up = solve_lp_warm(
            &LpView {
                a: lp.a,
                m: lp.m,
                n: lp.n,
                c: lp.c,
                l: &l,
                u: &u,
            },
            b,
            basis,
            simplex,
        );
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

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(ns: usize, int_cols: Vec<usize>) -> MilpOptions {
        MilpOptions {
            n_struct: ns,
            integer_cols: int_cols,
            max_nodes: 100_000,
            gap_tol: 1e-9,
            root_cuts: 16,
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
        assert!(r_on.nodes <= r_off.nodes, "{} vs {}", r_on.nodes, r_off.nodes);
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
