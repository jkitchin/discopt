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
use crate::lp::crossover::LpView;
use crate::lp::simplex::{solve_lp, solve_lp_warm, LpStatus, SimplexOptions};

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

    let glb = lp.l[..ns].to_vec();
    let gub = lp.u[..ns].to_vec();
    let mut tm = TreeManager::new(ns, glb, gub, int_info, SelectionStrategy::BestFirst);
    tm.initialize();

    let slack_l = lp.l[ns..].to_vec();
    let slack_u = lp.u[ns..].to_vec();

    let mut lp_iters = 0usize;
    let mut unbounded = false;
    let mut gap_certified = true;

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
        let mut results = Vec::with_capacity(batch.node_ids.len());
        for k in 0..batch.node_ids.len() {
            let id = batch.node_ids[k];
            let mut full_l = vec![0.0; n];
            let mut full_u = vec![0.0; n];
            full_l[..ns].copy_from_slice(&batch.lb[k]);
            full_u[..ns].copy_from_slice(&batch.ub[k]);
            full_l[ns..].copy_from_slice(&slack_l);
            full_u[ns..].copy_from_slice(&slack_u);
            let node_lp = LpView {
                a: lp.a,
                m: lp.m,
                n,
                c: lp.c,
                l: &full_l,
                u: &full_u,
            };

            let sol = match tm.node_basis(id) {
                Some(basis) => solve_lp_warm(&node_lp, b, &basis, &opts.simplex),
                None => solve_lp(&node_lp, b, &opts.simplex),
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
