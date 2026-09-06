//! Shared fixture for the two `DISCOPT_OBJ_LATTICE_SUBST` arms.
//!
//! An odd-cycle vertex cover on `K` nodes, whose objective is routed through a
//! single *continuous* column: `min z` subject to `z = sum x_j`. This is the shape
//! measured across MIPLIB -- 13 of a 104-instance draw put their whole objective
//! on one continuous column pinned by an equality row (the `neos-361*` family,
//! `misc07`, `markshare*`, `rout`).
//!
//! The cover is chosen because its LP relaxation is exactly `K/2` against an
//! integer optimum of `(K+1)/2`, so the entire root gap is a half unit that no cut
//! closes and only the objective lattice removes. With the lattice the root bound
//! lifts to the optimum; without it the search must branch. That makes the arms
//! separable by node count, not just by a counter.
//!
//! The two arms live in separate test binaries because the flag is read once per
//! process through a `OnceLock`.

use discopt_core::bnb::milp_driver::MilpOptions;
use discopt_core::lp::simplex::SimplexOptions;

/// Odd, so the cover LP bound `K/2` is genuinely fractional.
pub const K: usize = 15;
pub const OPTIMUM: f64 = ((K + 1) / 2) as f64;

const INF: f64 = 1e20;

/// Columns: `x_0..x_{K-1}` (binary), `z` (continuous), the `K` edge slacks, and
/// the defining row's slack (fixed, which is what makes that row an equality).
pub const NS: usize = K + 1;
pub const N: usize = 2 * K + 2;
pub const M: usize = K + 1;

/// `(A row-major, b, c, l, u)` in engine form `A z = b`.
pub fn build() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut a = vec![0.0; M * N];
    for i in 0..K {
        a[i * N + i] = 1.0;
        a[i * N + (i + 1) % K] = 1.0;
        a[i * N + (K + 1 + i)] = -1.0; // surplus: x_i + x_{i+1} - s_i = 0, s_i >= 1
    }
    let zr = K * N;
    a[zr + K] = 1.0; // z
    for j in 0..K {
        a[zr + j] = -1.0; // - sum x_j
    }
    a[zr + (2 * K + 1)] = -1.0; // slack fixed at 0 => equality

    let b = vec![0.0; M];

    let mut c = vec![0.0; N];
    c[K] = 1.0; // the whole objective, on a continuous column

    let mut l = vec![0.0; N];
    let mut u = vec![0.0; N];
    for j in 0..K {
        u[j] = 1.0; // binary
    }
    u[K] = K as f64; // z
    for i in 0..K {
        l[K + 1 + i] = 1.0;
        u[K + 1 + i] = INF;
    }
    l[2 * K + 1] = 0.0;
    u[2 * K + 1] = 0.0; // fixed

    (a, b, c, l, u)
}

pub fn options() -> MilpOptions {
    MilpOptions {
        n_struct: NS,
        integer_cols: (0..K).collect(),
        max_nodes: 200_000,
        time_limit_s: Some(60.0),
        gap_tol: 1e-9,
        root_cuts: 0,
        cut_rounds: 0,
        gmi_cuts: false,
        cut_select: false,
        node_cuts: false,
        max_pool_cuts: 0,
        heuristics: true,
        presolve: true,
        strong_branch: false,
        node_propagation: true,
        reduced_cost_fixing: true,
        sb_max_cands: 0,
        sb_node_budget: 0,
        initial_incumbent: None,
        node_hook_rounds: 0,
        node_hook_cut_cap: 0,
        root_cut_time_s: None,
        root_cut_prune: true,
        simplex: SimplexOptions::default(),
    }
}
