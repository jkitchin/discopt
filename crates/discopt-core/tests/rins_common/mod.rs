#![allow(dead_code)] // each arm binary uses a different subset of this fixture.
//! Shared fixture for the two RINS arms (`rins_on_improves_incumbent`,
//! `rins_default_off`). The arms are separate binaries because each one owns a
//! process-wide `DISCOPT_RINS` setting, and a shared module keeps the two from
//! silently drifting onto different models — which would make the comparison
//! between them meaningless.

use discopt_core::bnb::milp_driver::MilpOptions;
use discopt_core::lp::simplex::SimplexOptions;

pub const NF: usize = 6;
pub const NC: usize = 10;

/// A small capacitated facility-location instance, in the driver's standard form
/// `A x = b, l ≤ x ≤ u`:
///
/// ```text
/// min  Σ_j f_j y_j + Σ_ij c_ij x_ij
/// s.t. Σ_j x_ij = 1                       (customer i is served)
///      Σ_i d_i x_ij − K y_j + s_j = 0     (facility j's capacity), s_j ∈ [0, K]
/// ```
///
/// Columns: `x_ij` at `i*NF+j`, then `y_j` at `NX+j`, then `s_j` at `NX+NF+j`.
/// Capacity is tight enough (four facilities' worth spread over six) that the
/// root LP is fractional in many columns — which is the regime RINS is for, and
/// the reason a plain knapsack is not the fixture here.
pub struct Cfl {
    pub a: Vec<f64>,
    pub c: Vec<f64>,
    pub l: Vec<f64>,
    pub u: Vec<f64>,
    pub b: Vec<f64>,
    pub m: usize,
    pub n: usize,
    pub nx: usize,
}

pub fn demands() -> Vec<f64> {
    (0..NC).map(|i| 3.0 + ((i * 5) % 7) as f64).collect()
}

pub fn capacity() -> f64 {
    demands().iter().sum::<f64>() / (NF as f64 - 2.0)
}

pub fn build() -> Cfl {
    let nx = NF * NC;
    let n = nx + 2 * NF;
    let m = NC + NF;
    let d = demands();
    let cap = capacity();
    let f: Vec<f64> = (0..NF).map(|j| 20.0 + ((j * 7) % 11) as f64).collect();
    let cost: Vec<f64> = (0..nx)
        .map(|k| 1.0 + (((k / NF) * 13 + (k % NF) * 29) % 17) as f64)
        .collect();

    let mut a = vec![0.0; m * n];
    for i in 0..NC {
        for j in 0..NF {
            a[i * n + i * NF + j] = 1.0;
        }
    }
    for j in 0..NF {
        let r = NC + j;
        for i in 0..NC {
            a[r * n + i * NF + j] = d[i];
        }
        a[r * n + nx + j] = -cap;
        a[r * n + nx + NF + j] = 1.0;
    }

    let mut c = vec![0.0; n];
    c[..nx].copy_from_slice(&cost);
    c[nx..nx + NF].copy_from_slice(&f);
    let l = vec![0.0; n];
    let mut u = vec![1.0; n];
    for j in 0..NF {
        u[nx + NF + j] = cap;
    }
    let mut b = vec![0.0; m];
    b[..NC].fill(1.0);

    Cfl {
        a,
        c,
        l,
        u,
        b,
        m,
        n,
        nx,
    }
}

/// A feasible but deliberately poor starting point: every facility open, every
/// customer round-robined onto an arbitrary one. It is what a weak primal
/// heuristic produces and what the driver, having no improvement heuristic at
/// all, would otherwise carry to the end of the search.
pub fn poor_seed(p: &Cfl) -> Vec<f64> {
    let d = demands();
    let cap = capacity();
    let mut seed = vec![0.0; p.n];
    for j in 0..NF {
        seed[p.nx + j] = 1.0;
    }
    for i in 0..NC {
        seed[i * NF + (i % NF)] = 1.0;
    }
    for j in 0..NF {
        let load: f64 = (0..NC).map(|i| d[i] * seed[i * NF + j]).sum();
        seed[p.nx + NF + j] = cap * seed[p.nx + j] - load;
    }
    seed
}

/// Driver options for both arms. `heuristics: false` so the *only* incumbent
/// before RINS is the poor seed — otherwise the rounding heuristic supplies one
/// and the arms stop measuring the thing they are named for.
pub fn options(p: &Cfl, seed: Vec<f64>) -> MilpOptions {
    MilpOptions {
        n_struct: p.n,
        integer_cols: (0..p.nx + NF).collect(),
        max_nodes: 200_000,
        time_limit_s: Some(120.0),
        gap_tol: 1e-9,
        root_cuts: 0,
        cut_rounds: 0,
        gmi_cuts: false,
        cut_select: false,
        node_cuts: false,
        max_pool_cuts: 0,
        heuristics: false,
        presolve: true,
        strong_branch: false,
        node_propagation: true,
        reduced_cost_fixing: true,
        sb_max_cands: 0,
        sb_node_budget: 0,
        initial_incumbent: Some(seed),
        node_hook_rounds: 0,
        node_hook_cut_cap: 0,
        root_cut_time_s: None,
        root_cut_prune: true,
        simplex: SimplexOptions::default(),
    }
}

/// The instance's true optimum, independently pinned by the OFF arm (which never
/// runs RINS) so the ON arm's assertion is not circular.
pub const OPTIMUM: f64 = 148.0;

/// Objective of [`poor_seed`] — strictly worse than [`OPTIMUM`], so "RINS
/// improved the incumbent" is a claim with content.
pub const SEED_OBJ: f64 = 230.0;
