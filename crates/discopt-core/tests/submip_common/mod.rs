#![allow(dead_code)] // each arm binary uses a different subset of this fixture.
//! Shared fixture for the sub-MILP neighborhood arms — RINS
//! (`rins_on_improves_incumbent`, `rins_default_off`) and RENS
//! (`rens_on_finds_without_incumbent`, `rens_default_off`). Each arm is a
//! separate binary because each owns a process-wide `DISCOPT_RINS`/`DISCOPT_RENS`
//! setting, and a shared module keeps them from silently drifting onto different
//! models — which would make the comparisons between them meaningless.

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

// ---------------------------------------------------------------------------
// The RENS fixture. Deliberately a different model from the CFL above, for a
// measured reason.
//
// RENS was first tried on the CFL and it fails there — not through a defect but
// through a property of the instance. Every RENS box built from a CFL node LP was
// integer-infeasible: pinning the (many) assignment columns the relaxation had
// already driven to 0 leaves the remaining free columns unable to serve every
// customer within capacity. Re-solving each box with presolve, propagation and
// reduced-cost fixing all disabled still proved infeasibility in 3–7 nodes, so the
// infeasibility is the model's, not the presolver's. Tuning the fixing-rate floor
// until CFL passed would have been exactly the tolerance-tweak CLAUDE.md §3
// forbids; the honest move is a second fixture whose RENS box is feasible *by
// construction*.
// ---------------------------------------------------------------------------

/// Number of odd-cycle (fractional) columns and of forced (integral) columns.
pub const COVER_CYCLE: usize = 5;
pub const COVER_FORCED: usize = 5;

/// A minimum-weight vertex cover on an odd cycle, plus a block of columns the
/// rows force open, in the driver's standard form `A x = b, l ≤ x ≤ u`:
///
/// ```text
/// min  Σ_{j<5} x_j + Σ_{j≥5} w_j x_j
/// s.t. x_i + x_{i+1 mod 5} − s_i = 1     i = 0..4   (the C5 edges)
///      x_{5+k}            − s_{5+k} = 1  k = 0..4   (forces x_{5+k} = 1)
///      x ∈ {0,1}^10,  s ≥ 0
/// ```
///
/// Two properties make this the right fixture, and both are provable rather than
/// observed:
///
/// 1. **The root LP is fractional, uniquely and by exactly half.** Summing the
///    five cycle rows gives `2 Σ_{j<5} x_j ≥ 5`, so the LP cannot do better than
///    2.5, and attaining it forces every cycle row tight — which on an *odd*
///    cycle admits only `x_j = 1/2`. The forced block sits at 1. So the fixing
///    rate is exactly 5/10 = 0.5: above the floor, and not 1.0, which is the band
///    where `rens_box` is defined to act.
/// 2. **The RENS box is feasible.** It pins the forced block at 1 and leaves the
///    five half-valued cycle columns free over all of {0,1}, so the sub-MIP is an
///    unrestricted C5 vertex cover — cost 3 — and RENS returns the instance's true
///    optimum on its first attempt. That is the mechanism's claim stated as an
///    assertion rather than as a hit rate.
///
/// The integer optimum is 3 (cover 3 of the 5 cycle vertices; 2 cannot cover 5
/// edges) plus the forced block's weight 4+2+3+4+2 = 15.
pub struct Cover {
    pub a: Vec<f64>,
    pub c: Vec<f64>,
    pub l: Vec<f64>,
    pub u: Vec<f64>,
    pub b: Vec<f64>,
    pub m: usize,
    pub n: usize,
}

/// Weights of the forced block. Non-uniform so the objective is a real sum rather
/// than a count that could be reproduced by an off-by-one.
pub fn cover_weights() -> Vec<f64> {
    (0..COVER_FORCED)
        .map(|k| 2.0 + ((k * 2) % 3) as f64)
        .collect()
}

pub fn build_cover() -> Cover {
    let nbin = COVER_CYCLE + COVER_FORCED;
    let m = nbin; // one row per cycle edge, one per forced column
    let n = nbin + m; // binaries, then one surplus per row
    let w = cover_weights();

    let mut a = vec![0.0; m * n];
    for i in 0..COVER_CYCLE {
        a[i * n + i] = 1.0;
        a[i * n + (i + 1) % COVER_CYCLE] = 1.0;
        a[i * n + nbin + i] = -1.0;
    }
    for k in 0..COVER_FORCED {
        let r = COVER_CYCLE + k;
        a[r * n + COVER_CYCLE + k] = 1.0;
        a[r * n + nbin + r] = -1.0;
    }

    let mut c = vec![0.0; n];
    c[..COVER_CYCLE].fill(1.0);
    c[COVER_CYCLE..nbin].copy_from_slice(&w);

    let l = vec![0.0; n];
    // Every column's upper bound is 1: the binaries by definition, and the
    // surpluses because a cycle row can be covered at most twice and a forced row
    // never (`repeat_n` would be tidier but is stable only since 1.82; the
    // workspace MSRV is 1.75).
    let u = vec![1.0; n];

    let b = vec![1.0; m];

    Cover {
        a,
        c,
        l,
        u,
        b,
        m,
        n,
    }
}

/// Options with **no incumbent at all** and no rounding heuristic — the regime
/// RENS exists for and the one RINS structurally cannot enter. With both off, the
/// only ways an incumbent can appear are a node LP landing integral by luck or
/// RENS producing one, which is what makes the RENS arms' assertions sharp.
pub fn cover_options(p: &Cover) -> MilpOptions {
    MilpOptions {
        n_struct: p.n,
        integer_cols: (0..COVER_CYCLE + COVER_FORCED).collect(),
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
        initial_incumbent: None,
        node_hook_rounds: 0,
        node_hook_cut_cap: 0,
        root_cut_time_s: None,
        root_cut_prune: true,
        simplex: SimplexOptions::default(),
    }
}

/// The covering instance's true optimum: 3 for the odd cycle + 15 for the forced
/// block. Independently pinned by the RENS OFF arm, which never runs RENS, so the
/// ON arm's assertion is not circular.
pub const COVER_OPTIMUM: f64 = 18.0;
