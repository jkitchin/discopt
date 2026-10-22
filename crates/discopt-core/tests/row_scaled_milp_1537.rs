//! #1537: the MILP driver's certificate on the row-scaled witness shape found in
//! the m3 / flay03m OA masters.
//!
//! The LP is `min x + 2y` s.t. `x + y >= 3.5`, `x - y <= 1`, `x` integer in
//! [0, 10], `y` in [0, 100]: the optimum is 5 at `(2, 1.5)`. Written with row 0
//! multiplied by 1e5 and row 1 by 1e-6 it is the same model, but column `x` then
//! holds 1e5 and 1e-6, and the equilibration reads the 1e-6 as noise. The #1296
//! guard correctly withdraws the certificate: a lost certificate, never a false
//! bound. A geometric-mean pow2 row pre-pass (`DISCOPT_LP_ROW_PRESCALE`) restored
//! it here but was retired after its §5 panel (clay0303hfsg lost its certificate;
//! see `docs/dev/flag-retirement-audit.md`). This pins the behaviour main keeps;
//! a future row-scale-invariant fix flips the scaled arm and must pass the panel.
//!
//! One test in its own binary: it reads the process-wide profile counters.

use discopt_core::bnb::milp_driver::{solve_milp, MilpOptions, MilpResult, MilpStatus};
use discopt_core::lp::crossover::LpView;
use discopt_core::lp::simplex::SimplexOptions;
use discopt_core::profile::{counter, Ctr};

const INF: f64 = 1e20;
const OPTIMUM: f64 = 5.0;

fn options() -> MilpOptions {
    MilpOptions {
        n_struct: 2,
        integer_cols: vec![0],
        max_nodes: 10_000,
        max_open_nodes: None,
        time_limit_s: Some(30.0),
        gap_tol: 1e-9,
        abs_gap_tol: None,
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

/// Solve with rows multiplied by `(s0, s1)`; returns the result and the
/// `MilpTinyEntryDecert` counter for that solve.
fn solve(s0: f64, s1: f64) -> (MilpResult, u64) {
    #[rustfmt::skip]
    let a = [
        s0, s0, 1.0, 0.0,
        s1, -s1, 0.0, 1.0,
    ];
    let b = [3.5 * s0, s1];
    let c = [1.0, 2.0, 0.0, 0.0];
    // Columns x, y, s0 (<= 0: row 0 is `>=`), s1 (>= 0: row 1 is `<=`).
    let l = [0.0, 0.0, -INF, 0.0];
    let u = [10.0, 100.0, 0.0, INF];
    let lp = LpView {
        a: &a,
        m: 2,
        n: 4,
        c: &c,
        l: &l,
        u: &u,
    };
    let res = solve_milp(&lp, &b, 0.0, &options());
    (res, counter(Ctr::MilpTinyEntryDecert))
}

fn assert_certified(res: &MilpResult, label: &str) {
    assert_eq!(res.status, MilpStatus::Optimal, "{label}: {res:?}");
    assert!((res.obj - OPTIMUM).abs() < 1e-6, "{label}: {res:?}");
    assert!(
        res.bound <= OPTIMUM + 1e-6,
        "{label}: bound {} above the optimum {OPTIMUM} -- false certificate",
        res.bound
    );
    assert!(res.bound >= OPTIMUM - 1e-6, "{label}: {res:?}");
}

#[test]
fn row_scaled_witness_loses_its_certificate_soundly() {
    std::env::set_var("DISCOPT_PROFILE", "1");

    // Unit rows: the guard stays quiet and the model certifies the optimum.
    let (unit, d0) = solve(1.0, 1.0);
    assert_eq!(d0, 0);
    assert_certified(&unit, "unit rows");
    // The point is reported in original units.
    assert!((unit.x[0] - 2.0).abs() < 1e-6 && (unit.x[1] - 1.5).abs() < 1e-6);

    // Rows scaled by 1e5 / 1e-6. Anti-vacuity (CLAUDE.md §6): the guard must fire
    // here, or this test no longer reaches the code it pins. The certificate is
    // withdrawn -- never replaced by a bound above the optimum.
    let (scaled, d1) = solve(1e5, 1e-6);
    assert_eq!(
        d1, 1,
        "the #1296 guard no longer fires on the row-scaled witness"
    );
    assert_eq!(scaled.bound, f64::NEG_INFINITY, "{scaled:?}");
    assert_ne!(scaled.status, MilpStatus::Optimal);
}
