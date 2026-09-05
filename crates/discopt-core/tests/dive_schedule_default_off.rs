//! #1060, the other half: with `DISCOPT_MILP_DIVE_STRIDE` unset, the driver must
//! behave exactly as it did before the schedule existed — the dive runs at the
//! root and nowhere else.
//!
//! This is what makes the shipped default a no-op rather than a silent behavior
//! change, and it is the reason the golden/determinism panels in
//! `milp_driver.rs` still hold unmodified. See `dive_schedule_off_root.rs` for
//! the on-arm and for why the two halves are separate binaries.

use discopt_core::bnb::milp_driver::{solve_milp, MilpOptions};
use discopt_core::lp::crossover::LpView;
use discopt_core::lp::simplex::SimplexOptions;
use discopt_core::profile::{counter, Ctr};

const N: usize = 20;

#[test]
fn stride_unset_never_dives_away_from_the_root() {
    std::env::remove_var("DISCOPT_MILP_DIVE_STRIDE");
    std::env::set_var("DISCOPT_PROFILE", "1");

    let a: Vec<f64> = vec![2.0; N];
    let c: Vec<f64> = vec![1.0; N];
    let l: Vec<f64> = vec![0.0; N];
    let u: Vec<f64> = vec![1.0; N];
    let b: Vec<f64> = vec![7.0];
    let lp = LpView {
        a: &a,
        m: 1,
        n: N,
        c: &c,
        l: &l,
        u: &u,
    };
    let opts = MilpOptions {
        n_struct: N,
        integer_cols: (0..N).collect(),
        max_nodes: 4_000,
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
    };
    let res = solve_milp(&lp, &b, 0.0, &opts);

    // The same search that fires dives in the on-arm must fire none here. The
    // sibling test proves this model does reach many no-incumbent batches, so a
    // zero here is the default being off — not the probe failing to run.
    assert_eq!(
        counter(Ctr::DiveOffRoot),
        0,
        "off-root dive ran at the default stride"
    );
    assert!(
        !res.obj.is_finite(),
        "objective {} for a parity-infeasible model",
        res.obj
    );
}
