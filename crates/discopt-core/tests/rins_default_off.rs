//! A12, the OFF arm: with `DISCOPT_RINS` unset the driver must behave exactly as
//! it did before RINS existed — the heuristic is not merely quiet, it is never
//! reached.
//!
//! This is what makes the shipped default a no-op rather than a silent behavior
//! change (CLAUDE.md §5: a mechanism that changes what the search finds stays
//! default-off until a differential panel clears both the cert-clean and the
//! net-positive bar). Its sibling `rins_on_improves_incumbent.rs` proves this
//! same model *does* reach the attempt, so the zero below is the default being
//! off — not the probe failing to run.
//!
//! It also pins the optimum the ON arm asserts against, reached here without RINS
//! ever executing, so that assertion is not circular.

#[path = "submip_common/mod.rs"]
mod cfl;

use discopt_core::bnb::milp_driver::{solve_milp, MilpStatus};
use discopt_core::lp::crossover::LpView;
use discopt_core::profile::{counter, Ctr};

#[test]
fn rins_unset_never_runs_and_still_solves() {
    std::env::set_var("DISCOPT_PROFILE", "1");
    std::env::remove_var("DISCOPT_RINS");

    let p = cfl::build();
    let seed = cfl::poor_seed(&p);
    let lp = LpView {
        a: &p.a,
        m: p.m,
        n: p.n,
        c: &p.c,
        l: &p.l,
        u: &p.u,
    };
    let opts = cfl::options(&p, seed);
    let res = solve_milp(&lp, &p.b, 0.0, &opts);

    assert_eq!(
        counter(Ctr::RinsConsidered),
        0,
        "RINS was reached at the default (unset) setting"
    );
    assert_eq!(
        counter(Ctr::RinsRun),
        0,
        "a RINS sub-MILP ran at the default"
    );

    assert_eq!(res.status, MilpStatus::Optimal, "obj={}", res.obj);
    assert!(
        (res.obj - cfl::OPTIMUM).abs() < 1e-6,
        "objective {} != optimum {}",
        res.obj,
        cfl::OPTIMUM
    );
}
