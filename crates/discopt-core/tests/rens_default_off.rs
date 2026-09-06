//! A14, the OFF arm: with `DISCOPT_RENS` unset the driver must not run RENS at
//! all, and must still reach the same certified optimum.
//!
//! This is the opt-out guarantee. A14 is bound-changing in the CLAUDE.md §5 sense
//! -- it changes which incumbent is found and when -- so it ships default OFF
//! until a differential panel clears both the cert-clean and the net-positive bar.
//! The counter assertion is what makes "default off" a measured fact rather than a
//! claim in a doc: `RensConsidered == 0` means the schedule was never even
//! reached, which is stronger than `RensRun == 0` and is what distinguishes a
//! disabled feature from one that runs and declines every time.
//!
//! This arm also pins the optimum the ON arm asserts against, so that assertion is
//! not circular.
//!
//! Separate binary from `rens_on_finds_without_incumbent.rs`: `DISCOPT_RENS` is
//! process-wide.

#[path = "submip_common/mod.rs"]
mod cfl;

use discopt_core::bnb::milp_driver::{solve_milp, MilpStatus};
use discopt_core::lp::crossover::LpView;
use discopt_core::profile::{counter, Ctr};

#[test]
fn the_default_never_runs_rens() {
    std::env::set_var("DISCOPT_PROFILE", "1");
    std::env::remove_var("DISCOPT_RENS");

    let p = cfl::build_cover();
    let lp = LpView {
        a: &p.a,
        m: p.m,
        n: p.n,
        c: &p.c,
        l: &p.l,
        u: &p.u,
    };
    let res = solve_milp(&lp, &p.b, 0.0, &cfl::cover_options(&p));

    assert_eq!(
        counter(Ctr::RensConsidered),
        0,
        "RENS reached its call site with the flag unset -- the default is not off"
    );
    assert_eq!(counter(Ctr::RensRun), 0, "RENS ran with the flag unset");
    assert_eq!(counter(Ctr::RensImproved), 0);

    // The opt-out still solves the model, and to the same optimum the ON arm
    // asserts -- so that assertion is pinned by a run that never used RENS.
    assert_eq!(res.status, MilpStatus::Optimal, "obj={}", res.obj);
    assert!(
        (res.obj - cfl::COVER_OPTIMUM).abs() < 1e-6,
        "objective {} != optimum {}",
        res.obj,
        cfl::COVER_OPTIMUM
    );
}
