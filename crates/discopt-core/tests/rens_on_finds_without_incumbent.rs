//! A14, the ON arm: with `DISCOPT_RENS=1` the driver must produce an incumbent in
//! the regime where it has *no other way* to get one.
//!
//! The options carry **no seed incumbent** and `heuristics: false`, so the driver
//! holds nothing to improve and has no rounding heuristic. RINS cannot even be
//! asked the question here, because it fixes the columns on which an incumbent and
//! the relaxation agree and there is no incumbent. RENS reads only the relaxation,
//! so it can run from the first batch.
//!
//! Why this regime is the one that matters: measured on main+A13 over the
//! 38-instance parity panel, of the 14 unsolved instances that already held an
//! incumbent at 20 s, *none* improved it at 60 s -- 3x the wall and 2-5x the nodes
//! -- while the dual bound improved on 10 of 15. Two instances held no incumbent
//! at all. `enlight_hard` is the natural control: the only instance where the
//! primal machinery was still permitted to run is the only one that gained.
//!
//! Two assertions carry the correctness half and must never be relaxed to make
//! this pass: `RensRejected == 0` (RENS never proposed a point the driver could
//! not independently verify -- a nonzero value is a false-incumbent bug, not a
//! tuning signal) and the certified objective equalling the optimum the OFF arm
//! reaches without RENS.
//!
//! Separate binary from `rens_default_off.rs`: `DISCOPT_RENS` is process-wide, so
//! the two arms cannot share one.

#[path = "submip_common/mod.rs"]
mod cfl;

use discopt_core::bnb::milp_driver::{solve_milp, MilpStatus};
use discopt_core::lp::crossover::LpView;
use discopt_core::profile::{counter, Ctr};

#[test]
fn rens_on_finds_an_incumbent_with_nothing_to_improve() {
    std::env::set_var("DISCOPT_PROFILE", "1");
    std::env::set_var("DISCOPT_RENS", "1");

    let p = cfl::build_cover();
    let lp = LpView {
        a: &p.a,
        m: p.m,
        n: p.n,
        c: &p.c,
        l: &p.l,
        u: &p.u,
    };
    let opts = cfl::cover_options(&p);
    assert!(
        opts.initial_incumbent.is_none() && !opts.heuristics,
        "fixture drifted: this arm must run with no seed and no rounding heuristic"
    );
    let res = solve_milp(&lp, &p.b, 0.0, &opts);

    // The funnel must have reached the end: considered >= gated >= run >= improved.
    // Asserting the last of these is what fails before this change (RENS does not
    // exist) and passes after.
    assert!(
        counter(Ctr::RensRun) >= 1,
        "RENS never ran (considered={}, gated={})",
        counter(Ctr::RensConsidered),
        counter(Ctr::RensGated)
    );
    assert!(
        counter(Ctr::RensImproved) >= 1,
        "RENS ran {} time(s) but produced nothing -- the heuristic is inert",
        counter(Ctr::RensRun)
    );
    // The funnel ordering itself, so an implausible count reads as the arithmetic
    // error it is rather than as a plausible-looking hit rate (CLAUDE.md §6).
    assert!(
        counter(Ctr::RensConsidered) >= counter(Ctr::RensGated)
            && counter(Ctr::RensGated) >= counter(Ctr::RensRun)
            && counter(Ctr::RensRun) >= counter(Ctr::RensImproved),
        "funnel out of order: considered={} gated={} run={} improved={}",
        counter(Ctr::RensConsidered),
        counter(Ctr::RensGated),
        counter(Ctr::RensRun),
        counter(Ctr::RensImproved)
    );

    // Correctness half. A point RENS could not re-verify must never be injected.
    assert_eq!(
        counter(Ctr::RensRejected),
        0,
        "RENS proposed a point the driver could not verify -- a false-incumbent bug"
    );
    assert_eq!(res.status, MilpStatus::Optimal, "obj={}", res.obj);
    assert!(
        (res.obj - cfl::COVER_OPTIMUM).abs() < 1e-6,
        "objective {} != optimum {}",
        res.obj,
        cfl::COVER_OPTIMUM
    );
}
