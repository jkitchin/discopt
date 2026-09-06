//! A12, the ON arm: with `DISCOPT_RINS=1` the driver must actually *improve* an
//! incumbent it would otherwise carry unchanged to the end of the search.
//!
//! This is the behavior the whole change exists for. The driver has a rounding
//! heuristic and a no-incumbent dive (#1060) but no improvement heuristic at all,
//! which is what the A5 measurement located: on the parity panel 17/38 instances
//! were unsolved at an 89.6 % median primal share, 12 of them holding a merely
//! poor incumbent rather than none. Here the poor incumbent is supplied
//! explicitly (`poor_seed`, objective 230) so the arm measures RINS and nothing
//! else, and the optimum is 148.
//!
//! Two assertions carry the correctness half and must never be relaxed to make
//! this pass: `RinsRejected == 0` (RINS never proposed a point the driver could
//! not independently verify -- a nonzero value is a false-`Optimal` bug, not a
//! tuning signal) and the certified objective equalling the optimum the OFF arm
//! reaches without RINS.
//!
//! Separate binary from `rins_default_off.rs`: `DISCOPT_RINS` is process-wide, so
//! the two arms cannot share one.

#[path = "submip_common/mod.rs"]
mod cfl;

use discopt_core::bnb::milp_driver::{solve_milp, MilpStatus};
use discopt_core::lp::crossover::LpView;
use discopt_core::profile::{counter, Ctr};

#[test]
fn rins_on_improves_a_poor_incumbent() {
    std::env::set_var("DISCOPT_PROFILE", "1");
    std::env::set_var("DISCOPT_RINS", "1");

    let p = cfl::build();
    let seed = cfl::poor_seed(&p);
    let seed_obj: f64 = (0..p.n).map(|k| p.c[k] * seed[k]).sum();
    assert!(
        (seed_obj - cfl::SEED_OBJ).abs() < 1e-9,
        "fixture drifted: seed objective {seed_obj}, expected {}",
        cfl::SEED_OBJ
    );

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

    // The funnel must have reached the end: considered >= gated >= run >= improved.
    // Asserting the last of these is what fails before this change (RINS never
    // runs) and passes after.
    assert!(
        counter(Ctr::RinsRun) >= 1,
        "RINS never ran (considered={}, gated={})",
        counter(Ctr::RinsConsidered),
        counter(Ctr::RinsGated)
    );
    assert!(
        counter(Ctr::RinsImproved) >= 1,
        "RINS ran {} time(s) but improved nothing -- the heuristic is inert",
        counter(Ctr::RinsRun)
    );

    // Correctness half. A point RINS could not re-verify must never be injected.
    assert_eq!(
        counter(Ctr::RinsRejected),
        0,
        "RINS proposed a point the driver could not verify -- a false-incumbent bug"
    );
    assert_eq!(res.status, MilpStatus::Optimal, "obj={}", res.obj);
    assert!(
        (res.obj - cfl::OPTIMUM).abs() < 1e-6,
        "objective {} != optimum {}",
        res.obj,
        cfl::OPTIMUM
    );
}
