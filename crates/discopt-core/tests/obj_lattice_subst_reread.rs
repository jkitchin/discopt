//! The flag must be re-read on every solve, not cached for the life of the
//! process.
//!
//! This exists because the first A13 differential panel measured nothing: the gate
//! was a `OnceLock`, the panel's first solve was its OFF arm, and every later
//! solve -- the whole ON arm included -- inherited that cached `false`. The table
//! came back with both arms at an identical 313,441 nodes on `neos-3611447-jijia`,
//! which reads as "the lever does nothing" rather than "the lever never ran"
//! (CLAUDE.md §6).
//!
//! One binary, one test, two solves, because the thing under test is precisely
//! what happens on the *second* solve in a process.

mod objlat_common;

use discopt_core::bnb::milp_driver::solve_milp;
use discopt_core::lp::crossover::LpView;
use discopt_core::profile::{counter, Ctr};
use objlat_common::{build, options, M, N};

#[test]
fn the_flag_is_re_read_between_solves_in_one_process() {
    std::env::set_var("DISCOPT_PROFILE", "1");
    let (a, b, c, l, u) = build();
    let lp = LpView {
        a: &a,
        m: M,
        n: N,
        c: &c,
        l: &l,
        u: &u,
    };

    // OFF first, exactly as a differential panel interleaves them.
    std::env::set_var("DISCOPT_OBJ_LATTICE_SUBST", "0");
    let _ = solve_milp(&lp, &b, 0.0, &options());
    let off = counter(Ctr::ObjLatticeSubst);

    std::env::set_var("DISCOPT_OBJ_LATTICE_SUBST", "1");
    let _ = solve_milp(&lp, &b, 0.0, &options());
    let on = counter(Ctr::ObjLatticeSubst);

    assert_eq!(off, 0, "substitution engaged on the OFF solve");
    assert_eq!(
        on, 1,
        "substitution did not engage on a later ON solve -- the flag is cached, so \
         any single-process A/B silently compares the OFF arm against itself"
    );
}
