//! A13 opt-out arm: with `DISCOPT_OBJ_LATTICE_SUBST=0` the driver must behave
//! exactly as it did before the substitution existed -- the detector refuses a
//! costed continuous column, no lattice is wired in, and the root gap has to be
//! branched away.
//!
//! The lever graduated default-ON on its differential panel, so this is no longer
//! the shipped default; it is the legacy path CLAUDE.md §5 requires to stay
//! intact and reachable, and it is the half that gives the on-arm's single node
//! its meaning -- without it, "1 node" could just be an easy fixture. See
//! `obj_lattice_subst_default_on.rs` for the shipped arm.

mod objlat_common;

use discopt_core::bnb::milp_driver::{solve_milp, MilpStatus};
use discopt_core::lp::crossover::LpView;
use discopt_core::profile::{counter, Ctr};
use objlat_common::{build, options, M, N, OPTIMUM};

#[test]
fn the_opt_out_leaves_the_continuous_objective_column_unresolved() {
    std::env::set_var("DISCOPT_OBJ_LATTICE_SUBST", "0");
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
    let res = solve_milp(&lp, &b, 0.0, &options());

    assert_eq!(
        counter(Ctr::ObjLatticeSubst),
        0,
        "substitution engaged under the opt-out -- the legacy path is gone"
    );
    assert_eq!(res.status, MilpStatus::Optimal);
    assert!(
        (res.obj - OPTIMUM).abs() < 1e-6,
        "optimum is {OPTIMUM}, got {}",
        res.obj
    );
    // Anti-vacuity in the other direction: the fixture must genuinely need
    // branching without the lever. If this ever drops to 1, the on-arm's node
    // assertion has stopped proving anything and both tests must be revisited.
    assert!(
        res.nodes > 1,
        "fixture certified at the root without the lattice -- the on-arm's node \
         count no longer distinguishes the arms"
    );
}
