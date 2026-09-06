//! A13 shipped arm: with `DISCOPT_OBJ_LATTICE_SUBST` unset, the substitution runs.
//!
//! Separate from `obj_lattice_subst_on.rs` (which sets the variable explicitly)
//! because only an unset variable exercises the *default*, and env is
//! process-wide -- a test that shares a binary with one setting the variable
//! would be measuring that setting instead. The graduation panel is recorded on
//! `obj_lattice_subst_enabled` in `milp_driver.rs`; this pins the default it
//! chose, so flipping it back silently is a test failure rather than a quiet
//! performance regression.

mod objlat_common;

use discopt_core::bnb::milp_driver::{solve_milp, MilpStatus};
use discopt_core::lp::crossover::LpView;
use discopt_core::profile::{counter, Ctr};
use objlat_common::{build, options, M, N, OPTIMUM};

#[test]
fn the_default_resolves_the_continuous_objective_column() {
    std::env::remove_var("DISCOPT_OBJ_LATTICE_SUBST");
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

    // Anti-vacuity: without this the rest of the test passes just as well on a
    // build where the lever never ran (CLAUDE.md §6).
    assert_eq!(
        counter(Ctr::ObjLatticeSubst),
        1,
        "substitution did not engage at the default -- the flag did not graduate"
    );
    assert_eq!(res.status, MilpStatus::Optimal);
    assert!(
        (res.obj - OPTIMUM).abs() < 1e-6,
        "optimum is {OPTIMUM}, got {}",
        res.obj
    );
    assert!(
        res.bound <= OPTIMUM + 1e-6,
        "dual bound {} lifted past the optimum {OPTIMUM} -- a false certificate",
        res.bound
    );
    assert_eq!(
        res.nodes, 1,
        "the lattice should close this at the root; took {} nodes",
        res.nodes
    );
}
