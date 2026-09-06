//! A13 on-arm: with `DISCOPT_OBJ_LATTICE_SUBST=1` the objective lattice is
//! recovered through the equality row that pins the continuous objective column,
//! and the half-unit root gap of an odd-cycle cover closes at the root.
//!
//! This fails on the parent commit: the detector there refuses the moment a
//! continuous column carries cost, so the counter reads zero and the search has to
//! branch the gap away. See `obj_lattice_subst_optout.rs` for the other arm,
//! which pins that branching from the other side; they are separate binaries
//! because the flag is read once per solve (the `OnceLock` was removed; see
//! `obj_lattice_subst_reread.rs`, which pins the re-read).

mod objlat_common;

use discopt_core::bnb::milp_driver::{solve_milp, MilpStatus};
use discopt_core::lp::crossover::LpView;
use discopt_core::profile::{counter, Ctr};
use objlat_common::{build, options, M, N, OPTIMUM};

#[test]
fn substitution_recovers_the_lattice_and_closes_the_root_gap() {
    std::env::set_var("DISCOPT_OBJ_LATTICE_SUBST", "1");
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

    // Anti-vacuity (CLAUDE.md §6): this counter moves only when the base detector
    // refused AND the substitution succeeded, so a zero means the run measured
    // nothing and every assertion below is about the wrong code path.
    assert_eq!(
        counter(Ctr::ObjLatticeSubst),
        1,
        "substitution never engaged -- the fixture no longer presents the shape"
    );

    assert_eq!(
        res.status,
        MilpStatus::Optimal,
        "must certify, not merely find"
    );
    assert!(
        (res.obj - OPTIMUM).abs() < 1e-6,
        "optimum is {OPTIMUM}, got {} -- a wrong objective means the lattice cutoff \
         fathomed the optimum (CLAUDE.md §1)",
        res.obj
    );
    // The bound must never exceed the truth. This is the false certificate the
    // whole anchor argument exists to prevent: a lattice with the right spacing
    // but a wrong shift rounds the dual bound straight past the optimum.
    assert!(
        res.bound <= OPTIMUM + 1e-6,
        "dual bound {} exceeds the true optimum {OPTIMUM} -- false certificate",
        res.bound
    );
    // The cover's LP relaxation is K/2 = 7.5 and no cut moves it; only the lattice
    // lifts it onto 8. So the root alone certifies here, where the off-arm must
    // branch -- that difference is the whole lever, stated as a node count.
    assert_eq!(
        res.nodes, 1,
        "the lattice should close this at the root, took {} nodes",
        res.nodes
    );
}
