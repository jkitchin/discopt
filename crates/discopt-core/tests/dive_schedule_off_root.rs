//! #1060: the continuous-repair dive must be able to re-fire away from the root.
//!
//! `try_dive_repair` is the only heuristic in the MILP driver that re-solves the
//! continuous variables for a rounded integer assignment, so on a weak-relaxation
//! (big-M) model it is the only thing that can produce a *first* feasible point.
//! It used to run at the root and nowhere else, which leaves the search exactly
//! one chance: measured on the Quesada-Grossmann single-tree master for
//! `rsyn0840m`, 150,193 nodes produced exactly one integer-feasible candidate,
//! and once the lazy separator cut that one off the search ran to its node limit
//! with nothing to prune against.
//!
//! This file pins the schedule end to end: with `DISCOPT_MILP_DIVE_STRIDE` on,
//! dives actually happen off-root, they stay under the hard cap, and — on a model
//! with no integer point at all — the search still refuses to invent an
//! incumbent. Its sibling `dive_schedule_default_off.rs` pins the other half:
//! unset, not one off-root dive runs.
//!
//! Env-var state is process-global and `dive_stride()` latches it in a `OnceLock`,
//! so the two halves are separate test binaries (one `#[test]` each) rather than
//! two cases in one file.

use discopt_core::bnb::milp_driver::{solve_milp, MilpOptions, MilpStatus};
use discopt_core::lp::crossover::LpView;
use discopt_core::lp::simplex::SimplexOptions;
use discopt_core::profile::{counter, Ctr};

/// `min sum x  s.t.  sum_j 2*x_j = 7,  x binary`.
///
/// LP-feasible (x = 7/40 componentwise) but integer-**in**feasible: the left side
/// is even for every integer point and the right side is odd. Parity is invisible
/// to bound propagation, so the driver cannot fathom it at the root — it has to
/// branch, which is what gives the schedule several batches to fire in. Nothing
/// here is keyed to a named instance (CLAUDE.md §2); the property under test is
/// "the search spends many nodes holding no incumbent", which is the regime the
/// schedule exists for.
const N: usize = 20;

fn build() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let a: Vec<f64> = vec![2.0; N];
    let c: Vec<f64> = vec![1.0; N];
    let l: Vec<f64> = vec![0.0; N];
    let u: Vec<f64> = vec![1.0; N];
    let b: Vec<f64> = vec![7.0];
    (a, b, c, l, u)
}

fn options() -> MilpOptions {
    MilpOptions {
        n_struct: N,
        integer_cols: (0..N).collect(),
        // Bounded so the test is fast: parity infeasibility needs full enumeration,
        // and the assertions below are about the schedule, not the certificate.
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
    }
}

#[test]
fn stride_on_fires_the_dive_away_from_the_root_and_stays_capped() {
    // Must be set before the first `solve_milp`: the stride is latched once per
    // process, and `solve_milp` calls `profile::init_from_env`.
    std::env::set_var("DISCOPT_MILP_DIVE_STRIDE", "1");
    std::env::set_var("DISCOPT_PROFILE", "1");

    let (a, b, c, l, u) = build();
    let lp = LpView {
        a: &a,
        m: 1,
        n: N,
        c: &c,
        l: &l,
        u: &u,
    };
    let res = solve_milp(&lp, &b, 0.0, &options());

    let mut checks = 0;

    // 1. The schedule actually fired off-root. Without this the rest of the file
    //    would pass on a driver where the feature is silently dead.
    let dives = counter(Ctr::DiveOffRoot);
    assert!(dives > 0, "no off-root dive ran; the schedule never fired");
    checks += 1;

    // 2. It stayed under the hard cap. This model has no integer point, so the
    //    dive can never succeed — exactly the case that must not become a
    //    per-node tax on a large search.
    assert!(
        dives <= 64,
        "off-root dives ({dives}) exceeded the hard cap"
    );
    checks += 1;

    // 3. A dive that cannot repair must report nothing, not something.
    assert_eq!(
        counter(Ctr::DiveOffRootHits),
        0,
        "a model with no integer point reported a repaired incumbent"
    );
    checks += 1;

    // 4. Soundness: diving off-root must never manufacture an incumbent. The
    //    dive fixes inside the node box, a restriction of the root box, so any
    //    point it returns is feasible for the model — and there is none here.
    assert!(
        !matches!(res.status, MilpStatus::Optimal | MilpStatus::Feasible),
        "status {:?} claims a feasible point for a parity-infeasible model",
        res.status
    );
    checks += 1;
    assert!(
        !res.obj.is_finite(),
        "objective {} reported for a parity-infeasible model",
        res.obj
    );
    checks += 1;

    assert_eq!(checks, 5, "CHECKS_EXECUTED");
}
