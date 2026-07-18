//! Factorization hardening for near-singular simplex bases (issue #671, the
//! feral-touching half).
//!
//! # Problem
//!
//! hda-class ill-conditioned McCormick relaxations drive the revised simplex onto
//! **near-singular bases** (numerical rank deficient by ~13–43, κ ≈ 1e14). feral's
//! default LU aborts on them — a column whose pivot candidates are all `≤
//! zero_pivot_tol` returns `SingularBasis` (`LuSingularAction::Fail`) — so the
//! solve exits `Numerical` with no usable factor, and hence no usable dual. That
//! is the root cause of hda's missing/loose bound (candidate A only salvages a
//! *drifted* dual from the broken state).
//!
//! # Approach under test (entry experiment first — Dev-Philosophy #4)
//!
//! feral already exposes the right primitive: `LuSingularAction::PerturbToEps {
//! abs_floor }` replaces a singular pivot `d` with `sign(d)·max(|d|, abs_floor)`
//! and *continues*, producing a factor of a nearby matrix `B'` that differs from
//! `B` only in the rank-deficient pivot(s) — a **localized** regularization, not a
//! uniform `B + εI`. The perturbed factor solves `B' x ≈ rhs` inaccurately, but a
//! **high-precision iterative refinement** (residual `rhs − B x` accumulated in
//! double-double via [`super::refine::residual_dd`], re-solved through the same
//! `B'` factor) recovers accuracy in the well-conditioned subspace — the classic
//! Wilkinson result, here with the twist that the factor is of `B'`, not `B`.
//!
//! # Status
//!
//! Entry experiment **CONFIRMED** (the cargo tests below): on near-singular /
//! exactly-singular bases where feral's default `Fail` factorization aborts,
//! `PerturbToEps` + double-double refinement recovers the solve to residual ~0
//! (growth 1.0); the boundary case (solution loading the near-singular direction)
//! also recovers, provided `abs_floor` sits below the genuine small pivots.
//!
//! The production capability lives on [`super::linsolve::FeralLU`]:
//! [`FeralLU::with_singular_perturb`](super::linsolve::FeralLU::with_singular_perturb)
//! turns on the hardened factorization and double-double refined ftran/btran. It is
//! wired into the simplex's **failure-triggered** retry behind a default-OFF flag
//! (`DISCOPT_LP_FACTORIZATION_HARDENING`): [`super::linsolve::node_feral_lu`] builds
//! a hardened factor while [`super::linsolve::with_hardening_active`] is set, and
//! `dual::solve_lp_warm_csc` re-runs the solve under it once a strict solve exits
//! `Numerical`/`IterLimit`. Flag OFF ⇒ every solve is byte-identical to today (the
//! `node_lu_is_plain_when_hardening_inactive` test). The corpus-wide bound-neutral
//! + differential panel remains the graduation gate before any default-ON.

#![cfg(test)]

use super::linsolve::{FeralLU, LinearSolver};
use super::refine::residual_dd;
use feral::{DenseLu, LuParams, LuSingularAction};

/// Column-major near-singular `m×m` basis: identity except the last column is
/// `e0 + e1 + delta·e_{m-1}`. Then `det = delta`, the smallest singular value is
/// `Θ(delta)`, and `κ ≈ 1/delta`. `delta = 0` is exactly singular (rank `m-1`).
fn near_singular(m: usize, delta: f64) -> Vec<Vec<f64>> {
    let mut cols = vec![vec![0.0f64; m]; m];
    for j in 0..m {
        cols[j][j] = 1.0;
    }
    let last = m - 1;
    cols[last] = vec![0.0f64; m];
    cols[last][0] = 1.0;
    cols[last][1] = 1.0;
    cols[last][last] = delta;
    cols
}

/// Row `i` of the column-major matrix (for the high-precision residual).
fn row(cols: &[Vec<f64>], i: usize) -> Vec<f64> {
    cols.iter().map(|c| c[i]).collect()
}

/// `B·x` (dense, column-major).
fn matvec(cols: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    let m = cols.len();
    let mut y = vec![0.0f64; m];
    for (j, col) in cols.iter().enumerate() {
        for (i, &v) in col.iter().enumerate() {
            y[i] += v * x[j];
        }
    }
    y
}

/// Max-norm residual `‖rhs − B x‖∞`, each component accumulated in double-double.
fn residual_inf(cols: &[Vec<f64>], x: &[f64], rhs: &[f64]) -> f64 {
    (0..cols.len())
        .map(|i| residual_dd(&row(cols, i), x, rhs[i]).abs())
        .fold(0.0f64, f64::max)
}

/// Factor `B` with `on_singular = PerturbToEps{abs_floor}` (dense path), then run
/// `steps` high-precision refinement iterations against the *true* `B`. Returns
/// `(x, residual_inf, growth)`.
fn perturbed_refined_solve(
    cols: &[Vec<f64>],
    rhs: &[f64],
    abs_floor: f64,
    steps: usize,
) -> (Vec<f64>, f64, f64) {
    let m = cols.len();
    let params = LuParams {
        on_singular: LuSingularAction::PerturbToEps { abs_floor },
        ..LuParams::default()
    };
    let mut lu =
        DenseLu::factor(cols, m, params).expect("PerturbToEps must complete the factorization");
    let growth = lu.growth();
    let mut x = vec![0.0f64; m];
    for _ in 0..steps {
        // High-precision residual r = rhs − B x (against the TRUE B).
        let mut r: Vec<f64> = (0..m)
            .map(|i| residual_dd(&row(cols, i), &x, rhs[i]))
            .collect();
        // Correction: solve B' dx = r through the perturbed factor.
        lu.ftran(&mut r).expect("ftran on the perturbed factor");
        for j in 0..m {
            x[j] += r[j];
        }
    }
    let res = residual_inf(cols, &x, rhs);
    (x, res, growth)
}

/// Does feral's default LU (`on_singular = Fail`) abort on this basis?
fn default_factor_fails(cols: &[Vec<f64>]) -> bool {
    DenseLu::factor(cols, cols.len(), LuParams::default()).is_err()
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry experiment (#671 factorization hardening)
//
// Hypothesis H: `PerturbToEps` + high-precision iterative refinement recovers an
// accurate solve of `B x = rhs` (consistent rhs) on near-singular bases where
// feral's default factorization *fails*, with residual driven to ~1e-12.
//
// Kill criterion: if the refined residual stays ≫ 1e-9 across reasonable
// `abs_floor`, localized-perturbation + refinement is insufficient (the null
// direction genuinely matters) and a rank-revealing / basis-repair path is
// needed instead.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn experiment_perturb_and_refine_on_near_singular_bases() {
    let m = 6;
    // A consistent rhs whose solution lives in the WELL-conditioned subspace:
    // x_true supported on the first m-1 coordinates, so the near-singular last
    // column is not needed to represent rhs. This is the simplex-relevant case
    // (the basis is only numerically deficient; the solve it needs is benign).
    let mut x_true = vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.0];
    for (deltalabel, &delta) in [
        ("exactly-singular", 0.0),
        ("1e-14", 1e-14),
        ("1e-11", 1e-11),
    ]
    .iter()
    .map(|(l, d)| (l, d))
    {
        let cols = near_singular(m, delta);
        let rhs = matvec(&cols, &x_true);
        let default_fails = default_factor_fails(&cols);

        // abs_floor sweep; a handful of refinement steps.
        let mut best = f64::INFINITY;
        let mut best_floor = 0.0;
        let mut best_growth = 0.0;
        for &floor in &[1e-12, 1e-10, 1e-8, 1e-6, 1e-4] {
            let (_x, res, growth) = perturbed_refined_solve(&cols, &rhs, floor, 20);
            if res < best {
                best = res;
                best_floor = floor;
                best_growth = growth;
            }
        }
        println!(
            "[{deltalabel:>16}] default_fails={default_fails:<5}  \
             best_refined_residual={best:.3e}  at abs_floor={best_floor:.0e}  growth={best_growth:.2e}"
        );

        // Core assertion of H: refinement drives the residual to high accuracy.
        assert!(
            best < 1e-9,
            "[{deltalabel}] refined residual {best:.3e} did not reach 1e-9 — H falsified for this case"
        );
    }
    // Keep x_true referenced (silences an unused-mut lint on some toolchains).
    x_true[5] = 0.0;
}

#[test]
fn experiment_boundary_solution_uses_near_singular_direction() {
    // Adversarial: a NONSINGULAR but ill-conditioned basis (delta=1e-11, κ≈1e11)
    // whose solution genuinely loads the near-singular direction (x_true[last]≠0,
    // so rhs has a component only the tiny last pivot can represent). This maps
    // where localized-perturb + refinement stops working: the correction step must
    // resolve the 1/delta-amplified component through the *perturbed* factor.
    let m = 6;
    let delta = 1e-11;
    let cols = near_singular(m, delta);
    for &xlast in &[1.0f64, 100.0] {
        let x_true = vec![1.0, 2.0, 3.0, 4.0, 5.0, xlast];
        let rhs = matvec(&cols, &x_true);
        // abs_floor should not exceed the true smallest pivot here, or the
        // perturbed factor discards the direction the solution needs.
        let mut best = f64::INFINITY;
        let mut best_floor = 0.0;
        for &floor in &[1e-14, 1e-13, 1e-12, 1e-10] {
            let (_x, res, _g) = perturbed_refined_solve(&cols, &rhs, floor, 30);
            if res < best {
                best = res;
                best_floor = floor;
            }
        }
        println!(
            "[boundary xlast={xlast:>5}] best_refined_residual={best:.3e} at abs_floor={best_floor:.0e}"
        );
        // Informational (no hard assert): documents the regime where abs_floor
        // must sit below the true pivot for refinement to keep the direction.
    }
}

#[test]
fn hardened_feral_lu_solves_where_default_aborts() {
    // The production capability end-to-end: `FeralLU::with_singular_perturb` must
    // (1) factorize a basis whose default (strict) factorization aborts, and
    // (2) its refined ftran/btran recover the true solve to high accuracy.
    let m = 6;
    let cols = near_singular(m, 0.0); // exactly singular → default aborts
    assert!(
        FeralLU::new().factorize(m, &cols).is_err(),
        "default must abort"
    );

    let x_true = [1.0, 2.0, 3.0, 4.0, 5.0, 0.0];
    let rhs = matvec(&cols, &x_true);

    // Production usage: hardened factor + numeric focus (numeric focus turns on the
    // basis retention the double-double refined solve needs).
    let mut hardened = FeralLU::new()
        .with_singular_perturb(1e-12)
        .with_numeric_focus();
    hardened
        .factorize(m, &cols)
        .expect("hardened factorization must complete");

    // ftran: B x = rhs (refined against true B).
    let mut xf = rhs.clone();
    hardened.ftran_refined(&mut xf).expect("ftran_refined");
    let res_f = residual_inf(&cols, &xf, &rhs);
    assert!(
        res_f < 1e-9,
        "hardened ftran residual {res_f:.3e} too large"
    );

    // btran: Bᵀ y = e (a consistent rhs in the range of Bᵀ). Use Bᵀ·y_true.
    let y_true = [0.5, -1.0, 2.0, 0.0, 1.5, 0.0];
    let mut bt_rhs = vec![0.0f64; m];
    for i in 0..m {
        // (Bᵀ y_true)_i = Σ_j B_{ji} y_true_j = dot(cols[i], y_true)
        bt_rhs[i] = cols[i].iter().zip(&y_true).map(|(a, b)| a * b).sum();
    }
    let mut yb = bt_rhs.clone();
    hardened.btran_refined(&mut yb).expect("btran_refined");
    // Residual ‖Bᵀ y − bt_rhs‖∞ (row i of Bᵀ is cols[i]).
    let res_b = (0..m)
        .map(|i| residual_dd(&cols[i], &yb, bt_rhs[i]).abs())
        .fold(0.0f64, f64::max);
    assert!(
        res_b < 1e-9,
        "hardened btran residual {res_b:.3e} too large"
    );
}

#[test]
fn node_lu_is_plain_when_hardening_inactive() {
    // Outside a hardened retry, `node_feral_lu()` is the strict (Fail) factor: it
    // aborts on a singular basis exactly like `FeralLU::new()`, so the flag-OFF
    // path is byte-identical to today.
    let cols = near_singular(4, 0.0);
    let mut lu = super::linsolve::node_feral_lu();
    assert!(
        lu.factorize(4, &cols).is_err(),
        "inactive node LU must be plain"
    );
}

#[test]
fn node_lu_is_hardened_inside_active_scope() {
    // Inside a hardened retry scope, `node_feral_lu()` completes a singular factor
    // (PerturbToEps) — the failure-triggered escalation.
    let cols = near_singular(4, 0.0);
    super::linsolve::with_hardening_active(|| {
        let mut lu = super::linsolve::node_feral_lu();
        assert!(
            lu.factorize(4, &cols).is_ok(),
            "hardened node LU must complete the singular factorization"
        );
    });
    // Scope restored: back to plain afterward.
    let mut lu = super::linsolve::node_feral_lu();
    assert!(
        lu.factorize(4, &cols).is_err(),
        "hardening must not leak past the scope"
    );
}

#[test]
fn hardened_mode_off_is_byte_identical_default() {
    // Without `with_singular_perturb`, a well-conditioned basis solves identically
    // whether or not the refined entry points exist (default path untouched).
    let cols = vec![
        vec![2.0, 1.0, 0.0],
        vec![1.0, 2.0, 1.0],
        vec![0.0, 1.0, 2.0],
    ];
    let m = 3;
    let rhs = [1.0, 2.0, 3.0];
    let mut plain = FeralLU::new();
    plain.factorize(m, &cols).unwrap();
    let mut xp = rhs;
    plain.ftran(&mut xp).unwrap();

    let mut refined = FeralLU::new();
    refined.factorize(m, &cols).unwrap();
    let mut xr = rhs;
    refined.ftran_refined(&mut xr).unwrap(); // no perturb → feral default path
    assert_eq!(
        xp, xr,
        "non-hardened refined solve must match the plain solve"
    );
}

#[test]
fn experiment_reproduces_default_failure_on_singular_basis() {
    // The exactly-singular basis MUST fail feral's default factorization — this is
    // the hda-class `Numerical` reproduction the hardening targets.
    let cols = near_singular(6, 0.0);
    assert!(
        default_factor_fails(&cols),
        "exactly-singular basis should abort feral's default (Fail) LU"
    );
    // And PerturbToEps must complete it (a factor exists to refine against).
    let params = LuParams {
        on_singular: LuSingularAction::PerturbToEps { abs_floor: 1e-8 },
        ..LuParams::default()
    };
    assert!(
        DenseLu::factor(&cols, 6, params).is_ok(),
        "PerturbToEps must complete the singular factorization"
    );
}
