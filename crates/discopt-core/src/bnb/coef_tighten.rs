//! Big-M coefficient tightening for the MILP root presolve (#1917 G3.2 step 1).
//!
//! For a one-sided row `Σ_j a_j x_j <= R` whose maximal activity over the box,
//! `M = Σ_{a_j>0} a_j u_j + Σ_{a_j<0} a_j l_j`, is finite and exceeds `R`, let
//! `d = M - R > 0`. For an integer column `j` with `a_j > d`, setting `x_j <= u_j - 1`
//! makes the row redundant, so the row only matters at `x_j = u_j`. Replacing
//! `a_j` by `d` and `R` by `R + (d - a_j) u_j` keeps the row identical at
//! `x_j = u_j` and still redundant for every integer `x_j <= u_j - 1`. The mirror
//! case `a_j < -d` uses `-d` and `R + (-d - a_j) l_j`. `M - R` is unchanged by
//! each replacement, so one `d` serves the whole row.
//!
//! Ported from HiGHS, `highs/presolve/HPresolve.cpp`, `HPresolve::rowPresolve`,
//! the `strengthenCoefs` lambda (MIT license, Copyright (c) the HiGHS authors).
//! The same reduction is SCIP's `consdataTightenCoefs` in
//! `src/scip/cons_linear.c` (Apache-2.0).
//!
//! Soundness (both directions hold on the box `[l, u]`):
//! * **Integer projection is exact.** At every point of the box with integral
//!   integer columns, the original row holds iff the tightened row holds.
//! * **The LP relaxation only shrinks.** For any `x_j` in `[l_j, u_j]`, possibly
//!   fractional, the tightened row implies the original:
//!   `a_j x_j + rest = (d x_j + rest) + (a_j - d) x_j
//!    <= R + (d - a_j) u_j + (a_j - d) x_j = R - (a_j - d)(u_j - x_j) <= R`.
//!   So a point that verifies against the tightened rows also verifies against
//!   the original ones, with no larger residual.
//!
//! Floating point: `M` is summed with upward rounding, so `d` is an upper bound
//! on the exact gap, which only weakens the reduction. Each new right-hand side
//! is an upward-rounded upper bound of the exact one. Both errors leave the
//! tightened row valid.
//!
//! The bounds `l, u` must be valid for every feasible point. The driver passes
//! its FBBT-presolved root bounds, which are.
//!
//! Default-box and precision guards. A column side with `|bound| >=
//! EFFECTIVE_INF` (`1e19`) is unbounded for this pass, so its row has infinite
//! maximal activity and is skipped. This covers the Python default box
//! `±9.999e19`, which the `>= 1e20` sentinel test alone reads as finite. A row is
//! also skipped when the rounding error of its activity sum,
//! `gamma(k+2) * Σ|a_j bound_j|` (Higham §3.1, [`crate::numeric::gamma`]), exceeds
//! `COEF_TOL * d`. In that case the new coefficient `d` would be dominated by
//! cancellation in `M - R`. HiGHS computes the same gap in compensated
//! double-double (`HighsCDouble`) and refuses an infinite implied row bound
//! (`impliedRowUpper != kHighsInf`). SCIP flags any contribution of
//! `|a_j bound_j| >= numerics/hugeval` (1e15) as not tight
//! (`cons_linear.c` `getMaxActivity`). Both serve the same purpose: never build
//! a coefficient out of a cancelled huge activity.

use crate::lp::simplex::sparse::SparseCols;
use crate::numeric::{gamma, EFFECTIVE_INF};
use crate::presolve::directed::{add_up, mul_up, sub_down, sub_up};

/// The Rust LP layer's infinity sentinel (CLAUDE.md: `1e20`, not `f64::INFINITY`).
const INF: f64 = 1e20;
/// Minimum gap `d = M - R` worth reducing to. At or below it the row is
/// redundant or forcing, and HiGHS removes it elsewhere.
const GAP_TOL: f64 = 1e-6;
/// A coefficient is tightened only if it exceeds `d` by this relative margin
/// (HiGHS uses `primal_feastol` absolute).
const COEF_TOL: f64 = 1e-7;

/// What one pass changed, for the caller's anti-vacuity counters (CLAUDE.md §6).
#[derive(Debug, Default, Clone, PartialEq)]
pub struct CoefTightenStats {
    /// Rows the pass examined. A qualifying row is one-sided `<=`, has a
    /// standard logical, and has finite maximal activity.
    pub rows_examined: usize,
    /// Rows with at least one tightened coefficient.
    pub rows_tightened: Vec<usize>,
    /// Coefficients replaced.
    pub coefs_tightened: usize,
    /// Integer-bearing `<=` rows skipped because a side was at or past
    /// `EFFECTIVE_INF` (the default box included).
    pub rows_skipped_unbounded: usize,
    /// Rows skipped because `d` was not known to relative `COEF_TOL`.
    pub rows_skipped_precision: usize,
}

/// Upper bound of `delta * bound` for `delta` known only to lie in `[lo, hi]`.
fn prod_up(lo: f64, hi: f64, bound: f64) -> f64 {
    mul_up(lo, bound).max(mul_up(hi, bound))
}

/// One pass of big-M coefficient tightening over the standard-form matrix
/// `A z = b`. Columns `[0, ns)` are structural; row `i` owns the logical at
/// column `ns + i`.
///
/// A row is treated as `Σ_{j<ns} a_ij x_j <= b_i` only if its logical has a
/// single `+1` entry in row `i` and the ORIGINAL bounds `[0, INF]`
/// (`slack_l`, `slack_u` are the caller's pre-presolve bounds, indexed like `l`).
/// Every other row is left alone: equality logicals, ranged rows, and any
/// nonstandard logical. Only structural integer columns (`is_int[j]`, `j < ns`)
/// change, and their bounds `l[j], u[j]` are integral.
///
/// The matrix and `b` are edited in place. The column structure of the CSC is
/// unchanged; only values change, so no index anywhere goes stale.
#[allow(clippy::too_many_arguments)]
pub fn tighten_big_m_coefficients(
    csc: &mut SparseCols,
    m: usize,
    n: usize,
    ns: usize,
    b: &mut [f64],
    l: &[f64],
    u: &[f64],
    slack_l: &[f64],
    slack_u: &[f64],
    is_int: &[bool],
) -> CoefTightenStats {
    let mut stats = CoefTightenStats::default();
    if ns + m > n {
        return stats;
    }
    let (col_ptr, row_idx, vals) = csc.raw();

    // Which rows have a standard `<=` logical: one entry, +1, in its own row,
    // with original bounds [0, INF].
    let mut le_row = vec![false; m];
    for (i, flag) in le_row.iter_mut().enumerate() {
        let k = ns + i;
        let (s, e) = (col_ptr[k], col_ptr[k + 1]);
        *flag = e - s == 1
            && row_idx[s] == i
            && vals[s] == 1.0
            && slack_l[k] == 0.0
            && slack_u[k] >= INF
            && b[i].abs() < EFFECTIVE_INF;
    }

    // Row-wise view of the structural entries: (position in `vals`, column).
    let mut rows: Vec<Vec<(usize, usize)>> = vec![Vec::new(); m];
    for j in 0..ns {
        for p in col_ptr[j]..col_ptr[j + 1] {
            let i = row_idx[p];
            if i < m && le_row[i] && vals[p] != 0.0 {
                rows[i].push((p, j));
            }
        }
    }

    // Planned edits: (position, new value). Applied after the scan so that the
    // immutable borrow of `vals` ends first.
    let mut edits: Vec<(usize, f64)> = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        if row.is_empty() || !row.iter().any(|&(_, j)| is_int[j]) {
            continue;
        }
        // Upward-rounded maximal activity. A term whose bound is the sentinel is
        // unbounded. Test the BOUND, never a product (CLAUDE.md, INF note).
        // A side at or past EFFECTIVE_INF (the Python default box ±9.999e19
        // included) is unbounded.
        let mut maxact = 0.0f64;
        let mut abs_sum = 0.0f64;
        let mut finite = true;
        for &(p, j) in row {
            let a = vals[p];
            let bnd = if a > 0.0 { u[j] } else { l[j] };
            if bnd.abs() >= EFFECTIVE_INF || !bnd.is_finite() {
                finite = false;
                break;
            }
            let t = mul_up(a, bnd);
            maxact = add_up(maxact, t);
            abs_sum = add_up(abs_sum, t.abs());
        }
        if !finite || !maxact.is_finite() {
            stats.rows_skipped_unbounded += 1;
            continue;
        }
        stats.rows_examined += 1;
        let mut rhs = b[i];
        let d = sub_up(maxact, rhs);
        if d.is_nan() || d <= GAP_TOL || !d.is_finite() {
            continue;
        }
        // Precision: `d` must be known to relative COEF_TOL. Its error is at most
        // the activity sum's (k terms, plus the rhs subtraction).
        let err = gamma(row.len() + 2) * add_up(abs_sum, rhs.abs());
        if !err.is_finite() || err > COEF_TOL * d {
            stats.rows_skipped_precision += 1;
            continue;
        }
        let mut touched = false;
        for &(p, j) in row {
            if !is_int[j] {
                continue;
            }
            // The redundancy argument needs `x_j <= u_j - 1` (or `>= l_j + 1`)
            // to be the next integer value; a non-integral bound breaks it.
            if u[j].fract() != 0.0 || l[j].fract() != 0.0 {
                continue;
            }
            let a = vals[p];
            let margin = COEF_TOL * a.abs().max(1.0);
            if a > d + margin {
                // a_j -> d,  R -> R + (d - a_j) u_j   (HPresolve strengthenCoefs, val > max)
                let ub = u[j];
                let dl = prod_up(sub_down(d, a), sub_up(d, a), ub);
                rhs = add_up(rhs, dl);
                edits.push((p, d));
                touched = true;
                stats.coefs_tightened += 1;
            } else if a < -d - margin {
                // a_j -> -d, R -> R + (-d - a_j) l_j  (HPresolve strengthenCoefs, val < -max)
                let lb = l[j];
                let dl = prod_up(sub_down(-d, a), sub_up(-d, a), lb);
                rhs = add_up(rhs, dl);
                edits.push((p, -d));
                touched = true;
                stats.coefs_tightened += 1;
            }
        }
        if touched {
            b[i] = rhs;
            stats.rows_tightened.push(i);
        }
    }

    if !edits.is_empty() {
        let (cp, ri, v) = csc.raw();
        let (cp, ri, mut v) = (cp.to_vec(), ri.to_vec(), v.to_vec());
        for (p, val) in edits {
            v[p] = val;
        }
        *csc = SparseCols::from_csc(cp, ri, v);
    }
    stats
}

/// Whether `DISCOPT_RUST_MILP_COEF_TIGHTEN` asks for the pass (default OFF).
///
/// Read fresh on every solve, never cached: a differential panel that flips the
/// variable between solves in one process must get the arm it asked for. An
/// unrecognized value is a hard refusal, so a panel cannot silently run the
/// wrong arm.
pub fn coef_tighten_enabled() -> bool {
    match std::env::var("DISCOPT_RUST_MILP_COEF_TIGHTEN") {
        Err(_) => false,
        Ok(v) => match parse_coef_tighten_flag(&v) {
            Ok(on) => on,
            Err(msg) => panic!("{msg}"),
        },
    }
}

/// Parse a `DISCOPT_RUST_MILP_COEF_TIGHTEN` value. Kept pure so tests can assert the
/// refusal directly.
pub fn parse_coef_tighten_flag(raw: &str) -> Result<bool, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "0" | "false" | "no" => Ok(false),
        "1" | "true" | "yes" => Ok(true),
        other => Err(format!(
            "DISCOPT_RUST_MILP_COEF_TIGHTEN={other:?} is not recognized (expected \
             1/true/yes or 0/false/no). Refusing rather than silently picking an arm."
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build standard form `[A | I] z = b` from dense `<=` rows, with logicals at
    /// `[0, INF]`.
    fn std_form(rows: &[Vec<f64>], ns: usize) -> (SparseCols, usize, usize) {
        let m = rows.len();
        let n = ns + m;
        let mut dense = vec![0.0; m * n];
        for (i, r) in rows.iter().enumerate() {
            dense[i * n..i * n + ns].copy_from_slice(r);
            dense[i * n + ns + i] = 1.0;
        }
        (SparseCols::from_dense(&dense, m, n), m, n)
    }

    fn row_coefs(csc: &SparseCols, m: usize, ns: usize) -> Vec<Vec<f64>> {
        let (cp, ri, v) = csc.raw();
        let mut out = vec![vec![0.0; ns]; m];
        for j in 0..ns {
            for p in cp[j]..cp[j + 1] {
                out[ri[p]][j] = v[p];
            }
        }
        out
    }

    fn holds(a: &[f64], rhs: f64, x: &[f64]) -> bool {
        let act: f64 = a.iter().zip(x).map(|(p, q)| p * q).sum();
        act <= rhs + 1e-9 * (1.0 + rhs.abs())
    }

    fn bounds(lx: &[f64], ux: &[f64], m: usize) -> (Vec<f64>, Vec<f64>) {
        let mut l = lx.to_vec();
        let mut u = ux.to_vec();
        l.extend(std::iter::repeat_n(0.0, m));
        u.extend(std::iter::repeat_n(INF, m));
        (l, u)
    }

    #[test]
    fn classic_big_m_shrinks_to_the_bound() {
        // x - 1000 y <= 0 with x in [0, 64], y binary  ->  x - 64 y <= 0.
        let ns = 2;
        let (mut csc, m, n) = std_form(&[vec![1.0, -1000.0]], ns);
        let mut b = vec![0.0];
        let (l, u) = bounds(&[0.0, 0.0], &[64.0, 1.0], m);
        let st =
            tighten_big_m_coefficients(&mut csc, m, n, ns, &mut b, &l, &u, &l, &u, &[false, true]);
        assert_eq!(st.coefs_tightened, 1);
        assert_eq!(st.rows_tightened, vec![0]);
        let a = row_coefs(&csc, m, ns);
        // maxact = 64, d = 64: y -> -64, rhs -> 0 + (-64 + 1000)*0 = 0.
        assert_eq!(a[0], vec![1.0, -64.0]);
        assert_eq!(b[0], 0.0);
    }

    #[test]
    fn positive_big_m_on_the_y_eq_one_branch() {
        // 2x + 10100 y <= 10100 + 30, x in [0, 40]: maxact = 80 + 10100 = 10180,
        // d = 50 -> y coef 50, rhs = 10130 + (50 - 10100)*1 = 80.
        let ns = 2;
        let (mut csc, m, n) = std_form(&[vec![2.0, 10100.0]], ns);
        let mut b = vec![10130.0];
        let (l, u) = bounds(&[0.0, 0.0], &[40.0, 1.0], m);
        let st =
            tighten_big_m_coefficients(&mut csc, m, n, ns, &mut b, &l, &u, &l, &u, &[false, true]);
        assert_eq!(st.coefs_tightened, 1);
        let a = row_coefs(&csc, m, ns);
        assert!(
            (a[0][1] - 50.0).abs() < 1e-9 && (b[0] - 80.0).abs() < 1e-9,
            "{a:?} {b:?}"
        );
    }

    #[test]
    fn unbounded_rest_and_redundant_rows_are_left_alone() {
        let ns = 3;
        // Row 0: x0 unbounded above, so the max activity is infinite and the row
        // is skipped. Row 1: redundant (maxact 2 <= 5), so it is skipped.
        let (mut csc, m, n) = std_form(&[vec![1.0, -1000.0, 0.0], vec![1.0, 0.0, 1.0]], ns);
        let mut b = vec![0.0, 5.0];
        let (l, u) = bounds(&[0.0, 0.0, 0.0], &[INF, 1.0, 1.0], m);
        let mut u2 = u.clone();
        u2[0] = INF;
        let before = row_coefs(&csc, m, ns);
        let st = tighten_big_m_coefficients(
            &mut csc,
            m,
            n,
            ns,
            &mut b,
            &l,
            &u2,
            &l,
            &u,
            &[false, true, true],
        );
        assert_eq!(st.coefs_tightened, 0);
        assert_eq!(row_coefs(&csc, m, ns), before);
        assert_eq!(b, vec![0.0, 5.0]);
    }

    #[test]
    fn equality_and_nonstandard_logicals_are_skipped() {
        let ns = 2;
        let (mut csc, m, n) = std_form(&[vec![1.0, -1000.0]], ns);
        let mut b = vec![0.0];
        let (l, mut u) = bounds(&[0.0, 0.0], &[64.0, 1.0], m);
        u[ns] = 0.0; // logical fixed at 0: an equality row
        let st =
            tighten_big_m_coefficients(&mut csc, m, n, ns, &mut b, &l, &u, &l, &u, &[false, true]);
        assert_eq!(st.coefs_tightened, 0);
        assert_eq!(st.rows_examined, 0);
    }

    /// A Python default-box side (`±9.999e19`, below the `1e20` sentinel) is
    /// unbounded for the pass. `x_a - x_b + 100 y <= 10` with `x_a` in
    /// `[0, 9.999e19]` and `x_b` fixed at `9.999e19` has a float max activity of
    /// exactly 100, so without the guard the pass rewrites `100 y` to `90 y` from a
    /// cancelled default-box activity.
    #[test]
    fn default_box_sides_are_unbounded() {
        let ns = 3;
        let (mut csc, m, n) = std_form(&[vec![1.0, -1.0, 100.0]], ns);
        let mut b = vec![10.0];
        let big = 9.999e19;
        let (l, u) = bounds(&[0.0, big, 0.0], &[big, big, 1.0], m);
        let before = row_coefs(&csc, m, ns);
        let st = tighten_big_m_coefficients(
            &mut csc,
            m,
            n,
            ns,
            &mut b,
            &l,
            &u,
            &l,
            &u,
            &[false, false, true],
        );
        assert_eq!(st.coefs_tightened, 0, "{st:?}");
        assert_eq!(st.rows_skipped_unbounded, 1);
        assert_eq!(row_coefs(&csc, m, ns), before);
        assert_eq!(b, vec![10.0]);
        // The same row with an ordinary box still tightens. That keeps the test
        // from passing because the row shape never qualified at all.
        let (mut csc, m, n) = std_form(&[vec![1.0, -1.0, 100.0]], ns);
        let mut b = vec![10.0];
        let (l, u) = bounds(&[0.0, 5.0, 0.0], &[5.0, 5.0, 1.0], m);
        let st = tighten_big_m_coefficients(
            &mut csc,
            m,
            n,
            ns,
            &mut b,
            &l,
            &u,
            &l,
            &u,
            &[false, false, true],
        );
        assert_eq!(st.coefs_tightened, 1);
    }

    /// A large but finite cancelling activity is refused on precision: with
    /// `x_b` fixed at `1e17`, the float sum's error bound `gamma(k+2) * 2e17`
    /// (about 2e2) dwarfs `COEF_TOL * d`.
    #[test]
    fn cancelled_huge_activity_is_refused_on_precision() {
        let ns = 3;
        let (mut csc, m, n) = std_form(&[vec![1.0, -1.0, 100.0]], ns);
        let mut b = vec![10.0];
        let (l, u) = bounds(&[0.0, 1e17, 0.0], &[1e17, 1e17, 1.0], m);
        let st = tighten_big_m_coefficients(
            &mut csc,
            m,
            n,
            ns,
            &mut b,
            &l,
            &u,
            &l,
            &u,
            &[false, false, true],
        );
        assert_eq!(st.coefs_tightened, 0, "{st:?}");
        assert_eq!(st.rows_skipped_precision, 1);
        assert_eq!(b, vec![10.0]);
    }

    #[test]
    fn flag_parse_refuses_unknown_values() {
        assert_eq!(parse_coef_tighten_flag(""), Ok(false));
        assert_eq!(parse_coef_tighten_flag("0"), Ok(false));
        assert_eq!(parse_coef_tighten_flag(" Yes "), Ok(true));
        assert!(parse_coef_tighten_flag("on").is_err());
    }

    /// Tiny deterministic generator, so the property test needs no dev-dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.next()
        }
    }

    /// Property test over random big-M rows mixing binaries, a general integer
    /// and continuous columns:
    /// (1) at every enumerated integer assignment, with continuous columns at box
    ///     vertices and random interior points, the original row and the
    ///     tightened row agree (the integer projection is exact);
    /// (2) at random fractional points of the box, tightened implies original
    ///     (the LP region only shrinks).
    #[test]
    fn tightened_rows_keep_the_integer_projection_exactly() {
        let mut rng = Lcg(0x1917_6332);
        let mut checks = 0usize;
        let mut tightened = 0usize;
        for _case in 0..300 {
            // Columns: 2 continuous, 3 binaries, 1 general integer in [-2, 3].
            let ns = 6;
            let is_int = [false, false, true, true, true, true];
            let lx = [rng.range(-5.0, 0.0), 0.0, 0.0, 0.0, 0.0, -2.0];
            let ux = [
                rng.range(0.5, 6.0),
                rng.range(1.0, 50.0),
                1.0,
                1.0,
                1.0,
                3.0,
            ];
            let mut a = vec![0.0; ns];
            for (j, aj) in a.iter_mut().enumerate() {
                let big = if is_int[j] && rng.next() < 0.6 {
                    1000.0
                } else {
                    5.0
                };
                *aj = rng.range(-big, big);
                if rng.next() < 0.15 {
                    *aj = 0.0;
                }
            }
            if a.iter().all(|&v| v == 0.0) {
                continue;
            }
            // rhs drawn between min and max activity, so the row is non-trivial.
            let (mut mn, mut mx) = (0.0, 0.0);
            for j in 0..ns {
                let (p, q) = (a[j] * lx[j], a[j] * ux[j]);
                mn += p.min(q);
                mx += p.max(q);
            }
            let rhs = rng.range(mn, mx);
            let (mut csc, m, n) = std_form(&[a.clone()], ns);
            let mut b = vec![rhs];
            let (l, u) = bounds(&lx, &ux, 1);
            let st =
                tighten_big_m_coefficients(&mut csc, m, n, ns, &mut b, &l, &u, &l, &u, &is_int);
            tightened += st.coefs_tightened;
            let at = row_coefs(&csc, m, ns)[0].clone();
            // Continuous columns are never touched.
            assert_eq!(at[0], a[0]);
            assert_eq!(at[1], a[1]);
            // (1) Integer assignments x continuous vertices + interior samples.
            for y in 0..8u32 {
                for gi in -2..=3i32 {
                    let mut conts: Vec<[f64; 2]> = vec![
                        [lx[0], lx[1]],
                        [lx[0], ux[1]],
                        [ux[0], lx[1]],
                        [ux[0], ux[1]],
                    ];
                    for _ in 0..4 {
                        conts.push([rng.range(lx[0], ux[0]), rng.range(lx[1], ux[1])]);
                    }
                    for cv in conts {
                        let x = [
                            cv[0],
                            cv[1],
                            (y & 1) as f64,
                            ((y >> 1) & 1) as f64,
                            ((y >> 2) & 1) as f64,
                            gi as f64,
                        ];
                        let o = holds(&a, rhs, &x);
                        let t = holds(&at, b[0], &x);
                        // Allow a hair of slack at the boundary where the two
                        // rows' round-off differs.
                        if o != t {
                            let act_o: f64 = a.iter().zip(&x).map(|(p, q)| p * q).sum();
                            let act_t: f64 = at.iter().zip(&x).map(|(p, q)| p * q).sum();
                            let near = (act_o - rhs).abs() < 1e-7 * (1.0 + rhs.abs())
                                || (act_t - b[0]).abs() < 1e-7 * (1.0 + b[0].abs());
                            assert!(near, "integer projection changed: a={a:?} rhs={rhs} at={at:?} b={} x={x:?} o={o} t={t}", b[0]);
                        }
                        checks += 1;
                    }
                }
            }
            // (2) Fractional points: tightened => original.
            for _ in 0..64 {
                let x: Vec<f64> = (0..ns).map(|j| rng.range(lx[j], ux[j])).collect();
                if holds(&at, b[0], &x) {
                    assert!(
                        holds(&a, rhs, &x),
                        "LP region grew: a={a:?} at={at:?} x={x:?}"
                    );
                }
                checks += 1;
            }
        }
        assert!(
            tightened > 100,
            "the property test tightened only {tightened} coefficients"
        );
        assert!(checks > 100_000, "only {checks} checks executed");
    }
}
