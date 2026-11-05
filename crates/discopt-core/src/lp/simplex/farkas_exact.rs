//! Exact Farkas certificate over open columns (highs-parity P3.2,
//! `DISCOPT_FARKAS_EXACT_OPEN`, default OFF).
//!
//! **The gap it closes.** [`super::primal::farkas_ray_certifies_cols`] verifies a
//! ray `y` for `{A x = b, l ≤ x ≤ u}` in floating point: each reduced cost
//! `(Aᵀy)_j` is known only to an interval `[t − e, t + e]` with
//! `e = γ·Σ|a_ij·y_i| > 0`. For a column that is open on a side (free, or
//! `u_j = INF`), a true certificate needs `(Aᵀy)_j` exactly zero (or of the safe
//! sign), and the interval of a float ray *always* straddles zero, so it selects the
//! open side and the ray is rejected (`FarkasRejectOpen`). That is structural, not
//! noise: no float rescaling or projection of `y` can certify it. Measured
//! (highs-parity P3.2): on two P0.6 IIS fixtures the Rust MILP driver logged 516
//! `FarkasRejectOpen` and left the probes undecided, where HiGHS (and the Python
//! ladder's rational correction, P2.5) prove them infeasible.
//!
//! **What this does.** Over the rationals (`f64` values convert exactly): repeatedly
//! find the columns whose exact reduced cost selects an open side, and solve
//! `A_{R,S}ᵀ dy = −(Aᵀy)_S` for the held set `S` so their reduced costs become
//! exactly zero (sparse rational elimination, the P2.5 algorithm, with the same work
//! caps); when no column selects an open side, check the Farkas inequality
//! `bᵀy > Σ_j sup_{x_j ∈ [l_j, u_j]} (Aᵀy)_j x_j` **exactly**. Both orientations of
//! `y` are tried.
//!
//! **Why it is sound.** Weak duality holds for *every* `y`: if `bᵀy` exceeds that
//! supremum, no `x` in the box satisfies `A x = b`. The arithmetic here is exact, so
//! a `true` is a proof whatever float choices led to the candidate; the float ray
//! only decides whether a certificate is *found*, never whether it is valid. A refusal
//! (work cap, inconsistent system, still an open side, inequality fails) returns
//! `false`, which is the pre-flag behaviour.

use num_rational::BigRational;
use num_traits::{Signed, Zero};
use std::collections::{BTreeMap, BTreeSet};

use super::sparse::SparseCols;

/// The Rust LP layer's infinity sentinel.
const INF: f64 = 1e20;
/// Most columns one correction may hold at an exact zero (as `EXACT_MAX_COLUMNS`).
const MAX_COLUMNS: usize = 256;
/// Correction rounds (as `EXACT_MAX_ROUNDS`).
const MAX_ROUNDS: usize = 64;
/// Bit-operation budget of one elimination, and of all of one certificate's
/// eliminations together (as `EXACT_MAX_WORK` / `EXACT_MAX_TOTAL_WORK`): a count,
/// not a clock, so whether a certificate is found does not depend on machine speed.
const MAX_WORK: u64 = 200_000_000;
const MAX_TOTAL_WORK: u64 = 400_000_000;

/// Whether the exact open-column Farkas check runs (`DISCOPT_FARKAS_EXACT_OPEN`,
/// read once). **Default OFF:** it changes which node LPs are proven infeasible, so
/// which nodes the MILP and spatial trees fathom -- bound-changing per CLAUDE.md §5,
/// so it graduates only through a corpus panel (row in
/// `docs/dev/flag-retirement-audit.md`).
pub(crate) fn farkas_exact_open_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let raw = std::env::var("DISCOPT_FARKAS_EXACT_OPEN").unwrap_or_default();
        match raw.trim() {
            "" | "0" | "false" | "False" | "off" | "OFF" => false,
            "1" | "true" | "True" | "on" | "ON" => true,
            other => panic!(
                "DISCOPT_FARKAS_EXACT_OPEN={other:?} is not a recognized value; use 1/on or 0/off"
            ),
        }
    })
}

fn q(v: f64) -> BigRational {
    // Finite inputs only (the caller checks); `from_float` is exact for every finite f64.
    BigRational::from_float(v).expect("finite float")
}

fn bits(v: &BigRational) -> u64 {
    (v.numer().bits() + v.denom().bits()).max(1)
}

/// `(Aᵀy)_j` exactly.
fn reduced(cols: &SparseCols, j: usize, y: &[BigRational]) -> BigRational {
    let (rows, vals) = cols.col(j);
    let mut s = BigRational::zero();
    for (k, &i) in rows.iter().enumerate() {
        if !y[i].is_zero() && vals[k] != 0.0 {
            s += q(vals[k]) * &y[i];
        }
    }
    s
}

/// Does a reduced cost of this sign select an open side of column `j`?
fn selects_open(rc: &BigRational, l: f64, u: f64) -> bool {
    (rc.is_positive() && u >= INF) || (rc.is_negative() && l <= -INF)
}

/// Solve `M dy = rhs` where row `k` of `M` is column `cols_s[k]` of `A` restricted to
/// the rows it touches (unknowns: `dy_i` for every touched row; free unknowns are 0).
/// Sparse rational Gaussian elimination, fewest-nonzeros pivots. `None` when an
/// equation reduces to `0 = c ≠ 0` (no exact zero is reachable on this set) or the
/// work caps are hit.
fn solve_rect(
    cols: &SparseCols,
    cols_s: &[usize],
    rhs: &[BigRational],
    spent: &mut u64,
) -> Option<BTreeMap<usize, BigRational>> {
    // Equations as maps row-unknown -> coefficient, with the rhs alongside.
    let mut eqs: Vec<(BTreeMap<usize, BigRational>, BigRational)> = cols_s
        .iter()
        .zip(rhs)
        .map(|(&j, r)| {
            let (rows, vals) = cols.col(j);
            let mut m = BTreeMap::new();
            for (k, &i) in rows.iter().enumerate() {
                if vals[k] != 0.0 {
                    m.insert(i, q(vals[k]));
                }
            }
            (m, r.clone())
        })
        .collect();
    let mut work = 0u64;
    let mut pivots: Vec<(usize, usize)> = Vec::new(); // (equation, unknown)
    let mut done = vec![false; eqs.len()];
    loop {
        // Pick the unfinished equation with the fewest nonzeros; pivot on its first.
        let next = (0..eqs.len())
            .filter(|&k| !done[k])
            .min_by_key(|&k| (eqs[k].0.len(), k));
        let Some(k) = next else { break };
        done[k] = true;
        if eqs[k].0.is_empty() {
            if !eqs[k].1.is_zero() {
                return None; // 0 = c, c ≠ 0
            }
            continue; // a dependent equation, consistent
        }
        let (&var, p) = eqs[k].0.iter().next().map(|(a, b)| (a, b.clone())).unwrap();
        let prow: Vec<(usize, BigRational)> =
            eqs[k].0.iter().map(|(a, b)| (*a, b.clone())).collect();
        let prhs = eqs[k].1.clone();
        let row_bits: u64 = prow.iter().map(|(_, v)| bits(v)).max().unwrap_or(1) + bits(&prhs);
        for (r, eq) in eqs.iter_mut().enumerate() {
            if r == k || done[r] {
                continue;
            }
            let Some(c) = eq.0.remove(&var) else { continue };
            let f = c / &p;
            let step = (prow.len() as u64 + 1) * (bits(&f) + row_bits);
            work += step;
            *spent += step;
            if work > MAX_WORK || *spent > MAX_TOTAL_WORK {
                return None;
            }
            for (col, v) in &prow {
                if *col == var {
                    continue;
                }
                let nv = eq.0.get(col).cloned().unwrap_or_else(BigRational::zero) - &f * v;
                if nv.is_zero() {
                    eq.0.remove(col);
                } else {
                    eq.0.insert(*col, nv);
                }
            }
            eq.1 = &eq.1 - &f * &prhs;
        }
        pivots.push((k, var));
    }
    // Back-substitute in reverse pivot order; unknowns never pivoted stay 0.
    let mut sol: BTreeMap<usize, BigRational> = BTreeMap::new();
    for &(k, var) in pivots.iter().rev() {
        let (m, r) = &eqs[k];
        let mut acc = r.clone();
        for (col, v) in m {
            if *col != var {
                if let Some(x) = sol.get(col) {
                    acc -= v * x;
                }
            }
        }
        let piv = m.get(&var).expect("pivot entry");
        sol.insert(var, acc / piv);
    }
    Some(sol)
}

/// One orientation: try to turn `y` into an exact Farkas certificate.
fn certify_one(y: &[f64], cols: &SparseCols, n: usize, b: &[f64], l: &[f64], u: &[f64]) -> bool {
    let mut yq: Vec<BigRational> = y.iter().map(|&v| q(v)).collect();
    let open_cols: Vec<usize> = (0..n).filter(|&j| l[j] <= -INF || u[j] >= INF).collect();
    let mut held: BTreeSet<usize> = BTreeSet::new();
    let mut spent = 0u64;
    for round in 0..=MAX_ROUNDS {
        let wrong: Vec<usize> = open_cols
            .iter()
            .copied()
            .filter(|&j| selects_open(&reduced(cols, j, &yq), l[j], u[j]))
            .collect();
        if wrong.is_empty() {
            break;
        }
        if round == MAX_ROUNDS {
            return false;
        }
        held.extend(wrong);
        if held.len() > MAX_COLUMNS {
            return false;
        }
        let set: Vec<usize> = held.iter().copied().collect();
        // Drive every held column's reduced cost to exactly zero: (A_S)ᵀ dy = −(Aᵀy)_S.
        let rhs: Vec<BigRational> = set.iter().map(|&j| -reduced(cols, j, &yq)).collect();
        let Some(dy) = solve_rect(cols, &set, &rhs, &mut spent) else {
            return false;
        };
        for (i, v) in dy {
            yq[i] += v;
        }
    }
    // Exact Farkas inequality: bᵀy − Σ_j sup_{x_j ∈ box} rc_j x_j > 0.
    let mut g = BigRational::zero();
    for (bi, yi) in b.iter().zip(&yq) {
        if *bi != 0.0 && !yi.is_zero() {
            g += q(*bi) * yi;
        }
    }
    for j in 0..n {
        let rc = reduced(cols, j, &yq);
        if rc.is_zero() {
            continue;
        }
        if rc.is_positive() {
            if u[j] >= INF {
                return false;
            }
            g -= &rc * q(u[j]);
        } else {
            if l[j] <= -INF {
                return false;
            }
            g -= &rc * q(l[j]);
        }
    }
    g.is_positive()
}

/// Is `y` (in either orientation), corrected exactly on its open columns, a Farkas
/// certificate that `{A x = b, l ≤ x ≤ u}` is empty? `false` decides nothing.
pub(crate) fn farkas_exact_open(
    y: &[f64],
    cols: &SparseCols,
    n: usize,
    m: usize,
    b: &[f64],
    l: &[f64],
    u: &[f64],
) -> bool {
    if y.len() != m
        || b.len() != m
        || l.len() != n
        || u.len() != n
        || !y.iter().chain(b).chain(l).chain(u).all(|v| v.is_finite())
    {
        return false;
    }
    crate::profile::incr(crate::profile::Ctr::FarkasExactAttempt);
    for sgn in [1.0f64, -1.0] {
        let ys: Vec<f64> = y.iter().map(|v| sgn * v).collect();
        if certify_one(&ys, cols, n, b, l, u) {
            crate::profile::incr(crate::profile::Ctr::FarkasExactRescue);
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn csc(a: &[f64], m: usize, n: usize) -> SparseCols {
        SparseCols::from_dense(a, m, n)
    }

    /// The P3.2 probe: rows `-3x0+2x1+3x2+x3 - s3 = 0`, `2x0-x1+3x2 - s4 = 0`,
    /// `s3 <= 4.5`, `s4 <= -2.5`, `x1` free, `x0 >= 0`, `x2, x3 in [0, 3]`.
    /// Infeasible (row3 + 2 row4 forces `x0 + 9x2 + x3 <= -0.5`).
    fn probe() -> (SparseCols, usize, usize, Vec<f64>, Vec<f64>, Vec<f64>) {
        let a = vec![
            -3.0, 2.0, 3.0, 1.0, -1.0, 0.0, //
            2.0, -1.0, 3.0, 0.0, 0.0, -1.0,
        ];
        let (m, n) = (2, 6);
        let b = vec![0.0, 0.0];
        let l = vec![0.0, -INF, 0.0, 0.0, -INF, -INF];
        let u = vec![INF, INF, 3.0, 3.0, 4.5, -2.5];
        (csc(&a, m, n), m, n, b, l, u)
    }

    #[test]
    fn certifies_a_free_column_ray_the_float_check_cannot() {
        let (c, m, n, b, l, u) = probe();
        // A noisy multiple of the true ray (1, 2): x1's reduced cost is ~1e-17, not 0.
        let y = vec![0.333_333_333_333_333_3, 0.666_666_666_666_666_7];
        // The float verifier (flag unset in tests, so no exact fallback) rejects it.
        assert!(!super::super::primal::farkas_ray_certifies_cols(
            &y, &c, n, m, &b, &l, &u
        ));
        assert!(farkas_exact_open(&y, &c, n, m, &b, &l, &u));
    }

    #[test]
    fn never_certifies_a_feasible_system() {
        // Same rows with s4 <= +2.5: feasible (x = 0, x1 = 2, s3 = 4, s4 = -2).
        let (c, m, n, b, l, mut u) = probe();
        u[5] = 2.5;
        let mut checked = 0;
        for y in [
            vec![1.0, 2.0],
            vec![0.3, 0.7],
            vec![-1.0, 1.0],
            vec![1e-3, 5.0],
        ] {
            assert!(!farkas_exact_open(&y, &c, n, m, &b, &l, &u), "{y:?}");
            checked += 1;
        }
        assert_eq!(checked, 4);
    }

    #[test]
    fn inconsistent_correction_is_refused() {
        // x free in x = 1 and x = 2 written as two rows: any ray has rc_x = y0 + y1;
        // the exact system y0 + y1 = 0 is consistent and certifies (b'y = y0 + 2 y1 = y1
        // with y1 > 0 after correction). A ray whose correction cannot zero a free
        // column must be refused: a single row x - y_free = 3 with both free.
        let a = vec![1.0, -1.0];
        let c = csc(&a, 1, 2);
        let (b, l, u) = (vec![3.0], vec![-INF, -INF], vec![INF, INF]);
        assert!(!farkas_exact_open(&[1.0], &c, 2, 1, &b, &l, &u));
        let a2 = vec![1.0, 1.0];
        let c2 = csc(&a2, 2, 1);
        let (b2, l2, u2) = (vec![1.0, 2.0], vec![-INF], vec![INF]);
        assert!(farkas_exact_open(
            &[-0.5, 0.5000000000000001],
            &c2,
            1,
            2,
            &b2,
            &l2,
            &u2
        ));
    }
}
