//! Exact Farkas certificate over open columns (highs-parity P3.2,
//! `DISCOPT_FARKAS_EXACT_OPEN`, default ON; `=0` opts out).
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
use num_traits::{One, Signed, Zero};
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
/// read once). **Default ON** since its §5 panel (highs-parity P3.2, owner decision
/// 2026-11-05); `=0` opts out to the float-only verifier. Bound-changing (it decides
/// which node LPs are proven infeasible), so it was gated: 132-instance MINLP corpus ON
/// vs OFF cert-clean (0 problems, 244 incumbent checks, certified 107 = 107), L/M/INF on
/// the Rust route identical certification (130 runs, 0 incorrect), and decisive on its
/// class -- the Rust IIS engine returns P0.6's baseline IIS on 53/53 fixtures (51/53
/// without it) and the random-MILP IIS test's 131 uncertifiable node LPs certify. (A
/// first panel was retracted: it ran while a length check made this a no-op on the
/// primal simplex's call site.)
pub(crate) fn farkas_exact_open_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let raw = std::env::var("DISCOPT_FARKAS_EXACT_OPEN").unwrap_or_default();
        parse_farkas_exact_open(&raw)
    })
}

/// The parse table behind `DISCOPT_FARKAS_EXACT_OPEN`; garbage is refused.
fn parse_farkas_exact_open(raw: &str) -> bool {
    match raw.trim() {
        "0" | "false" | "False" | "off" | "OFF" => false,
        "" | "1" | "true" | "True" | "on" | "ON" => true,
        other => panic!(
            "DISCOPT_FARKAS_EXACT_OPEN={other:?} is not a recognized value; use 1/on or 0/off"
        ),
    }
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
    let _t = crate::profile::Timer::new(crate::profile::Phase::FarkasExactReduced);
    let (rows, vals) = cols.col(j);
    let mut s = BigRational::zero();
    let mut units = 0u64;
    for (k, &i) in rows.iter().enumerate() {
        if !y[i].is_zero() && vals[k] != 0.0 {
            // Limb operations of one exact multiply-add (`crate::work`).
            units += 1 + (bits(&y[i]) + bits(&s)) / 64;
            s += q(vals[k]) * &y[i];
        }
    }
    crate::work::add(crate::work::Kind::ExactLimb, units);
    s
}

/// `y` as floats, or `None` when a value does not convert to a finite, normal-range
/// float (then every sign is decided exactly).
fn to_floats(y: &[BigRational]) -> Option<Vec<f64>> {
    use num_traits::ToPrimitive;
    y.iter()
        .map(|v| {
            if v.is_zero() {
                return Some(0.0);
            }
            v.to_f64()
                .filter(|f| f.is_finite() && f.abs() >= f64::MIN_POSITIVE * 1e10)
        })
        .collect()
}

/// #2225: the sign of `(Aᵀy)_j` when floating point decides it, else `None`. `yf` is
/// `y` rounded to floats (each within a relative `2^-53`); the float sum over `k` terms
/// is then within `(k + 2) eps Σ|a_ij yf_i|` of the exact one (Higham's bound on a
/// recursive sum, plus the conversion), so a value past four times that has the exact
/// value's sign. Anything closer is left to the exact evaluation.
fn float_sign(cols: &SparseCols, j: usize, yf: &[f64]) -> Option<i8> {
    let (rows, vals) = cols.col(j);
    let (mut t, mut mag) = (0.0f64, 0.0f64);
    for (k, &i) in rows.iter().enumerate() {
        let p = vals[k] * yf[i];
        t += p;
        mag += p.abs();
    }
    if mag == 0.0 {
        // Exactly zero only when every term is (a product can underflow to 0).
        let exact_zero = rows
            .iter()
            .enumerate()
            .all(|(k, &i)| vals[k] == 0.0 || yf[i] == 0.0);
        return exact_zero.then_some(0);
    }
    let bound = 4.0 * (rows.len() as f64 + 2.0) * f64::EPSILON * mag;
    // Below this, products may have lost bits to gradual underflow: decide exactly.
    if !t.is_finite() || !mag.is_finite() || mag < 1e-280 || t.abs() <= bound {
        return None;
    }
    Some(if t > 0.0 { 1 } else { -1 })
}

/// #2225: does floating point already show that the exact Farkas inequality below
/// FAILS for `yq`? Only ever a refusal (the caller then returns `false`, as the exact
/// test would); a certificate is issued by the exact arithmetic alone.
///
/// `held` columns have an exactly zero reduced cost. Every other column must have its
/// sign decided by [`float_sign`], else nothing is concluded. Then (a) a column whose
/// sign selects an open side fails the exact test outright, and (b) the float value of
/// `g - margin` is within `TOL * S` of the exact one, `S` the sum of the magnitudes of
/// every term (each term's float error is a few `eps` of its magnitude, and the float
/// reduced cost's own error is below `1e-12` of `mag` by `float_sign`'s screen), so a
/// value below `-TOL * S` refutes.
fn float_refutes(
    yq: &[BigRational],
    held: &BTreeSet<usize>,
    cols: &SparseCols,
    n: usize,
    b: &[f64],
    l: &[f64],
    u: &[f64],
) -> bool {
    const TOL: f64 = 1e-9;
    if n + b.len() > 1_000_000 {
        return false;
    }
    let Some(yf) = to_floats(yq) else {
        return false;
    };
    let feas = super::primal::AUDIT_FEAS;
    let (mut g, mut margin, mut scale) = (0.0f64, 0.0f64, 0.0f64);
    for (bi, yi) in b.iter().zip(&yf) {
        g += bi * yi;
        margin += yi.abs() * feas * (1.0 + bi.abs());
        scale += (bi * yi).abs();
    }
    for j in 0..n {
        if held.contains(&j) {
            continue;
        }
        let (rows, vals) = cols.col(j);
        let (mut t, mut mag) = (0.0f64, 0.0f64);
        for (k, &i) in rows.iter().enumerate() {
            let p = vals[k] * yf[i];
            t += p;
            mag += p.abs();
        }
        let Some(sg) = float_sign(cols, j, &yf) else {
            // Sign unknown, `|rc| <= |t| + mag` for sure. The exact test either fails
            // on this column (an open side) or charges `-rc * side <= |rc| * |side|`
            // with `side` finite, and its margin term is `>= 0`: bound `g - margin`
            // above by that.
            let fin = [l[j], u[j]]
                .iter()
                .filter(|v| v.abs() < INF)
                .fold(0.0f64, |a, v| a.max(v.abs()));
            g += (t.abs() + mag) * fin;
            scale += mag * (1.0 + fin);
            continue;
        };
        if sg == 0 {
            continue;
        }
        let side = if sg > 0 { u[j] } else { l[j] };
        if side.abs() >= INF {
            return true; // selects an open side: the exact test refuses
        }
        g -= t * side;
        margin += t.abs() * feas * (1.0 + side.abs());
        scale += mag * (1.0 + side.abs());
    }
    let (v, s) = (g - margin, scale + margin);
    v.is_finite() && s.is_finite() && v < -TOL * s
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
    let _t = crate::profile::Timer::new(crate::profile::Phase::FarkasExactSolve);
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
            crate::work::add(crate::work::Kind::ExactLimb, 1 + step / 64);
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

/// One orientation: try to turn `y` into an exact Farkas certificate. With `seed`,
/// the open columns whose float reduced cost is at rounding-noise level are held from
/// the first round (see [`noise_open_columns`]); the rounds then only pick up what that
/// missed.
fn certify_one(
    y: &[f64],
    cols: &SparseCols,
    n: usize,
    b: &[f64],
    l: &[f64],
    u: &[f64],
    seed: &[usize],
) -> bool {
    let mut yq: Vec<BigRational> = y.iter().map(|&v| q(v)).collect();
    let open_cols: Vec<usize> = (0..n).filter(|&j| l[j] <= -INF || u[j] >= INF).collect();
    let mut held: BTreeSet<usize> = seed.iter().copied().collect();
    let mut spent = 0u64;
    if !held.is_empty() {
        if held.len() > MAX_COLUMNS {
            return false;
        }
        let set: Vec<usize> = held.iter().copied().collect();
        let rhs: Vec<BigRational> = set.iter().map(|&j| -reduced(cols, j, &yq)).collect();
        let Some(dy) = solve_rect(cols, &set, &rhs, &mut spent) else {
            return false;
        };
        for (i, v) in dy {
            yq[i] += v;
        }
    }
    for round in 0..=MAX_ROUNDS {
        // #2225: every held column's reduced cost is exactly zero here (the last
        // solve made it so, and a consistent exact solve satisfies every equation), so
        // it cannot select an open side and is not re-evaluated. The others are
        // screened in floating point; only a column the float value cannot sign goes
        // exact (`float_sign`).
        let yf = to_floats(&yq);
        let wrong: Vec<usize> = open_cols
            .iter()
            .copied()
            .filter(|j| !held.contains(j))
            .filter(
                |&j| match yf.as_deref().and_then(|yf| float_sign(cols, j, yf)) {
                    Some(sg) => (sg > 0 && u[j] >= INF) || (sg < 0 && l[j] <= -INF),
                    None => selects_open(&reduced(cols, j, &yq), l[j], u[j]),
                },
            )
            .collect();
        if wrong.is_empty() {
            break;
        }
        if round == MAX_ROUNDS {
            return false;
        }
        crate::profile::incr(crate::profile::Ctr::FarkasExactRounds);
        crate::profile::incr_by(crate::profile::Ctr::FarkasExactHeld, wrong.len() as u64);
        let fresh: BTreeSet<usize> = wrong.iter().copied().collect();
        held.extend(wrong);
        if held.len() > MAX_COLUMNS {
            return false;
        }
        let set: Vec<usize> = held.iter().copied().collect();
        // Drive every held column's reduced cost to exactly zero: (A_S)ᵀ dy = −(Aᵀy)_S,
        // whose right-hand side is zero on the columns held before this round.
        let rhs: Vec<BigRational> = set
            .iter()
            .map(|&j| {
                if fresh.contains(&j) {
                    -reduced(cols, j, &yq)
                } else {
                    BigRational::zero()
                }
            })
            .collect();
        let Some(dy) = solve_rect(cols, &set, &rhs, &mut spent) else {
            return false;
        };
        for (i, v) in dy {
            yq[i] += v;
        }
    }
    // Exact Farkas inequality, with the solver's own feasibility tolerance as the
    // margin: certify only if NO point the solver would accept as feasible exists.
    //
    // For `x` with `A x = b + r`, `|r_i| <= tol_i = AUDIT_FEAS (1 + |b_i|)` and
    // `l_j - tol_j <= x_j <= u_j + tol_j`, `tol_j = AUDIT_FEAS (1 + |bound_j|)` (the
    // row and bound tests of `primal::audit_feasibility`):
    //   bᵀy = rcᵀx − rᵀy <= Σ_j sup_box rc_j x_j + Σ_j |rc_j| tol_j + Σ_i |y_i| tol_i.
    // So `g = bᵀy − Σ_j sup_box rc_j x_j > Σ_i |y_i| tol_i + Σ_j |rc_j| tol_j` proves
    // the tolerance-widened system empty. A bare `g > 0` (P3.2 as first graduated)
    // proves only the EXACT system empty, and certified a #2142 cstr node LP that is
    // infeasible by ~1e-8 but feasible within tolerance (HiGHS solves it): a false
    // `Infeasible` that would fathom a node holding accepted points.
    let _tf = crate::profile::Timer::new(crate::profile::Phase::FarkasExactFinal);
    if float_refutes(&yq, &held, cols, n, b, l, u) {
        crate::profile::incr(crate::profile::Ctr::FarkasExactFloatRefuted);
        return false;
    }
    let feas = q(super::primal::AUDIT_FEAS);
    let one = BigRational::one();
    let mut g = BigRational::zero();
    let mut margin = BigRational::zero();
    for (bi, yi) in b.iter().zip(&yq) {
        if yi.is_zero() {
            continue;
        }
        if *bi != 0.0 {
            g += q(*bi) * yi;
        }
        margin += yi.abs() * &feas * (&one + q(bi.abs()));
    }
    for j in 0..n {
        let rc = reduced(cols, j, &yq);
        if rc.is_zero() {
            continue;
        }
        let side = if rc.is_positive() {
            if u[j] >= INF {
                return false;
            }
            u[j]
        } else {
            if l[j] <= -INF {
                return false;
            }
            l[j]
        };
        g -= &rc * q(side);
        margin += rc.abs() * &feas * (&one + q(side.abs()));
    }
    g > margin
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
    // The simplex passes its working box, which also carries its `m` artificial
    // columns after the `n` real ones (`lb.len() == n + m`); like the float verifier,
    // only the real columns' sides enter the certificate.
    if y.len() != m || b.len() != m || l.len() < n || u.len() < n {
        return false;
    }
    let (l, u) = (&l[..n], &u[..n]);
    if !y.iter().chain(b).chain(l).chain(u).all(|v| v.is_finite()) {
        return false;
    }
    crate::profile::incr(crate::profile::Ctr::FarkasExactAttempt);
    let _t = crate::profile::Timer::new(crate::profile::Phase::FarkasExactOpen);
    let seed = noise_open_columns(y, cols, n, l, u);
    for sgn in [1.0f64, -1.0] {
        let ys: Vec<f64> = y.iter().map(|v| sgn * v).collect();
        // #2225: hold the noise-level open columns from the start (one elimination
        // instead of a dozen growing ones); if that system is inconsistent, the
        // round-by-round correction decides, as before. Either way only the exact
        // inequality at the end certifies.
        if !seed.is_empty() && certify_one(&ys, cols, n, b, l, u, &seed) {
            crate::profile::incr(crate::profile::Ctr::FarkasExactRescue);
            crate::profile::incr(crate::profile::Ctr::FarkasExactSeeded);
            return true;
        }
        if certify_one(&ys, cols, n, b, l, u, &[]) {
            crate::profile::incr(crate::profile::Ctr::FarkasExactRescue);
            return true;
        }
    }
    false
}

/// #2225: the open columns of `y`'s support whose float reduced cost `(Aᵀy)_j` is
/// rounding noise, `|(Aᵀy)_j| <= 1e-9 Σ_i |a_ij y_i|`. A float ray of an infeasible
/// system with open columns is a rounded certificate whose reduced cost on those
/// columns is exactly zero, so the correction holds them at zero anyway; the round-by-
/// round loop found them a few at a time (measured on MIPcc27 `bell3ainf` IIS probes:
/// ~12 rounds and ~83 held columns per call, each round re-eliminating the whole held
/// set over the rationals). Only a choice of which columns to correct: the result is
/// still checked exactly.
fn noise_open_columns(y: &[f64], cols: &SparseCols, n: usize, l: &[f64], u: &[f64]) -> Vec<usize> {
    let open: Vec<usize> = (0..n).filter(|&j| l[j] <= -INF || u[j] >= INF).collect();
    let mut zero = Vec::new();
    let mut seed = Vec::new();
    for &j in &open {
        let (rows, vals) = cols.col(j);
        let (mut t, mut mag) = (0.0f64, 0.0f64);
        for (k, &i) in rows.iter().enumerate() {
            let p = vals[k] * y[i];
            t += p;
            mag += p.abs();
        }
        if mag > 0.0 && t.abs() <= 1e-9 * mag {
            seed.push(j);
        } else if mag == 0.0 {
            zero.push(j);
        }
    }
    {
        // Open columns off `y`'s support that share a row with a seed column: the
        // correction moves `y` on those rows, and would hold them a round later
        // (measured on the same probes: 1100 rounds -> 190, 1318 eliminations -> 408,
        // every certificate still found by the seeded attempt).
        let m = (0..n)
            .flat_map(|j| cols.col(j).0.iter().copied())
            .max()
            .map_or(0, |v| v + 1);
        let mut touched = vec![false; m];
        for &j in &seed {
            for &i in cols.col(j).0 {
                touched[i] = true;
            }
        }
        for j in zero {
            if cols.col(j).0.iter().any(|&i| touched[i]) {
                seed.push(j);
            }
        }
        seed.sort_unstable();
    }
    seed
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pre-#2225 `certify_one`, verbatim: every round re-evaluates every open
    /// column exactly and re-solves the whole held set. The reference the faster
    /// routine must agree with.
    fn certify_one_reference(
        y: &[f64],
        cols: &SparseCols,
        n: usize,
        b: &[f64],
        l: &[f64],
        u: &[f64],
    ) -> bool {
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
            let rhs: Vec<BigRational> = set.iter().map(|&j| -reduced(cols, j, &yq)).collect();
            let Some(dy) = solve_rect(cols, &set, &rhs, &mut spent) else {
                return false;
            };
            for (i, v) in dy {
                yq[i] += v;
            }
        }
        let feas = q(super::super::primal::AUDIT_FEAS);
        let one = BigRational::one();
        let mut g = BigRational::zero();
        let mut margin = BigRational::zero();
        for (bi, yi) in b.iter().zip(&yq) {
            if yi.is_zero() {
                continue;
            }
            if *bi != 0.0 {
                g += q(*bi) * yi;
            }
            margin += yi.abs() * &feas * (&one + q(bi.abs()));
        }
        for j in 0..n {
            let rc = reduced(cols, j, &yq);
            if rc.is_zero() {
                continue;
            }
            let side = if rc.is_positive() {
                if u[j] >= INF {
                    return false;
                }
                u[j]
            } else {
                if l[j] <= -INF {
                    return false;
                }
                l[j]
            };
            g -= &rc * q(side);
            margin += rc.abs() * &feas * (&one + q(side.abs()));
        }
        g > margin
    }

    fn reference(y: &[f64], c: &SparseCols, n: usize, b: &[f64], l: &[f64], u: &[f64]) -> bool {
        [1.0f64, -1.0].iter().any(|s| {
            let ys: Vec<f64> = y.iter().map(|v| s * v).collect();
            certify_one_reference(&ys, c, n, b, l, u)
        })
    }

    /// Deterministic LCG (no `rand` dependency in this crate's tests).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn int(&mut self, lo: i64, hi: i64) -> i64 {
            lo + (self.next() % ((hi - lo + 1) as u64)) as i64
        }
    }

    /// One random system `A x = b`, `l <= x <= u` with open columns, and a float ray.
    /// `planted`: an exact certificate `y*` (small integers) exists -- every open
    /// column's reduced cost is exactly zero under it and `b` beats the box supremum by
    /// a margin -- and the ray handed over is `y*/3` in floating point, so its reduced
    /// costs on the open columns are rounding noise, not zero (what the simplex hands
    /// over). Else `b = A x0` for a point `x0` of the box: feasible.
    #[allow(clippy::type_complexity)]
    fn random_case(
        r: &mut Lcg,
        planted: bool,
    ) -> (
        SparseCols,
        usize,
        usize,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
    ) {
        let m = r.int(2, 6) as usize;
        let n = r.int(3, 10) as usize;
        let ystar: Vec<f64> = (0..m).map(|_| r.int(-3, 3) as f64).collect();
        let mut a = vec![0.0; m * n];
        let (mut l, mut u) = (vec![0.0; n], vec![0.0; n]);
        for j in 0..n {
            for i in 0..m {
                if r.int(0, 2) > 0 {
                    // Decimal entries make the float ray's reduced costs inexact.
                    a[i * n + j] = r.int(-9, 9) as f64 * if r.int(0, 1) == 0 { 1.0 } else { 0.1 };
                }
            }
            match r.int(0, 3) {
                0 => (l[j], u[j]) = (-INF, INF),
                1 => (l[j], u[j]) = (r.int(-3, 0) as f64, INF),
                2 => (l[j], u[j]) = (-INF, r.int(0, 3) as f64),
                _ => (l[j], u[j]) = (r.int(-3, 0) as f64, r.int(0, 3) as f64),
            }
            let open = l[j] <= -INF || u[j] >= INF;
            if planted && open {
                // Make (A^T y*)_j exactly zero: put an integer entry on a row where
                // y* is nonzero, chosen to cancel the rest exactly (y* integer).
                if let Some(k) = (0..m).find(|&i| ystar[i] != 0.0) {
                    for i in 0..m {
                        a[i * n + j] = a[i * n + j].round();
                    }
                    let rest: f64 = (0..m)
                        .filter(|&i| i != k)
                        .map(|i| a[i * n + j] * ystar[i])
                        .sum();
                    a[k * n + j] = -rest / ystar[k];
                    if a[k * n + j] != a[k * n + j].round() {
                        // Not an exact integer: scale the column so it is.
                        for i in 0..m {
                            a[i * n + j] *= ystar[k];
                        }
                        a[k * n + j] = -rest;
                    }
                } else {
                    for i in 0..m {
                        a[i * n + j] = 0.0;
                    }
                }
            }
        }
        let c = csc(&a, m, n);
        let b: Vec<f64> = if planted {
            // b^T y* = sup over the box of (A^T y*)^T x, plus a clear margin: put it all
            // on one row where y* is nonzero.
            let mut sup = 0.0;
            for j in 0..n {
                let rc: f64 = (0..m).map(|i| a[i * n + j] * ystar[i]).sum();
                if rc > 0.0 {
                    sup += rc * if u[j] >= INF { 0.0 } else { u[j] };
                } else if rc < 0.0 {
                    sup += rc * if l[j] <= -INF { 0.0 } else { l[j] };
                }
            }
            let mut b = vec![0.0; m];
            if let Some(k) = (0..m).find(|&i| ystar[i] != 0.0) {
                b[k] = (sup + 1.0) / ystar[k];
            }
            b
        } else {
            let x0: Vec<f64> = (0..n)
                .map(|j| match (l[j] > -INF, u[j] < INF) {
                    (true, true) => 0.5 * (l[j] + u[j]),
                    (true, false) => l[j] + 1.0,
                    (false, true) => u[j] - 1.0,
                    (false, false) => r.int(-2, 2) as f64,
                })
                .collect();
            (0..m)
                .map(|i| (0..n).map(|j| a[i * n + j] * x0[j]).sum())
                .collect()
        };
        let y: Vec<f64> = if planted {
            ystar.iter().map(|v| v / 3.0).collect()
        } else {
            (0..m).map(|_| r.int(-5, 5) as f64 / 7.0).collect()
        };
        (c, m, n, b, l, u, y)
    }

    /// #2225: the faster routine (held columns not re-evaluated, float sign screens, a
    /// seeded first elimination, a float refutation of the final inequality) certifies
    /// every ray the reference certifies, and never certifies a feasible system.
    #[test]
    fn faster_routine_certifies_whatever_the_reference_does_2225() {
        let mut r = Lcg(0x2225);
        let (mut both, mut only_new, mut feasible_checked) = (0usize, 0usize, 0usize);
        for k in 0..3000 {
            let planted = k % 2 == 0;
            let (c, m, n, b, l, u, y) = random_case(&mut r, planted);
            let new = farkas_exact_open(&y, &c, n, m, &b, &l, &u);
            if planted {
                let old = reference(&y, &c, n, &b, &l, &u);
                assert!(
                    !old || new,
                    "case {k}: the reference certifies, the new routine refuses"
                );
                both += (old && new) as usize;
                only_new += (!old && new) as usize;
            } else {
                assert!(!new, "case {k}: a feasible system was certified empty");
                feasible_checked += 1;
            }
        }
        // The probe fired: planted certificates were found, feasible systems checked.
        assert!(both > 300, "only {both} planted certificates found by both");
        assert_eq!(feasible_checked, 1500);
        eprintln!("certified by both {both}, by the new routine only {only_new}");
    }

    /// #2225: `float_sign` never disagrees with the exact sign.
    #[test]
    fn float_sign_agrees_with_the_exact_sign() {
        let mut r = Lcg(7);
        let mut decided = 0;
        for _ in 0..3000 {
            let m = r.int(1, 8) as usize;
            let a: Vec<f64> = (0..m)
                .map(|_| r.int(-9, 9) as f64 * 0.1f64.powi(r.int(0, 3) as i32))
                .collect();
            let c = csc(&a, m, 1);
            // Rays with exactly cancelling and near-cancelling entries.
            let y: Vec<f64> = (0..m)
                .map(|_| r.int(-9, 9) as f64 / (r.int(1, 7) as f64))
                .collect();
            let yq: Vec<BigRational> = y.iter().map(|&v| q(v)).collect();
            let exact = reduced(&c, 0, &yq);
            if let Some(sg) = float_sign(&c, 0, &y) {
                decided += 1;
                let want = if exact.is_positive() {
                    1
                } else if exact.is_negative() {
                    -1
                } else {
                    0
                };
                assert_eq!(sg, want, "a={a:?} y={y:?}");
            }
        }
        assert!(
            decided > 2000,
            "only {decided} signs decided in floating point"
        );
    }

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
        // The float verifier alone rejects it (the exact fallback is what certifies).
        assert!(!super::super::primal::farkas_ray_certifies_raw(
            &y, &c, n, m, &b, &l, &u
        ));
        assert!(farkas_exact_open(&y, &c, n, m, &b, &l, &u));
    }

    #[test]
    fn accepts_a_working_box_that_carries_artificials() {
        // The simplex hands its `n + m`-long box (artificials appended); the extra
        // entries must be ignored, not make the check a silent no-op.
        let (c, m, n, b, mut l, mut u) = probe();
        l.extend([0.0, 0.0]);
        u.extend([0.0, 0.0]);
        let y = vec![0.333_333_333_333_333_3, 0.666_666_666_666_666_7];
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

    #[test]
    fn tolerance_feasible_system_is_not_certified_2142() {
        // x0 in [0, 1] and x0 - x1 = 1 + 1e-8 with x1 free in the row but fixed in
        // [0, 0]: exactly infeasible by 1e-8, yet x0 = 1 satisfies the row within the
        // solver's feasibility tolerance (1e-6 relative), so no certificate may be
        // issued. A bare exact `g > 0` (P3.2 as first graduated) certified it: the
        // #2142 cstr node LP failure, which HiGHS solves.
        let a = vec![1.0, -1.0];
        let c = csc(&a, 1, 2);
        let (b, l, u) = (vec![1.0 + 1e-8], vec![0.0, 0.0], vec![1.0, 0.0]);
        assert!(!farkas_exact_open(&[1.0], &c, 2, 1, &b, &l, &u));
        // Infeasible by a clear margin (0.5): certified.
        let b2 = vec![1.5];
        assert!(farkas_exact_open(&[1.0], &c, 2, 1, &b2, &l, &u));
    }
}
