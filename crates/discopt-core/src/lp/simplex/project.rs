//! Exact projection of a dual vector onto `{y : a_jᵀy = c_j, j ∈ C}` (#2115).
//!
//! The Neumaier–Shcherbina safe dual bound `g(y)` is `−∞` as soon as a column with
//! an **open** side has a
//! reduced cost `c_j − a_jᵀy` that can take that side's sign. In floating point a
//! reduced cost that is zero in exact arithmetic is only known up to its rounding,
//! so such a column defeats the bound even when the dual is, mathematically,
//! perfect. Measured on the MINLPLib `m6` OA master: 42 of its 86 columns are
//! translation-invariant layout positions (free variables, which reach the LP split
//! as `x⁺ − x⁻`), and the in-house engine's node bound abstained on 3194 of ~29k
//! nodes (`NodeNsAbstain`) for exactly this reason; those nodes, removed unbounded,
//! set the unresolved floor that held the tree at 62.83 against an optimum of 82.26.
//!
//! The remedy keeps the bound rigorous rather than loosening it. For a set `C` of
//! such columns, [`project_dual`] constructs a nearby `y*` with `a_jᵀy* = c_j`
//! **exactly** for every `j ∈ C`, as an *enclosure* `y* ∈ ŷ ± rad` (it never knows
//! `y*` itself), and [`ns_bound_enclosed`] evaluates the bound over the whole
//! enclosure, where every column of `C` contributes exactly `0`. That can only
//! *add* a bound the unprojected evaluation could not check: every quantity is
//! bounded over all `y'` in the box, so whichever `y'` is `y*` is covered, and a
//! projection that fails to verify reports nothing. The consumer is the MILP
//! driver's node bound (`node_ns_bound_projected`, `DISCOPT_NS_PROJECT`).
//!
//! The same construction applied to a Farkas ray (`c = 0`) was measured and not
//! kept: on the `m6` master it certified all 9 rays the raw and #1355-cleaned
//! verifiers refused, and changed neither the node count nor the bound (#2115).
//!
//! The construction (steps refer to [`project_dual`]):
//!
//! * **A — exact zeroing.** A single-entry column `a_j = a·e_i` with `c_j = 0` is
//!   annihilated by `y_i = 0`: no arithmetic at all. On `m6` these are the slacks
//!   of inactive rows whose `y_i` is 1e-11..1e-17·‖y‖∞ of rounding noise.
//! * **Frozen columns.** A `c_j = 0` column whose every `y_i` on its support is now
//!   exactly `0` has `a_jᵀy = 0` exactly. It needs no correction, only for the
//!   correction to stay off its rows.
//! * **B — verified block solve.** The rest need `A_Bᵀ(y0 + Δ) = c_B`. `Δ` is put
//!   on `k = |B|` rows `S` (row-pivoted elimination, outside the frozen rows) where
//!   `K = A_B[S,:]ᵀ` is nonsingular, and `K z = r := c_B − A_Bᵀy0` is solved with an
//!   approximate inverse `R` and then **enclosed**: `‖I − RK‖∞ ≤ α < 1` proves `K`
//!   nonsingular, and `‖z* − z̃‖∞ ≤ ‖|R|ρ‖∞ / (1 − α)` with `ρ ≥ |r − K z̃|` over
//!   the rounding of the residual *and* of `r` itself (Rump's verification of a
//!   linear system). Every sum is bounded by `γ_k·Σ|terms|` (Higham §3.1), the
//!   yardstick the raw verifiers already use ([`crate::numeric::gamma`]).
//!
//! Exact duplicates in `C` (a column bitwise equal to, or the exact negation of, an
//! earlier one, with `c` matching) need no row of their own: `x⁺/x⁻` would
//! otherwise make `K` singular.

#![allow(clippy::needless_range_loop)]

use super::sparse::SparseCols;
use crate::numeric::gamma;

const INF: f64 = 1e20;

/// Largest `|B|` [`project_dual`] solves (dense `O(k³)`; `m6` needs ≤ 28 after
/// step A). Above it the projection is declined.
pub const PROJECT_MAX_BLOCK: usize = 128;

/// Candidate selection for the projection: an open column whose reduced cost is on
/// (or straddles) its open side is projected when
/// `|c_j − a_jᵀy| ≤ τ·(|c_j| + ‖a_j‖∞·‖y‖∞)`. Selection only — what is selected is
/// verified rigorously — so it cannot affect soundness. On `m6` the noise reaches
/// 2.3e-11·‖y‖∞.
pub const PROJECT_NOISE_REL: f64 = 1e-9;

/// A dual known as an enclosure: the exact `y*` satisfies `|y*_i − y_i| ≤ rad[i]`
/// and `c_j − a_jᵀy* = 0` exactly for every `j` with `zero[j]`.
#[derive(Clone, Debug)]
pub struct ProjectedDual {
    /// Centre `ŷ`.
    pub y: Vec<f64>,
    /// Componentwise radius.
    pub rad: Vec<f64>,
    /// Columns whose reduced cost is exactly zero at `y*`.
    pub zero: Vec<bool>,
}

/// Why [`project_dual`] produced no enclosure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectFail {
    /// `C` was empty, or `y`/`c` malformed: nothing to do.
    Nothing,
    /// The block exceeded [`PROJECT_MAX_BLOCK`].
    TooLarge,
    /// The block is (numerically) rank-deficient, or its enclosure did not verify.
    Singular,
}

/// The columns [`project_dual`] should annihilate for the bound with costs `c`:
/// open columns whose reduced cost rides, or straddles, the open side within
/// [`PROJECT_NOISE_REL`], plus open columns whose reduced cost is already exactly
/// zero (so their rows are frozen). `hi_open(j)` decides whether column `j`'s
/// upper side is open. Empty when no column is noise-selected.
pub fn open_noise_candidates(
    y: &[f64],
    c: &[f64],
    cols: &SparseCols,
    n: usize,
    l: &[f64],
    hi_open: &mut impl FnMut(usize) -> bool,
) -> Vec<usize> {
    let ymax = y.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    let mut cand = Vec::new();
    let mut any_noise = false;
    for j in 0..n {
        let cj = c[j];
        let (aty, aty_abs, nnz) = cols.dot_with_magnitude(j, y);
        let lo_open = l[j] <= -INF;
        if aty_abs == 0.0 && cj == 0.0 {
            // Reduced cost exactly 0 already. An open such column is still a
            // candidate — [`project_dual`] freezes its rows, so the correction for
            // the others cannot move it off zero onto its open side (measured on
            // `m6`: without this every projected node bound failed on the slack
            // of a row the correction had been placed on).
            if lo_open || hi_open(j) {
                cand.push(j);
            }
            continue;
        }
        let rc = cj - aty;
        let e = gamma(nnz + 2) * (aty_abs + cj.abs());
        // Can the reduced cost take an open side's sign? (`rc > 0` rides an open
        // lower side to −∞; `rc < 0` an open upper side.)
        let may_pos = rc + e > 0.0;
        let may_neg = rc - e < 0.0;
        let bad = (lo_open && may_pos) || (may_neg && hi_open(j));
        if !bad {
            continue;
        }
        let amax = cols.col(j).1.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        if rc.abs() <= PROJECT_NOISE_REL * (cj.abs() + amax * ymax) {
            cand.push(j);
            any_noise = true;
        }
    }
    // Only exact zeros: nothing for the projection to do.
    if !any_noise {
        cand.clear();
    }
    cand
}

/// Gauss–Jordan inverse of a dense row-major `k×k` matrix, `None` on a zero or
/// non-finite pivot. Plain floating point: an *approximate* inverse the caller
/// verifies.
fn dense_approx_inverse(a: &[f64], k: usize) -> Option<Vec<f64>> {
    let w = 2 * k;
    let mut t = vec![0.0f64; k * w];
    for i in 0..k {
        t[i * w..i * w + k].copy_from_slice(&a[i * k..i * k + k]);
        t[i * w + k + i] = 1.0;
    }
    for c in 0..k {
        let mut p = c;
        let mut best = t[c * w + c].abs();
        for r in c + 1..k {
            let v = t[r * w + c].abs();
            if v > best {
                best = v;
                p = r;
            }
        }
        if !(best > 0.0 && best.is_finite()) {
            return None;
        }
        if p != c {
            for q in 0..w {
                t.swap(c * w + q, p * w + q);
            }
        }
        let piv = t[c * w + c];
        for q in 0..w {
            t[c * w + q] /= piv;
        }
        for r in 0..k {
            if r == c {
                continue;
            }
            let f = t[r * w + c];
            if f == 0.0 {
                continue;
            }
            for q in 0..w {
                t[r * w + q] -= f * t[c * w + q];
            }
        }
    }
    let mut inv = vec![0.0f64; k * k];
    for i in 0..k {
        inv[i * k..i * k + k].copy_from_slice(&t[i * w + k..i * w + w]);
    }
    inv.iter().all(|v| v.is_finite()).then_some(inv)
}

/// Project `y` so that `c_j − a_jᵀy* = 0` exactly for every `j ∈ cand` (see the
/// module docs).
pub fn project_dual(
    y: &[f64],
    c: &[f64],
    cols: &SparseCols,
    m: usize,
    n: usize,
    cand: &[usize],
) -> Result<ProjectedDual, ProjectFail> {
    if y.len() != m || !y.iter().all(|v| v.is_finite()) || cand.is_empty() {
        return Err(ProjectFail::Nothing);
    }
    if c.len() < n || !cand.iter().all(|&j| j < n && c[j].is_finite()) {
        return Err(ProjectFail::Nothing);
    }
    let cj = |j: usize| c[j];
    // Step A: zero the row of every single-entry `c_j = 0` candidate.
    let mut zrow = vec![false; m];
    for &j in cand {
        if cj(j) != 0.0 {
            continue;
        }
        let (rows, vals) = cols.col(j);
        let mut nz = rows.iter().zip(vals).filter(|(_, &a)| a != 0.0);
        if let (Some((&i, _)), None) = (nz.next(), nz.next()) {
            zrow[i] = true;
        }
    }
    let mut y0 = y.to_vec();
    for i in 0..m {
        if zrow[i] {
            y0[i] = 0.0;
        }
    }
    // Frozen: `c_j = 0` candidates with `y0` exactly zero on their support.
    let mut frozen = zrow;
    let exact_zero =
        |j: usize, y0: &[f64]| cj(j) == 0.0 && cols.col(j).0.iter().all(|&i| y0[i] == 0.0);
    for &j in cand {
        if exact_zero(j, &y0) {
            for &i in cols.col(j).0 {
                frozen[i] = true;
            }
        }
    }
    // Step B: the remaining candidates, exact `±` duplicates grouped.
    let same_up_to_sign = |p: usize, q: usize| -> bool {
        let (rp, vp) = cols.col(p);
        let (rq, vq) = cols.col(q);
        if rp != rq {
            return false;
        }
        (cj(p) == cj(q) && vp.iter().zip(vq).all(|(a, b)| a == b))
            || (cj(p) == -cj(q) && vp.iter().zip(vq).all(|(a, b)| *a == -b))
    };
    let mut block: Vec<usize> = Vec::new();
    for &j in cand {
        if exact_zero(j, &y0) || block.iter().any(|&p| same_up_to_sign(p, j)) {
            continue;
        }
        block.push(j);
        if block.len() > PROJECT_MAX_BLOCK {
            return Err(ProjectFail::TooLarge);
        }
    }
    let mut zero = vec![false; n];
    for &j in cand {
        zero[j] = true;
    }
    let k = block.len();
    if k == 0 {
        return Ok(ProjectedDual {
            y: y0,
            rad: vec![0.0; m],
            zero,
        });
    }
    // r = c_B − A_Bᵀy0, with its rounding radius.
    let mut r = vec![0.0f64; k];
    let mut er = vec![0.0f64; k];
    for (q, &j) in block.iter().enumerate() {
        let (aty, aty_abs, nnz) = cols.dot_with_magnitude(j, &y0);
        r[q] = cj(j) - aty;
        er[q] = gamma(nnz + 2) * (aty_abs + cj(j).abs());
    }
    // Rows outside `frozen` touched by the block; the dense block (rows × k).
    let mut row_pos = vec![usize::MAX; m];
    let mut rows_t: Vec<usize> = Vec::new();
    for &j in &block {
        for &i in cols.col(j).0 {
            if !frozen[i] && row_pos[i] == usize::MAX {
                row_pos[i] = rows_t.len();
                rows_t.push(i);
            }
        }
    }
    let mr = rows_t.len();
    if mr < k {
        return Err(ProjectFail::Singular);
    }
    let mut af = vec![0.0f64; mr * k];
    for (q, &j) in block.iter().enumerate() {
        let (rows, vals) = cols.col(j);
        for (&i, &a) in rows.iter().zip(vals) {
            if !frozen[i] {
                af[row_pos[i] * k + q] += a;
            }
        }
    }
    // Row selection: elimination on a scratch copy with row pivoting. A numerically
    // rank-deficient block has no well-defined square subsystem; the verification
    // below would refuse it anyway, so refuse it here and say so.
    let amax = af.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    if !(amax > 0.0 && amax.is_finite()) {
        return Err(ProjectFail::Singular);
    }
    let mut work = af.clone();
    let mut perm: Vec<usize> = (0..mr).collect();
    for col in 0..k {
        let mut p = col;
        let mut best = 0.0f64;
        for (rr, &pr) in perm.iter().enumerate().skip(col) {
            let v = work[pr * k + col].abs();
            if v > best {
                best = v;
                p = rr;
            }
        }
        if best.is_nan() || best <= 1e-12 * amax {
            return Err(ProjectFail::Singular);
        }
        perm.swap(col, p);
        let pr = perm[col];
        let piv = work[pr * k + col];
        for &rr in perm.iter().skip(col + 1) {
            let f = work[rr * k + col] / piv;
            if f == 0.0 {
                continue;
            }
            for q in col..k {
                work[rr * k + q] -= f * work[pr * k + q];
            }
        }
    }
    let sel = &perm[..k]; // indices into rows_t
                          // K[p][s] = a_{S_s, B_p}.
    let mut kmat = vec![0.0f64; k * k];
    for p in 0..k {
        for (s, &rs) in sel.iter().enumerate() {
            kmat[p * k + s] = af[rs * k + p];
        }
    }
    let rinv = dense_approx_inverse(&kmat, k).ok_or(ProjectFail::Singular)?;
    let mut z = vec![0.0f64; k];
    for p in 0..k {
        let mut acc = 0.0f64;
        for q in 0..k {
            acc += rinv[p * k + q] * r[q];
        }
        z[p] = acc;
    }
    if !z.iter().all(|v| v.is_finite()) {
        return Err(ProjectFail::Singular);
    }
    // α ≥ ‖I − RK‖∞, rounded up.
    let mut alpha = 0.0f64;
    for p in 0..k {
        let mut row = 0.0f64;
        for q in 0..k {
            let mut cpq = if p == q { 1.0 } else { 0.0 };
            let mut mag = cpq;
            for s in 0..k {
                let t = rinv[p * k + s] * kmat[s * k + q];
                cpq -= t;
                mag += t.abs();
            }
            row += cpq.abs() + gamma(k + 2) * mag;
        }
        row *= 1.0 + gamma(k + 1);
        alpha = alpha.max(row);
    }
    if alpha.is_nan() || alpha >= 0.5 {
        return Err(ProjectFail::Singular);
    }
    // ρ ≥ |r_true − K z̃| componentwise, rounded up.
    let mut rho = vec![0.0f64; k];
    for p in 0..k {
        let mut res = r[p];
        let mut mag = r[p].abs();
        for s in 0..k {
            let t = kmat[p * k + s] * z[s];
            res -= t;
            mag += t.abs();
        }
        rho[p] = (res.abs() + gamma(k + 2) * mag + er[p]) * (1.0 + 4.0 * f64::EPSILON);
    }
    // ‖z* − z̃‖∞ ≤ ‖|R|ρ‖∞ / (1 − α), rounded up.
    let mut rr_max = 0.0f64;
    for p in 0..k {
        let mut acc = 0.0f64;
        for q in 0..k {
            acc += rinv[p * k + q].abs() * rho[q];
        }
        rr_max = rr_max.max(acc * (1.0 + gamma(k + 1)));
    }
    let denom = (1.0 - alpha) * (1.0 - 4.0 * f64::EPSILON);
    let delta = rr_max / denom * (1.0 + 4.0 * f64::EPSILON);
    if !delta.is_finite() {
        return Err(ProjectFail::Singular);
    }
    // ŷ = y0 + z̃ on S; `2ε·|ŷ_i|` covers the rounding of that sum, MIN_POSITIVE a
    // subnormal.
    let mut yhat = y0;
    let mut rad = vec![0.0f64; m];
    for (s, &rs) in sel.iter().enumerate() {
        let i = rows_t[rs];
        yhat[i] += z[s];
        rad[i] = delta + 2.0 * f64::EPSILON * yhat[i].abs() + f64::MIN_POSITIVE;
    }
    if !yhat.iter().all(|v| v.is_finite()) {
        return Err(ProjectFail::Singular);
    }
    Ok(ProjectedDual { y: yhat, rad, zero })
}

/// Rounds of candidate closure in [`project_dual_closed`]. Each round adds the open
/// columns the previous projection pushed onto (or across) an open side; on `m6`
/// one extra round suffices.
pub const PROJECT_CLOSURE_ROUNDS: usize = 4;

/// The enclosed reduced cost `[rl, ru]` of column `j` over every `y'` in `pd`.
fn enclosed_rc(pd: &ProjectedDual, cj: f64, cols: &SparseCols, j: usize) -> (f64, f64) {
    let (aty, aty_abs, nnz) = cols.dot_with_magnitude(j, &pd.y);
    let (rows, vals) = cols.col(j);
    let mut shift = 0.0f64;
    for (&i, &a) in rows.iter().zip(vals) {
        shift += a.abs() * pd.rad[i];
    }
    let rc = cj - aty;
    let e = (gamma(nnz + 2) * (aty_abs + cj.abs()) + shift * (1.0 + gamma(nnz + 2)))
        * (1.0 + 4.0 * f64::EPSILON)
        + 2.0 * f64::EPSILON * rc.abs();
    (rc - e, rc + e)
}

/// [`open_noise_candidates`] then [`project_dual`], **closed** under the
/// projection: a correction placed on a row can move an open column that was safe
/// before (or exactly zero) onto its open side. After each projection such columns
/// are added to the candidates and the projection is redone, for up to
/// [`PROJECT_CLOSURE_ROUNDS`] rounds. Selection only — whatever is returned is an
/// enclosure [`project_dual`] verified, so this cannot affect soundness.
#[allow(clippy::too_many_arguments)]
pub fn project_dual_closed(
    y: &[f64],
    c: &[f64],
    cols: &SparseCols,
    m: usize,
    n: usize,
    l: &[f64],
    mut hi_open: impl FnMut(usize) -> bool,
) -> Result<ProjectedDual, ProjectFail> {
    let mut cand = open_noise_candidates(y, c, cols, n, l, &mut hi_open);
    if cand.is_empty() {
        return Err(ProjectFail::Nothing);
    }
    let mut pd = project_dual(y, c, cols, m, n, &cand)?;
    for _ in 0..PROJECT_CLOSURE_ROUNDS {
        let mut added = false;
        for j in 0..n {
            if pd.zero[j] {
                continue;
            }
            let lo_open = l[j] <= -INF;
            if !lo_open && !hi_open(j) {
                continue;
            }
            let (rl, ru) = enclosed_rc(&pd, c[j], cols, j);
            let hi = hi_open(j);
            if (lo_open && ru > 0.0) || (hi && rl < 0.0) {
                cand.push(j);
                added = true;
            }
        }
        if !added {
            break;
        }
        pd = project_dual(y, c, cols, m, n, &cand)?;
    }
    Ok(pd)
}

/// A rigorous lower bound on the Neumaier–Shcherbina bound
/// `g(y*) = bᵀy* + Σ_j min_{x_j ∈ [l_j, u_j]} (c_j − a_jᵀy*)·x_j` of
/// `min cᵀx s.t. Ax = b, l ≤ x ≤ u`, valid for **every** `y*` in the enclosure
/// `pd` — hence for the exact one, whose `zero` columns contribute exactly `0`.
/// `None` when an open side can be selected (no finite bound).
///
/// Every rounding is bounded by `γ_k·Σ|terms|` and taken in the bound's
/// disfavour; the result is then lowered by the crate's
/// [`super::refine::NS_MARGIN_REL`]` · S`, as [`super::refine::ns_safe_bound_csc`]
/// does, so it is never above what that evaluation would accept.
#[allow(clippy::too_many_arguments)]
pub fn ns_bound_enclosed(
    pd: &ProjectedDual,
    c: &[f64],
    cols: &SparseCols,
    m: usize,
    n: usize,
    b: &[f64],
    l: &[f64],
    u: &[f64],
) -> Option<f64> {
    let (y, rad) = (&pd.y, &pd.rad);
    if y.len() != m || rad.len() != m || pd.zero.len() < n || b.len() < m || c.len() < n {
        return None;
    }
    let (mut by, mut by_abs, mut by_rad) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..m {
        let t = b[i] * y[i];
        by += t;
        by_abs += t.abs();
        by_rad += b[i].abs() * rad[i];
    }
    // bᵀy* ≥ bᵀŷ − (rounding of bᵀŷ) − Σ|b_i|·rad_i (itself rounded up). The two
    // subtractions round too, and relative to `|by|`, not to the (possibly
    // cancelled) result -- the #1017 lesson -- so their bound is scaled by `|by|`.
    let by_err =
        (gamma(m + 1) * by_abs + by_rad * (1.0 + gamma(m + 1))) * (1.0 + 4.0 * f64::EPSILON);
    let by_lo = by - by_err - 4.0 * f64::EPSILON * (by.abs() + by_err);
    let (mut sum, mut sum_abs) = (0.0f64, 0.0f64);
    for j in 0..n {
        if pd.zero[j] {
            continue; // c_j − a_jᵀy* = 0 exactly
        }
        let (rl, ru) = enclosed_rc(pd, c[j], cols, j);
        let (l_open, u_open) = (l[j] <= -INF, u[j] >= INF);
        if (l_open && ru > 0.0) || (u_open && rl < 0.0) {
            return None;
        }
        // min over the finite corners of r·x, r ∈ [rl, ru]; an open side that
        // survived the test contributes nothing below them.
        let mut best = f64::INFINITY;
        for x in [l[j], u[j]] {
            if x <= -INF || x >= INF {
                continue;
            }
            for r in [rl, ru] {
                let t = r * x;
                best = best.min(t - t.abs() * f64::EPSILON);
            }
        }
        if best == f64::INFINITY {
            if rl >= 0.0 && ru <= 0.0 {
                continue; // interval is {0}
            }
            return None;
        }
        sum += best;
        sum_abs += best.abs();
    }
    // The sum of the `n` column terms and `by_lo`: `γ_{n+2}` over every magnitude
    // that entered it.
    let g = by_lo + sum - gamma(n + 2) * (sum_abs + by_lo.abs() + by.abs() + by_err);
    let s = 1.0 + by.abs() + sum_abs;
    let g = g - super::refine::NS_MARGIN_REL * s;
    g.is_finite().then_some(g)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `min f  s.t.  x0 − f = 0,  x1 + 49f = b1`, `x ∈ [0,1]²`, `f ∈ [0, ∞)` split
    /// as the engine sees a free column (`f⁺ − f⁻`). Duals from a vertex carry
    /// rounding (`49·fl(1/49) ≠ 1`), which defeats the plain NS bound on the open
    /// columns; the projected enclosure must give a finite bound that never exceeds
    /// the true optimum.
    #[test]
    fn ns_projection_bounds_split_free_column_2115() {
        let (m, n) = (2usize, 4usize);
        let sp = SparseCols::from_csc(
            vec![0, 1, 2, 4, 6],
            vec![0, 1, 0, 1, 0, 1],
            vec![1.0, 1.0, -1.0, 49.0, 1.0, -49.0],
        );
        let l = vec![0.0; 4];
        let u = vec![1.0, 1.0, INF, INF];
        // min x1: f = x0 ≤ 1 and x1 = 50 − 49·x0, so the optimum is x1 = 1 at
        // x0 = 1. The optimal dual is y = (49, 1) (from f: −y0 + 49·y1 = 0); a
        // one-ulp perturbation of y0 is what a vertex's btran leaves behind.
        let c = vec![0.0, 1.0, 0.0, 0.0];
        let b = vec![0.0, 50.0];
        let y = vec![49.0 * (1.0 + f64::EPSILON), 1.0];
        let mut checks = 0usize;
        let cand = open_noise_candidates(&y, &c, &sp, n, &l, &mut |j| u[j] >= INF);
        assert!(
            !cand.is_empty(),
            "premise: a split column rides an open side"
        );
        checks += 1;
        let (cp, ri, vv) = sp.raw();
        assert!(
            super::super::refine::ns_safe_bound_csc(&y, &c, cp, ri, vv, m, n, &b, &l, &u).is_none(),
            "premise: the plain NS bound abstains"
        );
        checks += 1;
        let pd = project_dual(&y, &c, &sp, m, n, &cand).expect("projection");
        let g = ns_bound_enclosed(&pd, &c, &sp, m, n, &b, &l, &u).expect("finite bound");
        assert!(g <= 1.0, "bound {g} above the LP optimum 1");
        assert!(g > 1.0 - 1e-6, "bound {g} not tight");
        checks += 2;
        assert!(checks == 4);
    }

    /// The projection must never certify a bound above the optimum, whatever `y`
    /// it is given: random duals around the true one, on the same LP.
    #[test]
    fn ns_projection_never_exceeds_optimum_2115() {
        let (m, n) = (2usize, 4usize);
        let sp = SparseCols::from_csc(
            vec![0, 1, 2, 4, 6],
            vec![0, 1, 0, 1, 0, 1],
            vec![1.0, 1.0, -1.0, 49.0, 1.0, -49.0],
        );
        let l = vec![0.0; 4];
        let u = vec![1.0, 1.0, INF, INF];
        let c = vec![0.0, 1.0, 0.0, 0.0];
        let b = vec![0.0, 50.0];
        let mut state = 0x2115u64;
        let mut rnd = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        let (mut evaluated, mut finite) = (0usize, 0usize);
        for _ in 0..2000 {
            let s = 10f64.powf(-16.0 + 15.0 * rnd().abs());
            // y0 ≈ 49·y1 (the open columns' balance) to relative noise `s`, y1 free.
            let y1 = 1.0 + 0.5 * rnd();
            let y = vec![49.0 * y1 * (1.0 + s * rnd()), y1];
            let cand = open_noise_candidates(&y, &c, &sp, n, &l, &mut |j| u[j] >= INF);
            if let Ok(pd) = project_dual(&y, &c, &sp, m, n, &cand) {
                evaluated += 1;
                if let Some(g) = ns_bound_enclosed(&pd, &c, &sp, m, n, &b, &l, &u) {
                    finite += 1;
                    assert!(g <= 1.0, "bound {g} above the optimum 1 from y = {y:?}");
                }
            }
        }
        assert!(
            evaluated > 100 && finite > 10,
            "probe fired {evaluated}/{finite}"
        );
    }
}
