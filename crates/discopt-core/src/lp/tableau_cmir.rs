//! Complemented-MIR cuts on tableau aggregations, with variable-bound
//! substitution (#1917 G3.2 step 2). Default OFF behind
//! `DISCOPT_RUST_MILP_TABLEAU_CMIR`.
//!
//! This follows HiGHS's tableau separator (`HighsTableauSeparator.cpp`), which
//! hands each fractional-basic tableau row, **both unnegated and negated**, to its
//! c-MIR cut generator (`HighsCutGeneration`). Bound substitution follows
//! `HighsTransformedLp`. Plain GMI (`gomory.rs`) reads one cut per row, from the
//! nonbasic bound each column sits at, with `delta = 1`. This module differs in
//! three ways:
//!
//! 1. **Variable-bound substitution.** A continuous column `x` with a detected
//!    variable bound `x <= c0 + sum U_v y_v` (or `>=`, `y_v` integer) can be
//!    rewritten through that bound instead of its simple bound. Its coefficient
//!    then moves onto the integers, where the MIR rounding can use it. The
//!    entry experiment attributes the step-2 gain to this
//!    (docs/dev/g32-tableau-cmir-entry-gate.md).
//! 2. **A delta search.** The row is divided by each candidate `delta` before
//!    rounding, and the most efficacious cut is kept.
//! 3. **Both signs of every row.**
//!
//! # Validity
//!
//! The aggregated row `sum_j a_j z_j = beta` with `a = w^T A`, `beta = w^T b` is a
//! linear combination of the equality rows. It holds at every feasible point
//! for **any** `w`, so validity does not depend on how accurate the tableau is.
//! Each step after that is a relaxation or an exact rewrite:
//!
//! - **Bound substitution is exact.** `x = c0 + sum U y + sigma t` *defines* the
//!   slack `t`, and `t >= 0` is the bound itself.
//! - **Dropping a term is a relaxation.** A transformed continuous term with a
//!   positive coefficient is dropped from the `<=` row. That is valid because
//!   `t >= 0`.
//! - **Tiny terms.** A tiny coefficient leaves the row only by charging its
//!   minimum over the box to the rhs (#1236). When that bound is infinite, the
//!   term stays in the row exactly.
//! - **Integer bounds.** Integers are complemented to `ceil(l)` or `floor(u)`,
//!   so the shifted variable is a nonnegative integer.
//! - **The MIR step itself.** Its validity for `sum a'_j x'_j + sum c_k s_k <=
//!   beta'`, with `x' in Z_+` and `s >= 0`, is Marchand and Wolsey (2001).
//!
//! # Floating point
//!
//! `a_j` and `beta` are computed in floating point. The `<=` rhs is relaxed by a
//! rigorous bound on the dot-product error, `gamma_k * sum |w_i A_ij| * max|z_j|`,
//! over every **bounded** column. A column with an infinite range cannot be
//! covered this way. That limitation is shared with `gomory.rs` and is
//! documented rather than hidden. The finished cut's rhs is relaxed once more by
//! a few ulps of its magnitude.

use super::basis::Basis;
use super::gomory::{btran_refined, GomoryCut};
use super::simplex::linsolve::{FeralLU, LinearSolver};
use super::simplex::sparse::SparseCols;
use crate::numeric::EFFECTIVE_INF;

/// Minimum fractionality of the basic integer whose row is aggregated (as GMI).
const FRAC_MIN: f64 = 1e-3;
/// A delta whose scaled rhs fractionality `f` falls outside `[F_MIN, 1-F_MIN]`
/// is skipped, because `1/(1-f)` and `1/f` blow up near the ends (HiGHS uses the
/// same window).
const F_MIN: f64 = 0.01;
/// Cap on a cut coefficient and on the rhs (as GMI).
const MAX_ABS_COEFF: f64 = 1e7;
const MAX_ABS_RHS: f64 = 1e9;
/// Below this magnitude a row coefficient is treated as noise. It is relaxed
/// into the rhs when bounded and otherwise kept exactly.
const TINY: f64 = 1e-12;
/// An integer counts as strictly interior (its `|a'|` becomes a delta candidate)
/// when it is at least this far from both bounds at the LP point.
const INTERIOR: f64 = 1e-6;
/// Minimum efficacy for a cut to be emitted.
const MIN_EFFICACY: f64 = 1e-5;

/// Is `v` a finite bound, not the `1e20` sentinel? Bounds are tested directly
/// and never through a product (CLAUDE.md, the INF note).
#[inline]
fn finite(v: f64) -> bool {
    v.is_finite() && v.abs() < EFFECTIVE_INF
}

/// A variable bound on a continuous column `x`:
/// `x = c0 + sum_v U_v y_v + sigma * t`, with `t >= 0`.
/// `sigma = +1` is a lower bound (VLB) and `sigma = -1` an upper bound (VUB).
/// Every `y_v` is integer-constrained.
#[derive(Debug, Clone, PartialEq)]
pub struct VarBound {
    /// `+1` for a lower bound (VLB), `-1` for an upper bound (VUB).
    pub sigma: f64,
    /// Constant term.
    pub c0: f64,
    /// `(integer column, U_v)` pairs.
    pub ys: Vec<(usize, f64)>,
}

/// Detect the variable bounds implied by the base rows `0..m_base`.
///
/// A row qualifies when it has exactly one continuous structural column `x`,
/// at least one integer structural column, and exactly one non-structural
/// (logical) column. That logical column must be a singleton. The row reads
/// `a x + sum c_v y_v + g s = b` with `s in [ls, us]`, so
/// `a x + sum c_v y_v` lies in an interval `[rl, ru]`, and each finite end
/// gives one bound on `x`. Returns one list per structural column.
#[allow(clippy::too_many_arguments)]
pub fn detect_variable_bounds(
    sp: &SparseCols,
    m_base: usize,
    n_base: usize,
    ns: usize,
    b: &[f64],
    l: &[f64],
    u: &[f64],
    is_int: &[bool],
) -> Vec<Vec<VarBound>> {
    let mut out: Vec<Vec<VarBound>> = vec![Vec::new(); ns];
    if m_base == 0 {
        return out;
    }
    // Row-major view of the structural block.
    let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m_base];
    for j in 0..ns {
        let (ri, rv) = sp.col(j);
        for (&r, &v) in ri.iter().zip(rv) {
            if r < m_base && v != 0.0 {
                rows[r].push((j, v));
            }
        }
    }
    // The logical columns that reach each row: (column, coefficient, singleton?).
    let mut logical: Vec<Vec<(usize, f64, bool)>> = vec![Vec::new(); m_base];
    for j in ns..n_base {
        let (ri, rv) = sp.col(j);
        let single = ri.len() == 1;
        for (&r, &v) in ri.iter().zip(rv) {
            if r < m_base && v != 0.0 {
                logical[r].push((j, v, single));
            }
        }
    }
    for r in 0..m_base {
        if logical[r].len() != 1 || !logical[r][0].2 {
            continue;
        }
        let (sj, g, _) = logical[r][0];
        let row = &rows[r];
        let conts: Vec<&(usize, f64)> = row.iter().filter(|(j, _)| !is_int[*j]).collect();
        if conts.len() != 1 || row.len() < 2 {
            continue;
        }
        let (xj, a) = *conts[0];
        let (ls, us) = (l[sj], u[sj]);
        // sum = b - g s.
        let (ru, rl) = if g > 0.0 {
            (
                finite(ls).then(|| b[r] - g * ls),
                finite(us).then(|| b[r] - g * us),
            )
        } else {
            (
                finite(us).then(|| b[r] - g * us),
                finite(ls).then(|| b[r] - g * ls),
            )
        };
        let ys: Vec<(usize, f64)> = row
            .iter()
            .filter(|(j, _)| *j != xj)
            .map(|&(j, c)| (j, -c / a))
            .collect();
        if ys.iter().any(|(_, uv)| !uv.is_finite()) {
            continue;
        }
        // a x + sum c y <= ru:  a > 0 -> x <= ru/a - sum (c/a) y (upper bound).
        if let Some(ru) = ru {
            let c0 = ru / a;
            if c0.is_finite() {
                out[xj].push(VarBound {
                    sigma: if a > 0.0 { -1.0 } else { 1.0 },
                    c0,
                    ys: ys.clone(),
                });
            }
        }
        if let Some(rl) = rl {
            let c0 = rl / a;
            if c0.is_finite() {
                out[xj].push(VarBound {
                    sigma: if a > 0.0 { 1.0 } else { -1.0 },
                    c0,
                    ys,
                });
            }
        }
    }
    out
}

/// Why a signed tableau row produced no cut.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// A continuous column has no finite simple bound and no variable bound,
    /// or an integer column has no finite bound. Without one it cannot be
    /// substituted, so the row is refused rather than mishandled.
    Free,
    /// No delta gave a usable fractionality, or no candidate cut was violated.
    NoDelta,
    /// The best cut failed a numerics gate (non-finite value, coefficient or rhs
    /// size, or every coefficient tiny).
    Numerics,
    /// The best cut's `max|c| / min|c|` exceeded `max_dynamism`.
    Dynamism,
}

/// Counters from one separator call; [`separate_tableau_cmir`] forwards them to
/// the profile.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CmirStats {
    /// Signed tableau rows handed to c-MIR.
    pub rows: u64,
    /// Cuts emitted.
    pub cuts: u64,
    /// Variable-bound substitutions in the emitted cuts.
    pub vb_subst: u64,
    /// Rows refused for a free column ([`Refusal::Free`]).
    pub refused_free: u64,
    /// Rows with no usable delta or no violated cut ([`Refusal::NoDelta`]).
    pub no_delta: u64,
    /// Rows whose best cut failed a numerics gate ([`Refusal::Numerics`]).
    pub refused_numerics: u64,
    /// Rows whose best cut failed the dynamism gate ([`Refusal::Dynamism`]).
    pub refused_dynamism: u64,
}

/// One kept continuous term after substitution: coefficient `c < 0` on the
/// slack `t` of the bound `x = c0 + sum U y + sigma t`.
struct STerm<'a> {
    c: f64,
    x: usize,
    sigma: f64,
    c0: f64,
    ys: &'a [(usize, f64)],
}

/// A sparse cut `(column, coefficient)` list.
type SparseCut = Vec<(usize, f64)>;

/// One row's cut: `(coeffs, rhs, variable-bound substitutions)`, read as
/// `coeffs . z <= rhs`.
pub type RowCut = (SparseCut, f64, u64);

/// A candidate bound for a continuous column: (rounded distance, priority,
/// sigma, c0, ys).
type Pick<'v> = (f64, u8, f64, f64, &'v [(usize, f64)]);

/// Offer a bound with LP-point distance `d`; keep it when it is closer (to 1e-9)
/// or equally close with a better priority.
#[inline]
fn offer<'v>(
    best: &mut Option<Pick<'v>>,
    d: f64,
    prio: u8,
    sigma: f64,
    c0: f64,
    ys: &'v [(usize, f64)],
) {
    if !d.is_finite() {
        return;
    }
    let k = (d / 1e-9).round();
    let better = match best {
        None => true,
        Some((bk, bp, ..)) => k < *bk || (k == *bk && prio < *bp),
    };
    if better {
        *best = Some((k, prio, sigma, c0, ys));
    }
}

/// Dense scratch accumulator with a touched list, reused across rows.
struct Accum {
    val: Vec<f64>,
    seen: Vec<bool>,
    touched: Vec<usize>,
}

impl Accum {
    fn new(n: usize) -> Self {
        Accum {
            val: vec![0.0; n],
            seen: vec![false; n],
            touched: Vec::new(),
        }
    }
    #[inline]
    fn add(&mut self, j: usize, v: f64) {
        if !self.seen[j] {
            self.seen[j] = true;
            self.touched.push(j);
        }
        self.val[j] += v;
    }
    fn clear(&mut self) {
        for &j in &self.touched {
            self.val[j] = 0.0;
            self.seen[j] = false;
        }
        self.touched.clear();
    }
}

/// c-MIR on the single row `sum_j row_j z_j <= beta`.
///
/// Returns the best cut as a sparse `<=` inequality `(coeffs, rhs)` and the
/// number of variable-bound substitutions it used. `x` is the LP point; it
/// steers bound choice, complementation and delta choice, but never validity.
#[allow(clippy::too_many_arguments)]
pub fn cmir_row(
    row: &[(usize, f64)],
    beta_in: f64,
    x: &[f64],
    l: &[f64],
    u: &[f64],
    is_int: &[bool],
    vbs: &[Vec<VarBound>],
    tol: f64,
    max_dynamism: f64,
) -> Result<RowCut, Refusal> {
    let n = l.len();
    let mut ints = Accum::new(n);
    let mut cut = Accum::new(n);
    cmir_row_ws(
        row,
        beta_in,
        x,
        l,
        u,
        is_int,
        vbs,
        tol,
        max_dynamism,
        &mut ints,
        &mut cut,
    )
}

#[allow(clippy::too_many_arguments)]
fn cmir_row_ws<'v>(
    row: &[(usize, f64)],
    beta_in: f64,
    x: &[f64],
    l: &[f64],
    u: &[f64],
    is_int: &[bool],
    vbs: &'v [Vec<VarBound>],
    tol: f64,
    max_dynamism: f64,
    ints: &mut Accum,
    cut: &mut Accum,
) -> Result<RowCut, Refusal> {
    ints.clear();
    cut.clear();
    let mut beta = beta_in;
    let mut sterms: Vec<STerm<'v>> = Vec::new();
    let mut vb_used = 0u64;
    const NO_YS: &[(usize, f64)] = &[];

    for &(j, a) in row {
        if a == 0.0 {
            continue;
        }
        if a.abs() <= TINY {
            // #1236: relax the term away, a z_j >= a * (minimising bound).
            let lo = if a > 0.0 { l[j] } else { u[j] };
            if finite(lo) {
                beta -= a * lo;
                continue;
            }
            // Unbounded: it cannot move, so it is kept exactly (below).
        }
        if is_int[j] {
            ints.add(j, a);
            continue;
        }
        // Continuous: the closest bound at the LP point, ties broken by
        // variable bound, then upper, then lower (the entry prototype's order).
        let mut best: Option<Pick<'v>> = None;
        if finite(l[j]) {
            offer(&mut best, x[j] - l[j], 2, 1.0, l[j], NO_YS);
        }
        if finite(u[j]) {
            offer(&mut best, u[j] - x[j], 1, -1.0, u[j], NO_YS);
        }
        if j < vbs.len() {
            for vb in &vbs[j] {
                let val = vb.c0 + vb.ys.iter().map(|&(y, uv)| uv * x[y]).sum::<f64>();
                offer(
                    &mut best,
                    vb.sigma * (x[j] - val),
                    0,
                    vb.sigma,
                    vb.c0,
                    &vb.ys,
                );
            }
        }
        let Some((_, _, sigma, c0, ys)) = best else {
            return Err(Refusal::Free);
        };
        if !ys.is_empty() {
            vb_used += 1;
        }
        // a x = a c0 + sum a U y + a sigma t.
        beta -= a * c0;
        for &(y, uv) in ys {
            ints.add(y, a * uv);
        }
        let ct = a * sigma;
        if ct < 0.0 {
            sterms.push(STerm {
                c: ct,
                x: j,
                sigma,
                c0,
                ys,
            });
        }
        // ct > 0: dropping `ct * t` (t >= 0) from a `<=` row is a relaxation.
    }

    // Complement the integers to an integral bound.
    // (column, a', complemented, bound used)
    let mut it: Vec<(usize, f64, bool, f64)> = Vec::new();
    let mut deltas: Vec<f64> = vec![1.0];
    for &j in &ints.touched {
        let al = ints.val[j];
        if al == 0.0 {
            continue;
        }
        let li = if finite(l[j]) {
            (l[j] - 1e-9).ceil()
        } else {
            f64::NEG_INFINITY
        };
        let ui = if finite(u[j]) {
            (u[j] + 1e-9).floor()
        } else {
            f64::INFINITY
        };
        let (lf, uf) = (li.is_finite(), ui.is_finite());
        if al.abs() <= TINY {
            let lo = if al > 0.0 { li } else { ui };
            if lo.is_finite() {
                beta -= al * lo;
                continue;
            }
        }
        let comp = match (lf, uf) {
            (true, true) => x[j] > 0.5 * (li + ui),
            (true, false) => false,
            (false, true) => true,
            (false, false) => return Err(Refusal::Free),
        };
        let (at, bnd) = if comp {
            beta -= al * ui;
            (-al, ui)
        } else {
            beta -= al * li;
            (al, li)
        };
        it.push((j, at, comp, bnd));
        if lf && uf && x[j] - li > INTERIOR && ui - x[j] > INTERIOR && at.abs() > 1e-9 {
            deltas.push(at.abs());
        }
    }
    if it.is_empty() || !beta.is_finite() {
        return Err(Refusal::NoDelta);
    }
    deltas.sort_by(|a, b| a.partial_cmp(b).unwrap());
    deltas.dedup_by(|a, b| (*a - *b).abs() <= 1e-9 * b.abs().max(1.0));
    let base = deltas.clone();
    for &d in &base {
        for k in [2.0, 4.0, 8.0] {
            deltas.push(d / k);
        }
    }

    // (efficacy, coeffs, rhs)
    let mut best: Option<(f64, SparseCut, f64)> = None;
    for &dl in &deltas {
        let bt = beta / dl;
        if !bt.is_finite() {
            continue;
        }
        let f = bt - bt.floor();
        if !(F_MIN..=1.0 - F_MIN).contains(&f) {
            continue;
        }
        cut.clear();
        let mut rhs = bt.floor();
        for &(j, at, comp, bnd) in &it {
            let vv = at / dl;
            let fj = vv - vv.floor();
            let g = vv.floor() + (fj - f).max(0.0) / (1.0 - f);
            if comp {
                // x' = bnd - x
                cut.add(j, -g);
                rhs -= g * bnd;
            } else {
                // x' = x - bnd
                cut.add(j, g);
                rhs += g * bnd;
            }
        }
        for s in &sterms {
            let w = s.c / dl / (1.0 - f);
            // t = sigma x - sigma c0 - sum sigma U y
            cut.add(s.x, w * s.sigma);
            for &(y, uv) in s.ys {
                cut.add(y, -w * s.sigma * uv);
            }
            rhs += w * s.sigma * s.c0;
        }
        let rhs = rhs * dl;
        let mut act = 0.0;
        let mut nrm2 = 0.0;
        let mut coeffs: Vec<(usize, f64)> = Vec::with_capacity(cut.touched.len());
        for &j in &cut.touched {
            let v = cut.val[j] * dl;
            if v != 0.0 {
                act += v * x[j];
                nrm2 += v * v;
                coeffs.push((j, v));
            }
        }
        if nrm2 == 0.0 || !act.is_finite() || !rhs.is_finite() {
            continue;
        }
        let eff = (act - rhs) / nrm2.sqrt();
        if best.as_ref().is_none_or(|b| eff > b.0) {
            best = Some((eff, coeffs, rhs));
        }
    }
    let Some((_, coeffs, mut rhs)) = best else {
        return Err(Refusal::NoDelta);
    };

    // Final cleanup. A coefficient that is tiny in absolute terms (`<= tol`) or
    // relative to the largest (`< max|c| / max_dynamism`) leaves the cut by
    // charging its minimum over the box to the rhs, which is a relaxation
    // (HiGHS removes small cut coefficients the same way). An unbounded one
    // cannot move and stays; above `tol` the dynamism gate below still sees it. Then add
    // a few ulps of slack and apply the numerics gates.
    if coeffs.iter().any(|(_, c)| !c.is_finite()) {
        return Err(Refusal::Numerics);
    }
    let max_all = coeffs.iter().fold(0.0_f64, |a, (_, c)| a.max(c.abs()));
    let small = tol.max(max_all / max_dynamism);
    let mut kept: Vec<(usize, f64)> = Vec::with_capacity(coeffs.len());
    let mut mag = rhs.abs();
    let mut max_c = 0.0_f64;
    let mut min_c = f64::INFINITY;
    for (j, c) in coeffs {
        let tiny = c.abs() <= small;
        if tiny {
            let lo = if c > 0.0 { l[j] } else { u[j] };
            if finite(lo) {
                rhs -= c * lo;
                mag += (c * lo).abs();
                continue;
            }
        }
        if finite(l[j]) && finite(u[j]) {
            mag += c.abs() * l[j].abs().max(u[j].abs());
        }
        max_c = max_c.max(c.abs());
        // As in `gomory.rs` (#1236): a coefficient at or below `tol` kept only
        // because its range is unbounded does not count toward dynamism.
        if c.abs() > tol {
            min_c = min_c.min(c.abs());
        }
        kept.push((j, c));
    }
    rhs += 8.0 * f64::EPSILON * mag;
    if kept.is_empty()
        || !min_c.is_finite()
        || max_c > MAX_ABS_COEFF
        || !rhs.is_finite()
        || rhs.abs() > MAX_ABS_RHS
    {
        return Err(Refusal::Numerics);
    }
    if max_c / min_c > max_dynamism {
        return Err(Refusal::Dynamism);
    }
    let act: f64 = kept.iter().map(|&(j, c)| c * x[j]).sum();
    let nrm = kept.iter().map(|&(_, c)| c * c).sum::<f64>().sqrt();
    let viol = act - rhs;
    if viol / nrm <= MIN_EFFICACY || viol <= 1e-6 * (1.0 + rhs.abs()) {
        return Err(Refusal::NoDelta);
    }
    Ok((kept, rhs, vb_used))
}

/// Separate tableau c-MIR cuts from the basis of the working LP
/// `[A | logicals] z = b`. Emits cuts as `coeffs . z >= rhs` (the GMI
/// convention), so the driver's substitution and selection treat them like GMI.
#[allow(clippy::too_many_arguments)]
pub fn separate_tableau_cmir(
    sp: &SparseCols,
    m: usize,
    n: usize,
    l: &[f64],
    u: &[f64],
    b: &[f64],
    basis: &Basis,
    is_int: &[bool],
    x: &[f64],
    vbs: &[Vec<VarBound>],
    tol: f64,
    max_dynamism: f64,
) -> (Vec<GomoryCut>, CmirStats) {
    let mut st = CmirStats::default();
    let mut cuts = Vec::new();
    // Counted on entry, so `Calls > 0` with `Rows == 0` shows an early refusal.
    crate::profile::incr(crate::profile::Ctr::SepCmirCalls);
    if m == 0
        || basis.basic_vars.len() != m
        || x.len() < n
        || is_int.len() < n
        || l.len() < n
        || u.len() < n
    {
        return (cuts, st);
    }
    let mut lu = FeralLU::new();
    let bcols: Vec<Vec<(usize, f64)>> = basis
        .basic_vars
        .iter()
        .map(|&bv| {
            let (rows, vals) = sp.col(bv);
            rows.iter().zip(vals).map(|(&r, &v)| (r, v)).collect()
        })
        .collect();
    if lu.factorize_sparse(m, &bcols).is_err() {
        return (cuts, st);
    }
    let (l, u) = (&l[..n], &u[..n]);
    let mut ints = Accum::new(n);
    let mut cutacc = Accum::new(n);
    let mut row: Vec<(usize, f64)> = Vec::new();
    let mut neg: Vec<(usize, f64)> = Vec::new();
    for (i, &bi) in basis.basic_vars.iter().enumerate() {
        if !is_int[bi] {
            continue;
        }
        let f0 = x[bi] - x[bi].floor();
        if !(FRAC_MIN..=1.0 - FRAC_MIN).contains(&f0) {
            continue;
        }
        let mut e_i = vec![0.0_f64; m];
        e_i[i] = 1.0;
        let Some(w) = btran_refined(&mut lu, sp, &basis.basic_vars, &e_i, tol) else {
            continue;
        };
        // a_j = w . A_j, and a rigorous bound on the rounding error of each dot
        // (gamma_k * sum|w_i A_ij|), charged at the column's largest |z_j|.
        row.clear();
        let mut err = 0.0_f64;
        for j in 0..n {
            let (a, absum, k) = sp.dot_with_magnitude(j, &w);
            if a == 0.0 && absum == 0.0 {
                continue;
            }
            if finite(l[j]) && finite(u[j]) {
                err += (k as f64 + 1.0) * absum * l[j].abs().max(u[j].abs());
            }
            if a != 0.0 {
                row.push((j, a));
            }
        }
        let mut beta = 0.0_f64;
        let mut babs = 0.0_f64;
        for (wi, bi) in w.iter().zip(b) {
            beta += wi * bi;
            babs += (wi * bi).abs();
        }
        err += (m as f64 + 1.0) * babs;
        let margin = 2.0 * f64::EPSILON * err;
        if !beta.is_finite() || !margin.is_finite() {
            st.refused_numerics += 2;
            continue;
        }
        neg.clear();
        neg.extend(row.iter().map(|&(j, a)| (j, -a)));
        for (r, bb) in [(&row, beta), (&neg, -beta)] {
            st.rows += 1;
            match cmir_row_ws(
                r,
                bb + margin,
                x,
                l,
                u,
                is_int,
                vbs,
                tol,
                max_dynamism,
                &mut ints,
                &mut cutacc,
            ) {
                Ok((coeffs, rhs, vbu)) => {
                    st.cuts += 1;
                    st.vb_subst += vbu;
                    let mut dense = vec![0.0_f64; n];
                    for (j, c) in coeffs {
                        dense[j] = -c;
                    }
                    cuts.push(GomoryCut {
                        coeffs: dense,
                        rhs: -rhs,
                    });
                }
                Err(Refusal::Free) => st.refused_free += 1,
                Err(Refusal::NoDelta) => st.no_delta += 1,
                Err(Refusal::Numerics) => st.refused_numerics += 1,
                Err(Refusal::Dynamism) => st.refused_dynamism += 1,
            }
        }
    }
    use crate::profile::{incr_by, Ctr};
    incr_by(Ctr::SepCmirRows, st.rows);
    incr_by(Ctr::SepCmirCutsEmitted, st.cuts);
    incr_by(Ctr::SepCmirVbSubst, st.vb_subst);
    incr_by(Ctr::SepCmirRefusedFree, st.refused_free);
    incr_by(Ctr::SepCmirNoDelta, st.no_delta);
    incr_by(Ctr::SepCmirRefusedNumerics, st.refused_numerics);
    incr_by(Ctr::SepCmirRefusedDynamism, st.refused_dynamism);
    (cuts, st)
}

/// Whether `DISCOPT_RUST_MILP_TABLEAU_CMIR` asks for the separator (default OFF).
/// Read fresh on each solve. An unrecognized value is a hard refusal.
pub fn tableau_cmir_enabled() -> bool {
    match std::env::var("DISCOPT_RUST_MILP_TABLEAU_CMIR") {
        Err(_) => false,
        Ok(v) => match parse_tableau_cmir_flag(&v) {
            Ok(on) => on,
            Err(msg) => panic!("{msg}"),
        },
    }
}

/// Parse a `DISCOPT_RUST_MILP_TABLEAU_CMIR` value (pure, for tests).
pub fn parse_tableau_cmir_flag(raw: &str) -> Result<bool, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "0" | "false" | "no" => Ok(false),
        "1" | "true" | "yes" => Ok(true),
        other => Err(format!(
            "DISCOPT_RUST_MILP_TABLEAU_CMIR={other:?} is not recognized (expected \
             1/true/yes or 0/false/no). Refusing rather than silently picking an arm."
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::basis::{AT_LOWER, BASIC};

    const INF: f64 = 1e20;

    /// Deterministic LCG so the tests need no RNG dependency.
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
        fn unit(&mut self) -> f64 {
            (self.next() % 1_000_000) as f64 / 1_000_000.0
        }
    }

    /// A small mixed instance in standard form `[A | I] z = b` (logicals in
    /// `[0, INF]`, so every row is `A_r . s <= b_r`). The structurals are `k`
    /// integers in `[0, ymax]` followed by one continuous `xc` in `[0, xmax]`.
    /// Row 0 is the variable upper bound `xc <= vub * y_0`.
    struct Inst {
        rows: Vec<Vec<f64>>,
        b: Vec<f64>,
        k: usize,
        ymax: i64,
        xmax: f64,
    }

    impl Inst {
        fn ns(&self) -> usize {
            self.k + 1
        }
        fn m(&self) -> usize {
            self.rows.len()
        }
        fn n(&self) -> usize {
            self.ns() + self.m()
        }
        fn csc(&self) -> SparseCols {
            let (m, n, ns) = (self.m(), self.n(), self.ns());
            let mut d = vec![0.0; m * n];
            for (i, r) in self.rows.iter().enumerate() {
                d[i * n..i * n + ns].copy_from_slice(r);
                d[i * n + ns + i] = 1.0;
            }
            SparseCols::from_dense(&d, m, n)
        }
        fn bounds(&self) -> (Vec<f64>, Vec<f64>, Vec<bool>) {
            let mut l = vec![0.0; self.n()];
            let mut u = vec![INF; self.n()];
            let mut is_int = vec![false; self.n()];
            for j in 0..self.k {
                u[j] = self.ymax as f64;
                is_int[j] = true;
            }
            u[self.k] = self.xmax;
            l[self.k] = 0.0;
            (l, u, is_int)
        }
        /// Every vertex of the feasible region's integer slices: for each
        /// integer assignment, the two ends of the feasible `xc` interval
        /// (a cut is linear in `xc`, so its minimum over the slice is at an end).
        fn points(&self) -> Vec<Vec<f64>> {
            let (k, ns, m) = (self.k, self.ns(), self.m());
            let mut out = Vec::new();
            let combos = (self.ymax + 1).pow(k as u32);
            for c in 0..combos {
                let mut y = vec![0.0; k];
                let mut cc = c;
                for v in y.iter_mut() {
                    *v = (cc % (self.ymax + 1)) as f64;
                    cc /= self.ymax + 1;
                }
                let (mut lo, mut hi) = (0.0_f64, self.xmax);
                let mut ok = true;
                for (r, row) in self.rows.iter().enumerate() {
                    let s: f64 = (0..k).map(|j| row[j] * y[j]).sum();
                    let d = row[k];
                    let rest = self.b[r] - s;
                    if d > 0.0 {
                        hi = hi.min(rest / d);
                    } else if d < 0.0 {
                        lo = lo.max(rest / d);
                    } else if rest < 0.0 {
                        ok = false;
                    }
                }
                if !ok || lo > hi + 1e-12 {
                    continue;
                }
                for xc in [lo, hi] {
                    let mut z = vec![0.0; ns + m];
                    z[..k].copy_from_slice(&y);
                    z[k] = xc;
                    for (r, row) in self.rows.iter().enumerate() {
                        let a: f64 = (0..ns).map(|j| row[j] * z[j]).sum();
                        z[ns + r] = (self.b[r] - a).max(0.0);
                    }
                    out.push(z);
                }
            }
            out
        }
    }

    fn random_inst(rng: &mut Lcg) -> Inst {
        let k = 3;
        let ymax = 2;
        let vub = rng.int(2, 7) as f64 + 0.5;
        let mut rows = vec![{
            let mut r = vec![0.0; k + 1];
            r[0] = -vub;
            r[k] = 1.0;
            r
        }];
        let mut b = vec![0.0];
        for _ in 0..3 {
            let mut r: Vec<f64> = (0..k).map(|_| rng.int(-4, 6) as f64).collect();
            r.push(rng.int(-3, 3) as f64 + 0.5 * rng.int(0, 1) as f64);
            b.push(rng.int(2, 14) as f64 + 0.25 * rng.int(0, 3) as f64);
            rows.push(r);
        }
        Inst {
            rows,
            b,
            k,
            ymax,
            xmax: 12.0,
        }
    }

    /// Check every cut at every feasible vertex; returns (checks, violations).
    fn check_cuts(cuts: &[GomoryCut], pts: &[Vec<f64>]) -> (usize, usize) {
        let (mut n, mut bad) = (0, 0);
        for c in cuts {
            for z in pts {
                let lhs: f64 = c.coeffs.iter().zip(z).map(|(a, v)| a * v).sum();
                n += 1;
                if lhs < c.rhs - 1e-7 * (1.0 + c.rhs.abs()) {
                    bad += 1;
                }
            }
        }
        (n, bad)
    }

    /// Run the separator over many random instances, bases and points, and
    /// return (cuts, checks, violations, vb substitutions, the cuts per instance
    /// with their feasible points).
    fn sweep(
        seed: u64,
        max_dyn: f64,
    ) -> (
        usize,
        usize,
        usize,
        u64,
        Vec<(Vec<GomoryCut>, Vec<Vec<f64>>)>,
    ) {
        let mut rng = Lcg(seed);
        let (mut ncuts, mut nchk, mut nbad, mut nvb) = (0, 0, 0, 0u64);
        let mut kept = Vec::new();
        for _ in 0..60 {
            let inst = random_inst(&mut rng);
            let pts = inst.points();
            if pts.is_empty() {
                continue;
            }
            let (m, n, ns) = (inst.m(), inst.n(), inst.ns());
            let sp = inst.csc();
            let (l, u, is_int) = inst.bounds();
            let vbs = detect_variable_bounds(&sp, m, n, ns, &inst.b, &l, &u, &is_int);
            for _ in 0..8 {
                // A random basis: always the integers, plus random others.
                let mut cols: Vec<usize> = (0..inst.k).collect();
                while cols.len() < m {
                    let j = rng.int(inst.k as i64, n as i64 - 1) as usize;
                    if !cols.contains(&j) {
                        cols.push(j);
                    }
                }
                let mut status = vec![AT_LOWER; n];
                for &j in &cols {
                    status[j] = BASIC;
                }
                let basis = Basis {
                    col_status: status,
                    basic_vars: cols,
                };
                // An arbitrary fractional point; validity cannot depend on it.
                let x: Vec<f64> = (0..n)
                    .map(|j| {
                        if j < inst.k {
                            rng.unit() * inst.ymax as f64
                        } else if j == inst.k {
                            rng.unit() * inst.xmax
                        } else {
                            rng.unit() * 3.0
                        }
                    })
                    .collect();
                let (cuts, st) = separate_tableau_cmir(
                    &sp, m, n, &l, &u, &inst.b, &basis, &is_int, &x, &vbs, 1e-9, max_dyn,
                );
                let (c, bad) = check_cuts(&cuts, &pts);
                ncuts += cuts.len();
                nchk += c;
                nbad += bad;
                nvb += st.vb_subst;
                if !cuts.is_empty() {
                    kept.push((cuts, pts.clone()));
                }
            }
        }
        (ncuts, nchk, nbad, nvb, kept)
    }

    #[test]
    fn cuts_are_valid_on_enumerated_feasible_points() {
        let (ncuts, nchk, nbad, nvb, _) = sweep(0x5eed_1917, 1e7);
        eprintln!("cuts {ncuts} checks {nchk} violations {nbad} vb_subst {nvb}");
        assert!(ncuts > 50, "probe did not fire: only {ncuts} cuts");
        assert!(nchk > 1000, "only {nchk} executed checks");
        assert!(nvb > 0, "the variable-bound path never engaged");
        assert_eq!(nbad, 0, "{nbad} feasible points cut off");
    }

    /// A tight dynamism cap forces the small-coefficient relaxation in the final
    /// cleanup on most cuts; they must stay valid, and every emitted cut must
    /// respect the cap over its coefficients above `tol` (the ones at or below
    /// `tol` on unbounded columns are exempt, as in `gomory.rs`).
    #[test]
    fn small_coefficient_relaxation_keeps_cuts_valid() {
        let (ncuts, nchk, nbad, _, kept) = sweep(0x5eed_1917, 4.0);
        let relaxed = kept
            .iter()
            .flat_map(|(c, _)| c.iter())
            .filter(|c| {
                let nz: Vec<f64> = c
                    .coeffs
                    .iter()
                    .filter(|v| v.abs() > 1e-9)
                    .map(|v| v.abs())
                    .collect();
                nz.iter().cloned().fold(0.0, f64::max)
                    / nz.iter().cloned().fold(f64::INFINITY, f64::min)
                    <= 4.0
            })
            .count();
        eprintln!("cuts {ncuts} checks {nchk} violations {nbad} within-cap {relaxed}");
        assert!(
            ncuts > 20 && nchk > 500,
            "probe did not fire: {ncuts} cuts, {nchk} checks"
        );
        assert_eq!(relaxed, ncuts, "a cut above the dynamism cap was emitted");
        assert_eq!(nbad, 0, "{nbad} feasible points cut off");
    }

    /// Mutation check: the validity checker above must catch an invalid cut.
    /// Each emitted cut's rhs is raised to just above its minimum over the
    /// feasible points, which makes it cut one of them off.
    #[test]
    fn checker_detects_a_strengthened_cut() {
        let (_, _, _, _, kept) = sweep(0x5eed_1917, 1e7);
        let mut caught = 0usize;
        let mut tried = 0usize;
        for (cuts, pts) in &kept {
            for c in cuts {
                let minv = pts
                    .iter()
                    .map(|z| c.coeffs.iter().zip(z).map(|(a, v)| a * v).sum::<f64>())
                    .fold(f64::INFINITY, f64::min);
                let bad = GomoryCut {
                    coeffs: c.coeffs.clone(),
                    rhs: minv + 1e-3 * (1.0 + minv.abs()),
                };
                tried += 1;
                if check_cuts(&[bad], pts).1 > 0 {
                    caught += 1;
                }
            }
        }
        assert!(tried > 50);
        assert_eq!(caught, tried);
    }

    #[test]
    fn variable_upper_bound_is_detected() {
        // xc - 5 y + s = 0, s in [0, INF]   =>   xc <= 5 y.
        let inst = Inst {
            rows: vec![vec![-5.0, 1.0]],
            b: vec![0.0],
            k: 1,
            ymax: 1,
            xmax: 10.0,
        };
        let (l, u, is_int) = inst.bounds();
        let vbs = detect_variable_bounds(
            &inst.csc(),
            inst.m(),
            inst.n(),
            inst.ns(),
            &inst.b,
            &l,
            &u,
            &is_int,
        );
        assert!(vbs[0].is_empty());
        assert_eq!(
            vbs[1],
            vec![VarBound {
                sigma: -1.0,
                c0: 0.0,
                ys: vec![(0, 5.0)]
            }]
        );
    }

    /// The classic case: the VUB `x <= 5 y` and the row `x >= 2`, with the LP
    /// point at `y = 0.4, x = 2`. Through the VUB the aggregation becomes
    /// `5y >= 2 + t`, whose MIR is `y >= 1` -- strictly stronger than anything
    /// from the simple bounds. Measures that the VUB is what gives the cut.
    #[test]
    fn vub_substitution_yields_the_rounded_cut() {
        // Columns: y (int, [0,1]), x (cont, [0,10]).  Row: -x <= -2.
        let row = vec![(1usize, -1.0)];
        let l = vec![0.0, 0.0];
        let u = vec![1.0, 10.0];
        let is_int = vec![true, false];
        let x = vec![0.4, 2.0];
        let vbs = vec![
            vec![],
            vec![VarBound {
                sigma: -1.0,
                c0: 0.0,
                ys: vec![(0, 5.0)],
            }],
        ];
        let (cut, rhs, vbu) =
            cmir_row(&row, -2.0, &x, &l, &u, &is_int, &vbs, 1e-9, 1e7).expect("a cut");
        assert_eq!(vbu, 1);
        // Expect a multiple of -y <= -1 (possibly with an x term at coefficient 0).
        let cy = cut.iter().find(|(j, _)| *j == 0).map(|p| p.1).unwrap();
        assert!(cy < 0.0);
        assert!((rhs / cy - 1.0).abs() < 1e-9, "cut {cut:?} <= {rhs}");
        // And without the VUB there is nothing to round: no cut.
        let none: Vec<Vec<VarBound>> = vec![vec![], vec![]];
        assert!(cmir_row(&row, -2.0, &x, &l, &u, &is_int, &none, 1e-9, 1e7).is_err());
    }

    #[test]
    fn integer_row_rounds_down() {
        let row = vec![(0usize, 1.0)];
        let (cut, rhs, _) = cmir_row(
            &row,
            1.5,
            &[1.5],
            &[0.0],
            &[3.0],
            &[true],
            &[vec![]],
            1e-9,
            1e7,
        )
        .unwrap();
        assert_eq!(cut.len(), 1);
        assert!((rhs / cut[0].1 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn free_continuous_column_refuses_the_row() {
        // y + x <= 1.5 with x free and no variable bound: no substitution exists.
        let row = vec![(0usize, 1.0), (1usize, 1.0)];
        let r = cmir_row(
            &row,
            1.5,
            &[0.5, 0.0],
            &[0.0, -INF],
            &[3.0, INF],
            &[true, false],
            &[vec![], vec![]],
            1e-9,
            1e7,
        );
        assert_eq!(r.unwrap_err(), Refusal::Free);
    }

    #[test]
    fn flag_parser_refuses_garbage() {
        assert_eq!(parse_tableau_cmir_flag("1"), Ok(true));
        assert_eq!(parse_tableau_cmir_flag(" Yes "), Ok(true));
        assert_eq!(parse_tableau_cmir_flag(""), Ok(false));
        assert_eq!(parse_tableau_cmir_flag("0"), Ok(false));
        assert!(parse_tableau_cmir_flag("on").is_err());
    }
}
