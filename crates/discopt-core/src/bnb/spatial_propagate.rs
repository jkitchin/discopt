//! Per-node FBBT fixpoint propagation for the native spatial B&B kernel
//! (issue #764 → C2, entry experiment GO 2026-07-19).
//!
//! The SCIP mechanism trace proved tanksize's dual bound climbs via cheap,
//! cutoff-coupled *nonlinear constraint propagation* (38k domain reductions; OBBT
//! and cuts near-irrelevant), and the C2 entry experiment reproduced it on the real
//! instance with a clean dose-response: propagation off → bound frozen at 0.838
//! forever; strict-sign reverse division → 0.891 hard stall (discopt's known FBBT
//! stall); adding the **extended zero-touching reverse division** → 0.956 @3000
//! nodes and still climbing. This module is the Rust port of that validated
//! propagator, run per node BEFORE the LP — replacing the ~95-probe OBBT sweep as
//! the default tightening.
//!
//! What it propagates, to a fixpoint (all interval-arithmetic, **zero LP solves**):
//! * the objective cutoff `cᵀx <= incumbent` (when an incumbent exists) — the
//!   coupling that lets the incumbent shrink boxes;
//! * every fixed linear row `Σ a_j x_j <= b` (standard activity-based tightening);
//! * every affine-form product `w = A·B` — forward (interval product) and reverse
//!   (interval division), including the one-sided **extended division** when a
//!   factor's interval touches zero (`G ∈ [0, g_hi]`, `w >= w_lo > 0` forces
//!   `G > 0` and `F >= w_lo / g_hi`) — the load-bearing case on boxes whose
//!   variables sit at 0, where strict-sign division is blocked;
//! * the fixed-width terms (bilinear / monomial / affine-square / sqrt), forward
//!   and (guarded, monotone) reverse;
//! * integer rounding.
//!
//! Soundness: every tightening step is a valid interval deduction whose endpoint
//! arithmetic is **directed** (outward-rounded, `presolve::directed`, #1504 /
//! #1537 D), then additionally relaxed by a small `EPS`-scaled margin;
//! infeasibility is declared only when a violation exceeds a conservative
//! tolerance. Skipping any step is always sound (the box just stays looser), so all
//! guarded cases degrade gracefully.
//!
//! #1537 D: the `EPS * (1 + |cap|)` margin alone is NOT an enclosure. It is relative
//! to the *result*, while the rounding error of an activity sum is relative to the
//! largest *term*: a row whose ~1e14-magnitude products cancel to a ~3e9 bound lost
//! ~10 units, and an exact-rational audit of 200,000 random rows found 3,125
//! feasible points cut. Every endpoint below is now computed in directed rounding,
//! so the margin is extra slack rather than the only guard.

use crate::bnb::spatial_kernel::{BlfTerm, EnvTerm, SpatialKernelSpec};
use crate::presolve::directed::{
    add_down, add_up, div_down, div_up, mul_down, mul_up, powi_down, powi_up, root_down, root_up,
    sqrt_down, sqrt_up, sub_down, sub_up,
};

/// Relative tolerance below which a bound crossing counts as real infeasibility.
const INFEAS_TOL: f64 = 1e-7;
/// Change-detection / outward-relaxation epsilon (relative).
const EPS: f64 = 1e-9;

#[inline]
fn rel(v: f64) -> f64 {
    1.0 + v.abs()
}

/// Lower `hi[j]` to `cap` (outward-guarded). True iff a real change was applied.
#[inline]
fn cap_hi(hi: &mut [f64], j: usize, cap: f64) -> bool {
    if !cap.is_finite() {
        return false;
    }
    let guarded = cap + EPS * rel(cap);
    if guarded < hi[j] - EPS * rel(hi[j]) {
        hi[j] = guarded;
        true
    } else {
        false
    }
}

/// Raise `lo[j]` to `cap` (outward-guarded). True iff a real change was applied.
#[inline]
fn raise_lo(lo: &mut [f64], j: usize, cap: f64) -> bool {
    if !cap.is_finite() {
        return false;
    }
    let guarded = cap - EPS * rel(cap);
    if guarded > lo[j] + EPS * rel(lo[j]) {
        lo[j] = guarded;
        true
    } else {
        false
    }
}

/// Propagate `Σ coeffs[k]·x[cols[k]] <= rhs`. `None` = proven infeasible.
fn tighten_le(
    cols: &[usize],
    coeffs: &[f64],
    rhs: f64,
    lo: &mut [f64],
    hi: &mut [f64],
) -> Option<bool> {
    // Minimum activity, rounded DOWN (#1537 D): a lower bound on the exact minimum
    // activity, so neither the infeasibility test nor a residual built from it can
    // exceed what exact arithmetic gives.
    let mut tot = 0.0f64;
    for (k, &c) in coeffs.iter().enumerate() {
        let j = cols[k];
        let m = if c > 0.0 {
            mul_down(c, lo[j])
        } else {
            mul_down(c, hi[j])
        };
        tot = add_down(tot, m);
    }
    if !tot.is_finite() {
        return Some(false); // an unbounded term: no deduction possible, not a proof
    }
    if tot > rhs + INFEAS_TOL * rel(rhs) {
        return None;
    }
    let mut changed = false;
    for (k, &c) in coeffs.iter().enumerate() {
        if c.abs() < 1e-12 {
            continue;
        }
        let j = cols[k];
        // Lower bound on the OTHER terms' min activity: tot_down - (this term, up).
        let mk_up = if c > 0.0 {
            mul_up(c, lo[j])
        } else {
            mul_up(c, hi[j])
        };
        let others = sub_down(tot, mk_up);
        // Upper bound on the residual `rhs - others`, then divide outward.
        let resid = sub_up(rhs, others);
        if c > 0.0 {
            changed |= cap_hi(hi, j, div_up(resid, c));
        } else {
            // c < 0: x_j >= resid / c, which falls as resid grows.
            changed |= raise_lo(lo, j, div_down(resid, c));
        }
    }
    Some(changed)
}

/// Interval enclosure of `cst + Σ coeffs·x[cols]` (directed rounding, #1537 D).
fn form_interval(cols: &[usize], coeffs: &[f64], cst: f64, lo: &[f64], hi: &[f64]) -> (f64, f64) {
    let mut l = cst;
    let mut h = cst;
    for (k, &c) in coeffs.iter().enumerate() {
        let j = cols[k];
        if c >= 0.0 {
            l = add_down(l, mul_down(c, lo[j]));
            h = add_up(h, mul_up(c, hi[j]));
        } else {
            l = add_down(l, mul_down(c, hi[j]));
            h = add_up(h, mul_up(c, lo[j]));
        }
    }
    (l, h)
}

/// Push `form ∈ [tlo, thi]` back onto the form's columns (two `<=` propagations).
fn tighten_form_to(
    cols: &[usize],
    coeffs: &[f64],
    cst: f64,
    tlo: f64,
    thi: f64,
    lo: &mut [f64],
    hi: &mut [f64],
) -> Option<bool> {
    let mut changed = false;
    // Each rhs is rounded OUTWARD (#1537 D): `thi - cst` up, `tlo - cst` down.
    if thi.is_finite() {
        changed |= tighten_le(cols, coeffs, sub_up(thi, cst), lo, hi)?;
    }
    if tlo.is_finite() {
        let neg: Vec<f64> = coeffs.iter().map(|c| -c).collect();
        changed |= tighten_le(cols, &neg, -sub_down(tlo, cst), lo, hi)?;
    }
    Some(changed)
}

/// Reverse-divide `w ∈ [w_lo, w_hi]` by the OTHER factor `G ∈ [g_lo, g_hi]` to get an
/// interval for the target factor `F = w / G`; `None` when no sound deduction exists.
///
/// Covers strict-sign full division plus the one-sided extended cases when `G`
/// touches zero (the entry experiment's load-bearing ingredient): e.g.
/// `G ∈ [0, g_hi]`, `w_lo > 0` ⇒ `G > 0` and `F >= w_lo / g_hi` (no finite upper —
/// `G → 0⁺`); mirrored for the other sign combinations.
///
/// #1537 D: the sign tests are exact (`0.0`), not a `1e-12` band. The band treated
/// `G ∈ [-5e-13, 1]` as `[0, 1]` and deduced `F >= w_lo / g_hi`, cutting the
/// feasible `F = -4e12, G = -3e-13`. Quotients are rounded outward.
fn reverse_div(w_lo: f64, w_hi: f64, g_lo: f64, g_hi: f64) -> Option<(f64, f64)> {
    if g_lo > 0.0 || g_hi < 0.0 {
        // 0 not in [g_lo, g_hi]: full interval division.
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for (a, b) in [(w_lo, g_lo), (w_lo, g_hi), (w_hi, g_lo), (w_hi, g_hi)] {
            let (ql, qh) = (div_down(a, b), div_up(a, b));
            if ql.is_nan() || qh.is_nan() {
                return None;
            }
            lo = lo.min(ql);
            hi = hi.max(qh);
        }
        return Some((lo, hi));
    }
    if g_lo == 0.0 && g_hi > 0.0 {
        // G in [0, g_hi] (touching zero from above).
        if w_lo > 0.0 {
            return Some((div_down(w_lo, g_hi), f64::INFINITY)); // F > 0, F >= w_lo/g_hi
        }
        if w_hi < 0.0 {
            return Some((f64::NEG_INFINITY, div_up(w_hi, g_hi))); // F < 0, F <= w_hi/g_hi
        }
    }
    if g_hi == 0.0 && g_lo < 0.0 {
        // G in [g_lo, 0] (touching zero from below).
        if w_lo > 0.0 {
            return Some((f64::NEG_INFINITY, div_up(w_lo, g_lo))); // F < 0
        }
        if w_hi < 0.0 {
            return Some((div_down(w_hi, g_lo), f64::INFINITY)); // F > 0
        }
    }
    None
}

/// One product `w = A·B` (affine forms): forward + reverse. `None` = infeasible.
#[allow(clippy::too_many_arguments)]
fn propagate_product(
    a_cols: &[usize],
    a_coeffs: &[f64],
    a_const: f64,
    b_cols: &[usize],
    b_coeffs: &[f64],
    b_const: f64,
    w: usize,
    lo: &mut [f64],
    hi: &mut [f64],
) -> Option<bool> {
    let (a_lo, a_hi) = form_interval(a_cols, a_coeffs, a_const, lo, hi);
    let (b_lo, b_hi) = form_interval(b_cols, b_coeffs, b_const, lo, hi);
    let mut changed = false;
    // Forward: w ∈ [A]·[B].
    if a_lo.is_finite() && a_hi.is_finite() && b_lo.is_finite() && b_hi.is_finite() {
        let (mut plo, mut phi) = (f64::INFINITY, f64::NEG_INFINITY);
        for (x, y) in [(a_lo, b_lo), (a_lo, b_hi), (a_hi, b_lo), (a_hi, b_hi)] {
            let (pl, ph) = (mul_down(x, y), mul_up(x, y));
            if !pl.is_nan() && !ph.is_nan() {
                plo = plo.min(pl);
                phi = phi.max(ph);
            }
        }
        if plo <= phi {
            changed |= raise_lo(lo, w, plo);
            changed |= cap_hi(hi, w, phi);
        }
    }
    if lo[w] > hi[w] + INFEAS_TOL * rel(hi[w]) {
        return None;
    }
    // Reverse onto A from w / [B].
    if let Some((qlo, qhi)) = reverse_div(lo[w], hi[w], b_lo, b_hi) {
        changed |= tighten_form_to(a_cols, a_coeffs, a_const, qlo, qhi, lo, hi)?;
    }
    // Reverse onto B from w / [A].
    if let Some((qlo, qhi)) = reverse_div(lo[w], hi[w], a_lo, a_hi) {
        changed |= tighten_form_to(b_cols, b_coeffs, b_const, qlo, qhi, lo, hi)?;
    }
    Some(changed)
}

/// One fixed-width term: forward + guarded reverse. `None` = infeasible.
fn propagate_env_term(t: &EnvTerm, lo: &mut [f64], hi: &mut [f64]) -> Option<bool> {
    let mut changed = false;
    match *t {
        EnvTerm::Bilinear { i, j, w } => {
            changed |= propagate_product(&[i], &[1.0], 0.0, &[j], &[1.0], 0.0, w, lo, hi)?;
        }
        EnvTerm::Monomial { i, s, p } => {
            // Sign-definite boxes only (the engine's registration precondition;
            // branching only shrinks boxes so it is preserved). Straddling: skip.
            // `p >= 2` is the registration contract; anything else is skipped (sound).
            if p >= 2 && (lo[i] >= 0.0 || hi[i] <= 0.0) {
                let pu = p as u64;
                // Forward: monotone on a sign-definite box (matches
                // monomial_aux_bounds), endpoints in directed rounding (#1537 D).
                let (flo, fhi) = if lo[i] >= 0.0 || p % 2 == 1 {
                    // x^p nondecreasing on this box.
                    (powi_down(lo[i], pu), powi_up(hi[i], pu))
                } else {
                    // hi[i] <= 0 and even p: nonincreasing.
                    (powi_down(hi[i], pu), powi_up(lo[i], pu))
                };
                changed |= raise_lo(lo, s, flo);
                changed |= cap_hi(hi, s, fhi);
                if lo[s] > hi[s] + INFEAS_TOL * rel(hi[s]) {
                    return None;
                }
                // Reverse: x = s^(1/p), monotone per regime. Verified directed
                // roots (`root_down`/`root_up`, #1504) replace a bare `powf(1/p)`.
                if p % 2 == 1 {
                    changed |= raise_lo(lo, i, root_down(lo[s], pu));
                    changed |= cap_hi(hi, i, root_up(hi[s], pu));
                } else if lo[i] >= 0.0 {
                    let s_lo = lo[s].max(0.0);
                    if hi[s] < -INFEAS_TOL {
                        return None;
                    }
                    changed |= raise_lo(lo, i, root_down(s_lo, pu));
                    changed |= cap_hi(hi, i, root_up(hi[s].max(0.0), pu));
                } else {
                    // hi[i] <= 0, even p: x in [-s_hi^(1/p), -s_lo^(1/p)].
                    let s_lo = lo[s].max(0.0);
                    if hi[s] < -INFEAS_TOL {
                        return None;
                    }
                    changed |= raise_lo(lo, i, -root_up(hi[s].max(0.0), pu));
                    changed |= cap_hi(hi, i, -root_down(s_lo, pu));
                }
            }
        }
        EnvTerm::AffineSquare { j, w, coeff, cst } => {
            // t = coeff*x + cst; w = t^2.
            let (t_lo, t_hi) = form_interval(&[j], &[coeff], cst, lo, hi);
            // Forward (exact square range).
            let (flo, fhi) = if t_lo >= 0.0 {
                (mul_down(t_lo, t_lo), mul_up(t_hi, t_hi))
            } else if t_hi <= 0.0 {
                (mul_down(t_hi, t_hi), mul_up(t_lo, t_lo))
            } else {
                (0.0, mul_up(t_lo, t_lo).max(mul_up(t_hi, t_hi)))
            };
            changed |= raise_lo(lo, w, flo);
            changed |= cap_hi(hi, w, fhi);
            if hi[w] < -INFEAS_TOL {
                return None;
            }
            if lo[w] > hi[w] + INFEAS_TOL * rel(hi[w]) {
                return None;
            }
            // Reverse: |t| <= sqrt(w_hi); sign-definite t also gets the lower root.
            let r = sqrt_up(hi[w].max(0.0));
            let (mut nlo, mut nhi) = (-r, r);
            let rl = sqrt_down(lo[w].max(0.0));
            if t_lo >= 0.0 {
                nlo = nlo.max(rl);
            } else if t_hi <= 0.0 {
                nhi = nhi.min(-rl);
            }
            changed |= tighten_form_to(&[j], &[coeff], cst, nlo, nhi, lo, hi)?;
        }
        EnvTerm::Sqrt { x, w, coeff, cst } => {
            // arg = coeff*x + cst >= 0; w = sqrt(arg) >= 0.
            let (arg_lo, arg_hi) = form_interval(&[x], &[coeff], cst, lo, hi);
            if arg_hi < -INFEAS_TOL {
                return None;
            }
            let alo = arg_lo.max(0.0);
            let ahi = arg_hi.max(0.0);
            changed |= raise_lo(lo, w, sqrt_down(alo));
            changed |= cap_hi(hi, w, sqrt_up(ahi));
            if lo[w] > hi[w] + INFEAS_TOL * rel(hi[w]) {
                return None;
            }
            // Reverse: arg ∈ [w_lo^2, w_hi^2] (w >= 0, monotone) and arg >= 0.
            let wl = lo[w].max(0.0);
            let wh = hi[w].max(0.0);
            let (al, ah) = (mul_down(wl, wl), mul_up(wh, wh));
            changed |= tighten_form_to(&[x], &[coeff], cst, al, ah, lo, hi)?;
        }
    }
    Some(changed)
}

/// Run the FBBT fixpoint over the spec's structure on the node box `(lo, hi)`
/// (length `n_cols`), with an optional objective cutoff `cᵀx <= cutoff`.
///
/// Returns `false` when the box is **proven empty** under the cutoff — i.e. the
/// region contains no feasible point with objective `<= cutoff` (or no feasible
/// point at all when `cutoff` is `None`); the caller may fathom it with region
/// lower bound `cutoff` (or `+inf`). Returns `true` otherwise, with `(lo, hi)`
/// tightened in place (tighten-only, outward-guarded).
pub fn propagate_spec_fixpoint(
    spec: &SpatialKernelSpec,
    lo: &mut [f64],
    hi: &mut [f64],
    cutoff: Option<f64>,
    max_rounds: usize,
) -> bool {
    // Objective-cutoff row support, built once. Every nonzero term is kept (#1955):
    // dropping a tiny `c_j` without accounting for `c_j·x_j` can make the row prove a
    // non-empty node empty when `|x_j|` is large. `tighten_le` already counts tiny
    // terms in the activity and only skips dividing by them.
    let (obj_cols, obj_coeffs): (Vec<usize>, Vec<f64>) = spec
        .c
        .iter()
        .enumerate()
        .filter(|(_, &c)| c != 0.0)
        .map(|(j, &c)| (j, c))
        .unzip();

    for _ in 0..max_rounds {
        let mut changed = false;
        // 1. cutoff row.
        if let Some(cut) = cutoff {
            match tighten_le(&obj_cols, &obj_coeffs, cut, lo, hi) {
                None => return false,
                Some(c) => changed |= c,
            }
        }
        // 2. fixed linear rows.
        for fr in &spec.fixed_rows {
            match tighten_le(&fr.cols, &fr.coeffs, fr.rhs, lo, hi) {
                None => return false,
                Some(c) => changed |= c,
            }
        }
        // 3. affine-form products.
        for t in &spec.blf_terms {
            let BlfTerm {
                a_cols,
                a_coeffs,
                a_const,
                b_cols,
                b_coeffs,
                b_const,
                w,
            } = t;
            match propagate_product(
                a_cols, a_coeffs, *a_const, b_cols, b_coeffs, *b_const, *w, lo, hi,
            ) {
                None => return false,
                Some(c) => changed |= c,
            }
        }
        // 4. fixed-width terms.
        for t in &spec.terms {
            match propagate_env_term(t, lo, hi) {
                None => return false,
                Some(c) => changed |= c,
            }
        }
        // 5. integer rounding.
        for (j, &is_int) in spec.integrality.iter().enumerate() {
            if !is_int {
                continue;
            }
            let nl = (lo[j] - 1e-6).ceil();
            let nh = (hi[j] + 1e-6).floor();
            if nl > lo[j] + EPS {
                lo[j] = nl;
                changed = true;
            }
            if nh < hi[j] - EPS {
                hi[j] = nh;
                changed = true;
            }
            if lo[j] > hi[j] + 1e-9 {
                return false;
            }
        }
        // Round-level empty-box scan.
        for j in 0..spec.n_cols {
            if lo[j] > hi[j] + INFEAS_TOL * rel(hi[j]) {
                return false;
            }
        }
        if !changed {
            break;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bnb::spatial_kernel::FixedRow;

    /// #1955: c = [1, −2^-40], x ∈ [1, 2], y fixed at 2^41, cutoff 0. cᵀx = x − 2 <= 0
    /// on the whole box, so the region is not empty. Filtering `|c| > 1e-12` dropped
    /// the −2 term and turned the row into x <= 0, which "proved" the box empty.
    #[test]
    fn cutoff_row_keeps_tiny_nonzero_term_1955() {
        let y = 2f64.powi(41);
        let spec = SpatialKernelSpec {
            n_cols: 2,
            n_orig: 2,
            c: vec![1.0, -(2f64.powi(-40))],
            integrality: vec![false; 2],
            global_lo: vec![1.0, y],
            global_hi: vec![2.0, y],
            fixed_rows: vec![],
            terms: vec![],
            blf_terms: vec![],
            obbt_candidates: vec![],
        };
        let (mut lo, mut hi) = (vec![1.0, y], vec![2.0, y]);
        assert!(propagate_spec_fixpoint(
            &spec,
            &mut lo,
            &mut hi,
            Some(0.0),
            10
        ));
        assert!(
            lo[0] <= 1.0 && hi[0] >= 2.0,
            "cutoff row cut x: [{}, {}]",
            lo[0],
            hi[0]
        );
    }

    fn empty_spec(n_cols: usize) -> SpatialKernelSpec {
        SpatialKernelSpec {
            n_cols,
            n_orig: n_cols,
            c: vec![0.0; n_cols],
            integrality: vec![false; n_cols],
            global_lo: vec![0.0; n_cols],
            global_hi: vec![10.0; n_cols],
            fixed_rows: vec![],
            terms: vec![],
            blf_terms: vec![],
            obbt_candidates: vec![],
        }
    }

    #[test]
    fn linear_row_tightens_activity() {
        // x + y <= 3, x >= 2  =>  y <= 1.
        let mut spec = empty_spec(2);
        spec.fixed_rows = vec![FixedRow {
            cols: vec![0, 1],
            coeffs: vec![1.0, 1.0],
            rhs: 3.0,
        }];
        let mut lo = vec![2.0, 0.0];
        let mut hi = vec![10.0, 10.0];
        assert!(propagate_spec_fixpoint(&spec, &mut lo, &mut hi, None, 10));
        assert!(hi[1] <= 1.0 + 1e-6, "y hi {} not tightened to 1", hi[1]);
        assert!(hi[0] <= 3.0 + 1e-6, "x hi {} not tightened to 3", hi[0]);
    }

    #[test]
    fn product_forward_and_strict_reverse() {
        // w = x*y, x in [1,2], y in [1,3], w capped at 2  =>  y <= 2.
        let mut spec = empty_spec(3);
        spec.blf_terms = vec![BlfTerm {
            a_cols: vec![0],
            a_coeffs: vec![1.0],
            a_const: 0.0,
            b_cols: vec![1],
            b_coeffs: vec![1.0],
            b_const: 0.0,
            w: 2,
        }];
        let mut lo = vec![1.0, 1.0, 0.0];
        let mut hi = vec![2.0, 3.0, 2.0];
        assert!(propagate_spec_fixpoint(&spec, &mut lo, &mut hi, None, 10));
        assert!(lo[2] >= 1.0 - 1e-6, "w lo {} (forward)", lo[2]);
        assert!(hi[1] <= 2.0 + 1e-6, "y hi {} (reverse w/x)", hi[1]);
    }

    /// The entry experiment's load-bearing case: factors touching zero block the
    /// strict-sign reverse; the extended one-sided division still deduces.
    #[test]
    fn product_extended_zero_touching_reverse() {
        // w = x*y, x in [0,2], y in [0,3], w >= 1  =>  x >= 1/3, y >= 1/2.
        let mut spec = empty_spec(3);
        spec.blf_terms = vec![BlfTerm {
            a_cols: vec![0],
            a_coeffs: vec![1.0],
            a_const: 0.0,
            b_cols: vec![1],
            b_coeffs: vec![1.0],
            b_const: 0.0,
            w: 2,
        }];
        let mut lo = vec![0.0, 0.0, 1.0];
        let mut hi = vec![2.0, 3.0, 6.0];
        assert!(propagate_spec_fixpoint(&spec, &mut lo, &mut hi, None, 10));
        assert!(
            lo[0] >= 1.0 / 3.0 - 1e-6,
            "x lo {} (extended reverse)",
            lo[0]
        );
        assert!(lo[1] >= 0.5 - 1e-6, "y lo {} (extended reverse)", lo[1]);
    }

    #[test]
    fn sqrt_forward_and_reverse() {
        // w = sqrt(x), x in [1,9], w capped at 2  =>  x <= 4; and w in [1,3] forward.
        let mut spec = empty_spec(2);
        spec.terms = vec![EnvTerm::Sqrt {
            x: 0,
            w: 1,
            coeff: 1.0,
            cst: 0.0,
        }];
        let mut lo = vec![1.0, 0.0];
        let mut hi = vec![9.0, 2.0];
        assert!(propagate_spec_fixpoint(&spec, &mut lo, &mut hi, None, 10));
        assert!(lo[1] >= 1.0 - 1e-6, "w lo {} (forward)", lo[1]);
        assert!(hi[0] <= 4.0 + 1e-5, "x hi {} (reverse)", hi[0]);
    }

    #[test]
    fn cutoff_proves_region_empty() {
        // min x with cutoff 1, but x >= 2 (via -x <= -2): no point with x <= 1.
        let mut spec = empty_spec(1);
        spec.c = vec![1.0];
        spec.fixed_rows = vec![FixedRow {
            cols: vec![0],
            coeffs: vec![-1.0],
            rhs: -2.0,
        }];
        let mut lo = vec![0.0];
        let mut hi = vec![10.0];
        assert!(!propagate_spec_fixpoint(
            &spec,
            &mut lo,
            &mut hi,
            Some(1.0),
            10
        ));
    }

    /// The tanksize chain in miniature: cutoff tightens the objective variable, a
    /// linear row ties it to a product output, reverse division tightens the
    /// operands — the coupled deduction no single step makes alone.
    #[test]
    fn cutoff_chains_through_linear_row_into_product() {
        // cols: x0 (obj), x1, x2, w=x1*x2 (col 3). Rows: w - x0 <= 0 (w <= x0).
        // cutoff x0 <= 2 => w <= 2; x1 in [1,4], x2 in [1,4] => forward w >= 1;
        // reverse: x1 <= 2/1 = 2, x2 <= 2.
        let mut spec = empty_spec(4);
        spec.c = vec![1.0, 0.0, 0.0, 0.0];
        spec.fixed_rows = vec![FixedRow {
            cols: vec![3, 0],
            coeffs: vec![1.0, -1.0],
            rhs: 0.0,
        }];
        spec.blf_terms = vec![BlfTerm {
            a_cols: vec![1],
            a_coeffs: vec![1.0],
            a_const: 0.0,
            b_cols: vec![2],
            b_coeffs: vec![1.0],
            b_const: 0.0,
            w: 3,
        }];
        let mut lo = vec![0.0, 1.0, 1.0, 0.0];
        let mut hi = vec![10.0, 4.0, 4.0, 100.0];
        assert!(propagate_spec_fixpoint(
            &spec,
            &mut lo,
            &mut hi,
            Some(2.0),
            15
        ));
        assert!(hi[0] <= 2.0 + 1e-6, "obj hi {}", hi[0]);
        assert!(hi[3] <= 2.0 + 1e-5, "w hi {} (via linear row)", hi[3]);
        assert!(
            hi[1] <= 2.0 + 1e-4,
            "x1 hi {} (via reverse division)",
            hi[1]
        );
        assert!(
            hi[2] <= 2.0 + 1e-4,
            "x2 hi {} (via reverse division)",
            hi[2]
        );
    }

    /// Tighten-only + outward guard: propagation never widens and never crosses.
    #[test]
    fn tighten_only_and_never_widens() {
        let mut spec = empty_spec(3);
        spec.blf_terms = vec![BlfTerm {
            a_cols: vec![0],
            a_coeffs: vec![1.0],
            a_const: 0.0,
            b_cols: vec![1],
            b_coeffs: vec![1.0],
            b_const: 0.0,
            w: 2,
        }];
        let lo0 = vec![0.5, 0.5, 0.0];
        let hi0 = vec![1.5, 1.5, 5.0];
        let mut lo = lo0.clone();
        let mut hi = hi0.clone();
        assert!(propagate_spec_fixpoint(&spec, &mut lo, &mut hi, None, 10));
        for j in 0..3 {
            assert!(lo[j] >= lo0[j] - 1e-9, "lo[{j}] widened");
            assert!(hi[j] <= hi0[j] + 1e-9, "hi[{j}] widened");
            assert!(lo[j] <= hi[j] + 1e-9, "crossed at {j}");
        }
        // The true point (1, 1, 1) must survive (w = x*y feasible).
        assert!(lo[0] <= 1.0 && hi[0] >= 1.0);
        assert!(lo[2] <= 1.0 + 1e-9 && hi[2] >= 1.0 - 1e-9);
    }

    /// #1537 D witness (exact-arithmetic audit, `probe_sp.py`, 3,125 of 200,000
    /// random rows cut a feasible point). The row's min activity is a sum of
    /// ~1e14-magnitude products; computing it in round-to-nearest and then taking
    /// `tot - mk` loses up to ~10 units, far past the `EPS * (1 + |cap|)` guard. The
    /// point below is feasible in exact rational arithmetic over the float data
    /// (all other columns at their min-activity bound, `x1 = 3415692969.048031…`,
    /// which makes the row hold with equality) yet the round-to-nearest code raised
    /// `lo[1]` to 3415692979.69 -- above `hi[1]`, so the box was declared EMPTY.
    #[test]
    fn issue_1537_linear_row_cancellation_never_cuts_a_feasible_point() {
        let mut spec = empty_spec(5);
        spec.fixed_rows = vec![FixedRow {
            cols: vec![0, 1, 2, 3, 4],
            coeffs: vec![
                0.01135996348320167,
                -0.0022948358799420505,
                -341.5045393785568,
                -0.029260775192544688,
                0.17995108280856198,
            ],
            rhs: -198764520876410.22,
        }];
        let lo0 = vec![
            132255245420.08032,
            3415688466.6468506,
            582286247020.8639,
            -305079229649.1662,
            435975525363.2826,
        ];
        let hi0 = vec![
            132255245500.33266,
            3415692969.404494,
            582286247020.8845,
            -305079229649.02423,
            435975525404.30225,
        ];
        let mut lo = lo0.clone();
        let mut hi = hi0.clone();
        // Exact x1 of the feasible point is 3415692969.048031...; this float is
        // below it, so a valid box must keep lo[1] <= it.
        let x1_feasible_floor = 3415692969.04803_f64;
        assert!(
            propagate_spec_fixpoint(&spec, &mut lo, &mut hi, None, 10),
            "box with an exactly feasible point declared empty (lo1={}, hi1={})",
            lo[1],
            hi[1]
        );
        assert!(
            lo[1] <= x1_feasible_floor,
            "lo[1] = {} cuts the feasible x1 = 3415692969.048031",
            lo[1]
        );
    }

    /// #1537 D witness: `reverse_div` treated a factor in `[-5e-13, 1]` as if it
    /// were `[0, 1]` (a hard-coded 1e-12 "touching zero" band), deducing `x >= 1`
    /// from `w = x*y >= 1`. The point `x = -4e12, y = -3e-13, w = x*y ~ 1.2` is
    /// feasible and was cut.
    #[test]
    fn issue_1537_reverse_division_respects_a_slightly_negative_factor() {
        let mut spec = empty_spec(3);
        spec.blf_terms = vec![BlfTerm {
            a_cols: vec![0],
            a_coeffs: vec![1.0],
            a_const: 0.0,
            b_cols: vec![1],
            b_coeffs: vec![1.0],
            b_const: 0.0,
            w: 2,
        }];
        let (x, y) = (-4e12_f64, -3e-13_f64);
        let w = x * y; // ~1.2, inside [1, 2]
        let mut lo = vec![-1e13, -5e-13, 1.0];
        let mut hi = vec![10.0, 1.0, 2.0];
        assert!(propagate_spec_fixpoint(&spec, &mut lo, &mut hi, None, 10));
        for (j, v) in [x, y, w].into_iter().enumerate() {
            assert!(
                lo[j] <= v && v <= hi[j],
                "feasible point coordinate {j} = {v} cut by [{}, {}]",
                lo[j],
                hi[j]
            );
        }
    }

    #[test]
    fn integer_rounding_applies() {
        let mut spec = empty_spec(1);
        spec.integrality = vec![true];
        let mut lo = vec![0.3];
        let mut hi = vec![2.7];
        assert!(propagate_spec_fixpoint(&spec, &mut lo, &mut hi, None, 5));
        assert!((lo[0] - 1.0).abs() < 1e-9 && (hi[0] - 2.0).abs() < 1e-9);
    }
}
