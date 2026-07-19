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
//! Soundness: every tightening step is a valid interval deduction, applied with a
//! small **outward relaxation** (`EPS`-scaled) so f64 roundoff can never cut a true
//! feasible point; infeasibility is declared only when a violation exceeds a
//! conservative tolerance. Skipping any step is always sound (the box just stays
//! looser), so all guarded cases degrade gracefully.

use crate::bnb::spatial_kernel::{BlfTerm, EnvTerm, SpatialKernelSpec};

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
    // Minimum activity.
    let mut tot = 0.0f64;
    for (k, &c) in coeffs.iter().enumerate() {
        let j = cols[k];
        tot += if c > 0.0 { c * lo[j] } else { c * hi[j] };
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
        let mk = if c > 0.0 { c * lo[j] } else { c * hi[j] };
        let cap = (rhs - (tot - mk)) / c;
        if c > 0.0 {
            changed |= cap_hi(hi, j, cap);
        } else {
            changed |= raise_lo(lo, j, cap);
        }
    }
    Some(changed)
}

/// Interval enclosure of `cst + Σ coeffs·x[cols]`.
fn form_interval(cols: &[usize], coeffs: &[f64], cst: f64, lo: &[f64], hi: &[f64]) -> (f64, f64) {
    let mut l = cst;
    let mut h = cst;
    for (k, &c) in coeffs.iter().enumerate() {
        let j = cols[k];
        if c >= 0.0 {
            l += c * lo[j];
            h += c * hi[j];
        } else {
            l += c * hi[j];
            h += c * lo[j];
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
    if thi.is_finite() {
        changed |= tighten_le(cols, coeffs, thi - cst, lo, hi)?;
    }
    if tlo.is_finite() {
        let neg: Vec<f64> = coeffs.iter().map(|c| -c).collect();
        changed |= tighten_le(cols, &neg, -(tlo - cst), lo, hi)?;
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
fn reverse_div(w_lo: f64, w_hi: f64, g_lo: f64, g_hi: f64) -> Option<(f64, f64)> {
    let e = 1e-12;
    if g_lo > e || g_hi < -e {
        // 0 not in [g_lo, g_hi]: full interval division.
        let qs = [w_lo / g_lo, w_lo / g_hi, w_hi / g_lo, w_hi / g_hi];
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for q in qs {
            if q.is_nan() {
                return None;
            }
            lo = lo.min(q);
            hi = hi.max(q);
        }
        return Some((lo, hi));
    }
    if g_lo >= -e && g_hi > e {
        // G in [0, g_hi] (touching zero from above).
        if w_lo > e {
            return Some((w_lo / g_hi, f64::INFINITY)); // F > 0, F >= w_lo/g_hi
        }
        if w_hi < -e {
            return Some((f64::NEG_INFINITY, w_hi / g_hi)); // F < 0, F <= w_hi/g_hi
        }
    }
    if g_hi <= e && g_lo < -e {
        // G in [g_lo, 0] (touching zero from below).
        if w_lo > e {
            return Some((f64::NEG_INFINITY, w_lo / g_lo)); // F < 0
        }
        if w_hi < -e {
            return Some((w_hi / g_lo, f64::INFINITY)); // F > 0
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
        let ps = [a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi];
        let (mut plo, mut phi) = (f64::INFINITY, f64::NEG_INFINITY);
        for p in ps {
            if !p.is_nan() {
                plo = plo.min(p);
                phi = phi.max(p);
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

/// Signed real p-th root (odd p handles negatives; even p requires `t >= 0`).
fn signed_root(t: f64, p: i32) -> f64 {
    if p % 2 == 1 {
        t.signum() * t.abs().powf(1.0 / p as f64)
    } else if t >= 0.0 {
        t.powf(1.0 / p as f64)
    } else {
        f64::NAN
    }
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
            if lo[i] >= 0.0 || hi[i] <= 0.0 {
                // Forward: monotone on a sign-definite box (matches monomial_aux_bounds).
                let a = lo[i].powi(p);
                let b = hi[i].powi(p);
                let (flo, fhi) = if a <= b { (a, b) } else { (b, a) };
                changed |= raise_lo(lo, s, flo);
                changed |= cap_hi(hi, s, fhi);
                if lo[s] > hi[s] + INFEAS_TOL * rel(hi[s]) {
                    return None;
                }
                // Reverse: x = s^(1/p), monotone per regime.
                if p % 2 == 1 {
                    let rlo = signed_root(lo[s], p);
                    let rhi = signed_root(hi[s], p);
                    changed |= raise_lo(lo, i, rlo);
                    changed |= cap_hi(hi, i, rhi);
                } else if lo[i] >= 0.0 {
                    let s_lo = lo[s].max(0.0);
                    if hi[s] < -INFEAS_TOL {
                        return None;
                    }
                    changed |= raise_lo(lo, i, s_lo.powf(1.0 / p as f64));
                    changed |= cap_hi(hi, i, hi[s].max(0.0).powf(1.0 / p as f64));
                } else {
                    // hi[i] <= 0, even p: x in [-s_hi^(1/p), -s_lo^(1/p)].
                    let s_lo = lo[s].max(0.0);
                    if hi[s] < -INFEAS_TOL {
                        return None;
                    }
                    changed |= raise_lo(lo, i, -(hi[s].max(0.0).powf(1.0 / p as f64)));
                    changed |= cap_hi(hi, i, -(s_lo.powf(1.0 / p as f64)));
                }
            }
        }
        EnvTerm::AffineSquare { j, w, coeff, cst } => {
            // t = coeff*x + cst; w = t^2.
            let (t_lo, t_hi) = form_interval(&[j], &[coeff], cst, lo, hi);
            // Forward (exact square range).
            let (flo, fhi) = if t_lo >= 0.0 {
                (t_lo * t_lo, t_hi * t_hi)
            } else if t_hi <= 0.0 {
                (t_hi * t_hi, t_lo * t_lo)
            } else {
                (0.0, (t_lo * t_lo).max(t_hi * t_hi))
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
            let r = hi[w].max(0.0).sqrt();
            let (mut nlo, mut nhi) = (-r, r);
            let rl = lo[w].max(0.0).sqrt();
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
            changed |= raise_lo(lo, w, alo.sqrt());
            changed |= cap_hi(hi, w, ahi.sqrt());
            if lo[w] > hi[w] + INFEAS_TOL * rel(hi[w]) {
                return None;
            }
            // Reverse: arg ∈ [w_lo^2, w_hi^2] (w >= 0, monotone) and arg >= 0.
            let wl = lo[w].max(0.0);
            let wh = hi[w].max(0.0);
            changed |= tighten_form_to(&[x], &[coeff], cst, wl * wl, wh * wh, lo, hi)?;
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
    // Objective-cutoff row support (nonzero coefficients only), built once.
    let (obj_cols, obj_coeffs): (Vec<usize>, Vec<f64>) = spec
        .c
        .iter()
        .enumerate()
        .filter(|(_, &c)| c.abs() > 1e-12)
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
        assert!(lo[0] >= 1.0 / 3.0 - 1e-6, "x lo {} (extended reverse)", lo[0]);
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
        assert!(!propagate_spec_fixpoint(&spec, &mut lo, &mut hi, Some(1.0), 10));
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
        assert!(propagate_spec_fixpoint(&spec, &mut lo, &mut hi, Some(2.0), 15));
        assert!(hi[0] <= 2.0 + 1e-6, "obj hi {}", hi[0]);
        assert!(hi[3] <= 2.0 + 1e-5, "w hi {} (via linear row)", hi[3]);
        assert!(hi[1] <= 2.0 + 1e-4, "x1 hi {} (via reverse division)", hi[1]);
        assert!(hi[2] <= 2.0 + 1e-4, "x2 hi {} (via reverse division)", hi[2]);
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
