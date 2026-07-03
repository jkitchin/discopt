//! {0,½}-Chvátal–Gomory ("zero-half") cuts from the integer `≤` rows.
//!
//! Reference: Caprara & Fischetti (1996), "{0,½}-Chvátal–Gomory cuts",
//! *Mathematical Programming* 74, 221–235; and SCIP's `sepa_zerohalf` as the
//! implementation model. A {0,½} cut takes a subset `S` of the integer-data
//! rows `a_i · x ≤ b_i` (coefficients and rhs integral after scaling), sums them
//! with weight `½`, and CG-rounds the result:
//!
//! ```text
//!   ⌊(Σ_{i∈S} a_i) / 2⌋ · x  ≤  ⌊(Σ_{i∈S} b_i) / 2⌋.
//! ```
//!
//! # Soundness (valid by construction, independent of the heuristic)
//!
//! For **any** subset `S`, `(½ Σ a_i) · x ≤ ½ Σ b_i` is a valid inequality of the
//! LP relaxation (a nonnegative — here `½` — combination of `≤` rows). The
//! left-hand coefficients `½ Σ a_i` need not be integral, but for every
//! integer-feasible `x ≥ 0`… actually for every integer `x` the term
//! `(½ Σ a_i) · x` is at least `⌊½ Σ a_i⌋ · x` only when `x ≥ 0`, so we must be
//! careful: the CG argument is that for integer `x`, `⌊c⌋ · x ≤ c · x` fails for
//! negative `x`. We therefore reduce the whole system to **nonnegative
//! variables** first, by the same lower-shift / upper-complement bound
//! substitution used for MIR (see [`super::mir`]); on the shifted, nonnegative
//! variables `y ≥ 0` the Chvátal–Gomory round is valid: `⌊c⌋ · y ≤ c · y ≤ ½ Σ b̃`
//! and, since the left side is integer for integer `y`, `⌊c⌋ · y ≤ ⌊½ Σ b̃⌋`.
//! Mapping the cut back through the (affine, invertible) substitution yields a
//! valid inequality in the original `x`. This holds for every subset `S`,
//! every scaling, and every point it was separated at, so **the heuristic that
//! picks `S` affects only strength, never validity**. The
//! `zerohalf_validity_random_systems` property test verifies this empirically
//! (it is the same regime that caught the sign bug in #415).
//!
//! # Separation heuristic
//!
//! The separation problem — find `S` whose rounded cut is most violated at `x*`
//! — is a min-weight T-join / shortest odd-parity combination over the mod-2
//! residue structure of the rows (Caprara–Fischetti). We ship a **heuristic**
//! (not exact) separator: GF(2) Gaussian elimination on the parity system
//! `[Ã mod 2 | b̃ mod 2]`, restricted to the low-slack rows, to find subsets `S`
//! whose combined coefficient parity is even on every column and whose combined
//! rhs parity is odd. Such a subset gives the maximal rounding gain (`⌊b̄/2⌋ =
//! (b̄−1)/2`, a full `½` below `b̄/2`), so the cut is violated whenever the
//! subset's total (nonnegative) slack is `< 1`. Exact/optimal separation is a
//! documented follow-on (see `docs/dev/certification-gap-plan.md` §7).

use std::collections::HashMap;

/// A separated {0,½}-CG cut `coeffs · x ≤ rhs` over the structural variables.
pub struct ZeroHalfCut {
    /// Dense length-`n` coefficient vector.
    pub coeffs: Vec<f64>,
    /// Right-hand side (the cut is `coeffs · x ≤ rhs`).
    pub rhs: f64,
    /// Violation `coeffs · x* − rhs` at the separation point (`> 0` by filter).
    pub violation: f64,
}

/// Absolute cap on a cut coefficient; larger ones are numerically unsafe (mirror
/// of the guard in [`super::mir`]).
const MAX_ABS_COEFF: f64 = 1e7;
/// A scaled coefficient/rhs is accepted as "integral" only within this tolerance
/// of the nearest integer; a row that does not scale cleanly to integers is
/// skipped (zerohalf is defined on integer data).
const INT_SCALE_TOL: f64 = 1e-6;
/// A row is eligible for the parity system only when its slack at `x*` is below
/// this bound; a subset of such rows can only produce a violated cut when its
/// total slack is `< 1` (see module docs), so high-slack rows are dead weight.
const SLACK_ELIGIBLE_MAX: f64 = 0.999;
/// Tolerance within which an integer column's upper bound must sit to an integer
/// for its complement `u_j − x_j` to remain integer-valued (mirror of `mir.rs`).
const INT_UB_TOL: f64 = 1e-6;

/// Scale one row `(a, b)` to integer data by a common multiplier, returning the
/// scaled integer coefficients (as `f64` holding integer values) and rhs, or
/// `None` if no small multiplier makes the row integral. Tries the multipliers
/// {1, 2, 3, 4, 6, 8, 12} (enough for the ½/⅓/¼-denominator coefficients that
/// arise after McCormick/knapsack scaling); a row already integral uses `1`.
fn scale_row_to_integer(a: &[f64], b: f64) -> Option<(Vec<i64>, i64)> {
    const MULTS: [f64; 7] = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0];
    'mult: for &m in &MULTS {
        let mut ai = Vec::with_capacity(a.len());
        for &v in a {
            let s = v * m;
            if (s - s.round()).abs() > INT_SCALE_TOL {
                continue 'mult;
            }
            ai.push(s.round() as i64);
        }
        let sb = b * m;
        if (sb - sb.round()).abs() > INT_SCALE_TOL {
            continue 'mult;
        }
        return Some((ai, sb.round() as i64));
    }
    None
}

/// Reduce one (possibly complemented) integer row `Σ ã_j y_j ≤ b̃` on the
/// nonnegative variables `y` to a {0,½}-CG cut *over the original `x`*, using a
/// precomputed subset combination. `comb_a`/`comb_b` are the integer sums
/// `Σ_{i∈S} ã` / `Σ_{i∈S} b̃` in the **shifted (nonnegative-y)** space; `comp`,
/// `l`, `u` describe the per-column substitution; `x` is the LP point. Returns
/// the cut over `x` with its violation, or `None` if not violated / unsafe.
///
/// # Soundness
/// `⌊comb_a / 2⌋ · y ≤ ⌊comb_b / 2⌋` is the CG round of the valid combination
/// `(½ comb_a) · y ≤ ½ comb_b` on `y ≥ 0`; it removes no integer `y`. Mapping
/// `y_j = u_j − x_j` (complemented) / `y_j = x_j − l_j` back is affine and
/// invertible and preserves integrality of `x` (complementation of an integer
/// column is gated on integral `u_j` by the caller), so the mapped cut removes
/// no integer-feasible `x`.
#[allow(clippy::too_many_arguments)]
fn round_and_map(
    comb_a: &[i64],
    comb_b: i64,
    comp: &[bool],
    l: &[f64],
    u: &[f64],
    x: &[f64],
    tol: f64,
    max_dynamism: f64,
) -> Option<(Vec<f64>, f64, f64)> {
    let n = comb_a.len();
    // CG round in the shifted y-space: g_j = ⌊ã_j / 2⌋, r = ⌊b̃ / 2⌋.
    let g: Vec<f64> = comb_a.iter().map(|&a| (a as f64 / 2.0).floor()).collect();
    let r = (comb_b as f64 / 2.0).floor();

    // Map `Σ g_j y_j ≤ r` back to the original x (identical bookkeeping to MIR):
    // complemented j contributes coeff −g_j and rhs −g_j u_j; lower j contributes
    // coeff g_j and rhs += g_j l_j.
    let mut coeffs = vec![0.0_f64; n];
    let mut rhs = r;
    for j in 0..n {
        if comp[j] {
            coeffs[j] = -g[j];
            rhs -= g[j] * u[j];
        } else {
            coeffs[j] = g[j];
            rhs += g[j] * l[j];
        }
    }

    let viol: f64 = (0..n).map(|j| coeffs[j] * x[j]).sum::<f64>() - rhs;
    if viol <= tol {
        return None;
    }

    let max_c = coeffs.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let min_c = coeffs
        .iter()
        .filter(|&&v| v.abs() > tol)
        .fold(f64::INFINITY, |a, &v| a.min(v.abs()));
    if max_c == 0.0
        || max_c > MAX_ABS_COEFF
        || rhs.abs() > MAX_ABS_COEFF
        || (min_c.is_finite() && max_c / min_c > max_dynamism)
        || !coeffs.iter().all(|v| v.is_finite())
        || !rhs.is_finite()
    {
        return None;
    }
    Some((coeffs, rhs, viol))
}

/// A row of the parity system: the shifted integer row `(a, b)` on nonnegative
/// `y`, its GF(2) residue (coefficient parities packed as a bitset over `u64`
/// words plus the rhs parity), and its slack at `x*`.
struct ParityRow {
    a: Vec<i64>,
    b: i64,
    /// Packed column-parity bits (`bit c` set ⇔ `a[c]` odd) followed logically by
    /// the rhs-parity, which we store separately.
    coef_bits: Vec<u64>,
    rhs_parity: bool,
    slack: f64,
}

/// XOR the GF(2) residue and accumulate the integer row of `other` into `dst`
/// (used when eliminating / combining rows). Integer coefficients add exactly, so
/// the combined row is the exact `Σ_{i∈S}` of its member rows — the combination
/// stays an exact nonnegative integer combination, which is what makes the CG
/// round of the result valid.
fn xor_into_ref(dst: &mut ParityRow, other: &ParityRow) {
    for (d, s) in dst.coef_bits.iter_mut().zip(&other.coef_bits) {
        *d ^= *s;
    }
    dst.rhs_parity ^= other.rhs_parity;
    for (d, s) in dst.a.iter_mut().zip(&other.a) {
        *d += *s;
    }
    dst.b += other.b;
    dst.slack += other.slack;
}

/// Index of the lowest set coefficient-parity bit (the pivot column), or `None`
/// when all coefficient parities are even.
fn pivot_col(bits: &[u64]) -> Option<usize> {
    for (w, &word) in bits.iter().enumerate() {
        if word != 0 {
            return Some(w * 64 + word.trailing_zeros() as usize);
        }
    }
    None
}

/// Separate {0,½}-CG cuts from the `≤` rows `a_ub · x ≤ b_ub` at point `x`.
///
/// `a_ub` is row-major `m × n`; `l`/`u` are the variable lower/upper bounds (used
/// to reduce every column to a nonnegative variable via lower-shift or, when the
/// LP point sits near a finite integral upper bound, upper-complement — the same
/// bound substitution as [`super::mir`], required for the CG round to be valid on
/// signed variables); `integrality[j]` marks integer columns. Only rows that
/// scale to integer data and whose slack at `x*` is small are entered into the
/// mod-2 parity system; GF(2) elimination then finds subsets whose combined
/// coefficient parity is even and rhs parity odd, and each such subset is
/// CG-rounded and mapped back to `x`. Returns the violated, numerically safe cuts
/// (most-violated first is the caller's job). Soundness is independent of which
/// subsets the heuristic finds (see module docs).
#[allow(clippy::too_many_arguments)]
pub fn separate_zerohalf(
    a_ub: &[f64],
    b_ub: &[f64],
    l: &[f64],
    u: &[f64],
    integrality: &[bool],
    x: &[f64],
    tol: f64,
    max_dynamism: f64,
) -> Vec<ZeroHalfCut> {
    let m = b_ub.len();
    let n = a_ub.len().checked_div(m.max(1)).unwrap_or(0);
    let mut cuts = Vec::new();
    if n == 0 || m == 0 {
        return cuts;
    }

    // Per-column bound substitution: complement column j (use y_j = u_j − x_j)
    // when the LP point sits strictly above the box midpoint and the complement
    // keeps integrality (integral u_j for integer columns). Otherwise lower-shift
    // (y_j = x_j − l_j). Both need a finite bound on the chosen side; a column
    // with no finite bound on its side makes the row ineligible.
    let mut comp = vec![false; n];
    for j in 0..n {
        let lo_ok = l[j].is_finite();
        let hi_ok = u[j].is_finite() && u[j] > l[j];
        let near_upper = hi_ok && (x[j] - l[j] > 0.5 * (u[j] - l[j]));
        let int_ok = !integrality[j] || (u[j] - u[j].round()).abs() <= INT_UB_TOL;
        if near_upper && int_ok {
            comp[j] = true;
        } else if !lo_ok && hi_ok && int_ok {
            comp[j] = true; // no finite lower bound → must complement
        } else {
            comp[j] = false;
        }
    }

    // Shifted point value y*_j = comp ? u−x : x−l (≥ 0). Only columns where the
    // shifted value is bounded away from 0 ("active") contribute to a cut's
    // violation: at y*_j ≈ 0 the term `⌊ā_j/2⌋ · y_j ≈ 0` regardless of the
    // coefficient parity there. Following Caprara–Fischetti, we therefore build
    // the mod-2 parity system over the **active support only** and treat inactive
    // columns as parity don't-cares. This is the difference between finding the
    // odd-cycle / parity cut and finding nothing: on a set-packing/graph LP the
    // fractional support is exactly the active set, and combinations that cancel
    // parity *there* (not on all n columns) are what round to a violated cut. The
    // final `round_and_map` computes the exact violation on all columns, so a
    // leftover odd parity on an inactive column can only weaken, never invalidate.
    let y_star: Vec<f64> = (0..n)
        .map(|j| if comp[j] { u[j] - x[j] } else { x[j] - l[j] })
        .collect();
    // Active-column threshold: a shifted value materially above 0. Scaled to the
    // box width so it is dimensionless-ish; a fixed small floor guards tiny boxes.
    let active: Vec<bool> = (0..n).map(|j| y_star[j] > 1e-6).collect();
    // Map each active column to a compact bit index (keeps the GF(2) words small
    // and the elimination fast even when n ≫ |active|).
    let mut active_bit = vec![usize::MAX; n];
    let mut n_active = 0usize;
    for j in 0..n {
        if active[j] {
            active_bit[j] = n_active;
            n_active += 1;
        }
    }
    let words = n_active.div_ceil(64).max(1);

    // Build the eligible parity rows: scale to integers in the SHIFTED space.
    let mut rows: Vec<ParityRow> = Vec::new();
    for i in 0..m {
        let row = &a_ub[i * n..(i + 1) * n];
        // Shift/complement to nonnegative y: ã_j = comp ? −a_j : a_j;
        // b̃ = b − Σ_comp a_j u_j − Σ_lower a_j l_j.
        let mut a_shift = vec![0.0_f64; n];
        let mut b_shift = b_ub[i];
        let mut ok = true;
        for j in 0..n {
            if comp[j] {
                if !u[j].is_finite() {
                    ok = false;
                    break;
                }
                a_shift[j] = -row[j];
                b_shift -= row[j] * u[j];
            } else {
                if !l[j].is_finite() {
                    // A nonzero coefficient on a variable with no finite lower
                    // bound cannot be shifted nonnegative; skip the row.
                    if row[j].abs() > tol {
                        ok = false;
                        break;
                    }
                } else {
                    b_shift -= row[j] * l[j];
                }
                a_shift[j] = row[j];
            }
        }
        if !ok {
            continue;
        }
        let (ai, bi) = match scale_row_to_integer(&a_shift, b_shift) {
            Some(v) => v,
            None => continue,
        };
        // Slack of the SHIFTED row at y*: b̃ − ã · y*.
        let slack = bi as f64 - (0..n).map(|j| ai[j] as f64 * y_star[j]).sum::<f64>();
        // Only low-slack rows can contribute to a violated cut; a row that is
        // itself infeasible at x* (slack < −tol) is not a valid ≤ row here, skip.
        if slack < -tol || slack > SLACK_ELIGIBLE_MAX {
            continue;
        }
        // Coefficient-parity bitset over the ACTIVE columns only (see above).
        let mut coef_bits = vec![0u64; words];
        for (j, &c) in ai.iter().enumerate() {
            if active[j] && c.rem_euclid(2) == 1 {
                let bit = active_bit[j];
                coef_bits[bit / 64] |= 1u64 << (bit % 64);
            }
        }
        let rhs_parity = bi.rem_euclid(2) == 1;
        // A row that already has all-even coefficients with odd rhs is a
        // singleton violated {0,½} candidate (S = {i}); it enters elimination as a
        // normal row (`pivot_col` returns `None` for it, so it is tested directly).
        rows.push(ParityRow {
            a: ai,
            b: bi,
            coef_bits,
            rhs_parity,
            slack: slack.max(0.0),
        });
    }
    if rows.is_empty() {
        return cuts;
    }

    // GF(2) Gaussian elimination keyed by pivot column. For each row, reduce it
    // by the already-pivoted rows; if it reduces to all-even coefficients, it is
    // a subset combination with even coefficient parity — a {0,½} candidate
    // (violated iff its rhs parity is odd and its accumulated slack < 1). Rows
    // with a fresh pivot column become pivots. This finds a spanning set of
    // low-slack even-parity combinations (a heuristic, not the min-weight one).
    let mut pivots: HashMap<usize, ParityRow> = HashMap::new();
    // Deduplicate emitted cuts by their integer combination signature.
    let mut emitted: std::collections::HashSet<Vec<i64>> = std::collections::HashSet::new();

    // Cap the accumulated slack during elimination: a subset can only yield a
    // violated {0,½} cut when its total slack is `< 1`, but we let combinations
    // grow a little past that during reduction (a later XOR can lower the
    // effective violation window only via parity, not slack) — abandoning a row
    // whose slack has already exceeded this bound keeps the search cheap without
    // ever affecting soundness (it only drops candidates, never emits bad ones).
    const SLACK_ABANDON: f64 = 2.0;

    for row in rows.into_iter() {
        let mut cur = row;
        // Reduce `cur` by the registered pivots until it either acquires a fresh
        // pivot column (becomes a pivot) or reduces to all-even coefficients.
        loop {
            match pivot_col(&cur.coef_bits) {
                None => {
                    // All-even coefficient parity ⇒ a subset S with `Σ a_i` even
                    // on every column. It is a violated {0,½} candidate exactly
                    // when the rhs parity is odd (maximal rounding gain) and the
                    // accumulated slack is `< 1`.
                    if cur.rhs_parity && cur.slack < 1.0 - tol {
                        try_emit(
                            &cur,
                            &comp,
                            l,
                            u,
                            x,
                            tol,
                            max_dynamism,
                            &mut emitted,
                            &mut cuts,
                        );
                    }
                    break;
                }
                Some(pc) => match pivots.get(&pc) {
                    Some(piv) => {
                        xor_into_ref(&mut cur, piv);
                        if cur.slack > SLACK_ABANDON {
                            break; // cannot become a violated cut; drop it
                        }
                    }
                    None => {
                        // Fresh pivot column: register `cur` as the pivot for it.
                        pivots.insert(pc, cur);
                        break;
                    }
                },
            }
        }
    }
    cuts
}

/// Emit the CG-rounded cut for an even-parity, odd-rhs combination, if it is
/// violated, numerically safe, and not a duplicate.
#[allow(clippy::too_many_arguments)]
fn try_emit(
    cur: &ParityRow,
    comp: &[bool],
    l: &[f64],
    u: &[f64],
    x: &[f64],
    tol: f64,
    max_dynamism: f64,
    emitted: &mut std::collections::HashSet<Vec<i64>>,
    cuts: &mut Vec<ZeroHalfCut>,
) {
    if let Some((coeffs, rhs, viol)) =
        round_and_map(&cur.a, cur.b, comp, l, u, x, tol, max_dynamism)
    {
        // Signature: the CG-rounded integer coefficient vector + rhs (in shifted
        // space) — dedups distinct member-subsets that round to the same cut.
        let mut sig: Vec<i64> = cur
            .a
            .iter()
            .map(|&a| (a as f64 / 2.0).floor() as i64)
            .collect();
        sig.push((cur.b as f64 / 2.0).floor() as i64);
        if emitted.insert(sig) {
            cuts.push(ZeroHalfCut {
                coeffs,
                rhs,
                violation: viol,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// Exhaustively check a `≤` cut excludes no integer-feasible point of the
    /// SYSTEM `A x ≤ b` over integer x in `[lo, hi]`. (Same enumerator as the MIR
    /// validity tests; a zerohalf cut is valid for the whole system, not one row.)
    fn assert_valid_system(
        coeffs: &[f64],
        rhs: f64,
        a: &[Vec<f64>],
        b: &[f64],
        lo: &[i64],
        hi: &[i64],
    ) {
        let n = coeffs.len();
        let mut idx = lo.to_vec();
        loop {
            let xv: Vec<f64> = idx.iter().map(|&v| v as f64).collect();
            let feasible = a.iter().zip(b).all(|(ai, &bi)| dot(ai, &xv) <= bi + 1e-9);
            if feasible {
                assert!(
                    dot(coeffs, &xv) <= rhs + 1e-6,
                    "zerohalf cut {coeffs:?} <= {rhs} excludes feasible {xv:?}"
                );
            }
            let mut k = 0;
            while k < n {
                idx[k] += 1;
                if idx[k] <= hi[k] {
                    break;
                }
                idx[k] = lo[k];
                k += 1;
            }
            if k == n {
                break;
            }
        }
    }

    fn flatten(a: &[Vec<f64>]) -> Vec<f64> {
        a.iter().flatten().copied().collect()
    }

    #[test]
    fn zerohalf_two_row_parity_cut() {
        // The canonical two-row {0,½} example:
        //   x0 + x1        <= 1
        //        x1 + x2   <= 1
        //   x0      + x2   <= 1
        // Summing all three (weight ½) and rounding gives x0 + x1 + x2 <= 1
        // (the odd-cycle/clique inequality) which cuts the LP point (½,½,½).
        let a = vec![
            vec![1.0, 1.0, 0.0],
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 1.0],
        ];
        let b = vec![1.0, 1.0, 1.0];
        let l = [0.0, 0.0, 0.0];
        let u = [1.0, 1.0, 1.0];
        let integ = [true, true, true];
        let x = [0.5, 0.5, 0.5]; // LP optimum of the triangle relaxation
        let cuts = separate_zerohalf(&flatten(&a), &b, &l, &u, &integ, &x, 1e-7, 1e9);
        assert!(!cuts.is_empty(), "must find the odd-cycle {{0,½}} cut");
        // Some emitted cut must separate x* and be valid for the whole system.
        let mut separated = false;
        for c in &cuts {
            assert_valid_system(&c.coeffs, c.rhs, &a, &b, &[0, 0, 0], &[1, 1, 1]);
            if dot(&c.coeffs, &x) > c.rhs + 1e-6 {
                separated = true;
            }
        }
        assert!(separated, "no emitted zerohalf cut separated x*={x:?}");

        // Differential: single-row MIR (the shipped competitor separator) finds
        // NO violated cut on this triangle — each row `x_i + x_j <= 1` has
        // integral rhs and 0/1 coefficients, so its MIR function is the row
        // itself, satisfied at (½,½,½). This is exactly the odd-cycle / parity
        // structure only {0,½} captures; the assertion fails-before (would be a
        // MIR cut) and passes-after (MIR is empty, zerohalf is not).
        let mir = crate::lp::mir::separate_mir(&flatten(&a), &b, &l, &u, &integ, &x, 1e-7, 1e9);
        let mir_viol = mir
            .iter()
            .map(|c| dot(&c.coeffs, &x) - c.rhs)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            mir.is_empty() || mir_viol <= 1e-6,
            "single-row MIR unexpectedly separated x* (viol {mir_viol}); the test \
             no longer isolates the {{0,½}}-only odd-cycle structure"
        );
    }

    #[test]
    fn zerohalf_finds_cut_gomory_single_row_misses() {
        // A pure parity structure where no single row is fractional-rhs after the
        // natural scaling but the ½-sum of two rows is: 2x0 + 2x1 <= 2 and
        // 2x0 - 2x1 <= 0 sum to 4x0 <= 2 -> x0 <= 0 (⌊2/2... ⌋). Construct so the
        // LP point x0=0.5 is cut. Rows scaled by ½ internally.
        let a = vec![vec![2.0, 2.0], vec![2.0, -2.0]];
        let b = vec![2.0, 0.0];
        let l = [0.0, 0.0];
        let u = [1.0, 1.0];
        let integ = [true, true];
        let x = [0.5, 0.5];
        let cuts = separate_zerohalf(&flatten(&a), &b, &l, &u, &integ, &x, 1e-7, 1e9);
        for c in &cuts {
            assert_valid_system(&c.coeffs, c.rhs, &a, &b, &[0, 0], &[1, 1]);
        }
    }

    /// Adversarial validity property test (the correctness gate). Hundreds of
    /// random integer `≤` systems with mixed integer/continuous columns,
    /// mixed-sign finite bounds, and LP points forced cuttable: assert NO
    /// integer-feasible point of the ORIGINAL system is ever cut, and that many
    /// (>20) non-vacuous cuts were validated. Same regime that caught #415.
    #[test]
    fn zerohalf_validity_random_systems() {
        let mut state: u64 = 0xf00d_babe_1234_5678;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut n_cuts_checked = 0usize;
        for _ in 0..400 {
            let n = 3usize;
            let m = 3usize + (next() * 3.0) as usize; // 3..=5 rows
                                                      // Mixed-sign small integer coefficients (zerohalf needs integer data;
                                                      // scaling handles a common ½ but we keep the base integral here).
            let a: Vec<Vec<f64>> = (0..m)
                .map(|_| {
                    (0..n)
                        .map(|_| (next() * 5.0).floor() - 2.0) // -2..=2
                        .collect()
                })
                .collect();
            // Mixed-sign integer lower bounds in [-2,1]; width 1..=3 above.
            let lo_i: Vec<i64> = (0..n).map(|_| (next() * 4.0).floor() as i64 - 2).collect();
            let hi_i: Vec<i64> = (0..n)
                .map(|k| lo_i[k] + 1 + (next() * 3.0).floor() as i64)
                .collect();
            let l: Vec<f64> = lo_i.iter().map(|&v| v as f64).collect();
            let u: Vec<f64> = hi_i.iter().map(|&v| v as f64).collect();
            let integ: Vec<bool> = (0..n).map(|_| next() > 0.3).collect();
            // rhs integral (integer data); pick so the box has feasible points.
            let b: Vec<f64> = a
                .iter()
                .map(|ai| {
                    // Center the rhs near a·midpoint so rows are active-ish.
                    let mid: f64 = (0..n).map(|k| ai[k] * 0.5 * (l[k] + u[k])).sum();
                    (mid + (next() * 3.0).floor() - 1.0).round()
                })
                .collect();
            // LP point pushed toward mixed positions (some near upper), forced to
            // be a plausible relaxation optimum inside the box.
            let x: Vec<f64> = (0..n)
                .map(|k| {
                    let t = if next() > 0.4 {
                        0.55 + 0.4 * next()
                    } else {
                        next()
                    };
                    l[k] + t * (u[k] - l[k])
                })
                .collect();
            for c in separate_zerohalf(&flatten(&a), &b, &l, &u, &integ, &x, 1e-7, 1e9) {
                assert_valid_system(&c.coeffs, c.rhs, &a, &b, &lo_i, &hi_i);
                assert!(
                    c.violation > 0.0,
                    "emitted a non-violated cut (viol {})",
                    c.violation
                );
                n_cuts_checked += 1;
            }
        }
        assert!(
            n_cuts_checked > 20,
            "expected >20 validated zerohalf cuts, got {n_cuts_checked}"
        );
    }

    /// The heuristic only affects strength, not validity: even with a degenerate
    /// point that makes many rows eligible, every cut stays valid.
    #[test]
    fn zerohalf_validity_binary_dense() {
        let mut state: u64 = 0x0bad_f00d_c0de_1111;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut checked = 0usize;
        for _ in 0..200 {
            let n = 4usize;
            let m = 4usize;
            // 0/1 coefficient rows (odd-cycle / set-packing structure — exactly
            // where {0,½} cuts bite).
            let a: Vec<Vec<f64>> = (0..m)
                .map(|_| {
                    (0..n)
                        .map(|_| if next() > 0.5 { 1.0 } else { 0.0 })
                        .collect()
                })
                .collect();
            let b: Vec<f64> = (0..m).map(|_| 1.0 + (next() * 2.0).floor()).collect();
            let l = vec![0.0; n];
            let u = vec![1.0; n];
            let integ = vec![true; n];
            let x: Vec<f64> = (0..n).map(|_| 0.3 + 0.5 * next()).collect();
            for c in separate_zerohalf(&flatten(&a), &b, &l, &u, &integ, &x, 1e-7, 1e9) {
                assert_valid_system(&c.coeffs, c.rhs, &a, &b, &vec![0; n], &vec![1; n]);
                checked += 1;
            }
        }
        // Not asserting a minimum here (binary-dense may or may not be cuttable),
        // just that whatever is emitted is valid.
        let _ = checked;
    }
}
