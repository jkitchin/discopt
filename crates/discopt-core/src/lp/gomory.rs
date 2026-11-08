//! Gomory mixed-integer (GMI) cuts from a recovered LP basis.
//!
//! Given a basis (from [`super::basis::recover_basis`]) the simplex tableau row
//! for a basic variable `x_{B_i}` is `x_{B_i} + Σ_{j∈N} ā_j x_j = b̄_i`, where
//! `ā = B⁻¹A` over the nonbasic columns `N`. When `x_{B_i}` is an
//! integer-constrained variable with a fractional value, this row yields a
//! **Gomory mixed-integer cut** — a valid inequality every integer-feasible
//! point satisfies but the current fractional vertex violates.
//!
//! The derivation works in the *shifted nonbasic space* `x̃_j ≥ 0`
//! (`x̃_j = x_j − l_j` at a lower bound, `u_j − x_j` at an upper bound). With the
//! row as `x_{B_i} + Σ_j ᾱ_j x̃_j = β` (`f₀ = β − ⌊β⌋`), the GMI cut is
//! `Σ_j ψ_j x̃_j ≥ 1` with, for `f_j = ᾱ_j − ⌊ᾱ_j⌋`:
//!
//! - integer nonbasic `j`: `ψ_j = f_j/f₀` if `f_j ≤ f₀`, else `(1−f_j)/(1−f₀)`;
//! - continuous nonbasic `j`: `ψ_j = ᾱ_j/f₀` if `ᾱ_j ≥ 0`, else `−ᾱ_j/(1−f₀)`.
//!
//! Substituting `x̃` back gives a cut `Σ_j γ_j x_j ≥ δ` over the original
//! standard-form variables (basic variables do not appear).
//!
//! **Numerical safety.** GMI is notoriously sensitive: an inaccurate tableau
//! coefficient `ā_j` near an integer flips `f_j` between ≈0 and ≈1, an O(1)
//! error that can make the cut *invalid* and cut off the true integer optimum.
//! Because the basis comes from an interior-point solve + crossover (accurate
//! only to ~1e-7), we do **not** trust the input vertex: the basic primal
//! values `x_B = B⁻¹(b − A_N x_N)` and the tableau rows `B⁻ᵀ e_i` are both
//! recomputed from the *exact* basis and bounds with **iterative refinement**
//! (driving the residual to ~machine precision).
//!
//! The refined `ā_j` are **not** snapped to the nearest integer. This module used
//! to snap them within [`SNAP_TOL`], justified as collapsing the flip error above.
//! That justification does not hold: the flip is in `f_j`, and `ψ(f)` is
//! *continuous* with `ψ → 0` at both `f → 0` and `f → 1`, so a coefficient
//! straddling an integer produces a near-zero `ψ` either way and there is nothing
//! to collapse. What snapping actually did was zero the `ψ` of a **continuous**
//! column whose `ā_j` was merely small, deleting `ψ_j·x̃_j` from the LHS of a `≥`
//! cut — a *strengthening*, and the one way this routine can emit an invalid cut.
//! See #1236: on the `fac2` OA master that deleted two terms worth 0.47 and 0.025
//! against a rhs of 1 (continuous slacks, `u = 1e20`, `x̃ ≈ 7e7` and `2.3e8`), and
//! the driver certified a false `optimal` 7839 above the true optimum.
//!
//! A tiny `ψ_j` may still leave the LHS, but only as a **relaxation**: its maximum
//! over the box, `ψ_j·(u_j − l_j)`, is charged to the rhs, and only when that
//! product is itself negligible. When the range is unbounded the term cannot move
//! and is kept exactly. Bounds are tested against the `1e20` sentinel directly and
//! never through a product — for a small `ψ` the product of an unbounded bound is
//! an ordinary finite number, which is precisely how this layer has produced a
//! false certified `optimal` before.

use super::basis::{Basis, AT_UPPER, BASIC};
use super::crossover::LpView;
use super::simplex::linsolve::{FeralLU, LinearSolver};
use super::simplex::sparse::SparseCols;

/// Tolerance for snapping a refined tableau coefficient to the nearest integer.
const SNAP_TOL: f64 = 1e-9;

/// Minimum fractionality of a basic variable to cut on. A value nearer than
/// this to an integer would divide by a tiny `f₀` (or `1−f₀`) and blow the cut
/// coefficients up to numerically unsafe magnitudes, so we skip it.
const FRAC_MIN: f64 = 1e-3;

/// Largest `Σ|ᾱ_j·x̃_j|` for which a globally complemented GMI row's β (and so
/// its f0) is trusted (PR #2196 review, item 3). β is a cancelling sum whose absolute
/// rounding error scales with this magnitude; with each `ᾱ_j` carrying a relative
/// error of ~1e-15 after btran refinement, 1e6 keeps that error near 1e-9, six orders
/// below `FRAC_MIN`, while 1e12 would put it at `FRAC_MIN` itself. A refusal drops a
/// cut, which can only weaken a bound, never invalidate one.
const GLOBAL_BETA_MAG_MAX: f64 = 1e6;

/// Absolute cap on a cut coefficient. A cut with a larger coefficient comes
/// from an ill-conditioned basis (or a near-integral pivot) and is dropped: it
/// would dominate the relaxation numerically and is unsafe even when formally
/// valid.
const MAX_ABS_COEFF: f64 = 1e7;

/// `DISCOPT_MILP_ROOT_GMI_REPAIR` (highs-parity P4.4): separate GMI cuts off a SHORT
/// basis by completing it with unit columns ([`separate_on_completed_basis`]); the
/// MILP driver's slack rewrite reads the same flag (`substitute_slacks_to_structural`'s
/// `slack_by_column`).
///
/// **Default OFF again (2026-11-06, owner).** It was graduated earlier the same day on a
/// §5 panel whose pre-registered bars passed (`p44_gmi_repair_panel.py`, data
/// `docs/dev/data/highs-parity/p4.4/`: cert-clean, certified 96 -> 99, nodes on the
/// always-certified set -19 %, 1 dual loss of a cap of 2). That panel's own rows also
/// show what the bars did not score: the incumbent LOST in every rep on `nexp-50-20-4-2`,
/// `sp150x300d` and `neos17`, and node throughput collapsed on the open population
/// (`fiber` median 2058 -> 13 nodes, `beavma` 5455 -> 35). Cause: the completed-basis
/// cuts are dense after the slack rewrite (`fiber` ~297 nonzeros per cut on a matrix with
/// ~8 per row), so every node LP carries them (`docs/dev/highs-parity-d2-analysis.md`).
/// What would re-graduate it: a density gate on these cuts and a re-run panel that also
/// bars incumbent loss and per-node cost on the open population. `=1` opts in.
pub(crate) fn gmi_repair_enabled() -> bool {
    let raw = std::env::var("DISCOPT_MILP_ROOT_GMI_REPAIR").unwrap_or_default();
    parse_gmi_repair(&raw)
}

/// The parse table behind [`gmi_repair_enabled`]: `1`/`true`/`yes` opt in, anything
/// else (unset included) reads as off, as before.
fn parse_gmi_repair(raw: &str) -> bool {
    match raw {
        "" | "0" | "false" | "no" => false,
        "1" | "true" | "yes" => true,
        _ => false,
    }
}

/// Work cap (`m·k²` flops) for the dense rank-revealing completion; above it the
/// structural matching is used (and a numerically singular result is declined).
const DENSE_COMPLETION_WORK: f64 = 5e8;

/// Rows NOT chosen as pivots by Gaussian elimination with partial (row) pivoting on
/// the dense `m × k` block of the basic columns. Completing with those rows' unit
/// columns gives a matrix that is nonsingular by construction: the eliminated block
/// is triangular on the pivot rows and the unit columns cover the rest. `None` when
/// some column has no pivot above `1e-9 · ‖column‖∞` (the columns are numerically
/// dependent: no completion is safe).
fn pivot_complement_rows(sp: &SparseCols, m: usize, basic: &[usize]) -> Option<Vec<usize>> {
    let k = basic.len();
    let mut mat = vec![0.0f64; m * k]; // column-major
    for (c, &col) in basic.iter().enumerate() {
        let (rows, vals) = sp.col(col);
        for (&r, &v) in rows.iter().zip(vals) {
            mat[c * m + r] = v;
        }
    }
    let mut used = vec![false; m];
    for c in 0..k {
        let colmax = (0..m).map(|r| mat[c * m + r].abs()).fold(0.0, f64::max);
        let (mut pr, mut pv) = (usize::MAX, 0.0f64);
        for r in 0..m {
            if !used[r] && mat[c * m + r].abs() > pv {
                pv = mat[c * m + r].abs();
                pr = r;
            }
        }
        if pr == usize::MAX || pv <= 1e-9 * colmax.max(f64::MIN_POSITIVE) {
            return None;
        }
        used[pr] = true;
        let piv = mat[c * m + pr];
        for c2 in c + 1..k {
            let f = mat[c2 * m + pr] / piv;
            if f != 0.0 {
                for r in 0..m {
                    if !used[r] || r == pr {
                        mat[c2 * m + r] -= f * mat[c * m + r];
                    }
                }
            }
        }
    }
    Some((0..m).filter(|&r| !used[r]).collect())
}

/// Rows left uncovered by a maximum bipartite matching of the basic columns to the
/// rows of their support (Kuhn's augmenting paths, iterative). `None` when some
/// basic column cannot be matched (the columns are structurally dependent, so no
/// completion is nonsingular).
fn unmatched_rows(sp: &SparseCols, m: usize, basic: &[usize]) -> Option<Vec<usize>> {
    let mut row_match: Vec<usize> = vec![usize::MAX; m]; // row -> index into `basic`
    let mut seen: Vec<usize> = vec![usize::MAX; m]; // row -> stamp of the last search
    for (k, &col) in basic.iter().enumerate() {
        // Cheap greedy first: a free row in the column's support.
        let (rows, _) = sp.col(col);
        if let Some(&r) = rows.iter().find(|&&r| row_match[r] == usize::MAX) {
            row_match[r] = k;
            continue;
        }
        // Augmenting path search from column k: stack of (column index, next row pos).
        let mut stack: Vec<(usize, usize)> = vec![(k, 0)];
        let mut path_rows: Vec<usize> = Vec::new();
        let mut found = false;
        while let Some(&mut (ck, ref mut pos)) = stack.last_mut() {
            let (rows, _) = sp.col(basic[ck]);
            if *pos >= rows.len() {
                stack.pop();
                path_rows.pop();
                continue;
            }
            let r = rows[*pos];
            *pos += 1;
            if seen[r] == k {
                continue;
            }
            seen[r] = k;
            path_rows.push(r);
            if row_match[r] == usize::MAX {
                found = true;
                break;
            }
            stack.push((row_match[r], 0));
        }
        if !found {
            return None;
        }
        // Flip the path: stack[i] column takes path_rows[i].
        for (i, &(ck, _)) in stack.iter().enumerate() {
            row_match[path_rows[i]] = ck;
        }
    }
    Some((0..m).filter(|&r| row_match[r] == usize::MAX).collect())
}

/// GMI separation off a basis that is SHORT (fewer than `m` basic columns), as the
/// primal export leaves it when a degenerate artificial cannot be pivoted out of a
/// redundant row (measured on `fiber`: 213 of 363 rows at the root, so the root got
/// no GMI cut at all).
///
/// The basis is completed with the unit columns `e_r` of the rows a maximum matching
/// leaves uncovered, appended as extra columns fixed at `[0, 0]`, continuous and
/// BASIC, and the ordinary separator runs on that augmented system. **Soundness:**
/// every tableau row is a linear combination of the rows of `[A | E] (x, a) = b`; a
/// feasible `x` of the original problem satisfies `A x = b`, i.e. the augmented system
/// with `a = 0`, which is inside the appended columns' `[0, 0]` box. So each tableau
/// identity, and every GMI cut derived from it, holds on the original feasible set.
/// The appended columns are basic, so the separator never gives them a coefficient;
/// the cuts are truncated back to the `n` original columns. `None` (decline, exactly
/// as before) when the matching or the factorization fails.
#[allow(clippy::too_many_arguments)]
fn separate_on_completed_basis(
    sp: &SparseCols,
    m: usize,
    n: usize,
    l: &[f64],
    u: &[f64],
    b: &[f64],
    basis: &Basis,
    integrality: &[bool],
    tol: f64,
    max_dynamism: f64,
) -> Option<Vec<GomoryCut>> {
    let k = basis.basic_vars.len();
    let missing = if (m as f64) * (k as f64) * (k as f64) <= DENSE_COMPLETION_WORK {
        pivot_complement_rows(sp, m, &basis.basic_vars)?
    } else {
        unmatched_rows(sp, m, &basis.basic_vars)?
    };
    if missing.len() + basis.basic_vars.len() != m {
        return None;
    }
    let (cp, ri, vv) = sp.raw();
    let mut col_ptr = cp.to_vec();
    let mut row_idx = ri.to_vec();
    let mut vals = vv.to_vec();
    for &r in &missing {
        row_idx.push(r);
        vals.push(1.0);
        col_ptr.push(row_idx.len());
    }
    let sp2 = SparseCols::from_csc(col_ptr, row_idx, vals);
    let n2 = n + missing.len();
    let mut l2 = l[..n].to_vec();
    let mut u2 = u[..n].to_vec();
    l2.resize(n2, 0.0);
    u2.resize(n2, 0.0);
    let mut int2 = integrality[..n].to_vec();
    int2.resize(n2, false);
    let mut status = basis.col_status[..n].to_vec();
    status.resize(n2, BASIC);
    let mut basic = basis.basic_vars.clone();
    basic.extend(n..n2);
    let full = Basis {
        col_status: status,
        basic_vars: basic,
    };
    crate::profile::incr(crate::profile::Ctr::SepGomoryBasisCompleted);
    let mut cuts = separate_gomory_cols(&sp2, m, n2, &l2, &u2, b, &full, &int2, tol, max_dynamism);
    for c in &mut cuts {
        debug_assert!(c.coeffs[n..].iter().all(|v| *v == 0.0));
        c.coeffs.truncate(n);
    }
    Some(cuts)
}

/// A generated cut `coeffs · x ≥ rhs` over the standard-form variables.
#[derive(Debug, Clone)]
pub struct GomoryCut {
    /// Dense length-`n` coefficient vector.
    pub coeffs: Vec<f64>,
    /// Right-hand side (the cut is `coeffs · x ≥ rhs`).
    pub rhs: f64,
}

/// Solve the dense `n × n` system `mat · x = rhs` (row-major) by Gaussian
/// elimination with partial pivoting. Returns `None` if singular to `tol`.
pub(crate) fn solve_dense(mat: &[f64], n: usize, rhs: &[f64], tol: f64) -> Option<Vec<f64>> {
    let mut a = mat.to_vec();
    let mut x = rhs.to_vec();
    for col in 0..n {
        let mut p = col;
        for i in (col + 1)..n {
            if a[i * n + col].abs() > a[p * n + col].abs() {
                p = i;
            }
        }
        if a[p * n + col].abs() <= tol {
            return None;
        }
        if p != col {
            for j in 0..n {
                a.swap(col * n + j, p * n + j);
            }
            x.swap(col, p);
        }
        let piv = a[col * n + col];
        for j in 0..n {
            a[col * n + j] /= piv;
        }
        x[col] /= piv;
        for i in 0..n {
            if i != col {
                let f = a[i * n + col];
                if f != 0.0 {
                    for j in 0..n {
                        a[i * n + j] -= f * a[col * n + j];
                    }
                    x[i] -= f * x[col];
                }
            }
        }
    }
    Some(x)
}

/// Solve `B x = rhs` with iterative refinement, where `B` is the basis whose
/// columns are `A[:, basis[k]]` (factorized into `lu`) and `sp` is the CSC view
/// of `A`. The feral factor gives the initial solve; each refinement round
/// computes the residual `rhs − B x` by a sparse matvec against the basis
/// columns and corrects with another ftran — the same machine-precision drive
/// as the dense [`solve_refined`], at O(nnz) per round instead of O(m²). Returns
/// `None` if the factor solve fails.
fn ftran_refined(
    lu: &mut FeralLU,
    sp: &SparseCols,
    basis: &[usize],
    rhs: &[f64],
    _tol: f64,
) -> Option<Vec<f64>> {
    let mut x = rhs.to_vec();
    lu.ftran(&mut x).ok()?;
    for _ in 0..3 {
        // r = rhs − B x  (B column k contributes x[k]·A[:, basis[k]]).
        let mut r = rhs.to_vec();
        for (k, &bv) in basis.iter().enumerate() {
            let xk = x[k];
            if xk != 0.0 {
                let (rows, vals) = sp.col(bv);
                for (t, &i) in rows.iter().enumerate() {
                    r[i] -= vals[t] * xk;
                }
            }
        }
        let mut dx = r;
        lu.ftran(&mut dx).ok()?;
        let mut maxdx = 0.0_f64;
        for (xi, dxi) in x.iter_mut().zip(&dx) {
            *xi += dxi;
            maxdx = maxdx.max(dxi.abs());
        }
        if maxdx <= 1e-15 {
            break;
        }
    }
    Some(x)
}

/// Solve `Bᵀ w = rhs` with iterative refinement (row of `B⁻¹` when `rhs = e_i`).
/// The residual `rhs − Bᵀ w` is `rhs[k] − w·A[:, basis[k]]`, a sparse column dot
/// per basis slot. O(nnz) per refinement round; mirrors [`ftran_refined`].
fn btran_refined(
    lu: &mut FeralLU,
    sp: &SparseCols,
    basis: &[usize],
    rhs: &[f64],
    _tol: f64,
) -> Option<Vec<f64>> {
    let mut w = rhs.to_vec();
    lu.btran(&mut w).ok()?;
    for _ in 0..3 {
        let mut r = rhs.to_vec();
        for (k, &bv) in basis.iter().enumerate() {
            r[k] -= sp.dot(bv, &w);
        }
        let mut dw = r;
        lu.btran(&mut dw).ok()?;
        let mut maxd = 0.0_f64;
        for (wi, dwi) in w.iter_mut().zip(&dw) {
            *wi += dwi;
            maxd = maxd.max(dwi.abs());
        }
        if maxd <= 1e-15 {
            break;
        }
    }
    Some(w)
}

/// Separate Gomory mixed-integer cuts from the basis of `lp` (`b` is the
/// length-`m` right-hand side of `A x = b`).
///
/// The vertex is reconstructed from the exact basis and bounds (not from any
/// approximate input point): nonbasic variables sit at the bound named by
/// `basis.col_status`, and the basic values come from a refined solve of
/// `B x_B = b − A_N x_N`. `integrality[j]` marks integer-constrained variables,
/// `tol` is the fractionality/zero tolerance, and `max_dynamism` caps a cut's
/// `max|coeff| / min nonzero |coeff|` ratio. Returns one cut per fractional
/// integer basic variable that yields a numerically sound inequality.
pub fn separate_gomory(
    lp: &LpView<'_>,
    b: &[f64],
    basis: &Basis,
    integrality: &[bool],
    tol: f64,
    max_dynamism: f64,
) -> Vec<GomoryCut> {
    // Dense-entry wrapper (used by tests): build the CSC once and delegate. The
    // driver calls `separate_gomory_cols` directly with its working CSC (no dense
    // matrix) — see docs/dev/sparse-milp-plan.md T3b5.
    let sp = SparseCols::from_dense(lp.a, lp.m, lp.n);
    separate_gomory_cols(
        &sp,
        lp.m,
        lp.n,
        lp.l,
        lp.u,
        b,
        basis,
        integrality,
        tol,
        max_dynamism,
    )
}

/// CSC-input GMI separation (the body of [`separate_gomory`]). Bit-identical: the
/// function already worked entirely through a `SparseCols` (`sp.col(j)` + a sparse
/// LU factorization); it just took the dense matrix and rebuilt the CSC internally.
#[allow(clippy::too_many_arguments)]
pub fn separate_gomory_cols(
    sp: &SparseCols,
    m: usize,
    n: usize,
    l: &[f64],
    u: &[f64],
    b: &[f64],
    basis: &Basis,
    integrality: &[bool],
    tol: f64,
    max_dynamism: f64,
) -> Vec<GomoryCut> {
    let cuts = Vec::new();
    crate::profile::incr(crate::profile::Ctr::SepGomoryCalls);
    if m == 0 {
        return cuts;
    }
    // A complete row-ordered basis is required to factorize B; a short basis
    // (degenerate phase-2 artifact the caller could not complete) is unusable —
    // decline rather than index past it. Matches the old dense path's `None`.
    if basis.basic_vars.len() != m {
        if basis.basic_vars.len() < m && gmi_repair_enabled() {
            if let Some(done) = separate_on_completed_basis(
                sp,
                m,
                n,
                l,
                u,
                b,
                basis,
                integrality,
                tol,
                max_dynamism,
            ) {
                return done;
            }
        }
        crate::profile::incr(crate::profile::Ctr::SepGomoryShortBasis);
        return cuts;
    }
    gomory_rows(
        sp,
        m,
        n,
        l,
        u,
        None,
        b,
        basis,
        integrality,
        tol,
        max_dynamism,
    )
}

/// GMI cuts that are valid for the **whole tree**, separated at a B&B node
/// (highs-parity D2, `DISCOPT_MILP_NODE_GMI`).
///
/// The tableau rows come from the node's optimal basis and the vertex from its
/// **local** bounds `l`/`u`. Each nonbasic column is then complemented at its
/// **global** bound on the same side (`gl` for at-lower, `gu` for at-upper), with
/// `x̃_j = |x*_j − bound|` and `β = x_B + Σ α_j x̃_j`. The tableau identity
/// `x_B + Σ α_j x̃_j = β` holds for every point with `A x = b`, and `x̃ ≥ 0` on the
/// global box, so the GMI cut `Σ ψ_j x̃_j ≥ 1` (with `f0 = frac(β)`) is globally valid.
/// This is how SCIP's in-tree Gomory cuts reach its global pool. Its integer
/// strengthening applies only to integer columns whose complementing *global* bound is
/// integral.
///
/// Columns `n_global..n` exist only in the node's local LP copy (rows appended there).
/// A cut with a nonzero coefficient on one of them is refused, and the cuts returned
/// are truncated to length `n_global`. A complementing global bound at the `1e20`
/// sentinel also refuses the cut (no finite `x̃`).
#[allow(clippy::too_many_arguments)]
pub fn separate_gomory_cols_global(
    sp: &SparseCols,
    m: usize,
    n: usize,
    l: &[f64],
    u: &[f64],
    gl: &[f64],
    gu: &[f64],
    n_global: usize,
    b: &[f64],
    basis: &Basis,
    integrality: &[bool],
    tol: f64,
    max_dynamism: f64,
) -> Vec<GomoryCut> {
    crate::profile::incr(crate::profile::Ctr::SepGomoryCalls);
    if m == 0 || basis.basic_vars.len() != m || gl.len() < n || gu.len() < n {
        return Vec::new();
    }
    let mut cuts = gomory_rows(
        sp,
        m,
        n,
        l,
        u,
        Some((gl, gu, n_global)),
        b,
        basis,
        integrality,
        tol,
        max_dynamism,
    );
    for c in &mut cuts {
        c.coeffs.truncate(n_global);
    }
    cuts
}

/// The GMI row loop shared by [`separate_gomory_cols`] (`global = None`: complement at
/// `l`/`u`, bit-identical to the historical body) and
/// [`separate_gomory_cols_global`] (`global = Some((gl, gu, n_global))`).
#[allow(clippy::too_many_arguments)]
fn gomory_rows(
    sp: &SparseCols,
    m: usize,
    n: usize,
    l: &[f64],
    u: &[f64],
    global: Option<(&[f64], &[f64], usize)>,
    b: &[f64],
    basis: &Basis,
    integrality: &[bool],
    tol: f64,
    max_dynamism: f64,
) -> Vec<GomoryCut> {
    let mut cuts = Vec::new();
    // Complementing bounds: the global ones when separating globally valid cuts.
    let (cl, cu) = match global {
        Some((gl, gu, _)) => (gl, gu),
        None => (l, u),
    };
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
        crate::profile::incr(crate::profile::Ctr::SepGomorySingular);
        return cuts; // singular basis → no cuts (as the dense solve's None did)
    }

    // Reconstruct the vertex exactly: nonbasic at bounds, x_B = B⁻¹(b − A_N x_N).
    let mut rhs_b = b.to_vec();
    for j in 0..n {
        if basis.col_status[j] == BASIC {
            continue;
        }
        let val = if basis.col_status[j] == AT_UPPER {
            u[j]
        } else {
            l[j]
        };
        if val != 0.0 {
            let (rows, vals) = sp.col(j);
            for (k, &i) in rows.iter().enumerate() {
                rhs_b[i] -= vals[k] * val;
            }
        }
    }
    let xb = match ftran_refined(&mut lu, sp, &basis.basic_vars, &rhs_b, tol) {
        Some(xb) => xb,
        None => {
            crate::profile::incr(crate::profile::Ctr::SepGomoryFtranFail);
            return cuts;
        }
    };

    for (i, &bi) in basis.basic_vars.iter().enumerate() {
        if !integrality[bi] {
            continue; // only integer basic variables yield cuts
        }
        if global.is_none() {
            let f0 = xb[i] - xb[i].floor();
            if !(FRAC_MIN..=1.0 - FRAC_MIN).contains(&f0) {
                continue; // integral, or too close to integral for a safe cut
            }
        } else if (xb[i] - xb[i].round()).abs() < FRAC_MIN {
            continue; // an integral basic value: nothing to separate at this vertex
        }

        // Row i of B⁻¹: refined solve of Bᵀ w = e_i (one btran + refinement).
        let mut e_i = vec![0.0_f64; m];
        e_i[i] = 1.0;
        let w = match btran_refined(&mut lu, sp, &basis.basic_vars, &e_i, tol) {
            Some(w) => w,
            None => {
                crate::profile::incr(crate::profile::Ctr::SepGomoryBtranFail);
                continue;
            }
        };

        // Global complementation: β = x_B + Σ α_j x̃_j with x̃_j = |x*_j − bound_j|
        // at the GLOBAL bound on the LP's side. With `global = None` every x̃_j is 0
        // and β is x_B exactly (the historical f0).
        let f0 = match global {
            None => xb[i] - xb[i].floor(),
            Some((_, _, n_global)) => {
                let mut beta = xb[i];
                // Magnitude of the complementation terms: the absolute rounding error they
                // add to β scales with it, so beyond `GLOBAL_BETA_MAG_MAX` the fractional
                // part f0 is not resolved and the cut may be invalid (PR #2196 review,
                // item 3). `x_B` itself is excluded, as on the local path.
                let mut beta_mag = 0.0f64;
                let mut refuse = false;
                for j in 0..n {
                    if basis.col_status[j] == BASIC {
                        continue;
                    }
                    let at_upper = basis.col_status[j] == AT_UPPER;
                    let local = if at_upper { u[j] } else { l[j] };
                    let bound = if at_upper { cu[j] } else { cl[j] };
                    let abar: f64 = sp.dot(j, &w);
                    if abar == 0.0 {
                        continue;
                    }
                    if j >= n_global || bound.abs() >= 1e20 {
                        refuse = true;
                        break;
                    }
                    let xt = if at_upper {
                        bound - local
                    } else {
                        local - bound
                    };
                    let alpha = if at_upper { -abar } else { abar };
                    beta += alpha * xt;
                    beta_mag += (alpha * xt).abs();
                }
                if refuse || !beta.is_finite() {
                    crate::profile::incr(crate::profile::Ctr::SepGomoryGlobalRefused);
                    continue;
                }
                if beta_mag > GLOBAL_BETA_MAG_MAX {
                    crate::profile::incr(crate::profile::Ctr::SepGomoryGlobalBetaUnresolved);
                    continue;
                }
                beta - beta.floor()
            }
        };
        if global.is_some() && !(FRAC_MIN..=1.0 - FRAC_MIN).contains(&f0) {
            continue;
        }

        // GMI cut Σ ψ_j x̃_j ≥ 1, accumulated into original-variable space.
        let mut coeffs = vec![0.0_f64; n];
        let mut rhs = 1.0_f64;
        let mut max_c = 0.0_f64;
        let mut min_c = f64::INFINITY;
        let mut ok = true;
        for j in 0..n {
            if basis.col_status[j] == BASIC {
                continue;
            }
            // ā_j = w · A[:,j], snapped to the nearest integer when very close
            // (the refined value is accurate, so this only removes ulp noise).
            // Sparse dot over column j's nonzeros (was an O(m) dense scan per j).
            let abar: f64 = sp.dot(j, &w);
            // #1236: `abar` is NOT snapped to the nearest integer. The header used
            // to justify snapping as collapsing a "flip error", but psi(f) is
            // continuous and zero at every integer, so there is no flip to
            // collapse -- and snapping a CONTINUOUS column's abar to 0 zeroes its
            // psi, which silently deletes `psi * xtilde_j` from the LHS of a `>=`
            // cut. Deleting a nonnegative term from the LHS of `sum psi xtilde >= 1`
            // STRENGTHENS it, so the cut can exclude feasible integer points.
            // Measured on the fac2 OA master: two such terms sat on continuous
            // slacks with u = 1e20 and xtilde ~ 7e7 / 2.3e8 at the optimum, so
            // coefficients of ~1e-10 were worth 0.47 and 0.025 against a rhs of 1.
            // The resulting cut cut off the optimum and the driver certified a
            // false `optimal` 7839 above it. `snap_eligible` survives only as the
            // "this integer column's abar is at an integer" flag used below.
            let snap_eligible = (abar - abar.round()).abs() < SNAP_TOL;
            // Nonbasic at its upper bound uses x̃_j = u_j − x_j (sign flip).
            let at_upper = basis.col_status[j] == AT_UPPER;
            let alpha = if at_upper { -abar } else { abar };

            // The integer GMI strengthening is valid only when the nonbasic
            // integer variable is pinned at an INTEGER bound, so the shifted
            // x̃_j = x_j − l_j (or u_j − x_j) takes integer values. If presolve
            // (coefficient strengthening / implied bounds) handed it a
            // fractional bound, that premise fails and the integer ψ can cut the
            // true optimum — fall back to the continuous formula, which is
            // always valid (a weaker but sound cut).
            let pinned = if at_upper { cu[j] } else { cl[j] };
            let use_integer = integrality[j] && (pinned - pinned.round()).abs() <= tol;
            let psi = if use_integer {
                let fj = alpha - alpha.floor();
                if fj <= f0 {
                    fj / f0
                } else {
                    (1.0 - fj) / (1.0 - f0)
                }
            } else if alpha >= 0.0 {
                alpha / f0
            } else {
                -alpha / (1.0 - f0)
            };
            if !psi.is_finite() {
                ok = false;
                break;
            }
            if psi == 0.0 {
                continue;
            }
            // #1236: a tiny term `psi * xtilde_j` (psi >= 0, xtilde >= 0) may only
            // leave the LHS of `sum psi xtilde >= 1` if its MAXIMUM over the box,
            // `psi * (u_j - l_j)`, is charged to the rhs -- that is a relaxation,
            // which is always sound. Dropping it outright is a strengthening and is
            // not. With an infinite range the term cannot move at all, so it is kept
            // exactly. The bound is tested against the 1e20 sentinel directly, never
            // via a product: for a small `psi` the product of an unbounded bound is
            // an ordinary finite number, which is exactly how this layer has
            // produced a false certified `optimal` before (CLAUDE.md, INF note).
            //
            // `substitute_slacks_to_structural` in the driver already applies this
            // rule correctly (it moves a small term to the rhs at its maximising
            // bound and refuses the cut when that pin is infinite); the separator
            // was the inconsistent one.
            // #1236 (review finding 3): gate `snap_eligible` on `use_integer`, not on
            // `integrality[j]`. When an integer column is pinned at a FRACTIONAL
            // bound the integer premise fails and the block above falls back to the
            // CONTINUOUS formula, where an integral `abar` gives psi = |alpha| / f0
            // -- which is >= 1/f0 and can be in the thousands, the opposite of tiny.
            // Calling that "tiny" excluded a large coefficient from the dynamism
            // gate below, so the gate silently stopped seeing the worst term in the
            // cut. Under `use_integer` an integral `abar` really does give psi = 0
            // (caught above) and a near-integral one gives ulp noise, which is the
            // only case the flag was ever meant to name.
            let tiny = psi.abs() <= tol || (use_integer && snap_eligible);
            if tiny {
                let range = cu[j] - cl[j];
                // Charge the term to the rhs only when its MAXIMUM contribution
                // over the box is itself negligible (<= tol): a bounded
                // weakening. Otherwise keep the exact coefficient.
                if cu[j] < 1e20 && cl[j] > -1e20 && range.is_finite() && psi * range <= tol {
                    rhs -= psi * range;
                    continue;
                }
            }
            if at_upper {
                coeffs[j] = -psi;
                rhs -= psi * cu[j];
            } else {
                coeffs[j] = psi;
                rhs += psi * cl[j];
            }
            max_c = max_c.max(psi.abs());
            // #1236: a tiny coefficient kept EXACTLY -- because its range was
            // unbounded and it therefore could not be charged to the rhs -- does not
            // count toward the dynamism gate. Counting it refuses nearly every cut
            // on a big-M master, which is sound but measurably harmful: on the
            // rsyn0830m master the refusals cost 6408 -> 17804 nodes. The cut itself
            // is exact either way; where a bound-based cleanup exists
            // (`substitute_slacks_to_structural`) it weakens these terms soundly or
            // refuses the cut, and where it does not (the convex kernel's
            // `substitute_slacks`) the row is still exact -- strictly better than the
            // deletion this replaces, which was unsound.
            if !tiny {
                min_c = min_c.min(psi.abs());
            }
        }

        // #1236 (review finding 2): `min_c` stays INFINITY when EVERY kept
        // coefficient was tiny, and `max_c / INFINITY == 0` slips past the dynamism
        // gate while `max_c != 0.0` slips past the emptiness gate -- so a cut whose
        // coefficients are all ~1e-12 would be emitted. Before the drop-term fix
        // above, such terms were skipped outright, `max_c` stayed 0.0 and the cut
        // was refused here; excluding tiny terms from `min_c` removed that refusal
        // as a side effect. `min_c.is_finite()` is exactly "at least one kept
        // coefficient was above tolerance", which restores it.
        //
        // It matters because the cut is not just weak, it is a FALSE PRUNE waiting
        // to happen: `sum 1e-12 xtilde >= 1` reads as `0 >= 1` at a 1e-9 LP
        // feasibility tolerance, so the node is fathomed as infeasible although it
        // contains feasible points, and that feeds a certified bound. The driver
        // path is shielded by `substitute_slacks_to_structural`, but the convex
        // kernel's `substitute_slacks` drops only exact zeros and would pass it
        // straight to the node LP. Refusing a cut is always sound.
        let all_kept_coefficients_are_tiny = !min_c.is_finite();
        // highs-parity P4.3 entry: which numerical gate refused the cut (counters
        // only; the decision is unchanged).
        let reject = if !ok {
            Some(crate::profile::Ctr::SepGomoryRejectNonFinite)
        } else if max_c == 0.0 || all_kept_coefficients_are_tiny {
            Some(crate::profile::Ctr::SepGomoryRejectTiny)
        } else if max_c > MAX_ABS_COEFF || rhs.abs() > MAX_ABS_COEFF {
            Some(crate::profile::Ctr::SepGomoryRejectMagnitude)
        } else if min_c > 0.0 && max_c / min_c > max_dynamism {
            Some(crate::profile::Ctr::SepGomoryRejectDynamism)
        } else {
            None
        };
        if let Some(ctr) = reject {
            crate::profile::incr(ctr);
            continue;
        }
        // #1236 review finding 4: record what was emitted, not just that it was.
        let nnz = coeffs.iter().filter(|v| **v != 0.0).count();
        crate::profile::incr(crate::profile::Ctr::SepGomoryCutsEmitted);
        crate::profile::incr_by(crate::profile::Ctr::SepGomoryCutNnz, nnz as u64);
        cuts.push(GomoryCut { coeffs, rhs });
    }
    cuts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::basis::{recover_basis, AT_LOWER};

    /// Every refusal inside `separate_gomory_cols` must leave a counter behind.
    ///
    /// This function had five uncounted early returns, so "no cuts separated" and
    /// "never looked at a single tableau row" were the same observation. That is
    /// the CLAUDE.md §6 failure mode, and it is what hid the short-basis export
    /// defect (A0'.1): on `neos-2624317-amur` the cold root LP exported 338 basic
    /// variables for m = 342, every separation call bailed at the length guard,
    /// and the panel reported `0/0` cuts — read as "nothing to cut".
    ///
    /// Asserted as *deltas* around each call, and with the call counter as the
    /// denominator, so the test itself cannot pass vacuously.
    #[test]
    fn a_short_basis_refusal_is_counted_not_silent() {
        let _guard = crate::profile::test_guard();
        crate::profile::set_enabled(true);

        // x0 + x1 + s = 1.5, x0,x1 ∈ {0,1} relaxed, s ≥ 0 — the LP of the first
        // test, whose complete basis is known to separate exactly one GMI cut.
        let a = [1.0, 1.0, 1.0];
        let l = [0.0, 0.0, 0.0];
        let u = [1.0, 1.0, f64::INFINITY];
        let b = [1.5];
        let integrality = [true, true, false];
        let sp = crate::lp::simplex::sparse::SparseCols::from_dense(&a, 1, 3);

        let calls = |()| crate::profile::counter(crate::profile::Ctr::SepGomoryCalls);
        let short = |()| crate::profile::counter(crate::profile::Ctr::SepGomoryShortBasis);

        let completed = |()| crate::profile::counter(crate::profile::Ctr::SepGomoryBasisCompleted);

        // Arm 1: a basis one column short of `m`. Whatever happens to it -- refused
        // (`DISCOPT_MILP_ROOT_GMI_REPAIR=0`) or completed with a unit column (the
        // default since P4.4) -- exactly one of the two counters must say so. Its only
        // basic slot is then the appended unit column, which is continuous: no cut.
        let c0 = calls(());
        let s0 = short(());
        let k0 = completed(());
        let stub = Basis {
            col_status: vec![AT_LOWER, AT_LOWER, AT_LOWER],
            basic_vars: vec![], // 0 != m = 1
        };
        let cuts = separate_gomory_cols(&sp, 1, 3, &l, &u, &b, &stub, &integrality, 1e-7, 1e9);
        assert!(cuts.is_empty(), "no integer basic variable: no cut");
        assert!(calls(()) - c0 >= 1, "the call counter is the denominator");
        assert_eq!(
            (short(()) - s0) + (completed(()) - k0),
            1,
            "a short basis must be counted as refused or as completed"
        );

        // Arm 2: the complete basis. The refusal counter must NOT move — without
        // this arm the test would pass on a counter that fires unconditionally.
        let x = [1.0, 0.5, 0.0];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 3,
            c: &[0.0, 0.0, 0.0],
            l: &l,
            u: &u,
        };
        let good = recover_basis(&x, &lp, 1e-7).expect("basis");
        let c1 = calls(());
        let s1 = short(());
        let cuts = separate_gomory_cols(&sp, 1, 3, &l, &u, &b, &good, &integrality, 1e-7, 1e9);
        assert_eq!(cuts.len(), 1, "the complete basis still separates its cut");
        assert_eq!(calls(()) - c1, 1);
        assert_eq!(
            short(()) - s1,
            0,
            "a complete basis must not count a refusal"
        );

        crate::profile::set_enabled(false);
    }

    /// #1236 review finding 2: a cut whose every coefficient is tiny must be REFUSED.
    ///
    /// Such a cut is mathematically valid and numerically a trap: at a 1e-9 LP
    /// feasibility tolerance `2e-12 y + 2e-12 s >= 1` reads as `0 >= 1`, so the node
    /// is fathomed as infeasible although it contains feasible points -- a FALSE
    /// PRUNE feeding a certified bound.
    ///
    /// Before the drop-term fix these terms were skipped outright, `max_c` stayed
    /// 0.0 and the `max_c == 0.0` arm refused the cut. Keeping them (which
    /// soundness requires) while excluding them from `min_c` left `min_c` at
    /// INFINITY, and `max_c / INFINITY == 0` passes the dynamism gate.
    #[test]
    fn a_cut_whose_every_coefficient_is_tiny_is_refused() {
        // 1e12·x0 + y + s = 5e11, x0 ∈ {0,1} relaxed, y,s continuous on [0, 1e20].
        // At x0 = 0.5 the basic row scales by w = 1e-12, so BOTH nonbasic columns
        // get abar = 1e-12 -> psi = 2e-12, and neither can be charged to the rhs
        // because its range is unbounded. Every kept coefficient is tiny.
        let a = [1e12, 1.0, 1.0];
        let c = [0.0, 0.0, 0.0];
        let l = [0.0, 0.0, 0.0];
        let u = [1.0, 1e20, 1e20];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 3,
            c: &c,
            l: &l,
            u: &u,
        };
        let b = [5e11];
        let x = [0.5, 0.0, 0.0];
        let integrality = [true, false, false];

        let basis = recover_basis(&x, &lp, 1e-7).expect("basis");
        assert_eq!(
            basis.basic_vars,
            vec![0],
            "x0 must be the basic fractional var"
        );

        let cuts = separate_gomory(&lp, &b, &basis, &integrality, 1e-7, 1e9);

        // The assertion is the SPEC, not the count: whatever is emitted must be
        // enforceable at LP tolerance. A cut whose largest coefficient is below the
        // feasibility tolerance cannot be, whatever its rhs says.
        let mut checked = 0usize;
        for cut in &cuts {
            let max_c = cut.coeffs.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            checked += 1;
            assert!(
                max_c > 1e-7,
                "emitted a cut no LP can enforce: max|coeff| = {max_c:e} vs rhs {} \
                 -- at a 1e-9 feasibility tolerance this reads as `0 >= 1` and \
                 fathoms a node that contains feasible points",
                cut.rhs
            );
        }
        assert!(
            cuts.is_empty(),
            "all {} coefficients are ~2e-12; the cut must be refused outright \
             (checked {checked} emitted cuts)",
            cuts.len()
        );
    }

    /// #1236 review finding 3: `snap_eligible` may only be consulted on the
    /// `use_integer` branch.
    ///
    /// An integer column pinned at a FRACTIONAL bound falls back to the continuous
    /// formula, where an integral `abar` yields psi = |alpha| / f0 >= 1 -- a LARGE
    /// coefficient. Flagging it `tiny` (because the column is integral and its abar
    /// sits on an integer) excluded it from `min_c`, so the dynamism gate stopped
    /// seeing the smallest coefficient in the cut and waved through a cut with a
    /// 1e6 spread.
    #[test]
    fn an_integer_column_pinned_at_a_fractional_bound_is_not_tiny() {
        // x0 + x1 + 1e6·s = 1; x0 ∈ {0,1} basic at 0.5; x1 INTEGER pinned at the
        // fractional bound l = 0.5; s continuous. w = 1, so abar_x1 = 1 (integral,
        // hence `snap_eligible`) while psi_x1 = 1 / 0.5 = 2, and psi_s = 2e6.
        let a = [1.0, 1.0, 1e6];
        let c = [0.0, 0.0, 0.0];
        let l = [0.0, 0.5, 0.0];
        let u = [1.0, 2.5, f64::INFINITY];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 3,
            c: &c,
            l: &l,
            u: &u,
        };
        let b = [1.0];
        let x = [0.5, 0.5, 0.0];
        let integrality = [true, true, false];

        let basis = recover_basis(&x, &lp, 1e-7).expect("basis");
        assert_eq!(basis.basic_vars, vec![0]);

        // Arm 1 (loose gate): the cut IS emitted, and x1's coefficient is the large
        // continuous-formula psi = 2 -- not zero, and not tiny.
        let cuts = separate_gomory(&lp, &b, &basis, &integrality, 1e-7, 1e9);
        assert_eq!(
            cuts.len(),
            1,
            "a loose dynamism gate must still emit the cut"
        );
        let cut = &cuts[0];
        assert!(
            (cut.coeffs[1] - 2.0).abs() < 1e-9,
            "x1's coefficient must be the continuous-formula psi = 2, got {}",
            cut.coeffs[1]
        );
        assert!(
            dot(&cut.coeffs, &x) < cut.rhs - 1e-6,
            "must cut off the vertex"
        );
        // The only integer-feasible point: x1 ∈ {1,2}, and x1 = 2 forces s < 0.
        let pt = [0.0, 1.0, 0.0];
        assert!(
            dot(&cut.coeffs, &pt) >= cut.rhs - 1e-6,
            "cut excludes feasible point {pt:?}"
        );

        // Arm 2 (tight gate): the true spread is max_c / min_c = 2e6 / 2 = 1e6, so a
        // gate of 1e3 must refuse it. Treating x1 as `tiny` hid it from `min_c`,
        // leaving min_c = max_c = 2e6, a reported spread of 1.0, and the cut passed.
        let cuts = separate_gomory(&lp, &b, &basis, &integrality, 1e-7, 1e3);
        assert!(
            cuts.is_empty(),
            "dynamism 1e6 must be refused by a 1e3 gate; the gate saw {:?}",
            cuts.iter().map(|c| c.coeffs.clone()).collect::<Vec<_>>()
        );
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn gmi_cut_separates_fractional_vertex_and_is_valid() {
        // x0 + x1 + s = 1.5, x0,x1 ∈ {0,1} (relaxed [0,1]), s >= 0.
        // Vertex (1, 0.5, 0): x1 basic & fractional → GMI cut 2 s ≥ 1
        // (equivalently x0 + x1 ≤ 1).
        let a = [1.0, 1.0, 1.0];
        let c = [0.0, 0.0, 0.0];
        let l = [0.0, 0.0, 0.0];
        let u = [1.0, 1.0, f64::INFINITY];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 3,
            c: &c,
            l: &l,
            u: &u,
        };
        let b = [1.5];
        let x = [1.0, 0.5, 0.0];
        let integrality = [true, true, false];

        let basis = recover_basis(&x, &lp, 1e-7).expect("basis");
        assert_eq!(basis.basic_vars, vec![1]); // x1 is the basic (free) var

        let cuts = separate_gomory(&lp, &b, &basis, &integrality, 1e-7, 1e9);
        assert_eq!(cuts.len(), 1);
        let cut = &cuts[0];
        assert!(dot(&cut.coeffs, &x) < cut.rhs - 1e-6); // cuts off the vertex
        for b0 in 0..=1 {
            for b1 in 0..=1 {
                let s = 1.5 - b0 as f64 - b1 as f64;
                if s < -1e-9 {
                    continue;
                }
                let pt = [b0 as f64, b1 as f64, s];
                assert!(
                    dot(&cut.coeffs, &pt) >= cut.rhs - 1e-6,
                    "cut excludes feasible point {pt:?}"
                );
            }
        }
    }

    #[test]
    fn gmi_cut_valid_for_general_integer_at_upper_bound() {
        // 2x0 + 2x1 + s = 5, x0,x1 integer in [0,2], s >= 0. Vertex
        // (0.5, 2, 0): x0 basic & fractional, x1 nonbasic at its UPPER bound 2
        // (the general-integer case that broke a naive, unrefined GMI). The cut
        // is s ≥ 1, i.e. x0 + x1 ≤ 2.
        let a = [2.0, 2.0, 1.0];
        let c = [0.0, 0.0, 0.0];
        let l = [0.0, 0.0, 0.0];
        let u = [2.0, 2.0, f64::INFINITY];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 3,
            c: &c,
            l: &l,
            u: &u,
        };
        let b = [5.0];
        let x = [0.5, 2.0, 0.0];
        let integrality = [true, true, false];

        let basis = recover_basis(&x, &lp, 1e-7).expect("basis");
        assert_eq!(basis.basic_vars, vec![0]);

        let cuts = separate_gomory(&lp, &b, &basis, &integrality, 1e-7, 1e9);
        assert_eq!(cuts.len(), 1);
        let cut = &cuts[0];
        assert!(dot(&cut.coeffs, &x) < cut.rhs - 1e-6); // separates (0.5, 2, 0)
        for i0 in 0..=2 {
            for i1 in 0..=2 {
                let s = 5.0 - 2.0 * i0 as f64 - 2.0 * i1 as f64;
                if s < -1e-9 {
                    continue;
                }
                let pt = [i0 as f64, i1 as f64, s];
                assert!(
                    dot(&cut.coeffs, &pt) >= cut.rhs - 1e-6,
                    "cut excludes feasible point {pt:?}"
                );
            }
        }
    }

    #[test]
    fn no_cut_when_vertex_is_integral() {
        // Same first system, integral vertex (1, 0, 0.5): the only basic var is
        // the continuous slack, so no GMI cut.
        let a = [1.0, 1.0, 1.0];
        let c = [0.0, 0.0, 0.0];
        let l = [0.0, 0.0, 0.0];
        let u = [1.0, 1.0, f64::INFINITY];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 3,
            c: &c,
            l: &l,
            u: &u,
        };
        let b = [1.5];
        let x = [1.0, 0.0, 0.5]; // s basic = 0.5, but s is continuous
        let integrality = [true, true, false];
        let basis = recover_basis(&x, &lp, 1e-7).expect("basis");
        let cuts = separate_gomory(&lp, &b, &basis, &integrality, 1e-7, 1e9);
        assert!(cuts.is_empty());
    }

    #[test]
    fn gmi_cut_two_constraints_is_valid() {
        // 2x0 + x1 + s0 = 3,  x0 + 2x1 + s1 = 3,  x0,x1 integer >= 0, s >= 0.
        // Vertex x0 = 1.5, x1 = 0, s0 = 0, s1 = 1.5 (x0 basic & fractional).
        let n = 4;
        let m = 2;
        let a = [
            2.0, 1.0, 1.0, 0.0, // 2x0 + x1 + s0 = 3
            1.0, 2.0, 0.0, 1.0, // x0 + 2x1 + s1 = 3
        ];
        let c = [0.0; 4];
        let l = [0.0; 4];
        let u = [f64::INFINITY; 4];
        let lp = LpView {
            a: &a,
            m,
            n,
            c: &c,
            l: &l,
            u: &u,
        };
        let b = [3.0, 3.0];
        let x = [1.5, 0.0, 0.0, 1.5];
        let integrality = [true, true, false, false];

        let basis = recover_basis(&x, &lp, 1e-7).expect("basis");
        let cuts = separate_gomory(&lp, &b, &basis, &integrality, 1e-7, 1e9);
        assert!(!cuts.is_empty(), "expected a cut from fractional x0");

        for cut in &cuts {
            assert!(dot(&cut.coeffs, &x) < cut.rhs - 1e-6);
            for i0 in 0..=3 {
                for i1 in 0..=3 {
                    let (x0, x1) = (i0 as f64, i1 as f64);
                    let s0 = 3.0 - 2.0 * x0 - x1;
                    let s1 = 3.0 - x0 - 2.0 * x1;
                    if s0 < -1e-9 || s1 < -1e-9 {
                        continue;
                    }
                    let pt = [x0, x1, s0, s1];
                    assert!(
                        dot(&cut.coeffs, &pt) >= cut.rhs - 1e-6,
                        "cut excludes feasible point {pt:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn gmi_cut_valid_when_integer_var_has_fractional_bound() {
        // x0 + x1 + s = 2, x0 integer in [0,2], x1 integer with a *fractional*
        // lower bound 0.5 (as presolve coefficient-strengthening / implied bounds
        // could produce), s >= 0. Vertex (1.5, 0.5, 0): x0 basic & fractional,
        // x1 nonbasic at its fractional lower bound. The integer GMI ψ assumes
        // the nonbasic integer sits at an integer bound, which is FALSE here; the
        // guard must fall back to the continuous ψ so the cut stays valid for
        // every feasible integer point (x1 ∈ {1,2}, since x1 is integer ≥ 0.5).
        let a = [1.0, 1.0, 1.0];
        let c = [0.0, 0.0, 0.0];
        let l = [0.0, 0.5, 0.0];
        let u = [2.0, 2.0, f64::INFINITY];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 3,
            c: &c,
            l: &l,
            u: &u,
        };
        let b = [2.0];
        let x = [1.5, 0.5, 0.0];
        let integrality = [true, true, false];

        let basis = recover_basis(&x, &lp, 1e-7).expect("basis");
        assert_eq!(basis.basic_vars, vec![0]); // x0 is the fractional basic var

        let cuts = separate_gomory(&lp, &b, &basis, &integrality, 1e-7, 1e9);
        for cut in &cuts {
            assert!(
                dot(&cut.coeffs, &x) < cut.rhs - 1e-6,
                "cut must separate vertex"
            );
            // Feasible integer points: x1 integer ≥ 0.5 → x1 ∈ {1, 2}.
            for x0i in 0..=2 {
                for x1i in 1..=2 {
                    let s = 2.0 - x0i as f64 - x1i as f64;
                    if s < -1e-9 {
                        continue;
                    }
                    let pt = [x0i as f64, x1i as f64, s];
                    assert!(
                        dot(&cut.coeffs, &pt) >= cut.rhs - 1e-6,
                        "cut excludes feasible integer point {pt:?}"
                    );
                }
            }
        }
    }

    /// #1236 regression: a continuous column with a TINY tableau coefficient and a
    /// huge range must not be silently dropped from a GMI cut. Row
    /// `x0 + 1e-10*y + s = 0.5`, `x0 ∈ {0,1}`, `y ∈ [0, 1e20]`, `s >= 0`; vertex
    /// (0.5, 0, 0). Snapping `abar_y = 1e-10` to 0 (or skipping `psi_y = 2e-10`)
    /// yields `2 s >= 1`, which cuts off the feasible integer point
    /// `(0, 5e9, 0)`. The exact cut `2e-10 y + 2 s >= 1` holds there with equality.
    #[test]
    fn gmi_tiny_coefficient_on_wide_continuous_column_is_not_dropped() {
        let a = [1.0, 1e-10, 1.0];
        let c = [0.0, 0.0, 0.0];
        let l = [0.0, 0.0, 0.0];
        let u = [1.0, 1e20, f64::INFINITY];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 3,
            c: &c,
            l: &l,
            u: &u,
        };
        let b = [0.5];
        let x = [0.5, 0.0, 0.0];
        let integrality = [true, false, false];
        let basis = recover_basis(&x, &lp, 1e-9).expect("basis");
        assert_eq!(basis.basic_vars, vec![0]);
        let cuts = separate_gomory(&lp, &b, &basis, &integrality, 1e-9, 1e12);
        assert_eq!(cuts.len(), 1, "one GMI cut off the fractional basic x0");
        let cut = &cuts[0];
        assert!(
            dot(&cut.coeffs, &x) < cut.rhs - 1e-6,
            "cut must separate the vertex"
        );
        let pt = [0.0, 5e9, 0.0]; // feasible: 0 + 1e-10*5e9 + 0 = 0.5
        assert!(
            dot(&cut.coeffs, &pt) >= cut.rhs - 1e-9,
            "cut excludes feasible integer point {pt:?}: lhs={} rhs={}",
            dot(&cut.coeffs, &pt),
            cut.rhs
        );
    }

    // ---- highs-parity P4.4: completing a short basis ----------------------------

    #[test]
    fn unmatched_rows_finds_the_redundant_row() {
        // Rows 0 and 1 are identical; column 1 covers both, column 2 only row 0.
        let a = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]; // 2x3 row-major
        let sp = SparseCols::from_dense(&a, 2, 3);
        assert_eq!(unmatched_rows(&sp, 2, &[1]).unwrap().len(), 1);
        assert_eq!(
            unmatched_rows(&sp, 2, &[1, 2]).unwrap(),
            Vec::<usize>::new()
        );
        // Two columns with the same single-row support cannot both be matched.
        let b = [1.0, 1.0, 0.0, 0.0];
        let sp2 = SparseCols::from_dense(&b, 2, 2);
        assert!(unmatched_rows(&sp2, 2, &[0, 1]).is_none());
        // An augmenting path is needed: col0 rows {0,1}, col1 row {0}.
        let c = [1.0, 1.0, 1.0, 0.0];
        let sp3 = SparseCols::from_dense(&c, 2, 2);
        assert_eq!(
            unmatched_rows(&sp3, 2, &[0, 1]).unwrap(),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn a_short_basis_on_a_redundant_row_is_completed_and_its_cut_is_valid() {
        // r0: x0 + x1 + s = 1.5 and r1, an exact duplicate (no slack of its own),
        // x0, x1 in {0,1}, s >= 0. Rank 1, so a basis of real columns is at most 1
        // long: the primal export leaves exactly this kind of short basis.
        let a = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let sp = SparseCols::from_dense(&a, 2, 3);
        let (l, u, b) = ([0.0, 0.0, 0.0], [1.0, 1.0, 1e20], [1.5, 1.5]);
        let integrality = [true, true, false];
        let short = Basis {
            col_status: vec![AT_UPPER, BASIC, AT_LOWER], // x = (1, 0.5, 0)
            basic_vars: vec![1],
        };
        let cuts =
            separate_on_completed_basis(&sp, 2, 3, &l, &u, &b, &short, &integrality, 1e-7, 1e9)
                .expect("completion must succeed");
        assert!(
            !cuts.is_empty(),
            "x1 = 0.5 is fractional: a GMI cut must be separated"
        );
        let feasible = [[0.0, 0.0, 1.5], [1.0, 0.0, 0.5], [0.0, 1.0, 0.5]];
        for cut in &cuts {
            assert_eq!(cut.coeffs.len(), 3, "truncated to the original columns");
            assert!(
                dot(&cut.coeffs, &[1.0, 0.5, 0.0]) < cut.rhs - 1e-9,
                "cuts off the vertex"
            );
            for x in &feasible {
                assert!(
                    dot(&cut.coeffs, x) >= cut.rhs - 1e-9,
                    "cut {cut:?} removes {x:?}"
                );
            }
        }
    }

    /// Exact-enumeration soundness on seeded random MILPs whose cold LP export is
    /// SHORT (a redundant row equal to the sum of two others): every cut separated
    /// off the completed basis must hold at every integer-feasible point.
    #[test]
    fn completed_basis_cuts_never_remove_an_integer_feasible_point() {
        use crate::lp::simplex::{solve_lp_cols_scaled, LpStatus, SimplexOptions};
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut rnd = |k: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % k
        };
        let (mut short_seen, mut cuts_checked, mut points_checked) = (0, 0, 0);
        for _trial in 0..400 {
            let ni = 3usize; // integer columns, each in [0, 3]
                             // rows r0, r1: a_r x + s_r = b_r (slacks s0, s1 >= 0); r2 = r0 + r1, slackless.
            let mut arow = [[0.0f64; 3]; 2];
            let mut bv = [0.0f64; 2];
            for r in 0..2 {
                for j in 0..ni {
                    arow[r][j] = (rnd(7) as f64) - 2.0;
                }
                bv[r] = (rnd(9) as f64) + 0.5;
            }
            let (m, n) = (3usize, ni + 2);
            let mut dense = vec![0.0f64; m * n];
            for j in 0..ni {
                dense[j] = arow[0][j];
                dense[n + j] = arow[1][j];
                dense[2 * n + j] = arow[0][j] + arow[1][j];
            }
            dense[ni] = 1.0;
            dense[n + ni + 1] = 1.0;
            dense[2 * n + ni] = 1.0;
            dense[2 * n + ni + 1] = 1.0;
            let b = [bv[0], bv[1], bv[0] + bv[1]];
            let mut l = vec![0.0; n];
            let mut u = vec![3.0; n];
            l[ni] = 0.0;
            l[ni + 1] = 0.0;
            u[ni] = 1e20;
            u[ni + 1] = 1e20;
            let c: Vec<f64> = (0..n)
                .map(|j| if j < ni { (rnd(9) as f64) - 4.0 } else { 0.0 })
                .collect();
            let integrality: Vec<bool> = (0..n).map(|j| j < ni).collect();
            let sp = SparseCols::from_dense(&dense, m, n);
            let sol =
                solve_lp_cols_scaled(sp.clone(), m, n, &c, &l, &u, &b, &SimplexOptions::default());
            if sol.status != LpStatus::Optimal || sol.basis.basic_vars.len() >= m {
                continue;
            }
            short_seen += 1;
            let Some(cuts) = separate_on_completed_basis(
                &sp,
                m,
                n,
                &l,
                &u,
                &b,
                &sol.basis,
                &integrality,
                1e-7,
                1e9,
            ) else {
                continue;
            };
            for cut in &cuts {
                cuts_checked += 1;
                for x0 in 0..=3 {
                    for x1 in 0..=3 {
                        for x2 in 0..=3 {
                            let xi = [x0 as f64, x1 as f64, x2 as f64];
                            let s0 = bv[0] - dot(&arow[0], &xi);
                            let s1 = bv[1] - dot(&arow[1], &xi);
                            if s0 < 0.0 || s1 < 0.0 {
                                continue;
                            }
                            let x = [xi[0], xi[1], xi[2], s0, s1];
                            points_checked += 1;
                            assert!(
                                dot(&cut.coeffs, &x) >= cut.rhs - 1e-7 * (1.0 + cut.rhs.abs()),
                                "cut {cut:?} removes feasible {x:?}"
                            );
                        }
                    }
                }
            }
        }
        assert!(
            short_seen > 0,
            "no trial produced a short basis -- the test measured nothing"
        );
        assert!(
            cuts_checked > 0 && points_checked > 0,
            "{short_seen} {cuts_checked} {points_checked}"
        );
        eprintln!("short={short_seen} cuts={cuts_checked} points={points_checked}");
    }

    /// PR #2196 review item 3: a globally complemented row whose complementation terms
    /// `Σ|ᾱ_j·x̃_j|` exceed `GLOBAL_BETA_MAG_MAX` is refused (f0 unresolved), while the
    /// same row with a short local-to-global distance still yields a cut (control).
    #[test]
    fn global_gmi_refuses_a_row_complemented_over_a_huge_distance() {
        use crate::lp::simplex::{solve_lp_cols_scaled, LpStatus, SimplexOptions};
        // min x0  s.t.  2 x1 - x0 = 0.5,  x0, x1 integer, global box [0, 1e9].
        // At a node with x0 >= L the LP vertex is x0 = L (nonbasic at its LOCAL
        // lower bound, global lower 0, so x~ = L) and x1 = (L + 0.5) / 2 basic.
        let dense = [-1.0, 2.0];
        let sp = SparseCols::from_dense(&dense, 1, 2);
        let (c, b) = ([1.0, 0.0], [0.5]);
        let (gl, gu) = (vec![0.0, 0.0], vec![1e9, 1e9]);
        let integrality = [true, true];
        let cuts_at = |lower: f64| {
            let (l, u) = (vec![lower, 0.0], gu.clone());
            let sol =
                solve_lp_cols_scaled(sp.clone(), 1, 2, &c, &l, &u, &b, &SimplexOptions::default());
            assert_eq!(sol.status, LpStatus::Optimal);
            assert!(
                (sol.x[0] - lower).abs() < 1e-9,
                "x0 not at its local lower bound"
            );
            separate_gomory_cols_global(
                &sp,
                1,
                2,
                &l,
                &u,
                &gl,
                &gu,
                2,
                &b,
                &sol.basis,
                &integrality,
                1e-7,
                1e12,
            )
        };
        let near = cuts_at(4.0); // x~ = 4: term 2, well inside the bound
        assert!(
            !near.is_empty(),
            "control: the short-distance row produced no cut"
        );
        let far = cuts_at(5e8); // x~ = 5e8: term 2.5e8 > GLOBAL_BETA_MAG_MAX
        assert!(
            far.is_empty(),
            "a row complemented over 5e8 was not refused: {far:?}"
        );
    }

    /// `separate_gomory_cols_global` (highs-parity D2): cuts separated at a node with
    /// LOCAL bounds but complemented at the GLOBAL bounds must hold at every integer-
    /// feasible point of the GLOBAL box. Control arm: the plain separator on the same
    /// local LPs is NOT globally valid, and the check must catch it at least once,
    /// otherwise this test could not tell a valid global cut from an invalid one.
    #[test]
    fn global_gmi_cuts_hold_on_the_whole_global_box() {
        use crate::lp::simplex::{solve_lp_cols_scaled, LpStatus, SimplexOptions};
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = |k: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % k
        };
        let (mut global_cuts, mut points, mut local_invalid) = (0usize, 0usize, 0usize);
        for _trial in 0..600 {
            let ni = 3usize;
            let mut arow = [[0.0f64; 3]; 2];
            let mut bv = [0.0f64; 2];
            for r in 0..2 {
                for j in 0..ni {
                    arow[r][j] = (rnd(7) as f64) - 2.0;
                }
                bv[r] = (rnd(9) as f64) + 0.5;
            }
            let (m, n) = (2usize, ni + 2);
            let mut dense = vec![0.0f64; m * n];
            for j in 0..ni {
                dense[j] = arow[0][j];
                dense[n + j] = arow[1][j];
            }
            dense[ni] = 1.0;
            dense[n + ni + 1] = 1.0;
            let b = [bv[0], bv[1]];
            let gl = vec![0.0; n];
            let mut gu = vec![3.0; n];
            gu[ni] = 1e20;
            gu[ni + 1] = 1e20;
            // A node box: one integer column branched up or down.
            let (mut l, mut u) = (gl.clone(), gu.clone());
            let j0 = rnd(ni as u64) as usize;
            if rnd(2) == 0 {
                l[j0] = (1 + rnd(2)) as f64;
            } else {
                u[j0] = (1 + rnd(2)) as f64;
            }
            let c: Vec<f64> = (0..n)
                .map(|j| if j < ni { (rnd(9) as f64) - 4.0 } else { 0.0 })
                .collect();
            let integrality: Vec<bool> = (0..n).map(|j| j < ni).collect();
            let sp = SparseCols::from_dense(&dense, m, n);
            let sol =
                solve_lp_cols_scaled(sp.clone(), m, n, &c, &l, &u, &b, &SimplexOptions::default());
            if sol.status != LpStatus::Optimal || sol.basis.basic_vars.len() != m {
                continue;
            }
            let glob = separate_gomory_cols_global(
                &sp,
                m,
                n,
                &l,
                &u,
                &gl,
                &gu,
                n,
                &b,
                &sol.basis,
                &integrality,
                1e-7,
                1e9,
            );
            let loc =
                separate_gomory_cols(&sp, m, n, &l, &u, &b, &sol.basis, &integrality, 1e-7, 1e9);
            let feasible = |xi: [f64; 3]| -> Option<[f64; 5]> {
                let s0 = bv[0] - dot(&arow[0], &xi);
                let s1 = bv[1] - dot(&arow[1], &xi);
                (s0 >= 0.0 && s1 >= 0.0).then_some([xi[0], xi[1], xi[2], s0, s1])
            };
            for x0 in 0..=3 {
                for x1 in 0..=3 {
                    for x2 in 0..=3 {
                        let Some(x) = feasible([x0 as f64, x1 as f64, x2 as f64]) else {
                            continue;
                        };
                        points += 1;
                        for cut in &glob {
                            assert!(
                                dot(&cut.coeffs, &x) >= cut.rhs - 1e-7 * (1.0 + cut.rhs.abs()),
                                "global cut {cut:?} removes feasible {x:?} (node box l={l:?} u={u:?})"
                            );
                        }
                        for cut in &loc {
                            if dot(&cut.coeffs, &x) < cut.rhs - 1e-6 * (1.0 + cut.rhs.abs()) {
                                local_invalid += 1;
                            }
                        }
                    }
                }
            }
            global_cuts += glob.len();
        }
        assert!(
            global_cuts > 0 && points > 0,
            "{global_cuts} {points}: the test measured nothing"
        );
        assert!(
            local_invalid > 0,
            "the local separator never removed a globally feasible point: the check has no power"
        );
        eprintln!("global_cuts={global_cuts} points={points} local_invalid={local_invalid}");
    }
}
