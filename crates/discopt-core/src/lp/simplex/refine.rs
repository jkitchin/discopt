//! LP iterative refinement (Gleixner–Steffy–Wolter) — a precision layer *above*
//! the double-precision simplex, for issue #671.
//!
//! # Why this exists
//!
//! hda-class flowsheet relaxations produce a root McCormick LP that is genuinely
//! **feasible** (elastic phase-1 minimum violation ≈ 1.8e-10) yet **every float64
//! LP engine false-infeasibles on it**: the constraint matrix is rank-deficient
//! with condition number ≈ 1e14, so the near-singular bases the simplex lands on
//! cannot be certified in double precision. Candidate A (#662) extracts a
//! Neumaier–Shcherbina safe lower bound from the simplex's *drifted* dual — sound
//! but loose (−1.80e10 vs opt −5964.53), because the drifted dual is a poor
//! multiplier.
//!
//! The entry experiment (#671, PR #708) confirmed that the loose bound is a
//! **precision artifact, not a relaxation property**: solving the same LP with
//! high-precision residuals recovers the true root value ≈ −6.47e4 (≈5.4 orders
//! tighter). The lever is exactly the GSW result: *solve in double, compute the
//! residual against the original data in higher-than-double precision, scale it
//! up, re-solve a correction LP in double, and iterate.* Only the **residual**
//! needs high precision — the correction solves stay in double.
//!
//! # What this module provides
//!
//! 1. [`residual_dd`] / [`dot_dd`] — the high-precision (≈106-bit *double-double*)
//!    residual primitive. Pure Rust, no new dependencies: `two_sum`/`two_prod`
//!    error-free transforms accumulate `b − A x` (or `c − Aᵀ y`) without the
//!    catastrophic cancellation that makes float64 drop the ~1e-12 balancing terms
//!    the hda Arrhenius coupling hinges on.
//! 2. [`refine`] — the GSW primal-dual iterative-refinement loop, generic over a
//!    [`CorrectionSolver`] (the fixed-`A` inner double solver). Returns a
//!    refined primal-dual pair whose accuracy is driven below what a single
//!    double solve can reach.
//! 3. [`ns_safe_bound`] — the Neumaier–Shcherbina safe lower bound `g(y)` (valid
//!    for *any* `y`), so a refined dual converts directly into a tight, rigorous
//!    certificate. This mirrors the formula the MILP boundary already uses
//!    (`milp_simplex._safe_lp_lower_bound`) and the crate's own `primal::safe_bound`.
//!
//! # Placement and soundness
//!
//! This layer is **root-only / numerical-failure-triggered** by design — the cost
//! is justified only on the pathological ill-conditioned relaxations float64
//! cannot certify, never the hot per-node engine. These functions are the
//! reusable *kernel*; the shipped hda bound (issue #671, flag
//! `DISCOPT_LP_ITERATIVE_REFINEMENT`, default OFF) uses the same soundness
//! principle via a τ-regularized-resolve schedule at the numerical-failure branch
//! (`solvers/milp_simplex.py::_refined_safe_bound_regularized`), because that is
//! what makes feral's *current* factorization return usable duals on hda;
//! `refine()` here becomes the engine once a rank-revealing LU lets feral return
//! consistent approximate solutions to refine. See
//! `docs/dev/issue-671-gsw-iterative-refinement-2026-07-18.md`. With the flag OFF
//! every existing solve is byte-identical, so the certifying panel is bound-neutral
//! by construction. The [`ns_safe_bound`] it produces is a weak-duality lower bound
//! valid for *any* multiplier, so a refined dual can only *tighten* the
//! certificate — it can never lift a bound above the true optimum, exactly like
//! candidate A. Refinement improves the multiplier; it never relaxes a guard.
//!
//! Reference: Gleixner, Steffy, Wolter, *Iterative Refinement for Linear
//! Programming*, INFORMS J. Comput. 28(3), 2016.

// Row/column index loops mirror `primal.rs`/`dual.rs` — the parallel index is used
// against several arrays at once, so a range loop is the clearest form here.
#![allow(clippy::needless_range_loop)]

/// A value held to roughly double the working precision as an unevaluated sum
/// `hi + lo` with `|lo| ≤ ½ ulp(hi)` ("double-double"). Enough extra precision to
/// compute `b − A x` accurately when the individual products nearly cancel — the
/// ~1e-12 residual buried under ~1e-10 terms in hda's Arrhenius core.
#[derive(Clone, Copy, Debug, Default)]
struct Dd {
    hi: f64,
    lo: f64,
}

/// Knuth's error-free transform: `a + b = s + e` exactly, with `s = fl(a+b)`.
#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let e = (a - (s - bb)) + (b - bb);
    (s, e)
}

/// Error-free transform of a product: `a * b = p + e` exactly, with `p = fl(a·b)`.
/// Uses a fused multiply-add so `e` is the exact rounding error of `p`.
#[inline]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let e = a.mul_add(b, -p);
    (p, e)
}

impl Dd {
    #[inline]
    fn zero() -> Self {
        Dd { hi: 0.0, lo: 0.0 }
    }

    /// Add an ordinary `f64`, keeping the trailing bits.
    #[inline]
    fn add_f64(self, x: f64) -> Self {
        let (s, e) = two_sum(self.hi, x);
        let lo = e + self.lo;
        // Renormalize so the invariant |lo| ≤ ½ ulp(hi) holds for the next step.
        let (hi, lo) = two_sum(s, lo);
        Dd { hi, lo }
    }

    /// Add the exact product `a·b`, keeping the trailing bits (a fused
    /// multiply-accumulate at ≈106-bit precision).
    #[inline]
    fn add_prod(self, a: f64, b: f64) -> Self {
        let (p, pe) = two_prod(a, b);
        let (s, e) = two_sum(self.hi, p);
        let lo = (e + pe) + self.lo;
        let (hi, lo) = two_sum(s, lo);
        Dd { hi, lo }
    }

    /// Round back to the nearest `f64` (`hi` already carries the correctly-rounded
    /// sum; `lo` only refines ties, so `hi` is the right answer).
    #[inline]
    fn to_f64(self) -> f64 {
        self.hi
    }
}

/// High-precision dot product `Σ_k v[k]·w[k]`, computed in double-double so that
/// near-total cancellation (products of magnitude `M` summing to `≪ M`) is
/// resolved instead of lost to float64 rounding. `v` and `w` must be equal length.
pub fn dot_dd(v: &[f64], w: &[f64]) -> f64 {
    debug_assert_eq!(v.len(), w.len());
    let mut acc = Dd::zero();
    for (&a, &b) in v.iter().zip(w.iter()) {
        acc = acc.add_prod(a, b);
    }
    acc.to_f64()
}

/// The correctly-rounded residual `rhs − Σ_k row[k]·x[k]`, accumulated at
/// double-double precision. This is the GSW primitive: even when the individual
/// `row[k]·x[k]` are orders of magnitude larger than the true residual, the
/// double-double accumulation recovers the residual to full `f64` accuracy, where
/// a naive float64 dot would return noise (or exactly `0`). `row` and `x` must be
/// equal length.
pub fn residual_dd(row: &[f64], x: &[f64], rhs: f64) -> f64 {
    debug_assert_eq!(row.len(), x.len());
    // Accumulate rhs − Σ row·x in one double-double sweep.
    let mut acc = Dd::zero().add_f64(rhs);
    for (&a, &xj) in row.iter().zip(x.iter()) {
        acc = acc.add_prod(-a, xj);
    }
    acc.to_f64()
}

/// The residual vector `rhs − B x` (`transpose = false`) or `rhs − Bᵀ x`
/// (`transpose = true`), each component accumulated in double-double, where `B` is
/// given by its **dense columns** `cols` (`cols[j][i] = B[i][j]`). Zero entries are
/// skipped, so on the sparse simplex bases (a handful of nonzeros per column) this
/// is ~`O(nnz)`, not `O(m²)` — the form the hardened refined solves
/// ([`super::linsolve::FeralLU`]) use so the residual is cheap on the hot path.
pub fn residual_matvec_dd(cols: &[Vec<f64>], x: &[f64], rhs: &[f64], transpose: bool) -> Vec<f64> {
    let m = cols.len();
    if transpose {
        // (Bᵀ x)_i = Σ_j B[j][i]·x[j] = dot(cols[i], x); rows are independent.
        (0..m)
            .map(|i| {
                let mut acc = Dd::zero().add_f64(rhs[i]);
                for (j, &a) in cols[i].iter().enumerate() {
                    if a != 0.0 {
                        acc = acc.add_prod(-a, x[j]);
                    }
                }
                acc.to_f64()
            })
            .collect()
    } else {
        // (B x)_i = Σ_j cols[j][i]·x[j]; scatter each column's contribution.
        let mut acc: Vec<Dd> = rhs.iter().map(|&b| Dd::zero().add_f64(b)).collect();
        for (j, col) in cols.iter().enumerate() {
            let xj = x[j];
            if xj == 0.0 {
                continue;
            }
            for (i, &a) in col.iter().enumerate() {
                if a != 0.0 {
                    acc[i] = acc[i].add_prod(-a, xj);
                }
            }
        }
        acc.iter().map(|d| d.to_f64()).collect()
    }
}

/// The sentinel treated as `±∞` when reading variable bounds. Matches the crate's
/// `CERT_INF` / MILP-boundary convention: a bound at or beyond this magnitude is
/// an open side.
pub const REFINE_INF: f64 = 1e20;

/// Neumaier–Shcherbina safe lower bound on `min cᵀx s.t. A x = b, l ≤ x ≤ u` from
/// **free-sign** equality multipliers `y` (length `m`). Weak duality gives, for
/// *any* `y`,
///
/// ```text
/// g(y) = bᵀy + Σ_j min_{x_j∈[l_j,u_j]} (c − Aᵀy)_j x_j  ≤  min cᵀx,
/// ```
///
/// so `g(y)` is a valid lower bound regardless of how `y` was obtained — a
/// refined dual only *tightens* it, never lifts it above the optimum. Returns
/// `None` when the bound is `−∞` (a reduced cost with the wrong-signed open box
/// side). `a` is row-major `m × n`.
///
/// This is the exact formula the MILP boundary applies to candidate A's drifted
/// dual (`milp_simplex._safe_lp_lower_bound_std`) and that `primal::safe_bound`
/// uses in its tests; refinement changes only the *quality* of `y` fed into it.
#[allow(clippy::too_many_arguments)]
pub fn ns_safe_bound(
    y: &[f64],
    c: &[f64],
    a: &[f64],
    m: usize,
    n: usize,
    b: &[f64],
    l: &[f64],
    u: &[f64],
) -> Option<f64> {
    if y.len() != m || !y.iter().all(|v| v.is_finite()) {
        return None;
    }
    // bᵀy at double-double precision (b and y can both be wide-ranged on these LPs).
    let mut g = dot_dd(b, y);
    for j in 0..n {
        // Reduced cost (c − Aᵀy)_j, high-precision so a tiny true rc is not lost.
        let mut aty = Dd::zero();
        for i in 0..m {
            aty = aty.add_prod(a[i * n + j], y[i]);
        }
        let rc = c[j] - aty.to_f64();
        if rc > 0.0 {
            if l[j] <= -REFINE_INF {
                return None;
            }
            g += rc * l[j];
        } else if rc < 0.0 {
            if u[j] >= REFINE_INF {
                return None;
            }
            g += rc * u[j];
        }
    }
    if g.is_finite() {
        Some(g)
    } else {
        None
    }
}

/// CSC twin of [`ns_safe_bound`]: the Neumaier–Shcherbina safe lower bound for
/// `min cᵀx s.t. A x = b, l ≤ x ≤ u` where `A` is given column-major as the raw
/// CSC arrays (`col_ptr` length `n+1`, `row_idx`/`vals` the nonzeros). Identical
/// arithmetic and rigor to the dense version — `bᵀy` and each column's `(Aᵀy)_j`
/// at double-double precision, then the reduced-cost box term — so the result is a
/// rigorous lower bound (`≤` the true optimum) at any conditioning. Returns `None`
/// when a nonzero reduced cost meets an infinite bound (no finite certificate) or a
/// dual is non-finite. Used by the native spatial node kernel, whose node LPs are
/// CSC, so no dense `m×n` matrix is ever materialized to certify the bound.
#[allow(clippy::too_many_arguments)]
pub fn ns_safe_bound_csc(
    y: &[f64],
    c: &[f64],
    col_ptr: &[usize],
    row_idx: &[usize],
    vals: &[f64],
    m: usize,
    n: usize,
    b: &[f64],
    l: &[f64],
    u: &[f64],
) -> Option<f64> {
    if y.len() != m || !y.iter().all(|v| v.is_finite()) {
        return None;
    }
    debug_assert_eq!(col_ptr.len(), n + 1);
    let mut g = dot_dd(b, y);
    for j in 0..n {
        // (Aᵀy)_j = sum over column j's nonzeros of vals[p]*y[row_idx[p]], in DD.
        let mut aty = Dd::zero();
        for p in col_ptr[j]..col_ptr[j + 1] {
            aty = aty.add_prod(vals[p], y[row_idx[p]]);
        }
        let rc = c[j] - aty.to_f64();
        if rc > 0.0 {
            if l[j] <= -REFINE_INF {
                return None;
            }
            g += rc * l[j];
        } else if rc < 0.0 {
            if u[j] >= REFINE_INF {
                return None;
            }
            g += rc * u[j];
        }
    }
    if g.is_finite() {
        Some(g)
    } else {
        None
    }
}

/// The primal-dual solution of a correction subproblem: `x` (length `n`) and the
/// row duals `y` (length `m`) under the crate's dual convention (so that
/// [`ns_safe_bound`] applied to `y` is a valid lower bound).
#[derive(Clone, Debug)]
pub struct CorrectionSolution {
    /// Primal point, length `n`.
    pub x: Vec<f64>,
    /// Row duals, length `m` (crate dual convention).
    pub y: Vec<f64>,
}

/// The fixed-`A` inner double solver GSW refines around. Each call solves
/// `min cᵀx s.t. A x = b, l ≤ x ≤ u` for the **same** constraint matrix `A`
/// (only the objective, right-hand side, and bounds vary between rounds — the
/// factorization/warm basis can be reused). Returns `None` on numerical failure,
/// which aborts refinement (the caller falls back to candidate A).
pub trait CorrectionSolver {
    /// Solve `min cᵀx s.t. A x = b, l ≤ x ≤ u` for this solver's fixed `A`,
    /// returning the primal-dual pair, or `None` on numerical failure.
    fn solve(&mut self, c: &[f64], b: &[f64], l: &[f64], u: &[f64]) -> Option<CorrectionSolution>;
}

/// Tunables for [`refine`].
#[derive(Clone, Debug)]
pub struct RefineOptions {
    /// Stop once the primal residual (max equality violation) is `≤` this.
    pub eps_primal: f64,
    /// Stop once the dual (reduced-cost KKT) residual is `≤` this.
    pub eps_dual: f64,
    /// Maximum refinement rounds. GSW converges geometrically (≈16 digits/round on
    /// a solver that returns ~`1e-16`-accurate corrections), so a handful suffices;
    /// the cap only bounds the cost when the inner solver is itself weak.
    pub max_rounds: usize,
    /// Largest residual scale-up factor `Δ`. Bounds how far a single round can push
    /// the correction data so the inner double solve stays in a sane range.
    pub scale_cap: f64,
}

impl Default for RefineOptions {
    fn default() -> Self {
        RefineOptions {
            eps_primal: 1e-12,
            eps_dual: 1e-12,
            max_rounds: 30,
            scale_cap: 1e12,
        }
    }
}

/// The refined solution and the residuals it reached.
#[derive(Clone, Debug)]
pub struct RefineResult {
    /// Refined primal point (length `n`), box-feasible by construction.
    pub x: Vec<f64>,
    /// Refined row duals (length `m`) — feed to [`ns_safe_bound`] for a tight,
    /// rigorous lower bound.
    pub y: Vec<f64>,
    /// Rounds actually performed.
    pub rounds: usize,
    /// Whether both residual tolerances were met.
    pub converged: bool,
    /// Final primal residual (max equality/bound violation).
    pub primal_res: f64,
    /// Final dual residual (max KKT reduced-cost violation).
    pub dual_res: f64,
}

/// Gleixner–Steffy–Wolter primal-dual iterative refinement of
/// `min cᵀx s.t. A x = b, l ≤ x ≤ u` (`a` row-major `m × n`).
///
/// Each round computes the primal residual `b − A x*` and the dual residual
/// `c − Aᵀ y*` **in double-double precision** ([`residual_dd`]), scales them up by
/// `Δp`/`Δd`, solves a correction subproblem in double via `solver`, and folds the
/// scaled-down correction back into the incumbent `(x*, y*)`. The incumbent is
/// kept **exactly box-feasible** every round because the correction's bounds are
/// `Δp·(l − x*) ≤ x̄ ≤ Δp·(u − x*)`, so `x* + x̄/Δp ∈ [l, u]`. Returns `None` if the
/// first correction solve fails (nothing to refine); a later failure returns the
/// best incumbent so far with `converged = false`.
#[allow(clippy::too_many_arguments)]
pub fn refine<S: CorrectionSolver>(
    a: &[f64],
    m: usize,
    n: usize,
    c: &[f64],
    b: &[f64],
    l: &[f64],
    u: &[f64],
    solver: &mut S,
    opts: &RefineOptions,
) -> Option<RefineResult> {
    assert_eq!(a.len(), m * n, "A must be row-major m×n");
    assert_eq!(c.len(), n);
    assert_eq!(b.len(), m);
    assert_eq!(l.len(), n);
    assert_eq!(u.len(), n);

    let mut xstar = vec![0.0f64; n];
    let mut ystar = vec![0.0f64; m];

    // Row slices of A, taken on demand (row-major layout).
    let row = |i: usize| &a[i * n..i * n + n];

    for round in 0..opts.max_rounds {
        // ---- primal residual r_b = b − A x*  (double-double per row) ----
        let mut r_b = vec![0.0f64; m];
        let mut primal_res = 0.0f64;
        for i in 0..m {
            let r = residual_dd(row(i), &xstar, b[i]);
            r_b[i] = r;
            primal_res = primal_res.max(r.abs());
        }

        // ---- dual residual d = c − Aᵀ y*  (double-double per column) ----
        let mut d = vec![0.0f64; n];
        for j in 0..n {
            let mut aty = Dd::zero();
            for i in 0..m {
                aty = aty.add_prod(a[i * n + j], ystar[i]);
            }
            d[j] = c[j] - aty.to_f64();
        }
        // KKT dual-infeasibility given which bound each x*_j rests at. This is a
        // stopping *diagnostic* only — g(y*) is a valid bound at every round, so
        // an imperfect measure never makes the certificate unsound.
        let dual_res = kkt_dual_residual(&d, &xstar, l, u);

        let converged = primal_res <= opts.eps_primal && dual_res <= opts.eps_dual;
        let snapshot = RefineResult {
            x: xstar.clone(),
            y: ystar.clone(),
            rounds: round,
            converged,
            primal_res,
            dual_res,
        };
        if converged {
            return Some(snapshot);
        }

        // ---- residual scale-ups: push both residuals toward O(1) ----
        let dp = scale_for(primal_res, opts.scale_cap);
        let dd = scale_for(dual_res, opts.scale_cap);

        // ---- correction subproblem (same A; shifted objective/rhs/bounds) ----
        let cc: Vec<f64> = d.iter().map(|&dj| dd * dj).collect();
        let bb: Vec<f64> = r_b.iter().map(|&ri| dp * ri).collect();
        let mut lc = vec![0.0f64; n];
        let mut uc = vec![0.0f64; n];
        for j in 0..n {
            lc[j] = shift_bound(l[j], xstar[j], dp, true);
            uc[j] = shift_bound(u[j], xstar[j], dp, false);
        }

        let sol = match solver.solve(&cc, &bb, &lc, &uc) {
            Some(s) => s,
            None => {
                // Inner solve failed this round — keep the best incumbent so far.
                return Some(snapshot);
            }
        };

        // ---- fold the scaled-down correction into the incumbent ----
        for j in 0..n {
            xstar[j] += sol.x[j] / dp;
        }
        for i in 0..m {
            ystar[i] += sol.y[i] / dd;
        }
    }

    // Recompute the final residuals of the last incumbent for an honest report.
    let mut primal_res = 0.0f64;
    for i in 0..m {
        primal_res = primal_res.max(residual_dd(row(i), &xstar, b[i]).abs());
    }
    let mut d = vec![0.0f64; n];
    for j in 0..n {
        let mut aty = Dd::zero();
        for i in 0..m {
            aty = aty.add_prod(a[i * n + j], ystar[i]);
        }
        d[j] = c[j] - aty.to_f64();
    }
    let dual_res = kkt_dual_residual(&d, &xstar, l, u);
    Some(RefineResult {
        x: xstar,
        y: ystar,
        rounds: opts.max_rounds,
        converged: primal_res <= opts.eps_primal && dual_res <= opts.eps_dual,
        primal_res,
        dual_res,
    })
}

/// Scale a residual toward O(1): `Δ = clamp(1/res, 1, cap)`. A residual already
/// below `1/cap` uses the cap; a tiny/zero residual uses `1` (nothing to scale).
#[inline]
fn scale_for(res: f64, cap: f64) -> f64 {
    if !res.is_finite() || res <= 0.0 {
        return 1.0;
    }
    (1.0 / res).clamp(1.0, cap)
}

/// Shifted correction bound `Δ·(bound − x*)`, preserving open (`±∞`) sides.
/// `lower` picks the `−∞` sentinel side; the caller passes the raw variable bound.
#[inline]
fn shift_bound(bound: f64, xstar: f64, dp: f64, lower: bool) -> f64 {
    if lower {
        if bound <= -REFINE_INF {
            return f64::NEG_INFINITY;
        }
    } else if bound >= REFINE_INF {
        return f64::INFINITY;
    }
    dp * (bound - xstar)
}

/// KKT dual-infeasibility: for each variable, its reduced cost must be `≥ 0` at a
/// lower bound, `≤ 0` at an upper bound, and `≈ 0` when strictly interior. Returns
/// the worst violation across all variables.
fn kkt_dual_residual(d: &[f64], x: &[f64], l: &[f64], u: &[f64]) -> f64 {
    let n = d.len();
    let mut worst = 0.0f64;
    for j in 0..n {
        let at_lower = l[j] > -REFINE_INF && (x[j] - l[j]).abs() <= 1e-9 * (1.0 + l[j].abs());
        let at_upper = u[j] < REFINE_INF && (x[j] - u[j]).abs() <= 1e-9 * (1.0 + u[j].abs());
        let v = if at_lower && !at_upper {
            (-d[j]).max(0.0)
        } else if at_upper && !at_lower {
            d[j].max(0.0)
        } else if at_lower && at_upper {
            // Fixed variable: any reduced-cost sign is dual-feasible.
            0.0
        } else {
            d[j].abs()
        };
        worst = worst.max(v);
    }
    worst
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::crossover::LpView;
    use crate::lp::simplex::primal::solve_lp;
    use crate::lp::simplex::{LpStatus, SimplexOptions};

    /// Naive left-to-right float64 dot — the baseline the double-double residual
    /// beats. Present only to *demonstrate* the cancellation loss in tests.
    fn naive_dot(v: &[f64], w: &[f64]) -> f64 {
        v.iter().zip(w).map(|(a, b)| a * b).sum()
    }

    #[test]
    fn two_sum_is_exact() {
        // 1 + 1e-20: the tail is exactly 1e-20, lost by fl(a+b) but kept in e.
        let (s, e) = two_sum(1.0, 1e-20);
        assert_eq!(s, 1.0);
        assert_eq!(e, 1e-20);
        // Reconstructs the exact sum.
        assert_eq!(s as f64 + e, 1.0 + 1e-20);
    }

    #[test]
    fn two_prod_is_exact() {
        // (1 + 2^-30)^2 needs 60 bits — the low word carries the overflow.
        let a = 1.0 + 2f64.powi(-30);
        let (p, e) = two_prod(a, a);
        // p is the rounded product; p + e is the exact product a*a.
        let exact = 1.0 + 2f64.powi(-29) + 2f64.powi(-60);
        assert_eq!(p + e, exact);
        assert!(e != 0.0, "the rounding error must be captured");
    }

    #[test]
    fn residual_dd_survives_catastrophic_cancellation() {
        // Σ = 1e16 + 1 − 1e16. Naive float64 loses the +1 (1e16 + 1 == 1e16),
        // returning 0; the true sum is 1.
        let row = [1e16, 1.0, -1e16];
        let x = [1.0, 1.0, 1.0];
        assert_eq!(naive_dot(&row, &x), 0.0, "naive dot loses the unit term");
        // residual = rhs − Σ, with rhs = 1 → true residual exactly 0.
        assert_eq!(residual_dd(&row, &x, 1.0), 0.0);
        // With rhs = 0 the true residual is −1, which naive would call 0.
        assert_eq!(residual_dd(&row, &x, 0.0), -1.0);
    }

    #[test]
    fn residual_dd_recovers_arrhenius_scale_product_error() {
        // hda's coupling (#671 E2): a 6.3e10 Arrhenius pre-exponential times a
        // ~1e-13 aux. The product carries a sub-ulp rounding error that naive
        // float64 silently drops — exactly the kind of ~1e-18 term whose loss lets
        // a near-feasible constraint read as violated. With `rhs = fl(big·aux)`,
        // the true residual `rhs − big·aux` is the *exact* product-rounding error,
        // which naive rounds to 0 but the double-double residual recovers.
        let big = 6.3e10_f64;
        let aux = 1.96e-13_f64;
        let p = big * aux; // rounded product ≈ 1.235e-2
        let e = big.mul_add(aux, -p); // exact product-rounding error (nonzero)
        assert!(e != 0.0, "the Arrhenius product must have a rounding tail");
        // Naive residual `rhs − fl(big·aux)` = p − p = 0 (the tail is lost).
        assert_eq!(p - naive_dot(&[big], &[aux]), 0.0);
        // Double-double residual recovers −e, the tail naive dropped.
        let r = residual_dd(&[big], &[aux], p);
        assert_eq!(
            r, -e,
            "double-double residual should recover the product tail"
        );
    }

    #[test]
    fn dot_dd_beats_naive_on_cancellation() {
        let v = [1e18, 3.0, -1e18, -2.0];
        let w = [1.0, 1.0, 1.0, 1.0];
        // True dot = 1. Naive collapses the 1e18 pair and the +3/−2 lands on noise.
        assert_eq!(dot_dd(&v, &w), 1.0);
        assert_ne!(naive_dot(&v, &w), 1.0);
    }

    /// A fixed-`A` inner solver backed by the crate's primal simplex. The GSW
    /// correction subproblems reuse this exactly as production would reuse the
    /// warm-started node simplex.
    struct SimplexInner<'a> {
        a: &'a [f64],
        m: usize,
        n: usize,
    }

    /// Translate refinement's mathematical `±∞` bounds to the simplex's `1e20`
    /// sentinel (open-side convention).
    fn to_sentinel(v: f64) -> f64 {
        if v == f64::INFINITY {
            REFINE_INF
        } else if v == f64::NEG_INFINITY {
            -REFINE_INF
        } else {
            v
        }
    }

    impl CorrectionSolver for SimplexInner<'_> {
        fn solve(
            &mut self,
            c: &[f64],
            b: &[f64],
            l: &[f64],
            u: &[f64],
        ) -> Option<CorrectionSolution> {
            let ls: Vec<f64> = l.iter().map(|&v| to_sentinel(v)).collect();
            let us: Vec<f64> = u.iter().map(|&v| to_sentinel(v)).collect();
            let view = LpView {
                a: self.a,
                m: self.m,
                n: self.n,
                c,
                l: &ls,
                u: &us,
            };
            let sol = solve_lp(&view, b, &SimplexOptions::default());
            if sol.status != LpStatus::Optimal {
                return None;
            }
            Some(CorrectionSolution {
                x: sol.x[..self.n].to_vec(),
                y: sol.dual,
            })
        }
    }

    /// Wraps an inner solver and rounds every returned coordinate to a coarse
    /// absolute grid — a faithful stand-in for a *double* solver that only reaches
    /// `grid`-level accuracy on an ill-conditioned LP. Refinement must recover
    /// accuracy far below `grid`.
    struct LossyInner<'a> {
        inner: SimplexInner<'a>,
        grid: f64,
    }

    impl CorrectionSolver for LossyInner<'_> {
        fn solve(
            &mut self,
            c: &[f64],
            b: &[f64],
            l: &[f64],
            u: &[f64],
        ) -> Option<CorrectionSolution> {
            let mut sol = self.inner.solve(c, b, l, u)?;
            let round = |v: f64, g: f64| (v / g).round() * g;
            for xj in sol.x.iter_mut() {
                *xj = round(*xj, self.grid);
            }
            for yi in sol.y.iter_mut() {
                *yi = round(*yi, self.grid);
            }
            Some(sol)
        }
    }

    // min −x0 − 2x1 s.t. x0 + x1 + s = 4, x0,x1 ∈ [0,5], s ∈ [0,∞). Optimum −8 at
    // (x1=4). The safe bound from the optimal dual reproduces −8.
    fn small_lp() -> (
        Vec<f64>,
        usize,
        usize,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
    ) {
        let a = vec![1.0, 1.0, 1.0];
        let c = vec![-1.0, -2.0, 0.0];
        let b = vec![4.0];
        let l = vec![0.0, 0.0, 0.0];
        let u = vec![5.0, 5.0, REFINE_INF];
        (a, 1, 3, c, b, l, u)
    }

    #[test]
    fn refine_exact_solver_reproduces_optimum_bound() {
        let (a, m, n, c, b, l, u) = small_lp();
        let mut inner = SimplexInner { a: &a, m, n };
        let res = refine(
            &a,
            m,
            n,
            &c,
            &b,
            &l,
            &u,
            &mut inner,
            &RefineOptions::default(),
        )
        .expect("refinement returns a result");
        assert!(res.converged, "exact inner solver converges: {res:?}");
        assert!(res.primal_res <= 1e-12 && res.dual_res <= 1e-12);
        let g = ns_safe_bound(&res.y, &c, &a, m, n, &b, &l, &u).expect("finite bound");
        assert!((g - (-8.0)).abs() < 1e-9, "safe bound {g} should be −8");
    }

    #[test]
    fn refine_extracts_accuracy_below_the_inner_solvers_grid() {
        // The inner solver only returns 1e-4-accurate corrections; refinement must
        // drive the true residuals orders of magnitude below that.
        let (a, m, n, c, b, l, u) = small_lp();
        let mut lossy = LossyInner {
            inner: SimplexInner { a: &a, m, n },
            grid: 1e-4,
        };
        let opts = RefineOptions {
            eps_primal: 1e-11,
            eps_dual: 1e-11,
            max_rounds: 40,
            scale_cap: 1e12,
        };
        let res = refine(&a, m, n, &c, &b, &l, &u, &mut lossy, &opts)
            .expect("refinement returns a result");
        assert!(
            res.primal_res <= 1e-9,
            "primal residual {} not driven below the 1e-4 grid",
            res.primal_res
        );
        // The refined dual yields a bound far tighter than the grid error, and
        // never above the true optimum (soundness).
        let g = ns_safe_bound(&res.y, &c, &a, m, n, &b, &l, &u).expect("finite bound");
        assert!(
            g <= -8.0 + 1e-9,
            "safe bound {g} above optimum −8 (unsound)"
        );
        assert!(
            (g - (-8.0)).abs() < 1e-6,
            "safe bound {g} not tightened to −8 (grid was 1e-4)"
        );
    }

    #[test]
    fn refine_tightens_a_drifted_dual_on_an_ill_conditioned_lp() {
        // Wide-coefficient LP (range 1e8) whose single lossy solve gives a drifted
        // dual → a loose safe bound; refinement tightens it toward the optimum.
        // min −x0 − x1  s.t.  1e8·x0 + x1 + s0 = 1e8,  x1 + s1 = 5,
        // x0∈[0,1], x1∈[0,10], s0,s1∈[0,∞).  Optimum: x0=1, x1=0 → −1? Let's keep
        // it simple: x0≤1 and 1e8 x0 ≤ 1e8 ⇒ x0≤1; x1≤5. Optimum −(1)−(5) = −6.
        let a = vec![
            1e8, 1.0, 1.0, 0.0, // row 0: 1e8 x0 + x1 + s0
            0.0, 1.0, 0.0, 1.0, // row 1: x1 + s1
        ];
        let m = 2;
        let n = 4;
        let c = vec![-1.0, -1.0, 0.0, 0.0];
        let b = vec![1e8, 5.0];
        let l = vec![0.0, 0.0, 0.0, 0.0];
        let u = vec![1.0, 10.0, REFINE_INF, REFINE_INF];

        // One lossy solve (grid 1e-3) → drifted dual → measure its safe bound.
        let mut lossy = LossyInner {
            inner: SimplexInner { a: &a, m, n },
            grid: 1e-3,
        };
        let single = lossy
            .solve(&c, &b, &l, &u)
            .expect("single lossy solve is optimal");
        let g_single = ns_safe_bound(&single.y, &c, &a, m, n, &b, &l, &u).expect("finite");

        // Refinement over the same lossy solver.
        let opts = RefineOptions {
            eps_primal: 1e-10,
            eps_dual: 1e-10,
            max_rounds: 40,
            scale_cap: 1e12,
        };
        let res = refine(&a, m, n, &c, &b, &l, &u, &mut lossy, &opts).expect("refined");
        let g_refined = ns_safe_bound(&res.y, &c, &a, m, n, &b, &l, &u).expect("finite");

        // Soundness: neither bound exceeds the true optimum −6.
        assert!(g_single <= -6.0 + 1e-6, "single bound {g_single} unsound");
        assert!(
            g_refined <= -6.0 + 1e-9,
            "refined bound {g_refined} unsound"
        );
        // The refined bound is at least as tight, and materially tighter than the
        // drifted single-solve bound.
        assert!(
            g_refined >= g_single - 1e-9,
            "refined {g_refined} looser than single {g_single}"
        );
        assert!(
            (g_refined - (-6.0)).abs() < 1e-6,
            "refined bound {g_refined} not tight to −6"
        );
    }

    #[test]
    fn ns_safe_bound_is_never_above_optimum_for_arbitrary_dual() {
        // g(y) ≤ opt for ANY y — the soundness property candidate A relies on.
        let (a, m, n, c, b, l, u) = small_lp();
        for &y0 in &[-3.7, -1.0, 0.0, 0.5, 2.3, 10.0] {
            if let Some(g) = ns_safe_bound(&[y0], &c, &a, m, n, &b, &l, &u) {
                assert!(g <= -8.0 + 1e-9, "g({y0}) = {g} exceeds optimum −8");
            }
        }
    }

    // Dense a (row-major m×n) -> CSC arrays (col_ptr, row_idx, vals).
    fn dense_to_csc(a: &[f64], m: usize, n: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let mut col_ptr = vec![0usize; n + 1];
        for j in 0..n {
            for i in 0..m {
                if a[i * n + j] != 0.0 {
                    col_ptr[j + 1] += 1;
                }
            }
            col_ptr[j + 1] += col_ptr[j];
        }
        let mut row_idx = vec![0usize; col_ptr[n]];
        let mut vals = vec![0.0f64; col_ptr[n]];
        let mut pos = col_ptr.clone();
        for j in 0..n {
            for i in 0..m {
                let v = a[i * n + j];
                if v != 0.0 {
                    row_idx[pos[j]] = i;
                    vals[pos[j]] = v;
                    pos[j] += 1;
                }
            }
        }
        (col_ptr, row_idx, vals)
    }

    /// The CSC safe bound is BIT-IDENTICAL to the dense one for arbitrary duals,
    /// on both a well-scaled and an ill-scaled (1e8) LP — same DD arithmetic.
    #[test]
    fn ns_safe_bound_csc_matches_dense() {
        // Well-scaled small LP.
        let (a, m, n, c, b, l, u) = small_lp();
        let (cp, ri, v) = dense_to_csc(&a, m, n);
        for &y0 in &[-3.7, -1.0, 0.0, 0.5, 2.3, 10.0] {
            let dense = ns_safe_bound(&[y0], &c, &a, m, n, &b, &l, &u);
            let csc = ns_safe_bound_csc(&[y0], &c, &cp, &ri, &v, m, n, &b, &l, &u);
            assert_eq!(dense, csc, "mismatch at y={y0} (well-scaled)");
        }
        // Ill-scaled LP (1e8 row) — the case DD precision exists for.
        let a2 = vec![
            1e8, 1.0, 1.0, 0.0, //
            0.0, 1.0, 0.0, 1.0, //
        ];
        let (m2, n2) = (2, 4);
        let c2 = vec![-1.0, -1.0, 0.0, 0.0];
        let b2 = vec![1e8, 5.0];
        let l2 = vec![0.0, 0.0, 0.0, 0.0];
        let u2 = vec![1.0, 10.0, REFINE_INF, REFINE_INF];
        let (cp2, ri2, v2) = dense_to_csc(&a2, m2, n2);
        for &yy in &[[-1.0, -1.0], [0.3, 2.7], [1e-7, -4.0]] {
            let dense = ns_safe_bound(&yy, &c2, &a2, m2, n2, &b2, &l2, &u2);
            let csc = ns_safe_bound_csc(&yy, &c2, &cp2, &ri2, &v2, m2, n2, &b2, &l2, &u2);
            assert_eq!(dense, csc, "mismatch at y={yy:?} (ill-scaled)");
        }
    }
}
