//! Basis recovery at an LP vertex.
//!
//! [`crossover::crossover_to_vertex`](super::crossover::crossover_to_vertex)
//! lands on a vertex of the optimal face; this module turns that vertex into a
//! *simplex basis* — the partition of the `n` variables into `m` **basic**
//! columns (whose `m × m` submatrix `A_B` is nonsingular) and `n − m`
//! **nonbasic** columns, each pinned at its lower or upper bound. The basis is
//! the prerequisite for basis-derived (Gomory / MIR) cuts, which read tableau
//! rows `B⁻¹A`.
//!
//! Recovery is purely combinatorial on `A` and the vertex `x`:
//!
//! 1. Every variable strictly inside its bounds at the vertex *must* be basic.
//!    (The crossover guarantees the free columns are independent, so there are
//!    at most `m` of them.)
//! 2. Complete the basis greedily with at-bound columns that raise the rank of
//!    `A_B`, via incremental Gaussian elimination, until `|B| = m`.
//! 3. Each remaining (nonbasic) variable is classified `AtLower`/`AtUpper` from
//!    which bound it sits on.
//!
//! On a degenerate vertex several distinct bases are valid; this returns *a*
//! valid one (the soundness property cuts rely on), which need not equal the
//! particular basis a simplex solver would report.

use super::crossover::LpView;

/// Nonbasic at lower bound — HiGHS `HighsBasisStatus::kLower`.
pub const AT_LOWER: i8 = 0;
/// Basic — HiGHS `HighsBasisStatus::kBasic`.
pub const BASIC: i8 = 1;
/// Nonbasic at upper bound — HiGHS `HighsBasisStatus::kUpper`.
pub const AT_UPPER: i8 = 2;

/// A recovered LP basis. `Clone` so a B&B node's optimal basis can be inherited
/// by its children as the warm-start state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Basis {
    /// Per-variable status, length `n`: [`BASIC`], [`AT_LOWER`] or [`AT_UPPER`]
    /// (HiGHS `HighsBasisStatus` codes).
    pub col_status: Vec<i8>,
    /// The `m` basic column indices, in the order they entered the basis.
    pub basic_vars: Vec<usize>,
}

impl Basis {
    /// Build a basis from an explicit set of `basic` column indices over `n`
    /// variables, with every other column nonbasic at its lower bound. For a
    /// standard-form LP with one slack per row this gives the all-slack starting
    /// basis (pass the slack column indices).
    pub fn from_basic(n: usize, basic: Vec<usize>) -> Self {
        let mut col_status = vec![AT_LOWER; n];
        for &j in &basic {
            col_status[j] = BASIC;
        }
        Self {
            col_status,
            basic_vars: basic,
        }
    }
}

/// Incremental rank tracker over `m`-vectors: accepts a column iff it is
/// linearly independent of those already accepted (forward Gaussian
/// elimination against stored pivot rows).
struct RankTracker {
    m: usize,
    /// `(pivot_index, reduced_row)` for each accepted column; the reduced row
    /// has a 1 at `pivot_index` and 0 at every earlier pivot index.
    pivots: Vec<(usize, Vec<f64>)>,
}

impl RankTracker {
    fn new(m: usize) -> Self {
        Self {
            m,
            pivots: Vec::new(),
        }
    }

    /// Reduce `v` against the stored pivots (zeros each pivot coordinate).
    fn reduce(&self, v: &mut [f64]) {
        for (p, row) in &self.pivots {
            let f = v[*p];
            if f != 0.0 {
                for i in 0..self.m {
                    v[i] -= f * row[i];
                }
            }
        }
    }

    /// Try to accept `col` (length `m`). Returns `true` and records a new pivot
    /// if `col` is independent of the current set; `false` otherwise.
    fn try_add(&mut self, col: &[f64], tol: f64) -> bool {
        let mut v = col.to_vec();
        self.reduce(&mut v);
        // Largest residual coordinate becomes the new pivot.
        let mut p = 0usize;
        let mut best = 0.0_f64;
        for (i, &vi) in v.iter().enumerate() {
            if vi.abs() > best {
                best = vi.abs();
                p = i;
            }
        }
        if best <= tol {
            return false; // dependent
        }
        let piv = v[p];
        for vi in v.iter_mut() {
            *vi /= piv;
        }
        self.pivots.push((p, v));
        true
    }

    fn rank(&self) -> usize {
        self.pivots.len()
    }
}

/// Recover a basis at the vertex `x` of `lp`.
///
/// `x` must be a vertex of the LP *polytope* `{A x = b, l ≤ x ≤ u}` — i.e. an
/// optimum after crossover, where at most `m` variables are strictly interior.
/// (A vertex of a higher-dimensional optimal face that is *not* a polytope
/// vertex can have up to `m + 1` interior variables; that is not a basic
/// feasible solution and is declined.)
///
/// `tol` is the bound/independence tolerance (e.g. `1e-7`). Returns `None` when
/// `x` is not such a vertex (too many, or dependent, interior columns) or `A`
/// lacks full row rank, so the caller can fall back.
pub fn recover_basis(x: &[f64], lp: &LpView<'_>, tol: f64) -> Option<Basis> {
    let (a, m, n, l, u) = (lp.a, lp.m, lp.n, lp.l, lp.u);
    if n == 0 {
        return None;
    }
    let col = |j: usize| -> Vec<f64> { (0..m).map(|i| a[i * n + j]).collect() };

    let mut tracker = RankTracker::new(m);
    let mut basic_vars: Vec<usize> = Vec::with_capacity(m);
    let mut col_status = vec![AT_LOWER; n];

    // 1. Free (strictly interior) variables must be basic.
    for (j, status) in col_status.iter_mut().enumerate() {
        if x[j] > l[j] + tol && x[j] < u[j] - tol {
            if !tracker.try_add(&col(j), tol) {
                return None; // interior var with a dependent column → not a vertex
            }
            basic_vars.push(j);
            *status = BASIC;
        } else if x[j] >= u[j] - tol && u[j] - l[j] > tol {
            *status = AT_UPPER; // pinned at upper (and not a fixed var)
        } else {
            *status = AT_LOWER;
        }
    }
    if basic_vars.len() > m {
        return None; // more interior vars than rows — x is not a vertex
    }

    // 2. Complete the basis with at-bound columns that raise the rank.
    for (j, status) in col_status.iter_mut().enumerate() {
        if tracker.rank() == m {
            break;
        }
        if *status == BASIC {
            continue;
        }
        if tracker.try_add(&col(j), tol) {
            basic_vars.push(j);
            *status = BASIC;
        }
    }

    if tracker.rank() < m {
        return None; // A has no full-rank set of m columns (rank-deficient rows)
    }
    Some(Basis {
        col_status,
        basic_vars,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::crossover::crossover_to_vertex;

    /// Solve the dense `m × m` system `A_B y = rhs` by Gaussian elimination.
    fn solve_dense(mut ab: Vec<f64>, m: usize, mut rhs: Vec<f64>) -> Vec<f64> {
        // ab is row-major m×m.
        for col in 0..m {
            // pivot
            let mut p = col;
            for i in col..m {
                if ab[i * m + col].abs() > ab[p * m + col].abs() {
                    p = i;
                }
            }
            for j in 0..m {
                ab.swap(col * m + j, p * m + j);
            }
            rhs.swap(col, p);
            let piv = ab[col * m + col];
            for j in 0..m {
                ab[col * m + j] /= piv;
            }
            rhs[col] /= piv;
            for i in 0..m {
                if i != col {
                    let f = ab[i * m + col];
                    for j in 0..m {
                        ab[i * m + j] -= f * ab[col * m + j];
                    }
                    rhs[i] -= f * rhs[col];
                }
            }
        }
        rhs
    }

    /// Assert `basis` is valid for vertex `x`: size `m`, free vars basic,
    /// nonbasics at the stated bound, and `A_B x_B = b − A_N x_N` reproduces x.
    fn assert_valid_basis(basis: &Basis, x: &[f64], lp: &LpView<'_>, b: &[f64], tol: f64) {
        let (a, m, n, l, u) = (lp.a, lp.m, lp.n, lp.l, lp.u);
        assert_eq!(basis.basic_vars.len(), m, "basis must have m columns");
        for j in 0..n {
            match basis.col_status[j] {
                BASIC => assert!(basis.basic_vars.contains(&j)),
                AT_LOWER => assert!((x[j] - l[j]).abs() < 1e-5, "var {j} not at lower"),
                AT_UPPER => assert!((x[j] - u[j]).abs() < 1e-5, "var {j} not at upper"),
                s => panic!("bad status {s}"),
            }
            // Strictly-interior vars must be basic.
            if x[j] > l[j] + tol && x[j] < u[j] - tol {
                assert_eq!(basis.col_status[j], BASIC, "interior var {j} must be basic");
            }
        }
        // Reconstruct: rhs = b − A_N x_N, solve A_B x_B = rhs, compare to x.
        let mut rhs = b.to_vec();
        for j in 0..n {
            if basis.col_status[j] != BASIC {
                for (i, ri) in rhs.iter_mut().enumerate() {
                    *ri -= a[i * n + j] * x[j];
                }
            }
        }
        let mut ab = vec![0.0_f64; m * m];
        for (bc, &j) in basis.basic_vars.iter().enumerate() {
            for i in 0..m {
                ab[i * m + bc] = a[i * n + j];
            }
        }
        let xb = solve_dense(ab, m, rhs);
        for (bc, &j) in basis.basic_vars.iter().enumerate() {
            assert!(
                (xb[bc] - x[j]).abs() < 1e-5,
                "reconstruction mismatch at var {j}"
            );
        }
    }

    #[test]
    fn recovers_basis_on_a_clean_vertex() {
        // x0 + x1 + x2 + s = 2, x in [0,1]^3, s >= 0. Vertex (1,1,0,0).
        let a = [1.0, 1.0, 1.0, 1.0];
        let c = [-1.0, -1.0, -1.0, 0.0];
        let l = [0.0, 0.0, 0.0, 0.0];
        let u = [1.0, 1.0, 1.0, f64::INFINITY];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 4,
            c: &c,
            l: &l,
            u: &u,
        };
        let x = [1.0, 1.0, 0.0, 0.0];
        let b = [2.0];
        let basis = recover_basis(&x, &lp, 1e-7).expect("basis");
        assert_valid_basis(&basis, &x, &lp, &b, 1e-7);
    }

    #[test]
    fn recovers_basis_with_a_free_variable() {
        // 2x0 + x1 + s = 2, x in [0,1]^2, s>=0. At c favouring x0, the optimum
        // has x0 basic and interior: x0 = (2 - x1 - s)/2. Take vertex x1=0,s=0
        // → x0 = 1 (at bound). Instead pick a vertex with x0 interior:
        // x1 = 1, s = 0 → 2x0 = 1 → x0 = 0.5 (free). Then x0 is basic.
        let a = [2.0, 1.0, 1.0];
        let c = [-1.0, 0.0, 0.0];
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
        let x = [0.5, 1.0, 0.0];
        let b = [2.0];
        let basis = recover_basis(&x, &lp, 1e-7).expect("basis");
        assert_eq!(basis.col_status[0], BASIC); // free var is basic
        assert_eq!(basis.col_status[1], AT_UPPER);
        assert_eq!(basis.col_status[2], AT_LOWER);
        assert_valid_basis(&basis, &x, &lp, &b, 1e-7);
    }

    #[test]
    fn crossover_then_recover_is_valid() {
        // Two constraints, slacks make it standard form:
        //   x0 + x1 + s0 = 1.5,  x0 + x2 + s1 = 1.5,  x in [0,1], s>=0.
        // With c = 0 the optimal face is the whole polytope, so an interior
        // feasible point crosses over to a genuine *polytope* vertex (<= m free
        // vars) — exactly where basis recovery is well-defined. (Recovery on an
        // interior point of a higher-dimensional optimal face is declined; see
        // `rejects_non_vertex`.)
        let n = 5; // x0,x1,x2,s0,s1
        let m = 2;
        let a = [
            1.0, 1.0, 0.0, 1.0, 0.0, // x0 + x1 + s0 = 1.5
            1.0, 0.0, 1.0, 0.0, 1.0, // x0 + x2 + s1 = 1.5
        ];
        let c = [0.0; 5];
        let l = [0.0; 5];
        let u = [1.0, 1.0, 1.0, f64::INFINITY, f64::INFINITY];
        let lp = LpView {
            a: &a,
            m,
            n,
            c: &c,
            l: &l,
            u: &u,
        };
        // interior feasible: x0=0.4,x1=0.5,x2=0.5,s0=0.6,s1=0.7
        let x0 = [0.4, 0.5, 0.5, 0.6, 0.7];
        let b = [
            a[0] * x0[0] + a[1] * x0[1] + a[2] * x0[2] + a[3] * x0[3] + a[4] * x0[4],
            a[5] * x0[0] + a[6] * x0[1] + a[7] * x0[2] + a[8] * x0[3] + a[9] * x0[4],
        ];
        let xv = crossover_to_vertex(&x0, &lp, 1e-7, 0);
        assert!((matvec(&a, m, n, &xv)[0] - b[0]).abs() < 1e-6);
        assert!((matvec(&a, m, n, &xv)[1] - b[1]).abs() < 1e-6);
        let basis = recover_basis(&xv, &lp, 1e-7).expect("basis");
        assert_valid_basis(&basis, &xv, &lp, &b, 1e-7);
    }

    fn matvec(a: &[f64], m: usize, n: usize, x: &[f64]) -> Vec<f64> {
        (0..m)
            .map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum())
            .collect()
    }

    #[test]
    fn rejects_non_vertex() {
        // Interior point of the face (two free vars but only one row) is not a
        // vertex: recovery must decline rather than fabricate a basis.
        let a = [1.0, 1.0, 1.0, 1.0];
        let c = [-1.0, -1.0, -1.0, 0.0];
        let l = [0.0, 0.0, 0.0, 0.0];
        let u = [1.0, 1.0, 1.0, f64::INFINITY];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 4,
            c: &c,
            l: &l,
            u: &u,
        };
        let x = [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.0]; // 3 free vars, m = 1
        assert!(recover_basis(&x, &lp, 1e-7).is_none());
    }
}
