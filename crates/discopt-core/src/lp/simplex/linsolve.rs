//! Basis factorization behind the revised simplex.
//!
//! The simplex only needs four operations on the current basis matrix `B`
//! (the `m × m` submatrix of basic columns): factorize it, solve `B x = rhs`
//! (`ftran`), solve `Bᵀ y = rhs` (`btran`), and replace one basic column with a
//! product-form update after a pivot. [`LinearSolver`] abstracts these so the
//! simplex is independent of the backend.
//!
//! - [`FeralLU`] — the production backend, wrapping [`feral`]'s sparse
//!   unsymmetric LU (`PLUQ` with `ftran`/`btran` and simplex-style product-form
//!   column updates). This is what makes warm re-solves cheap.
//! - [`DenseLU`] — a dependency-light oracle/fallback built on the crate's
//!   existing dense Gaussian elimination ([`crate::lp::gomory::solve_dense`]);
//!   it refactorizes on every solve, so it is for bring-up, tiny bases, and
//!   cross-checking `FeralLU` in tests — not performance.

use feral::{should_use_dense_lu, DenseLu, LuParams, SparseColMatrix, SparseLu, SparseLuSymbolic};

/// Error from a basis factorization/solve.
#[derive(Debug, Clone)]
pub enum LinError {
    /// The basis is singular (or numerically so).
    Singular,
    /// No factorization has been computed yet.
    NotFactorized,
    /// Backend error (message).
    Backend(String),
}

/// Operations the revised simplex needs from a basis factorization of `B`.
pub trait LinearSolver {
    /// Factorize the `m × m` basis whose columns are `cols[0..m]` (each an
    /// `m`-vector of that basic column's dense entries).
    fn factorize(&mut self, m: usize, cols: &[Vec<f64>]) -> Result<(), LinError>;

    /// Factorize the `m × m` basis from its **sparse** columns: `cols[slot]` lists
    /// the `(row, value)` nonzeros of basis slot `slot` (`0..m`). This avoids
    /// materializing the dense `m × m` basis (the O(m²) column build + nnz scan +
    /// dense→sparse conversion that dominated refactorization of row-heavy
    /// lifted-McCormick bases — see feral#87 / discopt#229/#268). The matrix built
    /// here is exactly the one [`factorize`](Self::factorize) would build from the
    /// same columns, so the factorization — and every downstream pivot — is
    /// bit-identical.
    ///
    /// The default scatters to dense and delegates to [`factorize`](Self::factorize)
    /// so backends without a native sparse path keep working; [`FeralLU`] overrides
    /// it to build feral's sparse matrix from the sparse columns directly.
    fn factorize_sparse(&mut self, m: usize, cols: &[Vec<(usize, f64)>]) -> Result<(), LinError> {
        let mut dense = vec![vec![0.0; m]; m];
        for (slot, col) in cols.iter().enumerate() {
            for &(row, v) in col {
                dense[slot][row] = v;
            }
        }
        self.factorize(m, &dense)
    }
    /// Solve `B x = rhs` in place.
    fn ftran(&mut self, rhs: &mut [f64]) -> Result<(), LinError>;
    /// Solve `Bᵀ y = rhs` in place.
    fn btran(&mut self, rhs: &mut [f64]) -> Result<(), LinError>;
    /// Replace the basic column in slot `leaving_slot` with `entering_col`
    /// (product-form update; the caller refactorizes when updates accumulate).
    fn update(&mut self, leaving_slot: usize, entering_col: &[f64]) -> Result<(), LinError>;

    /// Solve `B X = RHS` for several right-hand sides at once, each `rhs[k]` an
    /// `m`-vector solved in place. The point is to reuse the single existing
    /// factorization across all `k` solves (sensitivity / strong-branching probes
    /// / multi-rhs LPs) instead of refactorizing per system. The default loops
    /// over [`ftran`](Self::ftran); a backend with a true matrix solve may
    /// override. On the first failing column it stops and returns the error
    /// (later columns are left untouched).
    fn ftran_multi(&mut self, rhs: &mut [&mut [f64]]) -> Result<(), LinError> {
        for col in rhs.iter_mut() {
            self.ftran(col)?;
        }
        Ok(())
    }

    /// Solve `Bᵀ Y = RHS` for several right-hand sides at once (see
    /// [`ftran_multi`](Self::ftran_multi)).
    fn btran_multi(&mut self, rhs: &mut [&mut [f64]]) -> Result<(), LinError> {
        for col in rhs.iter_mut() {
            self.btran(col)?;
        }
        Ok(())
    }
}

/// Production backend: feral's unsymmetric LU, routed dense-or-sparse per basis.
///
/// B&B node bases are small and often dense (design / balance / big-M rows).
/// feral's *dense* LU factorizes in O(m³) with tiny constants and **no**
/// symbolic-analysis phase, while the sparse path re-runs
/// `SparseLuSymbolic::analyze` on every refactorization — measured at
/// ~440 µs/iteration (≈100× too slow) on a ~130-row dense basis, with healthy
/// iteration counts (so the cost was the LU, not pricing/degeneracy). We
/// therefore route small bases (and feral-judged dense ones) to `DenseLu`, and
/// leave genuinely large/sparse bases on `SparseLu`.
///
/// `Clone` duplicates the factorization itself (both feral LUs are `Clone`), so
/// a prepared basis factorization can be cloned and re-optimized from several
/// times — the dual warm-start reuses one factorization across a node's
/// strong-branching probes instead of refactorizing the identical basis per probe.
#[derive(Debug, Clone)]
enum Factored {
    // Both boxed: feral v0.11.2 grew `SparseLu` to ~736 B (vs `DenseLu` ~424 B),
    // so an unboxed enum trips clippy's `large_enum_variant` on the size gap (and
    // reserves the larger size for every `Factored`). Boxing both keeps the enum
    // to two pointers; method calls in ftran/btran/update auto-deref through the
    // boxes, so only the construction sites add `Box::new`.
    Sparse(Box<SparseLu>),
    Dense(Box<DenseLu>),
}

/// Force-dense cutoff: at or below this `m`, a dense LU is always at least as
/// fast as sparse (the O(m³) factor is cheap and there is no symbolic phase),
/// regardless of density — so route there even when feral's density test, which
/// also gates on sparsity, would not. Above it we defer to feral's heuristic.
const FORCE_DENSE_M: usize = 256;

/// Basis factorization backend: feral's LU, dispatched dense-or-sparse per
/// basis at factorize time (see the [`Factored`] docs for the rationale).
#[derive(Default, Clone)]
pub struct FeralLU {
    lu: Option<Factored>,
}

impl FeralLU {
    /// A new, unfactorized solver.
    pub fn new() -> Self {
        Self { lu: None }
    }

    /// Cumulative Forrest–Tomlin bump-update work (feral's `eta_ops`) accumulated
    /// on the current factorization since it was built. `0` for the dense backend
    /// (its updates are not product-form etas) and when unfactorized. The simplex
    /// uses this to refactorize *before* wide-bump updates compound into the
    /// O(bump²) FT blowup that dominates row-heavy McCormick bases
    /// (discopt#268 / feral#87).
    pub fn ft_update_work(&self) -> usize {
        match &self.lu {
            Some(Factored::Sparse(lu)) => lu.eta_ops(),
            _ => 0,
        }
    }

    /// nnz of the current sparse factor (`0` for the dense backend / unfactorized).
    /// The work budget the FT-update accumulation is compared against.
    pub fn factor_nnz(&self) -> usize {
        match &self.lu {
            Some(Factored::Sparse(lu)) => lu.factor_nnz(),
            _ => 0,
        }
    }
}

fn feral_err(e: feral::FeralError) -> LinError {
    LinError::Backend(format!("{e:?}"))
}

impl LinearSolver for FeralLU {
    fn factorize(&mut self, m: usize, cols: &[Vec<f64>]) -> Result<(), LinError> {
        debug_assert_eq!(cols.len(), m);
        let params = LuParams::default();
        let nnz: usize = cols
            .iter()
            .map(|c| c.iter().filter(|&&v| v != 0.0).count())
            .sum();
        self.lu = Some(
            if m <= FORCE_DENSE_M || should_use_dense_lu(m, nnz, &params) {
                Factored::Dense(Box::new(DenseLu::factor(cols, m, params).map_err(feral_err)?))
            } else {
                let a = SparseColMatrix::from_dense_columns(m, cols).map_err(feral_err)?;
                let sym = SparseLuSymbolic::analyze(&a).map_err(feral_err)?;
                Factored::Sparse(Box::new(SparseLu::factor(&a, &sym, params).map_err(feral_err)?))
            },
        );
        Ok(())
    }

    fn factorize_sparse(&mut self, m: usize, cols: &[Vec<(usize, f64)>]) -> Result<(), LinError> {
        let params = LuParams::default();
        // nnz is the sum of column lengths — O(m), not the O(m²) dense scan the
        // `Vec<Vec<f64>>` path pays.
        let nnz: usize = cols.iter().map(|c| c.len()).sum();
        self.lu = Some(
            if m <= FORCE_DENSE_M || should_use_dense_lu(m, nnz, &params) {
                // Small/dense basis: feral's dense LU wins and there is no symbolic
                // phase. Densifying is O(m² + nnz), affordable only because this
                // branch is gated to small `m` (<= FORCE_DENSE_M) or feral-judged-
                // dense bases.
                let mut dense = vec![vec![0.0; m]; m];
                for (slot, col) in cols.iter().enumerate() {
                    for &(row, v) in col {
                        dense[slot][row] = v;
                    }
                }
                Factored::Dense(Box::new(DenseLu::factor(&dense, m, params).map_err(feral_err)?))
            } else {
                // Build feral's sparse matrix directly from the sparse columns —
                // O(nnz), no dense m×m intermediate.
                let a = SparseColMatrix::from_sparse_columns(m, cols).map_err(feral_err)?;
                let sym = SparseLuSymbolic::analyze(&a).map_err(feral_err)?;
                Factored::Sparse(Box::new(SparseLu::factor(&a, &sym, params).map_err(feral_err)?))
            },
        );
        Ok(())
    }

    fn ftran(&mut self, rhs: &mut [f64]) -> Result<(), LinError> {
        match self.lu.as_mut().ok_or(LinError::NotFactorized)? {
            Factored::Sparse(lu) => lu.ftran(rhs).map_err(feral_err),
            Factored::Dense(lu) => lu.ftran(rhs).map_err(feral_err),
        }
    }

    fn btran(&mut self, rhs: &mut [f64]) -> Result<(), LinError> {
        match self.lu.as_mut().ok_or(LinError::NotFactorized)? {
            Factored::Sparse(lu) => lu.btran(rhs).map_err(feral_err),
            Factored::Dense(lu) => lu.btran(rhs).map_err(feral_err),
        }
    }

    fn update(&mut self, leaving_slot: usize, entering_col: &[f64]) -> Result<(), LinError> {
        match self.lu.as_mut().ok_or(LinError::NotFactorized)? {
            Factored::Sparse(lu) => lu.update(leaving_slot, entering_col).map_err(feral_err),
            Factored::Dense(lu) => lu.update(leaving_slot, entering_col).map_err(feral_err),
        }
    }
}

/// Dense oracle/fallback: stores the basis columns and refactorizes on every
/// solve via [`crate::lp::gomory::solve_dense`]. For bring-up / tiny bases /
/// cross-checking `FeralLU`, not performance.
#[derive(Default)]
pub struct DenseLU {
    m: usize,
    cols: Vec<Vec<f64>>,
}

impl DenseLU {
    /// A new, unfactorized solver.
    pub fn new() -> Self {
        Self {
            m: 0,
            cols: Vec::new(),
        }
    }
}

impl LinearSolver for DenseLU {
    fn factorize(&mut self, m: usize, cols: &[Vec<f64>]) -> Result<(), LinError> {
        self.m = m;
        self.cols = cols.to_vec();
        Ok(())
    }

    fn ftran(&mut self, rhs: &mut [f64]) -> Result<(), LinError> {
        // B x = rhs, with B[i][j] = cols[j][i] (column-major basis → row-major B).
        let m = self.m;
        let mut mat = vec![0.0_f64; m * m];
        for (j, col) in self.cols.iter().enumerate() {
            for (i, &v) in col.iter().enumerate() {
                mat[i * m + j] = v;
            }
        }
        let x = crate::lp::gomory::solve_dense(&mat, m, rhs, 1e-12).ok_or(LinError::Singular)?;
        rhs.copy_from_slice(&x);
        Ok(())
    }

    fn btran(&mut self, rhs: &mut [f64]) -> Result<(), LinError> {
        // Bᵀ y = rhs, with Bᵀ[i][j] = B[j][i] = cols[i][j].
        let m = self.m;
        let mut mat = vec![0.0_f64; m * m];
        for (i, col) in self.cols.iter().enumerate() {
            for (j, &v) in col.iter().enumerate() {
                mat[i * m + j] = v;
            }
        }
        let y = crate::lp::gomory::solve_dense(&mat, m, rhs, 1e-12).ok_or(LinError::Singular)?;
        rhs.copy_from_slice(&y);
        Ok(())
    }

    fn update(&mut self, leaving_slot: usize, entering_col: &[f64]) -> Result<(), LinError> {
        self.cols[leaving_slot] = entering_col.to_vec();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Matrix-vector B·x with B[i][j] = cols[j][i].
    fn bmatvec(cols: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
        let m = cols.len();
        let mut y = vec![0.0; m];
        for (j, col) in cols.iter().enumerate() {
            for (i, &v) in col.iter().enumerate() {
                y[i] += v * x[j];
            }
        }
        y
    }

    fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
    }

    #[test]
    fn feral_matches_dense_ftran_btran_and_update() {
        // B columns: [[2,1,0],[1,2,1],[0,1,2]] → SPD-ish nonsingular.
        let cols = vec![
            vec![2.0, 1.0, 0.0],
            vec![1.0, 2.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ];
        let m = 3;
        let mut fl = FeralLU::new();
        let mut dl = DenseLU::new();
        fl.factorize(m, &cols).unwrap();
        dl.factorize(m, &cols).unwrap();

        // ftran: B x = rhs.
        let rhs = [1.0, 2.0, 3.0];
        let mut xf = rhs;
        let mut xd = rhs;
        fl.ftran(&mut xf).unwrap();
        dl.ftran(&mut xd).unwrap();
        assert!(close(&xf, &xd, 1e-9), "ftran feral {xf:?} vs dense {xd:?}");
        assert!(close(&bmatvec(&cols, &xf), &rhs, 1e-9), "B·x != rhs");

        // btran: Bᵀ y = rhs.
        let mut yf = rhs;
        let mut yd = rhs;
        fl.btran(&mut yf).unwrap();
        dl.btran(&mut yd).unwrap();
        assert!(close(&yf, &yd, 1e-9), "btran feral {yf:?} vs dense {yd:?}");

        // product-form update: replace column 1 with [0,0,1] (keeps B
        // nonsingular — [1,1,1] would make it singular and feral would
        // correctly signal NeedsRefactor), then re-solve ftran.
        let newcol = [0.0, 0.0, 1.0];
        fl.update(1, &newcol).unwrap();
        dl.update(1, &newcol).unwrap();
        let mut xf2 = rhs;
        let mut xd2 = rhs;
        fl.ftran(&mut xf2).unwrap();
        dl.ftran(&mut xd2).unwrap();
        assert!(
            close(&xf2, &xd2, 1e-9),
            "post-update ftran feral {xf2:?} vs dense {xd2:?}"
        );
        let mut updated = cols.clone();
        updated[1] = newcol.to_vec();
        assert!(
            close(&bmatvec(&updated, &xf2), &rhs, 1e-9),
            "post-update B·x != rhs"
        );
    }

    #[test]
    fn multi_rhs_matches_per_column_solves() {
        // ftran_multi / btran_multi over one factorization must reproduce
        // solving each right-hand side individually.
        let cols = vec![
            vec![2.0, 1.0, 0.0],
            vec![1.0, 2.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ];
        let m = 3;
        let mut fl = FeralLU::new();
        fl.factorize(m, &cols).unwrap();

        let rhs0 = [1.0, 2.0, 3.0];
        let rhs1 = [3.0, 0.0, -1.0];

        // Reference: individual ftran.
        let (mut r0, mut r1) = (rhs0, rhs1);
        fl.ftran(&mut r0).unwrap();
        fl.ftran(&mut r1).unwrap();

        // Batched ftran_multi.
        let (mut b0, mut b1) = (rhs0, rhs1);
        {
            let mut batch: [&mut [f64]; 2] = [&mut b0, &mut b1];
            fl.ftran_multi(&mut batch).unwrap();
        }
        assert!(close(&b0, &r0, 1e-12) && close(&b1, &r1, 1e-12));
        // Each solved system genuinely satisfies B·x = rhs.
        assert!(close(&bmatvec(&cols, &b0), &rhs0, 1e-9));
        assert!(close(&bmatvec(&cols, &b1), &rhs1, 1e-9));

        // btran_multi likewise matches individual btran.
        let (mut y0, mut y1) = (rhs0, rhs1);
        fl.btran(&mut y0).unwrap();
        fl.btran(&mut y1).unwrap();
        let (mut c0, mut c1) = (rhs0, rhs1);
        {
            let mut batch: [&mut [f64]; 2] = [&mut c0, &mut c1];
            fl.btran_multi(&mut batch).unwrap();
        }
        assert!(close(&c0, &y0, 1e-12) && close(&c1, &y1, 1e-12));
    }
}
