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

use feral::{LuParams, SparseColMatrix, SparseLu, SparseLuSymbolic};

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

/// Production backend: feral's sparse unsymmetric LU.
#[derive(Default)]
pub struct FeralLU {
    lu: Option<SparseLu>,
}

impl FeralLU {
    /// A new, unfactorized solver.
    pub fn new() -> Self {
        Self { lu: None }
    }
}

fn feral_err(e: feral::FeralError) -> LinError {
    LinError::Backend(format!("{e:?}"))
}

impl LinearSolver for FeralLU {
    fn factorize(&mut self, m: usize, cols: &[Vec<f64>]) -> Result<(), LinError> {
        debug_assert_eq!(cols.len(), m);
        let a = SparseColMatrix::from_dense_columns(m, cols).map_err(feral_err)?;
        let sym = SparseLuSymbolic::analyze(&a).map_err(feral_err)?;
        let lu = SparseLu::factor(&a, &sym, LuParams::default()).map_err(feral_err)?;
        self.lu = Some(lu);
        Ok(())
    }

    fn ftran(&mut self, rhs: &mut [f64]) -> Result<(), LinError> {
        self.lu
            .as_mut()
            .ok_or(LinError::NotFactorized)?
            .ftran(rhs)
            .map_err(feral_err)
    }

    fn btran(&mut self, rhs: &mut [f64]) -> Result<(), LinError> {
        self.lu
            .as_mut()
            .ok_or(LinError::NotFactorized)?
            .btran(rhs)
            .map_err(feral_err)
    }

    fn update(&mut self, leaving_slot: usize, entering_col: &[f64]) -> Result<(), LinError> {
        self.lu
            .as_mut()
            .ok_or(LinError::NotFactorized)?
            .update(leaving_slot, entering_col)
            .map_err(feral_err)
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
