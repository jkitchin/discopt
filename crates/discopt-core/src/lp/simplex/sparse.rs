//! Compressed sparse-column (CSC) view of the constraint matrix.
//!
//! The revised simplex prices every nonbasic column each iteration (reduced cost
//! `d_j = c_j − yᵀA_j`) and, for Devex, dots the pivot row against every nonbasic
//! column. Done from the dense row-major matrix these are `O(n·m)` per pass and
//! rebuild a dense column vector each time. Building this CSC once per solve makes
//! a column dot `O(nnz_j)` and a scatter into a dense buffer `O(nnz_j)`, which is
//! what lets cut-augmented LPs (many sparse cut rows + unit slacks) stay cheap.

/// CSC storage of an `m × n` matrix: `col_ptr[j..j+1]` bounds column `j`'s
/// nonzeros in `row_idx`/`vals`.
pub struct SparseCols {
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    vals: Vec<f64>,
}

impl SparseCols {
    /// Build from a dense row-major `m × n` matrix (one `O(m·n)` pass per solve).
    pub fn from_dense(a: &[f64], m: usize, n: usize) -> Self {
        let mut col_ptr = vec![0usize; n + 1];
        for i in 0..m {
            let row = &a[i * n..(i + 1) * n];
            for j in 0..n {
                if row[j] != 0.0 {
                    col_ptr[j + 1] += 1;
                }
            }
        }
        for j in 0..n {
            col_ptr[j + 1] += col_ptr[j];
        }
        let nnz = col_ptr[n];
        let mut row_idx = vec![0usize; nnz];
        let mut vals = vec![0.0; nnz];
        let mut pos = col_ptr.clone();
        for i in 0..m {
            let row = &a[i * n..(i + 1) * n];
            for j in 0..n {
                let v = row[j];
                if v != 0.0 {
                    let p = pos[j];
                    row_idx[p] = i;
                    vals[p] = v;
                    pos[j] = p + 1;
                }
            }
        }
        Self {
            col_ptr,
            row_idx,
            vals,
        }
    }

    /// Read-only view of the raw CSC arrays (`col_ptr`, `row_idx`, `vals`). Lets
    /// the equilibration sweeps read every nonzero in storage order (cache-
    /// friendly) instead of striding the dense row-major matrix per column.
    #[inline]
    pub fn raw(&self) -> (&[usize], &[usize], &[f64]) {
        (&self.col_ptr, &self.row_idx, &self.vals)
    }

    /// Nonzero `(row, value)` pairs of column `j`.
    #[inline]
    pub fn col(&self, j: usize) -> (&[usize], &[f64]) {
        let s = self.col_ptr[j];
        let e = self.col_ptr[j + 1];
        (&self.row_idx[s..e], &self.vals[s..e])
    }

    /// Sparse dot `yᵀ A_j`.
    #[inline]
    pub fn dot(&self, j: usize, y: &[f64]) -> f64 {
        let (rows, vals) = self.col(j);
        let mut s = 0.0;
        for (k, &r) in rows.iter().enumerate() {
            s += y[r] * vals[k];
        }
        s
    }

    /// Scatter column `j` into a dense buffer (only the nonzero rows are written;
    /// the caller must supply a zeroed buffer of length `m`).
    #[inline]
    pub fn scatter(&self, j: usize, buf: &mut [f64]) {
        let (rows, vals) = self.col(j);
        for (k, &r) in rows.iter().enumerate() {
            buf[r] = vals[k];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csc_roundtrip_dot_scatter() {
        // 3×3 dense, row-major.
        let a = [2.0, 0.0, 1.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0];
        let sp = SparseCols::from_dense(&a, 3, 3);
        // column 0 = [2,0,4]
        let (r, v) = sp.col(0);
        assert_eq!(r, &[0, 2]);
        assert_eq!(v, &[2.0, 4.0]);
        // dot with y = [1,1,1] → col sums
        let y = [1.0, 1.0, 1.0];
        assert!((sp.dot(0, &y) - 6.0).abs() < 1e-12);
        assert!((sp.dot(1, &y) - 3.0).abs() < 1e-12);
        assert!((sp.dot(2, &y) - 6.0).abs() < 1e-12);
        // scatter column 2 = [1,0,5]
        let mut buf = vec![0.0; 3];
        sp.scatter(2, &mut buf);
        assert_eq!(buf, vec![1.0, 0.0, 5.0]);
    }
}
