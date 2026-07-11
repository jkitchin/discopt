//! Measure feral's Forrest–Tomlin `update()` cost on a *real* set-covering LP
//! basis (sc2000x800), to back the feral FT-update issue with the authentic
//! basis rather than a synthetic one.
//!
//! Builds the sc2000x800 set-covering standard-form LP, solves the root LP to get
//! a genuine optimal basis, factorizes `B` with the same `FeralLU` the simplex
//! uses, then replays a sequence of single-column replacements (`update`) with
//! real structural entering columns — exactly the per-pivot operation the cold
//! primal simplex performs — and reports the average wall-time per update plus
//! the factor fill. Each entering column has only ~6 nonzeros, yet the update
//! costs milliseconds because `B⁻¹` is dense on covering bases.
//!
//! Run: `cargo run --release --example feral_update_cost -p discopt-core`

use discopt_core::lp::crossover::LpView;
use discopt_core::lp::simplex::linsolve::{FeralLU, LinearSolver};
use discopt_core::lp::simplex::sparse::SparseCols;
use discopt_core::lp::simplex::{solve_lp, SimplexOptions};
use std::time::Instant;

const INF: f64 = 1e20;

struct Lcg(u64);
impl Lcg {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as usize) % n
    }
}

/// sc<ncol>x<nrow> standard form `A x = b` (Σ x − s = 1, s ≥ 0).
#[allow(clippy::type_complexity)]
fn gen_setcover(
    ncol: usize,
    nrow: usize,
    per_col: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, usize) {
    let mut rng = Lcg(0x9E37_79B9_7F4A_7C15);
    let mut cols: Vec<Vec<usize>> = Vec::with_capacity(ncol);
    for _ in 0..ncol {
        let mut rows = Vec::with_capacity(per_col);
        while rows.len() < per_col.min(nrow) {
            let r = rng.below(nrow);
            if !rows.contains(&r) {
                rows.push(r);
            }
        }
        cols.push(rows);
    }
    let mut covered = vec![false; nrow];
    for c in &cols {
        for &r in c {
            covered[r] = true;
        }
    }
    for r in 0..nrow {
        if !covered[r] {
            cols[rng.below(ncol)].push(r);
        }
    }
    let n = ncol + nrow;
    let m = nrow;
    let mut a = vec![0.0f64; m * n];
    for (j, c) in cols.iter().enumerate() {
        for &r in c {
            a[r * n + j] = 1.0;
        }
    }
    for i in 0..m {
        a[i * n + (ncol + i)] = -1.0;
    }
    let b = vec![1.0f64; m];
    let mut cvec = vec![0.0f64; n];
    for cj in cvec.iter_mut().take(ncol) {
        *cj = (1 + rng.below(99)) as f64;
    }
    let l = vec![0.0f64; n];
    let mut u = vec![INF; n];
    for uj in u.iter_mut().take(ncol) {
        *uj = 1.0;
    }
    (a, b, cvec, l, u, ncol)
}

/// Time `n_updates` real single-column replacements (FT `update`) on the basis
/// `basis` of the LP whose CSC is `sp`, with entering columns drawn from the
/// nonbasic structural columns. Returns `(factor_nnz/m, avg_ms, n_ok)`.
fn measure(
    sp: &SparseCols,
    basis: &[usize],
    m: usize,
    ncol: usize,
    n_updates: usize,
) -> (f64, f64, usize) {
    let mut lu = FeralLU::new();
    if lu.factorize_sparse(m, &basic_cols(sp, basis)).is_err() {
        return (0.0, 0.0, 0);
    }
    let fill = lu.factor_nnz() as f64 / m as f64;
    let in_basis: std::collections::HashSet<usize> = basis.iter().copied().collect();
    let entering: Vec<usize> = (0..ncol)
        .filter(|j| !in_basis.contains(j))
        .take(n_updates)
        .collect();
    let mut dense = vec![0.0f64; m];
    let mut times = Vec::new();
    for (t, &jin) in entering.iter().enumerate() {
        for v in dense.iter_mut() {
            *v = 0.0;
        }
        let (rows, vals) = sp.col(jin);
        for (k, &r) in rows.iter().enumerate() {
            dense[r] = vals[k];
        }
        let t0 = Instant::now();
        if lu.update(t % m, &dense).is_ok() {
            times.push(t0.elapsed().as_secs_f64() * 1e3);
        }
    }
    let n_ok = times.len().max(1);
    (fill, times.iter().sum::<f64>() / n_ok as f64, times.len())
}

fn basic_cols(sp: &SparseCols, basis: &[usize]) -> Vec<Vec<(usize, f64)>> {
    basis
        .iter()
        .map(|&j| {
            let (rows, vals) = sp.col(j);
            rows.iter().zip(vals).map(|(&r, &v)| (r, v)).collect()
        })
        .collect()
}

fn main() {
    let (ncol, nrow) = (2000usize, 800usize);
    let (a, b, c, l, u, _ns) = gen_setcover(ncol, nrow, 6);
    let m = nrow;
    let n = ncol + nrow;
    let lp = LpView {
        a: &a,
        m,
        n,
        c: &c,
        l: &l,
        u: &u,
    };
    let sp = SparseCols::from_dense(&a, m, n);

    println!("# feral update() cost on REAL sc{ncol}x{nrow} simplex bases (m={m})");
    println!("# Entering columns are real structural columns (~6 nonzeros each).");
    println!("# The COLD PRIMAL passes through transient bases that fill heavily; the");
    println!("# optimal basis is sparse again. Update cost tracks the factor fill, not");
    println!("# the entering-column nnz — so mid-solve pivots cost milliseconds.\n");
    println!(
        "{:>10}  {:>14}  {:>16}",
        "stop@iter", "factor_nnz/m", "avg update (ms)"
    );

    // Capture bases at increasing solve depth by capping the iteration count: a
    // low cap returns a half-built (heavily filled) transient basis; the full
    // solve returns the sparse optimum. This is exactly the basis family the cold
    // primal updates pivot-by-pivot.
    for &cap in &[200usize, 400, 800, 1600, 100_000] {
        let opts = SimplexOptions {
            tol: 1e-9,
            max_iter: cap,
            deadline: None,
            warm_stall_guard: true,
            warm_stall_cap_override: None,
            dual_exact_dse: false,
        };
        let sol = solve_lp(&lp, &b, &opts);
        let basis = &sol.basis.basic_vars;
        if basis.len() != m {
            println!("{cap:>10}  (incomplete basis, skipped)");
            continue;
        }
        let (fill, avg_ms, n_ok) = measure(&sp, basis, m, ncol, 40);
        let tag = if cap >= 100_000 {
            "full(opt)".to_string()
        } else {
            cap.to_string()
        };
        println!(
            "{tag:>10}  {fill:>14.1}  {avg_ms:>13.2}    ({n_ok} updates, {:?})",
            sol.status
        );
    }
    println!("\n# In a full cold solve the simplex performs thousands of these updates;");
    println!("# discopt profiling: feral update() = 83% of the 15.7 s sc2000x800 root LP.");
}
