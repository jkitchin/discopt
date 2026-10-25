//! #1935 entry experiment: how much of a node-LP basis factorization is spent on
//! the slack columns? Reads bases dumped from captured kernel node LPs and times
//! feral's `factorize_sparse` on (a) the whole `m x m` basis and (b) only its
//! structural nucleus -- the basic structural columns restricted to the rows no
//! basic slack covers. Every basic slack is an identity column, so the basis is
//! block-triangular `[[N, 0], [X, I]]` after a permutation and only `N` needs an
//! LU. Usage: `cargo run --release --example issue1935_lu_nucleus -- bases.txt`.
use discopt_core::lp::simplex::linsolve::{FeralLU, LinearSolver};
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).expect("bases file");
    let text = std::fs::read_to_string(path).expect("read");
    let mut lines = text.lines();
    let k: usize = lines.next().unwrap().trim().parse().unwrap();
    let (mut full_ns, mut nuc_ns, mut checked) = (0u128, 0u128, 0usize);
    let (mut sum_m, mut sum_nuc) = (0usize, 0usize);
    const REPS: usize = 20;
    for _ in 0..k {
        let hdr: Vec<usize> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|t| t.parse().unwrap())
            .collect();
        let (m, n_struct) = (hdr[0], hdr[1]);
        let mut cols: Vec<(usize, Vec<(usize, f64)>)> = Vec::with_capacity(m);
        for _ in 0..m {
            let t: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
            let j: usize = t[0].parse().unwrap();
            let nnz: usize = t[1].parse().unwrap();
            let col = (0..nnz)
                .map(|q| (t[2 + 2 * q].parse().unwrap(), t[3 + 2 * q].parse().unwrap()))
                .collect();
            cols.push((j, col));
        }
        let full: Vec<Vec<(usize, f64)>> = cols.iter().map(|(_, c)| c.clone()).collect();
        // Nucleus: rows not covered by a basic slack, reindexed.
        let mut covered = vec![false; m];
        for (j, _) in &cols {
            if *j >= n_struct {
                covered[*j - n_struct] = true;
            }
        }
        let mut new_row = vec![usize::MAX; m];
        let mut r2 = 0;
        for r in 0..m {
            if !covered[r] {
                new_row[r] = r2;
                r2 += 1;
            }
        }
        let nuc: Vec<Vec<(usize, f64)>> = cols
            .iter()
            .filter(|(j, _)| *j < n_struct)
            .map(|(_, c)| {
                c.iter()
                    .filter(|(r, _)| !covered[*r])
                    .map(|&(r, v)| (new_row[r], v))
                    .collect()
            })
            .collect();
        assert_eq!(nuc.len(), r2, "nucleus not square");
        sum_m += m;
        sum_nuc += r2;
        for _ in 0..REPS {
            let mut lu = FeralLU::new();
            let t = Instant::now();
            let ok = lu.factorize_sparse(m, &full).is_ok();
            full_ns += t.elapsed().as_nanos();
            let mut lu2 = FeralLU::new();
            let t = Instant::now();
            let ok2 = r2 == 0 || lu2.factorize_sparse(r2, &nuc).is_ok();
            nuc_ns += t.elapsed().as_nanos();
            // Same singularity verdict: the slack block is the identity.
            assert_eq!(ok, ok2, "full and nucleus disagree on singularity");
            checked += 1;
        }
    }
    let per = |ns: u128| ns as f64 / 1e6 / checked as f64;
    println!(
        "bases={k} reps={REPS} checked={checked} mean m={:.0} mean nucleus={:.0} | \
         full {:.3} ms/factorize, nucleus {:.3} ms/factorize, ratio {:.1}x",
        sum_m as f64 / k as f64,
        sum_nuc as f64 / k as f64,
        per(full_ns),
        per(nuc_ns),
        full_ns as f64 / nuc_ns.max(1) as f64
    );
    assert!(checked > 0, "measured nothing");
}
