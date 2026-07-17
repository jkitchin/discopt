//! Wall-clock benchmark for the Rust-internal MILP driver.
//!
//! Two regimes that exercise the data/cut layer differently:
//!   * sparse set-covering (each column covers ~6 of `nrow` rows) — the regime
//!     where dense O(m³) Gomory separation and cold-per-step diving used to
//!     dominate the root; sparse-LU Gomory + warm-start diving fixed both;
//!   * dense multidimensional knapsack (regime B) — the small/dense guard that
//!     must not regress.
//!
//! Reports status / nodes / objective / wall time per instance. No external
//! deps; for the SCIP comparison see `discopt_benchmarks/bench_milp_sparse.py`.
//!
//! Run: `cargo run --release --example milp_bench -p discopt-core`

use discopt_core::bnb::milp_driver::{solve_milp, MilpOptions, MilpStatus};
use discopt_core::lp::crossover::LpView;
use discopt_core::lp::simplex::SimplexOptions;
use std::time::Instant;

const INF: f64 = 1e20;

/// Deterministic LCG (no RNG dependency).
struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }
}

#[allow(clippy::type_complexity)]
fn gen_setcover(
    ncol: usize,
    nrow: usize,
    seed: u64,
    per_col: usize,
) -> (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    usize,
    Vec<usize>,
) {
    let mut rng = Lcg(seed.wrapping_add(0x9E37_79B9_7F4A_7C15));
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
            let j = rng.below(ncol);
            cols[j].push(r);
        }
    }
    let cost: Vec<f64> = (0..ncol).map(|_| (1 + rng.below(99)) as f64).collect();
    let n_struct = ncol;
    let n = ncol + nrow;
    let m = nrow;
    let mut a = vec![0.0f64; m * n];
    for (j, c) in cols.iter().enumerate() {
        for &r in c {
            a[r * n + j] = 1.0;
        }
    }
    for i in 0..m {
        a[i * n + (ncol + i)] = -1.0; // Σ x − s = 1, s ≥ 0
    }
    let b = vec![1.0f64; m];
    let mut cvec = vec![0.0f64; n];
    cvec[..ncol].copy_from_slice(&cost);
    let l = vec![0.0f64; n];
    let mut u = vec![INF; n];
    for uj in u.iter_mut().take(ncol) {
        *uj = 1.0;
    }
    (a, b, cvec, l, u, n_struct, (0..ncol).collect())
}

/// Dense multidimensional knapsack (regime B): `kdim` capacity rows over `n`
/// binaries, all coefficients dense. Standard form `Σ a_ij x_j + s_i = cap_i`.
#[allow(clippy::type_complexity)]
fn gen_mdknapsack(
    n: usize,
    kdim: usize,
    seed: u64,
) -> (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    usize,
    Vec<usize>,
) {
    let mut rng = Lcg(seed.wrapping_add(0x1234_5678));
    let ncol = n + kdim; // structural + one slack per capacity row
    let m = kdim;
    let mut a = vec![0.0f64; m * ncol];
    let mut b = vec![0.0f64; m];
    for i in 0..m {
        let mut rowsum = 0.0;
        for j in 0..n {
            let w = (1 + rng.below(50)) as f64;
            a[i * ncol + j] = w;
            rowsum += w;
        }
        a[i * ncol + (n + i)] = 1.0; // slack
        b[i] = 0.5 * rowsum; // ~half the total weight as capacity
    }
    let profit: Vec<f64> = (0..n).map(|_| -((1 + rng.below(50)) as f64)).collect(); // minimize −profit
    let mut c = vec![0.0f64; ncol];
    c[..n].copy_from_slice(&profit);
    let l = vec![0.0f64; ncol];
    let mut u = vec![INF; ncol];
    for uj in u.iter_mut().take(n) {
        *uj = 1.0;
    }
    (a, b, c, l, u, n, (0..n).collect())
}

fn opts(n_struct: usize, integer_cols: Vec<usize>, tl: f64) -> MilpOptions {
    MilpOptions {
        n_struct,
        integer_cols,
        max_nodes: 5_000_000,
        time_limit_s: Some(tl),
        gap_tol: 1e-6,
        root_cuts: 16,
        cut_rounds: 1,
        gmi_cuts: true,
        cut_select: false,
        node_cuts: false,
        max_pool_cuts: 128,
        heuristics: true,
        presolve: true,
        strong_branch: true,
        node_propagation: false,
        reduced_cost_fixing: false,
        sb_max_cands: 8,
        sb_node_budget: 128,
        initial_incumbent: None,
        simplex: SimplexOptions {
            tol: 1e-9,
            max_iter: 100_000,
            deadline: None,
            warm_stall_guard: true,
            warm_stall_cap_override: None,
        },
    }
}

fn st_name(s: MilpStatus) -> &'static str {
    match s {
        MilpStatus::Optimal => "optimal",
        MilpStatus::Feasible => "feasible",
        MilpStatus::Infeasible => "infeasible",
        MilpStatus::Unbounded => "unbounded",
        MilpStatus::NodeLimit => "node_limit",
    }
}

fn run(
    label: &str,
    a: &[f64],
    b: &[f64],
    c: &[f64],
    l: &[f64],
    u: &[f64],
    m: usize,
    n: usize,
    ns: usize,
    ints: Vec<usize>,
    tl: f64,
) {
    let lp = LpView { a, m, n, c, l, u };
    let t0 = Instant::now();
    let r = solve_milp(&lp, b, 0.0, &opts(ns, ints, tl));
    let wall = t0.elapsed().as_secs_f64();
    println!(
        "{label:16}  {:10}  nodes={:>6}  obj={:>10.2}  wall={wall:>7.2}s",
        st_name(r.status),
        r.nodes,
        r.obj
    );
}

fn main() {
    let tl: f64 = std::env::var("TL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60.0);
    println!("# MILP driver bench (TL={tl}s)");
    println!("## sparse set-covering (per_col=6)");
    for &(nc, nr) in &[(500usize, 250usize), (1000, 500), (2000, 800), (4000, 1500)] {
        let (a, b, c, l, u, ns, ints) = gen_setcover(nc, nr, 3, 6);
        run(
            &format!("sc{nc}x{nr}"),
            &a,
            &b,
            &c,
            &l,
            &u,
            nr,
            nc + nr,
            ns,
            ints,
            tl,
        );
    }
    println!("## dense multidim knapsack (regime B)");
    for &(n, k) in &[(50usize, 5usize), (150, 8), (250, 10)] {
        let (a, b, c, l, u, ns, ints) = gen_mdknapsack(n, k, 7);
        run(
            &format!("mdk{n}x{k}"),
            &a,
            &b,
            &c,
            &l,
            &u,
            k,
            n + k,
            ns,
            ints,
            tl,
        );
    }
}
