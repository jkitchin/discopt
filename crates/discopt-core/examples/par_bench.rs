//! Timing benchmark for the MILP B&B driver: serial vs. rayon-parallel.
//!
//! Builds a suite of deterministic multidimensional 0/1 knapsack instances
//! (hard enough to grow a real B&B tree) and times `solve_milp` over all of
//! them. The driver picks the serial or parallel batch loop depending on whether
//! the crate is built with the `parallel` feature, so compare:
//!
//! ```bash
//! cargo run --release --example par_bench                     # serial
//! cargo run --release --example par_bench --features parallel # parallel
//! RAYON_NUM_THREADS=4 cargo run --release --example par_bench --features parallel
//! ```
//!
//! It prints wall-clock time, total B&B nodes, and total simplex pivots so the
//! node/pivot counts can be checked identical (correctness) while time drops.

use std::time::Instant;

use discopt_core::bnb::milp_driver::{solve_milp, MilpOptions, MilpStatus};
use discopt_core::lp::crossover::LpView;
use discopt_core::lp::simplex::SimplexOptions;

const INF: f64 = 1e20;

/// Deterministic linear-congruential generator (reproducible across runs/builds).
struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        // Numerical Recipes constants.
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    /// Integer in `[lo, hi]`.
    fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.next_u64() % (hi - lo + 1)
    }
}

/// One built MILP instance in `LpView` form (owned storage).
struct Instance {
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    l: Vec<f64>,
    u: Vec<f64>,
    m: usize,
    n: usize,
    n_struct: usize,
    integer_cols: Vec<usize>,
}

/// Build a multidimensional 0/1 knapsack: maximize `vᵀx` s.t. `m` capacity rows
/// `Wᵢ·x ≤ capᵢ`, `x ∈ {0,1}^n`. Correlated weights/values make it non-trivial.
/// Converted to `min -vᵀx` with one nonnegative slack per row.
fn build_knapsack(n: usize, m: usize, seed: u64) -> Instance {
    let mut rng = Lcg(seed);
    let n_full = n + m; // structural + slacks
    let mut a = vec![0.0; m * n_full];
    let mut b = vec![0.0; m];
    let mut c = vec![0.0; n_full];
    let mut l = vec![0.0; n_full];
    let mut u = vec![0.0; n_full];

    // Weights per (row, item) and values (value correlated with total weight).
    let mut total_w = vec![0u64; m];
    for j in 0..n {
        let mut wsum = 0u64;
        for i in 0..m {
            let w = rng.range(1, 100);
            a[i * n_full + j] = w as f64;
            total_w[i] += w;
            wsum += w;
        }
        // Correlated value: average weight plus a small bonus.
        let avg = (wsum / m as u64) + rng.range(1, 20);
        c[j] = -(avg as f64); // minimize -value
        l[j] = 0.0;
        u[j] = 1.0;
    }
    // Slack columns: identity, nonnegative, zero cost.
    for i in 0..m {
        a[i * n_full + (n + i)] = 1.0;
        c[n + i] = 0.0;
        l[n + i] = 0.0;
        u[n + i] = INF;
        // Capacity ~ half the total weight (the classic hard regime).
        b[i] = (total_w[i] as f64) * 0.5;
    }

    Instance {
        a,
        b,
        c,
        l,
        u,
        m,
        n: n_full,
        n_struct: n,
        integer_cols: (0..n).collect(),
    }
}

fn opts(inst: &Instance) -> MilpOptions {
    MilpOptions {
        n_struct: inst.n_struct,
        integer_cols: inst.integer_cols.clone(),
        max_nodes: 2_000_000,
        time_limit_s: None,
        gap_tol: 1e-6,
        root_cuts: 16,
        cut_rounds: 2,
        node_cuts: true,
        max_pool_cuts: 256,
        heuristics: true,
        presolve: true,
        strong_branch: true,
        sb_max_cands: 8,
        sb_node_budget: 4096,
        simplex: SimplexOptions {
            tol: 1e-9,
            max_iter: 100_000,
        },
    }
}

fn main() {
    // A spread of sizes/seeds; each is solved to optimality.
    let configs: &[(usize, usize, u64)] = &[
        (34, 4, 1),
        (36, 5, 2),
        (38, 4, 3),
        (34, 5, 4),
        (40, 4, 5),
        (36, 4, 6),
    ];
    let instances: Vec<Instance> = configs
        .iter()
        .map(|&(n, m, s)| build_knapsack(n, m, s))
        .collect();

    #[cfg(feature = "parallel")]
    let mode = format!("PARALLEL (rayon, {} threads)", rayon::current_num_threads());
    #[cfg(not(feature = "parallel"))]
    let mode = "SERIAL".to_string();
    println!("mode: {mode}");

    let mut total_nodes = 0usize;
    let mut total_iters = 0usize;
    let mut checksum = 0.0f64;

    let start = Instant::now();
    for (k, inst) in instances.iter().enumerate() {
        let lp = LpView {
            a: &inst.a,
            m: inst.m,
            n: inst.n,
            c: &inst.c,
            l: &inst.l,
            u: &inst.u,
        };
        let res = solve_milp(&lp, &inst.b, 0.0, &opts(inst));
        assert_eq!(res.status, MilpStatus::Optimal, "instance {k} not optimal");
        total_nodes += res.nodes;
        total_iters += res.lp_iters;
        checksum += res.obj;
        println!(
            "  inst {k}: obj={:.1} nodes={} iters={}",
            res.obj, res.nodes, res.lp_iters
        );
    }
    let elapsed = start.elapsed();

    println!("---");
    println!("total wall: {:.3} s", elapsed.as_secs_f64());
    println!("total nodes: {total_nodes}");
    println!("total iters: {total_iters}");
    println!("obj checksum: {checksum:.4}");
}
