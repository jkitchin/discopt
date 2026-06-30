//! Issue #331 — pure-MILP node-efficiency ablation on multidim 0/1 knapsacks.
//!
//! Step 1 of the issue ("Reproduce & attribute") requires a *committed* micro-
//! bench that builds the multidim-knapsack family and **ablates the `MilpOptions`
//! levers independently**, so the per-lever node reduction is on record before any
//! engine change. This is that bench.
//!
//! For each instance it solves once under the production baseline (the exact
//! defaults the Python `solve_milp_py` binding passes) and once per single-lever
//! change, then prints node count / wall / objective for every config plus an
//! aggregate "node reduction attributable to each lever" summary. The objective is
//! asserted identical to the baseline across every config — a soundness tripwire:
//! no lever may change the optimum.
//!
//! SCIP is not a dependency here (and is absent from CI); this bench measures the
//! *engine's own* node response to each lever, which is what tells us whether the
//! extra nodes (vs SCIP) come from weak BOUNDS (cuts/presolve/fixing) or weak
//! BRANCHING. The SCIP wall/node comparison lives in
//! `discopt_benchmarks/bench_milp_sparse.py`.
//!
//! Run: `cargo run --release --example mdk_ablation -p discopt-core`
//!   env: `TL=<sec>` per-solve wall cap (default 30); `SIZES=small` for a quick subset.

// A standalone bench: the config table is naturally a vec of (label, fn-ptr), and
// the instance-builder loops index parallel slices by row/col (matches the cover/
// simplex modules), so silence the two style lints those shapes trip.
#![allow(clippy::type_complexity, clippy::needless_range_loop)]

use discopt_core::bnb::milp_driver::{solve_milp, MilpOptions, MilpStatus};
use discopt_core::lp::crossover::LpView;
use discopt_core::lp::simplex::SimplexOptions;
use std::time::Instant;

const INF: f64 = 1e20;

/// Deterministic LCG (no RNG dependency, fully reproducible across runs/machines).
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

/// Weakly-correlated multidimensional 0/1 knapsack (the classic *hard* family;
/// uncorrelated profits are trivially solved at the root and show nothing under
/// ablation). `kdim` capacity rows over `n` binaries, dense coefficients:
///   maximize  Σ p_j x_j           (engine minimizes −Σ p_j x_j)
///   s.t.      Σ_j w_ij x_j ≤ c_i   for each dim i,   x ∈ {0,1}.
/// Standard form `Σ w_ij x_j + s_i = c_i`, `s_i ≥ 0`. Weights w_ij ∈ [1,100],
/// capacity c_i = ⌊α·Σ_j w_ij⌋ (α = 0.5), profit p_j = ⌊avg_i w_ij⌋ + U[1,20]
/// (lightly correlated to the weights — the hardness knob). Calibrated so the
/// baseline proves optimality in well under the per-solve cap across the family,
/// which keeps the ablation apples-to-apples (every config solves to optimum).
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
    let mut wsum = vec![0.0f64; n]; // Σ_i w_ij, for the correlated profit
    for i in 0..m {
        let mut rowsum = 0.0;
        for j in 0..n {
            let w = (1 + rng.below(100)) as f64;
            a[i * ncol + j] = w;
            wsum[j] += w;
            rowsum += w;
        }
        a[i * ncol + (n + i)] = 1.0; // slack
        b[i] = (0.5 * rowsum).floor(); // α = 0.5 of the total weight
    }
    // Correlated profit: average weight across dims plus a bounded noise term.
    let profit: Vec<f64> = (0..n)
        .map(|j| {
            let avg = (wsum[j] / m as f64).floor();
            -(avg + (1 + rng.below(20)) as f64) // minimize −profit
        })
        .collect();
    let mut c = vec![0.0f64; ncol];
    c[..n].copy_from_slice(&profit);
    let l = vec![0.0f64; ncol];
    let mut u = vec![INF; ncol];
    for uj in u.iter_mut().take(n) {
        *uj = 1.0;
    }
    (a, b, c, l, u, n, (0..n).collect())
}

/// Sparse set-covering instance (the *other* MILP regime the repo guards — many
/// `≥` rows, each column covering ~`per_col` of `nrow` rows). Standard form
/// `Σ x − s = 1`, `s ≥ 0`. Used only by the cross-regime strong-branch-budget
/// safety check below: strong-branch probes are cheap on few-row knapsacks but
/// expensive here (each probe re-solves a many-row LP), so a budget that helps
/// knapsack can regress covering — exactly the wall-time trap the issue forbids.
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
    // Guarantee every row coverable (else trivially infeasible).
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
    let n = ncol + nrow;
    let m = nrow;
    let mut a = vec![0.0f64; m * n];
    for (j, c) in cols.iter().enumerate() {
        for &r in c {
            a[r * n + j] = 1.0;
        }
    }
    for i in 0..m {
        a[i * n + (ncol + i)] = -1.0; // Σ x − s = 1, s ≥ 0  (a ≥ row)
    }
    let b = vec![1.0f64; m];
    let mut cvec = vec![0.0f64; n];
    cvec[..ncol].copy_from_slice(&cost);
    let l = vec![0.0f64; n];
    let mut u = vec![INF; n];
    for uj in u.iter_mut().take(ncol) {
        *uj = 1.0;
    }
    (a, b, cvec, l, u, ncol, (0..ncol).collect())
}

/// Production baseline — the exact defaults the Python `solve_milp_py` PyO3
/// signature carries (so the ablation toggles against what actually ships).
fn baseline(n_struct: usize, integer_cols: Vec<usize>, tl: f64) -> MilpOptions {
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
        reduced_cost_fixing: true,
        sb_max_cands: 6,
        sb_node_budget: 48,
        simplex: SimplexOptions {
            tol: 1e-9,
            max_iter: 100_000,
            deadline: None,
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

struct Run {
    nodes: usize,
    wall: f64,
    obj: f64,
    status: MilpStatus,
}

#[allow(clippy::too_many_arguments)]
fn solve(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    l: &[f64],
    u: &[f64],
    m: usize,
    n: usize,
    opts: &MilpOptions,
) -> Run {
    let lp = LpView { a, m, n, c, l, u };
    let t0 = Instant::now();
    let r = solve_milp(&lp, b, 0.0, opts);
    Run {
        nodes: r.nodes,
        wall: t0.elapsed().as_secs_f64(),
        obj: r.obj,
        status: r.status,
    }
}

/// The ablation grid: a label, and a mutation applied to a fresh baseline.
/// Each is a *single* change vs. baseline except the two reference rows
/// (`min(no-levers)` and the two stacked-cut rows), which are labelled as such.
fn configs() -> Vec<(&'static str, fn(&mut MilpOptions))> {
    vec![
        ("baseline", |_o| {}),
        ("-root_cuts", |o| o.root_cuts = 0),
        ("-gmi(cover only)", |o| o.gmi_cuts = false),
        ("-presolve", |o| o.presolve = false),
        ("-strong_branch", |o| o.strong_branch = false),
        ("-reduced_cost_fix", |o| o.reduced_cost_fixing = false),
        ("-heuristics", |o| o.heuristics = false),
        ("+node_cuts", |o| o.node_cuts = true),
        ("+node_prop(FBBT)", |o| o.node_propagation = true),
        ("+cut_rounds=5", |o| o.cut_rounds = 5),
        ("+cutsel(r=8,c=48)", |o| {
            o.cut_rounds = 8;
            o.root_cuts = 48;
            o.cut_select = true;
        }),
        ("+sb_budget=5k", |o| o.sb_node_budget = 5_000),
        ("min(pure B&B)", |o| {
            o.root_cuts = 0;
            o.strong_branch = false;
            o.presolve = false;
            o.reduced_cost_fixing = false;
            o.heuristics = false;
            o.node_propagation = false;
        }),
    ]
}

fn main() {
    let tl: f64 = std::env::var("TL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30.0);
    let small = std::env::var("SIZES").map(|s| s == "small").unwrap_or(false);

    // Issue #331 instance family: mdk{items}x{dims}.
    let all_sizes: &[(usize, usize)] = &[
        (30, 5),
        (40, 5),
        (50, 8),
        (60, 8),
        (70, 10),
        (90, 12),
        (120, 15),
        (150, 20),
        (200, 25),
    ];
    let sizes: &[(usize, usize)] = if small { &all_sizes[..4] } else { all_sizes };
    let cfgs = configs();

    println!("# Issue #331 — MDK node-efficiency ablation (TL={tl}s/solve)");
    println!(
        "# baseline = production solve_milp_py defaults; each row = one lever changed vs baseline.\n"
    );

    // Aggregate: geometric-mean node ratio (config / baseline) across instances,
    // counting only instances both solved to optimality (apples-to-apples).
    let mut log_ratio_sum = vec![0.0f64; cfgs.len()];
    let mut ratio_count = vec![0usize; cfgs.len()];

    for &(n, k) in sizes {
        let (a, b, c, l, u, ns, ints) = gen_mdknapsack(n, k, 7);
        let nn = n + k;
        println!("## mdk{n}x{k}  (n_struct={ns}, rows={k})");
        println!(
            "{:<20} {:>8} {:>10} {:>9} {:>7}  {:>7}",
            "config", "status", "obj", "nodes", "wall", "ratio"
        );
        let mut base_nodes = 0usize;
        let mut base_obj = f64::NAN;
        for (ci, (label, mutate)) in cfgs.iter().enumerate() {
            let mut o = baseline(ns, ints.clone(), tl);
            mutate(&mut o);
            let r = solve(&a, &b, &c, &l, &u, k, nn, &o);
            if ci == 0 {
                base_nodes = r.nodes;
                base_obj = r.obj;
            } else if r.status == MilpStatus::Optimal
                && base_obj.is_finite()
                && (r.obj - base_obj).abs() > 1e-4 * (1.0 + base_obj.abs())
            {
                // Soundness tripwire: a lever changed the proven optimum.
                println!(
                    "  !! SOUNDNESS: {label} obj {:.4} != baseline {:.4}",
                    r.obj, base_obj
                );
            }
            let ratio = if base_nodes > 0 {
                r.nodes as f64 / base_nodes as f64
            } else {
                1.0
            };
            // Only accumulate apples-to-apples (both optimal, not capped).
            if ci > 0 && r.status == MilpStatus::Optimal && base_nodes > 0 {
                log_ratio_sum[ci] += ratio.max(1e-9).ln();
                ratio_count[ci] += 1;
            }
            println!(
                "{:<20} {:>8} {:>10.1} {:>9} {:>6.2}s  {:>6.2}x",
                label,
                st_name(r.status),
                r.obj,
                r.nodes,
                r.wall,
                ratio
            );
        }
        println!();
    }

    println!("## Aggregate node ratio vs baseline (geomean over optimally-solved instances)");
    println!("# >1 means the change EXPANDS the tree (so the lever, when on, shrinks it).");
    println!("{:<20} {:>10} {:>8}", "config", "geomean", "n_inst");
    for (ci, (label, _)) in cfgs.iter().enumerate() {
        if ci == 0 {
            continue;
        }
        if ratio_count[ci] == 0 {
            println!("{:<20} {:>10} {:>8}", label, "—", 0);
            continue;
        }
        let g = (log_ratio_sum[ci] / ratio_count[ci] as f64).exp();
        println!("{:<20} {:>9.2}x {:>8}", label, g, ratio_count[ci]);
    }

    // --- Cross-regime safety check: strong-branch budget vs problem shape ---
    // The knapsack ablation shows a deeper strong-branch budget is the one *sound*
    // (selection-only) lever that trims nodes there. But that is NOT a safe global
    // default: strong-branch probes re-solve the node LP, which is cheap on a
    // few-row knapsack and expensive on a many-row set-covering LP. This sweep
    // makes the trap reproducible — deepening the budget leaves covering wall flat
    // or *worse* even as it helps knapsack, so any change to `sb_node_budget` must
    // be conditioned on problem shape (few rows), never applied globally.
    println!("\n## Cross-regime strong-branch budget sweep (sound: selection-only)");
    println!("# knapsack probes are cheap; covering probes are expensive. A budget");
    println!("# that helps the former can regress the latter — so it can't go global.");
    println!(
        "{:<16} {:>8} {:>8} {:>10} {:>9} {:>8}",
        "instance", "sb_bud", "status", "obj", "nodes", "wall"
    );
    let budgets = [48usize, 2_000, 8_000];
    for &(n, k) in &[(50usize, 8usize)] {
        let (a, b, c, l, u, ns, ints) = gen_mdknapsack(n, k, 7);
        let nn = n + k;
        for &bud in &budgets {
            let mut o = baseline(ns, ints.clone(), tl);
            o.sb_node_budget = bud;
            let r = solve(&a, &b, &c, &l, &u, k, nn, &o);
            println!(
                "{:<16} {:>8} {:>8} {:>10.1} {:>9} {:>7.2}s",
                format!("mdk{n}x{k}"),
                bud,
                st_name(r.status),
                r.obj,
                r.nodes,
                r.wall
            );
        }
    }
    for &(nc, nr) in &[(1000usize, 500usize), (2000, 800)] {
        let (a, b, c, l, u, ns, ints) = gen_setcover(nc, nr, 3, 6);
        let nn = nc + nr;
        for &bud in &budgets {
            let mut o = baseline(ns, ints.clone(), tl);
            o.sb_node_budget = bud;
            let r = solve(&a, &b, &c, &l, &u, nr, nn, &o);
            println!(
                "{:<16} {:>8} {:>8} {:>10.1} {:>9} {:>7.2}s",
                format!("sc{nc}x{nr}"),
                bud,
                st_name(r.status),
                r.obj,
                r.nodes,
                r.wall
            );
        }
    }
}
