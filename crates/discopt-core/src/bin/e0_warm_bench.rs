//! E0 (scip-parity-kernel-plan): warm dual-simplex node-rate ceiling.
//!
//! Loads a standard-form LP exported by
//! `discopt_benchmarks/scripts/e0_export_lp.py` (E0LPBIN1 format) and measures
//! the pure-Rust warm re-solve rate under branching-shaped bound changes:
//!
//!   * breadth: every child = parent box with ONE integer column's bound
//!     tightened (down: `u_j = floor(x_j)`, up: `l_j = ceil(x_j)`), warm-started
//!     from the ROOT basis (the best-first pattern);
//!   * dive: fix the first fractional integer to its rounding, warm-start from
//!     the PARENT solve's basis, repeat until integral/infeasible, reset (the
//!     DFS pattern).
//!
//! Reports solves/s and per-solve latency percentiles — the kernel node-rate
//! ceiling of the in-house simplex on the REAL node LPs. Kill criterion
//! (plan §3 E0): < 500 warm re-solves/s on the rsyn-class LP.

use std::env;
use std::fs;
use std::process::ExitCode;
use std::time::Instant;

use discopt_core::lp::crossover::LpView;
use discopt_core::lp::simplex::{
    solve_lp, solve_lp_warm, LpStatus, PreparedDual, SimplexOptions, SparseCols,
};

const MAGIC: &[u8; 8] = b"E0LPBIN1";

struct E0Lp {
    m: usize,
    n: usize,
    c: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    l: Vec<f64>,
    u: Vec<f64>,
    cand: Vec<usize>,
}

fn read_f64s(buf: &[u8], off: &mut usize, k: usize) -> Vec<f64> {
    let mut v = Vec::with_capacity(k);
    for i in 0..k {
        let s = *off + 8 * i;
        v.push(f64::from_le_bytes(buf[s..s + 8].try_into().unwrap()));
    }
    *off += 8 * k;
    v
}

fn read_u64(buf: &[u8], off: &mut usize) -> u64 {
    let v = u64::from_le_bytes(buf[*off..*off + 8].try_into().unwrap());
    *off += 8;
    v
}

fn load(path: &str) -> E0Lp {
    let buf = fs::read(path).expect("read E0 LP file");
    assert_eq!(&buf[..8], MAGIC, "bad magic");
    let mut off = 8usize;
    let m = read_u64(&buf, &mut off) as usize;
    let n = read_u64(&buf, &mut off) as usize;
    let c = read_f64s(&buf, &mut off, n);
    let a = read_f64s(&buf, &mut off, m * n);
    let b = read_f64s(&buf, &mut off, m);
    let l = read_f64s(&buf, &mut off, n);
    let u = read_f64s(&buf, &mut off, n);
    let n_cand = read_u64(&buf, &mut off) as usize;
    let mut cand = Vec::with_capacity(n_cand);
    for _ in 0..n_cand {
        cand.push(read_u64(&buf, &mut off) as usize);
    }
    E0Lp {
        m,
        n,
        c,
        a,
        b,
        l,
        u,
        cand,
    }
}

fn frac_cands(x: &[f64], cand: &[usize]) -> Vec<usize> {
    let f: Vec<usize> = cand
        .iter()
        .copied()
        .filter(|&j| {
            let fr = x[j] - x[j].floor();
            fr > 1e-6 && fr < 1.0 - 1e-6
        })
        .collect();
    if f.is_empty() {
        cand.to_vec()
    } else {
        f
    }
}

fn pct(sorted_us: &[f64], p: f64) -> f64 {
    if sorted_us.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted_us.len() as f64 - 1.0) * p).round() as usize;
    sorted_us[idx]
}

fn report(tag: &str, times_us: &mut [f64], iters: &[usize], statuses: &[LpStatus]) {
    times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let total_s: f64 = times_us.iter().sum::<f64>() / 1e6;
    let nsolve = times_us.len();
    let mut opt = 0usize;
    let mut infeas = 0usize;
    let mut other = 0usize;
    for s in statuses {
        match s {
            LpStatus::Optimal => opt += 1,
            LpStatus::Infeasible => infeas += 1,
            _ => other += 1,
        }
    }
    let mean_it = if iters.is_empty() {
        0.0
    } else {
        iters.iter().sum::<usize>() as f64 / iters.len() as f64
    };
    println!(
        "  {tag}: {nsolve} solves in {total_s:.3}s -> {:.0}/s | us/solve p50={:.0} p90={:.0} p99={:.0} mean={:.0} | iters mean={mean_it:.1} | status opt={opt} infeas={infeas} other={other}",
        nsolve as f64 / total_s,
        pct(times_us, 0.5),
        pct(times_us, 0.9),
        pct(times_us, 0.99),
        times_us.iter().sum::<f64>() / nsolve as f64,
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: e0_warm_bench <file.e0lp> [trials=2000] [dive_depth=25]");
        return ExitCode::FAILURE;
    }
    let trials: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(2000);
    let dive_depth: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(25);
    let lp_data = load(&args[1]);
    println!(
        "{}: m={} n={} cand={}",
        args[1],
        lp_data.m,
        lp_data.n,
        lp_data.cand.len()
    );

    let opts = SimplexOptions {
        tol: 1e-7,
        max_iter: 100_000,
        deadline: None,
        warm_stall_guard: true,
        warm_stall_cap_override: None,
        expel_zero_artificials: true,
    };

    let mut l = lp_data.l.clone();
    let mut u = lp_data.u.clone();

    // cold root solve
    let t0 = Instant::now();
    let root = {
        let view = LpView {
            a: &lp_data.a,
            m: lp_data.m,
            n: lp_data.n,
            c: &lp_data.c,
            l: &l,
            u: &u,
        };
        solve_lp(&view, &lp_data.b, &opts)
    };
    let cold_us = t0.elapsed().as_secs_f64() * 1e6;
    println!(
        "  cold: status={:?} obj={:.6} iters={} in {:.0} us",
        root.status, root.obj, root.iters, cold_us
    );
    if !matches!(root.status, LpStatus::Optimal) {
        eprintln!("root LP not optimal — aborting");
        return ExitCode::FAILURE;
    }
    let root_basis = root.basis.clone();
    let fracs = frac_cands(&root.x, &lp_data.cand);
    println!("  fractional branch candidates at root: {}", fracs.len());

    // ── breadth: one bound flip from the root basis per child ──
    let mut times = Vec::with_capacity(trials);
    let mut iters = Vec::with_capacity(trials);
    let mut statuses = Vec::with_capacity(trials);
    for t in 0..trials {
        let j = fracs[t % fracs.len()];
        let (kl, ku) = (l[j], u[j]);
        if t % 2 == 0 {
            u[j] = root.x[j].floor();
        } else {
            l[j] = root.x[j].ceil();
        }
        let t1 = Instant::now();
        let sol = {
            let view = LpView {
                a: &lp_data.a,
                m: lp_data.m,
                n: lp_data.n,
                c: &lp_data.c,
                l: &l,
                u: &u,
            };
            solve_lp_warm(&view, &lp_data.b, &root_basis, &opts)
        };
        times.push(t1.elapsed().as_secs_f64() * 1e6);
        iters.push(sol.iters);
        statuses.push(sol.status);
        l[j] = kl;
        u[j] = ku;
    }
    report(
        "breadth (flip from root basis)",
        &mut times,
        &iters,
        &statuses,
    );

    // ── dive: chained fixings, basis carried from the parent ──
    let mut times = Vec::with_capacity(trials);
    let mut iters = Vec::with_capacity(trials);
    let mut statuses = Vec::with_capacity(trials);
    let mut resets = 0usize;
    let mut leaves = 0usize;
    while times.len() < trials {
        // reset to root box
        l.copy_from_slice(&lp_data.l);
        u.copy_from_slice(&lp_data.u);
        let mut basis = root_basis.clone();
        let mut x = root.x.clone();
        resets += 1;
        for _ in 0..dive_depth {
            if times.len() >= trials {
                break;
            }
            let fr = frac_cands(&x, &lp_data.cand);
            let j = *match fr.iter().find(|&&j| {
                let f = x[j] - x[j].floor();
                f > 1e-6 && f < 1.0 - 1e-6
            }) {
                Some(j) => j,
                None => {
                    leaves += 1;
                    break;
                }
            };
            let v = x[j].round().clamp(lp_data.l[j], lp_data.u[j]);
            l[j] = v;
            u[j] = v;
            let t1 = Instant::now();
            let sol = {
                let view = LpView {
                    a: &lp_data.a,
                    m: lp_data.m,
                    n: lp_data.n,
                    c: &lp_data.c,
                    l: &l,
                    u: &u,
                };
                solve_lp_warm(&view, &lp_data.b, &basis, &opts)
            };
            times.push(t1.elapsed().as_secs_f64() * 1e6);
            iters.push(sol.iters);
            let st = sol.status;
            statuses.push(st);
            if !matches!(st, LpStatus::Optimal) {
                break;
            }
            basis = sol.basis;
            x = sol.x;
        }
    }
    report(
        "dive (chained fixings, parent basis)",
        &mut times,
        &iters,
        &statuses,
    );
    println!("  dive resets={resets} integral_leaves={leaves}");

    // ── prepared: scale once, factorize the root basis ONCE, reoptimize per
    // flip — the kernel pattern (production: solve_lp_warm scales then
    // PreparedDual; here the scaling + CSC + LU are all amortized).
    l.copy_from_slice(&lp_data.l);
    u.copy_from_slice(&lp_data.u);
    let base_view = LpView {
        a: &lp_data.a,
        m: lp_data.m,
        n: lp_data.n,
        c: &lp_data.c,
        l: &l,
        u: &u,
    };
    use discopt_core::lp::simplex::scaling::ScaledLp;
    let scaled = ScaledLp::maybe_new(&base_view, &lp_data.b);
    let (s_view, s_b): (LpView<'_>, &[f64]) = match &scaled {
        Some(s) => (s.view(), s.b()),
        None => (base_view, &lp_data.b),
    };
    // fresh cold solve in the (scaled) space so the basis matches this exact view
    let mut s_root = solve_lp(&s_view, s_b, &opts);
    println!(
        "  prepared-space cold: status={:?} obj={:.6}",
        s_root.status, s_root.obj
    );
    // The cold primal path can return deficient/inconsistent basis bookkeeping
    // (measured: rsyn0805m 534/537 basics + 2 mislabeled nonbasics, reduced-cost
    // violation 2.6 — a discopt-core P1 hardening item). Route one solve through
    // the warm entry (its internal primal-warm fallback re-derives a consistent
    // basis) and prefer that basis when it is full-rank.
    {
        let re = solve_lp_warm(&s_view, s_b, &s_root.basis, &opts);
        if matches!(re.status, LpStatus::Optimal) && re.basis.basic_vars.len() == s_view.m {
            println!(
                "  prepared: adopting basis from warm-path fallback (obj={:.6})",
                re.obj
            );
            s_root = re;
        }
    }
    // The cold primal path can return a DEFICIENT basis (internal presolve
    // drops redundant rows; measured on rsyn0805m: 534 basic vars for m=537),
    // which PreparedDual (and the production warm path!) rejects. Repair by
    // crossing the optimal point over to a vertex and recovering a full basis.
    if s_root.basis.basic_vars.len() != s_view.m {
        println!(
            "  prepared: deficient cold basis ({} of {} rows) — completing with slacks",
            s_root.basis.basic_vars.len(),
            s_view.m
        );
        // Textbook completion: row-reduce the deficient basis matrix to find
        // the pivotless rows, then add each such row's slack column (the
        // single-entry +1 column) as basic. Bench-only dense elimination.
        let m = s_view.m;
        let n = s_view.n;
        let bv0 = s_root.basis.basic_vars.clone();
        let k = bv0.len();
        let mut mat = vec![0.0f64; m * k];
        for (slot, &j) in bv0.iter().enumerate() {
            for r in 0..m {
                mat[r * k + slot] = s_view.a[r * n + j];
            }
        }
        let mut pivot_row_used = vec![false; m];
        let mut col = 0usize;
        for _ in 0..k {
            if col >= k {
                break;
            }
            // find best pivot among unused rows for this column
            let mut best_r = usize::MAX;
            let mut best_v = 1e-10;
            for r in 0..m {
                if !pivot_row_used[r] && mat[r * k + col].abs() > best_v {
                    best_v = mat[r * k + col].abs();
                    best_r = r;
                }
            }
            if best_r != usize::MAX {
                pivot_row_used[best_r] = true;
                let d = mat[best_r * k + col];
                for r in 0..m {
                    if r == best_r {
                        continue;
                    }
                    let f = mat[r * k + col] / d;
                    if f != 0.0 {
                        for c2 in col..k {
                            mat[r * k + c2] -= f * mat[best_r * k + c2];
                        }
                    }
                }
            }
            col += 1;
        }
        let missing: Vec<usize> = (0..m).filter(|&r| !pivot_row_used[r]).collect();
        println!(
            "  prepared: pivotless rows: {:?}",
            &missing[..missing.len().min(8)]
        );
        // find each missing row's slack column: single nonzero (+1) at that row
        let mut basis = s_root.basis.clone();
        let is_basic: Vec<bool> = {
            let mut v = vec![false; n];
            for &j in &basis.basic_vars {
                v[j] = true;
            }
            v
        };
        for &r in &missing {
            let mut found = None;
            for j in (0..n).rev() {
                if is_basic[j] {
                    continue;
                }
                // single-entry column with support exactly {r}?
                let vr = s_view.a[r * n + j];
                if vr == 0.0 {
                    continue;
                }
                let mut single = true;
                for r2 in 0..m {
                    if r2 != r && s_view.a[r2 * n + j] != 0.0 {
                        single = false;
                        break;
                    }
                }
                if single {
                    found = Some(j);
                    break;
                }
            }
            match found {
                Some(j) => {
                    basis.basic_vars.push(j);
                    basis.col_status[j] = 1; // BASIC (basis.rs code)
                }
                None => println!("  prepared: no slack column for pivotless row {r} (eq row?)"),
            }
        }
        if basis.basic_vars.len() == m {
            s_root.basis = basis;
        }
    }
    let sp = SparseCols::from_dense(s_view.a, s_view.m, s_view.n);
    let t_prep = Instant::now();
    let prepared = PreparedDual::prepare(&s_view, &s_root.basis, &opts, &sp);
    let prep_us = t_prep.elapsed().as_secs_f64() * 1e6;
    match prepared {
        None => {
            println!("  prepared: root basis not usable (prepare returned None)");
            // (basis-deficiency diagnostics removed — P1.0 fixed; this arm
            // should no longer be reached on full-rank LPs.)
        }
        Some(pd) => {
            println!("  prepared: LU factorize+verify in {prep_us:.0} us");
            let mut l2 = s_view.l.to_vec();
            let mut u2 = s_view.u.to_vec();
            let base_l = l2.clone();
            let base_u = u2.clone();
            let mut times = Vec::with_capacity(trials);
            let mut iters = Vec::with_capacity(trials);
            let mut statuses = Vec::with_capacity(trials);
            let mut obj_moved = 0usize;
            // scaled-space fractional flips: use the scaled root x
            let s_fracs = frac_cands(&root.x, &lp_data.cand); // original-space fractionality
            for t in 0..trials {
                let j = s_fracs[t % s_fracs.len()];
                let (kl, ku) = (l2[j], u2[j]);
                // flip in the SCALED space: proportional tightening of the same
                // column (x'_j = x_j / s_j; floor/ceil applied in x-space then
                // scaled). Use the ratio of the scaled bound interval.
                if t % 2 == 0 {
                    // down-child: pull u to the box fraction where floor(x_j) sits
                    let span = lp_data.u[j] - lp_data.l[j];
                    let f = if span > 0.0 {
                        (root.x[j].floor() - lp_data.l[j]) / span
                    } else {
                        0.0
                    };
                    u2[j] = base_l[j] + f * (base_u[j] - base_l[j]);
                } else {
                    let span = lp_data.u[j] - lp_data.l[j];
                    let f = if span > 0.0 {
                        (root.x[j].ceil() - lp_data.l[j]) / span
                    } else {
                        1.0
                    };
                    l2[j] = base_l[j] + f * (base_u[j] - base_l[j]);
                }
                let t1 = Instant::now();
                let sol = pd.reoptimize(&l2, &u2, s_b, &opts);
                times.push(t1.elapsed().as_secs_f64() * 1e6);
                iters.push(sol.iters);
                if (sol.obj - s_root.obj).abs() > 1e-9 {
                    obj_moved += 1;
                }
                statuses.push(sol.status);
                l2[j] = kl;
                u2[j] = ku;
            }
            report(
                "prepared (scale+LU amortized, reoptimize per flip)",
                &mut times,
                &iters,
                &statuses,
            );
            println!(
                "  prepared: children with objective different from root: {obj_moved}/{trials}"
            );
        }
    }
    ExitCode::SUCCESS
}
