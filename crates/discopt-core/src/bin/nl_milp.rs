//! `.nl` → MILP certificate with **no Python in the process** (#1183 / #1212 Phase 1).
//!
//! Reads an AMPL `.nl` file with the in-tree Rust parser, builds the working form
//! `A x = b` (one slack/surplus column per inequality row) directly from
//! `ModelRepr`, and hands it to the same `solve_milp_csc` driver the Python path
//! calls. Nothing here is a new relaxation or a new search: it is the *existing*
//! engine entered without the interpreter, so the wall-clock it reports is the
//! core's own, attributable without the modeling layer.
//!
//! Refuses loudly rather than approximating: a nonlinear row, a nonlinear
//! objective, a maximize sense, or any constraint the linear extractor cannot
//! read is an error exit, never a silently-dropped term — dropping one would
//! certify an optimum for a different model.
//!
//! ```text
//! cargo run --release -p discopt-core --bin nl_milp -- model.nl [--root-cuts N]
//!     [--cut-rounds N] [--no-cut-select] [--time-limit S] [--gap G] [--max-nodes N]
//! ```

use std::env;
use std::process::ExitCode;
use std::time::Instant;

use discopt_core::bnb::milp_driver::{solve_milp_csc, MilpOptions};
use discopt_core::expr::{ConstraintSense, ObjectiveSense, VarType};
use discopt_core::lp::simplex::{SimplexOptions, SparseCols};
use discopt_core::nl_parser::parse_nl_file;
use discopt_core::presolve::extract_linear_rows;
use discopt_core::presolve::obbt::extract_linear_coeffs;

/// The LP layer's unboundedness sentinel. NOT `f64::INFINITY` (CLAUDE.md).
const INF: f64 = 1e20;

fn clamp_bound(v: f64) -> f64 {
    if !v.is_finite() {
        if v > 0.0 {
            INF
        } else {
            -INF
        }
    } else {
        v.clamp(-INF, INF)
    }
}

/// SplitMix64. A permutation seed must produce the SAME relabeling on every
/// run and every machine, so this is written out rather than pulled from a
/// crate whose stream could change across versions -- a drifting relabeling
/// would silently unpair the arms of a paired comparison.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Fisher-Yates over `0..n`, driven by `seed`.
fn permutation(n: usize, seed: u64) -> Vec<usize> {
    let mut p: Vec<usize> = (0..n).collect();
    let mut st = seed;
    for i in (1..n).rev() {
        let j = (splitmix64(&mut st) % (i as u64 + 1)) as usize;
        p.swap(i, j);
    }
    p
}

struct Args {
    path: String,
    root_cuts: usize,
    cut_rounds: usize,
    cut_select: bool,
    time_limit: f64,
    gap_tol: f64,
    max_nodes: usize,
    sb_node_budget: usize,
    sb_max_cands: usize,
    perm_seed: u64,
    /// Which labels the seed permutes: "rows", "cols", or "both".
    perm_mode: String,
}

fn parse_args() -> Result<Args, String> {
    let mut it = env::args().skip(1);
    let path = it.next().ok_or("usage: nl_milp <model.nl> [flags]")?;
    // Defaults mirror the SHIPPED Python configuration: `_milp_root_cut_budget`
    // (root_cuts=500, cut_rounds=50, cut_select=true, root_cut_time_s = half the
    // engine budget) plus the binding's own defaults for everything else. A probe
    // whose options differ from the shipped ones measures a solver nobody runs.
    let mut a = Args {
        path,
        root_cuts: 500,
        cut_rounds: 50,
        cut_select: true,
        time_limit: 120.0,
        gap_tol: 1e-4,
        max_nodes: 1_000_000,
        sb_node_budget: 48,
        sb_max_cands: 6,
        perm_seed: 0,
        perm_mode: "both".to_string(),
    };
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--root-cuts" => a.root_cuts = val()?.parse().map_err(|e| format!("{e}"))?,
            "--cut-rounds" => a.cut_rounds = val()?.parse().map_err(|e| format!("{e}"))?,
            "--no-cut-select" => a.cut_select = false,
            "--time-limit" => a.time_limit = val()?.parse().map_err(|e| format!("{e}"))?,
            "--gap" => a.gap_tol = val()?.parse().map_err(|e| format!("{e}"))?,
            "--max-nodes" => a.max_nodes = val()?.parse().map_err(|e| format!("{e}"))?,
            "--sb-node-budget" => a.sb_node_budget = val()?.parse().map_err(|e| format!("{e}"))?,
            "--sb-max-cands" => a.sb_max_cands = val()?.parse().map_err(|e| format!("{e}"))?,
            "--perm-seed" => a.perm_seed = val()?.parse().map_err(|e| format!("{e}"))?,
            "--perm-mode" => {
                a.perm_mode = val()?;
                if !matches!(a.perm_mode.as_str(), "rows" | "cols" | "both") {
                    return Err(format!(
                        "--perm-mode must be rows|cols|both, got {}",
                        a.perm_mode
                    ));
                }
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }
    Ok(a)
}

fn run() -> Result<(), String> {
    let args = parse_args()?;

    let t0 = Instant::now();
    let model = parse_nl_file(&args.path).map_err(|e| format!("parse failed: {e:?}"))?;
    let parse_s = t0.elapsed().as_secs_f64();

    if model.objective_sense != ObjectiveSense::Minimize {
        return Err(
            "maximize is not handled by this probe; refusing rather than \
                    flipping signs untested"
                .into(),
        );
    }

    // ---- columns: structural bounds and integrality straight off `ModelRepr` ----
    let n_struct = model.n_vars;
    let mut l = vec![0.0f64; n_struct];
    let mut u = vec![0.0f64; n_struct];
    let mut integer_cols: Vec<usize> = Vec::new();
    let mut n_placed = 0usize;
    for v in &model.variables {
        if v.lb.len() != v.size || v.ub.len() != v.size {
            return Err(format!(
                "variable block {} has {} elements but {}/{} bounds",
                v.name,
                v.size,
                v.lb.len(),
                v.ub.len()
            ));
        }
        for k in 0..v.size {
            let j = v.offset + k;
            l[j] = clamp_bound(v.lb[k]);
            u[j] = clamp_bound(v.ub[k]);
            if matches!(v.var_type, VarType::Binary | VarType::Integer) {
                integer_cols.push(j);
            }
            n_placed += 1;
        }
    }
    if n_placed != n_struct {
        return Err(format!(
            "variable blocks cover {n_placed} columns but n_vars = {n_struct}"
        ));
    }

    // ---- objective ----
    let (obj_terms, obj_const) = extract_linear_coeffs(&model.arena, model.objective)
        .ok_or("objective is not linear — refusing (a dropped term certifies the wrong model)")?;
    let mut c = vec![0.0f64; n_struct];
    for (j, coeff) in obj_terms {
        if j >= n_struct {
            return Err(format!("objective references column {j} >= {n_struct}"));
        }
        c[j] += coeff;
    }

    // ---- rows ----
    let mut rows = extract_linear_rows(&model);
    if rows.len() != model.constraints.len() {
        return Err(format!(
            "{} of {} constraints are not linear — refusing (this probe solves MILPs only)",
            model.constraints.len() - rows.len(),
            model.constraints.len()
        ));
    }
    // ---- optional relabeling (#1183 measurement harness) ----
    //
    // A pure relabeling -- permuting row order and structural column order --
    // leaves the feasible set, the optimum and every bound identical, so any
    // change in node count across seeds is the solver's own order-sensitivity
    // and nothing else. Measured spread on this instance is 2.96x, which is the
    // noise floor below which no single-instance ratio is believable; the point
    // of the flag is to let an A/B be PAIRED over the same seeds.
    //
    // Applied here, before the working form is built, so the slack columns are
    // generated in the permuted row order and stay consistent for free.
    if args.perm_seed != 0 {
        let ns = n_struct;
        // Identity for the half this mode does not relabel, so "rows" and
        // "cols" are the SAME perturbation family the numpy harness uses and
        // the two harnesses can be compared arm for arm.
        let rp = if args.perm_mode == "cols" {
            (0..rows.len()).collect::<Vec<_>>()
        } else {
            permutation(rows.len(), args.perm_seed)
        };
        let cp = if args.perm_mode == "rows" {
            (0..ns).collect::<Vec<_>>()
        } else {
            permutation(ns, args.perm_seed ^ 0xA5A5_A5A5_A5A5_A5A5)
        };

        // `rp[i]` is the row that moves INTO position i; `cp[j]` likewise for
        // structural columns. `inv_c[old] = new` is what the row index lists
        // and the integer-column list must be mapped through.
        let mut inv_c = vec![0usize; ns];
        for (new, &old) in cp.iter().enumerate() {
            inv_c[old] = new;
        }

        let old_rows = std::mem::take(&mut rows);
        let mut slot: Vec<Option<_>> = old_rows.into_iter().map(Some).collect();
        for &old in &rp {
            let mut r = slot[old].take().ok_or("permutation visited a row twice")?;
            for j in r.var_indices.iter_mut() {
                *j = inv_c[*j];
            }
            rows.push(r);
        }
        if slot.iter().any(|x| x.is_some()) {
            return Err("permutation did not cover every row".into());
        }

        let permute = |v: &[f64]| -> Vec<f64> { cp.iter().map(|&old| v[old]).collect() };
        c = permute(&c);
        l = permute(&l);
        u = permute(&u);
        for j in integer_cols.iter_mut() {
            *j = inv_c[*j];
        }
        integer_cols.sort_unstable();
        println!("perm_seed={} perm_mode={}", args.perm_seed, args.perm_mode);
    }

    let m = rows.len();
    let n_slack = rows
        .iter()
        .filter(|r| r.sense != ConstraintSense::Eq)
        .count();
    let n = n_struct + n_slack;

    // Working form: `A x = b`, one slack (`<=`, +1) or surplus (`>=`, -1) column
    // per inequality row, each in `[0, INF)`. Equalities carry none.
    let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
    let mut b = vec![0.0f64; m];
    c.resize(n, 0.0);
    l.resize(n, 0.0);
    u.resize(n, INF);
    let mut next_slack = n_struct;
    let mut nnz_struct = 0usize;
    for (i, row) in rows.iter().enumerate() {
        if row.var_indices.len() != row.coeffs.len() {
            return Err(format!("row {i} has mismatched index/coefficient lengths"));
        }
        for (&j, &v) in row.var_indices.iter().zip(&row.coeffs) {
            if j >= n_struct {
                return Err(format!("row {i} references column {j} >= {n_struct}"));
            }
            triplets.push((i, j, v));
            nnz_struct += 1;
        }
        b[i] = row.rhs;
        match row.sense {
            ConstraintSense::Le => {
                triplets.push((i, next_slack, 1.0));
                next_slack += 1;
            }
            ConstraintSense::Ge => {
                triplets.push((i, next_slack, -1.0));
                next_slack += 1;
            }
            ConstraintSense::Eq => {}
        }
    }
    if next_slack != n {
        return Err(format!(
            "placed {} slacks, expected {n_slack}",
            next_slack - n_struct
        ));
    }

    // triplets → CSC
    let mut counts = vec![0usize; n];
    for &(_, j, _) in &triplets {
        counts[j] += 1;
    }
    let mut col_ptr = vec![0usize; n + 1];
    for j in 0..n {
        col_ptr[j + 1] = col_ptr[j] + counts[j];
    }
    let nnz = triplets.len();
    let mut row_idx = vec![0usize; nnz];
    let mut vals = vec![0.0f64; nnz];
    let mut fill = col_ptr.clone();
    // Sort by row inside each column: the simplex reads columns in row order.
    triplets.sort_unstable_by_key(|&(i, j, _)| (j, i));
    for &(i, j, v) in &triplets {
        row_idx[fill[j]] = i;
        vals[fill[j]] = v;
        fill[j] += 1;
    }
    let csc = SparseCols::from_csc(col_ptr, row_idx, vals);

    let opts = MilpOptions {
        n_struct,
        integer_cols: integer_cols.clone(),
        max_nodes: args.max_nodes,
        time_limit_s: Some(args.time_limit),
        gap_tol: args.gap_tol,
        // This CLI exposes only the relative tolerance, so there is no
        // absolute criterion to apply (#1315).
        abs_gap_tol: None,
        root_cuts: args.root_cuts,
        cut_rounds: args.cut_rounds,
        gmi_cuts: true,
        cut_select: args.cut_select,
        node_cuts: false,
        max_pool_cuts: 128,
        heuristics: true,
        presolve: true,
        strong_branch: true,
        node_propagation: true,
        reduced_cost_fixing: true,
        sb_max_cands: args.sb_max_cands,
        sb_node_budget: args.sb_node_budget,
        initial_incumbent: None,
        node_hook_rounds: 0,
        node_hook_cut_cap: 0,
        root_cut_time_s: Some(args.time_limit * 0.5),
        root_cut_prune: true,
        simplex: SimplexOptions {
            tol: 1e-9,
            max_iter: 100_000,
            deadline: None,
            warm_stall_guard: true,
            warm_stall_cap_override: None,
            expel_zero_artificials: false,
            bank_deadline_duals: false,
            recover_unstable_pivot: false,
            dual_stall_patience: SimplexOptions::default().dual_stall_patience,
            dual_cost_perturb: SimplexOptions::default().dual_cost_perturb,
            cold_dual_start: false,
        },
    };

    println!(
        "file={} rows={m} struct_cols={n_struct} slack_cols={n_slack} int_cols={} nnz={nnz} \
         (structural {nnz_struct})",
        args.path,
        integer_cols.len()
    );
    println!(
        "opts: root_cuts={} cut_rounds={} cut_select={} gap_tol={} time_limit_s={}",
        args.root_cuts, args.cut_rounds, args.cut_select, args.gap_tol, args.time_limit
    );

    let t1 = Instant::now();
    let res = solve_milp_csc(&csc, m, n, &c, &l, &u, &b, obj_const, &opts);
    let solve_s = t1.elapsed().as_secs_f64();

    println!(
        "status={:?} objective={:.10} bound={:.10} nodes={} lp_iters={}",
        res.status, res.obj, res.bound, res.nodes, res.lp_iters
    );
    println!(
        "parse_s={parse_s:.4} solve_s={solve_s:.4} total_s={:.4}",
        parse_s + solve_s
    );

    // Anti-vacuity (CLAUDE.md §6): a probe that read no rows, branched on no
    // columns, or never entered the tree must not read as a solve.
    if m == 0 || integer_cols.is_empty() {
        return Err(format!(
            "vacuous probe: rows={m}, integer columns={}",
            integer_cols.len()
        ));
    }
    println!(
        "EXTRACTED_ROWS={m} INTEGER_COLS={} NODES={}",
        integer_cols.len(),
        res.nodes
    );
    // Counter dump under DISCOPT_PROFILE, so an experiment run through this
    // binary can tell "the arm is neutral" from "the arm never fired"
    // (CLAUDE.md §6). Without it every counter in the engine is invisible here.
    discopt_core::profile::dump();
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("nl_milp: {msg}");
            ExitCode::from(2)
        }
    }
}
