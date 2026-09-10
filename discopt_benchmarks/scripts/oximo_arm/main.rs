//! oximo arm of the cross-tool modelling benchmark (#1215).
//!
//! Four model families at three sizes, each timed from an empty model to `.nl`
//! text, so discopt and Pyomo can be compared against it on identical
//! mathematics. Emits TSV: `tool family idiom rows vars construct_s write_s`.
//!
//! Per-element is oximo's idiom -- it has no array-valued constraint body -- so
//! that is what is measured here, and it is the arm discopt's per-element
//! numbers are comparable to.
//!
//! `--memory` measures retained memory instead of wall time, emitting
//! `tool family idiom rows vars retained_b`. It re-executes itself once per
//! point (`--memory-child`) because an allocator pools what a previous model
//! freed, so a second measurement in the same process reads low. The Python
//! panel does exactly the same thing, reads the same `VmRSS` field of
//! `/proc/self/status`, and warms up on a tiny model first so that whatever the
//! first build touches is charged to the baseline rather than to the model.

use oximo::prelude::*;
use std::time::Instant;

const SIZES: [usize; 3] = [1_000, 10_000, 100_000];
const REPS: usize = 3;

fn linear(n: usize) -> Model {
    let m = Model::new("linear");
    variable!(m, x[i in 0..n] >= 0.0);
    variable!(m, y[i in 0..n] >= 0.0);
    variable!(m, z[i in 0..n] >= 0.0);
    constraint!(m, c[i in 0..n], x[i] + 2.0 * y[i] + 3.0 * z[i] <= 10.0 + (i % 7) as f64);
    objective!(m, Min, sum!(x[i] for i in 0..n));
    m
}

fn sep_nl(n: usize) -> Model {
    let m = Model::new("sep_nl");
    variable!(m, x[i in 0..n] >= 0.1);
    variable!(m, y[i in 0..n] >= 0.1);
    constraint!(m, c[i in 0..n], x[i].exp() + y[i] <= 9.0 + (i % 5) as f64);
    objective!(m, Min, sum!(x[i] for i in 0..n));
    m
}

fn coupled_nl(n: usize) -> Model {
    let m = Model::new("coupled_nl");
    variable!(m, x[i in 0..n] >= 0.1);
    variable!(m, y[i in 0..n] >= 0.1);
    constraint!(m, c[i in 0..n],
        x[i] * y[i] + (x[i] + 1.0).log() <= 9.0 + (i % 5) as f64);
    objective!(m, Min, sum!(x[i] for i in 0..n));
    m
}

fn minlp(n: usize) -> Model {
    let m = Model::new("minlp");
    variable!(m, x[i in 0..n] >= 0.0);
    variable!(m, y[i in 0..n] >= 0.0);
    variable!(m, z[i in 0..n], Bin);
    constraint!(m, c[i in 0..n], x[i] * y[i] + z[i] <= 4.0 + (i % 3) as f64);
    objective!(m, Min, sum!(x[i] for i in 0..n));
    m
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn families() -> Vec<(&'static str, fn(usize) -> Model)> {
    vec![
        ("linear", linear),
        ("sep_nl", sep_nl),
        ("coupled_nl", coupled_nl),
        ("minlp", minlp),
    ]
}

/// Resident set size in bytes, from `/proc/self/status`. `VmRSS` rather than
/// `statm` because the file states its unit, so no page-size assumption is
/// shared with the Python arm.
fn rss_bytes() -> std::io::Result<u64> {
    let status = std::fs::read_to_string("/proc/self/status")?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let mut it = rest.split_whitespace();
            let value: u64 = it.next().expect("VmRSS value").parse().expect("VmRSS is numeric");
            let unit = it.next().expect("VmRSS unit");
            assert_eq!(unit, "kB", "VmRSS reported in {unit}, expected kB");
            return Ok(value * 1024);
        }
    }
    panic!("no VmRSS line in /proc/self/status");
}

/// `(rows, vars)` read out of the `.nl` this model actually writes.
fn nl_shape(m: &Model) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let text = oximo::io::to_nl_string(m)?;
    let line1: Vec<usize> = text
        .lines()
        .nth(1)
        .unwrap_or("")
        .split_whitespace()
        .take(2)
        .map(|t| t.parse().unwrap_or(0))
        .collect();
    Ok((line1[1], line1[0]))
}

/// One memory point, in this (fresh) process. Prints the finished TSV row.
fn memory_child(name: &str, n: usize) -> Result<(), Box<dyn std::error::Error>> {
    let build = families()
        .into_iter()
        .find(|(f, _)| *f == name)
        .unwrap_or_else(|| panic!("unknown family {name}"))
        .1;

    // Warm-up: a tiny model of the same shape, dropped. Whatever the first build
    // makes the allocator reserve is then already resident, and is NOT charged
    // to the measured model.
    drop(build(8));

    let base = rss_bytes()?;
    let m = build(n);
    let retained = rss_bytes()?.saturating_sub(base);
    // `m` is still alive at the reading above; the shape must come from the file
    // it really writes, so read it after, not before.
    let (rows, vars) = nl_shape(&m)?;
    assert_eq!(rows, n, "{name}/{n}: wrote {rows} rows");
    println!("oximo\t{name}\tper-element\t{rows}\t{vars}\t{retained}");
    Ok(())
}

fn memory_parent() -> Result<(), Box<dyn std::error::Error>> {
    println!("tool\tfamily\tidiom\trows\tvars\tretained_b");
    let exe = std::env::current_exe()?;
    let mut emitted = 0usize;
    for (name, _) in families() {
        for n in SIZES {
            let out = std::process::Command::new(&exe)
                .args(["--memory-child", name, &n.to_string()])
                .output()?;
            assert!(out.status.success(), "{name}/{n}: child failed: {:?}", out.status);
            print!("{}", String::from_utf8(out.stdout)?);
            emitted += 1;
        }
    }
    eprintln!("# executed: {emitted} (family, size) memory points, row counts verified");
    assert!(emitted > 0);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--memory-child") => {
            return memory_child(&args[1], args[2].parse()?);
        }
        Some("--memory") => return memory_parent(),
        Some(other) => panic!("unknown argument {other}; expected --memory or nothing"),
        None => {}
    }

    let families: Vec<(&str, fn(usize) -> Model)> = families();
    println!("tool\tfamily\tidiom\trows\tvars\tconstruct_s\twrite_s");
    let mut emitted = 0usize;
    for (name, build) in families {
        for n in SIZES {
            let (mut cons, mut writes) = (Vec::new(), Vec::new());
            let (mut rows, mut vars) = (0usize, 0usize);
            for _ in 0..REPS {
                let t0 = Instant::now();
                let m = build(n);
                cons.push(t0.elapsed().as_secs_f64());
                let t1 = Instant::now();
                let text = oximo::io::to_nl_string(&m)?;
                writes.push(t1.elapsed().as_secs_f64());
                let line1: Vec<usize> = text
                    .lines()
                    .nth(1)
                    .unwrap_or("")
                    .split_whitespace()
                    .take(2)
                    .map(|t| t.parse().unwrap_or(0))
                    .collect();
                vars = line1[0];
                rows = line1[1];
            }
            // Row count comes from the written file's own header, so an arm
            // cannot look fast by building fewer rows than it claims.
            assert_eq!(rows, n, "{name}/{n}: wrote {rows} rows");
            println!(
                "oximo\t{name}\tper-element\t{rows}\t{vars}\t{:.6}\t{:.6}",
                median(cons),
                median(writes)
            );
            emitted += 1;
        }
    }
    eprintln!("# executed: {emitted} (family, size) points, row counts verified");
    assert!(emitted > 0);
    Ok(())
}
