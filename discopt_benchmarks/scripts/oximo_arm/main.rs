//! oximo arm of the cross-tool modelling benchmark (#1215).
//!
//! Four model families at three sizes, each timed from an empty model to `.nl`
//! text, so discopt and Pyomo can be compared against it on identical
//! mathematics. Emits TSV: `tool family rows vars construct_s write_s`.
//!
//! Per-element is oximo's idiom -- it has no array-valued constraint body -- so
//! that is what is measured here, and it is the arm discopt's per-element
//! numbers are comparable to.

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let families: Vec<(&str, fn(usize) -> Model)> = vec![
        ("linear", linear),
        ("sep_nl", sep_nl),
        ("coupled_nl", coupled_nl),
        ("minlp", minlp),
    ];
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
