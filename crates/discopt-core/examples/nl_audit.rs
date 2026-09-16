//! Differential audit probe for issue #1213: what does discopt's `.nl` parser
//! accept, and what does it refuse, over a directory of `.nl` files?
//!
//! Emits one JSON object per file on stdout so the result can be joined against
//! the pounce-nl side of the same corpus (`scratchpad/issue1213/pounce_side.py`).
//! Prints the number of files actually attempted and exits non-zero if that is
//! zero, so a probe that traverses nothing cannot read as a pass (CLAUDE.md §6).
//!
//! Usage: cargo run -p discopt-core --release --example nl_audit -- <dir> [<dir>...]

use discopt_core::nl_parser::{parse_nl_file_full, NlParseError};
use std::path::Path;

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', " ")
}

fn kind(e: &NlParseError) -> &'static str {
    match e {
        NlParseError::UnexpectedEof => "UnexpectedEof",
        NlParseError::InvalidHeader(_) => "InvalidHeader",
        NlParseError::UnknownOpcode(_) => "UnknownOpcode",
        NlParseError::UnsupportedOpcode { .. } => "UnsupportedOpcode",
        NlParseError::BinaryUnsupported => "BinaryUnsupported",
        NlParseError::Parse(_) => "Parse",
    }
}

/// `--eval <file.nl> <x.txt>`: print `objective(x)` plus the variable bounds,
/// so the same point can be evaluated by pounce-nl and the two compared.
fn eval_mode(nl: &str, xfile: &str) {
    let model = parse_nl_file_full(nl).expect("parse").model;
    let x: Vec<f64> = std::fs::read_to_string(xfile)
        .expect("x file")
        .split_whitespace()
        .map(|t| t.parse::<f64>().expect("x value"))
        .collect();
    assert_eq!(x.len(), model.n_vars, "x length != n_vars for {nl}");
    // JSON has no infinity; clamp to the 1e20 sentinel both sides agree on.
    let j = |v: f64| -> String {
        if v.is_nan() {
            "null".to_string()
        } else if v <= -1e20 {
            "-1e20".to_string()
        } else if v >= 1e20 {
            "1e20".to_string()
        } else {
            format!("{v:?}")
        }
    };
    let lo: Vec<String> = model.variables.iter().map(|v| j(v.lb[0])).collect();
    let hi: Vec<String> = model.variables.iter().map(|v| j(v.ub[0])).collect();
    println!(
        "{{\"obj\":{},\"minimize\":{},\"lo\":[{}],\"hi\":[{}]}}",
        j(model.evaluate_objective(&x)),
        matches!(
            model.objective_sense,
            discopt_core::expr::ObjectiveSense::Minimize
        ),
        lo.join(","),
        hi.join(",")
    );
}

fn main() {
    let dirs: Vec<String> = std::env::args().skip(1).collect();
    if dirs.first().map(|s| s.as_str()) == Some("--eval") {
        assert_eq!(dirs.len(), 3, "--eval needs <file.nl> <x.txt>");
        eval_mode(&dirs[1], &dirs[2]);
        return;
    }
    if dirs.is_empty() {
        eprintln!("usage: nl_audit <dir> [<dir>...]");
        std::process::exit(2);
    }
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    for d in &dirs {
        let p = Path::new(d);
        if p.is_file() {
            files.push(p.to_path_buf());
            continue;
        }
        let rd = std::fs::read_dir(p).unwrap_or_else(|e| panic!("read_dir {d}: {e}"));
        for ent in rd {
            let ent = ent.expect("dir entry");
            let path = ent.path();
            if path.extension().and_then(|s| s.to_str()) == Some("nl") {
                files.push(path);
            }
        }
    }
    files.sort();

    let mut attempted = 0usize;
    for f in &files {
        attempted += 1;
        let name = f.file_stem().unwrap().to_string_lossy().to_string();
        match parse_nl_file_full(f.to_str().unwrap()) {
            Ok(p) => {
                let m = &p.model;
                println!(
                    "{{\"name\":\"{}\",\"ok\":true,\"n_vars\":{},\"n_cons\":{},\
                     \"n_nodes\":{},\"n_compl\":{},\"n_x0\":{}}}",
                    esc(&name),
                    m.n_vars,
                    m.constraints.len(),
                    m.arena.len(),
                    p.complementarities.len(),
                    p.initial_point.len()
                );
            }
            Err(e) => {
                println!(
                    "{{\"name\":\"{}\",\"ok\":false,\"err_kind\":\"{}\",\"err\":\"{}\"}}",
                    esc(&name),
                    kind(&e),
                    esc(&e.to_string())
                );
            }
        }
    }
    eprintln!("nl_audit: attempted {attempted} file(s)");
    if attempted == 0 {
        eprintln!("PROBE DID NOT FIRE: zero .nl files attempted");
        std::process::exit(1);
    }
}
