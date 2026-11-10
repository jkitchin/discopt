//! Replay of HiGHS `generateCut` dumps through the Rust port, bit for bit.
//!
//! Record format (one HiGHS `generateCut` call; floats are C `%a` hex literals):
//!
//! ```text
//! GC_IN <id> numcols N feastol F epsilon F onlyinit B intspos B rng U rhs F
//! inds n ...   vals n ...   upper n ...   solval n ...   isint n ...
//! GC_CORE_FAIL <id> rng U                       | core rejected the row
//! GC_CORE_OUT <id> rng U rhshi F rhslo F isup B icoef B
//! inds n ...   vals n ...
//! GC_UNTRANSFORM_FAIL                           | untransform rejected it
//! GC_PP_IN rhs F isup B icoef B
//! inds n ...   vals n ...   lb n ...   ub n ...   isint n ...
//! GC_PP_FAIL | GC_PP_OUT rhs F isup B icoef B  inds n ...  vals n ...
//! GC_END
//! ```
//!
//! `fixture_replays_bit_identically` runs the committed sample. Set
//! `DISCOPT_HIGHS_CUTGEN_REPLAY=<file or dir>` and run the `#[ignore]`d
//! `full_dump_replay` to replay a complete dump.

use std::collections::HashMap;

use super::cdouble::CDouble;
use super::core::{BaseRow, CoreCut, CutGenCore, CutGenParams};
use super::random::HighsRandom;

pub(crate) fn parse_hex_f64(t: &str) -> f64 {
    match t {
        "inf" => return f64::INFINITY,
        "-inf" => return f64::NEG_INFINITY,
        "nan" | "-nan" => return f64::NAN,
        _ => {}
    }
    let (neg, body) = match t.strip_prefix('-') {
        Some(b) => (true, b),
        None => (false, t),
    };
    let body = body
        .strip_prefix("0x")
        .unwrap_or_else(|| panic!("not a hex float: {t}"));
    let (mant, exp) = body
        .split_once('p')
        .unwrap_or_else(|| panic!("no exponent: {t}"));
    let exp: i32 = exp.parse().unwrap();
    let (ip, fp) = mant.split_once('.').unwrap_or((mant, ""));
    let mut m: u64 = 0;
    for c in ip.chars().chain(fp.chars()) {
        m = (m << 4) | c.to_digit(16).unwrap() as u64;
    }
    let v = libm::ldexp(m as f64, exp - 4 * fp.len() as i32);
    if neg {
        -v
    } else {
        v
    }
}

#[derive(Debug, Default, Clone)]
struct Record {
    id: i64,
    raw: String,
    params: Option<CutGenParams>,
    only_init: bool,
    rng_in: u64,
    row: Option<BaseRow>,
    core: Option<(u64, Option<CoreCut>)>, // (rng after, cut)
    untransform_fail: bool,
    pp_in: Option<(CoreCut, Vec<f64>, Vec<f64>, Vec<bool>)>,
    pp_out: Option<Option<CoreCut>>,
}

fn kv<'a>(tok: &[&'a str], key: &str) -> &'a str {
    let p = tok
        .iter()
        .position(|t| *t == key)
        .unwrap_or_else(|| panic!("missing {key}"));
    tok[p + 1]
}

fn array_line<'a>(line: &'a str, tag: &str) -> Vec<&'a str> {
    let mut it = line.split_ascii_whitespace();
    assert_eq!(it.next(), Some(tag), "expected {tag}: {line}");
    let n: usize = it.next().unwrap().parse().unwrap();
    let v: Vec<&str> = it.collect();
    assert_eq!(v.len(), n, "{tag} length");
    v
}
fn fl(line: &str, tag: &str) -> Vec<f64> {
    array_line(line, tag)
        .iter()
        .map(|t| parse_hex_f64(t))
        .collect()
}
fn il(line: &str, tag: &str) -> Vec<i32> {
    array_line(line, tag)
        .iter()
        .map(|t| t.parse().unwrap())
        .collect()
}
fn bl(line: &str, tag: &str) -> Vec<bool> {
    array_line(line, tag).iter().map(|t| *t == "1").collect()
}

fn parse(text: &str) -> Vec<Record> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let l = lines[i];
        if !l.starts_with("GC_IN ") {
            i += 1;
            continue;
        }
        // A record truncated by the dump limit has no GC_END; skip it.
        let end = (i..lines.len())
            .find(|&k| lines[k] == "GC_END" || (k > i && lines[k].starts_with("GC_IN ")));
        let Some(end) = end else { break };
        if lines[end] != "GC_END" {
            i = end;
            continue;
        }
        let t: Vec<&str> = l.split_ascii_whitespace().collect();
        let mut r = Record {
            id: t[1].parse().unwrap(),
            ..Default::default()
        };
        r.params = Some(CutGenParams {
            feastol: parse_hex_f64(kv(&t, "feastol")),
            epsilon: parse_hex_f64(kv(&t, "epsilon")),
            num_cols: kv(&t, "numcols").parse().unwrap(),
        });
        r.only_init = kv(&t, "onlyinit") == "1";
        r.rng_in = kv(&t, "rng").parse().unwrap();
        r.row = Some(BaseRow {
            inds: il(lines[i + 1], "inds"),
            vals: fl(lines[i + 2], "vals"),
            rhs: parse_hex_f64(kv(&t, "rhs")),
            upper: fl(lines[i + 3], "upper"),
            solval: fl(lines[i + 4], "solval"),
            col_integral: bl(lines[i + 5], "isint"),
            ints_positive: kv(&t, "intspos") == "1",
        });
        let mut k = i + 6;
        let h: Vec<&str> = lines[k].split_ascii_whitespace().collect();
        if h[0] == "GC_CORE_FAIL" {
            r.core = Some((kv(&h, "rng").parse().unwrap(), None));
        } else {
            assert_eq!(h[0], "GC_CORE_OUT");
            let cut = CoreCut {
                inds: il(lines[k + 1], "inds"),
                vals: fl(lines[k + 2], "vals"),
                rhs: CDouble::from_parts(
                    parse_hex_f64(kv(&h, "rhshi")),
                    parse_hex_f64(kv(&h, "rhslo")),
                ),
                integral_support: kv(&h, "isup") == "1",
                integral_coefficients: kv(&h, "icoef") == "1",
            };
            r.core = Some((kv(&h, "rng").parse().unwrap(), Some(cut)));
            k += 3;
            if lines[k] == "GC_UNTRANSFORM_FAIL" {
                r.untransform_fail = true;
            } else {
                let p: Vec<&str> = lines[k].split_ascii_whitespace().collect();
                assert_eq!(p[0], "GC_PP_IN");
                let cut = CoreCut {
                    inds: il(lines[k + 1], "inds"),
                    vals: fl(lines[k + 2], "vals"),
                    rhs: CDouble::new(parse_hex_f64(kv(&p, "rhs"))),
                    integral_support: kv(&p, "isup") == "1",
                    integral_coefficients: kv(&p, "icoef") == "1",
                };
                let lb = fl(lines[k + 3], "lb");
                let ub = fl(lines[k + 4], "ub");
                let isint = bl(lines[k + 5], "isint");
                r.pp_in = Some((cut, lb, ub, isint));
                k += 6;
                let q: Vec<&str> = lines[k].split_ascii_whitespace().collect();
                if q[0] == "GC_PP_FAIL" {
                    r.pp_out = Some(None);
                } else {
                    assert_eq!(q[0], "GC_PP_OUT");
                    r.pp_out = Some(Some(CoreCut {
                        inds: il(lines[k + 1], "inds"),
                        vals: fl(lines[k + 2], "vals"),
                        rhs: CDouble::new(parse_hex_f64(kv(&q, "rhs"))),
                        integral_support: kv(&q, "isup") == "1",
                        integral_coefficients: kv(&q, "icoef") == "1",
                    }));
                }
            }
        }
        r.raw = lines[i..=end].join("\n");
        out.push(r);
        i = end + 1;
    }
    out
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[derive(Default, Debug)]
pub(crate) struct ReplayStats {
    pub records: usize,
    pub core_checked: usize,
    pub core_cuts: usize,
    pub pp_checked: usize,
    pub pp_cuts: usize,
    pub mismatches: Vec<String>,
    pub paths: HashMap<String, usize>,
    /// When set, records chosen for the committed fixture: per dump file at most
    /// `FIXTURE_PER_KIND` short records of each outcome kind.
    pub fixture: Option<String>,
}

const FIXTURE_PER_KIND: usize = 2;
const FIXTURE_MAX_ROWLEN: usize = 120;

/// Replays every record; a mismatch is recorded, not panicked on, so a run
/// reports how many records diverge and where.
pub(crate) fn replay(text: &str, tag: &str, stats: &mut ReplayStats) {
    let mut kinds: HashMap<String, usize> = HashMap::new();
    for r in parse(text) {
        stats.records += 1;
        let p = r.params.unwrap();
        let row = r.row.as_ref().unwrap();
        let mut g = CutGenCore::new(p, HighsRandom::from_state(r.rng_in));
        let got = g.generate_cut_core(row, r.only_init);
        let (rng_after, want) = r.core.as_ref().unwrap();
        stats.core_checked += 1;
        let mut bad = Vec::new();
        if g.rng().state() != *rng_after {
            bad.push("rng".to_string());
        }
        match (&got, want) {
            (None, None) => {}
            (Some(a), Some(b)) => {
                stats.core_cuts += 1;
                *stats
                    .paths
                    .entry(format!("{:?}", g.last_path()))
                    .or_default() += 1;
                if a.inds != b.inds {
                    bad.push("inds".into());
                }
                if bits(&a.vals) != bits(&b.vals) {
                    bad.push("vals".into());
                }
                if a.rhs.hi.to_bits() != b.rhs.hi.to_bits()
                    || a.rhs.lo.to_bits() != b.rhs.lo.to_bits()
                {
                    bad.push(format!(
                        "rhs {:e}/{:e} vs {:e}/{:e}",
                        a.rhs.hi, a.rhs.lo, b.rhs.hi, b.rhs.lo
                    ));
                }
                if a.integral_support != b.integral_support
                    || a.integral_coefficients != b.integral_coefficients
                {
                    bad.push("flags".into());
                }
            }
            (a, b) => bad.push(format!(
                "outcome rust={} highs={}",
                a.is_some(),
                b.is_some()
            )),
        }
        if let (Some((cut, lb, ub, isint)), Some(want_pp)) = (&r.pp_in, &r.pp_out) {
            stats.pp_checked += 1;
            let mut lbm = HashMap::new();
            for (k, &c) in cut.inds.iter().enumerate() {
                lbm.insert(c, (lb[k], ub[k], isint[k]));
            }
            let lbf = |c: i32| lbm[&c].0;
            let ubf = |c: i32| lbm[&c].1;
            let intf = |c: i32| lbm[&c].2;
            let got = g.postprocess_cut(cut.clone(), &lbf, &ubf, &intf);
            match (&got, want_pp) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    stats.pp_cuts += 1;
                    if a.inds != b.inds || bits(&a.vals) != bits(&b.vals) {
                        bad.push("pp row".into());
                    }
                    if a.rhs.to_f64().to_bits() != b.rhs.to_f64().to_bits() {
                        bad.push(format!(
                            "pp rhs {:e} vs {:e}",
                            a.rhs.to_f64(),
                            b.rhs.to_f64()
                        ));
                    }
                    if a.integral_support != b.integral_support
                        || a.integral_coefficients != b.integral_coefficients
                    {
                        bad.push("pp flags".into());
                    }
                }
                (a, b) => bad.push(format!(
                    "pp outcome rust={} highs={}",
                    a.is_some(),
                    b.is_some()
                )),
            }
        }
        if !bad.is_empty() {
            stats
                .mismatches
                .push(format!("{tag} #{}: {}", r.id, bad.join(", ")));
        }
        if let Some(fx) = stats.fixture.as_mut() {
            let kind = match (&r.core.as_ref().unwrap().1, r.untransform_fail, &r.pp_out) {
                (None, _, _) => "core_fail".to_string(),
                (Some(_), true, _) => "untransform_fail".to_string(),
                (Some(_), false, Some(None)) => "pp_fail".to_string(),
                _ => format!("{:?}", g.last_path()),
            };
            let n = kinds.entry(kind).or_default();
            if *n < FIXTURE_PER_KIND && row.inds.len() <= FIXTURE_MAX_ROWLEN {
                *n += 1;
                fx.push_str(&r.raw);
                fx.push('\n');
            }
        }
    }
}

#[test]
fn hex_float_parser_round_trips() {
    for v in [
        0.0,
        -0.0,
        1.0,
        -2.5,
        1.0 / 3.0,
        1e-300,
        5e-324,
        1.7976931348623157e308,
        -1e-17,
    ] {
        let s = hexf(v);
        assert_eq!(parse_hex_f64(&s).to_bits(), v.to_bits(), "{s}");
    }
    assert_eq!(parse_hex_f64("inf"), f64::INFINITY);
    assert_eq!(parse_hex_f64("0x1.8p+1"), 3.0);
    assert_eq!(parse_hex_f64("0x0.0000000000001p-1022"), 5e-324);
}

/// `%a` as glibc prints it (normal: `0x1.<13 hex>p<e>` with trailing zeros dropped).
#[cfg(test)]
fn hexf(v: f64) -> String {
    if v == 0.0 {
        return if v.is_sign_negative() {
            "-0x0p+0".into()
        } else {
            "0x0p+0".into()
        };
    }
    let b = v.to_bits();
    let neg = if b >> 63 == 1 { "-" } else { "" };
    let e = ((b >> 52) & 0x7ff) as i32;
    let f = b & ((1u64 << 52) - 1);
    let (lead, exp) = if e == 0 { (0, -1022) } else { (1, e - 1023) };
    let mut frac = format!("{f:013x}");
    while frac.ends_with('0') {
        frac.pop();
    }
    let dot = if frac.is_empty() {
        String::new()
    } else {
        format!(".{frac}")
    };
    format!("{neg}0x{lead}{dot}p{exp:+}")
}

#[test]
fn fixture_replays_bit_identically() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/highs_cutgen_replay.txt"
    );
    let text = std::fs::read_to_string(path).expect("replay fixture");
    let mut s = ReplayStats::default();
    replay(&text, "fixture", &mut s);
    // §6: the replay must have exercised every stage it claims to check.
    assert!(s.records >= 200, "{s:?}");
    assert!(s.core_cuts >= 50 && s.pp_cuts >= 50, "{s:?}");
    assert!(
        s.core_checked - s.core_cuts >= 20,
        "core failures exercised: {s:?}"
    );
    assert!(
        s.mismatches.is_empty(),
        "{} of {} records diverge: {:#?}",
        s.mismatches.len(),
        s.records,
        s.mismatches
    );
    eprintln!(
        "replayed {} records: {} core cuts, {} postprocessed cuts",
        s.records, s.core_cuts, s.pp_cuts
    );
}

#[test]
#[ignore]
fn full_dump_replay() {
    let p = std::env::var("DISCOPT_HIGHS_CUTGEN_REPLAY").expect("set DISCOPT_HIGHS_CUTGEN_REPLAY");
    let md = std::fs::metadata(&p).unwrap();
    let files: Vec<std::path::PathBuf> = if md.is_dir() {
        let mut v: Vec<_> = std::fs::read_dir(&p)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        v.sort();
        v
    } else {
        vec![p.into()]
    };
    let mut s = ReplayStats::default();
    let fixture_out = std::env::var("DISCOPT_HIGHS_CUTGEN_FIXTURE_OUT").ok();
    if fixture_out.is_some() {
        s.fixture = Some(String::new());
    }
    for f in &files {
        let text = std::fs::read_to_string(f).unwrap();
        let before = s.mismatches.len();
        let rec0 = s.records;
        replay(&text, &f.file_name().unwrap().to_string_lossy(), &mut s);
        eprintln!(
            "{}: {} records, {} mismatches",
            f.file_name().unwrap().to_string_lossy(),
            s.records - rec0,
            s.mismatches.len() - before
        );
    }
    eprintln!("paths: {:?}", s.paths);
    if let (Some(path), Some(fx)) = (fixture_out, s.fixture.as_ref()) {
        std::fs::write(path, fx).unwrap();
    }
    for m in s.mismatches.iter().take(40) {
        eprintln!("  {m}");
    }
    eprintln!(
        "TOTAL records {} core_cuts {} pp_checked {} pp_cuts {} mismatches {}",
        s.records,
        s.core_cuts,
        s.pp_checked,
        s.pp_cuts,
        s.mismatches.len()
    );
    assert!(s.records > 0);
    assert!(s.mismatches.is_empty());
}
