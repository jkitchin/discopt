//! Matched-box node comparison (`DISCOPT_NODEBOX_FILE`, diagnostic instrument).
//!
//! The file holds node boxes another solver evaluated: HiGHS's `HS_DUMP_NODEBOX` dump
//! (the patched build of `docs/dev/data/highs-parity/matched-box/`), one record per
//! node-LP solve (`NBE`) or node separation (`NBS`), each a header line followed by
//! `L <n> <lb...>` and `U <n> <ub...>` with every column's absolute bound as a C `%a`
//! hex float. Columns are the driver's structural columns, which for an MPS read
//! without presolve are the file's column order; a record with another column count
//! panics (a mapping error must not be reported as a comparison).
//!
//! For each box, at the top-level solve and before the search starts (root cuts done,
//! search LP and cut pool seeded, the driver's incumbent accepted), the harness:
//!
//! 1. intersects it with the driver's global box (`l_w`/`u_w`), counting columns
//!    whose external bound lies outside ours and boxes the intersection empties;
//! 2. installs it on a search environment built exactly as [`super::run`] builds its
//!    own (`HighsEnv::new`, `seed_pool`, `accept`) and evaluates it through
//!    [`SearchEnv::evaluate`] -- propagation, the node LP with the root cut rows, the
//!    NS-safe bound, reduced-cost fixing and re-solve: bound (a);
//! 3. for a separation record (`NBS`, a node where HiGHS separated) that (a) leaves
//!    open, runs [`SearchEnv::separate`] (the search's node separation) and evaluates
//!    the node again, as the search's dive does after separation: bound (b), which
//!    includes the separation lift. `NBE` records get (a) only, so in `cumulative`
//!    mode cuts are separated exactly at the boxes where HiGHS separated.
//!
//! Every bound printed is a node lower bound the search would use (`NodeData::
//! lower_bound`, from the certified LP bound; `inf` when propagation or the LP proves
//! the box empty). The LP's own objective is printed separately as `lpobj`, which is
//! not a bound. One `NODEBOX` line per record goes to stderr, then `NODEBOX_END` with
//! the counts. By default (`DISCOPT_NODEBOX_MODE=fresh`) every box gets a new
//! environment, so boxes do not see each other's cuts; `cumulative` evaluates them in
//! file order on one environment, so cut rows separated at one box persist (and age)
//! into the next, as in a search. Either way the search that follows uses its own
//! environment and is unchanged. Off when unset; read only at the top-level solve,
//! and a top-level call with no integer column (a relaxation solved through the
//! driver) is skipped with a `NODEBOX_SKIP` line; `NODEBOX_BEGIN` names the call.

use super::*;

/// One external node record: its kind and id (from the header) and its column box.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct BoxRecord {
    pub tag: String,
    pub id: i64,
    pub lo: Vec<f64>,
    pub hi: Vec<f64>,
}

/// A C `%a` hex float (`0x1.8p+3`, `-0x0p+0`), or `inf`/`-inf`/`nan`; decimal floats
/// are accepted too.
pub(super) fn parse_hex_f64(t: &str) -> Option<f64> {
    let (neg, body) = match t.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    let sign = if neg { -1.0 } else { 1.0 };
    let Some(hex) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) else {
        return body.parse::<f64>().ok().map(|v| sign * v);
    };
    let (mant, exp) = match hex.find(['p', 'P']) {
        Some(k) => (&hex[..k], hex[k + 1..].parse::<i32>().ok()?),
        None => (hex, 0),
    };
    let (int_part, frac_part) = match mant.find('.') {
        Some(k) => (&mant[..k], &mant[k + 1..]),
        None => (mant, ""),
    };
    if int_part.is_empty() && frac_part.is_empty() {
        return None;
    }
    let digits = int_part.len() + frac_part.len();
    if digits > 15 {
        // More than 60 bits: not a `%a` double.
        return None;
    }
    let mut m: u64 = 0;
    for ch in int_part.chars().chain(frac_part.chars()) {
        m = (m << 4) | ch.to_digit(16)? as u64;
    }
    let e = exp - 4 * frac_part.len() as i32;
    // `m` has at most 53 significant bits for a `%a` double, so `m as f64` is exact;
    // scale in two steps so no intermediate power of two overflows or underflows.
    let half = e / 2;
    Some(sign * (m as f64) * 2f64.powi(half) * 2f64.powi(e - half))
}

fn parse_vec(line: &str, want: char, lineno: usize) -> Result<Vec<f64>, String> {
    let mut it = line.split_ascii_whitespace();
    let tag = it.next().unwrap_or("");
    if tag != want.to_string() {
        return Err(format!("line {lineno}: expected `{want}`, found {tag:?}"));
    }
    let n: usize = it
        .next()
        .and_then(|t| t.parse().ok())
        .ok_or_else(|| format!("line {lineno}: missing count"))?;
    let v: Vec<f64> = it
        .map(|t| parse_hex_f64(t).ok_or_else(|| format!("line {lineno}: bad value {t:?}")))
        .collect::<Result<_, _>>()?;
    if v.len() != n {
        return Err(format!("line {lineno}: count {n}, {} values", v.len()));
    }
    Ok(v)
}

/// Parse a dump: every `NB*` header line followed by its `L` and `U` lines; other
/// lines are ignored. A truncated or malformed record is an error.
pub(super) fn parse_records(text: &str) -> Result<Vec<BoxRecord>, String> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        if !line.starts_with("NB") {
            i += 1;
            continue;
        }
        let mut it = line.split_ascii_whitespace();
        let tag = it.next().unwrap_or("").to_string();
        let id: i64 = it
            .next()
            .and_then(|t| t.parse().ok())
            .ok_or_else(|| format!("line {}: header without an id", i + 1))?;
        if i + 2 >= lines.len() {
            return Err(format!("line {}: record {tag} {id} is truncated", i + 1));
        }
        let lo = parse_vec(lines[i + 1], 'L', i + 2)?;
        let hi = parse_vec(lines[i + 2], 'U', i + 3)?;
        if lo.len() != hi.len() {
            return Err(format!(
                "record {tag} {id}: {} lower, {} upper",
                lo.len(),
                hi.len()
            ));
        }
        out.push(BoxRecord { tag, id, lo, hi });
        i += 3;
    }
    Ok(out)
}

/// An external box intersected with ours.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Intersection {
    pub lo: Vec<f64>,
    pub hi: Vec<f64>,
    /// Columns whose external lower bound is below ours / upper bound above ours.
    pub lo_out: Vec<usize>,
    pub hi_out: Vec<usize>,
    /// Columns the intersection leaves empty (`lo > hi`).
    pub empty: Vec<usize>,
}

/// `v` with the driver's infinity sentinel (`|v| >= INF`) as a true infinity.
fn unsentinel(v: f64) -> f64 {
    if v >= INF {
        f64::INFINITY
    } else if v <= -INF {
        f64::NEG_INFINITY
    } else {
        v
    }
}

/// Intersect the box `lo..hi` with ours `gl..gu` (bounds within `tol * max(1, |v|)`
/// count as equal).
pub(super) fn intersect(lo: &[f64], hi: &[f64], gl: &[f64], gu: &[f64], tol: f64) -> Intersection {
    let n = lo.len();
    let mut r = Intersection {
        lo: Vec::with_capacity(n),
        hi: Vec::with_capacity(n),
        lo_out: Vec::new(),
        hi_out: Vec::new(),
        empty: Vec::new(),
    };
    for j in 0..n {
        let (a, b) = (unsentinel(lo[j]), unsentinel(hi[j]));
        let (ga, gb) = (unsentinel(gl[j]), unsentinel(gu[j]));
        if a < ga && (ga.is_infinite() || ga - a > tol * ga.abs().max(1.0)) {
            r.lo_out.push(j);
        }
        if b > gb && (gb.is_infinite() || b - gb > tol * gb.abs().max(1.0)) {
            r.hi_out.push(j);
        }
        let (l, u) = (a.max(ga), b.min(gb));
        if l > u && l - u > tol * u.abs().max(1.0) {
            r.empty.push(j);
        }
        r.lo.push(l);
        r.hi.push(u);
    }
    r
}

fn bound_of(r: SearchResult, node: &NodeData) -> f64 {
    match r {
        SearchResult::DomainInfeasible | SearchResult::LpInfeasible => f64::INFINITY,
        _ => node.lower_bound,
    }
}

/// Fractional integer columns of an LP point (`-1` without one).
fn frac_count(ctx: &NodeCtx<'_>, x: Option<&[f64]>) -> i64 {
    x.map_or(-1, |x| {
        (0..ctx.ns)
            .filter(|&j| ctx.is_int[j] && frac(x[j]) > INT_TOL)
            .count() as i64
    })
}

fn fresh_env<'a, 'b>(
    ctx: &'a NodeCtx<'b>,
    lim: &'a Limits,
    initial: Option<&(Vec<f64>, f64)>,
    seed_pool: &[GomoryCut],
) -> HighsEnv<'a, 'b> {
    let mut env = HighsEnv::new(ctx, lim, None, None, None).unwrap_or_else(|| {
        panic!("DISCOPT_NODEBOX_FILE: the root working LP's rows are infeasible over the box")
    });
    env.seed_pool(seed_pool.to_vec());
    if let Some((x, v)) = initial {
        env.accept(x.clone(), *v);
    }
    env
}

/// Run the harness when `DISCOPT_NODEBOX_FILE` is set (see the module docs).
pub(super) fn harness(
    ctx: &NodeCtx<'_>,
    lim: &Limits,
    initial: Option<&(Vec<f64>, f64)>,
    seed_pool: &[GomoryCut],
) {
    let Ok(path) = std::env::var("DISCOPT_NODEBOX_FILE") else {
        return;
    };
    let cumulative = match std::env::var("DISCOPT_NODEBOX_MODE").as_deref() {
        Err(_) | Ok("fresh") => false,
        Ok("cumulative") => true,
        Ok(other) => panic!("DISCOPT_NODEBOX_MODE={other:?} (expected fresh or cumulative)"),
    };
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("DISCOPT_NODEBOX_FILE {path}: {e}"));
    let recs = parse_records(&text).unwrap_or_else(|e| panic!("DISCOPT_NODEBOX_FILE {path}: {e}"));
    if recs.is_empty() {
        panic!("DISCOPT_NODEBOX_FILE {path}: no records");
    }
    let ns = ctx.ns;
    let n_int = ctx.is_int[..ns].iter().filter(|&&b| b).count();
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let call = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if n_int == 0 {
        // A top-level search over a problem with no integer columns (a relaxation the
        // caller solves through the driver) is not a node comparison.
        eprintln!("NODEBOX_SKIP call={call} ns={ns} n_int=0");
        return;
    }
    let (gl, gu) = (&ctx.l_w[..ns], &ctx.u_w[..ns]);
    eprintln!(
        "NODEBOX_BEGIN call={call} records={} ns={ns} n_int={n_int} model_rows={} \
         root_lp_rows={} node_gmi={} cut_lp={} incumbent={:?} mode={}",
        recs.len(),
        ctx.n_orig_rows,
        ctx.m_w,
        ctx.node_gmi,
        search_cut_lp_enabled(),
        initial.map(|p| p.1),
        if cumulative { "cumulative" } else { "fresh" }
    );
    let mut out_cols: std::collections::BTreeSet<(usize, bool)> = Default::default();
    let mut shared: Option<HighsEnv<'_, '_>> = None;
    let (mut evaluated, mut separated) = (0usize, 0usize);
    for rec in &recs {
        if rec.lo.len() != ns {
            panic!(
                "DISCOPT_NODEBOX_FILE: record {} {} has {} columns, the driver {ns}: the \
                 column spaces differ",
                rec.tag,
                rec.id,
                rec.lo.len()
            );
        }
        let ix = intersect(&rec.lo, &rec.hi, gl, gu, 1e-9);
        for &j in &ix.lo_out {
            if out_cols.insert((j, false)) {
                eprintln!(
                    "NODEBOX_OUTSIDE col={j} side=lower external={:?} ours={:?} first={}:{}",
                    rec.lo[j], gl[j], rec.tag, rec.id
                );
            }
        }
        for &j in &ix.hi_out {
            if out_cols.insert((j, true)) {
                eprintln!(
                    "NODEBOX_OUTSIDE col={j} side=upper external={:?} ours={:?} first={}:{}",
                    rec.hi[j], gu[j], rec.tag, rec.id
                );
            }
        }
        let head = format!(
            "NODEBOX tag={} id={} lo_out={} hi_out={} empty_cols={}",
            rec.tag,
            rec.id,
            ix.lo_out.len(),
            ix.hi_out.len(),
            ix.empty.len()
        );
        if !ix.empty.is_empty() {
            eprintln!("{head} res_a=EmptyBox a=inf");
            continue;
        }
        let mut fresh;
        let env: &mut HighsEnv<'_, '_> = if cumulative {
            shared.get_or_insert_with(|| fresh_env(ctx, lim, initial, seed_pool))
        } else {
            fresh = fresh_env(ctx, lim, initial, seed_pool);
            &mut fresh
        };
        evaluated += 1;
        let cutoff = env.eval.cutoff;
        let rows_a = env.lp_rows();
        let u0 = env.eval.stats.untrusted;
        let i0 = env.eval.stats.integral;
        let mut node = NodeData::new(f64::NEG_INFINITY, f64::NEG_INFINITY);
        if !env.eval.install_box(&ix.lo, &ix.hi) {
            eprintln!("{head} res_a=DomainInfeasible a=inf rows_a={rows_a} cutoff={cutoff:?}");
            continue;
        }
        let ra = env.evaluate(&mut node, None, 0);
        let a = bound_of(ra, &node);
        let lpobj = node.lp_objective;
        let untrusted_a = env.eval.stats.untrusted - u0;
        let integral_a = env.eval.stats.integral - i0;
        let mut line = format!(
            "{head} res_a={ra:?} a={a:?} lpobj_a={lpobj:?} rows_a={rows_a} \
             untrusted_a={untrusted_a} integral_a={integral_a} frac_a={} cutoff={cutoff:?}",
            frac_count(ctx, env.eval.point.as_deref())
        );
        if ra == SearchResult::Open && rec.tag == "NBS" {
            let s0 = env.sep_calls;
            env.separate();
            let did = env.sep_calls > s0;
            separated += usize::from(did);
            let lift = env.eval.pending_lift;
            let rows_b = env.lp_rows();
            let u1 = env.eval.stats.untrusted;
            let rb = env.evaluate(&mut node, None, 0);
            let b = bound_of(rb, &node);
            line.push_str(&format!(
                " sep={} lift={lift:?} res_b={rb:?} b={b:?} lpobj_b={:?} rows_b={rows_b} \
                 untrusted_b={}",
                u8::from(did),
                node.lp_objective,
                env.eval.stats.untrusted - u1
            ));
        }
        if cumulative {
            env.delete_aged_cut_rows();
        }
        eprintln!("{line}");
    }
    eprintln!(
        "NODEBOX_END records={} evaluated={evaluated} separated={separated} \
         outside_cols={}",
        recs.len(),
        out_cols.len()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_floats_parse_as_c_prints_them() {
        let cases: [(&str, f64); 9] = [
            ("0x1.8p+3", 12.0),
            ("-0x1p-1", -0.5),
            ("0x0p+0", 0.0),
            ("-0x0p+0", -0.0),
            (
                "0x1.92980000218eap+15",
                f64::from_bits(0x40E9_2980_0002_18EA),
            ),
            ("inf", f64::INFINITY),
            ("-inf", f64::NEG_INFINITY),
            ("100", 100.0),
            ("0x1.fffffffffffffp+1023", f64::MAX),
        ];
        for (t, want) in cases {
            let got = parse_hex_f64(t).unwrap();
            assert_eq!(got.to_bits(), want.to_bits(), "{t}");
        }
        assert_eq!(parse_hex_f64("0x1p-1074"), Some(f64::from_bits(1)));
        assert!(parse_hex_f64("0xg").is_none());
        assert!(parse_hex_f64("0xp+1").is_none());
        assert!(parse_hex_f64("abc").is_none());
    }

    #[test]
    fn records_parse_and_reject_truncation() {
        let text =
            "junk\nNBE 7 0 1\nL 2 0x0p+0 -inf\nU 2 0x1p+0 inf\nNBS 9 3\nL 1 0x1p+1\nU 1 0x1p+2\n";
        let r = parse_records(text).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].tag, "NBE");
        assert_eq!(r[0].id, 7);
        assert_eq!(r[0].lo, vec![0.0, f64::NEG_INFINITY]);
        assert_eq!(r[0].hi, vec![1.0, f64::INFINITY]);
        assert_eq!(
            (r[1].tag.as_str(), r[1].id, r[1].lo[0], r[1].hi[0]),
            ("NBS", 9, 2.0, 4.0)
        );
        assert!(parse_records("NBE 1\nL 1 0x0p+0\n").is_err());
        assert!(parse_records("NBE 1\nL 2 0x0p+0\nU 2 0x0p+0 0x0p+0\n").is_err());
        assert!(parse_records("NBE 1\nU 1 0x0p+0\nL 1 0x0p+0\n").is_err());
    }

    #[test]
    fn intersection_reports_outside_and_empty_columns() {
        // col 0: inside ours; col 1: external looser on both sides (our sentinel
        // upper is infinite, so only the lower counts); col 2: disjoint from ours;
        // col 3: equal within tolerance.
        let lo = [1.0, -5.0, 10.0, 2.0 - 1e-12];
        let hi = [2.0, f64::INFINITY, 11.0, 3.0];
        let gl = [0.0, 0.0, 0.0, 2.0];
        let gu = [5.0, 1e20, 4.0, 3.0];
        let r = intersect(&lo, &hi, &gl, &gu, 1e-9);
        assert_eq!(r.lo_out, vec![1]);
        assert_eq!(r.hi_out, vec![2]);
        assert_eq!(r.empty, vec![2]);
        assert_eq!(r.lo, vec![1.0, 0.0, 10.0, 2.0]);
        assert_eq!(r.hi, vec![2.0, f64::INFINITY, 4.0, 3.0]);
    }
}
