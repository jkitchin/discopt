//! Replay of HiGHS `HighsTransformedLp` dumps (`HS_DUMP_TL`): each dumped object's
//! construction, then every `transform`/`untransform` call made on it, in order,
//! compared bit for bit. See `p3_highs_transformedlp_dump.patch` for the format
//! writer; the reader below is the format's specification:
//!
//! ```text
//! TL_NEW <id> ncols N nrows M mipcols K feastol F smv F
//!   colval N.. coldual N.. rowval M.. glb N.. gub N.. slb M.. sub M.. isint N+M..
//!   contcols k.. intcols k..
//!   VB <col> <nvub> (<col> <coef> <const> <ndown> <nup>)* <nvlb> (...)*   (0 or more)
//! TL_ABORT | TL_OUT <id>  lbdist ubdist slbdist subdist bdist (N+M each)
//!                         bestvlb N+M (<col> <coef> <const>)*  bestvub ...
//! TF <id> prefervbds B intspos B rhs F   inds.. vals..  bnd n (<col> <lb> <ub>)*
//! TF_OUT 0 [free] | TF_OUT 1 intspos B rhs F  inds.. vals.. upper.. solval..
//!   (a `TF_OUT 0` from the empty-row exit also carries the arrays)
//! UT <id> integral B rhs F  inds.. vals..  bnd n (<col> <lb> <ub>)*  rows k
//!   (row <r> <slb> <sub>  ri n..) (rv n..)  per row
//! UT_OUT 0 | UT_OUT 1 rhs F  inds.. vals..
//! ```

use std::collections::HashMap;

use super::replay::parse_hex_f64 as hx;
use super::transformed_lp::{TlContext, TlSetup, TransformedLp, VarBound, VbEntry};

/// `mipdata_->epsilon` (1e-9 in every dump: the GC dumps record it).
const EPSILON: f64 = 1e-9;

struct Ctx {
    ncols: usize,
    nrows: usize,
    feastol: f64,
    smv: f64,
    glb: Vec<f64>,
    gub: Vec<f64>,
    slb: Vec<f64>,
    sub: Vec<f64>,
    isint: Vec<bool>,
    rows: HashMap<i32, (Vec<i32>, Vec<f64>)>,
}

impl TlContext for Ctx {
    fn num_cols(&self) -> usize {
        self.ncols
    }
    fn num_rows(&self) -> usize {
        self.nrows
    }
    fn feastol(&self) -> f64 {
        self.feastol
    }
    fn small_matrix_value(&self) -> f64 {
        self.smv
    }
    fn col_lower(&self, col: i32) -> f64 {
        self.glb[col as usize]
    }
    fn col_upper(&self, col: i32) -> f64 {
        self.gub[col as usize]
    }
    fn slack_lower(&self, row: i32) -> f64 {
        self.slb[row as usize]
    }
    fn slack_upper(&self, row: i32) -> f64 {
        self.sub[row as usize]
    }
    fn is_col_integral(&self, col: i32) -> bool {
        self.isint[col as usize]
    }
    fn row(&self, row: i32) -> (&[i32], &[f64]) {
        let r = self
            .rows
            .get(&row)
            .unwrap_or_else(|| panic!("row {row} not dumped"));
        (&r.0, &r.1)
    }
}

fn arr<'a>(line: &'a str, tag: &str) -> Vec<&'a str> {
    let mut it = line.split_ascii_whitespace();
    assert_eq!(
        it.next(),
        Some(tag),
        "expected {tag}: {}",
        &line[..line.len().min(80)]
    );
    let n: usize = it.next().unwrap().parse().unwrap();
    let v: Vec<&str> = it.collect();
    assert_eq!(v.len(), n, "{tag} length");
    v
}
fn fa(line: &str, tag: &str) -> Vec<f64> {
    arr(line, tag).iter().map(|t| hx(t)).collect()
}
fn ia(line: &str, tag: &str) -> Vec<i32> {
    arr(line, tag).iter().map(|t| t.parse().unwrap()).collect()
}
fn kv<'a>(t: &[&'a str], key: &str) -> &'a str {
    t[t.iter()
        .position(|x| *x == key)
        .unwrap_or_else(|| panic!("missing {key}"))
        + 1]
}
/// `bnd n (col lb ub)*`
fn triples(line: &str, tag: &str) -> Vec<(i32, f64, f64)> {
    let mut it = line.split_ascii_whitespace();
    assert_eq!(it.next(), Some(tag));
    let n: usize = it.next().unwrap().parse().unwrap();
    let v: Vec<&str> = it.collect();
    assert_eq!(v.len(), 3 * n);
    v.chunks(3)
        .map(|c| (c[0].parse().unwrap(), hx(c[1]), hx(c[2])))
        .collect()
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Replay statistics.
#[derive(Default, Debug)]
pub(crate) struct TlStats {
    pub tls: usize,
    pub tl_vbd_cols: usize,
    pub tf: usize,
    pub tf_rows: usize,
    pub ut: usize,
    pub ut_rows: usize,
    pub mismatches: Vec<String>,
}

struct Live {
    ctx: Ctx,
    tl: TransformedLp,
}

pub(crate) fn replay(text: &str, tag: &str, s: &mut TlStats) {
    let lines: Vec<&str> = text.lines().collect();
    let mut live: HashMap<i64, Live> = HashMap::new();
    let mut i = 0;
    let n = lines.len();
    while i < n {
        let l = lines[i];
        let t: Vec<&str> = l.split_ascii_whitespace().collect();
        if t.is_empty() {
            i += 1;
            continue;
        }
        match t[0] {
            "TL_NEW" => {
                let id: i64 = t[1].parse().unwrap();
                let ncols: usize = kv(&t, "ncols").parse().unwrap();
                let nrows: usize = kv(&t, "nrows").parse().unwrap();
                let feastol = hx(kv(&t, "feastol"));
                let smv = hx(kv(&t, "smv"));
                let colval = fa(lines[i + 1], "colval");
                let coldual = fa(lines[i + 2], "coldual");
                let rowval = fa(lines[i + 3], "rowval");
                let glb = fa(lines[i + 4], "glb");
                let gub = fa(lines[i + 5], "gub");
                let slb = fa(lines[i + 6], "slb");
                let sub = fa(lines[i + 7], "sub");
                let isint: Vec<bool> = arr(lines[i + 8], "isint")
                    .iter()
                    .map(|x| *x == "1")
                    .collect();
                let contcols = ia(lines[i + 9], "contcols");
                let intcols = ia(lines[i + 10], "intcols");
                i += 11;
                let mut vubs: HashMap<i32, Vec<VbEntry>> = HashMap::new();
                let mut vlbs: HashMap<i32, Vec<VbEntry>> = HashMap::new();
                while lines[i].starts_with("VB ") {
                    let v: Vec<&str> = lines[i].split_ascii_whitespace().collect();
                    let col: i32 = v[1].parse().unwrap();
                    let mut k = 2;
                    let take = |k: &mut usize| -> Vec<VbEntry> {
                        let m: usize = v[*k].parse().unwrap();
                        *k += 1;
                        let mut out = Vec::new();
                        for _ in 0..m {
                            out.push(VbEntry {
                                col: v[*k].parse().unwrap(),
                                vb: VarBound {
                                    coef: hx(v[*k + 1]),
                                    constant: hx(v[*k + 2]),
                                },
                                nodes_down: v[*k + 3].parse().unwrap(),
                                nodes_up: v[*k + 4].parse().unwrap(),
                            });
                            *k += 5;
                        }
                        out
                    };
                    let vu = take(&mut k);
                    let vl = take(&mut k);
                    // A column appears once per cleanup loop it is in; keep the last.
                    vubs.insert(col, vu);
                    vlbs.insert(col, vl);
                    s.tl_vbd_cols += 1;
                    i += 1;
                }
                if lines[i] == "TL_ABORT" {
                    i += 1;
                    continue;
                }
                let h: Vec<&str> = lines[i].split_ascii_whitespace().collect();
                assert_eq!(h[0], "TL_OUT");
                let want_lb = fa(lines[i + 1], "lbdist");
                let want_ub = fa(lines[i + 2], "ubdist");
                let want_slb = fa(lines[i + 3], "slbdist");
                let want_sub = fa(lines[i + 4], "subdist");
                let want_bd = fa(lines[i + 5], "bdist");
                let pairs = |line: &str, tag: &str| -> Vec<(i32, f64, f64)> {
                    let mut it = line.split_ascii_whitespace();
                    assert_eq!(it.next(), Some(tag));
                    let m: usize = it.next().unwrap().parse().unwrap();
                    let v: Vec<&str> = it.collect();
                    assert_eq!(v.len(), 3 * m);
                    v.chunks(3)
                        .map(|c| (c[0].parse().unwrap(), hx(c[1]), hx(c[2])))
                        .collect()
                };
                let want_vlb = pairs(lines[i + 6], "bestvlb");
                let want_vub = pairs(lines[i + 7], "bestvub");
                i += 8;
                let ctx = Ctx {
                    ncols,
                    nrows,
                    feastol,
                    smv,
                    glb,
                    gub,
                    slb,
                    sub,
                    isint,
                    rows: HashMap::new(),
                };
                let empty: Vec<VbEntry> = Vec::new();
                let fvu = |c: i32| vubs.get(&c).cloned().unwrap_or_else(|| empty.clone());
                let fvl = |c: i32| vlbs.get(&c).cloned().unwrap_or_else(|| empty.clone());
                let setup = TlSetup {
                    col_value: &colval,
                    col_dual: &coldual,
                    row_value: &rowval,
                    continuous_cols: &contcols,
                    integral_cols: &intcols,
                    vubs: &fvu,
                    vlbs: &fvl,
                };
                let tl = TransformedLp::new(&ctx, &setup);
                s.tls += 1;
                let (lb, ub, slb, sub, bd, bvlb, bvub) = tl.debug_state();
                let mut bad = Vec::new();
                for (name, a, b) in [
                    ("lbdist", lb, &want_lb),
                    ("ubdist", ub, &want_ub),
                    ("slbdist", slb, &want_slb),
                    ("subdist", sub, &want_sub),
                    ("bdist", bd, &want_bd),
                ] {
                    if bits(a) != bits(b) {
                        let k = (0..a.len())
                            .find(|&k| a[k].to_bits() != b[k].to_bits())
                            .unwrap();
                        bad.push(format!("{name}[{k}] {:e} vs {:e}", a[k], b[k]));
                    }
                }
                for (name, a, b) in [("bestvlb", bvlb, &want_vlb), ("bestvub", bvub, &want_vub)] {
                    for k in 0..a.len() {
                        let (wc, wco, wcs) = b[k];
                        // Only the chosen column is meaningful when none was chosen.
                        if a[k].0 != wc
                            || (wc != -1
                                && (a[k].1.coef.to_bits() != wco.to_bits()
                                    || a[k].1.constant.to_bits() != wcs.to_bits()))
                        {
                            bad.push(format!(
                                "{name}[{k}] {:?} vs ({wc}, {wco:e}, {wcs:e})",
                                a[k]
                            ));
                            break;
                        }
                    }
                }
                if !bad.is_empty() {
                    s.mismatches
                        .push(format!("{tag} TL{id}: {}", bad.join("; ")));
                }
                live.insert(id, Live { ctx, tl });
            }
            "TF" => {
                let id: i64 = t[1].parse().unwrap();
                let prefer = kv(&t, "prefervbds") == "1";
                let intspos = kv(&t, "intspos") == "1";
                let rhs = hx(kv(&t, "rhs"));
                let inds = ia(lines[i + 1], "inds");
                let vals = fa(lines[i + 2], "vals");
                let bnd = triples(lines[i + 3], "bnd");
                let out: Vec<&str> = lines[i + 4].split_ascii_whitespace().collect();
                assert_eq!(out[0], "TF_OUT");
                let lv = live.get_mut(&id).expect("TF for unknown TL");
                for &(c, l, u) in &bnd {
                    let c = c as usize;
                    if c < lv.ctx.ncols {
                        lv.ctx.glb[c] = l;
                        lv.ctx.gub[c] = u;
                    } else {
                        lv.ctx.slb[c - lv.ctx.ncols] = l;
                        lv.ctx.sub[c - lv.ctx.ncols] = u;
                    }
                }
                s.tf += 1;
                let got = lv
                    .tl
                    .transform(&lv.ctx, EPSILON, inds, vals, rhs, intspos, prefer);
                let mut bad = Vec::new();
                let has_arrays = out.len() > 2 && out[2] == "intspos";
                if has_arrays {
                    let ret = out[1] == "1";
                    let w_intspos = kv(&out, "intspos") == "1";
                    let w_rhs = hx(kv(&out, "rhs"));
                    let wi = ia(lines[i + 5], "inds");
                    let wv = fa(lines[i + 6], "vals");
                    let wu = fa(lines[i + 7], "upper");
                    let ws = fa(lines[i + 8], "solval");
                    i += 9;
                    match (&got, ret) {
                        (None, false) => {}
                        (Some(r), true) => {
                            s.tf_rows += 1;
                            if r.inds != wi {
                                bad.push("inds".to_string());
                            }
                            if bits(&r.vals) != bits(&wv) {
                                bad.push("vals".into());
                            }
                            if bits(&r.upper) != bits(&wu) {
                                bad.push("upper".into());
                            }
                            if bits(&r.solval) != bits(&ws) {
                                bad.push("solval".into());
                            }
                            if r.rhs.to_bits() != w_rhs.to_bits() {
                                bad.push(format!("rhs {:e} vs {:e}", r.rhs, w_rhs));
                            }
                            if r.integers_positive != w_intspos {
                                bad.push("intspos".into());
                            }
                        }
                        (g, w) => bad.push(format!("outcome rust={} highs={w}", g.is_some())),
                    }
                } else {
                    i += 5;
                    if got.is_some() {
                        bad.push("outcome rust=true highs=false(free)".into());
                    }
                }
                if !bad.is_empty() {
                    s.mismatches
                        .push(format!("{tag} TF@{} TL{id}: {}", s.tf, bad.join(", ")));
                }
            }
            "UT" => {
                let id: i64 = t[1].parse().unwrap();
                let integral = kv(&t, "integral") == "1";
                let rhs = hx(kv(&t, "rhs"));
                let inds = ia(lines[i + 1], "inds");
                let vals = fa(lines[i + 2], "vals");
                let bnd = triples(lines[i + 3], "bnd");
                let rt: Vec<&str> = lines[i + 4].split_ascii_whitespace().collect();
                assert_eq!(rt[0], "rows");
                let nr: usize = rt[1].parse().unwrap();
                let lv = live.get_mut(&id).expect("UT for unknown TL");
                let mut k = i + 5;
                for _ in 0..nr {
                    let h: Vec<&str> = lines[k].split_ascii_whitespace().collect();
                    assert_eq!(h[0], "row");
                    let r: i32 = h[1].parse().unwrap();
                    lv.ctx.slb[r as usize] = hx(h[2]);
                    lv.ctx.sub[r as usize] = hx(h[3]);
                    assert_eq!(h[4], "ri");
                    let m: usize = h[5].parse().unwrap();
                    let ri: Vec<i32> = h[6..6 + m].iter().map(|x| x.parse().unwrap()).collect();
                    let rv = fa(lines[k + 1], "rv");
                    lv.ctx.rows.insert(r, (ri, rv));
                    k += 2;
                }
                for &(c, l, u) in &bnd {
                    lv.ctx.glb[c as usize] = l;
                    lv.ctx.gub[c as usize] = u;
                }
                let out: Vec<&str> = lines[k].split_ascii_whitespace().collect();
                assert_eq!(out[0], "UT_OUT");
                s.ut += 1;
                let got = lv.tl.untransform(&lv.ctx, &inds, &vals, rhs, integral);
                let mut bad = Vec::new();
                if out[1] == "1" {
                    let w_rhs = hx(kv(&out, "rhs"));
                    let wi = ia(lines[k + 1], "inds");
                    let wv = fa(lines[k + 2], "vals");
                    i = k + 3;
                    match &got {
                        Some((gi, gv, gr)) => {
                            s.ut_rows += 1;
                            if *gi != wi {
                                bad.push("inds".to_string());
                            }
                            if bits(gv) != bits(&wv) {
                                bad.push("vals".into());
                            }
                            if gr.to_bits() != w_rhs.to_bits() {
                                bad.push(format!("rhs {gr:e} vs {w_rhs:e}"));
                            }
                        }
                        None => bad.push("outcome rust=false highs=true".into()),
                    }
                } else {
                    i = k + 1;
                    if got.is_some() {
                        bad.push("outcome rust=true highs=false".into());
                    }
                }
                if !bad.is_empty() {
                    s.mismatches
                        .push(format!("{tag} UT@{} TL{id}: {}", s.ut, bad.join(", ")));
                }
            }
            _ => i += 1,
        }
    }
}

#[test]
#[ignore]
fn full_transformed_lp_replay() {
    let p = std::env::var("DISCOPT_HIGHS_TL_REPLAY").expect("set DISCOPT_HIGHS_TL_REPLAY");
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
    let mut s = TlStats::default();
    for f in &files {
        let text = std::fs::read_to_string(f).unwrap();
        let before = s.mismatches.len();
        let (tf0, ut0) = (s.tf, s.ut);
        replay(&text, &f.file_name().unwrap().to_string_lossy(), &mut s);
        eprintln!(
            "{}: tf {} ut {} mismatches {}",
            f.file_name().unwrap().to_string_lossy(),
            s.tf - tf0,
            s.ut - ut0,
            s.mismatches.len() - before
        );
    }
    for m in s.mismatches.iter().take(40) {
        eprintln!("  {m}");
    }
    eprintln!(
        "TOTAL tls {} vbd_cols {} tf {} tf_rows {} ut {} ut_rows {} mismatches {}",
        s.tls,
        s.tl_vbd_cols,
        s.tf,
        s.tf_rows,
        s.ut,
        s.ut_rows,
        s.mismatches.len()
    );
    assert!(s.tf > 0 && s.ut > 0);
    assert!(s.mismatches.is_empty());
}

#[test]
fn transformed_lp_fixture_replays_bit_identically() {
    // First transformed LP of the root (presolve off) of egout, gr4x6, bell5,
    // enlight8 and lseu, with every call made on it.
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/highs_transformedlp_replay.txt"
    );
    let text = std::fs::read_to_string(path).expect("transformed-LP fixture");
    let mut s = TlStats::default();
    replay(&text, "fixture", &mut s);
    // §6: the replay must exercise variable bounds, transforms and untransforms.
    assert_eq!(s.tls, 5, "{s:?}");
    assert!(s.tl_vbd_cols >= 40, "{s:?}");
    assert!(s.tf_rows >= 1000 && s.ut_rows >= 200, "{s:?}");
    assert!(s.mismatches.is_empty(), "{:#?}", s.mismatches);
}
