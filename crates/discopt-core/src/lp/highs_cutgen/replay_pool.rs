//! Replay of HiGHS cut-pool dumps (`HS_DUMP_POOL`, in `p3_highs_pool_dump.patch`):
//! every `addCut` (its returned index), `separate` (the selected cut indices in
//! order and the adaptive score state after), `performAging`, `lpCutRemoved` and
//! `resetAge` call on the main solve's global pool, in order.
//!
//! ```text
//! POOL ncols N agelim A softlimit S modelnnz Z modelrows R
//! PA rhs F integral B propagate B extract B conflict B   inds..  vals..   PA_RET r
//! PS feastol F threadsafe B cutset k   sol..  lb..  ub..   PS_OUT best F fac F sel n i..
//! PG | PR <cut> <threadsafe> | PZ <cut> <threadsafe>
//! ```

use super::pool::{CutPool, NoObserver};
use super::replay::parse_hex_f64 as hx;

fn arr<'a>(line: &'a str, tag: &str) -> Vec<&'a str> {
    let mut it = line.split_ascii_whitespace();
    assert_eq!(it.next(), Some(tag), "expected {tag}");
    let n: usize = it.next().unwrap().parse().unwrap();
    let v: Vec<&str> = it.collect();
    assert_eq!(v.len(), n, "{tag} length");
    v
}
fn kv<'a>(t: &[&'a str], key: &str) -> &'a str {
    t[t.iter()
        .position(|x| *x == key)
        .unwrap_or_else(|| panic!("missing {key}"))
        + 1]
}

#[derive(Default, Debug)]
pub(crate) struct PoolStats {
    pub adds: usize,
    pub accepted: usize,
    pub duplicates: usize,
    pub separates: usize,
    pub selected: usize,
    pub agings: usize,
    pub removed: usize,
    pub resets: usize,
    pub mismatches: Vec<String>,
}

pub(crate) fn replay(text: &str, tag: &str, s: &mut PoolStats) {
    let lines: Vec<&str> = text.lines().collect();
    let mut pool: Option<CutPool> = None;
    let mut obs = NoObserver;
    let mut i = 0;
    while i < lines.len() {
        let t: Vec<&str> = lines[i].split_ascii_whitespace().collect();
        if t.is_empty() {
            i += 1;
            continue;
        }
        match t[0] {
            "POOL" => {
                pool = Some(CutPool::new(
                    kv(&t, "ncols").parse().unwrap(),
                    kv(&t, "agelim").parse().unwrap(),
                    kv(&t, "softlimit").parse().unwrap(),
                    kv(&t, "modelnnz").parse().unwrap(),
                    kv(&t, "modelrows").parse().unwrap(),
                ));
                i += 1;
            }
            "PA" => {
                let p = pool.as_mut().expect("PA before POOL");
                let inds: Vec<i32> = arr(lines[i + 1], "inds")
                    .iter()
                    .map(|x| x.parse().unwrap())
                    .collect();
                let vals: Vec<f64> = arr(lines[i + 2], "vals").iter().map(|x| hx(x)).collect();
                let r: Vec<&str> = lines[i + 3].split_ascii_whitespace().collect();
                assert_eq!(r[0], "PA_RET");
                let want: i32 = r[1].parse().unwrap();
                let got = p.add_cut(
                    &mut obs,
                    &inds,
                    &vals,
                    hx(kv(&t, "rhs")),
                    kv(&t, "integral") == "1",
                    kv(&t, "propagate") == "1",
                    kv(&t, "conflict") == "1",
                );
                s.adds += 1;
                if want >= 0 {
                    s.accepted += 1;
                } else {
                    s.duplicates += 1;
                }
                if got != want {
                    s.mismatches
                        .push(format!("{tag} add #{}: index {got} vs {want}", s.adds));
                }
                i += 4;
            }
            "PS" => {
                let p = pool.as_mut().expect("PS before POOL");
                assert_eq!(kv(&t, "threadsafe"), "0");
                let sol: Vec<f64> = arr(lines[i + 1], "sol").iter().map(|x| hx(x)).collect();
                let lb: Vec<f64> = arr(lines[i + 2], "lb").iter().map(|x| hx(x)).collect();
                let ub: Vec<f64> = arr(lines[i + 3], "ub").iter().map(|x| hx(x)).collect();
                let o: Vec<&str> = lines[i + 4].split_ascii_whitespace().collect();
                assert_eq!(o[0], "PS_OUT");
                let want_best = hx(kv(&o, "best"));
                let want_fac = hx(kv(&o, "fac"));
                let nsel: usize = kv(&o, "sel").parse().unwrap();
                let pos = o.iter().position(|x| *x == "sel").unwrap();
                let want_sel: Vec<i32> = o[pos + 2..pos + 2 + nsel]
                    .iter()
                    .map(|x| x.parse().unwrap())
                    .collect();
                let mut cutset: Vec<i32> = Vec::new();
                let incoming: usize = kv(&t, "cutset").parse().unwrap();
                assert_eq!(
                    incoming, 0,
                    "separate with a pre-filled cut set is not replayed"
                );
                let got: Vec<i32> = p
                    .separate(&mut obs, &sol, &lb, &ub, hx(kv(&t, "feastol")), &mut cutset)
                    .iter()
                    .map(|c| c.index)
                    .collect();
                s.separates += 1;
                s.selected += want_sel.len();
                let (best, fac) = p.score_state();
                let mut bad = Vec::new();
                if got != want_sel {
                    bad.push(format!("selected {got:?} vs {want_sel:?}"));
                }
                if best.to_bits() != want_best.to_bits() || fac.to_bits() != want_fac.to_bits() {
                    bad.push(format!(
                        "score state {best:e}/{fac:e} vs {want_best:e}/{want_fac:e}"
                    ));
                }
                if !bad.is_empty() {
                    s.mismatches.push(format!(
                        "{tag} separate #{}: {}",
                        s.separates,
                        bad.join("; ")
                    ));
                }
                i += 5;
            }
            "PG" => {
                pool.as_mut().unwrap().perform_aging(&mut obs);
                s.agings += 1;
                i += 1;
            }
            "PR" => {
                assert_eq!(t[2], "0");
                pool.as_mut().unwrap().lp_cut_removed(t[1].parse().unwrap());
                s.removed += 1;
                i += 1;
            }
            "PZ" => {
                assert_eq!(t[2], "0");
                let c: i32 = t[1].parse().unwrap();
                let p = pool.as_mut().unwrap();
                if (c as usize) < p.num_slots() {
                    p.reset_age(c);
                } else {
                    s.mismatches
                        .push(format!("{tag} resetAge of unknown cut {c}"));
                }
                s.resets += 1;
                i += 1;
            }
            _ => i += 1,
        }
    }
}

#[test]
#[ignore]
fn full_pool_replay() {
    let p = std::env::var("DISCOPT_HIGHS_POOL_REPLAY").expect("set DISCOPT_HIGHS_POOL_REPLAY");
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
    let mut s = PoolStats::default();
    for f in &files {
        let text = std::fs::read_to_string(f).unwrap();
        let before = s.mismatches.len();
        let a0 = s.adds;
        replay(&text, &f.file_name().unwrap().to_string_lossy(), &mut s);
        eprintln!(
            "{}: adds {} mismatches {}",
            f.file_name().unwrap().to_string_lossy(),
            s.adds - a0,
            s.mismatches.len() - before
        );
    }
    for m in s.mismatches.iter().take(30) {
        eprintln!("  {m}");
    }
    eprintln!(
        "TOTAL adds {} accepted {} duplicates {} separates {} selected {} agings {} removed {} resets {} mismatches {}",
        s.adds, s.accepted, s.duplicates, s.separates, s.selected, s.agings, s.removed, s.resets, s.mismatches.len()
    );
    assert!(s.adds > 0 && s.separates > 0);
    assert!(s.mismatches.is_empty());
}

#[test]
fn cut_pool_fixture_replays_bit_identically() {
    // Root (presolve off) pool histories of egout, gr4x6, flugpl, bell5, issue-2173.
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/highs_cutpool_replay.txt"
    );
    let text = std::fs::read_to_string(path).expect("cut-pool fixture");
    let mut s = PoolStats::default();
    replay(&text, "fixture", &mut s);
    // §6: every operation kind must be exercised.
    assert!(
        s.adds >= 1500 && s.duplicates >= 500 && s.accepted >= 500,
        "{s:?}"
    );
    assert!(s.separates >= 40 && s.selected >= 100, "{s:?}");
    assert!(s.agings >= 10 && s.removed >= 20 && s.resets >= 20, "{s:?}");
    assert!(s.mismatches.is_empty(), "{:#?}", s.mismatches);
}
