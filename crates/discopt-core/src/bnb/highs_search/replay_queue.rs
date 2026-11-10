//! Replay of HiGHS node-queue dumps (`HS_DUMP_QUEUE`, written by
//! `docs/dev/data/highs-parity/search/hs_dump_queue.patch`): every operation applied to
//! the port, every recorded state compared exactly.
//!
//! ```text
//! QN <tag> <numcol>                         setNumCol
//! QE <tag> <slot> <lb> <est> <depth> <optlimit> <n> (<col> <0=lower|1=upper> <val>)* <nb> (<pos>)*
//! QP <tag> E|B <slot>                       popBestNode | popBestBoundNode
//! QB <tag> <upper_limit> <optlimit>         performBounding
//! QG <tag> <n> <slot>*                      nodes checkGlobalBounds deleted
//! QC <tag>                                  clear
//! QS <tag> <active> <lowerMin> <hybridEstimMin> <numSuboptimal> <optlimit>
//! ```

use std::collections::HashMap;

use super::node_queue::{DomainChange, NodeQueue};

fn hx(t: &str) -> f64 {
    if t == "inf" {
        return f64::INFINITY;
    }
    if t == "-inf" {
        return f64::NEG_INFINITY;
    }
    crate::lp::highs_cutgen::replay::parse_hex_f64(t)
}

/// Replay counts.
#[derive(Debug, Default)]
pub(crate) struct QueueStats {
    pub ops: usize,
    pub states: usize,
    pub pops: usize,
    pub queues: usize,
    pub mismatches: Vec<String>,
}

pub(crate) fn replay(text: &str, s: &mut QueueStats) {
    let mut qs: HashMap<i32, NodeQueue> = HashMap::new();
    for (ln, line) in text.lines().enumerate() {
        let t: Vec<&str> = line.split_ascii_whitespace().collect();
        if t.len() < 2 {
            continue;
        }
        let tag: i32 = t[1].parse().unwrap();
        match t[0] {
            "QN" => {
                qs.insert(tag, NodeQueue::new(t[2].parse().unwrap()));
                s.queues += 1;
            }
            "QE" => {
                let q = qs.entry(tag).or_insert_with(|| NodeQueue::new(0));
                let slot: i64 = t[2].parse().unwrap();
                let (lb, est) = (hx(t[3]), hx(t[4]));
                let depth: i32 = t[5].parse().unwrap();
                q.set_optimality_limit(hx(t[6]));
                let n: usize = t[7].parse().unwrap();
                let mut k = 8;
                let mut d = Vec::with_capacity(n);
                for _ in 0..n {
                    d.push(DomainChange {
                        col: t[k].parse().unwrap(),
                        upper: t[k + 1] == "1",
                        val: hx(t[k + 2]),
                    });
                    k += 3;
                }
                let nb: usize = t[k].parse().unwrap();
                let b: Vec<i32> = t[k + 1..k + 1 + nb]
                    .iter()
                    .map(|v| v.parse().unwrap())
                    .collect();
                let (got, _) = q.emplace_node(d, b, lb, est, depth);
                if got != slot {
                    s.mismatches
                        .push(format!("line {}: emplace slot {got} vs {slot}", ln + 1));
                }
                s.ops += 1;
            }
            "QP" => {
                let q = qs.get_mut(&tag).expect("pop on an unknown queue");
                let want: i64 = t[3].parse().unwrap();
                let got = if t[2] == "E" {
                    q.pop_best_node().0
                } else {
                    q.pop_best_bound_node().0
                };
                if got != want {
                    s.mismatches.push(format!(
                        "line {}: pop {} slot {got} vs {want}",
                        ln + 1,
                        t[2]
                    ));
                }
                s.ops += 1;
                s.pops += 1;
            }
            "QB" => {
                let q = qs.get_mut(&tag).expect("bounding on an unknown queue");
                q.set_optimality_limit(hx(t[3]));
                q.perform_bounding(hx(t[2]));
                s.ops += 1;
            }
            "QG" => {
                let q = qs.get_mut(&tag).expect("prune on an unknown queue");
                let n: usize = t[2].parse().unwrap();
                for v in &t[3..3 + n] {
                    q.prune_node(v.parse().unwrap());
                }
                s.ops += 1;
            }
            "QC" => {
                if let Some(q) = qs.get_mut(&tag) {
                    q.clear();
                }
                s.ops += 1;
            }
            "QS" => {
                let q = qs.get(&tag).expect("state of an unknown queue");
                let want: Vec<i64> = t[2..6].iter().map(|v| v.parse().unwrap()).collect();
                let got = vec![
                    q.num_active_nodes(),
                    q.lower_min(),
                    q.hybrid_estim_min(),
                    q.num_suboptimal(),
                ];
                if got != want {
                    s.mismatches
                        .push(format!("line {}: state {got:?} vs {want:?}", ln + 1));
                }
                s.states += 1;
            }
            _ => {}
        }
    }
}

#[test]
#[ignore]
fn full_queue_replay() {
    let p = std::env::var("DISCOPT_HIGHS_QUEUE_REPLAY").expect("set DISCOPT_HIGHS_QUEUE_REPLAY");
    let text = std::fs::read_to_string(&p).unwrap();
    let mut s = QueueStats::default();
    replay(&text, &mut s);
    for m in s.mismatches.iter().take(20) {
        eprintln!("  {m}");
    }
    eprintln!(
        "queues {} ops {} pops {} states {} mismatches {}",
        s.queues,
        s.ops,
        s.pops,
        s.states,
        s.mismatches.len()
    );
    assert!(
        s.pops > 0 && s.states > 0,
        "the replay checked nothing: {s:?}"
    );
    assert!(s.mismatches.is_empty());
}
