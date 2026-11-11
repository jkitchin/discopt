//! Provenance: a Rust translation of HiGHS source (https://github.com/ERGO-Code/HiGHS,
//! Copyright (c) HiGHS, MIT License), modified for discopt. See `THIRD_PARTY_NOTICES.md`
//! at the repository root for the license text, and `lp/highs_cutgen/mod.rs` for why
//! HiGHS is ported rather than linked.
//!
//! `HighsNodeQueue` (`mip/HighsNodeQueue.{h,cpp}`): the open nodes of the search, each a
//! stack of bound changes from the root with a lower bound, an estimate and a depth.
//!
//! HiGHS keeps three red-black trees keyed by tuples with the node's slot as the last
//! component, so every key is unique and every order is total; ordered sets over the same
//! keys give the same minima and the same iteration order, which is all HiGHS reads
//! (`lowerMin`, `hybridEstimMin`, `suboptimalMin`, predecessor walks in
//! `performBounding`). Slots are reused smallest-first (`freeslots` is a min-heap), as in
//! HiGHS, so slot numbers replay exactly (`replay_queue.rs`).

use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeSet, BinaryHeap};

/// A double ordered as C++'s `operator<` orders it (no NaN in a queue key: HiGHS
/// asserts finiteness or uses `kHighsInf` deliberately).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Key(pub f64);
impl Eq for Key {}
impl PartialOrd for Key {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Key {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.partial_cmp(&o.0).expect("NaN in a node-queue key")
    }
}

/// `HighsDomainChange`: `upper` is `HighsBoundType::kUpper`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DomainChange {
    /// Column.
    pub col: i32,
    /// An upper-bound change (else lower).
    pub upper: bool,
    /// New bound.
    pub val: f64,
}

/// `HighsNodeQueue::OpenNode`.
#[derive(Clone, Debug, Default)]
pub struct OpenNode {
    /// Bound changes from the root (`domchgstack`).
    pub domchgstack: Vec<DomainChange>,
    /// Positions of the branching changes in `domchgstack` (`branchings`).
    pub branchings: Vec<i32>,
    /// Lower bound.
    pub lower_bound: f64,
    /// Estimate (`kHighsInf` marks a suboptimal node).
    pub estimate: f64,
    /// Depth.
    pub depth: i32,
}

/// `NodeLowerRbTree` key: (lower bound, bound changes, estimate, slot).
type LowerKey = (Key, i32, Key, i64);
/// `NodeHybridEstimRbTree` key: (0.5 lb + 0.5 estimate, -bound changes, slot).
type HybridKey = (Key, i32, i64);
/// `SuboptimalNodeRbTree` key: (lower bound, slot).
type SubKey = (Key, i64);

/// `HighsNodeQueue`.
#[derive(Debug, Default)]
pub struct NodeQueue {
    nodes: Vec<OpenNode>,
    freeslots: BinaryHeap<Reverse<i64>>,
    lower: BTreeSet<LowerKey>,
    hybrid: BTreeSet<HybridKey>,
    suboptimal: BTreeSet<SubKey>,
    /// Per column, (bound, slot) of nodes holding a lower / upper bound change on it.
    col_lower: Vec<BTreeSet<SubKey>>,
    col_upper: Vec<BTreeSet<SubKey>>,
    optimality_limit: f64,
    num_col: usize,
}

const INF: f64 = f64::INFINITY;

impl NodeQueue {
    /// An empty queue over `num_col` columns (`setNumCol`).
    pub fn new(num_col: usize) -> Self {
        NodeQueue {
            col_lower: vec![BTreeSet::new(); num_col],
            col_upper: vec![BTreeSet::new(); num_col],
            optimality_limit: INF,
            num_col,
            ..Default::default()
        }
    }

    fn lower_key(&self, n: i64) -> LowerKey {
        let x = &self.nodes[n as usize];
        (
            Key(x.lower_bound),
            x.domchgstack.len() as i32,
            Key(x.estimate),
            n,
        )
    }
    fn hybrid_key(&self, n: i64) -> HybridKey {
        let x = &self.nodes[n as usize];
        (
            Key(0.5 * x.lower_bound + 0.5 * x.estimate),
            -(x.domchgstack.len() as i32),
            n,
        )
    }
    fn sub_key(&self, n: i64) -> SubKey {
        (Key(self.nodes[n as usize].lower_bound), n)
    }

    fn link_domchgs(&mut self, n: i64) {
        for i in 0..self.nodes[n as usize].domchgstack.len() {
            let d = self.nodes[n as usize].domchgstack[i];
            let set = if d.upper {
                &mut self.col_upper[d.col as usize]
            } else {
                &mut self.col_lower[d.col as usize]
            };
            // `NodeSet::emplace`: a duplicate (same bound twice for one column in one
            // stack) is not inserted twice, and erasing it later removes it once.
            set.insert((Key(d.val), n));
        }
    }
    fn unlink_domchgs(&mut self, n: i64) {
        for i in 0..self.nodes[n as usize].domchgstack.len() {
            let d = self.nodes[n as usize].domchgstack[i];
            let set = if d.upper {
                &mut self.col_upper[d.col as usize]
            } else {
                &mut self.col_lower[d.col as usize]
            };
            set.remove(&(Key(d.val), n));
        }
    }

    /// `link`: returns the tree weight a node above the optimality limit adds.
    fn link(&mut self, n: i64) -> f64 {
        if self.nodes[n as usize].lower_bound > self.optimality_limit {
            self.nodes[n as usize].estimate = INF;
            let k = self.sub_key(n);
            self.suboptimal.insert(k);
            self.link_domchgs(n);
            return 2f64.powi(1 - self.nodes[n as usize].depth);
        }
        let (h, l) = (self.hybrid_key(n), self.lower_key(n));
        self.hybrid.insert(h);
        self.lower.insert(l);
        self.link_domchgs(n);
        0.0
    }

    /// `unlink`: the slot becomes free.
    fn unlink(&mut self, n: i64) {
        if self.nodes[n as usize].estimate == INF {
            let k = self.sub_key(n);
            assert!(
                self.suboptimal.remove(&k),
                "unlink: slot {n} not suboptimal"
            );
        } else {
            let (h, l) = (self.hybrid_key(n), self.lower_key(n));
            assert!(
                self.hybrid.remove(&h),
                "unlink: slot {n} not in the estimate tree"
            );
            assert!(
                self.lower.remove(&l),
                "unlink: slot {n} not in the bound tree"
            );
        }
        self.unlink_domchgs(n);
        self.freeslots.push(Reverse(n));
    }

    /// `emplaceNode`: the slot used and the tree weight (non-zero when the node is
    /// already above the optimality limit).
    pub fn emplace_node(
        &mut self,
        domchgs: Vec<DomainChange>,
        branchings: Vec<i32>,
        lower_bound: f64,
        estimate: f64,
        depth: i32,
    ) -> (i64, f64) {
        assert!(estimate != INF, "emplace_node: estimate is +inf");
        let node = OpenNode {
            domchgstack: domchgs,
            branchings,
            lower_bound,
            estimate,
            depth,
        };
        let pos = match self.freeslots.pop() {
            Some(Reverse(p)) => {
                self.nodes[p as usize] = node;
                p
            }
            None => {
                self.nodes.push(node);
                (self.nodes.len() - 1) as i64
            }
        };
        (pos, self.link(pos))
    }

    /// `popBestNode`: the node with the smallest hybrid estimate.
    pub fn pop_best_node(&mut self) -> (i64, OpenNode) {
        let n = self
            .hybrid
            .first()
            .expect("pop_best_node on an empty queue")
            .2;
        self.unlink(n);
        (n, std::mem::take(&mut self.nodes[n as usize]))
    }

    /// `popBestBoundNode`: the node with the smallest lower bound.
    pub fn pop_best_bound_node(&mut self) -> (i64, OpenNode) {
        let n = self
            .lower
            .first()
            .expect("pop_best_bound_node on an empty queue")
            .3;
        self.unlink(n);
        (n, std::mem::take(&mut self.nodes[n as usize]))
    }

    /// `pruneNode`.
    pub fn prune_node(&mut self, n: i64) -> f64 {
        let w = if self.nodes[n as usize].estimate != INF {
            2f64.powi(1 - self.nodes[n as usize].depth)
        } else {
            0.0
        };
        self.unlink(n);
        w
    }

    /// `performBounding`: drop nodes at or above `upper_limit`, and move nodes at or
    /// above the optimality limit to the suboptimal set.
    pub fn perform_bounding(&mut self, upper_limit: f64) -> f64 {
        if self.lower.is_empty() {
            return 0.0;
        }
        let mut weight = 0.0f64;
        // Walk the bound tree from its largest key down (`predecessor` walk).
        let keys: Vec<LowerKey> = self.lower.iter().rev().copied().collect();
        let mut idx = 0usize;
        while idx < keys.len() {
            let n = keys[idx].3;
            if self.nodes[n as usize].lower_bound < upper_limit {
                break;
            }
            weight += self.prune_node(n);
            idx += 1;
        }
        if self.optimality_limit < upper_limit {
            while idx < keys.len() {
                let n = keys[idx].3;
                if self.nodes[n as usize].lower_bound < self.optimality_limit {
                    break;
                }
                let (h, l) = (self.hybrid_key(n), self.lower_key(n));
                self.hybrid.remove(&h);
                self.lower.remove(&l);
                weight += 2f64.powi(1 - self.nodes[n as usize].depth);
                self.nodes[n as usize].estimate = INF;
                let k = self.sub_key(n);
                self.suboptimal.insert(k);
                idx += 1;
            }
        }
        if !self.suboptimal.is_empty() {
            let subs: Vec<SubKey> = self.suboptimal.iter().rev().copied().collect();
            for (lb, n) in subs {
                if lb.0 < upper_limit {
                    break;
                }
                self.unlink(n);
            }
        }
        weight
    }

    /// `checkGlobalBounds`: drop nodes whose bound changes leave the global box
    /// `[lb, ub]` of `col` (beyond `feastol`). Returns the slots dropped, ascending (the
    /// order HiGHS's `std::set<int64_t> delnodes` unlinks them in), and the tree weight.
    pub fn check_global_bounds(
        &mut self,
        col: usize,
        lb: f64,
        ub: f64,
        feastol: f64,
    ) -> (Vec<i64>, f64) {
        let mut del: BTreeSet<i64> = BTreeSet::new();
        // `colLowerNodes[col].lower_bound({ub + feastol, -1})` to the end.
        for &(v, n) in self.col_lower[col].range((Key(ub + feastol), i64::MIN)..) {
            let _ = v;
            del.insert(n);
        }
        // `colUpperNodes[col]` from the start to `upper_bound({lb - feastol, kHighsIInf})`.
        for &(v, n) in self.col_upper[col].iter() {
            if v.0 > lb - feastol {
                break;
            }
            del.insert(n);
        }
        let mut weight = 0.0f64;
        for &n in &del {
            if self.nodes[n as usize].estimate != INF {
                weight += 2f64.powi(1 - self.nodes[n as usize].depth);
            }
            self.unlink(n);
        }
        (del.into_iter().collect(), weight)
    }

    /// `pruneInfeasibleNodes` without the domain: for every open node set of a column
    /// that covers all active nodes, the bound the global domain may take
    /// (`(col, Some(new lower), Some(new upper))`); the caller applies them to its
    /// global domain, propagates, and calls `check_global_bounds` again.
    pub fn implied_global_bounds(&self) -> Vec<(usize, Option<f64>, Option<f64>)> {
        let open = self.num_nodes();
        let mut out = Vec::new();
        if open == 0 {
            return out;
        }
        for c in 0..self.num_col {
            let lo = (self.col_lower[c].len() as i64 == open)
                .then(|| self.col_lower[c].first().unwrap().0 .0);
            let up = (self.col_upper[c].len() as i64 == open)
                .then(|| self.col_upper[c].last().unwrap().0 .0);
            if lo.is_some() || up.is_some() {
                out.push((c, lo, up));
            }
        }
        out
    }

    /// `numNodes` (active and suboptimal).
    pub fn num_nodes(&self) -> i64 {
        (self.nodes.len() - self.freeslots.len()) as i64
    }
    /// `numActiveNodes`.
    pub fn num_active_nodes(&self) -> i64 {
        self.num_nodes() - self.suboptimal.len() as i64
    }
    /// `empty`.
    pub fn is_empty(&self) -> bool {
        self.num_active_nodes() == 0
    }
    /// `lowerMin` (-1 when empty).
    pub fn lower_min(&self) -> i64 {
        self.lower.first().map_or(-1, |k| k.3)
    }
    /// `hybridEstimMin` (-1 when empty).
    pub fn hybrid_estim_min(&self) -> i64 {
        self.hybrid.first().map_or(-1, |k| k.2)
    }
    /// `numSuboptimal`.
    pub fn num_suboptimal(&self) -> i64 {
        self.suboptimal.len() as i64
    }
    /// `setOptimalityLimit`.
    pub fn set_optimality_limit(&mut self, v: f64) {
        self.optimality_limit = v;
    }
    /// The optimality limit.
    pub fn optimality_limit(&self) -> f64 {
        self.optimality_limit
    }
    /// `getBestLowerBound`.
    pub fn best_lower_bound(&self) -> f64 {
        let lb = self.lower.first().map_or(INF, |k| k.0 .0);
        match self.suboptimal.first() {
            None => lb,
            Some(k) => k.0 .0.min(lb),
        }
    }
    /// `getBestBoundDomchgStackSize` (`i32::MAX` for `kHighsIInf` when empty).
    pub fn best_bound_domchg_stack_size(&self) -> i32 {
        let a = self.lower.first().map_or(i32::MAX, |k| k.1);
        match self.suboptimal.first() {
            None => a,
            Some(k) => (self.nodes[k.1 as usize].domchgstack.len() as i32).min(a),
        }
    }
    /// `numNodesUp(col)`: open nodes with a lower-bound change on `col`.
    pub fn num_nodes_up(&self, col: usize) -> i64 {
        self.col_lower[col].len() as i64
    }
    /// `numNodesDown(col)`.
    pub fn num_nodes_down(&self, col: usize) -> i64 {
        self.col_upper[col].len() as i64
    }
    /// `clear`: as HiGHS's (the optimality limit resets to +inf).
    pub fn clear(&mut self) {
        *self = NodeQueue::new(self.num_col);
    }
    /// The node in a slot (for the search and the replay).
    pub fn node(&self, n: i64) -> &OpenNode {
        &self.nodes[n as usize]
    }
}
