//! Node pool with configurable selection strategy.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::bnb::node::{Node, NodeId, NodeStatus};

/// Node selection strategy for the B&B tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionStrategy {
    /// Select node with the lowest lower bound (best-first search).
    BestFirst,
    /// Select the deepest node, breaking ties by lowest lower bound (depth-first).
    DepthFirst,
    /// Select the node with the lowest best-estimate of its subtree optimum
    /// (Forrest/Achterberg best-estimate search). Finds good incumbents earlier;
    /// falls back to the lower bound when a node's estimate is unset.
    BestEstimate,
}

/// Entry in the priority queue, wrapping a NodeId with ordering metadata.
#[derive(Debug)]
struct HeapEntry {
    node_id: NodeId,
    lower_bound: f64,
    /// Best-estimate of the subtree optimum; equals `lower_bound` when the node's
    /// estimate was never set, so `BestEstimate` degrades to `BestFirst`.
    estimate: f64,
    depth: usize,
    strategy: SelectionStrategy,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.node_id == other.node_id
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.strategy {
            SelectionStrategy::BestFirst => {
                // BinaryHeap is a max-heap. We want lowest bound first,
                // so reverse the ordering on lower_bound.
                other
                    .lower_bound
                    .partial_cmp(&self.lower_bound)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| {
                        // Break ties: prefer deeper nodes.
                        self.depth.cmp(&other.depth)
                    })
                    .then_with(|| {
                        // Final tiebreak: lower NodeId for determinism.
                        other.node_id.0.cmp(&self.node_id.0)
                    })
            }
            SelectionStrategy::DepthFirst => {
                // Deepest node first (max depth).
                self.depth
                    .cmp(&other.depth)
                    .then_with(|| {
                        // Break ties: prefer lowest lower bound.
                        other
                            .lower_bound
                            .partial_cmp(&self.lower_bound)
                            .unwrap_or(Ordering::Equal)
                    })
                    .then_with(|| {
                        // Final tiebreak: higher NodeId (most recent) for LIFO behavior.
                        self.node_id.0.cmp(&other.node_id.0)
                    })
            }
            SelectionStrategy::BestEstimate => {
                // Max-heap: reverse so the lowest estimate is selected first.
                other
                    .estimate
                    .partial_cmp(&self.estimate)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| {
                        // Tie-break on the (valid) lower bound, then determinism.
                        other
                            .lower_bound
                            .partial_cmp(&self.lower_bound)
                            .unwrap_or(Ordering::Equal)
                    })
                    .then_with(|| other.node_id.0.cmp(&self.node_id.0))
            }
        }
    }
}

/// Pool of B&B tree nodes with a priority queue for open node selection.
pub struct NodePool {
    /// All nodes ever created (indexed by NodeId).
    nodes: Vec<Node>,
    /// `released[i]`: node `i` left the frontier and its per-node vectors (box,
    /// parent solution, basis) were freed by [`Self::release_storage`] (#1874).
    released: Vec<bool>,
    /// Priority queue of open (pending) nodes.
    open: BinaryHeap<HeapEntry>,
    /// Selection strategy.
    strategy: SelectionStrategy,
}

impl NodePool {
    /// Create a new empty node pool.
    pub fn new(strategy: SelectionStrategy) -> Self {
        Self {
            nodes: Vec::new(),
            released: Vec::new(),
            open: BinaryHeap::new(),
            strategy,
        }
    }

    /// Add a node to the pool. The node must have its `id` pre-assigned.
    /// If the node is Pending, it is added to the open set.
    pub fn add(&mut self, node: Node) -> NodeId {
        let id = node.id;
        assert_eq!(id.0, self.nodes.len(), "Node ID must match insertion order");
        if node.status == NodeStatus::Pending {
            // An unset estimate (NEG_INFINITY) falls back to the lower bound so
            // best-estimate ordering matches best-first until pseudocosts exist.
            let estimate = if node.best_estimate.is_finite() {
                node.best_estimate
            } else {
                node.local_lower_bound
            };
            self.open.push(HeapEntry {
                node_id: id,
                lower_bound: node.local_lower_bound,
                estimate,
                depth: node.depth,
                strategy: self.strategy,
            });
        }
        self.nodes.push(node);
        self.released.push(false);
        id
    }

    /// Select the next open node according to the strategy.
    ///
    /// Nodes that have been pruned or otherwise moved out of Pending status
    /// since they were enqueued are skipped (lazy deletion).
    pub fn select_next(&mut self) -> Option<NodeId> {
        while let Some(entry) = self.open.pop() {
            let node = &self.nodes[entry.node_id.0];
            if node.status == NodeStatus::Pending {
                return Some(entry.node_id);
            }
            // Stale entry (node was pruned/processed), skip.
        }
        None
    }

    /// Get a reference to a node by ID.
    ///
    /// # Panics
    /// Panics if the NodeId is out of range.
    pub fn get(&self, id: NodeId) -> &Node {
        &self.nodes[id.0]
    }

    /// Get a mutable reference to a node by ID.
    ///
    /// # Panics
    /// Panics if the NodeId is out of range.
    pub fn get_mut(&mut self, id: NodeId) -> &mut Node {
        &mut self.nodes[id.0]
    }

    /// Number of open (pending) nodes.
    ///
    /// This is an exact count computed by scanning the heap, since lazy
    /// deletion means some heap entries may be stale.
    pub fn open_count(&self) -> usize {
        self.open
            .iter()
            .filter(|e| self.nodes[e.node_id.0].status == NodeStatus::Pending)
            .count()
    }

    /// An upper bound on the open count in O(1): the heap's length, which also
    /// holds stale entries skipped by lazy deletion. Used for the frontier memory
    /// guard (#1874), where over-counting can only stop the search earlier.
    pub fn open_upper_bound(&self) -> usize {
        self.open.len()
    }

    /// Return an already-exported node to the open frontier so it is selected
    /// and re-evaluated again.
    ///
    /// This exists for the lazy-constraint separator (#1060). When a user
    /// separator VETOES the integer point a node's relaxation found, the node
    /// must be re-solved against the cut-augmented matrix — **not** excluded.
    /// Excluding it is the `_INFEASIBILITY_SENTINEL` path, a documented
    /// *non-rigorous* fathom (#748): it removes a box that was never proven to
    /// contain no acceptable point, so single-tree outer approximation built on
    /// it would silently drop the optimum. Re-opening keeps the box in the
    /// search; the cut that vetoed the point is globally valid and is folded
    /// into the shared matrix before the next selection, so the re-solve cannot
    /// return the same point.
    ///
    /// `select_next` consumed this node's heap entry, so a fresh one is pushed.
    /// Any stale entry left behind is skipped by the usual lazy-deletion rule
    /// once the node leaves `Pending` again.
    pub fn reopen(&mut self, id: NodeId) {
        assert!(!self.released[id.0], "reopen of a released node {:?}", id);
        let node = &mut self.nodes[id.0];
        node.status = NodeStatus::Pending;
        let lower_bound = node.local_lower_bound;
        let estimate = if node.best_estimate.is_finite() {
            node.best_estimate
        } else {
            lower_bound
        };
        let depth = node.depth;
        self.open.push(HeapEntry {
            node_id: id,
            lower_bound,
            estimate,
            depth,
            strategy: self.strategy,
        });
    }

    /// Mark a node as pruned and free its per-node vectors.
    pub fn prune(&mut self, id: NodeId) {
        self.nodes[id.0].status = NodeStatus::Pruned;
        // Lazy deletion: stale entry will be skipped in select_next.
        self.release_storage(id);
    }

    /// Free a node's box, parent solution and basis once it has left the frontier
    /// (#1874). The pool keeps every node ever created, so these vectors were the
    /// whole of its growth: measured on MINLPLib ``tln4`` (336 columns, 232 rows)
    /// ~5.5 KB per node, 645 MB at the 100k-node default reached in ~14 s, which
    /// made ``max_nodes`` a de-facto memory guard. The scalars a retired node is
    /// still read for -- status, local lower bound, depth, parent -- are kept, so
    /// the frontier minimum, the tight/unresolved floors and depth-stride gates are
    /// unaffected. Readers of a released box must refuse, not see an empty one:
    /// see [`Self::is_released`]. Releasing a node that is still open is a bug.
    pub fn release_storage(&mut self, id: NodeId) {
        let node = &mut self.nodes[id.0];
        debug_assert!(
            !matches!(node.status, NodeStatus::Pending | NodeStatus::Evaluated),
            "release_storage on a frontier node {:?}",
            id
        );
        node.lb = Vec::new();
        node.ub = Vec::new();
        node.parent_solution = None;
        node.basis = None;
        self.released[id.0] = true;
    }

    /// Whether [`Self::release_storage`] freed this node's vectors.
    pub fn is_released(&self, id: NodeId) -> bool {
        self.released[id.0]
    }

    /// Total number of nodes ever created.
    pub fn total_count(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pending_node(id: usize, depth: usize, lb: f64) -> Node {
        let mut node = Node::new(NodeId(id), None, depth, vec![], vec![]);
        node.local_lower_bound = lb;
        node
    }

    fn make_estimate_node(id: usize, lb: f64, estimate: f64) -> Node {
        let mut node = make_pending_node(id, 0, lb);
        node.best_estimate = estimate;
        node
    }

    #[test]
    fn test_best_first_selects_lowest_bound() {
        let mut pool = NodePool::new(SelectionStrategy::BestFirst);
        pool.add(make_pending_node(0, 0, 10.0));
        pool.add(make_pending_node(1, 0, 5.0));
        pool.add(make_pending_node(2, 0, 8.0));

        assert_eq!(pool.select_next(), Some(NodeId(1))); // lb=5
        assert_eq!(pool.select_next(), Some(NodeId(2))); // lb=8
        assert_eq!(pool.select_next(), Some(NodeId(0))); // lb=10
        assert_eq!(pool.select_next(), None);
    }

    #[test]
    fn test_depth_first_selects_deepest() {
        let mut pool = NodePool::new(SelectionStrategy::DepthFirst);
        pool.add(make_pending_node(0, 0, 5.0));
        pool.add(make_pending_node(1, 2, 8.0));
        pool.add(make_pending_node(2, 1, 3.0));

        assert_eq!(pool.select_next(), Some(NodeId(1))); // depth=2
        assert_eq!(pool.select_next(), Some(NodeId(2))); // depth=1
        assert_eq!(pool.select_next(), Some(NodeId(0))); // depth=0
        assert_eq!(pool.select_next(), None);
    }

    #[test]
    fn test_depth_first_lifo_on_same_depth() {
        let mut pool = NodePool::new(SelectionStrategy::DepthFirst);
        pool.add(make_pending_node(0, 1, 5.0));
        pool.add(make_pending_node(1, 1, 5.0));
        pool.add(make_pending_node(2, 1, 5.0));

        // Same depth and bound: highest ID first (LIFO).
        assert_eq!(pool.select_next(), Some(NodeId(2)));
        assert_eq!(pool.select_next(), Some(NodeId(1)));
        assert_eq!(pool.select_next(), Some(NodeId(0)));
    }

    #[test]
    fn test_prune_skips_in_selection() {
        let mut pool = NodePool::new(SelectionStrategy::BestFirst);
        pool.add(make_pending_node(0, 0, 10.0));
        pool.add(make_pending_node(1, 0, 5.0));
        pool.add(make_pending_node(2, 0, 8.0));

        pool.prune(NodeId(1)); // prune the best node
        assert_eq!(pool.select_next(), Some(NodeId(2))); // next best is lb=8
    }

    #[test]
    fn test_open_count() {
        let mut pool = NodePool::new(SelectionStrategy::BestFirst);
        pool.add(make_pending_node(0, 0, 10.0));
        pool.add(make_pending_node(1, 0, 5.0));
        assert_eq!(pool.open_count(), 2);

        pool.prune(NodeId(0));
        assert_eq!(pool.open_count(), 1);
    }

    #[test]
    fn test_empty_pool_select() {
        let mut pool = NodePool::new(SelectionStrategy::BestFirst);
        assert_eq!(pool.select_next(), None);
    }

    #[test]
    fn test_get_and_get_mut() {
        let mut pool = NodePool::new(SelectionStrategy::BestFirst);
        pool.add(make_pending_node(0, 0, 5.0));

        assert_eq!(pool.get(NodeId(0)).local_lower_bound, 5.0);

        pool.get_mut(NodeId(0)).local_lower_bound = 7.0;
        assert_eq!(pool.get(NodeId(0)).local_lower_bound, 7.0);
    }

    #[test]
    fn test_total_count() {
        let mut pool = NodePool::new(SelectionStrategy::BestFirst);
        assert_eq!(pool.total_count(), 0);
        pool.add(make_pending_node(0, 0, 1.0));
        pool.add(make_pending_node(1, 0, 2.0));
        assert_eq!(pool.total_count(), 2);
    }

    #[test]
    fn test_best_estimate_selects_lowest_estimate() {
        let mut pool = NodePool::new(SelectionStrategy::BestEstimate);
        // Lower bounds are equal; only the estimate differentiates the nodes.
        pool.add(make_estimate_node(0, 5.0, 12.0));
        pool.add(make_estimate_node(1, 5.0, 7.0));
        pool.add(make_estimate_node(2, 5.0, 9.0));

        assert_eq!(pool.select_next(), Some(NodeId(1))); // estimate 7
        assert_eq!(pool.select_next(), Some(NodeId(2))); // estimate 9
        assert_eq!(pool.select_next(), Some(NodeId(0))); // estimate 12
        assert_eq!(pool.select_next(), None);
    }

    #[test]
    fn test_best_estimate_falls_back_to_lower_bound_when_unset() {
        // With no estimate set, BestEstimate must behave like BestFirst.
        let mut pool = NodePool::new(SelectionStrategy::BestEstimate);
        pool.add(make_pending_node(0, 0, 10.0));
        pool.add(make_pending_node(1, 0, 5.0));
        pool.add(make_pending_node(2, 0, 8.0));

        assert_eq!(pool.select_next(), Some(NodeId(1))); // lb 5
        assert_eq!(pool.select_next(), Some(NodeId(2))); // lb 8
        assert_eq!(pool.select_next(), Some(NodeId(0))); // lb 10
    }

    #[test]
    fn test_best_estimate_differs_from_best_first() {
        // Node 0 has the lower bound but a worse (higher) estimate; best-estimate
        // should prefer node 1, whereas best-first would pick node 0.
        let mut pool = NodePool::new(SelectionStrategy::BestEstimate);
        pool.add(make_estimate_node(0, 1.0, 20.0));
        pool.add(make_estimate_node(1, 3.0, 6.0));
        assert_eq!(pool.select_next(), Some(NodeId(1)));
    }

    #[test]
    fn test_best_first_tiebreak_by_depth() {
        let mut pool = NodePool::new(SelectionStrategy::BestFirst);
        pool.add(make_pending_node(0, 0, 5.0)); // shallow
        pool.add(make_pending_node(1, 3, 5.0)); // deep, same bound

        // Same lower bound: prefer deeper node.
        assert_eq!(pool.select_next(), Some(NodeId(1)));
        assert_eq!(pool.select_next(), Some(NodeId(0)));
    }

    /// #1874: a retired node's vectors are freed, its scalars kept, and a frontier
    /// node is never released (reopen refuses a released one).
    #[test]
    fn release_storage_frees_vectors_and_keeps_scalars() {
        let mut pool = NodePool::new(SelectionStrategy::BestFirst);
        let mut n0 = Node::new(NodeId(0), None, 3, vec![], vec![]);
        n0.local_lower_bound = -2.5;
        n0.lb = vec![0.0; 50];
        n0.ub = vec![1.0; 50];
        n0.parent_solution = Some(vec![0.5; 50]);
        pool.add(n0);
        assert!(!pool.is_released(NodeId(0)));
        pool.prune(NodeId(0));
        assert!(pool.is_released(NodeId(0)));
        let n = pool.get(NodeId(0));
        assert!(n.lb.is_empty() && n.ub.is_empty() && n.parent_solution.is_none());
        assert_eq!(n.lb.capacity(), 0, "vector not freed");
        assert_eq!((n.depth, n.local_lower_bound), (3, -2.5));
        assert_eq!(n.status, NodeStatus::Pruned);
    }

    #[test]
    #[should_panic(expected = "reopen of a released node")]
    fn reopen_refuses_a_released_node() {
        let mut pool = NodePool::new(SelectionStrategy::BestFirst);
        pool.add(Node::new(NodeId(0), None, 0, vec![], vec![]));
        pool.prune(NodeId(0));
        pool.reopen(NodeId(0));
    }
}
