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
}

/// Entry in the priority queue, wrapping a NodeId with ordering metadata.
#[derive(Debug)]
struct HeapEntry {
    node_id: NodeId,
    lower_bound: f64,
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
        }
    }
}

/// Pool of B&B tree nodes with a priority queue for open node selection.
pub struct NodePool {
    /// All nodes ever created (indexed by NodeId).
    nodes: Vec<Node>,
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
            self.open.push(HeapEntry {
                node_id: id,
                lower_bound: node.local_lower_bound,
                depth: node.depth,
                strategy: self.strategy,
            });
        }
        self.nodes.push(node);
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

    /// Mark a node as pruned.
    pub fn prune(&mut self, id: NodeId) {
        self.nodes[id.0].status = NodeStatus::Pruned;
        // Lazy deletion: stale entry will be skipped in select_next.
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
    fn test_best_first_tiebreak_by_depth() {
        let mut pool = NodePool::new(SelectionStrategy::BestFirst);
        pool.add(make_pending_node(0, 0, 5.0)); // shallow
        pool.add(make_pending_node(1, 3, 5.0)); // deep, same bound

        // Same lower bound: prefer deeper node.
        assert_eq!(pool.select_next(), Some(NodeId(1)));
        assert_eq!(pool.select_next(), Some(NodeId(0)));
    }
}
