//! B&B tree node representation.

/// Unique identifier for a node in the B&B tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

/// Status of a node in the B&B tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeStatus {
    /// Not yet evaluated (relaxation not solved).
    Pending,
    /// Relaxation solved, waiting for processing (prune/branch/fathom).
    Evaluated,
    /// Children created from this node.
    Branched,
    /// Dominated or infeasible — pruned from search.
    Pruned,
    /// Integer-feasible solution found at this node.
    Fathomed,
}

/// A single node in the B&B search tree.
#[derive(Debug, Clone)]
pub struct Node {
    /// Unique identifier.
    pub id: NodeId,
    /// Parent node (None for root).
    pub parent: Option<NodeId>,
    /// Depth in the tree (root = 0).
    pub depth: usize,
    /// Variable lower bounds at this node.
    pub lb: Vec<f64>,
    /// Variable upper bounds at this node.
    pub ub: Vec<f64>,
    /// Relaxation objective bound at this node.
    pub local_lower_bound: f64,
    /// Current status.
    pub status: NodeStatus,
    /// Parent node's NLP solution, used as warm-start for child solves.
    pub parent_solution: Option<Vec<f64>>,
}

impl Node {
    /// Create a new pending node.
    pub fn new(
        id: NodeId,
        parent: Option<NodeId>,
        depth: usize,
        lb: Vec<f64>,
        ub: Vec<f64>,
    ) -> Self {
        Self {
            id,
            parent,
            depth,
            lb,
            ub,
            local_lower_bound: f64::NEG_INFINITY,
            status: NodeStatus::Pending,
            parent_solution: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let node = Node::new(NodeId(0), None, 0, vec![0.0, 0.0], vec![1.0, 1.0]);
        assert_eq!(node.id, NodeId(0));
        assert_eq!(node.parent, None);
        assert_eq!(node.depth, 0);
        assert_eq!(node.lb, vec![0.0, 0.0]);
        assert_eq!(node.ub, vec![1.0, 1.0]);
        assert_eq!(node.local_lower_bound, f64::NEG_INFINITY);
        assert_eq!(node.status, NodeStatus::Pending);
        assert!(node.parent_solution.is_none());
    }

    #[test]
    fn test_node_parent_solution() {
        let mut node = Node::new(NodeId(0), None, 0, vec![0.0], vec![1.0]);
        assert!(node.parent_solution.is_none());
        node.parent_solution = Some(vec![0.5]);
        assert_eq!(node.parent_solution.as_ref().unwrap(), &vec![0.5]);
    }

    #[test]
    fn test_node_with_parent() {
        let child = Node::new(
            NodeId(1),
            Some(NodeId(0)),
            1,
            vec![0.0, 0.0],
            vec![0.5, 1.0],
        );
        assert_eq!(child.parent, Some(NodeId(0)));
        assert_eq!(child.depth, 1);
    }

    #[test]
    fn test_node_status_transitions() {
        let mut node = Node::new(NodeId(0), None, 0, vec![], vec![]);
        assert_eq!(node.status, NodeStatus::Pending);

        node.status = NodeStatus::Evaluated;
        assert_eq!(node.status, NodeStatus::Evaluated);

        node.status = NodeStatus::Branched;
        assert_eq!(node.status, NodeStatus::Branched);
    }

    #[test]
    fn test_node_id_equality() {
        assert_eq!(NodeId(42), NodeId(42));
        assert_ne!(NodeId(1), NodeId(2));
    }

    #[test]
    fn test_node_clone() {
        let node = Node::new(NodeId(0), None, 0, vec![1.0, 2.0], vec![3.0, 4.0]);
        let cloned = node.clone();
        assert_eq!(cloned.id, node.id);
        assert_eq!(cloned.lb, node.lb);
        assert_eq!(cloned.ub, node.ub);
    }
}
