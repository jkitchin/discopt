//! B&B tree node representation.

use std::sync::Arc;

use crate::lp::basis::Basis;

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

/// The branching decision that created a node (#2043): `x[var] <= value` on the
/// down side, `x[var] >= value` on the up side. Recorded at child creation, so a
/// later bound tightening of the child's box does not change it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BranchRecord {
    /// Column index of the branching variable.
    pub var: usize,
    /// `false` for the down child (`x[var] <= value`), `true` for the up child.
    pub up: bool,
    /// The bound the branch imposed on `x[var]`.
    pub value: f64,
}

/// Where a node sits in the tree and how it got there (#2043): read-only
/// bookkeeping for the debugger, never consulted by the search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeLineage {
    /// The node.
    pub id: NodeId,
    /// Its parent (`None` for the root).
    pub parent: Option<NodeId>,
    /// Depth in the tree (root = 0).
    pub depth: usize,
    /// The branching decision that created it (`None` for the root).
    pub branch: Option<BranchRecord>,
    /// Its status now (pending, evaluated, branched, pruned or fathomed).
    pub status: NodeStatus,
}

impl NodeStatus {
    /// Lower-case name, as the debugger reports it (#2043).
    pub fn as_str(self) -> &'static str {
        match self {
            NodeStatus::Pending => "pending",
            NodeStatus::Evaluated => "evaluated",
            NodeStatus::Branched => "branched",
            NodeStatus::Pruned => "pruned",
            NodeStatus::Fathomed => "fathomed",
        }
    }
}

/// The box a pending child shares with its sibling (#2092): its parent's box,
/// before the branch's one-coordinate change ([`PendingBox`]).
#[derive(Debug)]
pub struct SharedBox {
    /// The parent's variable lower bounds.
    pub lb: Vec<f64>,
    /// The parent's variable upper bounds.
    pub ub: Vec<f64>,
}

/// A pending child's box in compact form (#2092): the parent box both siblings
/// share, plus the one bound the branch set. Two full copies per sibling pair
/// held the same data twice; on a long run whose gap never closes the open
/// frontier is the solve's memory. Materialized by [`Node::materialize_box`]
/// when the node leaves the frontier.
#[derive(Debug, Clone)]
pub struct PendingBox {
    /// The parent box, shared with the sibling.
    pub base: Arc<SharedBox>,
    /// The column the branch bounded.
    pub var: usize,
    /// `true`: the branch set `ub[var] = value`; `false`: `lb[var] = value`.
    pub upper: bool,
    /// The bound the branch set on `var`.
    pub value: f64,
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
    /// The branching decision that created this node (`None` for the root).
    pub branch: Option<BranchRecord>,
    /// Variable lower bounds at this node. **Empty while `pending_box` holds the
    /// box**: read a node's box through [`Self::box_owned`] or call
    /// [`Self::materialize_box`] first, never `lb`/`ub` of a pending child.
    pub lb: Vec<f64>,
    /// Variable upper bounds at this node (same caveat as `lb`).
    pub ub: Vec<f64>,
    /// A pending child's box in shared form (#2092); `None` once materialized,
    /// and for the root.
    pub pending_box: Option<PendingBox>,
    /// Relaxation objective bound at this node.
    pub local_lower_bound: f64,
    /// Best-estimate of the optimal objective in this node's subtree, used by the
    /// `BestEstimate` selection strategy. Computed from the parent's pseudocosts
    /// at branch time. `NEG_INFINITY` means "unset" — selection falls back to
    /// `local_lower_bound`, so best-estimate degrades gracefully to best-first
    /// before any pseudocost history exists.
    pub best_estimate: f64,
    /// Current status.
    pub status: NodeStatus,
    /// Parent node's NLP solution, used as warm-start for child solves.
    /// Shared by both siblings (#2092).
    pub parent_solution: Option<Arc<Vec<f64>>>,
    /// Parent node's optimal LP basis, inherited as the dual-simplex warm-start
    /// state for the MILP simplex node engine (roadmap P3). `None` for the root
    /// and for the POUNCE/IPM-driven path, which ignores it.
    /// Shared by both siblings (#2092).
    pub basis: Option<Arc<Basis>>,
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
            branch: None,
            lb,
            ub,
            pending_box: None,
            local_lower_bound: f64::NEG_INFINITY,
            best_estimate: f64::NEG_INFINITY,
            status: NodeStatus::Pending,
            parent_solution: None,
            basis: None,
        }
    }

    /// Turn a shared [`PendingBox`] into this node's own `lb`/`ub` (#2092). The
    /// last sibling to materialize takes the shared vectors without copying. A
    /// no-op on a node whose box is already materialized.
    pub fn materialize_box(&mut self) {
        if let Some(pb) = self.pending_box.take() {
            let (mut lb, mut ub) = match Arc::try_unwrap(pb.base) {
                Ok(b) => (b.lb, b.ub),
                Err(shared) => (shared.lb.clone(), shared.ub.clone()),
            };
            if pb.upper {
                ub[pb.var] = pb.value;
            } else {
                lb[pb.var] = pb.value;
            }
            self.lb = lb;
            self.ub = ub;
        }
    }

    /// This node's box as owned vectors, whether it is materialized or not.
    pub fn box_owned(&self) -> (Vec<f64>, Vec<f64>) {
        match &self.pending_box {
            None => (self.lb.clone(), self.ub.clone()),
            Some(pb) => {
                let (mut lb, mut ub) = (pb.base.lb.clone(), pb.base.ub.clone());
                if pb.upper {
                    ub[pb.var] = pb.value;
                } else {
                    lb[pb.var] = pb.value;
                }
                (lb, ub)
            }
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
        node.parent_solution = Some(Arc::new(vec![0.5]));
        assert_eq!(**node.parent_solution.as_ref().unwrap(), vec![0.5]);
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
