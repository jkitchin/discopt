//! Branching strategies for the B&B tree.

use crate::bnb::node::{Node, NodeId, NodeStatus};

/// Minimal variable info for branching decisions.
///
/// Decoupled from expr module (built separately). Each entry describes one
/// variable group: its flat offset in the variable vector, size (1 for scalars),
/// and whether it requires integrality.
#[derive(Debug, Clone)]
pub struct VarBranchInfo {
    /// Flat offset into the variable vector.
    pub offset: usize,
    /// Number of elements (1 for scalar, >1 for vector variables).
    pub size: usize,
    /// True for Binary and Integer variable types.
    pub is_integer: bool,
}

/// A decision about which variable to branch on and the split point.
#[derive(Debug, Clone)]
pub struct BranchDecision {
    /// Flat index into the variable vector.
    pub var_index: usize,
    /// Value to branch on (typically floor of fractional value for integers).
    pub branch_point: f64,
}

/// Integrality tolerance: values within this distance of an integer are
/// considered integral.
const INTEGRALITY_TOL: f64 = 1e-5;

/// Select the most-fractional variable for branching.
///
/// Among all integer variables with fractional values, selects the one whose
/// fractional part is closest to 0.5 (i.e., most ambiguous / most fractional).
/// Returns `None` if all integer variables are at integral values.
pub fn select_branch_variable(
    solution: &[f64],
    variables: &[VarBranchInfo],
) -> Option<BranchDecision> {
    let mut best: Option<BranchDecision> = None;
    let mut best_fractionality = f64::NEG_INFINITY;

    for var in variables {
        if !var.is_integer {
            continue;
        }
        for i in 0..var.size {
            let idx = var.offset + i;
            if idx >= solution.len() {
                continue;
            }
            let val = solution[idx];
            let frac = val - val.floor();

            // Skip if effectively integral.
            if !(INTEGRALITY_TOL..=1.0 - INTEGRALITY_TOL).contains(&frac) {
                continue;
            }

            // Fractionality metric: closeness to 0.5 (higher = more fractional).
            // We use 0.5 - |frac - 0.5|, which is maximized when frac == 0.5.
            let score = 0.5 - (frac - 0.5).abs();

            if score > best_fractionality {
                best_fractionality = score;
                best = Some(BranchDecision {
                    var_index: idx,
                    branch_point: val.floor(),
                });
            }
        }
    }

    best
}

/// Check if a solution is integer-feasible for all integer variables.
pub fn is_integer_feasible(solution: &[f64], variables: &[VarBranchInfo]) -> bool {
    for var in variables {
        if !var.is_integer {
            continue;
        }
        for i in 0..var.size {
            let idx = var.offset + i;
            if idx >= solution.len() {
                return false;
            }
            let val = solution[idx];
            let frac = val - val.floor();
            if frac > INTEGRALITY_TOL && frac < 1.0 - INTEGRALITY_TOL {
                return false;
            }
        }
    }
    true
}

/// Create two child nodes from a parent node and a branch decision.
///
/// Left child: upper bound on branch variable tightened to floor(value).
/// Right child: lower bound on branch variable tightened to ceil(value).
///
/// The `next_id` closure is called twice to assign IDs to the children.
pub fn create_children(
    parent: &Node,
    decision: &BranchDecision,
    mut next_id: impl FnMut() -> NodeId,
) -> (Node, Node) {
    let idx = decision.var_index;
    let bp = decision.branch_point;

    // Left child: x_i <= floor(val)
    let mut left_ub = parent.ub.clone();
    left_ub[idx] = bp; // bp is already floor(val)
    let left = Node {
        id: next_id(),
        parent: Some(parent.id),
        depth: parent.depth + 1,
        lb: parent.lb.clone(),
        ub: left_ub,
        local_lower_bound: f64::NEG_INFINITY,
        status: NodeStatus::Pending,
    };

    // Right child: x_i >= ceil(val)
    let mut right_lb = parent.lb.clone();
    right_lb[idx] = bp + 1.0; // ceil(val) = floor(val) + 1
    let right = Node {
        id: next_id(),
        parent: Some(parent.id),
        depth: parent.depth + 1,
        lb: right_lb,
        ub: parent.ub.clone(),
        local_lower_bound: f64::NEG_INFINITY,
        status: NodeStatus::Pending,
    };

    (left, right)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_most_fractional_selects_closest_to_half() {
        // x0=0.3 (frac=0.3), x1=0.5 (frac=0.5), x2=0.9 (frac=0.9)
        let solution = vec![0.3, 0.5, 0.9];
        let vars = vec![VarBranchInfo {
            offset: 0,
            size: 3,
            is_integer: true,
        }];
        let decision = select_branch_variable(&solution, &vars).unwrap();
        assert_eq!(decision.var_index, 1); // x1=0.5 is most fractional
        assert_eq!(decision.branch_point, 0.0); // floor(0.5) = 0
    }

    #[test]
    fn test_no_branch_when_all_integral() {
        let solution = vec![1.0, 2.0, 3.0];
        let vars = vec![VarBranchInfo {
            offset: 0,
            size: 3,
            is_integer: true,
        }];
        assert!(select_branch_variable(&solution, &vars).is_none());
    }

    #[test]
    fn test_no_branch_on_continuous() {
        let solution = vec![0.5, 0.5];
        let vars = vec![VarBranchInfo {
            offset: 0,
            size: 2,
            is_integer: false, // continuous, should be skipped
        }];
        assert!(select_branch_variable(&solution, &vars).is_none());
    }

    #[test]
    fn test_mixed_integer_continuous() {
        // x0 continuous=0.5, x1 integer=0.7
        let solution = vec![0.5, 0.7];
        let vars = vec![
            VarBranchInfo { offset: 0, size: 1, is_integer: false },
            VarBranchInfo { offset: 1, size: 1, is_integer: true },
        ];
        let decision = select_branch_variable(&solution, &vars).unwrap();
        assert_eq!(decision.var_index, 1); // only x1 is integer
        assert_eq!(decision.branch_point, 0.0); // floor(0.7)
    }

    #[test]
    fn test_near_integral_skipped() {
        let solution = vec![1.0 + 1e-7]; // nearly integral
        let vars = vec![VarBranchInfo {
            offset: 0,
            size: 1,
            is_integer: true,
        }];
        assert!(select_branch_variable(&solution, &vars).is_none());
    }

    #[test]
    fn test_is_integer_feasible_true() {
        let solution = vec![1.0, 2.0, 0.0];
        let vars = vec![VarBranchInfo {
            offset: 0,
            size: 3,
            is_integer: true,
        }];
        assert!(is_integer_feasible(&solution, &vars));
    }

    #[test]
    fn test_is_integer_feasible_false() {
        let solution = vec![1.0, 2.5, 0.0];
        let vars = vec![VarBranchInfo {
            offset: 0,
            size: 3,
            is_integer: true,
        }];
        assert!(!is_integer_feasible(&solution, &vars));
    }

    #[test]
    fn test_is_integer_feasible_continuous_ignored() {
        let solution = vec![1.5, 2.5];
        let vars = vec![VarBranchInfo {
            offset: 0,
            size: 2,
            is_integer: false,
        }];
        assert!(is_integer_feasible(&solution, &vars));
    }

    #[test]
    fn test_create_children_bounds() {
        let parent = Node::new(
            NodeId(0),
            None,
            0,
            vec![0.0, 0.0],
            vec![10.0, 10.0],
        );
        let decision = BranchDecision {
            var_index: 0,
            branch_point: 3.0, // branching x0 at floor(3.7) = 3
        };
        let mut counter = 1usize;
        let (left, right) = create_children(&parent, &decision, || {
            let id = NodeId(counter);
            counter += 1;
            id
        });

        // Left: x0 <= 3
        assert_eq!(left.id, NodeId(1));
        assert_eq!(left.parent, Some(NodeId(0)));
        assert_eq!(left.depth, 1);
        assert_eq!(left.lb, vec![0.0, 0.0]);
        assert_eq!(left.ub, vec![3.0, 10.0]);
        assert_eq!(left.status, NodeStatus::Pending);

        // Right: x0 >= 4
        assert_eq!(right.id, NodeId(2));
        assert_eq!(right.parent, Some(NodeId(0)));
        assert_eq!(right.depth, 1);
        assert_eq!(right.lb, vec![4.0, 0.0]);
        assert_eq!(right.ub, vec![10.0, 10.0]);
        assert_eq!(right.status, NodeStatus::Pending);
    }

    #[test]
    fn test_create_children_second_variable() {
        let parent = Node::new(
            NodeId(5),
            Some(NodeId(2)),
            3,
            vec![1.0, 2.0, 0.0],
            vec![5.0, 8.0, 1.0],
        );
        let decision = BranchDecision {
            var_index: 1,
            branch_point: 4.0,
        };
        let mut counter = 10usize;
        let (left, right) = create_children(&parent, &decision, || {
            let id = NodeId(counter);
            counter += 1;
            id
        });

        // Left: x1 <= 4
        assert_eq!(left.ub[1], 4.0);
        assert_eq!(left.lb[1], 2.0); // unchanged

        // Right: x1 >= 5
        assert_eq!(right.lb[1], 5.0);
        assert_eq!(right.ub[1], 8.0); // unchanged
    }
}
