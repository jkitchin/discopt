//! Branching strategies for the B&B tree.
//!
//! Supports most-fractional branching (default) and pseudocost-based branching.
//! Pseudocosts track the historical per-unit bound improvement when branching
//! on each variable, enabling reliability branching (use pseudocosts when
//! reliable, fall back to strong branching otherwise).

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

// ---------------------------------------------------------------------------
// Pseudocost tracking
// ---------------------------------------------------------------------------

/// Per-variable pseudocost tracker.
///
/// Pseudocosts measure the average bound improvement per unit of variable
/// change when branching on a variable:
///   - `down_cost`: average (parent_lb - child_lb) / (frac_part) for down branches
///   - `up_cost`: average (parent_lb - child_lb) / (1 - frac_part) for up branches
///
/// where child_lb is the relaxation lower bound of the child node.
#[derive(Debug, Clone)]
pub struct Pseudocosts {
    /// Sum of per-unit bound changes for down branches, per flat variable index.
    down_sum: Vec<f64>,
    /// Count of down-branch observations.
    down_count: Vec<u32>,
    /// Sum of per-unit bound changes for up branches.
    up_sum: Vec<f64>,
    /// Count of up-branch observations.
    up_count: Vec<u32>,
    /// Default pseudocost for variables with no observations.
    default_cost: f64,
}

impl Pseudocosts {
    /// Create a new pseudocost tracker for `n_vars` variables.
    pub fn new(n_vars: usize) -> Self {
        Self {
            down_sum: vec![0.0; n_vars],
            down_count: vec![0; n_vars],
            up_sum: vec![0.0; n_vars],
            up_count: vec![0; n_vars],
            default_cost: 1.0,
        }
    }

    /// Record a pseudocost observation.
    ///
    /// - `var_index`: which variable was branched on.
    /// - `parent_lb`: parent node's relaxation lower bound.
    /// - `child_lb`: child node's relaxation lower bound.
    /// - `frac_part`: fractional part of the variable at the parent.
    /// - `is_down`: true if this was the down branch (x <= floor).
    pub fn update(
        &mut self,
        var_index: usize,
        parent_lb: f64,
        child_lb: f64,
        frac_part: f64,
        is_down: bool,
    ) {
        if var_index >= self.down_sum.len() {
            return;
        }
        let delta_bound = child_lb - parent_lb;
        // delta_bound should be >= 0 (child is at least as tight as parent)
        let delta_bound = delta_bound.max(0.0);

        if is_down {
            let delta_var = frac_part.max(1e-10);
            self.down_sum[var_index] += delta_bound / delta_var;
            self.down_count[var_index] += 1;
        } else {
            let delta_var = (1.0 - frac_part).max(1e-10);
            self.up_sum[var_index] += delta_bound / delta_var;
            self.up_count[var_index] += 1;
        }
    }

    /// Get the average down pseudocost for a variable.
    pub fn down_cost(&self, var_index: usize) -> f64 {
        if var_index >= self.down_count.len() || self.down_count[var_index] == 0 {
            return self.default_cost;
        }
        self.down_sum[var_index] / self.down_count[var_index] as f64
    }

    /// Get the average up pseudocost for a variable.
    pub fn up_cost(&self, var_index: usize) -> f64 {
        if var_index >= self.up_count.len() || self.up_count[var_index] == 0 {
            return self.default_cost;
        }
        self.up_sum[var_index] / self.up_count[var_index] as f64
    }

    /// Total number of observations (down + up) for a variable.
    pub fn observation_count(&self, var_index: usize) -> u32 {
        if var_index >= self.down_count.len() {
            return 0;
        }
        self.down_count[var_index] + self.up_count[var_index]
    }

    /// Compute the pseudocost score for branching on a variable.
    ///
    /// Uses the product scoring rule: score = down_gain * up_gain,
    /// where gain = pseudocost * delta_var. This is the standard
    /// SCIP/Gurobi scoring heuristic.
    pub fn score(&self, var_index: usize, frac_part: f64) -> f64 {
        let d = self.down_cost(var_index) * frac_part;
        let u = self.up_cost(var_index) * (1.0 - frac_part);
        // Product score with small epsilon to avoid zero
        (1e-6 + d) * (1e-6 + u)
    }
}

// ---------------------------------------------------------------------------
// Branching variable selection
// ---------------------------------------------------------------------------

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

/// Select the branching variable using pseudocost scores.
///
/// Among all integer variables with fractional values, selects the one with
/// the highest pseudocost product score. Falls back to most-fractional if
/// no variable has pseudocost observations.
///
/// Returns a tuple of (decision, unreliable_candidates):
/// - `decision`: the best variable to branch on (or None).
/// - `unreliable_candidates`: flat indices of fractional variables with fewer
///   than `reliability_threshold` pseudocost observations. These are candidates
///   for strong branching.
pub fn select_branch_variable_pseudocost(
    solution: &[f64],
    variables: &[VarBranchInfo],
    pseudocosts: &Pseudocosts,
    reliability_threshold: u32,
) -> (Option<BranchDecision>, Vec<usize>) {
    let mut best: Option<BranchDecision> = None;
    let mut best_score = f64::NEG_INFINITY;
    let mut unreliable: Vec<usize> = Vec::new();

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

            if !(INTEGRALITY_TOL..=1.0 - INTEGRALITY_TOL).contains(&frac) {
                continue;
            }

            let obs = pseudocosts.observation_count(idx);
            if obs < reliability_threshold {
                unreliable.push(idx);
            }

            let score = pseudocosts.score(idx, frac);
            if score > best_score {
                best_score = score;
                best = Some(BranchDecision {
                    var_index: idx,
                    branch_point: val.floor(),
                });
            }
        }
    }

    (best, unreliable)
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

    // Warm-start: pass parent's stored solution to children.
    let parent_sol = parent.parent_solution.clone();

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
        parent_solution: parent_sol.clone(),
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
        parent_solution: parent_sol,
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

    // ----- Pseudocost tests -----

    #[test]
    fn test_pseudocosts_default_cost() {
        let pc = Pseudocosts::new(3);
        // No observations → default cost of 1.0.
        assert_eq!(pc.down_cost(0), 1.0);
        assert_eq!(pc.up_cost(0), 1.0);
        assert_eq!(pc.observation_count(0), 0);
    }

    #[test]
    fn test_pseudocosts_update_down() {
        let mut pc = Pseudocosts::new(3);
        // Branch on var 0 with frac=0.3, parent_lb=1.0, child_lb=2.0
        // delta_bound = 1.0, delta_var = 0.3, per-unit = 1.0/0.3 ≈ 3.333
        pc.update(0, 1.0, 2.0, 0.3, true);
        assert_eq!(pc.observation_count(0), 1);
        let cost = pc.down_cost(0);
        assert!((cost - 1.0 / 0.3).abs() < 1e-10);
        // Up cost should still be default.
        assert_eq!(pc.up_cost(0), 1.0);
    }

    #[test]
    fn test_pseudocosts_update_up() {
        let mut pc = Pseudocosts::new(3);
        // Branch on var 1 with frac=0.7, parent_lb=5.0, child_lb=6.5
        // delta_bound = 1.5, delta_var = 1-0.7 = 0.3, per-unit = 5.0
        pc.update(1, 5.0, 6.5, 0.7, false);
        assert_eq!(pc.observation_count(1), 1);
        let cost = pc.up_cost(1);
        assert!((cost - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_pseudocosts_average_multiple_observations() {
        let mut pc = Pseudocosts::new(2);
        // Two down observations on var 0:
        // obs1: delta_bound=2.0, frac=0.5 → per-unit = 4.0
        // obs2: delta_bound=1.0, frac=0.5 → per-unit = 2.0
        // average = 3.0
        pc.update(0, 0.0, 2.0, 0.5, true);
        pc.update(0, 1.0, 2.0, 0.5, true);
        assert_eq!(pc.observation_count(0), 2);
        assert!((pc.down_cost(0) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_pseudocosts_negative_delta_clamped() {
        let mut pc = Pseudocosts::new(2);
        // child_lb < parent_lb → delta clamped to 0.
        pc.update(0, 5.0, 3.0, 0.5, true);
        assert!((pc.down_cost(0)).abs() < 1e-10);
    }

    #[test]
    fn test_pseudocosts_score_product() {
        let mut pc = Pseudocosts::new(2);
        // Set known costs: down=2.0, up=3.0 for var 0.
        pc.update(0, 0.0, 1.0, 0.5, true); // per-unit = 2.0
        pc.update(0, 0.0, 1.5, 0.5, false); // per-unit = 3.0
        let score = pc.score(0, 0.5);
        // d = 2.0 * 0.5 = 1.0, u = 3.0 * 0.5 = 1.5
        // score = (1e-6 + 1.0) * (1e-6 + 1.5) ≈ 1.5
        assert!((score - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_pseudocosts_out_of_range() {
        let pc = Pseudocosts::new(2);
        // Out-of-range variable index → default.
        assert_eq!(pc.down_cost(99), 1.0);
        assert_eq!(pc.up_cost(99), 1.0);
        assert_eq!(pc.observation_count(99), 0);
    }

    #[test]
    fn test_select_branch_variable_pseudocost_basic() {
        let solution = vec![0.3, 0.7, 0.5];
        let vars = vec![VarBranchInfo {
            offset: 0,
            size: 3,
            is_integer: true,
        }];
        let mut pc = Pseudocosts::new(3);
        // Give var 2 a high pseudocost.
        pc.update(2, 0.0, 10.0, 0.5, true); // per-unit = 20.0
        pc.update(2, 0.0, 10.0, 0.5, false); // per-unit = 20.0

        let (decision, _unreliable) =
            select_branch_variable_pseudocost(&solution, &vars, &pc, 8);
        let d = decision.unwrap();
        assert_eq!(d.var_index, 2); // var 2 has highest score
        assert_eq!(d.branch_point, 0.0);
    }

    #[test]
    fn test_select_branch_variable_pseudocost_unreliable() {
        let solution = vec![0.5, 0.5];
        let vars = vec![VarBranchInfo {
            offset: 0,
            size: 2,
            is_integer: true,
        }];
        let pc = Pseudocosts::new(2); // no observations
        let (_decision, unreliable) =
            select_branch_variable_pseudocost(&solution, &vars, &pc, 8);
        // Both variables have 0 observations < 8 threshold → unreliable.
        assert_eq!(unreliable.len(), 2);
        assert!(unreliable.contains(&0));
        assert!(unreliable.contains(&1));
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
