//! TreeManager — orchestrates the B&B search loop.

use std::collections::HashMap;

use crate::bnb::branching::{
    create_children, is_integer_feasible, select_branch_variable,
    select_branch_variable_pseudocost, BranchDecision, Pseudocosts, VarBranchInfo,
};
use crate::bnb::node::{Node, NodeId, NodeStatus};
use crate::bnb::pool::{NodePool, SelectionStrategy};

/// A batch of nodes exported for relaxation evaluation.
#[derive(Debug)]
pub struct ExportBatch {
    /// Lower bounds for each node: `[N][n_vars]`.
    pub lb: Vec<Vec<f64>>,
    /// Upper bounds for each node: `[N][n_vars]`.
    pub ub: Vec<Vec<f64>>,
    /// Node IDs corresponding to each row.
    pub node_ids: Vec<NodeId>,
    /// Parent NLP solutions for warm-starting: `[N][n_vars]` (None if root).
    pub parent_solutions: Vec<Option<Vec<f64>>>,
}

/// Result of evaluating a single node's relaxation.
#[derive(Debug)]
pub struct NodeResult {
    /// Which node this result belongs to.
    pub node_id: NodeId,
    /// Relaxation objective lower bound.
    pub lower_bound: f64,
    /// Relaxation solution vector.
    pub solution: Vec<f64>,
    /// Whether the relaxation solution is integer-feasible.
    pub is_feasible: bool,
}

/// Statistics from processing a batch of evaluated nodes.
#[derive(Debug, Default)]
pub struct ProcessingStats {
    /// Number of nodes pruned (LB >= incumbent).
    pub pruned: usize,
    /// Number of nodes fathomed (integer-feasible solution).
    pub fathomed: usize,
    /// Number of nodes branched (children created).
    pub branched: usize,
    /// Number of incumbent updates.
    pub incumbent_updates: usize,
    /// Flat variable indices with unreliable pseudocost estimates.
    /// These are candidates for strong branching (LP/NLP re-solves).
    pub unreliable_candidates: Vec<usize>,
}

/// Aggregate tree statistics.
#[derive(Debug)]
pub struct TreeStats {
    /// Total number of nodes created.
    pub total_nodes: usize,
    /// Number of open (pending) nodes.
    pub open_nodes: usize,
    /// Best known feasible objective value.
    pub incumbent_value: f64,
    /// Global lower bound (min over open node bounds).
    pub global_lower_bound: f64,
    /// Current optimality gap (relative).
    pub gap: f64,
}

/// Internal buffer for results waiting to be processed.
#[derive(Debug, Clone)]
struct PendingResult {
    node_id: NodeId,
    solution: Vec<f64>,
    is_feasible: bool,
}

/// Record of a branching decision at a node, used for retroactive
/// pseudocost updates when child results arrive.
#[derive(Debug, Clone)]
struct BranchRecord {
    /// Which variable was branched on.
    var_index: usize,
    /// Fractional part of the variable at the parent.
    frac_part: f64,
    /// Parent node's relaxation lower bound.
    parent_lb: f64,
}

/// Orchestrates the Branch-and-Bound search.
pub struct TreeManager {
    pool: NodePool,
    incumbent_value: f64,
    incumbent_solution: Option<Vec<f64>>,
    global_lb: Vec<f64>,
    global_ub: Vec<f64>,
    integer_vars: Vec<VarBranchInfo>,
    node_counter: usize,
    global_lower_bound: f64,
    pending_results: Vec<PendingResult>,
    /// Per-node branch hints from Python (e.g. GNN branching).
    /// Maps node ID → variable index to branch on. Consumed after use.
    branch_hints: HashMap<NodeId, usize>,
    /// Pseudocost tracker for reliability branching.
    pseudocosts: Pseudocosts,
    /// Whether pseudocost-based branching is enabled (default: true).
    use_pseudocosts: bool,
    /// Minimum observations before a variable's pseudocost is considered reliable.
    reliability_threshold: u32,
    /// Records of branching decisions: maps child NodeId → parent's BranchRecord.
    /// Used to retroactively update pseudocosts when child results arrive.
    branch_records: HashMap<NodeId, BranchRecord>,
}

impl TreeManager {
    /// Create a new TreeManager.
    ///
    /// - `n_vars`: number of decision variables.
    /// - `lb`, `ub`: global variable bounds (must have length `n_vars`).
    /// - `integer_vars`: description of integer/binary variables for branching.
    /// - `strategy`: node selection strategy.
    pub fn new(
        n_vars: usize,
        lb: Vec<f64>,
        ub: Vec<f64>,
        integer_vars: Vec<VarBranchInfo>,
        strategy: SelectionStrategy,
    ) -> Self {
        assert_eq!(lb.len(), n_vars);
        assert_eq!(ub.len(), n_vars);
        let pseudocosts = Pseudocosts::new(n_vars);
        Self {
            pool: NodePool::new(strategy),
            incumbent_value: f64::INFINITY,
            incumbent_solution: None,
            global_lb: lb,
            global_ub: ub,
            integer_vars,
            node_counter: 0,
            global_lower_bound: f64::NEG_INFINITY,
            pending_results: Vec::new(),
            branch_hints: HashMap::new(),
            pseudocosts,
            use_pseudocosts: true,
            reliability_threshold: 8,
            branch_records: HashMap::new(),
        }
    }

    /// Allocate a fresh NodeId.
    fn next_id(&mut self) -> NodeId {
        let id = NodeId(self.node_counter);
        self.node_counter += 1;
        id
    }

    /// Initialize the tree with the root node.
    pub fn initialize(&mut self) {
        let id = self.next_id();
        let root = Node::new(id, None, 0, self.global_lb.clone(), self.global_ub.clone());
        self.pool.add(root);
    }

    /// Export a batch of up to `batch_size` pending nodes for relaxation evaluation.
    ///
    /// Selected nodes are moved to `Evaluated` status (so they won't be
    /// re-selected). Returns an empty batch if no open nodes remain.
    pub fn export_batch(&mut self, batch_size: usize) -> ExportBatch {
        let mut lb = Vec::with_capacity(batch_size);
        let mut ub = Vec::with_capacity(batch_size);
        let mut node_ids = Vec::with_capacity(batch_size);
        let mut parent_solutions = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            match self.pool.select_next() {
                Some(nid) => {
                    let node = self.pool.get(nid);
                    lb.push(node.lb.clone());
                    ub.push(node.ub.clone());
                    parent_solutions.push(node.parent_solution.clone());
                    node_ids.push(nid);
                    // Mark as evaluated so it won't be selected again.
                    self.pool.get_mut(nid).status = NodeStatus::Evaluated;
                }
                None => break,
            }
        }

        ExportBatch {
            lb,
            ub,
            node_ids,
            parent_solutions,
        }
    }

    /// Import relaxation results for a batch of nodes.
    ///
    /// Updates each node's lower bound and buffers solution data for processing.
    pub fn import_results(&mut self, results: &[NodeResult]) {
        for result in results {
            let node = self.pool.get_mut(result.node_id);
            debug_assert_eq!(
                node.status,
                NodeStatus::Evaluated,
                "import_results: node {:?} not in Evaluated status",
                result.node_id
            );
            node.local_lower_bound = result.lower_bound;
            // Store solution for warm-starting children.
            node.parent_solution = Some(result.solution.clone());
        }
        self.pending_results
            .extend(results.iter().map(|r| PendingResult {
                node_id: r.node_id,
                solution: r.solution.clone(),
                is_feasible: r.is_feasible,
            }));
    }

    /// Process all evaluated nodes: prune, check integrality, branch.
    ///
    /// Uses pseudocost-based branching when enabled and sufficient observations
    /// exist. Returns stats including indices of unreliable candidates for
    /// potential strong branching by the Python orchestrator.
    pub fn process_evaluated(&mut self) -> ProcessingStats {
        let mut stats = ProcessingStats::default();

        let pending: Vec<PendingResult> = self.pending_results.drain(..).collect();

        for result in &pending {
            let node_lb = self.pool.get(result.node_id).local_lower_bound;

            // 0. Retroactively update pseudocosts from this child's result.
            if let Some(record) = self.branch_records.remove(&result.node_id) {
                // Down branch has tightened ub (ub < global_ub for the branched var).
                // Up branch has tightened lb (lb > global_lb for the branched var).
                let node = self.pool.get(result.node_id);
                let is_down = record.var_index < node.ub.len()
                    && node.ub[record.var_index]
                        < self.global_ub.get(record.var_index).copied().unwrap_or(f64::INFINITY);
                self.pseudocosts.update(
                    record.var_index,
                    record.parent_lb,
                    node_lb,
                    record.frac_part,
                    is_down,
                );
            }

            // 1. Prune if lower bound >= incumbent (node can't improve).
            if node_lb >= self.incumbent_value {
                self.pool.prune(result.node_id);
                stats.pruned += 1;
                continue;
            }

            // 2. Check if solution is integer-feasible.
            let int_feasible =
                result.is_feasible || is_integer_feasible(&result.solution, &self.integer_vars);

            if int_feasible {
                self.pool.get_mut(result.node_id).status = NodeStatus::Fathomed;
                stats.fathomed += 1;

                if node_lb < self.incumbent_value {
                    self.incumbent_value = node_lb;
                    self.incumbent_solution = Some(result.solution.clone());
                    stats.incumbent_updates += 1;
                }
                continue;
            }

            // 3. Branch: use hint if available, else pseudocost/most-fractional.
            let (decision, unreliable) = self
                .branch_hints
                .remove(&result.node_id)
                .and_then(|hint_idx| {
                    let val = result.solution.get(hint_idx)?;
                    let frac = val - val.floor();
                    if frac > 1e-5 && frac < 1.0 - 1e-5 {
                        Some((
                            Some(BranchDecision {
                                var_index: hint_idx,
                                branch_point: val.floor(),
                            }),
                            Vec::new(),
                        ))
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| {
                    if self.use_pseudocosts {
                        select_branch_variable_pseudocost(
                            &result.solution,
                            &self.integer_vars,
                            &self.pseudocosts,
                            self.reliability_threshold,
                        )
                    } else {
                        (
                            select_branch_variable(&result.solution, &self.integer_vars),
                            Vec::new(),
                        )
                    }
                });

            // Collect unreliable candidates for potential strong branching.
            stats.unreliable_candidates.extend(unreliable);

            if let Some(ref decision) = decision {
                // Record the branching decision for retroactive pseudocost updates.
                let frac_part = {
                    let val = result.solution[decision.var_index];
                    val - val.floor()
                };

                let parent = self.pool.get(result.node_id).clone();
                self.pool.get_mut(result.node_id).status = NodeStatus::Branched;

                let (left, right) = create_children(&parent, decision, || self.next_id());

                // Record branch info for both children.
                let record = BranchRecord {
                    var_index: decision.var_index,
                    frac_part,
                    parent_lb: node_lb,
                };
                self.branch_records.insert(left.id, record.clone());
                self.branch_records.insert(right.id, record);

                self.pool.add(left);
                self.pool.add(right);
                stats.branched += 1;
            } else {
                // No fractional variable found — treat as fathomed.
                self.pool.get_mut(result.node_id).status = NodeStatus::Fathomed;
                stats.fathomed += 1;
                if node_lb < self.incumbent_value {
                    self.incumbent_value = node_lb;
                    self.incumbent_solution = Some(result.solution.clone());
                    stats.incumbent_updates += 1;
                }
            }
        }

        self.update_global_lower_bound();

        stats
    }

    /// Recompute the global lower bound as the minimum over all open nodes.
    fn update_global_lower_bound(&mut self) {
        let mut min_lb = f64::INFINITY;
        for i in 0..self.pool.total_count() {
            let node = self.pool.get(NodeId(i));
            match node.status {
                NodeStatus::Pending | NodeStatus::Evaluated => {
                    if node.local_lower_bound < min_lb {
                        min_lb = node.local_lower_bound;
                    }
                }
                _ => {}
            }
        }
        if min_lb == f64::INFINITY {
            self.global_lower_bound = self.incumbent_value;
        } else {
            self.global_lower_bound = min_lb;
        }
    }

    /// Check if the search is complete.
    ///
    /// The search is finished when there are no open nodes and no pending
    /// results, or the gap is closed (within tolerance).
    pub fn is_finished(&self) -> bool {
        if self.pool.open_count() == 0 && self.pending_results.is_empty() {
            return true;
        }
        if self.incumbent_value < f64::INFINITY && self.gap() < 1e-8 {
            return true;
        }
        false
    }

    /// Current relative optimality gap.
    ///
    /// gap = (incumbent - global_lb) / max(1, |incumbent|)
    pub fn gap(&self) -> f64 {
        if self.incumbent_value >= f64::INFINITY {
            return f64::INFINITY;
        }
        if self.global_lower_bound <= f64::NEG_INFINITY {
            return f64::INFINITY;
        }
        let denom = self.incumbent_value.abs().max(1.0);
        (self.incumbent_value - self.global_lower_bound) / denom
    }

    /// Get aggregate tree statistics.
    pub fn stats(&self) -> TreeStats {
        TreeStats {
            total_nodes: self.pool.total_count(),
            open_nodes: self.pool.open_count(),
            incumbent_value: self.incumbent_value,
            global_lower_bound: self.global_lower_bound,
            gap: self.gap(),
        }
    }

    /// Get the current incumbent solution, if any.
    pub fn incumbent(&self) -> Option<(&[f64], f64)> {
        self.incumbent_solution
            .as_ref()
            .map(|sol| (sol.as_slice(), self.incumbent_value))
    }

    /// Set a branch hint for a specific node.
    ///
    /// The next call to `process_evaluated` will use this variable index
    /// instead of most-fractional branching for the given node, provided
    /// the variable is actually fractional. The hint is consumed after use.
    pub fn set_branch_hint(&mut self, node_id: NodeId, var_index: usize) {
        self.branch_hints.insert(node_id, var_index);
    }

    /// Inject an externally-found incumbent (e.g. from a primal heuristic).
    ///
    /// Updates the incumbent only if `obj_val` is strictly better than the
    /// current incumbent value. Returns `true` if the incumbent was updated.
    pub fn inject_incumbent(&mut self, solution: Vec<f64>, obj_val: f64) -> bool {
        if obj_val < self.incumbent_value {
            self.incumbent_value = obj_val;
            self.incumbent_solution = Some(solution);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_integer_vars() -> Vec<VarBranchInfo> {
        vec![VarBranchInfo {
            offset: 0,
            size: 2,
            is_integer: true,
        }]
    }

    #[test]
    fn test_tree_manager_lifecycle() {
        let mut tm = TreeManager::new(
            2,
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            simple_integer_vars(),
            SelectionStrategy::BestFirst,
        );
        tm.initialize();

        let stats = tm.stats();
        assert_eq!(stats.total_nodes, 1);
        assert_eq!(stats.open_nodes, 1);
        assert_eq!(stats.incumbent_value, f64::INFINITY);

        // Export root node.
        let batch = tm.export_batch(10);
        assert_eq!(batch.node_ids.len(), 1);
        assert_eq!(batch.lb[0], vec![0.0, 0.0]);
        assert_eq!(batch.ub[0], vec![1.0, 1.0]);

        // Simulate relaxation: solution (0.5, 0.7), obj=1.2, not integer-feasible.
        tm.import_results(&[NodeResult {
            node_id: batch.node_ids[0],
            lower_bound: 1.2,
            solution: vec![0.5, 0.7],
            is_feasible: false,
        }]);

        let proc_stats = tm.process_evaluated();
        assert_eq!(proc_stats.branched, 1);
        assert_eq!(proc_stats.pruned, 0);
        assert_eq!(proc_stats.fathomed, 0);

        // Should have 2 children now.
        let stats = tm.stats();
        assert_eq!(stats.total_nodes, 3);
        assert_eq!(stats.open_nodes, 2);
    }

    #[test]
    fn test_pruning() {
        let mut tm = TreeManager::new(
            2,
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            simple_integer_vars(),
            SelectionStrategy::BestFirst,
        );
        tm.initialize();

        let batch = tm.export_batch(1);
        tm.import_results(&[NodeResult {
            node_id: batch.node_ids[0],
            lower_bound: 5.0,
            solution: vec![1.0, 0.0],
            is_feasible: true,
        }]);
        let stats = tm.process_evaluated();
        assert_eq!(stats.fathomed, 1);
        assert_eq!(stats.incumbent_updates, 1);
        assert_eq!(tm.incumbent().unwrap().1, 5.0);
    }

    #[test]
    fn test_node_pruned_by_bound() {
        let mut tm = TreeManager::new(
            1,
            vec![0.0],
            vec![10.0],
            vec![VarBranchInfo {
                offset: 0,
                size: 1,
                is_integer: true,
            }],
            SelectionStrategy::BestFirst,
        );
        tm.initialize();

        let batch = tm.export_batch(1);
        tm.import_results(&[NodeResult {
            node_id: batch.node_ids[0],
            lower_bound: 1.0,
            solution: vec![3.5],
            is_feasible: false,
        }]);
        tm.process_evaluated();

        // Manually set incumbent to test pruning.
        tm.incumbent_value = 2.0;
        tm.incumbent_solution = Some(vec![3.0]);

        let batch = tm.export_batch(2);
        assert_eq!(batch.node_ids.len(), 2);

        tm.import_results(&[
            NodeResult {
                node_id: batch.node_ids[0],
                lower_bound: 3.0,
                solution: vec![2.5],
                is_feasible: false,
            },
            NodeResult {
                node_id: batch.node_ids[1],
                lower_bound: 5.0,
                solution: vec![5.5],
                is_feasible: false,
            },
        ]);
        let stats = tm.process_evaluated();
        assert_eq!(stats.pruned, 2);
    }

    #[test]
    fn test_gap_computation() {
        let mut tm = TreeManager::new(
            1,
            vec![0.0],
            vec![1.0],
            vec![VarBranchInfo {
                offset: 0,
                size: 1,
                is_integer: true,
            }],
            SelectionStrategy::BestFirst,
        );
        assert_eq!(tm.gap(), f64::INFINITY);

        tm.incumbent_value = 10.0;
        tm.global_lower_bound = 5.0;
        assert!((tm.gap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_empty_batch_on_empty_pool() {
        let mut tm = TreeManager::new(
            2,
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            simple_integer_vars(),
            SelectionStrategy::BestFirst,
        );
        let batch = tm.export_batch(10);
        assert!(batch.node_ids.is_empty());
        assert!(batch.lb.is_empty());
        assert!(batch.ub.is_empty());
    }

    #[test]
    fn test_is_finished_no_open_nodes() {
        let mut tm = TreeManager::new(
            1,
            vec![0.0],
            vec![1.0],
            vec![VarBranchInfo {
                offset: 0,
                size: 1,
                is_integer: true,
            }],
            SelectionStrategy::BestFirst,
        );
        tm.initialize();

        let batch = tm.export_batch(1);
        tm.import_results(&[NodeResult {
            node_id: batch.node_ids[0],
            lower_bound: 3.0,
            solution: vec![1.0],
            is_feasible: true,
        }]);
        tm.process_evaluated();

        assert!(tm.is_finished());
    }

    #[test]
    fn test_determinism() {
        let mut results = Vec::new();
        for _ in 0..3 {
            let mut tm = TreeManager::new(
                2,
                vec![0.0, 0.0],
                vec![5.0, 5.0],
                vec![VarBranchInfo {
                    offset: 0,
                    size: 2,
                    is_integer: true,
                }],
                SelectionStrategy::BestFirst,
            );
            tm.initialize();

            let batch = tm.export_batch(1);
            tm.import_results(&[NodeResult {
                node_id: batch.node_ids[0],
                lower_bound: 1.0,
                solution: vec![2.3, 1.7],
                is_feasible: false,
            }]);
            let stats = tm.process_evaluated();

            let batch2 = tm.export_batch(2);
            results.push((
                tm.stats().total_nodes,
                stats.branched,
                batch2.node_ids.clone(),
                batch2.lb.clone(),
                batch2.ub.clone(),
            ));
        }
        assert_eq!(results[0], results[1]);
        assert_eq!(results[1], results[2]);
    }

    #[test]
    fn test_incumbent_update_on_better_feasible() {
        let mut tm = TreeManager::new(
            1,
            vec![0.0],
            vec![10.0],
            vec![VarBranchInfo {
                offset: 0,
                size: 1,
                is_integer: true,
            }],
            SelectionStrategy::BestFirst,
        );
        tm.initialize();

        let batch = tm.export_batch(1);
        tm.import_results(&[NodeResult {
            node_id: batch.node_ids[0],
            lower_bound: 10.0,
            solution: vec![5.0],
            is_feasible: true,
        }]);
        tm.process_evaluated();
        assert_eq!(tm.incumbent().unwrap().1, 10.0);
    }

    #[test]
    fn test_pseudocost_updates_retroactive() {
        // Verify that pseudocosts are updated when child results arrive.
        let mut tm = TreeManager::new(
            2,
            vec![0.0, 0.0],
            vec![10.0, 10.0],
            vec![VarBranchInfo {
                offset: 0,
                size: 2,
                is_integer: true,
            }],
            SelectionStrategy::BestFirst,
        );
        tm.initialize();

        // Root: fractional at (3.5, 4.5), lb=1.0.
        let batch = tm.export_batch(1);
        tm.import_results(&[NodeResult {
            node_id: batch.node_ids[0],
            lower_bound: 1.0,
            solution: vec![3.5, 4.5],
            is_feasible: false,
        }]);
        tm.process_evaluated();
        // Root branched on the most-fractional variable (both are 0.5 from
        // integer, tie-break picks first = var 0 or var 1).

        // Export the two children.
        let batch = tm.export_batch(2);
        assert_eq!(batch.node_ids.len(), 2);

        // Simulate child results with higher LBs.
        tm.import_results(&[
            NodeResult {
                node_id: batch.node_ids[0],
                lower_bound: 3.0, // LB improved by 2.0 from parent's 1.0
                solution: vec![3.0, 4.5],
                is_feasible: false,
            },
            NodeResult {
                node_id: batch.node_ids[1],
                lower_bound: 2.5, // LB improved by 1.5 from parent's 1.0
                solution: vec![4.0, 4.5],
                is_feasible: false,
            },
        ]);
        let stats = tm.process_evaluated();
        // Both children should have updated pseudocosts (branched further).
        assert_eq!(stats.branched, 2);
        // Pseudocosts should have been updated (2 observations total).
    }

    #[test]
    fn test_unreliable_candidates_returned() {
        let mut tm = TreeManager::new(
            2,
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            simple_integer_vars(),
            SelectionStrategy::BestFirst,
        );
        tm.initialize();

        let batch = tm.export_batch(1);
        tm.import_results(&[NodeResult {
            node_id: batch.node_ids[0],
            lower_bound: 0.0,
            solution: vec![0.5, 0.5],
            is_feasible: false,
        }]);
        let stats = tm.process_evaluated();
        // With no pseudocost observations and threshold=8, both fractional
        // variables should be unreliable.
        assert!(!stats.unreliable_candidates.is_empty());
        assert!(stats.branched > 0);
    }

    #[test]
    fn test_full_solve_two_binary_vars() {
        let mut tm = TreeManager::new(
            2,
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![VarBranchInfo {
                offset: 0,
                size: 2,
                is_integer: true,
            }],
            SelectionStrategy::BestFirst,
        );
        tm.initialize();

        // Root relaxation: fractional.
        let batch = tm.export_batch(1);
        tm.import_results(&[NodeResult {
            node_id: batch.node_ids[0],
            lower_bound: 0.0,
            solution: vec![0.5, 0.5],
            is_feasible: false,
        }]);
        tm.process_evaluated();

        // Two children.
        let batch = tm.export_batch(2);
        assert_eq!(batch.node_ids.len(), 2);

        tm.import_results(&[
            NodeResult {
                node_id: batch.node_ids[0],
                lower_bound: 1.0,
                solution: vec![0.0, 0.5],
                is_feasible: false,
            },
            NodeResult {
                node_id: batch.node_ids[1],
                lower_bound: 0.5,
                solution: vec![1.0, 0.5],
                is_feasible: false,
            },
        ]);
        tm.process_evaluated();

        // Export grandchildren.
        let batch = tm.export_batch(4);
        assert!(!batch.node_ids.is_empty());

        let mut node_results = Vec::new();
        for (i, &nid) in batch.node_ids.iter().enumerate() {
            node_results.push(NodeResult {
                node_id: nid,
                lower_bound: (i as f64) + 1.0,
                solution: batch.lb[i].clone(),
                is_feasible: true,
            });
        }
        tm.import_results(&node_results);
        tm.process_evaluated();

        assert!(tm.is_finished());
        assert!(tm.incumbent().is_some());
    }
}
