//! Branch-and-Bound engine — node pool, branching, pruning.

pub mod branching;
pub mod convex_kernel;
pub mod in_tree_presolve;
pub mod mccormick_patch;
pub mod milp_driver;
pub mod node;
pub mod obbt_sweep;
pub mod obj_integral;
pub mod pool;
pub mod spatial_kernel;
pub mod spatial_propagate;
pub mod spatial_tree;
pub mod tree_manager;

// Re-export primary public types for convenience.
pub use branching::{BranchDecision, Pseudocosts, VarBranchInfo};
pub use in_tree_presolve::{
    is_scalar_layout, run_in_tree_presolve, run_in_tree_presolve_scalar, run_in_tree_presolve_view,
    scalarize_for_fbbt, scalarize_for_fbbt_with, ArrayRowStats, InTreeDelta, InTreePresolveOptions,
    ScalarFbbtView, ARRAY_ROW_NODE_BUDGET,
};
pub use node::{Node, NodeId, NodeStatus};
pub use pool::{NodePool, SelectionStrategy};
pub use tree_manager::{ExportBatch, NodeResult, ProcessingStats, TreeManager, TreeStats};
