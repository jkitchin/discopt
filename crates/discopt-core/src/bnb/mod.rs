//! Branch-and-Bound engine — node pool, branching, pruning.

pub mod branching;
pub mod convex_kernel;
pub mod convex_quad_cuts;
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
    array_rows_enabled, expand_rows_for_fbbt, is_scalar_layout, run_in_tree_presolve,
    run_in_tree_presolve_scalar, run_in_tree_presolve_scalar_cached, run_in_tree_presolve_view,
    scalarize_for_fbbt, ArrayRowStats, ArrayRowViewCache, ArrayRowsOutcome, InTreeDelta,
    InTreePresolveOptions, ScalarFbbtView, ARRAY_ROW_INSTRUCTION_BUDGET,
};
pub use node::{BranchRecord, Node, NodeId, NodeLineage, NodeStatus};
pub use pool::{NodePool, SelectionStrategy};
pub use tree_manager::{ExportBatch, NodeResult, ProcessingStats, TreeManager, TreeStats};
