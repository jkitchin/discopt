//! Branch-and-Bound engine — node pool, branching, pruning.

pub mod branching;
pub mod in_tree_presolve;
pub mod node;
pub mod pool;
pub mod tree_manager;

// Re-export primary public types for convenience.
pub use branching::{BranchDecision, Pseudocosts, VarBranchInfo};
pub use in_tree_presolve::{run_in_tree_presolve, InTreeDelta, InTreePresolveOptions};
pub use node::{Node, NodeId, NodeStatus};
pub use pool::{NodePool, SelectionStrategy};
pub use tree_manager::{ExportBatch, NodeResult, ProcessingStats, TreeManager, TreeStats};
