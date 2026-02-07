//! Branch-and-Bound engine — node pool, branching, pruning.

pub mod branching;
pub mod node;
pub mod pool;
pub mod tree_manager;

// Re-export primary public types for convenience.
pub use branching::{BranchDecision, VarBranchInfo};
pub use node::{Node, NodeId, NodeStatus};
pub use pool::{NodePool, SelectionStrategy};
pub use tree_manager::{ExportBatch, NodeResult, ProcessingStats, TreeManager, TreeStats};
