//! Preprocessing and bound tightening for MINLP.
//!
//! This module implements:
//! - **FBBT**: Feasibility-Based Bound Tightening via forward/backward
//!   interval propagation through the expression DAG.
//! - **Probing**: Binary variable probing to detect implications and fixings.
//! - **Simplify**: Integer bound rounding, Big-M strengthening, and
//!   redundant constraint removal.

pub mod fbbt;
pub mod probing;
pub mod simplify;

pub use fbbt::{backward_propagate, fbbt, forward_propagate, Interval};
pub use probing::{probe_binary_vars, Implication, ProbingResult};
pub use simplify::{simplify, SimplifyResult};
