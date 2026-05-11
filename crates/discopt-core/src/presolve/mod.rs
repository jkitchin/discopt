//! Preprocessing and bound tightening for MINLP.
//!
//! Individual pass kernels:
//! - **FBBT** (`fbbt`): Feasibility-Based Bound Tightening via
//!   forward/backward interval propagation through the expression DAG.
//! - **Probing** (`probing`): Binary variable probing to detect
//!   implications and fixings.
//! - **Simplify** (`simplify`): Integer bound rounding, Big-M
//!   strengthening, and redundant constraint removal.
//! - **OBBT** (`obbt`): Optimality-Based Bound Tightening helpers
//!   (LP solving lives on the Python side).
//! - **Eliminate** (`eliminate`): M10 variable elimination via
//!   singleton equality detection.
//! - **Polynomial reformulation** (`polynomial`): M4+M5 polynomial-to-
//!   bilinear lowering.
//!
//! Orchestration layer (P1 of the roadmap, item A1+A2+A4):
//! - **`delta`** — `PresolveDelta` and friends; the uniform return type
//!   for any pass.
//! - **`pass`** — `PresolvePass` trait + `PresolveContext`.
//! - **`orchestrator`** — fixed-point loop driver under a global
//!   budget.
//! - **`passes`** — adapter shims wrapping each kernel as a
//!   `PresolvePass`.

pub mod aggregate;
pub mod cliques;
pub mod coefficient_strengthening;
pub mod delta;
pub mod duality;
pub mod eliminate;
pub mod factorable_elim;
pub mod fbbt;
pub mod fbbt_fp;
pub mod implied_bounds;
pub mod obbt;
pub mod orchestrator;
pub mod pass;
pub mod passes;
pub mod polynomial;
pub mod probing;
pub mod reduction_constraints;
pub mod redundancy;
pub mod scaling;
pub mod simplify;
pub mod symmetry;

pub use aggregate::{aggregate_variables, AggregationRecord, AggregationStats};
pub use cliques::{extract_cliques, CliqueSet, CliqueStats};
pub use coefficient_strengthening::{coefficient_strengthening, CoefficientStrengtheningStats};
pub use delta::{
    Implication as DeltaImplication, PresolveDelta, StructureManifest, TerminationReason,
    VarAggregation,
};
pub use duality::{reduced_cost_fixing, ReducedCostInfo, ReducedCostStats};
pub use eliminate::{eliminate_variables, EliminationStats};
pub use factorable_elim::{factorable_eliminate, FactorableElimStats};
pub use fbbt::{backward_propagate, fbbt, fbbt_with_cutoff, forward_propagate, Interval};
pub use fbbt_fp::{fbbt_fixed_point, FbbtFpOptions, FbbtFpStats};
pub use implied_bounds::{propagate_implied_bounds, ImpliedBoundsStats};
pub use obbt::{apply_obbt_bounds, extract_linear_rows, obbt_candidates, LinearRow, ObbtResult};
pub use orchestrator::{run as run_orchestrator, OrchestratorOptions, PresolveResult};
pub use pass::{PassCategory, PresolveContext, PresolvePass};
pub use passes::{
    AggregatePass, CliquePass, CoefficientStrengtheningPass, EliminatePass, FactorableElimPass,
    FbbtFixedPointPass, FbbtPass, ImpliedBoundsPass, PolynomialReformPass, ProbingPass,
    ReducedCostFixingPass, ReductionConstraintsPass, RedundancyPass, ScalingPass, SimplifyPass,
};
pub use polynomial::{
    reformulate_polynomial, try_polynomial, Monomial, Polynomial, ReformulationStats,
};
pub use probing::{probe_binary_vars, Implication, ProbingResult};
pub use reduction_constraints::{detect_reduction_constraints, ReductionStats};
pub use redundancy::{detect_row_redundancy, RedundancyStats};
pub use scaling::{compute_equilibration, ScalingFactors, ScalingStats};
pub use simplify::{simplify, SimplifyResult};
pub use symmetry::{detect_symmetries, Orbit, SymmetryStats};
