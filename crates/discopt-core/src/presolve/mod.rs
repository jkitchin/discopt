//! Preprocessing and bound tightening for MINLP.
//!
//! Individual pass kernels:
//! - **FBBT** (`fbbt`): Feasibility-Based Bound Tightening via
//!   forward/backward interval propagation through the expression DAG.
//! - **Probing** (`probing`): Binary variable probing to detect
//!   implications and fixings.
//! - **Simplify** (`simplify`): Integer bound rounding, Big-M
//!   strengthening, and redundant constraint removal.
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
pub mod eliminate;
pub mod factorable_elim;
pub mod fbbt;
pub mod fbbt_fp;
pub mod implied_bounds;
pub mod orchestrator;
pub mod pass;
pub mod passes;
pub mod polynomial;
pub mod probing;
pub mod redundancy;
pub mod simplify;
pub mod substitute;
pub mod symmetry;

pub use aggregate::{
    aggregate_variables, aggregate_variables_until, AggregationRecord, AggregationStats,
};
pub use cliques::{extract_cliques, CliqueSet, CliqueStats};
pub use coefficient_strengthening::{coefficient_strengthening, CoefficientStrengtheningStats};
pub use delta::{
    Implication as DeltaImplication, PresolveDelta, StructureManifest, TerminationReason,
    VarAggregation,
};
pub use eliminate::{eliminate_variables, eliminate_variables_until, EliminationStats};
pub use factorable_elim::{factorable_eliminate, FactorableElimStats};
pub use fbbt::{
    backward_propagate, fbbt, fbbt_until, fbbt_with_cutoff, fbbt_with_cutoff_until,
    forward_propagate, Interval,
};
pub use fbbt_fp::{fbbt_fixed_point, FbbtFpOptions, FbbtFpStats};
pub use implied_bounds::{propagate_implied_bounds, ImpliedBoundsStats};
pub use orchestrator::{run as run_orchestrator, OrchestratorOptions, PresolveResult};
pub use pass::{PassCategory, PresolveContext, PresolvePass};
pub use passes::{
    AggregatePass, CliquePass, CoefficientStrengtheningPass, EliminatePass, FactorableElimPass,
    FbbtFixedPointPass, FbbtPass, ImpliedBoundsPass, PolynomialReformPass, ProbingPass,
    RedundancyPass, SimplifyPass,
};
pub use polynomial::{
    reformulate_polynomial, try_polynomial, Monomial, Polynomial, ReformulationStats,
};
pub use probing::{
    probe_binary_vars, probe_node_bounds, Implication, NodeProbeResult, ProbingResult,
};
pub use redundancy::{detect_row_redundancy, RedundancyStats};
pub use simplify::{simplify, simplify_until, SimplifyResult};
pub use substitute::{
    postsolve_chain, postsolve_point, substitute_to_fixpoint, substitute_variables, SubstDef,
    SubstitutionRecord, SubstitutionStats,
};
pub use symmetry::{detect_symmetries, Orbit, SymmetryStats};
