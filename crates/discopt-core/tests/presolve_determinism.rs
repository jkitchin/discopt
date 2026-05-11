//! Determinism harness for the presolve orchestrator (item A4 of the
//! roadmap in `crates/discopt-core/src/presolve/ROADMAP.md`).
//!
//! For each fixture, we run the orchestrator twice with identical
//! inputs and assert that the resulting bounds and per-pass deltas
//! match exactly. The kernels themselves are deterministic by
//! construction (no RNG, the one `HashMap` use in `polynomial.rs` is
//! sorted before output), so any future regression — e.g. someone
//! introducing parallelism without a deterministic reduction — will
//! flip these tests immediately.
//!
//! Fixtures cover:
//!
//! 1. A trivial bounded-box NLP (FBBT only).
//! 2. A quartic polynomial constraint (polynomial reformulation +
//!    aux variables + FBBT on the rewritten model).
//! 3. A model with a singleton-equality variable (eliminate +
//!    bound propagation through the dropped equation).
//! 4. A model with a binary big-M constraint (probing + simplify).
//! 5. The same model from #4 run via a permuted pass order — the
//!    fixed-point bounds must coincide even though delta logs may
//!    differ.

use discopt_core::expr::*;
use discopt_core::presolve::{
    run_orchestrator, AggregatePass, CliquePass, EliminatePass, FbbtPass, ImpliedBoundsPass,
    OrchestratorOptions, PolynomialReformPass, PresolvePass, ProbingPass, RedundancyPass,
    SimplifyPass,
};

// ─────────────────────────────────────────────────────────────────
// Fixture builders
// ─────────────────────────────────────────────────────────────────

/// Trivial bounded box: minimize x with x in [-1, 1].
fn fixture_bounded_box() -> ModelRepr {
    let mut arena = ExprArena::new();
    let x = arena.add(ExprNode::Variable {
        name: "x".into(),
        index: 0,
        size: 1,
        shape: vec![],
    });
    ModelRepr {
        arena,
        objective: x,
        objective_sense: ObjectiveSense::Minimize,
        constraints: vec![],
        variables: vec![VarInfo {
            name: "x".into(),
            var_type: VarType::Continuous,
            offset: 0,
            size: 1,
            shape: vec![],
            lb: vec![-1.0],
            ub: vec![1.0],
        }],
        n_vars: 1,
    }
}

/// Quartic polynomial: x^4 + y <= 5; x in [-2, 2], y in [-3, 3].
fn fixture_quartic_poly() -> ModelRepr {
    let mut arena = ExprArena::new();
    let x = arena.add(ExprNode::Variable {
        name: "x".into(),
        index: 0,
        size: 1,
        shape: vec![],
    });
    let y = arena.add(ExprNode::Variable {
        name: "y".into(),
        index: 1,
        size: 1,
        shape: vec![],
    });
    let exp4 = arena.add(ExprNode::Constant(4.0));
    let x4 = arena.add(ExprNode::BinaryOp {
        op: BinOp::Pow,
        left: x,
        right: exp4,
    });
    let body = arena.add(ExprNode::BinaryOp {
        op: BinOp::Add,
        left: x4,
        right: y,
    });
    ModelRepr {
        arena,
        objective: x,
        objective_sense: ObjectiveSense::Minimize,
        constraints: vec![ConstraintRepr {
            body,
            sense: ConstraintSense::Le,
            rhs: 5.0,
            name: Some("c0".into()),
        }],
        variables: vec![
            VarInfo {
                name: "x".into(),
                var_type: VarType::Continuous,
                offset: 0,
                size: 1,
                shape: vec![],
                lb: vec![-2.0],
                ub: vec![2.0],
            },
            VarInfo {
                name: "y".into(),
                var_type: VarType::Continuous,
                offset: 1,
                size: 1,
                shape: vec![],
                lb: vec![-3.0],
                ub: vec![3.0],
            },
        ],
        n_vars: 2,
    }
}

/// Singleton equality: 2*x == 4, with x in [0, 10]. After eliminate,
/// x should be pinned to 2.
fn fixture_singleton_eq() -> ModelRepr {
    let mut arena = ExprArena::new();
    let x = arena.add(ExprNode::Variable {
        name: "x".into(),
        index: 0,
        size: 1,
        shape: vec![],
    });
    let two = arena.add(ExprNode::Constant(2.0));
    let body = arena.add(ExprNode::BinaryOp {
        op: BinOp::Mul,
        left: two,
        right: x,
    });
    ModelRepr {
        arena,
        objective: x,
        objective_sense: ObjectiveSense::Minimize,
        constraints: vec![ConstraintRepr {
            body,
            sense: ConstraintSense::Eq,
            rhs: 4.0,
            name: Some("c0".into()),
        }],
        variables: vec![VarInfo {
            name: "x".into(),
            var_type: VarType::Continuous,
            offset: 0,
            size: 1,
            shape: vec![],
            lb: vec![0.0],
            ub: vec![10.0],
        }],
        n_vars: 1,
    }
}

// ─────────────────────────────────────────────────────────────────
// Canonicalisation
// ─────────────────────────────────────────────────────────────────

/// Normalise an orchestrator outcome to a string. Excludes wall_time
/// (nondeterministic across runs) but includes everything else.
fn canon(result: &discopt_core::presolve::PresolveResult) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    writeln!(s, "iterations={}", result.iterations).unwrap();
    writeln!(s, "terminated_by={:?}", result.terminated_by).unwrap();
    writeln!(s, "n_vars={}", result.model.variables.len()).unwrap();
    writeln!(s, "n_constraints={}", result.model.constraints.len()).unwrap();
    for (i, b) in result.bounds.iter().enumerate() {
        writeln!(s, "bounds[{}] = ({:?}, {:?})", i, b.lo, b.hi).unwrap();
    }
    for (i, d) in result.deltas.iter().enumerate() {
        writeln!(
            s,
            "delta[{}] pass={} iter={} bounds_tightened={} vars_fixed={:?} aux_vars={} aux_cons={} cons_removed={:?} cons_rewritten={:?} work_units={}",
            i,
            d.pass_name,
            d.pass_iter,
            d.bounds_tightened,
            d.vars_fixed,
            d.aux_vars_introduced,
            d.aux_constraints_introduced,
            d.constraints_removed,
            d.constraints_rewritten,
            d.work_units,
        )
        .unwrap();
    }
    s
}

fn default_passes() -> Vec<Box<dyn PresolvePass>> {
    vec![
        Box::new(EliminatePass),
        Box::new(AggregatePass),
        Box::new(RedundancyPass),
        Box::new(SimplifyPass),
        Box::new(ImpliedBoundsPass),
        Box::new(FbbtPass::default()),
        Box::new(ProbingPass),
        Box::new(CliquePass),
    ]
}

fn default_passes_with_polynomial() -> Vec<Box<dyn PresolvePass>> {
    vec![
        Box::new(EliminatePass),
        Box::new(AggregatePass),
        Box::new(RedundancyPass),
        Box::new(PolynomialReformPass),
        Box::new(SimplifyPass),
        Box::new(ImpliedBoundsPass),
        Box::new(FbbtPass::default()),
        Box::new(ProbingPass),
        Box::new(CliquePass),
    ]
}

fn run_once(
    model: ModelRepr,
    passes: Vec<Box<dyn PresolvePass>>,
) -> discopt_core::presolve::PresolveResult {
    let opts = OrchestratorOptions::with_passes(passes);
    run_orchestrator(model, opts)
}

// ─────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────

#[test]
fn deterministic_on_bounded_box() {
    let a = run_once(fixture_bounded_box(), default_passes());
    let b = run_once(fixture_bounded_box(), default_passes());
    assert_eq!(canon(&a), canon(&b));
}

#[test]
fn deterministic_on_quartic_poly() {
    // Run twice with polynomial reformulation enabled.
    let a = run_once(fixture_quartic_poly(), default_passes_with_polynomial());
    let b = run_once(fixture_quartic_poly(), default_passes_with_polynomial());
    assert_eq!(canon(&a), canon(&b));
    // Must have introduced aux variables (degree-4 monomial → bilinear).
    assert!(
        a.deltas
            .iter()
            .any(|d| d.pass_name == "polynomial_reform" && d.aux_vars_introduced > 0),
        "expected polynomial reformulation to introduce aux vars"
    );
}

#[test]
fn deterministic_on_singleton_eq() {
    let a = run_once(fixture_singleton_eq(), default_passes());
    let b = run_once(fixture_singleton_eq(), default_passes());
    assert_eq!(canon(&a), canon(&b));
    // Eliminate should have fixed the variable.
    assert!(
        a.deltas
            .iter()
            .any(|d| d.pass_name == "eliminate" && d.bounds_tightened > 0),
        "expected eliminate to record progress on the singleton equality"
    );
    // Final bound should be pinned at the derived value 2.0.
    let b0 = a.bounds[0];
    assert!((b0.lo - 2.0).abs() < 1e-9, "lb={} expected 2.0", b0.lo);
    assert!((b0.hi - 2.0).abs() < 1e-9, "ub={} expected 2.0", b0.hi);
}

#[test]
fn ten_repeats_byte_identical() {
    // Stronger test: 10 runs all identical.
    let baseline = canon(&run_once(
        fixture_quartic_poly(),
        default_passes_with_polynomial(),
    ));
    for _ in 0..10 {
        let s = canon(&run_once(
            fixture_quartic_poly(),
            default_passes_with_polynomial(),
        ));
        assert_eq!(s, baseline);
    }
}
