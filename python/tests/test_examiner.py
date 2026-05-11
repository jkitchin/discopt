"""Unit tests for the Examiner-style post-solve validator.

Each test constructs a small model, fabricates a ``SolveResult`` at a chosen
point, and asserts which checks pass or fail. Solver involvement is avoided
so failures localise to the validator.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import SolveResult
from discopt.validation.examiner import (
    DUAL_FEAS_TOL,
    INTEGRALITY_TOL,
    PRIMAL_FEAS_TOL,
    assert_examined,
    examine,
)


def _result(x: dict[str, np.ndarray], obj: float | None) -> SolveResult:
    return SolveResult(status="optimal", objective=obj, x=x)


def _model_qp_box():
    """min (x-2)^2 + (y+1)^2 s.t. x+y <= 1, 0 <= x,y <= 5. Optimum at (1, 0)."""
    m = dm.Model("qp_box")
    x = m.continuous("x", lb=0.0, ub=5.0)
    y = m.continuous("y", lb=0.0, ub=5.0)
    m.minimize((x - 2) ** 2 + (y + 1) ** 2)
    m.subject_to(x + y <= 1)
    return m


def test_passes_at_known_optimum():
    m = _model_qp_box()
    # KKT: x*=1, y*=0; binding: x+y=1, y at lb=0; obj = (1-2)^2 + (0+1)^2 = 2.
    res = _result({"x": np.array(1.0), "y": np.array(0.0)}, obj=2.0)
    rep = examine(res, m)
    assert rep.passed, rep.summary(verbose=True)
    assert any(c.name == "stationarity" for c in rep.checks)


def test_flags_bound_violation():
    m = _model_qp_box()
    res = _result({"x": np.array(-0.01), "y": np.array(0.0)}, obj=5.0401)
    rep = examine(res, m)
    bad = next(c for c in rep.checks if c.name == "primal_var_feas")
    assert not bad.passed
    assert bad.max_violation > PRIMAL_FEAS_TOL


def test_flags_constraint_violation():
    m = _model_qp_box()
    # x=2, y=2 violates x+y<=1 by 3.
    res = _result({"x": np.array(2.0), "y": np.array(2.0)}, obj=9.0)
    rep = examine(res, m)
    bad = next(c for c in rep.checks if c.name.startswith("primal_con_feas"))
    assert not bad.passed
    assert bad.max_violation > 1.0


def test_flags_objective_inconsistency():
    m = _model_qp_box()
    # x=1, y=0 → true obj=2; report a wrong obj=99.
    res = _result({"x": np.array(1.0), "y": np.array(0.0)}, obj=99.0)
    rep = examine(res, m)
    bad = next(c for c in rep.checks if c.name == "obj_consistency")
    assert not bad.passed


def test_flags_integrality_violation():
    m = dm.Model("mip_int")
    z = m.integer("z", lb=0, ub=5)
    m.minimize(z)
    res = _result({"z": np.array(1.3)}, obj=1.3)
    rep = examine(res, m)
    bad = next(c for c in rep.checks if c.name == "integrality")
    assert not bad.passed
    assert bad.max_violation > INTEGRALITY_TOL


def test_flags_non_stationary_point():
    """A feasible-but-not-optimal interior point should fail stationarity."""
    m = dm.Model("qp_unconstrained_interior")
    x = m.continuous("x", lb=-10.0, ub=10.0)
    m.minimize((x - 2) ** 2)
    # Optimum is x=2; report x=0 (interior, no active constraints) → ∇f = -4 ≠ 0.
    res = _result({"x": np.array(0.0)}, obj=4.0)
    rep = examine(res, m)
    stat = next(c for c in rep.checks if c.name == "stationarity")
    assert not stat.passed
    assert stat.max_violation > DUAL_FEAS_TOL


def test_passes_at_unconstrained_interior_optimum():
    m = dm.Model("qp_unconstrained_interior_ok")
    x = m.continuous("x", lb=-10.0, ub=10.0)
    m.minimize((x - 2) ** 2)
    res = _result({"x": np.array(2.0)}, obj=0.0)
    rep = examine(res, m)
    assert rep.passed, rep.summary(verbose=True)


def test_assert_examined_raises_with_check_name():
    m = _model_qp_box()
    res = _result({"x": np.array(2.0), "y": np.array(2.0)}, obj=9.0)
    with pytest.raises(AssertionError, match="primal_con_feas"):
        assert_examined(res, m, "qp_box_bad")


def test_handles_maximize_objective_sign():
    m = dm.Model("max_qp")
    x = m.continuous("x", lb=-10.0, ub=10.0)
    m.maximize(-((x - 3) ** 2))
    # Optimum: x=3, obj=0 (in user-facing maximize sign).
    res = _result({"x": np.array(3.0)}, obj=0.0)
    rep = examine(res, m)
    assert rep.passed, rep.summary(verbose=True)


def test_recover_duals_can_be_disabled():
    m = _model_qp_box()
    res = _result({"x": np.array(1.0), "y": np.array(0.0)}, obj=2.0)
    rep = examine(res, m, recover_duals=False)
    assert rep.passed
    assert not any(c.name == "stationarity" for c in rep.checks)


# ─────────────────────────────────────────────────────────────────────────────
# Solver-supplied dual path
# ─────────────────────────────────────────────────────────────────────────────


def _result_with_duals(
    x: dict[str, np.ndarray],
    obj: float | None,
    *,
    constraint_duals=None,
    bound_duals_lower=None,
    bound_duals_upper=None,
) -> SolveResult:
    return SolveResult(
        status="optimal",
        objective=obj,
        x=x,
        constraint_duals=constraint_duals,
        bound_duals_lower=bound_duals_lower,
        bound_duals_upper=bound_duals_upper,
    )


def test_solver_duals_pass_at_correct_kkt_point():
    """KKT for min (x-2)^2+(y+1)^2 s.t. x+y<=1, y>=0 at (1,0).

    Stationarity:
      ∇f = (2(x-2), 2(y+1)) = (-2, 2).
      Active: row "x+y<=1" (row index 0), bound y>=0 (lb on y).
      ∇f + Jᵀ μ + λ_ub − λ_lb = 0
      → (-2, 2) + (1, 1)·μ + (0, -λ_lb_y) = 0
      → μ = 2, λ_lb_y = 4.
    """
    m = dm.Model("kkt_demo")
    x = m.continuous("x", lb=0.0, ub=5.0)
    y = m.continuous("y", lb=0.0, ub=5.0)
    m.minimize((x - 2) ** 2 + (y + 1) ** 2)
    m.subject_to(x + y <= 1, name="c0")

    res = _result_with_duals(
        {"x": np.array(1.0), "y": np.array(0.0)},
        obj=2.0,
        constraint_duals={"c0": np.array(2.0)},
        bound_duals_lower={"x": np.array(0.0), "y": np.array(4.0)},
        bound_duals_upper={"x": np.array(0.0), "y": np.array(0.0)},
    )
    rep = examine(res, m)
    assert rep.passed, rep.summary(verbose=True)
    assert rep.solver_duals_used
    assert any(c.name == "stationarity (solver duals)" for c in rep.checks)
    assert any(c.name == "stationarity (recovered)" for c in rep.checks)


def test_solver_duals_flag_wrong_sign():
    """A "<=" row supplied with negative μ violates dual feasibility."""
    m = dm.Model("wrong_sign")
    x = m.continuous("x", lb=0.0, ub=5.0)
    y = m.continuous("y", lb=0.0, ub=5.0)
    m.minimize((x - 2) ** 2 + (y + 1) ** 2)
    m.subject_to(x + y <= 1, name="c0")

    res = _result_with_duals(
        {"x": np.array(1.0), "y": np.array(0.0)},
        obj=2.0,
        constraint_duals={"c0": np.array(-2.0)},  # wrong sign for "<="
        bound_duals_lower={"x": np.array(0.0), "y": np.array(4.0)},
        bound_duals_upper={"x": np.array(0.0), "y": np.array(0.0)},
    )
    rep = examine(res, m)
    bad = next(c for c in rep.checks if c.name == "dual_var_feas (solver duals)")
    assert not bad.passed
    assert bad.max_violation > DUAL_FEAS_TOL


def test_solver_duals_flag_stationarity_when_inconsistent():
    """Supply duals that don't satisfy ∇f + Jᵀμ + λ_ub − λ_lb = 0."""
    m = dm.Model("bad_stat")
    x = m.continuous("x", lb=0.0, ub=5.0)
    y = m.continuous("y", lb=0.0, ub=5.0)
    m.minimize((x - 2) ** 2 + (y + 1) ** 2)
    m.subject_to(x + y <= 1, name="c0")

    res = _result_with_duals(
        {"x": np.array(1.0), "y": np.array(0.0)},
        obj=2.0,
        constraint_duals={"c0": np.array(0.5)},  # too small
        bound_duals_lower={"x": np.array(0.0), "y": np.array(0.5)},
        bound_duals_upper={"x": np.array(0.0), "y": np.array(0.0)},
    )
    rep = examine(res, m)
    bad = next(c for c in rep.checks if c.name == "stationarity (solver duals)")
    assert not bad.passed
    assert bad.max_violation > DUAL_FEAS_TOL


def test_solver_duals_flag_dual_cs_for_inactive_row():
    """An inactive "<=" row with nonzero μ violates dual complementary slack."""
    m = dm.Model("dual_cs")
    x = m.continuous("x", lb=-10.0, ub=10.0)
    m.minimize((x - 2) ** 2)
    m.subject_to(x <= 5, name="c0")  # inactive at x=2

    res = _result_with_duals(
        {"x": np.array(2.0)},
        obj=0.0,
        constraint_duals={"c0": np.array(1.0)},  # μ ≠ 0 but body-rhs = -3
        bound_duals_lower={"x": np.array(0.0)},
        bound_duals_upper={"x": np.array(0.0)},
    )
    rep = examine(res, m)
    bad = next(c for c in rep.checks if c.name == "dual_cs (solver duals)")
    assert not bad.passed
    assert bad.max_violation > 1e-6


def test_dual_consistency_check_appears_when_both_paths_run():
    m = dm.Model("xcheck")
    x = m.continuous("x", lb=0.0, ub=5.0)
    y = m.continuous("y", lb=0.0, ub=5.0)
    m.minimize((x - 2) ** 2 + (y + 1) ** 2)
    m.subject_to(x + y <= 1, name="c0")

    res = _result_with_duals(
        {"x": np.array(1.0), "y": np.array(0.0)},
        obj=2.0,
        constraint_duals={"c0": np.array(2.0)},
        bound_duals_lower={"x": np.array(0.0), "y": np.array(4.0)},
        bound_duals_upper={"x": np.array(0.0), "y": np.array(0.0)},
    )
    rep = examine(res, m)
    cross = next(
        (c for c in rep.checks if c.name == "dual_consistency (solver vs recovered)"),
        None,
    )
    assert cross is not None
    assert cross.passed, cross.detail


def test_end_to_end_nlp_duals_round_trip_through_examine():
    """Solve a small NLP via discopt; confirm duals populated and pass examine()."""
    pytest.importorskip("cyipopt")

    m = dm.Model("nlp_xyz")
    x = m.continuous("x", lb=0.0, ub=5.0)
    y = m.continuous("y", lb=0.0, ub=5.0)
    m.minimize(dm.exp(x - 2) + (y + 1) ** 2)
    m.subject_to(x + y <= 1, name="c0")

    res = m.solve()
    assert res.status == "optimal"
    assert res.convex_fast_path is True
    # NLP path should populate solver-supplied duals.
    assert res.constraint_duals is not None
    assert res.bound_duals_lower is not None
    assert res.bound_duals_upper is not None
    assert "c0" in res.constraint_duals
    assert "x" in res.bound_duals_lower
    assert "y" in res.bound_duals_lower

    rep = examine(res, m)
    assert rep.solver_duals_used
    # All KKT checks should pass at the IPOPT solution.
    assert rep.passed, rep.summary(verbose=True)
    # Both solver-direct and recovered checks should be present.
    names = {c.name for c in rep.checks}
    assert "stationarity (solver duals)" in names
    assert "stationarity (recovered)" in names
    # Cross-check should appear and pass (within permissive tol).
    assert "dual_consistency (solver vs recovered)" in names


def test_dual_consistency_check_flags_disagreement():
    """Solver duals that solve KKT but differ wildly from a recovered LSQ
    solution should still trip the consistency cross-check (large tol margin).
    """
    m = dm.Model("disagree")
    # Underdetermined active set: two equality redundant variants → many y solve KKT.
    x = m.continuous("x", lb=0.0, ub=5.0)
    y = m.continuous("y", lb=0.0, ub=5.0)
    m.minimize((x - 2) ** 2 + (y + 1) ** 2)
    m.subject_to(x + y <= 1, name="c0")

    # supply hugely off-magnitude (but still "satisfies" inequality sign) duals
    # by combining with a wrong stationarity offset injected via lb/ub fields
    res = _result_with_duals(
        {"x": np.array(1.0), "y": np.array(0.0)},
        obj=2.0,
        constraint_duals={"c0": np.array(2.0)},
        bound_duals_lower={"x": np.array(50.0), "y": np.array(54.0)},
        bound_duals_upper={"x": np.array(0.0), "y": np.array(0.0)},
    )
    rep = examine(res, m)
    cross = next(
        (c for c in rep.checks if c.name == "dual_consistency (solver vs recovered)"),
        None,
    )
    assert cross is not None
    assert not cross.passed
