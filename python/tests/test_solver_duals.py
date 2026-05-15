"""End-to-end tests for solver-supplied duals on the LP / QP / B&B paths.

Each test runs a real solve and asserts that ``constraint_duals`` /
``bound_duals_lower`` / ``bound_duals_upper`` are populated with the
expected sign and magnitude, then optionally re-runs through the
Examiner to confirm the duals satisfy KKT.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest


def _scalar(name: str, vec: dict[str, np.ndarray]) -> float:
    return float(np.asarray(vec[name]).ravel()[0])


# ── LP fast path ────────────────────────────────────────────────────────


def test_lp_path_returns_constraint_and_bound_duals():
    """min x + 2y s.t. x + y >= 4, 0<=x,y<=10 → x*=4, y*=0."""
    m = dm.Model("lp")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.minimize(x + 2 * y)
    m.subject_to(x + y >= 4, name="c1")

    res = m.solve()

    assert res.status == "optimal"
    assert res.constraint_duals is not None
    assert res.bound_duals_lower is not None
    assert res.bound_duals_upper is not None

    assert _scalar("x", res.x) == pytest.approx(4.0, abs=1e-6)
    assert _scalar("y", res.x) == pytest.approx(0.0, abs=1e-6)

    # ">=" row binding from below → μ ≥ 0 in the discopt convention.
    mu = _scalar("c1", res.constraint_duals)
    assert mu == pytest.approx(1.0, abs=1e-6)

    # Bound on y at lb=0 is active; reduced cost is 2 - mu = 1 (>= 0).
    lam_lb_y = _scalar("y", res.bound_duals_lower)
    assert lam_lb_y == pytest.approx(1.0, abs=1e-6)

    # Bound on x is inactive; multiplier ≈ 0.
    assert _scalar("x", res.bound_duals_lower) == pytest.approx(0.0, abs=1e-6)
    assert _scalar("x", res.bound_duals_upper) == pytest.approx(0.0, abs=1e-6)


def test_lp_equality_constraint_dual_sign():
    """Equality row dual is free; verify magnitude matches reduced LP."""
    m = dm.Model("lp_eq")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.minimize(x + 3 * y)
    m.subject_to(x + y == 4, name="ceq")

    res = m.solve()
    assert res.status == "optimal"
    assert res.constraint_duals is not None
    # x*=4, y*=0; ∇f = [1, 3]; KKT (Examiner convention ∇f + ∇body·μ = 0)
    # gives 1 + 1·μ = 0 → μ = -1 (equality multipliers are free in sign).
    mu = _scalar("ceq", res.constraint_duals)
    assert mu == pytest.approx(-1.0, abs=1e-6)


# ── QP fast path ────────────────────────────────────────────────────────


def test_qp_path_returns_duals_at_active_constraint():
    """min (x-0.5)^2 + (y-0.5)^2 s.t. x+y >= 5, 0<=x,y<=10."""
    m = dm.Model("qp_active")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.minimize((x - 0.5) ** 2 + (y - 0.5) ** 2)
    m.subject_to(x + y >= 5, name="c1")

    res = m.solve()
    assert res.status == "optimal"
    assert res.constraint_duals is not None
    assert res.bound_duals_lower is not None

    # Symmetric → x* = y* = 2.5; ∇f = [2(x-0.5), 2(y-0.5)] = [4, 4].
    # ">=" with μ ≥ 0 gives 4 - μ = 0 → μ = 4.
    mu = _scalar("c1", res.constraint_duals)
    assert mu == pytest.approx(4.0, abs=1e-4)


# ── MILP B&B path ───────────────────────────────────────────────────────


def test_milp_returns_relaxation_duals_at_incumbent():
    """min x + 2y, integer, s.t. x+y >= 5. Optimum x=5,y=0 with all-integer
    fix-and-resolve degenerating to zero free continuous columns; the
    recovery should still attach a dict (possibly all zeros) without
    raising."""
    m = dm.Model("milp")
    x = m.integer("x", lb=0, ub=10)
    y = m.integer("y", lb=0, ub=10)
    m.minimize(x + 2 * y)
    m.subject_to(x + y >= 5, name="c1")

    res = m.solve()
    assert res.status == "optimal"
    assert _scalar("x", res.x) == pytest.approx(5.0, abs=1e-6)
    assert _scalar("y", res.x) == pytest.approx(0.0, abs=1e-6)
    # Recovery returns dicts (even if all zero — fully integer fix gives
    # no free columns; what matters is the plumbing wires through).
    assert res.constraint_duals is not None
    assert "c1" in res.constraint_duals
    assert res.bound_duals_lower is not None
    assert "x" in res.bound_duals_lower
    assert "y" in res.bound_duals_lower


def test_miqp_returns_relaxation_duals_at_incumbent():
    """min (x-0.5)^2 + (y-0.5)^2 s.t. x+y >= 5, x cont, y int.

    With y fixed at incumbent (=3), the LP relaxation is min (x-0.5)^2
    s.t. x >= 2, x in [0,10] → x*=2, μ=3 on the ">=" row.
    """
    m = dm.Model("miqp")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.integer("y", lb=0, ub=10)
    m.minimize((x - 0.5) ** 2 + (y - 0.5) ** 2)
    m.subject_to(x + y >= 5, name="c1")

    res = m.solve()
    assert res.status == "optimal"
    assert res.constraint_duals is not None
    mu = _scalar("c1", res.constraint_duals)
    assert mu == pytest.approx(3.0, abs=1e-3)


# ── validate=True hook ──────────────────────────────────────────────────


def test_solve_with_validate_attaches_examiner_report():
    m = dm.Model("lp_validate")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.minimize(x + 2 * y)
    m.subject_to(x + y >= 4, name="c1")

    res = m.solve(validate=True)
    assert res.validation_report is not None
    rep = res.validation_report
    assert rep.passed, rep.summary(verbose=True)
    # The solver-duals branch should have run, since the LP fast path
    # populated constraint_duals.
    assert rep.solver_duals_used


def test_solve_without_validate_leaves_report_none():
    m = dm.Model("lp_no_validate")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.minimize(x)
    m.subject_to(x >= 1, name="c1")
    res = m.solve()
    assert res.validation_report is None


def test_validate_true_on_milp_attaches_report():
    m = dm.Model("milp_validate")
    x = m.integer("x", lb=0, ub=10)
    y = m.integer("y", lb=0, ub=10)
    m.minimize(x + 2 * y)
    m.subject_to(x + y >= 5, name="c1")

    res = m.solve(validate=True)
    assert res.validation_report is not None
    # We don't assert .passed — pure-integer fix-and-resolve degenerates;
    # the contract is that the report exists and primal checks pass.
    rep = res.validation_report
    primal = [c for c in rep.checks if c.name.startswith("primal_")]
    assert primal, "expected at least one primal check"
    assert all(c.passed for c in primal), rep.summary(verbose=True)
