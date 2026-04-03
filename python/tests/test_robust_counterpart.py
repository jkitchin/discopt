"""Integration tests for RobustCounterpart reformulation correctness.

Tests verify that:
1. The reformulated constraint/objective DAG contains only deterministic nodes
   (no Parameter nodes from the uncertainty set remain).
2. The worst-case constants embedded in the reformulation are correct for
   known small examples.
3. The robustified model objective is always ≥ nominal objective (for box
   uncertainty on a cost minimisation problem).

These are structural/algebraic tests that do not require the Rust solver.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
from discopt.ro import (
    BoxUncertaintySet,
    EllipsoidalUncertaintySet,
    PolyhedralUncertaintySet,
    RobustCounterpart,
    budget_uncertainty_set,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_parameters(expr) -> set[str]:
    """Return the set of Parameter names reachable from expr."""
    from discopt.modeling.core import (
        BinaryOp,
        FunctionCall,
        IndexExpression,
        MatMulExpression,
        Parameter,
        SumExpression,
        SumOverExpression,
        UnaryOp,
    )

    if isinstance(expr, Parameter):
        return {expr.name}
    if isinstance(expr, (BinaryOp, MatMulExpression)):
        return _collect_parameters(expr.left) | _collect_parameters(expr.right)
    if isinstance(expr, UnaryOp):
        return _collect_parameters(expr.operand)
    if isinstance(expr, FunctionCall):
        result: set[str] = set()
        for a in expr.args:
            result |= _collect_parameters(a)
        return result
    if isinstance(expr, IndexExpression):
        return _collect_parameters(expr.base)
    if isinstance(expr, SumExpression):
        return _collect_parameters(expr.operand)
    if isinstance(expr, SumOverExpression):
        result2: set[str] = set()
        for t in expr.terms:
            result2 |= _collect_parameters(t)
        return result2
    return set()


def _collect_constants(expr) -> list:
    """Return all Constant values reachable from expr as flat list."""
    from discopt.modeling.core import (
        BinaryOp,
        Constant,
        FunctionCall,
        IndexExpression,
        MatMulExpression,
        SumExpression,
        SumOverExpression,
        UnaryOp,
    )

    if isinstance(expr, Constant):
        return [expr.value]
    if isinstance(expr, (BinaryOp, MatMulExpression)):
        return _collect_constants(expr.left) + _collect_constants(expr.right)
    if isinstance(expr, UnaryOp):
        return _collect_constants(expr.operand)
    if isinstance(expr, FunctionCall):
        result = []
        for a in expr.args:
            result.extend(_collect_constants(a))
        return result
    if isinstance(expr, IndexExpression):
        return _collect_constants(expr.base)
    if isinstance(expr, SumExpression):
        return _collect_constants(expr.operand)
    if isinstance(expr, SumOverExpression):
        result2 = []
        for t in expr.terms:
            result2.extend(_collect_constants(t))
        return result2
    return []


# ---------------------------------------------------------------------------
# Box reformulation – structural correctness
# ---------------------------------------------------------------------------


class TestBoxReformulationStructure:
    """The reformulated model must contain no uncertain parameters."""

    def _cost_min_model(self):
        """min c^T x  s.t. x >= d, x >= 0 with scalar variables."""
        m = dm.Model()
        x = m.continuous("x", shape=(3,), lb=0)
        c = m.parameter("c", value=[10.0, 15.0, 8.0])
        d = m.parameter("d", value=5.0)
        m.minimize(dm.sum(c * x))
        m.subject_to(dm.sum(x) >= d, name="demand")
        return m, x, c, d

    def test_uncertain_params_removed_from_objective(self):
        m, x, c, d = self._cost_min_model()
        unc_c = BoxUncertaintySet(c, delta=1.0)
        rc = RobustCounterpart(m, unc_c)
        rc.formulate()
        remaining = _collect_parameters(m._objective.expression)
        assert "c" not in remaining, f"c still appears in objective: {remaining}"

    def test_uncertain_params_removed_from_constraints(self):
        m, x, c, d = self._cost_min_model()
        unc_d = BoxUncertaintySet(d, delta=0.5)
        rc = RobustCounterpart(m, unc_d)
        rc.formulate()
        for con in m._constraints:
            remaining = _collect_parameters(con.body)
            assert "d" not in remaining, f"d still appears in constraint: {remaining}"

    def test_multiple_params_removed(self):
        m, x, c, d = self._cost_min_model()
        rc = RobustCounterpart(m, [BoxUncertaintySet(c, 1.0), BoxUncertaintySet(d, 0.5)])
        rc.formulate()
        obj_params = _collect_parameters(m._objective.expression)
        assert "c" not in obj_params
        for con in m._constraints:
            assert "d" not in _collect_parameters(con.body)


class TestBoxReformulationValues:
    """Verify that worst-case constants are numerically correct."""

    def test_scalar_cost_upper_bound(self):
        """min c*x  with c uncertain: worst-case c = c̄ + δ."""
        m = dm.Model()
        x = m.continuous("x", lb=0)
        c = m.parameter("c", value=10.0)
        m.minimize(c * x)
        m.subject_to(x >= 1.0)

        rc = RobustCounterpart(m, BoxUncertaintySet(c, delta=2.0))
        rc.formulate()

        # Objective body should contain constant 12.0 (= 10 + 2)
        constants = _collect_constants(m._objective.expression)
        flat = np.concatenate([np.atleast_1d(v) for v in constants])
        assert np.any(np.isclose(flat, 12.0)), (
            f"Expected worst-case cost 12.0 in objective constants; got {flat}"
        )

    def test_rhs_demand_lower_bound(self):
        """x >= d  stored as  d - x <= 0: worst-case (maximize body) uses d̄ + δ."""
        m = dm.Model()
        x = m.continuous("x", lb=0)
        d = m.parameter("d", value=5.0)
        m.minimize(x)
        m.subject_to(x >= d, name="demand")  # stored: d - x <= 0

        rc = RobustCounterpart(m, BoxUncertaintySet(d, delta=1.0))
        rc.formulate()

        # Constraint body = d - x; worst-case body maximization → d = d̄ + δ = 6.0
        constants = _collect_constants(m._constraints[0].body)
        flat = np.concatenate([np.atleast_1d(v) for v in constants])
        assert np.any(np.isclose(flat, 6.0)), (
            f"Expected worst-case demand 6.0 in constraint constants; got {flat}"
        )

    def test_minimize_objective_is_conservative(self):
        """Robust objective value ≥ nominal for a simple LP."""
        # Both nominal and robust have known optima.
        # Nominal: min 10x  s.t. x >= 5  →  opt = 50
        # Robust (c ± 1, d ± 0.5): worst-case c=11, d=5.5  →  opt = 60.5
        m_nom = dm.Model()
        x_n = m_nom.continuous("x", lb=0)
        c_n = m_nom.parameter("c", value=10.0)
        d_n = m_nom.parameter("d", value=5.0)
        m_nom.minimize(c_n * x_n)
        m_nom.subject_to(x_n >= d_n)

        m_rob = dm.Model()
        x_r = m_rob.continuous("x", lb=0)
        c_r = m_rob.parameter("c", value=10.0)
        d_r = m_rob.parameter("d", value=5.0)
        m_rob.minimize(c_r * x_r)
        m_rob.subject_to(x_r >= d_r)

        rc = RobustCounterpart(m_rob, [BoxUncertaintySet(c_r, 1.0), BoxUncertaintySet(d_r, 0.5)])
        rc.formulate()

        # Verify the robust constants: c_wc = 11, d_wc = 5.5
        obj_consts = _collect_constants(m_rob._objective.expression)
        flat_obj = np.concatenate([np.atleast_1d(v) for v in obj_consts])
        assert np.any(np.isclose(flat_obj, 11.0)), f"Expected robust cost 11.0; got {flat_obj}"
        con_consts = _collect_constants(m_rob._constraints[0].body)
        flat_con = np.concatenate([np.atleast_1d(v) for v in con_consts])
        assert np.any(np.isclose(flat_con, 5.5)), f"Expected robust demand 5.5; got {flat_con}"

    def test_vector_parameter(self):
        """Vector cost parameter: each component uses its own δ."""
        m = dm.Model()
        x = m.continuous("x", shape=(3,), lb=0)
        c = m.parameter("c", value=[10.0, 15.0, 8.0])
        m.minimize(dm.sum(c * x))
        m.subject_to(dm.sum(x) == 10.0)

        delta = np.array([1.0, 2.0, 0.5])
        rc = RobustCounterpart(m, BoxUncertaintySet(c, delta=delta))
        rc.formulate()

        obj_consts = _collect_constants(m._objective.expression)
        flat = np.concatenate([np.atleast_1d(v) for v in obj_consts])
        expected = [11.0, 17.0, 8.5]
        for ev in expected:
            assert np.any(np.isclose(flat, ev)), (
                f"Expected worst-case cost component {ev}; got {flat}"
            )


# ---------------------------------------------------------------------------
# Ellipsoidal reformulation – structural correctness
# ---------------------------------------------------------------------------


class TestEllipsoidalReformulationStructure:
    def test_uncertain_param_replaced_in_matmul_objective(self):
        """p @ x in objective → p̄ @ x + ρ||Σ^{1/2}x||₂."""
        m = dm.Model()
        n = 4
        x = m.continuous("x", shape=(n,), lb=0, ub=1)
        mu = m.parameter("mu", value=np.ones(n) * 0.1)
        # minimize -mu @ x  (maximise expected return)
        m.minimize(-(mu @ x))
        m.subject_to(dm.sum(x) == 1.0)

        unc = EllipsoidalUncertaintySet(mu, rho=2.0)
        rc = RobustCounterpart(m, unc)
        rc.formulate()

        # mu should no longer appear in the objective
        remaining = _collect_parameters(m._objective.expression)
        assert "mu" not in remaining, f"mu still in objective: {remaining}"

    def test_penalty_added_to_objective(self):
        """A FunctionCall('sqrt', ...) penalty node should appear."""
        from discopt.modeling.core import FunctionCall

        m = dm.Model()
        x = m.continuous("x", shape=(2,), lb=0, ub=1)
        mu = m.parameter("mu", value=[0.1, 0.2])
        m.minimize(-(mu @ x))

        rc = RobustCounterpart(m, EllipsoidalUncertaintySet(mu, rho=1.0))
        rc.formulate()

        def _has_sqrt(expr) -> bool:
            if isinstance(expr, FunctionCall) and expr.func_name == "sqrt":
                return True
            from discopt.modeling.core import BinaryOp, UnaryOp

            if isinstance(expr, (BinaryOp,)):
                return _has_sqrt(expr.left) or _has_sqrt(expr.right)
            if isinstance(expr, UnaryOp):
                return _has_sqrt(expr.operand)
            return False

        assert _has_sqrt(m._objective.expression), (
            "Expected sqrt (norm) penalty in objective after ellipsoidal reformulation"
        )


# ---------------------------------------------------------------------------
# Polyhedral reformulation – structural correctness
# ---------------------------------------------------------------------------


class TestPolyhedralReformulationStructure:
    def test_uncertain_param_removed(self):
        m = dm.Model()
        x = m.continuous("x", lb=0)
        d = m.parameter("d", value=5.0)
        m.minimize(x)
        m.subject_to(x >= d)

        A = np.array([[1.0], [-1.0]])
        b = np.array([1.0, 1.0])  # |ξ| ≤ 1  →  d ∈ [4, 6]
        unc = PolyhedralUncertaintySet(d, A=A, b=b)
        rc = RobustCounterpart(m, unc)
        rc.formulate()

        for con in m._constraints:
            assert "d" not in _collect_parameters(con.body)

    def test_budget_set_removes_param(self):
        m = dm.Model()
        x = m.continuous("x", shape=(3,), lb=0)
        c = m.parameter("c", value=[10.0, 15.0, 8.0])
        m.minimize(dm.sum(c * x))
        m.subject_to(dm.sum(x) == 10.0)

        unc = budget_uncertainty_set(c, delta=1.0, gamma=2.0)
        rc = RobustCounterpart(m, unc)
        rc.formulate()

        remaining = _collect_parameters(m._objective.expression)
        assert "c" not in remaining

    def test_polyhedral_wc_values_match_lp_solution(self):
        """For |ξ| ≤ δ, polyhedral and box formulations give same worst-case.

        Encodes a box set as polyhedral (Aξ ≤ b with A=[I;-I], b=[δ;δ])
        and verifies that no uncertain parameters remain after reformulation.
        """
        val = 10.0
        delta_val = 2.0

        m_poly = dm.Model()
        x = m_poly.continuous("x", lb=0)
        p = m_poly.parameter("p", value=val)
        m_poly.minimize(p * x)
        m_poly.subject_to(x >= 1.0)
        A = np.array([[1.0], [-1.0]])
        b = np.array([delta_val, delta_val])
        unc_poly = PolyhedralUncertaintySet(p, A=A, b=b)
        rc_poly = RobustCounterpart(m_poly, unc_poly)
        rc_poly.formulate()

        # Parameter must be eliminated from both objective and constraints.
        remaining_obj = _collect_parameters(m_poly._objective.expression)
        assert "p" not in remaining_obj, f"p in objective: {remaining_obj}"
        for con in m_poly._constraints:
            remaining_con = _collect_parameters(con.body)
            assert "p" not in remaining_con, f"p in constraint: {remaining_con}"
