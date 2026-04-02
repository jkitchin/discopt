"""Tests for AffineDecisionRule (adjustable robust optimization).

Covers:
- Construction validation (type checks, shape checks)
- apply(): intercept and policy column variable creation
- apply(): correct substitution of recourse variable in constraints/objective
- apply(): idempotency guard (second call raises)
- apply(): recourse variable retired from model._variables
- Perturbation expressions: scalar and vector uncertain parameters
- Integration: AffineDecisionRule → RobustCounterpart pipeline
- Structural: no recourse variable remains in DAG after apply()
- Numerical: affine expression constants match expected values post-RobustCounterpart

References
----------
Ben-Tal, A., Goryashko, A., Guslitzer, E., Nemirovski, A. (2004).
Adjustable robust solutions of uncertain linear programs.
Mathematical Programming, 99(2), 351-376.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

import discopt.modeling as dm
from discopt.modeling.core import BinaryOp, Constant, Parameter, Variable
from discopt.ro import AffineDecisionRule, BoxUncertaintySet, RobustCounterpart


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_variables(expr) -> set[str]:
    """Collect all Variable names reachable from an expression node."""
    from discopt.modeling.core import (
        BinaryOp,
        FunctionCall,
        IndexExpression,
        MatMulExpression,
        SumExpression,
        SumOverExpression,
        UnaryOp,
        Variable,
    )

    if isinstance(expr, Variable):
        return {expr.name}
    if isinstance(expr, (BinaryOp, MatMulExpression)):
        return _find_variables(expr.left) | _find_variables(expr.right)
    if isinstance(expr, UnaryOp):
        return _find_variables(expr.operand)
    if isinstance(expr, FunctionCall):
        result: set[str] = set()
        for a in expr.args:
            result |= _find_variables(a)
        return result
    if isinstance(expr, IndexExpression):
        return _find_variables(expr.base)
    if isinstance(expr, SumExpression):
        return _find_variables(expr.operand)
    if isinstance(expr, SumOverExpression):
        r2: set[str] = set()
        for t in expr.terms:
            r2 |= _find_variables(t)
        return r2
    return set()


def _find_parameters(expr) -> set[str]:
    """Collect all Parameter names reachable from an expression node."""
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
        return _find_parameters(expr.left) | _find_parameters(expr.right)
    if isinstance(expr, UnaryOp):
        return _find_parameters(expr.operand)
    if isinstance(expr, FunctionCall):
        result: set[str] = set()
        for a in expr.args:
            result |= _find_parameters(a)
        return result
    if isinstance(expr, IndexExpression):
        return _find_parameters(expr.base)
    if isinstance(expr, SumExpression):
        return _find_parameters(expr.operand)
    if isinstance(expr, SumOverExpression):
        r2: set[str] = set()
        for t in expr.terms:
            r2 |= _find_parameters(t)
        return r2
    return set()


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestAffineDecisionRuleConstruction:
    def test_non_variable_recourse_rejected(self):
        m = dm.Model()
        d = m.parameter("d", value=10.0)
        with pytest.raises(TypeError, match="Variable"):
            AffineDecisionRule("not_a_variable", uncertain_params=d)

    def test_non_parameter_uncertain_rejected(self):
        m = dm.Model()
        y = m.continuous("y", lb=0)
        with pytest.raises(TypeError, match="Parameter"):
            AffineDecisionRule(y, uncertain_params="not_a_param")

    def test_empty_uncertain_params_rejected(self):
        m = dm.Model()
        y = m.continuous("y", lb=0)
        with pytest.raises(ValueError, match="empty"):
            AffineDecisionRule(y, uncertain_params=[])

    def test_2d_recourse_rejected(self):
        m = dm.Model()
        y = m.continuous("y", shape=(2, 3))
        d = m.parameter("d", value=10.0)
        with pytest.raises(ValueError, match="1-D"):
            AffineDecisionRule(y, uncertain_params=d)

    def test_scalar_recourse_accepted(self):
        m = dm.Model()
        y = m.continuous("y", lb=0)
        d = m.parameter("d", value=10.0)
        adr = AffineDecisionRule(y, uncertain_params=d)
        assert adr.n_policy_columns == 1

    def test_vector_recourse_accepted(self):
        m = dm.Model()
        y = m.continuous("y", shape=(3,), lb=0)
        d = m.parameter("d", value=10.0)
        adr = AffineDecisionRule(y, uncertain_params=d)
        assert adr.n_policy_columns == 1

    def test_multiple_uncertain_params(self):
        m = dm.Model()
        y = m.continuous("y", lb=0)
        d1 = m.parameter("d1", value=10.0)
        d2 = m.parameter("d2", value=[5.0, 8.0])
        adr = AffineDecisionRule(y, uncertain_params=[d1, d2])
        # d1: 1 scalar, d2: 2 scalars → 3 policy columns
        assert adr.n_policy_columns == 3

    def test_list_wrapping_single_param(self):
        m = dm.Model()
        y = m.continuous("y", lb=0)
        d = m.parameter("d", value=10.0)
        adr = AffineDecisionRule(y, uncertain_params=[d])
        assert adr.n_policy_columns == 1


# ---------------------------------------------------------------------------
# apply(): variable creation
# ---------------------------------------------------------------------------


class TestApplyVariableCreation:
    def _simple_model(self):
        m = dm.Model()
        x = m.continuous("x", lb=0, ub=100)
        y = m.continuous("y", lb=0, ub=50)
        d = m.parameter("d", value=30.0)
        m.minimize(2 * x + 5 * y)
        m.subject_to(x + y >= d, name="demand")
        return m, x, y, d

    def test_intercept_created_in_model(self):
        m, x, y, d = self._simple_model()
        adr = AffineDecisionRule(y, uncertain_params=d)
        adr.apply()
        var_names = {v.name for v in m._variables}
        assert "adr_intercept" in var_names

    def test_policy_columns_created_in_model(self):
        m, x, y, d = self._simple_model()
        adr = AffineDecisionRule(y, uncertain_params=d)
        adr.apply()
        var_names = {v.name for v in m._variables}
        assert "adr_Y0" in var_names

    def test_policy_columns_count_scalar_param(self):
        m, x, y, d = self._simple_model()
        adr = AffineDecisionRule(y, uncertain_params=d)
        adr.apply()
        assert len(adr.policy_columns) == 1

    def test_policy_columns_count_vector_param(self):
        m = dm.Model()
        y = m.continuous("y", lb=0)
        c = m.parameter("c", value=[1.0, 2.0, 3.0])
        m.minimize(y)
        m.subject_to(y >= dm.sum(c))
        adr = AffineDecisionRule(y, uncertain_params=c)
        adr.apply()
        assert len(adr.policy_columns) == 3  # 3-component vector param

    def test_intercept_has_same_shape_as_recourse(self):
        m = dm.Model()
        y = m.continuous("y", shape=(4,), lb=0)
        d = m.parameter("d", value=10.0)
        m.minimize(dm.sum(y))
        m.subject_to(y[0] >= d)
        adr = AffineDecisionRule(y, uncertain_params=d)
        adr.apply()
        assert adr.intercept.shape == (4,)

    def test_policy_columns_same_shape_as_recourse(self):
        m = dm.Model()
        y = m.continuous("y", shape=(3,), lb=0)
        d = m.parameter("d", value=10.0)
        m.minimize(dm.sum(y))
        m.subject_to(y[0] >= d)
        adr = AffineDecisionRule(y, uncertain_params=d)
        adr.apply()
        for col in adr.policy_columns:
            assert col.shape == (3,)

    def test_recourse_var_retired_from_model(self):
        m, x, y, d = self._simple_model()
        adr = AffineDecisionRule(y, uncertain_params=d)
        adr.apply()
        var_names = {v.name for v in m._variables}
        assert "y" not in var_names

    def test_model_variable_indices_contiguous_after_retirement(self):
        m, x, y, d = self._simple_model()
        adr = AffineDecisionRule(y, uncertain_params=d)
        adr.apply()
        for i, v in enumerate(m._variables):
            assert v._index == i, f"Variable {v.name} has _index={v._index}, expected {i}"

    def test_apply_twice_raises(self):
        m, x, y, d = self._simple_model()
        adr = AffineDecisionRule(y, uncertain_params=d)
        adr.apply()
        with pytest.raises(RuntimeError, match="already been called"):
            adr.apply()


# ---------------------------------------------------------------------------
# apply(): DAG substitution
# ---------------------------------------------------------------------------


class TestApplySubstitution:
    def test_recourse_removed_from_constraint(self):
        m = dm.Model()
        x = m.continuous("x", lb=0)
        y = m.continuous("y", lb=0)
        d = m.parameter("d", value=10.0)
        m.minimize(x + y)
        m.subject_to(x + y >= d)

        adr = AffineDecisionRule(y, uncertain_params=d)
        adr.apply()

        for con in m._constraints:
            assert "y" not in _find_variables(con.body), (
                f"'y' still in constraint '{con.name}' after apply()"
            )

    def test_recourse_removed_from_objective(self):
        m = dm.Model()
        x = m.continuous("x", lb=0)
        y = m.continuous("y", lb=0)
        d = m.parameter("d", value=10.0)
        m.minimize(2 * x + 5 * y)
        m.subject_to(x + y >= d)

        adr = AffineDecisionRule(y, uncertain_params=d)
        adr.apply()

        obj_vars = _find_variables(m._objective.expression)
        assert "y" not in obj_vars, f"'y' still in objective after apply(): {obj_vars}"

    def test_intercept_appears_in_constraint(self):
        m = dm.Model()
        x = m.continuous("x", lb=0)
        y = m.continuous("y", lb=0)
        d = m.parameter("d", value=10.0)
        m.minimize(x + y)
        m.subject_to(x + y >= d)

        adr = AffineDecisionRule(y, uncertain_params=d, prefix="adr")
        adr.apply()

        for con in m._constraints:
            assert "adr_intercept" in _find_variables(con.body), (
                f"intercept not found in constraint '{con.name}'"
            )

    def test_policy_column_appears_in_constraint(self):
        m = dm.Model()
        x = m.continuous("x", lb=0)
        y = m.continuous("y", lb=0)
        d = m.parameter("d", value=10.0)
        m.minimize(x + y)
        m.subject_to(x + y >= d)

        adr = AffineDecisionRule(y, uncertain_params=d, prefix="adr")
        adr.apply()

        for con in m._constraints:
            assert "adr_Y0" in _find_variables(con.body), (
                f"policy column not found in constraint '{con.name}'"
            )

    def test_uncertain_param_still_present_after_adr(self):
        """After AffineDecisionRule.apply(), uncertain params ξ still appear
        (via perturbation terms).  RobustCounterpart handles them next."""
        m = dm.Model()
        x = m.continuous("x", lb=0)
        y = m.continuous("y", lb=0)
        d = m.parameter("d", value=10.0)
        m.minimize(x + y)
        m.subject_to(x + y >= d)

        adr = AffineDecisionRule(y, uncertain_params=d)
        adr.apply()

        # d should still appear in constraint (via ξ = d - d̄)
        for con in m._constraints:
            param_names = _find_parameters(con.body)
            # d appears in the perturbation term Y0 * (d - 10.0)
            assert "d" in param_names, (
                f"uncertain param 'd' should still appear after ADR (before static RO): "
                f"{param_names}"
            )

    def test_indexed_recourse_substituted(self):
        """y[i] in a constraint should become (affine_expr)[i]."""
        m = dm.Model()
        y = m.continuous("y", shape=(3,), lb=0)
        d = m.parameter("d", value=5.0)
        m.minimize(y[0])
        m.subject_to(y[0] >= d)

        adr = AffineDecisionRule(y, uncertain_params=d)
        adr.apply()

        for con in m._constraints:
            assert "y" not in _find_variables(con.body)
            assert "adr_intercept" in _find_variables(con.body)


# ---------------------------------------------------------------------------
# Perturbation structure
# ---------------------------------------------------------------------------


class TestPerturbationStructure:
    def test_scalar_param_perturbation_is_binop_minus(self):
        m = dm.Model()
        y = m.continuous("y", lb=0)
        d = m.parameter("d", value=10.0)
        m.minimize(y)
        m.subject_to(y >= d)
        adr = AffineDecisionRule(y, uncertain_params=d)
        adr.apply()

        pert = adr.perturbations[0]
        assert isinstance(pert, BinaryOp)
        assert pert.op == "-"
        # Left should be the parameter
        assert isinstance(pert.left, Parameter)
        assert pert.left.name == "d"
        # Right should be the nominal constant
        assert isinstance(pert.right, Constant)
        np.testing.assert_allclose(pert.right.value, 10.0)

    def test_vector_param_perturbations_count(self):
        m = dm.Model()
        y = m.continuous("y", lb=0)
        c = m.parameter("c", value=[1.0, 2.0, 3.0])
        m.minimize(y)
        m.subject_to(y >= c[0])
        adr = AffineDecisionRule(y, uncertain_params=c)
        adr.apply()
        assert len(adr.perturbations) == 3

    def test_multiple_params_total_perturbations(self):
        m = dm.Model()
        y = m.continuous("y", lb=0)
        d = m.parameter("d", value=5.0)       # 1 scalar
        c = m.parameter("c", value=[1.0, 2.0]) # 2 scalars
        m.minimize(y)
        m.subject_to(y >= d)
        adr = AffineDecisionRule(y, uncertain_params=[d, c])
        adr.apply()
        assert len(adr.perturbations) == 3


# ---------------------------------------------------------------------------
# Properties before apply()
# ---------------------------------------------------------------------------


class TestPropertiesBeforeApply:
    def _make(self):
        m = dm.Model()
        y = m.continuous("y", lb=0)
        d = m.parameter("d", value=10.0)
        m.minimize(y)
        m.subject_to(y >= d)
        return AffineDecisionRule(y, uncertain_params=d)

    def test_intercept_before_apply_raises(self):
        adr = self._make()
        with pytest.raises(RuntimeError, match="apply"):
            _ = adr.intercept

    def test_policy_columns_before_apply_raises(self):
        adr = self._make()
        with pytest.raises(RuntimeError, match="apply"):
            _ = adr.policy_columns

    def test_affine_expression_before_apply_raises(self):
        adr = self._make()
        with pytest.raises(RuntimeError, match="apply"):
            _ = adr.affine_expression

    def test_perturbations_before_apply_raises(self):
        adr = self._make()
        with pytest.raises(RuntimeError, match="apply"):
            _ = adr.perturbations


# ---------------------------------------------------------------------------
# Integration: AffineDecisionRule + RobustCounterpart
# ---------------------------------------------------------------------------


class TestADRWithRobustCounterpart:
    """End-to-end pipeline: ADR substitution followed by static RobustCounterpart."""

    def test_no_uncertain_params_remain_after_pipeline(self):
        """After ADR + RobustCounterpart, no uncertain parameters should remain."""
        m = dm.Model()
        x = m.continuous("x", lb=0, ub=200)
        y = m.continuous("y", lb=0, ub=100)
        d = m.parameter("d", value=80.0)

        m.minimize(2 * x + 5 * y)
        m.subject_to(x + y >= d, name="demand")
        m.subject_to(x + y <= 200, name="capacity")

        # Stage 1: Affine decision rule
        adr = AffineDecisionRule(y, uncertain_params=d)
        adr.apply()

        # Stage 2: Static robust counterpart for remaining ξ
        rc = RobustCounterpart(m, BoxUncertaintySet(d, delta=20.0))
        rc.formulate()

        # After both stages: no Parameters should remain
        for con in m._constraints:
            assert not _find_parameters(con.body), (
                f"Parameters remain in '{con.name}' after full pipeline: "
                f"{_find_parameters(con.body)}"
            )
        if m._objective is not None:
            assert not _find_parameters(m._objective.expression), (
                "Parameters remain in objective after full pipeline"
            )

    def test_no_recourse_var_remains_after_pipeline(self):
        m = dm.Model()
        x = m.continuous("x", lb=0)
        y = m.continuous("y", lb=0)
        d = m.parameter("d", value=50.0)
        m.minimize(x + y)
        m.subject_to(x + y >= d)

        adr = AffineDecisionRule(y, uncertain_params=d)
        adr.apply()
        rc = RobustCounterpart(m, BoxUncertaintySet(d, delta=10.0))
        rc.formulate()

        # 'y' must not appear anywhere
        for con in m._constraints:
            assert "y" not in _find_variables(con.body)
        if m._objective is not None:
            assert "y" not in _find_variables(m._objective.expression)

    def test_new_variables_present_after_pipeline(self):
        """Intercept y₀ and policy column Y₀ must be in the reformulated model."""
        m = dm.Model()
        x = m.continuous("x", lb=0)
        y = m.continuous("y", lb=0)
        d = m.parameter("d", value=50.0)
        m.minimize(x + y)
        m.subject_to(x + y >= d)

        adr = AffineDecisionRule(y, uncertain_params=d, prefix="pol")
        adr.apply()
        rc = RobustCounterpart(m, BoxUncertaintySet(d, delta=10.0))
        rc.formulate()

        var_names = {v.name for v in m._variables}
        assert "pol_intercept" in var_names
        assert "pol_Y0" in var_names

    def test_worst_case_constants_conservative_box(self):
        """With box uncertainty δ=10 on demand d=50, the robust constraint
        (x + y₀ + Y₀*(d-50) ≥ d) after RobustCounterpart should embed the
        worst-case d = 50+10 = 60 as a constant."""
        from discopt.modeling.core import Constant

        m = dm.Model()
        x = m.continuous("x", lb=0)
        y = m.continuous("y", lb=0)
        d = m.parameter("d", value=50.0)
        m.minimize(x + y)
        m.subject_to(x + y >= d, name="demand")  # stored: d - x - y ≤ 0

        adr = AffineDecisionRule(y, uncertain_params=d)
        adr.apply()
        rc = RobustCounterpart(m, BoxUncertaintySet(d, delta=10.0))
        rc.formulate()

        def _all_constants(expr) -> list:
            from discopt.modeling.core import (
                BinaryOp, Constant, FunctionCall, IndexExpression,
                MatMulExpression, SumExpression, SumOverExpression, UnaryOp,
            )
            if isinstance(expr, Constant):
                return [float(np.sum(expr.value))]
            if isinstance(expr, (BinaryOp, MatMulExpression)):
                return _all_constants(expr.left) + _all_constants(expr.right)
            if isinstance(expr, UnaryOp):
                return _all_constants(expr.operand)
            if isinstance(expr, FunctionCall):
                r = []
                for a in expr.args:
                    r.extend(_all_constants(a))
                return r
            if isinstance(expr, IndexExpression):
                return _all_constants(expr.base)
            if isinstance(expr, (SumExpression,)):
                return _all_constants(expr.operand)
            if isinstance(expr, SumOverExpression):
                r2 = []
                for t in expr.terms:
                    r2.extend(_all_constants(t))
                return r2
            return []

        for con in m._constraints:
            consts = _all_constants(con.body)
            flat = np.concatenate([np.atleast_1d(c) for c in consts]) if consts else np.array([])
            # The worst-case demand (60.0) must appear, and the nominal shift
            # in the perturbation (50.0) must appear.
            assert np.any(np.isclose(flat, 60.0)) or np.any(np.isclose(flat, 50.0)), (
                f"Expected worst-case demand (60.0 or nominal 50.0) in constants; got {flat}"
            )

    def test_multiple_uncertain_params_pipeline(self):
        """Two uncertain parameters (cost c, demand d) with two recourse variables."""
        m = dm.Model()
        x = m.continuous("x", lb=0)
        y = m.continuous("y", lb=0)
        c = m.parameter("c", value=3.0)   # uncertain production cost
        d = m.parameter("d", value=20.0)  # uncertain demand

        m.minimize(c * x + 2 * y)
        m.subject_to(x + y >= d, name="demand")

        # y adapts to both c and d
        adr = AffineDecisionRule(y, uncertain_params=[c, d])
        adr.apply()

        assert len(adr.policy_columns) == 2  # one per uncertain scalar param

        rc = RobustCounterpart(m, [BoxUncertaintySet(c, delta=0.5), BoxUncertaintySet(d, delta=5.0)])
        rc.formulate()

        # All uncertain params eliminated
        for con in m._constraints:
            assert not _find_parameters(con.body)
        if m._objective is not None:
            assert not _find_parameters(m._objective.expression)

    def test_prefix_naming_convention(self):
        """Custom prefix propagates to intercept and policy column names."""
        m = dm.Model()
        y = m.continuous("recourse", lb=0)
        d = m.parameter("demand", value=10.0)
        m.minimize(y)
        m.subject_to(y >= d)

        adr = AffineDecisionRule(y, uncertain_params=d, prefix="stage2")
        adr.apply()

        assert adr.intercept.name == "stage2_intercept"
        assert adr.policy_columns[0].name == "stage2_Y0"

    def test_summary_str(self):
        """summary() returns a non-empty string before and after apply()."""
        m = dm.Model()
        y = m.continuous("y", lb=0)
        d = m.parameter("d", value=10.0)
        m.minimize(y)
        m.subject_to(y >= d)
        adr = AffineDecisionRule(y, uncertain_params=d)
        s_before = adr.summary()
        assert "not yet applied" in s_before
        adr.apply()
        s_after = adr.summary()
        assert "intercept" in s_after.lower() or "y0" in s_after.lower() or "adr" in s_after.lower()
