"""Tests for GDP (Generalized Disjunctive Programming) big-M reformulation.

Tests cover indicator constraints, disjunctive constraints, SOS1/SOS2
constraints, big-M computation, and end-to-end solve correctness.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.gdp_reformulate import (
    _bound_expression,
    _collect_variables,
    _compute_big_m,
    _extract_disjunct_bounds,
    _is_linear,
    _reformulate_indicator,
    _substitute_vars,
    reformulate_gdp,
)
from discopt.modeling.core import (
    Constraint,
    VarType,
    _IndicatorConstraint,
)

# ── Model with no GDP constraints passes through unchanged ──


class TestNoGDP:
    def test_no_gdp_returns_same_model(self):
        m = dm.Model("plain")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)
        m.subject_to(x >= 1)

        result = reformulate_gdp(m)
        assert result is m  # identity check -- no copy

    def test_empty_model(self):
        m = dm.Model("empty")
        x = m.continuous("x", lb=0, ub=5)
        m.minimize(x)
        result = reformulate_gdp(m)
        assert result is m


# ── Interval arithmetic / big-M computation ──


class TestBoundExpression:
    def test_variable_bounds(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=-3, ub=7)
        lo, hi = _bound_expression(x, m)
        assert lo == -3.0
        assert hi == 7.0

    def test_constant(self):
        from discopt.modeling.core import Constant

        m = dm.Model("t")
        c = Constant(5.0)
        lo, hi = _bound_expression(c, m)
        assert lo == 5.0
        assert hi == 5.0

    def test_addition(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=1, ub=4)
        y = m.continuous("y", lb=2, ub=6)
        expr = x + y
        lo, hi = _bound_expression(expr, m)
        assert lo == pytest.approx(3.0)
        assert hi == pytest.approx(10.0)

    def test_subtraction(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=1, ub=4)
        y = m.continuous("y", lb=2, ub=6)
        expr = x - y
        lo, hi = _bound_expression(expr, m)
        assert lo == pytest.approx(-5.0)
        assert hi == pytest.approx(2.0)

    def test_multiplication(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=-2, ub=3)
        y = m.continuous("y", lb=1, ub=4)
        expr = x * y
        lo, hi = _bound_expression(expr, m)
        assert lo == pytest.approx(-8.0)
        assert hi == pytest.approx(12.0)

    def test_power_squared(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=-3, ub=5)
        expr = x**2
        lo, hi = _bound_expression(expr, m)
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(25.0)

    def test_indexed_variable(self):
        m = dm.Model("t")
        x = m.continuous("x", shape=(3,), lb=np.array([1, 2, 3]), ub=np.array([4, 5, 6]))
        expr = x[1]
        lo, hi = _bound_expression(expr, m)
        assert lo == pytest.approx(2.0)
        assert hi == pytest.approx(5.0)


class TestComputeBigM:
    def test_le_constraint(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        # body = x - 5 (from x <= 5 => x - 5 <= 0)
        con = Constraint(body=x - dm.core.Constant(5.0), sense="<=", rhs=0.0)
        M = _compute_big_m(con, m)
        # Upper bound of body = 10 - 5 = 5, so M ~= 5 * 1.01
        assert M == pytest.approx(5.0 * 1.01)

    def test_ge_constraint(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        # body = x - 3 (from x >= 3 => x - 3 >= 0)
        con = Constraint(body=x - dm.core.Constant(3.0), sense=">=", rhs=0.0)
        M = _compute_big_m(con, m)
        # Lower bound of body = 0 - 3 = -3, so M = -(-3) * 1.01 = 3.03
        assert M == pytest.approx(3.0 * 1.01)

    def test_infinite_bounds_use_default(self):
        m = dm.Model("t")
        x = m.continuous("x")  # lb=-1e20, ub=1e20
        con = Constraint(body=x, sense="<=", rhs=0.0)
        M = _compute_big_m(con, m)
        # Should fallback to default 1e4
        assert M == pytest.approx(1e4 * 1.01)


# ── Indicator constraint reformulation ──


class TestIndicatorReformulation:
    def test_simple_le_indicator(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        con = Constraint(body=x - dm.core.Constant(5.0), sense="<=", rhs=0.0)
        ic = _IndicatorConstraint(indicator=y, constraint=con, active_value=1)

        new_cons = _reformulate_indicator(ic, m)
        assert len(new_cons) == 1
        assert new_cons[0].sense == "<="

    def test_simple_ge_indicator(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        con = Constraint(body=x - dm.core.Constant(3.0), sense=">=", rhs=0.0)
        ic = _IndicatorConstraint(indicator=y, constraint=con, active_value=1)

        new_cons = _reformulate_indicator(ic, m)
        assert len(new_cons) == 1
        assert new_cons[0].sense == ">="

    def test_eq_indicator_produces_two_constraints(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        con = Constraint(body=x - dm.core.Constant(5.0), sense="==", rhs=0.0, name="eq")
        ic = _IndicatorConstraint(indicator=y, constraint=con, active_value=1)

        new_cons = _reformulate_indicator(ic, m)
        assert len(new_cons) == 2
        assert new_cons[0].sense == "<="
        assert new_cons[1].sense == ">="

    def test_active_value_zero(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        con = Constraint(body=x - dm.core.Constant(5.0), sense="<=", rhs=0.0)
        ic = _IndicatorConstraint(indicator=y, constraint=con, active_value=0)

        new_cons = _reformulate_indicator(ic, m)
        assert len(new_cons) == 1
        # When y=0 constraint should be active (relaxed when y=1)


class TestReformulateGDP:
    def test_single_indicator(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        m.minimize(x)
        m.if_then(y, [x <= 5])

        new_m = reformulate_gdp(m)
        assert new_m is not m
        assert all(isinstance(c, Constraint) for c in new_m._constraints)
        assert new_m._objective is m._objective

    def test_multiple_indicators(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y", shape=(2,))
        m.minimize(x)
        m.if_then(y[0], [x <= 5])
        m.if_then(y[1], [x >= 2])

        new_m = reformulate_gdp(m)
        # Original has 2 _IndicatorConstraint objects
        # Each produces 1 standard Constraint
        assert all(isinstance(c, Constraint) for c in new_m._constraints)
        assert len(new_m._constraints) == 2

    def test_mixed_regular_and_gdp(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        m.minimize(x)
        m.subject_to(x >= 0)  # regular
        m.if_then(y, [x <= 5])  # GDP

        new_m = reformulate_gdp(m)
        # 1 regular + 1 from indicator = 2 total
        assert len(new_m._constraints) == 2
        assert all(isinstance(c, Constraint) for c in new_m._constraints)

    def test_variables_preserved(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        m.minimize(x)
        m.if_then(y, [x <= 5])

        new_m = reformulate_gdp(m)
        # Original vars should be in new model
        assert x in new_m._variables
        assert y in new_m._variables


# ── Disjunctive constraint reformulation ──


class TestDisjunctionReformulation:
    def test_two_disjuncts(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=20)
        m.minimize(x)
        m.either_or(
            [[x <= 5], [x >= 15]],
            name="modes",
        )

        new_m = reformulate_gdp(m)
        assert all(isinstance(c, Constraint) for c in new_m._constraints)
        # Should have: 1 selector sum ==, 2 big-M constraints (one per disjunct)
        assert len(new_m._constraints) == 3
        # Two auxiliary binaries added
        n_aux = len(new_m._variables) - len(m._variables)
        assert n_aux == 2

    def test_three_disjuncts(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=30)
        m.minimize(x)
        m.either_or(
            [[x <= 5], [x >= 10, x <= 15], [x >= 25]],
            name="three_modes",
        )

        new_m = reformulate_gdp(m)
        # 1 selector sum + (1 + 2 + 1) = 5 total constraints
        assert len(new_m._constraints) == 5
        # 3 auxiliary binaries
        n_aux = len(new_m._variables) - len(m._variables)
        assert n_aux == 3

    def test_disjunction_with_eq_constraint(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)
        m.either_or(
            [[x <= 3], [Constraint(body=x - dm.core.Constant(7.0), sense="==", rhs=0.0)]],
            name="eq_disj",
        )

        new_m = reformulate_gdp(m)
        # 1 selector + 1 LE + 2 (LE + GE for ==) = 4
        assert len(new_m._constraints) == 4


# ── SOS constraint reformulation ──


class TestSOSReformulation:
    def test_sos1_basic(self):
        m = dm.Model("t")
        x = m.continuous("x", shape=(3,), lb=0, ub=10)
        m.minimize(x[0] + x[1] + x[2])
        m.sos1([x[0], x[1], x[2]], name="set1")

        new_m = reformulate_gdp(m)
        assert all(isinstance(c, Constraint) for c in new_m._constraints)
        # 3 vars: 3 upper-bound linking + 1 sum constraint = 4
        # (No lower-bound linking since lb=0)
        assert len(new_m._constraints) == 4
        # 3 auxiliary binaries
        n_aux = len(new_m._variables) - len(m._variables)
        assert n_aux == 3

    def test_sos1_with_negative_bounds(self):
        m = dm.Model("t")
        x = m.continuous("x", shape=(2,), lb=-5, ub=5)
        m.minimize(x[0] + x[1])
        m.sos1([x[0], x[1]], name="neg")

        new_m = reformulate_gdp(m)
        # 2 upper-bound + 2 lower-bound + 1 sum = 5
        assert len(new_m._constraints) == 5

    def test_sos2_basic(self):
        m = dm.Model("t")
        x = m.continuous("x", shape=(4,), lb=0, ub=10)
        m.minimize(x[0] + x[1] + x[2] + x[3])
        m.sos2([x[0], x[1], x[2], x[3]], name="adj")

        new_m = reformulate_gdp(m)
        # 4 upper-bound + 1 sum<=2 + non-adjacency pairs
        # Non-adjacent: (0,2), (0,3), (1,3) = 3 pairs
        # Total: 4 + 1 + 3 = 8
        assert len(new_m._constraints) == 8
        # 4 auxiliary binaries
        n_aux = len(new_m._variables) - len(m._variables)
        assert n_aux == 4


# ── End-to-end solve tests ──


@pytest.mark.slow
class TestEndToEndSolve:
    def test_indicator_solve_active(self):
        """If y=1 then x <= 5; minimize x+y => should set y=0, x at lower bound."""
        m = dm.Model("ind_active")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        m.minimize(x + 10 * y)
        m.if_then(y, [x <= 5])
        m.subject_to(x >= 3)

        result = m.solve()
        assert result.status == "optimal"
        # y=0 is cheaper (saves 10), x=3 satisfies x>=3
        assert result.x["y"] == pytest.approx(0.0, abs=1e-3)
        assert result.x["x"] == pytest.approx(3.0, abs=1e-3)

    def test_indicator_solve_forced(self):
        """Force y=1 and check indicator constraint is respected."""
        m = dm.Model("ind_forced")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        m.minimize(x)
        m.if_then(y, [x <= 5])
        m.subject_to(y >= 1)  # force y=1

        result = m.solve()
        assert result.status == "optimal"
        # y must be 1, so x <= 5 is active; minimize x => x = 0
        assert result.x["y"] == pytest.approx(1.0, abs=1e-3)
        assert result.x["x"] == pytest.approx(0.0, abs=1e-3)

    def test_disjunction_solve(self):
        """Either x <= 5 or x >= 15; minimize x => x should be at lower bound."""
        m = dm.Model("disj_solve")
        x = m.continuous("x", lb=0, ub=20)
        m.minimize(x)
        m.either_or(
            [[x <= 5], [x >= 15]],
            name="modes",
        )

        result = m.solve()
        assert result.status == "optimal"
        # First disjunct allows x in [0, 5], which is cheaper => x = 0
        assert result.objective == pytest.approx(0.0, abs=1e-2)

    def test_disjunction_forced_second(self):
        """Either x <= 5 or x >= 15; maximize -x with x >= 16 => forces second."""
        m = dm.Model("disj_forced")
        x = m.continuous("x", lb=0, ub=20)
        m.minimize(x)
        m.either_or(
            [[x <= 5], [x >= 15]],
            name="modes",
        )
        m.subject_to(x >= 16)

        result = m.solve()
        assert result.status == "optimal"
        # Must use second disjunct: x >= 16 forces x = 16
        assert result.x["x"] == pytest.approx(16.0, abs=1e-2)

    def test_sos1_solve(self):
        """SOS1: at most one of x1, x2, x3 nonzero. Minimize -x1 - 2*x2 - x3."""
        m = dm.Model("sos1_solve")
        x1 = m.continuous("x1", lb=0, ub=10)
        x2 = m.continuous("x2", lb=0, ub=10)
        x3 = m.continuous("x3", lb=0, ub=10)
        m.minimize(-x1 - 2 * x2 - x3)
        m.sos1([x1, x2, x3], name="s1")

        result = m.solve()
        assert result.status == "optimal"
        # Best: x2 = 10, rest = 0 => obj = -20
        assert result.objective == pytest.approx(-20.0, abs=0.5)
        assert result.x["x2"] == pytest.approx(10.0, abs=0.5)

    def test_no_gdp_solve_unchanged(self):
        """A plain MINLP should solve correctly without GDP overhead."""
        m = dm.Model("plain")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        m.minimize(x + 5 * y)
        m.subject_to(x >= 1)

        result = m.solve()
        assert result.status == "optimal"
        assert result.x["x"] == pytest.approx(1.0, abs=1e-3)
        assert result.x["y"] == pytest.approx(0.0, abs=1e-3)


# ── Additional reformulation correctness tests ──


class TestReformulationCorrectness:
    def test_indicator_constraint_name_propagation(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        m.minimize(x)
        m.if_then(y, [x <= 5], name="test_ind")

        new_m = reformulate_gdp(m)
        # Named constraints should preserve names
        named = [c for c in new_m._constraints if c.name is not None]
        assert len(named) >= 1

    def test_disjunction_selector_sum(self):
        """Verify the selector sum == 1 constraint is present."""
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=20)
        m.minimize(x)
        m.either_or([[x <= 5], [x >= 15]], name="sel")

        new_m = reformulate_gdp(m)
        eq_cons = [c for c in new_m._constraints if c.sense == "=="]
        assert len(eq_cons) == 1  # selector sum == 1

    def test_multiple_constraints_in_if_then(self):
        """if_then with multiple then-constraints."""
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        m.minimize(x)
        m.if_then(y, [x <= 5, x >= 2])

        new_m = reformulate_gdp(m)
        # 2 indicator constraints => 2 standard constraints
        assert len(new_m._constraints) == 2

    def test_parameters_preserved(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        p = m.parameter("p", value=5.0)
        m.minimize(x)
        m.if_then(y, [x <= 5])

        new_m = reformulate_gdp(m)
        assert len(new_m._parameters) == 1
        assert new_m._parameters[0] is p

    def test_model_name_preserved(self):
        m = dm.Model("my_model")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        m.minimize(x)
        m.if_then(y, [x <= 5])

        new_m = reformulate_gdp(m)
        assert new_m.name == "my_model"

    def test_auxiliary_variables_are_binary(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=20)
        m.minimize(x)
        m.either_or([[x <= 5], [x >= 15]], name="modes")

        new_m = reformulate_gdp(m)
        orig_vars = set(id(v) for v in m._variables)
        aux_vars = [v for v in new_m._variables if id(v) not in orig_vars]
        assert len(aux_vars) == 2
        for v in aux_vars:
            assert v.var_type == VarType.BINARY
            assert float(v.lb) == 0.0
            assert float(v.ub) == 1.0

    def test_big_m_finite(self):
        """All big-M values should be finite."""
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=100)
        con = Constraint(body=x - dm.core.Constant(50.0), sense="<=", rhs=0.0)
        M = _compute_big_m(con, m)
        assert np.isfinite(M)
        assert M > 0

    def test_big_m_tightness(self):
        """Big-M from bounds should be tighter than default when bounds are finite."""
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=5)
        con = Constraint(body=x - dm.core.Constant(3.0), sense="<=", rhs=0.0)
        M = _compute_big_m(con, m)
        # Upper bound of (x - 3) is 5 - 3 = 2, so M should be ~2.02
        assert M < 100  # much tighter than default 1e4

    def test_sos2_non_adjacency_count(self):
        """SOS2 with 5 vars should have correct number of non-adjacency constraints."""
        m = dm.Model("t")
        x = m.continuous("x", shape=(5,), lb=0, ub=10)
        m.minimize(x[0])
        m.sos2([x[0], x[1], x[2], x[3], x[4]], name="s2")

        new_m = reformulate_gdp(m)
        # Non-adjacent pairs for n=5: (0,2), (0,3), (0,4), (1,3), (1,4), (2,4) = 6
        # Plus 5 upper-bound + 1 sum constraint = 12 total
        le_cons = [c for c in new_m._constraints if c.sense == "<="]
        # 5 ub + 1 sum + 6 nonadj = 12 LE constraints
        assert len(le_cons) == 12


# ── Hull reformulation helpers ──


class TestHullHelpers:
    """Tests for hull reformulation helper functions."""

    # -- _collect_variables --

    def test_collect_variables_simple(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        expr = x + y * 2
        result = _collect_variables(expr)
        assert set(result.keys()) == {"x", "y"}
        assert result["x"] is x
        assert result["y"] is y

    def test_collect_variables_nested(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        z = m.continuous("z", lb=0, ub=10)
        expr = dm.exp(x) + y**2 - z
        result = _collect_variables(expr)
        assert set(result.keys()) == {"x", "y", "z"}

    def test_collect_variables_constant_only(self):
        from discopt.modeling.core import Constant

        expr = Constant(5.0)
        result = _collect_variables(expr)
        assert result == {}

    def test_collect_variables_index_expr(self):
        m = dm.Model("t")
        x = m.continuous("x", shape=(3,), lb=0, ub=10)
        expr = x[0] + x[1]
        result = _collect_variables(expr)
        # Both index expressions refer to the same base variable x
        assert set(result.keys()) == {"x"}
        assert result["x"] is x

    # -- _is_linear --

    def test_is_linear_true(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        expr = 2 * x + 3 * y - 5
        assert _is_linear(expr) is True

    def test_is_linear_false_bilinear(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        expr = x * y
        assert _is_linear(expr) is False

    def test_is_linear_false_nonlinear(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        expr = dm.exp(x)
        assert _is_linear(expr) is False

    # -- _substitute_vars --

    def test_substitute_vars_simple(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        v = m.continuous("v", lb=0, ub=10)
        expr = x + dm.core.Constant(3.0)
        result = _substitute_vars(expr, {"x": v})
        # The left child of the addition should now be v
        assert isinstance(result, dm.core.BinaryOp)
        assert result.left is v

    def test_substitute_vars_preserves_constants(self):
        c = dm.core.Constant(42.0)
        result = _substitute_vars(c, {"x": dm.core.Constant(0.0)})
        assert result is c  # identity — unchanged

    # -- _extract_disjunct_bounds --

    def test_extract_disjunct_bounds_simple(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=0, ub=10)
        disjunct = [
            Constraint(body=x, sense="<=", rhs=5.0),
            Constraint(body=x, sense=">=", rhs=2.0),
        ]
        result = _extract_disjunct_bounds(disjunct, m)
        assert result["x"] == (2.0, 5.0)

    def test_extract_disjunct_bounds_tightens_global(self):
        m = dm.Model("t")
        x = m.continuous("x", lb=-100, ub=100)
        disjunct = [Constraint(body=x, sense="<=", rhs=3.0)]
        result = _extract_disjunct_bounds(disjunct, m)
        assert result["x"] == (-100.0, 3.0)


# ── Logical propositions ──


class TestLogicalPropositions:
    def test_at_least(self):
        m = dm.Model("at_least")
        y = [m.binary(f"y{i}") for i in range(3)]
        m.minimize(sum(y))
        m.at_least(2, y)
        r = m.solve()
        assert r.status == "optimal"
        active = sum(int(np.round(r.x[v.name])) for v in y)
        assert active >= 2

    def test_at_most(self):
        m = dm.Model("at_most")
        y = [m.binary(f"y{i}") for i in range(3)]
        m.maximize(sum(y))
        m.at_most(1, y)
        r = m.solve()
        assert r.status == "optimal"
        active = sum(int(np.round(r.x[v.name])) for v in y)
        assert active <= 1

    def test_exactly(self):
        m = dm.Model("exactly")
        y = [m.binary(f"y{i}") for i in range(4)]
        m.minimize(sum(y))
        m.exactly(2, y)
        r = m.solve()
        assert r.status == "optimal"
        active = sum(int(np.round(r.x[v.name])) for v in y)
        assert active == 2

    def test_implies(self):
        m = dm.Model("implies")
        x = m.continuous("x", lb=0, ub=10)
        y1 = m.binary("y1")
        y2 = m.binary("y2")
        # Force y1=1 via constraint, then implies should force y2=1
        m.minimize(x)
        m.subject_to(y1 >= 1)
        m.implies(y1, y2)
        r = m.solve()
        assert r.status == "optimal"
        assert r.x["y1"] == pytest.approx(1.0, abs=1e-3)
        assert r.x["y2"] == pytest.approx(1.0, abs=1e-3)

    def test_iff(self):
        m = dm.Model("iff")
        x = m.continuous("x", lb=0, ub=10)
        y1 = m.binary("y1")
        y2 = m.binary("y2")
        m.minimize(x)
        m.subject_to(y1 >= 1)
        m.iff(y1, y2)
        r = m.solve()
        assert r.status == "optimal"
        assert r.x["y1"] == pytest.approx(1.0, abs=1e-3)
        assert r.x["y2"] == pytest.approx(1.0, abs=1e-3)

    def test_implies_chain(self):
        m = dm.Model("chain")
        x = m.continuous("x", lb=0, ub=10)
        y1 = m.binary("y1")
        y2 = m.binary("y2")
        y3 = m.binary("y3")
        m.minimize(x)
        m.subject_to(y1 >= 1)
        m.implies(y1, y2)
        m.implies(y2, y3)
        r = m.solve()
        assert r.status == "optimal"
        assert r.x["y1"] == pytest.approx(1.0, abs=1e-3)
        assert r.x["y3"] == pytest.approx(1.0, abs=1e-3)

    def test_at_least_validates_binary(self):
        m = dm.Model("val")
        x = m.continuous("x", lb=0, ub=1)
        m.minimize(x)
        with pytest.raises(ValueError, match="requires binary"):
            m.at_least(1, [x])

    def test_implies_validates_binary(self):
        m = dm.Model("val")
        y = m.binary("y")
        z = m.integer("z", lb=0, ub=1)
        m.minimize(y)
        with pytest.raises(ValueError, match="requires binary"):
            m.implies(y, z)

    def test_at_least_with_indexed_binary(self):
        m = dm.Model("idx")
        y = m.binary("y", shape=(3,))
        m.minimize(sum(y[i] for i in range(3)))
        m.at_least(1, [y[0], y[1], y[2]])
        r = m.solve()
        assert r.status == "optimal"
        active = sum(int(np.round(r.x["y"][i])) for i in range(3))
        assert active >= 1


# ── Hull reformulation tests ──


class TestHullReformulation:
    """Tests for convex hull reformulation of disjunctions."""

    def test_hull_two_disjuncts_structure(self):
        """Verify disaggregated vars, aggregation, bound linking, selector."""
        m = dm.Model("hull2")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)
        m.either_or([[x <= 3], [x >= 7]], name="modes")

        new_m = reformulate_gdp(m, method="hull")
        assert all(isinstance(c, Constraint) for c in new_m._constraints)

        orig_var_ids = {id(v) for v in m._variables}
        aux_vars = [v for v in new_m._variables if id(v) not in orig_var_ids]
        # 2 selectors + 2 disaggregated (1 var * 2 disjuncts)
        assert len(aux_vars) == 4

        # Selector sum == 1
        eq_cons = [c for c in new_m._constraints if c.sense == "=="]
        assert len(eq_cons) >= 1  # selector + aggregation

        # Bound linking: 2 disjuncts * 1 var * 2 (ub + lb) = 4
        hull_ub = [c for c in new_m._constraints if c.name and "_hull_ub_" in c.name]
        hull_lb = [c for c in new_m._constraints if c.name and "_hull_lb_" in c.name]
        assert len(hull_ub) == 2
        assert len(hull_lb) == 2

    def test_hull_three_disjuncts(self):
        """3 disjuncts with 2 variables => 6 disaggregated variables."""
        m = dm.Model("hull3")
        x = m.continuous("x", lb=0, ub=30)
        y = m.continuous("y", lb=0, ub=30)
        m.minimize(x + y)
        m.either_or(
            [[x <= 5, y <= 5], [x >= 10, y >= 10], [x >= 25]],
            name="tri",
        )

        new_m = reformulate_gdp(m, method="hull")
        orig_var_ids = {id(v) for v in m._variables}
        aux_vars = [v for v in new_m._variables if id(v) not in orig_var_ids]
        # 3 selectors + disagg: disjuncts 0 and 1 have x,y; disjunct 2 has x
        # _collect_variables scans all disjuncts, so all 3 get both x and y
        n_selectors = 3
        n_disagg = 2 * 3  # 2 vars * 3 disjuncts
        assert len(aux_vars) == n_selectors + n_disagg

    def test_hull_linear_constraint(self):
        """Linear constraints produce perspective-form constraints."""
        m = dm.Model("hull_lin")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)
        m.either_or([[x <= 3], [x >= 7]], name="lin")

        new_m = reformulate_gdp(m, method="hull")
        # Constraint names with _hull_lin_d should exist (one per disjunct)
        d_cons = [c for c in new_m._constraints if c.name and "_hull_lin_d" in c.name]
        assert len(d_cons) == 2

    def test_hull_vs_bigm_tighter_relaxation(self):
        """Hull should produce tighter or equal relaxation compared to big-M.

        Build either_or([[x<=3],[x>=7]]) with x in [0,10], minimize x.
        Both methods should give optimal x=0, but hull reformulation has
        more structural constraints (disaggregated vars + bound linking).
        """
        # Big-M version
        m_bm = dm.Model("bigm")
        x_bm = m_bm.continuous("x", lb=0, ub=10)
        m_bm.minimize(x_bm)
        m_bm.either_or([[x_bm <= 3], [x_bm >= 7]], name="d")
        r_bm = m_bm.solve()

        # Hull version
        m_hull = dm.Model("hull")
        x_hull = m_hull.continuous("x", lb=0, ub=10)
        m_hull.minimize(x_hull)
        m_hull.either_or([[x_hull <= 3], [x_hull >= 7]], name="d")
        r_hull = m_hull.solve(gdp_method="hull")

        # Both should find optimal
        assert r_bm.status == "optimal"
        assert r_hull.status == "optimal"
        # Both should find x=0 (first disjunct)
        assert r_bm.objective == pytest.approx(0.0, abs=0.1)
        assert r_hull.objective == pytest.approx(0.0, abs=0.1)

        # Hull has more variables (disaggregated)
        m_hull_ref = reformulate_gdp(m_hull, method="hull")
        m_bm_ref = reformulate_gdp(m_bm, method="big-m")
        assert len(m_hull_ref._variables) > len(m_bm_ref._variables)

    def test_hull_solve_correct_objective(self):
        """MINLP with both methods should give matching objectives."""
        m = dm.Model("hull_obj")
        x = m.continuous("x", lb=0, ub=20)
        y = m.binary("y_int")
        m.minimize(x + 5 * y)
        m.either_or([[x <= 5], [x >= 15]], name="d")
        m.subject_to(x >= 1)

        r_bm = m.solve(gdp_method="big-m")
        r_hull = m.solve(gdp_method="hull")
        assert r_bm.status == "optimal"
        assert r_hull.status == "optimal"
        assert r_bm.objective == pytest.approx(r_hull.objective, abs=0.5)

    def test_hull_nonlinear_constraint(self):
        """x**2 <= 5 in disjunct => perspective form, correct solve."""
        m = dm.Model("hull_nl")
        x = m.continuous("x", lb=-5, ub=5)
        m.minimize(x)
        # Disjunct 0: x**2 <= 5 (nonlinear), Disjunct 1: x >= 3
        m.either_or(
            [
                [Constraint(body=x**2 - dm.core.Constant(5.0), sense="<=", rhs=0.0)],
                [x >= 3],
            ],
            name="nl",
        )

        r = m.solve(gdp_method="hull")
        assert r.status == "optimal"
        # First disjunct allows x in [-sqrt(5), sqrt(5)] ~ [-2.24, 2.24]
        # min x => x ~ -2.24
        assert r.x["x"] == pytest.approx(-np.sqrt(5), abs=0.5)

    def test_hull_method_parameter_endtoend(self):
        """m.solve(gdp_method='hull') works end-to-end."""
        m = dm.Model("e2e")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)
        m.either_or([[x <= 3], [x >= 7]], name="d")

        r = m.solve(gdp_method="hull")
        assert r.status == "optimal"
        assert r.objective == pytest.approx(0.0, abs=0.1)

    def test_hull_no_gdp_passthrough(self):
        """No GDP constraints => original model returned unchanged."""
        m = dm.Model("no_gdp")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)
        m.subject_to(x >= 1)

        result = reformulate_gdp(m, method="hull")
        assert result is m

    def test_hull_overlapping_disjuncts(self):
        """[[x<=5],[x>=3]] with overlapping feasible regions."""
        m = dm.Model("overlap")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)
        m.either_or([[x <= 5], [x >= 3]], name="ov")

        r = m.solve(gdp_method="hull")
        assert r.status == "optimal"
        # First disjunct: x in [0,5], min x=0
        assert r.objective == pytest.approx(0.0, abs=0.1)

    def test_hull_disjunct_local_bounds(self):
        """Verify disaggregated variable gets disjunct-local bounds."""
        m = dm.Model("local_bds")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)
        # Disjunct 0: x <= 3, disjunct 1: x >= 7
        m.either_or([[x <= 3], [x >= 7]], name="bd")

        new_m = reformulate_gdp(m, method="hull")
        # Find disaggregated variables
        disagg_vars = [v for v in new_m._variables if v.name.startswith("_hull_bd_v_x_")]
        assert len(disagg_vars) == 2
        # Disjunct 0 (x<=3): dlb=0, dub=3 => v bounds [0, 3]
        v0 = [v for v in disagg_vars if v.name.endswith("_0")][0]
        assert float(v0.ub) == pytest.approx(3.0)
        # Disjunct 1 (x>=7): dlb=7, dub=10 => v bounds [0, 10]
        v1 = [v for v in disagg_vars if v.name.endswith("_1")][0]
        assert float(v1.ub) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# BooleanVar + Propositional Logic
# ---------------------------------------------------------------------------


class TestBooleanVar:
    def test_scalar_creation(self):
        m = dm.Model("test")
        Y = m.boolean("y")
        assert isinstance(Y, dm.BooleanVar)

    def test_array_creation(self):
        m = dm.Model("test")
        Y = m.boolean("y", shape=(3,))
        assert isinstance(Y, dm.BooleanVarArray)
        assert len(Y) == 3
        assert isinstance(Y[0], dm.BooleanVar)
        assert isinstance(Y[2], dm.BooleanVar)

    def test_and_operator(self):
        m = dm.Model("test")
        Y = m.boolean("y", shape=(2,))
        from discopt.modeling.core import LogicalAnd

        expr = Y[0] & Y[1]
        assert isinstance(expr, LogicalAnd)

    def test_or_operator(self):
        m = dm.Model("test")
        Y = m.boolean("y", shape=(2,))
        from discopt.modeling.core import LogicalOr

        expr = Y[0] | Y[1]
        assert isinstance(expr, LogicalOr)

    def test_not_operator(self):
        m = dm.Model("test")
        Y = m.boolean("y")
        from discopt.modeling.core import LogicalNot

        expr = ~Y
        assert isinstance(expr, LogicalNot)

    def test_implies_method(self):
        m = dm.Model("test")
        Y = m.boolean("y", shape=(2,))
        from discopt.modeling.core import LogicalImplies

        expr = Y[0].implies(Y[1])
        assert isinstance(expr, LogicalImplies)

    def test_equivalent_to_method(self):
        m = dm.Model("test")
        Y = m.boolean("y", shape=(2,))
        from discopt.modeling.core import LogicalEquivalent

        expr = Y[0].equivalent_to(Y[1])
        assert isinstance(expr, LogicalEquivalent)

    def test_complex_nesting(self):
        m = dm.Model("test")
        Y = m.boolean("y", shape=(3,))
        from discopt.modeling.core import LogicalImplies

        expr = Y[0].implies(Y[1] & ~Y[2])
        assert isinstance(expr, LogicalImplies)

    def test_iterate_array(self):
        m = dm.Model("test")
        Y = m.boolean("y", shape=(3,))
        items = list(Y)
        assert len(items) == 3
        assert all(isinstance(b, dm.BooleanVar) for b in items)


class TestLogicalConstraints:
    def test_logical_implies_solve(self):
        """Y[0]=1 and Y[0].implies(Y[1]) should force Y[1]=1."""
        m = dm.Model("test_implies")
        Y = m.boolean("y", shape=(2,))
        x = m.continuous("x", lb=0, ub=10)

        m.logical(Y[0].implies(Y[1]))
        m.subject_to(Y[0].variable == 1)  # force Y[0] = 1
        m.minimize(x + Y[1].variable)

        result = m.solve(time_limit=30)
        assert result.status == "optimal"
        # Y[1] must be 1 due to implication
        y_vals = result.x["y"]
        assert y_vals[1] == pytest.approx(1.0, abs=1e-4)

    def test_logical_or_solve(self):
        """Y[0] | Y[1] forces at least one to be true."""
        m = dm.Model("test_or")
        Y = m.boolean("y", shape=(2,))

        m.logical(Y[0] | Y[1])
        # Minimize sum to prefer both 0 — but OR forces at least one to 1
        m.minimize(Y[0].variable + Y[1].variable)

        result = m.solve(time_limit=30)
        assert result.status == "optimal"
        y_vals = result.x["y"]
        assert y_vals[0] + y_vals[1] >= 1.0 - 1e-4

    def test_logical_complex_solve(self):
        """Y[0].implies(Y[1] & ~Y[2])."""
        m = dm.Model("test_complex")
        Y = m.boolean("y", shape=(3,))

        m.logical(Y[0].implies(Y[1] & ~Y[2]))
        m.subject_to(Y[0].variable == 1)  # force Y[0] = 1
        m.minimize(Y[1].variable + Y[2].variable)

        result = m.solve(time_limit=30)
        assert result.status == "optimal"
        y_vals = result.x["y"]
        # Y[0]=1 implies Y[1]=1 and Y[2]=0
        assert y_vals[1] == pytest.approx(1.0, abs=1e-4)
        assert y_vals[2] == pytest.approx(0.0, abs=1e-4)

    def test_logical_atleast(self):
        """atleast(2, Y) forces at least 2 true."""
        m = dm.Model("test_atleast")
        Y = m.boolean("y", shape=(3,))

        m.logical(dm.atleast(2, Y))
        m.minimize(Y[0].variable + Y[1].variable + Y[2].variable)

        result = m.solve(time_limit=30)
        assert result.status == "optimal"
        y_sum = sum(result.x["y"])
        assert y_sum >= 2.0 - 1e-4

    def test_logical_atmost(self):
        """atmost(1, Y) forces at most 1 true."""
        m = dm.Model("test_atmost")
        Y = m.boolean("y", shape=(3,))

        m.logical(dm.atmost(1, Y))
        # Maximize to prefer all 1 — but atmost forces at most 1
        m.maximize(Y[0].variable + Y[1].variable + Y[2].variable)

        result = m.solve(time_limit=30)
        assert result.status == "optimal"
        y_sum = sum(result.x["y"])
        assert y_sum <= 1.0 + 1e-4

    def test_logical_exactly(self):
        """exactly(2, Y) forces exactly 2 true."""
        m = dm.Model("test_exactly")
        Y = m.boolean("y", shape=(3,))

        m.logical(dm.exactly(2, Y))
        m.minimize(Y[0].variable)  # any feasible objective

        result = m.solve(time_limit=30)
        assert result.status == "optimal"
        y_sum = sum(result.x["y"])
        assert y_sum == pytest.approx(2.0, abs=1e-4)

    def test_logical_equivalent_to_solve(self):
        """Y[0].equivalent_to(Y[1]) forces Y[0] == Y[1]."""
        m = dm.Model("test_equiv")
        Y = m.boolean("y", shape=(2,))

        m.logical(Y[0].equivalent_to(Y[1]))
        m.subject_to(Y[0].variable == 1)
        m.minimize(Y[1].variable)

        result = m.solve(time_limit=30)
        assert result.status == "optimal"
        y_vals = result.x["y"]
        assert y_vals[1] == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Nested Disjunctions
# ---------------------------------------------------------------------------


class TestNestedDisjunctions:
    def test_disjunction_factory(self):
        """m.disjunction() returns object without adding to model."""
        m = dm.Model("test")
        x = m.continuous("x", lb=0, ub=10)
        inner = m.disjunction([[x <= 3], [x >= 7]], name="inner")

        from discopt.modeling.core import _DisjunctiveConstraint

        assert isinstance(inner, _DisjunctiveConstraint)
        # Should NOT be in model constraints yet
        assert inner not in m._constraints

    def test_nested_solve_two_level(self):
        """Nested disjunction: outer picks mode, inner refines."""
        m = dm.Model("nested")
        x = m.continuous("x", lb=0, ub=20)

        # Inner: x in [1, 3] or x in [5, 7]
        inner = m.disjunction(
            [
                [x >= 1, x <= 3],
                [x >= 5, x <= 7],
            ],
            name="inner",
        )

        # Outer: (inner disjunction) or x in [15, 20]
        m.either_or(
            [
                [inner],
                [x >= 15, x <= 20],
            ],
            name="outer",
        )

        m.minimize(x)
        result = m.solve(time_limit=60)
        assert result.status == "optimal"
        # Minimum is x=1 from inner disjunct 1
        assert result.objective == pytest.approx(1.0, abs=1e-3)

    def test_nested_solve_picks_outer(self):
        """Nested: when inner options are worse, solver picks other outer."""
        m = dm.Model("nested2")
        x = m.continuous("x", lb=0, ub=20)

        inner = m.disjunction(
            [
                [x >= 10, x <= 12],
                [x >= 14, x <= 16],
            ],
            name="inner",
        )

        m.either_or(
            [
                [inner],
                [x >= 2, x <= 4],
            ],
            name="outer",
        )

        m.minimize(x)
        result = m.solve(time_limit=60)
        assert result.status == "optimal"
        # x=2 is best (from outer disjunct 2)
        assert result.objective == pytest.approx(2.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Disjunct Block Abstraction
# ---------------------------------------------------------------------------


class TestDisjunctBlock:
    def test_create_disjunct(self):
        m = dm.Model("test")
        d = m.make_disjunct("mode_a")
        assert d.name == "mode_a"
        assert isinstance(d.active, dm.BooleanVar)
        assert len(d.constraints) == 0

    def test_add_constraints_to_disjunct(self):
        m = dm.Model("test")
        x = m.continuous("x", lb=0, ub=10)
        d = m.make_disjunct("mode_a")
        d.subject_to(x <= 3)
        d.subject_to(x >= 1)
        assert len(d.constraints) == 2

    def test_add_disjunction_solve(self):
        """Disjunct block solve matches either_or result."""
        m = dm.Model("test_block")
        x = m.continuous("x", lb=0, ub=10)

        d1 = m.make_disjunct("low")
        d1.subject_to(x <= 3)

        d2 = m.make_disjunct("high")
        d2.subject_to(x >= 7)

        m.add_disjunction([d1, d2], name="mode")
        m.minimize(x)

        result = m.solve(time_limit=30)
        assert result.status == "optimal"
        assert result.objective == pytest.approx(0.0, abs=1e-3)

    def test_disjunct_indicator_in_logical(self):
        """Can use disjunct indicators in logical constraints."""
        m = dm.Model("test")
        x = m.continuous("x", lb=0, ub=10)

        d1 = m.make_disjunct("a")
        d1.subject_to(x <= 3)

        d2 = m.make_disjunct("b")
        d2.subject_to(x >= 7)

        m.add_disjunction([d1, d2])

        # Force d1 active via logical constraint
        m.subject_to(d1.active.variable == 1)
        m.minimize(x)

        result = m.solve(time_limit=30)
        assert result.status == "optimal"
        assert result.objective == pytest.approx(0.0, abs=1e-3)

    def test_disjunct_list_constraints(self):
        """Can add a list of constraints at once."""
        m = dm.Model("test")
        x = m.continuous("x", lb=0, ub=10)
        d = m.make_disjunct("mode")
        d.subject_to([x <= 5, x >= 2])
        assert len(d.constraints) == 2
