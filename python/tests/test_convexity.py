"""Tests for convexity detection (Phase E1)."""

import discopt.modeling as dm
from discopt._jax.convexity import (
    Curvature,
    classify_constraint,
    classify_expr,
    classify_model,
)
from discopt.modeling.core import Constant, Model


class TestLeafExpressions:
    """Leaf nodes are always AFFINE."""

    def test_constant(self):
        assert classify_expr(Constant(5.0)) == Curvature.AFFINE

    def test_variable(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        assert classify_expr(x, m) == Curvature.AFFINE

    def test_parameter(self):
        m = Model("test")
        p = dm.Parameter("p", 3.0, m)
        assert classify_expr(p, m) == Curvature.AFFINE

    def test_index_expression(self):
        m = Model("test")
        x = m.continuous("x", shape=(3,), lb=0, ub=10)
        assert classify_expr(x[0], m) == Curvature.AFFINE


class TestLinearExpressions:
    """Linear expressions are AFFINE."""

    def test_sum(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        assert classify_expr(x + y, m) == Curvature.AFFINE

    def test_difference(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        assert classify_expr(x - y, m) == Curvature.AFFINE

    def test_scalar_multiply(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        assert classify_expr(3.0 * x, m) == Curvature.AFFINE

    def test_negation(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        assert classify_expr(-x, m) == Curvature.AFFINE

    def test_div_by_constant(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        assert classify_expr(x / 2.0, m) == Curvature.AFFINE

    def test_linear_combination(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        expr = 2.0 * x + 3.0 * y - 1.0
        assert classify_expr(expr, m) == Curvature.AFFINE


class TestConvexExpressions:
    """Known convex expressions."""

    def test_x_squared(self):
        m = Model("test")
        x = m.continuous("x", lb=-5, ub=5)
        assert classify_expr(x**2, m) == Curvature.CONVEX

    def test_exp(self):
        m = Model("test")
        x = m.continuous("x", lb=-5, ub=5)
        assert classify_expr(dm.exp(x), m) == Curvature.CONVEX

    def test_exp_of_affine(self):
        m = Model("test")
        x = m.continuous("x", lb=-5, ub=5)
        assert classify_expr(dm.exp(2.0 * x + 1.0), m) == Curvature.CONVEX

    def test_exp_of_convex(self):
        """exp(convex) = convex because exp is convex & nondecreasing."""
        m = Model("test")
        x = m.continuous("x", lb=-5, ub=5)
        assert classify_expr(dm.exp(x**2), m) == Curvature.CONVEX

    def test_sum_of_convex(self):
        m = Model("test")
        x = m.continuous("x", lb=-5, ub=5)
        y = m.continuous("y", lb=-5, ub=5)
        expr = x**2 + y**2
        assert classify_expr(expr, m) == Curvature.CONVEX

    def test_positive_scale_of_convex(self):
        m = Model("test")
        x = m.continuous("x", lb=-5, ub=5)
        assert classify_expr(3.0 * (x**2), m) == Curvature.CONVEX

    def test_abs_of_affine(self):
        m = Model("test")
        x = m.continuous("x", lb=-5, ub=5)
        assert classify_expr(dm.abs(x), m) == Curvature.CONVEX

    def test_cosh_of_affine(self):
        m = Model("test")
        x = m.continuous("x", lb=-5, ub=5)
        from discopt.modeling.core import FunctionCall

        assert classify_expr(FunctionCall("cosh", x), m) == Curvature.CONVEX

    def test_even_power(self):
        m = Model("test")
        x = m.continuous("x", lb=-5, ub=5)
        assert classify_expr(x**4, m) == Curvature.CONVEX

    def test_odd_power_nonneg(self):
        """x^3 on [0, inf) is convex."""
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        assert classify_expr(x**3, m) == Curvature.CONVEX

    def test_fractional_power_gt1_nonneg(self):
        """x^1.5 on [0, inf) is convex."""
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        assert classify_expr(x**1.5, m) == Curvature.CONVEX

    def test_convex_plus_affine(self):
        m = Model("test")
        x = m.continuous("x", lb=-5, ub=5)
        expr = x**2 + 3.0 * x + 1.0
        assert classify_expr(expr, m) == Curvature.CONVEX


class TestConcaveExpressions:
    """Known concave expressions."""

    def test_log(self):
        m = Model("test")
        x = m.continuous("x", lb=0.1, ub=10)
        assert classify_expr(dm.log(x), m) == Curvature.CONCAVE

    def test_log_of_affine(self):
        m = Model("test")
        x = m.continuous("x", lb=0.1, ub=10)
        assert classify_expr(dm.log(2.0 * x + 1.0), m) == Curvature.CONCAVE

    def test_sqrt(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        assert classify_expr(dm.sqrt(x), m) == Curvature.CONCAVE

    def test_negative_convex_is_concave(self):
        m = Model("test")
        x = m.continuous("x", lb=-5, ub=5)
        assert classify_expr(-1.0 * (x**2), m) == Curvature.CONCAVE

    def test_neg_of_convex(self):
        m = Model("test")
        x = m.continuous("x", lb=-5, ub=5)
        assert classify_expr(-(x**2), m) == Curvature.CONCAVE

    def test_sum_of_concave(self):
        m = Model("test")
        x = m.continuous("x", lb=0.1, ub=10)
        y = m.continuous("y", lb=0.1, ub=10)
        assert classify_expr(dm.log(x) + dm.log(y), m) == Curvature.CONCAVE

    def test_fractional_power_01(self):
        """x^0.5 on [0, inf) is concave."""
        m = Model("test")
        x = m.continuous("x", lb=0.1, ub=10)
        assert classify_expr(x**0.5, m) == Curvature.CONCAVE

    def test_concave_plus_affine(self):
        m = Model("test")
        x = m.continuous("x", lb=0.1, ub=10)
        expr = dm.log(x) + 3.0 * x - 2.0
        assert classify_expr(expr, m) == Curvature.CONCAVE


class TestUnknownExpressions:
    """Non-convex/non-concave expressions."""

    def test_bilinear(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        assert classify_expr(x * y, m) == Curvature.UNKNOWN

    def test_sin(self):
        m = Model("test")
        x = m.continuous("x", lb=-5, ub=5)
        assert classify_expr(dm.sin(x), m) == Curvature.UNKNOWN

    def test_cos(self):
        m = Model("test")
        x = m.continuous("x", lb=-5, ub=5)
        assert classify_expr(dm.cos(x), m) == Curvature.UNKNOWN

    def test_convex_minus_convex(self):
        m = Model("test")
        x = m.continuous("x", lb=-5, ub=5)
        y = m.continuous("y", lb=-5, ub=5)
        assert classify_expr(x**2 - y**2, m) == Curvature.UNKNOWN

    def test_exp_of_concave(self):
        """exp(concave) is neither convex nor concave."""
        m = Model("test")
        x = m.continuous("x", lb=0.1, ub=10)
        assert classify_expr(dm.exp(dm.log(x)), m) == Curvature.UNKNOWN

    def test_log_of_convex(self):
        """log(convex) is neither convex nor concave."""
        m = Model("test")
        x = m.continuous("x", lb=0.1, ub=10)
        assert classify_expr(dm.log(dm.exp(x)), m) == Curvature.UNKNOWN

    def test_odd_power_mixed_sign(self):
        """x^3 on [-5, 5] is neither convex nor concave."""
        m = Model("test")
        x = m.continuous("x", lb=-5, ub=5)
        assert classify_expr(x**3, m) == Curvature.UNKNOWN

    def test_x_div_y(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=1, ub=10)
        assert classify_expr(x / y, m) == Curvature.UNKNOWN


class TestConstraintConvexity:
    """Constraint-level convexity checks."""

    def test_convex_leq(self):
        """x^2 <= 10 is convex."""
        m = Model("test")
        x = m.continuous("x", lb=-5, ub=5)
        c = x**2 <= 10
        assert classify_constraint(c, m) is True

    def test_concave_geq(self):
        """log(x) >= 0 is convex (concave body, >= sense)."""
        m = Model("test")
        x = m.continuous("x", lb=0.1, ub=10)
        c = dm.log(x) >= 0
        assert classify_constraint(c, m) is True

    def test_affine_eq(self):
        """2x + 3y == 10 is convex."""
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        c = 2 * x + 3 * y == 10
        assert classify_constraint(c, m) is True

    def test_nonconvex_leq(self):
        """sin(x) <= 0 is not convex."""
        m = Model("test")
        x = m.continuous("x", lb=-5, ub=5)
        c = dm.sin(x) <= 0
        assert classify_constraint(c, m) is False

    def test_convex_geq_is_nonconvex(self):
        """x^2 >= 1 is non-convex (convex body, >= sense)."""
        m = Model("test")
        x = m.continuous("x", lb=-5, ub=5)
        c = x**2 >= 1
        assert classify_constraint(c, m) is False

    def test_nonlinear_eq(self):
        """x^2 == 1 is non-convex."""
        m = Model("test")
        x = m.continuous("x", lb=-5, ub=5)
        c = x**2 == 1
        assert classify_constraint(c, m) is False


class TestModelConvexity:
    """Model-level convexity detection."""

    def test_convex_qp(self):
        """min x^2 + y^2 s.t. x + y <= 10."""
        m = Model("convex_qp")
        x = m.continuous("x", lb=-5, ub=5)
        y = m.continuous("y", lb=-5, ub=5)
        m.minimize(x**2 + y**2)
        m.subject_to(x + y <= 10)
        is_cvx, mask = classify_model(m)
        assert is_cvx is True
        assert all(mask)

    def test_convex_with_log_objective(self):
        """max log(x) + log(y) s.t. x + y <= 10."""
        m = Model("convex_log")
        x = m.continuous("x", lb=0.1, ub=10)
        y = m.continuous("y", lb=0.1, ub=10)
        m.maximize(dm.log(x) + dm.log(y))
        m.subject_to(x + y <= 10)
        is_cvx, mask = classify_model(m)
        assert is_cvx is True

    def test_nonconvex_bilinear(self):
        """min x*y is nonconvex."""
        m = Model("bilinear")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.minimize(x * y)
        is_cvx, mask = classify_model(m)
        assert is_cvx is False

    def test_mixed_convex_constraints(self):
        """Some constraints convex, some not."""
        m = Model("mixed")
        x = m.continuous("x", lb=0.1, ub=10)
        y = m.continuous("y", lb=0.1, ub=10)
        m.minimize(x + y)
        m.subject_to(x + y <= 10)  # convex
        m.subject_to(x * y <= 5)  # non-convex
        m.subject_to(dm.log(x) >= 0.1)  # convex (concave >= rhs)
        is_cvx, mask = classify_model(m)
        assert is_cvx is False
        assert mask == [True, False, True]

    def test_lp_is_convex(self):
        """Linear programs are convex."""
        m = Model("lp")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.minimize(x + 2 * y)
        m.subject_to(x + y <= 10)
        m.subject_to(x - y >= -5)
        is_cvx, mask = classify_model(m)
        assert is_cvx is True
        assert all(mask)

    def test_no_objective(self):
        """Feasibility problem with convex constraints."""
        m = Model("feasibility")
        x = m.continuous("x", lb=-5, ub=5)
        m.subject_to(x**2 <= 25)
        is_cvx, mask = classify_model(m)
        assert is_cvx is True

    def test_maximize_convex_is_nonconvex(self):
        """max x^2 is nonconvex (maximizing a convex function)."""
        m = Model("max_convex")
        x = m.continuous("x", lb=-5, ub=5)
        m.maximize(x**2)
        is_cvx, _mask = classify_model(m)
        assert is_cvx is False


class TestCaching:
    """Verify that caching works correctly."""

    def test_shared_subexpression(self):
        m = Model("test")
        x = m.continuous("x", lb=-5, ub=5)
        # x^2 appears twice in x^2 + x^2
        sq = x**2
        expr = sq + sq
        cache: dict = {}
        result = classify_expr(expr, m, cache)
        assert result == Curvature.CONVEX
        # x^2 should be cached
        assert id(sq) in cache


class TestSumOverExpression:
    """Test SumOverExpression handling."""

    def test_sum_of_convex_terms(self):
        m = Model("test")
        x = m.continuous("x", shape=(3,), lb=-5, ub=5)
        terms = [x[i] ** 2 for i in range(3)]
        from discopt.modeling.core import SumOverExpression

        expr = SumOverExpression(terms)
        assert classify_expr(expr, m) == Curvature.CONVEX

    def test_sum_of_mixed_terms(self):
        m = Model("test")
        x = m.continuous("x", lb=-5, ub=5)
        y = m.continuous("y", lb=0.1, ub=10)
        from discopt.modeling.core import SumOverExpression

        expr = SumOverExpression([x**2, dm.log(y)])
        assert classify_expr(expr, m) == Curvature.UNKNOWN


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_x_power_1(self):
        m = Model("test")
        x = m.continuous("x", lb=-5, ub=5)
        assert classify_expr(x**1, m) == Curvature.AFFINE

    def test_x_power_0(self):
        m = Model("test")
        x = m.continuous("x", lb=-5, ub=5)
        assert classify_expr(x**0, m) == Curvature.AFFINE

    def test_negative_scale_flips(self):
        m = Model("test")
        x = m.continuous("x", lb=-5, ub=5)
        # -x^2 should be concave
        assert classify_expr(-1 * (x**2), m) == Curvature.CONCAVE

    def test_div_by_negative_constant(self):
        m = Model("test")
        x = m.continuous("x", lb=-5, ub=5)
        # x^2 / (-2) should be concave
        assert classify_expr((x**2) / (-2.0), m) == Curvature.CONCAVE
