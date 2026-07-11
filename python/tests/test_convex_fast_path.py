"""Tests for the convex NLP fast path.

Verifies that convex continuous problems are solved via a single NLP call
(no branch-and-bound), with global optimality guaranteed, and that
nonconvex or integer problems correctly fall back to the standard solver.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import Model
from test_minlptests import MINLPTESTS_CVX_BY_ID


class TestConvexFastPathDetection:
    """Verify that the fast path is triggered for convex problems."""

    def test_convex_nlp_uses_fast_path(self):
        """A convex NLP (exp + quadratic, linear constraints) uses fast path."""
        m = Model("convex_nlp")
        x = m.continuous("x", lb=-5, ub=5)

        # minimize exp(x) + x^2 (convex: sum of convex functions)
        m.minimize(dm.exp(x) + x**2)
        m.subject_to(x >= -2)

        result = m.solve()
        assert result.status == "optimal"
        assert result.convex_fast_path is True
        assert result.node_count == 0

    def test_convex_exp_quadratic_with_linear_constraint(self):
        """Convex NLP with exp and quadratic terms uses fast path."""
        m = Model("convex_exp_quad")
        x = m.continuous("x", lb=-10, ub=10)
        y = m.continuous("y", lb=-10, ub=10)

        # minimize exp(x) + y^2 (convex)
        m.minimize(dm.exp(x) + y**2)
        m.subject_to(x + y >= 1)

        result = m.solve()
        assert result.status == "optimal"
        assert result.convex_fast_path is True
        assert result.node_count == 0

    def test_nonconvex_nlp_no_fast_path(self):
        """A nonconvex NLP (sin) does NOT use the fast path."""
        m = Model("nonconvex_nlp")
        x = m.continuous("x", lb=-5, ub=5)

        # minimize sin(x) (nonconvex)
        m.minimize(dm.sin(x))

        result = m.solve()
        # Should still solve, but not via fast path
        assert result.convex_fast_path is False

    @pytest.mark.slow
    def test_integer_variables_no_fast_path(self):
        """Integer variables prevent the fast path, even if relaxation is convex."""
        m = Model("integer_problem")
        x = m.integer("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)

        # minimize exp(x) + y^2 (convex objective, but integer variable)
        m.minimize(dm.exp(x) + y**2)
        m.subject_to(x + y >= 1)

        result = m.solve()
        # Should NOT use fast path because of integer variable
        assert result.convex_fast_path is False

    def test_maximize_concave_uses_fast_path(self):
        """Maximizing a concave function uses the fast path."""
        m = Model("maximize_concave")
        x = m.continuous("x", lb=0.1, ub=10)

        # maximize log(x) (concave objective, so max is convex problem)
        m.maximize(dm.log(x))
        m.subject_to(x <= 5)

        result = m.solve()
        assert result.status == "optimal"
        assert result.convex_fast_path is True
        assert result.node_count == 0


class TestNonconvexContinuousSpatialRegressions:
    """Regression tests for nonconvex pure-continuous spatial dispatch."""

    def test_nonconvex_nlp_root_local_point_is_not_certified_optimal(self):
        m = Model("narrow_nonconvex_well")
        x = m.continuous("x", lb=0.0, ub=1.0)
        m.minimize(-dm.exp(-10000.0 * (x - 0.05) ** 2))

        result = m.solve(nlp_solver="ipm", time_limit=10.0, gap_tolerance=1e-6, max_nodes=1)

        assert result.convex_fast_path is False
        assert result.status != "optimal"
        assert result.objective is not None
        assert result.objective > -0.1
        assert result.node_count > 1
        assert result.bound is None or result.bound < -0.5

    def test_nonconvex_continuous_qp_minimize_uses_spatial_bb(self):
        m = Model("nonconvex_qp_min")
        x = m.continuous("x", lb=-1.0, ub=1.0)
        m.minimize(-(x**2))

        result = m.solve(nlp_solver="ipm", time_limit=10.0, gap_tolerance=1e-6, max_nodes=500)

        assert result.status == "optimal"
        assert result.convex_fast_path is False
        assert result.objective == pytest.approx(-1.0, abs=1e-6)
        assert abs(float(result.value(x))) == pytest.approx(1.0, abs=1e-6)
        assert result.node_count > 0

    def test_nonconvex_continuous_qp_maximize_uses_spatial_bb(self):
        m = Model("nonconvex_qp_max")
        x = m.continuous("x", lb=-1.0, ub=1.0)
        m.maximize(x**2)

        result = m.solve(nlp_solver="ipm", time_limit=10.0, gap_tolerance=1e-6, max_nodes=500)

        assert result.status == "optimal"
        assert result.convex_fast_path is False
        assert result.objective == pytest.approx(1.0, abs=1e-6)
        assert abs(float(result.value(x))) == pytest.approx(1.0, abs=1e-6)
        assert result.node_count > 0

    def test_skip_convex_check_nonconvex_continuous_qp_minimize_uses_spatial_bb(self):
        m = Model("skip_check_nonconvex_qp_min")
        x = m.continuous("x", lb=-1.0, ub=1.0)
        m.minimize(-(x**2))

        result = m.solve(
            skip_convex_check=True,
            nlp_solver="ipm",
            time_limit=10.0,
            gap_tolerance=1e-6,
            max_nodes=500,
        )

        assert result.status == "optimal"
        assert result.convex_fast_path is False
        assert result.objective == pytest.approx(-1.0, abs=1e-6)
        assert abs(float(result.value(x))) == pytest.approx(1.0, abs=1e-6)
        assert result.node_count > 0

    def test_skip_convex_check_nonconvex_continuous_qp_maximize_uses_spatial_bb(self):
        m = Model("skip_check_nonconvex_qp_max")
        x = m.continuous("x", lb=-1.0, ub=1.0)
        m.maximize(x**2)

        result = m.solve(
            skip_convex_check=True,
            nlp_solver="ipm",
            time_limit=10.0,
            gap_tolerance=1e-6,
            max_nodes=500,
        )

        assert result.status == "optimal"
        assert result.convex_fast_path is False
        assert result.objective == pytest.approx(1.0, abs=1e-6)
        assert abs(float(result.value(x))) == pytest.approx(1.0, abs=1e-6)
        assert result.node_count > 0


class TestConvexFastPathOptOut:
    """Verify that skip_convex_check disables the fast path."""

    def test_skip_convex_check(self):
        """Passing skip_convex_check=True bypasses convex detection."""
        m = Model("skip_check")
        x = m.continuous("x", lb=-10, ub=10)

        # Use a nonlinear convex objective so it goes through NLP path
        m.minimize(dm.exp(x))

        result = m.solve(skip_convex_check=True)
        assert result.status == "optimal"
        # With skip_convex_check, the fast path is not used
        assert result.convex_fast_path is False


class TestConvexFastPathSolutions:
    """Verify that solutions from the fast path are globally optimal."""

    def test_convex_nlp_optimal_value(self):
        """Verify NLP solution: min exp(x) + y^2 s.t. x + y >= 1."""
        m = Model("nlp_optimal")
        x = m.continuous("x", lb=-10, ub=10)
        y = m.continuous("y", lb=-10, ub=10)

        m.minimize(dm.exp(x) + y**2)
        m.subject_to(x + y >= 1)

        result = m.solve()
        assert result.status == "optimal"
        assert result.convex_fast_path is True
        assert result.objective is not None
        # The optimal should be finite and achievable
        assert np.isfinite(result.objective)

    def test_exp_nlp_optimal_value(self):
        """Verify NLP solution: min exp(x) s.t. x >= 0."""
        m = Model("exp_optimal")
        x = m.continuous("x", lb=-10, ub=10)

        m.minimize(dm.exp(x))
        m.subject_to(x >= 0)

        result = m.solve()
        assert result.status == "optimal"
        assert result.convex_fast_path is True

        # Optimal: x = 0, objective = exp(0) = 1.0
        assert result.objective is not None
        assert abs(result.objective - 1.0) < 1e-4

    def test_maximize_log_optimal_value(self):
        """Verify: max log(x) s.t. x <= 5."""
        m = Model("log_optimal")
        x = m.continuous("x", lb=0.1, ub=10)

        m.maximize(dm.log(x))
        m.subject_to(x <= 5)

        result = m.solve()
        assert result.status == "optimal"
        assert result.convex_fast_path is True

        # Optimal: x = 5, objective = log(5) ≈ 1.6094 (must be positive — not negated)
        assert result.objective is not None
        assert abs(result.objective - np.log(5.0)) < 1e-2

    def test_maximize_objective_sign_not_negated(self):
        """Regression test for issue #28: maximize result.objective must not be negated.

        max -(x-1)^2 + 4  s.t. x in [0,2] → optimal value = 4.0 at x=1.
        _solve_continuous was returning -4.0 (the internal minimization value).
        """
        m = Model("maximize_mwe")
        x = m.continuous("x", lb=0.0, ub=2.0)
        m.maximize(-((x - 1.0) ** 2) + 4.0)
        result = m.solve()
        assert result.status == "optimal"
        assert result.objective is not None
        assert result.objective > 0, f"Expected +4.0, got {result.objective}"
        assert abs(result.objective - 4.0) < 1e-4

    def test_gap_is_zero_for_convex(self):
        """Convex fast path should report zero gap (global optimality)."""
        m = Model("zero_gap")
        x = m.continuous("x", lb=-5, ub=5)

        # Use exp to avoid QP classification
        m.minimize(dm.exp(x))

        result = m.solve()
        assert result.convex_fast_path is True
        assert result.gap == 0.0


class TestConvexFastPathConstraints:
    """Test constraint handling in convex detection."""

    def test_affine_equality_with_convex_obj(self):
        """Affine equality constraints with convex NLP objective use fast path."""
        m = Model("affine_eq")
        x = m.continuous("x", lb=-10, ub=10)
        y = m.continuous("y", lb=-10, ub=10)

        # exp is nonlinear convex, so classifier won't pick it up as QP
        m.minimize(dm.exp(x) + dm.exp(y))
        m.subject_to(x + y == 1)

        result = m.solve()
        assert result.convex_fast_path is True
        assert result.status == "optimal"

    def test_nonlinear_equality_blocks_fast_path(self):
        """Nonlinear equality constraints are not convex."""
        m = Model("nonlinear_eq")
        x = m.continuous("x", lb=-10, ub=10)
        y = m.continuous("y", lb=-10, ub=10)

        m.minimize(dm.exp(x) + dm.exp(y))
        # x^2 == 1 is NOT affine, so convexity fails
        m.subject_to(x**2 == 1)

        result = m.solve()
        assert result.convex_fast_path is False

    def test_convex_inequality_constraint(self):
        """Convex inequality constraints allow fast path."""
        m = Model("convex_ineq")
        x = m.continuous("x", lb=-10, ub=10)
        y = m.continuous("y", lb=-10, ub=10)

        # minimize exp(x) + exp(y) subject to x^2 + y^2 <= 1
        m.minimize(dm.exp(x) + dm.exp(y))
        m.subject_to(x**2 + y**2 <= 1)

        result = m.solve()
        assert result.convex_fast_path is True
        assert result.status == "optimal"


class TestTranslatedLPRegressions:
    """Translated convex LPs from MINLPTests must take the convex fast path."""

    @pytest.mark.parametrize(
        "problem_id",
        [
            "nlp_cvx_001_010",
            "nlp_cvx_002_010",
        ],
    )
    @pytest.mark.parametrize("solver_name", [None, "amp"], ids=["default", "amp"])
    def test_translated_lp_uses_convex_fast_path(self, problem_id, solver_name):
        instance = MINLPTESTS_CVX_BY_ID[problem_id]
        m = instance.build_fn()
        solve_kwargs = {"time_limit": 60.0, "gap_tolerance": 1e-6}
        if solver_name == "amp":
            solve_kwargs["solver"] = "amp"
            solve_kwargs["nlp_solver"] = "ipm"

        result = m.solve(**solve_kwargs)

        assert result.status == "optimal"
        assert result.convex_fast_path is True
        assert result.objective is not None
        tol = 1e-6 + 1e-4 * abs(instance.expected_obj)
        assert abs(result.objective - instance.expected_obj) <= tol


class TestTranslatedSpecialConvexRegressions:
    """Translated convex forms beyond LP/QP must take the convex fast path."""

    @pytest.mark.parametrize(
        "problem_id",
        [
            "nlp_cvx_105_011",
            "nlp_cvx_108_010",
            "nlp_cvx_108_011",
            "nlp_cvx_108_012",
            "nlp_cvx_108_013",
            "nlp_cvx_203_010",
            "nlp_cvx_204_010",
            "nlp_cvx_205_010",
            "nlp_cvx_206_010",
        ],
    )
    @pytest.mark.parametrize("solver_name", [None, "amp"], ids=["default", "amp"])
    def test_translated_special_convex_uses_fast_path(self, problem_id, solver_name):
        instance = MINLPTESTS_CVX_BY_ID[problem_id]
        m = instance.build_fn()
        solve_kwargs = {"time_limit": 60.0, "gap_tolerance": 1e-6}
        if solver_name == "amp":
            solve_kwargs["solver"] = "amp"
            solve_kwargs["nlp_solver"] = "ipm"

        result = m.solve(**solve_kwargs)

        assert result.status in ("optimal", "feasible")
        assert result.convex_fast_path is True
        assert result.objective is not None
        tol = 1e-6 + 1e-4 * abs(instance.expected_obj)
        assert abs(result.objective - instance.expected_obj) <= tol
