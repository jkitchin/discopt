"""Tests for RLT and Outer Approximation cutting planes.

Validates:
  - LinearCut structure and representation
  - RLT cuts: McCormick envelope validity for bilinear terms
  - OA cuts: tangent hyperplane validity for convex/nonlinear constraints
  - Separation: only violated cuts are returned
  - Integration with NLPEvaluator
"""

from __future__ import annotations

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

sys.path.insert(0, "/Users/jkitchin/Dropbox/projects/discopt/python")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from discopt._jax.cutting_planes import (
    BilinearTerm,
    LinearCut,
    detect_bilinear_terms,
    generate_cuts_at_node,
    generate_oa_cut,
    generate_oa_cuts_from_evaluator,
    generate_objective_oa_cut,
    generate_rlt_cuts,
    is_cut_violated,
    separate_oa_cuts,
    separate_rlt_cuts,
)

TOL = 1e-8
N_POINTS = 5_000


def _random_points_in_box(key, lb, ub, n=N_POINTS):
    """Generate n random points uniformly in [lb, ub] for each dimension."""
    d = len(lb)
    u = jax.random.uniform(key, shape=(n, d), dtype=jnp.float64)
    return np.asarray(lb + (ub - lb) * u)


# ===================================================================
# LinearCut basics
# ===================================================================


class TestLinearCut:
    def test_named_tuple_fields(self):
        coeffs = np.array([1.0, -2.0, 3.0])
        cut = LinearCut(coeffs=coeffs, rhs=5.0, sense="<=")
        assert cut.sense == "<="
        assert cut.rhs == 5.0
        np.testing.assert_array_equal(cut.coeffs, coeffs)

    def test_equality_sense(self):
        cut = LinearCut(coeffs=np.array([1.0]), rhs=0.0, sense="==")
        assert cut.sense == "=="

    def test_ge_sense(self):
        cut = LinearCut(coeffs=np.array([1.0, 1.0]), rhs=2.0, sense=">=")
        assert cut.sense == ">="


# ===================================================================
# RLT cuts: validity for bilinear terms
# ===================================================================


class TestRLTCutsWithAuxiliary:
    """Test RLT cuts when an auxiliary variable w = x[i]*x[j] is present."""

    def test_generates_four_cuts(self):
        bt = BilinearTerm(i=0, j=1, w_index=2)
        lb = np.array([1.0, 2.0, 0.0])
        ub = np.array([3.0, 5.0, 100.0])
        cuts = generate_rlt_cuts(bt, lb, ub, n_vars=3)
        assert len(cuts) == 4

    def test_underestimators_valid(self):
        """The two underestimator cuts must be satisfied when w = x[i]*x[j]."""
        bt = BilinearTerm(i=0, j=1, w_index=2)
        lb = np.array([1.0, 2.0, 0.0])
        ub = np.array([4.0, 6.0, 100.0])
        cuts = generate_rlt_cuts(bt, lb, ub, n_vars=3)

        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        xi = np.asarray(1.0 + 3.0 * jax.random.uniform(k1, (N_POINTS,), dtype=jnp.float64))
        xj = np.asarray(2.0 + 4.0 * jax.random.uniform(k2, (N_POINTS,), dtype=jnp.float64))
        w = xi * xj

        # Underestimator cuts have sense ">="
        ge_cuts = [c for c in cuts if c.sense == ">="]
        assert len(ge_cuts) == 2

        for cut in ge_cuts:
            for k in range(N_POINTS):
                x = np.array([xi[k], xj[k], w[k]])
                lhs = np.dot(cut.coeffs, x)
                assert lhs >= cut.rhs - TOL, f"Underestimator violated: {lhs} < {cut.rhs}"

    def test_overestimators_valid(self):
        """The two overestimator cuts must be satisfied when w = x[i]*x[j]."""
        bt = BilinearTerm(i=0, j=1, w_index=2)
        lb = np.array([-2.0, -3.0, 0.0])
        ub = np.array([4.0, 5.0, 100.0])
        cuts = generate_rlt_cuts(bt, lb, ub, n_vars=3)

        key = jax.random.PRNGKey(99)
        k1, k2 = jax.random.split(key)
        xi = np.asarray(-2.0 + 6.0 * jax.random.uniform(k1, (N_POINTS,), dtype=jnp.float64))
        xj = np.asarray(-3.0 + 8.0 * jax.random.uniform(k2, (N_POINTS,), dtype=jnp.float64))
        w = xi * xj

        le_cuts = [c for c in cuts if c.sense == "<="]
        assert len(le_cuts) == 2

        for cut in le_cuts:
            for k in range(N_POINTS):
                x = np.array([xi[k], xj[k], w[k]])
                lhs = np.dot(cut.coeffs, x)
                assert lhs <= cut.rhs + TOL, f"Overestimator violated: {lhs} > {cut.rhs}"

    def test_tight_at_corners(self):
        """Each McCormick cut should be tight at one of the four box corners."""
        bt = BilinearTerm(i=0, j=1, w_index=2)
        x_lb = np.array([1.0, 2.0, 0.0])
        x_ub = np.array([3.0, 5.0, 100.0])
        cuts = generate_rlt_cuts(bt, x_lb, x_ub, n_vars=3)

        corners = [
            np.array([1.0, 2.0, 2.0]),  # (lb, lb)
            np.array([1.0, 5.0, 5.0]),  # (lb, ub)
            np.array([3.0, 2.0, 6.0]),  # (ub, lb)
            np.array([3.0, 5.0, 15.0]),  # (ub, ub)
        ]

        for cut in cuts:
            tight_at_any = False
            for corner in corners:
                lhs = np.dot(cut.coeffs, corner)
                if abs(lhs - cut.rhs) < TOL:
                    tight_at_any = True
                    break
            assert tight_at_any, f"Cut not tight at any corner: {cut}"


class TestRLTCutsWithoutAuxiliary:
    """Test RLT cuts expressed only in original variables (no w)."""

    def test_generates_four_cuts(self):
        bt = BilinearTerm(i=0, j=1)
        lb = np.array([1.0, 2.0])
        ub = np.array([3.0, 5.0])
        cuts = generate_rlt_cuts(bt, lb, ub, n_vars=2)
        assert len(cuts) == 4

    def test_envelope_bounds_product(self):
        """Without auxiliary, the cuts bound the bilinear product at any point."""
        bt = BilinearTerm(i=0, j=1)
        lb = np.array([1.0, 2.0])
        ub = np.array([4.0, 6.0])
        cuts = generate_rlt_cuts(bt, lb, ub, n_vars=2)

        key = jax.random.PRNGKey(7)
        k1, k2 = jax.random.split(key)
        xi = np.asarray(1.0 + 3.0 * jax.random.uniform(k1, (N_POINTS,), dtype=jnp.float64))
        xj = np.asarray(2.0 + 4.0 * jax.random.uniform(k2, (N_POINTS,), dtype=jnp.float64))
        product = xi * xj

        # Without auxiliary, the "<=" cuts give underestimators of the product
        # and ">=" cuts give overestimators.
        le_cuts = [c for c in cuts if c.sense == "<="]
        ge_cuts = [c for c in cuts if c.sense == ">="]

        for k in range(N_POINTS):
            x = np.array([xi[k], xj[k]])
            # Each underestimator: coeffs @ x <= rhs should mean
            # the linear combination is an underestimator of the product
            for cut in le_cuts:
                linear_val = np.dot(cut.coeffs, x)
                # linear_val <= product (underestimator form)
                assert linear_val - cut.rhs <= product[k] + TOL

            for cut in ge_cuts:
                linear_val = np.dot(cut.coeffs, x)
                # linear_val - rhs >= product (overestimator form)
                assert linear_val - cut.rhs >= product[k] - TOL


class TestRLTSeparation:
    """Test that separation returns only violated cuts."""

    def test_feasible_point_no_cuts(self):
        """At a point where w = x[i]*x[j], no cuts should be violated."""
        bt = BilinearTerm(i=0, j=1, w_index=2)
        lb = np.array([1.0, 2.0, 0.0])
        ub = np.array([3.0, 5.0, 100.0])
        x_sol = np.array([2.0, 3.0, 6.0])  # w = 2*3 = 6
        violated = separate_rlt_cuts(bt, x_sol, lb, ub, n_vars=3)
        assert len(violated) == 0

    def test_violated_point_returns_cuts(self):
        """At a point where w != x[i]*x[j], some cuts should be violated."""
        bt = BilinearTerm(i=0, j=1, w_index=2)
        lb = np.array([1.0, 2.0, 0.0])
        ub = np.array([3.0, 5.0, 100.0])
        # w = 20 but x[0]*x[1] = 6, so w is too large -> overestimator violated
        x_sol = np.array([2.0, 3.0, 20.0])
        violated = separate_rlt_cuts(bt, x_sol, lb, ub, n_vars=3)
        assert len(violated) > 0


# ===================================================================
# OA cuts: tangent hyperplane validity
# ===================================================================


class TestOACutGeneration:
    def test_linear_function_exact(self):
        """OA cut of a linear function should reproduce it exactly."""
        # g(x) = 2*x[0] + 3*x[1] - 5
        grad = np.array([2.0, 3.0])
        x_star = np.array([1.0, 2.0])
        func_val = 2.0 * 1.0 + 3.0 * 2.0 - 5.0  # = 3.0
        cut = generate_oa_cut(grad, func_val, x_star, sense="<=")

        # At x_star: cut.coeffs @ x_star should equal cut.rhs + func_val... no:
        # The cut is: grad @ x <= grad @ x* - g(x*)
        # = 2*1 + 3*2 - 3 = 5
        expected_rhs = np.dot(grad, x_star) - func_val
        assert abs(cut.rhs - expected_rhs) < TOL
        np.testing.assert_allclose(cut.coeffs, grad)

    def test_quadratic_underestimates(self):
        """OA cut of a convex quadratic should underestimate everywhere."""
        # g(x) = x[0]^2 + x[1]^2
        # grad at x* = [2*x*[0], 2*x*[1]]
        x_star = np.array([1.0, 2.0])
        func_val = 1.0 + 4.0  # = 5.0
        grad = np.array([2.0, 4.0])  # 2*x*
        cut = generate_oa_cut(grad, func_val, x_star, sense="<=")

        # For any x, g(x) >= g(x*) + grad @ (x - x*)  (convexity)
        # So: g(x) >= grad @ x - (grad @ x* - g(x*))
        # i.e., g(x) >= cut.coeffs @ x - cut.rhs
        key = jax.random.PRNGKey(11)
        points = _random_points_in_box(key, np.array([-5.0, -5.0]), np.array([5.0, 5.0]))

        for k in range(N_POINTS):
            x = points[k]
            g_x = x[0] ** 2 + x[1] ** 2
            linear_val = np.dot(cut.coeffs, x)
            # g(x) >= linear_val - rhs => linear_val - rhs <= g(x)
            assert linear_val - cut.rhs <= g_x + TOL

    def test_oa_cut_tight_at_linearization_point(self):
        """OA cut should be tight (equality) at the linearization point."""
        x_star = np.array([3.0, -1.0])
        func_val = 9.0 + 1.0  # x^2 + y^2 = 10
        grad = np.array([6.0, -2.0])
        cut = generate_oa_cut(grad, func_val, x_star, sense="<=")

        lhs = np.dot(cut.coeffs, x_star)
        # At x_star: grad @ x* = grad @ x*, rhs = grad @ x* - g(x*)
        # So lhs - rhs = g(x*), meaning the linearization equals g(x*)
        assert abs(lhs - cut.rhs - func_val) < TOL


class TestOACutsFromEvaluator:
    """Test OA cut generation using a real NLPEvaluator."""

    def _make_model_and_evaluator(self):
        """Build a simple model: min x0^2 + x1^2 s.t. x0 + x1 >= 1."""
        from discopt._jax.nlp_evaluator import NLPEvaluator
        from discopt.modeling.core import Model

        m = Model("test_oa")
        x = m.continuous("x", shape=(2,), lb=-5.0, ub=5.0)
        m.minimize(x[0] ** 2 + x[1] ** 2)
        m.subject_to(x[0] + x[1] >= 1, name="sum_lb")
        evaluator = NLPEvaluator(m)
        return m, evaluator

    def test_generates_one_cut_per_constraint(self):
        _, evaluator = self._make_model_and_evaluator()
        x_sol = np.array([0.5, 0.5])
        cuts = generate_oa_cuts_from_evaluator(evaluator, x_sol)
        assert len(cuts) == 1

    def test_constraint_cut_valid(self):
        """OA cut should be valid linearization of the constraint."""
        _, evaluator = self._make_model_and_evaluator()
        x_sol = np.array([0.5, 0.5])
        # The constraint is x0 + x1 >= 1, which the evaluator stores as
        # (x0 + x1) - 1 >= 0, i.e., body = x0 + x1 - 1.
        # Since the body is linear, the OA cut is exact.
        cuts = generate_oa_cuts_from_evaluator(evaluator, x_sol, constraint_senses=["<="])
        assert len(cuts) == 1
        cut = cuts[0]
        assert cut.sense == "<="

    def test_objective_oa_cut(self):
        """Test OA cut generation for the objective."""
        _, evaluator = self._make_model_and_evaluator()
        x_sol = np.array([1.0, 2.0])
        n_vars = 2
        cut = generate_objective_oa_cut(evaluator, x_sol, n_vars)

        # f(x) = x0^2 + x1^2, grad = [2, 4], f(x*) = 5
        # cut: [2, 4] @ x <= [2,4]@[1,2] - 5 = 10-5 = 5
        np.testing.assert_allclose(cut.coeffs, [2.0, 4.0], atol=1e-6)
        assert abs(cut.rhs - 5.0) < 1e-6

    def test_objective_oa_cut_with_epigraph(self):
        """OA cut with epigraph variable z."""
        _, evaluator = self._make_model_and_evaluator()
        x_sol = np.array([1.0, 2.0])
        cut = generate_objective_oa_cut(evaluator, x_sol, n_vars=3, z_index=2)

        # coeffs should be [2, 4, -1], rhs = 5
        np.testing.assert_allclose(cut.coeffs, [2.0, 4.0, -1.0], atol=1e-6)
        assert abs(cut.rhs - 5.0) < 1e-6


class TestOASeparation:
    """Test that OA separation returns only violated cuts."""

    def _make_evaluator(self):
        from discopt._jax.nlp_evaluator import NLPEvaluator
        from discopt.modeling.core import Model

        m = Model("test_sep")
        x = m.continuous("x", shape=(2,), lb=-5.0, ub=5.0)
        m.minimize(x[0] + x[1])
        # Constraint: x0^2 + x1^2 <= 1 => x0^2 + x1^2 - 1 <= 0
        m.subject_to(x[0] ** 2 + x[1] ** 2 <= 1, name="circle")
        return NLPEvaluator(m)

    def test_feasible_point_no_violated_cuts(self):
        evaluator = self._make_evaluator()
        x_sol = np.array([0.0, 0.0])  # clearly inside circle
        cuts = separate_oa_cuts(evaluator, x_sol, constraint_senses=["<="])
        assert len(cuts) == 0

    def test_infeasible_point_returns_cut(self):
        evaluator = self._make_evaluator()
        x_sol = np.array([1.0, 1.0])  # x0^2+x1^2 = 2 > 1, violated
        cuts = separate_oa_cuts(evaluator, x_sol, constraint_senses=["<="])
        assert len(cuts) == 1
        assert cuts[0].sense == "<="


# ===================================================================
# is_cut_violated utility
# ===================================================================


class TestIsCutViolated:
    def test_le_not_violated(self):
        cut = LinearCut(coeffs=np.array([1.0, 1.0]), rhs=5.0, sense="<=")
        x = np.array([2.0, 2.0])  # 4 <= 5
        assert not is_cut_violated(cut, x)

    def test_le_violated(self):
        cut = LinearCut(coeffs=np.array([1.0, 1.0]), rhs=3.0, sense="<=")
        x = np.array([2.0, 2.0])  # 4 > 3
        assert is_cut_violated(cut, x)

    def test_ge_not_violated(self):
        cut = LinearCut(coeffs=np.array([1.0, 1.0]), rhs=3.0, sense=">=")
        x = np.array([2.0, 2.0])  # 4 >= 3
        assert not is_cut_violated(cut, x)

    def test_ge_violated(self):
        cut = LinearCut(coeffs=np.array([1.0, 1.0]), rhs=5.0, sense=">=")
        x = np.array([2.0, 2.0])  # 4 < 5
        assert is_cut_violated(cut, x)

    def test_eq_not_violated(self):
        cut = LinearCut(coeffs=np.array([1.0, 1.0]), rhs=4.0, sense="==")
        x = np.array([2.0, 2.0])
        assert not is_cut_violated(cut, x)

    def test_eq_violated(self):
        cut = LinearCut(coeffs=np.array([1.0, 1.0]), rhs=5.0, sense="==")
        x = np.array([2.0, 2.0])
        assert is_cut_violated(cut, x)


# ===================================================================
# Integration: RLT cut validity over many random points
# ===================================================================


class TestRLTSoundness:
    """Exhaustive soundness check: all four McCormick cuts hold at x[i]*x[j]."""

    @pytest.mark.parametrize(
        "bounds",
        [
            ((1.0, 3.0), (2.0, 5.0)),  # positive-positive
            ((-3.0, -1.0), (2.0, 5.0)),  # negative-positive
            ((-4.0, 2.0), (-3.0, 5.0)),  # mixed-mixed
            ((-5.0, -1.0), (-4.0, -2.0)),  # negative-negative
            ((0.0, 3.0), (0.0, 5.0)),  # zero-bounded
        ],
    )
    def test_all_cuts_valid_with_auxiliary(self, bounds):
        (xi_lb, xi_ub), (xj_lb, xj_ub) = bounds
        bt = BilinearTerm(i=0, j=1, w_index=2)
        lb = np.array([xi_lb, xj_lb, -1000.0])
        ub = np.array([xi_ub, xj_ub, 1000.0])
        cuts = generate_rlt_cuts(bt, lb, ub, n_vars=3)

        key = jax.random.PRNGKey(123)
        k1, k2 = jax.random.split(key)
        xi = np.asarray(
            xi_lb + (xi_ub - xi_lb) * jax.random.uniform(k1, (N_POINTS,), dtype=jnp.float64)
        )
        xj = np.asarray(
            xj_lb + (xj_ub - xj_lb) * jax.random.uniform(k2, (N_POINTS,), dtype=jnp.float64)
        )
        w = xi * xj

        for cut in cuts:
            for k in range(N_POINTS):
                x = np.array([xi[k], xj[k], w[k]])
                lhs = np.dot(cut.coeffs, x)
                if cut.sense == ">=":
                    assert lhs >= cut.rhs - TOL, (
                        f"Cut violated at ({xi[k]:.4f}, {xj[k]:.4f}): {lhs:.6f} < {cut.rhs:.6f}"
                    )
                elif cut.sense == "<=":
                    assert lhs <= cut.rhs + TOL, (
                        f"Cut violated at ({xi[k]:.4f}, {xj[k]:.4f}): {lhs:.6f} > {cut.rhs:.6f}"
                    )


# ===================================================================
# Bilinear term detection
# ===================================================================


class TestDetectBilinearTerms:
    """Test automatic detection of bilinear products in model expressions."""

    def test_simple_bilinear_constraint(self):
        from discopt.modeling.core import Model

        m = Model("bilinear")
        x = m.continuous("x", shape=(2,), lb=0.0, ub=3.0)
        m.minimize(x[0] + x[1])
        m.subject_to(x[0] * x[1] <= 2, name="bilinear")

        terms = detect_bilinear_terms(m)
        assert len(terms) == 1
        assert terms[0].i == 0
        assert terms[0].j == 1

    def test_no_bilinear_terms(self):
        from discopt.modeling.core import Model

        m = Model("linear")
        x = m.continuous("x", shape=(2,), lb=0.0, ub=3.0)
        m.minimize(x[0] + x[1])
        m.subject_to(x[0] + x[1] <= 5, name="linear")

        terms = detect_bilinear_terms(m)
        assert len(terms) == 0

    def test_multiple_bilinear_terms(self):
        from discopt.modeling.core import Model

        m = Model("multi_bilinear")
        x = m.continuous("x", shape=(3,), lb=0.0, ub=3.0)
        m.minimize(x[0] * x[1] + x[1] * x[2])
        m.subject_to(x[0] * x[2] <= 5, name="c1")

        terms = detect_bilinear_terms(m)
        # Should find x0*x1, x1*x2, and x0*x2
        assert len(terms) == 3
        pairs = {(t.i, t.j) for t in terms}
        assert (0, 1) in pairs
        assert (1, 2) in pairs
        assert (0, 2) in pairs

    def test_deduplication(self):
        """Same bilinear term in multiple places should be detected once."""
        from discopt.modeling.core import Model

        m = Model("dedup")
        x = m.continuous("x", shape=(2,), lb=0.0, ub=3.0)
        m.minimize(x[0] * x[1])
        m.subject_to(x[0] * x[1] <= 5, name="c1")

        terms = detect_bilinear_terms(m)
        assert len(terms) == 1


# ===================================================================
# Combined cut generation (generate_cuts_at_node)
# ===================================================================


class TestCombinedCutGeneration:
    """Test the combined OA + RLT cut generator for the solver loop."""

    def _make_convex_model_and_evaluator(self):
        from discopt._jax.nlp_evaluator import NLPEvaluator
        from discopt.modeling.core import Model

        m = Model("convex")
        x = m.continuous("x", shape=(2,), lb=-3.0, ub=3.0)
        m.minimize(x[0] + x[1])
        m.subject_to(x[0] ** 2 + x[1] ** 2 <= 4, name="circle")
        return m, NLPEvaluator(m)

    def _make_bilinear_model_and_evaluator(self):
        from discopt._jax.nlp_evaluator import NLPEvaluator
        from discopt.modeling.core import Model

        m = Model("bilinear")
        x = m.continuous("x", shape=(2,), lb=0.0, ub=3.0)
        m.minimize(x[0] + x[1])
        m.subject_to(x[0] * x[1] <= 2, name="bilinear")
        return m, NLPEvaluator(m)

    def test_convex_violated_generates_oa(self):
        """Violated convex constraint should produce OA cut."""
        m, evaluator = self._make_convex_model_and_evaluator()
        x_sol = np.array([2.0, 2.0])  # x0^2+x1^2 = 8 > 4
        lb = np.array([-3.0, -3.0])
        ub = np.array([3.0, 3.0])
        cuts = generate_cuts_at_node(
            evaluator,
            m,
            x_sol,
            lb,
            ub,
            constraint_senses=["<="],
        )
        # Should get at least 1 OA cut (no bilinear terms in this model)
        assert len(cuts) >= 1
        assert any(c.sense == "<=" for c in cuts)

    def test_convex_feasible_no_oa(self):
        """Feasible point should not produce OA cuts."""
        m, evaluator = self._make_convex_model_and_evaluator()
        x_sol = np.array([0.5, 0.5])  # x0^2+x1^2 = 0.5 < 4
        lb = np.array([-3.0, -3.0])
        ub = np.array([3.0, 3.0])
        cuts = generate_cuts_at_node(
            evaluator,
            m,
            x_sol,
            lb,
            ub,
            constraint_senses=["<="],
        )
        assert len(cuts) == 0

    def test_bilinear_model_detects_rlt(self):
        """Bilinear model should produce RLT cuts."""
        m, evaluator = self._make_bilinear_model_and_evaluator()
        # At (2, 1.5), x0*x1 = 3 > 2 so constraint is violated
        x_sol = np.array([2.0, 1.5])
        lb = np.array([0.0, 0.0])
        ub = np.array([3.0, 3.0])
        cuts = generate_cuts_at_node(
            evaluator,
            m,
            x_sol,
            lb,
            ub,
            constraint_senses=["<="],
        )
        # Should get OA cut(s) and/or RLT cut(s)
        assert len(cuts) >= 1

    def test_convex_oa_cut_validity(self):
        """OA cut from convex constraint must not cut off feasible region."""
        m, evaluator = self._make_convex_model_and_evaluator()
        x_star = np.array([1.5, 1.5])  # violated: 4.5 > 4
        lb = np.array([-3.0, -3.0])
        ub = np.array([3.0, 3.0])
        cuts = generate_cuts_at_node(
            evaluator,
            m,
            x_star,
            lb,
            ub,
            constraint_senses=["<="],
        )
        assert len(cuts) >= 1

        # Every OA cut should be satisfied at every feasible point
        # g(x) = x0^2 + x1^2 - 4 <= 0
        key = jax.random.PRNGKey(77)
        points = _random_points_in_box(key, lb, ub)

        for cut in cuts:
            if cut.sense != "<=":
                continue
            for k in range(N_POINTS):
                x = points[k]
                g_x = x[0] ** 2 + x[1] ** 2 - 4.0
                if g_x <= 0:  # feasible
                    lhs = float(np.dot(cut.coeffs, x))
                    assert lhs <= cut.rhs + 1e-6, (
                        f"OA cut violated at feasible point: lhs={lhs:.6f} > rhs={cut.rhs:.6f}"
                    )

    def test_precomputed_bilinear_terms(self):
        """Pre-detected bilinear terms should be reused."""
        m, evaluator = self._make_bilinear_model_and_evaluator()
        bilinear_terms = detect_bilinear_terms(m)
        assert len(bilinear_terms) == 1

        x_sol = np.array([2.0, 1.5])
        lb = np.array([0.0, 0.0])
        ub = np.array([3.0, 3.0])
        cuts = generate_cuts_at_node(
            evaluator,
            m,
            x_sol,
            lb,
            ub,
            constraint_senses=["<="],
            bilinear_terms=bilinear_terms,
        )
        assert len(cuts) >= 1
