"""Tests for RLT, Outer Approximation, Lift-and-Project cutting planes and CutPool.

Validates:
  - LinearCut structure and representation
  - RLT cuts: McCormick envelope validity for bilinear terms
  - OA cuts: tangent hyperplane validity for convex/nonlinear constraints
  - Separation: only violated cuts are returned
  - Integration with NLPEvaluator
  - CutPool: add/remove/purge/duplicate detection
  - Lift-and-project: valid separating hyperplane on small MIPs
  - Integration: solve_model with cutting_planes=True
"""

from __future__ import annotations

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from discopt._jax.cutting_planes import (
    BilinearTerm,
    CutPool,
    LinearCut,
    OACutGenerationReport,
    OACutSkip,
    detect_bilinear_terms,
    generate_cuts_at_node,
    generate_lift_and_project_cut,
    generate_oa_cut,
    generate_oa_cuts_from_evaluator,
    generate_oa_cuts_from_evaluator_report,
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
    """RLT cuts need an auxiliary variable; without one, none are generated."""

    def test_no_cuts_without_w_index(self):
        bt = BilinearTerm(i=0, j=1)
        lb = np.array([1.0, 2.0])
        ub = np.array([3.0, 5.0])
        assert generate_rlt_cuts(bt, lb, ub, n_vars=2) == []

    def test_separation_returns_empty_without_w_index(self):
        bt = BilinearTerm(i=0, j=1)
        lb = np.array([0.1, 0.1])
        ub = np.array([5.0, 5.0])
        x_sol = np.array([2.0, 1.0])
        assert separate_rlt_cuts(bt, x_sol, lb, ub, n_vars=2) == []


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

    def test_convex_mask_filters_nonconvex_constraints(self):
        """The convex mask should suppress OA cuts for nonconvex constraint rows."""
        from discopt._jax.nlp_evaluator import NLPEvaluator
        from discopt.modeling.core import Model

        m = Model("test_oa_mask")
        x = m.continuous("x", shape=(2,), lb=-2.0, ub=2.0)
        m.minimize(x[0] + x[1])
        m.subject_to(x[0] + x[1] <= 1.0, name="linear")
        m.subject_to(x[0] * x[1] <= 0.25, name="bilinear")
        evaluator = NLPEvaluator(m)

        cuts = generate_oa_cuts_from_evaluator(
            evaluator,
            np.array([0.5, 0.5]),
            constraint_senses=["<=", "<="],
            convex_mask=[True, False],
        )

        assert len(cuts) == 1
        np.testing.assert_allclose(cuts[0].coeffs, [1.0, 1.0], atol=1e-6)
        assert cuts[0].sense == "<="

    def test_report_records_nonconvex_mask_skip_reason(self):
        """The report exposes why direct evaluator OA skipped a row."""
        from discopt._jax.nlp_evaluator import NLPEvaluator
        from discopt.modeling.core import Model

        m = Model("test_oa_report")
        x = m.continuous("x", shape=(2,), lb=-2.0, ub=2.0)
        m.minimize(x[0] + x[1])
        m.subject_to(x[0] + x[1] <= 1.0, name="linear")
        m.subject_to(x[0] * x[1] <= 0.25, name="bilinear")
        evaluator = NLPEvaluator(m)

        report = generate_oa_cuts_from_evaluator_report(
            evaluator,
            np.array([0.5, 0.5]),
            constraint_senses=["<=", "<="],
            convex_mask=[True, False],
            skip_reasons=[None, "bilinear_not_direct_oa"],
        )

        assert isinstance(report, OACutGenerationReport)
        assert len(report.cuts) == 1
        assert report.skipped == [OACutSkip(constraint_index=1, reason="bilinear_not_direct_oa")]


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

    def test_convex_mask_skips_nonconvex_violations(self):
        """Separation should ignore violated rows that are marked nonconvex."""
        from discopt._jax.nlp_evaluator import NLPEvaluator
        from discopt.modeling.core import Model

        m = Model("test_sep_mask")
        x = m.continuous("x", shape=(2,), lb=-2.0, ub=2.0)
        m.minimize(x[0] + x[1])
        m.subject_to(x[0] ** 2 + x[1] ** 2 <= 1.0, name="circle")
        m.subject_to(x[0] * x[1] <= 0.25, name="bilinear")
        evaluator = NLPEvaluator(m)

        cuts = separate_oa_cuts(
            evaluator,
            np.array([1.0, 1.0]),
            constraint_senses=["<=", "<="],
            convex_mask=[True, False],
        )

        assert len(cuts) == 1
        np.testing.assert_allclose(cuts[0].coeffs, [2.0, 2.0], atol=1e-6)
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


# ===================================================================
# CutPool tests
# ===================================================================


class TestCutPool:
    """Test CutPool add/remove/purge/deduplication."""

    def test_empty_pool(self):
        pool = CutPool()
        assert len(pool) == 0
        assert pool.cuts == []
        A, b, senses = pool.to_constraint_arrays()
        assert A.shape[0] == 0
        assert b.shape[0] == 0
        assert senses == []

    def test_add_single_cut(self):
        pool = CutPool()
        cut = LinearCut(np.array([1.0, 2.0]), rhs=3.0, sense="<=")
        assert pool.add(cut) is True
        assert len(pool) == 1

    def test_add_duplicate_rejected(self):
        pool = CutPool()
        cut = LinearCut(np.array([1.0, 2.0]), rhs=3.0, sense="<=")
        assert pool.add(cut) is True
        assert pool.add(cut) is False
        assert len(pool) == 1

    def test_add_many(self):
        pool = CutPool()
        cuts = [
            LinearCut(np.array([1.0, 0.0]), rhs=1.0, sense="<="),
            LinearCut(np.array([0.0, 1.0]), rhs=2.0, sense="<="),
            LinearCut(np.array([1.0, 0.0]), rhs=1.0, sense="<="),  # dup
        ]
        added = pool.add_many(cuts)
        assert added == 2
        assert len(pool) == 2

    def test_get_active_cuts_violated(self):
        pool = CutPool()
        # Cut: x[0] + x[1] <= 3
        pool.add(LinearCut(np.array([1.0, 1.0]), rhs=3.0, sense="<="))
        # Cut: x[0] <= 1
        pool.add(LinearCut(np.array([1.0, 0.0]), rhs=1.0, sense="<="))

        # x = [2, 2]: first cut violated (4 > 3), second violated (2 > 1)
        active = pool.get_active_cuts(np.array([2.0, 2.0]))
        assert len(active) == 2

    def test_get_active_cuts_none_violated(self):
        pool = CutPool()
        pool.add(LinearCut(np.array([1.0, 1.0]), rhs=10.0, sense="<="))
        active = pool.get_active_cuts(np.array([1.0, 1.0]))
        assert len(active) == 0

    def test_age_and_purge(self):
        pool = CutPool()
        # Binding cut at x = [1, 1]
        pool.add(LinearCut(np.array([1.0, 1.0]), rhs=2.0, sense="<="))
        # Non-binding cut at x = [1, 1]
        pool.add(LinearCut(np.array([1.0, 0.0]), rhs=10.0, sense="<="))

        x = np.array([1.0, 1.0])
        # Age multiple times to make the non-binding cut stale
        for _ in range(12):
            pool.age_cuts(x)

        pool.purge_inactive(max_age=10)
        # Non-binding cut should be purged
        assert len(pool) == 1
        assert pool.cuts[0].rhs == 2.0

    def test_max_cuts_triggers_purge(self):
        pool = CutPool(max_cuts=5, purge_fraction=0.4)
        for i in range(6):
            pool.add(LinearCut(np.array([float(i), 0.0]), rhs=float(i), sense="<="))
        # Should have purged some (max_cuts=5, added 6)
        assert len(pool) <= 5

    def test_to_constraint_arrays(self):
        pool = CutPool()
        pool.add(LinearCut(np.array([1.0, 2.0]), rhs=3.0, sense="<="))
        pool.add(LinearCut(np.array([4.0, 5.0]), rhs=6.0, sense=">="))

        A, b, senses = pool.to_constraint_arrays()
        assert A.shape == (2, 2)
        assert b.shape == (2,)
        np.testing.assert_allclose(A[0], [1.0, 2.0])
        np.testing.assert_allclose(A[1], [4.0, 5.0])
        np.testing.assert_allclose(b, [3.0, 6.0])
        assert senses == ["<=", ">="]

    def test_different_sense_not_duplicate(self):
        pool = CutPool()
        pool.add(LinearCut(np.array([1.0, 2.0]), rhs=3.0, sense="<="))
        pool.add(LinearCut(np.array([1.0, 2.0]), rhs=3.0, sense=">="))
        assert len(pool) == 2

    def test_nearly_identical_coeffs_deduplicated(self):
        """Cuts with coefficients differing by < 1e-8 should be deduplicated."""
        pool = CutPool()
        pool.add(LinearCut(np.array([1.0, 2.0]), rhs=3.0, sense="<="))
        pool.add(LinearCut(np.array([1.0 + 1e-10, 2.0 - 1e-10]), rhs=3.0, sense="<="))
        assert len(pool) == 1


# ===================================================================
# Lift-and-Project cuts
# ===================================================================


class TestLiftAndProject:
    """Test lift-and-project cut generation for fractional binary variables."""

    def test_no_cut_for_integer_value(self):
        """Should return None when variable is already integral."""
        x = np.array([0.0, 1.0, 0.5])
        A = np.array([[1.0, 1.0, 0.0]])
        b = np.array([1.5])
        lb = np.zeros(3)
        ub = np.ones(3)
        # x[0] = 0 is integral => no cut
        assert generate_lift_and_project_cut(x, A, b, lb, ub, 0) is None
        # x[1] = 1 is integral => no cut
        assert generate_lift_and_project_cut(x, A, b, lb, ub, 1) is None

    def test_cut_for_fractional_value_with_opposite_signs(self):
        """Should return a cut when constraints have opposite-sign a_j."""
        x = np.array([0.5, 0.3, 0.8])
        # Row 0: a_j = 1 > 0, Row 1: a_j = -1 < 0 => can combine
        A = np.array([[1.0, 1.0, 1.0], [-1.0, 0.0, 1.0]])
        b = np.array([1.6, 0.4])  # slacks: 0.0 and 0.1
        lb = np.zeros(3)
        ub = np.ones(3)
        cut = generate_lift_and_project_cut(x, A, b, lb, ub, 0)
        assert cut is not None
        assert isinstance(cut, LinearCut)

    def test_cut_from_violated_constraint(self):
        """A violated constraint row should be returned as a cut."""
        x = np.array([0.5, 0.7])
        A = np.array([[1.0, 1.0]])
        b = np.array([1.0])  # slack = -0.2 (violated)
        lb = np.zeros(2)
        ub = np.ones(2)
        cut = generate_lift_and_project_cut(x, A, b, lb, ub, 0)
        assert cut is not None
        # Cut should be violated at x_sol
        assert is_cut_violated(cut, x)

    def test_cut_valid_at_integer_points(self):
        """Disjunctive cut must be valid at all LP-feasible integer points."""
        # Two constraints with opposite a_j signs
        x = np.array([0.5, 0.7, 0.3])
        A = np.array(
            [
                [1.0, 1.0, 1.0],  # a_j = 1 > 0
                [-1.0, 0.5, 0.5],  # a_j = -1 < 0
            ]
        )
        b = np.array([1.5, 0.3])
        lb = np.zeros(3)
        ub = np.ones(3)
        cut = generate_lift_and_project_cut(x, A, b, lb, ub, 0)
        if cut is None:
            pytest.skip("No cut generated for this instance")

        # Check all eight binary corners
        for x0 in [0.0, 1.0]:
            for x1 in [0.0, 1.0]:
                for x2 in [0.0, 1.0]:
                    corner = np.array([x0, x1, x2])
                    if np.all(A @ corner <= b + 1e-8):
                        lhs = float(np.dot(cut.coeffs, corner))
                        assert lhs <= cut.rhs + 1e-6, (
                            f"Cut invalid at ({x0},{x1},{x2}): {lhs:.6f} > {cut.rhs:.6f}"
                        )

    def test_no_constraints_returns_none(self):
        """With no constraints, cannot derive a valid global cut."""
        x = np.array([0.3, 0.7])
        cut = generate_lift_and_project_cut(x, None, None, np.zeros(2), np.ones(2), 0)
        assert cut is None

    def test_opposite_sign_rows_produce_cut(self):
        """Combining rows with positive and negative a_j eliminates x[j]."""
        x = np.array([0.5, 0.6])
        # Row 0: x[0] + x[1] <= 1.0 (a_j = 1 > 0, slack = -0.1, violated)
        # Row 1: -x[0] + x[1] <= 0.2 (a_j = -1 < 0, slack = 0.1 - 0.6 = -0.3)
        A = np.array([[1.0, 1.0], [-1.0, 1.0]])
        b = np.array([1.0, 0.2])
        lb = np.zeros(2)
        ub = np.ones(2)
        cut = generate_lift_and_project_cut(x, A, b, lb, ub, 0)
        assert cut is not None
        assert cut.sense == "<="

    def test_convex_combination_validity(self):
        """Cut formed from convex combination of rows must be valid everywhere."""
        x = np.array([0.5, 0.8])
        A = np.array([[1.0, 1.0], [-1.0, 2.0]])
        b = np.array([1.2, 1.0])
        lb = np.zeros(2)
        ub = np.ones(2)
        cut = generate_lift_and_project_cut(x, A, b, lb, ub, 0)
        if cut is None:
            pytest.skip("No cut generated")
        # Since the cut is a convex combination of valid constraints,
        # it must be valid at all feasible points
        for x0 in np.linspace(0, 1, 5):
            for x1 in np.linspace(0, 1, 5):
                pt = np.array([x0, x1])
                if np.all(A @ pt <= b + 1e-8):
                    lhs = float(np.dot(cut.coeffs, pt))
                    assert lhs <= cut.rhs + 1e-6


# ===================================================================
# AugmentedEvaluator tests
# ===================================================================


class TestAugmentedEvaluator:
    """Test the _AugmentedEvaluator wrapper for injecting cuts into NLP."""

    def _make_evaluator_and_pool(self):
        from discopt._jax.nlp_evaluator import NLPEvaluator
        from discopt.modeling.core import Model

        m = Model("augtest")
        x = m.continuous("x", shape=(2,), lb=-5.0, ub=5.0)
        m.minimize(x[0] ** 2 + x[1] ** 2)
        m.subject_to(x[0] + x[1] >= 1, name="sum_lb")
        evaluator = NLPEvaluator(m)

        pool = CutPool()
        # Add cut: x[0] <= 3
        pool.add(LinearCut(np.array([1.0, 0.0]), rhs=3.0, sense="<="))
        # Add cut: x[1] >= -2 => -x[1] <= 2
        pool.add(LinearCut(np.array([0.0, 1.0]), rhs=-2.0, sense=">="))

        return evaluator, pool

    def test_n_constraints_augmented(self):
        from discopt.solver import _AugmentedEvaluator

        evaluator, pool = self._make_evaluator_and_pool()
        aug = _AugmentedEvaluator(evaluator, pool)
        assert aug.n_constraints == evaluator.n_constraints + 2

    def test_evaluate_constraints_shape(self):
        from discopt.solver import _AugmentedEvaluator

        evaluator, pool = self._make_evaluator_and_pool()
        aug = _AugmentedEvaluator(evaluator, pool)
        x = np.array([1.0, 1.0])
        cons = aug.evaluate_constraints(x)
        assert cons.shape == (3,)  # 1 original + 2 cuts

    def test_evaluate_jacobian_shape(self):
        from discopt.solver import _AugmentedEvaluator

        evaluator, pool = self._make_evaluator_and_pool()
        aug = _AugmentedEvaluator(evaluator, pool)
        x = np.array([1.0, 1.0])
        jac = aug.evaluate_jacobian(x)
        assert jac.shape == (3, 2)

    def test_objective_unchanged(self):
        from discopt.solver import _AugmentedEvaluator

        evaluator, pool = self._make_evaluator_and_pool()
        aug = _AugmentedEvaluator(evaluator, pool)
        x = np.array([1.5, 2.0])
        assert abs(aug.evaluate_objective(x) - evaluator.evaluate_objective(x)) < 1e-10

    def test_gradient_unchanged(self):
        from discopt.solver import _AugmentedEvaluator

        evaluator, pool = self._make_evaluator_and_pool()
        aug = _AugmentedEvaluator(evaluator, pool)
        x = np.array([1.5, 2.0])
        np.testing.assert_allclose(
            aug.evaluate_gradient(x), evaluator.evaluate_gradient(x), atol=1e-10
        )

    def test_augmented_constraint_bounds(self):
        from discopt.solver import _AugmentedEvaluator

        evaluator, pool = self._make_evaluator_and_pool()
        aug = _AugmentedEvaluator(evaluator, pool)
        original_bounds = [(0.0, 1e20)]
        new_bounds = aug.get_augmented_constraint_bounds(original_bounds)
        assert len(new_bounds) == 3
        assert new_bounds[0] == (0.0, 1e20)
        assert new_bounds[1] == (-1e20, 0.0)
        assert new_bounds[2] == (-1e20, 0.0)

    def test_augmented_jax_constraint_fn(self):
        from discopt.solver import _AugmentedEvaluator

        evaluator, pool = self._make_evaluator_and_pool()
        aug = _AugmentedEvaluator(evaluator, pool)
        cons_fn = aug._cons_fn
        assert cons_fn is not None
        x_jax = jnp.array([2.0, 1.0])
        result = cons_fn(x_jax)
        assert result.shape == (3,)

    def test_empty_pool_passthrough(self):
        from discopt.solver import _AugmentedEvaluator

        evaluator, _ = self._make_evaluator_and_pool()
        empty_pool = CutPool()
        aug = _AugmentedEvaluator(evaluator, empty_pool)
        assert aug.n_constraints == evaluator.n_constraints
        x = np.array([1.0, 1.0])
        np.testing.assert_allclose(
            aug.evaluate_constraints(x),
            evaluator.evaluate_constraints(x),
            atol=1e-10,
        )


# ===================================================================
# Integration: solve_model with cutting_planes=True
# ===================================================================


class TestSolverCutIntegration:
    """Integration tests: solve_model with cutting_planes=True."""

    def test_simple_minlp_with_cuts(self):
        """A simple MINLP should solve correctly with cuts enabled."""
        from discopt.modeling.core import Model

        m = Model("minlp_cuts")
        x = m.continuous("x", lb=0.0, ub=5.0)
        y = m.binary("y")
        m.minimize(x + 2 * y)
        m.subject_to(x + y >= 1.5, name="c1")

        result = m.solve(cutting_planes=True, max_nodes=200)
        assert result.status in ("optimal", "feasible")
        assert result.objective is not None

    @pytest.mark.slow
    def test_bilinear_minlp_with_cuts(self):
        """Bilinear MINLP should solve with OA + RLT cuts."""
        from discopt.modeling.core import Model

        m = Model("bilinear_minlp")
        x = m.continuous("x", lb=0.0, ub=3.0)
        y = m.binary("y")
        m.minimize(x**2 + 3 * y)
        m.subject_to(x + y >= 1, name="c1")
        m.subject_to(x * y <= 2, name="bilinear")

        result = m.solve(cutting_planes=True, max_nodes=500)
        assert result.status in ("optimal", "feasible")

    def test_cuts_do_not_cut_off_optimum(self):
        """Cuts should not invalidate the correct optimum."""
        from discopt.modeling.core import Model

        # Known optimal: x=0.5, y=1, obj = 0.5 + 2 = 2.5
        m = Model("correctness")
        x = m.continuous("x", lb=0.0, ub=5.0)
        y = m.binary("y")
        m.minimize(x + 2 * y)
        m.subject_to(x + y >= 1.5, name="c1")

        result_with = m.solve(cutting_planes=True, max_nodes=200)
        result_without = m.solve(cutting_planes=False, max_nodes=200)

        if result_with.status == "optimal" and result_without.status == "optimal":
            # Objectives should match (cuts should not change optimum)
            assert abs(result_with.objective - result_without.objective) < 0.1
