"""Tests for the general-purpose Outer Approximation (OA) solver.

Requires highspy for the MILP master problem.
"""

import discopt.modeling as dm
import numpy as np
import pytest

try:
    import highspy  # noqa: F401

    HAS_HIGHS = True
except ImportError:
    HAS_HIGHS = False

pytestmark = pytest.mark.skipif(not HAS_HIGHS, reason="highspy not installed")

ABS_TOL = 1e-3
REL_TOL = 1e-3
INTEGRALITY_TOL = 1e-5


# ── Helper ────────────────────────────────────────────────────


def _solve_oa(model, **kwargs):
    """Solve model with OA and return result."""
    defaults = dict(gdp_method="oa", time_limit=60)
    defaults.update(kwargs)
    return model.solve(**defaults)


def _assert_optimal(result, expected_obj, abs_tol=ABS_TOL):
    assert result.status in ("optimal", "feasible"), (
        f"Expected optimal/feasible, got {result.status}"
    )
    assert result.objective == pytest.approx(expected_obj, abs=abs_tol)


def _assert_integer_feasible(result, int_var_names, model):
    """Check that integer/binary variables are integral."""
    for name in int_var_names:
        vals = np.atleast_1d(result.x[name])
        for v in vals.flat:
            assert abs(v - round(v)) < INTEGRALITY_TOL, f"Variable {name} = {v} is not integral"


# ── Convex MINLP ─────────────────────────────────────────────


class TestOAConvexMINLP:
    """Convex MINLP problems where OA should find the global optimum."""

    @pytest.mark.slow
    def test_simple_quadratic_binary(self):
        """min x^2 + y, y in {0,1}, x + y >= 1, x in [0, 2].

        y=0 → x >= 1 → obj = 1
        y=1 → x >= 0 → obj = 1
        Both give obj=1; OA should find optimal=1.
        """
        m = dm.Model("simple_qb")
        x = m.continuous("x", lb=0, ub=2)
        y = m.binary("y")
        m.minimize(x**2 + y)
        m.subject_to(x + y >= 1)

        result = _solve_oa(m)
        _assert_optimal(result, 1.0)

    def test_simple_minlp_from_examples(self):
        """Example simple MINLP: min x1^2 + x2^2 + x3, x3 binary.

        With x1 + x2 >= 1 and x1^2 + x2 <= 3.
        Optimal: x1=x2=0.5, x3=0, obj=0.5.
        """
        from discopt.modeling.examples import example_simple_minlp

        m = example_simple_minlp()
        result = _solve_oa(m)
        _assert_optimal(result, 0.5, abs_tol=0.05)
        _assert_integer_feasible(result, ["x3"], m)

    @pytest.mark.slow
    def test_convex_with_multiple_binaries(self):
        """min (x-3)^2 + 2*y1 + 3*y2, y1+y2 <= 1, x <= 2*y1 + 4*y2.

        y1=0,y2=0 → x <= 0 → obj = 9
        y1=1,y2=0 → x <= 2 → obj = 1 + 2 = 3
        y1=0,y2=1 → x <= 4 → obj = (3-3)^2 + 3 = 3
        Optimal: y1=1,y2=0,x=2 or y1=0,y2=1,x=3, both obj=3.
        """
        m = dm.Model("multi_binary")
        x = m.continuous("x", lb=0, ub=5)
        y1 = m.binary("y1")
        y2 = m.binary("y2")
        m.minimize((x - 3) ** 2 + 2 * y1 + 3 * y2)
        m.subject_to(y1 + y2 <= 1)
        m.subject_to(x <= 2 * y1 + 4 * y2)

        result = _solve_oa(m)
        _assert_optimal(result, 3.0, abs_tol=0.1)

    def test_linear_objective_nonlinear_constraints(self):
        """min x + 10*y, x^2 <= 4*y, x >= 0.5, y in {0,1}.

        y=0 → x^2 <= 0 → infeasible (x >= 0.5)
        y=1 → x^2 <= 4, x >= 0.5 → x=0.5, obj = 0.5 + 10 = 10.5
        """
        m = dm.Model("lin_obj")
        x = m.continuous("x", lb=0.5, ub=3)
        y = m.binary("y")
        m.minimize(x + 10 * y)
        m.subject_to(x**2 - 4 * y <= 0)

        result = _solve_oa(m)
        _assert_optimal(result, 10.5, abs_tol=0.5)
        assert result.x["y"] == pytest.approx(1.0, abs=INTEGRALITY_TOL)


# ── Non-convex MINLP ─────────────────────────────────────────


class TestOANonConvex:
    """Non-convex problems: OA may find local optimum, not global."""

    @pytest.mark.slow
    def test_nonconvex_finds_feasible(self):
        """min -x*y_bin, x in [0,2], y_bin in {0,1}, x <= 1 + y_bin.

        y=0 → x <= 1 → obj = 0 (x*0)
        y=1 → x <= 2 → obj = -2 (x=2, y=1)
        OA should at least find a feasible solution.
        """
        m = dm.Model("nonconvex")
        x = m.continuous("x", lb=0, ub=2)
        y = m.binary("y")
        m.minimize(-(x * y))
        m.subject_to(x - y <= 1)

        result = _solve_oa(m)
        assert result.status in ("optimal", "feasible")
        assert result.objective is not None

    def test_nonconvex_objective_skips_objective_oa_cuts(self, monkeypatch):
        """A nonconvex objective must not produce OA objective cuts or certified bounds."""
        from discopt._jax import cutting_planes

        calls = []
        real_generate = cutting_planes.generate_objective_oa_cut

        def wrapped_generate(*args, **kwargs):
            calls.append((args, kwargs))
            return real_generate(*args, **kwargs)

        monkeypatch.setattr(cutting_planes, "generate_objective_oa_cut", wrapped_generate)

        m = dm.Model("oa_nonconvex_objective")
        x = m.continuous("x", lb=0, ub=2)
        y = m.binary("y")
        m.subject_to(x <= 1 + y)
        m.minimize(-(x * y))

        result = _solve_oa(m, max_nodes=6)

        assert calls == []
        assert result.status == "feasible"
        assert result.objective is not None
        assert result.bound is None
        assert result.gap is None


# ── Edge Cases ────────────────────────────────────────────────


class TestOAEdgeCases:
    """Edge cases and degenerate problems."""

    def test_pure_nlp_no_integers(self):
        """No integer variables: OA should solve a single NLP."""
        m = dm.Model("pure_nlp")
        x = m.continuous("x", lb=-5, ub=5)
        m.minimize(x**2)

        result = _solve_oa(m)
        _assert_optimal(result, 0.0, abs_tol=0.01)

    def test_pure_milp_all_linear(self):
        """All-linear MINLP: OA should converge in one iteration."""
        m = dm.Model("pure_milp")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        m.minimize(x + 5 * y)
        m.subject_to(x + y >= 1)

        result = _solve_oa(m)
        _assert_optimal(result, 1.0, abs_tol=0.1)

    def test_infeasible_model(self):
        """Infeasible MINLP: contradictory constraints."""
        m = dm.Model("infeasible")
        x = m.continuous("x", lb=0, ub=1)
        y = m.binary("y")
        m.minimize(x + y)
        m.subject_to(x >= 2)  # infeasible: x in [0,1] but x >= 2

        result = _solve_oa(m)
        assert result.status == "infeasible"

    def test_single_iteration_optimal(self):
        """NLP relaxation is already integer-feasible → immediate convergence."""
        m = dm.Model("trivial")
        x = m.continuous("x", lb=0, ub=5)
        y = m.binary("y")
        # Optimal at y=1, x=0 → obj=1. Relaxation likely finds this.
        m.minimize(x**2 + y)
        m.subject_to(y >= 1)

        result = _solve_oa(m)
        _assert_optimal(result, 1.0, abs_tol=0.1)


# ── Infeasible NLP Handling ───────────────────────────────────


class TestOAInfeasibleNLP:
    """Tests for handling infeasible NLP subproblems."""

    def test_some_assignments_infeasible(self):
        """Problem where one binary assignment makes NLP infeasible.

        y=0: x^2 <= -1 (infeasible)
        y=1: x^2 <= 3, min x^2 + 1 → x=0, obj=1
        """
        m = dm.Model("partial_infeas")
        x = m.continuous("x", lb=-2, ub=2)
        y = m.binary("y")
        m.minimize(x**2 + y)
        m.subject_to(x**2 - 4 * y + 1 <= 0)

        result = _solve_oa(m)
        _assert_optimal(result, 1.0, abs_tol=0.5)
        assert result.x["y"] == pytest.approx(1.0, abs=INTEGRALITY_TOL)


# ── ECP Mode ─────────────────────────────────────────────────


class TestECPMode:
    """Extended Cutting Plane mode (no NLP subproblem solves)."""

    def test_ecp_convex_minlp(self):
        """ECP should converge on a convex MINLP."""
        m = dm.Model("ecp_test")
        x = m.continuous("x", lb=0, ub=5)
        y = m.binary("y")
        m.minimize(x**2 + y)
        m.subject_to(x + y >= 1)

        result = _solve_oa(m, ecp_mode=True)
        _assert_optimal(result, 1.0, abs_tol=0.2)

    def test_ecp_linear_objective(self):
        """ECP with linear objective + nonlinear constraints."""
        m = dm.Model("ecp_lin")
        x = m.continuous("x", lb=0, ub=3)
        y = m.binary("y")
        m.minimize(x + 10 * y)
        m.subject_to(x**2 - 4 * y <= 0)
        m.subject_to(x >= 0.5)

        result = _solve_oa(m, ecp_mode=True)
        assert result.status in ("optimal", "feasible")


# ── Equality Relaxation ──────────────────────────────────────


class TestEqualityRelaxation:
    """Tests for equality relaxation (ER) strategy."""

    def test_er_helps_nonlinear_equality(self):
        """Nonlinear equality that may cause master infeasibility.

        min x^2 + y, x^2 == y (nonlinear equality), x in [0,2], y in {0,1}.
        y=0 → x=0, obj=0
        y=1 → x=1, obj=2

        With ER, the equality is relaxed to x^2 <= y in OA cuts.
        """
        m = dm.Model("er_test")
        x = m.continuous("x", lb=0, ub=2)
        y = m.binary("y")
        m.minimize(x**2 + y)
        m.subject_to(x**2 - y == 0)

        result = _solve_oa(m, equality_relaxation=True)
        assert result.status in ("optimal", "feasible")


# ── Regression vs B&B ────────────────────────────────────────


class TestOAMatchesBnB:
    """OA results should be close to B&B on shared test problems."""

    def test_simple_minlp_matches(self):
        """Compare OA and default B&B on simple MINLP."""
        from discopt.modeling.examples import example_simple_minlp

        m_oa = example_simple_minlp()
        m_bb = example_simple_minlp()

        result_oa = _solve_oa(m_oa)
        result_bb = m_bb.solve(time_limit=60)

        # Both should find feasible solutions
        assert result_oa.status in ("optimal", "feasible")
        assert result_bb.status in ("optimal", "feasible")

        # Objectives should be close (within tolerance)
        if result_oa.objective is not None and result_bb.objective is not None:
            assert result_oa.objective == pytest.approx(result_bb.objective, abs=0.5)
