"""Tests for GDPopt LOA solver.

These tests require highspy for the MILP master problem.
Skipped if highspy is not installed.
"""

import numpy as np
import pytest

try:
    import highspy  # noqa: F401

    HAS_HIGHS = True
except ImportError:
    HAS_HIGHS = False

pytestmark = [pytest.mark.slow, pytest.mark.skipif(not HAS_HIGHS, reason="highspy not installed")]


class TestGDPoptLOA:
    def test_simple_disjunction(self):
        """Simple 2-disjunct problem: min x s.t. (x<=3) or (x>=7)."""
        import discopt.modeling as dm

        m = dm.Model("loa_simple")
        x = m.continuous("x", lb=0, ub=10)
        m.either_or([[x <= 3], [x >= 7]], name="choice")
        m.minimize(x)

        result = m.solve(time_limit=60, gdp_method="loa")
        assert result.status in ("optimal", "feasible")
        assert result.objective == pytest.approx(0.0, abs=1e-2)

    def test_indicator_constraint(self):
        """Indicator: if y=1 then x<=5. Minimize x + 10*y."""
        import discopt.modeling as dm

        m = dm.Model("loa_indicator")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        m.if_then(y, [x <= 5])
        m.minimize(x + 10 * y)

        result = m.solve(time_limit=60, gdp_method="loa")
        assert result.status in ("optimal", "feasible")
        # Optimal: y=0, x=0 → obj=0
        assert result.objective == pytest.approx(0.0, abs=1e-2)

    def test_nonlinear_constraints(self):
        """GDP with nonlinear constraints — tests OA cut generation."""
        import discopt.modeling as dm

        m = dm.Model("loa_nonlinear")
        x = m.continuous("x", lb=0, ub=5)
        y = m.continuous("y", lb=0, ub=5)

        # Either (x^2 + y <= 4) or (x + y^2 <= 4)
        m.either_or(
            [
                [x**2 + y <= 4, x >= 0.5],
                [x + y**2 <= 4, y >= 0.5],
            ],
            name="mode",
        )
        m.minimize(x + y)

        result = m.solve(time_limit=60, gdp_method="loa")
        assert result.status in ("optimal", "feasible")
        # Should find a feasible solution with moderate objective
        assert result.objective < 5.0

    def test_matches_bigm_result(self):
        """LOA and big-M should produce same optimal on a simple problem."""
        import discopt.modeling as dm

        def build_model():
            m = dm.Model("compare")
            x = m.continuous("x", lb=0, ub=10)
            m.either_or([[x <= 3], [x >= 7]], name="choice")
            m.minimize(x)
            return m

        # Solve with big-M
        m1 = build_model()
        r1 = m1.solve(time_limit=30, gdp_method="big-m")

        # Solve with LOA
        m2 = build_model()
        r2 = m2.solve(time_limit=60, gdp_method="loa")

        assert r1.status in ("optimal", "feasible")
        assert r2.status in ("optimal", "feasible")
        assert r1.objective == pytest.approx(r2.objective, abs=1e-2)

    def test_disjunct_block_with_loa(self):
        """Disjunct block API works with LOA solver."""
        import discopt.modeling as dm

        m = dm.Model("loa_block")
        x = m.continuous("x", lb=0, ub=10)

        d1 = m.make_disjunct("low")
        d1.subject_to(x <= 3)

        d2 = m.make_disjunct("high")
        d2.subject_to(x >= 7)

        m.add_disjunction([d1, d2])
        m.minimize(x)

        result = m.solve(time_limit=60, gdp_method="loa")
        assert result.status in ("optimal", "feasible")
        assert result.objective == pytest.approx(0.0, abs=1e-2)


class TestMILPHiGHS:
    def test_simple_milp(self):
        """Simple knapsack-style MILP."""
        from discopt.solvers.milp_highs import solve_milp

        # max 5x + 4y s.t. x + y <= 5, x,y integer in [0, 5]
        result = solve_milp(
            c=np.array([-5.0, -4.0]),  # minimize negative = maximize
            A_ub=np.array([[1.0, 1.0]]),
            b_ub=np.array([5.0]),
            bounds=[(0, 5), (0, 5)],
            integrality=np.array([1, 1]),
        )
        assert result.status.value == "optimal"
        assert result.objective == pytest.approx(-25.0, abs=1e-6)
        # x=5, y=0 → 5*5 + 4*0 = 25
        assert result.x[0] == pytest.approx(5.0, abs=1e-4)

    def test_lp_fallback(self):
        """Without integrality, degenerates to LP."""
        from discopt.solvers.milp_highs import solve_milp

        result = solve_milp(
            c=np.array([1.0, 1.0]),
            A_ub=np.array([[1.0, 0.0], [0.0, 1.0]]),
            b_ub=np.array([3.0, 3.0]),
            bounds=[(0, 5), (0, 5)],
        )
        assert result.status.value == "optimal"
        assert result.objective == pytest.approx(0.0, abs=1e-6)

    def test_infeasible(self):
        """Infeasible MILP."""
        from discopt.solvers.milp_highs import solve_milp

        result = solve_milp(
            c=np.array([1.0]),
            A_eq=np.array([[1.0]]),
            b_eq=np.array([0.5]),  # x must be 0.5, but integer
            bounds=[(0, 1)],
            integrality=np.array([1]),
        )
        assert result.status.value == "infeasible"
