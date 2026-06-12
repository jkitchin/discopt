"""Validation of the POUNCE QP path (roadmap P0.4, mirrors test_lp_pounce).

Covers convex QPs across inequality / equality / bounded / free-variable
cases with HiGHS as the cross-check oracle (P0.5), the IPM corner cases
(infeasible -> INFEASIBLE with the elastic Phase-1 certificate on request,
unbounded -> UNBOUNDED), dual sign parity with HiGHS, and the explicit
refusal of integrality (POUNCE has no MIQP).

Skipped entirely when POUNCE is not importable.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pounce")

from discopt.solvers import SolveStatus  # noqa: E402
from discopt.solvers.qp_pounce import solve_qp  # noqa: E402

try:
    from discopt.solvers.qp_highs import solve_qp as solve_qp_highs

    _HIGHS = True
except ImportError:  # pragma: no cover
    _HIGHS = False

_OBJ_TOL = 1e-5


class TestBasicQP:
    """min (x1-1)^2 + (x2-2)^2 s.t. x1+x2 <= 2, 0<=x<=10.

    In 0.5 x'Qx + c'x form: Q = 2I, c = (-2, -4) (constant +5 dropped).
    Optimum x = (0.5, 1.5), objective -4.5 (i.e. true value 0.5).
    """

    _KW = dict(
        Q=2 * np.eye(2),
        c=np.array([-2.0, -4.0]),
        A_ub=np.array([[1.0, 1.0]]),
        b_ub=np.array([2.0]),
        bounds=[(0.0, 10.0), (0.0, 10.0)],
    )

    def test_optimal(self):
        r = solve_qp(**self._KW)
        assert r.status == SolveStatus.OPTIMAL
        assert abs(r.objective - (-4.5)) < _OBJ_TOL
        np.testing.assert_allclose(r.x, [0.5, 1.5], atol=1e-4)

    @pytest.mark.skipif(not _HIGHS, reason="HiGHS oracle unavailable")
    def test_duals_match_highs(self):
        rp = solve_qp(**self._KW)
        rh = solve_qp_highs(**self._KW)
        np.testing.assert_allclose(rp.dual_values, rh.dual_values, atol=1e-5)
        np.testing.assert_allclose(rp.reduced_costs, rh.reduced_costs, atol=1e-5)


class TestEqualityFreeVars:
    """min x1^2 + x2^2 s.t. x1+x2 = 4, x free -> x=(2,2), obj 8, dual 4."""

    def test_solution_and_dual(self):
        r = solve_qp(
            Q=2 * np.eye(2),
            c=np.zeros(2),
            A_eq=np.array([[1.0, 1.0]]),
            b_eq=np.array([4.0]),
        )
        assert r.status == SolveStatus.OPTIMAL
        assert abs(r.objective - 8.0) < _OBJ_TOL
        np.testing.assert_allclose(r.x, [2.0, 2.0], atol=1e-4)
        np.testing.assert_allclose(r.dual_values, [4.0], atol=1e-4)


class TestBoundsOnly:
    """min (x1-3)^2 over 0<=x1<=2 -> x1=2 at the bound, obj (in Qc form) -8."""

    def test_active_bound(self):
        r = solve_qp(Q=np.array([[2.0]]), c=np.array([-6.0]), bounds=[(0.0, 2.0)])
        assert r.status == SolveStatus.OPTIMAL
        np.testing.assert_allclose(r.x, [2.0], atol=1e-4)
        assert abs(r.objective - (-8.0)) < _OBJ_TOL


class TestInfeasible:
    _KW = dict(
        Q=2 * np.eye(2),
        c=np.zeros(2),
        A_ub=np.array([[1.0, 1.0], [-1.0, -1.0]]),
        b_ub=np.array([1.0, -10.0]),
        bounds=[(0.0, 10.0), (0.0, 10.0)],
    )

    def test_infeasible_status(self):
        r = solve_qp(**self._KW)
        assert r.status == SolveStatus.INFEASIBLE
        assert r.x is None

    def test_certificate_on_request(self):
        r = solve_qp(**self._KW, certificate=True)
        assert r.status == SolveStatus.INFEASIBLE
        cert = r.infeasibility_certificate
        assert cert is not None
        assert cert.total_violation > 1.0  # gap between <=1 and >=10

    def test_inconsistent_equalities_certified(self):
        """The Phase-1 disambiguation applies to QPs unchanged: the quadratic
        objective is irrelevant to the (linear) feasibility question."""
        r = solve_qp(
            Q=2 * np.eye(2),
            c=np.zeros(2),
            A_eq=np.array([[1.0, 1.0], [1.0, 1.0]]),
            b_eq=np.array([1.0, 5.0]),
            bounds=[(0.0, 10.0), (0.0, 10.0)],
            options={"max_iter": 300},
        )
        assert r.status == SolveStatus.INFEASIBLE
        assert r.infeasibility_certificate is not None
        assert r.infeasibility_certificate.total_violation == pytest.approx(4.0, abs=1e-4)


class TestUnbounded:
    def test_unbounded_null_direction(self):
        """Q singular with linear descent along its null space -> unbounded."""
        r = solve_qp(
            Q=np.diag([2.0, 0.0]),
            c=np.array([0.0, -1.0]),
            bounds=[(0.0, 10.0), (0.0, float("inf"))],
        )
        assert r.status == SolveStatus.UNBOUNDED
        assert r.x is None


class TestIntegralityRefused:
    def test_integer_marks_raise(self):
        with pytest.raises(ValueError, match="integrality"):
            solve_qp(Q=2 * np.eye(2), c=np.zeros(2), integrality=np.array([1, 0]))

    def test_all_continuous_integrality_ok(self):
        r = solve_qp(
            Q=2 * np.eye(2),
            c=np.zeros(2),
            integrality=np.zeros(2),
            bounds=[(1.0, 5.0), (1.0, 5.0)],
        )
        assert r.status == SolveStatus.OPTIMAL
        np.testing.assert_allclose(r.x, [1.0, 1.0], atol=1e-4)


class TestDimensionMismatch:
    def test_Q_shape(self):
        with pytest.raises(ValueError, match="Q has shape"):
            solve_qp(Q=np.eye(3), c=np.zeros(2))

    def test_bounds_length(self):
        with pytest.raises(ValueError, match="bounds"):
            solve_qp(Q=2 * np.eye(2), c=np.zeros(2), bounds=[(0.0, 1.0)])


@pytest.mark.skipif(not _HIGHS, reason="HiGHS oracle unavailable")
class TestHighsOracle:
    """Cross-check on random strictly convex QPs — with independent
    verification, because the oracle itself can fail: on seed=1 HiGHS returns
    a constraint-violating point labeled kOptimal (max violation ~7.5, obj
    83.18 vs the true 1.672 confirmed by SLSQP). POUNCE's point is therefore
    verified feasible directly, and objectives are only compared on instances
    where HiGHS's own point is feasible.
    """

    @pytest.mark.parametrize("seed", list(range(8)))
    def test_pounce_feasible_and_matches_feasible_highs(self, seed):
        rng = np.random.default_rng(seed)
        n, m = 4, 5
        M = rng.standard_normal((n, n))
        Q = M @ M.T + n * np.eye(n)  # strictly convex
        c = rng.standard_normal(n)
        A_ub = rng.standard_normal((m, n))
        x_feas = rng.uniform(0, 1, n)
        b_ub = A_ub @ x_feas + rng.uniform(0.5, 1.5, m)
        bounds = [(-5.0, 5.0)] * n

        rp = solve_qp(Q=Q, c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        assert rp.status == SolveStatus.OPTIMAL
        # Independent feasibility verification of POUNCE's point.
        assert np.max(A_ub @ rp.x - b_ub) < 1e-6
        assert np.all(np.abs(rp.x) <= 5.0 + 1e-6)

        rh = solve_qp_highs(Q=Q, c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        if rh.status != SolveStatus.OPTIMAL or np.max(A_ub @ rh.x - b_ub) > 1e-6:
            return  # oracle failed on this instance (e.g. seed=1); nothing to compare
        assert abs(rp.objective - rh.objective) < 1e-4, (
            f"seed={seed}: POUNCE {rp.objective} vs HiGHS {rh.objective}"
        )
        # Strictly convex -> unique optimum: primals agree too.
        np.testing.assert_allclose(rp.x, rh.x, atol=1e-3)

    def test_seed1_highs_failure_is_caught_by_pounce(self):
        """The known oracle-failure instance: POUNCE finds the true optimum
        (1.672, SLSQP-confirmed) where HiGHS returns an infeasible point."""
        rng = np.random.default_rng(1)
        n, m = 4, 5
        M = rng.standard_normal((n, n))
        Q = M @ M.T + n * np.eye(n)
        c = rng.standard_normal(n)
        A_ub = rng.standard_normal((m, n))
        x_feas = rng.uniform(0, 1, n)
        b_ub = A_ub @ x_feas + rng.uniform(0.5, 1.5, m)
        rp = solve_qp(Q=Q, c=c, A_ub=A_ub, b_ub=b_ub, bounds=[(-5.0, 5.0)] * n)
        assert rp.status == SolveStatus.OPTIMAL
        assert np.max(A_ub @ rp.x - b_ub) < 1e-6
        assert rp.objective == pytest.approx(1.671976, abs=1e-3)
