"""Matrix-form MILP via the self-hosted B&B (roadmap P4).

``milp_pounce.solve_milp`` exposes the discopt Rust-tree B&B (POUNCE LP
relaxations) behind the ``milp_simplex.solve_milp`` signature/``MILPResult``
contract, by building a Model from the matrices and running
``_solve_milp_bb(prefer_pounce=True)``. This is what lets the matrix-form
MILP consumers (OA/GDP-LOA masters, milp_relaxation) run with only POUNCE.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pounce")

from discopt.solvers import SolveStatus  # noqa: E402
from discopt.solvers.milp_pounce import solve_milp  # noqa: E402

# Cross-check oracle: the exact Rust simplex B&B (HiGHS was removed, issue #356).
try:
    from discopt.solvers.milp_simplex import solve_milp as ref_milp

    _HIGHS = True
except ImportError:  # pragma: no cover
    _HIGHS = False


class TestKnownInstances:
    def test_knapsack(self):
        r = solve_milp(
            c=np.array([-3.0, -4.0, -5.0]),
            A_ub=np.array([[2.0, 3.0, 4.0]]),
            b_ub=np.array([6.0]),
            bounds=[(0, 1)] * 3,
            integrality=np.array([1, 1, 1]),
        )
        assert r.status == SolveStatus.OPTIMAL
        assert abs(r.objective - (-8.0)) < 1e-4  # x1=1,x3=1 (weight 6)

    def test_mixed_integer_continuous(self):
        # min -x1 - 2*x2 + x3, x1+x2+x3 <= 4.5, x1,x3 int, x2 cont, all in [0,5]
        r = solve_milp(
            c=np.array([-1.0, -2.0, 1.0]),
            A_ub=np.array([[1.0, 1.0, 1.0]]),
            b_ub=np.array([4.5]),
            bounds=[(0, 5), (0, 5), (0, 5)],
            integrality=np.array([1, 0, 1]),
        )
        assert r.status == SolveStatus.OPTIMAL
        assert abs(r.objective - (-9.0)) < 1e-4

    def test_infeasible(self):
        r = solve_milp(
            c=np.array([1.0, 1.0]),
            A_ub=np.array([[1.0, 1.0], [-1.0, -1.0]]),
            b_ub=np.array([1.0, -10.0]),
            bounds=[(0, 5), (0, 5)],
            integrality=np.array([1, 1]),
        )
        assert r.status == SolveStatus.INFEASIBLE
        assert r.x is None

    def test_all_continuous_degenerates_to_lp(self):
        r = solve_milp(
            c=np.array([-1.0, -2.0]),
            A_ub=np.array([[1.0, 1.0]]),
            b_ub=np.array([10.0]),
            bounds=[(0, 10), (0, 10)],
            integrality=None,
        )
        assert r.status == SolveStatus.OPTIMAL
        assert abs(r.objective - (-20.0)) < 1e-4
        np.testing.assert_allclose(r.x, [0.0, 10.0], atol=1e-4)


@pytest.mark.skipif(not _HIGHS, reason="HiGHS oracle unavailable")
class TestHighsAgreement:
    @pytest.mark.parametrize("seed", list(range(12)))
    def test_objective_matches_highs(self, seed):
        rng = np.random.default_rng(seed)
        n, m = 4, 3
        c = rng.integers(-5, 6, n).astype(float)
        A = rng.integers(0, 4, (m, n)).astype(float)
        b = (A @ rng.integers(0, 4, n) + rng.integers(1, 5, m)).astype(float)
        kw = dict(c=c, A_ub=A, b_ub=b, bounds=[(0, 5)] * n, integrality=np.ones(n))
        rp = solve_milp(**kw)
        rh = ref_milp(**kw)
        if rh.status != SolveStatus.OPTIMAL:
            return  # only compare where HiGHS certified an optimum
        assert rp.status in (SolveStatus.OPTIMAL, SolveStatus.ITERATION_LIMIT)
        assert rp.objective is not None
        assert abs(rp.objective - rh.objective) < 1e-3, (
            f"seed={seed}: POUNCE {rp.objective} vs HiGHS {rh.objective}"
        )


class TestObjectiveOffsetAndEmpty:
    def test_zero_objective_feasibility(self):
        # All-zero c: any feasible integer point; status optimal, obj 0.
        r = solve_milp(
            c=np.zeros(2),
            A_ub=np.array([[1.0, 1.0]]),
            b_ub=np.array([3.0]),
            bounds=[(0, 5), (0, 5)],
            integrality=np.array([1, 1]),
        )
        assert r.status == SolveStatus.OPTIMAL
        assert abs(r.objective) < 1e-6
