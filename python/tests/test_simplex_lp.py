"""Rust revised-simplex LP solver vs HiGHS (roadmap P1).

``discopt._rust.solve_lp_py`` solves the standard-form LP
``min cᵀx s.t. A x = b, lb ≤ x ≤ ub`` with the bounded-variable two-phase primal
simplex (feral LU backend). This is the cold-solve correctness gate: on random
feasible/bounded LPs its optimum must match HiGHS within 1e-6, and it must
detect infeasibility/unboundedness on constructed cases.
"""

from __future__ import annotations

import numpy as np
import pytest

rust = pytest.importorskip("discopt._rust")
if not hasattr(rust, "solve_lp_py"):
    pytest.skip("simplex LP binding not built", allow_module_level=True)
highspy = pytest.importorskip("highspy")  # noqa: F841

from discopt.solvers import SolveStatus  # noqa: E402
from discopt.solvers.lp_highs import solve_lp as highs_lp  # noqa: E402


def _rand_standard_form(seed, m, n):
    """Random feasible+bounded standard-form LP: A x = b with x in [0,5]^n and
    b = A x0 for an interior x0, so a feasible (hence optimal) point exists."""
    rng = np.random.default_rng(seed)
    A = rng.integers(-3, 4, (m, n)).astype(float)
    x0 = rng.uniform(0.5, 4.5, n)
    b = A @ x0
    c = rng.integers(-5, 6, n).astype(float)
    lb = np.zeros(n)
    ub = np.full(n, 5.0)
    return c, A, b, lb, ub


class TestSimplexVsHighs:
    @pytest.mark.parametrize("seed", list(range(40)))
    def test_matches_highs_on_random_lps(self, seed):
        m = 2 + (seed % 4)
        n = m + 2 + (seed % 3)
        c, A, b, lb, ub = _rand_standard_form(seed, m, n)

        status, x, obj, _iters = rust.solve_lp_py(
            np.ascontiguousarray(c),
            np.ascontiguousarray(A),
            np.ascontiguousarray(b),
            np.ascontiguousarray(lb),
            np.ascontiguousarray(ub),
        )
        hi = highs_lp(c=c, A_eq=A, b_eq=b, bounds=list(zip(lb.tolist(), ub.tolist())))
        if hi.status != SolveStatus.OPTIMAL:
            return  # only compare where HiGHS certified an optimum
        assert status == "optimal", f"seed={seed}: simplex status {status}"
        assert abs(obj - hi.objective) < 1e-6, f"seed={seed}: simplex {obj} vs HiGHS {hi.objective}"
        # returned point is feasible: A x = b, bounds respected.
        x = np.asarray(x)
        assert np.allclose(A @ x, b, atol=1e-6)
        assert np.all(x >= lb - 1e-7) and np.all(x <= ub + 1e-7)

    def test_knapsack_relaxation(self):
        a = np.array([[5.0, 5.0, 5.0, 5.0, 1.0]])
        b = np.array([9.0])
        c = np.array([-16.0, -16.0, -16.0, -16.0, 0.0])
        lb = np.zeros(5)
        ub = np.array([1.0, 1.0, 1.0, 1.0, 1e20])
        status, _x, obj, _ = rust.solve_lp_py(c, a, b, lb, ub)
        assert status == "optimal"
        assert abs(obj - (-28.8)) < 1e-6

    def test_infeasible(self):
        # x0 + s = 1, s>=0, x0 in [2,inf): x0>=2 but x0<=1 → infeasible.
        a = np.array([[1.0, 1.0]])
        b = np.array([1.0])
        c = np.array([1.0, 0.0])
        lb = np.array([2.0, 0.0])
        ub = np.array([1e20, 1e20])
        status, _x, _obj, _ = rust.solve_lp_py(c, a, b, lb, ub)
        assert status == "infeasible"

    def test_unbounded(self):
        a = np.array([[1.0, -1.0]])
        b = np.array([0.0])
        c = np.array([-1.0, 0.0])
        lb = np.array([0.0, 0.0])
        ub = np.array([1e20, 1e20])
        status, _x, _obj, _ = rust.solve_lp_py(c, a, b, lb, ub)
        assert status == "unbounded"


@pytest.mark.skipif(
    not hasattr(rust, "solve_lp_warm_py"),
    reason="warm-start LP binding not built",
)
class TestWarmStartLp:
    """``solve_lp_warm_py`` is the cutting-plane re-solve engine: it must (a) match
    the cold solver when given no basis, (b) return the SAME optimum when warm-
    started from a previous basis after rows (cuts) are appended, and (c) ignore a
    garbage basis (cold fallback) rather than return a wrong value. Warm-start is a
    speed optimization only — the optimum, hence any relaxation bound, is invariant.
    """

    def _std(self, A_ub, b_ub, c, lb, ub):
        """[A_ub | I] z = b_ub standard form with one slack per row."""
        A_ub = np.asarray(A_ub, float)
        m, n = A_ub.shape
        a_std = np.zeros((m, n + m))
        a_std[:, :n] = A_ub
        a_std[:, n:] = np.eye(m)
        c_std = np.concatenate([np.asarray(c, float), np.zeros(m)])
        lb_std = np.concatenate([np.asarray(lb, float), np.zeros(m)])
        ub_std = np.concatenate([np.asarray(ub, float), np.full(m, 1e20)])
        return (
            np.ascontiguousarray(c_std),
            np.ascontiguousarray(a_std),
            np.ascontiguousarray(np.asarray(b_ub, float)),
            np.ascontiguousarray(lb_std),
            np.ascontiguousarray(ub_std),
        )

    @pytest.mark.parametrize("seed", list(range(20)))
    def test_cold_matches_solve_lp_py(self, seed):
        # n > m (as the cutting-plane form ``[A_ub | I]`` always is), so a full
        # basis of m columns exists.
        m = 2 + seed % 4
        n = m + 2 + seed % 3
        c, A, b, lb, ub = _rand_standard_form(seed, m, n)
        s0, x0, o0, _ = rust.solve_lp_py(
            np.ascontiguousarray(c),
            np.ascontiguousarray(A),
            np.ascontiguousarray(b),
            np.ascontiguousarray(lb),
            np.ascontiguousarray(ub),
        )
        s1, x1, o1, _i, cs, bv, _d, _r = rust.solve_lp_warm_py(
            np.ascontiguousarray(c),
            np.ascontiguousarray(A),
            np.ascontiguousarray(b),
            np.ascontiguousarray(lb),
            np.ascontiguousarray(ub),
        )
        assert s1 == s0
        if s0 == "optimal":
            assert abs(o1 - o0) < 1e-9
            assert len(cs) == A.shape[1] and len(bv) == A.shape[0]

    def test_rowappend_warmstart_matches_cold(self):
        # min -x0 - x1 s.t. x0 + x1 <= 1, x in [0,1]; then append cut x0 <= 0.5.
        c, a, b, lb, ub = self._std([[1.0, 1.0]], [1.0], [-1.0, -1.0], [0, 0], [1, 1])
        st, _x, obj, _i, cs, bv, _d, _r = rust.solve_lp_warm_py(c, a, b, lb, ub)
        assert st == "optimal" and abs(obj - (-1.0)) < 1e-9

        c2, a2, b2, lb2, ub2 = self._std(
            [[1.0, 1.0], [1.0, 0.0]], [1.0, 0.5], [-1.0, -1.0], [0, 0], [1, 1]
        )
        # warm-start from the 1-row basis (Rust extends it with the new slack basic)
        sw, xw, ow, _iw, _csw, _bvw, _dw, _rw = rust.solve_lp_warm_py(
            c2, a2, b2, lb2, ub2, cs.astype(np.int8), bv.astype(np.int64)
        )
        sc, _xc, oc, _ic, _csc, _bvc, _dc, _rc = rust.solve_lp_warm_py(c2, a2, b2, lb2, ub2)
        assert sw == "optimal" == sc
        assert abs(ow - oc) < 1e-9
        assert abs(ow - (-1.0)) < 1e-9  # x0 = x1 = 0.5
        assert abs(np.asarray(xw)[0] - 0.5) < 1e-9

    def test_garbage_basis_is_ignored(self):
        # A dimensionally-inconsistent basis must be ignored (cold fallback), not
        # trusted into a wrong optimum.
        c, a, b, lb, ub = self._std([[1.0, 1.0]], [1.0], [-1.0, -1.0], [0, 0], [1, 1])
        bad_cs = np.array([9, 9, 9, 9, 9], dtype=np.int8)  # wrong length & values
        bad_bv = np.array([7, 7, 7], dtype=np.int64)  # out-of-range indices
        st, _x, obj, _i, _cs, _bv, _d, _r = rust.solve_lp_warm_py(c, a, b, lb, ub, bad_cs, bad_bv)
        assert st == "optimal"
        assert abs(obj - (-1.0)) < 1e-9


class TestSimplexLpDuals:
    """``lp_simplex.solve_lp`` exposes vertex duals/reduced costs (issue #356), in
    HiGHS's convention, so the dual-consuming seams (Benders, DBBT) run on it."""

    @staticmethod
    def _lp(seed):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(2, 6))
        mub = int(rng.integers(1, 4))
        meq = int(rng.integers(0, 2))
        A_ub = rng.integers(-3, 4, size=(mub, n)).astype(float)
        b_ub = rng.integers(1, 9, size=mub).astype(float)
        A_eq = rng.integers(-2, 3, size=(meq, n)).astype(float) if meq else None
        b_eq = (A_eq @ rng.uniform(0, 3, size=n)) if meq else None
        c = rng.integers(-3, 4, size=n).astype(float)
        bounds = [(0.0, 5.0)] * n
        return c, A_ub, b_ub, A_eq, b_eq, bounds

    def test_objective_matches_highs(self):
        from discopt.solvers.lp_simplex import solve_lp as rust_lp

        for seed in range(40):
            c, A_ub, b_ub, A_eq, b_eq, bounds = self._lp(seed)
            h = highs_lp(c, A_ub, b_ub, A_eq, b_eq, bounds)
            r = rust_lp(c, A_ub, b_ub, A_eq, b_eq, bounds)
            assert r.status == h.status
            if r.status == SolveStatus.OPTIMAL:
                assert abs(r.objective - h.objective) < 1e-6

    def test_duals_are_populated_and_satisfy_strong_duality(self):
        # The duals need not equal HiGHS's on a degenerate LP, but they must be a
        # *valid* optimal dual: strong duality g(y) == obj. (Skip LPs whose dual
        # points along a free direction, where the box term is unbounded.)
        from discopt.solvers.lp_simplex import solve_lp as rust_lp

        checked = 0
        for seed in range(60):
            c, A_ub, b_ub, A_eq, b_eq, bounds = self._lp(seed)
            r = rust_lp(c, A_ub, b_ub, A_eq, b_eq, bounds)
            if r.status != SolveStatus.OPTIMAL:
                continue
            assert r.dual_values is not None and r.reduced_costs is not None
            c_arr = np.asarray(c, float)
            n = len(c_arr)
            y = np.asarray(r.dual_values, float)
            rc = np.asarray(r.reduced_costs, float)
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            mub = A_ub.shape[0]
            rhs = np.concatenate(
                [np.asarray(b_ub, float), np.asarray(b_eq, float) if b_eq is not None else []]
            )
            # g(y) = b·y + Σ_j min_{x∈[lo,hi]} rc_j x_j  == obj at a valid optimum.
            contrib = np.where(rc > 0, rc * lo, np.where(rc < 0, rc * hi, 0.0))
            g = float(rhs @ y) + float(contrib.sum())
            assert abs(g - r.objective) <= 1e-6 * (1.0 + abs(r.objective))
            # reduced costs are exactly c − Aᵀy.
            recomputed = c_arr - A_ub.T @ y[:mub]
            if A_eq is not None:
                recomputed = recomputed - A_eq.T @ y[mub:]
            assert np.allclose(rc, recomputed, atol=1e-9)
            assert len(rc) == n
            checked += 1
        assert checked >= 20
