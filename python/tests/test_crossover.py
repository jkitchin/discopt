"""Interior-point -> vertex crossover (roadmap Phase 2 keystone).

The crossover pushes an interior LP optimum (analytic center of the optimal
face) to a vertex, preserving objective and feasibility, so cover/clique cuts
and branching see the sharp fractional structure a vertex exposes. Tests:
the push preserves objective/feasibility and lands on a vertex; it converts
the symmetric knapsack's analytic center into a cover-separable vertex; the
size guard; and an end-to-end node reduction that was impossible from the
interior point.
"""

from __future__ import annotations

import discopt.modeling as dm
import discopt.solver as S
import jax.numpy as jnp
import numpy as np
import pytest
from discopt._jax.cover_cuts import separate_cover_cuts
from discopt._jax.crossover import _MAX_CROSSOVER_VARS, _null_direction, crossover_to_vertex
from discopt._jax.lp_ipm import lp_ipm_solve
from discopt._jax.problem_classifier import extract_lp_data
from discopt.solver import _decompose_eq_slack_form


def _lp_optimum(model):
    ld = extract_lp_data(model)
    st = lp_ipm_solve(
        jnp.asarray(ld.c),
        jnp.asarray(ld.A_eq),
        jnp.asarray(ld.b_eq),
        jnp.asarray(ld.x_l),
        jnp.asarray(ld.x_u),
    )
    return ld, np.asarray(st.x)


def _is_vertex(x, A, c, xl, xu, tol=1e-6):
    free = np.where((x > xl + tol) & (x < xu - tol))[0]
    if free.size == 0:
        return True
    M = np.vstack([A[:, free], c[free][None, :]]) if A.shape[0] else c[free][None, :]
    return _null_direction(M, 1e-7) is None


class TestCrossoverProperties:
    def _sym_knapsack(self):
        m = dm.Model("sym")
        xs = [m.binary(f"x{i}") for i in range(4)]
        m.minimize(-sum(16 * x for x in xs))
        m.subject_to(sum(5 * x for x in xs) <= 9)
        return m

    def test_preserves_objective_and_feasibility_and_is_vertex(self):
        ld, x_int = _lp_optimum(self._sym_knapsack())
        A, b, c = np.asarray(ld.A_eq), np.asarray(ld.b_eq), np.asarray(ld.c)
        xl, xu = np.asarray(ld.x_l), np.asarray(ld.x_u)
        xv = crossover_to_vertex(x_int, A, b, c, xl, xu)
        assert abs(c @ xv - c @ x_int) < 1e-5  # objective preserved
        assert np.allclose(A @ xv, b, atol=1e-5)  # feasibility preserved
        assert np.all(xv >= xl - 1e-6) and np.all(xv <= xu + 1e-6)
        assert _is_vertex(xv, A, c, xl, xu)  # lands on a vertex of the optimal face

    def test_interior_center_becomes_cover_separable(self):
        ld, x_int = _lp_optimum(self._sym_knapsack())
        A, b, c = np.asarray(ld.A_eq), np.asarray(ld.b_eq), np.asarray(ld.c)
        xl, xu = np.asarray(ld.x_l), np.asarray(ld.x_u)
        n = 4
        A_ub, b_ub, _, _ = _decompose_eq_slack_form(A, b, n, A.shape[1] - n)
        # Interior center violates no cover; the crossed-over vertex does.
        assert separate_cover_cuts(A_ub, b_ub, x_int[:n], np.ones(n, bool)) == []
        xv = crossover_to_vertex(x_int, A, b, c, xl, xu)
        assert separate_cover_cuts(A_ub, b_ub, xv[:n], np.ones(n, bool))

    def test_already_vertex_is_stable(self):
        # A point already at a vertex (all but one var at bounds) is unchanged.
        A = np.array([[1.0, 1.0, 1.0]])
        b, c = np.array([2.0]), np.array([-1.0, -1.0, -1.0])
        xl, xu = np.zeros(3), np.ones(3)
        x = np.array([1.0, 1.0, 0.0])  # vertex of {sum=2, [0,1]^3}
        xv = crossover_to_vertex(x, A, b, c, xl, xu)
        np.testing.assert_allclose(xv, x, atol=1e-6)

    def test_size_guard_returns_unchanged(self):
        n = _MAX_CROSSOVER_VARS + 5
        x = np.full(n, 0.5)
        xv = crossover_to_vertex(
            x, np.zeros((0, n)), np.zeros(0), np.ones(n), np.zeros(n), np.ones(n)
        )
        np.testing.assert_array_equal(xv, x)

    def test_random_lps_preserve_objective_and_reach_vertex(self):
        rng = np.random.default_rng(3)
        for _ in range(40):
            n, mrows = 5, 2
            A = rng.standard_normal((mrows, n))
            x_feas = rng.uniform(0.2, 0.8, n)
            b = A @ x_feas
            c = rng.standard_normal(n)
            xl, xu = np.zeros(n), np.ones(n)
            # Start from an interior feasible point.
            xv = crossover_to_vertex(x_feas, A, b, c, xl, xu)
            assert abs(c @ xv - c @ x_feas) < 1e-5
            assert np.allclose(A @ xv, b, atol=1e-5)
            assert np.all(xv >= -1e-6) and np.all(xv <= 1 + 1e-6)
            assert _is_vertex(xv, A, c, xl, xu)


class TestEndToEnd:
    def _sym_knapsack(self):
        m = dm.Model("sym")
        xs = [m.binary(f"x{i}") for i in range(4)]
        m.minimize(-sum(16 * x for x in xs))
        m.subject_to(sum(5 * x for x in xs) <= 9)
        return m

    def test_crossover_makes_cuts_bite_on_symmetric_problem(self, monkeypatch):
        pytest.importorskip("pounce")
        import time as _time

        # This validates the Python self-hosted B&B crossover + cover-cut
        # machinery (_solve_milp_bb with IPM nodes), where the interior relaxation
        # point needs crossover for cover cuts to bite. That path is now only a
        # *fallback* for the default MILP route (the Rust whole-search is
        # primary), so call it directly to keep the _root_cover_cut_loop
        # monkeypatch meaningful. With crossover + cuts the symmetric knapsack
        # solves at/near the root; without cuts it needs more nodes.
        def _run():
            return S._solve_milp_bb(
                self._sym_knapsack(),
                60.0,
                1e-4,
                16,
                "best_first",
                100_000,
                _time.perf_counter(),
                prefer_pounce=True,
                node_engine="pounce",
            )

        r_cut = _run()
        monkeypatch.setattr(S, "_root_cover_cut_loop", lambda ld, *a, **k: (ld, 0))
        r_nocut = _run()

        assert r_cut.status == "optimal" and r_nocut.status == "optimal"
        assert abs(r_cut.objective - r_nocut.objective) < 1e-4
        assert abs(r_cut.objective - (-16.0)) < 1e-3
        assert r_cut.node_count < r_nocut.node_count  # strictly fewer: cuts now bite
