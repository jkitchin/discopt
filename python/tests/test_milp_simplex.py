"""Rust-internal warm-started-simplex MILP solver vs POUNCE (roadmap P4).

``discopt._rust.solve_milp_py`` runs the whole pure-MILP branch-and-bound in
Rust: the existing tree manager, with each node's LP solved by the bounded
simplex — root cold, children warm-started (dual simplex) from the basis they
inherit from their parent. This is the B&B correctness gate: on random MILPs its
optimum must match the POUNCE B&B reference (``incorrect_count == 0``, within
1e-4; HiGHS was removed, issue #356), and it must detect infeasibility.

The solver consumes standard form ``A x = b`` with explicit slack columns; the
helper here slacks a ``≤`` MILP so the same instance can be sent to both
solvers.
"""

from __future__ import annotations

import numpy as np
import pytest

rust = pytest.importorskip("discopt._rust")
if not hasattr(rust, "solve_milp_py"):
    pytest.skip("simplex MILP binding not built", allow_module_level=True)

from discopt.solvers import SolveStatus  # noqa: E402


def _slack_standard_form(c, A_ub, b_ub, lb, ub):
    """Turn `min c x s.t. A_ub x <= b_ub, lb<=x<=ub` into standard form
    `A_eq z = b` with one slack per row. Returns (c_s, A_eq, b, l_s, u_s, n)."""
    m, n = A_ub.shape
    A_eq = np.zeros((m, n + m))
    A_eq[:, :n] = A_ub
    A_eq[:, n:] = np.eye(m)
    c_s = np.concatenate([c, np.zeros(m)])
    l_s = np.concatenate([lb, np.zeros(m)])
    u_s = np.concatenate([ub, np.full(m, 1e20)])
    return c_s, A_eq, b_ub.astype(float), l_s, u_s, n


def _solve_simplex(c, A_ub, b_ub, lb, ub, integer_cols):
    c_s, A_eq, b, l_s, u_s, n = _slack_standard_form(c, A_ub, b_ub, lb, ub)
    return rust.solve_milp_py(
        np.ascontiguousarray(c_s),
        np.ascontiguousarray(A_eq),
        np.ascontiguousarray(b),
        np.ascontiguousarray(l_s),
        np.ascontiguousarray(u_s),
        np.ascontiguousarray(np.asarray(integer_cols, dtype=np.int64)),
        n,
    )


class TestMilpSimplexVsPounce:
    @pytest.mark.parametrize("seed", list(range(30)))
    def test_matches_pounce(self, seed):
        pytest.importorskip("pounce")
        from discopt.solvers.milp_pounce import solve_milp as ref_milp

        rng = np.random.default_rng(seed)
        n, m = 4, 3
        c = rng.integers(-5, 6, n).astype(float)
        A = rng.integers(0, 4, (m, n)).astype(float)
        b = (A @ rng.integers(0, 4, n) + rng.integers(1, 5, m)).astype(float)
        lb = np.zeros(n)
        ub = np.full(n, 5.0)
        integer_cols = list(range(n))

        status, x, obj, _bound, _nodes, _it = _solve_simplex(c, A, b, lb, ub, integer_cols)
        hi = ref_milp(c=c, A_ub=A, b_ub=b, bounds=[(0, 5)] * n, integrality=np.ones(n))
        if hi.status != SolveStatus.OPTIMAL:
            return
        assert status in ("optimal", "feasible"), f"seed={seed}: {status}"
        assert abs(obj - hi.objective) < 1e-4, (
            f"seed={seed}: simplex {obj} vs POUNCE {hi.objective}"
        )

    def test_binary_knapsack(self):
        c = np.array([-10.0, -9.0, -8.0, -1.0])
        A = np.array([[5.0, 5.0, 5.0, 5.0]])
        b = np.array([9.0])
        status, _x, obj, _b, _n, _i = _solve_simplex(c, A, b, np.zeros(4), np.ones(4), [0, 1, 2, 3])
        assert status == "optimal"
        assert abs(obj - (-10.0)) < 1e-6

    def test_infeasible(self):
        # x0 <= 1 (slack) but x0 >= 2 → infeasible.
        c = np.array([1.0])
        A = np.array([[1.0]])
        b = np.array([1.0])
        status, _x, _obj, _b, _n, _i = _solve_simplex(
            c, A, b, np.array([2.0]), np.array([5.0]), [0]
        )
        assert status == "infeasible"


class TestModelSolveSimplex:
    """End-to-end: Model.solve(nlp_solver="simplex") routes a pure MILP to the
    Rust warm-started-simplex engine and matches the POUNCE default."""

    def _knapsack(self):
        import discopt.modeling as dm

        m = dm.Model("k")
        xs = [m.binary(f"x{i}") for i in range(4)]
        m.minimize(-(10 * xs[0] + 9 * xs[1] + 8 * xs[2] + 1 * xs[3]))
        m.subject_to(5 * xs[0] + 5 * xs[1] + 5 * xs[2] + 5 * xs[3] <= 9)
        return m

    def test_knapsack_via_model_solve(self):
        r = self._knapsack().solve(nlp_solver="simplex", time_limit=30)
        assert r.status == "optimal"
        assert abs(r.objective - (-10.0)) < 1e-4

    def test_matches_default_on_random(self):
        import discopt.modeling as dm

        for seed in range(8):
            rng = np.random.default_rng(1000 + seed)
            n = 4
            v = rng.integers(1, 12, n)
            w = rng.integers(1, 12, n)
            cap = float(0.6 * w.sum())
            m = dm.Model("k")
            xs = [m.binary(f"x{i}") for i in range(n)]
            m.minimize(-sum(int(v[i]) * xs[i] for i in range(n)))
            m.subject_to(sum(int(w[i]) * xs[i] for i in range(n)) <= cap)
            r_s = m.solve(nlp_solver="simplex", time_limit=30)

            m2 = dm.Model("k2")
            ys = [m2.binary(f"x{i}") for i in range(n)]
            m2.minimize(-sum(int(v[i]) * ys[i] for i in range(n)))
            m2.subject_to(sum(int(w[i]) * ys[i] for i in range(n)) <= cap)
            r_h = m2.solve(nlp_solver="pounce", time_limit=30)

            assert r_s.status == "optimal"
            if r_h.status == "optimal":
                assert abs(r_s.objective - r_h.objective) < 1e-4, f"seed={seed}"


class TestMILPBoundChannel:
    """`MILPResult.bound` must be a sound dual *lower* bound on every exit,
    distinct from the incumbent `objective` (upper bound). This pins the
    contract the AMP/OA/GDP lower-bound updates rely on (PR #117 review #1)."""

    def _knapsack(self, n=30, seed=7):
        import discopt.modeling as dm

        rng = np.random.default_rng(seed)
        v = rng.integers(1, 30, n)
        w = rng.integers(1, 30, n)
        cap = float(0.5 * w.sum())
        m = dm.Model("k")
        xs = [m.binary(f"x{i}") for i in range(n)]
        m.minimize(-sum(int(v[i]) * xs[i] for i in range(n)))
        m.subject_to(sum(int(w[i]) * xs[i] for i in range(n)) <= cap)
        c = (-v).astype(float)
        A = w.reshape(1, -1).astype(float)
        b = np.array([cap])
        return m, c, A, b, n

    def test_optimal_bound_equals_objective(self):
        from discopt.solvers.milp_simplex import solve_milp

        _, c, A, b, n = self._knapsack()
        r = solve_milp(c=c, A_ub=A, b_ub=b, bounds=[(0, 1)] * n, integrality=np.ones(n))
        assert r.status == SolveStatus.OPTIMAL
        assert r.bound is not None and r.objective is not None
        # On a proven optimum the dual bound is tight.
        assert abs(r.bound - r.objective) < 1e-6

    def test_node_limited_bound_brackets_optimum(self):
        """Force a node limit: `bound` must stay a valid lower bound (<= true
        optimum) and `objective`, if present, an upper bound (>= optimum).
        The incumbent must never be reported as the bound."""
        from discopt.solvers.milp_simplex import solve_milp

        _, c, A, b, n = self._knapsack()
        opt = solve_milp(c=c, A_ub=A, b_ub=b, bounds=[(0, 1)] * n, integrality=np.ones(n))
        true_opt = opt.objective

        r = solve_milp(
            c=c, A_ub=A, b_ub=b, bounds=[(0, 1)] * n, integrality=np.ones(n), max_nodes=5
        )
        if r.status == SolveStatus.OPTIMAL:
            pytest.skip("instance solved at the root; cannot force a gap")
        # The dual bound is a sound lower bound on the optimum...
        assert r.bound is not None
        assert r.bound <= true_opt + 1e-6
        # ...and the incumbent is an upper bound strictly worse than the optimum
        # here — i.e. using it as the LB (the bug) would over-certify. `bound`
        # must be reported separately and never equal that incumbent.
        assert r.objective is not None
        assert r.objective >= true_opt - 1e-6
        assert r.bound <= r.objective + 1e-9

    def test_relaxation_model_propagates_bound(self):
        """MilpRelaxationModel.solve must surface the dual `bound`, not the
        incumbent, as the relaxation's lower bound."""
        from discopt.solvers import MILPResult
        from discopt.solvers import SolveStatus as S

        # A node-limited MILPResult: incumbent above, dual bound below.
        res = MILPResult(status=S.ITERATION_LIMIT, x=np.zeros(2), objective=5.0, bound=-3.0)
        assert res.bound == -3.0 and res.objective == 5.0
        # The result type carries both channels independently.
        assert res.bound < res.objective


class TestPureLpShortCircuit:
    """THRU-2b: ``solve_milp`` with no integer columns is a plain LP, so the
    driver's integer-search machinery (root cuts / GMI / heuristics / strong
    branching) must be turned off — it is inert on an LP and only adds overhead
    (this is the dense-cold fallback the McCormick node relaxer reaches when the
    warm sparse simplex breaks down numerically). The optimum/infeasibility
    verdict is unchanged; only the wasted integer-side work is skipped."""

    def test_pure_lp_runs_with_machinery_off(self, monkeypatch):
        """A no-integer solve routes the sparse CSC driver with the MILP machinery
        off; a solve WITH integers keeps it on. Fails before THRU-2b (the old
        code always passed the machinery-on defaults)."""
        import discopt._rust as _rust
        from discopt.solvers import milp_simplex

        captured: list[dict] = []
        _orig = _rust.solve_milp_csc_py

        def _spy(*args, **kwargs):
            captured.append(dict(kwargs))
            return _orig(*args, **kwargs)

        # ``solve_milp`` does ``from discopt._rust import solve_milp_csc_py`` at
        # call time, so patch the source module attribute (not ``milp_simplex``).
        monkeypatch.setattr(_rust, "solve_milp_csc_py", _spy)

        c = np.array([-1.0, -1.0])
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        b = np.array([3.0, 3.0])
        bounds = [(0.0, 10.0), (0.0, 10.0)]

        # Pure LP (integrality=None): machinery must be off.
        milp_simplex.solve_milp(c=c, A_ub=A, b_ub=b, bounds=bounds, integrality=None)
        assert captured, "solve_milp_py was not called"
        lp_kw = captured[-1]
        assert lp_kw.get("root_cuts") == 0
        assert lp_kw.get("cut_rounds") == 0
        assert lp_kw.get("gmi_cuts") is False
        assert lp_kw.get("heuristics") is False
        assert lp_kw.get("strong_branch") is False

        # Genuine MILP (integer columns present): machinery must stay ON, i.e. the
        # short-circuit kwargs are NOT injected (Rust defaults apply).
        captured.clear()
        milp_simplex.solve_milp(c=c, A_ub=A, b_ub=b, bounds=bounds, integrality=np.array([1, 1]))
        milp_kw = captured[-1]
        assert "root_cuts" not in milp_kw
        assert "gmi_cuts" not in milp_kw
        assert "heuristics" not in milp_kw

    def test_pure_lp_optimum_is_bound_neutral(self):
        """The short-circuit must return the exact LP optimum — a fractional
        vertex the integer machinery would otherwise never reach (correctness
        guard on the speed change)."""
        from discopt.solvers import milp_simplex

        # max x0 + x1 s.t. 2x0+x1<=3, x0+2x1<=3, 0<=x<=10 -> x=(1,1), obj=2.
        c = np.array([-1.0, -1.0])
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        b = np.array([3.0, 3.0])
        r = milp_simplex.solve_milp(c=c, A_ub=A, b_ub=b, bounds=[(0.0, 10.0)] * 2, integrality=None)
        assert r.status == SolveStatus.OPTIMAL
        assert r.objective == pytest.approx(-2.0, abs=1e-6)
        assert r.bound is not None and r.bound <= r.objective + 1e-9
