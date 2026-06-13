"""Phase 1 soundness: self-hosted MILP/MIQP B&B must not certify "optimal"
off a non-KKT (max-iter) relaxation bound.

The node LP/QP relaxation IPM reports ``converged==1`` (optimal/KKT) or
``converged==3`` (max-iter). A code-3 objective f(x~) >= f* is not a valid
lower bound, so accepting it and reporting a certified "optimal" can prune the
true integer optimum. These tests pin that ``_solve_milp_bb`` /
``_solve_miqp_bb`` decertify the gap (status "feasible",
``gap_certified=False``, no rigorous bound) when any node bound came from a
stalled relaxation, while normal (converged) solves still certify.
"""

from __future__ import annotations

import discopt.modeling as dm
import discopt.solver as S
import numpy as np


# ---------------------------------------------------------------------------
# Controls: normal solves certify
# ---------------------------------------------------------------------------
def _knapsack_milp() -> dm.Model:
    m = dm.Model("milp")
    x1, x2, x3 = m.binary("x1"), m.binary("x2"), m.binary("x3")
    m.minimize(-3 * x1 - 4 * x2 - 5 * x3)
    m.subject_to(2 * x1 + 3 * x2 + 4 * x3 <= 6)
    return m  # optimum x1=x2=1,x3=0 -> -7? no: weights 2+4=6, obj -3-5=-8 (x1,x3)


def _miqp() -> dm.Model:
    m = dm.Model("miqp")
    y1 = m.integer("y1", lb=0, ub=5)
    y2 = m.integer("y2", lb=0, ub=5)
    m.minimize((y1 - 2.3) ** 2 + (y2 - 1.7) ** 2)
    return m  # optimum y1=2, y2=2 -> 0.09 + 0.09 = 0.18


class TestControlsCertify:
    def test_milp_bb_certifies(self):
        r = _knapsack_milp().solve(use_highs_milp=False, time_limit=60)
        assert r.status == "optimal"
        assert r.gap_certified is True
        assert abs(r.objective - (-8.0)) < 1e-4

    def test_miqp_bb_certifies(self, monkeypatch):
        monkeypatch.setattr(S, "_solve_qp_highs", lambda *a, **k: None)
        r = _miqp().solve(time_limit=60)
        assert r.status == "optimal"
        assert r.gap_certified is True
        assert abs(r.objective - 0.18) < 1e-4


# ---------------------------------------------------------------------------
# Force a non-KKT (code-3) relaxation: gap must not be certified
# ---------------------------------------------------------------------------
class _Code3State:
    """Stand-in IPM state: real (feasible) obj/x, but flagged max-iter."""

    def __init__(self, state):
        self.converged = np.full_like(np.asarray(state.converged), 3)
        self.obj = np.asarray(state.obj)
        self.x = np.asarray(state.x)


def _force_code3(monkeypatch, module, name):
    """Wrap one IPM entry point to relabel every result max-iter (non-KKT).

    Note: only one path (serial or batch) is patched per test, to a config
    that uses it, because mixing a relabeled root with the other path is a
    mock artifact (the stand-in state perturbs the mixed serial/batch search,
    not the decertification under test).
    """
    orig = getattr(module, name)

    def wrapper(*a, **k):
        return _Code3State(orig(*a, **k))

    monkeypatch.setattr(module, name, wrapper)


class TestNonKKTRecoveredByPounce:
    """Increment 2: a stalled (code-3) node is re-solved with POUNCE, whose
    KKT-valid optimum restores the bound — the solve certifies normally."""

    def test_milp_batch_non_kkt_recovered(self, monkeypatch):
        import pytest as _pytest

        _pytest.importorskip("pounce")
        import discopt._jax.lp_ipm as lp_ipm

        # batch_size=8 lets the tree export >1 node -> batch LP path.
        _force_code3(monkeypatch, lp_ipm, "lp_ipm_solve_batch")
        r = _knapsack_milp().solve(use_highs_milp=False, time_limit=60, batch_size=8)
        assert r.status == "optimal"
        assert r.gap_certified is True
        assert abs(r.objective - (-8.0)) < 1e-4

    def test_milp_serial_non_kkt_recovered(self, monkeypatch):
        import pytest as _pytest

        _pytest.importorskip("pounce")
        import discopt._jax.lp_ipm as lp_ipm

        # batch_size=1 forces the serial per-node LP path.
        _force_code3(monkeypatch, lp_ipm, "lp_ipm_solve")
        r = _knapsack_milp().solve(use_highs_milp=False, time_limit=60, batch_size=1)
        assert r.status == "optimal"
        assert r.gap_certified is True
        assert abs(r.objective - (-8.0)) < 1e-4

    def test_miqp_batch_non_kkt_recovered(self, monkeypatch):
        import pytest as _pytest

        _pytest.importorskip("pounce")
        import discopt._jax.qp_ipm as qp_ipm

        monkeypatch.setattr(S, "_solve_qp_highs", lambda *a, **k: None)
        _force_code3(monkeypatch, qp_ipm, "qp_ipm_solve_batch")
        r = _miqp().solve(time_limit=60, batch_size=8)
        assert r.status == "optimal"
        assert r.gap_certified is True
        assert abs(r.objective - 0.18) < 1e-3


class TestNonKKTDecertifiesWhenUnrecoverable:
    """When POUNCE recovery also fails, the gap must not be certified."""

    def test_milp_batch_decertifies(self, monkeypatch):
        import discopt._jax.lp_ipm as lp_ipm

        _force_code3(monkeypatch, lp_ipm, "lp_ipm_solve_batch")
        monkeypatch.setattr(S, "_pounce_recover_node_bound", lambda *a, **k: None)
        r = _knapsack_milp().solve(use_highs_milp=False, time_limit=60, batch_size=8)
        # Bounds are the real LP optima (just relabeled non-KKT), so the answer
        # is still found, but optimality must not be certified.
        assert r.gap_certified is False
        assert r.status == "feasible"
        assert r.bound is None and r.gap is None
        assert abs(r.objective - (-8.0)) < 1e-4  # incumbent still correct

    def test_milp_serial_decertifies(self, monkeypatch):
        import discopt._jax.lp_ipm as lp_ipm

        _force_code3(monkeypatch, lp_ipm, "lp_ipm_solve")
        monkeypatch.setattr(S, "_pounce_recover_node_bound", lambda *a, **k: None)
        r = _knapsack_milp().solve(use_highs_milp=False, time_limit=60, batch_size=1)
        assert r.gap_certified is False
        assert r.status == "feasible"
        assert abs(r.objective - (-8.0)) < 1e-4

    def test_miqp_batch_decertifies(self, monkeypatch):
        import discopt._jax.qp_ipm as qp_ipm

        monkeypatch.setattr(S, "_solve_qp_highs", lambda *a, **k: None)
        _force_code3(monkeypatch, qp_ipm, "qp_ipm_solve_batch")
        monkeypatch.setattr(S, "_pounce_recover_node_bound", lambda *a, **k: None)
        r = _miqp().solve(time_limit=60, batch_size=8)
        assert r.gap_certified is False
        assert r.status == "feasible"
        assert r.bound is None and r.gap is None
        assert abs(r.objective - 0.18) < 1e-3  # incumbent still correct


# ---------------------------------------------------------------------------
# Increment 3: snap-fix-resolve purification of interior relaxation points
# ---------------------------------------------------------------------------
class TestSnapFixResolvePurification:
    def _lp_parts(self):
        # min -x1 - 2*x2 s.t. x1 + x2 <= 10, both integer in [0, 10].
        c = np.array([-1.0, -2.0])
        A_ub = np.array([[1.0, 1.0]])
        b_ub = np.array([10.0])
        return c, A_ub, b_ub

    def test_near_integral_point_purified(self):
        import pytest as _pytest

        _pytest.importorskip("pounce")
        c, A_ub, b_ub = self._lp_parts()
        # Smeared interior point: ints at 0.0003 / 9.99996 (beyond 1e-5 tol).
        inc = S._pounce_snap_incumbent(
            np.array([0.0003 - 0.0003, 9.99996]),  # x1 ~ 0, x2 ~ 10
            [0, 1],
            [1, 1],
            np.array([0.0, 0.0]),
            np.array([10.0, 10.0]),
            c,
            0.0,
            A_ub,
            b_ub,
            None,
            None,
            0.0,
            30.0,
        )
        assert inc is not None
        obj, x = inc
        np.testing.assert_allclose(x[:2], [0.0, 10.0], atol=1e-6)
        assert abs(obj - (-20.0)) < 1e-6  # exact objective at the integer point

    def test_far_from_integral_returns_none(self):
        c, A_ub, b_ub = self._lp_parts()
        inc = S._pounce_snap_incumbent(
            np.array([0.4, 9.5]),  # genuinely fractional
            [0, 1],
            [1, 1],
            np.array([0.0, 0.0]),
            np.array([10.0, 10.0]),
            c,
            0.0,
            A_ub,
            b_ub,
            None,
            None,
            0.0,
            30.0,
        )
        assert inc is None

    def test_snap_outside_node_box_returns_none(self):
        c, A_ub, b_ub = self._lp_parts()
        # x2 ~ 5.99997 snaps to 6, but the node box caps x2 at 5.99998.
        inc = S._pounce_snap_incumbent(
            np.array([0.0, 5.99997]),
            [0, 1],
            [1, 1],
            np.array([0.0, 0.0]),
            np.array([10.0, 5.99998]),
            c,
            0.0,
            A_ub,
            b_ub,
            None,
            None,
            0.0,
            30.0,
        )
        assert inc is None

    def test_no_integer_vars_returns_none(self):
        c, A_ub, b_ub = self._lp_parts()
        inc = S._pounce_snap_incumbent(
            np.array([0.5, 0.5]),
            [],
            [],
            np.zeros(2),
            np.full(2, 10.0),
            c,
            0.0,
            A_ub,
            b_ub,
            None,
            None,
            0.0,
            30.0,
        )
        assert inc is None

    def test_purification_yields_exact_incumbent_end_to_end(self):
        import pytest as _pytest

        _pytest.importorskip("pounce")
        r = _knapsack_milp().solve(use_highs_milp=False, time_limit=60)
        assert r.status == "optimal"
        # The snapped incumbent is exact, not the smeared IPM objective.
        assert r.objective == -8.0


# ---------------------------------------------------------------------------
# Increment 4: reduced-cost fixing via relaxation duals
# ---------------------------------------------------------------------------
class TestReducedCostFixing:
    def test_tightens_and_fixes(self):
        # z_lp=10, z_inc=12, gap=2 (+ tiny margin).
        #  d0=5  (lb): x0 <= 0 + floor(2/5)=0   -> fixed to 0
        #  d1=0.5(lb): x1 <= 0 + floor(2/0.5)=4 -> ub 10->4
        #  d2=-3 (ub): x2 >= 8 - floor(2/3)=8   -> fixed to 8
        lb = np.array([0.0, 0.0, 0.0])
        ub = np.array([5.0, 10.0, 8.0])
        rc = np.array([5.0, 0.5, -3.0])
        nlb, nub, nch = S._reduced_cost_fixing(lb, ub, [0, 1, 2], rc, z_lp=10.0, z_inc=12.0)
        assert nch == 3
        assert nlb[0] == 0.0 and nub[0] == 0.0
        assert nub[1] == 4.0
        assert nlb[2] == 8.0 and nub[2] == 8.0

    def test_negative_gap_is_noop(self):
        lb = np.array([0.0])
        ub = np.array([5.0])
        nlb, nub, nch = S._reduced_cost_fixing(lb, ub, [0], np.array([5.0]), z_lp=12.0, z_inc=10.0)
        assert nch == 0 and nub[0] == 5.0

    def test_near_zero_reduced_cost_skipped(self):
        # A basic / degenerate variable (|d| below tol) is never fixed.
        lb = np.array([0.0])
        ub = np.array([5.0])
        nlb, nub, nch = S._reduced_cost_fixing(lb, ub, [0], np.array([1e-9]), z_lp=10.0, z_inc=12.0)
        assert nch == 0 and nub[0] == 5.0

    def test_never_cuts_the_optimum(self):
        # The true optimum x* with objective <= z_inc must survive RCF: any
        # integer x* satisfies d_j*(x*_j - bound_j) <= gap term-by-term.
        rng = np.random.default_rng(0)
        for _ in range(200):
            n = 4
            lb = np.zeros(n)
            ub = rng.integers(1, 6, n).astype(float)
            rc = rng.uniform(-4, 4, n)
            z_lp = rng.uniform(-5, 5)
            gap = rng.uniform(0, 6)
            z_inc = z_lp + gap
            nlb, nub, _ = S._reduced_cost_fixing(lb, ub, list(range(n)), rc, z_lp, z_inc)
            # Any integer point whose reduced-cost objective estimate is within
            # the gap must lie inside the tightened box.
            for x in rng.integers(0, 6, (50, n)).astype(float):
                if np.any(x < lb) or np.any(x > ub):
                    continue
                est = z_lp + sum(rc[j] * (x[j] - (lb[j] if rc[j] > 0 else ub[j])) for j in range(n))
                if est <= z_inc + 1e-12:  # an "improving" point per the LP bound
                    assert np.all(x >= nlb - 1e-9) and np.all(x <= nub + 1e-9)

    def test_end_to_end_answer_unchanged_with_and_without_rcf(self, monkeypatch):
        import pytest as _pytest

        _pytest.importorskip("pounce")

        def _knap():
            m = dm.Model("k")
            xs = [m.binary(f"x{i}") for i in range(5)]
            vals, wts = [8, 5, 3, 2, 1], [5, 4, 3, 2, 1]
            m.minimize(-sum(v * x for v, x in zip(vals, xs)))
            m.subject_to(sum(w * x for w, x in zip(wts, xs)) <= 7)
            return m

        with_rcf = _knap().solve(use_highs_milp=False, time_limit=60)
        monkeypatch.setattr(
            S, "_root_reduced_cost_fixing", lambda lp, n, lb, ub, *a: (lb, ub, None)
        )
        without_rcf = _knap().solve(use_highs_milp=False, time_limit=60)
        assert with_rcf.status == without_rcf.status == "optimal"
        assert abs(with_rcf.objective - without_rcf.objective) < 1e-6
        assert with_rcf.objective == -10.0
