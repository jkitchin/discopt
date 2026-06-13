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
