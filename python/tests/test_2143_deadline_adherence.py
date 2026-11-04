"""#2143: a root primal heuristic's sub-solve stops at the solve's deadline.

Measured on GDPlib ``syngas`` under ``gdp_method="hull"`` at 120 s, with the stage-2
flags ON: the solve ended at 123.6-160.1 s. Two sub-solves outlived the deadline:

* a POUNCE NLP that ran past its own ``max_wall_time``. One pump-seed start was
  launched at 86 s with a 30 s cap and returned at 159.8 s;
* one QP of the pump's feasibility repair, from 105.95 s to 123.48 s.
  ``pounce.project_to_feasible`` takes no time limit.

The NLP is now stopped from POUNCE's per-iteration ``intermediate`` hook once the
live solve's deadline passes, and the repair hands each QP the time left. Both
stop only work past the deadline: the helpers are compared bit for bit against
the legacy path while the deadline does not bind.
"""

from __future__ import annotations

import time
import types

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import primal_heuristics as PH
from discopt._tape_nlp_evaluator import make_evaluator
from discopt.solver_tuning import ROLE1_DEADLINE
from discopt.solvers import NLPResult, SolveStatus
from discopt.solvers._nlp_hard_stop import (
    HardStopCallbacks,
    armed_deadline,
    stop_at_solve_deadline,
)

pounce = pytest.importorskip("pounce")


def _model():
    m = dm.Model("repair")
    x = m.continuous("x", lb=-5.0, ub=5.0)
    y = m.continuous("y", lb=-5.0, ub=5.0)
    m.minimize(x + y)
    m.subject_to(x * x + y * y <= 1.0)
    m.subject_to(x + 2.0 * y >= 0.5)
    m.subject_to(x * y >= -0.3)
    return m


class _SlowHessian:
    """An evaluator whose every Hessian evaluation costs ``delay`` seconds."""

    def __init__(self, base, delay: float) -> None:
        self._base = base
        self._delay = delay
        self.hessian_calls = 0

    def __getattr__(self, name):
        return getattr(self._base, name)

    def evaluate_hessian_values(self, *a, **k):
        self.hessian_calls += 1
        time.sleep(self._delay)
        return self._base.evaluate_hessian_values(*a, **k)

    def evaluate_lagrangian_hessian(self, *a, **k):
        self.hessian_calls += 1
        time.sleep(self._delay)
        return self._base.evaluate_lagrangian_hessian(*a, **k)


@pytest.fixture
def role1_deadline():
    """Publish a role-1 deadline the way ``solve_model`` does, and clear it after."""
    tokens = []

    def _set(deadline):
        tokens.append(ROLE1_DEADLINE.set(deadline))

    yield _set
    for t in reversed(tokens):
        ROLE1_DEADLINE.reset(t)


# ---------------------------------------------------------------- the NLP stop


def test_arming_is_a_noop_outside_a_solve():
    assert ROLE1_DEADLINE.get() is None
    with stop_at_solve_deadline():
        assert armed_deadline() is None


def test_arming_takes_the_solve_deadline_and_keeps_the_earlier(role1_deadline):
    role1_deadline(100.0)
    with stop_at_solve_deadline():
        assert armed_deadline() == 100.0
        role1_deadline(50.0)
        with stop_at_solve_deadline():
            assert armed_deadline() == 50.0
    assert armed_deadline() is None


def test_heuristics_arm_the_stop(role1_deadline):
    seen = []

    def backend(ev, x0, options=None, **kw):
        seen.append(armed_deadline())
        return NLPResult(status=SolveStatus.ERROR, x=np.asarray(x0, dtype=float))

    role1_deadline(time.perf_counter() + 1e6)
    m = _model()
    PH.subnlp(m, np.array([0.5, 0.5]), backend=backend, evaluator=make_evaluator(m))
    assert seen, "the backend was never called"
    assert seen[0] == ROLE1_DEADLINE.get()


def test_slow_nlp_stops_at_the_solve_deadline(role1_deadline):
    """Fails before #2143: the solve runs every iteration, ~0.15 s each."""
    from discopt.solvers.nlp_pounce import solve_nlp

    ev = _SlowHessian(make_evaluator(_model()), delay=0.15)
    t0 = time.perf_counter()
    role1_deadline(t0 + 0.4)
    with stop_at_solve_deadline():
        r = solve_nlp(ev, np.array([3.0, -3.0]), options={"max_wall_time": 1e4})
    wall = time.perf_counter() - t0
    assert r.status == SolveStatus.TIME_LIMIT
    assert wall < 1.0, wall
    assert ev.hessian_calls >= 1


def test_nlp_past_the_deadline_is_not_started(role1_deadline):
    from discopt.solvers.nlp_pounce import solve_nlp

    ev = _SlowHessian(make_evaluator(_model()), delay=10.0)
    role1_deadline(time.perf_counter() - 1.0)
    x0 = np.array([3.0, -3.0])
    with stop_at_solve_deadline():
        r = solve_nlp(ev, x0)
    assert r.status == SolveStatus.TIME_LIMIT
    assert ev.hessian_calls == 0
    np.testing.assert_array_equal(r.x, x0)


def test_unbinding_stop_leaves_the_solve_bit_identical(role1_deadline):
    from discopt.solvers.nlp_pounce import solve_nlp

    ev = make_evaluator(_model())
    x0 = np.array([3.0, -3.0])
    plain = solve_nlp(ev, x0)
    role1_deadline(time.perf_counter() + 1e6)
    with stop_at_solve_deadline():
        armed = solve_nlp(ev, x0)
    assert plain.status == armed.status == SolveStatus.OPTIMAL
    np.testing.assert_array_equal(plain.x, armed.x)
    assert plain.iterations == armed.iterations


def test_hard_stop_respects_an_inner_intermediate():
    class Inner:
        def intermediate(self, **kw):
            return False

    cb = HardStopCallbacks(Inner(), deadline=time.perf_counter() + 1e6)
    assert cb.intermediate(iter_count=0) is False
    assert cb.fired is False


# ------------------------------------------------------------- the repair QPs


def _bounds(ev):
    from discopt.solvers.nlp_pounce import _infer_constraint_bounds

    lb, ub = ev.variable_bounds
    cl, cu = _infer_constraint_bounds(ev)
    return (np.asarray(v, dtype=float) for v in (lb, ub, cl, cu))


@pytest.mark.parametrize("x0", [[3.0, -3.0], [-4.0, 4.5], [0.0, 0.0], [5.0, 5.0]])
def test_time_limited_projection_matches_pounce(x0):
    """While the deadline does not bind, the repair is pounce's, bit for bit."""
    from discopt.solvers.nlp_pounce import _IpoptCallbacks

    ev = make_evaluator(_model())
    lb, ub, cl, cu = _bounds(ev)
    x0 = np.asarray(x0, dtype=float)
    ref = pounce.project_to_feasible(
        _IpoptCallbacks(ev),
        x0,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
        tol=PH._PUMP_REPAIR_TOL,
        max_iter=PH._PUMP_REPAIR_OUTER_ITERS,
    )
    ours = PH._project_to_feasible_until(
        _IpoptCallbacks(ev),
        x0,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
        tol=PH._PUMP_REPAIR_TOL,
        max_iter=PH._PUMP_REPAIR_OUTER_ITERS,
        deadline=time.perf_counter() + 1e6,
    )
    np.testing.assert_array_equal(ref, ours)


def test_slow_repair_qp_gets_the_time_left(monkeypatch):
    """Fails before #2143: pounce's projection passes no time limit, so the QP runs 5 s."""
    import pounce.qp as pqp

    real = pqp.solve_qp
    limits = []

    def slow_qp(*a, time_limit=None, **k):
        limits.append(time_limit)
        budget = 5.0 if time_limit is None else min(5.0, time_limit)
        time.sleep(budget)
        if time_limit is not None and time_limit <= 5.0:
            return types.SimpleNamespace(status="time_limit", success=False, x=None)
        return real(*a, **k)

    monkeypatch.setattr(pqp, "solve_qp", slow_qp)
    ev = make_evaluator(_model())
    t0 = time.perf_counter()
    out = PH._repair_to_feasible(ev, np.array([3.0, -3.0]), deadline=t0 + 0.3)
    wall = time.perf_counter() - t0
    assert limits and limits[0] is not None and limits[0] <= 0.3
    assert out is None
    assert wall < 1.5, wall


def test_repair_without_a_deadline_still_uses_pounce(monkeypatch):
    called = []
    real = pounce.project_to_feasible

    def spy(*a, **k):
        called.append(1)
        return real(*a, **k)

    monkeypatch.setattr(pounce, "project_to_feasible", spy)
    ev = make_evaluator(_model())
    assert PH._repair_to_feasible(ev, np.array([3.0, -3.0]), deadline=None) is not None
    assert called == [1]
