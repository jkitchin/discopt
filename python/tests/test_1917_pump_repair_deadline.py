"""#1917: the feasibility pump's repair stops at the caller's deadline.

``pounce.project_to_feasible`` takes no time limit, and each of its outer steps is
a sparse QP. Measured on GDPlib ``gdp_col`` under hull: each QP took ~27 s and the
repair ran 10 of them, so a 120 s solve returned after 359 s. The repair now
checks the deadline each time pounce calls back for ``constraints``/``jacobian``,
which happens between QPs.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import primal_heuristics as PH
from discopt._tape_nlp_evaluator import make_evaluator

pytest.importorskip("pounce")


def _model():
    m = dm.Model("repair")
    x = m.continuous("x", lb=-5.0, ub=5.0)
    y = m.continuous("y", lb=-5.0, ub=5.0)
    m.minimize(x + y)
    m.subject_to(x * x + y * y <= 1.0)
    m.subject_to(x + 2.0 * y >= 0.5)
    return m


class _Inner:
    def __init__(self):
        self.calls = []

    def constraints(self, x):
        self.calls.append("constraints")
        return np.zeros(1)

    def jacobian(self, x):
        self.calls.append("jacobian")
        return np.zeros(1)

    def jacobianstructure(self):
        return (np.array([0]), np.array([0]))


def test_callbacks_pass_through_before_the_deadline(monkeypatch):
    monkeypatch.setattr(PH, "_now", lambda: 0.0)
    inner = _Inner()
    cb = PH._DeadlineCallbacks(inner, deadline=1.0)
    cb.constraints(np.zeros(1))
    cb.jacobian(np.zeros(1))
    rows, cols = cb.jacobianstructure()
    assert inner.calls == ["constraints", "jacobian"]
    assert rows.tolist() == [0] and cols.tolist() == [0]


def test_callbacks_raise_once_the_deadline_passes(monkeypatch):
    monkeypatch.setattr(PH, "_now", lambda: 2.0)
    inner = _Inner()
    cb = PH._DeadlineCallbacks(inner, deadline=1.0)
    for f in (cb.constraints, cb.jacobian):
        with pytest.raises(PH._RepairDeadline):
            f(np.zeros(1))
    assert inner.calls == []


def test_repair_without_a_deadline_returns_a_point():
    ev = make_evaluator(_model())
    x = PH._repair_to_feasible(ev, np.array([3.0, -3.0]), deadline=None)
    assert x is not None and x.shape == (2,)


def test_repair_stops_when_the_deadline_passes_during_it(monkeypatch):
    """The entry check passes; the clock then runs out before pounce's first callback."""
    clock = iter([0.0] + [10.0] * 1000)
    monkeypatch.setattr(PH, "_now", lambda: next(clock))
    ev = make_evaluator(_model())
    assert PH._repair_to_feasible(ev, np.array([3.0, -3.0]), deadline=1.0) is None
