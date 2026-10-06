"""#1668: a feasible ``initial_solution`` must survive the solve-time lift.

The factorable lift appends aux columns ``w`` with rows ``w - f(x) == 0``. They
were padded with box midpoints, so the extended point violated those rows and a
start feasible on the user's model was rejected as an incumbent.
"""

import io
import logging

import discopt.modeling as dm
import numpy as np
import pytest


def _model():
    m = dm.Model("ws")
    y = m.binary("y", shape=(2,))
    q = m.continuous("q", lb=-2, ub=2)
    h = m.continuous("h", lb=-10, ub=10)
    m.subject_to(y[0] + y[1] == 1)
    m.subject_to(h == (1.0 * y[0] + 3.0 * y[1]) * (q * abs(q) ** 0.852))
    m.subject_to(h >= 1.5)
    m.minimize(1.0 * y[0] + 2.0 * y[1] + 0.1 * q)
    return m, y, q, h


def test_feasible_start_accepted_after_lift():
    m, y, q, h = _model()
    qq = 0.8
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    lg = logging.getLogger("discopt")
    old = lg.level
    lg.addHandler(handler)
    lg.setLevel(logging.INFO)
    try:
        r = m.solve(
            initial_solution={y: np.array([0.0, 1.0]), q: qq, h: 3 * qq**1.852}, time_limit=20
        )
    finally:
        lg.removeHandler(handler)
        lg.setLevel(old)
    log = buf.getvalue()
    assert "extended from" in log  # the probe fired: columns really were added
    assert "Warm-start incumbent injected: obj=2.08" in log
    assert "violates constraints" not in log
    assert r.status == "optimal"
    assert r.objective == pytest.approx(1.1244748858573679, abs=1e-5)
