"""Reported bound duals are judged against the box the USER declared (#1037 follow-up).

The solve tightens variable bounds in place and restores them on exit, so the
continuous route's "declared" box was already the working box: on
``nlp_cvx_204_010`` it read ``x in [-1, 1]`` where the user declared the default
+/-9.999e19 box, a 1.8e-9 barrier residue on the default side passed the #1037
check, and the examiner found ``lambda * |x - bound|`` = 1.75e12.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import discopt.solver as solver_mod
import numpy as np
from discopt.modeling.core import solve_entry_box


def _rsoc() -> dm.Model:
    m = dm.Model("nlp_cvx_204_010")
    x = m.continuous("x")
    y = m.continuous("y")
    z = m.continuous("z", lb=1e-8)
    m.minimize(-y - x)
    m.subject_to(x**2 / z <= y)
    m.subject_to(x**2 + y**2 <= -z + 1)
    return m


def test_entry_box_is_the_declared_box_during_the_solve(monkeypatch):
    m = _rsoc()
    seen = []
    real = solver_mod._solve_continuous

    def spy(model, *a, **k):
        seen.append((solve_entry_box(model), [float(v.ub) for v in model._variables]))
        return real(model, *a, **k)

    monkeypatch.setattr(solver_mod, "_solve_continuous", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.solve()
    assert seen, "the continuous route never ran"
    box, working_ub = seen[0]
    assert box is not None
    np.testing.assert_array_equal(box[1], [9.999e19, 9.999e19, 9.999e19])
    assert working_ub[0] < 1e3  # the working box really was tightened in place
    assert solve_entry_box(m) is None  # nothing leaks past the solve


def test_bound_duals_satisfy_cs_against_the_declared_box():
    m = _rsoc()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve()
    assert r.status == "optimal"
    checks = 0
    for v in m._variables:
        xv = float(np.asarray(r.x[v.name]))
        for duals, bound in ((r.bound_duals_lower, v.lb), (r.bound_duals_upper, v.ub)):
            if duals is None:
                continue
            lam = float(np.asarray(duals[v.name]))
            assert abs(lam) * abs(xv - float(bound)) <= 1e-7, (v.name, lam, bound)
            checks += 1
    assert checks > 0
