"""The structure-cut recognizer must keep the objective sense and its own verdicts.

``inject_cuts`` called ``model.minimize(...)`` unconditionally and injected cuts
whose derivation had reported ``h`` non-convex and the check unsound; it also
checked convexity on ``(0, sqrt(3.5*(2500 - 30**2)))`` -- one gas instance's pipe
bound, hardcoded. ``structure_cuts`` is on by default, so ``maximize(-cost)`` and a
negative power coefficient both returned ``infeasible`` with ``gap_certified=True``
on feasible models (found by the #1952-#1958 adjacent-bug sweep).
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest

pytest.importorskip("sympy")

from discopt._relax.symbolic import cut_recognizer as R  # noqa: E402


def _gas(kp=0.828, sense="min", wmax=100.0):
    m = dm.Model("gas")
    p = m.continuous("p_node", shape=(5,), lb=30.0, ub=70.0)
    p_d1 = m.continuous("p_d1", lb=45.0, ub=70.0)
    p_d2 = m.continuous("p_d2", lb=45.0, ub=70.0)
    f = m.continuous("f_pipe", shape=(6,), lb=0.0, ub=150.0)
    w = m.continuous("w_cs", shape=(2,), lb=0.0, ub=wmax)
    beta = m.continuous("beta", shape=(2,), lb=1.0, ub=2.0)
    y = m.binary("y_cs", shape=(2,))
    cost = kp * w[0] * (beta[0] ** 0.2857 - 1.0) + kp * w[1] * (beta[1] ** 0.2857 - 1.0)
    cost = cost + 0.5 * (y[0] + y[1])
    m.minimize(cost) if sense == "min" else m.maximize(-cost)
    m.subject_to(f[0] ** 2 == 3.5 * (50.0**2 - p[0] ** 2))
    m.subject_to(f[1] ** 2 == 3.5 * (50.0**2 - p[2] ** 2))
    m.subject_to(f[2] ** 2 == 5.0 * (p[1] ** 2 - p[4] ** 2))
    m.subject_to(f[3] ** 2 == 5.0 * (p[3] ** 2 - p[4] ** 2))
    m.subject_to(f[4] ** 2 == 3.5 * (p[4] ** 2 - p_d1**2))
    m.subject_to(f[5] ** 2 == 3.5 * (p[4] ** 2 - p_d2**2))
    m.subject_to(p[1] == beta[0] * p[0])
    m.subject_to(p[3] == beta[1] * p[2])
    m.subject_to(w[0] <= wmax * y[0])
    m.subject_to(w[1] <= wmax * y[1])
    m.subject_to(beta[0] <= 1.0 + y[0])
    m.subject_to(beta[1] <= 1.0 + y[1])
    m.subject_to(f[0] == w[0])
    m.subject_to(w[0] == f[2])
    m.subject_to(f[1] == w[1])
    m.subject_to(w[1] == f[3])
    m.subject_to(f[2] + f[3] == f[4] + f[5])
    m.subject_to(f[4] == 40.0)
    m.subject_to(f[5] == 30.0)
    m.subject_to(y[0] + y[1] >= 1)
    return m


@pytest.mark.smoke
@pytest.mark.parametrize(
    "kw, opt",
    [({"sense": "max"}, -3.0025943), ({"kp": -0.828}, -11.6933318)],
    ids=["maximize", "negative_coefficient"],
)
def test_default_solve_certifies_true_optimum(kw, opt):
    r = _gas(**kw).solve(time_limit=60, threads=1)
    assert r.status == "optimal", (r.status, r.objective, r.bound)
    assert r.objective == pytest.approx(opt, abs=1e-5)


@pytest.mark.smoke
def test_unsound_or_nonconvex_cuts_are_not_injected():
    for kw in ({"sense": "max"}, {"kp": -0.828}, {}):  # {}: h undefined past w ~ 93.5
        m = _gas(**kw)
        n_cons = len(m._constraints)
        assert R.recognize_and_inject(m) == 0
        assert len(m._constraints) == n_cons


@pytest.mark.smoke
def test_injection_keeps_maximize_sense():
    # On a box where the cut is provable it fires, and the objective keeps its sense.
    m = _gas(wmax=74.0)
    cuts = R.recognize_and_derive_cuts(m)
    assert len(cuts) == 2 and all(c.domain == (0.0, 74.0) for c in cuts)
    sense_before = m._objective.sense
    assert R.inject_cuts(m, cuts) == 2
    assert m._objective.sense == sense_before
