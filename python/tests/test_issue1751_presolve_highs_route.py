"""#1751: ``solve(presolve=False)`` switches HiGHS's presolve off on the HiGHS route.

On that route HiGHS's presolve is the one that runs, so the argument was a silent
no-op: the HiGHS log was identical for ``presolve=False`` and ``True``.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.solvers import lp_milp_highs

pytestmark = pytest.mark.smoke


def _knapsack():
    rng = np.random.default_rng(3)
    n = 30
    w = rng.integers(5, 40, (5, n))
    v = rng.integers(10, 50, n)
    cap = (w.sum(1) * 0.3).astype(int)
    m = dm.Model("mk")
    x = m.binary("x", shape=(n,))
    m.subject_to(w @ x <= cap)
    m.maximize(v @ x)
    return m


def _lp():
    m = dm.Model("lp")
    x = m.continuous("x", shape=(3,), lb=0, ub=10)
    m.subject_to(x[0] + 2 * x[1] + x[2] >= 4)
    m.minimize(dm.sum(x))
    return m


def _presolve_settings(monkeypatch, model, **kw):
    seen = []
    real = lp_milp_highs._new_highs

    def spy(highspy, opts):
        h = real(highspy, opts)
        seen.append(h.getOptionValue("presolve")[1])
        return h

    monkeypatch.setattr(lp_milp_highs, "_new_highs", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = model.solve(**kw)
    assert seen, "no HiGHS instance was built"
    return r, seen


@pytest.mark.parametrize("build", [_knapsack, _lp])
def test_presolve_false_reaches_highs(monkeypatch, build):
    r, seen = _presolve_settings(monkeypatch, build(), presolve=False)
    assert r.status == "optimal"
    assert set(seen) == {"off"}


def test_presolve_default_leaves_highs_presolve_on(monkeypatch):
    r, seen = _presolve_settings(monkeypatch, _knapsack())
    assert r.status == "optimal"
    assert "choose" in seen


def test_same_answer_either_way():
    assert _knapsack().solve(presolve=False).objective == _knapsack().solve().objective


def test_contradicting_highs_option_is_refused():
    with pytest.raises(ValueError, match="contradict"):
        _knapsack().solve(presolve=False, highs_options={"presolve": "on"})


def test_agreeing_highs_option_is_accepted(monkeypatch):
    r, seen = _presolve_settings(
        monkeypatch, _knapsack(), presolve=False, highs_options={"presolve": "off"}
    )
    assert set(seen) == {"off"}
