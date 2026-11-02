"""#1917: a heuristic sub-NLP can be solved on its FBBT-implied box.

Under ``gdp_method="hull"``, a heuristic that fixes a disjunct's indicator to 0 left
the disjunct's disaggregated copy ``v`` on its whole box. POUNCE's bound push then
started ``v`` at 0.01, and the hull row ``yhat * exp(a v / yhat)`` overflowed at
``yhat = eps``: ``Invalid_Number_Detected`` before the first iteration. The model is
the Trespalacios ex. 6.3.3 GDP (``pyomo.ex_633_trespalacios`` in the parity panel).
``DISCOPT_SUBNLP_IMPLIED_BOX=1`` solves the sub-NLP on the box the node FBBT
implies, which pins ``v`` to 0. It is OFF by default, and OFF must call the backend
exactly as before.
"""

from __future__ import annotations

import collections

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import primal_heuristics as PH

ORACLE = 4.46037  # baron+scip (gdplib_runner.panel_oracle)


def _ex633():
    """The parity-panel model, converted from Pyomo.

    Not rebuilt natively: the converted model orders the indicators first, and a
    native build of the same GDP (indicators after ``x1, x2``) sends the heuristics
    down other dives and hit no ``Invalid_Number_Detected`` at all (measured: 0 of
    184, 235 nodes), so it would not exercise the class.
    """
    gr = pytest.importorskip("benchmarks.gdplib_runner")
    pytest.importorskip("pyomo")
    from discopt.pyomo.gdp import convert_pyomo_gdp

    spec = gr.discover_models(include=["pyomo.ex_633_trespalacios"])
    if not spec:
        pytest.skip("pyomo.ex_633_trespalacios not available")
    return convert_pyomo_gdp(spec[0].builder()).model


def _count_statuses(monkeypatch):
    from discopt.solvers import nlp_pounce

    counts: collections.Counter = collections.Counter()
    real = nlp_pounce._run_solve

    def counting(problem, x0, warm_start, solve_report=False):
        x, info, report = real(problem, x0, warm_start, solve_report)
        counts[int(info.get("status", -100))] += 1
        return x, info, report

    monkeypatch.setattr(nlp_pounce, "_run_solve", counting)
    return counts


@pytest.mark.slow
@pytest.mark.parametrize("flag", ["0", "1"])
def test_hull_solve_is_correct_either_way(monkeypatch, flag):
    monkeypatch.setenv("DISCOPT_SUBNLP_IMPLIED_BOX", flag)
    counts = _count_statuses(monkeypatch)
    r = _ex633().solve(gdp_method="hull", time_limit=120)
    assert str(r.status).endswith("optimal")
    assert r.objective == pytest.approx(ORACLE, abs=1e-4)
    assert r.bound <= r.objective + 1e-6
    assert sum(counts.values()) > 0


@pytest.mark.slow
def test_flag_on_removes_the_invalid_number_starts(monkeypatch):
    """OFF: most heuristic sub-NLPs die at the first iterate. ON: few do."""
    counts = {}
    for flag in ("0", "1"):
        with monkeypatch.context() as mp:
            mp.setenv("DISCOPT_SUBNLP_IMPLIED_BOX", flag)
            c = _count_statuses(mp)
            _ex633().solve(gdp_method="hull", time_limit=120)
            counts[flag] = dict(c)
    off, on = counts["0"], counts["1"]
    assert sum(off.values()) > 0 and sum(on.values()) > 0
    invalid_off, invalid_on = off.get(-13, 0), on.get(-13, 0)
    # Measured 164 vs 6 on the GDPlib instance at 6d3e2f79; leave room for drift.
    assert invalid_off >= 50
    assert invalid_on <= invalid_off // 5


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, evaluator, x0, options=None):
        self.calls.append((evaluator, np.array(x0, dtype=np.float64)))
        return "result"


def _fixed_off_disjunct_model():
    """Hull of ``d`` with the first indicator fixed to 0 in the model's bounds."""
    from discopt._relax.gdp_reformulate import reformulate_gdp
    from discopt._tape_nlp_evaluator import make_evaluator

    m = dm.Model("fixed")
    x = m.continuous("x", lb=0.0, ub=5.0)
    z = m.continuous("z", lb=0.0, ub=10.0)
    m.minimize(z - x)
    m.either_or([[z - dm.exp(x) >= 0.0, x >= 1.0], [x <= 0.5, z == 0.0]], name="d")
    h = reformulate_gdp(m, method="hull")
    names = [v.name for v in h._variables]
    ind = [v for v in h._variables if v.var_type == dm.VarType.BINARY][0]
    ind.lb = np.zeros_like(np.asarray(ind.lb, dtype=np.float64))
    ind.ub = np.zeros_like(np.asarray(ind.ub, dtype=np.float64))
    return h, make_evaluator(h), names


def test_off_calls_the_backend_with_the_same_arguments(monkeypatch):
    monkeypatch.delenv("DISCOPT_SUBNLP_IMPLIED_BOX", raising=False)
    h, ev, _ = _fixed_off_disjunct_model()
    rec = _Recorder()
    x0 = np.full(ev.n_variables, 0.3)
    assert PH._solve_sub_nlp(rec, ev, x0, {}) == "result"
    ((got_ev, got_x0),) = rec.calls
    assert got_ev is ev
    np.testing.assert_array_equal(got_x0, x0)


def test_on_pins_the_off_disjunct_copies_and_keeps_the_rest(monkeypatch):
    monkeypatch.setenv("DISCOPT_SUBNLP_IMPLIED_BOX", "1")
    h, ev, names = _fixed_off_disjunct_model()
    lb0, ub0 = (np.asarray(a, dtype=np.float64) for a in ev.variable_bounds)
    rec = _Recorder()
    x0 = np.full(ev.n_variables, 0.3)
    PH._solve_sub_nlp(rec, ev, x0, {})
    ((got_ev, got_x0),) = rec.calls
    assert got_ev is not ev
    lb, ub = got_ev.variable_bounds
    pinned = {names[i] for i in range(len(names)) if ub[i] < ub0[i] and ub[i] <= 1e-9}
    # The disjunct whose indicator is 0 has disaggregated copies of x and z; each
    # is forced to 0 by its ``v <= ub * y`` row.
    assert {"_hull_d_v_x_0", "_hull_d_v_z_0"} <= pinned, (names, lb, ub)
    # Never a bound outside the declared box, and the start is in the new box.
    assert np.all(lb >= lb0) and np.all(ub <= ub0)
    assert np.all(got_x0 >= lb) and np.all(got_x0 <= ub)
