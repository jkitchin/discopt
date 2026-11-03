"""#1917: ``subnlp`` does not solve a fixed-integer box that node FBBT proves empty.

Measured on GDPlib ``modprodnet`` under hull: the root integer local search's
objective descent issued 101 sub-NLPs, and every one had an integer assignment
refuted by a row that also holds continuous columns. Each one returned ERROR after
about 1 s of NLP time, and together they used the whole 120 s before B&B left the
root. The integer-only row test (#1775 G6.2) cannot see such a row. Node FBBT can,
and B&B already prunes a node on the same proof.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
from discopt._relax import primal_heuristics as PH
from discopt._tape_nlp_evaluator import make_evaluator


class _Recorder:
    def __init__(self):
        self.calls = 0

    def __call__(self, evaluator, x0, options=None):
        self.calls += 1
        raise RuntimeError("recorder: no NLP result")


def _mixed_row_model():
    m = dm.Model("mixed_row")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.binary("y")
    m.minimize(x + y)
    # Mixed row: with y = 1 it needs x >= 2 > ub(x). Not an integer-only row.
    m.subject_to(x >= 2.0 * y)
    return m


def test_an_fbbt_refuted_assignment_is_not_solved():
    m = _mixed_row_model()
    ev = make_evaluator(m)
    rec = _Recorder()
    assert PH.subnlp(m, np.array([0.5, 1.0]), backend=rec, evaluator=ev) is None
    assert rec.calls == 0


def test_a_feasible_assignment_is_still_solved():
    m = _mixed_row_model()
    ev = make_evaluator(m)
    rec = _Recorder()
    PH.subnlp(m, np.array([0.5, 0.0]), backend=rec, evaluator=ev)
    assert rec.calls == 1


def test_the_feasible_assignment_returns_its_point():
    m = _mixed_row_model()
    ev = make_evaluator(m)
    got = PH.subnlp(m, np.array([0.5, 0.0]), evaluator=ev)
    assert got is not None
    x, obj = got
    assert x[1] == 0.0
    assert obj == np.float64(x[0] + x[1])
    assert 0.0 <= x[0] <= 1.0


def test_the_model_bounds_are_restored_after_a_refusal():
    m = _mixed_row_model()
    ev = make_evaluator(m)
    before = [(np.array(v.lb).copy(), np.array(v.ub).copy()) for v in m._variables]
    PH.subnlp(m, np.array([0.5, 1.0]), backend=_Recorder(), evaluator=ev)
    after = [(np.array(v.lb), np.array(v.ub)) for v in m._variables]
    checked = 0
    for (lb0, ub0), (lb1, ub1) in zip(before, after):
        np.testing.assert_array_equal(lb0, lb1)
        np.testing.assert_array_equal(ub0, ub1)
        checked += 1
    assert checked == 2
