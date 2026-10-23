"""#1775 G6.2: ``subnlp`` refuses an integer assignment its linear rows already refute.

``subnlp`` fixes every integer to its rounded value and solves the continuous NLP
that is left. When a linear row's support is entirely integer, that row is a
constant once the integers are fixed, and if the constant violates the row no
NLP can be feasible. Measured on the GDP parity panel (big-M ``eight_process``):
``integer_local_search`` issued 24 sub-NLPs in 4.9 s and every one fixed an
assignment breaking a ``sum_k y_k == 1`` disjunction row, so all 24 were decided
before the NLP ran. ``subnlp`` now checks those rows first and returns ``None``
without calling the backend.

Soundness: the refusal threshold is above what ``subnlp``'s own final
``_check_constraint_feasibility`` would accept on the same row at the same integer
values, so a refused assignment is one that check would have rejected anyway.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import primal_heuristics as PH

pytestmark = pytest.mark.smoke


class _Spy:
    """An NLP backend that records its calls and delegates to the real one."""

    def __init__(self):
        from discopt.solvers.nlp_backend import get_nlp_solver

        self.real = get_nlp_solver("auto")
        self.calls = 0

    def __call__(self, evaluator, x0, options=None):
        self.calls += 1
        return self.real(evaluator, x0, options=options)


def _model():
    """Three binaries in one ``== 1`` row, an integer row, and a nonlinear part."""
    m = dm.Model("rows")
    x = m.continuous("x", lb=0, ub=4)
    y = [m.binary(f"y{i}") for i in range(3)]
    n = m.integer("n", lb=0, ub=5)
    m.subject_to(y[0] + y[1] + y[2] == 1)
    m.subject_to(n - 2 * y[1] <= 2)  # integer-only inequality
    m.subject_to(x >= 1 + y[0] + 2 * y[1] + 3 * y[2])
    m.minimize((x - 2.5) ** 2 + 0.1 * n)
    return m


def _point(m, y, n, x=2.0):
    out = np.zeros(sum(v.size for v in m._variables))
    out[0], out[1:4], out[4] = x, y, n
    return out


@pytest.mark.parametrize(
    "y, n",
    [
        ((1, 1, 0), 0),  # two disjuncts selected: the == 1 row reads 1
        ((0, 0, 0), 0),  # none selected: the == 1 row reads -1
        ((0, 1, 0), 5),  # the <= row reads 5 - 2 - 2 = 1 > 0
    ],
)
def test_a_refuted_assignment_never_reaches_the_backend(y, n):
    m = _model()
    spy = _Spy()
    assert PH.subnlp(m, _point(m, y, n), backend=spy) is None
    assert spy.calls == 0


@pytest.mark.parametrize("y, n", [((1, 0, 0), 0), ((0, 1, 0), 4), ((0, 0, 1), 2)])
def test_a_consistent_assignment_is_still_solved(y, n):
    m = _model()
    spy = _Spy()
    r = PH.subnlp(m, _point(m, y, n), backend=spy)
    assert spy.calls == 1
    assert r is not None
    x, _ = r
    assert x[1:4].tolist() == list(y) and x[4] == n


def test_rounding_happens_before_the_check():
    """A fractional point rounds to a consistent assignment and is solved."""
    m = _model()
    spy = _Spy()
    r = PH.subnlp(m, _point(m, (0.9, 0.2, 0.1), 0.3), backend=spy)
    assert spy.calls == 1 and r is not None and r[0][1:4].tolist() == [1, 0, 0]


def test_rows_with_a_continuous_variable_are_not_prejudged():
    """``x >= 1 + ...`` mentions x, so it is left to the NLP even when x0 breaks it."""
    m = _model()
    spy = _Spy()
    r = PH.subnlp(m, _point(m, (0, 0, 1), 0, x=0.0), backend=spy)
    assert spy.calls == 1 and r is not None


def test_the_scan_is_cached_and_follows_new_constraints():
    m = _model()
    mask = PH._get_integer_mask(m)
    first = PH._integer_linear_rows(m, mask)
    assert PH._integer_linear_rows(m, mask) is first
    assert len(first) == 2
    y = [v for v in m._variables if v.name.startswith("y")]
    m.subject_to(y[0] <= 0)
    assert len(PH._integer_linear_rows(m, mask)) == 3
