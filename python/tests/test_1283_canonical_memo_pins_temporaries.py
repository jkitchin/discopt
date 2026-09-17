"""The canonical memo must not hand a stale CNode to a recycled ``id()`` (#1283).

``sum(0.5**x) + sum(10**Y)`` is canonicalized element by element: the sum branch
builds TEMPORARY scalar expressions with ``scalar_elements`` and canonicalizes
them. The memo was keyed by ``id(expr)`` without holding the expression, so once
the first sum's temporaries were freed, the second sum's temporaries could reuse
their addresses and receive the first sum's CNodes. ``Y`` then vanished from the
row, and the solve certified 3.87 or 5.77 (it varied by run) against the true
optimum ``3*0.5**1.5 + 4*0.1``.
"""

import gc

import discopt.modeling as dm
import pytest
from discopt._relax.canonical_expr import canonicalize, var_support

TRUE_OPTIMUM = 3 * 0.5**1.5 + 4 * 0.1


def _model():
    m = dm.Model("epi")
    x = m.continuous("x", lb=-1.5, ub=1.5, shape=(3,))
    Y = m.continuous("Y", lb=-1, ub=1, shape=(2, 2))
    t = m.continuous("t", lb=-100, ub=100)
    m.subject_to(dm.sum(0.5**x) + dm.sum(10.0**Y) <= t)
    m.minimize(t)
    return m


def test_row_support_covers_every_variable():
    m = _model()
    for _ in range(20):
        gc.collect()
        dag = canonicalize(m)
        assert len(dag.constraints) == 1
        assert var_support(dag.constraints[0]) == frozenset(range(8))


@pytest.mark.parametrize("run", range(3))
def test_solve_never_certifies_above_the_optimum(run):
    r = _model().solve(time_limit=60)
    assert r.status in ("optimal", "feasible", "time_limit")
    if r.bound is not None:
        assert r.bound <= TRUE_OPTIMUM + 1e-6
    if r.status == "optimal":
        assert r.objective == pytest.approx(TRUE_OPTIMUM, abs=1e-4)
