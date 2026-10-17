"""#1714: a column an ``==`` row pins on its own starts at its implied value.

``_solve_continuous`` started the NLP at the midpoint of the tightened box, which
ignores the equalities. On the #1711 hanging chain the four ``ends`` columns
(``x[0] == 0``, ``y[0] == 0``, ``x[n] == 1``, ``y[n] == 0``) started at -5, the
first iterate had ``inf_pr = 50``, and POUNCE needed 101 iterations at
N = 10,000 instead of 47.

The fix moves only the START. The box handed to the NLP is unchanged, so no
bound or certificate depends on it. (The first attempt, ``DISCOPT_EQ_MIRROR_BOUNDS``,
tightened the box instead; it was retired after two panels showed it did not help --
see ``docs/dev/flag-retirement-audit.md``.)
"""

from __future__ import annotations

import discopt.modeling as dm
import discopt.solvers.nlp_pounce as nlp_pounce
import numpy as np
import pytest
from discopt._relax.nonlinear_bound_tightening import affine_implied_values


def _chain(n: int) -> dm.Model:
    m = dm.Model("chain")
    x = m.continuous("x", shape=(n + 1,), lb=-10, ub=10)
    y = m.continuous("y", shape=(n + 1,), lb=-10, ub=10)
    m.subject_to([x[0] == 0, y[0] == 0, x[n] == 1, y[n] == 0], name="ends")
    m.subject_to((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2 <= (1.5 / n) ** 2, name="link")
    m.minimize(0.5 * dm.sum(y[1:] + y[:-1]))
    return m


def test_implied_values_on_the_chain_ends():
    n = 20
    vals = affine_implied_values(_chain(n))
    # flat layout: x[0..n], y[0..n]
    assert vals == {0: 0.0, n: 1.0, n + 1: 0.0, 2 * n + 1: 0.0}


def test_only_single_variable_affine_rows_qualify():
    m = dm.Model("rows")
    v = m.continuous("v", lb=-10, ub=10)
    w = m.continuous("w", lb=-10, ub=10)
    u = m.continuous("u", lb=-10, ub=10)
    z = m.continuous("z", shape=(3,), lb=-10, ub=10)
    m.subject_to(2 * v + 1 == 3)  # v = 1
    m.subject_to(v + w == 1)  # two variables: skipped
    m.subject_to(w**2 == 4)  # nonlinear: skipped
    m.subject_to(u <= 2)  # not an equality: skipped
    m.subject_to(z[1:] == 0)  # a slice, not a scalar column: skipped
    m.subject_to(-3 * z[0] + 6 == 0)  # z[0] = 2
    m.minimize(v + w + u)
    vals = affine_implied_values(m)
    assert vals == pytest.approx({0: 1.0, 3: 2.0})


def test_the_continuous_solve_starts_the_end_columns_at_their_values(monkeypatch):
    n = 40
    starts = []
    orig = nlp_pounce.solve_nlp

    def spy(evaluator, x0, **kw):
        starts.append(np.array(x0, dtype=float))
        return orig(evaluator, x0, **kw)

    monkeypatch.setattr(nlp_pounce, "solve_nlp", spy)
    r = _chain(n).solve(time_limit=120)
    assert starts, "the continuous NLP was never called -- the probe did not fire"
    x0 = starts[0]
    for idx, val in {0: 0.0, n: 1.0, n + 1: 0.0, 2 * n + 1: 0.0}.items():
        assert x0[idx] == val, (idx, x0[idx])
    assert r.status == "optimal"
    assert r.bound is not None and r.bound <= r.objective + 1e-9
