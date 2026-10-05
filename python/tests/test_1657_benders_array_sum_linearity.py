"""#1657: ``decomposition="benders"`` must recognise a linear MILP however its sums
are spelled.

``dm.sum(x[i, :])`` (a full reduction over a slice), ``dm.sum(c * x)`` and
``c @ x`` are linear, and plain ``Model.solve()`` routes all of them to HiGHS. The
Benders dispatcher and the decomposition advisor used the GDP ``_is_linear``
predicate, which has no arm for those nodes, so the same linear MILP went to
Generalized Benders (the nonlinear-recourse algorithm) and the advisor called it
nonlinear. Both now take :func:`extract_linear` as the witness.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.decomposition._linear import extract_linear, first_nonlinear_reason

COST = np.arange(12.0).reshape(3, 4) % 5 + 1
FC = np.array([3.0, 4.0, 5.0])

SPELLINGS = [
    ("terms", "terms"),
    ("slice", "terms"),
    ("terms", "sum(c*x)"),
    ("terms", "c@x"),
    ("slice", "c@x"),
]


def _build(row: str, obj: str) -> dm.Model:
    m = dm.Model("t1657")
    y = m.binary("y", shape=(3,))
    x = m.continuous("x", shape=(3, 4), lb=0, ub=10)
    for c in range(4):
        m.subject_to(dm.sum([x[i, c] for i in range(3)]) == 2.0)
    for i in range(3):
        body = dm.sum([x[i, c] for c in range(4)]) if row == "terms" else dm.sum(x[i, :])
        m.subject_to(body <= 8.0 * y[i])
    if obj == "terms":
        o = dm.sum(
            [float(COST[i, c]) * x[i, c] for i in range(3) for c in range(4)]
            + [float(FC[i]) * y[i] for i in range(3)]
        )
    elif obj == "sum(c*x)":
        o = dm.sum(COST * x) + dm.sum(FC * y)
    else:
        o = dm.sum(lambda i: COST[i] @ x[i, :], over=range(3)) + FC @ y
    m.minimize(o)
    m.first_stage(y)
    return m


def test_every_spelling_extracts_the_same_rows():
    ref = extract_linear(_build("terms", "terms"))
    n = 0
    for row, obj in SPELLINGS:
        lin = extract_linear(_build(row, obj))
        np.testing.assert_allclose(lin.dense(), ref.dense(), rtol=0, atol=0)
        np.testing.assert_allclose(lin.b, ref.b, rtol=0, atol=0)
        np.testing.assert_allclose(lin.c, ref.c, rtol=0, atol=1e-15)
        assert lin.sense == ref.sense
        n += 1
    assert n == len(SPELLINGS)


@pytest.mark.parametrize("row,obj", SPELLINGS)
def test_benders_routes_linear_spellings_to_classical_benders(row, obj):
    m = _build(row, obj)
    assert first_nonlinear_reason(m) is None
    assert m.analyze_decomposition().structure().model_is_nonlinear is False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve(decomposition="benders", time_limit=30)
    route = r.algorithm_route.split(":")[0]
    assert route == "benders", r.algorithm_route
    assert r.status == "optimal"
    assert r.objective == pytest.approx(20.0, abs=1e-6)


def test_nonlinear_model_still_goes_to_gbd_and_says_why():
    m = dm.Model("nl1657")
    y = m.binary("y", shape=(2,))
    x = m.continuous("x", shape=(2,), lb=0, ub=4)
    m.subject_to(x[0] + x[1] >= 1.0)
    m.subject_to(x[0] <= 4 * y[0])
    m.subject_to(x[1] <= 4 * y[1])
    m.minimize(x[0] ** 2 + x[1] ** 2 + y[0] + y[1])
    m.first_stage(y)
    reason = first_nonlinear_reason(m)
    assert reason is not None and "objective" in reason
    assert m.analyze_decomposition().structure().model_is_nonlinear is True
    with pytest.warns(UserWarning, match="not recognised as linear"):
        r = m.solve(decomposition="benders", time_limit=30)
    assert r.algorithm_route.startswith("benders/gbd"), r.algorithm_route
