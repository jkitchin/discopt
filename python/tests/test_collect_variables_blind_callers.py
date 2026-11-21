"""Callers of ``gdp_reformulate._collect_variables`` that acted on its blind spot.

That walker has no arm for ``SumExpression``/``MatMulExpression``/``CustomCall``
(it is left so on purpose: the GDP hull coverage guard cross-checks against it).
Three callers made decisions from what it could not see:

* the GDP advisor read ``(A @ X)[0] + y`` as a small linear body and chose hull,
  whose own coverage guard then refused it, so ``solve(gdp_method="auto")`` raised
  on a model big-M and mbigm solve (the default ``gdp_method`` is ``"big-m"``);
* the disjunct-local-variable guard missed a local variable used through
  ``dm.sum`` outside its disjunct;
* ``detect_separability`` split ``dm.sum(x) + y <= 1`` into separate blocks.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.presolve.separability import detect_separability

A = np.array([[1.0, 1.0, 1.0]])


def _gdp(body):
    m = dm.Model("gdp_blind")
    x = m.continuous("X", shape=(3,), lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.either_or([[body(x, y) <= 3.0], [x[0] >= 8.0]])
    m.minimize(-(dm.sum(x) + y))
    return m


@pytest.mark.parametrize(
    "body", [lambda x, y: (A @ x)[0] + y, lambda x, y: dm.sum(x) + y], ids=["matmul", "sum"]
)
def test_gdp_auto_routes_around_hull_it_cannot_take(body):
    res = _gdp(body).solve(gdp_method="auto")  # the default is "big-m"
    assert res.status == "optimal"
    # The second disjunct (X[0] >= 8) with everything else at its upper bound.
    assert res.objective == pytest.approx(-40.0, abs=1e-6)


def test_advisor_never_recommends_hull_that_hull_refuses():
    from discopt._relax.gdp_advisor import recommend_method

    m = _gdp(lambda x, y: (A @ x)[0] + y)
    dc = next(c for c in m._constraints if type(c).__name__ == "_DisjunctiveConstraint")
    advice = recommend_method(dc, m)
    assert advice.recommendation in ("big-m", "mbigm")
    assert "hull cannot see every variable" in advice.rationale


def test_advisor_unchanged_where_hull_sees_everything():
    from discopt._relax.gdp_advisor import recommend_method

    m = dm.Model("gdp_plain")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.either_or([[x + y <= 3.0], [x >= 8.0]])
    m.minimize(-(x + y))
    dc = next(c for c in m._constraints if type(c).__name__ == "_DisjunctiveConstraint")
    assert recommend_method(dc, m).recommendation == "hull"


def test_local_variable_escaping_through_dm_sum_is_refused():
    m = dm.Model("escape_sum")
    x = m.continuous("x", lb=0, ub=10)
    a = m.make_disjunct("A")
    z = a.variable("z", shape=(2,), lb=0, ub=10)
    a.subject_to(x >= z[0])
    b = m.make_disjunct("B")
    b.subject_to(x >= 6)
    m.add_disjunction([a, b])
    m.subject_to(dm.sum(z) <= 4)  # outside A, through a reduction
    m.minimize(x)
    with pytest.raises(ValueError, match="declared local to disjunct 'A'"):
        m.solve()


def test_separability_sees_reductions():
    m = dm.Model("sep")
    x = m.continuous("x", shape=(3,), lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    m.subject_to(dm.sum(x) + y <= 1)
    m.minimize(-dm.sum(x) - y)
    report = detect_separability(m)
    assert len(report.blocks) == 1
    assert not report.separable
