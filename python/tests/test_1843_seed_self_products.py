"""#1843: ``_seed_self_products`` matched distinct Constant arrays by ``str()``.

``Constant.__str__`` prints only the shape, so ``(a @ v) * (b @ v)`` with different
vectors ``a`` and ``b`` compared equal to a square and was enclosed in
``[0, max^2]``. ``_domain_gap`` then accepted ``sqrt((a@v)*(b@v) + 1)`` on a box
where its argument goes negative, the lowered body imposed its domain on the other
disjunct, and big-M certified 2.5 where the optimum is 4 (at ``v = (-2, 2)``).
Operands are now compared structurally, constants by value.
"""

import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.gdp_reformulate import _BoundMemo, _seed_self_products
from discopt.modeling.core import Constant

pytestmark = pytest.mark.smoke


def _model(b):
    m = dm.Model()
    v = m.continuous("v", shape=(2,), lb=-2, ub=2)
    a = np.array([1.0, 0.0])
    m.either_or([[dm.sqrt((a @ v) * (b @ v) + 1) >= 1.5], [v[0] <= -1.5]], name="d")
    m.maximize(v[1] - v[0])
    return m


@pytest.mark.parametrize("method", ["big-m", "mbigm", "binary_multiplication"])
def test_distinct_constant_vectors_are_not_a_square(method):
    # Before the fix: big-M "optimal" 2.5 at v=(-2, 0.5). The truth is 4; the row is
    # undefined on part of the box, so the lowerings that keep the body must refuse.
    with pytest.raises(ValueError, match="undefined on part of the variable box"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _model(np.array([0.0, 1.0])).solve(gdp_method=method, time_limit=60)


def test_equal_valued_distinct_constants_still_a_square():
    # (a @ v) * (a' @ v) with a' a *different object* equal to a is a genuine square:
    # the #1787 behaviour (accept, solve to the true optimum 4) is kept.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _model(np.array([1.0, 0.0])).solve(gdp_method="big-m", time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(4.0, abs=1e-5)


def test_seed_memo_unit():
    m = dm.Model()
    v = m.continuous("v", shape=(2,), lb=-2, ub=2)
    a, b = Constant(np.array([1.0, 0.0])), Constant(np.array([0.0, 1.0]))
    assert str(a @ v) == str(b @ v)  # the collision the old check relied on
    distinct = (a @ v) * (b @ v)
    square = (a @ v) * (Constant(np.array([1.0, 0.0])) @ v)
    memo = _BoundMemo(array_reductions=True)
    _seed_self_products(distinct + square, m, memo)
    assert memo.bounds[id(square)][1] == (0.0, 4.0)
    assert id(distinct) not in memo.bounds
