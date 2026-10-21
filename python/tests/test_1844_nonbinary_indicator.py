"""#1844: ``if_then`` accepted integer and continuous indicators.

``m.if_then(z, [x >= 6])`` with an integer ``z in [0, 3]`` solved to -4 under
big-M (the row over-enforced for ``z >= 2``) and to -24 under
``binary_multiplication`` (any ``z > 0`` active): the optimum depended on
``gdp_method``. A non-binary or non-scalar indicator is now refused with a
``TypeError`` at ``if_then``, and again by ``Model.validate`` for one whose type
changed after the call.
"""

import warnings

import discopt.modeling as dm
import pytest
from discopt.modeling.core import VarType

pytestmark = pytest.mark.smoke


def _base():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=10)
    return m, x


@pytest.mark.parametrize(
    "make",
    [
        lambda m: m.integer("z", lb=0, ub=3),
        lambda m: m.integer("z", lb=0, ub=1),
        lambda m: m.continuous("z", lb=0, ub=1),
        lambda m: m.binary("z", shape=(2,)),
        lambda m: m.integer("z", shape=(2,), lb=0, ub=1)[0],
    ],
    ids=["int03", "int01", "continuous", "binary_vector", "int_indexed"],
)
def test_non_binary_indicator_refused(make):
    m, x = _base()
    z = make(m)
    with pytest.raises(TypeError, match="must be a single binary variable"):
        m.if_then(z, [x >= 6])
    assert not m._constraints  # nothing recorded


def test_complement_of_integer_refused():
    m, x = _base()
    z = m.integer("z", lb=0, ub=3)
    with pytest.raises(TypeError):
        m.if_then(1 - z, [x >= 6])


@pytest.mark.parametrize("method", ["big-m", "binary_multiplication", "hull", "mbigm"])
def test_binary_indicator_agrees_across_methods(method):
    m, x = _base()
    z = m.binary("z")
    m.if_then(z, [x >= 6])
    m.minimize(x - 10 * z)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve(gdp_method=method, time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-4.0, abs=1e-6)


def test_indexed_binary_and_complement_accepted():
    m, x = _base()
    z = m.binary("z", shape=(2,))
    m.if_then(z[1], [x >= 6])
    m.if_then(1 - z[0], [x <= 8])
    assert len(m._constraints) == 2


def test_validate_rechecks_a_retyped_indicator():
    m, x = _base()
    z = m.binary("z")
    m.if_then(z, [x >= 6])
    m.minimize(x - 10 * z)
    z.var_type = VarType.INTEGER
    z.ub = 3
    with pytest.raises(TypeError, match="must be a single binary variable"):
        m.solve(time_limit=10)
