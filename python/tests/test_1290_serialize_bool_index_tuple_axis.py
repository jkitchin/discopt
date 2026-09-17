"""Native save/load keeps boolean indices and tuple sum axes (#1290).

``bool`` subclasses ``int``, so ``_enc_index`` wrote ``x[True]`` as ``x[1]``. numpy
reads ``x[True]`` as a new leading axis over the whole array, so the reloaded
model maximized ``x[1]`` (2.0) instead of ``x.sum()`` (6.0) and still reported
optimal. A tuple-axis ``mean`` crashed the writer with a bare ``TypeError``.
"""

import discopt.modeling as dm
import discopt.serialize as ser
import numpy as np
import pytest
from discopt.modeling.core import SumExpression


def _bool_model(index):
    m = dm.Model("b")
    x = m.continuous("x", shape=(3,), lb=0, ub=[1, 2, 3])
    m.maximize(x[index].sum())
    return m


@pytest.mark.parametrize("index", [True, np.bool_(True)])
def test_bool_index_round_trips(index):
    m = _bool_model(index)
    text = ser.dumps(m, provenance=False)
    m2 = ser.loads(text)
    assert ser.dumps(m2, provenance=False) == text
    r = m.solve(time_limit=30)
    r2 = m2.solve(time_limit=30)
    assert r.status == r2.status == "optimal"
    assert r.objective == pytest.approx(6.0, abs=1e-6)
    assert r2.objective == pytest.approx(r.objective, abs=1e-9)


def test_bool_index_is_not_an_int_index():
    doc_true = ser.dumps(_bool_model(True), provenance=False)
    doc_one = ser.dumps(_bool_model(1), provenance=False)
    assert doc_true != doc_one


def test_tuple_axis_round_trips():
    m = dm.Model("t")
    X = m.continuous("X", shape=(2, 3), lb=0, ub=[[1, 2, 3], [4, 5, 6]])
    obj = X.mean(axis=(0, 1))
    m.maximize(obj)

    def axes(model):
        found = []
        stack = [model._objective.expression]
        while stack:
            nd = stack.pop()
            if isinstance(nd, SumExpression):
                found.append(nd.axis)
            stack.extend(v for v in vars(nd).values() if isinstance(v, dm.Expression))
        return found

    assert (0, 1) in axes(m)
    text = ser.dumps(m, provenance=False)
    m2 = ser.loads(text)
    assert (0, 1) in axes(m2)
    assert ser.dumps(m2, provenance=False) == text
    r2 = m2.solve(time_limit=30)
    assert r2.status == "optimal"
    assert r2.objective == pytest.approx(21.0 / 6.0, abs=1e-6)


@pytest.mark.parametrize("axis", [1.5, (0, 1.0), True])
def test_unsupported_axis_is_a_serialization_error(axis):
    m = dm.Model("u")
    x = m.continuous("x", shape=(2, 2), lb=0, ub=1)
    m.maximize(SumExpression(x, axis).sum())
    with pytest.raises(ser.SerializationError):
        ser.dumps(m)
