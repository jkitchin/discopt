"""``readonly_model_repr``: one marshal per model state, never a stale repr.

``telecomsp_nor_sun`` paid ~12 s per ``model_to_repr`` and six of them per
solve on an unchanged model. The memo is only worth having if it can never hand
back a repr of a *different* model state, so most of this file is the misses:
one test per kind of in-place change the key has to see.
"""

from __future__ import annotations

import pickle

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import _repr_cache as rc
from discopt._relax.problem_classifier import ProblemClass, classify_problem
from discopt.modeling.core import VarType


def _model():
    m = dm.Model("rc")
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=-1, ub=3)
    p = m.parameter("p", value=2.0)
    m.subject_to(x + p * y <= 5, name="c0")
    m.subject_to(x * y >= -2, name="c1")
    m.minimize(x + y)
    return m, x, y, p


@pytest.fixture
def counted(monkeypatch):
    import discopt._rust as rust

    real = rust.model_to_repr
    calls: list[int] = []

    def spy(model, *a, **k):
        calls.append(1)
        return real(model, *a, **k)

    monkeypatch.setattr(rust, "model_to_repr", spy)
    return calls


def test_unchanged_model_is_marshalled_once(counted):
    m, *_ = _model()
    r1 = rc.readonly_model_repr(m)
    r2 = rc.readonly_model_repr(m)
    assert r1 is r2
    assert len(counted) == 1


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda m, x, y, p: setattr(x, "ub", 3.0), id="bound"),
        pytest.param(lambda m, x, y, p: setattr(y, "var_type", VarType.INTEGER), id="var_type"),
        pytest.param(lambda m, x, y, p: setattr(p, "value", 7.0), id="parameter"),
        pytest.param(lambda m, x, y, p: m.subject_to(x <= 3), id="new_row"),
        pytest.param(lambda m, x, y, p: setattr(m._constraints[0], "rhs", 4.0), id="rhs"),
        pytest.param(lambda m, x, y, p: setattr(m._constraints[0], "sense", ">="), id="sense"),
        pytest.param(lambda m, x, y, p: setattr(m._constraints[1], "name", "zz"), id="name"),
        pytest.param(lambda m, x, y, p: m._constraints.pop(), id="dropped_row"),
        pytest.param(lambda m, x, y, p: m.minimize(x * x), id="objective"),
        pytest.param(lambda m, x, y, p: m.maximize(x + y), id="sense_flip"),
    ],
)
def test_every_in_place_change_is_a_miss(counted, mutate):
    m, x, y, p = _model()
    r1 = rc.readonly_model_repr(m)
    mutate(m, x, y, p)
    r2 = rc.readonly_model_repr(m)
    assert r2 is not r1
    assert len(counted) == 2
    # and the fresh repr is the one an uncached call builds
    import discopt._rust as rust

    fresh = rust.model_to_repr(m, None)
    assert r2.n_constraints == fresh.n_constraints
    assert r2.is_objective_linear() == fresh.is_objective_linear()


def test_rows_outside_the_key_are_never_cached(counted):
    """An indicator row carries state the key does not read: never cached."""
    m, x, y, p = _model()
    b = m.binary("b")
    m.if_then(b, [x <= 1])
    assert rc._state_key(m) is None
    assert getattr(m, rc._ATTR, None) is None


def test_classification_is_unchanged_by_the_cache():
    checked = 0
    for build in (_model,):
        m, x, y, p = build()
        first = classify_problem(m)
        assert classify_problem(m) == first
        x.var_type = VarType.INTEGER  # in place: must be re-read
        assert classify_problem(m) in (ProblemClass.MINLP, ProblemClass.MIQCP)
        checked += 1
    assert checked == 1


def test_cache_is_dropped_by_pickle():
    m, *_ = _model()
    rc.readonly_model_repr(m)
    assert getattr(m, rc._ATTR, None) is not None
    m2 = pickle.loads(pickle.dumps(m))
    assert getattr(m2, rc._ATTR, None) is None


def test_bound_values_not_identities_decide(counted):
    """B&B rebinds bounds with fresh arrays of equal value: still a hit."""
    m, x, *_ = _model()
    rc.readonly_model_repr(m)
    x.lb = np.array(0.0)
    rc.readonly_model_repr(m)
    assert len(counted) == 1
