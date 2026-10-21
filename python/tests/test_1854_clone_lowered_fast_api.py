"""#1854: ``reformulate_gdp(m, "big-m").clone()`` raised on a fast-API model.

A GDP lowering shares the source model's ``Variable`` objects, whose ``model`` is
the source. Deep-copying the lowering therefore copies the source too -- reached
through a shared leaf's state, so the source's ``__deepcopy__`` replayed its Rust
builder while that leaf's copy was still empty: ``AttributeError: 'Variable'
object has no attribute 'name'``. ``Model.__deepcopy__`` now copies every such
owning model to completion before reconstructing its own nodes.
"""

import copy
import pickle
import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.gdp_reformulate import reformulate_gdp

pytestmark = pytest.mark.smoke


def _fast_model():
    # min x0 + 2 x1, x0 + x1 <= 8, x0 >= 6 or x1 >= 6, x in [0, 10]^2: (6, 0) = 6.
    m = dm.Model("fast")
    x = m.continuous("x", shape=(2,), lb=0, ub=10)
    m.add_linear_constraints(np.array([[1.0, 1.0]]), x, "<=", np.array([8.0]))
    m.either_or([[x[0] >= 6], [x[1] >= 6]])
    m.add_linear_objective(np.array([1.0, 2.0]), x)
    return m


def _solve(m):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m.solve(time_limit=60)


@pytest.mark.parametrize("method", ["big-m", "mbigm", "hull"])
@pytest.mark.parametrize("how", ["clone", "deepcopy", "pickle"])
def test_lowered_fast_api_model_copies(method, how):
    m = _fast_model()
    lowered = reformulate_gdp(m, method)
    if how == "clone":
        c = lowered.clone()
    elif how == "deepcopy":
        c = copy.deepcopy(lowered)
    else:
        c = pickle.loads(pickle.dumps(lowered))
    assert c is not lowered
    assert len(c._constraints) == len(lowered._constraints)
    r = _solve(c)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(6.0, abs=1e-6)
    # The copy is independent: the source and its lowering still solve.
    assert _solve(lowered).objective == pytest.approx(6.0, abs=1e-6)
    assert _solve(m).objective == pytest.approx(6.0, abs=1e-6)


def test_copy_keeps_shared_leaves_consistent():
    m = _fast_model()
    lowered = reformulate_gdp(m, "big-m")
    c = lowered.clone()
    x_copy = c._variables[0]
    assert x_copy is not m._variables[0]
    assert x_copy.name == "x"
    # The shared leaf's owner was copied with it, not left pointing at ``m``.
    assert x_copy.model is not m
