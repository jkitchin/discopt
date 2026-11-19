"""A scipy sparse matrix combined with a discopt expression fails with guidance.

``scipy_sparse @ x`` raised ``AttributeError: 'Variable' object has no attribute
'ndim'`` from inside scipy's matmul dispatch, and ``x @ scipy_sparse`` raised numpy's
``ValueError: setting an array element with a sequence``. Neither said what was wrong
or what to do instead. Both now name ``Model.add_linear_constraints``. Found while
reviewing PR #2260.
"""

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt.modeling.core import Expression

HINT = "add_linear_constraints"


@pytest.fixture
def x():
    m = dm.Model()
    return m.continuous("x", shape=(3,), lb=0, ub=1)


@pytest.mark.parametrize("make", [sp.csr_matrix, sp.csr_array, sp.coo_array, sp.csc_matrix])
def test_sparse_matmul_variable_names_the_remedy(x, make):
    A = make(np.eye(3))
    with pytest.raises(AttributeError, match=HINT):
        A @ x


@pytest.mark.parametrize("make", [sp.csr_matrix, sp.csr_array])
def test_variable_matmul_sparse_names_the_remedy(x, make):
    A = make(np.eye(3))
    with pytest.raises(TypeError, match=HINT):
        x @ A
    with pytest.raises(TypeError, match=HINT):
        x + A


def test_expression_still_has_no_ndim(x):
    # ``hasattr`` keeps its meaning for callers that probe for array-likes.
    assert not hasattr(x, "ndim")
    assert not hasattr(x + 1, "ndim")
    assert getattr(x, "ndim", None) is None


def test_dense_matmul_and_the_remedy_still_work(x):
    A = np.eye(3)
    assert isinstance(A @ x, Expression)
    m = dm.Model()
    y = m.continuous("y", shape=(3,), lb=0, ub=1)
    m.add_linear_constraints(sp.csr_matrix(np.eye(3)), y, ">=", np.full(3, 0.5))
    m.minimize(dm.sum(y))
    r = m.solve()
    assert r.status == "optimal"
    assert r.objective == pytest.approx(1.5, abs=1e-6)
