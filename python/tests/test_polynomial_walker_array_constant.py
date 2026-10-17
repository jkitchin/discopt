"""Scalar polynomial walkers must decline array-valued ``Constant`` leaves.

``_product_to_monomial`` / ``_expr_to_polynomial`` in ``_relax/milp_relaxation.py``
called ``float(e.value)`` on every ``Constant``. An objective like
``dm.sum(C[1:, :] * (v[1:, :] - v[:-1, :]) ** 2)`` (COPS ``bearing``-style
per-row weights on squared finite differences) holds the (3, 3) array
``C[1:, :]`` in a ``Constant``, so the root edge-concave separator crashed
``Model.solve()`` with ``TypeError: only 0-dimensional arrays can be converted
to Python scalars``. The walkers now decline (return ``None``) on a non-scalar
constant, which only drops a candidate cut and is therefore sound; scalar and
size-1 constants are folded exactly as before.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.milp_relaxation import _expr_to_polynomial, _product_to_monomial


def _bearing_like_model() -> dm.Model:
    m = dm.Model("t")
    v = m.continuous("v", shape=(4, 3), lb=-5, ub=5)
    w = m.continuous("w", shape=(4, 2, 3), lb=-5, ub=5)
    th = m.continuous("th", lb=0, ub=3)
    C = np.arange(12.0).reshape(4, 3)
    m.subject_to(th * w[:, 0, 1] - v[:, 2] / (1 + v[:, 0] ** 2) == 0)
    m.subject_to(v[0, :] == 0)
    m.subject_to(dm.sqrt(1 + v[:, 1] ** 2) <= 3)
    m.minimize(dm.sum(C[1:, :] * (v[1:, :] - v[:-1, :]) ** 2) + 1.0)
    return m


def test_walkers_decline_array_constant_factor():
    m = dm.Model("u")
    x = m.continuous("x", shape=(3,), lb=-1, ub=1)
    coeff = np.array([1.0, 2.0, 3.0])
    prod = coeff * x[0] ** 2
    assert _product_to_monomial(prod, m) is None
    assert _expr_to_polynomial(prod, m) is None
    # An array-valued additive constant is not one scalar polynomial either.
    assert _expr_to_polynomial(x[0] + coeff, m) is None


def test_walkers_unchanged_on_scalar_constants():
    m = dm.Model("u")
    x = m.continuous("x", shape=(3,), lb=-1, ub=1)
    i0 = _expr_to_polynomial(x[0], m)[1][0][1][0]
    i1 = _expr_to_polynomial(x[1], m)[1][0][1][0]
    assert _product_to_monomial(3.0 * x[0] ** 2 * x[1], m) == (3.0, tuple(sorted((i0, i0, i1))))
    const, terms = _expr_to_polynomial(2.0 * x[0] * x[1] + 4.0 - x[1] / 2.0, m)
    assert const == 4.0
    assert sorted(terms) == sorted([(2.0, tuple(sorted((i0, i1)))), (-0.5, (i1,))])


def test_walkers_unwrap_size1_array_constant():
    # A size-1 array is a scalar: unwrap it rather than decline. (NumPy >= 2.5
    # removed ``float()`` on ndim > 0 arrays, so the old walker crashed here too;
    # older NumPy unwrapped it with a deprecation warning, as this does.)
    m = dm.Model("u")
    x = m.continuous("x", shape=(3,), lb=-1, ub=1)
    i0 = _expr_to_polynomial(x[0], m)[1][0][1][0]
    assert _product_to_monomial(np.array([2.5]) * x[0], m) == (2.5, (i0,))


@pytest.mark.slow
def test_solve_array_coefficient_times_square_does_not_crash():
    m = _bearing_like_model()
    r = m.solve(time_limit=20)
    assert r.status is not None
    assert r.objective is not None and np.isfinite(r.objective)
    # Objective is C-weighted squares (C >= 0) plus 1, so it is >= 1.
    assert r.objective >= 1.0 - 1e-6
    bound = getattr(r, "bound", None)
    if bound is not None and np.isfinite(bound):
        assert bound <= r.objective + 1e-6
