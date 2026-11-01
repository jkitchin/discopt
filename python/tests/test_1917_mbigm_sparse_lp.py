"""#1917: mbigm's LP relaxation is built sparse, memoized, and never swallowed.

Measured on GDPlib ``stranded_gas``: ``_precompute_lp_relaxation`` built
``A_ub``/``A_eq`` dense (14887 x 57810 at 0.018% density, 6.4 GiB), and every
``_compute_big_m_lp`` call re-sparsified them inside ``solve_lp``: 97% of the LP
time was ``ndarray.nonzero``, and mbigm lowering took more than 600 s. The
matrices are now CSR, identical to ``csr_matrix(dense)``, so each LP is the same
LP; a repeated objective is answered from a per-lowering memo; and an extraction
failure raises instead of silently turning every LP M into an interval M.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt._relax import gdp_reformulate as G
from discopt._relax.obbt import _extract_linear_constraints


def _model():
    m = dm.Model("sparse_lp")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=-3.0, ub=4.0)
    z = m.continuous("z", shape=(3,), lb=0.0, ub=2.0)
    m.minimize(x + y)
    m.subject_to(x + 2.0 * y <= 8.0)
    m.subject_to(3.0 * x - y >= -1.0)
    m.subject_to(z[0] + z[2] - x == 0.5)
    m.subject_to(0.0 * y + z[1] <= 1.5)  # a stored zero must not appear
    m.subject_to(x * y <= 3.0)  # nonlinear: skipped by both forms
    return m


def test_sparse_extraction_is_csr_of_the_dense_extraction():
    m = _model()
    Ad_ub, bd_ub, Ad_eq, bd_eq, nd = _extract_linear_constraints(m)
    As_ub, bs_ub, As_eq, bs_eq, ns = _extract_linear_constraints(m, sparse=True)
    assert nd == ns == 5
    checked = 0
    for dense, sparse in ((Ad_ub, As_ub), (Ad_eq, As_eq)):
        assert isinstance(dense, np.ndarray)
        assert sp.issparse(sparse) and sparse.format == "csr"
        ref = sp.csr_matrix(dense)
        # Same structure, not merely the same values: solve_lp builds exactly
        # ``csr_matrix(dense)`` from a dense input.
        assert sparse.has_sorted_indices
        np.testing.assert_array_equal(sparse.indptr, ref.indptr)
        np.testing.assert_array_equal(sparse.indices, ref.indices)
        np.testing.assert_array_equal(sparse.data, ref.data)
        checked += 1
    np.testing.assert_array_equal(bd_ub, bs_ub)
    np.testing.assert_array_equal(bd_eq, bs_eq)
    assert checked == 2


def test_precompute_is_sparse_and_keeps_the_five_field_shape():
    data = G._precompute_lp_relaxation(_model())
    A_ub, b_ub, A_eq, b_eq, n_vars = data
    assert sp.issparse(A_ub) and sp.issparse(A_eq)
    assert n_vars == 5


def test_an_extraction_failure_raises(monkeypatch):
    import discopt._relax.obbt as obbt

    def boom(model, sparse=False):
        raise RuntimeError("extraction defect")

    monkeypatch.setattr(obbt, "_extract_linear_constraints", boom)
    with pytest.raises(RuntimeError, match="extraction defect"):
        G._precompute_lp_relaxation(_model())


def test_a_repeated_objective_is_solved_once(monkeypatch):
    from discopt.solvers import lp_backend

    m = _model()
    x = m._variables[0]
    data = G._precompute_lp_relaxation(m)
    real = lp_backend.get_exact_lp_solver()
    if real is None:
        pytest.skip("no exact LP oracle")
    calls = []

    def counting(**kw):
        calls.append(kw["c"].copy())
        return real(**kw)

    monkeypatch.setattr(lp_backend, "get_exact_lp_solver", lambda: counting)
    first = G._compute_big_m_lp(x <= 1.0, m, data)
    # Same objective, other constant: one LP, and the constant still applied.
    other = G._compute_big_m_lp(x <= 7.0, m, data)
    assert len(calls) == 1
    assert G._compute_big_m_lp(x <= 7.0, m, G._precompute_lp_relaxation(m)) == other
    assert len(calls) == 2
    calls.clear()
    # A fresh precompute has a fresh memo, and solves again to the same M.
    assert G._compute_big_m_lp(x <= 1.0, m, G._precompute_lp_relaxation(m)) == first
    assert len(calls) == 1
