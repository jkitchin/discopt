"""#2153: the GDP master keeps its rows by their nonzeros.

GDPlib ``stranded_gas`` under ``gdp_method="loa"`` overran a 60 s ``time_limit``
by ~7 s or ~50 s (bimodal). Two of the steps behind it were in the master:

* ``build_master`` allocated one dense ``n``-long vector per row and
  ``MasterMILP.add_row`` kept it: 15,205 rows x 57,828 columns (367,404
  nonzeros) is 7 GB, and allocating it took 2.8 s or 27 s depending on the
  allocator's state. Every master solve then rescanned the dense rows for their
  nonzeros (2.7 s per solve).
* the Rust MILP driver's in-engine refined recovery built the ``m x m`` basis
  densely (m = 22,501), 13-21 s of a 2 s budget. Fixed in
  ``crates/discopt-core/src/lp/simplex/{linsolve,dual,primal}.rs``, pinned by
  that crate's ``*_2153`` tests.

The change is bound-neutral (CLAUDE.md section 5, regime 1): the rows reach the
engine as the same CSR, bit for bit. These tests pin that, and pin the storage.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt._relax.gdp_reformulate import _extract_body_coeffs, _extract_body_terms
from discopt.gdp.algorithms.master import (
    MasterMILP,
    RelaxationMaster,
    _stack_sparse,
    build_master,
)
from discopt.solvers.oa import _SparseRow

pytestmark = pytest.mark.smoke


def _wide_linear_model(n=4000, m=300, seed=0):
    rng = np.random.default_rng(seed)
    mod = dm.Model("wide_2153")
    x = mod.continuous("x", shape=(n,), lb=0.0, ub=1.0)
    z = mod.binary("z", shape=(3,))
    for _ in range(m):
        cols = rng.choice(n, 4, replace=False)
        coef = rng.uniform(0.5, 2.0, 4)
        mod.subject_to(dm.sum(lambda j: float(coef[j]) * x[int(cols[j])], over=range(4)) <= 2.0)
    mod.subject_to(x[0] + x[1] - 0.0 * x[2] == z[0])  # an explicit zero coefficient
    mod.subject_to(x[3] >= z[1])
    mod.subject_to(x[4] ** 2 <= z[2])  # nonlinear: not a master row
    mod.minimize(-x[0] - 2.0 * x[3] + z[0] + z[1])
    return mod


def test_extract_body_terms_is_the_dense_extraction_without_the_vector():
    mod = _wide_linear_model(n=50, m=20)
    n = sum(int(v.size) for v in mod._variables)
    compared = 0
    for c in mod._constraints:
        dense = _extract_body_coeffs(c.body, mod, n)
        terms = _extract_body_terms(c.body, mod, n)
        assert (dense is None) == (terms is None)
        if dense is None:
            continue
        coeffs, off = terms
        scattered = np.zeros(n)
        for k, v in coeffs.items():
            scattered[k] = v
        np.testing.assert_array_equal(scattered, dense[0])
        assert off == dense[1]
        compared += 1
    assert compared == 22, "every linear row was compared (section 6)"


def test_build_master_stores_rows_by_their_nonzeros_and_matches_the_dense_rows():
    mod = _wide_linear_model()
    master = build_master(mod)
    n = master.n_vars
    assert len(master.rows) == 302
    compared = 0
    for row, c in zip(master.rows, [c for c in mod._constraints][:302]):
        assert isinstance(row, _SparseRow), "a dense row is 8 n bytes (#2153)"
        assert row.idx.size <= 4 and len(row) == n
        dense, _ = _extract_body_coeffs(c.body, mod, n)
        np.testing.assert_array_equal(np.asarray(row), dense)
        assert np.all(np.diff(row.idx) > 0) and np.all(row.val != 0.0)
        compared += 1
    assert compared == 302
    # The explicit 0 * x[2] is not stored.
    assert 2 not in set(master.rows[300].idx.tolist())


def test_the_stacked_master_is_the_dense_stack_bit_for_bit():
    rng = np.random.default_rng(7)
    dense = np.where(rng.random((30, 19)) < 0.7, 0.0, rng.normal(size=(30, 19)))
    dense[4] = 0.0
    signs = [(-1.0 if i % 3 == 0 else 1.0) for i in range(30)]
    want = _stack_sparse(list(dense), 21, signs)  # dense rows: the pre-#2153 input
    got = _stack_sparse([_SparseRow.from_dense(r) for r in dense], 21, signs)
    for a, b in ((want, got),):
        assert a.shape == b.shape == (30, 21)
        np.testing.assert_array_equal(a.indptr, b.indptr)
        np.testing.assert_array_equal(a.indices, b.indices)
        assert a.data.tobytes() == b.data.tobytes()
    with pytest.raises(ValueError, match="entries"):
        _stack_sparse([_SparseRow.from_dense(np.ones(22))], 21)


def test_a_sparse_and_a_dense_master_solve_identically():
    mod = _wide_linear_model(n=300, m=60, seed=3)
    sparse_master = build_master(mod)
    dense_master = MasterMILP(
        sparse_master.n_vars,
        sparse_master.lb,
        sparse_master.ub,
        sparse_master.integrality,
        objective=sparse_master.objective,
    )
    for row, sense, rhs in zip(sparse_master.rows, sparse_master.senses, sparse_master.rhs):
        dense_master.add_row(np.asarray(row), sense, rhs)  # a dense caller still works
    a = sparse_master.solve(30.0)
    b = dense_master.solve(30.0)
    assert a.status == b.status and a.x is not None
    np.testing.assert_array_equal(a.x, b.x)
    assert a.bound == b.bound


def test_add_row_refuses_a_row_of_the_wrong_width():
    master = MasterMILP(3, np.zeros(3), np.ones(3), np.zeros(3, dtype=np.int32))
    with pytest.raises(ValueError, match="expected 3"):
        master.add_row(np.ones(4), "<=", 1.0)
    with pytest.raises(ValueError, match="expected 3"):
        master.add_row(_SparseRow.from_dense(np.ones(2)), "<=", 1.0)
    with pytest.raises(ValueError, match="expected 3"):
        master.add_row(np.ones((3, 1)), "<=", 1.0)
    with pytest.raises(ValueError, match="unknown sense"):
        master.add_row(np.ones(3), "<", 1.0)


@pytest.mark.parametrize("as_sparse", [True, False], ids=["csr", "dense"])
def test_the_relaxation_master_reads_a_sparse_relaxation_without_densifying(as_sparse):
    A = np.array([[-1.0, -1.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0]])

    class _Relax:
        _c = np.array([1.0, 1.0, 0.0])
        _obj_offset = 0.0
        _objective_bound_valid = True
        _b_ub = np.array([-1.0, -0.5, 0.0])
        _bounds = [(0.0, 1.0), (0.0, 2.0), (0.0, 3.0)]
        _integrality = np.array([1, 0, 0])
        _build_truncated = False

    # A CSR with an explicit stored zero and unsorted column indices.
    csr = sp.csr_matrix(
        (np.array([-1.0, 0.0, -1.0, -1.0]), np.array([1, 2, 0, 2]), np.array([0, 3, 4, 4])),
        shape=(3, 3),
    )
    _Relax._A_ub = csr if as_sparse else A
    master = RelaxationMaster(_Relax(), 2)
    assert len(master.rows) == 3
    for row, want in zip(master.rows, A):
        assert isinstance(row, _SparseRow)
        np.testing.assert_array_equal(np.asarray(row), want)
        assert np.all(row.val != 0.0) and np.all(np.diff(row.idx) > 0)
    r = master.solve(10.0)
    assert r.bound == pytest.approx(1.0, abs=1e-6)
