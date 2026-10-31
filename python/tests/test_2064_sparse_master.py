"""#2064 review: the OA / LP/NLP-BB masters are assembled sparse.

Every master builder (``_build_master_milp_data``, ``_solve_regularized_master``,
``_solve_integer_projection_mip``) used to stack full-width ``np.ndarray`` rows,
and ``_decompose_model`` stored every linear row at full width, so a model with
``m`` linear rows and ``n`` columns cost ``8 m n`` bytes twice before any solver
ran. Measured on a 20,000-column, 8,020-row model: peak RSS over the model
3,619 MB -> 117 MB, master build 10.95 s -> 0.05 s (performance-plan 103.16).

The refactor is bound-neutral (CLAUDE.md section 5, regime 1): the CSR equals the
old dense stack bit for bit. Checked across 178 deterministic solves / 463 masters
of the 102 routed instances (default) and 30 of them under ``oa``, ``oa`` +
``level_L1`` and ``fp``. These tests pin the pieces.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt.solvers import oa


def _wide_model(n=3000, m=1000, seed=0):
    rng = np.random.default_rng(seed)
    mod = dm.Model("wide_sparse")
    x = mod.continuous("x", shape=(n,), lb=0.0, ub=1.0)
    z = mod.binary("z", shape=(5,))
    for _ in range(m):
        cols = rng.choice(n, 5, replace=False)
        coef = rng.uniform(0.5, 2.0, 5)
        mod.subject_to(dm.sum(lambda j: float(coef[j]) * x[int(cols[j])], over=range(5)) <= 3.0)
    for k in range(5):
        mod.subject_to(x[k] <= z[k])
    mod.subject_to(dm.sum(lambda k: x[k] ** 2, over=range(5)) <= 1.0)
    mod.minimize(-dm.sum(lambda k: x[k], over=range(n)))
    return mod


def test_decomposition_stores_linear_rows_by_their_nonzeros():
    d = oa._decompose_model(_wide_model(n=400, m=50))
    assert d.linear_A_rows, "no linear rows: the probe tested nothing (section 6)"
    for row in d.linear_A_rows:
        assert isinstance(row, oa._SparseRow)
        assert len(row) == d.n_vars
        dense = np.asarray(row)
        assert dense.shape == (d.n_vars,)
        assert np.count_nonzero(dense) == row.idx.size <= 5
        np.testing.assert_array_equal(dense[row.idx], row.val)
        np.testing.assert_array_equal(np.asarray(-row), -dense)


def test_master_is_csr_with_nonzeros_proportional_to_the_model():
    d = oa._decompose_model(_wide_model())
    master = oa._build_master_milp_data(
        d.linear_A_rows,
        d.linear_b_rows,
        d.linear_senses,
        [],
        [],
        d.n_vars,
        d.integrality,
        d.lb,
        d.ub,
        d.obj_coeffs,
        d.obj_is_linear,
        True,
    )
    A = master.A_ub
    assert sp.issparse(A) and A.format == "csr"
    assert A.shape == (len(d.linear_A_rows), len(master.c))
    # 1000 five-term rows + 5 two-term links: O(nnz), not O(rows x cols).
    assert A.nnz == 5 * 1000 + 2 * 5
    for i, row in enumerate(d.linear_A_rows):
        got = A.getrow(i).toarray().ravel()
        np.testing.assert_array_equal(got[: d.n_vars], np.asarray(row))
        assert not np.any(got[d.n_vars :])


def test_master_rows_keep_order_drop_zeros_and_match_the_dense_stack():
    rng = np.random.default_rng(3)
    dense = np.where(rng.random((40, 17)) < 0.7, 0.0, rng.normal(size=(40, 17)))
    dense[5] = 0.0  # an all-zero row keeps its place
    acc = oa._MasterRows()
    for i, row in enumerate(dense):
        if i % 2:
            acc.append(row)
        else:
            nz = np.flatnonzero(row)
            acc.append_entries(np.concatenate([nz, [0]]), np.concatenate([row[nz], [0.0]]))
    csr = acc.to_csr(17)
    assert csr.has_canonical_format
    assert csr.nnz == np.count_nonzero(dense)  # the explicit 0.0 entries were dropped
    np.testing.assert_array_equal(csr.toarray(), dense)
    np.testing.assert_array_equal(csr.toarray(), sp.csr_matrix(dense).toarray())
    assert oa._MasterRows().to_csr(17) is None
    with pytest.raises(ValueError, match="outside"):
        bad = oa._MasterRows()
        bad.append_entries([17], [1.0])
        bad.to_csr(17)


def test_state_fingerprint_is_exact_on_sparse_rows():
    a = oa._SparseRow.from_dense([0.0, 1.0, 0.0, 2.0])
    b = oa._SparseRow.from_dense([0.0, 1.0, 0.0, 2.0])
    c = oa._SparseRow.from_dense([0.0, 1.0, 0.0, 2.0 + 2**-40])
    d = oa._SparseRow.from_dense([0.0, 1.0, 0.0, 2.0, 0.0])  # same nonzeros, wider
    assert oa._state_fingerprint([a]) == oa._state_fingerprint([b])
    assert oa._state_fingerprint([a]) != oa._state_fingerprint([c])
    assert oa._state_fingerprint([a]) != oa._state_fingerprint([d])


@pytest.mark.parametrize(
    "method, options",
    [("oa", None), ("oa", {"add_regularization": "level_L1"}), ("fp", None)],
    ids=["oa", "roa", "fp"],
)
def test_sparse_masters_solve_a_small_convex_minlp(method, options):
    """End to end through each builder the refactor touched."""
    from discopt.solvers.mip_nlp import solve_mip_nlp

    m = dm.Model("small")
    x = m.continuous("x", shape=(3,), lb=0.0, ub=4.0)
    y = m.binary("y", shape=(3,))
    for i in range(3):
        m.subject_to(x[i] <= 4.0 * y[i])
    m.subject_to(y[0] + y[1] + y[2] <= 2)
    m.subject_to(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 <= 8.0)
    m.minimize(-(3.0 * x[0] + 2.0 * x[1] + x[2]) + 0.5 * (y[0] + y[1] + y[2]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = solve_mip_nlp(m, method=method, time_limit=30, mip_nlp_options=options)
    # Optimum: y = (1, 1, 0), x proportional to (3, 2) on the ball of radius sqrt(8).
    opt = -np.sqrt(8.0) * np.sqrt(13.0) + 1.0
    assert r.objective is not None
    assert r.objective == pytest.approx(opt, abs=1e-4)
    if method != "fp":
        assert r.status == "optimal" and r.bound <= r.objective + 1e-9
