"""highs-parity D2: ``_milp_aligned_lp`` gives the native MILP driver an aligned standard
form, where column ``n_orig + r`` is row ``r``'s slack and each equality row gets a slack
fixed at ``[0, 0]``.

The route's ``extract_lp_data`` puts slacks on inequality rows only, in arbitrary order,
which broke the driver's root GMI loop (``DISCOPT_MILP_ALIGNED_SLACKS`` docstring). These
tests pin the layout and show the feasible set is unchanged: the same model solves to the
same certified optimum with the flag on and off.
"""

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt._relax.problem_classifier import extract_lp_data
from discopt.solver import _milp_aligned_lp


def _model():
    m = dm.Model("d2_aligned")
    x = m.integer("x", shape=(4,), lb=0, ub=5)
    m.subject_to(x[0] + 2 * x[1] - x[2] == 3)
    m.subject_to(3 * x[0] + x[3] <= 9)
    m.subject_to(x[1] + x[2] + x[3] >= 2)
    m.subject_to(2 * x[0] + 2 * x[1] + 2 * x[2] == 6)
    m.minimize(-x[0] - 2 * x[1] + x[2] + 0.5 * x[3])
    return m


@pytest.mark.smoke
def test_aligned_layout_puts_row_r_slack_at_n_orig_plus_r():
    m = _model()
    lp = extract_lp_data(m)
    n_orig = sum(v.size for v in m._variables)
    al = _milp_aligned_lp(lp, n_orig)
    A = sp.csc_matrix(al.A_eq)
    rows = A.shape[0]
    assert A.shape[1] >= n_orig + rows
    checked = 0
    for r in range(rows):
        k = n_orig + r
        col = A[:, k].tocoo()
        assert list(col.row) == [r], f"column {k} is not row {r}'s singleton slack"
        checked += 1
    # Structural columns, costs and bounds are untouched.
    assert np.array_equal(sp.csc_matrix(lp.A_eq)[:, :n_orig].toarray(), A[:, :n_orig].toarray())
    assert np.array_equal(np.asarray(lp.c)[:n_orig], al.c[:n_orig])
    # Equality rows got a [0, 0] slack (the model has two).
    fixed = [r for r in range(rows) if al.x_l[n_orig + r] == 0.0 and al.x_u[n_orig + r] == 0.0]
    assert len(fixed) >= 2
    assert checked == rows


@pytest.mark.smoke
def test_aligned_slacks_solve_to_the_same_certified_optimum(monkeypatch):
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    out = {}
    for arm in ("0", "1"):
        monkeypatch.setenv("DISCOPT_MILP_ALIGNED_SLACKS", arm)
        r = _model().solve(time_limit=30)
        assert str(r.status).endswith("optimal"), (arm, r.status)
        out[arm] = r.objective
    assert out["0"] == pytest.approx(out["1"], abs=1e-9)


def _aligned_by_column_stack(lp_data, n_orig):
    """The pre-2026-11-13 ``_milp_aligned_lp`` (one ``A[:, k]`` slice per column,
    then ``hstack``): the reference the array assembly must reproduce exactly."""
    A = sp.csc_matrix(lp_data.A_eq)
    m, n = A.shape
    slack_of_row = [-1] * m
    for k in range(n_orig, n):
        lo, hi = A.indptr[k], A.indptr[k + 1]
        if hi - lo == 1:
            r = int(A.indices[lo])
            if slack_of_row[r] < 0:
                slack_of_row[r] = k
    used = {k for k in slack_of_row if k >= 0}
    rest = [k for k in range(n_orig, n) if k not in used]
    c, xl, xu = (np.asarray(v, dtype=np.float64) for v in (lp_data.c, lp_data.x_l, lp_data.x_u))
    cols = [A[:, :n_orig]]
    cc, ll, uu = [c[:n_orig]], [xl[:n_orig]], [xu[:n_orig]]
    for r in range(m):
        k = slack_of_row[r]
        if k >= 0:
            cols.append(A[:, k])
            cc.append(c[k : k + 1])
            ll.append(xl[k : k + 1])
            uu.append(xu[k : k + 1])
        else:
            cols.append(sp.csc_matrix(([1.0], ([r], [0])), shape=(m, 1)))
            cc.append(np.zeros(1))
            ll.append(np.zeros(1))
            uu.append(np.zeros(1))
    for k in rest:
        cols.append(A[:, k])
        cc.append(c[k : k + 1])
        ll.append(xl[k : k + 1])
        uu.append(xu[k : k + 1])
    return sp.hstack(cols, format="csc"), np.concatenate(cc), np.concatenate(ll), np.concatenate(uu)


def _assert_same(lp, n_orig):
    want_a, want_c, want_l, want_u = _aligned_by_column_stack(lp, n_orig)
    got = _milp_aligned_lp(lp, n_orig)
    ga = sp.csc_matrix(got.A_eq)
    assert ga.shape == want_a.shape
    assert np.array_equal(ga.indptr, want_a.indptr)
    assert np.array_equal(ga.indices, want_a.indices)
    assert np.array_equal(ga.data, want_a.data)
    for g, w in ((got.c, want_c), (got.x_l, want_l), (got.x_u, want_u)):
        assert np.array_equal(np.asarray(g), w)
    return ga.nnz


@pytest.mark.smoke
def test_milp_aligned_lp_matches_the_column_stack():
    """The array assembly is the column stack exactly (indptr, indices, values, costs,
    bounds): on the model above, and on 200 random standard forms with singleton slacks
    in random order, rows without a slack, two slacks on one row, and non-slack columns
    after the structural block. Counts what it compared (CLAUDE.md section 6)."""
    from collections import namedtuple

    compared = _assert_same(extract_lp_data(_model()), 4)
    Lp = namedtuple("Lp", "A_eq c x_l x_u")
    rng = np.random.default_rng(7)
    for _ in range(200):
        m_rows = int(rng.integers(1, 12))
        n_orig = int(rng.integers(1, 10))
        struct = sp.random(m_rows, n_orig, density=0.4, random_state=rng, format="csc")
        extra = []
        for r in rng.permutation(m_rows):
            if rng.random() < 0.7:  # a singleton slack
                extra.append(
                    sp.csc_matrix(([rng.choice([1.0, -1.0])], ([r], [0])), shape=(m_rows, 1))
                )
            if rng.random() < 0.15:  # a second singleton on the same row
                extra.append(sp.csc_matrix(([2.0], ([r], [0])), shape=(m_rows, 1)))
        if rng.random() < 0.5:  # a non-singleton extra column
            extra.append(sp.random(m_rows, 1, density=0.6, random_state=rng, format="csc"))
        A = sp.hstack([struct] + extra, format="csc")
        n = A.shape[1]
        lp = Lp(A, rng.normal(size=n), -rng.random(n), rng.random(n) + 1.0)
        compared += _assert_same(lp, n_orig)
    assert compared > 1000, compared
