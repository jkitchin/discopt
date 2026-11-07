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
