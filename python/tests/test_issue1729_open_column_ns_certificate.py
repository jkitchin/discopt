"""#1729: a covering LP with no explicit upper bounds was never certified.

Every structural column is open above, FBBT cannot close them, and one row
multiplier a rounding error the wrong side of zero drove the NS bound to -inf.
The repaired dual (sign projection on logical columns, then a sign-preserving
scale) is only a candidate: ``ns_bound`` computes the bound from it rigorously.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt.solvers.lp_milp_highs import INF, StdForm, ns_bound, scaled_dual_for_open_columns


def _covering(mrows: int, n: int, ub, seed: int = 0):
    rng = np.random.default_rng(seed)
    A = rng.uniform(0, 1, (mrows, n))
    b = rng.uniform(1, 10, mrows)
    c = rng.uniform(1, 2, n)
    m = dm.Model("lp")
    kw = {} if ub is None else {"ub": ub}
    x = m.continuous("x", shape=(n,), lb=0, **kw)
    m.subject_to(A @ x >= b)
    m.minimize(c @ x)
    return m


@pytest.mark.parametrize("shape", [(300, 60), (1000, 200)])
def test_open_columns_certify_like_redundant_bounds(shape):
    closed = _covering(*shape, ub=10).solve()
    open_ = _covering(*shape, ub=None).solve()
    assert closed.status == "optimal"
    assert open_.status == "optimal", open_.status
    assert open_.gap_certified
    assert open_.objective == pytest.approx(closed.objective, rel=1e-9)
    # A valid lower bound, never above the objective.
    assert open_.bound <= open_.objective + 1e-12


def test_repaired_dual_bound_never_exceeds_the_optimum():
    """Differential soundness: the candidate's NS bound against HiGHS's optimum on
    random covering LPs (std form with >= logicals), perturbing the dual by
    rounding-sized noise of both signs."""
    from scipy.optimize import linprog

    rng = np.random.default_rng(1729)
    checked = 0
    for trial in range(40):
        m_rows, n = int(rng.integers(5, 40)), int(rng.integers(3, 20))
        A = rng.uniform(0, 1, (m_rows, n))
        b = rng.uniform(1, 10, m_rows)
        c = rng.uniform(0.5, 2, n)
        res = linprog(c, A_ub=-A, b_ub=-b, bounds=(0, None), method="highs")
        assert res.status == 0
        opt = float(res.fun)
        y = -np.asarray(res.ineqlin.marginals, dtype=np.float64)  # >= 0 row duals
        y = y + rng.normal(scale=1e-14, size=y.shape)
        Astd = sp.hstack([sp.csc_matrix(A), -sp.identity(m_rows)], format="csc")
        sf = StdForm.from_arrays(
            np.concatenate([c, np.zeros(m_rows)]),
            Astd,
            b,
            np.zeros(n + m_rows),
            np.full(n + m_rows, INF),
        )
        y_s = scaled_dual_for_open_columns(y, sf)
        assert y_s is not None, trial
        g = ns_bound(y_s, sf)
        assert g is not None, trial
        assert g <= opt + 1e-9 * max(1.0, abs(opt)), (trial, g, opt)
        assert opt - g <= 1e-6 * max(1.0, abs(opt)), (trial, g, opt)
        checked += 1
    assert checked == 40


def test_free_column_with_nonzero_reduced_cost_is_refused():
    A = sp.csc_matrix(np.array([[1.0, -1.0]]))
    sf = StdForm.from_arrays(
        np.array([1.0, 0.0]), A, np.array([1.0]), np.array([-INF, 0.0]), np.array([INF, INF])
    )
    assert scaled_dual_for_open_columns(np.array([0.5]), sf) is None
