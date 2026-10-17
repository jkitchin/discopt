"""#1730: row-matched duals were O(rows^2) ``allclose`` calls on a dense LP.

Every row of a dense ``A`` shares one sparsity pattern, so bucketing by support
left all rows in one bucket. The projection-key window keeps the exact test but
visits only candidates it could accept.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np


def _dense_lp(m_rows: int = 300, n: int = 40):
    rng = np.random.default_rng(0)
    A = rng.uniform(0, 1, (m_rows, n))
    b = rng.uniform(1, 10, m_rows)
    c = rng.uniform(1, 2, n)
    m = dm.Model("lp")
    x = m.continuous("x", shape=(n,), lb=0, ub=10)
    m.subject_to(A @ x >= b, name="cov")
    m.minimize(c @ x)
    return m, A, c


def test_dense_row_match_is_not_quadratic(monkeypatch):
    m, A, c = _dense_lp()
    calls = {"n": 0}
    real = np.allclose

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(np, "allclose", counting)
    r = m.solve()
    assert r.status == "optimal"
    rows = A.shape[0]
    assert calls["n"] > 0  # the probe fired: matching ran
    # The full scan made 2 * rows * rows comparisons (180,000 here).
    assert calls["n"] < 10 * rows, calls["n"]


def test_dense_row_match_duals_satisfy_stationarity():
    m, A, c = _dense_lp()
    r = m.solve()
    assert r.status == "optimal"
    assert r.constraint_duals is not None and "cov" in r.constraint_duals
    mu = np.asarray(r.constraint_duals["cov"])
    assert mu.shape == (A.shape[0],)
    assert np.all(mu >= -1e-9)
    lo = np.asarray(r.bound_duals_lower["x"])
    up = np.asarray(r.bound_duals_upper["x"])
    # min c'x s.t. A x >= b, 0 <= x <= 10:  c - A' mu - lo + up = 0.
    np.testing.assert_allclose(c - A.T @ mu - lo + up, 0.0, atol=1e-6)
