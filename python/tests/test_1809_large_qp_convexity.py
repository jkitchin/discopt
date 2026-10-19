"""#1809: the exact-Hessian QP convexity cap left large convex QPs uncertified.

``certify_quadratic_objective_convex`` refused any model with ``n + m > 2000``
before extraction, and any extracted ``Q`` wider than 2000 after it. The cap
guards a *dense* eigensolve, but it refused the proof outright, so a separable
convex QP of 2500 variables fell through to spatial McCormick B&B and returned
``feasible`` with bound 1.34 against an incumbent of 105.5 at a 60 s limit.

The cap now bounds the dense work only. A larger ``Q`` goes to the shared
``psd_certified`` predicate (#1679), which drops the zero slack padding and
decides a sparse block by exact rational elimination: a proof, or "not proved".
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt._relax.convexity.certificate import (
    _QP_EXACT_CONVEXITY_MAX_N,
    certify_quadratic_objective_convex,
)

N = 2500


def _qp(n, arm, q=None):
    """The issue's repro: ``Q = diag(q)``, ``x in [0, 10]^n``, ``n/10`` covering rows."""
    rng = np.random.default_rng(0)
    q = rng.random(n) + 0.1 if q is None else q
    c = rng.random(n)
    k = n // 10
    m = dm.Model("qp")
    x = m.continuous("x", shape=(n,), lb=0, ub=10)
    A = sp.random(k, n, density=5 / n, random_state=1, format="csr") + sp.csr_matrix(
        (np.ones(k), (np.arange(k), np.arange(k))), shape=(k, n)
    )
    m.add_linear_constraints(A, x, ">=", np.ones(k))
    if arm == "fast":
        m.add_quadratic_objective(sp.diags(q, format="csr"), c, x)
    else:
        m.minimize(dm.sum(0.5 * q * x * x) + dm.sum(c * x))
    return m


@pytest.mark.smoke
@pytest.mark.parametrize("arm", ["fast", "pure"])
def test_large_separable_convex_qp_is_certified(arm):
    assert N > _QP_EXACT_CONVEXITY_MAX_N
    assert certify_quadratic_objective_convex(_qp(N, arm)) is True


@pytest.mark.smoke
@pytest.mark.parametrize("arm", ["fast", "pure"])
def test_large_nonconvex_qp_is_not_certified(arm):
    q = np.random.default_rng(0).random(N) + 0.1
    q[N // 2] = -1e-3  # one concave direction
    assert certify_quadratic_objective_convex(_qp(N, arm, q=q)) is False


@pytest.mark.smoke
def test_large_indefinite_tridiagonal_is_not_certified():
    # Every diagonal is positive, so no row is refuted on sight: the elimination
    # has to find the negative pivot. lambda_min = 1.9 - 2 cos(pi/(n+1)) < 0.
    n = N
    Q = sp.diags(
        [np.full(n - 1, -1.0), np.full(n, 1.9), np.full(n - 1, -1.0)], [-1, 0, 1], format="csr"
    )
    m = dm.Model("tri")
    x = m.continuous("x", shape=(n,), lb=0, ub=1)
    m.add_quadratic_objective(Q, np.ones(n), x)
    assert certify_quadratic_objective_convex(m) is False


@pytest.mark.smoke
def test_row_count_no_longer_counts_against_the_cap():
    # n = 1900, m = 190: n + m = 2090 > 2000 was refused before any extraction,
    # although the original-variable block is only 1900 wide.
    assert certify_quadratic_objective_convex(_qp(1900, "pure")) is True


@pytest.mark.slow
@pytest.mark.parametrize("arm", ["fast", "pure"])
def test_large_convex_qp_solves_to_optimality(arm):
    r = _qp(N, arm).solve(time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(105.515282, rel=1e-6)
    assert r.bound <= r.objective + 1e-6
    assert r.bound == pytest.approx(r.objective, rel=1e-4)
