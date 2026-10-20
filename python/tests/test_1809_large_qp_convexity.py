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


# Review of #1809: with the pre-extraction gate gone, a dense builder QP just above
# the cap paid ``classify_problem`` + ``extract_qp_data`` (~10 s at n=2100) for a
# verdict that could only be "not proved". ``_builder_hessian_certain_unproved``
# refuses exactly those from the builder block, before any extraction.


def _dense_pd_builder_qp(n, n_dense=None):
    rng = np.random.default_rng(0)
    k = n if n_dense is None else n_dense
    A = rng.standard_normal((k, k))
    Q = np.zeros((n, n))
    Q[:k, :k] = A @ A.T / k + np.eye(k)
    m = dm.Model("dense")
    x = m.continuous("x", shape=(n,), lb=0, ub=1)
    m.add_quadratic_objective(Q, np.ones(n), x)
    return m


def _refuse_extraction(monkeypatch):
    import discopt._relax.problem_classifier as pc

    def boom(*_a, **_k):
        raise AssertionError("extraction ran on a model predicted not-proved")

    monkeypatch.setattr(pc, "extract_qp_data", boom)
    monkeypatch.setattr(pc, "classify_problem", boom)


@pytest.mark.smoke
@pytest.mark.parametrize("sense", ["minimize", "maximize"])
@pytest.mark.parametrize("kind", ["sym", "upper", "lower", "asym"])
def test_builder_prediction_matches_extracted_hessian(kind, sense):
    """The prediction's premise: the extracted Hessian is exactly +-S, zero elsewhere."""
    from discopt._relax.problem_classifier import extract_qp_data

    n = 60
    rng = np.random.default_rng(3)
    Q = sp.random(n, n, density=0.1, random_state=rng, format="csr")
    Q = {"sym": Q + Q.T, "upper": sp.triu(Q), "lower": sp.tril(Q), "asym": Q}[kind].tocsr()
    m = dm.Model("rel")
    m.continuous("y", shape=(3,), lb=0, ub=1)
    x = m.continuous("x", shape=(n,), lb=-1, ub=1)
    m.add_linear_constraints(sp.csr_matrix(np.ones((2, n))), x, "<=", np.ones(2))
    m.add_quadratic_objective(Q, np.ones(n), x, sense=sense)
    # #1818: a non-symmetric Q is stored as its symmetric part (same x'Qx); the
    # builder then reads S = triu + triu' of that, which is the matrix itself.
    Qd = Q.toarray()
    S = Qd if np.array_equal(Qd, Qd.T) else 0.5 * Qd + 0.5 * Qd.T
    qd = extract_qp_data(m)
    E = qd.Q.toarray() if sp.issparse(qd.Q) else np.asarray(qd.Q)
    expected = np.zeros_like(E)
    expected[3 : 3 + n, 3 : 3 + n] = S if sense == "minimize" else -S
    assert np.array_equal(E, expected)


@pytest.mark.smoke
def test_dense_builder_qp_above_cap_skips_extraction(monkeypatch):
    m = _dense_pd_builder_qp(_QP_EXACT_CONVEXITY_MAX_N + 100)
    _refuse_extraction(monkeypatch)
    assert certify_quadratic_objective_convex(m) is False


@pytest.mark.smoke
def test_prediction_leaves_provable_models_alone():
    from discopt._relax.convexity.certificate import _builder_hessian_certain_unproved

    # Wider than the cap, but only 1500 active rows: the dense eigensolve proves it.
    m = _dense_pd_builder_qp(_QP_EXACT_CONVEXITY_MAX_N + 100, n_dense=1500)
    assert not _builder_hessian_certain_unproved(m)
    # Sparse and wide: the exact elimination proves it.
    assert not _builder_hessian_certain_unproved(_qp(N, "fast"))
    # An expression objective is never predicted.
    assert not _builder_hessian_certain_unproved(_qp(N, "pure"))


@pytest.mark.smoke
def test_deadline_is_checked_after_extraction(monkeypatch):
    import time

    import discopt._relax.problem_classifier as pc

    real = pc.extract_qp_data

    def slow(model):
        out = real(model)
        time.sleep(0.2)
        return out

    m = _qp(200, "pure")
    assert certify_quadratic_objective_convex(m) is True
    monkeypatch.setattr(pc, "extract_qp_data", slow)
    assert certify_quadratic_objective_convex(m, deadline=time.perf_counter() + 0.1) is False


@pytest.mark.slow
def test_wide_dense_block_within_cap_is_still_proved():
    m = _dense_pd_builder_qp(_QP_EXACT_CONVEXITY_MAX_N + 100, n_dense=1500)
    assert certify_quadratic_objective_convex(m) is True


@pytest.mark.slow
def test_predicted_verdict_equals_full_proof(monkeypatch):
    """With the prediction disabled the full route reaches the same False."""
    import discopt._relax.convexity.certificate as cert

    m = _dense_pd_builder_qp(_QP_EXACT_CONVEXITY_MAX_N + 100)
    assert cert._builder_hessian_certain_unproved(m)
    monkeypatch.setattr(cert, "_builder_hessian_certain_unproved", lambda _m: False)
    assert certify_quadratic_objective_convex(m) is False
