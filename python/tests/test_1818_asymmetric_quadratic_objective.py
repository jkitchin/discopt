"""#1818: ``add_quadratic_objective`` read only ``triu(Q)``.

The objective is documented as ``0.5 x'Qx``, and ``x'Qx`` is defined for any
``Q``. The Rust builder read only the upper triangle, so an upper-only ``Q``
doubled its cross terms and a lower-only ``Q`` lost them (the objective was
wiped to its linear part). ``Q`` is now stored as its symmetric part, which has
the same ``x'Qx``; a symmetric ``Q`` is stored unchanged.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt._relax.convexity.certificate import certify_quadratic_objective_convex
from discopt._relax.problem_classifier import extract_qp_data

pytestmark = pytest.mark.smoke


def _model(Q, sense="minimize"):
    n = Q.shape[0]
    m = dm.Model("q")
    x = m.continuous("x", shape=(n,), lb=-1, ub=1)
    m.add_quadratic_objective(Q, np.zeros(n), x, sense=sense)
    return m


def _rust_objective(m, pt):
    from discopt._rust import model_to_repr

    return model_to_repr(m, m._builder).evaluate_objective(np.asarray(pt, dtype=np.float64))


def _shapes(rng, n=6):
    M = rng.standard_normal((n, n))
    return {
        "upper": np.triu(M),
        "lower": np.tril(M),
        "asym": M,
        "sym": M + M.T,
    }


@pytest.mark.parametrize("kind", ["upper", "lower", "asym", "sym"])
@pytest.mark.parametrize("fmt", ["dense", "csr", "coo"])
def test_objective_is_half_xQx(kind, fmt):
    rng = np.random.default_rng(7)
    Q = _shapes(rng)[kind]
    Qin = {"dense": Q, "csr": sp.csr_matrix(Q), "coo": sp.coo_matrix(Q)}[fmt]
    m = _model(Qin)
    checks = 0
    for _ in range(5):
        p = rng.uniform(-1, 1, Q.shape[0])
        want = 0.5 * p @ Q @ p
        assert _rust_objective(m, p) == pytest.approx(want, rel=1e-12, abs=1e-12)
        checks += 1
    assert checks == 5


def test_issue_examples():
    upper = np.array([[0.0, 2.0], [0.0, 0.0]])
    lower = upper.T.copy()
    p = np.array([0.7, -0.3])
    for Q in (upper, lower):
        assert _rust_objective(_model(Q), p) == pytest.approx(p[0] * p[1], rel=1e-15)
        H = extract_qp_data(_model(Q)).Q
        H = H.toarray() if sp.issparse(H) else np.asarray(H)
        np.testing.assert_array_equal(H[:2, :2], [[0.0, 1.0], [1.0, 0.0]])


def test_symmetric_q_is_stored_bit_identically():
    rng = np.random.default_rng(3)
    M = rng.standard_normal((5, 5))
    Q = sp.csr_matrix(M + M.T)
    m = _model(Q)
    stored = m._builder_quadratic_objective[0]
    np.testing.assert_array_equal(stored.toarray(), Q.toarray())


@pytest.mark.parametrize("sense", ["minimize", "maximize"])
def test_certificate_matches_symmetrized(sense):
    """Same verdict as the symmetrized Q, for a convex and a nonconvex case."""
    # Upper-only spelling of the PSD matrix [[2, 1], [1, 2]]: x'Qx = 2x0^2 + 2x0x1 + 2x1^2.
    convex_upper = np.array([[2.0, 2.0], [0.0, 2.0]])
    # Lower-only spelling of the indefinite [[1, 3], [3, 1]].
    indefinite_lower = np.array([[1.0, 0.0], [6.0, 1.0]])
    for Q in (convex_upper, indefinite_lower):
        S = 0.5 * Q + 0.5 * Q.T
        assert certify_quadratic_objective_convex(
            _model(Q, sense)
        ) == certify_quadratic_objective_convex(_model(S, sense))
    assert certify_quadratic_objective_convex(_model(convex_upper)) is True
    assert certify_quadratic_objective_convex(_model(indefinite_lower)) is False
