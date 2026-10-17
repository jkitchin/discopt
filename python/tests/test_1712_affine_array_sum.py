"""#1712 follow-up: scalar-term passes read ``dm.sum(C * x)`` instead of refusing it.

Three passes walked ``C[i] * x[i] + ...`` term by term and refused the same sum
written vectorized, ``dm.sum(C * x)``, because its array ``Constant`` is not a
scalar: the interval Hessian (abstained), the sparse linear/quadratic extractors
(so the affine-square proof, and with it POUNCE's QP route, declined), and the
binary-multilinear expansion (aborted). The vectorized form is what the
``DISCOPT_SUM_CANONICAL`` fold emits, and what a user can write by hand, so each
test here builds it by hand and compares it with the term-by-term sum.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import SumOverExpression, affine_array_sum

pytestmark = [pytest.mark.smoke]


def _pair(n=40, seed=0, shape=None):
    """The same affine sum two ways: vectorized and term by term."""
    rng = np.random.default_rng(seed)
    shape = shape or (n,)
    m = dm.Model("pair")
    x = m.continuous("x", shape=shape, lb=-1.0, ub=2.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    C = rng.normal(size=shape)
    vec = dm.sum(C * x)
    idx = list(np.ndindex(*shape))
    terms = SumOverExpression([float(C[i]) * x[i if len(shape) > 1 else i[0]] for i in idx])
    return m, x, y, C, vec, terms


def test_recognizer():
    m, x, _, C, vec, _ = _pair(shape=(3, 4))
    var, coefs = affine_array_sum(vec)
    assert var is x and np.array_equal(coefs, C.ravel())
    var, coefs = affine_array_sum(dm.sum(x))
    assert var is x and np.array_equal(coefs, np.ones(12))
    assert affine_array_sum(dm.sum(x * C)) is not None
    # Broadcast, axis reductions and Parameters are not this pattern.
    assert affine_array_sum(dm.sum(np.ones((1, 4)) * x)) is None
    assert affine_array_sum(dm.sum(C * x, axis=0)) is None
    assert affine_array_sum(dm.sum(m.parameter("p", C) * x)) is None


def test_sparse_linear_extractor_reads_the_coefficients():
    from discopt._relax.problem_classifier import _extract_linear_coefficients_sparse

    m, _, _, _, vec, terms = _pair(shape=(5, 7))
    n = sum(v.size for v in m._variables)
    a, ca = _extract_linear_coefficients_sparse(2.0 * vec - 3.0, m, n)
    b, cb = _extract_linear_coefficients_sparse(2.0 * terms - 3.0, m, n)
    assert a == b and ca == cb
    assert len(a) == 35


def test_affine_square_proof_reaches_the_vectorized_rows():
    from discopt._relax.convexity.patterns import weighted_affine_square_decomposition

    rng = np.random.default_rng(1)
    K, b = rng.normal(size=(6, 40)), rng.normal(size=6)
    m = dm.Model("ls")
    x = m.continuous("x", shape=(40,), lb=-10, ub=10)
    expr = dm.sum([(dm.sum(K[i] * x) - b[i]) ** 2 for i in range(6)])
    dec = weighted_affine_square_decomposition(expr, m)
    assert dec is not None
    w, A, bb, _, _ = dec
    np.testing.assert_array_equal(A.toarray(), K)
    np.testing.assert_array_equal(bb, -b)
    assert np.all(w == 1.0)


def test_interval_hessian_walks_the_vectorized_sum():
    from discopt._relax.convexity import interval_ad as iad

    m, _, y, _, vec, terms = _pair(n=40)
    a = iad.interval_hessian(dm.sin(vec / 40.0) * y, m)
    b = iad.interval_hessian(dm.sin(terms / 40.0) * y, m)
    a_lo, a_hi = np.asarray(a.hess.lo), np.asarray(a.hess.hi)
    b_lo, b_hi = np.asarray(b.hess.lo), np.asarray(b.hess.hi)
    assert np.all(np.isfinite(a_lo)) and np.all(np.isfinite(a_hi))
    # Before: the vectorized form raised (array ``Constant`` leaf). Now it gets the
    # term walk's enclosure, up to the outward rounding of the coefficient products.
    np.testing.assert_allclose(a_lo, b_lo, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(a_hi, b_hi, rtol=1e-12, atol=1e-15)


def test_binary_multilinear_reads_the_vectorized_objective():
    from discopt._relax import binary_multilinear_reform as bml

    def build(vectorized: bool):
        m = dm.Model("bml")
        b = m.binary("b", shape=(12,))
        v = np.arange(1.0, 13.0)
        m.maximize(
            dm.sum(v * b) if vectorized else SumOverExpression([v[i] * b[i] for i in range(12)])
        )
        for i in range(0, 10, 3):
            m.subject_to(b[i] * b[i + 1] * b[i + 2] <= 0)
        return m

    outs = []
    for vectorized in (True, False):
        m = build(vectorized)
        out = bml.reformulate_binary_multilinear(m)
        assert out is not m, "the pass declined"
        outs.append(out.solve(time_limit=60))
    assert all(r.status == "optimal" for r in outs)
    assert outs[0].objective == pytest.approx(outs[1].objective, abs=1e-9)
