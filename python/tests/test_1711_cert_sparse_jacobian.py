"""#1711: the convex-NLP certificate must not build a dense ``(m, n)`` Jacobian.

A hanging-chain NLP with 100k links was OS-killed at 32 GB inside
``_convex_nlp_certificate``: ``_cert_jacobian`` scattered the tape's sparse
Jacobian into a dense array, the certificate copied it several times, and
``_lagrangian_slope``'s dense exact-sum refinement took a ``rows x unsure`` block
of it. Above ``_MAX_DENSE_JACOBIAN_ELEMS`` the certificate now works in CSR.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest

pytestmark = pytest.mark.smoke


def _chain(n):
    L = 1.5 / n
    m = dm.Model("chain")
    x = m.continuous("x", shape=(n + 1,), lb=-10, ub=10)
    y = m.continuous("y", shape=(n + 1,), lb=-10, ub=10)
    m.subject_to([x[0] == 0, y[0] == 0, x[n] == 1, y[n] == 0], name="ends")
    m.subject_to((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2 <= L**2, name="link")
    m.minimize(0.5 * dm.sum(y[1:] + y[:-1]))
    return m


def _tape_evaluator(model):
    from discopt._tape_nlp_evaluator import TapeNLPEvaluator

    return TapeNLPEvaluator(model)


def test_cert_jacobian_is_csr_above_threshold_and_exact():
    import scipy.sparse as sp
    from discopt.solver import _MAX_DENSE_JACOBIAN_ELEMS, _cert_jacobian

    ev = _tape_evaluator(_chain(800))
    m, n = ev.n_constraints, ev.n_variables
    assert m * n > _MAX_DENSE_JACOBIAN_ELEMS
    x = np.random.default_rng(0).uniform(-1, 1, n)
    J = _cert_jacobian(ev, x, m, n)
    assert sp.issparse(J) and J.format == "csr" and J.shape == (m, n)
    dense = np.asarray(ev.evaluate_jacobian(x), dtype=np.float64).reshape(m, n)
    assert np.array_equal(J.toarray(), dense)


def test_cert_jacobian_stays_dense_below_threshold():
    from discopt.solver import _MAX_DENSE_JACOBIAN_ELEMS, _cert_jacobian

    ev = _tape_evaluator(_chain(20))
    m, n = ev.n_constraints, ev.n_variables
    assert m * n <= _MAX_DENSE_JACOBIAN_ELEMS
    J = _cert_jacobian(ev, np.zeros(n), m, n)
    assert isinstance(J, np.ndarray) and J.shape == (m, n)


def test_cert_jacobian_ignores_a_non_matrix_sparse_return():
    """A COO tuple (CUTEst's ``(vals, rows, cols)``) or a dense fallback from
    ``evaluate_sparse_jacobian`` must not be reshaped as if it were the matrix."""
    from discopt.solver import _MAX_DENSE_JACOBIAN_ELEMS, _cert_jacobian

    m, n = 1100, 1000
    assert m * n > _MAX_DENSE_JACOBIAN_ELEMS
    dense = np.zeros((m, n))
    dense[np.arange(m) % n, np.arange(m) % n] = 2.0

    class _Ev:
        def evaluate_sparse_jacobian(self, x):
            r, c = np.nonzero(dense)
            return dense[r, c], r, c

        def evaluate_jacobian(self, x):
            return dense

    J = _cert_jacobian(_Ev(), np.zeros(n), m, n)
    assert isinstance(J, np.ndarray) and np.array_equal(J, dense)


def test_chain_nlp_certificate_never_takes_the_dense_jacobian(monkeypatch):
    from discopt._tape_nlp_evaluator import TapeNLPEvaluator

    calls = []
    real = TapeNLPEvaluator.evaluate_jacobian

    def spy(self, x):
        calls.append((self.n_constraints, self.n_variables))
        return real(self, x)

    monkeypatch.setattr(TapeNLPEvaluator, "evaluate_jacobian", spy)
    r = _chain(1500).solve()
    assert r.status == "optimal" and r.gap_certified
    assert r.bound <= r.objective
    assert calls == [], calls


def test_chain_nlp_peak_memory_is_sparse_sized():
    import tracemalloc

    _chain(5).solve()  # first-solve imports and compiles are not the measurement
    m = _chain(1500)
    tracemalloc.start()
    try:
        r = m.solve()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert r.status == "optimal" and r.gap_certified
    # The dense Jacobian alone is 1504 * 3002 * 8 = 36 MB; the solve peaked at
    # 148.9 MB before #1711 and 4.8 MB after.
    assert peak < 30e6, peak / 1e6
