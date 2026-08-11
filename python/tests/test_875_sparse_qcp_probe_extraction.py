"""#875: the QCP *probe* extractor's Hessian may be sparse, and must agree exactly.

``_extract_quadratic_coefficients_from_values`` is the numeric-probe counterpart of
``_extract_qp_data_from_repr`` and the sibling of the algebraic walk covered by
``test_863_sparse_algebraic_Q.py``. #863 made both of those support-restricted and
sparse; this one was left dense because ``watercontamination0202`` does not route
through it, and #868 declined to widen speculatively. It is the same shape, and it is
worse: ``_extract_qcp_data_from_repr`` calls it **once per constraint**, so the dense
``(n, n)`` and the all-pairs sweep are paid per row.

Two independent economies, each tested here:

* **support restriction** — a variable absent from the evaluator has
  ``f(e_j) == f(-e_j) == d``, so every product involving it is identically zero. The
  ``O(n)`` diagonal probes already identify the support, so the pair sweep drops from
  ``O(n^2)`` to ``O(|support|^2)`` for free.
* **sparse materialisation** — through ``_materialise_Q``, dense while ``(n, n)``
  float64 fits the budget and scipy CSR beyond it.

Every forced-sparse arm asserts ``sp.issparse``: without that, a sparse branch that
quietly failed would leave the comparison dense-against-dense and prove nothing — the
trap recorded in ``test_863_sparse_algebraic_Q.py``'s header.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt._relax.problem_classifier as pc  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from discopt._relax.problem_classifier import (  # noqa: E402
    _extract_quadratic_coefficients_from_values,
    _quadratic_row_has_terms,
    dense_Q,
)


@pytest.fixture
def q_budget(monkeypatch):
    def _set(nbytes):
        monkeypatch.setattr(pc, "_QP_DENSE_Q_MAX_BYTES", nbytes)

    return _set


def _quadratic_evaluator(n: int, support: int = 5):
    """``0.5 x'Qx + c'x + d`` touching only the first ``support`` variables.

    Returned as a plain callable so the test exercises the probe arithmetic rather
    than the model plumbing; ``probes`` counts evaluator calls.
    """
    rng = np.random.default_rng(7)
    Q = np.zeros((n, n), dtype=np.float64)
    block = rng.standard_normal((support, support))
    Q[:support, :support] = block + block.T  # symmetric
    c = np.zeros(n, dtype=np.float64)
    c[:support] = rng.standard_normal(support)
    d = 1.25
    probes = [0]

    def evaluate(x):
        probes[0] += 1
        x = np.asarray(x, dtype=np.float64)
        return 0.5 * float(x @ (Q @ x)) + float(c @ x) + d

    return evaluate, probes, Q, c, d


@pytest.mark.parametrize("n", [30, 60])
def test_probe_extraction_recovers_the_quadratic(q_budget, n):
    q_budget(10**12)
    evaluate, _probes, Q, c, d = _quadratic_evaluator(n)
    Q_out, c_out, d_out = _extract_quadratic_coefficients_from_values(evaluate, n)
    assert not sp.issparse(Q_out), "a 10^12-byte budget should have stayed dense"
    assert d_out == pytest.approx(d)
    assert np.allclose(dense_Q(Q_out), Q, atol=1e-9)
    assert np.allclose(np.asarray(c_out), c, atol=1e-9)


@pytest.mark.parametrize("n", [30, 60])
def test_forced_sparse_arm_equals_the_dense_arm(q_budget, n):
    """The whole safety argument: flipping the representation must not move an entry."""
    q_budget(10**12)
    evaluate, _p, _Q, _c, _d = _quadratic_evaluator(n)
    Q_dense, c_dense, d_dense = _extract_quadratic_coefficients_from_values(evaluate, n)
    assert not sp.issparse(Q_dense)

    q_budget(1)
    evaluate2, _p2, _Q2, _c2, _d2 = _quadratic_evaluator(n)
    Q_sparse, c_sparse, d_sparse = _extract_quadratic_coefficients_from_values(evaluate2, n)
    assert sp.issparse(Q_sparse), "a 1-byte budget should have forced a sparse Q"

    assert np.array_equal(dense_Q(Q_dense), dense_Q(Q_sparse))
    assert np.array_equal(np.asarray(c_dense), np.asarray(c_sparse))
    assert d_dense == d_sparse


def test_off_diagonal_probing_is_restricted_to_the_support():
    """Probe COUNT, not wall clock: the all-pairs sweep is ``n(n-1)/2`` off-diagonal
    probes; restricted to a support of ``s`` it is ``s(s-1)/2``. This is the test that
    fails before the fix (n=120, support=5: 7,140 probes vs 10)."""
    n, support = 120, 5
    evaluate, probes, _Q, _c, _d = _quadratic_evaluator(n, support=support)
    _extract_quadratic_coefficients_from_values(evaluate, n)

    fixed = 1 + 2 * n  # f(0) plus f(e_j) / f(-e_j) for every j
    off_diagonal = probes[0] - fixed
    all_pairs = n * (n - 1) // 2
    support_pairs = support * (support - 1) // 2
    assert off_diagonal <= support_pairs, (
        f"off-diagonal probing is not support-restricted: {off_diagonal} probes "
        f"(support allows {support_pairs}, all pairs would be {all_pairs})"
    )


def test_a_row_with_no_quadratic_terms_is_still_seen_as_linear(q_budget):
    """``_quadratic_row_has_terms`` decides the linear/quadratic split for every QCP
    row. It used ``np.any(np.abs(Q) > tol)``, which does not mean what it looks like
    on a scipy sparse matrix — the split must not move when Q sparsifies."""
    n = 40

    def linear(x):
        return float(np.asarray(x, dtype=np.float64)[:3].sum()) + 2.0

    for budget in (10**12, 1):
        q_budget(budget)
        Q, c, d = _extract_quadratic_coefficients_from_values(linear, n)
        assert _quadratic_row_has_terms(Q) is False
        assert d == pytest.approx(2.0)
        assert np.allclose(np.asarray(c)[:3], 1.0)
        assert np.allclose(np.asarray(c)[3:], 0.0)

    q_budget(1)
    evaluate, _p, _Q, _c, _d = _quadratic_evaluator(n)
    Q_q, _c_q, _d_q = _extract_quadratic_coefficients_from_values(evaluate, n)
    assert sp.issparse(Q_q)
    assert _quadratic_row_has_terms(Q_q) is True


def test_quadratic_row_has_terms_respects_its_tolerance_when_sparse(q_budget):
    """A stored value below ``tol`` must not count, sparse or dense — otherwise the
    sparse arm would classify a numerically-linear row as quadratic."""
    tiny = sp.csr_matrix(([1e-15, -1e-15], ([0, 1], [1, 0])), shape=(3, 3))
    assert _quadratic_row_has_terms(tiny) is False
    real = sp.csr_matrix(([2.0], ([0], [0])), shape=(3, 3))
    assert _quadratic_row_has_terms(real) is True
    assert _quadratic_row_has_terms(np.zeros((3, 3))) is False
