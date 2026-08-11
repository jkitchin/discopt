"""#863: QPData.Q may be sparse, and must agree exactly with the dense form.

A wide model with a narrow objective cannot hold a dense ``(n, n)`` ``Q``:
``watercontamination0202`` is 106,711 variables whose objective touches 101, and
its dense Q would be **91 GB**. The extractor therefore emits a scipy sparse ``Q``
above ``_QP_DENSE_Q_MAX_BYTES`` and a dense one below it, so existing small-model
behaviour is bit-identical.

Consumers must never call ``np.asarray`` on ``Q`` directly: on a sparse matrix that
does **not** raise, it returns a 0-d object array wrapping the matrix, which would
silently feed garbage into a solver. Everything goes through ``dense_Q()``, which
raises loudly if it ever produces an object array or a non-2-D result.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt._relax.problem_classifier as pc  # noqa: E402
import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from discopt._relax.problem_classifier import dense_Q, extract_qp_data  # noqa: E402


def _model(n: int, support: int = 6):
    """``n`` variables; objective touches only the first ``support``, including a
    genuine off-diagonal cross term so the test covers more than the diagonal."""
    m = dm.Model(f"wide{n}")
    xs = [m.continuous(f"x{i}", lb=0.0, ub=10.0) for i in range(n)]
    m.minimize(sum((xs[i] - 1.0) ** 2 for i in range(support)) + 3.0 * xs[0] * xs[1])
    m.subject_to(sum(xs) >= 1)
    return m


@pytest.fixture
def q_budget(monkeypatch):
    def _set(nbytes):
        monkeypatch.setattr(pc, "_QP_DENSE_Q_MAX_BYTES", nbytes)

    return _set


@pytest.mark.parametrize("n", [60, 120])
def test_sparse_and_dense_paths_agree_exactly(q_budget, n):
    """The whole safety argument: flipping the representation must not change a
    single entry of Q."""
    q_budget(10**12)  # force dense
    q_dense = dense_Q(extract_qp_data(_model(n)).Q)
    q_budget(1)  # force sparse
    q_sparse_data = extract_qp_data(_model(n)).Q
    assert sp.issparse(q_sparse_data), "budget of 1 byte should have forced a sparse Q"
    q_sparse = dense_Q(q_sparse_data)

    assert np.array_equal(q_dense, q_sparse), (
        f"sparse and dense Q differ (max |diff| = {np.abs(q_dense - q_sparse).max()})"
    )
    # and the values are actually right: 2 on the squared terms, 3 on the cross term
    assert q_dense[0, 1] == pytest.approx(3.0)
    assert q_dense[1, 0] == pytest.approx(3.0)
    assert q_dense[2, 2] == pytest.approx(2.0)


def test_small_models_stay_dense(q_budget):
    """Default budget: an ordinary small QP must keep the pre-#863 dense Q, so
    nothing about existing behaviour changes."""
    q = extract_qp_data(_model(30)).Q
    assert not sp.issparse(q), "a small model should not have been sparsified"


def test_dense_Q_rejects_a_smuggled_object_array():
    """``np.asarray`` on a sparse matrix yields a 0-d object array rather than
    raising. ``dense_Q`` must refuse that instead of passing it to a solver — this
    is the failure mode the helper exists to prevent."""
    smuggled = np.asarray(sp.csr_matrix((3, 3)))  # 0-d object array, no exception
    assert smuggled.dtype == object
    with pytest.raises(TypeError, match="dense_Q"):
        dense_Q(smuggled)


def test_dense_Q_round_trips_both_representations():
    m = _model(40)
    q = dense_Q(extract_qp_data(m).Q)
    assert isinstance(q, np.ndarray) and q.ndim == 2 and q.dtype == np.float64
    assert np.array_equal(q, dense_Q(sp.csr_matrix(q)))
