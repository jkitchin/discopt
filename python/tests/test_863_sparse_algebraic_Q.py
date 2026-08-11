"""#863: the ALGEBRAIC extractor's Hessian may be sparse, and must agree exactly.

``c525f519`` gated the dense ``Q`` in ``_extract_qp_data_from_repr`` (the numeric
probe extractor) only. ``_extract_quadratic_coefficients`` — the DAG-walking
extractor behind ``extract_qp_data_algebraic``, ``extract_qcp_data_algebraic`` and
``_extract_qcp_constraints_algebraic``'s per-row Hessians — still opened with
``np.zeros((n, n))``. On ``watercontamination0202`` (106,711 variables) that is 91 GB,
which macOS *allows*, because ``np.zeros`` maps zero pages lazily; it does not raise,
it just makes the first full read ruinous. A single ``Q @ x`` against that array
(holding 4,017 nonzeros) was measured at **16.0 s**.

These tests are named for the extractor they cover, because the repr and algebraic
extractors contain near-identical blocks and a previous patch on this issue landed in
the wrong one. Each test asserts which function produced the value, and every
forced-sparse arm asserts ``sp.issparse``: without that a raising sparse branch makes
the dispatcher fall through to a dense extractor and the comparison silently becomes
dense-against-dense, which is how an earlier "parity confirmed" result on the repr Q
turned out to be worthless.
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
from discopt._relax.problem_classifier import (  # noqa: E402
    _extract_quadratic_coefficients,
    _extract_quadratic_terms,
    _materialise_Q,
    dense_Q,
    extract_qcp_data_algebraic,
    extract_qp_data_algebraic,
)


@pytest.fixture
def q_budget(monkeypatch):
    def _set(nbytes):
        monkeypatch.setattr(pc, "_QP_DENSE_Q_MAX_BYTES", nbytes)

    return _set


def _qp_model(n: int, support: int = 6):
    """``n`` variables, objective touching only the first ``support`` — the shape that
    breaks a dense Hessian.

    Every quadratic term is written in a form the ALGEBRAIC walk accepts: ``x ** 2``
    on a *bare* variable, ``var * var``, and ``(const * var) * var``, which are the
    three ``_qadd`` branches. (``(x - 1) ** 2`` is deliberately avoided: the walk
    refuses a power whose base is not a bare variable reference, so a model built
    that way silently never reaches this extractor — it falls through to the repr
    probe path, which is the *other* extractor and already covered by
    test_863_sparse_qpdata.py.)
    """
    m = dm.Model(f"algqp{n}")
    xs = [m.continuous(f"x{i}", lb=0.0, ub=10.0) for i in range(n)]
    m.minimize(
        sum(xs[i] ** 2 for i in range(support))
        + 3.0 * xs[0] * xs[1]
        + (2.5 * xs[2]) * xs[3]
        - 4.0 * xs[4] * xs[4]
        - 7.0 * xs[0]
    )
    m.subject_to(sum(xs) >= 1)
    m.subject_to(xs[0] + 2.0 * xs[1] == 3.0)
    return m


def _qcp_model(n: int, support: int = 5):
    """Quadratic OBJECTIVE and a quadratic ROW — the row Hessians are the other
    dense-(n, n) site inside the algebraic path."""
    m = dm.Model(f"algqcp{n}")
    xs = [m.continuous(f"x{i}", lb=0.0, ub=5.0) for i in range(n)]
    m.minimize(sum(xs[i] * xs[i] for i in range(support)) + 2.0 * xs[0] * xs[1])
    m.subject_to(xs[0] * xs[1] + xs[2] * xs[2] <= 6.0)
    m.subject_to(sum(xs) >= 1)
    return m


# --------------------------------------------------------------------------
# _extract_quadratic_coefficients (the algebraic walk itself)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [40, 90])
def test_extract_quadratic_coefficients_sparse_equals_dense(q_budget, n):
    """The whole safety argument for the ALGEBRAIC walk: flipping the representation
    must not move a single Hessian entry."""
    m = _qp_model(n)
    obj = m._objective.expression

    q_budget(10**12)
    Q_dense, c_dense, k_dense = _extract_quadratic_coefficients(obj, m, n)
    assert not sp.issparse(Q_dense), "a 10^12-byte budget should have stayed dense"

    q_budget(1)
    Q_sparse, c_sparse, k_sparse = _extract_quadratic_coefficients(obj, m, n)
    assert sp.issparse(Q_sparse), "a 1-byte budget should have forced a sparse Q"

    assert np.array_equal(Q_dense, dense_Q(Q_sparse))
    assert np.array_equal(c_dense, c_sparse)
    assert k_dense == k_sparse

    # ...and the values are actually right, not merely equal to each other.
    assert Q_dense[0, 1] == pytest.approx(3.0)
    assert Q_dense[1, 0] == pytest.approx(3.0)
    assert Q_dense[2, 3] == pytest.approx(2.5)
    assert Q_dense[3, 2] == pytest.approx(2.5)
    assert Q_dense[4, 4] == pytest.approx(2.0 - 8.0)  # x4**2 gives 2, -4*x4*x4 gives -8
    assert Q_dense[5, 5] == pytest.approx(2.0)
    assert c_dense[0] == pytest.approx(-7.0)


def test_extract_quadratic_terms_matches_the_materialised_matrix():
    """The sparse accumulator and the materialised matrix are the same object seen
    two ways — that equivalence is what lets a LINEAR QCP row skip materialising."""
    m = _qp_model(30)
    obj = m._objective.expression
    terms, c_terms, k_terms = _extract_quadratic_terms(obj, m, 30)
    Q, c, k = _extract_quadratic_coefficients(obj, m, 30)
    assert np.array_equal(dense_Q(_materialise_Q(terms, 30)), dense_Q(Q))
    assert np.array_equal(c_terms, c)
    assert k_terms == k
    # accumulated sparsely, so only the touched cells exist
    assert len(terms) <= 12, f"accumulator holds {len(terms)} cells for a 6-var objective"


def test_a_negative_hessian_slot_refuses_instead_of_wrapping(monkeypatch):
    """The dense predecessor got bounds-checking free from numpy, and for a NEGATIVE
    index silently wrote the WRONG cell via wraparound. The dict accumulator must
    refuse so the dispatcher falls through as it did on the old IndexError."""
    m = _qp_model(10)
    obj = m._objective.expression
    monkeypatch.setattr(pc, "_compute_var_offset", lambda var, model: -5)
    with pytest.raises(pc._NotQuadraticError, match="outside the model"):
        _extract_quadratic_coefficients(obj, m, 10)


def test_materialise_Q_of_no_terms_is_an_all_zero_matrix(q_budget):
    q_budget(1)
    Q = _materialise_Q({}, 7)
    assert sp.issparse(Q)
    assert np.array_equal(dense_Q(Q), np.zeros((7, 7)))
    q_budget(10**12)
    assert np.array_equal(_materialise_Q({}, 7), np.zeros((7, 7)))


# --------------------------------------------------------------------------
# extract_qp_data_algebraic  (NOT _extract_qp_data_from_repr)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [40, 90])
def test_extract_qp_data_algebraic_sparse_equals_dense(q_budget, n):
    q_budget(10**12)
    dense = extract_qp_data_algebraic(_qp_model(n))
    assert not sp.issparse(dense.Q)

    q_budget(1)
    sparse = extract_qp_data_algebraic(_qp_model(n))
    assert sp.issparse(sparse.Q), (
        "extract_qp_data_algebraic did not emit a sparse Q under a 1-byte budget; "
        "without this the comparison below is dense-against-dense and proves nothing"
    )

    assert np.array_equal(dense_Q(dense.Q), dense_Q(sparse.Q))
    assert np.array_equal(np.asarray(dense.c), np.asarray(sparse.c))
    assert dense.obj_const == sparse.obj_const


def test_extract_qp_data_algebraic_keeps_slack_padding_sparse(q_budget):
    """The model has an inequality, so slacks are appended and Q is block-padded.
    Padding must not densify — that would defeat the whole point."""
    q_budget(1)
    data = extract_qp_data_algebraic(_qp_model(50))
    assert sp.issparse(data.Q)
    n_total = np.asarray(data.c).shape[0]
    assert data.Q.shape == (n_total, n_total)
    assert n_total > 50, "expected at least one slack column"
    Q = dense_Q(data.Q)
    assert np.array_equal(Q[50:, :], np.zeros((n_total - 50, n_total)))
    assert np.array_equal(Q[:, 50:], np.zeros((n_total, n_total - 50)))


def test_extract_qp_data_algebraic_maximize_negates_sparse_Q(q_budget):
    """Sense handling is ``Q_full = -Q_full``; on scipy sparse that must still be the
    exact negation of the dense result."""
    m = _qp_model(60)
    m._objective.sense = type(m._objective.sense).MAXIMIZE
    q_budget(10**12)
    dense = extract_qp_data_algebraic(m)
    q_budget(1)
    sparse = extract_qp_data_algebraic(m)
    assert sp.issparse(sparse.Q)
    assert np.array_equal(dense_Q(dense.Q), dense_Q(sparse.Q))
    assert dense_Q(sparse.Q)[0, 1] == pytest.approx(-3.0)


# --------------------------------------------------------------------------
# extract_qcp_data_algebraic + per-row Hessians
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [40, 90])
def test_extract_qcp_data_algebraic_sparse_equals_dense(q_budget, n):
    q_budget(10**12)
    dense = extract_qcp_data_algebraic(_qcp_model(n))
    assert not sp.issparse(dense.Q)

    q_budget(1)
    sparse = extract_qcp_data_algebraic(_qcp_model(n))
    assert sp.issparse(sparse.Q), "extract_qcp_data_algebraic did not emit a sparse Q"

    assert np.array_equal(dense_Q(dense.Q), dense_Q(sparse.Q))
    assert len(dense.quadratic_constraints) == len(sparse.quadratic_constraints) == 1
    for rd, rs in zip(dense.quadratic_constraints, sparse.quadratic_constraints):
        assert sp.issparse(rs.Q), "a quadratic ROW's Hessian must sparsify too"
        assert np.array_equal(dense_Q(rd.Q), dense_Q(rs.Q))
        assert np.array_equal(np.asarray(rd.c), np.asarray(rs.c))
        assert rd.sense == rs.sense
        assert rd.rhs == pytest.approx(rs.rhs)


def test_linear_qcp_rows_are_still_classified_linear(q_budget):
    """``_quadratic_terms_nonempty`` replaces ``_quadratic_row_has_terms`` on the
    pre-materialisation accumulator; the linear/quadratic split must not move."""
    for budget in (10**12, 1):
        q_budget(budget)
        data = extract_qcp_data_algebraic(_qcp_model(45))
        # one quadratic row, one linear (sum(xs) >= 1) row
        assert len(data.quadratic_constraints) == 1
        assert np.asarray(data.b_ub).shape[0] == 1


def test_small_models_stay_dense_on_the_algebraic_path():
    """Default budget: ordinary small models keep the pre-#863 dense Hessian, so
    nothing about existing behaviour changes."""
    data = extract_qp_data_algebraic(_qp_model(30))
    assert not sp.issparse(data.Q)
    qcp = extract_qcp_data_algebraic(_qcp_model(30))
    assert not sp.issparse(qcp.Q)
    assert not sp.issparse(qcp.quadratic_constraints[0].Q)


# --------------------------------------------------------------------------
# consumers
# --------------------------------------------------------------------------


def test_quadratic_row_feasibility_check_handles_a_sparse_row_Q(q_budget):
    """``_quadratic_rows_solution_feasible`` used ``np.asarray(row.Q)``, which on a
    sparse matrix yields a 0-d object array rather than raising — the check would
    have become meaningless instead of failing."""
    from discopt.solver import _quadratic_rows_solution_feasible

    q_budget(10**12)
    dense_rows = extract_qcp_data_algebraic(_qcp_model(40)).quadratic_constraints
    q_budget(1)
    sparse_rows = extract_qcp_data_algebraic(_qcp_model(40)).quadratic_constraints
    assert sp.issparse(sparse_rows[0].Q)

    for x in (np.zeros(40), np.full(40, 0.5), np.full(40, 3.0)):
        assert _quadratic_rows_solution_feasible(
            x, dense_rows
        ) == _quadratic_rows_solution_feasible(x, sparse_rows)
    # and the verdicts are not all-True (otherwise the parity above is vacuous)
    assert _quadratic_rows_solution_feasible(np.zeros(40), sparse_rows)
    assert not _quadratic_rows_solution_feasible(np.full(40, 3.0), sparse_rows)


def test_convexity_pattern_analysis_declines_a_sparse_Q(q_budget):
    """``patterns._quadratic_data`` is unavoidably dense (np.diag, masks, eigvalsh).
    On a sparse Q it must decline — ``None`` is the conservative answer every caller
    already handles — rather than densify 91 GB or smuggle an object array."""
    from discopt._relax.convexity.patterns import _quadratic_data, quadratic_curvature

    m = _qp_model(40)
    obj = m._objective.expression
    q_budget(10**12)
    assert _quadratic_data(obj, m) is not None
    assert quadratic_curvature(obj, m) is not None
    q_budget(1)
    assert _quadratic_data(obj, m) is None
    assert quadratic_curvature(obj, m) is None
