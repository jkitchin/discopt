"""#863: ``A_eq`` / ``A_ub`` may be sparse, and must agree exactly with the dense form.

``QPData.Q`` became sparse-capable in c525f519. The constraint matrix is an
equal-sized wall: on ``watercontamination0202`` (106,711 variables / 107,209 rows)
the dense ``Q`` would have been 91 GB and the dense ``A_eq`` is **86 GiB**. Both
paths built it as ``np.stack`` of dense full-width rows, which additionally needs
every row resident at once.

The extractors now accumulate COO triples — a row is reduced to its nonzeros as soon
as it is produced — and materialise dense while ``(m, n)`` float64 fits
``_DENSE_A_MAX_BYTES`` (256 MB), scipy CSR beyond it. Measured on that instance:
``extract_qp_data`` now returns in **15.0 s** with ``A_eq`` CSR, nnz = 208,240.

Consumers must never call ``np.asarray`` on these fields directly: on a sparse matrix
that does **not** raise, it returns a 0-d object array wrapping the matrix, which
would silently feed garbage into a solver. Everything goes through ``dense_A()``,
which raises loudly if it ever produces an object array or a non-2-D result.

The forced-sparse arm of every parity test asserts ``scipy.sparse.issparse`` on the
result. Without that assertion the test proves nothing: if the sparse branch raised,
the dispatcher would silently fall through to a dense extractor and the "parity"
comparison would be dense-against-dense. That exact mistake wasted a measurement
round while c525f519 was being built.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt._relax.problem_classifier as pc  # noqa: E402
import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from discopt._relax.problem_classifier import (  # noqa: E402
    ProblemClass,
    _extract_linear_coefficients,
    _extract_linear_coefficients_sparse,
    _NotLinearError,
    classify_problem,
    dense_A,
    extract_lp_data,
    extract_qcp_data,
    extract_qp_data,
)
from discopt.modeling.core import Constraint, from_nl  # noqa: E402

NL_DIR = Path(__file__).parent / "data" / "minlplib_nl"


# --------------------------------------------------------------------------- #
# A corpus of model shapes that between them cover every branch of the COO
# assembly: equalities, both inequality senses (slack +1 / -1), the row ordering
# (equalities first), maximize, vector variables, matmul rows, repeated variable
# references in one row, and negative / fractional coefficients.
# --------------------------------------------------------------------------- #
def _lp_eq_only():
    m = dm.Model("eq_only")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.minimize(x + 2.0 * y)
    m.subject_to(x + y == 3.0)
    m.subject_to(2.0 * x - 0.5 * y == 1.0)
    return m


def _lp_both_inequality_senses():
    m = dm.Model("both_senses")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    z = m.continuous("z", lb=0.0, ub=10.0)
    m.minimize(x + y + z)
    m.subject_to(x + 2.0 * y <= 8.0)  # slack +1
    m.subject_to(y - 3.0 * z >= -4.0)  # slack -1
    m.subject_to(x + z == 2.0)  # equality: must come FIRST in the row order
    return m


def _lp_repeated_reference():
    """``x + 2*x`` must accumulate to 3, not overwrite: the dict accumulator has to
    reproduce the dense ``c[i] += scale`` exactly."""
    m = dm.Model("repeated")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.minimize(x + y)
    m.subject_to(x + 2.0 * x - 0.25 * y <= 5.0)
    return m


def _lp_vector_variable():
    m = dm.Model("vector")
    v = m.continuous("v", shape=(4,), lb=0.0, ub=5.0)
    w = m.continuous("w", lb=0.0, ub=5.0)
    m.minimize(sum(v) + w)
    m.subject_to(sum(v) + w >= 2.0)
    m.subject_to(v[0] - v[3] == 0.5)
    return m


def _lp_matmul():
    m = dm.Model("matmul")
    v = m.continuous("v", shape=(3,), lb=0.0, ub=5.0)
    m.minimize(sum(v))
    m.subject_to(np.array([1.0, -2.0, 0.5]) @ v <= 4.0)
    return m


def _lp_maximize():
    m = dm.Model("maximize")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.maximize(3.0 * x + y)
    m.subject_to(x + y <= 4.0)
    m.subject_to(x - y >= -1.0)
    return m


def _lp_no_constraints():
    m = dm.Model("bounds_only")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    m.minimize(x)
    return m


def _milp():
    m = dm.Model("milp")
    b = [m.binary(f"b{i}") for i in range(4)]
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.minimize(x - sum(b))
    m.subject_to(sum(b) <= 2)
    m.subject_to(x - 3.0 * b[0] <= 0.0)
    return m


def _qp():
    m = dm.Model("qp")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.minimize((x - 1.0) ** 2 + 2.0 * (y - 2.0) ** 2 + 3.0 * x * y)
    m.subject_to(x + y <= 6.0)
    m.subject_to(x - y == 0.5)
    return m


def _miqp():
    m = dm.Model("miqp")
    b = m.binary("b")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.minimize((x - 2.0) ** 2 + 4.0 * b)
    m.subject_to(x - 5.0 * b <= 0.0)
    return m


LP_MODELS = {
    "eq_only": _lp_eq_only,
    "both_senses": _lp_both_inequality_senses,
    "repeated_reference": _lp_repeated_reference,
    "vector_variable": _lp_vector_variable,
    "matmul": _lp_matmul,
    "maximize": _lp_maximize,
    "no_constraints": _lp_no_constraints,
    "milp": _milp,
}
QP_MODELS = {"qp": _qp, "miqp": _miqp}

# Real in-repo corpus instances that classify onto the QP/MIQP extraction path.
NL_MIQP = [
    "alan",
    "gbd",
    "nvs15",
    "st_miqp1",
    "st_miqp2",
    "st_miqp3",
    "st_miqp4",
    "st_miqp5",
    "st_test1",
    "st_testgr3",
]


@pytest.fixture
def a_budget(monkeypatch):
    """Force the dense/sparse decision. Read as a module global at call time."""

    def _set(nbytes):
        monkeypatch.setattr(pc, "_DENSE_A_MAX_BYTES", nbytes)

    return _set


def _both_arms(a_budget, extract, build):
    """Extract with the budget forced dense, then forced sparse.

    Returns ``(dense_fields, sparse_fields)`` as dicts of densified matrices. Asserts
    the sparse arm really is sparse -- otherwise the comparison is worthless.
    """
    a_budget(10**15)
    d = extract(build())
    a_budget(1)
    s = extract(build())

    names = [n for n in ("A_eq", "A_ub") if hasattr(d, n)]
    any_sparse = False
    out_d, out_s = {}, {}
    for name in names:
        raw_d, raw_s = getattr(d, name), getattr(s, name)
        assert not sp.issparse(raw_d), f"{name}: a 10^15-byte budget should stay dense"
        # A matrix with zero rows is returned as the (0, n) empty array in both arms.
        if raw_s.shape[0] > 0:
            assert sp.issparse(raw_s), (
                f"{name}: a 1-byte budget should have forced a sparse matrix, got "
                f"{type(raw_s).__name__} -- the sparse branch probably raised and the "
                "dispatcher fell through to a dense extractor"
            )
            any_sparse = True
        out_d[name] = dense_A(raw_d)
        out_s[name] = dense_A(raw_s)
    assert any_sparse, "no matrix in this model was sparsified; the test proves nothing"
    return d, s, out_d, out_s


def _assert_parity(d, s, out_d, out_s, label):
    for name in out_d:
        assert out_d[name].shape == out_s[name].shape, f"{label}/{name}: shape differs"
        assert np.array_equal(out_d[name], out_s[name]), (
            f"{label}/{name}: dense and sparse differ, max |diff| = "
            f"{np.abs(out_d[name] - out_s[name]).max()}"
        )
    for name in ("c", "b_eq", "b_ub", "x_l", "x_u"):
        if hasattr(d, name):
            assert np.array_equal(np.asarray(getattr(d, name)), np.asarray(getattr(s, name))), (
                f"{label}/{name}: the sparse arm perturbed a vector field"
            )
    assert d.obj_const == s.obj_const, f"{label}: obj_const differs"


@pytest.mark.parametrize("name", sorted(set(LP_MODELS) - {"no_constraints"}))
def test_lp_dense_and_sparse_A_eq_agree_exactly(a_budget, name):
    d, s, out_d, out_s = _both_arms(a_budget, extract_lp_data, LP_MODELS[name])
    _assert_parity(d, s, out_d, out_s, name)


@pytest.mark.parametrize("name", sorted(QP_MODELS))
def test_qp_dense_and_sparse_A_eq_agree_exactly(a_budget, name):
    d, s, out_d, out_s = _both_arms(a_budget, extract_qp_data, QP_MODELS[name])
    _assert_parity(d, s, out_d, out_s, name)


@pytest.mark.parametrize("stem", NL_MIQP)
def test_real_corpus_instances_dense_and_sparse_agree_exactly(a_budget, stem):
    """The same parity over real MINLPLib instances, not synthetic shapes."""
    path = NL_DIR / f"{stem}.nl"
    if not path.exists():  # pragma: no cover - corpus is checked in
        pytest.skip(f"{path} missing")
    d, s, out_d, out_s = _both_arms(a_budget, extract_qp_data, lambda p=path: from_nl(str(p)))
    _assert_parity(d, s, out_d, out_s, stem)


def test_qcp_dense_and_sparse_A_ub_and_A_eq_agree_exactly(a_budget):
    """``QCPData`` carries two linear matrices and no slacks; both must round-trip."""
    path = NL_DIR / "dispatch.nl"
    if not path.exists():  # pragma: no cover - corpus is checked in
        pytest.skip("dispatch.nl missing")
    assert classify_problem(from_nl(str(path))) in (
        ProblemClass.QCP,
        ProblemClass.QCQP,
        ProblemClass.MIQCP,
        ProblemClass.MIQCQP,
    )
    d, s, out_d, out_s = _both_arms(a_budget, extract_qcp_data, lambda p=path: from_nl(str(p)))
    _assert_parity(d, s, out_d, out_s, "dispatch")


def test_row_order_and_slack_signs_are_preserved(a_budget):
    """The riskiest part of the COO rewrite: rows are ``[equalities..., inequalities]``
    and each inequality gets exactly one slack, +1 for ``<=`` and -1 for ``>=``.
    Pinned against an explicit expectation, in BOTH representations."""
    # The comparison operators normalise every inequality to ``body <= 0``, so a
    # ``>=`` written in the API arrives as ``<=`` on the negated body. The ``>=``
    # branch (slack -1) is reached from producers that build Constraints directly --
    # gdp_reformulate emits `sense=">="` in eight places -- so append one that way.
    model = _lp_both_inequality_senses()
    x, y, z = model._variables
    model._constraints.append(Constraint(body=x - 2.0 * z + 1.0, sense=">=", rhs=0.0))

    expected = np.array(
        [
            # x     y     z  | s0   s1   s2
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # x + z == 2       (equality, FIRST)
            [1.0, 2.0, 0.0, 1.0, 0.0, 0.0],  # x + 2y <= 8      (slack +1)
            [0.0, -1.0, 3.0, 0.0, 1.0, 0.0],  # -(−4 −(y−3z)) <= 0 (slack +1)
            [1.0, 0.0, -2.0, 0.0, 0.0, -1.0],  # x - 2z + 1 >= 0  (slack -1)
        ]
    )
    for nbytes in (10**15, 1):
        a_budget(nbytes)
        data = extract_lp_data(model)
        if nbytes == 1:
            assert sp.issparse(data.A_eq), "1-byte budget did not force a sparse A_eq"
        assert np.array_equal(dense_A(data.A_eq), expected)
        assert np.array_equal(np.asarray(data.b_eq), np.array([2.0, 8.0, 4.0, -1.0]))


def test_a_model_with_no_constraints_stays_an_empty_dense_matrix(a_budget):
    """A (0, n) matrix has nothing to sparsify; both arms must return the same empty
    dense array rather than an empty CSR that consumers would have to special-case."""
    for nbytes in (10**15, 1):
        a_budget(nbytes)
        a = extract_lp_data(_lp_no_constraints()).A_eq
        assert not sp.issparse(a)
        assert a.shape == (0, 1)


def test_small_models_stay_dense_under_the_default_budget():
    """Ordinary small models must keep the pre-#863 dense matrix, so nothing about
    existing behaviour changes."""
    for name, build in {**LP_MODELS, **QP_MODELS}.items():
        extract = extract_qp_data if name in QP_MODELS else extract_lp_data
        a = extract(build()).A_eq
        assert not sp.issparse(a), f"{name}: a small model should not have been sparsified"


def test_a_wide_model_is_sparsified_under_the_DEFAULT_budget():
    """No monkeypatching: a model whose (m, n) genuinely exceeds
    ``_DENSE_A_MAX_BYTES`` must come back sparse on its own.

    5,000 variables with 5,000 inequality rows is (5000, 10000) float64 = 381 MB
    against the 256 MB budget. Verified against the pre-fix revision, where this was
    an unconditional ``np.stack`` of 5,000 dense full-width rows:

        before   A_eq type=ndarray     issparse=False   -> FAIL
        after    A_eq type=csr_matrix  issparse=True    nnz=15000
    """
    n = 5000
    m = dm.Model("wide")
    xs = [m.continuous(f"x{i}", lb=0.0, ub=10.0) for i in range(n)]
    m.minimize(xs[0])
    for i in range(n):
        m.subject_to(xs[i] + xs[(i + 1) % n] <= 3.0)

    data = extract_lp_data(m)
    assert sp.issparse(data.A_eq), (
        f"(5000, 10000) float64 is 381 MB, above the "
        f"{pc._DENSE_A_MAX_BYTES / 2**20:.0f} MB budget, but A_eq came back "
        f"{type(data.A_eq).__name__}"
    )
    assert data.A_eq.shape == (n, 2 * n)
    # 2 structural coefficients + 1 slack per row
    assert data.A_eq.nnz == 3 * n
    # spot-check a row against the dense expectation
    row = dense_A(data.A_eq)[7]
    assert row[7] == 1.0 and row[8] == 1.0 and row[n + 7] == 1.0
    assert np.count_nonzero(row) == 3


# --------------------------------------------------------------------------- #
# dense_A itself
# --------------------------------------------------------------------------- #
def test_dense_A_rejects_a_smuggled_object_array():
    """``np.asarray`` on a sparse matrix yields a 0-d object array rather than
    raising. ``dense_A`` must refuse that instead of passing it to a solver — this is
    the failure mode the helper exists to prevent."""
    smuggled = np.asarray(sp.csr_matrix((3, 3)))  # 0-d object array, no exception
    assert smuggled.dtype == object
    with pytest.raises(TypeError, match="dense_A"):
        dense_A(smuggled)


def test_dense_A_rejects_a_non_2d_result():
    with pytest.raises(TypeError, match="dense_A"):
        dense_A(np.zeros(5))


def test_dense_A_round_trips_both_representations():
    a = np.array([[1.0, 0.0, -2.5], [0.0, 3.0, 0.0]])
    out = dense_A(a)
    assert isinstance(out, np.ndarray) and out.ndim == 2 and out.dtype == np.float64
    assert np.array_equal(out, dense_A(sp.csr_matrix(a)))


# --------------------------------------------------------------------------- #
# the sparse coefficient walk
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(LP_MODELS))
def test_sparse_walk_matches_the_dense_walk_bit_for_bit(name):
    """``_extract_linear_coefficients`` is now a wrapper over the sparse walk. Its
    output must be bit-identical to densifying the sparse walk's dict."""
    model = LP_MODELS[name]()
    n = sum(v.size for v in model._variables)
    for con in (c for c in model._constraints if isinstance(c, Constraint)):
        dense, const_d = _extract_linear_coefficients(con.body, model, n)
        terms, const_s = _extract_linear_coefficients_sparse(con.body, model, n)
        rebuilt = np.zeros(n, dtype=np.float64)
        for i, v in terms.items():
            rebuilt[i] = v
        assert np.array_equal(dense, rebuilt)
        assert const_d == const_s
        # and the dict holds no entry the dense form does not
        assert set(terms) <= set(np.nonzero(dense)[0].tolist()) | {
            i for i, v in terms.items() if v == 0.0
        }


def test_sparse_walk_refuses_an_out_of_range_variable_slot():
    """The dense walk got a bounds check for free from numpy indexing (and for a
    NEGATIVE index silently wrote the wrong slot via numpy wraparound). The dict has
    no such check, so it must reject explicitly rather than emit a column that does
    not exist — a silently misplaced coefficient is a wrong-answer bug."""
    donor = dm.Model("donor")
    for i in range(6):
        donor.continuous(f"d{i}", lb=0.0, ub=1.0)
    stranger = donor._variables[5]

    host = dm.Model("host")
    host.continuous("h", lb=0.0, ub=1.0)
    with pytest.raises(_NotLinearError, match="outside the model"):
        _extract_linear_coefficients_sparse(stranger + 0.0, host, 1)
