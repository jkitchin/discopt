"""Issue #1230 V3: the producer carries the row senses instead of the consumer
re-deriving them from the marshaled matrix.

``extract_lp_data`` turns every constraint into an equality with a logical (slack)
column, and ``solver._decompose_eq_slack_form`` projects back to
``A_ub x <= b_ub`` / ``A_eq x = b_eq``. It used to recover which rows were
inequalities by testing a slack coefficient against ``1e-15`` and taking its sign --
re-deriving a structural fact the producer knew exactly and discarded. The producer
now records it on ``LPData.row_sense`` (from ``std_form.logical_block``, the code
that chose the slack layout), and the consumer reads it.

This is a **bound-neutral** change in CLAUDE.md §5's sense, so it is verified as one:
the declared-sense projection must be *exactly* -- not approximately -- what the
inference produced, on both the dense and the sparse path, under both logical
layouts. Any drift means the change is wrong.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from discopt import Model
from discopt._relax.problem_classifier import extract_lp_data
from discopt.solver import _decompose_eq_slack_form

pytestmark = pytest.mark.unit


def _only_le():
    m = Model()
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.subject_to(x + y <= 4)
    m.subject_to(2 * x - y <= 3)
    m.minimize(-x - 2 * y)
    return m


def _only_ge():
    m = Model()
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.subject_to(x + y >= 2)
    m.subject_to(x - 3 * y >= -8)
    m.minimize(x + y)
    return m


def _only_eq():
    m = Model()
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.subject_to(x + y == 4)
    m.subject_to(x - y == 1)
    m.minimize(x + 2 * y)
    return m


def _mixed():
    m = Model()
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    z = m.continuous("z", lb=-5, ub=5)
    m.subject_to(x + y + z == 4)
    m.subject_to(x - y <= 3)
    m.subject_to(2 * x + z >= -1)
    m.subject_to(y + 3 * z <= 12)
    m.minimize(-x - 2 * y + z)
    return m


def _wide():
    """Enough rows that the sparse path is worth exercising, all three senses."""
    m = Model()
    xs = [m.continuous(f"x{i}", lb=0, ub=10) for i in range(12)]
    for i in range(0, 12, 3):
        m.subject_to(xs[i] + xs[i + 1] <= 5)
        m.subject_to(xs[i + 1] - xs[i + 2] >= -4)
        m.subject_to(xs[i] + xs[i + 2] == 3)
    m.minimize(sum(xs))
    return m


MODELS = [_only_le, _only_ge, _only_eq, _mixed, _wide]


def _same(a, b, what):
    """Exact identity, including which side is ``None``."""
    if a is None or b is None:
        assert a is None and b is None, f"{what}: one side is None ({a is None} vs {b is None})"
        return
    a_d = a.toarray() if sp.issparse(a) else np.asarray(a)
    b_d = b.toarray() if sp.issparse(b) else np.asarray(b)
    assert a_d.shape == b_d.shape, f"{what}: shape {a_d.shape} vs {b_d.shape}"
    assert np.array_equal(a_d, b_d), f"{what}: values differ (max |d| {np.abs(a_d - b_d).max()})"


@pytest.mark.parametrize("build", MODELS, ids=[f.__name__ for f in MODELS])
@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("row_logicals", ["0", "1"], ids=["legacy-layout", "row-logicals"])
def test_declared_senses_reproduce_the_inferred_projection_exactly(
    build, sparse, row_logicals, monkeypatch
):
    monkeypatch.setenv("DISCOPT_LP_ROW_LOGICALS", row_logicals)
    lp = extract_lp_data(build())
    assert lp.row_sense is not None, "the extractor must carry the senses"

    A = lp.A_eq.toarray() if sp.issparse(lp.A_eq) else np.asarray(lp.A_eq, dtype=np.float64)
    m_rows, n_total = A.shape
    assert lp.row_sense.shape == (m_rows,)
    # Structural width: total columns minus the logical block this layout produced.
    n_slack = int(np.count_nonzero(lp.row_sense != 0.0)) if row_logicals == "0" else m_rows
    n_orig = n_total - n_slack
    assert n_orig > 0

    b = np.asarray(lp.b_eq, dtype=np.float64)
    x_u = np.asarray(lp.x_u, dtype=np.float64)
    A_in = sp.csr_matrix(A) if sparse else A

    inferred = _decompose_eq_slack_form(A_in, b, n_orig, n_slack, x_u)
    declared = _decompose_eq_slack_form(A_in, b, n_orig, n_slack, x_u, row_sense=lp.row_sense)

    for name, got, want in zip(("A_ub", "b_ub", "A_eq", "b_eq"), declared, inferred):
        _same(got, want, f"{build.__name__}/{'sparse' if sparse else 'dense'}/{name}")

    # The comparison must not have been vacuous: every model here has rows, and the
    # projection must have produced at least one of the two blocks.
    assert m_rows > 0
    assert declared[0] is not None or declared[2] is not None


def test_a_row_count_mismatch_is_an_error_not_a_silent_fallback():
    """Rows changed after extraction -> the declared senses no longer describe them.
    Inferring quietly would hide exactly the drift ``row_sense`` removes."""
    lp = extract_lp_data(_mixed())
    A = lp.A_eq.toarray() if sp.issparse(lp.A_eq) else np.asarray(lp.A_eq, dtype=np.float64)
    m_rows, n_total = A.shape
    n_slack = int(np.count_nonzero(lp.row_sense != 0.0))
    b = np.asarray(lp.b_eq, dtype=np.float64)

    for A_in in (A, sp.csr_matrix(A)):
        with pytest.raises(ValueError, match="row_sense"):
            _decompose_eq_slack_form(
                A_in, b, n_total - n_slack, n_slack, None, row_sense=lp.row_sense[:-1]
            )


def test_every_extractor_rung_records_the_senses():
    """Not just the rung this machine happens to reach: each extractor that builds a
    logical block must put the senses on its ``LPData``."""
    from discopt._relax import problem_classifier as pc

    checked = 0
    for name in (
        "extract_lp_data_algebraic",
        "_extract_lp_data_from_repr",
        "_extract_lp_data_tape",
    ):
        fn = getattr(pc, name, None)
        if fn is None:
            continue
        try:
            data = fn(_mixed())
        except Exception:  # this rung cannot represent the model; not its turn
            continue
        if data is None:
            continue
        assert data.row_sense is not None, f"{name} dropped the senses"
        assert data.row_sense.shape[0] == np.asarray(data.b_eq).shape[0]
        checked += 1
    assert checked > 0, "no extractor rung ran — the test measured nothing"
