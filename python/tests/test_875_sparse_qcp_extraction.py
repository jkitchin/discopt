"""#875: the QCP row extractor's Hessian may be sparse, and must agree exactly.

``_quadratic_coefficients`` recovers ``(Q, c, d)`` for one QCP row -- the objective
or one quadratic constraint. ``_extract_qcp_data_from_repr`` calls it **once per
constraint**, so whatever it costs is paid per row, and whatever it allocates is
allocated per row.

It used to be a numeric probe (``_extract_quadratic_coefficients_from_values``):
``O(n)`` evaluations to find the support, then one evaluation per support pair to
recover an off-diagonal. #875 made that sweep support-restricted and its ``Q``
sparse, which is what this file originally tested. The probe is now **deleted** --
the coefficients are read off the expression arena, exactly, in time proportional
to the DAG -- so the tests below assert the stronger property that replaced the
economy: the extractor does not evaluate the model *at all*.

What survives unchanged is the sparse/dense parity discipline, because the sparse
materialisation did not go away. Every forced-sparse arm asserts ``sp.issparse``:
without that, a sparse branch that quietly failed would leave the comparison
dense-against-dense and prove nothing -- the trap recorded in
``test_863_sparse_algebraic_Q.py``'s header, and CLAUDE.md §6.

The last section covers the arm that *replaced* the probe as the fallback when the
arena walk declines: one AD-tape Hessian for the row. It is exact, so unlike the
probe it has no accuracy caveat -- but it has a convention caveat, which is why it
is tested against the walk rather than only against hand-computed numbers. The
walk returns the full coefficient of ``x_i x_j`` and its consumer doubles the
diagonal; a Hessian is *already* doubled there. Getting that wrong is a factor of
two on every diagonal entry, silently.
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
    _quadratic_coefficients,
    _quadratic_row_has_terms,
    _tape_quadratic_coefficients,
    dense_Q,
)


@pytest.fixture
def q_budget(monkeypatch):
    def _set(nbytes):
        monkeypatch.setattr(pc, "_QP_DENSE_Q_MAX_BYTES", nbytes)

    return _set


def _qcp_model(n: int, support: int = 5):
    """``n`` variables; the objective and one quadratic row touch only the first
    ``support`` of them -- the wide-model/narrow-row shape #875 is about.

    Returns ``(model, Q_obj, c_obj, d_obj)`` with the objective's coefficients in
    the ``0.5 x'Qx + c'x + d`` convention every consumer here uses, so a test can
    compare against them without re-deriving the factor of two.
    """
    m = dm.Model(f"qcp{n}")
    xs = [m.continuous(f"x{i}", lb=-5.0, ub=5.0) for i in range(n)]

    # 0.5 * (2*x0^2 + 4*x1^2) + 3*x0*x1 + (x0 - 2*x1) + 1.25
    m.minimize(
        xs[0] * xs[0] + 2.0 * xs[1] * xs[1] + 3.0 * xs[0] * xs[1] + xs[0] - 2.0 * xs[1] + 1.25
    )
    m.subject_to(xs[2] * xs[2] + xs[3] * xs[4] <= 9.0)
    m.subject_to(sum(xs) >= 1.0)

    Q = np.zeros((n, n), dtype=np.float64)
    Q[0, 0] = 2.0
    Q[1, 1] = 4.0
    Q[0, 1] = Q[1, 0] = 3.0
    c = np.zeros(n, dtype=np.float64)
    c[0], c[1] = 1.0, -2.0
    return m, Q, c, 1.25


def _repr_of(model):
    from discopt._rust import model_to_repr

    return model_to_repr(model, getattr(model, "_builder", None))


class _CountingRepr:
    """Forwards to a real repr, counting every numeric evaluation of the model.

    The point of the change is that there are none. A wrapper rather than a
    monkeypatched counter because the extractor may reach the repr by more than
    one method name, and an uncounted route would read as a pass.
    """

    _EVAL = ("evaluate_objective", "evaluate_constraint", "evaluate_constraints")

    def __init__(self, inner):
        self._inner = inner
        self.evals = 0

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if name in self._EVAL:

            def _counted(*a, **k):
                self.evals += 1
                return attr(*a, **k)

            return _counted
        return attr


# --------------------------------------------------------------------------
# the arena walk: exact, sparse-safe, and it never evaluates the model
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [30, 60])
def test_row_extraction_recovers_the_quadratic(q_budget, n):
    q_budget(10**12)
    model, Q, c, d = _qcp_model(n)
    Q_out, c_out, d_out = _quadratic_coefficients(_repr_of(model), n, None)
    assert not sp.issparse(Q_out), "a 10^12-byte budget should have stayed dense"
    assert d_out == pytest.approx(d)
    assert np.allclose(dense_Q(Q_out), Q, atol=1e-9)
    assert np.allclose(np.asarray(c_out), c, atol=1e-9)


@pytest.mark.parametrize("n", [30, 60])
def test_forced_sparse_arm_equals_the_dense_arm(q_budget, n):
    """The whole safety argument: flipping the representation must not move an entry."""
    model, _Q, _c, _d = _qcp_model(n)

    q_budget(10**12)
    Q_dense, c_dense, d_dense = _quadratic_coefficients(_repr_of(model), n, None)
    assert not sp.issparse(Q_dense)

    q_budget(1)
    Q_sparse, c_sparse, d_sparse = _quadratic_coefficients(_repr_of(model), n, None)
    assert sp.issparse(Q_sparse), "a 1-byte budget should have forced a sparse Q"

    assert np.array_equal(dense_Q(Q_dense), dense_Q(Q_sparse))
    assert np.array_equal(np.asarray(c_dense), np.asarray(c_sparse))
    assert d_dense == d_sparse


def test_extraction_does_not_evaluate_the_model():
    """The property that superseded the support restriction.

    #875 reduced the off-diagonal sweep from ``n(n-1)/2`` evaluations to
    ``s(s-1)/2`` over the support (n=120, s=5: 7,140 -> 10). Reading the arena
    takes it to **zero**, for the objective and for every constraint row, so this
    asserts a count rather than a bound -- a support-restricted probe would fail it
    just as an all-pairs probe would.
    """
    n = 120
    model, _Q, _c, _d = _qcp_model(n, support=5)
    counting = _CountingRepr(_repr_of(model))

    _quadratic_coefficients(counting, n, None)
    for i in range(counting.n_constraints):
        _quadratic_coefficients(counting, n, i)

    assert counting.evals == 0, (
        f"the extractor evaluated the model {counting.evals} times; the arena walk "
        "reads coefficients and must never probe"
    )


def test_a_row_with_no_quadratic_terms_is_still_seen_as_linear(q_budget):
    """``_quadratic_row_has_terms`` decides the linear/quadratic split for every QCP
    row. It used ``np.any(np.abs(Q) > tol)``, which does not mean what it looks like
    on a scipy sparse matrix -- the split must not move when Q sparsifies."""
    n = 40
    model, _Q, _c, _d = _qcp_model(n)
    repr_ = _repr_of(model)
    # Row 1 of the model is ``sum(xs) >= 1``: linear. Row 0 is quadratic.
    for budget in (10**12, 1):
        q_budget(budget)
        Q, c, d = _quadratic_coefficients(repr_, n, 1)
        assert _quadratic_row_has_terms(Q) is False
        # The repr normalises ``sum(xs) >= 1`` to a body of ``-sum(xs)``, so the
        # coefficients are -1 -- the sign lives in ``constraint_sense``/``_rhs``,
        # which this helper does not see and does not apply.
        assert np.allclose(np.asarray(c), -1.0)

    q_budget(1)
    Q_q, _c_q, _d_q = _quadratic_coefficients(repr_, n, 0)
    assert sp.issparse(Q_q)
    assert _quadratic_row_has_terms(Q_q) is True


def test_quadratic_row_has_terms_respects_its_tolerance_when_sparse(q_budget):
    """A stored value below ``tol`` must not count, sparse or dense -- otherwise the
    sparse arm would classify a numerically-linear row as quadratic."""
    tiny = sp.csr_matrix(([1e-15, -1e-15], ([0, 1], [1, 0])), shape=(3, 3))
    assert _quadratic_row_has_terms(tiny) is False
    real = sp.csr_matrix(([2.0], ([0], [0])), shape=(3, 3))
    assert _quadratic_row_has_terms(real) is True
    assert _quadratic_row_has_terms(np.zeros((3, 3))) is False


# --------------------------------------------------------------------------
# the fallback that replaced the probe: one AD-tape Hessian per row
# --------------------------------------------------------------------------


def _tape_of(model):
    from discopt._tape_nlp_evaluator import try_build

    tape = try_build(model)
    assert tape is not None, "the tape declined a plain quadratic model"
    return tape


def test_tape_fallback_matches_the_walk_on_the_objective():
    """Both arms are exact, so this is an equality test, not a tolerance test --
    and it is the one that catches the factor of two. ``_quadratic_form_to_
    coefficients`` doubles the walk's diagonal to reach the ``0.5 x'Qx``
    convention; a Hessian arrives already doubled, so doubling it again would
    show up here as ``Q[0,0] == 4`` against the walk's 2."""
    n = 12
    model, Q_true, c_true, d_true = _qcp_model(n)
    Q_walk, c_walk, d_walk = _quadratic_coefficients(_repr_of(model), n, None)
    Q_tape, c_tape, d_tape = _tape_quadratic_coefficients(_tape_of(model), n, None)

    assert np.allclose(dense_Q(Q_tape), Q_true, atol=1e-9)
    assert np.allclose(dense_Q(Q_tape), dense_Q(Q_walk), atol=1e-9)
    assert np.allclose(np.asarray(c_tape), np.asarray(c_walk), atol=1e-9)
    assert d_tape == pytest.approx(d_walk)


def test_tape_fallback_matches_the_walk_on_a_constraint_row():
    """``evaluate_lagrangian_hessian(x, 0.0, e_i)`` is the only way to isolate one
    constraint's curvature. A nonzero ``obj_factor`` would fold the objective's
    Hessian into every row, which this model would show as a nonzero ``Q[0,0]``
    on a row that does not mention ``x0``."""
    n = 12
    model, _Q, _c, _d = _qcp_model(n)
    repr_ = _repr_of(model)
    Q_walk, c_walk, d_walk = _quadratic_coefficients(repr_, n, 0)
    Q_tape, c_tape, d_tape = _tape_quadratic_coefficients(_tape_of(model), n, 0)

    assert np.allclose(dense_Q(Q_tape), dense_Q(Q_walk), atol=1e-9)
    assert np.allclose(np.asarray(c_tape), np.asarray(c_walk), atol=1e-9)
    assert d_tape == pytest.approx(d_walk)
    assert dense_Q(Q_tape)[0, 0] == 0.0, (
        "the objective's Hessian leaked into a constraint row -- obj_factor is not 0"
    )


def test_tape_fallback_returns_the_objective_in_the_models_own_sense():
    """The tape negates a MAXIMIZE objective internally. Every caller of this
    helper applies its own negation, so returning the tape's already-flipped sign
    hands a maximisation to a minimiser -- the failure ``_qp_terms_tape``
    documents. Compared against the walk, which is always in model sense."""
    n = 8
    m = dm.Model("maxq")
    xs = [m.continuous(f"x{i}", lb=-5.0, ub=5.0) for i in range(n)]
    m.maximize(-(xs[0] * xs[0]) - 2.0 * xs[1] * xs[1] + 3.0 * xs[0])
    m.subject_to(sum(xs) <= 1.0)

    Q_walk, c_walk, d_walk = _quadratic_coefficients(_repr_of(m), n, None)
    Q_tape, c_tape, d_tape = _tape_quadratic_coefficients(_tape_of(m), n, None, maximize=True)

    assert np.allclose(dense_Q(Q_tape), dense_Q(Q_walk), atol=1e-9), (
        f"sense mismatch: tape Q diag {np.diag(dense_Q(Q_tape))[:2]} vs "
        f"walk {np.diag(dense_Q(Q_walk))[:2]}"
    )
    assert np.allclose(np.asarray(c_tape), np.asarray(c_walk), atol=1e-9)
    assert d_tape == pytest.approx(d_walk)


def test_a_declined_row_with_no_tape_refuses_loudly():
    """The one thing the fallback must never do is guess. With no tape on offer, a
    declining walk raises rather than returning a zero Q -- a silent zero Q is
    exactly the #866 false optimum."""

    class _AlwaysDeclines:
        def objective_quadratic_form(self):
            return None

        def constraint_quadratic_form(self, i):
            return None

    with pytest.raises(pc._NotQuadraticError) as exc:
        _quadratic_coefficients(_AlwaysDeclines(), 4, None, tape_factory=None)
    assert "no AD tape" in str(exc.value), str(exc.value)

    with pytest.raises(pc._NotQuadraticError):
        _quadratic_coefficients(_AlwaysDeclines(), 4, 0, tape_factory=lambda: None)
