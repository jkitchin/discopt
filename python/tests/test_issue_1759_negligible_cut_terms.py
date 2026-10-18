"""#1759: a cut term too small to move its row must not set the row's scale.

OA on a big-M portfolio (optimization-book II.22) linearizes ``d_i x_i**2`` at
points with ``x_i ~ 1e-22``, so a cut carries ``2 d_i x_i ~ 1e-22`` next to the
``-1`` on its epigraph column. ``_prepare_cut_row`` let that entry choose the
power-of-two scale that lifts it over HiGHS's ``small_matrix_value``: the row was
scaled by ``2**44``, entries reached ``-1.76e13``, and HiGHS answered "Solve error"
on a master that plain HiGHS solves to optimality. OA filed each one as
``master_no_point_error`` and looped to the time limit (152 of 178 masters on the
reporter's machine).

A term whose largest magnitude over its finite column box is at most
``_NEGLIGIBLE_REL`` of the row's largest term is now dropped on its valid side
(rhs loosened by its worst case) instead.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("highspy")

from discopt.solvers import milp_highs  # noqa: E402
from discopt.solvers.milp_highs import (  # noqa: E402
    DROP_NEGLIGIBLE_ENV,
    _fit_rows_to_window,
    _prepare_cut_row,
)

SMALL, LARGE = 1e-9, 1e15


def _portfolio_cut():
    """The shape of the failing OA row: epigraph ``-1``, a 1e-22 gradient entry."""
    coeffs = np.array([0.0411, -0.1248, 1.7e-22, 2.3e-23, -0.0368, -1.0])
    lb = np.array([0.0, 0.0, 0.0, 0.0, -1.0, 0.0])
    ub = np.array([0.4, 0.4, 0.4, 0.4, 1.0, 1.0])
    return coeffs, -0.0023, lb, ub


def test_a_negligible_term_does_not_scale_the_row_toward_large_matrix_value(monkeypatch):
    monkeypatch.delenv(DROP_NEGLIGIBLE_ENV, raising=False)
    coeffs, rhs, lb, ub = _portfolio_cut()
    idx, vals, out_rhs = _prepare_cut_row(coeffs, rhs, lb, ub, SMALL, LARGE)

    # Before the fix: 2**44 * 1 = 1.76e13.
    assert float(np.max(np.abs(vals))) == 1.0, "row was rescaled for a term it cannot use"
    assert idx.tolist() == [0, 1, 4, 5], "the negligible terms reached HiGHS"
    np.testing.assert_array_equal(vals, coeffs[idx])
    # Both dropped terms are positive on non-negative columns: they drop for free.
    assert out_rhs == rhs


def test_a_negative_negligible_term_loosens_rhs_by_at_most_its_worst_case(monkeypatch):
    monkeypatch.delenv(DROP_NEGLIGIBLE_ENV, raising=False)
    coeffs, rhs, lb, ub = _portfolio_cut()
    coeffs[2] = -1.7e-22
    idx, vals, out_rhs = _prepare_cut_row(coeffs, rhs, lb, ub, SMALL, LARGE)
    assert 2 not in idx.tolist()
    assert out_rhs >= rhs, "a dropped negative term left the row tighter"
    assert out_rhs - rhs <= 1e-12, "the row was loosened by more than its negligible term"
    _assert_implied(coeffs, rhs, idx, vals, out_rhs, lb, ub)


def test_opt_out_restores_the_legacy_scale(monkeypatch):
    """Anti-vacuity and the ``=0`` opt-out: without the rule the row IS scaled."""
    monkeypatch.setenv(DROP_NEGLIGIBLE_ENV, "0")
    coeffs, rhs, lb, ub = _portfolio_cut()
    _idx, vals, _ = _prepare_cut_row(coeffs, rhs, lb, ub, SMALL, LARGE)
    assert float(np.max(np.abs(vals))) > 1e12


def test_a_term_on_an_open_column_is_still_rescued(monkeypatch):
    """No finite box, no bound on the term's effect: it keeps the #1066 rescue."""
    monkeypatch.delenv(DROP_NEGLIGIBLE_ENV, raising=False)
    coeffs, rhs, lb, ub = _portfolio_cut()
    ub[2] = np.inf
    idx, vals, _ = _prepare_cut_row(coeffs, rhs, lb, ub, SMALL, LARGE)
    assert 2 in idx.tolist()
    assert float(np.min(np.abs(vals))) > SMALL


def test_equality_rows_are_never_dropped_into(monkeypatch):
    """An equality has no valid side: it is scaled exactly, as before."""
    monkeypatch.delenv(DROP_NEGLIGIBLE_ENV, raising=False)
    a = sp.csr_matrix(np.array([[1.0, 1e-11]]))
    out_a, out_rhs = _fit_rows_to_window(
        a, np.array([2.0]), np.zeros(2), np.ones(2), SMALL, LARGE, equality=True
    )
    row = out_a.toarray().ravel()
    assert row[1] == 1e-11 * row[0] and out_rhs[0] == 2.0 * row[0]


def _assert_implied(coeffs, rhs, idx, vals, rhs_out, lb, ub, n_samples=600):
    """Every box point the original row keeps, the prepared row keeps (CLAUDE.md §6).

    Points are put ON the original row's boundary, where a tightening would show,
    and both rows are evaluated in exact rational arithmetic: the loosening under
    test (~1e-19) is far below float round-off of the activities (~1e-17).
    """
    from fractions import Fraction as F

    rng = np.random.default_rng(1759)
    x = rng.uniform(lb, ub, size=(n_samples, lb.size))
    x[:, 5] = np.clip(x[:, :5] @ coeffs[:5] - rhs, lb[5], ub[5])
    checked = 0
    for row in x:
        xf = [F(float(v)) for v in row]
        if sum(F(float(a)) * v for a, v in zip(coeffs, xf)) > F(float(rhs)):
            continue
        checked += 1
        assert sum(F(float(a)) * xf[j] for j, a in zip(idx, vals)) <= F(float(rhs_out))
    assert checked > n_samples // 10, f"only {checked} sample points were checked"


def _portfolio_model():
    import discopt.modeling as dm

    rng = np.random.default_rng(21)
    n_a = 10
    mu = np.round(rng.uniform(0.04, 0.14, n_a), 3)
    B = np.round(rng.normal(0.0, 1.0, (n_a, 2)) * [0.15, 0.08], 3)
    d = np.round(rng.uniform(0.15, 0.35, n_a) ** 2, 4)
    m = dm.Model("portfolio")
    x = m.continuous("x", shape=(n_a,), lb=0, ub=0.4)
    z = m.binary("z", shape=(n_a,))
    f = m.continuous("f", shape=(2,), lb=-1, ub=1)
    for k in range(2):
        m.subject_to(f[k] == dm.sum(lambda i: B[i, k] * x[i], over=range(n_a)))
    m.minimize(f[0] ** 2 + f[1] ** 2 + dm.sum(lambda i: d[i] * x[i] ** 2, over=range(n_a)))
    m.subject_to(dm.sum(lambda i: x[i], over=range(n_a)) == 1)
    m.subject_to(dm.sum(lambda i: mu[i] * x[i], over=range(n_a)) >= 0.10)
    m.subject_to(dm.sum(lambda i: z[i], over=range(n_a)) <= 3)
    for i in range(n_a):
        m.subject_to(x[i] <= 0.4 * z[i])
        m.subject_to(x[i] >= 0.1 * z[i])
    return m


def test_oa_on_the_portfolio_gets_a_point_from_every_highs_master(monkeypatch, caplog):
    """End to end: the II.22 model certifies with no ``master_no_point_error``.

    Before the fix the scaled masters errored (1 of 33 here, 152 of 178 on the
    reporter's machine); the probe also counts master solves so it cannot pass
    by running none.
    """
    monkeypatch.delenv(DROP_NEGLIGIBLE_ENV, raising=False)
    solves = {"n": 0, "no_point": 0}
    real = milp_highs.solve_milp

    def spy(*a, **k):
        r = real(*a, **k)
        solves["n"] += 1
        solves["no_point"] += r.x is None
        return r

    monkeypatch.setattr(milp_highs, "solve_milp", spy)
    with caplog.at_level(logging.INFO, logger="discopt"):
        r = _portfolio_model().solve(
            solver="mip-nlp", milp_solver="highs", mip_nlp_method="oa", time_limit=120
        )
    assert solves["n"] >= 10, f"only {solves['n']} HiGHS master solves ran"
    assert solves["no_point"] == 0
    assert not [m for m in caplog.messages if "master_no_point" in m]
    assert r.status == "optimal"
    assert r.objective == pytest.approx(0.0161695584, rel=1e-6)
    assert r.bound <= r.objective + 1e-9


# ---------------------------------------------------------------------------
# #1759, part 2: a primary HiGHS "Solve error" is recovered, not looped on.
# OA reads a master with no point as "add cuts at the box midpoint"; on a repeat
# error that adds the same cuts every iteration, so one error became a loop to
# the time limit. The reporter's masters solved with presolve off.


def _small_milp():
    return dict(
        c=np.array([-1.0, -2.0]),
        A_ub=np.array([[1.0, 1.0]]),
        b_ub=np.array([3.5]),
        A_eq=None,
        b_eq=None,
        bounds=[(0, 10), (0, 2)],
        integrality=np.array([1, 0]),
    )


def _erroring_primary(monkeypatch, retry_errors=False):
    from discopt.solvers import MILPResult, SolveStatus

    real = milp_highs._solve_milp_once
    calls = []

    def once(*a, presolve, **k):
        calls.append(presolve)
        if presolve or retry_errors:
            # What HiGHS returned on the II.22 masters: ERROR, no point, and a
            # stale finite dual bound.
            return MILPResult(status=SolveStatus.ERROR, x=None, objective=None, bound=0.0)
        return real(*a, presolve=presolve, **k)

    monkeypatch.setattr(milp_highs, "_solve_milp_once", once)
    return calls


@pytest.mark.parametrize("confirm_from", [None, -1e9])
def test_an_errored_primary_is_resolved_without_presolve(monkeypatch, confirm_from):
    from discopt.solvers import SolveStatus

    calls = _erroring_primary(monkeypatch)
    r = milp_highs.solve_milp(**_small_milp(), confirm_bound_from=confirm_from)
    assert calls == [True, False], calls
    assert r.x is not None, "the recovered master has no point -- OA would loop"
    assert r.objective == pytest.approx(-5.0)  # x0 = 1 (integer), x1 = 2
    # One HiGHS configuration alone never certifies (#1634): no bound claim.
    assert r.bound is None
    assert r.status == SolveStatus.ITERATION_LIMIT
    assert not milp_highs.is_unconfirmed(r)


def test_a_second_error_is_reported_as_the_error_it_is(monkeypatch):
    from discopt.solvers import SolveStatus

    calls = _erroring_primary(monkeypatch, retry_errors=True)
    r = milp_highs.solve_milp(**_small_milp())
    assert calls == [True, False]
    assert r.status == SolveStatus.ERROR and r.x is None


def test_oa_with_the_legacy_scaling_still_gets_a_point_from_every_master(monkeypatch):
    """The legacy row scaling (``=0``) produced the "Solve error" masters here."""
    monkeypatch.setenv(DROP_NEGLIGIBLE_ENV, "0")
    solves = {"n": 0, "no_point": 0}
    real = milp_highs.solve_milp

    def spy(*a, **k):
        r = real(*a, **k)
        solves["n"] += 1
        solves["no_point"] += r.x is None
        return r

    monkeypatch.setattr(milp_highs, "solve_milp", spy)
    r = _portfolio_model().solve(
        solver="mip-nlp", milp_solver="highs", mip_nlp_method="oa", time_limit=120
    )
    assert solves["n"] >= 10
    assert solves["no_point"] == 0
    assert r.status == "optimal"
    assert r.objective == pytest.approx(0.0161695584, rel=1e-6)
