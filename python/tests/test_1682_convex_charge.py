"""#1682: "convex up to a charge" acceptance of a rank-deficient Gram QP objective.

Since #1679 the float Gram matrix ``2 K'K`` of a rank-deficient ``K`` is (correctly)
not proved PSD -- it is indefinite in exact arithmetic at ``lambda_min ~ -1e-14``.
By default (opt-out ``DISCOPT_CONVEX_CHARGE=0``) a solve may still route it to the convex path
when ``lambda_min >= -delta`` is PROVED (``rigorous_psd_shift``) and the charge
``delta/2 * D**2`` over the bounded box is tiny; the charge is then subtracted
from the published bound.
"""

from __future__ import annotations

from fractions import Fraction

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import Model
from discopt._relax.convexity.certificate import (
    certify_quadratic_objective_convex,
    convexity_charge_scope,
    quadratic_objective_charge,
    set_charge_scope_abs_gap_tol,
)
from discopt._relax.convexity.eigenvalue import rigorous_psd_shift
from scipy.optimize import lsq_linear


def _exact_psd_fraction(A: list[list[Fraction]]) -> bool:
    """Exact PSD test of a rational symmetric matrix (symmetric elimination)."""
    n = len(A)
    A = [row[:] for row in A]
    alive = list(range(n))
    while alive:
        p = next((i for i in alive if A[i][i] > 0), None)
        if p is None:
            # every remaining diagonal <= 0: PSD iff the remaining block is zero
            return all(A[i][j] == 0 for i in alive for j in alive)
        alive.remove(p)
        for i in alive:
            if A[i][p] == 0:
                continue
            f = A[i][p] / A[p][p]
            for j in alive:
                A[i][j] -= f * A[p][j]
        if any(A[i][i] < 0 for i in alive):
            return False
    return True


def _cases():
    rng = np.random.default_rng(1682)
    out = []
    for scale in (1e-6, 1.0, 1e6):
        for kind in ("gram_rank_def", "indefinite", "psd", "diag_neg"):
            n = 7
            if kind == "gram_rank_def":
                K = rng.normal(size=(9, n))
                K[:, -1] = K[:, 0] + K[:, 1]
                Q = 2.0 * (K.T @ K)
            elif kind == "indefinite":
                A = rng.normal(size=(n, n))
                Q = A + A.T
            elif kind == "psd":
                A = rng.normal(size=(n, n))
                Q = A @ A.T
            else:
                Q = np.diag(rng.normal(size=n))
            out.append((f"{kind}@{scale:g}", scale * Q))
    return out


@pytest.mark.parametrize("label,Q", _cases(), ids=[c[0] for c in _cases()])
def test_rigorous_shift_is_proved_in_exact_arithmetic(label, Q):
    """``S + delta I`` is PSD over the RATIONALS, S the exact symmetric part of Q."""
    delta = rigorous_psd_shift(Q)
    assert delta is not None and delta >= 0.0
    n = Q.shape[0]
    d = Fraction(delta)
    S = [
        [
            (Fraction(float(Q[i, j])) + Fraction(float(Q[j, i]))) / 2 + (d if i == j else 0)
            for j in range(n)
        ]
        for i in range(n)
    ]
    assert _exact_psd_fraction(S), label
    # ...and it is a tight proof, not a licence: within O(n u ||Q||) of -lambda_min.
    lam = float(np.linalg.eigvalsh(0.5 * (Q + Q.T))[0])
    slack = 200.0 * n * 2.0**-53 * float(np.linalg.norm(Q))
    assert delta <= max(-lam, 0.0) + slack, (delta, lam, slack)


def test_rigorous_shift_refuses_nonfinite():
    assert rigorous_psd_shift(np.array([[1.0, np.nan], [np.nan, 1.0]])) is None
    assert rigorous_psd_shift(np.zeros((3, 3))) == 0.0


def test_charge_needs_bounded_quadratic_variables_only():
    Q = np.zeros((3, 3))
    Q[:2, :2] = [[1.0, -1.0], [-1.0, 1.0 - 1e-15]]
    # an unbounded LINEAR-only variable (index 2) does not block the charge
    c = quadratic_objective_charge(Q, [-1, -1, -1e20], [1, 1, 1e20])
    assert c is not None and 0.0 < c < 1e-12
    # an unbounded QUADRATIC variable does
    assert quadratic_objective_charge(Q, [-1, -1e20, 0], [1, 1, 0]) is None
    assert quadratic_objective_charge(Q, [-1, -np.inf, 0], [1, 1, 0]) is None


def _ls_model(lb=-10.0, ub=10.0, maximize=False):
    rng = np.random.default_rng(0)
    K = rng.normal(size=(40, 35))
    K[:, -1] = K[:, 0] + K[:, 1]
    b = rng.normal(size=40)
    m = Model("ls")
    x = m.continuous("x", shape=(35,), lb=lb, ub=ub)
    f = x @ (K.T @ K) @ x - 2 * ((K.T @ b) @ x) + float(b @ b)
    if maximize:
        m.maximize(-f)
    else:
        m.minimize(f)
    return m, K, b


def test_acceptance_only_inside_a_solve_scope(monkeypatch):
    """``Model.convexity()`` and any caller outside a solve keep the strict proof."""
    monkeypatch.setenv("DISCOPT_CONVEX_CHARGE", "1")
    m, _, _ = _ls_model()
    assert certify_quadratic_objective_convex(m) is False
    assert m.convexity().is_convex is False
    with convexity_charge_scope() as scope:
        assert certify_quadratic_objective_convex(m) is False  # no tolerance known yet
        set_charge_scope_abs_gap_tol(1e-6)
        assert certify_quadratic_objective_convex(m) is True
        assert 0.0 < scope.total <= 1e-7
    with convexity_charge_scope():
        set_charge_scope_abs_gap_tol(1e-12)  # charge too large for this tolerance
        assert certify_quadratic_objective_convex(m) is False


def test_flag_off_keeps_refusal(monkeypatch):
    monkeypatch.setenv("DISCOPT_CONVEX_CHARGE", "0")
    m, _, _ = _ls_model()
    with convexity_charge_scope() as scope:
        set_charge_scope_abs_gap_tol(1e-6)
        assert certify_quadratic_objective_convex(m) is False
        assert scope.total == 0.0


@pytest.mark.parametrize("maximize", [False, True])
@pytest.mark.parametrize("box", [(-10.0, 10.0), (-0.3, 0.2)])
def test_charged_solve_certifies_with_a_valid_bound(monkeypatch, maximize, box):
    """Regression for the #1682 table row (0.22 s optimal before #1679, 60 s
    ``feasible`` after): with the flag the solve certifies again, the published
    bound is valid against an independent box optimum, the charge was subtracted,
    and no sampled feasible point beats the bound."""
    monkeypatch.setenv("DISCOPT_CONVEX_CHARGE", "1")
    m, K, b = _ls_model(*box, maximize=maximize)
    r = m.solve(time_limit=60)
    assert r.status == "optimal" and r.gap_certified
    charge = r.solver_stats["convexity/charge"]
    assert 0.0 < charge <= 1e-7
    ref = lsq_linear(K, b, bounds=box, tol=1e-14, max_iter=10000)
    f_ref = float(np.sum((K @ ref.x - b) ** 2))
    sign = -1.0 if maximize else 1.0
    lower = sign * r.bound  # internal-minimize lower bound
    assert lower <= f_ref + 1e-12 * (1 + abs(f_ref))
    assert lower <= sign * r.objective - charge  # charge subtracted
    assert abs(sign * r.objective - f_ref) <= 1e-6 * (1 + abs(f_ref))
    rng = np.random.default_rng(7)
    n_checked = 0
    for _ in range(2000):
        xs = rng.uniform(box[0], box[1], size=35)
        assert float(np.sum((K @ xs - b) ** 2)) >= lower
        n_checked += 1
    assert n_checked == 2000


def test_flag_off_does_not_charge(monkeypatch):
    monkeypatch.setenv("DISCOPT_CONVEX_CHARGE", "0")
    m, _, _ = _ls_model()
    r = m.solve(time_limit=3)
    assert "convexity/charge" not in (r.solver_stats or {})
    assert not (r.status == "optimal" and r.node_count == 0)


def test_structural_spelling_unaffected(monkeypatch):
    """The sum-of-squares spelling is proved convex structurally; no charge."""
    monkeypatch.setenv("DISCOPT_CONVEX_CHARGE", "1")
    rng = np.random.default_rng(0)
    K = rng.normal(size=(40, 35))
    K[:, -1] = K[:, 0] + K[:, 1]
    b = rng.normal(size=40)
    m = Model("ls_vec")
    x = m.continuous("x", shape=(35,), lb=-10, ub=10)
    m.minimize(dm.sum((K @ x - b) ** 2))
    r = m.solve(time_limit=60)
    assert r.status == "optimal"
    assert "convexity/charge" not in (r.solver_stats or {})


def test_nested_scope_charges_reach_the_enclosing_scope(monkeypatch):
    """A verdict memoized inside a nested solve must still be charged by the outer."""
    monkeypatch.setenv("DISCOPT_CONVEX_CHARGE", "1")
    m, _, _ = _ls_model()
    with convexity_charge_scope() as outer:
        set_charge_scope_abs_gap_tol(1e-6)
        with convexity_charge_scope() as inner:
            set_charge_scope_abs_gap_tol(1e-6)
            assert certify_quadratic_objective_convex(m) is True
        assert inner.total > 0.0
        assert outer.total == inner.total


def test_default_is_on(monkeypatch):
    monkeypatch.delenv("DISCOPT_CONVEX_CHARGE", raising=False)
    m, _, _ = _ls_model()
    with convexity_charge_scope() as scope:
        set_charge_scope_abs_gap_tol(1e-6)
        assert certify_quadratic_objective_convex(m) is True
        assert scope.total > 0.0
