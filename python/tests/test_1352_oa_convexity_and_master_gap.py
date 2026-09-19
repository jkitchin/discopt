"""Issue #1352: OA objective-convexity false negative, and the master-MILP gap stall
that the obvious fix exposed.

**(a) Master gap units.** OA hands its *relative* ``gap_tolerance`` to the master
MILP, whose in-house engine normalises the gap by ``max(|incumbent|, 1.0)``. Below
unit objective scale that is an ABSOLUTE ``1e-4``: on the Markowitz model below
(optimum ~1.1e-3) the master exited ``optimal`` 5.7% short of its optimum,
re-proposed an already-visited assignment, and OA stalled to its iteration cap.
:func:`master_gap_tolerance` converts OA's closing window into the engine's units.
This is independent of (b): the sum-of-squares formulation here is convex to the
legacy rules.

**(b) Convexity classification.** ``classify_oa_cut_convexity`` used only the
syntactic rules, so a PSD objective written with ``dm.sum`` over array variables
read as non-convex and OA dropped its objective cuts. Behind
``DISCOPT_OA_CONVEXITY_CERTIFICATE`` (default OFF) it now also consults the sound
interval-Hessian certificate and the exact-QP Hessian route (#936) — the same
routes the convex-MINLP router already trusts.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.convexity import classify_oa_cut_convexity
from discopt.solvers._gap import GAP_ABS_TOL, master_gap_tolerance

N = 8
MU = np.array([0.12, 0.10, 0.07, 0.03, 0.15, 0.08, 0.11, 0.06])
_L = np.diag([0.10, 0.08, 0.06, 0.04, 0.12, 0.07, 0.09, 0.05])
for (_i, _j), _v in {
    (1, 0): 0.02,
    (2, 0): 0.01,
    (3, 1): 0.01,
    (4, 0): 0.03,
    (5, 2): 0.01,
    (6, 1): 0.02,
    (7, 3): 0.01,
}.items():
    _L[_i, _j] = _v
SIGMA = _L @ _L.T

# Brute force over all C(8,3) supports (SLSQP per support), from the issue.
TRUE_OPT_K3 = 0.0020154286


def _portfolio(K: int, objective: str) -> dm.Model:
    m = dm.Model("portfolio")
    z = m.binary("z", shape=(N,))
    w = m.continuous("w", shape=(N,), lb=0.0, ub=0.4)
    if objective == "dm.sum":
        m.minimize(
            dm.sum(
                lambda i: dm.sum(lambda j: SIGMA[i, j] * w[i] * w[j], over=range(N)),
                over=range(N),
            )
        )
    else:  # sum of squares of L^T w: convex to the syntactic rules
        m.minimize(sum((sum(_L[i, k] * w[i] for i in range(N))) ** 2 for k in range(N)))
    m.subject_to(dm.sum(lambda i: w[i], over=range(N)) == 1.0)
    m.subject_to(dm.sum(lambda i: MU[i] * w[i], over=range(N)) >= 0.09)
    m.subject_to(dm.sum(z) <= K)
    for i in range(N):
        m.subject_to(w[i] >= 0.02 * z[i])
        m.subject_to(w[i] <= 0.40 * z[i])
    return m


def _qp_model(Q: np.ndarray) -> dm.Model:
    n = Q.shape[0]
    m = dm.Model("qp")
    x = m.continuous("x", shape=(n,), lb=-1.0, ub=1.0)
    y = m.binary("y")
    m.minimize(
        dm.sum(lambda i: dm.sum(lambda j: Q[i, j] * x[i] * x[j], over=range(n)), over=range(n))
    )
    m.subject_to(dm.sum(lambda i: x[i], over=range(n)) >= 0.5 * y)
    return m


# ── (a) master gap tolerance ─────────────────────────────────────────────────


@pytest.mark.smoke
def test_master_gap_tolerance_is_unchanged_at_unit_scale_and_above(monkeypatch):
    monkeypatch.delenv("DISCOPT_OA_MASTER_GAP_SCALED", raising=False)
    for ub in (1.0, -1.0, 3.7, -250.0, 1e8):
        assert master_gap_tolerance(1e-4, ub) == 1e-4
    # No incumbent yet (None / the 1e20 sentinel / non-finite): scale unknown.
    for ub in (None, 1e20, -1e20, float("inf"), float("nan")):
        assert master_gap_tolerance(1e-4, ub) == 1e-4


@pytest.mark.smoke
def test_master_gap_tolerance_is_oa_window_below_unit_scale(monkeypatch):
    monkeypatch.delenv("DISCOPT_OA_MASTER_GAP_SCALED", raising=False)
    # OA closes at max(abs_tol, rel * |ub|); the engine's gap below unit scale is
    # the absolute gap, so the master must be asked for exactly that window.
    assert master_gap_tolerance(1e-4, 2e-3) == GAP_ABS_TOL
    assert master_gap_tolerance(1e-4, -0.5) == pytest.approx(5e-5)
    assert master_gap_tolerance(1e-4, 0.0) == GAP_ABS_TOL
    # Never looser than what the caller asked for.
    assert master_gap_tolerance(1e-9, 2e-3) == 1e-9


@pytest.mark.smoke
def test_master_gap_tolerance_opt_out(monkeypatch):
    monkeypatch.setenv("DISCOPT_OA_MASTER_GAP_SCALED", "0")
    assert master_gap_tolerance(1e-4, 2e-3) == 1e-4


def test_oa_master_no_longer_stalls_below_unit_scale(monkeypatch):
    """K=6: the master used to exit 5.7% short, re-propose a visited assignment,
    and leave OA ``feasible`` at its iteration cap (72 MILPs, 3.9 s). Measured
    with the fix: ``optimal`` in 17 MILPs."""
    from discopt.solvers.oa import solve_oa

    monkeypatch.delenv("DISCOPT_OA_MASTER_GAP_SCALED", raising=False)
    model = _portfolio(6, "sum-of-squares")
    assert classify_oa_cut_convexity(model).objective_is_convex is True  # independent of (b)
    r = solve_oa(model, time_limit=60)
    assert r.status == "optimal"
    assert r.bound is not None and r.bound <= r.objective
    assert r.objective - r.bound <= max(GAP_ABS_TOL, 1e-4 * abs(r.objective))
    assert r.mip_count < 40


# ── (b) convexity classification behind the flag ────────────────────────────


@pytest.mark.smoke
def test_flag_off_keeps_legacy_verdict(monkeypatch):
    monkeypatch.delenv("DISCOPT_OA_CONVEXITY_CERTIFICATE", raising=False)
    model = _portfolio(3, "dm.sum")
    assert classify_oa_cut_convexity(model).objective_is_convex is False


@pytest.mark.smoke
def test_flag_on_certifies_dm_sum_psd_objective(monkeypatch):
    monkeypatch.setenv("DISCOPT_OA_CONVEXITY_CERTIFICATE", "1")
    model = _portfolio(3, "dm.sum")
    assert classify_oa_cut_convexity(model).objective_is_convex is True


@pytest.mark.smoke
def test_flag_on_uses_exact_qp_for_non_diagonally_dominant_psd(monkeypatch):
    """A PSD matrix Gershgorin cannot certify: only the exact-QP route proves it."""
    from discopt._relax.convexity.certificate import certify_convex
    from discopt._relax.convexity.rules import Curvature

    v = np.array([1.0, 1.0, 1.0, 1.0])
    Q = np.outer(v, v) + 1e-3 * np.eye(4)  # PSD, far from diagonally dominant
    model = _qp_model(Q)
    assert certify_convex(model._objective.expression, model) is not Curvature.CONVEX
    monkeypatch.setenv("DISCOPT_OA_CONVEXITY_CERTIFICATE", "1")
    assert classify_oa_cut_convexity(model).objective_is_convex is True
    monkeypatch.setenv("DISCOPT_OA_CONVEXITY_CERTIFICATE", "0")
    assert classify_oa_cut_convexity(model).objective_is_convex is False


@pytest.mark.smoke
def test_flag_on_refuses_indefinite_objective(monkeypatch):
    monkeypatch.setenv("DISCOPT_OA_CONVEXITY_CERTIFICATE", "1")
    Q = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3, -1
    assert classify_oa_cut_convexity(_qp_model(Q)).objective_is_convex is False


def test_flag_on_oa_certifies_issue_model_within_contract(monkeypatch):
    """The issue's Finding 2 case. The 4.1e-7 miss is inside discopt's absolute
    contract (the fixed NLP stops short of an active bound); the bound is valid."""
    monkeypatch.setenv("DISCOPT_OA_CONVEXITY_CERTIFICATE", "1")
    r = _portfolio(3, "dm.sum").solve(time_limit=60)
    assert r.status == "optimal"
    assert r.node_count == 0
    assert "fell back" not in (r.algorithm_route or "")
    assert r.bound <= TRUE_OPT_K3 + 1e-9
    assert abs(r.objective - TRUE_OPT_K3) <= GAP_ABS_TOL + 1e-4 * TRUE_OPT_K3
