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
``DISCOPT_OA_CONVEXITY_CERTIFICATE`` (default ON, opt-out ``=0``) it now also consults the sound
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


# ── (b) convexity classification (flag default ON) ──────────────────────────


@pytest.mark.smoke
def test_flag_default_on_certifies_dm_sum_objective(monkeypatch):
    monkeypatch.delenv("DISCOPT_OA_CONVEXITY_CERTIFICATE", raising=False)
    model = _portfolio(3, "dm.sum")
    assert classify_oa_cut_convexity(model).objective_is_convex is True


@pytest.mark.smoke
@pytest.mark.parametrize("value", ["0", "false", "no", "off"])
def test_flag_opt_out_keeps_legacy_verdict(monkeypatch, value):
    monkeypatch.setenv("DISCOPT_OA_CONVEXITY_CERTIFICATE", value)
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


def test_explicit_oa_certifies_issue_model_within_contract(monkeypatch):
    """The issue's Finding 2 case, through explicit OA. The 4.1e-7 miss is inside
    discopt's absolute contract (the fixed NLP stops short of an active bound); the
    bound is valid. With the certificate opted out OA cannot certify at all."""
    from discopt.solvers.oa import solve_oa

    monkeypatch.delenv("DISCOPT_OA_CONVEXITY_CERTIFICATE", raising=False)
    r = solve_oa(_portfolio(3, "dm.sum"), time_limit=60)
    assert r.status == "optimal"
    assert r.bound <= TRUE_OPT_K3 + 1e-9
    assert abs(r.objective - TRUE_OPT_K3) <= GAP_ABS_TOL + 1e-4 * TRUE_OPT_K3

    monkeypatch.setenv("DISCOPT_OA_CONVEXITY_CERTIFICATE", "0")
    legacy = solve_oa(_portfolio(3, "dm.sum"), time_limit=60)
    assert legacy.status != "optimal"
    assert legacy.bound is None or legacy.bound <= TRUE_OPT_K3 + 1e-9


def test_router_sends_certificate_only_objective_to_bnb(monkeypatch):
    """The router gate: a model whose objective is convex only by the numerical
    certificate goes straight to B&B, not through an OA attempt that falls back.
    The opt-out restores the OA route, which then certifies at the root."""
    from discopt.solver import _convex_minlp_auto_route

    monkeypatch.delenv("DISCOPT_OA_CONVEXITY_CERTIFICATE", raising=False)
    monkeypatch.delenv("DISCOPT_CONVEX_ROUTE_SYNTACTIC_OBJECTIVE", raising=False)
    monkeypatch.delenv("DISCOPT_CONVEX_MINLP_ROUTE", raising=False)
    model = _portfolio(3, "dm.sum")
    route, reason, _ = _convex_minlp_auto_route(model)
    assert route is None
    assert "numerical certificate" in reason
    # A syntactically convex objective of the same model class is still routed.
    assert _convex_minlp_auto_route(_portfolio(3, "sumsq"))[0] is not None

    r = model.solve(time_limit=60)
    assert r.status == "optimal"
    assert "oa" not in (r.algorithm_route or "").lower()
    assert r.bound <= TRUE_OPT_K3 + 1e-9
    assert abs(r.objective - TRUE_OPT_K3) <= GAP_ABS_TOL + 1e-4 * TRUE_OPT_K3

    monkeypatch.setenv("DISCOPT_CONVEX_ROUTE_SYNTACTIC_OBJECTIVE", "0")
    assert _convex_minlp_auto_route(_portfolio(3, "dm.sum"))[0] is not None
    r = _portfolio(3, "dm.sum").solve(time_limit=60)
    assert r.status == "optimal"
    assert r.node_count == 0
    assert "fell back" not in (r.algorithm_route or "")
    assert r.bound <= TRUE_OPT_K3 + 1e-9


# ── (c) fixed-NLP scale-aware tolerance ──────────────────────────────────────────


def _markowitz(n: int, K: int, seed: int) -> tuple[dm.Model, np.ndarray]:
    """Seeded cardinality Markowitz model written with ``dm.sum`` (not diag-dominant)."""
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(n, 3)) * 0.1
    sigma = F @ F.T + np.diag(rng.uniform(0.001, 0.02, n))
    mu = rng.uniform(0.02, 0.15, n)
    rmin = float(np.quantile(mu, 0.6))
    m = dm.Model(f"markowitz-{n}-{K}-{seed}")
    z = m.binary("z", shape=(n,))
    w = m.continuous("w", shape=(n,), lb=0.0, ub=0.5)
    m.minimize(
        dm.sum(lambda i: dm.sum(lambda j: sigma[i, j] * w[i] * w[j], over=range(n)), over=range(n))
    )
    m.subject_to(dm.sum(lambda i: w[i], over=range(n)) == 1.0)
    m.subject_to(dm.sum(lambda i: mu[i] * w[i], over=range(n)) >= rmin)
    m.subject_to(dm.sum(z) <= K)
    for i in range(n):
        m.subject_to(w[i] >= 0.02 * z[i])
        m.subject_to(w[i] <= 0.5 * z[i])
    return m, sigma


# Exact optimum of the (10, 3, 3) model: its optimal support {1, 5, 6} solved as a
# convex QP by HiGHS at 1e-10 tolerances. POUNCE at default tolerances returned
# 2.9469633e-3 (status 0) for the same fixed assignment.
MARKOWITZ_10_3_3_OPT = 0.0029462641054


class _Obj:
    def __init__(self, value):
        self.value = value

    def evaluate_objective(self, x):
        return self.value


@pytest.mark.smoke
def test_nlp_scaled_tol_rule(monkeypatch):
    from discopt.solvers.oa import _nlp_scaled_tol

    monkeypatch.delenv("DISCOPT_OA_NLP_SCALED_TOL", raising=False)
    x0 = np.zeros(2)
    # Unit scale and above: solver default (every such solve is unchanged).
    for f in (1.0, -1.0, 3.7, -250.0, 1e8, float("inf"), float("nan")):
        assert _nlp_scaled_tol(_Obj(f), x0) is None
    assert _nlp_scaled_tol(_Obj(0.5), x0) == pytest.approx(5e-9)
    assert _nlp_scaled_tol(_Obj(-0.05), x0) == pytest.approx(5e-10)
    # Never tighter than 1e-10, however small the objective.
    for f in (2.9e-3, 1e-9, 0.0):
        assert _nlp_scaled_tol(_Obj(f), x0) == pytest.approx(1e-10)
    monkeypatch.setenv("DISCOPT_OA_NLP_SCALED_TOL", "0")
    assert _nlp_scaled_tol(_Obj(2.9e-3), x0) is None


def test_oa_closes_when_fixed_nlp_is_scale_aware(monkeypatch):
    """Without the scaled tolerance OA re-proposes a visited assignment 50 times, its master
    1.04e-6 below an NLP incumbent 7e-7 above the true optimum, and ends
    ``stalling`` (61 MILPs). With it: ``optimal`` at the exact optimum in 11."""
    from discopt.solvers.oa import solve_oa

    monkeypatch.setenv("DISCOPT_OA_CONVEXITY_CERTIFICATE", "1")
    monkeypatch.delenv("DISCOPT_OA_NLP_SCALED_TOL", raising=False)
    model, _ = _markowitz(10, 3, 3)
    r = solve_oa(model, time_limit=60)
    assert r.status == "optimal"
    assert r.mip_count < 30
    assert r.bound <= MARKOWITZ_10_3_3_OPT + 1e-9
    assert abs(r.objective - MARKOWITZ_10_3_3_OPT) <= 1e-8

    monkeypatch.setenv("DISCOPT_OA_NLP_SCALED_TOL", "0")
    legacy = solve_oa(_markowitz(10, 3, 3)[0], time_limit=60)
    assert legacy.status == "feasible"  # the stall this fixes; the opt-out keeps it
    assert legacy.bound <= MARKOWITZ_10_3_3_OPT + 1e-9
