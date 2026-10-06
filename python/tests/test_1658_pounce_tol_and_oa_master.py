"""#1658: two performance regressions, each with the arm that must still fire.

1. ``tol*sigma`` (#1537) handed POUNCE a tolerance the engine could not reach on
   an LP with no strict interior: complementarity bottoms out at roundoff
   (~5e-9 in engine units on the blending LP below), so the engine ran 59
   iterations to its stall exit instead of 16 and returned no better a point.
   The engine now runs at the caller's ``tol`` first and is re-run at
   ``tol*sigma`` only when the mapped-back residual fails the caller-unit test.
   The ``3200*j**2`` class #1537 was built for must still take the re-run.
2. The HiGHS OA master ran the #1634 presolve-free cross-solve on every master,
   half of all master time, though OA certifies a master bound only when it
   closes the gap. The cross-solve now runs on demand; skipped bounds are never
   certified.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("pounce")

from discopt.solvers import SolveStatus  # noqa: E402
from discopt.solvers import convex_ipm_pounce as cvx  # noqa: E402

# ── item 1: POUNCE tolerance under objective scaling ─────────────────────────

_STOCKS = {  # name: (avail bbl/d, cost $/bbl, RON, RVP psi, S ppm, SG)
    "butane": (3000, 45, 93, 52.0, 10.0, 0.584),
    "LSR": (6000, 70, 70, 11.0, 5.0, 0.664),
    "reformate": (9000, 98, 98, 3.5, 1.0, 0.810),
    "FCC": (12000, 92, 92, 7.0, 20.0, 0.745),
    "alkylate": (4000, 105, 96, 4.6, 6.0, 0.700),
}
_PRODS = {"regular": (110, 20000, 91, 9.0, 10), "premium": (122, 8000, 96, 9.0, 10)}


def _blend(contract: bool):
    m = dm.Model("blend")
    x = {
        (i, j): m.continuous(f"x[{i},{j}]", lb=0, ub=float(_PRODS[j][1]))
        for i in _STOCKS
        for j in _PRODS
    }
    for i, s in _STOCKS.items():
        m.subject_to(sum(x[i, j] for j in _PRODS) <= float(s[0]))
    for j, (_price, mx, ron, rvp, smax) in _PRODS.items():
        m.subject_to(sum(x[i, j] for i in _STOCKS) <= float(mx))
        m.subject_to(sum(float(s[2] - ron) * x[i, j] for i, s in _STOCKS.items()) >= 0)
        m.subject_to(
            sum(float(s[3] ** 1.25 - rvp**1.25) * x[i, j] for i, s in _STOCKS.items()) <= 0
        )
        m.subject_to(sum(float(s[5] * (s[4] - smax)) * x[i, j] for i, s in _STOCKS.items()) <= 0)
    if contract:
        # Tight at every feasible point: the LP has no strict interior.
        m.subject_to(sum(x[i, "premium"] for i in _STOCKS) >= 8000)
    m.maximize(
        sum(float(_PRODS[j][0] - s[1]) * x[i, j] for i, s in _STOCKS.items() for j in _PRODS)
    )
    return m


@pytest.mark.parametrize(
    "contract, objective", [(False, 574487.40), (True, 563213.29)], ids=["interior", "no_interior"]
)
def test_blend_lp_does_not_stall_at_the_scaled_tolerance(contract, objective):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _blend(contract).solve(solver="pounce")
    assert r.status == "optimal", (r.status, r.error)
    assert r.algorithm_route == "pounce:lp-ipm"
    assert r.objective == pytest.approx(objective, abs=0.01)
    # Measured: 16 on both (was 18 and 59 under an unconditional tol*sigma).
    assert r.solver_stats["pounce/iterations"] <= 20


def _rb(k):
    """``k/2*j**2 + (x-0.4)**2 - 1.6*j`` s.t. ``x - j = 0.4`` as matrix data."""
    P = sp.csc_matrix(np.diag([2.0, float(k)]))
    c = np.array([-0.8, -1.6])
    A_eq = np.array([[1.0, -1.0]])
    b_eq = np.array([0.4])
    lb = np.array([0.0, -1e3])
    ub = np.array([1.0, 1e3])
    return P, c, A_eq, b_eq, lb, ub


@pytest.mark.parametrize("k", [64 / 0.01, 1e6])
def test_large_quadratic_still_takes_the_scaled_resolve(k):
    """The #1537 class: the caller-tol point fails the caller-unit test, so the
    ``tol*sigma`` re-solve must run (the arm this change keeps)."""
    P, c, A_eq, b_eq, lb, ub = _rb(k)
    status, res, *_ = cvx._solve(P, c, None, None, A_eq, b_eq, lb, ub, None, None)
    assert status == "optimal"
    assert getattr(res, "retried_at_scaled_tol", False) is True
    assert float(res.x[1]) == pytest.approx(1.6 / (k + 2), rel=1e-5)
    assert cvx.caller_unit_converged(res, P, c, cvx._ENGINE_DEFAULT_TOL)


def test_lp_with_a_reachable_tolerance_runs_once():
    """A scaled LP whose caller-``tol`` point already passes is not re-solved."""
    c = np.array([-50.0, -77.0])
    A_ub = np.array([[1.0, 1.0], [2.0, 1.0]])
    b_ub = np.array([10.0, 15.0])
    lb, ub = np.zeros(2), np.full(2, 20.0)
    status, res, *_ = cvx._solve(None, c, A_ub, b_ub, None, None, lb, ub, None, None)
    assert status == "optimal"
    assert res.objective_scale < 1.0  # the scaled path, so the choice was made
    assert not getattr(res, "retried_at_scaled_tol", False)
    assert cvx.caller_unit_converged(res, None, c, cvx._ENGINE_DEFAULT_TOL)


def test_caller_unit_converged_reads_each_residual():
    from types import SimpleNamespace

    def res(pr, du, co, y=100.0):
        return SimpleNamespace(
            x=np.zeros(1),
            y=np.array([y]),
            z=None,
            z_lb=None,
            z_ub=None,
            residuals={
                "primal_infeasibility": pr,
                "dual_infeasibility": du,
                "complementarity": co,
            },
        )

    c = np.array([1.0])
    # Dual/complementarity are judged against tol * max(1, |c|, |y|) = 1e-6 here.
    assert cvx.caller_unit_converged(res(1e-9, 9e-7, 9e-7), None, c, 1e-8)
    assert not cvx.caller_unit_converged(res(1e-9, 2e-6, 0.0), None, c, 1e-8)
    assert not cvx.caller_unit_converged(res(1e-9, 0.0, 2e-6), None, c, 1e-8)
    # The primal residual does not scale with the objective: absolute.
    assert not cvx.caller_unit_converged(res(2e-8, 0.0, 0.0), None, c, 1e-8)
    # No breakdown -> not judged converged (gets the re-solve).
    nobreak = res(0.0, 0.0, 0.0)
    nobreak.residuals = {}
    assert not cvx.caller_unit_converged(nobreak, None, c, 1e-8)


# ── item 2: the HiGHS OA master's cross-solve on demand ─────────────────────

pytest.importorskip("highspy")

from discopt.solvers import milp_highs  # noqa: E402
from discopt.solvers.oa import _master_bound_confirmed  # noqa: E402


def _knapsack():
    """``min -x0 - 2 x1 - 3 x2`` s.t. ``x0 + x1 + x2 <= 2``, binaries; optimum -5."""
    return dict(
        c=np.array([-1.0, -2.0, -3.0]),
        A_ub=np.array([[1.0, 1.0, 1.0]]),
        b_ub=np.array([2.0]),
        bounds=[(0.0, 1.0)] * 3,
        integrality=np.ones(3, dtype=int),
    )


def test_confirm_bound_from_skips_the_cross_solve_below_it(monkeypatch):
    calls = []
    real = milp_highs._cross_check_presolve
    monkeypatch.setattr(
        milp_highs, "_cross_check_presolve", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    below = milp_highs.solve_milp(**_knapsack(), confirm_bound_from=0.0)
    assert below.bound == pytest.approx(-5.0)
    assert calls == []
    assert below.callback_stats["presolve_cross_check"]["confirmed"] is False
    assert not _master_bound_confirmed(below)

    at = milp_highs.solve_milp(**_knapsack(), confirm_bound_from=-5.0)
    assert calls == [1]
    assert at.callback_stats["presolve_cross_check"]["confirmed"] is True
    assert _master_bound_confirmed(at)

    default = milp_highs.solve_milp(**_knapsack())
    assert calls == [1, 1]
    assert _master_bound_confirmed(default)


def test_infeasible_claim_is_always_cross_checked(monkeypatch):
    calls = []
    real = milp_highs._cross_check_presolve
    monkeypatch.setattr(
        milp_highs, "_cross_check_presolve", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    data = _knapsack()
    data["A_ub"] = np.array([[-1.0, -1.0, -1.0]])
    data["b_ub"] = np.array([-4.0])  # x0 + x1 + x2 >= 4 with three binaries
    r = milp_highs.solve_milp(**data, confirm_bound_from=1e9)
    assert r.status == SolveStatus.INFEASIBLE
    assert calls == [1]


def test_unmarked_master_reads_confirmed():
    from discopt.solvers import MILPResult

    assert _master_bound_confirmed(MILPResult(status=SolveStatus.OPTIMAL, bound=1.0))


def _portfolio(n=12, seed=21, K=3, eps=1e-3, perspective=False):
    rng = np.random.default_rng(seed)
    mu = np.round(rng.uniform(0.04, 0.14, n), 3)
    B = np.round(rng.normal(0, 1, (n, 2)) * [0.15, 0.08], 3)
    d = np.round(rng.uniform(0.15, 0.35, n) ** 2, 4)
    m = dm.Model(f"portfolio_{n}")
    x = m.continuous("x", shape=(n,), lb=0, ub=0.4)
    z = m.binary("z", shape=(n,))
    f = m.continuous("f", shape=(2,), lb=-1, ub=1)
    for k in range(2):
        m.subject_to(f[k] == dm.sum(lambda i: B[i, k] * x[i], over=range(n)))
    if perspective:
        w = m.continuous("w", shape=(n,), lb=eps, ub=1)
        for i in range(n):
            m.subject_to(w[i] == (1 - eps) * z[i] + eps)
        own = dm.sum(lambda i: d[i] * x[i] ** 2 / w[i], over=range(n))
    else:
        own = dm.sum(lambda i: d[i] * x[i] ** 2, over=range(n))
    m.minimize(f[0] ** 2 + f[1] ** 2 + own)
    m.subject_to(dm.sum(lambda i: x[i], over=range(n)) == 1)
    m.subject_to(dm.sum(lambda i: mu[i] * x[i], over=range(n)) >= 0.10)
    m.subject_to(dm.sum(lambda i: z[i], over=range(n)) <= K)
    for i in range(n):
        m.subject_to(x[i] <= 0.4 * z[i])
        m.subject_to(x[i] >= 0.1 * z[i])
    return m


@pytest.mark.parametrize("perspective", [False, True], ids=["big_m", "perspective"])
def test_oa_certifies_with_fewer_cross_solves(monkeypatch, perspective):
    from discopt.solvers.mip_nlp import solve_mip_nlp

    calls = []
    real = milp_highs._cross_check_presolve
    monkeypatch.setattr(
        milp_highs, "_cross_check_presolve", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = solve_mip_nlp(
            _portfolio(perspective=perspective),
            method="oa",
            time_limit=120,
            milp_solver="highs",
            max_iterations=1000,
        )
    assert r.status == "optimal" and r.gap_certified, (r.status, r.objective, r.bound)
    assert r.bound <= r.objective
    assert (r.objective - r.bound) <= 1e-4 * abs(r.objective)
    # Both arms fired: some masters were cross-checked (the certificate's), and
    # some were not (measured: 37 masters / 5 cross-solves big-M, 16 / 3 perspective).
    assert 1 <= len(calls) < r.mip_count


def test_opt_out_cross_checks_every_master(monkeypatch):
    from discopt.solvers.mip_nlp import solve_mip_nlp

    monkeypatch.setenv("DISCOPT_OA_MASTER_CONFIRM_ON_DEMAND", "0")
    calls = []
    real = milp_highs._cross_check_presolve
    monkeypatch.setattr(
        milp_highs, "_cross_check_presolve", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = solve_mip_nlp(
            _portfolio(n=8, perspective=True),
            method="oa",
            time_limit=120,
            milp_solver="highs",
            max_iterations=1000,
        )
    assert r.status == "optimal"
    assert len(calls) == r.mip_count


def test_route_fallback_merge_keeps_the_route_masters():
    from discopt.modeling.core import SolveResult
    from discopt.solver import _merge_route_and_fallback

    route = SolveResult(status="feasible", objective=2.0, bound=1.0, mip_count=51, node_count=0)
    fallback = SolveResult(status="optimal", objective=1.5, bound=1.5, mip_count=0, node_count=11)
    merged = _merge_route_and_fallback(route, fallback, is_maximize=False)
    assert merged.objective == 1.5
    assert merged.mip_count == 51
    assert merged.node_count == 11


# ── item 2, continued: LP/NLP-BB on HiGHS, now the route's target ───────────


def _portfolio_seeded(n, seed, perspective=False):
    return _portfolio(n=n, seed=seed, perspective=perspective)


@pytest.mark.parametrize("n, seed", [(18, 22), (22, 23), (30, 23)])
def test_lazy_master_does_not_judge_offers_from_a_stale_tree(monkeypatch, n, seed):
    """HiGHS keeps offering improving solutions of a tree whose cut is already
    pending. Judged there, an offer that violates a pending row looked clean to a
    separator that reports only rows it has not emitted, and was accepted: on the
    n=20 big-M portfolio a point with master objective 0.0058 and true objective
    0.0301 (optimum 0.01044) became HiGHS's incumbent, capped the dual bound at
    0.0058, and LP/NLP-BB came back ``feasible``.

    Fail-before is timing-dependent: whether HiGHS offers a stale point depends on
    where its interrupt polls land. On ``main`` at 1d9d27ac, on the Linux CI-class
    container this was written on, all three instances here (and 9 more of the
    family) return ``feasible`` with the bound capped, 2/2 runs each; a macOS run
    certified the n=20 seed-21 case on ``main`` (review of #1673). The status
    assertions catch the defect where it shows; the ``stale_offers`` counter is
    the probe that the guard fired on every platform (CLAUDE.md §6)."""
    from discopt.solvers.mip_nlp import solve_mip_nlp

    seen = []
    real = milp_highs.solve_milp_with_lazy_cuts

    def _spy(*a, **kw):
        r = real(*a, **kw)
        seen.append(dict(r.callback_stats or {}))
        return r

    monkeypatch.setattr(milp_highs, "solve_milp_with_lazy_cuts", _spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = solve_mip_nlp(
            _portfolio_seeded(n, seed),
            method="lp_nlp_bb",
            milp_solver="highs",
            time_limit=60,
        )
    assert r.status == "optimal" and r.gap_certified, (r.status, r.objective, r.bound)
    assert r.bound <= r.objective
    assert seen and sum(int(s.get("stale_offers", 0)) for s in seen) >= 1, seen


def test_hook_stop_does_not_spend_the_budget_on_the_cross_check():
    """#1673 review B2: when the caller's termination hook stops the lazy master
    (the #1066 guard handing over to the fallback), the #1634 cross-solve must not
    run on the route's remaining budget -- on rsyn0815m03m it ran 17.0 s -> 26.2 s
    of a 30 s limit and the fallback overran to 36.3 s. The bound it would have
    confirmed is withdrawn instead."""
    import time

    from discopt.solvers.mip_nlp import solve_mip_nlp

    calls = []

    def hook(ctx):
        calls.append(ctx["elapsed"])
        return True

    t = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = solve_mip_nlp(
            _portfolio(n=20, perspective=True),
            method="lp_nlp_bb",
            milp_solver="highs",
            time_limit=60,
            mip_nlp_options={"termination_hook": hook},
        )
    wall = time.perf_counter() - t
    assert calls, "the hook was never consulted"
    stats = r.mip_nlp_trace["summary"]["callback_stats"]
    assert stats["abandoned"] is True
    assert stats["presolve_cross_check"].get("ran") is not True
    assert "no time budget" in str(stats["presolve_cross_check"].get("withdrawn"))
    assert r.status != "optimal" and not r.gap_certified
    assert wall < 20.0, wall


@pytest.mark.parametrize("perspective", [False, True], ids=["big_m", "perspective"])
def test_issue_portfolio_certifies_on_the_route(perspective):
    """The issue's reproducer: both formulations certify on the auto-route, with
    no fallback to the default path."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _portfolio(n=20, perspective=perspective).solve(time_limit=60)
    route = r.algorithm_route or ""
    assert r.status == "optimal" and r.gap_certified, (r.status, route)
    assert route.startswith("mip-nlp/lp_nlp_bb:"), route
    assert "fell back" not in route, route
    assert r.objective == pytest.approx(0.0104420249, rel=1e-6)
    assert r.bound <= r.objective


def test_lazy_cut_rows_below_unit_norm_are_lifted_exactly():
    """A lazy row whose largest coefficient is below 1 reaches HiGHS lifted by a
    power of two into [1, 2): HiGHS's absolute feasibility tolerance otherwise
    let a point violating an OA cut by 1.2e-2 per unit coefficient through as
    6.6e-7 raw (flay03m, rows scaled by 10^U(-6,6))."""
    coeffs = np.array([5.5e-5, -2.0e-5, 0.0, 1.0e-6])
    rhs = 3.0e-5
    lb, ub = np.zeros(4), np.ones(4)
    # Off by default: the whole-model fitting that serves the OA/GOA/GDP
    # masters (``_stack_rows``) does not lift, so those masters are unchanged.
    _, vals0, rhs0 = milp_highs._prepare_cut_row(coeffs, rhs, lb, ub, 1e-9, 1e15)
    np.testing.assert_array_equal(vals0, coeffs[np.flatnonzero(coeffs)])
    assert rhs0 == rhs
    idx, vals, row_rhs = milp_highs._prepare_cut_row(
        coeffs, rhs, lb, ub, 1e-9, 1e15, lift_to_unit=True
    )
    scale = vals[0] / coeffs[idx[0]]
    mant, _ = np.frexp(scale)
    assert mant == 0.5  # an exact power of two: the same inequality bit-for-bit
    assert 1.0 <= float(np.max(np.abs(vals))) < 2.0
    np.testing.assert_array_equal(vals, coeffs[idx] * scale)
    assert row_rhs == rhs * scale
    # A row already at unit norm is untouched.
    idx1, vals1, rhs1 = milp_highs._prepare_cut_row(
        np.array([1.5, -0.25]), 2.0, np.zeros(2), np.ones(2), 1e-9, 1e15, lift_to_unit=True
    )
    np.testing.assert_array_equal(vals1, [1.5, -0.25])
    assert rhs1 == 2.0


def test_lp_nlp_bb_never_certifies_an_unverified_incumbent():
    """``portfol_roundlot`` with rows scaled by 10^U(-3,3): LP/NLP-BB adopted a
    fixed-NLP point violating row 5 by 3.0e-3 (allowed 1e-6), certified it, and
    ``Model.solve``'s #772 guard withheld it (``status="error"``). Candidates now
    pass ``_exit_verified_incumbent`` first, as in ``solve_oa``."""
    import pathlib
    import sys
    import zlib

    from discopt.modeling.core import Constraint, from_nl
    from discopt.solvers.mip_nlp import solve_mip_nlp
    from discopt.validation.feasibility import verify_point

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from _invariance import _rebuild

    path = pathlib.Path(__file__).parent / "data" / "minlplib_nl" / "portfol_roundlot.nl"
    if not path.exists():
        path = pathlib.Path(__file__).parent / "data" / "minlplib" / "portfol_roundlot.nl"
    base = from_nl(str(path))
    model = _rebuild(base, lambda v: np.zeros(v.lb.shape), 1.0, "portfol_roundlot_pr3")
    rng = np.random.default_rng(zlib.crc32(path.name.encode()))
    model._constraints = [
        Constraint(
            body=float(10.0 ** rng.uniform(-3.0, 3.0)) * c.body,
            sense=c.sense,
            rhs=0.0,
            name=c.name,
        )
        for c in model._constraints
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = solve_mip_nlp(model, method="lp_nlp_bb", milp_solver="highs", time_limit=30)
    assert r.x is not None
    flat = np.concatenate([np.ravel(np.asarray(r.x[v.name], float)) for v in model._variables])
    verdict = verify_point(model, flat)
    unverified = bool((r.solver_stats or {}).get("oa/unverified_incumbent"))
    if r.status == "optimal" or r.gap_certified:
        assert verdict.ok, verdict.reason
        assert not unverified
    else:
        assert verdict.ok or unverified, verdict.reason


def test_lazy_master_separates_a_final_solution_no_callback_judged(monkeypatch):
    """HiGHS does not route every solution through its improving-solution
    callback (one found in presolve/postsolve is never offered), so a tree could
    finish ``optimal`` on a point the separator never saw. ``tls2`` with rows
    scaled by 10^U(-6,6), under #1667's presolve rules: the master finished at
    4.3 with the separator's only accepted point at 5.3 (optimum 5.3), and
    LP/NLP-BB returned ``feasible``. The final solution is now separated when it
    beats the best accepted point."""
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from discopt.solvers.mip_nlp import solve_mip_nlp
    from test_1537_row_scaling import _per_row_scaled

    seen = []
    real = milp_highs.solve_milp_with_lazy_cuts

    def _spy(*a, **kw):
        r = real(*a, **kw)
        seen.append(dict(r.callback_stats or {}))
        return r

    monkeypatch.setattr(milp_highs, "solve_milp_with_lazy_cuts", _spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = solve_mip_nlp(
            _per_row_scaled("tls2.nl", 6.0),
            method="lp_nlp_bb",
            milp_solver="highs",
            time_limit=60,
        )
    assert r.status == "optimal" and r.gap_certified, (r.status, r.objective, r.bound)
    assert r.objective == pytest.approx(5.3, rel=1e-6)
    assert r.bound <= r.objective + 1e-9
    assert seen and sum(int(s.get("final_offers", 0)) for s in seen) >= 1, seen
