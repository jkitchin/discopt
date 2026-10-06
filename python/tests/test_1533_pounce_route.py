"""``Model.solve(solver="pounce")``: one POUNCE interior-point solve, nothing else (#1533).

The route sends an LP to POUNCE's convex LP IPM (``lp-ipm``), a PSD QP to its
``qp-ipm``, and every other continuous model to the filter line-search NLP IPM
once. Pinned here:

* which engine each class reaches, and that nothing else does (no HiGHS, no
  convexity classification, no second NLP solve);
* certificate honesty: LP/convex QP report ``optimal``; the NLP arm reports
  ``local_optimal`` with no bound, and an engine's infeasible/unbounded verdict
  is only certified once verified;
* ``pounce_options`` reaches POUNCE, and an option the convex engine lacks is
  refused rather than dropped;
* the loud refusals (integers, GDP constraints, callbacks).
"""

from __future__ import annotations

import discopt.modeling as dm
import discopt.solver as solver_mod
import numpy as np
import pounce.qp
import pytest
from discopt.solvers import convex_ipm_pounce

pytestmark = [pytest.mark.requires_pounce]


def _lp():
    m = dm.Model("lp")
    x = m.continuous("x", shape=2, lb=0, ub=10)
    m.minimize(-x[0] - 2 * x[1])
    m.subject_to(x[0] + x[1] <= 4)
    m.subject_to(x[0] - x[1] == 1)
    return m


def _convex_qp():
    m = dm.Model("qp")
    x = m.continuous("x", shape=2, lb=-5, ub=5)
    m.minimize(x[0] ** 2 + x[1] ** 2 - 3 * x[0] - 4 * x[1])
    m.subject_to(x[0] + x[1] <= 1)
    return m


def _double_well():
    """Nonconvex NLP with a local minimum near x=+1 and the global one near x=-1."""
    m = dm.Model("well")
    x = m.continuous("x", lb=-3, ub=3)
    y = m.continuous("y", lb=-3, ub=3)
    m.minimize((x**2 - 1) ** 2 + y**2 + 0.3 * x)
    m.subject_to(x + y >= -2)
    return m, x, y


@pytest.fixture
def spy_convex(monkeypatch):
    calls: list[dict] = []
    real = pounce.qp.solve_qp

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(pounce.qp, "solve_qp", spy)
    return calls


@pytest.fixture
def spy_nlp(monkeypatch):
    from discopt.solvers import nlp_pounce

    calls: list[dict] = []
    real = nlp_pounce.solve_nlp

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(nlp_pounce, "solve_nlp", spy)
    return calls


@pytest.fixture
def no_other_route(monkeypatch):
    """Anything outside the route that a default solve would call raises."""

    def boom(*_a, **_k):
        raise AssertionError("solver='pounce' reached a non-POUNCE route")

    for name in ("_solve_lp_highs", "_solve_lp", "_solve_qp", "_classify_model_convexity"):
        monkeypatch.setattr(solver_mod, name, boom)


# ── routing ──────────────────────────────────────────────────────────────────


@pytest.mark.smoke
def test_lp_reaches_lp_ipm_and_matches_default(spy_convex, spy_nlp, no_other_route):
    res = _lp().solve(solver="pounce")
    assert res.algorithm_route == "pounce:lp-ipm"
    assert len(spy_convex) == 1 and spy_convex[0]["P"] is None  # P=None: the LP engine
    assert spy_convex[0]["method"] == "ipm"
    assert spy_nlp == []
    assert res.status == "optimal" and res.gap_certified
    assert res.objective == pytest.approx(-5.5, abs=1e-7)
    assert res.solver_stats["pounce/iterations"] >= 1
    # Duals are in discopt's convention, identical to the default route's.
    assert res.constraint_duals["c0"] == pytest.approx(1.5, abs=1e-6)
    assert res.constraint_duals["c1"] == pytest.approx(-0.5, abs=1e-6)


def test_convex_qp_reaches_qp_ipm(spy_convex, spy_nlp, no_other_route):
    res = _convex_qp().solve(solver="pounce")
    assert res.algorithm_route == "pounce:qp-ipm"
    assert len(spy_convex) == 1 and spy_convex[0]["P"] is not None
    assert spy_nlp == []
    assert res.status == "optimal"
    # min (x0-1.5)^2 + (x1-2)^2 - 6.25 on x0 + x1 <= 1: x = (0.25, 0.75).
    np.testing.assert_allclose(res.x["x"], [0.25, 0.75], atol=1e-6)
    assert res.objective == pytest.approx(-3.125, abs=1e-6)


def test_indefinite_qp_is_solved_locally_by_the_nlp_ipm(spy_nlp, no_other_route):
    m = dm.Model("ncqp")
    x = m.continuous("x", shape=2, lb=-5, ub=5)
    m.minimize(x[0] ** 2 - x[1] ** 2)
    m.subject_to(x[0] + x[1] <= 1)
    res = m.solve(solver="pounce")
    assert res.algorithm_route == "pounce:nlp"
    assert len(spy_nlp) == 1
    assert res.status == "local_optimal"
    assert res.bound is None and not res.gap_certified


@pytest.mark.smoke
def test_nonconvex_nlp_is_one_local_ipm_call(spy_nlp, no_other_route):
    m, x, y = _double_well()
    res = m.solve(solver="pounce", initial_solution={x: 2.0, y: 0.0})
    assert len(spy_nlp) == 1
    assert res.algorithm_route == "pounce:nlp"
    # The local minimum near x=+1, not the global one near x=-1 (obj ~ -0.305):
    # the route did not search, and it does not claim otherwise.
    assert res.status == "local_optimal"
    assert res.x["x"] == pytest.approx(0.96, abs=1e-2)
    assert res.objective == pytest.approx(0.2941, abs=1e-3)
    assert res.bound is None and res.gap is None and not res.gap_certified


def test_nlp_route_does_not_tighten_the_declared_box(monkeypatch, spy_nlp):
    def boom(*_a, **_k):
        raise AssertionError("bound tightening ran on solver='pounce'")

    monkeypatch.setattr(solver_mod, "_apply_nonlinear_tightening_with_status", boom)
    m, _x, _y = _double_well()
    assert m.solve(solver="pounce").status == "local_optimal"


# ── options ──────────────────────────────────────────────────────────────────


def test_pounce_options_reach_the_nlp_engine(spy_nlp):
    m, _x, _y = _double_well()
    opts = {"print_level": 0, "mu_strategy": "adaptive", "tol": 1e-9}
    m.solve(solver="pounce", pounce_options=opts)
    sent = spy_nlp[0]["options"]
    assert sent["mu_strategy"] == "adaptive" and sent["tol"] == 1e-9
    assert sent["print_level"] == 0


def test_pounce_options_reach_the_convex_engine(spy_convex, capsys):
    _lp().solve(solver="pounce", pounce_options={"tol": 1e-10, "max_iter": 50, "print_level": 1})
    call = spy_convex[0]
    # The engine solves ``sigma * objective``. It first receives the caller's
    # ``tol``; only a point that fails the caller-unit test is re-solved at
    # ``tol * sigma`` (#1537 ``engine_tol``, #1658).
    sigma = convex_ipm_pounce.objective_scale(None, np.array([-1.0, -2.0]))  # _lp's c
    assert sigma == 0.5
    assert call["tol"] == 1e-10
    assert len(spy_convex) <= 2
    if len(spy_convex) == 2:
        assert spy_convex[1]["tol"] == 1e-10 * sigma
    assert call["max_iter"] == 50
    assert spy_convex[0]["collect_iterates"] is True
    assert "lp-ipm" in capsys.readouterr().out  # print_level>0 prints the trace


def test_nlp_engine_option_on_an_lp_is_refused():
    with pytest.raises(ValueError, match="mu_strategy"):
        _lp().solve(solver="pounce", pounce_options={"mu_strategy": "adaptive"})


def test_pounce_options_is_an_alias_on_the_default_route_too(spy_nlp):
    m, _x, _y = _double_well()
    m.solve(pounce_options={"tol": 1e-9, "print_level": 0}, time_limit=20)
    assert spy_nlp and all(c["options"]["tol"] == 1e-9 for c in spy_nlp)


def test_conflicting_aliases_are_refused():
    with pytest.raises(ValueError, match="pounce_options and ipopt_options"):
        _lp().solve(solver="pounce", pounce_options={"tol": 1e-8}, ipopt_options={"tol": 1e-6})


# ── refusals ─────────────────────────────────────────────────────────────────


@pytest.mark.smoke
def test_integer_variables_are_refused():
    m = dm.Model("milp")
    x = m.integer("x", lb=0, ub=3)
    m.minimize(x)
    with pytest.raises(ValueError, match="integer or binary"):
        m.solve(solver="pounce")


def test_disjunctive_constraints_are_refused():
    m = dm.Model("gdp")
    x = m.continuous("x", lb=0, ub=10)
    m.minimize(x)
    m.either_or([[x <= 2], [x >= 8]])
    with pytest.raises(ValueError, match="logical/disjunctive"):
        m.solve(solver="pounce")


def test_feasibility_callbacks_are_refused():
    with pytest.raises(ValueError, match="pounce"):
        _lp().solve(solver="pounce", lazy_constraints=lambda *a: [])


# ── certificates ─────────────────────────────────────────────────────────────


def test_infeasible_lp_is_certified_after_verification():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=1)
    m.minimize(x)
    m.subject_to(x >= 2)
    assert m.solve(solver="pounce").status == "infeasible"


def test_unverified_infeasibility_is_not_certified(monkeypatch):
    """POUNCE's own verdict alone never becomes a certificate."""
    monkeypatch.setattr(convex_ipm_pounce, "_simplex_feasibility_verdict", lambda *a: "undecided")
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=1)
    m.minimize(x)
    m.subject_to(x >= 2)
    res = m.solve(solver="pounce")
    assert res.status == "error" and res.error


def test_unbounded_lp_with_infinite_bounds():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=np.inf)
    y = m.continuous("y", lb=0, ub=np.inf)
    m.minimize(-x)
    m.subject_to(x - y <= 1)
    assert m.solve(solver="pounce").status == "unbounded"


def test_unbounded_over_the_default_box_is_not_certified():
    """The 9.999e19 default box is finite as declared; 'unbounded' would be false."""
    m = dm.Model()
    x = m.continuous("x", lb=0)
    m.minimize(-x)
    m.subject_to(x >= 1)
    res = m.solve(solver="pounce")
    assert res.status == "error"
    assert "9.999e19" in res.error


# ── #1539 review ─────────────────────────────────────────────────────────────


@pytest.mark.smoke
def test_slightly_indefinite_qp_is_not_certified():
    """Review blocker: POUNCE's PSD tolerance (1e-8 * max|P|) admits this Hessian,
    and the convex IPM stops at the saddle x=0, which used to be certified optimal
    (objective 0) on a problem whose optimum is -0.1."""
    m = dm.Model("saddle")
    x = m.continuous("x", lb=-1e4, ub=1e4)
    y = m.continuous("y", lb=-1, ub=1)
    m.minimize(-1e-9 * x**2 + y**2)
    res = m.solve(solver="pounce")
    assert res.algorithm_route == "pounce:nlp"
    assert res.status != "optimal"
    assert res.bound is None and not res.gap_certified


def test_certify_psd_is_exact_on_small_matrices():
    from discopt.solvers.convex_ipm_pounce import certify_psd

    cases = [
        (np.diag([-2e-9, 2.0]), False),  # the review's Hessian
        (np.diag([0.0, 2.0]), True),  # a variable that appears only linearly
        (np.array([[1.0, -1.0], [-1.0, 1.0]]), True),  # (x - y)**2, singular PSD
        (np.array([[1.0, 1.0], [1.0, 1.0 - 1e-15]]), False),  # det < 0 by 1e-15
        (np.array([[0.0, 1.0], [1.0, 0.0]]), False),  # x*y
        (np.array([[1.0, 0.0], [0.0, np.nan]]), False),
    ]
    for Q, expected in cases:
        assert certify_psd(Q) is expected, Q
    assert len(cases) == 6


def test_certify_psd_eigen_fallback_above_the_exact_cap():
    from discopt.solvers import convex_ipm_pounce as cvx

    n = cvx._EXACT_PSD_MAX_N + 10
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n))
    pd = A.T @ A + np.eye(n)
    assert cvx.certify_psd(pd) is True
    bad = pd.copy()
    bad[0, 0] = -1e-6  # a tiny negative diagonal makes it indefinite
    assert cvx.certify_psd(bad) is False


def test_local_infeasibility_is_local_infeasible_not_error():
    m = dm.Model("noroom")
    x = m.continuous("x", lb=-1, ub=1)
    m.minimize(x)
    m.subject_to(x**2 >= 4)
    res = m.solve(solver="pounce")
    assert res.status == "local_infeasible"
    assert res.bound is None and not res.gap_certified


@pytest.mark.parametrize("nlp_solver", ["ipm", "sparse_ipm", "ipopt"])
def test_other_nlp_solver_values_are_refused(nlp_solver):
    with pytest.raises(ValueError, match="nlp_solver"):
        _lp().solve(solver="pounce", nlp_solver=nlp_solver)


def test_every_ignored_option_is_named():
    with pytest.warns(UserWarning, match="ignores.*skip_convex_check") as rec:
        _lp().solve(solver="pounce", skip_convex_check=True, presolve=False)
    msg = " ".join(str(w.message) for w in rec)
    assert "presolve" in msg


def test_a_plain_call_warns_about_nothing():
    import warnings

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _lp().solve(solver="pounce")
    assert not [w for w in rec if "solver='pounce' ignores" in str(w.message)]


def test_ignored_option_table_tracks_the_signature():
    import inspect

    from discopt.solver import _POUNCE_ROUTE_HONOURED, solve_model

    params = set(inspect.signature(solve_model).parameters)
    assert _POUNCE_ROUTE_HONOURED <= params
    assert len(params - _POUNCE_ROUTE_HONOURED) >= 40


def _relabel_inaccurate(monkeypatch, perturb=0.0):
    """Make every qp-ipm solve report ``optimal_inaccurate`` (POUNCE gh#984).

    pounce ``main`` re-judges an ``optimal`` IPM iterate on its normalized KKT
    measure and reports ``optimal_inaccurate`` when it is above ``tol``; the 0.12.0
    wheel did not. Relabelling pins the route's handling on either build. Returns
    the list of statuses handed to the route, so a test can prove the relabel fired.
    """
    import dataclasses

    real = pounce.qp.solve_qp
    seen: list[str] = []

    def relabel(**kw):
        res = real(**kw)
        if res.status in ("optimal", "optimal_inaccurate"):
            res = dataclasses.replace(
                res, status="optimal_inaccurate", x=np.asarray(res.x) + perturb
            )
        seen.append(res.status)
        return res

    monkeypatch.setattr(pounce.qp, "solve_qp", relabel)
    return seen


def _scaled_row_qp(scale):
    m = dm.Model("scaled")
    x = m.continuous("x", lb=-10, ub=10)
    y = m.continuous("y", lb=-10, ub=10)
    m.subject_to(scale * (x + y) >= scale * 2.0)
    m.minimize(x**2 + 2 * y**2)
    return m, x, y


@pytest.mark.parametrize("scale", [1e-6, 1.0])
def test_optimal_inaccurate_qp_is_a_candidate_not_an_iteration_limit(monkeypatch, scale):
    """An ``optimal_inaccurate`` qp-ipm point goes through the #1596 certificate.

    It was mapped to ``iteration_limit`` and dropped -- after 17 iterations, with no
    cap hit -- which is what CI's Linux runner (pounce ``main``) reported for the
    1e-6 row of #1617 while macOS (the 0.12.0 wheel) returned the same iterate as
    ``optimal``. Whether it is ``optimal`` or ``feasible`` is now the certificate's
    call, as it is for an ``optimal`` label.
    """
    seen = _relabel_inaccurate(monkeypatch)
    m, x, y = _scaled_row_qp(scale)
    r = m.solve(solver="pounce")
    assert seen == ["optimal_inaccurate"]
    assert r.algorithm_route == "pounce:qp-ipm"
    assert r.status in ("optimal", "feasible"), (r.status, r.error)
    np.testing.assert_allclose([r.x["x"], r.x["y"]], [4 / 3, 2 / 3], rtol=1e-6)
    assert r.objective == pytest.approx(8 / 3, rel=1e-7)
    if r.status == "optimal":
        assert r.bound is not None and r.bound <= r.objective + 1e-9
    else:
        assert r.bound is None and r.gap_certified is False


def test_optimal_inaccurate_qp_still_faces_the_feasibility_guard(monkeypatch):
    """The candidate is not trusted: an infeasible ``optimal_inaccurate`` is refused."""
    seen = _relabel_inaccurate(monkeypatch, perturb=-0.1)  # x + y = 1.8 < 2
    m, _, _ = _scaled_row_qp(1.0)
    r = m.solve(solver="pounce")
    assert seen == ["optimal_inaccurate"]
    assert r.status == "error"
    assert "infeasible point" in (r.error or "")
