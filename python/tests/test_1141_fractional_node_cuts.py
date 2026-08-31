"""#1141 — fractional-node cut separation on the in-house MILP driver.

Before this change the only backend that could separate cuts at a *fractional*
node relaxation was Gurobi (MIPNODE). ``milp_simplex`` and ``milp_highs`` both
refused ``node_callback``, so on the certified-convex class every node either paid
a full NLP (measured 38.9 ms/node against SCIP's 3.0) or ran on a relaxation that
ignored the nonlinear constraint. The Rust driver now has the hook, so the
Quesada–Grossmann LP/NLP branch-and-cut runs in-house.

Every end-to-end assertion here checks the separator actually fired
(``mipnode_calls``/``driver_node_cuts``): a run that never calls the hook would
still answer these models correctly off the lazy separator alone and would read as
a pass while testing nothing (CLAUDE.md §6).
"""

import itertools

import numpy as np
import pytest
from discopt.solvers import SolveStatus
from discopt.solvers.milp_simplex import solve_milp, solve_milp_with_lazy_cuts

# ---------------------------------------------------------------------------
# a tiny convex MINLP whose nonlinear row lives ONLY in the separators
# ---------------------------------------------------------------------------

_R2 = 5.0
_LO, _HI = -3.0, 3.0


def _ball_cut(x):
    """Tangent of the convex ``g(x) = x0² + x1² − R²`` at ``x`` (globally valid)."""
    x = np.asarray(x, dtype=float)
    g = float(x[0] ** 2 + x[1] ** 2 - _R2)
    if g <= 1e-9:
        return []
    grad = 2.0 * x[:2]
    return [(grad, float(grad @ x[:2] - g))]


def _ball_kwargs():
    return dict(
        c=np.array([-1.0, -1.0]),
        # One inert row: the master needs a matrix, and the ball lives in the
        # callbacks so a run that never separates cannot answer correctly.
        A_ub=np.zeros((1, 2)),
        b_ub=np.array([1.0]),
        bounds=[(_LO, _HI), (_LO, _HI)],
        integrality=np.array([1, 1]),
        lazy_callback=_ball_cut,
        time_limit=60.0,
        gap_tolerance=1e-9,
    )


@pytest.mark.smoke
def test_node_hook_fires_and_is_counted():
    r = solve_milp_with_lazy_cuts(
        **_ball_kwargs(), node_callback=_ball_cut, node_hook_rounds=4, node_hook_cut_cap=200
    )
    stats = dict(r.callback_stats or {})
    # Anti-vacuity first: without these the assertions below are about a run that
    # never separated at a fractional point.
    assert stats["mipnode_calls"] > 0, "the fractional separator never fired"
    assert stats["driver_node_cuts"] > 0, "the fractional separator added no rows"
    assert stats["driver_node_calls"] == stats["mipnode_calls"]
    assert r.status == SolveStatus.OPTIMAL
    assert r.x is not None
    assert float(r.x[0] ** 2 + r.x[1] ** 2) <= _R2 + 1e-6
    assert r.objective == pytest.approx(-3.0, abs=1e-6)


@pytest.mark.smoke
def test_no_node_hook_means_no_node_calls():
    """Absent hook ⇒ bit-identical search: the counters must stay at zero."""
    r = solve_milp_with_lazy_cuts(**_ball_kwargs())
    stats = dict(r.callback_stats or {})
    assert stats["mipnode_calls"] == 0
    assert stats["node_cuts"] == 0
    assert stats["driver_node_cuts"] == 0
    assert r.objective == pytest.approx(-3.0, abs=1e-6)


@pytest.mark.smoke
def test_zero_budget_node_callback_is_refused_not_silently_dropped():
    # A separator that can never fire reports ``mipnode_calls == 0``, which is
    # indistinguishable from one that ran and found nothing (CLAUDE.md §6).
    with pytest.raises(ValueError, match="node_hook_rounds"):
        solve_milp_with_lazy_cuts(**_ball_kwargs(), node_callback=_ball_cut, node_hook_rounds=0)
    with pytest.raises(ValueError, match="node_hook_cut_cap"):
        solve_milp_with_lazy_cuts(**_ball_kwargs(), node_callback=_ball_cut, node_hook_cut_cap=0)


@pytest.mark.smoke
def test_node_separator_failure_propagates():
    """A separator that raises must stop the solve, not read as 'found nothing'."""

    def boom(_x):
        raise ZeroDivisionError("node separator blew up")

    with pytest.raises(Exception, match="node separator blew up"):
        solve_milp_with_lazy_cuts(
            **_ball_kwargs(), node_callback=boom, node_hook_rounds=2, node_hook_cut_cap=50
        )


@pytest.mark.smoke
def test_node_separator_rejects_an_over_wide_row():
    """A row longer than the structural width would address the driver's slacks."""

    def too_wide(_x):
        return [(np.ones(5), 1.0)]

    with pytest.raises(ValueError, match="coefficients"):
        solve_milp_with_lazy_cuts(
            **_ball_kwargs(), node_callback=too_wide, node_hook_rounds=2, node_hook_cut_cap=50
        )


# ---------------------------------------------------------------------------
# soundness: certified answer vs brute force, with the hook on
# ---------------------------------------------------------------------------


def _random_convex_binary_minlp(rng, n=10):
    """min cᵀx over binaries, with one convex quadratic row held in the separator."""
    F = rng.normal(size=(n, 3))
    Q = F @ F.T / 3 + np.diag(0.5 + rng.random(n))
    q = rng.normal(size=n)
    rhs = float(np.trace(Q) / n * rng.uniform(1.5, 4.0))
    c = rng.normal(size=n).round(3)
    A = rng.normal(size=(2, n)).round(3)
    b = (np.abs(A) @ np.full(n, 0.6)).round(3)

    def g(x):
        x = np.asarray(x, float)
        return float(x @ Q @ x + q @ x - rhs)

    def cut(x):
        x = np.asarray(x, float)
        gv = g(x)
        if gv <= 1e-9:
            return []
        grad = 2.0 * (Q @ x) + q
        return [(grad, float(grad @ x - gv))]

    best = None
    for pt in itertools.product((0.0, 1.0), repeat=n):
        p = np.array(pt)
        if np.any(A @ p > b + 1e-9) or g(p) > 1e-9:
            continue
        v = float(c @ p)
        best = v if best is None or v < best else best
    return c, A, b, cut, best


@pytest.mark.smoke
def test_fractional_separation_certifies_the_brute_force_optimum():
    """The cuts are supporting hyperplanes, so the certificate must not move."""
    rng = np.random.default_rng(20260830)
    fired = 0
    checked = 0
    for _ in range(6):
        c, A, b, cut, best = _random_convex_binary_minlp(rng)
        if best is None:
            continue  # infeasible draw: nothing to certify against
        n = c.shape[0]
        base = dict(
            c=c,
            A_ub=A,
            b_ub=b,
            bounds=[(0.0, 1.0)] * n,
            integrality=np.ones(n, int),
            lazy_callback=cut,
            time_limit=60.0,
            gap_tolerance=1e-9,
        )
        off = solve_milp_with_lazy_cuts(**base)
        on = solve_milp_with_lazy_cuts(
            **base, node_callback=cut, node_hook_rounds=3, node_hook_cut_cap=500
        )
        fired += int(dict(on.callback_stats or {})["mipnode_calls"] > 0)
        for arm in (off, on):
            checked += 1
            assert arm.objective == pytest.approx(best, rel=1e-6, abs=1e-6)
            assert arm.bound is not None
            # The invariant that matters: a dual bound above the true optimum is a
            # false certificate, however good the incumbent looks.
            assert arm.bound <= best + 1e-6 * max(1.0, abs(best))
    assert checked > 0, "no draw was certifiable; the test measured nothing"
    assert fired > 0, "the fractional separator never fired on any draw"


# ---------------------------------------------------------------------------
# the bound-validity defect #1141 exposed
# ---------------------------------------------------------------------------


def _small_objective_knapsack():
    """A MILP whose optimum has magnitude ~0.5, so the engine's ``max(|U|, 1)``
    gap denominator makes a *relative* tolerance an *absolute* one."""
    rng = np.random.default_rng(5)
    n = 40
    w = rng.integers(3, 40, size=n).astype(float)
    p = rng.integers(3, 40, size=n).astype(float) / 1000.0
    a2 = rng.integers(1, 20, size=n).astype(float)
    A = np.vstack([w, a2])
    b = np.array([w.sum() * 0.35, 0.4 * a2.sum()])
    return -p, A, b, [(0.0, 1.0)] * n, np.ones(n, int)


@pytest.mark.smoke
def test_optimal_exit_publishes_the_dual_bound_not_the_incumbent():
    """``"optimal"`` from the driver means optimal WITHIN ``gap_tolerance``.

    Publishing ``bound = objective`` there over-states the dual bound by up to the
    tolerance — and the engine normalises its gap by ``max(|incumbent|, 1.0)``, so
    on a small-magnitude objective a 1 % "relative" tolerance is 1 % absolute,
    i.e. ~2 % relative. This asserts the published bound never rises above the
    true optimum, which is what a dual bound means.
    """
    c, A, b, bounds, integrality = _small_objective_knapsack()
    exact = solve_milp(
        c=c,
        A_ub=A,
        b_ub=b,
        bounds=bounds,
        integrality=integrality,
        time_limit=120.0,
        gap_tolerance=1e-12,
    )
    assert exact.status == SolveStatus.OPTIMAL and exact.objective is not None
    optimum = float(exact.objective)

    loose = solve_milp(
        c=c,
        A_ub=A,
        b_ub=b,
        bounds=bounds,
        integrality=integrality,
        time_limit=120.0,
        gap_tolerance=1e-2,
    )
    assert loose.status == SolveStatus.OPTIMAL
    assert loose.bound is not None
    # The regression: before the fix this was `bound = objective = -0.542`, i.e.
    # 1.4e-3 ABOVE the true optimum -0.543 — a false lower bound.
    assert loose.bound <= optimum + 1e-9, (
        f"published dual bound {loose.bound!r} is above the true optimum {optimum!r}"
    )
    assert loose.objective is not None and loose.objective >= optimum - 1e-9
    # And the reported gap must describe that interval rather than claim zero.
    assert loose.gap is None or loose.gap >= 0.0


@pytest.mark.smoke
def test_lazy_optimal_exit_publishes_the_dual_bound_not_the_incumbent():
    """Same invariant on the lazy entry point, which is what the OA master uses."""
    c, A, b, bounds, integrality = _small_objective_knapsack()
    exact = solve_milp(
        c=c,
        A_ub=A,
        b_ub=b,
        bounds=bounds,
        integrality=integrality,
        time_limit=120.0,
        gap_tolerance=1e-12,
    )
    optimum = float(exact.objective)
    loose = solve_milp_with_lazy_cuts(
        c=c,
        A_ub=A,
        b_ub=b,
        bounds=bounds,
        integrality=integrality,
        lazy_callback=lambda _x: [],
        time_limit=120.0,
        gap_tolerance=1e-2,
    )
    assert loose.status == SolveStatus.OPTIMAL
    assert loose.bound is not None
    assert loose.bound <= optimum + 1e-9, (
        f"published dual bound {loose.bound!r} is above the true optimum {optimum!r}"
    )
    assert dict(loose.callback_stats or {})["mipsol_calls"] > 0


# ---------------------------------------------------------------------------
# #1141 item 2: stop collapsing every fixed-NLP failure into one word
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_ipopt_code_2_is_not_mapped_to_a_global_infeasibility_verdict():
    """Restoration converging to a local violation minimum is not a proof.

    Ipopt code 2 (``Infeasible_Problem_Detected``) says the algorithm is stuck at
    an infeasible point. On a convex subproblem that is a genuine proof; on a
    nonconvex one it is not, and the map serves both — including the pure-NLP path
    in ``solve_model``, which would otherwise publish ``status="infeasible"`` for a
    model it never proved infeasible. The raw code is carried separately for
    callers that hold a convexity certificate.
    """
    from discopt.solvers.nlp_ipopt import _IPOPT_STATUS_MAP, IPOPT_LOCALLY_INFEASIBLE

    assert IPOPT_LOCALLY_INFEASIBLE == 2
    assert _IPOPT_STATUS_MAP[IPOPT_LOCALLY_INFEASIBLE] is not SolveStatus.INFEASIBLE


@pytest.mark.smoke
def test_nlp_result_carries_the_subsolvers_own_code():
    import discopt.modeling as dm
    from discopt._tape_nlp_evaluator import make_evaluator
    from discopt.solvers.nlp_pounce import solve_nlp

    m = dm.Model("tiny")
    v = m.continuous("v", lb=0.0, ub=1.0)
    m.subject_to(v <= 0.5)
    m.minimize((v - 0.25) ** 2)
    r = solve_nlp(make_evaluator(m), np.array([0.4]))
    assert r.raw_status is not None, "the subsolver's own terminal code was dropped"
    assert r.status == SolveStatus.OPTIMAL


@pytest.mark.smoke
def test_fixed_nlp_status_label_names_the_outcome():
    from discopt.solvers.oa import _fixed_nlp_status_label, _NLPAttempt

    assert _fixed_nlp_status_label(_NLPAttempt(x=np.zeros(1), objective=0.0, multipliers=None)) == (
        "feasible"
    )
    # The two outcomes that used to be one word: a genuinely infeasible assignment
    # and a subsolver that fell over are different problems with different fixes.
    infeasible = _NLPAttempt(
        x=None, objective=None, multipliers=None, status=SolveStatus.ERROR, raw_status=2
    )
    broke = _NLPAttempt(
        x=None, objective=None, multipliers=None, status=SolveStatus.ERROR, raw_status=-3
    )
    assert _fixed_nlp_status_label(infeasible) == "infeasible_local"
    assert _fixed_nlp_status_label(broke) == "failed:-3"
    # No code at all (the subsolver raised) still reports honestly.
    empty = _NLPAttempt(x=None, objective=None, multipliers=None)
    assert _fixed_nlp_status_label(empty) == "failed"


# ---------------------------------------------------------------------------
# #1141 item 3: the feasibility subproblem's formulation
# ---------------------------------------------------------------------------


def _mixed_sense_model():
    """Convex rows of all three senses, so the elastic rewrite is exercised fully."""
    import discopt.modeling as dm

    m = dm.Model("elastic")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    m.subject_to(x * x + y * y <= 1.0)
    m.subject_to(x + 2 * y >= 0.5)
    m.subject_to(dm.exp(x) - y == 0.25)
    m.minimize(x + y)
    return m


@pytest.mark.smoke
@pytest.mark.parametrize("norm", ["L1", "L2", "L_infinity"])
def test_elastic_restoration_derivatives_match_finite_differences(norm):
    """A wrong elastic model would not raise — it would restore to the wrong point.

    So the Jacobian, gradient and Lagrangian Hessian are checked against finite
    differences of this class's own constraint and objective functions.
    """
    from discopt._tape_nlp_evaluator import make_evaluator
    from discopt.solvers.oa import _ElasticFeasibilityEvaluator

    ev = make_evaluator(_mixed_sense_model())
    lo, hi = np.array([-2.0, -2.0]), np.array([2.0, 2.0])
    el = _ElasticFeasibilityEvaluator(ev, lo, hi, norm)
    z = el.start_point(np.array([0.7, 0.9]))

    assert el.n_constraints == 4, "an equality must contribute BOTH elastic rows"
    assert el.n_variables == (2 + (1 if norm == "L_infinity" else 3))

    h = 1e-6
    g0 = el.evaluate_constraints(z)
    jac = el.evaluate_jacobian(z)
    jac_fd = np.column_stack(
        [
            (el.evaluate_constraints(z + h * np.eye(el.n_variables)[j]) - g0) / h
            for j in range(el.n_variables)
        ]
    )
    assert np.abs(jac - jac_fd).max() < 1e-4

    f0 = el.evaluate_objective(z)
    grad = el.evaluate_gradient(z)
    grad_fd = np.array(
        [
            (el.evaluate_objective(z + h * np.eye(el.n_variables)[j]) - f0) / h
            for j in range(el.n_variables)
        ]
    )
    assert np.abs(grad - grad_fd).max() < 1e-4

    rng = np.random.default_rng(0)
    lam = rng.normal(size=el.n_constraints)
    sigma = 1.3
    hess = el.evaluate_lagrangian_hessian(z, sigma, lam)

    def lagrangian(zz):
        return sigma * el.evaluate_objective(zz) + float(lam @ el.evaluate_constraints(zz))

    hh = 1e-5
    n = el.n_variables
    hess_fd = np.zeros((n, n))
    eye = np.eye(n)
    for i in range(n):
        for j in range(n):
            hess_fd[i, j] = (
                lagrangian(z + hh * eye[i] + hh * eye[j])
                - lagrangian(z + hh * eye[i] - hh * eye[j])
                - lagrangian(z - hh * eye[i] + hh * eye[j])
                + lagrangian(z - hh * eye[i] - hh * eye[j])
            ) / (4 * hh * hh)
    assert np.abs(hess - hess_fd).max() < 1e-3
    # The defect this class exists for: the shipped merit formulation reports an
    # identically zero Hessian, which is what makes the KKT system singular.
    assert np.abs(hess).max() > 0.0


@pytest.mark.smoke
@pytest.mark.parametrize("norm", ["L1", "L2", "L_infinity"])
def test_elastic_restoration_start_point_is_feasible(norm):
    """The IPM starts inside its own feasible set, so any progress reduces violation."""
    from discopt._tape_nlp_evaluator import make_evaluator
    from discopt.solvers.oa import _ElasticFeasibilityEvaluator

    ev = make_evaluator(_mixed_sense_model())
    el = _ElasticFeasibilityEvaluator(ev, np.array([-2.0, -2.0]), np.array([2.0, 2.0]), norm)
    for x0 in (np.array([0.7, 0.9]), np.array([-1.5, 1.9]), np.array([0.0, 0.0])):
        z = el.start_point(x0)
        rows = el.evaluate_constraints(z)
        lo = np.array([b[0] for b in el.constraint_bounds()])
        hi = np.array([b[1] for b in el.constraint_bounds()])
        assert np.all(rows <= hi + 1e-9) and np.all(rows >= lo - 1e-9), (
            f"elastic start is infeasible for {norm} at {x0}"
        )


@pytest.mark.smoke
def test_merit_formulation_reports_a_zero_hessian():
    """Pins the defect itself, so a future rewrite of the merit path is noticed."""
    from discopt._tape_nlp_evaluator import make_evaluator
    from discopt.solvers.oa import _FeasibilityEvaluator

    ev = make_evaluator(_mixed_sense_model())
    merit = _FeasibilityEvaluator(ev, np.array([-2.0, -2.0]), np.array([2.0, 2.0]), "L1")
    z = np.array([0.7, 0.9])
    assert merit.n_constraints == 0
    assert np.abs(np.asarray(merit.evaluate_lagrangian_hessian(z, 1.0, np.empty(0)))).max() == 0.0


# ---------------------------------------------------------------------------
# a false certificate #1141's corpus panel uncovered (pre-existing)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize("method", ["lp_nlp_bb", "oa"])
def test_integer_free_nonconvex_model_is_not_certified_optimal(method):
    """One local NLP solve is a global proof only on a CONVEX model.

    An integer-free OA loop is a single NLP solve, and both drivers reported it as
    ``status="optimal", bound=objective, gap=0.0`` unconditionally. On MINLPLib
    ``trig`` — one continuous variable on ``[-2, 5]``, one nonconvex row — that
    returned ``-2.479027828`` as *optimal* while the true minimum over the declared
    box is ``-3.762500358`` (MINLPLib's value; reproduced here by brute force in
    ``scratchpad/1141/``). A local minimum was handed back as a certificate.
    """
    from discopt.modeling.core import from_nl

    model = from_nl("python/tests/data/minlplib/trig.nl")
    r = model.solve(
        solver="mip-nlp",
        mip_nlp_method=method,
        milp_solver="simplex",
        time_limit=60,
        gap_tolerance=1e-4,
    )
    true_min = -3.762500358
    assert r.objective is not None
    # The point found is a genuine local minimum well above the global one, which
    # is exactly why certifying it would be a false claim.
    assert r.objective > true_min + 1e-3
    assert r.status != "optimal", f"{method} certified a local minimum as optimal"
    assert r.bound is None, f"{method} published {r.bound!r} as a dual bound"
    assert getattr(r, "gap_certified", False) is False


@pytest.mark.smoke
@pytest.mark.parametrize("method", ["lp_nlp_bb", "oa"])
def test_integer_free_convex_model_still_certifies(method):
    """The guard must not cost a certificate a convex model has actually earned."""
    import discopt.modeling as dm

    m = dm.Model("convex_continuous")
    x = m.continuous("x", lb=-3.0, ub=3.0)
    y = m.continuous("y", lb=-3.0, ub=3.0)
    m.subject_to(x * x + y * y <= 4.0)
    m.subject_to(x + y >= -1.0)
    m.minimize((x - 0.5) ** 2 + (y + 0.25) ** 2)
    r = m.solve(
        solver="mip-nlp",
        mip_nlp_method=method,
        milp_solver="simplex",
        time_limit=60,
        gap_tolerance=1e-6,
    )
    assert r.status == "optimal", f"{method} lost a certificate it had earned"
    assert r.bound is not None
    assert r.objective == pytest.approx(0.0, abs=1e-6)
