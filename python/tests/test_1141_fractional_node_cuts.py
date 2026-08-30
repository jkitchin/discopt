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
