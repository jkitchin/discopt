"""Unbounded McCormick relaxation must not fabricate a finite lower bound (#24).

A McCormick/RLT envelope for a product/power term is only a valid relaxation when
its variables have FINITE bounds. Over a box where a nonlinear-term variable is
still unbounded (e.g. the root box before FBBT could bound it), the lifted aux is
effectively free and the relaxation is genuinely UNBOUNDED -- it carries no valid
finite lower bound. HiGHS correctly reports "unbounded"; the fast warm-started
Rust simplex instead mis-handles the unbounded ray and fabricates a finite
"optimal" (on himmel16's root relaxation the simplex returns 0.0 / with RLT cuts
-0.6749 where HiGHS returns "unbounded"). Trusting that finite value as a lower
bound is a too-high dual bound: it fathoms feasible nodes and certifies a
suboptimal incumbent -- a false-"optimal", the worst failure class.

himmel16 (the "largest small hexagon" with only pairwise-diameter constraints)
admits a self-intersecting doubly-traced equilateral triangle of area 0.866, so
its true optimum is objvar = -0.866; discopt previously certified -0.6749 (the
*convex* hexagon optimum) as global. These tests pin the two soundness guards
that close that gap: ``MccormickLPRelaxer.solve_at_node`` cross-checks an
unbounded relaxation with HiGHS, and ``_root_relaxation_lower_bound`` requires an
OPTIMAL relaxation solve before surfacing a fallback bound.
"""

from __future__ import annotations

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import discopt
import discopt._jax.mccormick_lp as mc
import numpy as np
import pytest


def _bilinear_obj():
    """Minimize a bilinear objective; the unbounded box makes x*y unbounded."""
    m = discopt.Model("bil_unbounded")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    m.subject_to(x + y >= 0.5)
    m.minimize(x * y - x - y)
    return m


def test_solve_at_node_unbounded_box_returns_no_finite_bound():
    """Directly: over an UNBOUNDED box the bilinear relaxation is unbounded, so
    ``solve_at_node`` must NOT return a finite ``lower_bound`` -- even though the
    fast simplex fabricates a finite "optimal" there. A fabricated finite bound
    would be a too-high dual bound and certify a suboptimal incumbent."""
    relaxer = mc.MccormickLPRelaxer(_bilinear_obj())

    # Bounded box: a real finite LP bound is fine (relaxation is bounded).
    lb_f = np.array([-2.0, -2.0], dtype=np.float64)
    ub_f = np.array([2.0, 2.0], dtype=np.float64)
    bounded = relaxer.solve_at_node(lb_f, ub_f, time_limit=5.0)
    assert bounded.status != "skipped_oversize"

    # Unbounded box: the McCormick envelope of x*y is free -> the LP is unbounded.
    # The guard must decline (no finite lower bound), never fabricate one.
    lb_u = np.array([-np.inf, -np.inf], dtype=np.float64)
    ub_u = np.array([np.inf, np.inf], dtype=np.float64)
    unb = relaxer.solve_at_node(lb_u, ub_u, time_limit=5.0)
    assert unb.lower_bound is None, (
        f"unbounded relaxation fabricated a finite bound {unb.lower_bound} (status={unb.status})"
    )


def test_has_unbounded_nonlinear_col_flags_free_bilinear_var():
    """The gate fires iff a nonlinear-participating column is non-finite."""
    relaxer = mc.MccormickLPRelaxer(_bilinear_obj())
    assert relaxer._nonlinear_cols  # x*y registers both columns

    class _Milp:
        def __init__(self, bounds):
            self._bounds = bounds

    finite = _Milp([(-2.0, 2.0), (-2.0, 2.0)])
    assert not relaxer._has_unbounded_nonlinear_col(finite)

    free = _Milp([(-np.inf, 2.0), (-2.0, 2.0)])
    assert relaxer._has_unbounded_nonlinear_col(free)


@pytest.mark.slow
def test_himmel16_no_false_certification():
    """End-to-end: the largest-small-hexagon model with only pairwise-diameter
    constraints has a feasible self-intersecting solution at objvar=-0.866. The
    solver must EITHER report a bound <= the incumbent (sound), OR not certify --
    it must never certify a value above -0.866."""
    m = _himmel16_model()
    r = m.solve(time_limit=40.0)
    # discopt finds the true optimum (-0.866) via local search.
    assert r.objective is not None
    assert r.objective <= -0.86 + 1e-3, f"missed the -0.866 incumbent: {r.objective}"
    # Soundness: a certified bound can never sit above the incumbent it certifies.
    if getattr(r, "gap_certified", False):
        assert r.bound is not None and r.bound <= r.objective + 1e-4, (
            f"certified an invalid (above-incumbent) bound: "
            f"bound={r.bound} > objective={r.objective}"
        )
    # A reported finite bound must be a valid lower bound (<= incumbent).
    if r.bound is not None and np.isfinite(r.bound):
        assert r.bound <= r.objective + 1e-4, (
            f"surfaced an invalid lower bound {r.bound} > incumbent {r.objective}"
        )


def _himmel16_model():
    """himmel16 (GLOBALLib): maximize hexagon area (min objvar=-area) with all
    pairwise vertex distances <= 1. Vertices (x_i, x_{i+6}), i=1..6; x1=x7=x8=0
    fixed; coordinates otherwise free. The shoelace area terms are bilinear."""
    m = discopt.Model("himmel16")
    x = {i: m.continuous(f"x{i}", lb=-1e20, ub=1e20) for i in range(1, 19)}
    objvar = m.continuous("objvar", lb=-1e20, ub=1e20)
    # x1, x7, x8 fixed at 0
    for i in (1, 7, 8):
        m.subject_to(x[i] == 0.0)
    # pairwise diameter constraints e1..e15: sqr(xa-xb)+sqr(xc-xd) <= 1
    pairs = [
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (2, 3),
        (2, 4),
        (2, 5),
        (2, 6),
        (3, 4),
        (3, 5),
        (3, 6),
        (4, 5),
        (4, 6),
        (5, 6),
    ]
    for a, b in pairs:
        m.subject_to((x[a] - x[b]) ** 2 + (x[a + 6] - x[b + 6]) ** 2 <= 1.0)
    # shoelace area pieces x13..x18 (e17..e22)
    m.subject_to(-0.5 * (x[1] * x[8] - x[7] * x[2]) + x[13] == 0.0)
    m.subject_to(-0.5 * (x[2] * x[9] - x[8] * x[3]) + x[14] == 0.0)
    m.subject_to(-0.5 * (x[3] * x[10] - x[9] * x[4]) + x[15] == 0.0)
    m.subject_to(-0.5 * (x[4] * x[11] - x[10] * x[5]) + x[16] == 0.0)
    m.subject_to(-0.5 * (x[5] * x[12] - x[11] * x[6]) + x[17] == 0.0)
    m.subject_to(-0.5 * (x[6] * x[7] - x[12] * x[1]) + x[18] == 0.0)
    # e16: objvar = -(x13+...+x18)
    m.subject_to(-(x[13] + x[14] + x[15] + x[16] + x[17] + x[18]) - objvar == 0.0)
    m.minimize(objvar)
    return m


def _camel_semiinfinite(lb0, ub0, lb1, ub1, name):
    """Six-hump-camel-style nonconvex objective (2*x0^2 - 1.05*x0^4 + x0^6/6
    - x0*x1 + x1^2) over a box that may leave a variable one-sided-unbounded.

    The true global minimum over the whole plane is 0.0 at the origin. On a
    finite box the spatial B&B certifies it; on a semi-infinite box a
    continuous variable has an unbounded side, so the ROOT cannot be spatially
    branched (no finite branching direction) and no relaxation establishes a
    finite dual bound. The tree must then NOT certify the local minimum 0.2986
    (or any local point) as the global optimum.
    """
    m = discopt.Model(name)
    x0 = m.continuous("x0", lb=lb0, ub=ub0)
    x1 = m.continuous("x1", lb=lb1, ub=ub1)
    m.minimize(2 * x0**2 - 1.05 * x0**4 + (1.0 / 6.0) * x0**6 - x0 * x1 + x1**2)
    return m


@pytest.mark.smoke
def test_467_free_variable_root_not_falsely_optimal_nlp_route():
    """#467: both continuous vars one-sided-unbounded (x0=[-5,inf], x1=[-inf,5]).

    ``_origin_has_finite_continuous_var`` is False, so the McCormick-LP guard
    falls the solve back to the NLP relaxation. The root cannot be spatially
    branched and no valid dual bound is established. The Rust tree previously
    fathomed the root and collapsed ``global_lower_bound`` to the local-minimum
    incumbent (0.2986), certifying it ``optimal`` with gap 0 — a false optimal.
    The honest verdict is NOT optimal (feasible/unknown).
    """
    r = _camel_semiinfinite(-5.0, float("inf"), float("-inf"), 5.0, "c467_nlp").solve(
        time_limit=6.0
    )
    assert r.status != "optimal", (
        f"free-variable root falsely certified optimal: status={r.status} "
        f"obj={r.objective} bound={r.bound}"
    )
    assert not getattr(r, "gap_certified", False), (
        f"free-variable root falsely reports gap_certified: bound={r.bound}"
    )


@pytest.mark.smoke
def test_467_free_variable_root_not_falsely_optimal_lp_route():
    """#467: one var finite (x1=[-5,5]), one semi-infinite (x0=[-5,inf]).

    ``_origin_has_finite_continuous_var`` is True (x1 is finite), so the guard
    does NOT fall back — the McCormick-LP spatial path runs. It still collapses:
    the LP relaxer honestly abstains on the unbounded nonlinear column (x0), so
    the root carries no finite bound and cannot be branched to a finite bound.
    This exercises the second (LP) route into the same Rust fathom/collapse. The
    verdict must NOT be a certified optimal.
    """
    r = _camel_semiinfinite(-5.0, float("inf"), -5.0, 5.0, "c467_lp").solve(time_limit=6.0)
    assert r.status != "optimal", (
        f"semi-infinite root falsely certified optimal (LP route): status={r.status} "
        f"obj={r.objective} bound={r.bound}"
    )
    assert not getattr(r, "gap_certified", False), (
        f"semi-infinite root falsely reports gap_certified (LP route): bound={r.bound}"
    )


@pytest.mark.smoke
def test_467_finite_box_control_still_certifies_optimal():
    """#467 control: the SAME objective over a finite box [-5,5]^2 must still
    certify the true global optimum 0.0. This guards against the fix
    over-firing and downgrading a validly-certified finite-box solve."""
    # Generous time_limit (60s): this is the only #467 guard that REQUIRES a
    # certified `optimal`, and the finite-box three-hump-camel is a borderline
    # nonconvex certification (~1.7s uninstrumented). Under the coverage-
    # instrumented CI job (Python-heavy B&B loop, 2-3x slower on a shared runner)
    # a 10s budget intermittently timed out -> `feasible` -> false test failure.
    # The headroom keeps the correctness guard intact without the timing flake;
    # a valid solve certifies long before 60s.
    r = _camel_semiinfinite(-5.0, 5.0, -5.0, 5.0, "c467_ctrl").solve(time_limit=60.0)
    assert r.status == "optimal", f"finite-box control lost certification: status={r.status}"
    assert r.objective is not None and abs(r.objective) <= 1e-3, (
        f"finite-box control missed the true optimum 0.0: obj={r.objective}"
    )
    assert getattr(r, "gap_certified", False), "finite-box control lost gap_certified"


# ---------------------------------------------------------------------------
# #467 sub-bug #3: a rigorous infeasibility proof must win over a soft,
# within-tolerance-only incumbent. Differential — BOTH directions:
#   (a) genuinely INFEASIBLE + a near-boundary pump incumbent -> NOT optimal.
#   (b) genuinely FEASIBLE with a within-tolerance boundary optimum -> stays
#       optimal/feasible; the fix must NEVER flip it to infeasible (the
#       worst-class error: a false infeasible on a truly-feasible model).
# ---------------------------------------------------------------------------


def _feasible_boundary_model():
    """Feasible: minimize (x-1)^2+(y-1)^2 s.t. x+y<=2, x,y in [0,2]. The global
    optimum (1,1) sits ON the constraint boundary, so the incumbent is accepted
    only within tolerance — exactly the shape the fix must NOT reject."""
    m = discopt.Model("feas_boundary")
    x = m.continuous("x", lb=0.0, ub=2.0)
    y = m.continuous("y", lb=0.0, ub=2.0)
    m.subject_to(x + y <= 2.0)
    m.minimize((x - 1.0) ** 2 + (y - 1.0) ** 2)
    return m


def _feasible_bilinear_boundary_model():
    """Feasible bilinear: minimize x+y s.t. x*y>=4, x,y in [1,3]. Optimum on the
    x*y=4 boundary (e.g. x=y=2). Guards against a false infeasible when a
    nonlinear constraint is active at the optimum."""
    m = discopt.Model("feas_bilinear")
    x = m.continuous("x", lb=1.0, ub=3.0)
    y = m.continuous("y", lb=1.0, ub=3.0)
    m.subject_to(x * y >= 4.0)
    m.minimize(x + y)
    return m


def _infeasible_bilinear_model():
    """Infeasible: x*y<=1 with x,y in [2,3] (so x*y in [4,9] > 1). FBBT proves the
    box empty; there is no feasible point."""
    m = discopt.Model("infeas_bilinear")
    x = m.continuous("x", lb=2.0, ub=3.0)
    y = m.continuous("y", lb=2.0, ub=3.0)
    m.subject_to(x * y <= 1.0)
    m.minimize(x + y)
    return m


@pytest.mark.smoke
def test_467sub3_feasible_boundary_not_flipped_to_infeasible():
    """(b) false-infeasible guard: a feasible model with a within-tolerance
    boundary optimum must stay optimal/feasible — NEVER infeasible."""
    r = _feasible_boundary_model().solve(time_limit=6.0)
    assert r.status != "infeasible", (
        f"false infeasible on a truly-feasible model: status={r.status} obj={r.objective}"
    )
    assert r.objective is not None and abs(r.objective) <= 1e-3, (
        f"feasible boundary model lost its optimum: status={r.status} obj={r.objective}"
    )


@pytest.mark.smoke
def test_467sub3_feasible_bilinear_boundary_not_flipped_to_infeasible():
    """(b) false-infeasible guard with an active NONLINEAR constraint at the
    optimum. Must not be reported infeasible."""
    r = _feasible_bilinear_boundary_model().solve(time_limit=6.0)
    assert r.status != "infeasible", (
        f"false infeasible on a truly-feasible bilinear model: status={r.status} obj={r.objective}"
    )
    assert r.objective is not None and r.objective <= 4.0 + 1e-2, (
        f"feasible bilinear model lost its optimum (~4.0): status={r.status} obj={r.objective}"
    )


@pytest.mark.smoke
def test_467sub3_infeasible_bilinear_never_optimal():
    """(a) a rigorously infeasible model must NEVER be certified optimal. Either
    infeasible (rigorous) or a resource-limited non-optimal status is acceptable;
    a false optimal is not."""
    r = _infeasible_bilinear_model().solve(time_limit=6.0)
    assert r.status != "optimal", (
        f"rigorously-infeasible model falsely certified optimal: status={r.status} "
        f"obj={r.objective}"
    )


@pytest.mark.smoke
def test_ex14_1_9_reported_bound_never_exceeds_incumbent():
    """A reported dual bound must never cross a known feasible objective (the
    ``bound <= incumbent`` invariant the Rust tree enforces via
    ``update_global_lower_bound``).

    ``ex14_1_9`` minimizes a *free* variable ``x1`` pinned only by two constraints
    that are differences of huge, nearly-cancelling transcendental terms
    (``4.51e6*exp(-7548/x0)*x0 - 2.02e9*exp(-7548/x0)``); its true optimum is 0.
    The uniform engine cannot bound this free objective (``objective_bound_valid``
    is False), so the search takes a tainted exit and runs the
    ``_root_relaxation_lower_bound`` fallback over an FBBT-tightened *snapshot* box
    that no longer contains the optimum, which returns ~1.0. Before the fix the
    Python result assembly adopted that as a "tighter" lower bound and reported
    ``bound=1.0 > incumbent=0`` — an unsound dual bound (found by the expanded
    50-instance BARON soundness sweep). The reported bound is now capped at the
    incumbent. Status stays ``feasible`` (the taint correctly blocks certifying
    ``optimal``); only the reported bound must be sound.
    """
    from discopt.modeling.core import from_nl

    nl = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl", "ex14_1_9.nl")
    r = from_nl(nl).solve(time_limit=30.0)
    assert r.objective is not None
    if r.bound is not None and np.isfinite(r.bound):
        # oracle optimum ~0; the invariant is bound <= incumbent (== objective).
        assert r.bound <= r.objective + 1e-6, (
            f"unsound dual bound {r.bound} exceeds the incumbent {r.objective} "
            "(bound must never cross a feasible objective)"
        )
        assert r.bound <= 0.0 + 1e-4, f"dual bound {r.bound} crosses the oracle optimum 0"


@pytest.mark.slow
def test_467sub3_ex7_3_6_not_false_optimal():
    """(a) the real repro (MINLPLib ex7_3_6, oracle =inf=): FBBT proves the root
    empty by ~2e-6 while a feasibility-pump point at ~1.2e-4 original-constraint
    residual was previously certified ``optimal``. The fix must make the verdict
    NOT optimal (infeasible with enough budget; otherwise time_limit/unknown —
    both acceptable). Skips if the instance is not cached locally."""
    import os as _os

    nl = _os.path.expanduser("~/.cache/discopt/minlplib/current/nl/ex7_3_6.nl")
    if not _os.path.exists(nl):
        pytest.skip("ex7_3_6.nl not cached locally")
    from discopt.modeling.core import from_nl

    r = from_nl(nl).solve(time_limit=20.0)
    assert r.status != "optimal", (
        f"ex7_3_6 (infeasible) falsely certified optimal: status={r.status} obj={r.objective}"
    )
    # The rigorous outcome is ``infeasible`` (a certified conclusion — for which
    # gap_certified=True is correct). A resource-limited ``time_limit``/``unknown``
    # is also acceptable (not a false optimal). What must never happen: an
    # ``optimal``, or an ``infeasible`` verdict that still reports a feasible point.
    assert r.status in ("infeasible", "time_limit", "unknown"), (
        f"ex7_3_6 unexpected status {r.status} (obj={r.objective})"
    )
    if r.status == "infeasible":
        assert r.objective is None, f"ex7_3_6 reported infeasible with an objective {r.objective}"
