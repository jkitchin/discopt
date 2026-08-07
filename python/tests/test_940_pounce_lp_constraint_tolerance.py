"""Regression tests for #940: POUNCE's LP/QP points must meet discopt's own
constraint tolerance, so the #850 guard stops rejecting correct solves.

POUNCE inherits Ipopt's default ``constr_viol_tol`` of ``1e-4``. discopt checks
every matrix-form point against ``solver._matrix_solution_feasible`` — the #850
Obs 3 guard, ``|viol_i| ≤ 1e-6 + 1e-9·Σ_j|A_ij||x_j|``. The two are three orders
of magnitude apart, so on any LP whose rows carry term magnitudes above a few
hundred POUNCE returned a point on the infeasible side of a row, labeled it
``optimal``, and the guard — correctly — threw it away, logged a WARNING and
re-solved with the exact simplex. Measured before the fix: **50%** of POUNCE LP
solves over a 148-solve population tripped the guard, including all four
``docs/notebooks/tutorial_lp.ipynb`` models and 100% of a random sweep with row
term scale in ``[1e2, 1e5)``.

The fix is at the source, not at the guard: ``lp_pounce._CONSTR_VIOL_TOL = 1e-9``
is requested for the LP and QP backends, which makes POUNCE's own convergence
criterion imply discopt's. The guard is untouched and remains the arbiter.

These tests fail on the pre-fix tree (worst violation ``1e-4``, guard trips,
warning emitted) and pass after.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import logging  # noqa: E402

import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from discopt.solver import _matrix_solution_feasible  # noqa: E402
from discopt.solvers import SolveStatus  # noqa: E402
from discopt.solvers.lp_pounce import POUNCE_AVAILABLE  # noqa: E402

pytestmark = pytest.mark.skipif(not POUNCE_AVAILABLE, reason="pounce not installed")

# discopt's absolute constraint tolerance (conftest ``abs``; the guard's ``tol``).
DISCOPT_CONSTR_TOL = 1e-6


def _diet_matrices():
    """The tutorial_lp.ipynb diet LP in ``min cᵀx  s.t.  A_ub x ≤ b_ub`` form.

    Worst row term scale ≈ 800, which is exactly the regime the guard rejected.
    """
    c = np.array([2.0, 3.5, 8.0, 11.0, 25.0])
    A = np.array(
        [
            [3.0, 8.0, 15.0, 22.0, 31.0],
            [25.0, 120.0, 200.0, 10.0, 15.0],
            [1.0, 0.1, 0.5, 3.0, 5.5],
            [250.0, 150.0, 400.0, 200.0, 450.0],
        ]
    )
    req = np.array([55.0, 800.0, 12.0, 2000.0])
    # ``A x ≥ req``  ⇒  ``-A x ≤ -req``
    return c, -A, -req, [(0.0, 10.0)] * 5


def _worst_violation(x, A_ub, b_ub, bounds):
    """Max absolute amount by which ``x`` breaks a row or a bound."""
    x = np.asarray(x, dtype=np.float64)
    viol = float(np.max(np.asarray(A_ub, dtype=np.float64) @ x - np.asarray(b_ub, float)))
    for xi, (lo, hi) in zip(x, bounds):
        viol = max(viol, lo - xi, xi - hi)
    return viol


def test_pounce_lp_point_meets_discopt_constraint_tolerance():
    """POUNCE's own LP convergence must imply discopt's constraint tolerance."""
    from discopt.solvers.lp_pounce import solve_lp

    c, A_ub, b_ub, bounds = _diet_matrices()
    res = solve_lp(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

    assert res.status == SolveStatus.OPTIMAL
    worst = _worst_violation(res.x, A_ub, b_ub, bounds)
    # Pre-fix this is ~5.8e-6 (and up to Ipopt's 1e-4 default on other data).
    assert worst <= DISCOPT_CONSTR_TOL, f"worst violation {worst:.3e} exceeds 1e-6"
    assert _matrix_solution_feasible(res.x, A_ub, b_ub, None, None, bounds)


@pytest.mark.parametrize("scale", [1e2, 1e3, 1e4])
def test_pounce_lp_guard_holds_across_data_scales(scale):
    """The guard trips as a function of row term scale, so pin the whole band.

    Pre-fix, 100% of solves with row term scale in ``[1e2, 1e5)`` tripped it;
    this is the general statement, not the tutorial instance.
    """
    from discopt.solvers.lp_pounce import solve_lp

    rng = np.random.default_rng(940)
    n, m = 20, 10
    A_ub = -np.abs(rng.uniform(0.5, 5.0, size=(m, n)))
    b_ub = -scale * np.abs(rng.uniform(0.5, 2.0, size=m)) * n * 0.25
    c = np.abs(rng.uniform(1.0, 10.0, size=n))
    bounds = [(0.0, 10.0 * scale)] * n

    res = solve_lp(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    assert res.status == SolveStatus.OPTIMAL
    assert _matrix_solution_feasible(res.x, A_ub, b_ub, None, None, bounds), (
        f"guard rejected a converged POUNCE LP point at data scale {scale:g}; "
        f"worst violation {_worst_violation(res.x, A_ub, b_ub, bounds):.3e}"
    )


def test_pounce_lp_respects_binding_bound_at_large_magnitude():
    """A variable pinned at a large upper bound must not overshoot it.

    Ipopt's ``bound_relax_factor`` lets iterates sit outside a bound by
    ``1e-8·(1+|bound|)``; pre-fix the returned point sat 1e-4 above ``ub=1e4``,
    which the guard (threshold ``1e-6 + 1e-9·1e4 ≈ 1.1e-5``) rejected.
    """
    from discopt.solvers.lp_pounce import solve_lp

    n, B = 4, 1e4
    res = solve_lp(
        c=-np.ones(n),
        A_ub=np.ones((1, n)),
        b_ub=np.array([10.0 * n * B]),  # slack row; the BOUND is what binds
        bounds=[(0.0, B)] * n,
    )
    assert res.status == SolveStatus.OPTIMAL
    over = float(np.max(np.asarray(res.x) - B))
    assert over <= DISCOPT_CONSTR_TOL, f"x overshot its upper bound by {over:.3e}"


def test_pounce_qp_point_meets_discopt_constraint_tolerance():
    """The QP backend goes through the same guard and needs the same tolerance."""
    from discopt.solvers.qp_pounce import solve_qp

    # Seed 20 is one that demonstrably trips the guard with POUNCE's old 1e-4
    # tolerance (found by sweeping seeds with ``options={"constr_viol_tol": 1e-4}``),
    # so this test genuinely fails on the pre-fix tree rather than passing by luck.
    rng = np.random.default_rng(20)
    n, m, scale = 10, 6, 1e3
    B = rng.normal(size=(n, n))
    Q = B @ B.T + n * np.eye(n)
    c = -np.abs(rng.uniform(1.0, 10.0, size=n)) * scale
    A_ub = -np.abs(rng.uniform(0.5, 5.0, size=(m, n)))
    b_ub = -scale * np.abs(rng.uniform(0.5, 2.0, size=m)) * n * 0.25
    bounds = [(0.0, 10.0 * scale)] * n

    res = solve_qp(Q=Q, c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    assert res.status == SolveStatus.OPTIMAL
    assert _matrix_solution_feasible(res.x, A_ub, b_ub, None, None, bounds), (
        "guard rejected a converged POUNCE QP point; worst violation "
        f"{_worst_violation(res.x, A_ub, b_ub, bounds):.3e}"
    )


@pytest.mark.parametrize("flat", [True, False], ids=["lp_path", "nlp_path"])
def test_returned_point_stays_inside_its_declared_box(flat):
    """Every POUNCE entry point must return a point inside the declared bounds.

    Pins the *property* directly rather than through an objective comparison.
    Ipopt's ``bound_relax_factor`` (default 1e-8) relaxes every bound by
    ``1e-8*(1 + |bound|)``, so a converged point legitimately sits outside the box
    the caller declared — which is what discopt's feasibility guards were
    rejecting POUNCE points for.

    Both entry points are exercised because they are genuinely different code
    paths: ``dm.sum(y.flat)`` classifies linear and routes to ``lp_pounce``,
    while ``dm.sum(y)`` on the bare indexed container routes to the general NLP
    path. Seeding the option in the LP/QP backends alone left this second path
    returning points ~7.5e-9 below ``lb``; the fix lives in
    ``solvers.pounce_option_defaults`` so every entry point inherits it (#940).
    """
    m = dm.Model()
    s = m.set("S", [10, 20, 30])
    y = m.continuous("y", lb=1.0, ub=5.0, over=s)
    m.minimize(dm.sum(y.flat) if flat else dm.sum(y))
    res = m.solve()

    assert res.status == "optimal"
    x = np.asarray(res.value(y), dtype=np.float64).ravel()
    assert x.size == 3
    below = float(np.max(1.0 - x))
    assert below <= 1e-12, f"returned point sits {below:.3e} below its declared lb=1"
    # The optimum is exactly 3.0; a point outside the box buys a super-optimal
    # objective, so pin that it is not below the true value either.
    assert res.objective >= 3.0 - 1e-12


def test_shared_pounce_defaults_are_the_single_source_of_truth():
    """The option baseline must not be re-spelled per backend.

    A second copy is how one entry point silently keeps Ipopt's defaults while
    the rest move — the #940 failure mode exactly.
    """
    from discopt.solvers import pounce_option_defaults

    defaults = pounce_option_defaults()
    assert defaults["bound_relax_factor"] == 0.0
    assert defaults["constr_viol_tol"] <= DISCOPT_CONSTR_TOL
    # A fresh dict each call: a caller mutating it must not poison the next solve.
    defaults["bound_relax_factor"] = 999.0
    assert pounce_option_defaults()["bound_relax_factor"] == 0.0


def test_caller_supplied_constr_viol_tol_still_wins():
    """The tightened value is a default, not an override of the caller."""
    from discopt.solvers.lp_pounce import solve_lp

    c, A_ub, b_ub, bounds = _diet_matrices()
    # Hand back BOTH of POUNCE's own defaults. Overriding only one is not enough:
    # the two requests target different mechanisms and either alone keeps the
    # point tight, which is the whole point of setting both (#940).
    res = solve_lp(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        options={"constr_viol_tol": 1e-4, "bound_relax_factor": 1e-8},
    )
    assert res.status == SolveStatus.OPTIMAL
    # With POUNCE's own defaults restored the point is allowed back out past
    # discopt's tolerance — the pre-fix behavior. This is proof the requests are
    # honored end to end, so the tests above are measuring them and not a
    # coincidence of this POUNCE build.
    assert _worst_violation(res.x, A_ub, b_ub, bounds) > DISCOPT_CONSTR_TOL


def test_tutorial_lp_solves_without_the_fallback_warning(caplog):
    """End to end: a tutorial LP solves on the first engine, warning-free.

    This is the user-visible symptom from #940 — all four ``tutorial_lp.ipynb``
    LP cells printed "POUNCE LP returned an infeasible point labeled optimal"
    above otherwise-correct output.
    """
    cost = np.array([2.0, 3.5, 8.0, 11.0, 25.0])
    A = np.array(
        [
            [3.0, 8.0, 15.0, 22.0, 31.0],
            [25.0, 120.0, 200.0, 10.0, 15.0],
            [1.0, 0.1, 0.5, 3.0, 5.5],
            [250.0, 150.0, 400.0, 200.0, 450.0],
        ]
    )
    req = np.array([55.0, 800.0, 12.0, 2000.0])

    m = dm.Model("diet")
    x = m.continuous("x", shape=(5,), lb=0, ub=10)
    m.minimize(dm.sum(lambda j: cost[j] * x[j], over=range(5)))
    for i in range(4):
        m.subject_to(dm.sum(lambda j: A[i, j] * x[j], over=range(5)) >= req[i], name=f"n{i}")

    with caplog.at_level(logging.WARNING, logger="discopt.solver"):
        res = m.solve()

    assert res.status == "optimal"
    assert res.objective == pytest.approx(41.559888579, rel=1e-6)
    offending = [
        r.message for r in caplog.records if "infeasible point labeled optimal" in str(r.msg)
    ]
    assert not offending, f"the #850 fallback warning still fires: {offending}"


def test_unbounded_is_never_certified_on_a_compact_box():
    """A finite box makes ``UNBOUNDED`` impossible, so POUNCE must not report it.

    POUNCE reaches ``UNBOUNDED`` from Ipopt codes 3/4 (stalled search direction /
    diverging iterates), which are ambiguous numerical-failure signals. On data
    of magnitude ~1e7 over an ordinary ``[0, 1e8]`` box it hit that exit on LPs
    whose true status is ``optimal``, and since those bounds are far below the
    ``[1e15, 1e20)`` window of the #850 Obs 1 deferral, ``_solve_lp_matrix``
    certified a **false ``unbounded``** end to end. Predates #940 and reproduced
    at POUNCE's own 1e-4 default; closed by ``_certify_unbounded_ray``, which
    finds no improving recession direction here (the only one is ``d = 0``).
    """
    from discopt.solvers.lp_pounce import solve_lp
    from discopt.solvers.lp_simplex import solve_lp as simplex_solve_lp

    # Seed 0 at this shape/scale reaches the ambiguous code-3/4 exit at BOTH
    # POUNCE's own 1e-4 default and the #940 value, so this test fails on the
    # pre-fix tree (it certified ``unbounded``) independently of the tolerance.
    rng = np.random.default_rng(0)
    n, m, scale = 20, 10, 1e7
    A_ub = -np.abs(rng.uniform(0.5, 5.0, size=(m, n)))
    b_ub = -scale * np.abs(rng.uniform(0.5, 2.0, size=m)) * n * 0.25
    c = np.abs(rng.uniform(1.0, 10.0, size=n))
    bounds = [(0.0, 10.0 * scale)] * n

    ref = simplex_solve_lp(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    assert ref.status == SolveStatus.OPTIMAL, "oracle says this LP is bounded"

    res = solve_lp(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    assert res.status != SolveStatus.UNBOUNDED, (
        "POUNCE certified UNBOUNDED for an LP whose every bound is finite; "
        "on a compact box that verdict is impossible"
    )

    # End to end: the model-level certificate must be the oracle's.
    mdl = dm.Model("scale1e7")
    x = mdl.continuous("x", shape=(n,), lb=0.0, ub=10.0 * scale)
    mdl.minimize(dm.sum(lambda j: float(c[j]) * x[j], over=range(n)))
    for i in range(m):
        row = A_ub[i]
        mdl.subject_to(
            dm.sum(lambda j: float(row[j]) * x[j], over=range(n)) <= float(b_ub[i]), name=f"r{i}"
        )
    out = mdl.solve(nlp_solver="pounce")
    assert out.status == "optimal"
    assert out.objective == pytest.approx(ref.objective, rel=1e-9)


def test_unbounded_is_never_certified_on_a_non_compact_box():
    """An infinite bound does not make a ray exist — the ray must be exhibited.

    The stronger half of the fix. With ``min c'x`` under ``c >= 0, x >= 0`` no ray
    can lower the objective *by construction*, yet POUNCE's ambiguous exit
    certified ``unbounded`` on 42 of 90 such instances at data scale 1e7-1e8, and
    an infinite upper bound puts them outside any compact-box argument. Only a
    genuine recession direction may keep the verdict.
    """
    from discopt.solvers.lp_pounce import solve_lp
    from discopt.solvers.lp_simplex import solve_lp as simplex_solve_lp

    rng = np.random.default_rng(0)
    n, m, scale = 20, 10, 1e7
    A_ub = -np.abs(rng.uniform(0.5, 5.0, size=(m, n)))
    b_ub = -scale * np.abs(rng.uniform(0.5, 2.0, size=m)) * n * 0.25
    c = np.abs(rng.uniform(1.0, 10.0, size=n))  # c >= 0 with x >= 0: no improving ray
    bounds = [(0.0, np.inf)] + [(0.0, 10.0 * scale)] * (n - 1)

    ref = simplex_solve_lp(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    assert ref.status == SolveStatus.OPTIMAL, "oracle says this LP is bounded"

    res = solve_lp(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    assert res.status != SolveStatus.UNBOUNDED, (
        "POUNCE certified UNBOUNDED for an LP with c >= 0 and x >= 0, where no "
        "recession direction can lower the objective"
    )

    mdl = dm.Model("noncompact")
    x = mdl.continuous("x", shape=(n,), lb=0.0, ub=np.inf)
    mdl.minimize(dm.sum(lambda j: float(c[j]) * x[j], over=range(n)))
    for i in range(m):
        row = A_ub[i]
        mdl.subject_to(
            dm.sum(lambda j: float(row[j]) * x[j], over=range(n)) <= float(b_ub[i]), name=f"r{i}"
        )
    out = mdl.solve(nlp_solver="pounce")
    assert out.status == "optimal"
    assert out.objective == pytest.approx(ref.objective, rel=1e-6)


def test_genuinely_unbounded_lp_is_still_reported_unbounded():
    """The refusal must not swallow a real unbounded ray.

    This is what a bare "codes 3/4 never certify anything" rule would have cost:
    the Benders dual seam relies on an unbounded dual LP as its feasibility-cut
    signal, so the verdict has to survive when a ray genuinely exists.
    """
    from discopt.solvers.lp_pounce import solve_lp

    # min -x0 - x1 with x free above: d = (1, 1) is a recession direction of
    # strictly negative cost, so UNBOUNDED is earned and must stand.
    res = solve_lp(
        c=np.array([-1.0, -1.0]),
        A_ub=np.array([[-1.0, -1.0]]),
        b_ub=np.array([-1.0]),
        bounds=[(0.0, np.inf)] * 2,
    )
    assert res.status == SolveStatus.UNBOUNDED


def test_genuinely_unbounded_qp_is_still_reported_unbounded():
    """A convex QP that is flat along its improving ray stays UNBOUNDED.

    ``Q = diag(1, 0)`` and ``c = (0, -1)``: the objective is ``½x0² - x1``, which
    falls without bound along ``d = (0, 1)`` — and ``Qd = 0`` there, so the
    quadratic extension of the ray test admits it.
    """
    from discopt.solvers.qp_pounce import solve_qp

    res = solve_qp(
        Q=np.diag([1.0, 0.0]),
        c=np.array([0.0, -1.0]),
        A_ub=np.array([[1.0, 0.0]]),
        b_ub=np.array([10.0]),
        bounds=[(0.0, np.inf)] * 2,
    )
    assert res.status == SolveStatus.UNBOUNDED


def test_qp_unbounded_refused_when_the_ray_is_curved():
    """A ray the quadratic term curves upward is not an unbounded direction.

    ``Q = I``, ``c = (0, -1)``: the objective ``½‖x‖² - x1`` grows along every
    direction, so ``Qd = 0`` has no nonzero solution and no verdict may be
    certified even though the box is unbounded above.
    """
    from discopt.solvers.qp_pounce import solve_qp

    res = solve_qp(
        Q=np.eye(2),
        c=np.array([0.0, -1.0]),
        A_ub=np.array([[1.0, 0.0]]),
        b_ub=np.array([10.0]),
        bounds=[(0.0, np.inf)] * 2,
    )
    assert res.status != SolveStatus.UNBOUNDED


def test_infeasible_and_unbounded_lps_still_certified():
    """The tighter tolerance must not cost POUNCE its infeasible/unbounded verdicts.

    ``solve_lp``'s Phase-1 elastic disambiguation inherits the same option dict,
    so pin that it still converges and still certifies.
    """
    from discopt.solvers.lp_pounce import solve_lp

    # Infeasible: x0 + x1 ≥ 5 with both in [0, 1].
    res = solve_lp(
        c=np.array([1.0, 1.0]),
        A_ub=np.array([[-1.0, -1.0]]),
        b_ub=np.array([-5.0]),
        bounds=[(0.0, 1.0)] * 2,
        certificate=True,
    )
    assert res.status == SolveStatus.INFEASIBLE
    assert res.infeasibility_certificate is not None
    assert res.infeasibility_certificate.total_violation > 0.0

    # Feasible and bounded on a large-but-finite box: still an ordinary optimum.
    res2 = solve_lp(
        c=np.array([-1.0, -1.0]),
        A_ub=np.array([[-1.0, -1.0]]),
        b_ub=np.array([-1.0]),
        bounds=[(0.0, 1e6)] * 2,
    )
    assert res2.status == SolveStatus.OPTIMAL
    assert res2.objective == pytest.approx(-2e6, rel=1e-6)
