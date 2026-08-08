"""Regression tests for #940: POUNCE's LP/QP points must stay inside the box and
meet discopt's own constraint tolerance, so the #850 guard stops rejecting correct
solves.

Two independent mechanisms put a returned point where discopt rejects it, and
which one dominates depends on the POUNCE build — so both are pinned, in
``discopt.solvers.pounce_option_defaults``:

* ``bound_relax_factor = 0``. Ipopt's default 1e-8 deliberately relaxes bounds,
  including the slack bounds standing in for inequality rows, by
  ``1e-8*(1 + |bound|)``. The solve converges honestly but to a RELAXED box, so
  the point sits outside the declared one and the error grows with the data
  (8.1e-6 / 8.1e-5 / 8.1e-4 at data scale 1e2 / 1e3 / 1e4). No convergence
  tolerance can fix this. Dominant on pounce @ main.
* ``constr_viol_tol = 1e-8``. POUNCE inherits Ipopt's 1e-4, two orders looser
  than the guard's 1e-6 floor. Dominant on the published wheel; a no-op on main.

Measured before the fix: **50%** of POUNCE LP solves over a 148-solve population
tripped the guard, including all four ``docs/notebooks/tutorial_lp.ipynb`` models
and 100% of a random sweep with row term scale in ``[1e2, 1e5)``.

The fix is at the source, not at the guard — the guard is untouched and remains
the arbiter. These tests pin BEHAVIOUR (the returned point), never the option
names, which is why they caught a ``constr_viol_tol``-only version of this fix
being a complete no-op on the build CI actually installs.

8 of these cases fail on the pre-#940 tree. ``nlp_path`` was a live strict xfail
until #945 seeded the NLP path from the same baseline; it is now a plain pass.
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
from discopt.solvers import POUNCE_BOUND_RELAX_FACTOR, SolveStatus  # noqa: E402
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


def _model_from_matrices(c, A_ub, b_ub, bounds, name="m940"):
    """Build a `dm.Model` equivalent to ``min cᵀx s.t. A_ub x ≤ b_ub, bounds``.

    The tests below go through ``model.solve()`` deliberately. ``bound_relax_factor``
    is requested by the model-level POUNCE call sites (``solver._solve_lp_pounce`` /
    ``_solve_qp_pounce``) rather than as a backend-wide default, because those are
    the call sites whose points ``_matrix_solution_feasible`` checks — and applying
    it backend-wide reaches the Benders dual LP, where it costs convergence (#945).
    So calling the backend directly would test a different contract than the one
    #940 is about.
    """
    n = len(c)
    m = dm.Model(name)
    lo = float(bounds[0][0])
    hi = float(bounds[0][1])
    x = m.continuous("x", shape=(n,), lb=lo, ub=hi)
    m.minimize(dm.sum(lambda j: float(c[j]) * x[j], over=range(n)))
    for i in range(A_ub.shape[0]):
        row = A_ub[i]
        m.subject_to(
            dm.sum(lambda j: float(row[j]) * x[j], over=range(n)) <= float(b_ub[i]),
            name=f"r{i}",
        )
    return m, x


def _worst_violation(x, A_ub, b_ub, bounds):
    """Max absolute amount by which ``x`` breaks a row or a bound."""
    x = np.asarray(x, dtype=np.float64)
    viol = float(np.max(np.asarray(A_ub, dtype=np.float64) @ x - np.asarray(b_ub, float)))
    for xi, (lo, hi) in zip(x, bounds):
        viol = max(viol, lo - xi, xi - hi)
    return viol


def test_pounce_lp_point_meets_discopt_constraint_tolerance():
    """The model-level LP fast path must return a point discopt's guard accepts."""
    c, A_ub, b_ub, bounds = _diet_matrices()
    model, x = _model_from_matrices(c, A_ub, b_ub, bounds, name="diet940")
    res = model.solve(nlp_solver="pounce")

    assert res.status == "optimal"
    sol = np.asarray(res.value(x), dtype=np.float64).ravel()
    worst = _worst_violation(sol, A_ub, b_ub, bounds)
    # Pre-fix this is ~5.8e-6 (and up to Ipopt's 1e-4 default on other data).
    assert worst <= DISCOPT_CONSTR_TOL, f"worst violation {worst:.3e} exceeds 1e-6"
    assert _matrix_solution_feasible(sol, A_ub, b_ub, None, None, bounds)


@pytest.mark.parametrize("scale", [1e2, 1e3, 1e4])
def test_pounce_lp_guard_holds_across_data_scales(scale):
    """The guard trips as a function of row term scale, so pin the whole band.

    Pre-fix, 100% of solves with row term scale in ``[1e2, 1e5)`` tripped it;
    this is the general statement, not the tutorial instance.
    """
    rng = np.random.default_rng(940)
    n, m = 20, 10
    A_ub = -np.abs(rng.uniform(0.5, 5.0, size=(m, n)))
    b_ub = -scale * np.abs(rng.uniform(0.5, 2.0, size=m)) * n * 0.25
    c = np.abs(rng.uniform(1.0, 10.0, size=n))
    bounds = [(0.0, 10.0 * scale)] * n

    model, x = _model_from_matrices(c, A_ub, b_ub, bounds, name=f"sw{scale:g}")
    res = model.solve(nlp_solver="pounce")
    assert res.status == "optimal"
    sol = np.asarray(res.value(x), dtype=np.float64).ravel()
    assert _matrix_solution_feasible(sol, A_ub, b_ub, None, None, bounds), (
        f"guard rejected the model-level POUNCE LP point at data scale {scale:g}; "
        f"worst violation {_worst_violation(sol, A_ub, b_ub, bounds):.3e}"
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

    # Mirrors exactly what solver._solve_qp_pounce requests at the guard-checked
    # call site; the wiring itself is pinned by the test below.
    res = solve_qp(
        Q=Q,
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        options={"bound_relax_factor": POUNCE_BOUND_RELAX_FACTOR},
    )
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
    ``solvers.pounce_option_defaults`` so every entry point inherits it — the
    LP/QP backends in #940, the NLP path in #945.
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


def test_option_requests_are_wired_where_the_guard_checks():
    """Pin WHERE each request lives, because the two have different blast radii.

    ``constr_viol_tol`` is a backend-wide default: it is harmless everywhere and
    carries the fix on the published pounce wheel.

    ``bound_relax_factor = 0`` is NOT backend-wide. It is requested by the
    model-level POUNCE call sites, whose points ``_matrix_solution_feasible``
    checks. Applied backend-wide it also reaches the Benders dual LP, where it
    costs convergence — two correctness-lane tests go 1.6s -> 79s and end at
    iteration/time limit (#945). Keeping the request at the consumer that needs
    the guarantee is the difference between fixing #940 and breaking Benders.
    """
    import inspect

    from discopt import solver as S
    from discopt.solvers import POUNCE_BOUND_RELAX_FACTOR, pounce_option_defaults

    defaults = pounce_option_defaults()
    assert defaults["constr_viol_tol"] <= DISCOPT_CONSTR_TOL
    assert "bound_relax_factor" not in defaults, (
        "bound_relax_factor must NOT be a backend-wide default — it breaks the "
        "Benders dual LP (#945)"
    )
    # A fresh dict each call: a caller mutating it must not poison the next solve.
    defaults["constr_viol_tol"] = 999.0
    assert pounce_option_defaults()["constr_viol_tol"] <= DISCOPT_CONSTR_TOL

    assert POUNCE_BOUND_RELAX_FACTOR == 0.0
    for fn in (S._solve_lp_pounce, S._solve_qp_pounce):
        src = inspect.getsource(fn)
        assert "bound_relax_factor" in src, (
            f"{fn.__name__} must request bound_relax_factor — it is the call site "
            "whose point the #850 guard checks"
        )


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
