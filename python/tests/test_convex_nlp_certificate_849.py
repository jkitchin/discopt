"""Regression: false optimality certificate on a bounded convex QCP (#849).

The adversary found that on a bounded, feasible, *convex* quadratically
constrained problem with a large constraint coefficient, discopt returned
``status=optimal, gap=0.0, gap_certified=True`` at a point ~20x short of the true
optimum, with the certified dual bound *crossing* the true optimum (an unsound
lower bound). Minimal reproduction::

    min -x  s.t.  x**2 <= 1e18,  x in [0, 1e12]     # true opt x=1e9, obj=-1e9

The convexity-certified single-NLP fast path took the backend's reported
``optimal`` as a global optimality certificate. But the NLP backend's stopping
test runs on its internally *scaled* problem, so under the large coefficient it
declared success at a point that satisfies the *scaled* test yet grossly violates
the *unscaled* KKT complementarity (the constraint was nowhere near active while
its multiplier was nonzero). Under convexity KKT is sufficient for global
optimality, so the fix recomputes the *unscaled* KKT residuals at the returned
primal-dual point and withholds the certificate (reporting ``iteration_limit``,
uncertified) when they are not met — exactly as the neighbouring problem scales
(R=1e12..1e15) already do. It never upgrades a certificate or emits a bound, so
it cannot introduce a false optimum or a wrong bound.

The scale sweep is the discriminator (true opt ``-sqrt(R)``): only R=1e18 flipped
to a false ``optimal`` before the fix; R=1e6 is a genuine certified optimum and
must stay certified (no false negative).
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402


def _qcp(R: float) -> dm.Model:
    """``min -x s.t. x**2 <= R, x in [0, 1e12]`` — convex, feasible, true opt -sqrt(R)."""
    m = dm.Model("qcp")
    x = m.continuous("x", lb=0.0, ub=1e12)
    m.minimize(-x)
    m.subject_to(x * x <= R)
    return m


def _assert_sound(name: str, r, opt: float) -> None:
    """A certified optimum must sit at the true optimum, and any dual bound must
    not cross it (for a minimization a valid lower bound satisfies ``bound <= opt``)."""
    tol = max(5e-3 * abs(opt), 1e-4)
    if r.status == "optimal" and r.gap_certified:
        assert r.objective is not None and abs(r.objective - opt) <= tol, (
            f"{name}: FALSE-OPTIMAL — certified {r.objective!r} != true opt {opt:.6g}"
        )
    if r.bound is not None:
        # Sound lower bound: never above the true optimum (never crosses it).
        assert r.bound <= opt + tol + 1e-6 * abs(opt), (
            f"{name}: UNSOUND BOUND {r.bound!r} > true opt {opt:.6g}"
        )
        if r.objective is not None:
            assert r.bound <= r.objective + tol + 1e-6 * abs(r.objective), (
                f"{name}: UNSOUND CERT bound {r.bound!r} > incumbent {r.objective!r}"
            )


@pytest.mark.parametrize("solver", ["ipm", "pounce"])
def test_large_coefficient_qcp_not_false_optimal(solver):
    """The exact #849 reproduction: R=1e18 must NOT be certified optimal at the
    wrong point. Before the fix: status=optimal, gap_certified=True, obj~-5e7,
    bound~-5e7 (crossing the true opt -1e9)."""
    r = _qcp(1e18).solve(nlp_solver=solver)
    _assert_sound("R=1e18", r, -1e9)
    # It specifically must not claim a certified optimum at the ~20x-short point.
    assert not (r.status == "optimal" and r.gap_certified), (
        f"R=1e18: still emits a certificate (status={r.status}, "
        f"gap_certified={r.gap_certified}, obj={r.objective!r})"
    )


@pytest.mark.parametrize(
    "R,opt",
    [
        (1e6, -1_000.0),
        (1e9, -math.sqrt(1e9)),
        (1e12, -1e6),
        (1e15, -math.sqrt(1e15)),
        (1e18, -1e9),
    ],
)
def test_qcp_scale_sweep_is_sound(R, opt):
    """Across the whole scale sweep the result is always sound: no false optimum
    and no dual bound crossing the true optimum."""
    r = _qcp(R).solve(nlp_solver="ipm")
    _assert_sound(f"R={R:.0e}", r, opt)


def test_small_coefficient_qcp_still_certifies():
    """A well-conditioned convex QCP must still certify its genuine optimum — the
    gate only withholds bad certificates, it must not cause a false negative."""
    r = _qcp(1e6).solve(nlp_solver="ipm")
    assert r.status == "optimal" and r.gap_certified, (
        f"R=1e6 lost its genuine certificate (status={r.status}, gap_certified={r.gap_certified})"
    )
    assert r.objective == pytest.approx(-1_000.0, rel=1e-4)


def _quad_bound_active() -> tuple[dm.Model, float]:
    # min (x-10)^2 s.t. x <= 5  -> opt 25 at the active linear bound.
    m = dm.Model("q")
    x = m.continuous("x", lb=0, ub=20)
    m.minimize((x - 10) ** 2)
    m.subject_to(x <= 5)
    return m, 25.0


def _circle_active() -> tuple[dm.Model, float]:
    # min -x-y s.t. x^2+y^2 <= 1 -> opt -sqrt(2), the QCP active at the boundary.
    m = dm.Model("qcp2")
    x = m.continuous("x", lb=-2, ub=2)
    y = m.continuous("y", lb=-2, ub=2)
    m.minimize(-x - y)
    m.subject_to(x * x + y * y <= 1)
    return m, -math.sqrt(2)


@pytest.mark.parametrize(
    "builder", [_quad_bound_active, _circle_active], ids=["quad-linear-active", "qcp-circle-active"]
)
def test_well_posed_convex_problems_still_certify(builder):
    """Genuine convex optima at active constraints/bounds must keep their
    certificate: the KKT gate must not reject a converged solve."""
    model, opt = builder()
    r = model.solve(nlp_solver="ipm")
    assert r.status == "optimal" and r.gap_certified, (
        f"lost certificate: status={r.status}, gap_certified={r.gap_certified}"
    )
    assert r.objective == pytest.approx(opt, rel=5e-3, abs=1e-3)


def test_certificate_gap_helper_flags_noncomplementary_point():
    """Unit test of the duality-gap helper: a stationary point whose multiplier is
    complementary-violating (nonzero λ on a far-from-active inequality) must show a
    large relative gap, while a true KKT point shows ~0."""
    from discopt.solver import _convex_nlp_certificate_gap, _make_evaluator

    R = 1e18
    m = _qcp(R)
    ev = _make_evaluator(m)
    from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

    cl, cu = _infer_constraint_bounds(ev)
    lb, ub = np.array([0.0]), np.array([1e12])

    # Non-KKT point: x with x^2 << R (constraint slack huge) but a positive
    # multiplier tuned to satisfy stationarity of min -x (grad -1, jac 2x).
    x_bad = np.array([6.4e7])
    lam_bad = np.array([1.0 / (2.0 * x_bad[0])])  # makes grad + J^T lam = 0
    stat, comp = _convex_nlp_certificate_gap(ev, x_bad, lam_bad, lb, ub, cl, cu, -x_bad[0])
    assert stat < 1e-4, f"constructed point should be projected-gradient stationary, got {stat}"
    assert comp > 1.0, f"far-from-active nonzero multiplier must show large gap, got {comp}"

    # True KKT point: x = sqrt(R) (constraint active), lam = 1/(2x).
    x_ok = np.array([math.sqrt(R)])
    lam_ok = np.array([1.0 / (2.0 * x_ok[0])])
    stat2, comp2 = _convex_nlp_certificate_gap(ev, x_ok, lam_ok, lb, ub, cl, cu, -x_ok[0])
    assert stat2 < 1e-4 and comp2 < 1e-6, f"true KKT point must certify, got {stat2}, {comp2}"


def test_certificate_gap_helper_certifies_bound_active_and_fixed_vars():
    """The gap must NOT flag a genuine optimum resting on a variable bound, nor a
    variable pinned by lb==ub — even with zero bound multipliers (POUNCE/Ipopt
    return ~0 there). Regression for the GP-MINLP node false-negative uncovered
    while fixing #849: min exp(y)+2.25 exp(-y)."""
    from discopt import exp as dexp
    from discopt.solver import _convex_nlp_certificate_gap, _make_evaluator

    # Bound-active: y in [log2, log5], optimum at the lower bound y=log2.
    lo, hi = math.log(2), math.log(5)
    m1 = dm.Model("ba")
    y = m1.continuous("y", lb=lo, ub=hi)
    m1.minimize(dexp(y) + 2.25 * dexp(-y))
    ev1 = _make_evaluator(m1)
    stat1, comp1 = _convex_nlp_certificate_gap(
        ev1, np.array([lo]), None, np.array([lo]), np.array([hi]), np.empty(0), np.empty(0), 3.125
    )
    assert stat1 < 1e-4 and comp1 < 1e-6, f"bound-active optimum rejected: {stat1}, {comp1}"

    # Fixed variable: y pinned at log2 (lb==ub). Gradient there is nonzero (0.875)
    # but the variable cannot move, so the projected gradient is zero.
    m2 = dm.Model("fx")
    yf = m2.continuous("y", lb=lo, ub=lo)
    m2.minimize(dexp(yf) + 2.25 * dexp(-yf))
    ev2 = _make_evaluator(m2)
    stat2, comp2 = _convex_nlp_certificate_gap(
        ev2, np.array([lo]), None, np.array([lo]), np.array([lo]), np.empty(0), np.empty(0), 3.125
    )
    assert stat2 < 1e-4 and comp2 < 1e-6, f"fixed-variable optimum rejected: {stat2}, {comp2}"


def test_certificate_gap_helper_none_without_multipliers():
    """Constraints present but no multipliers -> cannot assess -> None (caller
    treats as not-certified)."""
    from discopt.solver import _convex_nlp_certificate_gap, _make_evaluator

    m = _qcp(1e18)
    ev = _make_evaluator(m)
    from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

    cl, cu = _infer_constraint_bounds(ev)
    out = _convex_nlp_certificate_gap(
        ev, np.array([1.0]), None, np.array([0.0]), np.array([1e12]), cl, cu, -1.0
    )
    assert out is None
