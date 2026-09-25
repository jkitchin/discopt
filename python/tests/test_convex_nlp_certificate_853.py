"""Regression: false optimality certificate on a convex vanishing-gradient objective (#853).

The adversary found that on a convex, explicitly-bounded problem whose objective
gradient asymptotically vanishes, discopt's convex single-NLP fast path certifies
(``status=optimal, gap_certified=True``) an *interior* stall point whose objective
is strictly worse than a feasible point at the box boundary. Minimal reproduction::

    min -log(x)  s.t.  x in [1, 1e12]        # true opt x=1e12, obj=-log(1e12)=-27.631

``-log`` is convex and strictly decreasing, so the minimum is at the upper bound.
The fast path instead stalls at an interior x≈1.77e8 where |∇(-log)| = 1/x ≈ 5.6e-9
falls below the KKT stationarity tolerance and is accepted as stationary — a false
certificate whose dual bound (-18.99) crosses the true optimum (-27.63).

This is the same fast-path certificate-soundness surface as #849/#850 but a DISTINCT
failure mode: #849 gates on the *magnitude* of the KKT residual, which does not help
here because the residual is genuinely small — a small stationarity residual does not
bound the objective gap when the Hessian (1/x^2 for -log) flattens. The fix adds a
sound better-point refutation: it moves along the reduced-gradient Frank-Wolfe
direction to the box bound, 1-D-minimizes the Lagrangian, and withholds the
certificate when it EXHIBITS a feasible in-box point beating the incumbent by more
than the optimality tolerance. It only ever downgrades (never upgrades or emits a
bound) and only with an exhibited witness, so it cannot introduce a false optimum and
cannot reject a genuine optimum (which has no better feasible point).

The ub sweep is the discriminator: for ub <= 1e8 the stall point IS the true boundary
optimum, so it must stay certified (no false negative); ub >= 1e9 flipped to a false
``optimal`` before the fix and must now withhold.
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402


def _neglog(ub: float) -> dm.Model:
    """``min -log(x) s.t. x in [1, ub]`` — convex, feasible, true opt -log(ub) at x=ub."""
    m = dm.Model("neg_log")
    x = m.continuous("x", lb=1.0, ub=ub)
    m.minimize(-dm.log(x))
    return m


def _assert_sound(name: str, r, opt: float) -> None:
    """A certified optimum must sit at the true optimum, and any reported bound must
    not cross it (for a minimization a valid lower bound satisfies ``bound <= opt``)."""
    tol = max(5e-3 * abs(opt), 1e-4)
    if r.status == "optimal" and r.gap_certified:
        assert r.objective is not None and abs(r.objective - opt) <= tol, (
            f"{name}: FALSE-OPTIMAL — certified {r.objective!r} != true opt {opt:.6g}"
        )
    if r.bound is not None:
        assert r.bound <= opt + tol + 1e-6 * abs(opt), (
            f"{name}: UNSOUND BOUND {r.bound!r} > true opt {opt:.6g}"
        )
        if r.objective is not None:
            assert r.bound <= r.objective + tol + 1e-6 * abs(r.objective), (
                f"{name}: UNSOUND CERT bound {r.bound!r} > incumbent {r.objective!r}"
            )


@pytest.mark.parametrize("solver", ["ipm", "ipopt", "pounce"])
def test_neglog_interior_stall_not_false_optimal(solver):
    """The exact #853 reproduction: ub=1e12 must NOT be certified optimal at the
    interior stall point. Before the fix: status=optimal, gap_certified=True,
    obj~-18.99, bound~-18.99 (crossing the true opt -27.63)."""
    r = _neglog(1e12).solve(nlp_solver=solver)
    _assert_sound("ub=1e12", r, -math.log(1e12))
    assert not (r.status == "optimal" and r.gap_certified), (
        f"ub=1e12: still emits a certificate (status={r.status}, "
        f"gap_certified={r.gap_certified}, obj={r.objective!r})"
    )


@pytest.mark.parametrize("ub", [1e4, 1e6, 1e8, 1e9, 1e10, 1e12], ids=lambda v: f"ub={v:.0e}")
def test_neglog_scale_sweep_is_sound(ub):
    """Across the ub sweep the result is always sound: no false optimum and no
    reported bound crossing the true optimum."""
    r = _neglog(ub).solve(nlp_solver="ipm")
    _assert_sound(f"ub={ub:.0e}", r, -math.log(ub))


@pytest.mark.parametrize("ub", [1e4, 1e6, 1e8], ids=lambda v: f"ub={v:.0e}")
def test_neglog_reachable_boundary_still_certifies(ub):
    """When the boundary optimum is reachable (ub <= 1e8) the genuine optimum must
    still certify — the refutation must not cause a false negative."""
    r = _neglog(ub).solve(nlp_solver="ipm")
    assert r.status == "optimal" and r.gap_certified, (
        f"ub={ub:.0e} lost its genuine certificate (status={r.status}, "
        f"gap_certified={r.gap_certified})"
    )
    assert r.objective == pytest.approx(-math.log(ub), rel=1e-4, abs=1e-4)


def test_certificate_gap_helper_refutes_vanishing_gradient_stall():
    """Unit test of the refutation: at the interior -log stall point (tiny reduced
    gradient, distant descent bound) the helper returns None (refuted), because a
    strictly better feasible box point is exhibited. A genuine boundary optimum of
    the same objective still returns a valid (stat, comp) tuple."""
    from discopt.solver import _convex_nlp_certificate_gap, _make_evaluator

    m = _neglog(1e12)
    ev = _make_evaluator(m)
    lb, ub = np.array([1.0]), np.array([1e12])
    cl = cu = np.empty(0)

    # Interior stall point: gradient -1/x ~ -5.6e-9 (below the unit-step stationarity
    # tol) but the true min is at the distant upper bound -> must be refuted (None).
    x_stall = np.array([1.7693e8])
    out = _convex_nlp_certificate_gap(ev, x_stall, None, lb, ub, cl, cu, -math.log(x_stall[0]))
    assert out is None, f"interior vanishing-gradient stall must be refuted, got {out}"

    # Genuine boundary optimum x=ub: the gradient points OUT of the box (toward a
    # larger x that is infeasible), so no in-box descent exists -> certifiable tuple.
    x_opt = np.array([1e12])
    out2 = _convex_nlp_certificate_gap(ev, x_opt, None, lb, ub, cl, cu, -math.log(x_opt[0]))
    assert out2 is not None, "genuine boundary optimum must not be refuted"
    stat, comp = out2
    assert stat < 1e-4 and comp < 1e-6, f"boundary optimum must certify, got {stat}, {comp}"


# --- Default / large box: ub in [1e19, 1e20) (#853 regression, 2026-09-25) ---------
#
# The refutation above first formed its Frank-Wolfe vertex from ``_INF = 1e19``-capped
# bounds, so the default box ``DEFAULT_VARIABLE_BOUND = 9.999e19`` — FINITE per #850 —
# read as +inf and got no vertex. Every continuous variable declared WITHOUT an upper
# bound therefore still certified the interior -log stall (obj = bound = -18.97) while
# the box optimum is -log(9.999e19) = -46.05: a dual bound crossing the box optimum by
# ~27. The vertex now uses the true 1e20 bound-infinity cutoff.

from discopt.constants import DEFAULT_VARIABLE_BOUND  # noqa: E402

_BOX_UB = DEFAULT_VARIABLE_BOUND


def _assert_no_certificate(name: str, r) -> None:
    assert not (r.status == "optimal" and r.gap_certified), (
        f"{name}: still emits a certificate (status={r.status}, "
        f"gap_certified={r.gap_certified}, obj={r.objective!r}, bound={r.bound!r})"
    )


@pytest.mark.parametrize("solver", ["ipm", "ipopt", "pounce"])
def test_neglog_default_box_not_false_optimal(solver):
    """``min -log(x), x >= 1`` with NO declared ub (default box 9.999e19). Before the
    fix: optimal, gap_certified=True, obj=bound=-18.97 vs box optimum -46.05."""
    m = dm.Model("neg_log_default")
    x = m.continuous("x", lb=1.0)
    m.minimize(-dm.log(x))
    r = m.solve(nlp_solver=solver)
    _assert_sound("default box", r, -math.log(_BOX_UB))
    _assert_no_certificate("default box", r)


@pytest.mark.parametrize("ub", [1e19, 5e19, 9e19], ids=lambda v: f"ub={v:.0e}")
def test_neglog_large_explicit_ub_not_false_optimal(ub):
    """Explicit ub in [1e19, 1e20) is a finite bound and must be refuted like 1e12."""
    r = _neglog(ub).solve(nlp_solver="ipm")
    _assert_sound(f"ub={ub:.0e}", r, -math.log(ub))
    _assert_no_certificate(f"ub={ub:.0e}", r)


def test_neglog_shifted_default_box_not_false_optimal():
    """``min -log(1+x), x >= 0`` on the default box (box optimum -log(1+9.999e19))."""
    m = dm.Model("neg_log1p_default")
    x = m.continuous("x", lb=0.0)
    m.minimize(-dm.log(1 + x))
    r = m.solve(nlp_solver="ipm")
    _assert_sound("-log(1+x) default box", r, -math.log(1.0 + _BOX_UB))
    _assert_no_certificate("-log(1+x) default box", r)


def test_neglog_two_var_constrained_default_box_not_false_optimal():
    """``min -log(x)-log(y) s.t. x+y >= 2`` on the default box: the constraint is slack
    at the stall (zero multiplier), so the refutation must still reach the corner."""
    m = dm.Model("neg_log2_default")
    x = m.continuous("x", lb=1e-3)
    y = m.continuous("y", lb=1e-3)
    m.subject_to(x + y >= 2)
    m.minimize(-dm.log(x) - dm.log(y))
    r = m.solve(nlp_solver="ipm")
    _assert_sound("two-var default box", r, -2.0 * math.log(_BOX_UB))
    _assert_no_certificate("two-var default box", r)


def test_maximize_log_default_box_not_false_optimal():
    """``max log(x), x >= 1`` on the default box. Box optimum +46.05; a valid upper
    bound must be >= it. Before the fix the certified bound was +18.97 (below max)."""
    m = dm.Model("max_log_default")
    x = m.continuous("x", lb=1.0)
    m.maximize(dm.log(x))
    r = m.solve(nlp_solver="ipm")
    opt = math.log(_BOX_UB)
    if r.bound is not None:
        assert r.bound >= opt - 1e-4 * opt, f"UNSOUND upper bound {r.bound!r} < max {opt:.6g}"
    _assert_no_certificate("max log default box", r)


def test_default_box_interior_optimum_still_certifies():
    """A GENUINE interior optimum on the default box must stay certified: the
    refutation now forms a vertex at 9.999e19, but ``L`` there is far worse, so no
    witness exists. ``min x^2/2 - log(x)`` has its optimum at x=1, obj=0.5."""
    m = dm.Model("interior_default")
    x = m.continuous("x", lb=1e-3)
    m.minimize(0.5 * x**2 - dm.log(x))
    r = m.solve(nlp_solver="ipm")
    assert r.status == "optimal" and r.gap_certified, (
        f"genuine interior optimum lost its certificate (status={r.status}, "
        f"gap_certified={r.gap_certified})"
    )
    assert r.objective == pytest.approx(0.5, rel=1e-5, abs=1e-6)


def test_certificate_gap_helper_refutes_on_default_box():
    """Unit test: with ub = DEFAULT_VARIABLE_BOUND the helper must refute the stall
    (it returned a certifiable tuple before the fix)."""
    from discopt.solver import _convex_nlp_certificate_gap, _make_evaluator

    m = dm.Model("neg_log_default_unit")
    xv = m.continuous("x", lb=1.0)
    m.minimize(-dm.log(xv))
    ev = _make_evaluator(m)
    lb, ub = np.array([1.0]), np.array([_BOX_UB])
    cl = cu = np.empty(0)
    x_stall = np.array([1.7339e8])
    out = _convex_nlp_certificate_gap(ev, x_stall, None, lb, ub, cl, cu, -math.log(x_stall[0]))
    assert out is None, f"default-box stall must be refuted, got {out}"
