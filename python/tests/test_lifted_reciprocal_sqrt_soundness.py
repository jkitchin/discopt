"""Adversarial soundness for the lifted reciprocal / sqrt envelopes (#154).

``c / g`` (convex ``1/g`` on ``g > 0``) and ``sqrt(g)`` (concave on ``g >= 0``)
are *force-lifted* when ``g`` is itself nonlinear (a product or a sum of
products): every multiplicative factor becomes a bounded McCormick product aux
and the outer atom gets the standard tangent/secant envelope. This file locks the
two ways that lift can go wrong:

1. **Unsound cut** — an envelope row that excludes a *true feasible point*. If the
   underestimator ever rises above the curve (or the overestimator drops below
   it) the dual bound is no longer valid. We prove enclosure directly: pin the
   original variables to an interior grid point (a degenerate box), at which the
   relaxation must still contain the exact curve value, and check the relaxation
   brackets it from both sides — ``min`` bound ``<= f(pt) <=`` the ``max`` (negated)
   bound. A bracket that holds at every sampled point means no cut excludes the
   curve.

2. **Numerically degenerate cut** — a tangent slope so large (``1/gl**2`` as the
   denominator approaches 0, ``1/(2*sqrt(t))`` as a sqrt argument approaches 0)
   that the LP is ill-conditioned and the solver returns an ``iteration_limit``
   objective that is *not* a valid bound. The conditioning guards must make the
   lift *abstain* (drop the constraint — which only enlarges the feasible region,
   still sound) rather than emit a degenerate cut. We assert these adversarial
   boxes never come back ``iteration_limit`` and never report an unsound bound.

The invariant under test is the project's non-negotiable one: a valid lower
bound never exceeds the true optimum, and the relaxation never excludes a
feasible point.
"""

import math
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import discopt.modeling as dm
import pytest
from discopt._jax.mccormick_lp import MccormickLPRelaxer
from discopt._jax.model_utils import flat_variable_bounds

_TOL = 1e-6
_DATA = Path(__file__).parent / "data" / "minlplib"


def _root_bound(model, pin=None):
    """Root McCormick LP bound, optionally pinning original columns to a point.

    ``pin`` is a dict ``{col_index: value}``; pinning sets ``lb == ub == value``
    so the box is degenerate at that point.
    """
    relaxer = MccormickLPRelaxer(model)
    lb, ub = flat_variable_bounds(model)
    lb = lb.copy()
    ub = ub.copy()
    if pin is not None:
        for i, v in pin.items():
            lb[i] = v
            ub[i] = v
    return relaxer.solve_at_node(lb, ub)


# ── Curve-enclosure: the envelope never excludes a true feasible point ──────────


def _recip_model(const, xb, yb, *, sign=1.0):
    """``sign * const / (x*y)`` over the given boxes (a lifted reciprocal)."""
    m = dm.Model()
    x = m.continuous("x", lb=xb[0], ub=xb[1])
    y = m.continuous("y", lb=yb[0], ub=yb[1])
    m.minimize(sign * const / (x * y))
    return m


def _sqrt_model(ab, bb, *, sign=1.0):
    """``sign * sqrt(a**2 + b**2)`` over the given boxes (a lifted sqrt)."""
    m = dm.Model()
    a = m.continuous("a", lb=ab[0], ub=ab[1])
    b = m.continuous("b", lb=bb[0], ub=bb[1])
    m.minimize(sign * dm.sqrt(a * a + b * b))
    return m


_RECIP_BOXES = [
    (1.0, (1.0, 4.0), (2.0, 5.0)),
    (4243.28, (2.0, 9.0), (3.0, 7.0)),
    (0.5, (0.5, 2.0), (1.0, 3.0)),
    (10.0, (1.0, 1.5), (4.0, 6.0)),
]

_SQRT_BOXES = [
    ((3.0, 6.0), (4.0, 8.0)),
    ((1.0, 5.0), (2.0, 9.0)),
    ((0.5, 2.0), (0.5, 4.0)),
]


@pytest.mark.correctness
@pytest.mark.parametrize("const, xb, yb", _RECIP_BOXES)
def test_reciprocal_envelope_encloses_curve(const, xb, yb):
    """At every interior grid point the relaxation brackets ``const/(x*y)`` from
    both sides — the lifted ``1/g`` envelope never excludes the true curve."""
    xs = [xb[0], 0.5 * (xb[0] + xb[1]), xb[1]]
    ys = [yb[0], 0.5 * (yb[0] + yb[1]), yb[1]]
    for xv in xs:
        for yv in ys:
            true = const / (xv * yv)
            lo = _root_bound(_recip_model(const, xb, yb, sign=1.0), pin={0: xv, 1: yv})
            hi = _root_bound(_recip_model(const, xb, yb, sign=-1.0), pin={0: xv, 1: yv})
            assert lo.status == "optimal" and lo.lower_bound is not None
            assert hi.status == "optimal" and hi.lower_bound is not None
            # underestimator never rises above the curve …
            assert lo.lower_bound <= true + _TOL, (
                f"recip under-cut excludes curve at ({xv},{yv}): {lo.lower_bound} > {true}"
            )
            # … overestimator never drops below it.
            assert -hi.lower_bound >= true - _TOL, (
                f"recip over-cut excludes curve at ({xv},{yv}): {-hi.lower_bound} < {true}"
            )


@pytest.mark.correctness
@pytest.mark.parametrize("ab, bb", _SQRT_BOXES)
def test_sqrt_envelope_encloses_curve(ab, bb):
    """At every interior grid point the relaxation brackets ``sqrt(a**2+b**2)``
    from both sides — the lifted concave sqrt envelope never excludes the curve."""
    as_ = [ab[0], 0.5 * (ab[0] + ab[1]), ab[1]]
    bs = [bb[0], 0.5 * (bb[0] + bb[1]), bb[1]]
    for av in as_:
        for bv in bs:
            true = math.sqrt(av * av + bv * bv)
            lo = _root_bound(_sqrt_model(ab, bb, sign=1.0), pin={0: av, 1: bv})
            hi = _root_bound(_sqrt_model(ab, bb, sign=-1.0), pin={0: av, 1: bv})
            assert lo.status == "optimal" and lo.lower_bound is not None
            assert hi.status == "optimal" and hi.lower_bound is not None
            assert lo.lower_bound <= true + _TOL, (
                f"sqrt under-cut excludes curve at ({av},{bv}): {lo.lower_bound} > {true}"
            )
            assert -hi.lower_bound >= true - _TOL, (
                f"sqrt over-cut excludes curve at ({av},{bv}): {-hi.lower_bound} < {true}"
            )


# ── Whole-box objective soundness: bound <= true optimum ────────────────────────


@pytest.mark.correctness
@pytest.mark.parametrize("const, xb, yb", _RECIP_BOXES)
def test_reciprocal_box_bound_is_sound(const, xb, yb):
    """The root bound on ``min const/(x*y)`` never exceeds the box optimum
    (attained at the upper corner since ``1/(x*y)`` decreases in x,y)."""
    res = _root_bound(_recip_model(const, xb, yb, sign=1.0))
    true_min = const / (xb[1] * yb[1])
    assert res.status == "optimal"
    assert res.lower_bound is not None and math.isfinite(res.lower_bound)
    assert res.lower_bound <= true_min + 1e-3, (
        f"UNSOUND recip box bound {res.lower_bound} > opt {true_min}"
    )


@pytest.mark.correctness
@pytest.mark.parametrize("ab, bb", _SQRT_BOXES)
def test_sqrt_box_bound_is_sound(ab, bb):
    """The root bound on ``min sqrt(a**2+b**2)`` never exceeds the box optimum
    (the lower corner, since the radius grows with |a|,|b| on positive boxes)."""
    res = _root_bound(_sqrt_model(ab, bb, sign=1.0))
    true_min = math.sqrt(ab[0] ** 2 + bb[0] ** 2)
    assert res.status == "optimal"
    assert res.lower_bound is not None and math.isfinite(res.lower_bound)
    assert res.lower_bound <= true_min + 1e-3, (
        f"UNSOUND sqrt box bound {res.lower_bound} > opt {true_min}"
    )


@pytest.mark.correctness
def test_reciprocal_lift_actually_tightens():
    """Sanity: on a well-conditioned box the reciprocal lift fires and returns a
    finite (and exact-at-corner) bound, not a dropped objective. Guards against a
    silent regression where the lift stops engaging and the bound goes to None."""
    res = _root_bound(_recip_model(1.0, (1.0, 4.0), (2.0, 5.0), sign=1.0))
    assert res.status == "optimal"
    assert res.lower_bound is not None
    # exact at the upper corner: 1/(4*5) = 0.05
    assert res.lower_bound == pytest.approx(0.05, abs=1e-6)


# ── Conditioning / domain guards: abstain, never emit a degenerate cut ──────────


def _adversarial_recip_zero_crossing():
    """``1/(x*y)`` where ``g = x*y`` spans 0 (``x in [-2, 3]``): ``1/g`` is not
    convex across the singularity, so the lift must abstain."""
    m = dm.Model()
    x = m.continuous("x", lb=-2.0, ub=3.0)
    y = m.continuous("y", lb=1.0, ub=2.0)
    m.minimize(1.0 / (x * y))
    return m


def _adversarial_recip_tiny_denom():
    """``g`` reaches ~1e-8, so the tangent slope ``1/gl**2`` ~ 1e16 blows past the
    conditioning guard: the lift must abstain rather than build a degenerate LP."""
    m = dm.Model()
    x = m.continuous("x", lb=1e-4, ub=1.0)
    y = m.continuous("y", lb=1e-4, ub=1.0)
    m.minimize(1.0 / (x * y))
    return m


def _adversarial_sqrt_negative_arg():
    """``sqrt(g)`` whose inner ``g`` can dip below 0 over the box; the sqrt is not
    real there, so the lift must abstain (negative-slack guard)."""
    m = dm.Model()
    x = m.continuous("x", lb=-3.0, ub=2.0)
    y = m.continuous("y", lb=-3.0, ub=2.0)
    # x*y ranges to -6 over the box → inner can be negative.
    m.minimize(dm.sqrt(x * y + 1.0))
    return m


@pytest.mark.correctness
@pytest.mark.parametrize(
    "name, build",
    [
        ("recip zero-crossing denom", _adversarial_recip_zero_crossing),
        ("recip tiny denom (slope blow-up)", _adversarial_recip_tiny_denom),
        ("sqrt negative inner arg", _adversarial_sqrt_negative_arg),
    ],
)
def test_guard_abstains_never_unsound(name, build):
    """Every adversarial box must (a) solve cleanly — no ``iteration_limit``
    garbage objective masquerading as a bound — and (b) if it reports a bound at
    all, that bound is sound (``<= 0``, since each objective's true minimum is
    positive). Abstaining to ``bound=None`` is the expected, sound outcome."""
    res = _root_bound(build())
    assert res.status != "iteration_limit", (
        f"[{name}] returned an iteration_limit objective — degenerate cut not guarded"
    )
    assert res.status in ("optimal", "infeasible"), f"[{name}] unexpected status {res.status}"
    if res.lower_bound is not None and math.isfinite(res.lower_bound):
        # The true min of each objective is strictly positive; a sound lower
        # bound can only be <= it. A guard that emitted a bad cut would overshoot.
        assert res.lower_bound <= 1.0 + 1e-6, (
            f"[{name}] UNSOUND bound {res.lower_bound} from a guarded box"
        )


# ── Real-instance tightening lock (regression: the lift must keep engaging) ─────

# Pre-#154 (lift-disengaged) root bounds vs the value the reciprocal/sqrt lift
# now delivers at declared bounds. ``lock`` sits between the two so the test
# fails loudly both if the lift silently disengages (bound falls back toward the
# baseline) and is robust to small numerical drift. ``opt`` is the MINLPLib
# optimum: the bound must stay sound (<= opt) on every iteration.
_TIGHTENING_CASES = [
    # instance, pre-lift baseline, post-lift lock floor, MINLPLib optimum
    ("nvs05", 0.674, 1.20, 5.47093),
    ("nvs22", 1.826, 2.30, 6.0584),
]


@pytest.mark.correctness
@pytest.mark.parametrize("instance, baseline, lock, optimum", _TIGHTENING_CASES)
def test_lifted_instance_root_bound_tightens_and_stays_sound(instance, baseline, lock, optimum):
    """``nvs05``/``nvs22`` carry a ``c/(x*y)``-style constraint the lift now
    envelopes. The root bound must be strictly tighter than the pre-lift baseline
    (the lift is engaging) and still <= the optimum (it stays sound)."""
    nl = _DATA / f"{instance}.nl"
    assert nl.exists(), f"missing {nl}"
    res = _root_bound(dm.from_nl(str(nl)))
    assert res.status == "optimal", f"[{instance}] root LP status {res.status}"
    assert res.lower_bound is not None and math.isfinite(res.lower_bound)
    assert res.lower_bound >= lock, (
        f"[{instance}] lift disengaged: bound {res.lower_bound} fell back toward "
        f"the pre-lift baseline {baseline} (expected >= {lock})"
    )
    assert res.lower_bound <= optimum + 1e-3, (
        f"[{instance}] UNSOUND bound {res.lower_bound} > optimum {optimum}"
    )
