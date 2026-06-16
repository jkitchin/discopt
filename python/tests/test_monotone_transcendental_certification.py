"""Soundness + certification locks for monotone transcendental atoms (issue #136).

The global benchmark corpus exercised no inverse-trig / inverse-hyperbolic /
error / ``log1p`` atoms, even though every one of them is **monotone** on its
domain (or piecewise-monotone with a single known inflection at the origin), so
a *sound* interval enclosure is just the image of the endpoints. This is the
"easy, sound-today" half of the transcendental certification gap (the periodic
group — sin/cos/tan — is tracked in issue #137).

This module pins a growing set of ``atan`` / ``asin`` / ``acos`` / ``asinh`` /
``acosh`` / ``atanh`` / ``erf`` / ``log1p`` instances with **known/calculable
global optima**. Each asserts the acceptance invariant from #136:

    a valid dual bound never exceeds the known optimum (``bound <= opt + tol``)
    and the solver never reports a false certified optimum.

The companion change wires monotone interval enclosures for these atoms into
``convexity/interval.py`` + ``interval_eval.py`` and curvature profiles into
``convexity/lattice.py``, so the convexity certificate stops abstaining on them
(it previously returned an unbounded enclosure / ``UNKNOWN`` curvature). The
unit tests below lock that behavior directly.

Scope note: surveying the full MINLPLib for o37/o38/o40/o45/o47/o49–o53 (plus
erf/log1p) instances with published optima — the issue's "then survey full
MINLPLib" item — needs network access to minlplib.org and is a follow-on; these
hand-constructed instances are the self-contained ground truth.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import math

import discopt.modeling as dm
import numpy as np
import pytest

# Soundness slack: a valid dual bound must not exceed the known optimum by more
# than this.
_SOUND_TOL = 1e-3


# ──────────────────────────────────────────────────────────────────────
# Benchmark instances: single monotone atom over a box with a known optimum.
# For a nondecreasing atom the global min is at the left endpoint; for the
# nonincreasing ``acos`` it is at the right endpoint.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.correctness
@pytest.mark.parametrize(
    "name, func, lb, ub, opt",
    [
        # Defined on all of R (inflection at the origin):
        ("atan", dm.atan, -2.0, 3.0, math.atan(-2.0)),
        ("asinh", dm.asinh, -2.0, 3.0, math.asinh(-2.0)),
        ("erf", dm.erf, -2.0, 2.0, math.erf(-2.0)),
        # Restricted domain [-1, 1]:
        ("asin", dm.asin, -0.9, 0.8, math.asin(-0.9)),
        ("acos", dm.acos, -0.9, 0.8, math.acos(0.8)),  # nonincreasing -> min at ub
        # Restricted domain (-1, 1) / [1, inf):
        ("atanh", dm.atanh, -0.9, 0.9, math.atanh(-0.9)),
        ("acosh", dm.acosh, 1.5, 5.0, math.acosh(1.5)),  # min at lb (away from x=1)
        # Domain (-1, inf):
        ("log1p", dm.log1p, 0.0, 5.0, math.log1p(0.0)),
    ],
)
def test_monotone_atom_certifies_sound_bound(name, func, lb, ub, opt):
    """``min f(x)`` for a single monotone atom certifies its optimum soundly."""
    m = dm.Model()
    x = m.continuous("x", lb=lb, ub=ub)
    m.minimize(func(x))
    r = m.solve(time_limit=20, gap_tolerance=1e-4)

    assert r.objective is not None and r.bound is not None, f"[{name}] no bound produced"
    # Incumbent reaches the known global optimum.
    assert math.isclose(r.objective, opt, abs_tol=1e-4), f"[{name}] obj={r.objective} != {opt}"
    # Soundness invariant (#136): the dual bound never exceeds the optimum.
    assert r.bound <= opt + _SOUND_TOL, f"[{name}] unsound dual bound {r.bound} > {opt}"
    assert r.bound <= r.objective + 1e-6, f"[{name}] dual bound {r.bound} > incumbent"
    assert r.gap_certified, f"[{name}] expected certified optimality on a monotone branch"


@pytest.mark.correctness
def test_acosh_singular_boundary_stays_sound():
    """``min acosh(x)`` over ``[1, 5]``: the optimum sits at the singular x=1.

    ``acosh'(1) = +inf``, so the local NLP solver cannot reach the boundary
    optimum and the relaxation gap does not close. The sound behavior is to
    return a *feasible* (suboptimal) incumbent without certifying — never a
    false "optimal" and never ``bound > opt``. This locks that the singular
    boundary degrades gracefully rather than fabricating a certificate.
    """
    m = dm.Model()
    x = m.continuous("x", lb=1.0, ub=5.0)
    m.minimize(dm.acosh(x))
    r = m.solve(time_limit=20, gap_tolerance=1e-4)

    opt = 0.0  # acosh(1) = 0
    # No false certificate: either uncertified, or a sound certified bound.
    if r.bound is not None:
        assert r.bound <= opt + _SOUND_TOL, f"unsound dual bound {r.bound} > {opt}"
    if r.gap_certified:
        # If it ever does certify, the incumbent must be the true optimum.
        assert math.isclose(r.objective, opt, abs_tol=1e-2), "certified a non-optimal incumbent"
    # The feasible incumbent is itself a sound upper bound (>= opt).
    assert r.objective is not None and r.objective >= opt - 1e-6


# ──────────────────────────────────────────────────────────────────────
# Unit locks for the wiring that makes the certificate stop abstaining.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.correctness
@pytest.mark.parametrize(
    "name, func, lb, ub, expected",
    [
        ("asin", dm.asin, 0.0, 0.9, "convex"),  # asin'' = x/(1-x^2)^1.5 > 0 on (0,1)
        ("asin", dm.asin, -0.9, 0.0, "concave"),
        ("atanh", dm.atanh, 0.0, 0.9, "convex"),
        ("atanh", dm.atanh, -0.9, 0.0, "concave"),
        ("atan", dm.atan, 0.0, 3.0, "concave"),  # atan'' = -2x/(1+x^2)^2 < 0 on x>0
        ("atan", dm.atan, -3.0, 0.0, "convex"),
        ("asinh", dm.asinh, 0.0, 3.0, "concave"),
        ("asinh", dm.asinh, -3.0, 0.0, "convex"),
        ("erf", dm.erf, 0.0, 2.0, "concave"),
        ("erf", dm.erf, -2.0, 0.0, "convex"),
        ("acos", dm.acos, 0.0, 0.9, "concave"),
        ("acos", dm.acos, -0.9, 0.0, "convex"),
        ("acosh", dm.acosh, 1.0, 5.0, "concave"),  # concave on its whole domain
        ("log1p", dm.log1p, 0.0, 5.0, "concave"),  # concave on its whole domain
    ],
)
def test_monotone_atom_curvature_classified(name, func, lb, ub, expected):
    """The convexity certificate now classifies these atoms (issue #136 wiring).

    Before the wiring, every atom here returned ``UNKNOWN`` (the certificate's
    interval evaluator produced an unbounded enclosure and the lattice had no
    profile). It must now prove the correct sign-restricted curvature.
    """
    from discopt._jax.convexity import Curvature, classify_expr

    m = dm.Model()
    x = m.continuous("x", lb=lb, ub=ub)
    curv = classify_expr(func(x), m)
    want = Curvature.CONVEX if expected == "convex" else Curvature.CONCAVE
    assert curv == want, f"[{name} on [{lb},{ub}]] classified {curv}, expected {want}"


@pytest.mark.correctness
@pytest.mark.parametrize(
    "fn, npf, lb, ub",
    [
        ("atan", np.arctan, -2.0, 3.0),
        ("asin", np.arcsin, -0.9, 0.8),
        ("acos", np.arccos, -0.9, 0.8),
        ("asinh", np.arcsinh, -2.0, 3.0),
        ("acosh", np.arccosh, 1.2, 5.0),
        ("atanh", np.arctanh, -0.9, 0.9),
        ("log1p", np.log1p, 0.0, 5.0),
    ],
)
def test_monotone_interval_enclosure_is_sound(fn, npf, lb, ub):
    """The interval enclosure soundly contains the true image of the box."""
    from discopt._jax.convexity import interval as iv
    from discopt._jax.convexity.interval import Interval

    enc = getattr(iv, fn)(Interval.from_bounds(lb, ub))
    samples = npf(np.linspace(lb, ub, 200))
    assert float(enc.lo) <= float(samples.min()) + 1e-9, f"[{fn}] lower endpoint not sound"
    assert float(enc.hi) >= float(samples.max()) - 1e-9, f"[{fn}] upper endpoint not sound"


@pytest.mark.correctness
def test_out_of_domain_interval_abstains():
    """Out-of-domain / asymptote endpoints collapse to a conservative ±inf."""
    from discopt._jax.convexity import interval as iv
    from discopt._jax.convexity.interval import Interval

    # asin below its domain -> unbounded lower endpoint (sound abstention).
    assert not np.isfinite(iv.asin(Interval.from_bounds(-2.0, 0.5)).lo)
    # acosh below its domain (x < 1) -> unbounded lower endpoint.
    assert not np.isfinite(iv.acosh(Interval.from_bounds(0.5, 5.0)).lo)
    # atanh touching its asymptote at +1 -> unbounded upper endpoint.
    assert not np.isfinite(iv.atanh(Interval.from_bounds(0.0, 1.0)).hi)
