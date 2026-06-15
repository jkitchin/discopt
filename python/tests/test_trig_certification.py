"""Soundness locks for periodic transcendental instances (issue #137).

The global benchmark corpus exercised no ``sin`` / ``cos`` / ``tan`` (nor
``tanh``) terms, and the periodic family has no DCP curvature rule, so the
convexity certificate classifies them as UNKNOWN. The concern in #137 is
soundness: a periodic term on a wide interval can collapse the McCormick
enclosure to the trivial ``[-1, 1]`` range, and ``tan`` has asymptotes that, if
mishandled, would fabricate an invalid objective bound.

This module pins a growing set of ``sin`` / ``cos`` / ``tan`` / ``tanh``
instances with **known or calculable global optima**, each asserting the
acceptance invariant from #137:

    a valid dual bound never exceeds the known optimum (``bound <= opt + tol``),
    and the solver never reports a false certified optimum.

These are hand-constructed / classic global-optimization test functions (their
optima are derivable in closed form), so the suite is self-contained and needs
no network fetch. They are the ground truth against which tighter periodic
certification rules (piecewise-McCormick keyed to monotone branches,
asymptote-aware ``tan`` handling, spatial branching on the periodic argument)
can later be measured.

Note on the current engine: discopt has no periodic-specific envelope yet, but
the alphaBB underestimator fallback already yields *sound* bounds on these boxed
instances and certifies the simple ones. The ``tan`` asymptote case is handled
by *declining* to bound (``bound=None``) rather than fabricating one — also
sound. Scope note: surveying the full MINLPLib for o37/o38/o41/o46 instances
with published optima (issue #137, second bullet) needs network access to
minlplib.org and is a follow-on; the periodic family is essentially absent from
the vendored subset and rare upstream.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import math

import discopt.modeling as dm
import pytest

# Soundness slack: a valid dual bound must not exceed the known optimum by more
# than this. Kept tight; periodic terms must not silently produce bound > opt.
_SOUND_TOL = 1e-3


@pytest.mark.correctness
@pytest.mark.parametrize(
    "name, func, lb, ub, opt",
    [
        # Monotone branches (no extremum interior to the box): the relaxation
        # has a single convex/concave piece, so it should certify exactly.
        ("sin_increasing", dm.sin, 0.0, math.pi / 2, 0.0),  # sin(0) = 0
        ("cos_monotone", dm.cos, 0.0, math.pi, -1.0),  # cos(pi) = -1
        ("tan_no_asymptote", dm.tan, -1.0, 1.0, math.tan(-1.0)),  # tan(-1)
        ("tanh_monotone", dm.tanh, -2.0, 2.0, math.tanh(-2.0)),  # tanh(-2)
        # Periodic full-range boxes: the interior extremum is the global min.
        ("sin_full_period", dm.sin, 0.0, 2 * math.pi, -1.0),  # min at 3pi/2
        ("cos_full_period", dm.cos, 0.0, 2 * math.pi, -1.0),  # min at pi
    ],
)
def test_single_trig_term_certifies_sound_bound(name, func, lb, ub, opt):
    """``min f(x)`` for a single periodic atom over a box: sound, certified bound."""
    m = dm.Model()
    x = m.continuous("x", lb=lb, ub=ub)
    m.minimize(func(x))
    r = m.solve(time_limit=20, gap_tolerance=1e-4)

    assert r.objective is not None and r.bound is not None, f"[{name}] no bound produced"
    # Incumbent reaches the known global optimum.
    assert math.isclose(r.objective, opt, abs_tol=1e-4), f"[{name}] obj={r.objective} != {opt}"
    # Soundness invariant (#137): the dual bound never exceeds the optimum.
    assert r.bound <= opt + _SOUND_TOL, f"[{name}] unsound dual bound {r.bound} > {opt}"
    assert r.bound <= r.objective + 1e-6, f"[{name}] dual bound {r.bound} > incumbent"
    assert r.gap_certified, f"[{name}] expected certified optimality on a single branch"


@pytest.mark.correctness
def test_multimodal_x_plus_sin_certifies_sound_bound():
    """``min x + sin(3x)`` over ``[0, 2pi]``: global min is 0 at x=0, sound bound.

    On ``[0, 2pi]`` the term ``sin(3x)`` is positive wherever it could pull the
    objective below ``f(0)=0`` (``x < 1`` requires ``3x < pi``), so the global
    minimum is exactly ``0`` at the left endpoint.
    """
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=2 * math.pi)
    m.minimize(x + dm.sin(3 * x))
    r = m.solve(time_limit=25, gap_tolerance=1e-4)

    assert r.objective is not None and r.bound is not None
    assert math.isclose(r.objective, 0.0, abs_tol=1e-4), f"obj={r.objective} != 0"
    assert r.bound <= 0.0 + _SOUND_TOL, f"unsound dual bound {r.bound} > 0"
    assert r.bound <= r.objective + 1e-6, "dual bound must not exceed the incumbent"


@pytest.mark.correctness
@pytest.mark.parametrize("dim", [1, 2])
def test_rastrigin_restricted_certifies_global_zero(dim):
    """Rastrigin restriction to ``[-1, 1]^dim`` (cos-based): global min 0 at the origin."""
    m = dm.Model()
    xs = [m.continuous(f"x{i}", lb=-1.0, ub=1.0) for i in range(dim)]
    expr = 10 * dim
    for xi in xs:
        expr = expr + xi * xi - 10 * dm.cos(2 * math.pi * xi)
    m.minimize(expr)
    r = m.solve(time_limit=30, gap_tolerance=1e-4)

    assert r.objective is not None and r.bound is not None, f"[rastrigin{dim}d] no bound"
    assert math.isclose(r.objective, 0.0, abs_tol=1e-4), f"[rastrigin{dim}d] obj={r.objective} != 0"
    assert r.bound <= 0.0 + _SOUND_TOL, f"[rastrigin{dim}d] unsound bound {r.bound} > 0"
    assert r.bound <= r.objective + 1e-6


@pytest.mark.correctness
def test_schwefel_restricted_certifies_sound_bound():
    """Schwefel restriction (``sin`` of a nonlinear argument) certifies soundly.

    ``f(x) = 418.9829 - x*sin(sqrt(|x|))`` over ``[400, 440]`` has its global
    minimum near ``x* = 420.9687`` with ``f(x*) ~= 0`` (the constant ``418.9829``
    is the canonical per-dimension offset). The point of the instance is to
    exercise ``sin`` of a *nonlinear* argument, not the offset's last digits, so
    soundness is asserted tightly and the incumbent to a looser tolerance.
    """
    m = dm.Model()
    x = m.continuous("x", lb=400.0, ub=440.0)
    m.minimize(418.9829 - x * dm.sin(dm.sqrt(dm.abs(x))))
    r = m.solve(time_limit=30, gap_tolerance=1e-4)

    assert r.objective is not None and r.bound is not None
    assert abs(r.objective - 0.0) <= 1e-2, f"obj={r.objective} not near the known min 0"
    assert r.bound <= 0.0 + 1e-2, f"unsound dual bound {r.bound} > known optimum 0"
    assert r.bound <= r.objective + 1e-6, "dual bound must not exceed the incumbent"


@pytest.mark.correctness
def test_tan_crossing_asymptote_stays_sound():
    """``min tan(x)`` over a box straddling ``pi/2`` must not fabricate a bound.

    ``tan`` is unbounded below as ``x -> (pi/2)+`` inside ``[1, 2]``, so there is
    no finite optimum. The sound behavior (documented in ``test_amp.py``) is to
    *decline* to relax the asymptote-crossing term: discopt returns a feasible
    incumbent with ``bound=None`` and does **not** certify a false optimum. This
    locks that a periodic asymptote never yields ``bound > opt`` or a bogus
    "optimal".
    """
    m = dm.Model()
    x = m.continuous("x", lb=1.0, ub=2.0)
    m.minimize(dm.tan(x))
    r = m.solve(time_limit=20, gap_tolerance=1e-4)

    # The solver must not claim a certified optimum across the asymptote.
    assert not r.gap_certified, "must not certify optimality across a tan asymptote"
    assert r.bound is None, f"expected declined bound (None), got {r.bound}"
