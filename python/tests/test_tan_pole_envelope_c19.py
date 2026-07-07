"""C-19 regression: relax_tan must not draw a secant across a tan pole.

``tan`` diverges at each pole ``pi/2 + k*pi``. The pre-fix code centered on the
nearest inflection ``k*pi`` and classified the box as a single convex/concave
branch via ``lb >= center`` / ``ub <= center`` without checking that ``[lb, ub]``
stays inside one continuous branch ``(-pi/2 + k*pi, pi/2 + k*pi)``. For a
pole-straddling box (e.g. ``[1.4, 1.8]`` straddling ``pi/2 ~= 1.5708``) it drew a
secant ACROSS the pole -> the "convex underestimator" ``cv`` exceeded ``tan`` by
hundreds of thousands on part of the branch (invalid envelope -> invalid dual
bound -> risk of a wrong certificate).

The fix *abstains* on any pole-straddling box: it emits the no-information
envelope ``(-inf, +inf)`` (never a crossing finite envelope), leaving FBBT /
spatial branching to shrink the box below the pole spacing. Pole-free boxes keep
the tight branch envelope.

These tests call ``relax_tan`` DIRECTLY on straddling and pole-free boxes so the
buggy branch is exercised on the JAX backend. The property test
encodes the false-certificate CLASS (a finite envelope that crosses ``tan`` on a
pole-straddling box), not a single named box.
"""

import numpy as np
import pytest
from discopt._jax import mccormick as jm

pytestmark = [pytest.mark.unit, pytest.mark.smoke]

TOL = 1e-7
# Stay away from the pole itself where tan -> +-inf and float values are useless.
_POLE_GUARD = 5e-3

_BACKENDS = [
    pytest.param(jm, id="jax"),
]


def _pole_distance(xs):
    """Distance from each x to the nearest tan pole pi/2 + k*pi."""
    return np.abs(((xs + np.pi / 2) % np.pi) - np.pi / 2)


def _finite_max_crossing(relax_fn, lb, ub, n=401):
    """Max (cv - tan) and (tan - cc) over FINITE envelope points on [lb, ub].

    Abstained points (cv=-inf, cc=+inf) carry no information and are excluded;
    what must never happen is a *finite* envelope that crosses tan.
    """
    xs = np.linspace(lb, ub, n)
    keep = _pole_distance(xs) > _POLE_GUARD
    xs = xs[keep]
    cv, cc = relax_fn(xs, lb, ub)
    cv = np.asarray(cv, dtype=float)
    cc = np.asarray(cc, dtype=float)
    fx = np.tan(xs)
    finite = np.isfinite(cv) & np.isfinite(cc) & np.isfinite(fx)
    if not finite.any():
        return -np.inf, -np.inf, 0
    over = float(np.max(cv[finite] - fx[finite]))
    under = float(np.max(fx[finite] - cc[finite]))
    return over, under, int(finite.sum())


@pytest.mark.parametrize("mod", _BACKENDS)
def test_tan_straddling_pole_literal_repro_is_sound(mod):
    """The card's literal repro: relax_tan on [1.4, 1.8] evaluated at x=1.5.

    Pre-fix ``cv`` (secant across the pole) massively exceeds tan(1.5)~=14.10.
    Post-fix the box straddles a pole so the envelope must abstain (cv=-inf).
    """
    cv, cc = mod.relax_tan(np.array(1.5), 1.4, 1.8)
    cv = float(cv)
    cc = float(cc)
    true = float(np.tan(1.5))
    # Sound either way: abstain (-inf/+inf) or a non-crossing finite envelope.
    assert cv <= true + TOL, f"cv={cv} > tan(1.5)={true} (secant across pole)"
    assert cc >= true - TOL, f"cc={cc} < tan(1.5)={true} (secant across pole)"
    # The fix's chosen behavior is to abstain across the pole.
    assert cv == -np.inf and cc == np.inf, "straddling box must abstain, not emit a finite envelope"


@pytest.mark.parametrize("mod", _BACKENDS)
@pytest.mark.parametrize(
    "lb,ub",
    [
        (1.4, 1.8),  # straddles +pi/2
        (-1.8, -1.4),  # straddles -pi/2
        (1.0, 2.2),  # wide straddle of +pi/2
        (1.5, 4.8),  # spans more than one full period (contains a pole)
        (4.4, 4.9),  # straddles 3*pi/2 ~= 4.712
    ],
)
def test_tan_pole_straddling_never_crosses(mod, lb, ub):
    """A finite tan envelope must never cross the function across a pole."""
    over, under, _ = _finite_max_crossing(mod.relax_tan, lb, ub)
    assert over <= TOL, f"tan cv exceeds f by {over} on straddling [{lb},{ub}]"
    assert under <= TOL, f"tan cc below f by {under} on straddling [{lb},{ub}]"


@pytest.mark.parametrize("mod", _BACKENDS)
@pytest.mark.parametrize(
    "lb,ub",
    [
        (0.1, 1.4),  # pole-free, convex half of principal branch
        (-1.4, -0.1),  # pole-free, concave half
        (-1.4, 1.4),  # pole-free, spans the inflection at 0
        (3.3, 4.6),  # pole-free branch centered at pi (k=1)
        (-4.6, -3.3),  # pole-free branch centered at -pi
    ],
)
def test_tan_pole_free_stays_tight_and_sound(mod, lb, ub):
    """Pole-free boxes must keep a finite, sound (non-crossing) envelope."""
    over, under, n_finite = _finite_max_crossing(mod.relax_tan, lb, ub)
    assert n_finite > 0, f"pole-free [{lb},{ub}] unexpectedly abstained everywhere"
    assert over <= TOL, f"tan cv exceeds f by {over} on pole-free [{lb},{ub}]"
    assert under <= TOL, f"tan cc below f by {under} on pole-free [{lb},{ub}]"


@pytest.mark.parametrize("mod", _BACKENDS)
def test_tan_envelope_property_random_subboxes(mod):
    """Property test: no FINITE envelope crossing over random sub-boxes.

    Draws random boxes across several branches (some pole-straddling, some not).
    Whatever the box, a returned finite envelope must satisfy cv <= tan <= cc.
    """
    rng = np.random.default_rng(20260703)
    n_cross = 0
    worst = 0.0
    for _ in range(600):
        a, b = np.sort(rng.uniform(-5.0, 5.0, size=2))
        if b - a < 1e-3:
            continue
        over, under, _ = _finite_max_crossing(mod.relax_tan, a, b, n=61)
        crossing = max(over, under)
        if crossing > TOL:
            n_cross += 1
            worst = max(worst, crossing)
    assert n_cross == 0, f"relax_tan: {n_cross} crossing sub-boxes, worst={worst}"
