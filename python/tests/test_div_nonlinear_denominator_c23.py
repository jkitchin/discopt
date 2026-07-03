"""C-23 regression: relax_div must be a sound envelope for NONLINEAR denominators.

``relax_div`` composes ``x/y = x * (1/y)`` and (pre-fix) reciprocated the
denominator at the MIDPOINT of its relaxation interval. When the denominator is a
bare variable or affine, that interval point-collapses so ``1/mid`` is exact and
the composition is sound. When the denominator is NONLINEAR (``x*y``, ``x*x``,
``sqrt(x*y)``, ``x*y+1``, ...) the interval is non-degenerate, ``1/mid`` sits
ABOVE the true ``1/(.)``, and the "convex underestimator" ``cv`` exceeds ``f``
(e.g. ``1/(x*y)`` on ``[0.3,2]x[0.4,1.8]`` gave ``cv=1.334 > 1.0``) -> invalid
dual bound -> risk of a wrong certificate.

The fix keeps the tight bilinear composition where the denominator relaxation
collapses to a point, and falls back to the SOUND interval enclosure
``[x_lb,x_ub] * [1/y_ub, 1/y_lb]`` when the denominator interval is
non-degenerate — never emitting ``cv > f``.

Two layers of coverage:
  1. Direct primitive test on ``relax_div`` with a non-degenerate denominator
     interval (JAX and numpy backends).
  2. Full-compiler containment on ``1/(x*y)``, ``x/(y*z)``, ``1/(x*x)``,
     ``1/sqrt(x*y)`` plus the sound ``1/x`` / ``x/y`` controls — the
     reciprocal-/division-of-nonlinear-inner case the harness previously omitted.
"""

import numpy as np
import pytest
from discopt._jax import mccormick as jm
from discopt._numpy import mccormick as nm
from relaxation_harness import build_relaxation, evaluate

pytestmark = [pytest.mark.unit, pytest.mark.smoke]

TOL = 1e-7

_BACKENDS = [
    pytest.param(jm, id="jax"),
    pytest.param(nm, id="numpy"),
]


# --------------------------------------------------------------------------
# Layer 1: direct primitive test on relax_div with a non-degenerate denominator
# --------------------------------------------------------------------------


def _div_max_crossing(relax_fn, x_lb, x_ub, y_lb, y_ub, n=21):
    """Max (cv - x/y) and (x/y - cc) over the numerator x denominator box.

    ``relax_div`` returns a scalar (cv, cc) that must bracket x/y for EVERY
    (x_true, y_true) in [x_lb,x_ub] x [y_lb,y_ub]. The compiler evaluates it at
    the interval midpoints, so we call it the same way and check containment over
    the whole box.
    """
    x = 0.5 * (x_lb + x_ub)
    y = 0.5 * (y_lb + y_ub)
    cv, cc = relax_fn(np.array(x), np.array(y), x_lb, x_ub, y_lb, y_ub)
    cv = float(cv)
    cc = float(cc)
    xs = np.linspace(x_lb, x_ub, n)
    ys = np.linspace(y_lb, y_ub, n)
    XX, YY = np.meshgrid(xs, ys)
    fx = XX / YY
    return float(np.max(cv - fx)), float(np.max(fx - cc))


@pytest.mark.parametrize("mod", _BACKENDS)
def test_div_nonlinear_denominator_literal_repro(mod):
    """1/(x*y) style: numerator collapsed to 1, denominator interval [0.58, 1.56].

    Pre-fix cv=1.334 > true 1.0. Post-fix the enclosure must bracket every point.
    """
    # numerator '1' -> x_lb == x_ub == 1 ; denominator relax interval [0.58, 1.56]
    over, under = _div_max_crossing(mod.relax_div, 1.0, 1.0, 0.58, 1.56)
    assert over <= TOL, f"div cv exceeds f by {over} (nonlinear-denominator underestimator)"
    assert under <= TOL, f"div cc below f by {under}"


@pytest.mark.parametrize("mod", _BACKENDS)
@pytest.mark.parametrize(
    "x_lb,x_ub,y_lb,y_ub",
    [
        (1.0, 1.0, 0.58, 1.56),  # 1 / nonlinear positive denom
        (0.3, 2.0, 0.5, 1.5),  # numerator interval too, positive denom
        (-2.0, 2.0, 0.5, 1.5),  # numerator straddles 0, positive denom
        (1.0, 1.0, -1.56, -0.58),  # nonlinear NEGATIVE denom
        (-2.0, 1.0, -1.5, -0.5),  # both straddle / negative denom
    ],
)
def test_div_nondegenerate_denominator_never_crosses(mod, x_lb, x_ub, y_lb, y_ub):
    over, under = _div_max_crossing(mod.relax_div, x_lb, x_ub, y_lb, y_ub)
    assert over <= TOL, f"div cv exceeds f by {over} on [{x_lb},{x_ub}]/[{y_lb},{y_ub}]"
    assert under <= TOL, f"div cc below f by {under} on [{x_lb},{x_ub}]/[{y_lb},{y_ub}]"


@pytest.mark.parametrize("mod", _BACKENDS)
def test_div_point_denominator_stays_tight(mod):
    """Collapsed (linear/constant) denominator keeps the exact bilinear value."""
    # y_lb == y_ub -> 1/y exact; cv == cc == x/y at the point.
    cv, cc = mod.relax_div(np.array(3.0), np.array(2.0), 3.0, 3.0, 2.0, 2.0)
    assert abs(float(cv) - 1.5) <= TOL and abs(float(cc) - 1.5) <= TOL


@pytest.mark.parametrize("mod", _BACKENDS)
def test_div_random_subboxes_no_crossing(mod):
    """Property test over random positive/negative denominator sub-boxes."""
    rng = np.random.default_rng(20260703)
    n_cross = 0
    worst = 0.0
    for _ in range(600):
        # positive or negative denominator (0 excluded)
        if rng.random() < 0.5:
            y_lb = rng.uniform(0.2, 3.0)
            y_ub = y_lb + rng.uniform(0.0, 3.0)
        else:
            y_ub = rng.uniform(-3.0, -0.2)
            y_lb = y_ub - rng.uniform(0.0, 3.0)
        x_lb = rng.uniform(-3.0, 3.0)
        x_ub = x_lb + rng.uniform(0.0, 3.0)
        over, under = _div_max_crossing(mod.relax_div, x_lb, x_ub, y_lb, y_ub, n=13)
        crossing = max(over, under)
        if crossing > TOL:
            n_cross += 1
            worst = max(worst, crossing)
    assert n_cross == 0, f"relax_div: {n_cross} crossing sub-boxes, worst={worst}"


# --------------------------------------------------------------------------
# Layer 2: full-compiler containment (the harness's missing nonlinear-inner case)
# --------------------------------------------------------------------------


def _compiler_worst_crossing(expr_fn, bounds, n=4000, seed=0):
    relax_fn, true_fn, lb, ub = build_relaxation(expr_fn, bounds)
    rng = np.random.default_rng(seed)
    w = np.asarray(ub) - np.asarray(lb)
    xs = np.asarray(lb) + rng.uniform(0.0, 1.0, size=(n, len(bounds))) * w
    import jax.numpy as jnp

    cv, cc, f = evaluate(relax_fn, true_fn, lb, ub, jnp.array(xs, dtype=jnp.float64))
    fin = np.isfinite(cv) & np.isfinite(cc) & np.isfinite(f)
    over = float(np.max((cv - f)[fin])) if fin.any() else 0.0
    under = float(np.max((f - cc)[fin])) if fin.any() else 0.0
    return over, under


@pytest.mark.parametrize(
    "expr_fn,bounds,label",
    [
        (lambda x, y: 1 / (x * y), [(0.3, 2.0), (0.4, 1.8)], "1/(x*y)"),
        (lambda x, y, z: x / (y * z), [(0.5, 2.0), (0.4, 1.8), (0.5, 1.5)], "x/(y*z)"),
        (lambda x: 1 / (x * x), [(0.3, 2.0)], "1/(x*x)"),
        (lambda x, y: 1 / (x * y) ** 0.5, [(0.3, 2.0), (0.4, 1.8)], "1/sqrt(x*y)"),
        (lambda x, y: (x + 1) / (x * y), [(0.3, 2.0), (0.4, 1.8)], "(x+1)/(x*y)"),
        (lambda x, y: 1 / (x * y + 1), [(0.3, 2.0), (0.4, 1.8)], "1/(x*y+1)"),
        # sound controls (linear/point denominators) must remain sound.
        (lambda x: 1 / x, [(0.3, 2.0)], "1/x"),
        (lambda x, y: x / y, [(0.3, 2.0), (0.4, 1.8)], "x/y"),
        (lambda x, y: 1 / (x + y), [(0.3, 2.0), (0.4, 1.8)], "1/(x+y)"),
    ],
)
def test_compiler_division_containment(expr_fn, bounds, label):
    over, under = _compiler_worst_crossing(expr_fn, bounds)
    assert over <= 1e-6, f"[{label}] cv exceeds f by {over} (invalid underestimator)"
    assert under <= 1e-6, f"[{label}] cc below f by {under} (invalid overestimator)"
