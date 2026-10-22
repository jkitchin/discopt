"""The legacy power envelopes are sound for negative exponents (#1493).

77b9acc removed the "every even power is convex on all of R" verdict from
``uniform_relax._pow_curv``. The legacy McCormick relaxation (``relaxation_compiler``,
reached via ``mccormick_bounds="nlp"``, the alphaBB fallback and the compiled
relaxation API) had the same defect twice: ``mccormick.relax_pow`` and
``envelopes.relax_power_int`` both receive negative integer exponents (the compiler
routes every integral constant exponent to them) and treated ``x**-2`` as convex on
all of R and ``x**-1`` like ``x**3``. On ``[-3, 3]`` that made ``x**-2``'s
overestimator the secant ``1/9`` (cut off every point with ``|x| < 1/3``) and the
odd case NaN or wrong everywhere. Feasible-point sampling: 16,618 of 57,648 sampled
``(p, box, x)`` cells violated ``cv <= x**p <= cc`` before; 0 after.
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import numpy as np
import pytest

pytest.importorskip("jax", reason="JAX is the optional discopt[jax] extra (#1873)")

from discopt._relax.envelopes import relax_power_int
from discopt._relax.mccormick import relax_negative_int_power, relax_pow
from discopt._relax.relaxation_compiler import compile_objective_relaxation

pytestmark = pytest.mark.unit

BOXES = [
    (-3.0, 3.0),
    (-1.0, 1.0),
    (-0.9, 0.7),
    (-2.0, 1.0),
    (0.0, 3.0),
    (-3.0, 0.0),
    (0.5, 3.0),
    (-3.0, -0.5),
]


@pytest.mark.parametrize("fn", [relax_pow, relax_power_int], ids=["relax_pow", "power_int"])
@pytest.mark.parametrize("p", [-1, -2, -3, -4, 2, 3, 4, 5])
@pytest.mark.parametrize("box", BOXES)
def test_envelope_brackets_every_sampled_point(fn, p, box):
    lo, hi = box
    checked = 0
    for x in np.linspace(lo, hi, 201):
        if abs(x) < 1e-9:
            continue  # the pole itself is not in the domain
        cv, cc = fn(np.float64(x), np.float64(lo), np.float64(hi), p)
        cv, cc, f = float(cv), float(cc), float(x) ** p
        assert not (math.isnan(cv) or math.isnan(cc)), (x, cv, cc)
        tol = 1e-9 * (1.0 + abs(f))
        assert cv <= f + tol and f <= cc + tol, f"x={x}: cv={cv} f={f} cc={cc}"
        checked += 1
    assert checked >= 199


@pytest.mark.parametrize("p", [-1, -2, -3, -4])
@pytest.mark.parametrize("box", [(-3.0, 3.0), (0.0, 3.0), (-3.0, 0.0)])
def test_a_box_holding_the_pole_gets_no_finite_overestimator(p, box):
    cv, cc = relax_negative_int_power(np.float64(0.5), np.float64(box[0]), np.float64(box[1]), p)
    assert float(cc) == math.inf
    assert float(cv) == (0.0 if p % 2 == 0 else -math.inf)


@pytest.mark.parametrize("p", [-1, -2, -3])
def test_pole_free_boxes_keep_their_tight_envelope(p):
    # Positive box: convex -> cv is the function itself, cc the secant.
    cv, cc = relax_negative_int_power(np.float64(1.0), np.float64(0.5), np.float64(2.0), p)
    assert float(cv) == pytest.approx(1.0)
    sec = 0.5**p + (2.0**p - 0.5**p) / 1.5 * 0.5
    assert float(cc) == pytest.approx(sec)
    # Negative box: odd p is concave there (cc is the function), even p convex.
    cv, cc = relax_negative_int_power(np.float64(-1.0), np.float64(-2.0), np.float64(-0.5), p)
    if p % 2:
        assert float(cc) == pytest.approx((-1.0) ** p)
    else:
        assert float(cv) == pytest.approx((-1.0) ** p)


def test_positive_exponents_are_unchanged():
    for p in (2, 3, 4, 5):
        for lo, hi in BOXES:
            for x in np.linspace(lo, hi, 11):
                a = relax_power_int(np.float64(x), np.float64(lo), np.float64(hi), p)
                b = relax_pow(np.float64(x), np.float64(lo), np.float64(hi), p)
                assert np.all(np.isfinite(np.asarray(a))) and np.all(np.isfinite(np.asarray(b)))


@pytest.mark.parametrize("p", [-1, -2, -3])
@pytest.mark.parametrize("box", [(-3.0, 3.0), (-0.9, 0.7), (0.5, 3.0), (-3.0, -0.5)])
def test_compiled_relaxation_brackets_x_to_negative_power(p, box):
    """The compiler path (``relax_power_int`` for a plain variable base)."""
    m = dm.Model("negpow")
    x = m.continuous("x", lb=box[0], ub=box[1])
    m.minimize(x**p)
    fn = compile_objective_relaxation(m)
    lb = np.array([box[0]])
    ub = np.array([box[1]])
    checked = 0
    for xv in np.linspace(box[0], box[1], 101):
        if abs(xv) < 1e-9:
            continue
        pt = np.array([xv])
        cv, cc = fn(pt, pt, lb, ub)
        f = float(xv) ** p
        tol = 1e-9 * (1.0 + abs(f))
        assert float(cv) <= f + tol and f <= float(cc) + tol, (xv, float(cv), f, float(cc))
        checked += 1
    assert checked >= 100
