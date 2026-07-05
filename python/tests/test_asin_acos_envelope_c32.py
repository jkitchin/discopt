"""C-32 regression: relax_asin / relax_acos must be sound envelopes.

`asin''(x) = x*(1-x**2)**(-3/2)` -> asin is CONVEX on [0, 1] and CONCAVE on
[-1, 0]; `acos''(x) = -asin''(x)` -> acos is the mirror. The pre-fix code
inverted the curvature regime (treated `lb >= 0` as concave for asin), so the
"convex underestimator" `cv` was returned ABOVE the function -> an unsound
McCormick envelope in the LIVE JAX relaxation layer -> invalid dual bound ->
risk of certifying a wrong optimum.

These tests call the relaxation primitives DIRECTLY on OFF-DIAGONAL boxes (a
grid of x-values spread across a non-degenerate box, plus randomized sub-boxes
of [-1, 1]) so the buggy branch is exercised. On the pre-fix code every
crossing assertion fails; after the fix `cv <= f <= cc` holds with zero
crossings on the JAX backend.
"""

import numpy as np
import pytest
from discopt._jax import mccormick as jm

pytestmark = [pytest.mark.unit, pytest.mark.smoke]

TOL = 1e-9

_BACKENDS = [
    pytest.param(jm, id="jax"),
]


def _envelope_max_crossing(relax_fn, f, lb, ub, n=41):
    """Return (max(cv - f), max(f - cc)) over a grid across [lb, ub].

    Both should be <= 0 (up to tol) for a sound envelope: cv underestimates
    and cc overestimates f everywhere on the box.
    """
    xs = np.linspace(lb, ub, n)
    cv, cc = relax_fn(xs, lb, ub)
    cv = np.asarray(cv, dtype=float)
    cc = np.asarray(cc, dtype=float)
    fx = f(xs)
    return float(np.max(cv - fx)), float(np.max(fx - cc))


@pytest.mark.parametrize("mod", _BACKENDS)
def test_asin_literal_repro_is_sound(mod):
    """The issue's literal repro: relax_asin(0.5) on [0.1, 0.9].

    Pre-fix returned cv=0.609968 > true=0.523599 (unsound). Post-fix cv must
    not exceed the true value.
    """
    cv, cc = mod.relax_asin(np.array(0.5), 0.1, 0.9)
    true = float(np.arcsin(0.5))
    assert float(cv) <= true + TOL, f"cv={float(cv)} > asin(0.5)={true} (unsound)"
    assert float(cc) >= true - TOL, f"cc={float(cc)} < asin(0.5)={true} (unsound)"


@pytest.mark.parametrize("mod", _BACKENDS)
@pytest.mark.parametrize(
    "lb,ub",
    [
        (0.1, 0.9),  # one-signed, convex region for asin
        (0.0, 0.95),
        (-0.9, -0.1),  # one-signed, concave region for asin
        (-0.95, 0.0),
        (-0.9, 0.9),  # zero-straddling (inflection at 0)
        (-0.99, 0.5),
        (-0.5, 0.99),
    ],
)
def test_asin_envelope_no_crossing(mod, lb, ub):
    over, under = _envelope_max_crossing(mod.relax_asin, np.arcsin, lb, ub)
    assert over <= TOL, f"asin cv exceeds f by {over} on [{lb},{ub}]"
    assert under <= TOL, f"asin cc below f by {under} on [{lb},{ub}]"


@pytest.mark.parametrize("mod", _BACKENDS)
@pytest.mark.parametrize(
    "lb,ub",
    [
        (0.1, 0.9),  # one-signed, concave region for acos
        (0.0, 0.95),
        (-0.9, -0.1),  # one-signed, convex region for acos
        (-0.95, 0.0),
        (-0.9, 0.9),  # zero-straddling (inflection at 0)
        (-0.99, 0.5),
        (-0.5, 0.99),
    ],
)
def test_acos_envelope_no_crossing(mod, lb, ub):
    over, under = _envelope_max_crossing(mod.relax_acos, np.arccos, lb, ub)
    assert over <= TOL, f"acos cv exceeds f by {over} on [{lb},{ub}]"
    assert under <= TOL, f"acos cc below f by {under} on [{lb},{ub}]"


@pytest.mark.parametrize("mod", _BACKENDS)
@pytest.mark.parametrize("fn_name", ["relax_asin", "relax_acos"])
def test_envelope_property_random_subboxes(mod, fn_name):
    """Property test: no envelope crossing over random sub-boxes of [-1, 1].

    Encodes the false-certificate CLASS (off-diagonal, univariate over a
    non-degenerate box), not a single named box, so a future refactor that
    reintroduces the inverted regime in any guise trips it.
    """
    relax_fn = getattr(mod, fn_name)
    f = np.arcsin if fn_name == "relax_asin" else np.arccos
    rng = np.random.default_rng(20260703)
    n_cross = 0
    worst = 0.0
    for _ in range(500):
        a, b = np.sort(rng.uniform(-0.99, 0.99, size=2))
        if b - a < 1e-3:
            continue
        over, under = _envelope_max_crossing(relax_fn, f, a, b, n=25)
        crossing = max(over, under)
        if crossing > TOL:
            n_cross += 1
            worst = max(worst, crossing)
    assert n_cross == 0, f"{fn_name}: {n_cross} crossing sub-boxes, worst={worst}"
