"""#1908: FBBT inverse rules must not cut feasible points near a function's limits.

Two defects in ``crates/discopt-core/src/presolve/fbbt.rs``, both of which removed
feasible points and so produced a wrong ``optimal`` or a false ``infeasible``:

1. The backward inverses of tanh/atan/sigmoid/erf/softplus clamped the output
   interval to ``limit -/+ 1e-12`` before inverting, so an output reaching the
   limit (``tanh(x) >= 0.5`` means ``tanh(x) in [0.5, 1]``) imposed a finite
   argument bound (``x <= 14.16``) instead of leaving that side open. The forward
   atanh image clamped its argument the same way.
2. The tan branch test, ``round(x / PI)`` per endpoint, put the double ``pi/2``
   (below the real pole) on branch 1 with ``x = 2``, so ``[pi/2, 2]`` -- which
   straddles the pole -- was treated as one branch with the inverted image
   ``[1.6e16, -2.19]``. The Python interval ``tan`` had the same test.

Each model below is from the issue; the oracle is the closed-form optimum.
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.convexity import interval as iv
from discopt._relax.convexity.interval import Interval
from discopt.tightening import fbbt_box


def _bimodal(f, c):
    m = dm.Model("bimodal")
    x = m.continuous("x", lb=0.0, ub=50.0)
    m.subject_to(f(x) >= c)
    # Two bumps: 10 at x = 2 and the global 20 at x = 45, both feasible.
    m.maximize(10 * dm.exp(-((x - 2) ** 2)) + 20 * dm.exp(-((x - 45) ** 2)))
    return m


@pytest.mark.parametrize(
    "name,f,c,threshold",
    [
        ("tanh", dm.tanh, 0.5, math.atanh(0.5)),
        ("sigmoid", dm.sigmoid, 0.6, math.log(0.6 / 0.4)),
        ("erf", dm.erf, 0.5, 0.4769362762044699),
    ],
)
def test_output_at_the_range_limit_keeps_the_far_optimum(name, f, c, threshold):
    m = _bimodal(f, c)
    b = fbbt_box(m)
    # The feasible set is [threshold, 50]; the clamp used to cap it near 14-27.
    assert float(b.ub[0]) == 50.0, (name, b.ub)
    assert threshold - 1e-9 <= float(b.lb[0]) <= threshold, (name, b.lb)
    r = m.solve(time_limit=60)
    assert r.status == "optimal", (name, r.status)
    assert r.objective == pytest.approx(20.0, abs=1e-6), (name, r.objective)
    assert float(r.x["x"]) == pytest.approx(45.0, abs=1e-4)


def test_atan_upper_side_stays_open():
    # atan(x) <= 2 holds for every x (2 > pi/2); the clamp imposed x <= ~1e12.
    m = dm.Model("atan")
    x = m.continuous("x", lb=0.0, ub=1e13)
    m.subject_to(dm.atan(x) <= 2.0)
    m.maximize(x)
    assert float(fbbt_box(m).ub[0]) == 1e13
    r = m.solve(time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(1e13, rel=1e-9)


def test_softplus_lower_side_stays_open():
    # softplus(x) >= 0 holds for every x; the clamp imposed x >= -27.6.
    m = dm.Model("softplus")
    x = m.continuous("x", lb=-50.0, ub=50.0)
    m.subject_to(dm.softplus(x) >= 0.0)
    m.minimize(x)
    assert float(fbbt_box(m).lb[0]) == -50.0
    r = m.solve(time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-50.0, abs=1e-6)


def test_atanh_near_one_is_feasible():
    m = dm.Model("atanh")
    x = m.continuous("x", lb=0.9, ub=1 - 1e-14)
    m.subject_to(dm.atanh(x) >= 14.5)
    m.minimize(x)
    witness = math.tanh(14.5)  # inside the box, atanh(witness) >= 14.5
    assert 0.9 <= witness <= 1 - 1e-14 and math.atanh(witness) >= 14.5
    # The limit caps the solve; it is not the claim under test. On CI the coverage
    # lane runs this solve 2-5x slower than the fast lane (measured in #1915:
    # 6.3 s fast vs 30.65 s, its 30 s limit), so the old 30 s limit let a
    # slow runner stop it before it certified.
    r = m.solve(time_limit=100)
    assert r.status == "optimal", r.status
    assert r.objective == pytest.approx(witness, abs=1e-6)
    assert math.atanh(float(r.x["x"])) >= 14.5 - 1e-6


def test_tan_box_starting_at_the_double_pi_over_2():
    m = dm.Model("tan")
    x = m.continuous("x", lb=math.pi / 2, ub=2.0)
    m.subject_to(dm.tan(x) <= -1.0)  # every x in (pi/2, 2] satisfies it
    m.minimize((x - 1.9) ** 2)
    r = m.solve(time_limit=30)
    assert r.status == "optimal", r.status
    assert r.objective == pytest.approx(0.0, abs=1e-6)
    assert float(r.x["x"]) == pytest.approx(1.9, abs=1e-3)


def test_python_interval_tan_across_a_pole_is_unbounded():
    out = iv.tan(Interval(np.float64(math.pi / 2), np.float64(2.0)))
    assert float(out.lo) == -math.inf and float(out.hi) == math.inf


def test_python_interval_tan_contains_sampled_points():
    rng = np.random.default_rng(1908)
    checks = 0
    pole_starts = [(k + 0.5) * math.pi for k in range(-6, 7)]
    los = np.concatenate([rng.uniform(-20, 20, 3000), pole_starts])
    widths = np.concatenate([4.0 * 1e-12 ** rng.uniform(0, 1, 3000), np.full(13, 0.3)])
    for a, w in zip(los, widths):
        b = a + w
        out = iv.tan(Interval(np.float64(a), np.float64(b)))
        assert float(out.lo) <= float(out.hi)
        for p in np.linspace(a, b, 5):
            v = math.tan(p)
            assert float(out.lo) <= v <= float(out.hi), (a, b, p, v, out)
            checks += 1
    assert checks == 5 * 3013
    # A box well inside one branch keeps a finite enclosure.
    inner = iv.tan(Interval(np.float64(1.6), np.float64(2.0)))
    assert math.isfinite(float(inner.lo)) and math.isfinite(float(inner.hi))
