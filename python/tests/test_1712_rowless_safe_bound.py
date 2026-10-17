"""#1712 follow-up: a row-less node LP certifies from a rigorous bound, not on trust.

``min asin(x)`` relaxes to an LP with no rows -- a cost over a box. The warm simplex,
which computes the Neumaier-Shcherbina bound, needs rows, so the generic path
solved it and ``_certify`` took the vertex objective through its ``trusted_vertex``
arm. Over a box the LP optimum is closed form; ``_rowless_lp_safe_bound`` computes
it rounded down, and the generic path now reports it as ``safe_bound``.
"""

from __future__ import annotations

import math
from fractions import Fraction

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.mccormick_lp import BOUND_PROVENANCE, reset_bound_provenance
from discopt._relax.milp_relaxation import _rowless_lp_safe_bound

pytestmark = [pytest.mark.smoke]


def _exact_box_min(c, bounds, offset) -> Fraction:
    total = Fraction(offset)
    for cj, (lo, hi) in zip(c, bounds):
        if cj != 0.0:
            total += Fraction(cj) * Fraction(lo if cj > 0 else hi)
    return total


def test_rowless_bound_never_exceeds_the_exact_optimum():
    rng = np.random.default_rng(1712)
    checked = tight = 0
    for _ in range(2000):
        n = int(rng.integers(1, 12))
        scale = 10.0 ** rng.uniform(-8, 8, n)
        c = rng.normal(size=n) * scale
        c[rng.random(n) < 0.2] = 0.0
        lo = rng.normal(size=n) * 10.0 ** rng.uniform(-3, 6, n)
        hi = lo + rng.uniform(0, 1e3, n)
        bounds = list(zip(lo.tolist(), hi.tolist()))
        offset = float(rng.normal() * 1e3)
        got = _rowless_lp_safe_bound(c, bounds, offset)
        assert got is not None
        exact = _exact_box_min(c.tolist(), bounds, offset)
        assert Fraction(got) <= exact, (got, float(exact))
        # Rounded down by a few ulps, not by a meaningful amount.
        if abs(float(exact) - got) <= 1e-9 * max(1.0, abs(float(exact))):
            tight += 1
        checked += 1
    assert checked == 2000
    assert tight == checked


def test_rowless_bound_declines_an_unbounded_improving_side():
    inf = math.inf
    assert _rowless_lp_safe_bound(np.array([1.0]), [(-inf, 0.0)], 0.0) is None
    assert _rowless_lp_safe_bound(np.array([-1.0]), [(0.0, 1e20)], 0.0) is None
    # A free column with zero cost costs nothing, and the improving side is all
    # that matters for a nonzero cost.
    assert _rowless_lp_safe_bound(np.array([0.0, 2.0]), [(-inf, inf), (1.0, inf)], 0.0) == (
        pytest.approx(2.0)
    )


@pytest.mark.parametrize(
    "fn,sense", [(dm.asin, "min"), (dm.asin, "max"), (dm.acos, "min"), (dm.acos, "max")]
)
def test_inverse_trig_certifies_from_the_safe_bound(fn, sense):
    reset_bound_provenance()
    m = dm.Model("trig")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    (m.minimize if sense == "min" else m.maximize)(fn(x))
    r = m.solve(time_limit=60)
    assert r.status == "optimal"
    tally = dict(BOUND_PROVENANCE)
    assert tally.get("ns_safe_bound", 0) > 0, tally  # the probe saw certified nodes
    assert tally.get("trusted_vertex", 0) == 0, tally
