"""Gershgorin decision slack must be sized by the side under test.

The PSD/NSD tests in ``eigenvalue.py`` accept a row whose Gershgorin bound
misses zero by a rounding-sized slack. That slack used to be sized by each
diagonal entry's ``max(|inf|, |sup|)``, so a huge *negative* ``inf(H_ii)``
licensed a genuinely positive ``sup(H_ii)`` in the NSD test (and the mirror for
PSD). ``sqrt(exp(x))`` on ``[-20, 20]`` has ``H = [-6.3e29, 5.3e12]``; the slack
was ``2e15`` and the convex function was certified *concave*, which made a
nonconvex ``g >= c`` row look convex and sent the model to the local-NLP fast
path with ``gap_certified=True`` on a wrong optimum.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.convexity import certify_convex
from discopt._relax.convexity.eigenvalue import (
    gershgorin_certifies_nsd,
    gershgorin_certifies_psd,
)
from discopt._relax.convexity.interval import Interval
from discopt._relax.convexity.lattice import Curvature


def test_one_sided_diagonal_does_not_license_other_side():
    # 1x1 interval Hessian [-6.3e29, 5.3e12]: indefinite in sign, neither test passes.
    H = Interval(lo=np.array([[-6.3e29]]), hi=np.array([[5.3e12]]))
    assert not gershgorin_certifies_nsd(H)
    H2 = Interval(lo=np.array([[-5.3e12]]), hi=np.array([[6.3e29]]))
    assert not gershgorin_certifies_psd(H2)


def test_exact_zero_boundary_still_certifies():
    # The slack's purpose is unchanged: a boundary-PSD row whose Gershgorin
    # bound rounds a hair below zero still certifies.
    H = Interval(lo=np.array([[1.0, -1.0], [-1.0, 1.0]]), hi=np.array([[1.0, -1.0], [-1.0, 1.0]]))
    assert gershgorin_certifies_psd(H)
    assert gershgorin_certifies_nsd(-H)


# (id, builder, lb, ub, verdict that would be wrong)
_CASES = [
    ("sqrt(exp(x))", lambda x: dm.sqrt(dm.exp(x)), -20.0, 20.0, Curvature.CONCAVE),
    ("log(exp(x*(x+1)))", lambda x: dm.log(dm.exp(x * (x + 1.0))), 1e-3, 10.0, Curvature.CONCAVE),
    ("sqrt(x**2+1) x<=0", lambda x: dm.sqrt(x**2 + 1.0), None, 0.0, Curvature.CONCAVE),
    ("sqrt(1/x) x>=2", lambda x: dm.sqrt(1.0 / x), 2.0, None, Curvature.CONCAVE),
    ("x/(x+1) x>=2", lambda x: x / (x + 1.0), 2.0, None, Curvature.CONVEX),
    ("x**4-x**2 wide", lambda x: x**4 - x**2, -1e7, 1e7, Curvature.CONVEX),
]


@pytest.mark.parametrize("name,build,lb,ub,forbidden", _CASES, ids=[c[0] for c in _CASES])
def test_certify_convex_never_returns_wrong_curvature(name, build, lb, ub, forbidden):
    m = dm.Model()
    kw = {}
    if lb is not None:
        kw["lb"] = lb
    if ub is not None:
        kw["ub"] = ub
    x = m.continuous("x", **kw)
    verdict = certify_convex(build(x), m)
    assert verdict is not forbidden, f"{name}: certified {verdict}"


def test_sqrt_exp_row_not_sent_to_local_fast_path():
    # min (x-1.5)^2 s.t. 3*sqrt(exp(-x)) + sqrt(exp(x)) >= 10 on [-20, 20].
    # The feasible set is two disjoint intervals; the global optimum is at
    # x ~ 4.5423 (obj ~ 9.2556). Before the fix the model was classified
    # convex and the local solve certified x ~ -2.345 (obj 14.784).
    m = dm.Model()
    x = m.continuous("x", lb=-20, ub=20)
    m.minimize((x - 1.5) ** 2)
    m.subject_to(3 * dm.sqrt(dm.exp(-x)) + dm.sqrt(dm.exp(x)) >= 10)
    r = m.solve(time_limit=60)
    xs = np.linspace(-20, 20, 800001)
    feas = 3 * np.exp(-xs / 2) + np.exp(xs / 2) >= 10
    best = float(((xs[feas] - 1.5) ** 2).min())
    assert r.status == "optimal"
    assert r.objective <= best + 1e-3, (r.objective, best)
