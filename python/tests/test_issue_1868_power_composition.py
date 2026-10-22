"""Issue #1868: the convexity detector returned UNKNOWN for ``c ** f`` (c > 0) and for
``g ** n`` with a non-affine ``g`` of proven sign.

``cvxnonsep_pcon20/30/40`` (and their ``r`` variants) are convex in MINLPLib: their
one nonlinear row is ``sqr(sum_k 2 ** (affine_k)) <= 57600``. Both pieces classified
UNKNOWN, so the solve went to spatial B&B and never closed the gap (SCIP: 1.1 s / 6.3 s).

Measured over the 161 MINLPLib instances flagged convex with <= 200 vars/cons:
``classify_model`` proved 149 convex before, 155 after (the six ``pcon``), none lost;
over 230 instances flagged NOT convex, 0 newly classified convex.

A false CONVEX verdict is a false certificate, so every verdict here is also checked
numerically -- midpoint convexity/concavity on sampled pairs -- independently of the
rules that produced it.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.convexity import classify_expr
from discopt._relax.convexity.lattice import Curvature
from discopt.modeling.core import Model

pytestmark = pytest.mark.unit

BOX = {"x": (0.0, 5.0), "y": (0.0, 5.0), "z": (-5.0, 5.0)}


def _model():
    m = Model("pow")
    v = {k: m.continuous(k, lb=lo, ub=hi) for k, (lo, hi) in BOX.items()}
    return m, v["x"], v["y"], v["z"]


# (label, builder(x, y, z) -> expression, numpy f(x, y, z), expected curvature)
CASES = [
    ("2**(x+y)", lambda x, y, z: 2 ** (x + y), lambda x, y, z: 2.0 ** (x + y), "CONVEX"),
    ("0.5**x", lambda x, y, z: 0.5**x, lambda x, y, z: 0.5**x, "CONVEX"),
    ("0.5**(-x*x)", lambda x, y, z: 0.5 ** (-(x * x)), lambda x, y, z: 0.5 ** (-(x * x)), "CONVEX"),
    ("1**z", lambda x, y, z: 1**z, lambda x, y, z: 1.0**z, "AFFINE"),
    (
        "(2**x+2**y)**2",
        lambda x, y, z: (2**x + 2**y) ** 2,
        lambda x, y, z: (2.0**x + 2.0**y) ** 2,
        "CONVEX",
    ),
    (
        "(exp(x)+exp(y))**2",
        lambda x, y, z: (dm.exp(x) + dm.exp(y)) ** 2,
        lambda x, y, z: (np.exp(x) + np.exp(y)) ** 2,
        "CONVEX",
    ),
    ("exp(z)**3", lambda x, y, z: dm.exp(z) ** 3, lambda x, y, z: np.exp(z) ** 3, "CONVEX"),
    ("(-exp(z))**2", lambda x, y, z: (-dm.exp(z)) ** 2, lambda x, y, z: np.exp(z) ** 2, "CONVEX"),
    (
        "(-exp(z))**3",
        lambda x, y, z: (-dm.exp(z)) ** 3,
        lambda x, y, z: -(np.exp(z) ** 3),
        "CONCAVE",
    ),
    ("(z*z)**2", lambda x, y, z: (z * z) ** 2, lambda x, y, z: (z * z) ** 2, "CONVEX"),
    # Must abstain: the inner has no proven sign, and the composite is nonconvex
    # where 2**x < 1.5 (second derivative 2 ln^2(2) 2^x (2*2^x - 3)).
    ("(2**x-3)**2", lambda x, y, z: (2**x - 3) ** 2, lambda x, y, z: (2.0**x - 3) ** 2, "UNKNOWN"),
    # Must abstain: a non-positive base is not a real exponential.
    ("(-2)**z", lambda x, y, z: (-2) ** z, None, "UNKNOWN"),
    ("0**z", lambda x, y, z: 0**z, None, "UNKNOWN"),
]


def _midpoint_violation(f, sense: str, n: int = 4000, seed: int = 1868) -> float:
    """Largest midpoint-inequality violation over random pairs in BOX (0 if none)."""
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in BOX.values()])
    hi = np.array([b[1] for b in BOX.values()])
    a = rng.uniform(lo, hi, size=(n, 3))
    b = rng.uniform(lo, hi, size=(n, 3))
    fa, fb, fm = f(*a.T), f(*b.T), f(*((a + b) / 2).T)
    gap = fm - (fa + fb) / 2  # <= 0 for convex, >= 0 for concave
    scale = 1.0 + np.abs(fa) + np.abs(fb)
    viol = (gap if sense == "CONVEX" else -gap) / scale
    return float(max(0.0, viol.max()))


@pytest.mark.parametrize("label,build,f,expected", CASES, ids=[c[0] for c in CASES])
def test_power_composition_verdicts(label, build, f, expected):
    m, x, y, z = _model()
    got = classify_expr(build(x, y, z), m)
    assert got == getattr(Curvature, expected), (label, got)
    # Independent numerical check of every non-abstaining curvature claim.
    if expected in ("CONVEX", "CONCAVE"):
        assert _midpoint_violation(f, expected) <= 1e-9, label


def test_the_abstaining_composite_really_is_nonconvex():
    """Pin the guard's premise: ``(2**x - 3)**2`` violates midpoint convexity on the
    box, so classifying it CONVEX would be a false certificate."""
    assert _midpoint_violation(lambda x, y, z: (2.0**x - 3) ** 2, "CONVEX") > 1e-3


def test_cvxnonsep_pcon_shape_classifies_convex():
    """The pcon row shape -- ``sqr(sum 2 ** (x_i + x_{i+1}))`` over a box -- end to end
    through ``classify_model``, without needing the MINLPLib file."""
    from discopt._relax.convexity import classify_model

    m = Model("pcon_shape")
    xs = [m.continuous(f"x{i}", lb=0.0, ub=4.0) for i in range(6)]
    ints = [m.integer(f"i{i}", lb=0, ub=4) for i in range(4)]
    allv = ints + xs
    row = sum(2 ** (allv[k] + allv[k + 1]) for k in range(len(allv) - 1)) ** 2
    m.subject_to(row <= 57600)
    m.minimize(-sum(0.5 * v for v in allv))
    ok, mask = classify_model(m)
    assert ok and all(mask), mask
