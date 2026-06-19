"""Comprehensive factorable-lift coverage probe (promoted from an ad-hoc probe).

This guards the *whole class* of deficiency that ex1226 and the composite-base
fractional powers were instances of: a nonlinear term that is mathematically
liftable (each factor has a known envelope and the composition is a McCormick /
multilinear product of lifted columns) but that some recognizer in the lift chain
fails to see — so the constraint silently drops from the MILP relaxation (or gets
a loose envelope) and the dual bound freezes. The targeted regression tests in
``test_factorable_reform`` / ``test_bucket2_sound_bounds`` lock the specific
instances; this sweep locks the surrounding pattern space so a new representation
(power vs call, int/float/fraction/negative exponent, uni/multivariate, product /
sum / transcendental factor, ratios) cannot regress unnoticed.

For each pattern we mirror the solver: apply ``factorable_reformulate`` when
``has_factorable_work`` fires, build the MILP relaxation, and assert

  * the constraint is NOT dropped (no "omitting constraint" log), and
  * the root dual bound is finite and SOUND (``<= grid optimum``).

Tagged ``slow`` (~50 builds) so it is excluded from the default run; execute with
``pytest -m slow python/tests/test_factorable_lift_coverage.py``.
"""

from __future__ import annotations

import itertools
import logging
import math

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax import milp_relaxation as MR
from discopt._jax.factorable_reform import factorable_reformulate, has_factorable_work
from discopt._jax.mccormick_lp import MccormickLPRelaxer
from discopt._jax.model_utils import flat_variable_bounds

pytestmark = pytest.mark.slow


class _DropCatcher(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs: list[str] = []

    def emit(self, record):
        m = record.getMessage()
        if "omitting constraint" in m or "Cannot decompose" in m:
            self.msgs.append(m)


def _build_and_probe(build_model):
    """Reform (mirroring the solver) → build relaxation → (dropped, bound, status)."""
    MR._warned_messages.clear()
    catcher = _DropCatcher()
    lg = logging.getLogger("discopt._jax.milp_relaxation")
    prev = lg.level
    lg.setLevel(logging.WARNING)
    lg.addHandler(catcher)
    try:
        m = build_model()
        if has_factorable_work(m):
            m = factorable_reformulate(m)
        lb, ub = flat_variable_bounds(m)
        res = MccormickLPRelaxer(m).solve_at_node(lb, ub)
        return bool(catcher.msgs), res.lower_bound, res.status
    finally:
        lg.removeHandler(catcher)
        lg.setLevel(prev)


def _grid_opt(con, rhs, sense, coeffs, box, n=25):
    """Brute-force min of the linear objective over the box subject to the constraint.
    Coarse grid → an *upper* bound on the true optimum, fine for a sound-bound check."""
    axes = [np.linspace(lo, hi, n) for (lo, hi) in box]
    best = math.inf
    for combo in itertools.product(*axes):
        v = np.array(combo)
        cv = con(v)
        ok = (cv <= rhs + 1e-9) if sense == "<=" else (cv >= rhs - 1e-9)
        if ok:
            best = min(best, float(np.dot(coeffs, v)))
    return best


# (name, discopt-constraint builder, numpy evaluator, rhs, sense, obj coeffs, box)
B2 = [(1.0, 3.0)] * 2
B3 = [(1.0, 3.0)] * 3
B4 = [(1.0, 3.0)] * 4

_PATTERNS = [
    # single-factor powers — int / fraction / negative
    ("x**2", lambda v: v[0] ** 2, lambda v: v[0] ** 2, 4.0, "<=", [-1, 0], B2),
    ("x**3", lambda v: v[0] ** 3, lambda v: v[0] ** 3, 8.0, "<=", [-1, 0], B2),
    ("x**0.5", lambda v: v[0] ** 0.5, lambda v: v[0] ** 0.5, 1.5, "<=", [-1, 0], B2),
    ("x**1.5", lambda v: v[0] ** 1.5, lambda v: v[0] ** 1.5, 4.0, "<=", [-1, 0], B2),
    ("x**2.5", lambda v: v[0] ** 2.5, lambda v: v[0] ** 2.5, 9.0, "<=", [-1, 0], B2),
    ("x**-1", lambda v: v[0] ** -1, lambda v: v[0] ** -1.0, 0.9, ">=", [1, 0], B2),
    ("x**-2", lambda v: v[0] ** -2, lambda v: v[0] ** -2.0, 0.5, ">=", [1, 0], B2),
    ("x**-0.5", lambda v: v[0] ** -0.5, lambda v: v[0] ** -0.5, 0.8, ">=", [1, 0], B2),
    # two-factor products
    ("x*y", lambda v: v[0] * v[1], lambda v: v[0] * v[1], 4.0, "<=", [-1, -1], B2),
    ("x*y**2", lambda v: v[0] * v[1] ** 2, lambda v: v[0] * v[1] ** 2, 8.0, "<=", [-1, -1], B2),
    (
        "x*y**0.5",
        lambda v: v[0] * v[1] ** 0.5,
        lambda v: v[0] * v[1] ** 0.5,
        3.0,
        "<=",
        [-1, -1],
        B2,
    ),
    (
        "x**2*y**2",
        lambda v: v[0] ** 2 * v[1] ** 2,
        lambda v: v[0] ** 2 * v[1] ** 2,
        16.0,
        "<=",
        [-1, -1],
        B2,
    ),
    (
        "x**2*y**3",
        lambda v: v[0] ** 2 * v[1] ** 3,
        lambda v: v[0] ** 2 * v[1] ** 3,
        30.0,
        "<=",
        [-1, -1],
        B2,
    ),
    (
        "x**0.5*y**2 (ex1226)",
        lambda v: v[0] ** 0.5 * v[1] ** 2,
        lambda v: v[0] ** 0.5 * v[1] ** 2,
        6.0,
        "<=",
        [-1, -1],
        B2,
    ),
    (
        "x**2*y**0.5",
        lambda v: v[0] ** 2 * v[1] ** 0.5,
        lambda v: v[0] ** 2 * v[1] ** 0.5,
        6.0,
        "<=",
        [-1, -1],
        B2,
    ),
    (
        "x**0.5*y**0.5",
        lambda v: v[0] ** 0.5 * v[1] ** 0.5,
        lambda v: v[0] ** 0.5 * v[1] ** 0.5,
        2.0,
        "<=",
        [-1, -1],
        B2,
    ),
    (
        "x**1.5*y**1.5",
        lambda v: v[0] ** 1.5 * v[1] ** 1.5,
        lambda v: v[0] ** 1.5 * v[1] ** 1.5,
        8.0,
        "<=",
        [-1, -1],
        B2,
    ),
    # coefficient variants (float / negative)
    (
        "3.5*x**0.5*y**2",
        lambda v: 3.5 * v[0] ** 0.5 * v[1] ** 2,
        lambda v: 3.5 * v[0] ** 0.5 * v[1] ** 2,
        20.0,
        "<=",
        [-1, -1],
        B2,
    ),
    (
        "-2*x**2*y**2",
        lambda v: -2 * v[0] ** 2 * v[1] ** 2,
        lambda v: -2 * v[0] ** 2 * v[1] ** 2,
        -8.0,
        ">=",
        [-1, -1],
        B2,
    ),
    # transcendental factors
    ("exp(x)", lambda v: dm.exp(v[0]), lambda v: np.exp(v[0]), 10.0, "<=", [-1, 0], B2),
    ("sqrt(x)", lambda v: dm.sqrt(v[0]), lambda v: np.sqrt(v[0]), 1.5, "<=", [-1, 0], B2),
    ("log(x)", lambda v: dm.log(v[0]), lambda v: np.log(v[0]), 1.0, "<=", [-1, 0], B2),
    (
        "exp(x)*y",
        lambda v: dm.exp(v[0]) * v[1],
        lambda v: np.exp(v[0]) * v[1],
        15.0,
        "<=",
        [-1, -1],
        B2,
    ),
    (
        "sqrt(x)*y",
        lambda v: dm.sqrt(v[0]) * v[1],
        lambda v: np.sqrt(v[0]) * v[1],
        4.0,
        "<=",
        [-1, -1],
        B2,
    ),
    (
        "exp(x)*y**2",
        lambda v: dm.exp(v[0]) * v[1] ** 2,
        lambda v: np.exp(v[0]) * v[1] ** 2,
        30.0,
        "<=",
        [-1, -1],
        B2,
    ),
    (
        "x**2*exp(y)",
        lambda v: v[0] ** 2 * dm.exp(v[1]),
        lambda v: v[0] ** 2 * np.exp(v[1]),
        30.0,
        "<=",
        [-1, -1],
        B2,
    ),
    (
        "sin(x)*cos(y)",
        lambda v: dm.sin(v[0]) * dm.cos(v[1]),
        lambda v: np.sin(v[0]) * np.cos(v[1]),
        0.5,
        "<=",
        [-1, -1],
        B2,
    ),
    (
        "exp(x)*exp(y)",
        lambda v: dm.exp(v[0]) * dm.exp(v[1]),
        lambda v: np.exp(v[0]) * np.exp(v[1]),
        40.0,
        "<=",
        [-1, -1],
        B2,
    ),
    (
        "log(x)*y",
        lambda v: dm.log(v[0]) * v[1],
        lambda v: np.log(v[0]) * v[1],
        3.0,
        "<=",
        [-1, -1],
        B2,
    ),
    # composite-base powers (the gate-fix family)
    (
        "(x*y)**0.5",
        lambda v: (v[0] * v[1]) ** 0.5,
        lambda v: (v[0] * v[1]) ** 0.5,
        2.0,
        "<=",
        [-1, -1],
        B2,
    ),
    (
        "(x*y)**1.5",
        lambda v: (v[0] * v[1]) ** 1.5,
        lambda v: (v[0] * v[1]) ** 1.5,
        8.0,
        "<=",
        [-1, -1],
        B2,
    ),
    (
        "(x+y)**0.5",
        lambda v: (v[0] + v[1]) ** 0.5,
        lambda v: (v[0] + v[1]) ** 0.5,
        2.0,
        "<=",
        [-1, -1],
        B2,
    ),
    (
        "(x+y)**0.7",
        lambda v: (v[0] + v[1]) ** 0.7,
        lambda v: (v[0] + v[1]) ** 0.7,
        2.0,
        "<=",
        [-1, -1],
        B2,
    ),
    (
        "(x*y)**2",
        lambda v: (v[0] * v[1]) ** 2,
        lambda v: (v[0] * v[1]) ** 2,
        16.0,
        "<=",
        [-1, -1],
        B2,
    ),
    # affine-square as a product factor
    (
        "(x+1)**2*y",
        lambda v: (v[0] + 1) ** 2 * v[1],
        lambda v: (v[0] + 1) ** 2 * v[1],
        20.0,
        "<=",
        [-1, -1],
        B2,
    ),
    (
        "y*(x+1)**2",
        lambda v: v[1] * (v[0] + 1) ** 2,
        lambda v: v[1] * (v[0] + 1) ** 2,
        20.0,
        "<=",
        [-1, -1],
        B2,
    ),
    # transcendental of affine, in a product
    (
        "exp(x+y)*z",
        lambda v: dm.exp(v[0] + v[1]) * v[2],
        lambda v: np.exp(v[0] + v[1]) * v[2],
        60.0,
        "<=",
        [-1, -1, -1],
        B3,
    ),
    (
        "sqrt(x+y)*z",
        lambda v: dm.sqrt(v[0] + v[1]) * v[2],
        lambda v: np.sqrt(v[0] + v[1]) * v[2],
        6.0,
        "<=",
        [-1, -1, -1],
        B3,
    ),
    # trilinear with lifted factors
    (
        "x*y*z",
        lambda v: v[0] * v[1] * v[2],
        lambda v: v[0] * v[1] * v[2],
        12.0,
        "<=",
        [-1, -1, -1],
        B3,
    ),
    (
        "x**2*y*z",
        lambda v: v[0] ** 2 * v[1] * v[2],
        lambda v: v[0] ** 2 * v[1] * v[2],
        20.0,
        "<=",
        [-1, -1, -1],
        B3,
    ),
    (
        "x**2*y**2*z",
        lambda v: v[0] ** 2 * v[1] ** 2 * v[2],
        lambda v: v[0] ** 2 * v[1] ** 2 * v[2],
        30.0,
        "<=",
        [-1, -1, -1],
        B3,
    ),
    (
        "x**0.5*y*z",
        lambda v: v[0] ** 0.5 * v[1] * v[2],
        lambda v: v[0] ** 0.5 * v[1] * v[2],
        10.0,
        "<=",
        [-1, -1, -1],
        B3,
    ),
    (
        "sqrt(x)*y**2*z",
        lambda v: dm.sqrt(v[0]) * v[1] ** 2 * v[2],
        lambda v: np.sqrt(v[0]) * v[1] ** 2 * v[2],
        40.0,
        "<=",
        [-1, -1, -1],
        B3,
    ),
    # repeated-factor forms (the collector-fix family)
    ("x*x*y", lambda v: v[0] * v[0] * v[1], lambda v: v[0] ** 2 * v[1], 8.0, "<=", [-1, -1], B2),
    (
        "x*x*y*y",
        lambda v: v[0] * v[0] * v[1] * v[1],
        lambda v: v[0] ** 2 * v[1] ** 2,
        16.0,
        "<=",
        [-1, -1],
        B2,
    ),
    (
        "x**2*y**2*z**2",
        lambda v: v[0] ** 2 * v[1] ** 2 * v[2] ** 2,
        lambda v: (v[0] * v[1] * v[2]) ** 2,
        60.0,
        "<=",
        [-1, -1, -1],
        B3,
    ),
    # ratios
    (
        "(x*y)/z",
        lambda v: (v[0] * v[1]) / v[2],
        lambda v: (v[0] * v[1]) / v[2],
        4.0,
        "<=",
        [-1, -1, 1],
        B3,
    ),
    (
        "(x*y)/(z*w)",
        lambda v: (v[0] * v[1]) / (v[2] * v[3]),
        lambda v: (v[0] * v[1]) / (v[2] * v[3]),
        2.0,
        "<=",
        [-1, -1, 1, 1],
        B4,
    ),
    # mixed trip
    (
        "x**2*y**0.5*z**2",
        lambda v: v[0] ** 2 * v[1] ** 0.5 * v[2] ** 2,
        lambda v: v[0] ** 2 * np.sqrt(v[1]) * v[2] ** 2,
        40.0,
        "<=",
        [-1, -1, -1],
        B3,
    ),
]


@pytest.mark.parametrize(
    "name,mk,con_np,rhs,sense,coeffs,box", _PATTERNS, ids=[p[0] for p in _PATTERNS]
)
def test_factorable_pattern_kept_and_sound(name, mk, con_np, rhs, sense, coeffs, box):
    """Every catalogued liftable term must stay in the relaxation with a sound bound."""

    def build():
        m = dm.Model("lift_probe")
        vs = [m.continuous(f"x{i}", lb=lo, ub=hi) for i, (lo, hi) in enumerate(box)]
        m.minimize(sum(c * v for c, v in zip(coeffs, vs)))
        e = mk(vs)
        m.subject_to(e <= rhs if sense == "<=" else e >= rhs)
        return m

    dropped, bound, status = _build_and_probe(build)

    assert not dropped, (
        f"[{name}] constraint dropped from the relaxation ('omitting constraint') — "
        "a liftable term was not recognized, which freezes the dual bound"
    )
    assert status == "optimal", f"[{name}] root relaxation LP status {status}"
    assert bound is not None and math.isfinite(bound), f"[{name}] no finite root bound"

    opt = _grid_opt(con_np, rhs, sense, coeffs, box)
    # Soundness: a valid lower bound never exceeds the true optimum. The grid optimum
    # is an upper bound on the true optimum; a small slack absorbs grid coarseness.
    assert bound <= opt + 5e-2, f"[{name}] UNSOUND root bound {bound} > grid optimum {opt}"
