"""#2142 A: regression shapes left by the retired ``DISCOPT_HULL_LIFTED_PERSPECTIVE``.

The flag hulled a nonlinear disjunct row by the absorbed exact perspective: ratio
columns ``t`` with ``v - y t = 0`` and the body ``y * g(t)`` with ``y`` folded into one
factor per monomial (no ``eps``, no ``1/yhat``). It reached the native kernel on
GDPlib ``cstr`` under hull, but its §5 panel was not cert-clean
(``docs/dev/flag-retirement-audit.md``): ``eight_process`` and
``eight_process_logical`` lost their certificates, and hda's root bound was looser in
2/2 reps (the pre-registered kill criterion), also with the class restricted to
polynomial bodies. Flag and implementation were retired together; the
implementation is in this file's history (commit "feat(gdp): #2142 A").

Kept: the two shapes it added -- a non-polynomial body (``exp(x z) == y``) and a
polynomial with a constant term (``x z + 1 == y``) -- under the shipped hull, the
differential battery on both, and a check that the retired variable is inert.
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import pytest
from discopt._relax.gdp_reformulate import _is_linear, reformulate_gdp
from discopt.transformations import create_using, model_fingerprint
from gdp_differential import Shape, run_shape


def _exp_eq():
    """Disjunct 1: ``exp(x z) == y`` (non-polynomial, ``g(0) = 1``); disjunct 2: a box."""
    m = dm.Model("exp_eq")
    x = m.continuous("x", lb=0, ub=1)
    z = m.continuous("z", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=3)
    m.either_or([[dm.exp(x * z) - y == 0], [x + z <= 0.5, y <= 0.5]], name="d")
    m.minimize(-x - z + 0.1 * y)
    return m


def _const_term_eq():
    """``x z + 1 == y``: a polynomial with a constant term (``g(0) = 1``)."""
    m = dm.Model("const_term_eq")
    x = m.continuous("x", lb=0, ub=2)
    z = m.continuous("z", lb=0, ub=2)
    y = m.continuous("y", lb=0, ub=5)
    m.either_or([[x * z + 1 - y == 0], [x <= 0.5, z <= 0.5, y <= 0.5]], name="d")
    m.minimize(-x - z + 0.1 * y)
    return m


SHAPES = [
    Shape(
        "exp_eq",
        _exp_eq,
        lambda v, t: (
            abs(math.exp(v["x"] * v["z"]) - v["y"]) <= t
            or (v["x"] + v["z"] <= 0.5 + t and v["y"] <= 0.5 + t)
        ),
        points=[
            {"x": 1.0, "z": 1.0, "y": math.e},
            {"x": 0.5, "z": 0.5, "y": math.exp(0.25)},
            {"x": 0.2, "z": 0.2, "y": 0.3},  # disjunct 2
            {"x": 1.0, "z": 1.0, "y": 2.0},  # infeasible in both
            {"x": 0.5, "z": 0.5, "y": 2.0},
        ],
    ),
    Shape(
        "const_term_eq",
        _const_term_eq,
        lambda v, t: (
            abs(v["x"] * v["z"] + 1 - v["y"]) <= t
            or (v["x"] <= 0.5 + t and v["z"] <= 0.5 + t and v["y"] <= 0.5 + t)
        ),
        points=[
            {"x": 2.0, "z": 2.0, "y": 5.0},
            {"x": 1.0, "z": 1.0, "y": 2.0},
            {"x": 0.4, "z": 0.3, "y": 0.2},  # disjunct 2
            {"x": 1.0, "z": 1.0, "y": 3.0},  # infeasible in both
            {"x": 2.0, "z": 2.0, "y": 4.0},
        ],
    ),
]


_EXP_UNDECIDABLE = pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason="#2142: under the eps-perspective the off disjunct evaluates exp(v / eps), "
    "which overflows at every probed assignment, so the battery cannot decide the "
    "point (a harness limit, not a soundness verdict; the retired absorbed form "
    "passed this shape)",
)


@pytest.mark.parametrize(
    "shape",
    [pytest.param(s, marks=_EXP_UNDECIDABLE) if s.name == "exp_eq" else s for s in SHAPES],
    ids=lambda s: s.name,
)
def test_differential_battery_under_hull(shape):
    rep = run_shape(shape, "hull")
    assert rep.refused is None, rep.refused
    assert rep.feasible_checked > 0 and rep.infeasible_checked > 0, rep
    rep.assert_sound()


@pytest.mark.parametrize("build", [_exp_eq, _const_term_eq])
def test_rows_keep_the_eps_perspective(build, monkeypatch):
    # Pinned to the eps-perspective lowering: since #2142 graduated
    # DISCOPT_HULL_EXACT_PERSPECTIVE, a nonconvex polynomial row is homogenized by
    # default; this test is about the eps form, so it opts out.
    monkeypatch.setenv("DISCOPT_HULL_EXACT_PERSPECTIVE", "0")
    rows = [c for c in reformulate_gdp(build(), "hull")._constraints if not _is_linear(c.body)]
    assert rows and all("/" in str(c.body) for c in rows), rows


@pytest.mark.parametrize("build", [_exp_eq, _const_term_eq])
def test_retired_flag_is_inert(build, monkeypatch):
    """Setting the retired variable changes nothing (no dead flag left behind)."""
    monkeypatch.delenv("DISCOPT_HULL_LIFTED_PERSPECTIVE", raising=False)
    off = model_fingerprint(create_using("gdp.hull", build()))
    monkeypatch.setenv("DISCOPT_HULL_LIFTED_PERSPECTIVE", "1")
    on = model_fingerprint(create_using("gdp.hull", build()))
    assert off == on
