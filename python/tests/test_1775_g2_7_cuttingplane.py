"""#1775 G2.7: ``gdp.cuttingplane`` (Sawaya & Grossmann 2005; Pyomo's transformation).

Before: no such transformation.

After (NC: a new, user-selected transformation; no route changes): the big-M
lowering plus cuts ``g.x >= v`` that separate the big-M relaxation's point from
the hull relaxation, where ``v`` is a certified lower bound on ``min g.x`` over the
hull relaxation -- so every cut holds at every feasible point, however inexact the
separation. Disjuncts must be linear or proven convex; anything else is refused.
"""

from __future__ import annotations

import itertools
import warnings

import discopt.modeling as dm
import discopt.transformations as dt
import numpy as np
import pytest
from discopt._relax.gdp_reformulate import reformulate_gdp
from discopt.gdp.cuttingplane import cuttingplane
from discopt.validation.feasibility import verify_point

pytestmark = pytest.mark.smoke


def _squares(obj):
    """[x, y <= 2] | [x, y >= 8, x <= 9], and [x + y <= 3] | [x + y >= 15] on [0, 10]^2."""
    m = dm.Model("squares")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.either_or([[x <= 2, y <= 2], [x >= 8, y >= 8, x <= 9]], name="a")
    m.either_or([[x + y <= 3], [x + y >= 15]], name="b")
    m.minimize(obj(x, y))
    return m


def _circles(obj):
    """Two convex disjuncts: a disc of radius 1 at (2, 2), or one at (7, 6)."""
    m = dm.Model("circles")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.either_or([[(x - 2) ** 2 + (y - 2) ** 2 <= 1], [(x - 7) ** 2 + (y - 6) ** 2 <= 1]], name="c")
    m.minimize(obj(x, y))
    return m


# Optima by hand. squares: A = [0,2]^2 needs x + y <= 3 (B's x + y >= 15 is out of
# reach there); C = [8,9]x[8,10] needs x + y >= 15 (x + y <= 3 is out of reach).
#   min x - 2y: A -> (0, 2) = -4; C -> (8, 10) = -12            -> -12
#   min -x - y: A -> 3; C -> (9, 10) = -19                       -> -19
# circles: min x + y -> 4 - sqrt(2); min -x - y -> -(13 + sqrt(2)).
CASES = [
    (_squares, lambda x, y: x - 2 * y, -12.0),
    (_squares, lambda x, y: -x - y, -19.0),
    (_circles, lambda x, y: x + y, 4.0 - np.sqrt(2.0)),
    (_circles, lambda x, y: -x - y, -(13.0 + np.sqrt(2.0))),
]


def _solve(m):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return m.solve(time_limit=60)


@pytest.mark.parametrize("norm", [2, "inf"])
@pytest.mark.parametrize("build, obj, truth", CASES)
def test_cuts_keep_the_optimum_and_never_loosen_the_relaxation(build, obj, truth, norm):
    m = build(obj)
    out, rep = cuttingplane(m, norm=norm, max_cuts=10)
    assert rep.relaxation_before is not None and rep.relaxation_after is not None
    assert rep.relaxation_after >= rep.relaxation_before - 1e-6  # min: never looser
    r = _solve(out)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(truth, abs=1e-5)


def test_the_cuts_tighten_the_big_m_relaxation():
    out, rep = cuttingplane(_squares(lambda x, y: x - 2 * y), max_cuts=10)
    assert rep.cuts >= 1
    assert rep.relaxation_before == pytest.approx(-12.08, abs=1e-4)
    assert rep.relaxation_after == pytest.approx(-12.0, abs=1e-4)  # the hull bound
    assert sum(c.name is not None and c.name.startswith("_gdp_cp_cut_") for c in out._constraints)


@pytest.mark.parametrize("build", [_squares, _circles])
def test_every_cut_holds_at_every_feasible_point(build):
    """A grid over the box; every point the declared model accepts satisfies every
    cut. The probe must have checked feasible points and cuts."""
    m = build(lambda x, y: x - 2 * y)
    _, rep = cuttingplane(m, max_cuts=10)
    assert rep.cuts >= 1
    checks = 0
    for xv, yv in itertools.product(np.linspace(0, 10, 41), repeat=2):
        if not verify_point(m, np.array([xv, yv])).ok:
            continue
        for g, v in rep.cut_rows:
            assert g["x"] * xv + g["y"] * yv >= v - 1e-7, (xv, yv, g, v)
            checks += 1
    assert checks >= 20


def test_registered_transformation():
    m = _squares(lambda x, y: -x - y)
    t = dt.create_using("gdp.cuttingplane", m, max_cuts=5)
    assert t is not m and len(m._constraints) == 2  # the source is untouched
    assert _solve(t).objective == pytest.approx(-19.0, abs=1e-5)


def test_max_cuts_zero_is_the_big_m_lowering():
    m = _squares(lambda x, y: x - 2 * y)
    out, rep = cuttingplane(m, max_cuts=0)
    assert rep.cuts == 0 and rep.stopped == "max_cuts=0 reached"
    assert [str(c) for c in out._constraints] == [
        str(c) for c in reformulate_gdp(m, "big-m")._constraints
    ]


def test_a_model_without_gdp_is_returned_unchanged():
    m = dm.Model("plain")
    x = m.continuous("x", lb=0, ub=1)
    m.minimize(x)
    out, rep = cuttingplane(m)
    assert out is m and rep.cuts == 0


@pytest.mark.parametrize(
    "row",
    [
        lambda x, y: x * y <= 1,  # bilinear: not convex
        lambda x, y: x**2 + y**2 >= 4,  # convex body, >=: not a convex set
        lambda x, y: x**2 == y,  # nonlinear equality
    ],
)
def test_a_nonconvex_disjunct_is_refused(row):
    m = dm.Model("nc")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.either_or([[row(x, y)], [x >= 8]])
    m.minimize(x + y)
    with pytest.raises(ValueError, match="not linear or proven convex"):
        cuttingplane(m)


def test_or_semantics_is_refused_by_the_hull_relaxation():
    m = dm.Model("or")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x <= 2], [x >= 5]], semantics="or")
    m.minimize(x)
    with pytest.raises(NotImplementedError, match="OR"):
        cuttingplane(m)


def test_bad_options_are_refused():
    m = _squares(lambda x, y: x)
    with pytest.raises(ValueError, match="norm"):
        cuttingplane(m, norm=1)
    with pytest.raises(ValueError, match="max_cuts"):
        cuttingplane(m, max_cuts=-1)
