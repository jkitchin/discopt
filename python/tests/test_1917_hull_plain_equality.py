"""Regression cases left by the retired ``DISCOPT_HULL_PLAIN_EQUALITY`` (#1917).

The flag hulled a nonlinear disjunct *equality* with ``g(0) = 0`` as the plain
disaggregated row ``g(v_k) = 0`` instead of the eps-perspective. It was retired after
its CLAUDE.md §5 panel (``docs/dev/flag-retirement-audit.md``):

* sound everywhere;
* large wins on cstr, gdp_col, mod_hens, modprodnet and eight_process;
* deterministically looser root bound on hda (117483 vs 121141);
* incumbent lost on methanol in 3/3 reps.

At fractional ``y`` the plain row and the perspective are different relaxations,
and neither dominates. What stays pinned here:

* every nonlinear disjunct row, equalities included, gets the eps-perspective (a ``/``);
* the differential battery (no feasible point cut, no infeasible point admitted) on
  two nonlinear-equality shapes under hull;
* hull solves of those shapes agree with big-M;
* GDPlib ``cstr`` under hull reaches the native spatial kernel on the default route
  (#2142: the 1/yhat products used to make the probe-box and real-box relaxations
  differ in shape; the producer now keeps row placeholders and the kernel has an
  ``AffinePow`` term). The kernel solves the same relaxation, so cstr-hull's bound
  does not move. (The retired ``DISCOPT_HULL_LIFTED_PERSPECTIVE`` reached the kernel
  here by changing the rows and failed its §5 panel; see the audit.)
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest
from discopt._relax.gdp_reformulate import _is_linear, reformulate_gdp
from gdp_differential import Shape, run_shape


def _nonlinear_rows(low):
    return [c for c in low._constraints if not _is_linear(c.body)]


def _bilinear_eq():
    """Disjunct 1: ``x*z == y, y >= 1`` (``g(0) = 0``); disjunct 2: a small box."""
    m = dm.Model("bilinear_eq")
    x = m.continuous("x", lb=0, ub=4)
    z = m.continuous("z", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=16)
    m.either_or([[x * z - y == 0, y >= 1], [x <= 1, z <= 1, y <= 0.5]], name="d")
    m.minimize(-x - z + 0.1 * y)
    return m


def _bilinear_eq_offset():
    """``x*z == 2``: ``g(0) = -2 != 0``."""
    m = dm.Model("bilinear_eq_offset")
    x = m.continuous("x", lb=0, ub=4)
    z = m.continuous("z", lb=0, ub=4)
    m.either_or([[x * z == 2], [x + z <= 1]], name="d")
    m.minimize(-x - z)
    return m


def _inside_1(v, t):
    return abs(v["x"] * v["z"] - v["y"]) <= t and v["y"] >= 1 - t


def _inside_2(v, t):
    return v["x"] <= 1 + t and v["z"] <= 1 + t and v["y"] <= 0.5 + t


EQ_SHAPES = [
    Shape(
        "bilinear_eq",
        _bilinear_eq,
        lambda v, t: _inside_1(v, t) or _inside_2(v, t),
        points=[
            {"x": 2.0, "z": 1.5, "y": 3.0},  # on the surface, disjunct 1
            {"x": 4.0, "z": 4.0, "y": 16.0},
            {"x": 0.5, "z": 2.0, "y": 1.0},
            {"x": 2.0, "z": 1.5, "y": 3.5},  # off the surface: infeasible
            {"x": 0.5, "z": 0.5, "y": 0.25},  # disjunct 2 (and off the surface)
            {"x": 0.0, "z": 0.0, "y": 0.0},
            {"x": 1.0, "z": 1.0, "y": 0.75},  # infeasible in both
        ],
    ),
    Shape(
        "bilinear_eq_offset",
        _bilinear_eq_offset,
        lambda v, t: abs(v["x"] * v["z"] - 2) <= t or v["x"] + v["z"] <= 1 + t,
        points=[
            {"x": 1.0, "z": 2.0},
            {"x": 4.0, "z": 0.5},
            {"x": 1.0, "z": 1.0},
            {"x": 0.25, "z": 0.25},
        ],
    ),
]


def test_nonlinear_equalities_keep_the_perspective():
    for build in (_bilinear_eq, _bilinear_eq_offset):
        rows = _nonlinear_rows(reformulate_gdp(build(), "hull"))
        assert rows and all("/" in str(c.body) for c in rows), (build.__name__, rows)


def test_retired_flag_is_inert(monkeypatch):
    """Setting the retired env var changes nothing (no dead flag left behind)."""
    monkeypatch.setenv("DISCOPT_HULL_PLAIN_EQUALITY", "1")
    rows = _nonlinear_rows(reformulate_gdp(_bilinear_eq(), "hull"))
    assert len(rows) == 2 and all("/" in str(c.body) for c in rows), rows


@pytest.mark.parametrize("shape", EQ_SHAPES, ids=lambda s: s.name)
def test_differential_battery_nonlinear_equalities_under_hull(shape):
    rep = run_shape(shape, "hull")
    assert rep.refused is None, rep.refused
    assert rep.feasible_checked > 0 and rep.infeasible_checked > 0, rep
    rep.assert_sound()


def test_hull_solve_agrees_with_big_m():
    r = _bilinear_eq().solve(gdp_method="hull", time_limit=60)
    ref = _bilinear_eq().solve(gdp_method="big-m", time_limit=60)
    assert r.status == "optimal" and ref.status == "optimal", (r.status, ref.status)
    # x = z = 4, y = 16: -8 + 1.6
    assert ref.objective == pytest.approx(-6.4, abs=1e-5)
    assert r.objective == pytest.approx(ref.objective, abs=1e-5)


class _ReachedProducer(Exception):
    """Raised by the spy at the first producer call: the route is decided there."""


def _cstr_hull_first_kernel_spec(monkeypatch):
    """Solve GDPlib cstr under hull up to the solver's FIRST native-kernel spec build
    and return that spec (``None`` = declined). The solve stops there: what happens
    after is the kernel's search (or the Python tree's), not the routing under test.

    The time limit is only a backstop. It used to be 5 s, which on this model ends
    the solve before the producer is ever asked (measured 2026-11-04: no call at
    5 s), so the strict xfail this replaced passed on the missing call as well as
    on the decline.
    """
    pytest.importorskip("pyomo")
    cstr = pytest.importorskip("gdplib.cstr")
    import discopt._relax.spatial_producer as sp
    import discopt.solver as solver
    from discopt.pyomo.gdp import convert_pyomo_gdp

    orig = sp.build_spatial_kernel_spec
    specs = []

    def spy(*a, **k):
        specs.append(orig(*a, **k))
        raise _ReachedProducer

    monkeypatch.setattr(sp, "build_spatial_kernel_spec", spy)
    for name in dir(solver):
        if getattr(solver, name, None) is orig:
            monkeypatch.setattr(solver, name, spy)
    model = convert_pyomo_gdp(cstr.build_model()).model
    with pytest.raises(_ReachedProducer):
        model.solve(gdp_method="hull", time_limit=300)
    assert len(specs) == 1
    return specs[0]


@pytest.mark.slow  # ~13 s locally (presolve before the producer); ~3x that on CI
def test_cstr_hull_reaches_the_native_kernel(monkeypatch):
    """#2142: on the real box the eps-perspective's ``yc**-1`` chains reach ``_BIG``
    and ``_emit_mccormick`` drops 44 rows the probe box keeps; the producer now
    holds placeholders for them and hands the ``yc**-1`` atoms to the kernel as
    ``AffinePow`` terms, so the spec is accepted."""
    spec = _cstr_hull_first_kernel_spec(monkeypatch)
    assert spec is not None, "the native kernel declined cstr under hull"
    assert int((spec["term_kind"] == 4).sum()) == 20
