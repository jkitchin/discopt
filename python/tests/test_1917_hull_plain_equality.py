"""``DISCOPT_HULL_PLAIN_EQUALITY`` (#1917): the hull of a nonlinear disjunct
*equality* with ``g(0) = 0`` as the plain disaggregated row ``g(v_k) = 0`` instead
of the eps-perspective.

Bound-changing, so default OFF (CLAUDE.md §5). What is pinned here:

* flag OFF (and unset): every nonlinear row keeps the eps-perspective (a ``/``);
* flag ON: a ``g(0) = 0`` equality is emitted with no division, and every row
  outside that class (an inequality, ``g(0) != 0``) keeps the perspective;
* the differential battery (no feasible point cut, no infeasible point admitted)
  on every shape, plus two nonlinear-equality shapes, under hull with the flag ON;
* solves agree with big-M;
* GDPlib ``cstr``: OFF, the 1/yhat products make the probe-box and real-box
  relaxations differ in shape and the native spatial kernel declines the model
  (the Python tree then reaches bound 0.094 in 300 s); ON, the kernel accepts it.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest
from discopt._relax.gdp_reformulate import _is_linear, reformulate_gdp
from gdp_differential import Shape, run_shape
from test_gdp_differential_battery import SHAPES

FLAG = "DISCOPT_HULL_PLAIN_EQUALITY"


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
    """``x*z == 2``: ``g(0) = -2 != 0``, so the row is out of the class."""
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


def test_flag_off_keeps_the_perspective(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    rows = _nonlinear_rows(reformulate_gdp(_bilinear_eq(), "hull"))
    assert len(rows) == 2 and all("/" in str(c.body) for c in rows), rows


def test_flag_on_emits_the_plain_row(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    rows = _nonlinear_rows(reformulate_gdp(_bilinear_eq(), "hull"))
    # the equality's _le/_ge pair, neither with a division
    assert len(rows) == 2 and not any("/" in str(c.body) for c in rows), rows


def test_out_of_class_rows_keep_the_perspective(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    rows = _nonlinear_rows(reformulate_gdp(_bilinear_eq_offset(), "hull"))
    assert rows and all("/" in str(c.body) for c in rows), rows  # g(0) != 0

    m = dm.Model("ineq")
    x = m.continuous("x", lb=0, ub=4)
    z = m.continuous("z", lb=0, ub=4)
    m.either_or([[x * z <= 2], [x >= 3]], name="d")
    m.minimize(-x - z)
    rows = _nonlinear_rows(reformulate_gdp(m, "hull"))
    assert rows and all("/" in str(c.body) for c in rows), rows  # not an equality


@pytest.mark.parametrize("shape", [*SHAPES, *EQ_SHAPES], ids=lambda s: s.name)
def test_differential_battery_under_hull_with_the_flag(monkeypatch, shape):
    monkeypatch.setenv(FLAG, "1")
    rep = run_shape(shape, "hull")
    if rep.refused is not None:
        assert "hull" in shape.refuses
        return
    assert rep.feasible_checked > 0 and rep.infeasible_checked > 0, rep
    rep.assert_sound()


@pytest.mark.parametrize("flag", ["0", "1"])
def test_solves_agree_with_big_m(monkeypatch, flag):
    monkeypatch.setenv(FLAG, flag)
    r = _bilinear_eq().solve(gdp_method="hull", time_limit=60)
    ref = _bilinear_eq().solve(gdp_method="big-m", time_limit=60)
    assert r.status == "optimal" and ref.status == "optimal", (r.status, ref.status)
    # x = z = 4, y = 16: -8 + 1.6
    assert ref.objective == pytest.approx(-6.4, abs=1e-5)
    assert r.objective == pytest.approx(ref.objective, abs=1e-5)


def test_cstr_hull_reaches_the_native_kernel(monkeypatch):
    """The #1917 entry: GDPlib ``cstr`` under hull is declined by the native spatial
    kernel's spec producer with the eps-perspective rows, and accepted with the plain
    rows. Fails before the flag existed (the producer returned ``None``)."""
    pytest.importorskip("pyomo")
    cstr = pytest.importorskip("gdplib.cstr")
    import discopt._relax.spatial_producer as sp
    import discopt.solver as solver
    from discopt.pyomo.gdp import convert_pyomo_gdp

    monkeypatch.setenv(FLAG, "1")
    orig = sp.build_spatial_kernel_spec
    specs = []

    def spy(*a, **k):
        out = orig(*a, **k)
        specs.append(out)
        return out

    monkeypatch.setattr(sp, "build_spatial_kernel_spec", spy)
    for name in dir(solver):
        if getattr(solver, name, None) is orig:
            monkeypatch.setattr(solver, name, spy)
    model = convert_pyomo_gdp(cstr.build_model()).model
    model.solve(gdp_method="hull", time_limit=5)
    assert specs, "the spec producer was never called"
    assert specs[0] is not None, "the native kernel declined cstr under hull"
