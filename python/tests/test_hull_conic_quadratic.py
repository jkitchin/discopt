"""``DISCOPT_HULL_CONIC_QUADRATIC``: the hull of a separable convex quadratic row as a
smoothed second-order cone instead of the eps-perspective.

Bound-changing, so default OFF (CLAUDE.md §5). What is pinned here:

* flag OFF: no cone row anywhere (today's lowering);
* flag ON: a ball/ellipse row becomes ``sqrt(S + d^2) <= sqrt(R + d^2) y + d (1 - y)``
  with no division, and every row outside the recognized class keeps the
  perspective;
* the emitted expression, evaluated as built: exact at ``y = 1`` and ``y = 0``, and
  containing the exact perspective ``S <= R y^2`` at fractional ``y`` (a relaxation
  of the hull, never tighter);
* the differential battery (no feasible point cut, no infeasible point admitted)
  on every shape under hull with the flag ON;
* solves agree with big-M, nested and flat, on the certified convex route.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
import test_1775_g2_1_hull_nesting as nesting
from discopt._relax.dag_compiler import compile_expression
from discopt._relax.gdp_reformulate import _collect_variables, _is_linear, reformulate_gdp
from gdp_differential import run_shape
from test_gdp_differential_battery import SHAPES

FLAG = "DISCOPT_HULL_CONIC_QUADRATIC"


def _nonlinear_rows(low):
    return [c for c in low._constraints if not _is_linear(c.body)]


def _is_cone(con) -> bool:
    body = str(con.body)
    return "sqrt(" in body and "/" not in body


def _ellipse(sense="<="):
    m = dm.Model("ellipse")
    x = m.continuous("x", lb=-4, ub=4)
    y = m.continuous("y", lb=-4, ub=4)
    g = 2 * (x - 1) ** 2 + 0.5 * (y + 1) ** 2
    row = g <= 2 if sense == "<=" else -g >= -2
    m.either_or([[row], [x >= 3]], name="d")
    m.minimize(x + y)
    return m


def test_flag_off_emits_no_cone(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    m, x, y = nesting._circles()
    m.minimize(x + y)
    rows = _nonlinear_rows(reformulate_gdp(m, "hull"))
    assert rows and not any(_is_cone(c) for c in rows)
    assert all("/" in str(c.body) for c in rows)  # today's eps-perspective


@pytest.mark.parametrize("sense", ["<=", ">="])
def test_flag_on_ellipse_is_a_cone(monkeypatch, sense):
    monkeypatch.setenv(FLAG, "1")
    rows = _nonlinear_rows(reformulate_gdp(_ellipse(sense), "hull"))
    assert len(rows) == 1 and _is_cone(rows[0]), rows


def test_a_sumover_circle_is_a_cone(monkeypatch):
    """The Pyomo converter writes sums as ``Σ[...]``: ``Σ[x, -2.5, -15]**2 + ...``.
    ``distribute_products`` expands ``(a + b)**2`` but not ``(Σ[a, b])**2``, so the
    extractor abstained on every such row (all 24 disjunct rows of
    ``pyomo.constrained_layout_clay0203``) until the cone path desugared ``Σ``."""
    from discopt.modeling.core import Constant, SumOverExpression

    monkeypatch.setenv(FLAG, "1")
    m = dm.Model("sumover_circle")
    x = m.continuous("x", lb=0, ub=40)
    y = m.continuous("y", lb=0, ub=40)
    circle = (
        SumOverExpression([x, Constant(-2.5), Constant(-15.0)]) ** Constant(2.0)
        + SumOverExpression([y, Constant(3.0), Constant(-10.0)]) ** Constant(2.0)
        <= 36.0
    )
    m.either_or([[circle], [x >= 30]], name="d")
    m.minimize(x + y)
    rows = _nonlinear_rows(reformulate_gdp(m, "hull"))
    assert len(rows) == 1 and _is_cone(rows[0]), rows


def _abstaining_models():
    def build(row_fn):
        m = dm.Model("abstain")
        x = m.continuous("x", lb=-4, ub=4)
        y = m.continuous("y", lb=-4, ub=4)
        p = m.parameter("p", value=1.0)
        m.either_or([[row_fn(x, y, p)], [x >= 3]], name="d")
        m.minimize(x + y)
        return m

    return {
        "equality": build(lambda x, y, p: x**2 + y**2 == 1),
        "cross_term": build(lambda x, y, p: x**2 + x * y + y**2 <= 1),
        "linear_only_variable": build(lambda x, y, p: (x - 1) ** 2 + y <= 1),
        "parameter": build(lambda x, y, p: x**2 + y**2 <= p),
        "empty_ball": build(lambda x, y, p: x**2 + y**2 <= -1),
        "concave": build(lambda x, y, p: -(x**2) - y**2 <= -1),
        "not_quadratic": build(lambda x, y, p: dm.exp(x) + y**2 <= 2),
    }


@pytest.mark.parametrize("name", list(_abstaining_models()))
def test_rows_outside_the_class_keep_the_perspective(monkeypatch, name):
    monkeypatch.setenv(FLAG, "1")
    rows = _nonlinear_rows(reformulate_gdp(_abstaining_models()[name], "hull"))
    assert rows and not any(_is_cone(c) for c in rows), (name, rows)


def test_the_cone_row_is_exact_at_integer_points_and_relaxes_the_perspective(monkeypatch):
    """Evaluate the emitted row itself. ``g = 2 (x-1)^2 + 0.5 (y+1)^2 - 2``."""
    monkeypatch.setenv(FLAG, "1")
    low = reformulate_gdp(_ellipse(), "hull")
    (row,) = _nonlinear_rows(low)
    names = set(_collect_variables(row.body))
    (sel,) = [n for n in names if n.startswith("_gdp_aux_")]
    vx, vy = sorted(n for n in names if n not in (sel,))  # ..._v_x_0, ..._v_y_0
    assert vx.endswith("_v_x_0") and vy.endswith("_v_y_0")
    fn = compile_expression(row.body, low)
    index = {v.name: low._flat_var_offset(v) for v in low._variables}
    n_flat = sum(v.size for v in low._variables)

    def value(px, py, s):
        z = np.zeros(n_flat)
        z[index[vx]], z[index[vy]], z[index[sel]] = px, py, s
        return float(fn(z))

    def g(px, py):
        return 2 * (px - 1) ** 2 + 0.5 * (py + 1) ** 2 - 2

    rng = np.random.default_rng(1775)
    checks = 0
    assert value(0.0, 0.0, 0.0) == 0.0  # the off disjunct, exactly (the #1043 class)
    checks += 1
    for _ in range(400):
        px, py = rng.uniform(-4, 4, size=2)
        gv = g(px, py)
        if gv <= 0:
            # y = 1: the disjunct's own feasible points are kept.
            assert value(px, py, 1.0) <= 1e-9, (px, py)
            # 0 < y < 1: the perspective point (y x, y) is kept -- a relaxation.
            s = rng.uniform(0.01, 0.99)
            assert value(s * px, s * py, s) <= 1e-9, (px, py, s)
            checks += 2
        elif gv >= 1e-3:
            # y = 1: an infeasible point is cut, the row is exact there.
            assert value(px, py, 1.0) > 0.0, (px, py)
            checks += 1
    assert checks > 300, checks


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: s.name)
def test_differential_battery_under_hull_with_the_flag(monkeypatch, shape):
    monkeypatch.setenv(FLAG, "1")
    rep = run_shape(shape, "hull")
    if rep.refused is not None:
        assert "hull" in shape.refuses
        return
    assert rep.feasible_checked > 0 and rep.infeasible_checked > 0, rep
    rep.assert_sound()


def test_the_battery_actually_saw_a_cone(monkeypatch):
    """The battery run above would pass vacuously if no shape produced a cone."""
    monkeypatch.setenv(FLAG, "1")
    shape = next(s for s in SHAPES if s.name == "separable_convex")
    rows = _nonlinear_rows(reformulate_gdp(shape.build(), "hull"))
    assert len(rows) == 2 and all(_is_cone(c) for c in rows), rows


def test_nested_cone_rows_are_passed_up_unchanged(monkeypatch):
    """The inner circles' cones mention only the child's copies and selector."""
    monkeypatch.setenv(FLAG, "1")
    m, x, y = nesting._circles()
    m.minimize(x + y)
    rows = _nonlinear_rows(reformulate_gdp(m, "hull"))
    assert len(rows) == 3 and all(_is_cone(c) for c in rows), rows
    inner = [c for c in rows if any("_inner_v_" in n for n in _collect_variables(c.body))]
    assert len(inner) == 2
    for c in inner:
        outer_sel = [n for n in _collect_variables(c.body) if n.startswith("_gdp_aux_hull_outer_")]
        assert outer_sel and all("_inner_" in n for n in outer_sel), outer_sel


@pytest.mark.parametrize("builder", ["nested", "flat"])
def test_solves_agree_with_big_m_on_the_convex_route(monkeypatch, builder):
    monkeypatch.setenv(FLAG, "1")

    def build():
        if builder == "nested":
            return nesting._circles()
        m = dm.Model("flat")
        x = m.continuous("x", lb=0, ub=4)
        y = m.continuous("y", lb=0, ub=4)
        m.either_or(
            [
                [y <= 2, (x - 1) ** 2 + (y - 1) ** 2 <= 0.25],
                [y <= 2, (x - 3) ** 2 + (y - 1) ** 2 <= 0.25],
                [(x - 2) ** 2 + (y - 3.5) ** 2 <= 0.25],
            ],
            name="o",
        )
        return m, x, y

    m, x, y = build()
    m.minimize(x + y)
    r = m.solve(gdp_method="hull", time_limit=60)
    m_b, x_b, y_b = build()
    m_b.minimize(x_b + y_b)
    ref = m_b.solve(gdp_method="big-m", time_limit=60)
    assert r.status == "optimal" and ref.status == "optimal"
    assert r.objective == pytest.approx(ref.objective, abs=1e-5)
    assert "convex" in (r.algorithm_route or ""), r.algorithm_route
