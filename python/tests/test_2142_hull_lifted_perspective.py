"""#2142 A: the absorbed exact perspective for nonlinear hull rows
(``DISCOPT_HULL_LIFTED_PERSPECTIVE``, default OFF).

With the flag on, every nonlinear disjunct row that has no Parameter and a finite
enclosure on its disjunct box gets ratio columns ``t`` (``v - y t = 0``, the eps-free
lifted perspective's links) and the body ``_absorb_perspective(g)``, equal to
``y * g(t)`` with ``y`` folded into one factor per monomial. Pinned here:

* flag OFF: the eps-perspective (a ``/``) on every nonlinear row, as before;
* flag ON: no ``/`` and no ``eps``, and the ratio links are emitted;
* the differential battery (no feasible point cut, no infeasible point admitted)
  under hull with the flag ON, on the polynomial equality shapes of #1917 plus a
  non-polynomial body (``exp(x z)``) and one with a constant term;
* flag ON hull solve agrees with big-M;
* the absorber is an identity: ``absorbed(t, v = y t, y) == y * g(t)`` on random
  polynomial (and partly non-polynomial) bodies at random points.
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.convexity.interval import Interval
from discopt._relax.convexity.interval_eval import evaluate_interval
from discopt._relax.gdp_reformulate import _absorb_perspective, _is_linear, reformulate_gdp
from discopt.modeling.core import Constant
from gdp_differential import Shape, run_shape
from test_1917_hull_plain_equality import EQ_SHAPES, _bilinear_eq

FLAG = "DISCOPT_HULL_LIFTED_PERSPECTIVE"


def _nonlinear_rows(low):
    return [c for c in low._constraints if not _is_linear(c.body)]


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


EXTRA_SHAPES = [
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


@pytest.mark.parametrize("build", [_bilinear_eq, _exp_eq, _const_term_eq])
def test_flag_off_keeps_the_eps_perspective(build, monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    rows = _nonlinear_rows(reformulate_gdp(build(), "hull"))
    assert rows and all("/" in str(c.body) for c in rows), rows


@pytest.mark.parametrize("build", [_bilinear_eq, _exp_eq, _const_term_eq])
def test_flag_on_emits_the_absorbed_perspective(build, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    low = reformulate_gdp(build(), "hull")
    rows = _nonlinear_rows(low)
    links = [c for c in low._constraints if c.name and c.name.startswith("_hull_ratio0_")]
    assert links, "the ratio links v - y t = 0 must be emitted"
    hull_rows = [c for c in rows if c not in links]
    assert hull_rows and all("/" not in str(c.body) for c in hull_rows), hull_rows


@pytest.mark.parametrize("shape", [*EQ_SHAPES, *EXTRA_SHAPES], ids=lambda s: s.name)
def test_differential_battery_flag_on(shape, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    rep = run_shape(shape, "hull")
    assert rep.refused is None, rep.refused
    assert rep.feasible_checked > 0 and rep.infeasible_checked > 0, rep
    rep.assert_sound()


def test_flag_on_hull_solve_agrees_with_big_m(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    r = _bilinear_eq().solve(gdp_method="hull", time_limit=60)
    ref = _bilinear_eq().solve(gdp_method="big-m", time_limit=60)
    assert r.status == "optimal" and ref.status == "optimal", (r.status, ref.status)
    assert ref.objective == pytest.approx(-6.4, abs=1e-5)
    assert r.objective == pytest.approx(ref.objective, abs=1e-5)


def _random_body(rng, xs):
    """A random sum of monomials ``c * prod x_i^p`` over ``xs`` (degree <= 3), with a
    constant term, a negation, a constant quotient and, sometimes, an ``exp`` node."""
    body = Constant(float(rng.uniform(-3, 3)))
    for _ in range(int(rng.integers(1, 5))):
        k = int(rng.integers(1, 4))
        idx = rng.choice(len(xs), size=k)
        mono = xs[idx[0]]
        for i in idx[1:]:
            mono = mono * xs[i]
        if rng.random() < 0.3:
            mono = xs[int(rng.integers(len(xs)))] ** int(rng.integers(2, 4))
        c = float(rng.uniform(-5, 5))
        term = c * mono
        if rng.random() < 0.2:
            term = -term
        if rng.random() < 0.2:
            term = term / 2.5
        if rng.random() < 0.15:
            term = term + dm.exp(xs[int(rng.integers(len(xs)))])
        body = body + term
    return body


def _point_value(expr, box):
    iv = evaluate_interval(expr, None, box)
    lo, hi = float(np.asarray(iv.lo)), float(np.asarray(iv.hi))
    return 0.5 * (lo + hi), hi - lo


def test_absorber_is_the_perspective_identity():
    rng = np.random.default_rng(2142)
    m = dm.Model("absorb")
    xs = [m.continuous(f"x{i}", lb=-2, ub=2) for i in range(3)]
    ts = {x.name: m.continuous(f"t_{x.name}", lb=-2, ub=2) for x in xs}
    vs = {x.name: m.continuous(f"v_{x.name}", lb=-2, ub=2) for x in xs}
    y = m.continuous("y", lb=0, ub=1)
    from discopt._relax.gdp_reformulate import _substitute_vars

    executed = 0
    for _ in range(50):
        g = _random_body(rng, xs)
        absorbed = _absorb_perspective(g, y, dict(ts), dict(vs))
        reference = y * _substitute_vars(g, dict(ts))
        for _ in range(4):
            yv = float(rng.choice([0.0, 1.0, rng.uniform(0, 1)]))
            box = {y: Interval(np.float64(yv), np.float64(yv))}
            for x in xs:
                tv = float(rng.uniform(-2, 2))
                box[ts[x.name]] = Interval(np.float64(tv), np.float64(tv))
                box[vs[x.name]] = Interval(np.float64(yv * tv), np.float64(yv * tv))
            a, wa = _point_value(absorbed, box)
            r, wr = _point_value(reference, box)
            assert np.isfinite(a) and np.isfinite(r)
            tol = 1e-12 * (1.0 + abs(r)) + 4 * (wa + wr)
            assert abs(a - r) <= tol, (str(g), yv, a, r)
            executed += 1
    print(f"absorber identity: executed assertions: {executed}")
    assert executed == 200
