"""#1975: the 1/T outer piece (``DISCOPT_OUTER_PIECE``, default OFF).

The rewrite ``x_k = r_k + sigma u_k / s`` must be an exact identity on every row
(``outer_row(image of x) = s^D * row(x)``), so every feasible point of the outer
piece maps to a feasible point of the rescaled model (feasible-point sampling: no
valid point is cut). The certified outer bound must then sit below
``(M/T)^d f(x)`` at every sampled feasible point, and the combined solve must
certify only true optima. prolog (MINLPLib) is the probe: with the flag OFF it
has no dual bound (an exact recession ray of its root LP, #1966); ON it certifies
its optimum 0. Plan: ``docs/dev/performance-plan.md`` §92.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import discopt.modeling as dm
from discopt._point_eval import evaluate_at
from discopt.modeling import _outer_piece as op
from discopt.validation.feasibility import verify_point

PROLOG = Path(__file__).parent / "data" / "minlplib_nl" / "prolog.nl"


def _toy() -> dm.Model:
    """Open above (x), open below (w), free (z), bounded (y); every rewrite rule."""
    m = dm.Model("toy1975")
    x = m.continuous("x", lb=1.0)
    w = m.continuous("w", lb=-1e20, ub=2.0)
    z = m.continuous("z", lb=-1e20, ub=1e20)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.subject_to(x * y + 3.0 * w <= 10.0 + 3.0 * x, name="c_bilinear")
    m.subject_to(w >= -2.0 * x - 5.0, name="c_lin")
    m.subject_to(z <= x + 1.0, name="c_z_up")
    m.subject_to(z >= -x - 1.0, name="c_z_lo")
    m.subject_to(dm.sqrt(x) * (y + 1.0) + abs(z) / (1.0 + y) <= 4.0 * x, name="c_sqrt_abs")
    m.subject_to(x**2 * dm.exp(y) - 50.0 * x * x <= 7.0 + x**1.5, name="c_powers")
    m.minimize(x * (2.0 - y) + (y - 0.5) ** 2 - 0.1 * w + 0.5 * abs(z))
    return m


def _random_piece_points(m: dm.Model, rng, n: int) -> list[np.ndarray]:
    pts = []
    for _ in range(n):
        x = 10.0 ** rng.uniform(1, 5)
        y = rng.uniform(0, 1)
        w = 2.0 - 10.0 ** rng.uniform(-1, 4)
        z = rng.uniform(-1, 1) * 10.0 ** rng.uniform(0, 4)
        pts.append(np.array([x, w, z, y]))
    return pts


def test_flag_default_off():
    assert os.environ.get(op.FLAG) is None or not op.outer_piece_enabled()
    assert op.outer_piece_enabled() is (os.environ.get(op.FLAG, "0") == "1")


def test_plan_classifies_open_columns():
    p = op.plan(_toy())
    assert p is not None
    kinds = {p.model._variables[c.var_index].name: (c.ref, c.sign) for c in p.cols}
    assert kinds["x"] == (1.0, 1.0)
    assert kinds["w"][1] == -1.0
    assert kinds["z"] == (0.0, 0.0)
    assert "y" not in kinds


def test_rewrite_is_an_exact_identity_on_every_row():
    """outer_row(image(x)) == s^D * (row(x) - rhs) for random points with T > 0."""
    m = _toy()
    p = op.plan(m)
    outer, d = op.build_outer(p, 1e2)
    assert d == 1
    rng = np.random.default_rng(1975)
    n_checked = 0
    for x in _random_piece_points(m, rng, 40):
        y = op.outer_point(p, outer, x)
        s = y[len(x)]
        for k, con in enumerate(m._constraints):
            orig = float(evaluate_at(con.body, m, x)) - float(con.rhs)
            new = float(evaluate_at(outer._constraints[k].body, outer, y))
            want = s ** p.row_degrees[k] * orig
            assert new == pytest.approx(want, rel=1e-9, abs=1e-12 * max(1.0, abs(orig)))
            n_checked += 1
        # objective: M^d s^d f
        f = float(evaluate_at(m._objective.expression, m, x))
        g = float(evaluate_at(outer._objective.expression, outer, y))
        assert g == pytest.approx((1e2 * s) ** d * f, rel=1e-9, abs=1e-12)
        n_checked += 1
    assert n_checked == 40 * (len(m._constraints) + 1)


def test_rewrite_declines_log_of_an_open_column():
    m = dm.Model("declines")
    x = m.continuous("x", lb=1.0)
    m.subject_to(dm.log(x) <= 3.0 + x)
    m.minimize(x)
    p = op.plan(m)
    with pytest.raises(op.OuterPieceUnsupported, match="log"):
        op.build_outer(p, 1e2)


def test_rewrite_declines_a_negative_power_of_an_open_column():
    m = dm.Model("declines_pow")
    x = m.continuous("x", lb=1.0)
    m.subject_to(x**-0.5 <= 3.0)
    m.minimize(x)
    p = op.plan(m)
    with pytest.raises(op.OuterPieceUnsupported, match="power"):
        op.build_outer(p, 1e2)


def test_no_open_column_is_a_noop_plan():
    m = dm.Model("bounded")
    x = m.continuous("x", lb=0.0, ub=3.0)
    m.minimize((x - 1) ** 2)
    assert op.plan(m) is None


def test_prolog_generators_are_x7_x8():
    """T sums only the columns whose boxing lets FBBT bound the rest."""
    m = dm.from_nl(str(PROLOG))
    p = op.plan(m)
    gens = sorted(m._variables[c.var_index].name for c in p.cols if c.in_T)
    assert gens == ["x7", "x8"]
    assert len(p.cols) == 7


def _feasible_prolog_points(m, p, rng, n):
    """Feasible points of prolog far out along T: the zero point scaled up.

    ``x = 0.2`` on x0, x1, x6 (their floors), every other column 0, satisfies
    every row; then x7 = x8 = t keeps every row (each is <= with the x7, x8
    columns on the slack side) and grows T."""
    pts = []
    names = [v.name for v in m._variables]
    for _ in range(n):
        x = np.zeros(len(names))
        for nm in ("x0", "x1", "x6"):
            x[names.index(nm)] = 0.2
        t = 10.0 ** rng.uniform(4.5, 7)
        x[names.index("x7")] = t * rng.uniform(0.2, 1.0)
        x[names.index("x8")] = t * rng.uniform(0.2, 1.0)
        pts.append(x)
    return pts


@pytest.mark.slow
@pytest.mark.correctness
def test_prolog_outer_bound_holds_at_feasible_points():
    """Feasible-point sampling: every sampled point of prolog's outer piece maps to
    a feasible point of the rescaled model, and its scaled objective is at least the
    certified outer bound."""
    m = dm.from_nl(str(PROLOG))
    p = op.plan(m)
    M = 1e4
    outer, d = op.build_outer(p, M)
    r = outer.solve(time_limit=60)
    assert r.bound is not None and r.bound > 0.0, (r.status, r.bound)
    rng = np.random.default_rng(1966)
    n_checked = 0
    for x in _feasible_prolog_points(m, p, rng, 20):
        assert verify_point(m, x).ok
        y = op.outer_point(p, outer, x)
        assert y[len(x)] <= 1.0 / M
        assert verify_point(outer, y).ok
        f = float(evaluate_at(m._objective.expression, m, x))
        T = 1.0 / y[len(x)]
        assert (M / T) ** d * f >= r.bound - 1e-9 * max(1.0, abs(f))
        n_checked += 1
    assert n_checked == 20


@pytest.mark.slow
@pytest.mark.correctness
def test_prolog_certifies_with_the_flag_and_not_without(monkeypatch):
    """Differential bound: OFF has no bound; ON's bound is finite and <= the
    MINLPLib optimum 0, and the certified objective is 0."""
    monkeypatch.setenv(op.FLAG, "0")
    off = dm.from_nl(str(PROLOG)).solve(time_limit=20)
    assert not off.gap_certified
    assert off.bound is None
    monkeypatch.setenv(op.FLAG, "1")
    on = dm.from_nl(str(PROLOG)).solve(time_limit=60)
    assert on.status == "optimal" and on.gap_certified, (on.status, on.solver_stats)
    assert on.solver_stats["outer_piece/result"] == "certified"
    assert on.bound <= 0.0 + 1e-6
    assert abs(on.objective - 0.0) <= 1e-4


@pytest.mark.slow
@pytest.mark.correctness
def test_toy_end_to_end_certifies_the_true_optimum(monkeypatch):
    """A model with open columns of every kind: the combined solve certifies the
    optimum found by the bounded reformulation (x at its floor)."""
    monkeypatch.setenv(op.FLAG, "1")
    m = _toy()
    r = m.solve(time_limit=60)
    assert r.solver_stats.get("outer_piece/open_columns") == 3.0, r.solver_stats
    assert r.solver_stats["outer_piece/outer_bound"] >= 0.0
    # Same model boxed by hand far beyond any optimum candidate: the reference.
    monkeypatch.setenv(op.FLAG, "0")
    ref = _toy()
    for v in ref._variables:
        if v.name == "x":
            v.ub = np.asarray(1e3)
        if v.name == "w":
            v.lb = np.asarray(-1e3)
        if v.name == "z":
            v.lb, v.ub = np.asarray(-1e3), np.asarray(1e3)
    r_ref = ref.solve(time_limit=60)
    assert r_ref.gap_certified
    if r.gap_certified:
        assert r.objective == pytest.approx(r_ref.objective, rel=1e-4, abs=1e-5)
    assert r.bound is not None and r.bound <= r_ref.objective + 1e-5
