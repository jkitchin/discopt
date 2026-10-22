"""Issue #1867: integer-domain minimum in ``SeparableQuadraticUpperBoundRule``.

``ball_mk3_*`` are one row ``sum_i a_i (x_i**2 - x_i) <= -1e-4`` over integers in
``[-1, 2]``. ``x(x-1) >= 0`` at every integer, so the row is infeasible by
inspection -- but the rule minimised each term over the *continuous* interval
(``-a/4`` at ``x = 0.5``) and proved nothing; the tree then enumerated (SCIP: 0.00 s,
discopt: time limit). With ``DISCOPT_INTEGER_SEPARABLE_QUAD`` an integer/binary
column's minimum is taken over its integer points (each widened by the integrality
tolerance), so presolve proves infeasibility and ``ball_mk2_*``'s domains shrink to
``{0, 1}``.

Every claim the rule makes is checked against an independent oracle here: brute-force
enumeration for the minimum, and feasible-point sampling for the tightened box.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt._relax import nonlinear_bound_tightening as N
from discopt.modeling.core import Model

pytestmark = pytest.mark.unit

FLAG = "DISCOPT_INTEGER_SEPARABLE_QUAD"
A = np.random.default_rng(1867).uniform(0.03, 0.18, 10)


def _ball(rhs: float, n: int = 10) -> Model:
    m = Model("ball")
    x = [m.integer(f"x{i}", lb=-1, ub=2) for i in range(n)]
    # Spelled as MINLPLib writes it (``a*sqr(x) - a*x``), the form the rule's term
    # flattener decomposes.
    m.subject_to(sum(float(A[i]) * x[i] ** 2 - float(A[i]) * x[i] for i in range(n)) <= rhs)
    m.minimize(sum(x))
    return m


def _tighten(m: Model):
    meta = N.build_flat_variable_metadata(m)
    from discopt._relax.primal_heuristics import _get_variable_bounds

    lb, ub = _get_variable_bounds(m)
    return N.SeparableQuadraticUpperBoundRule().tighten(m, lb.copy(), ub.copy(), meta)


def test_integer_minimum_matches_brute_force_and_never_undercuts_the_continuous_one():
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(500):
        a = float(rng.choice([0.0, rng.uniform(0.01, 5.0)]))
        b = float(rng.uniform(-10, 10))
        lo = float(rng.integers(-6, 4))
        hi = lo + float(rng.integers(0, 8))
        lb, ub = lo - float(rng.uniform(0, 0.9)), hi + float(rng.uniform(0, 0.9))
        got = N._min_univariate_quadratic_integer(a, b, lb, ub)
        ints = np.arange(np.ceil(lb - 1e-9), np.floor(ub + 1e-9) + 1)
        brute = float(np.min(a * ints**2 + b * ints))
        cont = N._min_univariate_quadratic(a, b, lb, ub)
        # Within the integrality-tolerance window of the true integer minimum...
        slack = abs(2 * a * ints).max() * N._INTEGER_DOMAIN_TOL + abs(b) * N._INTEGER_DOMAIN_TOL
        assert brute - slack - 1e-12 <= got <= brute + 1e-12, (a, b, lb, ub, got, brute)
        # ...and never below the continuous minimum (a pure tightening).
        assert got >= cont - 1e-12
        checked += 1
    assert checked == 500


def test_flag_on_proves_the_ball_mk3_shape_infeasible(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    with pytest.raises(N.NonlinearBoundTighteningInfeasible):
        _tighten(_ball(-1e-4))


def test_flag_off_is_the_continuous_rule(monkeypatch):
    """OFF: the continuous minimum (-sum(a)/4) proves nothing. (It still rounds some
    leave-one-out intervals inward -- the pre-#1867 behaviour, kept.)"""
    monkeypatch.setenv(FLAG, "0")
    lb, ub = _tighten(_ball(-1e-4))  # must not raise
    assert np.all(lb >= -1) and np.all(ub <= 2) and np.all(lb <= ub)


def test_tightened_box_keeps_every_feasible_integer_point(monkeypatch):
    """Feasible-point sampling: with rhs = 0 (the ball_mk2 shape, optimum 0) the ON
    box shrinks to {0, 1}; every integer point satisfying the row must stay in it."""
    monkeypatch.setenv(FLAG, "1")
    lb, ub = _tighten(_ball(0.0))
    assert np.all(lb == 0) and np.all(ub == 1), (lb, ub)
    rng = np.random.default_rng(7)
    inside = feasible = 0
    for _ in range(20000):
        x = rng.integers(-1, 3, size=10)
        if float(np.sum(A * (x**2 - x))) <= 0.0:
            feasible += 1
            inside += bool(np.all(x >= lb) and np.all(x <= ub))
    assert feasible > 0, "sampling found no feasible point -- the check measured nothing"
    assert inside == feasible


def test_flag_defaults_on(monkeypatch):
    """Graduated under CLAUDE.md section 5 (panel in the gate's docstring)."""
    monkeypatch.delenv(FLAG, raising=False)
    assert N.integer_separable_quadratic_enabled()
    with pytest.raises(N.NonlinearBoundTighteningInfeasible):
        _tighten(_ball(-1e-4))
    monkeypatch.setenv(FLAG, "0")
    assert not N.integer_separable_quadratic_enabled()
