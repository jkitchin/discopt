"""#1731: objective-lattice rounding in the spatial B&B tree (default OFF).

CLAUDE.md §5 bound-changing regime: the flag-ON bound must never cross the true
optimum (checked against brute-force enumeration), must be at least as tight as
flag OFF, and both arms must return the same certified objective.
"""

from __future__ import annotations

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.solver import _objective_lattice, _round_bounds_to_lattice


def _knapsack(w, cap, *, maximize: bool, offset: float = 0.0, nl: bool = True):
    n = len(w)
    m = dm.Model("ks")
    x = m.binary("x", shape=(n,))
    m.subject_to(np.abs(w) @ x <= cap)
    if maximize:
        m.maximize(w @ x + offset)
    else:
        m.minimize(w @ x + offset)
        m.subject_to(np.abs(w) @ x >= cap / 2)
    if nl:
        m.subject_to(x[0] * x[1] <= 1)  # redundant bilinear row -> spatial B&B
    return m


def _brute(w, cap, *, maximize: bool, offset: float = 0.0):
    best = None
    for bits in itertools.product((0, 1), repeat=len(w)):
        x = np.asarray(bits, dtype=float)
        if np.abs(w) @ x > cap + 1e-9:
            continue
        if not maximize and np.abs(w) @ x < cap / 2 - 1e-9:
            continue
        v = float(w @ x) + offset
        if best is None or (v > best if maximize else v < best):
            best = v
    return best


CASES = []
_rng = np.random.default_rng(1731)
for _k in range(6):
    _w = _rng.integers(3, 40, size=9).astype(float)
    if _k % 2:
        _w[::3] *= -1.0
    CASES.append((_w, float(np.abs(_w).sum() / 2) + 0.5, bool(_k % 3), float(_k) * 0.25))


@pytest.mark.parametrize("w, cap, maximize, offset", CASES)
def test_differential_bound_against_enumeration(monkeypatch, w, cap, maximize, offset):
    opt = _brute(w, cap, maximize=maximize, offset=offset)
    assert opt is not None
    monkeypatch.delenv("DISCOPT_SPATIAL_OBJ_LATTICE", raising=False)
    off = _knapsack(w, cap, maximize=maximize, offset=offset).solve(time_limit=30)
    monkeypatch.setenv("DISCOPT_SPATIAL_OBJ_LATTICE", "1")
    on = _knapsack(w, cap, maximize=maximize, offset=offset).solve(time_limit=30)
    for r in (off, on):
        assert r.objective == pytest.approx(opt, abs=1e-6)
    assert on.status == "optimal"
    assert on.bound is not None
    tol = 1e-6 * max(1.0, abs(opt))
    if maximize:
        assert on.bound >= opt - tol  # never below the true maximum
        if off.bound is not None:
            assert on.bound <= off.bound + tol
    else:
        assert on.bound <= opt + tol  # never above the true minimum
        if off.bound is not None:
            assert on.bound >= off.bound - tol


def test_issue_repro_certifies_with_flag(monkeypatch):
    monkeypatch.setenv("DISCOPT_SPATIAL_OBJ_LATTICE", "1")
    rng = np.random.default_rng(1)
    n = 30
    w = rng.integers(10, 100, size=n).astype(float)
    W = int(w.sum() / 2) + 0.5
    m = dm.Model("ss")
    x = m.binary("x", shape=(n,))
    m.subject_to(w @ x <= W)
    m.maximize(w @ x)
    m.subject_to(x[0] * x[1] <= 1)
    r = m.solve(time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(834.0)
    assert r.bound == pytest.approx(834.0)


def test_detector_refuses_anything_inexact():
    m = dm.Model()
    x = m.binary("x", shape=(3,))
    y = m.continuous("y", lb=0, ub=1)
    k = m.integer("k", lb=0, ub=5)
    m.minimize(4 * x[0] + 6 * x[1] - 10 * k + 0.5)
    assert _objective_lattice(m) == (2.0, 0.5)
    m.minimize(4 * x[0] + y)  # costed continuous column
    assert _objective_lattice(m) is None
    m.minimize(4 * x[0] + 0.5 * x[1])  # non-integer coefficient
    assert _objective_lattice(m) is None
    m.minimize(x[0] * x[1] * x[2])  # cubic: beyond the degree-2 extractor
    assert _objective_lattice(m) is None
    m.minimize(dm.exp(k))  # nonpolynomial
    assert _objective_lattice(m) is None
    m.minimize(0.5 * x[0] * x[1])  # non-integer monomial coefficient
    assert _objective_lattice(m) is None
    m.minimize(x[0] * y + 2 * x[1])  # bilinear term on a continuous column
    assert _objective_lattice(m) is None


def test_detector_accepts_integer_quadratics():
    m = dm.Model()
    x = m.binary("x", shape=(3,))
    k = m.integer("k", lb=-3, ub=3)
    m.minimize(3 * x[0] * x[1] + 6 * k * k - 9 * x[2] + 1)
    assert _objective_lattice(m) == (3.0, 1.0)
    # Two half-coefficients on the same monomial sum to an integer one.
    m.minimize(0.5 * x[0] * x[1] + 0.5 * x[1] * x[0] + 2 * x[2])
    assert _objective_lattice(m) == (1.0, 0.0)


def _bqp(Q, c, *, maximize: bool):
    n = len(c)
    m = dm.Model("bqp")
    x = m.binary("x", shape=(n,))
    obj = sum(int(Q[i, j]) * x[i] * x[j] for i in range(n) for j in range(i, n) if Q[i, j])
    obj = obj + c @ x
    m.subject_to(x[0] * x[1] <= 1)  # redundant bilinear row -> spatial B&B
    m.subject_to(dm.sum(x) <= n - 2)
    (m.maximize if maximize else m.minimize)(obj)
    return m


def _bqp_brute(Q, c, *, maximize: bool):
    n = len(c)
    vals = [
        float(np.asarray(b) @ np.triu(Q) @ np.asarray(b) + c @ np.asarray(b))
        for b in itertools.product((0, 1), repeat=n)
        if sum(b) <= n - 2
    ]
    return max(vals) if maximize else min(vals)


QCASES = []
for _k in range(4):
    _Q = np.triu(_rng.integers(-6, 7, size=(7, 7))).astype(float) * 2.0
    QCASES.append((_Q, _rng.integers(-9, 10, size=7).astype(float) * 2.0, bool(_k % 2)))


@pytest.mark.parametrize("Q, c, maximize", QCASES)
def test_quadratic_differential_bound_against_enumeration(monkeypatch, Q, c, maximize):
    opt = _bqp_brute(Q, c, maximize=maximize)
    assert _objective_lattice(_bqp(Q, c, maximize=maximize)) is not None
    monkeypatch.delenv("DISCOPT_SPATIAL_OBJ_LATTICE", raising=False)
    off = _bqp(Q, c, maximize=maximize).solve(time_limit=30)
    monkeypatch.setenv("DISCOPT_SPATIAL_OBJ_LATTICE", "1")
    on = _bqp(Q, c, maximize=maximize).solve(time_limit=30)
    for r in (off, on):
        assert r.objective == pytest.approx(opt, abs=1e-6)
    assert on.bound is not None
    tol = 1e-6 * max(1.0, abs(opt))
    if maximize:
        assert on.bound >= opt - tol
    else:
        assert on.bound <= opt + tol


def test_rounding_slack_never_jumps_a_step_on_noise():
    lbs = np.array([834.0 + 1e-9, 834.5, -834.5, 1e30, -np.inf, 833.9999999])
    out = _round_bounds_to_lattice(lbs, 1.0, 0.0)
    np.testing.assert_array_equal(out[:3], [834.0, 835.0, -834.0])
    assert out[3] == 1e30 and out[4] == -np.inf
    assert out[5] == 834.0
    np.testing.assert_array_equal(_round_bounds_to_lattice(np.array([1.3]), 2.0, 0.5), [2.5])
