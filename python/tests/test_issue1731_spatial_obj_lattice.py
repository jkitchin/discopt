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
    m.minimize(x[0] * x[1])  # nonlinear
    assert _objective_lattice(m) is None


def test_rounding_slack_never_jumps_a_step_on_noise():
    lbs = np.array([834.0 + 1e-9, 834.5, -834.5, 1e30, -np.inf, 833.9999999])
    out = _round_bounds_to_lattice(lbs, 1.0, 0.0)
    np.testing.assert_array_equal(out[:3], [834.0, 835.0, -834.0])
    assert out[3] == 1e30 and out[4] == -np.inf
    assert out[5] == 834.0
    np.testing.assert_array_equal(_round_bounds_to_lattice(np.array([1.3]), 2.0, 0.5), [2.5])
