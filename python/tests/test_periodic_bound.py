"""Tests for the periodic-variable bound reduction.

A continuous variable used only inside sin/cos can be restricted to one period
without loss of optimality. The rule must (a) reduce such variables, (b) leave
variables used anywhere else untouched, and (c) only ever shrink the box (sound).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.nonlinear_bound_tightening import (
    PeriodicVariableBoundRule,
    build_flat_variable_metadata,
)

_RULE = PeriodicVariableBoundRule()
_PI = np.pi


def _apply(m, lb, ub):
    md = build_flat_variable_metadata(m)
    return _RULE.tighten(m, np.array(lb, float), np.array(ub, float), md)


def test_reduces_free_periodic_only_variable():
    m = dm.Model("p")
    y = m.continuous("y")
    m.minimize(dm.cos(y))
    nlb, nub = _apply(m, [-1e20], [1e20])
    assert np.isclose(nlb[0], -_PI) and np.isclose(nub[0], _PI)


def test_skips_variable_used_outside_cos():
    m = dm.Model("p")
    y = m.continuous("y")
    m.minimize(y + dm.cos(y))  # y also appears linearly -> not periodic-only
    nlb, nub = _apply(m, [-1e20], [1e20])
    assert nlb[0] == -1e20 and nub[0] == 1e20


def test_skips_non_bare_argument():
    m = dm.Model("p")
    y = m.continuous("y")
    m.minimize(dm.cos(2 * y))  # period is pi, not 2pi -> conservatively skip
    nlb, nub = _apply(m, [-1e20], [1e20])
    assert nlb[0] == -1e20 and nub[0] == 1e20


def test_shrinks_wide_finite_box_to_one_period():
    m = dm.Model("p")
    y = m.continuous("y", lb=0.0, ub=100.0)
    m.minimize(dm.sin(y))
    nlb, nub = _apply(m, [0.0], [100.0])
    # Anchored at the lower bound, shrunk to exactly one period.
    assert np.isclose(nlb[0], 0.0) and np.isclose(nub[0], 2 * _PI)


def test_no_change_when_already_within_period():
    m = dm.Model("p")
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.minimize(dm.cos(y))
    nlb, nub = _apply(m, [0.0], [1.0])
    assert nlb[0] == 0.0 and nub[0] == 1.0


def test_integer_periodic_variable_untouched():
    m = dm.Model("p")
    y = m.integer("y", lb=-1000, ub=1000)
    m.minimize(dm.cos(y))
    nlb, nub = _apply(m, [-1000.0], [1000.0])
    assert nlb[0] == -1000.0 and nub[0] == 1000.0  # only continuous vars reduced


@pytest.mark.slow
def test_end_to_end_unblocks_free_angular_variable():
    # nlp_001-style: cos(y) over a free y blocks convergence without the rule.
    m = dm.Model("nlp001")
    x = m.continuous("x")
    y = m.continuous("y")
    z = m.continuous("z", lb=1.0)
    m.minimize(x * dm.exp(x) + dm.cos(y) + z**3 - z**2)
    res = m.solve(time_limit=60)
    assert res.status == "optimal"
    assert abs(float(res.objective) - (-1.3679)) < 1e-2
    assert res.node_count <= 5  # was 237 (non-converging) before the rule
