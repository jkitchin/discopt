"""Tests for the improvement heuristics: diving, RINS, local branching.

These heuristics may only *propose* feasible incumbents; they must never mutate
the model permanently nor (by construction) affect the dual bound. The tests
assert: produced points are integer- and constraint-feasible, never beat the
known global optimum, and the model's variable bounds are restored afterwards.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

pytest.importorskip("pounce")

from discopt._jax.nlp_evaluator import NLPEvaluator  # noqa: E402
from discopt._jax.primal_heuristics import (  # noqa: E402
    _get_integer_mask,
    _is_integer_feasible,
    diving,
    fractional_diving,
    local_branching,
    objective_diving,
    rins,
)
from discopt.modeling.core import Model  # noqa: E402
from discopt.solvers.nlp_backend import get_nlp_solver  # noqa: E402

pytestmark = pytest.mark.requires_pounce


def _bounds_snapshot(model: Model):
    return [(np.array(v.lb), np.array(v.ub)) for v in model._variables]


def _assert_bounds_restored(model: Model, snap) -> None:
    for v, (lb, ub) in zip(model._variables, snap):
        assert np.allclose(np.array(v.lb), lb), "variable lb not restored"
        assert np.allclose(np.array(v.ub), ub), "variable ub not restored"


def _quad_minlp() -> tuple[Model, np.ndarray]:
    """min (x-2.3)^2 + (y-0.4)^2 ; x integer in [0,5], y cont in [0,5].

    Integer optimum: x=2, y=0.4, objective 0.09.
    """
    m = Model("quad_minlp")
    x = m.integer("x", lb=0, ub=5)
    y = m.continuous("y", lb=0, ub=5)
    m.minimize((x - 2.3) ** 2 + (y - 0.4) ** 2)
    return m, np.array([2.3, 0.4])


def _relax(model: Model, x0: np.ndarray) -> np.ndarray:
    ev = NLPEvaluator(model)
    res = get_nlp_solver("auto")(ev, x0, options={"print_level": 0})
    assert res.x is not None
    return np.asarray(res.x)


# ───────────────────────────── diving ─────────────────────────────


def test_fractional_diving_finds_feasible_incumbent():
    m, x0 = _quad_minlp()
    snap = _bounds_snapshot(m)
    x_relax = _relax(m, x0)

    out = fractional_diving(m, x_relax)
    assert out is not None
    x, obj = out
    int_mask = _get_integer_mask(m)
    assert _is_integer_feasible(x, int_mask)
    # Nearest-integer optimum is x=2 -> objective 0.09; never beats it.
    assert obj >= 0.09 - 1e-6
    assert abs(obj - 0.09) < 1e-4
    _assert_bounds_restored(m, snap)


def test_objective_diving_feasible():
    m, x0 = _quad_minlp()
    x_relax = _relax(m, x0)
    out = objective_diving(m, x_relax)
    assert out is not None
    x, obj = out
    assert _is_integer_feasible(x, _get_integer_mask(m))
    assert obj >= 0.09 - 1e-6


def test_diving_pure_continuous_returns_none():
    m = Model("cont")
    x = m.continuous("x", lb=0, ub=5)
    m.minimize((x - 1.5) ** 2)
    assert diving(m, np.array([1.5])) is None


# ───────────────────────────── RINS ─────────────────────────────


def test_rins_improves_on_poor_incumbent():
    m, x0 = _quad_minlp()
    snap = _bounds_snapshot(m)
    x_relax = _relax(m, x0)
    # Poor incumbent x=3 (objective 0.49) disagrees with the relaxation (x≈2.3).
    incumbent = np.array([3.0, 0.4])
    out = rins(m, incumbent, x_relax)
    assert out is not None
    x, obj = out
    assert _is_integer_feasible(x, _get_integer_mask(m))
    assert obj < 0.49  # strictly better than the seed incumbent
    _assert_bounds_restored(m, snap)


def test_rins_full_agreement_returns_none():
    m, x0 = _quad_minlp()
    x_relax = _relax(m, x0)
    # Build an incumbent that agrees with the (rounded) relaxation on all ints
    # by first rounding the relaxation to an integer-feasible point.
    rounded = x_relax.copy()
    int_mask = _get_integer_mask(m)
    rounded[int_mask] = np.round(rounded[int_mask])
    # Make the relaxation itself integral so agreement is full.
    assert rins(m, rounded, rounded) is None


# ─────────────────────────── local branching ───────────────────────────


def _binary_model() -> Model:
    """min (a+b-1)^2 + (z-0.5)^2 ; a,b binary, z cont. Optimum a+b=1, z=0.5."""
    m = Model("binmodel")
    a = m.binary("a")
    b = m.binary("b")
    z = m.continuous("z", lb=-5, ub=5)
    m.minimize((a + b - 1) ** 2 + (z - 0.5) ** 2)
    return m


def test_local_branching_improves_incumbent():
    m = _binary_model()
    snap = _bounds_snapshot(m)
    # Start from a=0,b=0 (objective 1.0 + 0 = 1.0 at z=0.5).
    incumbent = np.array([0.0, 0.0, 0.5])
    out = local_branching(m, incumbent, k=1)
    assert out is not None
    x, obj = out
    assert _is_integer_feasible(x, _get_integer_mask(m))
    assert obj < 1.0 - 1e-6  # found a strictly better neighbour
    assert abs(obj) < 1e-4  # the true optimum (objective 0)
    _assert_bounds_restored(m, snap)


def test_local_branching_too_many_binaries_returns_none():
    m = _binary_model()
    out = local_branching(m, np.array([0.0, 0.0, 0.5]), k=1, max_binaries=1)
    assert out is None
