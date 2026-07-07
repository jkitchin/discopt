"""Relaxation coverage audit: every common univariate operator must yield a
*valid, tight* global bound.

Each operator is solved as a single-operator model (minimize and maximize) over a
representative box.  Because these are one-dimensional we compute the reference
optimum by a dense brute-force grid, so the test is self-validating: it pins both

* **soundness** — the relaxation never excludes the true optimum, and
* **tightness** — spatial B&B closes the gap and reports ``status == "optimal"``
  with the correct objective.

This guards against regressions in the operator-coverage matrix and documents the
operators discopt can globally bound.  ``asin``/``acos``/``acosh`` were added here
after the Wave-2 coverage audit found they previously produced no valid bound.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest

_TOL = 1e-2


def _reciprocal(x):
    return 1.0 / x


def _pow2(x):
    return x**2


def _pow3(x):
    return x**3


def _pow_half(x):
    return x**0.5


# (id, model-builder fn, numpy reference fn, lb, ub)
_OPERATORS = [
    ("exp", dm.exp, np.exp, -1.0, 1.0),
    ("log", dm.log, np.log, 0.5, 3.0),
    ("log2", dm.log2, np.log2, 0.5, 3.0),
    ("log10", dm.log10, np.log10, 0.5, 3.0),
    ("sqrt", dm.sqrt, np.sqrt, 0.1, 4.0),
    ("reciprocal", _reciprocal, lambda x: 1.0 / x, 0.5, 3.0),
    ("sin", dm.sin, np.sin, 0.0, 3.0),
    ("cos", dm.cos, np.cos, 0.0, 3.0),
    ("tan", dm.tan, np.tan, -1.0, 1.0),
    ("atan", dm.atan, np.arctan, -2.0, 2.0),
    ("asin", dm.asin, np.arcsin, -1.0, 1.0),
    ("acos", dm.acos, np.arccos, -1.0, 1.0),
    ("acosh", dm.acosh, np.arccosh, 1.0, 3.0),
    ("sinh", dm.sinh, np.sinh, -2.0, 2.0),
    ("cosh", dm.cosh, np.cosh, -2.0, 2.0),
    ("tanh", dm.tanh, np.tanh, -2.0, 2.0),
    ("asinh", dm.asinh, np.arcsinh, -2.0, 2.0),
    ("atanh", dm.atanh, np.arctanh, -0.9, 0.9),
    ("erf", dm.erf, lambda x: np.vectorize(__import__("math").erf)(x), -2.0, 2.0),
    ("log1p", dm.log1p, np.log1p, -0.5, 3.0),
    ("sigmoid", dm.sigmoid, lambda x: 1.0 / (1.0 + np.exp(-x)), -3.0, 3.0),
    ("softplus", dm.softplus, lambda x: np.log1p(np.exp(x)), -3.0, 3.0),
    # abs certifies via the exact piecewise spatial B&B, which is ~15 s (kink at
    # 0); heavy for the PR-fast budget, so this one operator runs on the slow
    # path. The other operators stay in the PR-fast set.
    pytest.param("abs", dm.abs, np.abs, -2.0, 2.0, marks=pytest.mark.slow, id="abs"),
    ("pow2", _pow2, lambda x: x**2, -2.0, 2.0),
    ("pow3", _pow3, lambda x: x**3, -2.0, 2.0),
    ("pow_half", _pow_half, lambda x: x**0.5, 0.1, 4.0),
]


def _brute_optima(ref_fn, lb, ub):
    xs = np.linspace(lb, ub, 200_001)
    vals = ref_fn(xs)
    return float(np.min(vals)), float(np.max(vals))


def _solve(build_fn, lb, ub, sense):
    m = dm.Model("op")
    x = m.continuous("x", lb=lb, ub=ub)
    expr = build_fn(x)
    if sense == "min":
        m.minimize(expr)
    else:
        m.maximize(expr)
    return m.solve(time_limit=60)


@pytest.mark.relaxation
@pytest.mark.parametrize(
    "name,build_fn,ref_fn,lb,ub",
    _OPERATORS,
    ids=[getattr(o, "id", None) or o[0] for o in _OPERATORS],
)
def test_operator_has_valid_tight_bound(name, build_fn, ref_fn, lb, ub):
    """Each operator solves to a certified global optimum matching brute force."""
    true_min, true_max = _brute_optima(ref_fn, lb, ub)

    res_min = _solve(build_fn, lb, ub, "min")
    assert res_min.status == "optimal", f"{name}: minimize did not certify optimality"
    assert abs(float(res_min.objective) - true_min) < _TOL, (
        f"{name}: min objective {float(res_min.objective)} != brute force {true_min}"
    )

    res_max = _solve(build_fn, lb, ub, "max")
    assert res_max.status == "optimal", f"{name}: maximize did not certify optimality"
    assert abs(float(res_max.objective) - true_max) < _TOL, (
        f"{name}: max objective {float(res_max.objective)} != brute force {true_max}"
    )


@pytest.mark.relaxation
def test_bilinear_lifted_moment_bound():
    """The lifted-moment path bounds a bilinear product (min x*y over [0,1]^2 = 0)."""
    m = dm.Model("bilinear")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.minimize(x * y)
    res = m.solve(time_limit=60)
    assert res.status == "optimal"
    assert abs(float(res.objective)) < _TOL


@pytest.mark.slow
@pytest.mark.relaxation
@pytest.mark.parametrize("lb,ub", [(-1.0, 1.0), (-2.0, 2.0), (-2.0, 3.0)])
def test_abs_zero_crossing_certifies(lb, ub):
    """Regression guard: minimize |x| over an interval straddling 0 certifies 0.

    A smooth gradient-based NLP oscillates at the kink (it used to return ~0.99
    with ``iteration_limit``); non-smooth models now fall back to the spatial
    McCormick B&B, whose exact piecewise ``abs`` relaxation certifies the
    optimum of 0.
    """
    res = _solve(dm.abs, lb, ub, "min")
    assert res.status == "optimal"
    assert abs(float(res.objective)) < _TOL


@pytest.mark.relaxation
@pytest.mark.parametrize("name", ["asin", "acos", "acosh"])
def test_inverse_trig_gap_closed(name):
    """Regression guard for the three operators that previously had no valid bound.

    Before the Wave-2 fix these returned ``status == "feasible"`` with an objective
    that missed the true optimum (no dual bound).  They must now certify optimality.
    """
    build_fn, ref_fn, lb, ub = {
        "asin": (dm.asin, np.arcsin, -1.0, 1.0),
        "acos": (dm.acos, np.arccos, -1.0, 1.0),
        "acosh": (dm.acosh, np.arccosh, 1.0, 3.0),
    }[name]
    true_min, _ = _brute_optima(ref_fn, lb, ub)
    res = _solve(build_fn, lb, ub, "min")
    assert res.status == "optimal"
    assert abs(float(res.objective) - true_min) < _TOL
