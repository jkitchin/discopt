"""Regression tests for robust-optimization soundness fixes RO-1 and RO-2.

- **RO-1** (`box.py`): sign-tracking treated a bare ``Parameter * Variable`` as if
  the parameter were the coefficient, correct only when the variable is provably
  ``>= 0``. For a sign-indefinite variable the "robust" counterpart under-protected
  (returned a point that violates the constraint at an in-set realization — not a
  counterpart). Such terms now route through the ``|coeff|`` linearization path.
- **RO-2** (universal guard): several formulations silently left an uncertain
  parameter at its nominal value when the expression pattern was unsupported (e.g.
  ellipsoidal robustifies only ``p @ x``). The counterpart now refuses loudly if
  any uncertain parameter survives ``formulate()`` — no silently non-robust models.

Each fails on the pre-fix code (RO-1 returns the wrong optimum; RO-2 returns the
nominal model with no error).
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.ro import (
    BoxUncertaintySet,
    EllipsoidalUncertaintySet,
    RobustCounterpart,
)

pytestmark = pytest.mark.smoke


# --------------------------------------------------------------------------- RO-1
def test_ro1_sign_indefinite_variable_coefficient():
    """max x s.t. p*x <= -1, x in [-10,10], pbar=1, delta=0.5 -> robust x = -2."""
    m = dm.Model("ro1")
    x = m.continuous("x", shape=(1,), lb=-10, ub=10)
    p = m.parameter("p", value=1.0)
    m.maximize(x[0])
    m.subject_to(p * x[0] <= -1, name="c")
    RobustCounterpart(m, BoxUncertaintySet(p, delta=0.5)).formulate()
    r = m.solve()
    xval = float(r.value(x)[0])
    # Pre-fix returned x = -0.667 (violates the constraint at p=0.5).
    assert xval == pytest.approx(-2.0, abs=1e-4)


def test_ro1_solution_is_actually_robust():
    """The RO-1 solution must satisfy the constraint at the worst-case realization."""
    m = dm.Model("ro1r")
    x = m.continuous("x", shape=(1,), lb=-10, ub=10)
    p = m.parameter("p", value=1.0)
    m.maximize(x[0])
    m.subject_to(p * x[0] <= -1, name="c")
    RobustCounterpart(m, BoxUncertaintySet(p, delta=0.5)).formulate()
    xval = float(m.solve().value(x)[0])
    # For every p in [0.5, 1.5], p*x <= -1 must hold (x < 0, so p=0.5 is worst).
    for pval in np.linspace(0.5, 1.5, 11):
        assert pval * xval <= -1 + 1e-6, f"violated at p={pval}"


def test_ro1_nonnegative_variable_fast_path_still_correct():
    """Control: x >= 0 keeps the sign-tracking fast path and stays correct."""
    m = dm.Model("ro1pos")
    x = m.continuous("x", shape=(1,), lb=0, ub=10)
    p = m.parameter("p", value=1.0)
    m.maximize(x[0])
    m.subject_to(p * x[0] <= 4, name="c")
    RobustCounterpart(m, BoxUncertaintySet(p, delta=0.5)).formulate()
    xval = float(m.solve().value(x)[0])
    # Worst case p = 1.5, so 1.5*x <= 4 -> x <= 2.667.
    assert xval == pytest.approx(4.0 / 1.5, abs=1e-4)
    for pval in np.linspace(0.5, 1.5, 11):
        assert pval * xval <= 4 + 1e-6


# --------------------------------------------------------------------------- RO-2
def test_ro2_scalar_ellipsoidal_refuses_loudly():
    """Scalar p*x is not the p@x pattern -> must raise, not silently no-op."""
    m = dm.Model("ro2s")
    x = m.continuous("x", shape=(1,), lb=0, ub=10)
    p = m.parameter("p", value=1.0)
    m.maximize(x[0])
    m.subject_to(p * x[0] <= 4, name="c")
    with pytest.raises(NotImplementedError, match="silently non-robust|could not robustify"):
        RobustCounterpart(m, EllipsoidalUncertaintySet(p, rho=0.5)).formulate()


def test_ro2_elementwise_sum_ellipsoidal_refuses_loudly():
    """dm.sum(p * x) (elementwise, not MatMul) must raise, not silently no-op."""
    m = dm.Model("ro2e")
    x = m.continuous("x", shape=(3,), lb=0, ub=1)
    p = m.parameter("p", value=np.array([1.0, 2.0, 3.0]))
    m.maximize(dm.sum(p * x))
    m.subject_to(dm.sum(x) <= 1, name="budget")
    with pytest.raises(NotImplementedError, match="silently non-robust|could not robustify"):
        RobustCounterpart(m, EllipsoidalUncertaintySet(p, rho=0.5)).formulate()


def test_ro2_blessed_matmul_still_formulates():
    """Control: the supported p @ x pattern must NOT trip the guard."""
    m = dm.Model("blessed")
    x = m.continuous("x", shape=(3,), lb=0, ub=1)
    p = m.parameter("p", value=np.array([1.0, 2.0, 3.0]))
    m.maximize(p @ x)
    m.subject_to(dm.sum(x) <= 1, name="budget")
    RobustCounterpart(m, EllipsoidalUncertaintySet(p, rho=0.5)).formulate()
    assert m.solve().status == "optimal"


def test_ro2_blessed_box_still_formulates():
    """Control: constant-coeff box counterpart must NOT trip the guard."""
    m = dm.Model("blessedbox")
    x = m.continuous("x", shape=(3,), lb=0, ub=1)
    c = m.parameter("c", value=np.array([1.0, 2.0, 3.0]))
    m.minimize(dm.sum(c * x))
    m.subject_to(dm.sum(x) >= 1, name="cover")
    RobustCounterpart(m, BoxUncertaintySet(c, delta=0.1 * np.ones(3))).formulate()
    assert m.solve().status == "optimal"
