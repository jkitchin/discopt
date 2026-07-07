"""Regression tests for robust-optimization soundness fixes RO-1, RO-2, RO-3.

- **RO-1** (`box.py`): sign-tracking treated a bare ``Parameter * Variable`` as if
  the parameter were the coefficient, correct only when the variable is provably
  ``>= 0``. For a sign-indefinite variable the "robust" counterpart under-protected
  (returned a point that violates the constraint at an in-set realization — not a
  counterpart). Such terms now route through the ``|coeff|`` linearization path.
- **RO-2** (universal guard): several formulations silently left an uncertain
  parameter at its nominal value when the expression pattern was unsupported (e.g.
  ellipsoidal robustifies only ``p @ x``). The counterpart now refuses loudly if
  any uncertain parameter survives ``formulate()`` — no silently non-robust models.
- **RO-3** (`uncertainty.py:budget_uncertainty_set` + `polyhedral.py`): the budget
  set stored only the all-plus / all-minus budget facets (2 of the 2^k needed) — a
  strict *superset* of the true Bertsimas–Sim set, so the "robust" counterpart was
  SOUND but up to 2× over-conservative in mixed-sign directions (support of
  ``[1,-1]`` at k=2, δ=1, Γ=1 was 2.0 vs the true 1.0). The set now stores the exact
  compact lifted ``(ξ,u)`` polytope and the polyhedral dualization dualizes it
  exactly. The RO-3 tests assert BOTH directions: the new solution is still robust
  at every in-set realization (the load-bearing check — a violation here would be
  under-protection, worse than the bug) AND it is less conservative / matches the
  analytical B–S optimum.

Each fails on the pre-fix code (RO-1 returns the wrong optimum; RO-2 returns the
nominal model with no error; RO-3 support is 2.0 not 1.0 and the solve is
over-conservative).
"""

from __future__ import annotations

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.ro import (
    BoxUncertaintySet,
    EllipsoidalUncertaintySet,
    RobustCounterpart,
)
from discopt.ro.formulations.polyhedral import _support_function_lp
from discopt.ro.uncertainty import budget_uncertainty_set

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


# --------------------------------------------------------------------------- RO-3
def _bs_support(coeff, delta, gamma):
    """Analytical Bertsimas–Sim support function: sum of the Γ largest |c_j|·δ_j."""
    w = np.sort(np.abs(coeff) * delta)[::-1]
    g, total = gamma, 0.0
    for wi in w:
        take = min(1.0, g)
        total += take * wi
        g -= take
        if g <= 0:
            break
    return total


def _in_budget(xi, delta, gamma, tol=1e-9):
    return bool(np.all(np.abs(xi) <= delta + tol) and np.sum(np.abs(xi) / delta) <= gamma + tol)


@pytest.mark.parametrize(
    "direction",
    [[1.0, -1.0], [1.0, 1.0], [1.0, 0.0], [-1.0, 1.0], [2.0, -1.0], [-3.0, 2.0]],
)
def test_ro3_support_function_matches_bertsimas_sim(direction):
    """Budget-set support in mixed-sign directions = analytical B–S value.

    Pre-fix, the stored A,b was the all-plus/all-minus superset: support([1,-1]) at
    k=2,δ=1,Γ=1 was 2.0 (the box), not the true 1.0.
    """
    delta = np.array([1.0, 1.0])
    gamma = 1.0
    m = dm.Model("sf")
    p = m.parameter("p", value=np.zeros(2))
    unc = budget_uncertainty_set(p, delta=delta, gamma=gamma)
    coeff = np.array(direction)
    coeff_lifted = np.concatenate([coeff, np.zeros(unc.A.shape[1] - 2)])
    sf = _support_function_lp(coeff_lifted, unc.A, unc.b, maximize=True)
    true = _bs_support(coeff, delta, gamma)
    # LP applies a tiny conservative (upward) margin on a max; allow it.
    assert sf == pytest.approx(true, abs=1e-4), f"{direction}: {sf} != BS {true}"


def test_ro3_support_endpoints_nominal_and_box():
    """Γ→0 gives ~nominal (0 protection); Γ=k recovers the full box."""
    delta = np.array([1.0, 1.0])
    coeff = np.array([1.0, 1.0])
    m = dm.Model("ep")
    p = m.parameter("p", value=np.zeros(2))
    unc0 = budget_uncertainty_set(p, delta=delta, gamma=1e-9)
    c_lift0 = np.concatenate([coeff, np.zeros(unc0.A.shape[1] - 2)])
    assert _support_function_lp(c_lift0, unc0.A, unc0.b, maximize=True) == pytest.approx(
        0.0, abs=1e-4
    )
    unc_box = budget_uncertainty_set(p, delta=delta, gamma=2.0)
    c_liftb = np.concatenate([coeff, np.zeros(unc_box.A.shape[1] - 2)])
    # box worst case = Σ|c_j|δ_j = 2.0
    assert _support_function_lp(c_liftb, unc_box.A, unc_box.b, maximize=True) == pytest.approx(
        2.0, abs=1e-4
    )


def _solve_budget_model(gamma):
    """max x0-x1 s.t. (ξ)·x + x0 - x1 <= 3, ξ ∈ budget(δ=1, Γ=γ), x ∈ [-5,5]^2.

    coeff(x)=x is sign-indefinite at the optimum (x=[t,-t]), so the mixed-sign
    over-conservatism of the pre-fix budget set is exercised end-to-end.
    """
    m = dm.Model("bs")
    x = m.continuous("x", shape=(2,), lb=-5, ub=5)
    a = m.parameter("a", value=np.zeros(2))
    m.subject_to(dm.sum(a * x) + x[0] - x[1] <= 3, name="c")
    m.maximize(x[0] - x[1])
    RobustCounterpart(m, budget_uncertainty_set(a, delta=np.ones(2), gamma=gamma)).formulate()
    r = m.solve()
    return r, np.asarray(r.value(x))


def test_ro3_price_of_robustness_matches_analytical():
    """(b) Less conservative: objective matches the closed-form B–S optimum.

    Constraint reduces to 2t + min(Γ,2)·t ≤ 3 with x=[t,-t], obj=2t:
      Γ=0 → t=1.5, obj=3.0 ;  Γ=1 → t=1, obj=2.0 ;  Γ=2 → t=0.75, obj=1.5.
    Pre-fix the budget facets made support([t,-t])=2t for all Γ, so Γ=1 gave the
    over-conservative obj≈1.5 instead of 2.0.
    """
    expected = {0.0: 3.0, 1.0: 2.0, 2.0: 1.5}
    prev = None
    for gamma, want in expected.items():
        r, _ = _solve_budget_model(gamma)
        assert r.status == "optimal"
        assert r.objective == pytest.approx(want, abs=1e-3), f"Γ={gamma}: {r.objective}"
        if prev is not None:  # price of robustness is monotone non-increasing
            assert r.objective <= prev + 1e-6
        prev = r.objective


def test_ro3_less_conservative_than_prefix_superset():
    """(b) The fixed Γ=1 objective (2.0) strictly beats the pre-fix superset (1.5)."""
    r, _ = _solve_budget_model(1.0)
    assert r.objective == pytest.approx(2.0, abs=1e-3)
    assert r.objective > 1.5 + 1e-3  # strictly less conservative than the old set


def test_ro3_solution_is_robust_at_every_in_set_realization():
    """(a) LOAD-BEARING: the robust solution must satisfy the constraint at EVERY
    realization in the TRUE Bertsimas–Sim budget set.

    Samples the budget polytope exhaustively (vertices + dense grid + random
    interior). A violation here is under-protection — worse than the original bug.
    """
    delta = np.ones(2)
    gamma = 1.0
    _, xv = _solve_budget_model(gamma)

    # Vertices of the budget set at k=2, Γ=1: ±δ_j e_j and 0.
    samples = [
        np.array([1.0, 0.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, -1.0]),
        np.zeros(2),
    ]
    grid = np.linspace(-1.0, 1.0, 41)
    for a1, a2 in itertools.product(grid, grid):
        xi = np.array([a1, a2])
        if _in_budget(xi, delta, gamma):
            samples.append(xi)
    rng = np.random.default_rng(0)
    for _ in range(20000):
        xi = rng.uniform(-1.0, 1.0, 2)
        if _in_budget(xi, delta, gamma):
            samples.append(xi)

    worst = -np.inf
    for xi in samples:
        lhs = float(xi @ xv + xv[0] - xv[1])  # constraint body
        worst = max(worst, lhs - 3.0)  # <= 0 means satisfied
    assert len(samples) > 5000
    assert worst <= 1e-6, f"UNDER-PROTECTION: worst in-set violation {worst}"
