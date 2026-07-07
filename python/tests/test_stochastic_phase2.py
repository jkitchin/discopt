"""SP Phase 2: progressive hedging + convex-nonlinear recourse routing.

The PH numerical core is verified against analytic separable-quadratic subproblems
(no optimization backend needed): PH must converge to the expected-value first-stage
`x̄ = Σ_s p_s a_s`, keep the multiplier invariant `Σ_s p_s w_s = 0`, and drive the
nonanticipativity gap to zero. Plus the L-shaped `method` routing (linear vs GBD)
validation. Needs NumPy; no JAX/Rust.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt.modeling.core import Constant, Model  # noqa: E402
from discopt.stochastic import ScenarioSet, solve_lshaped
from discopt.stochastic.ph import progressive_hedging, quadratic_subproblem_solver

pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------
# Progressive hedging (analytic subproblems).
# ---------------------------------------------------------------------------


def test_ph_converges_to_expected_value_solution():
    scen = ScenarioSet.from_list([(0.5, {}), (0.3, {}), (0.2, {})])
    targets = {0: np.array([1.0, 2.0]), 1: np.array([3.0, -1.0]), 2: np.array([0.0, 5.0])}
    solver = quadratic_subproblem_solver(targets)
    res = progressive_hedging(
        scen, first_stage_dim=2, subproblem_solver=solver, rho=1.0, max_iter=2000, tol=1e-10
    )
    probs = scen.probabilities
    expected = sum(probs[s] * targets[s] for s in range(3))  # [1.4, 1.7]
    assert res.converged
    assert np.allclose(res.x_bar, expected, atol=1e-6)
    # every scenario agrees on the nonanticipative decision
    assert np.allclose(res.x_scenarios, res.x_bar, atol=1e-6)


def test_ph_multiplier_invariant_probability_weighted_zero():
    scen = ScenarioSet.from_list([(0.4, {}), (0.6, {})])
    targets = {0: np.array([2.0]), 1: np.array([-3.0])}
    res = progressive_hedging(
        scen, 1, quadratic_subproblem_solver(targets), rho=0.7, max_iter=2000, tol=1e-11
    )
    probs = scen.probabilities
    assert abs(float(np.average(res.weights[:, 0], weights=probs))) < 1e-6


def test_ph_gap_converges_and_rho_positive():
    scen = ScenarioSet.from_samples({"_": np.zeros(4)})  # 4 equal-weight scenarios
    targets = {s: np.array([float(s)]) for s in range(4)}
    res = progressive_hedging(
        scen, 1, quadratic_subproblem_solver(targets), rho=1.0, max_iter=1000, tol=1e-9
    )
    assert res.converged and res.gap < 1e-9
    assert res.history[-1] <= res.history[0]  # gap shrank
    # rho must be positive
    with pytest.raises(ValueError, match="rho > 0"):
        progressive_hedging(scen, 1, quadratic_subproblem_solver(targets), rho=0.0)


def test_ph_converges_regardless_of_rho():
    scen = ScenarioSet.from_list([(0.5, {}), (0.5, {})])
    targets = {0: np.array([10.0]), 1: np.array([-4.0])}
    expected = np.array([3.0])  # 0.5*10 + 0.5*(-4)
    for rho in (0.1, 1.0, 5.0):
        res = progressive_hedging(
            scen, 1, quadratic_subproblem_solver(targets), rho=rho, max_iter=5000, tol=1e-11
        )
        assert np.allclose(res.x_bar, expected, atol=1e-6), f"rho={rho}"


# ---------------------------------------------------------------------------
# L-shaped method routing (linear vs GBD for convex-nonlinear recourse).
# ---------------------------------------------------------------------------


def _tiny_lshaped(method):
    m = Model("route")
    q = m.continuous("q", lb=0, ub=10)
    scen = ScenarioSet.from_list([(0.5, {"demand": 4}), (0.5, {"demand": 8})])

    def recourse(model, data, s):
        d = float(data["demand"])
        y = model.continuous(f"y_{s}", lb=0, ub=10)
        model.subject_to(y <= q)
        model.subject_to(y <= Constant(d))
        return Constant(-1.0) * y

    return solve_lshaped(
        m,
        first_stage_vars=[q],
        scenarios=scen,
        recourse_builder=recourse,
        first_stage_cost=Constant(1.0) * q,
        method=method,
        solve=False,
    )


def test_lshaped_method_validation():
    with pytest.raises(ValueError, match="method must be"):
        _tiny_lshaped("nope")


@pytest.mark.parametrize("method", ["auto", "benders", "gbd"])
def test_lshaped_accepts_valid_methods_and_builds_structure(method):
    res = _tiny_lshaped(method)
    assert res.structure is not None
    assert res.structure.complicating_vars == ["q"]
    assert res.result is None  # solve=False
