"""Stochastic programming for discopt.

Two-stage stochastic programs with recourse: a here-and-now first-stage decision
made before the uncertainty is realized, and a wait-and-see recourse per scenario.
The distributional sibling of :mod:`discopt.ro` (which optimizes the worst case).

Phase 0 ships the scenario layer, risk measures, and the extensive-form
(deterministic-equivalent) builder. See ``docs/dev/stochastic-module-plan.md``.
"""

from __future__ import annotations

from discopt.stochastic.dro import worst_case_distribution, worst_case_expected_cost
from discopt.stochastic.extensive_form import ExtensiveForm, build_extensive_form
from discopt.stochastic.lshaped import LShapedResult, solve_lshaped
from discopt.stochastic.multistage import integer_lshaped, solve_multistage
from discopt.stochastic.ph import PHResult, progressive_hedging, quadratic_subproblem_solver
from discopt.stochastic.risk import CVaR, Expectation, MeanCVaR, RiskMeasure, chance_constraint
from discopt.stochastic.saa import (
    SAABound,
    optimality_gap_estimate,
    saa_lower_bound_estimate,
    saa_upper_bound_estimate,
)
from discopt.stochastic.scenario import Scenario, ScenarioSet

__all__ = [
    "Scenario",
    "ScenarioSet",
    "Expectation",
    "CVaR",
    "MeanCVaR",
    "RiskMeasure",
    "chance_constraint",
    "ExtensiveForm",
    "build_extensive_form",
    "LShapedResult",
    "solve_lshaped",
    "PHResult",
    "progressive_hedging",
    "quadratic_subproblem_solver",
    "SAABound",
    "saa_lower_bound_estimate",
    "saa_upper_bound_estimate",
    "optimality_gap_estimate",
    "worst_case_distribution",
    "worst_case_expected_cost",
    "solve_multistage",
    "integer_lshaped",
]
