"""Stochastic programming for discopt.

Two-stage stochastic programs with recourse: a here-and-now first-stage decision
made before the uncertainty is realized, and a wait-and-see recourse per scenario.
The distributional sibling of :mod:`discopt.ro` (which optimizes the worst case).

Phase 0 ships the scenario layer, risk measures, and the extensive-form
(deterministic-equivalent) builder. See ``docs/dev/stochastic-module-plan.md``.
"""

from __future__ import annotations

from discopt.stochastic.extensive_form import ExtensiveForm, build_extensive_form
from discopt.stochastic.risk import CVaR, Expectation, MeanCVaR, RiskMeasure, chance_constraint
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
]
