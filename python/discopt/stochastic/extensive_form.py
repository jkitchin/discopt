"""Extensive form (deterministic equivalent) of a two-stage stochastic program.

Builds the single monolithic model ``min c·x + risk_s[ Q_s(x) ]`` by calling a
**scenario-creator callback** once per scenario (the mpi-sppy / PySP pattern): the
callback constructs that scenario's second-stage variables and constraints (given
the realized data) and returns its recourse cost expression. The risk measure
(:mod:`~discopt.stochastic.risk`) then assembles the objective.

This is correct by construction (an exact deterministic equivalent), so it is the
Phase 0 baseline *and* the oracle the decomposed methods (L-shaped, PH) are checked
against. See ``docs/dev/stochastic-module-plan.md`` §1, §3.
"""

from __future__ import annotations

from dataclasses import dataclass

from discopt.modeling.core import Constant, Expression, Model
from discopt.stochastic.risk import Expectation, RiskMeasure
from discopt.stochastic.scenario import ScenarioSet

__all__ = ["ExtensiveForm", "build_extensive_form"]


@dataclass
class ExtensiveForm:
    """Result of :func:`build_extensive_form` (kept for inspection/testing)."""

    scenario_costs: list[Expression]
    objective: Expression
    risk: RiskMeasure


def build_extensive_form(
    model: Model,
    *,
    scenarios: ScenarioSet,
    recourse_builder,
    first_stage_cost: Expression | None = None,
    risk: RiskMeasure | None = None,
    minimize: bool = True,
) -> ExtensiveForm:
    """Assemble the deterministic-equivalent objective onto ``model`` (in place).

    Parameters
    ----------
    scenarios
        The probability-weighted scenario set.
    recourse_builder
        Callable ``(model, scenario_data, s) -> recourse_cost_expr``. It must create
        scenario ``s``'s second-stage variables (uniquely named, e.g. suffixed by
        ``s``) and add its recourse constraints to ``model``, returning the scenario's
        recourse **cost** expression (to be minimized in expectation / risk).
    first_stage_cost
        The here-and-now cost ``c·x`` (default 0).
    risk
        A :class:`~discopt.stochastic.risk.RiskMeasure` (default :class:`Expectation`).
    """
    risk = risk or Expectation()
    fsc = first_stage_cost if first_stage_cost is not None else Constant(0.0)

    costs: list[Expression] = []
    for s, scen in enumerate(scenarios):
        cost_s = recourse_builder(model, scen.data, s)
        if cost_s is None:
            raise ValueError(
                f"recourse_builder returned None for scenario {s}; it must "
                f"return the scenario's recourse cost expression"
            )
        costs.append(cost_s)

    objective = risk.build(model, fsc, costs, scenarios.probabilities)
    if minimize:
        model.minimize(objective)
    else:
        model.maximize(objective)
    return ExtensiveForm(scenario_costs=costs, objective=objective, risk=risk)
