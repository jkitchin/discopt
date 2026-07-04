"""Risk measures for stochastic programming.

A risk measure turns first-stage cost plus per-scenario recourse costs into a
single objective expression, adding any auxiliary variables/constraints to the
model. Phase 0 (``docs/dev/stochastic-module-plan.md`` §3.3) provides:

* :class:`Expectation` — the risk-neutral ``E[·] = Σ_s p_s Q_s``;
* :class:`CVaR` — Conditional Value-at-Risk via the Rockafellar–Uryasev form
  ``η + 1/(1-α) Σ_s p_s (Q_s - η)^+``;
* :class:`MeanCVaR` — a convex combination of the two;
* :func:`chance_constraint` — an SAA chance constraint
  ``P(g(x, ξ) ≤ 0) ≥ 1 - ε`` via per-scenario indicators.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from discopt.modeling.core import Constant, Expression, Model

__all__ = ["RiskMeasure", "Expectation", "CVaR", "MeanCVaR", "chance_constraint"]


def _weighted_sum(pairs: list[tuple[float, Expression]]) -> Expression:
    """Σ coeff · expr, skipping zero coefficients; returns Constant(0) if empty."""
    acc: Expression | None = None
    for coeff, expr in pairs:
        if coeff == 0.0:
            continue
        term = expr if coeff == 1.0 else Constant(float(coeff)) * expr
        acc = term if acc is None else acc + term
    return acc if acc is not None else Constant(0.0)


class RiskMeasure:
    """Interface: build the objective from first-stage + per-scenario recourse costs."""

    def build(
        self, model: Model, first_stage_cost: Expression, scenario_costs, probs
    ) -> Expression:  # pragma: no cover - interface
        raise NotImplementedError


class Expectation(RiskMeasure):
    """Risk-neutral expected cost: ``first_stage + Σ_s p_s Q_s``."""

    def build(self, model, first_stage_cost, scenario_costs, probs) -> Expression:
        exp = _weighted_sum([(float(p), c) for p, c in zip(probs, scenario_costs)])
        return cast(Expression, first_stage_cost + exp)


class CVaR(RiskMeasure):
    """Conditional Value-at-Risk at level ``alpha`` (Rockafellar–Uryasev).

    ``first_stage + η + 1/(1-α) Σ_s p_s u_s`` with ``u_s ≥ Q_s - η``, ``u_s ≥ 0``.
    ``alpha`` is the confidence level (e.g. 0.95 penalizes the worst 5% tail).
    """

    def __init__(self, alpha: float, *, prefix: str = "cvar"):
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"CVaR alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)
        self.prefix = prefix

    def build(self, model, first_stage_cost, scenario_costs, probs) -> Expression:
        inv = 1.0 / (1.0 - self.alpha)
        eta = model.continuous(f"{self.prefix}_eta")  # VaR (free)
        tail_pairs: list[tuple[float, Expression]] = []
        for s, (c, p) in enumerate(zip(scenario_costs, probs)):
            u = model.continuous(f"{self.prefix}_u{s}", lb=0.0)
            model.subject_to(u >= c - eta, name=f"{self.prefix}_shortfall{s}")
            tail_pairs.append((float(p) * inv, u))
        return cast(Expression, first_stage_cost + eta + _weighted_sum(tail_pairs))


class MeanCVaR(RiskMeasure):
    """Convex combination ``(1-λ)·E[·] + λ·CVaR_α`` — a mean/risk trade-off."""

    def __init__(self, lam: float, alpha: float, *, prefix: str = "mcvar"):
        if not (0.0 <= lam <= 1.0):
            raise ValueError(f"MeanCVaR lambda must be in [0, 1], got {lam}")
        self.lam = float(lam)
        self.alpha = float(alpha)
        self.prefix = prefix

    def build(self, model, first_stage_cost, scenario_costs, probs) -> Expression:
        zero = Constant(0.0)
        exp = Expectation().build(model, zero, scenario_costs, probs)
        cvar = CVaR(self.alpha, prefix=self.prefix).build(model, zero, scenario_costs, probs)
        return cast(
            Expression,
            first_stage_cost + Constant(1.0 - self.lam) * exp + Constant(self.lam) * cvar,
        )


def chance_constraint(
    model: Model,
    scenario_bodies: list[Expression],
    probs,
    epsilon: float,
    *,
    big_m: float,
    prefix: str = "cc",
) -> list:
    """SAA chance constraint ``P(g(x, ξ) ≤ 0) ≥ 1 - ε``.

    For each scenario, a binary ``z_s`` with ``g_s ≤ M z_s`` lets the constraint be
    violated only when ``z_s = 1``; ``Σ_s p_s z_s ≤ ε`` caps the violation
    probability. Documented as a *finite-sample* (SAA) approximation. Returns the
    indicator variables.
    """
    if big_m <= 0:
        raise ValueError("chance_constraint needs a positive big_m")
    if not (0.0 <= epsilon < 1.0):
        raise ValueError(f"epsilon must be in [0, 1), got {epsilon}")
    probs = np.asarray(probs, float)
    zs = []
    cov_pairs: list[tuple[float, Expression]] = []
    for s, (g, p) in enumerate(zip(scenario_bodies, probs)):
        z = model.binary(f"{prefix}_z{s}")
        model.subject_to(g <= Constant(float(big_m)) * z, name=f"{prefix}_link{s}")
        zs.append(z)
        cov_pairs.append((float(p), z))
    model.subject_to(_weighted_sum(cov_pairs) <= epsilon, name=f"{prefix}_coverage")
    return zs
