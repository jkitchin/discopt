"""Distributionally-robust bridge: worst case over a set of distributions.

Where :mod:`discopt.ro` optimizes the worst case over an uncertainty set of
*parameter values*, and the rest of :mod:`discopt.stochastic` optimizes the
expectation under a *fixed* distribution, **distributionally-robust optimization
(DRO)** optimizes the worst case over an *ambiguity set of distributions*:

    min_x  max_{P ∈ 𝒫}  E_P[ f(x, ξ) ].

For a finite :class:`~discopt.stochastic.scenario.ScenarioSet` the inner problem is
a linear program over the scenario probabilities ``p`` constrained to an ambiguity
set 𝒫 around the nominal ``p₀``. This module provides the core primitive — the
**worst-case distribution** for a total-variation (L1) ambiguity ball — which is the
heart of the inner maximization; embedding its dual as constraints yields the
single-level DRO model, exactly the robust-counterpart pattern of :mod:`discopt.ro`
(that full reformulation is the CI/next step; the worst-case weights here are
pure-NumPy and testable).

See ``docs/dev/stochastic-module-plan.md`` §1 (DRO bridge).
"""

from __future__ import annotations

import numpy as np

__all__ = ["worst_case_distribution", "worst_case_expected_cost"]


def worst_case_distribution(nominal_probs, costs, budget: float) -> np.ndarray:
    """Worst-case distribution for a total-variation (L1) ambiguity ball.

    Solves ``max_p Σ_s p_s c_s`` s.t. ``Σ_s |p_s - p0_s| ≤ budget``, ``p ≥ 0``,
    ``Σ_s p_s = 1``. The maximizer shifts probability mass from the lowest-cost
    scenarios to the highest-cost ones; moving mass ``ε`` between two scenarios
    changes the L1 distance by ``2ε``, so at most ``budget/2`` total mass moves.
    Returns the worst-case probability vector.
    """
    p0 = np.asarray(nominal_probs, dtype=float).copy()
    c = np.asarray(costs, dtype=float)
    if p0.shape != c.shape:
        raise ValueError("nominal_probs and costs must have the same shape")
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    p: np.ndarray = p0.copy()
    movable = min(budget / 2.0, 1.0)  # total mass we may relocate

    order = np.argsort(c)  # ascending cost: donors first, receivers last
    # Remove up to `movable` mass from the cheapest scenarios.
    remove = movable
    removed_total = 0.0
    for i in order:
        if remove <= 0:
            break
        take = min(p[i], remove)
        p[i] -= take
        remove -= take
        removed_total += take
    # Add the removed mass to the most expensive scenarios (descending cost).
    add = removed_total
    for i in order[::-1]:
        if add <= 0:
            break
        # room to fill this scenario toward mass 1 (a single scenario can hold all mass)
        give = min(1.0 - p[i], add)
        p[i] += give
        add -= give
    return p


def worst_case_expected_cost(nominal_probs, costs, budget: float) -> float:
    """The worst-case expected cost ``max_{P ∈ 𝒫} E_P[c]`` for the L1 ball."""
    p = worst_case_distribution(nominal_probs, costs, budget)
    return float(np.dot(p, np.asarray(costs, float)))
