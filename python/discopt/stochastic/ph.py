"""Progressive hedging (PH) for two-stage stochastic programs.

PH {cite:p}`RockafellarWets1991` is a scenario-decomposition method: each scenario
solves an augmented copy of the problem with its *own* first-stage decision, and a
proximal term + multipliers drive those copies to a common **nonanticipative**
value ``x̄ = Σ_s p_s x_s`` (the here-and-now decision cannot depend on which
scenario is realized).

Each PH iteration:

1. every scenario ``s`` solves ``min f_s(x_s, y_s) + w_s·x_s + (ρ/2)‖x_s - x̄‖²``,
   returning its first-stage vector ``x_s``;
2. consensus ``x̄ ← Σ_s p_s x_s``;
3. multipliers ``w_s ← w_s + ρ (x_s - x̄)``;
4. stop when the probability-weighted nonanticipativity gap ``‖x_s - x̄‖`` is small.

**Design for testability.** The subproblem solve is *injected*: the PH loop here is
pure NumPy given a ``subproblem_solver`` callback. That makes the numerical core
(consensus, multiplier update, convergence) verifiable against analytic subproblems
with no optimization backend (see :func:`quadratic_subproblem_solver`); the real
discopt-NLP subproblem solver plugs into the same interface (used under the backend,
so exercised in CI). PH gives a **primal** solution; a valid dual bound comes from
the Lagrangian relaxation and is reported separately.

See ``docs/dev/stochastic-module-plan.md`` §5 (Phase 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from discopt.stochastic.scenario import ScenarioSet

__all__ = ["PHResult", "progressive_hedging", "quadratic_subproblem_solver"]


@dataclass
class PHResult:
    """Result of :func:`progressive_hedging`."""

    x_bar: np.ndarray  # the nonanticipative first-stage decision
    x_scenarios: np.ndarray  # (n_scenarios, d) per-scenario first-stage solutions
    weights: np.ndarray  # (n_scenarios, d) PH multipliers
    iterations: int
    gap: float  # final nonanticipativity gap
    converged: bool
    history: list  # per-iteration gap


def progressive_hedging(
    scenarios: ScenarioSet,
    first_stage_dim: int,
    subproblem_solver,
    *,
    rho: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-6,
    x0=None,
) -> PHResult:
    """Run progressive hedging.

    Parameters
    ----------
    scenarios
        The probability-weighted scenario set.
    first_stage_dim
        Dimension ``d`` of the first-stage decision vector.
    subproblem_solver
        Callable ``(s, scenario_data, x_bar, w_s, rho) -> x_s`` returning scenario
        ``s``'s first-stage vector (length ``d``) for the augmented subproblem. This
        is where the actual optimization happens; PH itself only coordinates.
    rho
        Proximal / augmented-Lagrangian penalty (> 0).
    max_iter, tol
        Iteration cap and nonanticipativity-gap tolerance.
    x0
        Optional initial consensus ``x̄`` (default zeros).
    """
    if rho <= 0:
        raise ValueError("progressive hedging needs rho > 0")
    n = len(scenarios)
    d = int(first_stage_dim)
    probs = scenarios.probabilities

    w = np.zeros((n, d))
    x = np.zeros((n, d))
    x_bar = np.zeros(d) if x0 is None else np.asarray(x0, float).reshape(d)

    history: list = []
    it = 0
    gap = np.inf
    converged = False
    for it in range(1, max_iter + 1):
        x_bar_old = x_bar.copy()
        for s in range(n):
            xs = np.asarray(subproblem_solver(s, scenarios[s].data, x_bar, w[s], rho), float)
            x[s] = xs.reshape(d)
        x_bar = np.average(x, axis=0, weights=probs)
        w += rho * (x - x_bar)  # multiplier update (keeps Σ p_s w_s = 0 if it started so)
        # Two residuals (ADMM-style): the primal / nonanticipativity residual is the
        # scenario dispersion; the dual residual is the consensus movement. The gap
        # must include the dual term — when ρ is large the scenarios can agree
        # (primal ≈ 0) while x̄ is still drifting to its optimum.
        primal = float(np.sqrt(np.average(np.sum((x - x_bar) ** 2, axis=1), weights=probs)))
        dual = float(np.linalg.norm(x_bar - x_bar_old))
        gap = max(primal, dual)
        history.append(gap)
        if primal < tol and dual < tol:
            converged = True
            break

    return PHResult(
        x_bar=x_bar,
        x_scenarios=x.copy(),
        weights=w.copy(),
        iterations=it,
        gap=gap,
        converged=converged,
        history=history,
    )


def quadratic_subproblem_solver(targets, hessian=2.0):
    """A closed-form subproblem solver for separable quadratic scenarios (test/illustration).

    Scenario ``s`` minimizes ``(h/2)‖x - a_s‖² + w_s·x + (ρ/2)‖x - x̄‖²``, whose
    minimizer is ``x_s = (h·a_s - w_s + ρ·x̄) / (h + ρ)``. With PH this converges to
    the expected-value solution ``x̄ = Σ_s p_s a_s``. ``targets`` maps scenario index
    to its ideal point ``a_s``.
    """
    h = float(hessian)

    def solver(s, data, x_bar, w_s, rho):
        a_s = np.asarray(targets[s], float)
        return (h * a_s - np.asarray(w_s, float) + rho * np.asarray(x_bar, float)) / (h + rho)

    return solver
