"""Acquisition functions for active-learning optimization.

An acquisition function scores candidate design points by how
*valuable* they are to evaluate next. Higher scores are better; the
optimizer selects the top-``batch_size`` candidates.

Every function in this module has the signature::

    score(surrogate, X_candidates, *, y_best, direction, **kw) -> ndarray

* ``surrogate`` is a fitted object satisfying the
  :class:`~discopt.doe.surrogate.Surrogate` protocol.
* ``X_candidates`` is a 2D array of shape ``(n, d)``.
* ``y_best`` is the best observed response so far (incumbent).
* ``direction`` is ``+1`` for *maximize* or ``-1`` for *minimize*.

Implementations
---------------

* :func:`expected_improvement` -- the standard EI used in Bayesian
  optimization. Balances exploitation (high predicted mean in the
  desired direction) and exploration (high uncertainty).
* :func:`upper_confidence_bound` / :func:`lower_confidence_bound` --
  UCB / LCB. A direction-aware wrapper :func:`confidence_bound` picks
  the right sign automatically.
* :func:`steepest_ascent` -- Box-Wilson style: score points by the
  predicted improvement only (ignores uncertainty). Useful with a
  response-surface surrogate when you want classical RSM behaviour.
"""

from __future__ import annotations

import math
from typing import Callable, Literal

import numpy as np

from discopt.doe.surrogate import Surrogate

Direction = Literal[1, -1]


def _erf(x: np.ndarray) -> np.ndarray:
    return np.asarray(np.vectorize(math.erf)(x), dtype=float)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.asarray(np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi), dtype=float)


def expected_improvement(
    surrogate: Surrogate,
    X_candidates: np.ndarray,
    *,
    y_best: float,
    direction: Direction,
    xi: float = 0.0,
) -> np.ndarray:
    """Expected improvement over the incumbent ``y_best``.

    For maximization (``direction = +1``)::

        EI(x) = (μ - y_best - xi) Φ(z) + σ φ(z)
        z     = (μ - y_best - xi) / σ

    For minimization (``direction = -1``), the sign flips so EI is
    always non-negative for points that improve on the incumbent.

    ``xi`` is an exploration knob -- a small positive value
    (e.g. ``0.01``) demands a slightly stronger improvement and
    encourages exploration.
    """
    mu, sigma = surrogate.predict(np.asarray(X_candidates, dtype=float))
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    dir_sign = int(direction)
    if dir_sign not in (1, -1):
        raise ValueError(f"direction must be +1 or -1, got {dir_sign}")

    improvement = dir_sign * (mu - y_best) - xi
    safe_sigma = np.where(sigma > 0.0, sigma, 1.0)
    z = improvement / safe_sigma
    ei = improvement * _norm_cdf(z) + safe_sigma * _norm_pdf(z)
    ei = np.where(sigma > 0.0, ei, np.maximum(improvement, 0.0))
    return ei


def upper_confidence_bound(
    surrogate: Surrogate,
    X_candidates: np.ndarray,
    *,
    kappa: float = 2.0,
    **_: object,
) -> np.ndarray:
    """``μ + κ σ`` -- maximize when ``direction = +1``."""
    mu, sigma = surrogate.predict(np.asarray(X_candidates, dtype=float))
    return np.asarray(mu).ravel() + float(kappa) * np.asarray(sigma).ravel()


def lower_confidence_bound(
    surrogate: Surrogate,
    X_candidates: np.ndarray,
    *,
    kappa: float = 2.0,
    **_: object,
) -> np.ndarray:
    """``-(μ - κ σ)`` -- larger is *better* under minimization."""
    mu, sigma = surrogate.predict(np.asarray(X_candidates, dtype=float))
    return -(np.asarray(mu).ravel() - float(kappa) * np.asarray(sigma).ravel())


def confidence_bound(
    surrogate: Surrogate,
    X_candidates: np.ndarray,
    *,
    direction: Direction,
    kappa: float = 2.0,
    **_: object,
) -> np.ndarray:
    """Direction-aware UCB/LCB. Higher score = better candidate."""
    if direction == 1:
        return upper_confidence_bound(surrogate, X_candidates, kappa=kappa)
    if direction == -1:
        return lower_confidence_bound(surrogate, X_candidates, kappa=kappa)
    raise ValueError(f"direction must be +1 or -1, got {direction}")


def steepest_ascent(
    surrogate: Surrogate,
    X_candidates: np.ndarray,
    *,
    direction: Direction,
    y_best: float | None = None,
    **_: object,
) -> np.ndarray:
    """Predicted improvement only -- ignores uncertainty.

    Score is ``direction * μ(x)`` (or, if ``y_best`` is supplied,
    ``direction * (μ(x) - y_best)``, which only shifts the score and
    does not change the ranking). Use this with a response-surface
    surrogate to reproduce classical Box-Wilson behaviour.
    """
    mu, _sigma = surrogate.predict(np.asarray(X_candidates, dtype=float))
    mu = np.asarray(mu, dtype=float).ravel()
    if y_best is None:
        return int(direction) * mu
    return int(direction) * (mu - float(y_best))


ACQUISITIONS: dict[str, Callable[..., np.ndarray]] = {
    "expected_improvement": expected_improvement,
    "ei": expected_improvement,
    "ucb": confidence_bound,
    "lcb": confidence_bound,
    "confidence_bound": confidence_bound,
    "steepest_ascent": steepest_ascent,
}


def resolve_acquisition(name_or_fn):
    """Look up an acquisition function by string or pass through a callable."""
    if callable(name_or_fn):
        return name_or_fn
    try:
        return ACQUISITIONS[name_or_fn]
    except KeyError as e:
        raise ValueError(
            f"unknown acquisition {name_or_fn!r}; available: {sorted(set(ACQUISITIONS))}"
        ) from e


__all__ = [
    "ACQUISITIONS",
    "confidence_bound",
    "expected_improvement",
    "lower_confidence_bound",
    "resolve_acquisition",
    "steepest_ascent",
    "upper_confidence_bound",
]
