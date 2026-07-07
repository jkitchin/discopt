"""Sample Average Approximation (SAA) statistical bounds.

SAA solves the stochastic program on a finite sample and studies how the sample
optimum relates to the true optimum. Two classic estimators (Mak–Morton–Wood 1999)
bracket the true optimal value ``v*`` of a **minimization** SP:

* **Statistical lower bound.** The SAA optimum ``v_N`` is biased *low*:
  ``E[v_N] ≤ v*``. Averaging ``M`` independent SAA batches and taking a one-sided
  Student-t confidence bound gives a valid statistical **lower** bound on ``v*``.
* **Statistical upper bound.** For a fixed candidate first-stage ``x̂``, the true
  objective ``E_ξ[f(x̂, ξ)] ≥ v*``; a sample mean over a large independent draw with
  a one-sided t bound gives a statistical **upper** bound.

The gap between the two estimates is a statistical optimality-gap estimate.

**Design.** The statistics here are pure NumPy over arrays of already-computed
values, so they are fully testable without an optimization backend (the batch
solves / candidate evaluations are the caller's job, and are exercised in CI). See
``docs/dev/stochastic-module-plan.md`` §5 (Phase 3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SAABound",
    "saa_lower_bound_estimate",
    "saa_upper_bound_estimate",
    "optimality_gap_estimate",
]


def _t_quantile(p: float, df: int) -> float:
    """One-sided Student-t quantile t_{p, df} (scipy if available, else normal)."""
    try:
        from scipy.stats import t as _t

        return float(_t.ppf(p, df))
    except Exception:  # pragma: no cover - scipy expected in this project
        from math import erf, sqrt

        # Normal approximation (df large); invert Φ by bisection.
        lo, hi = -20.0, 20.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if 0.5 * (1 + erf(mid / sqrt(2))) < p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


@dataclass
class SAABound:
    """A one-sided SAA statistical bound.

    ``bound`` is the confidence bound (a *lower* bound for the lower-bound
    estimator, an *upper* bound for the upper-bound estimator); ``mean`` is the
    point estimate; ``std_error`` the standard error; ``half_width`` the t-scaled
    margin (``|bound - mean|``).
    """

    mean: float
    std_error: float
    n: int
    confidence: float
    side: str  # "lower" | "upper"
    bound: float
    half_width: float


def _check(values, confidence):
    vals = np.asarray(values, dtype=float).ravel()
    if vals.size < 2:
        raise ValueError("SAA bounds need at least 2 samples (to estimate the variance)")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    return vals


def saa_lower_bound_estimate(batch_values, confidence: float = 0.95) -> SAABound:
    """One-sided statistical **lower** bound on ``v*`` from ``M`` SAA batch optima.

    ``batch_values`` are the optimal values of ``M`` independently-sampled SAA
    instances (minimization). Returns the lower confidence bound
    ``mean - t_{conf, M-1}·se`` (Mak–Morton–Wood).
    """
    vals = _check(batch_values, confidence)
    m = vals.size
    mean = float(vals.mean())
    se = float(vals.std(ddof=1) / np.sqrt(m))
    hw = _t_quantile(confidence, m - 1) * se
    return SAABound(mean, se, m, confidence, "lower", mean - hw, hw)


def saa_upper_bound_estimate(candidate_objectives, confidence: float = 0.95) -> SAABound:
    """One-sided statistical **upper** bound on ``v*`` from candidate evaluations.

    ``candidate_objectives`` are ``N'`` independent objective samples of a *fixed*
    first-stage candidate ``x̂``. Returns ``mean + t_{conf, N'-1}·se``.
    """
    vals = _check(candidate_objectives, confidence)
    n = vals.size
    mean = float(vals.mean())
    se = float(vals.std(ddof=1) / np.sqrt(n))
    hw = _t_quantile(confidence, n - 1) * se
    return SAABound(mean, se, n, confidence, "upper", mean + hw, hw)


def optimality_gap_estimate(lower: SAABound, upper: SAABound) -> dict:
    """Combine a lower and an upper SAA bound into an optimality-gap estimate.

    ``point_gap`` = ``upper.mean - lower.mean``; ``conservative_gap`` =
    ``upper.bound - lower.bound`` (a high-confidence gap using both one-sided bounds).
    """
    if lower.side != "lower" or upper.side != "upper":
        raise ValueError("optimality_gap_estimate expects a lower-side and an upper-side bound")
    return {
        "point_gap": upper.mean - lower.mean,
        "conservative_gap": upper.bound - lower.bound,
        "lower_bound": lower.bound,
        "upper_bound": upper.bound,
    }
