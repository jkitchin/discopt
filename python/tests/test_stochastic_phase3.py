"""SP Phase 3: SAA statistical bounds, discrete DRO, and reserved-method stubs.

All solver-free: the SAA confidence intervals are checked for correct one-sided
*coverage* over many synthetic trials; the DRO worst-case distribution is checked
against a scipy LP oracle; multistage / integer-L-shaped refuse loudly. Needs
NumPy + scipy; no JAX/Rust.
"""

from __future__ import annotations

import numpy as np
import pytest

scipy_opt = pytest.importorskip("scipy.optimize")

from discopt.stochastic import (  # noqa: E402
    integer_lshaped,
    optimality_gap_estimate,
    saa_lower_bound_estimate,
    saa_upper_bound_estimate,
    solve_multistage,
    worst_case_distribution,
    worst_case_expected_cost,
)

pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------
# SAA statistical bounds — one-sided coverage calibration.
# ---------------------------------------------------------------------------


def test_saa_lower_bound_one_sided_coverage():
    rng = np.random.default_rng(0)
    mu, sigma, m, conf = 10.0, 2.0, 25, 0.9
    trials = 3000
    covered = sum(
        saa_lower_bound_estimate(rng.normal(mu, sigma, m), confidence=conf).bound <= mu
        for _ in range(trials)
    )
    assert abs(covered / trials - conf) < 0.03  # calibrated: P(bound <= μ) ≈ conf


def test_saa_upper_bound_one_sided_coverage():
    rng = np.random.default_rng(1)
    mu, sigma, n, conf = -3.0, 1.5, 40, 0.95
    trials = 3000
    covered = sum(
        saa_upper_bound_estimate(rng.normal(mu, sigma, n), confidence=conf).bound >= mu
        for _ in range(trials)
    )
    assert abs(covered / trials - conf) < 0.03


def test_saa_bounds_need_two_samples():
    with pytest.raises(ValueError, match="at least 2"):
        saa_lower_bound_estimate([1.0])


def test_optimality_gap_estimate():
    lo = saa_lower_bound_estimate([8.0, 9.0, 10.0, 11.0, 12.0], confidence=0.9)  # mean 10
    up = saa_upper_bound_estimate([11.0, 12.0, 13.0, 12.0, 12.0], confidence=0.9)  # mean 12
    gap = optimality_gap_estimate(lo, up)
    assert abs(gap["point_gap"] - 2.0) < 1e-9
    assert gap["conservative_gap"] >= gap["point_gap"]  # one-sided bounds widen the gap
    with pytest.raises(ValueError, match="lower-side and an upper-side"):
        optimality_gap_estimate(up, lo)  # swapped


# ---------------------------------------------------------------------------
# Discrete DRO worst-case distribution vs a scipy LP oracle.
# ---------------------------------------------------------------------------


def _dro_lp_worstcost(p0, c, budget):
    """max Σ p_s c_s s.t. Σ|p_s-p0_s|<=budget, p>=0, Σp=1 — via linprog (min -c·p)."""
    n = len(c)
    ident = np.eye(n)
    obj = np.concatenate([-c, np.zeros(n)])  # vars [p, t]
    a_ub = np.vstack(
        [
            np.hstack([ident, -ident]),  # p - t <= p0
            np.hstack([-ident, -ident]),  # -p - t <= -p0
            np.concatenate([np.zeros(n), np.ones(n)]).reshape(1, -1),  # Σ t <= budget
        ]
    )
    b_ub = np.concatenate([p0, -p0, [budget]])
    a_eq = np.concatenate([np.ones(n), np.zeros(n)]).reshape(1, -1)
    res = scipy_opt.linprog(
        obj,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=[1.0],
        bounds=[(0, None)] * (2 * n),
        method="highs",
    )
    assert res.success, res.message
    return -res.fun


def test_dro_worst_case_cost_matches_linprog():
    rng = np.random.default_rng(3)
    for _ in range(25):
        n = int(rng.integers(3, 8))
        p0 = rng.dirichlet(np.ones(n))
        c = rng.normal(0, 5, n)
        budget = float(rng.uniform(0.0, 1.5))
        got = worst_case_expected_cost(p0, c, budget)
        ref = _dro_lp_worstcost(p0, c, budget)
        assert abs(got - ref) < 1e-6, f"n={n} budget={budget}"


def test_dro_budget_zero_is_nominal():
    p0 = np.array([0.2, 0.3, 0.5])
    c = np.array([1.0, 5.0, 2.0])
    assert np.allclose(worst_case_distribution(p0, c, 0.0), p0)


def test_dro_large_budget_concentrates_on_max_cost():
    p0 = np.array([0.25, 0.25, 0.25, 0.25])
    c = np.array([1.0, 7.0, 3.0, 2.0])
    p = worst_case_distribution(p0, c, budget=2.0)  # full TV budget -> all mass movable
    assert abs(p[1] - 1.0) < 1e-9 and np.allclose(np.delete(p, 1), 0.0)


# ---------------------------------------------------------------------------
# Reserved methods refuse loudly.
# ---------------------------------------------------------------------------


def test_multistage_and_integer_lshaped_refuse():
    with pytest.raises(NotImplementedError, match="multistage"):
        solve_multistage()
    with pytest.raises(NotImplementedError, match="integer"):
        integer_lshaped()
