"""``pounce_sensitivity`` -- sIPOPT parametric sensitivity (#1216).

Two things are pinned here.

**The active set is part of the system.** The sensitivity system is
``[W J_Aᵀ; J_A 0][dx; dλ] = -[∂²L/∂x∂p; ∂g_A/∂p]``, where every row of ``J_A``
is a constraint that is *active* at the solution. Assembling it over all rows
instead pins the solution to constraints that are slack. The witness is one
line of algebra: on ``min (y-q)² s.t. y ≤ 1`` at ``q = 0.5`` the bound is slack,
``y* = q``, and ``dy*/dq = 1`` -- the all-rows assembly returned ``-2e-10``.

**Second order.** ``d²x*/dp²`` is compared against the closed form on the
projection follower, where ``y0(p) = p/sqrt(p²+1)``.

Every derivative assertion compares against a closed form or an independent
finite difference of a *re-solved* problem, never against the other code path
alone.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling import Model
from discopt.solvers.sipopt import pounce_sensitivity

pytestmark = pytest.mark.requires_pounce


def _clip_model(q_value: float) -> tuple[Model, object]:
    """``min (y - q)^2  s.t.  y <= 1``: y* = min(q, 1), dy*/dq = 1 then 0."""
    m = Model("clip")
    y = m.continuous("y", lb=-5.0, ub=5.0)
    q = m.parameter("q", value=q_value)
    m.minimize((y - q) * (y - q))
    m.subject_to(y <= 1.0)
    return m, q


def _projection_model(p_value: float) -> tuple[Model, object]:
    m = Model("projection")
    y = m.continuous("y", shape=(2,), lb=-2.0, ub=2.0)
    q = m.parameter("q", value=p_value)
    m.minimize((y[0] - q) ** 2 + (y[1] - 1.0) ** 2)
    m.subject_to(y[0] * y[0] + y[1] * y[1] == 1.0)
    return m, q


@pytest.mark.slow
@pytest.mark.parametrize("method", ["exact", "fd"])
@pytest.mark.parametrize(("q_value", "expected"), [(0.5, 1.0), (2.0, 0.0)])
def test_inactive_inequality_does_not_pin_the_solution(method, q_value, expected):
    m, q = _clip_model(q_value)
    sens = pounce_sensitivity(m, [q], method=method)
    assert float(sens.dx_dp[0, 0]) == pytest.approx(expected, abs=1e-5)
    assert bool(sens.active[0]) is (q_value > 1.0)


@pytest.mark.slow
def test_first_order_matches_a_black_box_finite_difference():
    """The reported dx*/dp reproduces a re-solve at perturbed parameter values."""
    m, q = _projection_model(1.3)
    sens = pounce_sensitivity(m, [q])

    eps = 1e-5

    def solved_x(value):
        mm, qq = _projection_model(value)
        return pounce_sensitivity(mm, [qq]).x_star

    fd = (solved_x(1.3 + eps) - solved_x(1.3 - eps)) / (2 * eps)
    assert sens.dx_dp.ravel() == pytest.approx(fd, abs=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("p_value", [0.5, 2.0])
def test_second_order_matches_the_closed_form(p_value):
    m, q = _projection_model(p_value)
    sens = pounce_sensitivity(m, [q], order=2)
    assert sens.d2x_dp2 is not None
    assert sens.d2x_dp2.shape == (2, 1, 1)
    assert float(sens.dx_dp[0, 0]) == pytest.approx((p_value**2 + 1.0) ** -1.5, rel=1e-6)
    assert float(sens.d2x_dp2[0, 0, 0]) == pytest.approx(
        -3.0 * p_value * (p_value**2 + 1.0) ** -2.5, rel=1e-5
    )


@pytest.mark.slow
def test_first_order_default_leaves_second_order_unset():
    m, q = _projection_model(1.0)
    assert pounce_sensitivity(m, [q]).d2x_dp2 is None


@pytest.mark.slow
def test_exact_and_fd_right_hand_sides_agree():
    """The two right-hand-side constructions are independent; they must agree."""
    m, q = _projection_model(1.7)
    exact = pounce_sensitivity(m, [q], method="exact")
    fd = pounce_sensitivity(m, [q], method="fd")
    assert exact.dx_dp == pytest.approx(fd.dx_dp, abs=1e-6)
    assert exact.dlambda_dp == pytest.approx(fd.dlambda_dp, abs=1e-5)


@pytest.mark.slow
def test_predict_uses_the_first_order_expansion():
    m, q = _projection_model(1.0)
    sens = pounce_sensitivity(m, [q], order=2)
    dp = 0.05
    predicted = sens.predict([1.0 + dp])
    truth = np.array([1.0 + dp, 1.0]) / np.sqrt((1.0 + dp) ** 2 + 1.0)
    # A first-order expansion is wrong at O(dp^2 * d2x/dp2) and no better: the
    # bound below is the second-order term the result itself reports, so this
    # asserts the expansion is right rather than merely close.
    curvature = np.abs(sens.d2x_dp2[:, 0, 0])
    assert np.all(np.abs(predicted - truth) <= 0.6 * curvature * dp**2 + 1e-8)


@pytest.mark.unit
def test_bad_order_and_method_are_refused():
    m, q = _projection_model(1.0)
    with pytest.raises(ValueError, match="order must be"):
        pounce_sensitivity(m, [q], order=3)
    with pytest.raises(ValueError, match="method must be"):
        pounce_sensitivity(m, [q], method="magic")
    with pytest.raises(ValueError, match="order=2 requires"):
        pounce_sensitivity(m, [q], order=2, method="fd")


def _portfolio_model(r_value: float):
    """The sIPOPT tutorial's 6-asset portfolio (docs/notebooks/tutorial_pounce_sipopt).

    At a binding return target two weights are driven onto their lower bound but
    stop ~2e-6 short of it -- POUNCE is an interior-point method -- while carrying
    bound multipliers of ~1e-3. Identifying the active set by distance alone calls
    them free and their stationarity rows then miss the bound duals by 1e-3.
    """
    rng = np.random.default_rng(7)
    n_assets = 6
    mu_vec = np.array([0.15, 0.10, 0.08, 0.12, 0.07, 0.05])
    cov_root = rng.normal(0, 0.04, (n_assets, n_assets))
    sigma = cov_root @ cov_root.T + 0.01 * np.eye(n_assets)

    m = Model("portfolio")
    r_min = m.parameter("r_min", value=r_value)
    w = m.continuous("w", shape=(n_assets,), lb=0, ub=1)
    m.minimize(
        dm.sum(
            lambda i: dm.sum(lambda j: float(sigma[i, j]) * w[i] * w[j], over=range(n_assets)),
            over=range(n_assets),
        )
    )
    m.subject_to(dm.sum(w) == 1.0, name="budget")
    m.subject_to(
        dm.sum(lambda i: float(mu_vec[i]) * w[i], over=range(n_assets)) >= r_min,
        name="return",
    )
    return m, r_min


@pytest.mark.slow
def test_interior_point_active_set_is_identified_against_the_multipliers():
    """A binding target: the sensitivity must exist and match a re-solve."""
    m, r_min = _portfolio_model(0.12)
    sens = pounce_sensitivity(m, [r_min])
    assert bool(sens.active[1]) is True  # the return constraint binds
    assert int(sens.at_bound.sum()) == 2  # two weights are on their lower bound

    h = 1e-5

    def solved_w(value):
        mm, rr = _portfolio_model(value)
        return pounce_sensitivity(mm, [rr]).x_star

    fd = (solved_w(0.12 + h) - solved_w(0.12 - h)) / (2 * h)
    assert sens.dx_dp.ravel() == pytest.approx(fd, abs=2e-3)
    # The two bound-pinned weights do not move with the target.
    assert sens.dx_dp[sens.at_bound].ravel() == pytest.approx(np.zeros(2), abs=1e-9)


@pytest.mark.slow
def test_slack_constraint_contributes_no_sensitivity():
    """At a target the min-variance portfolio already meets, dw*/dr_min is 0."""
    m, r_min = _portfolio_model(0.09)
    sens = pounce_sensitivity(m, [r_min])
    assert bool(sens.active[1]) is False
    assert sens.dx_dp.ravel() == pytest.approx(np.zeros(6), abs=1e-12)

    h = 1e-5

    def solved_w(value):
        mm, rr = _portfolio_model(value)
        return pounce_sensitivity(mm, [rr]).x_star

    fd = (solved_w(0.09 + h) - solved_w(0.09 - h)) / (2 * h)
    assert np.max(np.abs(fd)) < 1e-3  # the re-solve agrees: the target is slack
