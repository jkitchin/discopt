"""Paper-scale gate for simultaneous neural-DAE training (HM2.4).

Ports ``scripts/hybrid_ml/exp_c_paper_scale.py`` onto the HM1+HM2 API: a
1-30-30-1 softplus network (1021 weights) trained across three trajectories that
share the weights, full-space, no decomposition.

**This test is the canary for the falsified-decomposition premise**
(implementation plan §0.4 / §11): the exploration measured that full-space
training converges in 17 iterations / ~40 s at this scale, so no decomposition
machinery was built. If this gate starts failing, that premise needs
re-examination — record it in the plan's §11 rather than silently loosening the
thresholds.
"""

import numpy as np
import pytest
from discopt.dae import Trajectory, fit_trajectories
from discopt.modeling import Model
from discopt.nn import TrainableNetwork, train

pytestmark = pytest.mark.slow

TF = 2.0


def _r_true(c):
    return 1.5 * c**2 / (0.3 + c)


def _make_data(seed, ca0):
    from scipy.integrate import solve_ivp

    rng = np.random.default_rng(seed)
    t = np.linspace(0.05, TF, 15)
    sol = solve_ivp(
        lambda t, y: [-_r_true(y[0]), _r_true(y[0])],
        (0, TF),
        [ca0, 0.0],
        t_eval=t,
        rtol=1e-10,
        atol=1e-12,
    )
    yA = sol.y[0] + 0.01 * rng.standard_normal(15)
    yB = sol.y[1] + 0.01 * rng.standard_normal(15)
    return t, yA, yB


def test_paper_scale_full_space():
    """1-30-30-1 softplus, 3 trajectories, shared weights, full-space local solve.

    Thresholds carry margin on the measured 17 iters / ~40 s (implementation
    plan §2): iterations <= 40, wall <= 5 min.
    """
    import time

    ca0s = [1.0, 0.8, 0.6]
    trajs = [
        Trajectory(
            t_data=_make_data(k, ca0)[0],
            y_data={"cA": _make_data(k, ca0)[1], "cB": _make_data(k, ca0)[2]},
            initial={"cA": ca0, "cB": 0.0},
        )
        for k, ca0 in enumerate(ca0s)
    ]

    m = Model()
    net = TrainableNetwork(
        m, [1, 30, 30, 1], activation="softplus", weight_bounds=(-8, 8), name="rate"
    )
    assert net.n_parameters() == 1021

    def rhs(t, s, a, u):
        r = net(s["cA"])
        return {"cA": -r, "cB": r}

    fit = fit_trajectories(
        m,
        trajectories=trajs,
        states=[("cA", dict(bounds=(0.0, 1.5))), ("cB", dict(bounds=(0.0, 1.5)))],
        rhs=rhs,
        t_span=(0.0, TF),
        nfe=20,
        ncp=2,
    )
    m.minimize(fit.least_squares() + 1e-4 * net.l2_penalty())

    t0 = time.perf_counter()
    res = train(
        m,
        initial_solution=fit.warm_start() | net.initial_values(seed=0),
        options={"max_iter": 3000, "tol": 1e-8},
    )
    wall = time.perf_counter() - t0

    assert res.status.name == "OPTIMAL"
    assert res.iterations <= 40, f"iterations {res.iterations} exceeds 40 (measured 17)"
    assert wall <= 300, f"wall {wall:.0f}s exceeds 300s"

    # Recovered rate law is accurate over the visited range.
    frozen = net.freeze(res)
    c_grid = np.linspace(0.05, 1.0, 100)
    r_hat = frozen.forward(c_grid.reshape(-1, 1)).ravel()
    rate_rmse = float(np.sqrt(np.mean((r_hat - _r_true(c_grid)) ** 2)))
    assert rate_rmse <= 0.05, f"rate RMSE {rate_rmse:.4f} exceeds 0.05 (measured 0.013)"
